"""Step 0 (operator learning): the exact teacher on the port set  box nodes  U  cut-band nodes.

One cell (body from fast_prep4, element moments from the polyhedral integrator, ghost penalty from the face list):
  K = K_body + gamma K_ghost over all active Q2 DOFs (node-major xyz, NODES order).
  Ports P = box nodes U cut-band nodes (fixed background grid, masks only); interior I = the rest.
Actions (all fp64, sparse direct factorizations with nvmath/cuDSS):
  extend(q)    u = E q : u_P = q, u_I = -K_II^-1 K_IP q          (the solution operator the network learns)
  apply(q)     S q = (K E q)_P                                    (exact, since the interior residual is zero)
  neumann(f)   q = S^+ f : K u = [f_eq; 0] with a statically determinate 3-2-1 support, rigid part removed
  sens(u)      -u^T (dK/dtau_c) u for the 8 thickness corners (dK/dtau from central differences of the moments at fixed
               topology; the ghost penalty does not depend on tau)
Self-test (main): rigid modes, symmetry, Neumann residual, apply vs an independent dense Schur complement (cuDSS Schur
mode on the same port set), sensitivity vs finite differences of the compliance under force control, timings.
Usage: teacher.py <body_dir> <out_json> <case> [<case> ...]
"""
import json, sys, time, gc, os
from pathlib import Path
from fractions import Fraction
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
dev, dt = torch.device('cuda:0'), torch.float64
ROOT = Path('/root/autodl-tmp/CUTFEM_FRESH_GP_20260921')


def sync():
    torch.cuda.synchronize()


def rigid_basis(node_ids, n):
    """Orthonormal rigid modes (6 columns) on the given grid nodes, node-major xyz."""
    g = np.stack(np.unravel_index(node_ids, (2 * n + 1,) * 3), 1) / (2 * n)
    c = g - g.mean(0)
    R = np.zeros((len(g), 3, 6))
    for a in range(3):
        R[:, a, a] = 1.0
        e = np.zeros(3); e[a] = 1.0
        R[:, :, 3 + a] = np.cross(e[None, :], c)
    Q, _ = np.linalg.qr(R.reshape(-1, 6))
    return torch.as_tensor(Q, dtype=dt, device=dev)


def _set_current():
    try:
        from cuda.core.experimental import Device
    except ImportError:
        from cuda.core import Device
    Device(torch.cuda.current_device()).set_current()


class SPDSolver:
    """nvmath DirectSolver on an SPD upper CSR (already Jacobi scaled), fp64 (or fp32), panels of width w."""

    def __init__(self, crow, col, vals, n, w=16, threads=16, fdt=None):
        self.fdt = vals.dtype if fdt is None else fdt
        vals = vals.to(self.fdt)
        from nvmath.sparse.advanced import DirectSolver, DirectSolverOptions, DirectSolverMatrixType, DirectSolverMatrixViewType
        lib = os.environ.get('CUDSS_MT')
        opts = DirectSolverOptions(sparse_system_type=DirectSolverMatrixType.SPD, sparse_system_view=DirectSolverMatrixViewType.UPPER,
                                   **(dict(multithreading_lib=lib) if lib else {}))
        self.n, self.w = n, w
        self.U = torch.sparse_csr_tensor(crow, col, vals, size=(n, n))
        b = torch.zeros((w, n), dtype=self.fdt, device=dev).T
        self.solver = DirectSolver(self.U, b, options=opts)
        if lib:
            self.solver.plan_config.host_nthreads = threads
        self.solver.plan()
        self.solver.factorize()

    def solve(self, r):
        _set_current()                                                     # autograd may call from its own thread
        out = torch.empty_like(r)
        for c0 in range(0, r.shape[1], self.w):
            k = min(self.w, r.shape[1] - c0)
            b = torch.zeros((self.w, self.n), dtype=self.fdt, device=dev).T
            b[:, :k] = r[:, c0:c0 + k].to(self.fdt)
            self.solver.reset_operands(b=b)
            out[:, c0:c0 + k] = self.solver.solve()[:, :k].to(r.dtype)
        return out

    def free(self):
        self.solver.free()


class Cell:
    def __init__(self, case, body_dir, ports=('box', 'cut'), s=4, levels=1, log=print):
        import element_moments as EM
        import polyref_torch_fast as PT
        import box_encode as BX
        self.PT, self.s, self.levels, self.log = PT, s, levels, log
        t0 = time.perf_counter()
        ctx = json.loads((ROOT / 'packets' / case / 'FRESH_CONTEXT.json').read_text())
        self.case, self.n = case, int(ctx['n'])
        Emod = float(ctx['material']['E']); nu = float(ctx['material']['nu'])
        lam = Emod * nu / ((1 + nu) * (1 - 2 * nu)); mu = Emod / (2 * (1 + nu))
        cs = ctx['case']
        self.kind = cs.get('kind')
        self.taus0 = [float(Fraction(v)) for v in cs['tau_corners']]
        self.normal = None if cs.get('normal') is None else [float(Fraction(v)) for v in cs['normal']]
        self.offset = None if self.normal is None else float(Fraction(cs['offset']))
        self.gamma = float(json.loads((ROOT / 'packets' / case / 'SAMPLE.json').read_text())['gp']['gamma'])
        d = Path(body_dir) / case
        self.nodes = np.load(d / 'NODES.npy'); self.cells = np.load(d / 'CELL_INDICES.npy')
        dofs = np.load(d / 'dofs.npy')
        arr = {'NODES.npy': self.nodes, 'dofs.npy': dofs, 'CELL_INDICES.npy': self.cells}
        xi = EM.local_coordinates(case, self.n, arr)
        keys = np.unique(xi.reshape(len(xi), -1), axis=0)
        if len(keys) != 1:
            raise ValueError('ONE_NODE_ORDERING_EXPECTED')
        self.Tm = torch.tensor(EM.pattern_operators(keys[0].reshape(27, 3), lam, mu, self.n)[1], dtype=dt, device=dev).reshape(125, 81, 81)
        iu = torch.triu_indices(81, 81, device=dev)
        self.iu = iu
        self.Tm_up = self.Tm[:, iu[0], iu[1]].contiguous()
        nb = 3 * len(self.nodes); self.nb = nb
        self.dofs = torch.as_tensor(dofs, dtype=torch.long, device=dev)
        r, c = self.dofs[:, iu[0]], self.dofs[:, iu[1]]
        key_e = torch.minimum(r, c) * nb + torch.maximum(r, c)
        gcache = d / 'GP_UPPER.npz'                                          # unit ghost matrix, upper, cached
        if gcache.exists():
            z = np.load(gcache)
            key_g, val_g = torch.as_tensor(z['keys'], device=dev), torch.as_tensor(z['vals'], device=dev)
        else:
            G = BX.ghost_faces_gpu(body_dir, case, self.n, nb)
            gi, gv = G.indices(), G.values()
            up = gi[0] <= gi[1]
            key_g, val_g = gi[0][up] * nb + gi[1][up], gv[up]
            del G, gi, gv
            gc.collect(); torch.cuda.empty_cache()
            np.savez(gcache, keys=key_g.cpu().numpy(), vals=val_g.cpu().numpy())
        U = torch.unique(torch.cat([key_e.reshape(-1), key_g]))
        self.pos_e = torch.searchsorted(U, key_e.reshape(-1)).int()
        del key_e
        self.base = torch.zeros(len(U), dtype=dt, device=dev)
        self.base.index_add_(0, torch.searchsorted(U, key_g), self.gamma * val_g)
        del key_g, val_g
        # memory-lean upper pattern (int32 indices; nb < 2^31, nnz < 2^31)
        self.ru, self.cu = (U // nb).int(), (U % nb).int()
        del U
        gc.collect(); torch.cuda.empty_cache()
        self.crow = torch.cat([torch.zeros(1, dtype=torch.long, device=dev), torch.cumsum(torch.bincount(self.ru, minlength=nb), 0)])
        self.diag = torch.nonzero(self.ru == self.cu).squeeze(1)
        # transposed pattern (CSR of U^T): permutation of the upper entries sorted by (col, row)
        self.tperm = torch.argsort(self.cu.long() * nb + self.ru.long()).int()
        self.crow_t = torch.cat([torch.zeros(1, dtype=torch.long, device=dev), torch.cumsum(torch.bincount(self.cu, minlength=nb), 0)]).int()
        self.col_t = self.ru[self.tperm.long()]
        gc.collect(); torch.cuda.empty_cache()
        if len(self.diag) != nb:
            raise ValueError('DIAGONAL_INCOMPLETE')
        box = np.load(d / 'BOX_NODES.npy'); cut = np.load(d / 'CUT_NODES.npy') if 'cut' in ports else np.zeros(0, np.int64)
        self.is_box = np.isin(self.nodes, box); self.is_cut = np.isin(self.nodes, cut)
        onport = self.is_box | self.is_cut if 'box' in ports else self.is_cut
        self.port_node_ids = self.nodes[onport]
        self.port_is_box = self.is_box[onport]; self.port_is_cut = self.is_cut[onport]
        pm = torch.as_tensor(np.repeat(onport, 3), device=dev)
        self.P = torch.nonzero(pm).squeeze(1); self.I = torch.nonzero(~pm).squeeze(1)
        self.np_, self.ni = len(self.P), len(self.I)
        self.Q = rigid_basis(self.port_node_ids, self.n)
        self.Qall = rigid_basis(self.nodes, self.n)
        self.sol_I = self.sol_N = None
        sync(); self.setup_seconds = time.perf_counter() - t0
        log(json.dumps(dict(event='SETUP', case=case, seconds=self.setup_seconds, dofs=nb, ports=self.np_, interior=self.ni,
                            box_nodes=int(self.is_box.sum()), cut_nodes=int(self.is_cut.sum()), elements=int(len(self.cells)))))

    # ---------------------------------------------------------------- assembly
    def moments(self, taus):
        return torch.as_tensor(self.PT.cell_moments(self.cells, self.n, taus, self.normal, self.offset, self.s, levels=self.levels),
                               dtype=dt, device=dev)

    def assemble(self, taus=None):
        taus = self.taus0 if taus is None else taus
        self.taus = list(taus)
        self._free(); self.U = self.Ut = None; gc.collect(); torch.cuda.empty_cache()
        self.M = self.moments(taus)
        vals = self.base.clone()
        for lo in range(0, len(self.M), 4096):
            ke = self.M[lo:lo + 4096] @ self.Tm_up
            vals.index_add_(0, self.pos_e[lo * 3321:(lo + len(ke)) * 3321].long(), ke.reshape(-1))
        self.vals = vals
        self.U = torch.sparse_csr_tensor(self.crow.int(), self.cu, vals, size=(self.nb, self.nb))
        self.Ut = torch.sparse_csr_tensor(self.crow_t, self.col_t, vals[self.tperm.long()], size=(self.nb, self.nb))
        self.dK = vals[self.diag]
        self.K = self                                                          # K @ x -> symmetric product
        gc.collect(); torch.cuda.empty_cache()
        return self

    def Kx(self, x):
        """Symmetric product K x = U x + U^T x - diag(K) x from the upper CSR (and its transpose)."""
        return self.U @ x + self.Ut @ x - self.dK[:, None] * x

    def __matmul__(self, x):                                                   # so that cell.K @ x keeps working
        return self.Kx(x)

    def _free(self):
        for s in (self.sol_I, self.sol_N):
            if s is not None:
                s.free()
        self.sol_I = self.sol_N = None
        gc.collect(); torch.cuda.empty_cache()

    # ---------------------------------------------------------------- factorizations
    def factor(self, neumann=True, fp32=False):
        self._free()
        self.fp32 = fp32
        st = {}
        sync(); t = time.perf_counter()
        # interior block K_II (ports held)
        pm = torch.zeros(self.nb, dtype=torch.bool, device=dev); pm[self.P] = True
        new = torch.full((self.nb,), -1, dtype=torch.long, device=dev); new[self.I] = torch.arange(self.ni, device=dev)
        ru, cu = self.ru.long(), self.cu.long()
        sel = torch.nonzero(~pm[ru] & ~pm[cu]).squeeze(1)
        rA, cA, vA = new[ru[sel]], new[cu[sel]], self.vals[sel]
        dA = vA[rA == cA]
        sA = torch.zeros(self.ni, dtype=dt, device=dev); sA[rA[rA == cA]] = 1 / torch.sqrt(dA)
        crow = torch.cat([torch.zeros(1, dtype=torch.long, device=dev), torch.cumsum(torch.bincount(rA, minlength=self.ni), 0)])
        self.sA = sA
        self.sol_I = SPDSolver(crow.int(), cA.int(), (vA * sA[rA] * sA[cA]).contiguous(), self.ni,
                               fdt=torch.float32 if fp32 else None)
        sync(); st['factor_interior'] = time.perf_counter() - t; t = time.perf_counter()
        if neumann:
            self.pin = self._pick_pins()
            pin = torch.zeros(self.nb, dtype=torch.bool, device=dev); pin[self.pin] = True
            v = self.vals.clone()
            v[pin[ru] | pin[cu]] = 0
            v[self.diag[self.pin]] = 1.0
            sN = 1 / torch.sqrt(v[self.diag])
            self.sN = sN
            del ru, cu
            self.sol_N = SPDSolver(self.crow.int(), self.cu, (v * sN[self.ru.long()] * sN[self.cu.long()]).contiguous(), self.nb)
            sync(); st['factor_neumann'] = time.perf_counter() - t
        return st

    def _pick_pins(self):
        """3-2-1 support on three far-apart, well-supported nodes: 6 DOFs whose restriction of the rigid modes is
        nonsingular (statically determinate: equilibrated loads produce no reactions)."""
        g = np.stack(np.unravel_index(self.nodes, (2 * self.n + 1,) * 3), 1).astype(float)
        dg = self.vals[self.diag].reshape(-1, 3).sum(1).cpu().numpy()
        strong = dg >= np.quantile(dg, 0.5)
        idx = np.flatnonzero(strong)
        a = idx[np.argmin(g[idx].sum(1))]
        b = idx[np.argmax(((g[idx] - g[a]) ** 2).sum(1))]
        ab = g[b] - g[a]
        area = np.linalg.norm(np.cross(g[idx] - g[a], ab[None, :]), axis=1)
        c = idx[np.argmax(area)]
        Q = self.Qall.cpu().numpy()
        best = None
        for cb in ((1, 2), (0, 2), (0, 1)):
            for cc in range(3):
                pins = [3 * a, 3 * a + 1, 3 * a + 2, 3 * b + cb[0], 3 * b + cb[1], 3 * c + cc]
                sv = np.linalg.svd(Q[pins], compute_uv=False)
                cond = sv[0] / max(sv[-1], 1e-300)
                if best is None or cond < best[0]:
                    best = (cond, pins)
        self.pin_cond = float(best[0])
        return torch.as_tensor(best[1], device=dev)

    # ---------------------------------------------------------------- actions
    def extend(self, q):
        """u = E q on all DOFs; q: (ports, k)."""
        x = torch.zeros((self.nb, q.shape[1]), dtype=dt, device=dev)
        x[self.P] = q
        r = -(self.K @ x)[self.I]
        x[self.I] = self.sA[:, None] * self.sol_I.solve(self.sA[:, None] * r)
        for _ in range(3 if getattr(self, 'fp32', False) else 0):             # fp64 iterative refinement
            res = -(self.K @ x)[self.I]                                         # interior residual (ports fixed)
            x[self.I] += self.sA[:, None] * self.sol_I.solve(self.sA[:, None] * res)
        return x

    def apply(self, q):
        return (self.K @ self.extend(q))[self.P]

    def neumann(self, f):
        """q = S^+ f (rigid-free); f: (ports, k) is projected onto the equilibrated subspace first."""
        f = f - self.Q @ (self.Q.T @ f)
        F = torch.zeros((self.nb, f.shape[1]), dtype=dt, device=dev); F[self.P] = f
        F[self.pin] = 0
        u = self.sN[:, None] * self.sol_N.solve(self.sN[:, None] * F)
        q = u[self.P]
        return q - self.Q @ (self.Q.T @ q)

    # ---------------------------------------------------------------- sensitivities
    def dmoments(self, rel=1e-5):
        """dM/dtau_c (8, E, 125) by central differences at fixed topology."""
        out = []
        for c in range(8):
            h = rel * self.taus[c]
            tp = list(self.taus); tp[c] += h
            tm = list(self.taus); tm[c] -= h
            out.append((self.moments(tp) - self.moments(tm)) / (2 * h))
        self.dM = torch.stack(out)
        return self.dM

    def energy_density(self, u, chunk=2048):
        """g[e, m, k] = u_e^T Tm_m u_e for full fields u (nb, k)."""
        g = torch.empty((len(self.cells), 125, u.shape[1]), dtype=dt, device=dev)
        for lo in range(0, len(self.cells), chunk):
            ue = u[self.dofs[lo:lo + chunk]]                              # c x 81 x k
            z = torch.einsum('eik,mij->emjk', ue, self.Tm)
            g[lo:lo + chunk] = torch.einsum('emjk,ejk->emk', z, ue)
            del z
        return g

    def sens(self, u):
        """-u^T (dK/dtau_c) u, (8, k)."""
        g = self.energy_density(u)
        return -torch.einsum('cem,emk->ck', self.dM, g)


def selftest(case, body_dir, log):
    rec = dict(case=case)
    t0 = time.perf_counter()
    C = Cell(case, body_dir, log=log)
    rec.update(dofs=C.nb, ports=C.np_, interior=C.ni, box_nodes=int(C.is_box.sum()), cut_nodes=int(C.is_cut.sum()))
    sync(); t = time.perf_counter(); C.assemble(); sync(); rec['assemble_s'] = time.perf_counter() - t
    rec.update({k + '_s': v for k, v in C.factor().items()}); rec['pin_cond'] = C.pin_cond
    gen = torch.Generator(device=dev).manual_seed(0)
    # rigid: E R must be the rigid motion on all DOFs, S R = 0
    uR = C.extend(C.Q)
    Rall = C.Qall
    resid = uR - Rall @ torch.linalg.lstsq(Rall, uR).solution
    rec['rigid_extension_rel'] = float(resid.norm() / uR.norm())
    q = torch.randn((C.np_, 8), dtype=dt, device=dev, generator=gen)
    sync(); t = time.perf_counter(); Sq = C.apply(q); sync(); rec['apply8_s'] = time.perf_counter() - t
    rec['rigid_force_rel'] = float((C.apply(C.Q)).norm() / Sq.norm() * q.norm() / C.Q.norm())
    M = q.T @ Sq
    rec['symmetry_rel'] = float((M - M.T).norm() / M.norm())
    # Neumann: S S^+ f = f_eq
    f = torch.randn((C.np_, 8), dtype=dt, device=dev, generator=gen)
    feq = f - C.Q @ (C.Q.T @ f)
    sync(); t = time.perf_counter(); qn = C.neumann(f); sync(); rec['neumann8_s'] = time.perf_counter() - t
    rec['neumann_residual_rel'] = float((C.apply(qn) - feq).norm() / feq.norm())
    # independent path: dense Schur complement on the same port set (cuDSS Schur mode)
    try:
        import schur_encoder as SE
        pd = Path(body_dir) / (case + '_portview'); pd.mkdir(exist_ok=True)
        for fn in ('NODES.npy', 'CELL_INDICES.npy', 'dofs.npy', 'GP_FACES.npy'):
            if not (pd / fn).exists():
                os.symlink(Path(body_dir) / case / fn, pd / fn)
        np.save(pd / 'BOX_NODES.npy', C.port_node_ids)
        big = C.np_ > 20000
        C._free()                                                          # room for the Schur-mode factorization
        enc = SE.SchurEncoder(case, body_dir, precision='fp64', log=lambda s_: None, sub=case + '_portview', out_host=big)
        T, _ = enc.update()
        Tq = torch.cat([(T[r0:r0 + 4096].to(dev) @ q) for r0 in range(0, T.shape[0], 4096)])
        rec['vs_dense_schur_rel'] = float((Tq - Sq).norm() / Sq.norm())
        enc.free(); del T, enc; gc.collect(); torch.cuda.empty_cache()
    except Exception as e:
        rec['vs_dense_schur_error'] = repr(e)[:300]
    gc.collect(); torch.cuda.empty_cache()
    if C.sol_I is None:
        C.factor()
    # sensitivity: C(tau) = f^T S(tau)^+ f (force control, fixed topology); dC/dtau_c = -u^T K'_c u, u = E S^+ f
    sync(); t = time.perf_counter(); C.dmoments(); sync(); rec['dmoments_s'] = time.perf_counter() - t
    fs = feq[:, :2]
    qs = C.neumann(fs)
    s_an = C.sens(C.extend(qs))
    fd = []
    for c in (0, 5):
        h = 1e-4 * C.taus0[c]
        vals = []
        for sgn in (1, -1):
            tp = list(C.taus0); tp[c] += sgn * h
            C.assemble(tp); C.factor()
            vals.append((fs * C.neumann(fs)).sum(0))
        fd.append((vals[0] - vals[1]) / (2 * h))
        C.assemble(); C.factor()
    fd = torch.stack(fd)
    an = s_an[[0, 5]]
    rec['sens_analytic'] = an.tolist(); rec['sens_fd'] = fd.tolist()
    rec['sens_rel_err'] = float(((an - fd).abs() / fd.abs()).max())
    rec['compliance'] = (fs * qs).sum(0).tolist()
    rec['total_s'] = time.perf_counter() - t0
    rec['gpu_peak_gb'] = torch.cuda.max_memory_allocated() / 2 ** 30
    C._free()
    return rec


if __name__ == '__main__':
    body_dir, out = sys.argv[1], sys.argv[2]
    recs = []
    for case in sys.argv[3:]:
        torch.cuda.reset_peak_memory_stats()
        r = selftest(case, body_dir, log=lambda s_: print(s_, flush=True))
        print(json.dumps(r), flush=True)
        recs.append(r)
        gc.collect(); torch.cuda.empty_cache()
    Path(out).write_text(json.dumps(recs, indent=2))
    print('DONE', flush=True)
