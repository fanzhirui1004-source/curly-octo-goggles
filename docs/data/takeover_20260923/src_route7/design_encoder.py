"""Route 7: a persistent per-cell encoder for design loops (exact box operator, no trace coordinates).

Setup once per topology (active cells, element DOF map and ghost faces from fast_prep3):
  - the upper-triangle sparsity pattern of K = K_body + gamma K_ghost as sorted unique keys,
  - for every element upper entry its position in that pattern (scatter map),
  - the ghost contribution as a constant value vector (the ghost matrix depends only on the faces),
  - the split into A = K_ii (upper, CSR, cuDSS), C = K_ib and D = K_bb with fixed index arrays,
  - the cuDSS plan (on the first factorization).
Per design update (new thickness corners, same topology): moments -> 125 x 3321 element upper entries -> one
index_add into the value vector -> Jacobi scaling -> in-place values -> cuDSS refactorization (fp32 or fp64).
Queries T q = D q + C^T z, z = -A^-1 C q; with fp32 factors, z is refined twice in fp64.
"""
import json, sys, time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
dev, dt = torch.device('cuda:0'), torch.float64
ROOT = Path('/root/autodl-tmp/CUTFEM_FRESH_GP_20260921')


def sync():
    torch.cuda.synchronize()


class CellEncoder:
    def __init__(self, case, body_dir, precision='fp32', threads=16, s=4, levels=1, log=print, sub=None):
        import element_moments as EM
        import polyref_torch_fast as PT        # closed-form full sub-cubes, per-tetrahedron reduction
        from fractions import Fraction
        self.PT, self.s, self.levels, self.precision, self.threads, self.log = PT, s, levels, precision, threads, log
        self.T = {}
        t0 = time.perf_counter()
        ctx = json.loads((ROOT / 'packets' / case / 'FRESH_CONTEXT.json').read_text())
        self.case, self.n = case, int(ctx['n'])
        Emod = float(ctx['material']['E']); nu = float(ctx['material']['nu'])
        lam = Emod * nu / ((1 + nu) * (1 - 2 * nu)); mu = Emod / (2 * (1 + nu))
        self.taus0 = [float(Fraction(v)) for v in ctx['case']['tau_corners']]
        self.normal = None if ctx['case'].get('normal') is None else [float(v) for v in ctx['case']['normal']]
        self.offset = None if self.normal is None else float(ctx['case']['offset'])
        self.gamma = float(json.loads((ROOT / 'packets' / case / 'SAMPLE.json').read_text())['gp']['gamma'])
        d = Path(body_dir) / (sub or case)
        nodes = np.load(d / 'NODES.npy'); cells = np.load(d / 'CELL_INDICES.npy'); dofs = np.load(d / 'dofs.npy')
        faces = np.load(d / 'GP_FACES.npy')
        self.cells = cells
        arr = {'NODES.npy': nodes, 'dofs.npy': dofs, 'CELL_INDICES.npy': cells}
        xi = EM.local_coordinates(case, self.n, arr)
        keys = np.unique(xi.reshape(len(xi), -1), axis=0)
        if len(keys) != 1:
            raise ValueError('ONE_NODE_ORDERING_EXPECTED')
        Tm = torch.tensor(EM.pattern_operators(keys[0].reshape(27, 3), lam, mu, self.n)[1], dtype=dt, device=dev)
        iu = torch.triu_indices(81, 81, device=dev)
        self.Tm_up = Tm.reshape(125, 81, 81)[:, iu[0], iu[1]].contiguous()           # 125 x 3321
        nb = 3 * len(nodes); self.nb = nb
        dofs_t = torch.as_tensor(dofs, dtype=torch.long, device=dev)
        r, c = dofs_t[:, iu[0]], dofs_t[:, iu[1]]
        key_e = torch.minimum(r, c) * nb + torch.maximum(r, c)                         # E x 3321
        # ghost upper entries (unit penalty), coalesced in groups
        import box_encode as BX
        G = BX.ghost_faces_gpu(body_dir, sub or case, self.n, nb)                       # full symmetric COO
        gi, gv = G.indices(), G.values()
        up = gi[0] <= gi[1]
        key_g, val_g = gi[0][up] * nb + gi[1][up], gv[up]
        del G, gi, gv
        U = torch.unique(torch.cat([key_e.reshape(-1), key_g]))                          # sorted pattern
        self.pos_e = torch.searchsorted(U, key_e.reshape(-1))
        self.base = torch.zeros(len(U), dtype=dt, device=dev)
        self.base.index_add_(0, torch.searchsorted(U, key_g), self.gamma * val_g)
        del key_e, key_g, val_g
        ru, cu = U // nb, U % nb
        del U
        onbox = np.isin(nodes, np.load(d / 'BOX_NODES.npy'))
        self.box_node_ids = nodes[onbox]
        box = torch.as_tensor(np.repeat(onbox, 3), device=dev)
        bi = torch.nonzero(box).squeeze(1); ii = torch.nonzero(~box).squeeze(1)
        new = torch.empty(nb, dtype=torch.long, device=dev)
        new[bi] = torch.arange(len(bi), device=dev); new[ii] = torch.arange(len(ii), device=dev)
        self.ni, self.nbx = len(ii), len(bi)
        br, bc = box[ru], box[cu]
        selA = torch.nonzero(~br & ~bc).squeeze(1)                                      # interior-interior, upper, CSR order
        self.idxA = selA; self.rowA, self.colA = new[ru[selA]], new[cu[selA]]
        self.diagA = torch.nonzero(self.rowA == self.colA).squeeze(1)
        if len(self.diagA) != self.ni:
            raise ValueError('A_DIAGONAL_INCOMPLETE')
        counts = torch.bincount(self.rowA, minlength=self.ni)
        self.crowA = torch.cat([torch.zeros(1, dtype=torch.long, device=dev), torch.cumsum(counts, 0)]).int()
        self.colA32 = self.colA.int()
        # refinement residuals use the fp64 upper CSR: A x = U x + U^T x - diag(A) x
        # C = K_ib: upper entries with exactly one box index, oriented (interior row, box col)
        selC1 = torch.nonzero(~br & bc).squeeze(1); selC2 = torch.nonzero(br & ~bc).squeeze(1)
        self.idxC = torch.cat([selC1, selC2])
        self.C_rows = torch.cat([new[ru[selC1]], new[cu[selC2]]]); self.C_cols = torch.cat([new[cu[selC1]], new[ru[selC2]]])
        selD = torch.nonzero(br & bc).squeeze(1)
        self.idxD = selD; self.D_r, self.D_c = new[ru[selD]], new[cu[selD]]
        self.D_off = self.D_r != self.D_c
        self.solver = None
        sync(); self.T['setup'] = time.perf_counter() - t0
        log(json.dumps(dict(event='SETUP', case=case, seconds=self.T['setup'], nnz_upper=int(len(self.base)),
                            interior=self.ni, box=self.nbx, elements=int(len(cells)), faces=int(len(faces)))))

    def update(self, taus=None):
        """New thickness corners (same topology). Returns stage times."""
        from nvmath.sparse.advanced import DirectSolver, DirectSolverOptions, DirectSolverMatrixType, DirectSolverMatrixViewType
        import os
        st = {}
        taus = self.taus0 if taus is None else taus
        sync(); t = time.perf_counter()
        M = torch.as_tensor(self.PT.cell_moments(self.cells, self.n, taus, self.normal, self.offset, self.s, levels=self.levels),
                            dtype=dt, device=dev)
        sync(); st['moments'] = time.perf_counter() - t; t = time.perf_counter()
        vals = self.base.clone()
        for lo in range(0, len(M), 4096):
            ke = M[lo:lo + 4096] @ self.Tm_up                                           # chunk x 3321
            vals.index_add_(0, self.pos_e[lo * 3321:(lo + len(ke)) * 3321], ke.reshape(-1))
        del M
        a = vals[self.idxA]
        s = 1 / torch.sqrt(a[self.diagA])
        scaled = a * s[self.rowA] * s[self.colA]
        self.s_scale = s
        if self.precision == 'fp32':
            self.A_up = torch.sparse_csr_tensor(self.crowA.long(), self.colA, a, size=(self.ni, self.ni))
            self.A_upT = self.A_up.t().to_sparse_csr()
            self.A_diag = a[self.diagA]
        cv = vals[self.idxC]
        self.C = torch.sparse_coo_tensor(torch.stack([self.C_rows, self.C_cols]), cv, (self.ni, self.nbx)).coalesce()
        self.Ct = self.C.t().coalesce()
        dv = vals[self.idxD]
        self.D = torch.sparse_coo_tensor(torch.stack([torch.cat([self.D_r, self.D_c[self.D_off]]), torch.cat([self.D_c, self.D_r[self.D_off]])]),
                                         torch.cat([dv, dv[self.D_off]]), (self.nbx, self.nbx)).coalesce()
        del vals
        sync(); st['assemble'] = time.perf_counter() - t; t = time.perf_counter()
        fdt = torch.float32 if self.precision == 'fp32' else dt
        if self.solver is None:
            self.U = torch.sparse_csr_tensor(self.crowA, self.colA32, scaled.to(fdt), size=(self.ni, self.ni))
            lib = os.environ.get('CUDSS_MT')
            opts = DirectSolverOptions(sparse_system_type=DirectSolverMatrixType.SPD, sparse_system_view=DirectSolverMatrixViewType.UPPER,
                                       **(dict(multithreading_lib=lib) if lib else {}))
            self.w = 8
            self.b = torch.zeros((self.w, self.ni), dtype=fdt, device=dev).T
            self.solver = DirectSolver(self.U, self.b, options=opts)
            if lib:
                self.solver.plan_config.host_nthreads = self.threads
            torch.cuda.empty_cache()
            self.solver.plan()
            sync(); st['plan'] = time.perf_counter() - t; t = time.perf_counter()
        else:
            self.U.values().copy_(scaled.to(fdt))
        self.solver.factorize()
        sync(); st['factorize'] = time.perf_counter() - t
        return st

    def _solve_scaled(self, r):
        """(s A s) y = r in the factor's precision, panels of width w."""
        fdt = torch.float32 if self.precision == 'fp32' else dt
        out = torch.empty_like(r)
        for c0 in range(0, r.shape[1], self.w):
            k = min(self.w, r.shape[1] - c0)
            b = torch.zeros((self.w, self.ni), dtype=fdt, device=dev).T
            b[:, :k] = r[:, c0:c0 + k].to(fdt)
            self.solver.reset_operands(b=b)
            out[:, c0:c0 + k] = self.solver.solve()[:, :k].to(dt)
        return out

    def solve(self, rhs, refine=None):
        s = self.s_scale[:, None]
        z = s * self._solve_scaled(s * rhs)
        steps = (2 if self.precision == 'fp32' else 0) if refine is None else refine
        for _ in range(steps):
            Az = self.A_up @ z + self.A_upT @ z - self.A_diag[:, None] * z
            z = z + s * self._solve_scaled(s * (rhs - Az))
        return z

    def apply(self, q):
        """T q = D q + C^T z, z = -A^-1 C q (box DOFs in fast_prep3 node order, node-major xyz)."""
        z = -self.solve(torch.sparse.mm(self.C, q))
        return torch.sparse.mm(self.D, q) + torch.sparse.mm(self.Ct, z)


def main(case, body_dir, out):
    rec = dict(case=case)
    q = None
    for precision in ('fp64', 'fp32'):
        enc = CellEncoder(case, body_dir, precision=precision)
        st0 = enc.update()                                                              # first: includes plan
        if q is None:
            g = torch.Generator(device=dev).manual_seed(0)
            q = torch.randn((enc.nbx, 16), dtype=dt, device=dev, generator=g)
        y0 = enc.apply(q)
        steps = []
        for eps in (1e-6, 1e-4):
            st = enc.update([t * (1 + eps) for t in enc.taus0])
            sync(); t = time.perf_counter(); enc.apply(q[:, :8]); sync(); st['query8'] = time.perf_counter() - t
            steps.append(st)
        enc.update()                                                                    # back to the original design
        y1 = enc.apply(q)
        if precision == 'fp64':
            import box_encode as BX
            from types import SimpleNamespace
            with torch.no_grad():
                rec['validation_vs_packet'] = BX.validate(SimpleNamespace(case=case), lambda x: enc.apply(x), enc.box_node_ids, enc.n)
            print(json.dumps(dict(validation=rec['validation_vs_packet'])), flush=True)
        rec[precision] = dict(setup=enc.T['setup'], first=st0, steps=steps,
                              back_to_start_rel=float((y1 - y0).norm() / y0.norm()), y0=y0)
        print(json.dumps(dict(precision=precision, setup=round(enc.T['setup'], 2), first={k: round(v, 3) for k, v in st0.items()},
                              steps=[{k: round(v, 3) for k, v in s_.items()} for s_ in steps],
                              back_to_start=rec[precision]['back_to_start_rel'])), flush=True)
        del enc; torch.cuda.empty_cache()
    rec['fp32_vs_fp64'] = float((rec['fp32']['y0'] - rec['fp64']['y0']).norm() / rec['fp64']['y0'].norm())
    for p in ('fp64', 'fp32'):
        rec[p].pop('y0')
    print(json.dumps(dict(fp32_vs_fp64=rec['fp32_vs_fp64'])), flush=True)
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / f'DESIGN_{case}.json').write_text(json.dumps(rec, indent=2, default=float))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])
