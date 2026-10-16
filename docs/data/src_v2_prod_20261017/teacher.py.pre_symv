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
import fastidx as FI

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
dev, dt = torch.device(os.environ.get('OPL_DEV', 'cuda:0')), torch.float64   # OPL_DEV=cpu: local unit tests (no factorizations)
ROOT = Path('/root/autodl-tmp/CUTFEM_FRESH_GP_20260921')
# Extra packet roots (colon separated, e.g. the expansion packets S3/packets): a case missing from ROOT/packets is looked up
# there. Unset (the default): every lookup is ROOT/packets/<case>, as before.
PACKETS_EXTRA = [Path(p_) for p_ in os.environ.get('OPL_PACKETS_EXTRA', '').split(':') if p_]


def packet_dir(case):
    d = ROOT / 'packets' / case
    if d.exists():
        return d
    for r in PACKETS_EXTRA:
        if (r / case).exists():
            return r / case
    return d


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


class GhostFaces:
    """Matrix-free unit ghost-penalty product G x = sum_f P_f^T B_d(f)^T B_d(f) P_f x from the face list (GP_FACES.npy) and
    the three fixed face templates B_d (GP_TEMPLATES_n<n>.npz, 54 x 135): the same sum as box_encode.ghost_faces_gpu,
    which assembles it (reassociated; no stored matrix). Face DOFs int32, faces grouped by template."""

    def __init__(self, C, dtype, chunk=2048, like=None):
        self.chunk, self.dtype = chunk, dtype
        if like is not None:                                                    # same faces, other precision
            self.idx, self.faces = like.idx, like.faces
            self.B = [B.to(dtype) for B in like.B]
            return
        d = Path(C._body_dir) / C.case
        faces = np.load(d / 'GP_FACES.npy').astype(np.int64)
        tpl = np.load(Path(C._body_dir) / f'GP_TEMPLATES_n{C.n}.npz')
        offs = tpl['offsets'].astype(np.int64)
        g = 2 * C.cells[faces[:, 0]][:, None, :] + offs[faces[:, 2]]
        M = 2 * C.n + 1
        local = np.searchsorted(C.nodes, (g[..., 0] * M + g[..., 1]) * M + g[..., 2])
        if not np.array_equal(C.nodes[local], (g[..., 0] * M + g[..., 1]) * M + g[..., 2]):
            raise ValueError('GHOST_FACE_NODE_MISSING')
        dofs = (3 * local[:, :, None] + np.arange(3)).reshape(len(faces), -1)
        self.idx = [torch.as_tensor(dofs[faces[:, 2] == k], dtype=torch.int32, device=dev) for k in range(len(offs))]
        self.B = [torch.as_tensor(tpl['canonical'][k], dtype=dtype, device=dev) for k in range(len(offs))]
        self.faces = len(faces)

    def matvec(self, x, y):
        """y += G x (x, y: nb x b, self.dtype)."""
        b = x.shape[1]
        if FI.ON:                                                               # one block per face template
            for idx, B in zip(self.idx, self.B):
                X = FI.rows(x, idx).permute(1, 0, 2).reshape(idx.shape[1], -1)
                Y = B.t() @ (B @ X)
                y.index_add_(0, idx.reshape(-1), Y.reshape(idx.shape[1], len(idx), b).permute(1, 0, 2).reshape(-1, b))
            return y
        for idx, B in zip(self.idx, self.B):
            for lo in range(0, len(idx), self.chunk):
                ii = idx[lo:lo + self.chunk].long()
                X = FI.rows(x, ii).permute(1, 0, 2).reshape(ii.shape[1], -1)                 # 135 x (faces * b)
                Y = B.t() @ (B @ X)
                y.index_add_(0, ii.reshape(-1), Y.reshape(ii.shape[1], len(ii), b).permute(1, 0, 2).reshape(-1, b))
        return y


class _Kfp32:
    """K x in float32 (element matrices and the symmetric ghost-penalty part, int32 indices) for the equilibrium correction's
    smoothing and coarse residuals (teacher.Cell.lean(correction_fp32=True)); input and output keep their dtype. The
    condensed product F^T K F and all energies keep the float64 K.
    ke_moments: element matrices recomputed per chunk from float32 moments (M T in float32) instead of stored;
    ghost_faces: ghost part by face templates (GhostFaces, float32) instead of the stored CSR."""

    def __init__(self, C, ke_moments=False, ghost_faces=False, gf_like=None):
        f32 = torch.float32
        self.C, self.ke_moments, self.gf = C, ke_moments, None
        self._ke_tmp, self._hold = None, 0
        if ke_moments:
            self.Ke, self.M32, self.Tf32 = None, C.M.to(f32), C.Tm.reshape(125, 81 * 81).to(f32)
        else:
            self.Ke = C.Ke.to(f32)
        if ghost_faces:
            self.gf, self.G, self.G64vals = GhostFaces(C, f32, like=gf_like), None, None
            self.gamma = C.gamma
            return
        Gf = (C.G.to_sparse_coo() + C.Gt.to_sparse_coo()).coalesce()
        nb = C.nb
        Gf = (torch.sparse_coo_tensor(Gf.indices(), Gf.values(), Gf.shape)
              - torch.sparse_coo_tensor(torch.stack([torch.arange(nb, device=Gf.device)] * 2), C.dG, Gf.shape)).coalesce().to_sparse_csr()
        self.G = torch.sparse_csr_tensor(Gf.crow_indices().int(), Gf.col_indices().int(), Gf.values().to(f32), Gf.shape)
        self.G64vals = Gf.values()                                              # kept only until lean(deploy=True) takes it
        del Gf

    def __matmul__(self, x):
        C, f32 = self.C, torch.float32
        x32 = x.to(f32)
        if self.G64vals is not None and not getattr(C, '_deploy', False):
            self.G64vals = None                                                 # not needed outside deploy mode
        if self.gf is not None:
            y = self.gf.matvec(x32, torch.zeros_like(x32)).mul_(self.gamma)
        else:
            y = self.G @ x32
        ch = C._lean_chunk
        Xe = FI.rows(x32, C.dofs) if FI.ON else None                           # all element vectors at once
        for j, lo in enumerate(range(0, len(C.dofs), ch)):
            de = C.dofs[lo:lo + ch]
            if not self.ke_moments:
                Ke = self.Ke[lo:lo + ch]
            elif self._ke_tmp is not None:
                Ke = self._ke_tmp[j]
            else:
                Ke = (self.M32[lo:lo + ch] @ self.Tf32).reshape(-1, 81, 81)
            xe = Xe[lo:lo + ch] if Xe is not None else FI.rows(x32, de)
            y.index_add_(0, de.reshape(-1), torch.bmm(Ke, xe).reshape(-1, x.shape[1]))
        return y.to(x.dtype)

    def hold(self):
        """Context: with ke_moments and OPL_FASTIDX=1, the float32 element matrices are formed once (same chunks, same
        products) and reused by every product inside the block, then released."""
        import contextlib
        me = self

        @contextlib.contextmanager
        def ctx():
            if me.ke_moments and FI.ON and me._hold == 0:
                ch = me.C._lean_chunk
                me._ke_tmp = [(me.M32[lo:lo + ch] @ me.Tf32).reshape(-1, 81, 81) for lo in range(0, len(me.M32), ch)]
            me._hold += 1
            try:
                yield
            finally:
                me._hold -= 1
                if me._hold == 0:
                    me._ke_tmp = None
        return ctx()


class Cell:
    def __init__(self, case, body_dir, ports=('box', 'cut'), s=4, levels=1, log=print, deploy=False):
        import element_moments as EM
        import polyref_torch_fast as PT
        import box_encode as BX
        self.PT, self.s, self.levels, self.log = PT, s, levels, log
        t0 = time.perf_counter()
        ctx = json.loads((packet_dir(case) / 'FRESH_CONTEXT.json').read_text())
        self.case, self.n = case, int(ctx['n'])
        Emod = float(ctx['material']['E']); nu = float(ctx['material']['nu'])
        lam = Emod * nu / ((1 + nu) * (1 - 2 * nu)); mu = Emod / (2 * (1 + nu))
        cs = ctx['case']
        self.kind = cs.get('kind')
        self.surface = cs.get('surface', 'P') or 'P'
        self.taus0 = [float(Fraction(v)) for v in cs['tau_corners']]
        self.normal = None if cs.get('normal') is None else [float(Fraction(v)) for v in cs['normal']]
        self.offset = None if self.normal is None else float(Fraction(cs['offset']))
        self.gamma = float(json.loads((packet_dir(case) / 'SAMPLE.json').read_text())['gp']['gamma'])
        d = Path(body_dir) / case
        self._body_dir = body_dir
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
        if deploy:                                                              # deploy_from_scratch: no global pattern
            self._scratch = True
        else:
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
                if os.environ.get('OPL_GP_CACHE', '1') != '0':                 # OPL_GP_CACHE=0: rebuild every time, never
                    np.savez(gcache, keys=key_g.cpu().numpy(), vals=val_g.cpu().numpy())   # write (disk for thousands of cells)
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
        return torch.as_tensor(self.PT.cell_moments(self.cells, self.n, taus, self.normal, self.offset, self.s, levels=self.levels,
                                                    surface=self.surface),
                               dtype=dt, device=dev)

    def assemble(self, taus=None):
        if getattr(self, '_scratch', False):
            raise RuntimeError('DEPLOY_CELL_USE_assemble_deploy')
        if getattr(self, '_lean', False):
            raise RuntimeError('LEAN_CELL_HAS_NO_ASSEMBLY_MAPS')
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
        """Symmetric product K x = U x + U^T x - diag(K) x from the upper CSR (and its transpose); after lean(), element by
        element plus the ghost-penalty part."""
        if getattr(self, '_lean', False):
            return self._Kx_lean(x)
        return self.U @ x + self.Ut @ x - self.dK[:, None] * x

    def lean(self, keep_kpp=True, chunk=4096, correction_fp32=False, deploy=False, ke_moments=False, ghost_faces=False):
        """Deployment storage (called explicitly; default off). K x = sum_e P_e^T K_e P_e x + G x with the element matrices
        K_e = sum_m M_em T_m (fp64, E x 81 x 81) and the ghost-penalty part G = gamma * (unit ghost matrix) as its own upper
        CSR and transpose. The assembled upper CSR, its transpose, the ghost baseline and the assembly index maps are released;
        K_PP's upper triplets are cached first for lattice preconditioners (lat_multi.from_teacher). factor() and assemble()
        are unavailable afterwards. The arithmetic of K x is the same sum, reassociated.
        ke_moments (with correction_fp32 / deploy): the float32 correction stiffness recomputes its element matrices from the
        moments per chunk (no stored float32 K_e). ghost_faces (with deploy): the ghost part of both the float32 correction
        stiffness and the float64 K by face templates (GhostFaces), no stored ghost matrix."""
        if getattr(self, '_lean', False):
            return self
        d = self.vals.device
        if keep_kpp:
            pm = torch.zeros(self.nb, dtype=torch.bool, device=d); pm[self.P.to(d)] = True
            pnew = torch.full((self.nb,), -1, dtype=torch.long, device=d); pnew[self.P.to(d)] = torch.arange(self.np_, device=d)
            ru, cu = self.ru.long(), self.cu.long()
            sel = torch.nonzero(pm[ru] & pm[cu] & (self.vals != 0)).squeeze(1)
            self._kpp_cache = (pnew[ru[sel]], pnew[cu[sel]], self.vals[sel].clone())
            del pm, pnew, ru, cu, sel
        faces_only = deploy and ghost_faces                                     # no stored ghost matrix / K_e at any point
        nb = self.nb
        if faces_only:
            self.G = self.Gt = self.dG = None
            self.lean_info = dict(ghost_nnz_upper=int(torch.count_nonzero(self.base)), elements=int(len(self.cells)))
        else:
            g = torch.nonzero(self.base).squeeze(1)
            gr, gcl, gv = self.ru[g].long(), self.cu[g].long(), self.base[g].clone()
            del g
            self.G = torch.sparse_coo_tensor(torch.stack([gr, gcl]), gv, (nb, nb)).coalesce().to_sparse_csr()
            self.Gt = torch.sparse_coo_tensor(torch.stack([gcl, gr]), gv, (nb, nb)).coalesce().to_sparse_csr()
            on = gr == gcl
            self.dG = torch.zeros(nb, dtype=dt, device=d).index_add_(0, gr[on], gv[on])
            self.lean_info = dict(ghost_nnz_upper=int(len(gv)), elements=int(len(self.cells)))
            del gr, gcl, gv, on
        if faces_only and ke_moments:
            self.Ke = None
        else:
            self.Ke = torch.empty((len(self.M), 81, 81), dtype=dt, device=d)
            Tf = self.Tm.reshape(125, 81 * 81)
            for lo in range(0, len(self.M), chunk):
                self.Ke[lo:lo + chunk] = (self.M[lo:lo + chunk] @ Tf).reshape(-1, 81, 81)
        self._lean_chunk = chunk
        self.U = self.Ut = None
        self.vals = self.base = self.ru = self.cu = self.tperm = self.col_t = self.crow_t = self.pos_e = None
        self._lean = True
        if faces_only:
            self.G64, self.GF64 = None, GhostFaces(self, dt)
        if correction_fp32 or deploy:                                           # trainlib uses C._Kc inside the correction only
            self._Kc = _Kfp32(self, ke_moments=ke_moments, ghost_faces=faces_only, gf_like=getattr(self, 'GF64', None))
        if faces_only:
            self.G = self.Gt = self.dG = None
            self.Ke = None
            self._deploy = True
        elif deploy:
            # deployment memory: one symmetric float64 ghost matrix sharing the float32 copy's int32 indices, element
            # matrices recomputed from the moments for every float64 product (K_e = M_e T), no stored float64 K_e
            g32 = self._Kc.G
            vals64 = self._Kc.G64vals
            self.G64 = torch.sparse_csr_tensor(g32.crow_indices(), g32.col_indices(), vals64, g32.shape)
            self._Kc.G64vals = None
            self.G = self.Gt = self.dG = None
            self.Ke = None
            self._deploy = True
        gc.collect(); torch.cuda.empty_cache()
        return self

    def assemble_deploy(self, taus=None, chunk=4096, ke_moments=True):
        """Deployment state directly from the moments (Cell(..., deploy=True); no global pattern or assembled matrix at any
        point): the state of lean(deploy=True, ke_moments, ghost_faces=True) with K_PP's upper triplets (the same entries as
        the assembled upper CSR restricted to port pairs, nonzeros only), the diagonal of K and the node 3 x 3 diagonal blocks
        (diag3, for bench_deploy.netdata) summed from element and face blocks (same sums, reassociated)."""
        if not getattr(self, '_scratch', False):
            raise RuntimeError('assemble_deploy needs Cell(..., deploy=True)')
        taus = self.taus0 if taus is None else taus
        self.taus = list(taus)
        self._free()
        for k in ('_cV', '_cL', '_c_space', '_tail_bounds', '_cL32', '_cV32', '_cV32t', '_cL32_keep', '_Kc', 'GF64', 'M', 'dM',
                  '_kpp_cache', 'diag3'):                                       # state of the previous design (if any)
            setattr(self, k, None)
        gc.collect(); torch.cuda.empty_cache()
        self.M = self.moments(taus)
        nb, N, E = self.nb, len(self.nodes), len(self.M)
        self.GF64 = GF = GhostFaces(self, dt)
        # node diagonal blocks and diag(K)
        T5 = self.Tm.reshape(125, 27, 3, 27, 3)
        Td = torch.diagonal(T5, dim1=1, dim2=3).permute(0, 3, 1, 2).reshape(125, 27 * 9)      # (125, 27 * 3 * 3)
        nodes_e = self.dofs[:, ::3] // 3
        d3 = torch.zeros((N, 9), dtype=dt, device=dev)
        for lo in range(0, E, chunk):
            d3.index_add_(0, nodes_e[lo:lo + chunk].reshape(-1), (self.M[lo:lo + chunk] @ Td).reshape(-1, 9))
        for idx, B in zip(GF.idx, GF.B):
            BtB = B.t() @ B
            bd = torch.diagonal(BtB.reshape(45, 3, 45, 3), dim1=0, dim2=2).permute(2, 0, 1).reshape(45, 9)
            nf = idx[:, ::3].long() // 3
            d3.index_add_(0, nf.reshape(-1), (self.gamma * bd).repeat(len(idx), 1))
        self.diag3 = d3.reshape(N, 3, 3)
        self.dK = torch.diagonal(self.diag3, dim1=1, dim2=2).reshape(-1).clone()
        # K_PP upper triplets (port numbering; P sorted, so upper stays upper)
        pm = torch.zeros(nb, dtype=torch.bool, device=dev); pm[self.P] = True
        pnew = torch.full((nb,), -1, dtype=torch.long, device=dev); pnew[self.P] = torch.arange(self.np_, device=dev)
        rows, cols, vals = [], [], []
        Tf = self.Tm.reshape(125, 81 * 81)
        el = torch.nonzero(pm[self.dofs].sum(1) >= 1).squeeze(1)
        for lo in range(0, len(el), chunk):
            e = el[lo:lo + chunk]; de = self.dofs[e]
            Ke = (self.M[e] @ Tf).reshape(-1, 81, 81)
            r, c = de[:, :, None].expand_as(Ke), de[:, None, :].expand_as(Ke)
            sel = pm[r] & pm[c] & (r <= c)
            rows.append(pnew[r[sel]]); cols.append(pnew[c[sel]]); vals.append(Ke[sel])
            del Ke, r, c, sel
        for idx, B in zip(GF.idx, GF.B):
            BtB = self.gamma * (B.t() @ B)
            ii = idx.long()
            ii = ii[pm[ii].sum(1) >= 1]
            for lo in range(0, len(ii), 1024):
                de = ii[lo:lo + 1024]
                r, c = de[:, :, None].expand(-1, 135, 135), de[:, None, :].expand(-1, 135, 135)
                sel = pm[r] & pm[c] & (r <= c)
                rows.append(pnew[r[sel]]); cols.append(pnew[c[sel]]); vals.append(BtB.expand(len(de), -1, -1)[sel])
                del r, c, sel
        kp = torch.sparse_coo_tensor(torch.stack([torch.cat(rows), torch.cat(cols)]), torch.cat(vals), (self.np_, self.np_)).coalesce()
        del rows, cols, vals
        ki, kv = kp.indices(), kp.values()
        nz = kv != 0
        self._kpp_cache = (ki[0][nz], ki[1][nz], kv[nz])
        del kp, ki, kv, nz
        # deployment state (as lean(deploy=True, ke_moments, ghost_faces=True))
        self._lean_chunk = chunk
        self.U = self.Ut = self.vals = self.base = None
        self.G = self.Gt = self.dG = self.G64 = None
        self.lean_info = dict(ghost_faces=GF.faces, elements=int(E))
        if ke_moments:
            self.Ke = None
        else:
            self.Ke = torch.empty((E, 81, 81), dtype=dt, device=dev)
            for lo in range(0, E, chunk):
                self.Ke[lo:lo + chunk] = (self.M[lo:lo + chunk] @ Tf).reshape(-1, 81, 81)
        self._lean = self._deploy = True
        self._Kc = _Kfp32(self, ke_moments=ke_moments, ghost_faces=True, gf_like=GF)
        self.Ke = None
        self.K = self
        gc.collect(); torch.cuda.empty_cache()
        return self

    def _Kx_lean(self, x):
        if getattr(self, '_deploy', False):
            y = self.G64 @ x if self.G64 is not None else self.GF64.matvec(x, torch.zeros_like(x)).mul_(self.gamma)
            Tf = self.Tm.reshape(125, 81 * 81)
            held = getattr(self, '_ke64_tmp', None)
            for j, lo in enumerate(range(0, len(self.M), self._lean_chunk)):
                de = self.dofs[lo:lo + self._lean_chunk]
                Ke = held[j] if held is not None else self._ke64(lo, lo + self._lean_chunk)
                y.index_add_(0, de.reshape(-1), torch.bmm(Ke, FI.rows(x, de)).reshape(-1, x.shape[1]))
            return y
        y = self.G @ x + self.Gt @ x - self.dG[:, None] * x
        for lo in range(0, len(self.Ke), self._lean_chunk):
            de = self.dofs[lo:lo + self._lean_chunk]
            ye = torch.bmm(self.Ke[lo:lo + self._lean_chunk], FI.rows(x, de))
            y.index_add_(0, de.reshape(-1), ye.reshape(-1, x.shape[1]))
        return y

    def _ke64(self, lo, hi):
        """float64 element matrices K_e = M_e T of rows lo:hi. OPL_FASTIDX=1: T_m is exactly symmetric, so only the upper
        triangle (3321 of 6561 entries) is formed by the GEMM and mirrored (half the float64 flops; same products)."""
        if not FI.ON:
            return (self.M[lo:hi] @ self.Tm.reshape(125, 81 * 81)).reshape(-1, 81, 81)
        sym = getattr(self, '_sym_idx', None)
        if sym is None or sym.device != self.M.device:
            iu = torch.triu_indices(81, 81, device=self.M.device)
            pos = torch.full((81, 81), -1, dtype=torch.long, device=self.M.device)
            pos[iu[0], iu[1]] = torch.arange(iu.shape[1], device=self.M.device)
            pos = torch.where(pos >= 0, pos, pos.t())
            self._sym_idx = sym = pos.reshape(-1)
            self._Tup = self.Tm[:, iu[0], iu[1]].contiguous()
        return (self.M[lo:hi] @ self._Tup)[:, sym].reshape(-1, 81, 81)

    def hold64(self):
        """Context: in deployment mode with OPL_FASTIDX=1, the float64 element matrices K_e = M_e T are formed once (same
        chunks, same products) and reused by every float64 K product inside the block (setup loops: coarse probing,
        power iteration), then released."""
        import contextlib
        me = self

        @contextlib.contextmanager
        def ctx():
            own = FI.ON and getattr(me, '_deploy', False) and getattr(me, '_ke64_tmp', None) is None
            if own:
                Tf, ch = me.Tm.reshape(125, 81 * 81), me._lean_chunk
                me._ke64_tmp = [me._ke64(lo, lo + ch) for lo in range(0, len(me.M), ch)]
            try:
                yield
            finally:
                if own:
                    me._ke64_tmp = None
        return ctx()

    def __matmul__(self, x):                                                   # so that cell.K @ x keeps working
        return self.Kx(x)

    def _free(self):
        for s in (self.sol_I, self.sol_N):
            if s is not None:
                s.free()
        self.sol_I = self.sol_N = None
        gc.collect(); torch.cuda.empty_cache()

    # ---------------------------------------------------------------- factorizations
    def factor(self, neumann=True, fp32=False, interior=True, fp32_neumann=False):
        if getattr(self, '_lean', False):
            raise RuntimeError('LEAN_CELL_HAS_NO_ASSEMBLED_MATRIX')
        self._free()
        self.fp32 = fp32
        st = {}
        sync(); t = time.perf_counter()
        if not interior:                                                      # Neumann factor only
            ru, cu = self.ru.long(), self.cu.long()
            return self._factor_neumann(ru, cu, st, t, fp32_neumann)
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
            return self._factor_neumann(ru, cu, st, t, fp32_neumann)
        return st

    def _factor_neumann(self, ru, cu, st, t, fp32_neumann=False):
        if True:
            self.pin = self._pick_pins()
            pin = torch.zeros(self.nb, dtype=torch.bool, device=dev); pin[self.pin] = True
            v = self.vals.clone()
            v[pin[ru] | pin[cu]] = 0
            v[self.diag[self.pin]] = 1.0
            sN = 1 / torch.sqrt(v[self.diag])
            self.sN = sN
            del ru, cu
            self.sol_N = SPDSolver(self.crow.int(), self.cu, (v * sN[self.ru.long()] * sN[self.cu.long()]).contiguous(), self.nb,
                                   fdt=torch.float32 if fp32_neumann else None)
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
            ue = FI.rows(u, self.dofs[lo:lo + chunk])                              # c x 81 x k
            z = torch.einsum('eik,mij->emjk', ue, self.Tm)
            g[lo:lo + chunk] = torch.einsum('emjk,ejk->emk', z, ue)
            del z
        return g

    def sens(self, u):
        """-u^T (dK/dtau_c) u, (8, k)."""
        g = self.energy_density(u)
        return -torch.einsum('cem,emk->ck', self.dM, g)

    def sens2(self, u, chunk=1024):
        """Same as sens, reassociated for many fields: the 8 element derivatives A_ce = sum_m dM_cem Tm_m (81 x 81) are
        formed per element chunk, then s_ck = -sum_e u_ek^T A_ce u_ek (8 instead of 125 contractions per field)."""
        s = torch.zeros((8, u.shape[1]), dtype=dt, device=dev)
        for lo in range(0, len(self.cells), chunk):
            A = torch.einsum('cem,mij->ceij', self.dM[:, lo:lo + chunk], self.Tm)       # 8 x c x 81 x 81
            ue = FI.rows(u, self.dofs[lo:lo + chunk])                                            # c x 81 x k
            z = torch.einsum('ceij,ejk->ceik', A, ue)
            s -= (z * ue[None]).sum((1, 2))
            del A, z
        return s


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
