"""Deployable preconditioners for the multi-cell port lattice (lat_multi.MultiLattice): A = sum_c R_c^T S_c R_c on the free
port DOFs, S_c applied through the cell operators only (exact: teacher interior factor; learned: FastNet).
Allowed ingredients: operator applications, the assembled port stiffness K_PP (sum of the cells' K restricted to their port
DOFs, interior clamped; K_PP >= S_c in the Loewner order, so K_PP^-1 A has its spectrum in (0, 1]), sparse factorisations of
lattice-level port matrices, small dense coarse solves. No interior factor of a cell's CutFEM system is used here.

Fine levels (B):   'jac'  inverse diagonal of the assembled K_PP
                   'kpp'  sparse Cholesky of the assembled K_PP (cuDSS through teacher.SPDSolver on CUDA; CHOLMOD if
                          scikit-sparse is importable; else scipy SuperLU in symmetric mode on the host)
Coarse spaces (Z, columns = lattice vectors on the free DOFs):
  'q1t' / 'q1r' / 'q1a'  macro-Q1 partition of unity: hat_v(x) = prod_d max(0, 1 - |x_d - v_d|) over the lattice vertices v
                         (cell corners, cell units) times 3 translations / + 3 rotations about v / + 6 symmetric affine
                         strains about v. The hats are ONE global continuous field evaluated at each DOF's absolute
                         position, so shared port nodes get one consistent value by construction (no averaging), and the
                         hats sum to 1: span(q1t) contains every global affine displacement (rigid motions + 6 constant
                         strains), q1r adds per-vertex rotations (bending of slender lattices). Default: q1r.
  'pu6' / 'pu12'          per-cell rigid (6) / rigid + affine strain (12) modes about the cell centre, multiplied by the
                         multiplicity partition of unity w_c = 1 / (number of cells sharing the DOF) (Nicolaides / balancing
                         Neumann-Neumann style). Per-cell modes restricted to a cell disagree between the two cells on a
                         shared node, so they are not lattice vectors by themselves; with the weights the columns are
                         lattice vectors and their sum over cells reproduces every global rigid (pu6) / affine (pu12) field.
                         The weighted modes jump across cell faces (value 1/2 on a shared face, 0 in the next cell), which
                         costs energy: expected weaker than q1* on smooth lattice-scale modes.
  Columns with zero norm on the free DOFs are dropped; A_c = Z^T A Z (A Z by matmat_sparse: each cell applies its
  operator to the <= 8 * modes (q1) or 27 * modes (pu) columns it touches, batched over cells sharing an operator);
  A_c is Jacobi-scaled and eigendecomposed; eigenvalues below rcond * max are dropped (robust to dependent columns);
  W = Z T with W^T A W = I, so Q = Z A_c^+ Z^T = W W^T; A W is kept (no operator application per preconditioner call).
Preconditioners / solvers (spec strings for build()):
  'jacobi', 'kpp'            one-level, M = B
  'add:<fine>:<coarse>'      two-level additive, M = Q + B
  'bnn:<fine>:<coarse>'      hybrid (multiplicative, balancing Neumann-Neumann form), M = Q + (I - Q A) B (I - A Q), SPD;
                             with A W stored it costs one fine solve + 4 skinny products (no extra operator application)
  'defl:<fine>:<coarse>'     deflated PCG (A-DEF2): x0 = Q b, search directions (I - Q A) B r (Saad et al. 2000)
"""
import os
import time
import numpy as np
import torch
import gpuprof as GP

dt = torch.float64


def sync(dev):
    if torch.device(dev).type == 'cuda':
        torch.cuda.synchronize()


class _T:
    def __init__(self, dev):
        self.dev = dev

    def __enter__(self):
        sync(self.dev); self.t = time.perf_counter(); return self

    def __exit__(self, *a):
        sync(self.dev); self.s = time.perf_counter() - self.t


# ---------------------------------------------------------------------- sparse SPD factorisation (fine level)
SLIM = os.environ.get('OPL_V3_SLIM', '1') == '1'
KPP_FP32 = os.environ.get('OPL_V3_KPP_FP32', '0') == '1'                # category B: K_PP factor in float32 (half the memory)
KPP_CSR = os.environ.get('OPL_V3_KPP_CSR', '1') == '1'                  # src_v3: K_PP assembled in row blocks as CSR


def _has_nvmath():
    try:
        import nvmath  # noqa: F401
        return True
    except ImportError:
        return False


class SparseSPD:
    """Factor of an SPD matrix given by upper triplets on n unknowns (Jacobi scaled first).
    src_v3: rows / cols / vals may also be passed as one list (holder) that this constructor empties, so that the
    triplets are freed (and the allocator cache returned to the device) before cuDSS plans and factorises."""

    def __init__(self, rows, cols=None, vals=None, n=None, backend='auto'):
        crow = None
        if cols is None:                                                    # holder list [rows, cols, vals]
            if isinstance(rows[0], str) and rows[0] == 'csr':                # src_v3: ['csr', crow, cols, vals] (upper,
                _, crow, cols, vals = rows.pop(0), rows.pop(0), rows.pop(0), rows.pop(0)   # MultiLattice.assemble_kpp_csr)
                rows = None
            else:
                rows, cols, vals = rows.pop(0), rows.pop(0), rows.pop(0)
        self.n, self.dev = n, vals.device
        if crow is not None:
            if backend == 'cudss' or (backend == 'auto' and vals.is_cuda and _has_nvmath()):
                self._init_csr(crow, cols, vals, n)
                return
            rows = torch.repeat_interleave(torch.arange(n, device=vals.device), (crow[1:] - crow[:-1]).long())
            cols = cols.long()
        dg = torch.zeros(n, dtype=dt, device=vals.device); d = rows == cols
        dg[rows[d]] = vals[d]
        del d
        if not bool((dg > 0).all()):
            raise ValueError('KPP_DIAGONAL_NOT_POSITIVE')
        self.s = 1 / torch.sqrt(dg)
        sv = vals * self.s[rows] * self.s[cols]
        if backend == 'auto':
            backend = 'splu'
            if vals.is_cuda:
                try:
                    import nvmath  # noqa: F401
                    backend = 'cudss'
                except ImportError:
                    pass
            if backend == 'splu':
                try:
                    import sksparse.cholmod  # noqa: F401
                    backend = 'cholmod'
                except ImportError:
                    pass
        self.backend = backend
        if backend == 'cudss':
            import teacher as TE
            crow = torch.zeros(n + 1, dtype=torch.long, device=vals.device)
            crow[1:] = torch.cumsum(torch.bincount(rows, minlength=n), 0)
            crow, cols = crow.int(), cols.int()
            if SLIM:                                                        # src_v3: free the triplets and the cache
                import gc
                del rows, vals, dg
                gc.collect(); torch.cuda.empty_cache()
            fdt = torch.float32 if KPP_FP32 else None                          # category B (OPL_V3_KPP_FP32=1): float32
            import stream_ops as SO                                             # src_v3 S25: cells spilled to disk
            with SO.spill_guard(SO.SPILL_BPNZ * len(sv)) as self.spill:          # around the host peak (OPL_V3_SPILL)
                self.sol = TE.SPDSolver(crow, cols, sv.contiguous(), n, fdt=fdt)  # factor and solves (not bitwise)
            self.mem_est = getattr(self.sol, 'mem_est', None)
        else:
            import scipy.sparse as sp
            r, c, v = rows.cpu().numpy(), cols.cpu().numpy(), sv.cpu().numpy()
            off = r != c
            A = sp.csc_matrix((np.concatenate([v, v[off]]), (np.concatenate([r, c[off]]), np.concatenate([c, r[off]]))), shape=(n, n))
            if backend == 'cholmod':
                from sksparse.cholmod import cholesky
                self.fac = cholesky(A)
                self._solve = self.fac
            else:
                import scipy.sparse.linalg as spl
                self.fac = spl.splu(A, permc_spec='MMD_AT_PLUS_A', diag_pivot_thresh=0.0, options=dict(SymmetricMode=True))
                self._solve = self.fac.solve
            self.nnz_factor = int(self.fac.L.nnz + self.fac.U.nnz) if backend == 'splu' else None

    def _init_csr(self, crow, cols, vals, n):
        """src_v3 (OPL_V3_KPP_CSR): the cuDSS factor from the upper CSR (crow, cols int32; vals float64, owned: scaled in
        place). The same Jacobi scaling as the triplet path (vals * s[row] * s[col], in that order: bitwise the same
        values), done in blocks of rows so that no int64 row or column array of the whole matrix is formed."""
        import gc
        import teacher as TE
        self.backend = 'cudss'
        nnz = len(vals)
        first = crow[:-1].long()
        has = (crow[1:] > crow[:-1]) & (cols[first.clamp(max=max(nnz - 1, 0))].long() == torch.arange(n, device=vals.device))
        dg = torch.where(has, vals[first.clamp(max=max(nnz - 1, 0))], torch.zeros((), dtype=dt, device=vals.device))
        del first, has
        if not bool((dg > 0).all()):
            raise ValueError('KPP_DIAGONAL_NOT_POSITIVE')
        self.s = 1 / torch.sqrt(dg)
        del dg
        cro = crow.long()
        step = max(1, n // max(1, (nnz + 2 ** 25 - 1) // 2 ** 25))         # ~32M entries per block
        for a in range(0, n, step):
            b = min(n, a + step)
            lo, hi = int(cro[a]), int(cro[b])
            if hi <= lo:
                continue
            ridx = torch.repeat_interleave(torch.arange(a, b, device=vals.device), cro[a + 1:b + 1] - cro[a:b])
            seg = vals[lo:hi]
            seg.mul_(self.s[ridx]).mul_(self.s[cols[lo:hi].long()])
            del ridx, seg
        del cro
        sv = vals
        del vals
        if SLIM:
            gc.collect(); torch.cuda.empty_cache()
        fdt = torch.float32 if KPP_FP32 else None
        import stream_ops as SO
        with SO.spill_guard(SO.SPILL_BPNZ * nnz) as self.spill:
            self.sol = TE.SPDSolver(crow, cols, sv, n, fdt=fdt)
        self.mem_est = getattr(self.sol, 'mem_est', None)

    def __call__(self, R):
        x = self.s[:, None] * R
        if self.backend == 'cudss':
            y = self.sol.solve(x.contiguous())
        else:
            y = torch.as_tensor(np.asarray(self._solve(x.cpu().numpy())).reshape(x.shape), dtype=dt, device=R.device)
        return self.s[:, None] * y

    def free(self):
        if self.backend == 'cudss':
            self.sol.free()
        self.fac = self._solve = self.sol = None


# ---------------------------------------------------------------------- coarse spaces
def mode_values(dx, comp, kind):
    """Displacement modes about a point at DOFs with offsets dx (m x 3) and components comp (m,): (m x nm).
    kind: 't' translations (3), 'r' + rotations (6), 'a' + symmetric affine strains (12)."""
    m = len(comp)
    e = torch.nn.functional.one_hot(comp, 3).to(dt)                                    # m x 3
    cols = [e[:, 0], e[:, 1], e[:, 2]]
    if kind in ('r', 'a'):
        x, y, z = dx[:, 0], dx[:, 1], dx[:, 2]
        cols += [e[:, 2] * y - e[:, 1] * z,                                             # rot about x: (0, -z, y)
                 e[:, 0] * z - e[:, 2] * x,                                             # rot about y: (z, 0, -x)
                 e[:, 1] * x - e[:, 0] * y]                                             # rot about z: (-y, x, 0)
    if kind == 'a':
        cols += [e[:, 0] * x, e[:, 1] * y, e[:, 2] * z,                                  # e_xx, e_yy, e_zz
                 e[:, 0] * y + e[:, 1] * x, e[:, 1] * z + e[:, 2] * y, e[:, 0] * z + e[:, 2] * x]
    return torch.stack(cols, 1).reshape(m, -1)


def coarse_basis(lat, kind):
    """Z (free x nc) on the lattice's device and a description."""
    xyz, comp = lat.xyz, lat.comp
    dev = xyz.device
    if kind.startswith('q1'):
        mk = {'q1t': 't', 'q1r': 'r', 'q1a': 'a'}[kind]
        nm = {'t': 3, 'r': 6, 'a': 12}[mk]
        offs = np.asarray(lat.positions)
        verts = sorted({tuple(p + np.asarray(c)) for p in offs for c in np.ndindex(2, 2, 2)})
        Z = torch.zeros((lat.nfree, nm * len(verts)), dtype=dt, device=dev)            # preallocated (large lattices)
        for j, v in enumerate(verts):
            vv = torch.as_tensor(v, dtype=dt, device=dev)
            h = torch.clamp(1 - (xyz - vv).abs(), min=0).prod(1)
            sup = torch.nonzero(h > 0).squeeze(1)
            if len(sup):
                Z[sup, nm * j:nm * (j + 1)] = h[sup, None] * mode_values(xyz[sup] - vv, comp[sup], mk)
        desc = dict(vertices=len(verts))
    elif kind in ('pu6', 'pu12'):
        mk = 'r' if kind == 'pu6' else 'a'
        nm = 6 if mk == 'r' else 12
        Z = torch.zeros((lat.nfree, nm * len(lat.positions)), dtype=dt, device=dev)
        for i, p in enumerate(lat.positions):
            kp, fk = lat._keep[i]
            ctr = torch.as_tensor(np.asarray(p) + 0.5, dtype=dt, device=dev)
            Z[fk, nm * i:nm * (i + 1)] = mode_values(xyz[fk] - ctr, comp[fk], mk) / lat.mult[fk, None]
        desc = dict(cells=len(lat.positions))
    else:
        raise ValueError(f'COARSE:{kind}')
    nrm = Z.norm(dim=0)
    keep = nrm > 1e-12 * nrm.max()
    Z = Z[:, keep]
    desc.update(kind=kind, columns=int(Z.shape[1]), dropped_zero=int((~keep).sum()))
    return Z, desc


class Coarse:
    """Q = Z A_c^+ Z^T = W W^T with W^T A W = I; A W stored."""

    def __init__(self, lat, ops, kind, rcond=1e-10, batch=True):
        dev = lat.device
        self.setup = {}
        with _T(dev) as t:
            Z, self.desc = coarse_basis(lat, kind)
        self.setup['basis_s'] = t.s
        with _T(dev) as t:
            AZ = lat.matmat_sparse(ops, Z, batch=batch)
        self.setup['AZ_s'] = t.s
        with _T(dev) as t:
            Ac = Z.T @ AZ
            asym = float((Ac - Ac.T).norm() / Ac.norm())
            Ac = (Ac + Ac.T) / 2
            s = 1 / torch.sqrt(torch.clamp(torch.diagonal(Ac), min=1e-300))
            lam, V = torch.linalg.eigh(s[:, None] * Ac * s[None, :])
            ok = lam > rcond * lam.max()
            T = s[:, None] * V[:, ok] / torch.sqrt(lam[ok])[None, :]
            self.W, self.AW = Z @ T, AZ @ T
            del Z, AZ
        self.setup['factor_s'] = t.s
        self.desc.update(rank=int(ok.sum()), Ac_asym=asym, Ac_cond_scaled=float(lam.max() / lam[ok].min()),
                         negative_eigs=int((lam < 0).sum()))

    def WT(self, R):
        return self.W.T @ R

    def Wm(self, a):
        return self.W @ a

    def AWm(self, a):
        return self.AW @ a

    def AWT(self, y):
        return self.AW.T @ y

    def Q(self, R):
        return self.W @ (self.W.T @ R)

    def proj_T(self, Zr):
        """(I - Q A) z = z - W (A W)^T z."""
        return Zr - self.W @ (self.AW.T @ Zr)

    def free(self):
        self.W = self.AW = None


def _csr(r, c, v, shape):
    return torch.sparse_coo_tensor(torch.stack([r, c]), v, shape).coalesce().to_sparse_csr()


def _csr32(A):
    return torch.sparse_csr_tensor(A.crow_indices().int(), A.col_indices().int(), A.values(), A.shape)


BLOCKS = os.environ.get('OPL_COARSE_BLOCKS', '1') == '1'                     # src_v3: SparseCoarse A Z per cell
TT_KEEP = os.environ.get('OPL_V3_COARSE_TT', '1') == '1'                     # src_v3 item 2c: T^T as its own copy (0: view)
ZT_KEEP = os.environ.get('OPL_V3_COARSE_ZT', '1') == '1'                     # src_v3 item 2c: Z^T as its own CSR (0: none)


def coarse_z(lat, kind):
    """SparseCoarse's basis: Z (free DOFs x columns, CSR) and Z^T (CSR), the vertex list and the kept-column mask."""
    dev = lat.device
    xyz, comp = lat.xyz, lat.comp
    n = lat.nfree
    mk = {'q1t': 't', 'q1r': 'r', 'q1a': 'a'}[kind]
    nm = {'t': 3, 'r': 6, 'a': 12}[mk]
    offs = np.asarray(lat.positions)
    verts = sorted({tuple(p + np.asarray(c)) for p in offs for c in np.ndindex(2, 2, 2)})
    R_, C_, V_ = [], [], []
    for j, v in enumerate(verts):                                                 # same entries as coarse_basis
        vv = torch.as_tensor(v, dtype=dt, device=dev)
        h = torch.clamp(1 - (xyz - vv).abs(), min=0).prod(1)
        sup = torch.nonzero(h > 0).squeeze(1)
        if len(sup):
            blk = h[sup, None] * mode_values(xyz[sup] - vv, comp[sup], mk)
            R_.append(sup[:, None].expand(-1, nm).reshape(-1))
            C_.append((nm * j + torch.arange(nm, device=dev))[None].expand(len(sup), -1).reshape(-1))
            V_.append(blk.reshape(-1))
    r, c, v = torch.cat(R_), torch.cat(C_), torch.cat(V_)
    del R_, C_, V_
    nz = v != 0
    r, c, v = r[nz], c[nz], v[nz]
    ncol0 = nm * len(verts)
    nrm = torch.zeros(ncol0, dtype=dt, device=dev).index_add_(0, c, v * v).sqrt()
    keep = nrm > 1e-12 * nrm.max()
    newc = torch.full((ncol0,), -1, dtype=torch.long, device=dev); newc[keep] = torch.arange(int(keep.sum()), device=dev)
    ok = keep[c]
    r, c, v = r[ok], newc[c[ok]], v[ok]
    nc = int(keep.sum())
    Z = _csr(r, c, v, (n, nc)); ZT = _csr(c, r, v, (nc, n))
    if BLOCKS:                                                                    # src_v3: int32 index arrays
        Z, ZT = _csr32(Z), _csr32(ZT)
    return Z, ZT, verts, keep


class SparseCoarse:
    """Coarse with the same Q = Z A_c^+ Z^T for large lattices (OPL_COARSE_SPARSE=1; q1* spaces): Z and A Z are kept
    sparse (Z: the hats of a DOF's own cell corners only; A Z: the ports of the <= 8 cells around a vertex) and
    W = Z T, A W = (A Z) T are never formed; A_c = sum_cells (R_i Z)^T S_i (R_i Z) (the same sum as Z^T (A Z),
    reassociated). Applications: W^T R = T^T (Z^T R), W a = Z (T a), A W a = (A Z)(T a), (A W)^T y = T^T ((A Z)^T y)."""

    def __init__(self, lat, ops, kind, rcond=1e-10, batch=True):
        if not kind.startswith('q1'):
            raise ValueError(f'SPARSE_COARSE_Q1_ONLY:{kind}')
        dev = lat.device
        self.setup = {}
        n = lat.nfree
        with _T(dev) as t:
            self.Z, self.ZT, verts, keep = coarse_z(lat, kind)
            nc = self.Z.shape[1]
            if not ZT_KEEP:                                                  # src_v3 item 2c: Z^T R by the transposed
                self.ZT = None                                               # product of Z (no second copy)
        self.desc = dict(vertices=len(verts), kind=kind, columns=nc, dropped_zero=int((~keep).sum()), sparse=True,
                         Z_nnz=int(self.Z.values().numel()))
        self.setup['basis_s'] = t.s
        with _T(dev) as t:
            crow, col, val = self.Z.crow_indices(), self.Z.col_indices(), self.Z.values()
            blocks, Js, rows = [], [], []
            for i in range(len(lat.geoms)):
                kp, fk = lat._keep[i]
                s, e = crow[fk], crow[fk + 1]
                cnt = e - s
                tot = int(cnt.sum())
                rid = torch.repeat_interleave(torch.arange(len(fk), device=dev), cnt)
                idx = torch.arange(tot, device=dev) - torch.repeat_interleave(torch.cumsum(cnt, 0) - cnt, cnt) \
                    + torch.repeat_interleave(s, cnt)
                J, jinv = torch.unique(col[idx], return_inverse=True)
                q = torch.zeros((lat.geoms[i].nport, len(J)), dtype=dt, device=dev)
                q[kp[rid], jinv] = val[idx]
                blocks.append(q); Js.append(J)
            outs = lat.apply_blocks(ops, blocks, batch)
            Ac = torch.zeros((nc, nc), dtype=dt, device=dev)
            AR, AC, AV = [], [], []
            self.azb = [] if BLOCKS else None
            for i, (q, y, J) in enumerate(zip(blocks, outs, Js)):
                kp, fk = lat._keep[i]
                zb, yb = q[kp], y[kp].to(dt)
                Ac.index_put_((J[:, None], J[None, :]), zb.T @ yb, accumulate=True)
                if BLOCKS:                                                   # src_v3: (A Z) as per-cell dense blocks
                    self.azb.append((fk, J, yb.contiguous()))
                    continue
                AR.append(fk[:, None].expand(-1, len(J)).reshape(-1)); AC.append(J[None, :].expand(len(fk), -1).reshape(-1))
                AV.append(yb.reshape(-1))
            del blocks, outs
            if BLOCKS:
                self.AZ = self.AZT = None
                az_nnz = sum(int(Y.numel()) for _, _, Y in self.azb)
            else:
                ar, ac, av = torch.cat(AR), torch.cat(AC), torch.cat(AV)
                del AR, AC, AV
                self.AZ = _csr(ar, ac, av, (n, nc)); self.AZT = _csr(ac, ar, av, (nc, n))
                del ar, ac, av
                az_nnz = int(self.AZ.values().numel())
        self.setup['AZ_s'] = t.s
        with _T(dev) as t:
            asym = float((Ac - Ac.T).norm() / Ac.norm())
            Ac = (Ac + Ac.T) / 2
            s = 1 / torch.sqrt(torch.clamp(torch.diagonal(Ac), min=1e-300))
            lam, V = torch.linalg.eigh(s[:, None] * Ac * s[None, :])
            ok = lam > rcond * lam.max()
            self.T = s[:, None] * V[:, ok] / torch.sqrt(lam[ok])[None, :]
            self.TT = self.T.T.contiguous() if TT_KEEP else self.T.t()       # src_v3 item 2c: a view (cuBLAS transposes)
        self.setup['factor_s'] = t.s
        self.desc.update(rank=int(ok.sum()), Ac_asym=asym, Ac_cond_scaled=float(lam.max() / lam[ok].min()),
                         negative_eigs=int((lam < 0).sum()), AZ_nnz=az_nnz, AZ_blocks=BLOCKS)

    def WT(self, R):
        if self.ZT is None:
            return self.TT @ torch.sparse.mm(self.Z.t(), R)
        return self.TT @ torch.sparse.mm(self.ZT, R)

    def Wm(self, a):
        return torch.sparse.mm(self.Z, self.T @ a)

    def AWm(self, a):
        if self.azb is not None:                                             # sum over cells of (A Z)_i (T a)[J_i]
            ta = self.T @ a
            out = torch.zeros((self.Z.shape[0], a.shape[1]), dtype=ta.dtype, device=ta.device)
            for fk, J, Y in self.azb:
                out.index_add_(0, fk, Y @ ta[J])
            return out
        return torch.sparse.mm(self.AZ, self.T @ a)

    def AWT(self, y):
        if self.azb is not None:                                             # T^T sum over cells of (A Z)_i^T y[fk_i]
            z = torch.zeros((self.T.shape[0], y.shape[1]), dtype=y.dtype, device=y.device)
            for fk, J, Y in self.azb:
                z.index_add_(0, J, Y.T @ y[fk])
            return self.TT @ z
        return self.TT @ torch.sparse.mm(self.AZT, y)

    def Q(self, R):
        return self.Wm(self.WT(R))

    def proj_T(self, Zr):
        return Zr - self.Wm(self.AWT(Zr))

    def free(self):
        self.Z = self.ZT = self.AZ = self.AZT = self.T = self.TT = self.azb = None


# ---------------------------------------------------------------------- preconditioners
class Jacobi:
    def __init__(self, lat, kpp=None):
        dev = lat.device
        with _T(dev) as t:
            r, c, v = kpp if kpp is not None else lat.assemble_kpp()
            dg = torch.zeros(lat.nfree, dtype=dt, device=dev)
            d = r == c
            dg[r[d]] = v[d]
            self.inv = 1 / dg
        self.setup = dict(jacobi_s=t.s)

    def __call__(self, R):
        return self.inv[:, None] * R


class KppFine:
    def __init__(self, lat, kpp=None, backend='auto', holder=None):
        dev = lat.device
        if holder is not None:                                              # src_v3: triplets owned by this call
            self.nnz_upper = int(len(holder[3] if isinstance(holder[0], str) else holder[2]))   # (or ['csr', crow, cols, vals])
            with _T(dev) as t2:
                self.fac = SparseSPD(holder, None, None, lat.nfree, backend)
            self.setup = dict(kpp_assemble_s=0.0, kpp_factor_s=t2.s, **getattr(self.fac, 'spill', {}))
            self.backend = self.fac.backend
            return
        with _T(dev) as t:
            r, c, v = kpp if kpp is not None else lat.assemble_kpp()
        self.nnz_upper = int(len(v))
        with _T(dev) as t2:
            self.fac = SparseSPD(r, c, v, lat.nfree, backend)
        self.setup = dict(kpp_assemble_s=t.s, kpp_factor_s=t2.s, **getattr(self.fac, 'spill', {}))
        self.backend = self.fac.backend

    def __call__(self, R):
        return self.fac(R)

    def free(self):
        self.fac.free()


class Additive:
    def __init__(self, coarse, fine):
        self.coarse, self.fine = coarse, fine

    def __call__(self, R):
        return self.coarse.Q(R) + self.fine(R)


class BNN:
    """M = Q + (I - Q A) B (I - A Q)."""

    def __init__(self, coarse, fine):
        self.coarse, self.fine = coarse, fine

    def __call__(self, R):
        c = self.coarse
        a = c.WT(R)
        y = self.fine(R - c.AWm(a))
        return c.Wm(a) + y - c.Wm(c.AWT(y))


class Deflated:
    """Marker for dpcg: fine B and coarse (W, A W)."""

    def __init__(self, coarse, fine):
        self.coarse, self.fine = coarse, fine

    def __call__(self, R):                                                              # one application inside dpcg
        return self.coarse.proj_T(self.fine(R))


# ---------------------------------------------------------------------- solvers
def pcg(lat, ops, prec, F=None, tol=1e-8, maxit=3000, max_seconds=None, batch=True, X0=None, snaps=None):
    """PCG on the retained system. X0: initial guess (default zero; one extra operator application); snaps: descending
    relative-residual levels at which a copy of X and the iteration count are recorded (returned as 'snaps')."""
    F = lat.F if F is None else F
    if BLOCK and F.shape[1] > 1:                                         # src_v3 P1 (category B): block PCG
        return bpcg(lat, ops, prec, F=F, tol=tol, maxit=maxit, max_seconds=max_seconds, batch=batch, X0=X0, snaps=snaps)
    if RITZ and isinstance(prec, BNN):                                   # src_v3 P1 (category B): Ritz-deflated restart
        return rpcg(lat, ops, prec, F=F, tol=tol, maxit=maxit, max_seconds=max_seconds, batch=batch, X0=X0, snaps=snaps)
    dev = F.device
    if X0 is None:
        X = torch.zeros_like(F); R = F.clone()
    else:
        X = X0.clone(); R = F - lat.matvec(ops, X, batch=batch)
    Z = prec(R); P = Z.clone()
    rz = (R * Z).sum(0); r0 = F.norm(dim=0)
    sync(dev); t = time.perf_counter(); it = 0
    hist = []
    todo = sorted(snaps or [], reverse=True); got = []
    lz = [] if LANCZOS else None                                        # src_v3 (diagnostic): CG coefficients
    prof = None
    for it in range(1, maxit + 1):
        if GP.ITER and it == GP.ITER:                                   # src_v3 diagnostic: device-level profile
            with GP.Probe() as pr_:
                AP = lat.matvec(ops, P, batch=batch)
            prof = pr_.summary
        else:
            AP = lat.matvec(ops, P, batch=batch)
        alpha = rz / (P * AP).sum(0)
        X += alpha * P; R -= alpha * AP
        rel = float((R.norm(dim=0) / r0).max()); hist.append(rel)
        while todo and rel < todo[0]:
            got.append(dict(level=todo.pop(0), iterations=it, X=X.clone()))
        if rel < tol or (max_seconds and time.perf_counter() - t > max_seconds):
            if lz is not None:
                lz.append((alpha.clone(), None))
            break
        Z = prec(R)
        rz_new = (R * Z).sum(0)
        if lz is not None:
            lz.append((alpha.clone(), rz_new / rz))
        P = Z + (rz_new / rz) * P; rz = rz_new
    sync(dev)
    out = dict(X=X, iterations=it, seconds=time.perf_counter() - t, residual=hist[-1] if hist else 0.0, history=hist, snaps=got)
    if lz:
        out['lanczos'] = ritz_summary(lz)
    if prof is not None:
        out['prof'] = prof
    return out


LANCZOS = os.environ.get('OPL_V3_PCG_LANCZOS', '0') == '1'
BLOCK = os.environ.get('OPL_V3_PCG_BLOCK', '0') == '1'
BLOCK_DROP = float(os.environ.get('OPL_V3_PCG_BLOCK_DROP', '1e-12'))


def _orth(V, drop):
    """Orthonormal basis of the numerically independent part of span(V) (n x s, s small): eigen-decomposition of the
    Gram matrix of the column-normalised V; directions with eigenvalue below drop (relative to s) are left out."""
    nv = V.norm(dim=0)
    ok = nv > 0
    V = V[:, ok] / nv[ok][None, :]
    G = V.T @ V
    lam, U = torch.linalg.eigh((G + G.T) / 2)
    keep = lam > drop * max(float(lam.max()), 1.0)
    return V @ (U[:, keep] / torch.sqrt(lam[keep])[None, :])


_RZ = os.environ.get('OPL_V3_PCG_RITZ', '')                           # "m,k": restart after m steps with k Ritz vectors
RITZ = tuple(int(v) for v in _RZ.split(',')) if _RZ else None


class Augmented:
    """Coarse space W = [W_c, W_r] with W^T A W = I: the lattice coarse space plus extra A-orthonormal columns W_r
    (A-orthogonal to W_c), A W_r stored; the same methods as Coarse / SparseCoarse."""

    def __init__(self, c, Wr, AWr):
        self.c, self.Wr, self.AWr = c, Wr, AWr
        self.k = c.WT(Wr[:, :1]).shape[0]

    def WT(self, R):
        return torch.cat([self.c.WT(R), self.Wr.T @ R], 0)

    def Wm(self, a):
        return self.c.Wm(a[:self.k]) + self.Wr @ a[self.k:]

    def AWm(self, a):
        return self.c.AWm(a[:self.k]) + self.AWr @ a[self.k:]

    def AWT(self, y):
        return torch.cat([self.c.AWT(y), self.AWr.T @ y], 0)

    def Q(self, R):
        return self.Wm(self.WT(R))

    def proj_T(self, Zr):
        return Zr - self.Wm(self.AWT(Zr))


def ritz_vectors(Ps, APs, al, be, rz, k, coarse, rcond=1e-10):
    """k smallest Ritz pairs of the preconditioned operator per column from m CG steps (Ps / APs: the m search blocks
    p_j and A p_j; al, be, rz: alpha_j, beta_j = rz_{j+1} / rz_j, rz_j per column), pooled over the columns, made
    A-orthogonal to the coarse space and A-orthonormal: -> (W_r, A W_r, Ritz values used).
    Lanczos vectors v_j = (-1)^j z_j / sqrt(rz_j), z_j = p_j - beta_{j-1} p_{j-1}; a Ritz vector sum_j s_j v_j is
    sum_j (c_j - beta_j c_{j+1}) p_j with c_j = (-1)^j s_j / sqrt(rz_j): no operator application."""
    m = len(Ps)
    ncol = Ps[0].shape[1]
    Ws, AWs, th = [], [], []
    for col in range(ncol):
        a = np.array([float(x[col]) for x in al]); b = np.array([float(x[col]) for x in be]); r = np.array([float(x[col]) for x in rz])
        d = np.empty(m); e = np.empty(max(m - 1, 0))
        for i in range(m):
            d[i] = 1 / a[i] + (b[i - 1] / a[i - 1] if i > 0 else 0.0)
            if i < m - 1:
                e[i] = np.sqrt(b[i]) / a[i]
        w, S = np.linalg.eigh(np.diag(d) + np.diag(e, 1) + np.diag(e, -1))
        S = S[:, :k]
        c = ((-1.0) ** np.arange(m))[:, None] * S / np.sqrt(r)[:, None]
        cn = np.vstack([c[1:], np.zeros((1, c.shape[1]))])
        D = torch.as_tensor(c - b[:, None] * cn, dtype=Ps[0].dtype, device=Ps[0].device)          # m x k
        Ws.append(sum(Ps[j][:, col:col + 1] * D[j][None, :] for j in range(m)))
        AWs.append(sum(APs[j][:, col:col + 1] * D[j][None, :] for j in range(m)))
        th += [float(x) for x in w[:k]]
    W, AW = torch.cat(Ws, 1), torch.cat(AWs, 1)
    h = coarse.AWT(W)                                                   # A-orthogonal to the coarse space
    W, AW = W - coarse.Wm(h), AW - coarse.AWm(h)
    G = W.T @ AW
    s_ = 1 / torch.sqrt(torch.clamp(torch.diagonal(G), min=1e-300))
    lam, V = torch.linalg.eigh(s_[:, None] * ((G + G.T) / 2) * s_[None, :])
    ok = lam > rcond * lam.max()
    T = s_[:, None] * V[:, ok] / torch.sqrt(lam[ok])[None, :]
    return W @ T, AW @ T, th


def rpcg(lat, ops, prec, F=None, tol=1e-8, maxit=3000, max_seconds=None, batch=True, X0=None, snaps=None):
    """src_v3 P1 (OPL_V3_PCG_RITZ=m,k, default off; a different solver, results agree to the tolerance only): PCG with
    the BNN preconditioner for m steps, then the k smallest Ritz vectors of the preconditioned operator (per column,
    from the stored search directions, no operator application) are added to the coarse space (deflation of the
    slowest modes) and PCG restarts from the current iterate and residual with the augmented BNN preconditioner.
    Costs: 2 m stored vectors per column during the first phase; k more coarse columns per column afterwards."""
    F = lat.F if F is None else F
    m_, k_ = RITZ
    dev = F.device
    if X0 is None:
        X = torch.zeros_like(F); R = F.clone()
    else:
        X = X0.clone(); R = F - lat.matvec(ops, X, batch=batch)
    r0 = F.norm(dim=0)
    sync(dev); t = time.perf_counter(); it = 0
    hist = []
    todo = sorted(snaps or [], reverse=True); got = []
    M = prec
    Z = M(R); P = Z.clone(); rz = (R * Z).sum(0)
    Ps, APs, al, be, rzs = [], [], [], [], []
    info = dict(m=m_, k=k_)
    for it in range(1, maxit + 1):
        AP = lat.matvec(ops, P, batch=batch)
        alpha = rz / (P * AP).sum(0)
        if Ps is not None:
            Ps.append(P); APs.append(AP); al.append(alpha); rzs.append(rz)
        X += alpha * P; R -= alpha * AP
        rel = float((R.norm(dim=0) / r0).max()); hist.append(rel)
        while todo and rel < todo[0]:
            got.append(dict(level=todo.pop(0), iterations=it, X=X.clone()))
        if rel < tol or (max_seconds and time.perf_counter() - t > max_seconds):
            break
        if Ps is not None and it == m_:                                 # deflate and restart
            Zt = M(R)
            be.append((R * Zt).sum(0) / rz)
            sync(dev); t1 = time.perf_counter()
            Wr, AWr, th = ritz_vectors(Ps, APs, al, be, rzs, k_, prec.coarse)
            Ps = APs = None
            M = BNN(Augmented(prec.coarse, Wr, AWr), prec.fine)
            Z = M(R); P = Z.clone(); rz = (R * Z).sum(0)
            sync(dev)
            info.update(added=int(Wr.shape[1]), ritz=[round(x, 5) for x in sorted(th)[:12]], setup_s=time.perf_counter() - t1,
                        restart_at=it)
            continue
        Z = M(R)
        rz_new = (R * Z).sum(0)
        if Ps is not None:
            be.append(rz_new / rz)
        P = Z + (rz_new / rz) * P; rz = rz_new
    sync(dev)
    return dict(X=X, iterations=it, seconds=time.perf_counter() - t, residual=hist[-1] if hist else 0.0, history=hist,
                snaps=got, ritz=info)


def bpcg(lat, ops, prec, F=None, tol=1e-8, maxit=3000, max_seconds=None, batch=True, X0=None, snaps=None):
    """src_v3 P1 (OPL_V3_PCG_BLOCK=1, default off; a different solver, results agree to the tolerance only): block PCG
    for several right-hand sides (breakdown-free form, Ji & Li 2017): the columns share one block Krylov space instead of
    running independent recurrences (pcg). Per iteration one operator application to the search block P (at most as
    many columns as F; fewer once directions become dependent), one preconditioner application to R:
        Q = A P, alpha = (P^T Q)^-1 P^T R, X += P alpha, R -= Q alpha, Z = M R,
        beta = -(P^T Q)^-1 Q^T Z, P = orth(Z + P beta).
    Convergence test, snapshots and result as pcg (relative residual per column, the largest one)."""
    F = lat.F if F is None else F
    dev = F.device
    if X0 is None:
        X = torch.zeros_like(F); R = F.clone()
    else:
        X = X0.clone(); R = F - lat.matvec(ops, X, batch=batch)
    r0 = F.norm(dim=0)
    P = _orth(prec(R), BLOCK_DROP)
    sync(dev); t = time.perf_counter(); it = 0
    hist, widths = [], []
    todo = sorted(snaps or [], reverse=True); got = []
    for it in range(1, maxit + 1):
        Q = lat.matvec(ops, P, batch=batch)
        PQ = P.T @ Q
        L = torch.linalg.cholesky((PQ + PQ.T) / 2)
        alpha = torch.cholesky_solve(P.T @ R, L)
        X += P @ alpha; R -= Q @ alpha
        widths.append(int(P.shape[1]))
        rel = float((R.norm(dim=0) / r0).max()); hist.append(rel)
        while todo and rel < todo[0]:
            got.append(dict(level=todo.pop(0), iterations=it, X=X.clone()))
        if rel < tol or (max_seconds and time.perf_counter() - t > max_seconds):
            break
        Z = prec(R)
        beta = -torch.cholesky_solve(Q.T @ Z, L)
        P = _orth(Z + P @ beta, BLOCK_DROP)
    sync(dev)
    return dict(X=X, iterations=it, seconds=time.perf_counter() - t, residual=hist[-1] if hist else 0.0, history=hist,
                snaps=got, block_widths=dict(min=min(widths) if widths else 0, max=max(widths) if widths else 0,
                                             mean=float(np.mean(widths)) if widths else 0.0))


def ritz_summary(lz, n_small=40, n_large=5):
    """src_v3 diagnostic (OPL_V3_PCG_LANCZOS=1): the Ritz values of the preconditioned operator from the CG coefficients
    (Lanczos tridiagonal T: T_kk = 1/alpha_k + beta_{k-1}/alpha_{k-1}, T_k,k+1 = sqrt(beta_k)/alpha_k), per column: the
    smallest and largest ones and their ratio (an estimate of the effective condition number seen by this solve)."""
    al = torch.stack([a for a, _ in lz]).double().cpu().numpy()          # its x cols
    be = np.array([b.double().cpu().numpy() if b is not None else np.full(al.shape[1], np.nan) for _, b in lz])
    res = []
    for col in range(al.shape[1]):
        a, b = al[:, col], be[:, col]
        k = len(a)
        d = np.empty(k); e = np.empty(max(k - 1, 0))
        for i in range(k):
            d[i] = 1 / a[i] + (b[i - 1] / a[i - 1] if i > 0 else 0.0)
            if i < k - 1:
                e[i] = np.sqrt(b[i]) / a[i]
        T = np.diag(d) + np.diag(e, 1) + np.diag(e, -1)
        w = np.linalg.eigvalsh(T)
        res.append(dict(steps=k, smallest=[float(x) for x in w[:n_small]], largest=[float(x) for x in w[-n_large:]],
                        cond=float(w[-1] / w[0]) if w[0] > 0 else None,
                        below_1pct_of_max=int((w < 1e-2 * w[-1]).sum()), below_10pct_of_max=int((w < 1e-1 * w[-1]).sum())))
    return res


def dpcg(lat, ops, defl, F=None, tol=1e-8, maxit=3000, max_seconds=None, batch=True):
    """Deflated PCG (A-DEF2): x0 = Q b, r0 = b - A W W^T b (no operator application), p = (I - Q A) B r."""
    F = lat.F if F is None else F
    c, B = defl.coarse, defl.fine
    dev = F.device
    sync(dev); t = time.perf_counter()
    a = c.WT(F)
    X = c.Wm(a); R = F - c.AWm(a)
    r0 = F.norm(dim=0)
    Zr = B(R); P = c.proj_T(Zr)
    rz = (R * Zr).sum(0)
    it = 0; hist = []
    rel = float((R.norm(dim=0) / r0).max())
    if rel < tol:
        maxit = 0
    for it in range(1, maxit + 1):
        AP = lat.matvec(ops, P, batch=batch)
        alpha = rz / (P * AP).sum(0)
        X += alpha * P; R -= alpha * AP
        rel = float((R.norm(dim=0) / r0).max()); hist.append(rel)
        if rel < tol or (max_seconds and time.perf_counter() - t > max_seconds):
            break
        Zr = B(R)
        rz_new = (R * Zr).sum(0)
        P = c.proj_T(Zr) + (rz_new / rz) * P; rz = rz_new
    sync(dev)
    return dict(X=X, iterations=it, seconds=time.perf_counter() - t, residual=hist[-1] if hist else rel, history=hist)


def solve(lat, ops, pc, **kw):
    return dpcg(lat, ops, pc, **kw) if isinstance(pc, Deflated) else pcg(lat, ops, pc, **kw)


# ---------------------------------------------------------------------- factory with shared setups
class Factory:
    """Builds preconditioners from spec strings for one lattice and one operator set; the fine levels (operator
    independent) can be shared across operator sets by passing the same `shared` dict."""

    def __init__(self, lat, ops, shared=None, rcond=1e-10, kpp_backend='auto', log=None):
        self.lat, self.ops, self.rcond, self.kpp_backend = lat, ops, rcond, kpp_backend
        self.shared = {} if shared is None else shared
        self.coarse = {}
        self.log = log or (lambda s_: None)

    def _kpp(self):
        if 'kpp_triplets' not in self.shared:
            with _T(self.lat.device) as t:
                self.shared['kpp_triplets'] = self.lat.assemble_kpp()
            self.shared['kpp_triplets_s'] = t.s
        return self.shared['kpp_triplets']

    def fine(self, name):
        if name not in self.shared:
            if name == 'jac':
                self.shared[name] = Jacobi(self.lat, self._kpp())
            elif name == 'kpp':
                if SLIM and KPP_CSR and 'kpp_triplets' not in self.shared and hasattr(self.lat, 'assemble_kpp_csr'):
                    with _T(self.lat.device) as t:                          # src_v3: row blocks straight to CSR
                        holder = ['csr'] + list(self.lat.assemble_kpp_csr())
                    self.shared['kpp_triplets_s'] = t.s
                    self.shared[name] = KppFine(self.lat, backend=self.kpp_backend, holder=holder)
                    self.shared[name].setup['kpp_csr'] = dict(getattr(self.lat, 'kpp_csr_info', {}))
                elif SLIM:                                                  # src_v3: the factor takes the triplets over
                    self._kpp()
                    holder = list(self.shared.pop('kpp_triplets'))
                    self.shared[name] = KppFine(self.lat, backend=self.kpp_backend, holder=holder)
                else:
                    self.shared[name] = KppFine(self.lat, self._kpp(), self.kpp_backend)
            else:
                raise ValueError(f'FINE:{name}')
            self.shared[name].setup['kpp_triplets_s'] = self.shared.get('kpp_triplets_s', 0.0)
        return self.shared[name]

    def coarse_space(self, kind):
        if kind not in self.coarse:
            sparse = os.environ.get('OPL_COARSE_SPARSE') == '1' and kind.startswith('q1')
            self.coarse[kind] = (SparseCoarse if sparse else Coarse)(self.lat, self.ops, kind, self.rcond)
            self.log(dict(event='COARSE', **self.coarse[kind].desc, **self.coarse[kind].setup))
        return self.coarse[kind]

    def build(self, spec):
        """-> (preconditioner, setup dict (seconds of every ingredient; 'total_s'), description)."""
        parts = spec.split(':')
        if parts[0] in ('jacobi', 'kpp'):
            f = self.fine('jac' if parts[0] == 'jacobi' else 'kpp')
            st = dict(f.setup)
            return f, dict(st, total_s=sum(v for k, v in st.items() if k.endswith('_s'))), dict(fine=parts[0])
        comb, fname, cname = parts
        c = self.coarse_space(cname)                                        # coarse first: its setup temporaries
        f = self.fine(fname)                                                # are freed before the K_PP factor
        pc = dict(add=Additive, bnn=BNN, defl=Deflated)[comb](c, f)
        st = dict(f.setup); st.update({'coarse_' + k: v for k, v in c.setup.items()})
        return pc, dict(st, total_s=sum(v for k, v in st.items() if k.endswith('_s'))), dict(fine=fname, coarse=dict(c.desc))

    def free(self):
        for c in self.coarse.values():
            c.free()
        self.coarse.clear()


def apply_cost(pc, R, reps=3):
    """Seconds per preconditioner application on the block R (for Deflated: B + the projection)."""
    dev = R.device
    pc(R)
    sync(dev); t = time.perf_counter()
    for _ in range(reps):
        pc(R)
    sync(dev)
    return (time.perf_counter() - t) / reps


def true_residual(lat, ops, X, F=None, batch=True):
    F = lat.F if F is None else F
    return float(((F - lat.matvec(ops, X, batch=batch)).norm(dim=0) / F.norm(dim=0)).max())
