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
import time
import numpy as np
import torch

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
class SparseSPD:
    """Factor of an SPD matrix given by upper triplets on n unknowns (Jacobi scaled first)."""

    def __init__(self, rows, cols, vals, n, backend='auto'):
        self.n, self.dev = n, vals.device
        dg = torch.zeros(n, dtype=dt, device=vals.device); d = rows == cols
        dg[rows[d]] = vals[d]
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
            self.sol = TE.SPDSolver(crow.int(), cols.int(), sv.contiguous(), n)
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

    def Q(self, R):
        return self.W @ (self.W.T @ R)

    def proj_T(self, Zr):
        """(I - Q A) z = z - W (A W)^T z."""
        return Zr - self.W @ (self.AW.T @ Zr)

    def free(self):
        self.W = self.AW = None


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
    def __init__(self, lat, kpp=None, backend='auto'):
        dev = lat.device
        with _T(dev) as t:
            r, c, v = kpp if kpp is not None else lat.assemble_kpp()
        self.nnz_upper = int(len(v))
        with _T(dev) as t2:
            self.fac = SparseSPD(r, c, v, lat.nfree, backend)
        self.setup = dict(kpp_assemble_s=t.s, kpp_factor_s=t2.s)
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
        a = c.W.T @ R
        y = self.fine(R - c.AW @ a)
        return c.W @ a + y - c.W @ (c.AW.T @ y)


class Deflated:
    """Marker for dpcg: fine B and coarse (W, A W)."""

    def __init__(self, coarse, fine):
        self.coarse, self.fine = coarse, fine

    def __call__(self, R):                                                              # one application inside dpcg
        return self.coarse.proj_T(self.fine(R))


# ---------------------------------------------------------------------- solvers
def pcg(lat, ops, prec, F=None, tol=1e-8, maxit=3000, max_seconds=None, batch=True):
    F = lat.F if F is None else F
    dev = F.device
    X = torch.zeros_like(F); R = F.clone()
    Z = prec(R); P = Z.clone()
    rz = (R * Z).sum(0); r0 = F.norm(dim=0)
    sync(dev); t = time.perf_counter(); it = 0
    hist = []
    for it in range(1, maxit + 1):
        AP = lat.matvec(ops, P, batch=batch)
        alpha = rz / (P * AP).sum(0)
        X += alpha * P; R -= alpha * AP
        rel = float((R.norm(dim=0) / r0).max()); hist.append(rel)
        if rel < tol or (max_seconds and time.perf_counter() - t > max_seconds):
            break
        Z = prec(R)
        rz_new = (R * Z).sum(0)
        P = Z + (rz_new / rz) * P; rz = rz_new
    sync(dev)
    return dict(X=X, iterations=it, seconds=time.perf_counter() - t, residual=hist[-1] if hist else 0.0, history=hist)


def dpcg(lat, ops, defl, F=None, tol=1e-8, maxit=3000, max_seconds=None, batch=True):
    """Deflated PCG (A-DEF2): x0 = Q b, r0 = b - A W W^T b (no operator application), p = (I - Q A) B r."""
    F = lat.F if F is None else F
    c, B = defl.coarse, defl.fine
    dev = F.device
    sync(dev); t = time.perf_counter()
    a = c.W.T @ F
    X = c.W @ a; R = F - c.AW @ a
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
                self.shared[name] = KppFine(self.lat, self._kpp(), self.kpp_backend)
            else:
                raise ValueError(f'FINE:{name}')
            self.shared[name].setup['kpp_triplets_s'] = self.shared.get('kpp_triplets_s', 0.0)
        return self.shared[name]

    def coarse_space(self, kind):
        if kind not in self.coarse:
            self.coarse[kind] = Coarse(self.lat, self.ops, kind, self.rcond)
            self.log(dict(event='COARSE', **self.coarse[kind].desc, **self.coarse[kind].setup))
        return self.coarse[kind]

    def build(self, spec):
        """-> (preconditioner, setup dict (seconds of every ingredient; 'total_s'), description)."""
        parts = spec.split(':')
        if parts[0] in ('jacobi', 'kpp'):
            f = self.fine('jac' if parts[0] == 'jacobi' else 'kpp')
            st = dict(f.setup)
            return f, dict(st, total_s=sum(st.values())), dict(fine=parts[0])
        comb, fname, cname = parts
        f, c = self.fine(fname), self.coarse_space(cname)
        pc = dict(add=Additive, bnn=BNN, defl=Deflated)[comb](c, f)
        st = dict(f.setup); st.update({'coarse_' + k: v for k, v in c.setup.items()})
        return pc, dict(st, total_s=sum(st.values())), dict(fine=fname, coarse=dict(c.desc))

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
