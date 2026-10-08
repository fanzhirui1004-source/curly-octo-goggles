"""Route 7: how many boundary-operator queries (S q per cell) does ONE lattice solve need?

Conventional thickness optimization does one global solve per design iteration (compliance is self-adjoint, so the
adjoint is the displacement itself). With full-resolution cell interfaces the global system is the interface
problem sum_c R_c^T S_c R_c u = f (about half of 23 616 box DOFs per FULL cell), too large to factor, so it is
solved iteratively and every iteration queries every cell once. This measures the iteration count.

Lattice: nx x ny x nz identical FULL cells of uniform thickness (opposite faces match), exact fp64 box operator
T from the Schur mode; plane x = 0 clamped; loads on the plane x = nx: uniform nodal forces in x (axial) and in -z
(bending). Preconditioners:
  jacobi      diagonal of the assembled interface matrix (needs only forward queries);
  coarse+jac  hybrid two-level: coarse space of partition-of-unity weighted rigid modes of the floating cells,
              Jacobi smoother (forward queries only);
  bdd         balancing domain decomposition (Mandel): same coarse space, local Neumann solves with each cell's
              pseudo-inverse S_c^+ (needs an inverse action per cell, not only S q).
Counts: CG iterations and forward queries per cell to relative residual 1e-8; compliance error along the way.
Usage: lattice_pcg.py <body_dir> <out_dir> <case> <sub> <tau_uniform> <dims;dims;...> [precs]
"""
import json, sys, time, gc
from fractions import Fraction
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import schur_encoder as SE
import precision_study as PS

dev, dt = SE.dev, SE.dt


class Lattice:
    def __init__(self, T, node_ids, n, dims, log=print):
        self.T, self.n, self.dims = T, n, dims
        nx, ny, nz = dims
        g = np.stack(np.unravel_index(node_ids, (2 * n + 1,) * 3), 1).astype(np.int64)
        self.nbn = len(g); self.nbx = 3 * self.nbn
        cells = np.array([(i, j, k) for i in range(nx) for j in range(ny) for k in range(nz)], dtype=np.int64)
        self.cells = cells; self.Nc = len(cells)
        G = g[None, :, :] + 2 * n * cells[:, None, :]                                 # Nc x nbn x 3 global grid
        L = 2 * n * np.asarray(dims) + 1
        keys = (G[..., 0] * L[1] + G[..., 1]) * L[2] + G[..., 2]
        uniq, inv = np.unique(keys.reshape(-1), return_inverse=True)
        cell_nodes = inv.reshape(self.Nc, self.nbn)
        gx = uniq // (L[1] * L[2]); gz_ = uniq % L[2]
        self.N_nodes = len(uniq)
        clamped_node = gx == 0
        loaded_node = gx == 2 * n * nx
        ndof = 3 * self.N_nodes
        clamped = np.repeat(clamped_node, 3)
        free = np.flatnonzero(~clamped); fmap = np.full(ndof, -1, dtype=np.int64); fmap[free] = np.arange(len(free))
        self.nfree = len(free)
        cell_dofs = (3 * cell_nodes[:, :, None] + np.arange(3)).reshape(self.Nc, self.nbx)
        cf = fmap[cell_dofs]
        self.touch_clamp = (cf < 0).any(1)
        self.cell_free = torch.as_tensor(np.where(cf < 0, self.nfree, cf), device=dev)    # dump row for clamped
        self.cf_mask = torch.as_tensor(cf >= 0, device=dev)
        F = np.zeros((ndof, 2))
        on = np.flatnonzero(loaded_node)
        F[3 * on + 0, 0] = 1.0; F[3 * on + 2, 1] = -1.0
        F /= np.abs(F).sum(0, keepdims=True)
        self.F = torch.as_tensor(F[free], dtype=dt, device=dev)
        mult = torch.zeros(self.nfree + 1, dtype=dt, device=dev)
        mult.index_add_(0, self.cell_free.reshape(-1), torch.ones(self.Nc * self.nbx, dtype=dt, device=dev))
        self.mult = mult[:self.nfree]
        self.queries = 0
        shared = int((np.bincount(cell_nodes.reshape(-1), minlength=self.N_nodes) > 1).sum())
        log(json.dumps(dict(event='LATTICE', dims=list(dims), cells=self.Nc, nodes=self.N_nodes, shared_nodes=shared,
                            free_dofs=self.nfree, clamped_cells=int(self.touch_clamp.sum()))))

    def gather(self, X):
        """global free vectors (nfree x k) -> per-cell box vectors (Nc x nbx x k), zero at clamped DOFs."""
        Xp = torch.cat([X, torch.zeros((1, X.shape[1]), dtype=X.dtype, device=dev)])
        return Xp[self.cell_free]

    def scatter(self, Y):
        k = Y.shape[2]
        out = torch.zeros((self.nfree + 1, k), dtype=Y.dtype, device=dev)
        out.index_add_(0, self.cell_free.reshape(-1), Y.reshape(-1, k))
        return out[:self.nfree]

    def cell_apply(self, Xc, M):
        """M @ Xc for every cell: Xc (Nc x nbx x k) -> (Nc x nbx x k), M symmetric nbx x nbx."""
        Nc, nbx, k = Xc.shape
        Y = M @ Xc.permute(1, 0, 2).reshape(nbx, Nc * k)
        return Y.reshape(nbx, Nc, k).permute(1, 0, 2)

    def matvec(self, X):
        self.queries += 1
        return self.scatter(self.cell_apply(self.gather(X), self.T))


def build_coarse(lat, R, log=print):
    """Phi = [R_c^T D_c Z_c for floating cells]; A_c = Phi^T A Phi assembled cell by cell (local columns only)."""
    fl = np.flatnonzero(~lat.touch_clamp)
    D = 1.0 / lat.mult
    Dp = torch.cat([D, torch.zeros(1, dtype=dt, device=dev)])
    Dc = Dp[lat.cell_free]                                                          # Nc x nbx (0 at clamped)
    nco = 6 * len(fl)
    col_of = {int(c): 6 * i for i, c in enumerate(fl)}
    # which floating cells touch each cell: share at least one free DOF
    owner = torch.full((lat.nfree + 1,), -1, dtype=torch.long, device=dev)
    Ac = torch.zeros((nco, nco), dtype=dt, device=dev)
    cf = lat.cell_free
    # cells are neighbours if their grid indices differ by at most 1 in every direction
    cells = lat.cells
    for c in range(lat.Nc):
        nb = np.flatnonzero((np.abs(cells[fl] - cells[c]).max(1) <= 1))
        nb_cells = fl[nb]
        if len(nb_cells) == 0:
            continue
        # local basis: for each neighbouring floating cell d, its weighted rigid modes restricted to cell c's DOFs
        cols = []
        pos_c = torch.full((lat.nfree + 1,), -1, dtype=torch.long, device=dev)
        pos_c[cf[c]] = torch.arange(lat.nbx, device=dev)
        pos_c[lat.nfree] = -1
        for d in nb_cells:
            B = torch.zeros((lat.nbx, 6), dtype=dt, device=dev)
            pd = pos_c[cf[d]]                                                       # where d's DOFs sit in c
            m = (pd >= 0) & lat.cf_mask[d]
            B[pd[m]] = (Dc[d][:, None] * R)[m]
            cols.append(B)
        Bc = torch.cat(cols, 1)                                                     # nbx x 6 len(nb)
        Bc = Bc * lat.cf_mask[c][:, None]
        loc = Bc.T @ (lat.T @ Bc)
        idx = torch.as_tensor(np.concatenate([np.arange(col_of[int(d)], col_of[int(d)] + 6) for d in nb_cells]), device=dev)
        Ac[idx[:, None], idx[None, :]] += loc
    Ac = 0.5 * (Ac + Ac.T)
    Lc = torch.linalg.cholesky(Ac)
    lat.queries += 0
    log(json.dumps(dict(event='COARSE', floating=int(len(fl)), coarse_dim=nco)))
    return dict(fl=torch.as_tensor(fl, device=dev), Dc=Dc, Lc=Lc, R=R, nco=nco)


def coarse_T(lat, co, X):
    """Phi^T X: per floating cell Z^T (D_c * X_c) -> (nco x k)."""
    Xc = lat.gather(X)[co['fl']]                                                    # Nf x nbx x k
    Y = torch.einsum('bm,cbk->cmk', co['R'], co['Dc'][co['fl']][:, :, None] * Xc)
    return Y.reshape(-1, X.shape[1])


def coarse_(lat, co, lam):
    """Phi lam -> global free vector."""
    k = lam.shape[1]
    lam = lam.reshape(len(co['fl']), 6, k)
    Yc = torch.zeros((lat.Nc, lat.nbx, k), dtype=dt, device=dev)
    Yc[co['fl']] = co['Dc'][co['fl']][:, :, None] * torch.einsum('bm,cmk->cbk', co['R'], lam)
    return lat.scatter(Yc)


def coarse_solve(co, b):
    return torch.cholesky_solve(b, co['Lc'])


def local_factors(lat, R, alpha):
    """pseudo-inverse of the floating cell (Cholesky of T + alpha R R^T) and inverse of a clamped cell (T on its
    free DOFs; all clamped cells of this lattice share the same clamped face)."""
    Lf = torch.linalg.cholesky(lat.T + alpha * (R @ R.T))
    c0 = int(np.flatnonzero(lat.touch_clamp)[0])
    fm = lat.cf_mask[c0]
    for c in np.flatnonzero(lat.touch_clamp):
        if not bool((lat.cf_mask[int(c)] == fm).all()):
            raise ValueError('CLAMP_PATTERN_DIFFERS')
    idx = torch.nonzero(fm).squeeze(1)
    Ld = torch.linalg.cholesky(lat.T[idx][:, idx])
    return dict(Lf=Lf, Ld=Ld, didx=idx, alpha=alpha)


def nn_apply(lat, co, lf, R, r):
    """sum_c R_c^T D_c S_c^+ D_c R_c r."""
    k = r.shape[1]
    D = co['Dc'][:, :, None]
    Xc = D * lat.gather(r)
    Yc = torch.zeros_like(Xc)
    fl = co['fl']
    if len(fl):
        Xf = Xc[fl]                                                                  # Nf x nbx x k
        Nf = len(fl)
        B = Xf.permute(1, 0, 2).reshape(lat.nbx, Nf * k)
        Z = torch.cholesky_solve(B, lf['Lf'])
        Z = Z - R @ (R.T @ Z)                     # (T + a R R^T)^-1 B = T^+ B + R R^T B / a; T^+ B is orthogonal to R
        Yc[fl] = Z.reshape(lat.nbx, Nf, k).permute(1, 0, 2)
    cl = torch.as_tensor(np.flatnonzero(lat.touch_clamp), device=dev)
    if len(cl):
        Xd = Xc[cl][:, lf['didx'], :]
        Nd = len(cl)
        B = Xd.permute(1, 0, 2).reshape(len(lf['didx']), Nd * k)
        Z = torch.cholesky_solve(B, lf['Ld']).reshape(len(lf['didx']), Nd, k).permute(1, 0, 2)
        tmp = torch.zeros((Nd, lat.nbx, k), dtype=dt, device=dev)
        tmp[:, lf['didx'], :] = Z
        Yc[cl] = tmp
    return lat.scatter(D * Yc)


def pcg(lat, apply_M, tol=1e-8, maxit=3000, x0=None, log=print, label=''):
    F = lat.F
    x = torch.zeros_like(F) if x0 is None else x0.clone()
    r = F - lat.matvec(x) if x0 is not None else F.clone()
    z = apply_M(r); p = z.clone(); rz = (r * z).sum(0)
    nF = F.norm(dim=0)
    hist = []
    it = 0
    for it in range(1, maxit + 1):
        Ap = lat.matvec(p)
        a = rz / (p * Ap).sum(0)
        x += a * p; r -= a * Ap
        rel = float((r.norm(dim=0) / nF).max())
        if it % 10 == 0 or rel < tol:
            hist.append((it, rel, lat.queries, (F * x).sum(0).tolist()))
        if rel < tol:
            break
        z = apply_M(r); rz2 = (r * z).sum(0)
        p = z + (rz2 / rz) * p; rz = rz2
    return x, it, hist


def run_case(lat, R, precs, log=print):
    out = {}
    alpha = None
    co = lf = None
    if any(p in precs for p in ('coarse+jac', 'bdd')):
        t = time.perf_counter(); co = build_coarse(lat, R, log); out['coarse_seconds'] = time.perf_counter() - t
    if 'bdd' in precs:
        alpha = float(torch.linalg.matrix_norm(lat.T, ord=2)) if lat.nbx < 6000 else float(lat.T.diagonal().max()) * 10
        t = time.perf_counter(); lf = local_factors(lat, R, alpha); out['local_factor_seconds'] = time.perf_counter() - t
    diag = lat.scatter(lat.T.diagonal()[None, :, None].expand(lat.Nc, lat.nbx, 1).clone() * lat.cf_mask[:, :, None])[:, 0]
    jac = lambda r: r / diag[:, None]
    for name in precs:
        lat.queries = 0
        t = time.perf_counter()
        if name == 'jacobi':
            M = jac
        elif name in ('coarse+jac', 'bdd'):
            smoother = jac if name == 'coarse+jac' else (lambda r: nn_apply(lat, co, lf, R, r))
            def M(r, smoother=smoother):
                a = coarse_(lat, co, coarse_solve(co, coarse_T(lat, co, r)))
                r1 = r - lat.matvec(a)
                z1 = smoother(r1)
                return a + z1 - coarse_(lat, co, coarse_solve(co, coarse_T(lat, co, lat.matvec(z1))))
        maxit = 3000 if name == 'jacobi' else 1000
        x, it, hist = pcg(lat, M, maxit=maxit, log=log, label=name)
        torch.cuda.synchronize(); sec = time.perf_counter() - t
        c = (lat.F * x).sum(0).tolist()
        out[name] = dict(iterations=it, queries_per_cell=lat.queries, seconds=sec, compliance=c,
                         final_rel_residual=hist[-1][1], history=hist)
        log(json.dumps(dict(prec=name, iterations=it, queries_per_cell=lat.queries, seconds=round(sec, 1),
                            rel=hist[-1][1], compliance=c)))
    # compliance convergence: queries needed to get within 1e-3 and 1e-6 of the final compliance
    best = None
    for name in precs:
        if out[name]['final_rel_residual'] < 1e-8:
            best = np.asarray(out[name]['compliance']); break
    if best is not None:
        for name in precs:
            for tol in (1e-2, 1e-3, 1e-6):
                q = next((h[2] for h in out[name]['history'] if np.abs(np.asarray(h[3]) / best - 1).max() < tol), None)
                out[name][f'queries_to_compliance_{tol:g}'] = q
    return out


def main(body_dir, out, case, sub, tau, dims_list, precs):
    Path(out).mkdir(parents=True, exist_ok=True)
    taus = [float(Fraction(tau))] * 8
    enc = SE.SchurEncoder(case, body_dir, precision='fp64', log=print, sub=sub, out_host=True)
    Th, st = enc.update(taus)
    ids, n = enc.box_node_ids.copy(), enc.n
    enc.free(); enc.Tbuf = None; del enc
    gc.collect(); torch.cuda.empty_cache()
    T = Th.T.contiguous().to(dev)                                                    # symmetric
    del Th
    R = PS.rigid_basis(ids, n)
    T -= R @ (R.T @ T); T -= (T @ R) @ R.T                                           # exact rigid null space
    T = 0.5 * (T + T.T)
    rec = dict(case=case, sub=sub, tau=tau, box=int(T.shape[0]), schur=st, lattices=[])
    print(json.dumps(dict(event='T', box=int(T.shape[0]), schur=st)), flush=True)
    for dims in dims_list:
        lat = Lattice(T, ids, n, dims)
        res = run_case(lat, R, precs)
        rec['lattices'].append(dict(dims=list(dims), cells=lat.Nc, free_dofs=lat.nfree, **res))
        (Path(out) / f'LATTICE_PCG_{sub}.json').write_text(json.dumps(rec, indent=2, default=float))
        del lat; gc.collect(); torch.cuda.empty_cache()
    print('DONE', flush=True)


if __name__ == '__main__':
    dims_list = [tuple(int(v) for v in d.split('x')) for d in sys.argv[6].split(';')]
    precs = sys.argv[7].split(',') if len(sys.argv) > 7 else ['bdd', 'coarse+jac', 'jacobi']
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], dims_list, precs)
