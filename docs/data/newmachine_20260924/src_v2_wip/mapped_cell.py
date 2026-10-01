"""Direction A, step 0: a cell under a smooth volume map x = Phi(X), discretised isoparametrically (default off).

Nothing in the existing code imports this module; it only adds a new path.

Setting. The level set, the background grid, the active elements, ghost faces and ports stay those of the reference cell
(X in the unit box). The displacement unknowns stay Cartesian physical components at the Q2 nodes. The geometry is the Q2
interpolant of the map, x_h(t) = sum_a x_a N_a(t) with x_a = Phi(X_a) at the 65^3 grid nodes (isoparametric), so
u = c + W x_h is in the discrete space and the rigid motions of the mapped cell are exact zero-energy modes.

Element stiffness. K_e[a i, b k] = int dN_a/dt_J  A_iJkL(t)  dN_b/dt_L dt over the material part of the element, with
  A_iJkL = det(J) (J^-1)_Jj C_ijkl (J^-1)_Ll,   J = dx_h/dt   (derivative indices only; A has major, not minor symmetry).
It is integrated with the integrator's own positive-weight rule (polyref_torch_fast.cell_moments: Kuhn tetrahedra of the
s^3 sub-grid clipped by the three level-set functions, collapsed Gauss-Jacobi on the pieces), except that sub-cubes
entirely inside the material use the tensor 3-point Gauss rule instead of the closed form, and elements whose base
sub-cubes are all inside use the 5-point tensor Gauss rule on the whole element. With A constant both rules reproduce
the moments exactly, so for the identity map K_e equals sum_m M_em T_m to rounding. The quadrature sum is reorganised as
45 weighted moment sets, Mt[e, c, m] = sum_q w_q A_c(t_q) t_q^m (c over the 45 entries p <= q of the symmetric 9 x 9
matrix A_(iJ),(kL)), and fixed templates Tt (45 * 125, 81, 81): K_e = Mt_e . Tt. This is the same quadrature sum
evaluated in another order: the products of shape-function derivatives lie in the span of the 125 monomials. Positive
weights and det J > 0 give K_e >= 0.

Ghost penalty. The P1 form (lambda_0 + 2 mu_0) sum_f sum_{j=1,2} h^(2j-1) int_f [d^j u / dn^j]^2 dA is evaluated in
physical coordinates: the normal derivatives use the physical gradient and the full isoparametric Hessian
(H_x = J^-T (d2N/dt2 - sum_k dN/dx_k d2x_k/dt2) J^-1), the physical unit normal of the mapped face, the physical area
element |x_u x x_v| and h_x = h det(dx/dX)^(1/3). For a rigid motion the physical gradient is constant and the physical
Hessian vanishes on both sides, so the penalty does not see it; for the identity map the form is the P1 one.

MappedCell(teacher.Cell) uses these for K, and the physical node coordinates for the rigid bases (Q, Qall). Paths that
use the isotropic templates directly (lean/deploy storage, thickness sensitivities) raise NotImplementedError.
Usage: see t_mapped.py (unit tests) and a0_eval.py (step-0 evaluation).
"""
import gc, json, os, time
from itertools import product
from pathlib import Path
import numpy as np
import torch

import teacher as TE
import element_moments as EM
import polyref_torch_fast as PT
from element_polyref import KUHN, CUBE, tet_rule, cube_rule

dev, dt = TE.dev, TE.dt
PAIRS = [(p, q) for p in range(9) for q in range(p, 9)]                       # 45 entries of the symmetric 9 x 9 A


# ------------------------------------------------------------------------------------------------------------- maps
def make_map(spec):
    """spec (dict) -> Phi: (N, 3) reference points of the unit cell -> (N, 3) physical points (numpy, float64).
    kinds: identity; rotation (axis, deg); scale (s); affine (A, 3 x 3, about the cell centre); twist (deg: rotation
    angle per cell length about the z axis through the centre); bend (R: radius / cell length, x bent in the x-z plane);
    grade (ratio: local size ratio between the faces x = 1 and x = 0); bezier (amp, seed: cubic Bezier volume
    displacement with random control vectors, amp in cell lengths)."""
    k = spec['kind']
    c = np.full(3, 0.5)
    if k == 'identity':
        return lambda X: np.array(X, dtype=float)
    if k in ('rotation', 'affine', 'scale'):
        if k == 'rotation':
            a = np.deg2rad(spec['deg']); ax = spec.get('axis', 2)
            i, j = [d for d in range(3) if d != ax]
            A = np.eye(3); A[i, i] = A[j, j] = np.cos(a); A[i, j] = -np.sin(a); A[j, i] = np.sin(a)
        elif k == 'scale':
            A = spec['s'] * np.eye(3)
        else:
            A = np.asarray(spec['A'], float)
        if np.linalg.det(A) <= 0:
            raise ValueError('MAP_NOT_ORIENTATION_PRESERVING')
        return lambda X: c + (np.asarray(X, float) - c) @ A.T
    if k == 'twist':
        rate = np.deg2rad(spec['deg'])

        def phi(X):
            X = np.asarray(X, float); th = rate * (X[:, 2] - 0.5)
            u, v = X[:, 0] - 0.5, X[:, 1] - 0.5
            return np.stack([0.5 + np.cos(th) * u - np.sin(th) * v, 0.5 + np.sin(th) * u + np.cos(th) * v, X[:, 2]], 1)
        return phi
    if k == 'bend':
        R = float(spec['R'])

        def phi(X):
            X = np.asarray(X, float); rho = R + (X[:, 2] - 0.5); a = (X[:, 0] - 0.5) / R
            return np.stack([0.5 + rho * np.sin(a), X[:, 1], 0.5 - R + rho * np.cos(a)], 1)
        return phi
    if k == 'grade':
        g = float(spec['ratio']) - 1.0

        def phi(X):
            X = np.asarray(X, float); s = 1 + g * X[:, 0]
            return np.stack([X[:, 0] + 0.5 * g * X[:, 0] ** 2, 0.5 + s * (X[:, 1] - 0.5), 0.5 + s * (X[:, 2] - 0.5)], 1)
        return phi
    if k == 'bezier':
        rng = np.random.default_rng(int(spec.get('seed', 0)))
        P = float(spec['amp']) * rng.uniform(-1, 1, (4, 4, 4, 3))
        from math import comb

        def bern(t):
            return np.stack([comb(3, i) * t ** i * (1 - t) ** (3 - i) for i in range(4)], -1)

        def phi(X):
            X = np.asarray(X, float)
            B = [bern(X[:, d]) for d in range(3)]
            return X + np.einsum('ni,nj,nk,ijkd->nd', B[0], B[1], B[2], P)
        return phi
    raise ValueError(f'MAP_KIND:{k}')


def node_xyz(node_ids, n, phi):
    """Physical coordinates (N, 3) of grid nodes (ids on the (2n+1)^3 grid) under the map."""
    X = np.stack(np.unravel_index(np.asarray(node_ids), (2 * n + 1,) * 3), 1) / (2 * n)
    return phi(X)


def rigid_basis_xyz(x):
    """Orthonormal rigid modes (6 columns, node-major xyz) on nodes with physical coordinates x (N, 3)."""
    c = x - x.mean(0)
    R = np.zeros((len(x), 3, 6))
    for a in range(3):
        R[:, a, a] = 1.0
        e = np.zeros(3); e[a] = 1.0
        R[:, :, 3 + a] = np.cross(e[None, :], c)
    Q, _ = np.linalg.qr(R.reshape(-1, 6))
    return torch.as_tensor(Q, dtype=dt, device=dev)


def rigid_raw_xyz(x, center):
    """ops.rigid_raw with physical coordinates (unnormalised rigid modes about `center`)."""
    g = x - center
    R = np.zeros((len(x), 3, 6))
    for a in range(3):
        R[:, a, a] = 1.0
        e = np.zeros(3); e[a] = 1.0
        R[:, :, 3 + a] = np.cross(e[None, :], g)
    return torch.as_tensor(R.reshape(-1, 6), dtype=dt, device=dev)


# ------------------------------------------------------------------------------------------------------ Q2 basis
def _lagrange(t):
    """1-D Q2 Lagrange values, first and second derivatives at nodes -1, 0, 1: three (..., 3) tensors."""
    L = torch.stack([0.5 * t * (t - 1), 1 - t * t, 0.5 * t * (t + 1)], -1)
    dL = torch.stack([t - 0.5, -2 * t, t + 0.5], -1)
    d2L = torch.stack([torch.ones_like(t), -2 * torch.ones_like(t), torch.ones_like(t)], -1)
    return L, dL, d2L


def q2_basis(t, xi_nodes, second=False):
    """Values N (..., 27), t-gradients dN (..., 27, 3) and optionally t-Hessians d2N (..., 27, 3, 3) at local points
    t (..., 3) in [-1, 1]^3, for the element node ordering xi_nodes (27, 3) in {-1, 0, 1}."""
    o = torch.as_tensor(np.asarray(xi_nodes) + 1, device=t.device)          # (27, 3) in {0, 1, 2}
    vals = [_lagrange(t[..., d]) for d in range(3)]                            # per axis: (L, dL, d2L), each (..., 3)
    f = [[v[k][..., o[:, d]] for k in range(3)] for d, v in enumerate(vals)]   # f[d][k]: (..., 27)
    N = f[0][0] * f[1][0] * f[2][0]
    dN = torch.stack([f[0][1] * f[1][0] * f[2][0], f[0][0] * f[1][1] * f[2][0], f[0][0] * f[1][0] * f[2][1]], -1)
    if not second:
        return N, dN
    H = torch.empty(N.shape + (3, 3), dtype=t.dtype, device=t.device)
    for i in range(3):
        for j in range(3):
            k = [0, 0, 0]; k[i] += 1; k[j] += 1
            H[..., i, j] = f[0][k[0]] * f[1][k[1]] * f[2][k[2]]
    return N, dN, H


def _adj3(J):
    """Adjugate (adj = det J * J^-1) and determinant of 3 x 3 matrices (..., 3, 3), closed form."""
    a, b, c = J[..., 0, 0], J[..., 0, 1], J[..., 0, 2]
    d, e, f = J[..., 1, 0], J[..., 1, 1], J[..., 1, 2]
    g, h, i = J[..., 2, 0], J[..., 2, 1], J[..., 2, 2]
    adj = torch.stack([torch.stack([e * i - f * h, c * h - b * i, b * f - c * e], -1),
                       torch.stack([f * g - d * i, a * i - c * g, c * d - a * f], -1),
                       torch.stack([d * h - e * g, b * g - a * h, a * e - b * d], -1)], -2)
    det = a * adj[..., 0, 0] + b * adj[..., 1, 0] + c * adj[..., 2, 0]
    return adj, det


_IDX = {}


def a_pairs(J, lam, mu):
    """The 45 entries (PAIRS) of A = det J (J^-1)_Jj C_ijkl (J^-1)_Ll for isotropic C, and det J; closed form through the
    adjugate: A_iJkL = (lam adj_Ji adj_Lk + mu delta_ik (adj adj^T)_JL + mu adj_Jk adj_Li) / det."""
    key = J.device
    if key not in _IDX:
        p = torch.tensor([p for p, _ in PAIRS], device=J.device); q = torch.tensor([q for _, q in PAIRS], device=J.device)
        _IDX[key] = (p // 3, p % 3, q // 3, q % 3)
    i, Jj, k, L = _IDX[key]
    adj, det = _adj3(J)
    G = adj @ adj.transpose(-1, -2)
    dik = (i == k).to(J.dtype)                                                 # float64 mask (a bool times a Python float is float32)
    A = lam * adj[..., Jj, i] * adj[..., L, k] + mu * adj[..., Jj, k] * adj[..., L, i] + mu * dik * G[..., Jj, L]
    return A / det[..., None], det


def a_tensor(J, lam, mu):
    """A[..., p, q] (9 x 9, p = 3 i + J, q = 3 k + L) = det J (J^-1)_Jj C_ijkl (J^-1)_Ll for isotropic C; and det J."""
    det = torch.linalg.det(J)
    Ji = torch.linalg.inv(J)                                                   # Ji[J, j]
    G = Ji @ Ji.transpose(-1, -2)                                              # G[J, L] = sum_j Ji[J, j] Ji[L, j]
    I3 = torch.eye(3, dtype=J.dtype, device=J.device)
    A = (lam * torch.einsum('...Ji,...Lk->...iJkL', Ji, Ji)
         + mu * torch.einsum('ik,...JL->...iJkL', I3, G)
         + mu * torch.einsum('...Jk,...Li->...iJkL', Ji, Ji))
    return (det[..., None, None] * A.reshape(A.shape[:-4] + (9, 9))), det


# ------------------------------------------------------------------------------------------------------ templates
def unit_templates(xi_nodes):
    """Tt (45, 125, 81, 81): K_e[a i, b k] = sum_c sum_m Mt[c, m] Tt[c, m, a i, b k] for the 45 entries (p <= q) of A,
    with Mt[c, m] = int A_c t^m dt over the material part (t-measure)."""
    N = [[EM.L1D[int(xi_nodes[a, d])] for d in range(3)] for a in range(27)]
    DN = [[EM.DL1D[int(xi_nodes[a, d])] for d in range(3)] for a in range(27)]
    P = np.zeros((27, 3, 27, 3, 5, 5, 5))                                       # coefficients of dN_a/dt_J dN_b/dt_L
    for a in range(27):
        for J in range(3):
            pa = [DN[a][d] if d == J else N[a][d] for d in range(3)]
            for b in range(27):
                for L in range(3):
                    pb = [DN[b][d] if d == L else N[b][d] for d in range(3)]
                    pr = [np.convolve(pa[d], pb[d]) for d in range(3)]
                    P[a, J, b, L, :len(pr[0]), :len(pr[1]), :len(pr[2])] = np.einsum('a,b,c->abc', *pr)
    P = P.reshape(27, 3, 27, 3, 125)
    T = np.zeros((45, 125, 27, 3, 27, 3))
    for c, (p, q) in enumerate(PAIRS):
        i, J = divmod(p, 3); k, L = divmod(q, 3)
        T[c, :, :, i, :, k] += P[:, J, :, L].transpose(2, 0, 1)
        if p != q:
            T[c, :, :, k, :, i] += P[:, L, :, J].transpose(2, 0, 1)
    return T.reshape(45, 125, 81, 81)


# ------------------------------------------------------------------------------------------------------ quadrature
def _gauss_cube(m, device):
    g, w = np.polynomial.legendre.leggauss(m)
    P = np.array(list(product(g, g, g))); W = np.prod(np.array(list(product(w, w, w))), 1)
    return torch.tensor(P, dtype=dt, device=device), torch.tensor(W, dtype=dt, device=device)  # on [-1,1]^3, sum 8


def pieces(cells, n, taus, normal, offset, s, levels=0, surface='P', device=None, rule_order=4, batch=256,
           piece_chunk=8192):
    """Positive-weight quadrature of every active element's material part, as pieces (owner, points (T, q, 3) in local
    t in [-1, 1]^3, weights (T, q) in t-measure). Same sub-grid, refinement and clipping as polyref_torch_fast.cell_moments;
    sub-cubes entirely inside: tensor 3-point Gauss (exact for the products of Q2 derivatives); elements whose s^3 base
    sub-cubes are all inside: one tensor 5-point Gauss rule. Yields lists of pieces per cell batch."""
    import surfaces as SF
    device = device or dev
    trule = tet_rule(rule_order)
    tref = torch.tensor(trule[0], dtype=dt, device=device); tw = torch.tensor(trule[1], dtype=dt, device=device)
    g3, w3 = _gauss_cube(3, device); g5, w5 = _gauss_cube(5, device)
    kuhn = torch.tensor(KUHN, device=device); cube = torch.tensor(CUBE, dtype=dt, device=device)
    taus_t = torch.tensor(np.asarray(taus, float), dtype=dt, device=device)
    nrm = None if normal is None else torch.tensor(np.asarray(normal, float), dtype=dt, device=device)
    idx = torch.tensor(list(product(range(s), repeat=3)), dtype=dt, device=device)
    cells_t = torch.as_tensor(np.asarray(cells), dtype=dt, device=device)

    def psi_at(cell_xyz, xi):                                                  # as polyref_torch_fast.cell_moments
        phys = (cell_xyz + (xi + 1) / 2) / n
        f = torch.cos(2 * torch.pi * phys).sum(-1) if surface in (None, 'P') else SF.f_torch(phys, surface)
        w8 = torch.where(cube.bool().view(8, *([1] * (phys.dim() - 1)), 3), phys[None], 1 - phys[None]).prod(-1)
        tau = torch.einsum('c,c...->...', taus_t, w8)
        p3 = (offset - phys @ nrm) if nrm is not None else torch.ones_like(f)
        return torch.stack([tau - f, tau + f, p3], -1)

    for b0 in range(0, len(cells), batch):
        cb = cells_t[b0:b0 + batch]; nb = len(cb)
        out = []
        h = 2 / s
        owner = torch.arange(nb, device=device).repeat_interleave(len(idx))
        lo = (-1 + h * idx).repeat(nb, 1)
        for level in range(levels + 1):
            cx = lo[:, None, :] + h * cube[None]
            cp = psi_at(cb[owner][:, None, :], cx)
            full = (cp >= 0).all(-1).all(-1); empty = (cp < 0).all(1).any(-1)
            part = ~full & ~empty
            if level == 0:                                                     # whole elements inside: one 5^3 rule
                nfull = torch.zeros(nb, dtype=torch.long, device=device).index_add_(0, owner, full.long())
                whole = nfull == len(idx)
                if whole.any():
                    ow = torch.nonzero(whole).squeeze(1)
                    out.append((ow + b0, g5[None].expand(len(ow), -1, -1), w5[None].expand(len(ow), -1)))
                full = full & ~whole[owner]
            if full.any():
                lf = lo[full]
                pts = lf[:, None, :] + h * (g3[None] + 1) / 2
                out.append((owner[full] + b0, pts, (h / 2) ** 3 * w3[None].expand(len(lf), -1)))
            if not part.any():
                break
            if level < levels:
                lo = (lo[part][:, None, :] + (h / 2) * cube[None]).reshape(-1, 3)
                owner = owner[part].repeat_interleave(8)
                h = h / 2
                continue
            tets = cx[part][:, kuhn].reshape(-1, 4, 3)
            attr = cp[part][:, kuhn].reshape(-1, 4, 3)
            town = owner[part].repeat_interleave(6)
            for kk in range(3):
                tets, attr, town = PT._clip(tets, attr, town, kk)
            if len(tets):
                E = torch.stack([tets[:, 1] - tets[:, 0], tets[:, 2] - tets[:, 0], tets[:, 3] - tets[:, 0]], 1)
                Jt = torch.linalg.det(E).abs()
                keep = Jt > 0
                tets, E, Jt, town = tets[keep], E[keep], Jt[keep], town[keep]
                for lo_t in range(0, len(tets), piece_chunk):
                    sl = slice(lo_t, lo_t + piece_chunk)
                    P = tets[sl, None, 0, :] + torch.einsum('qk,tkd->tqd', tref, E[sl])
                    out.append((town[sl] + b0, P, Jt[sl, None] * tw[None]))
        yield out


def mapped_moments(cells, n, taus, normal, offset, s, levels, surface, xe, xi_nodes, lam, mu, piece_chunk=4096,
                   geometric=False, stats=None):
    """Mt (E, 45, 125): sum over quadrature points of w A_c(t) t^m (t-measure; A includes det J of x_h(t)).
    xe (E, 27, 3) physical node coordinates of every element (torch, dev). geometric=True also returns the plain moments
    M (E, 125) of the same rule times (1/(2n))^3 (the physical-measure convention of cell_moments). stats (dict) collects
    min det J (all points), max condition number of J (first point of every piece) and the number of points."""
    E = len(cells)
    Mt = torch.zeros((E, 45 * 125), dtype=dt, device=dev)
    M = torch.zeros((E, 125), dtype=dt, device=dev) if geometric else None
    pi = torch.tensor([p for p, _ in PAIRS], device=dev); qi = torch.tensor([q for _, q in PAIRS], device=dev)
    dmin, kmax, npts = float('inf'), 0.0, 0
    for out in pieces(cells, n, taus, normal, offset, s, levels, surface, piece_chunk=piece_chunk):
        for own, P, W in out:
            for lo in range(0, len(own), piece_chunk):
                o, p, w = own[lo:lo + piece_chunk], P[lo:lo + piece_chunk], W[lo:lo + piece_chunk]
                _, dN = q2_basis(p, xi_nodes)                                   # (T, q, 27, 3)
                J = torch.einsum('tai,tqaj->tqij', xe[o], dN)                    # J_ij = dx_i / dt_j
                A, det = a_pairs(J, lam, mu)
                dmin = min(dmin, float(det.min()))
                sv = torch.linalg.svdvals(J[:, 0])                               # first point of every piece (cost)
                kmax = max(kmax, float((sv[..., 0] / sv[..., -1]).max()))
                npts += p.shape[0] * p.shape[1]
                Ac = A * w[..., None]                                            # (T, q, 45)
                pw = [torch.stack([torch.ones_like(p[..., d]), p[..., d], p[..., d] ** 2, p[..., d] ** 3, p[..., d] ** 4], -1)
                      for d in range(3)]                                        # (T, q, 5) each
                X = (Ac[..., :, None] * pw[0][..., None, :]).reshape(p.shape[0], p.shape[1], 45 * 5)
                YZ = (pw[1][..., :, None] * pw[2][..., None, :]).reshape(p.shape[0], p.shape[1], 25)
                Mt.index_add_(0, o, torch.bmm(X.transpose(1, 2), YZ).reshape(len(o), 45 * 125))
                if geometric:
                    Xg = pw[0] * w[..., None]
                    M.index_add_(0, o, torch.bmm(Xg.transpose(1, 2), YZ).reshape(len(o), 125))
    if stats is not None:
        stats.update(det_t_min=dmin, cond_J_max=kmax, points=npts)
    Mt = Mt.reshape(E, 45, 125)
    if geometric:
        return Mt, M * (1 / (2 * n)) ** 3
    return Mt


# ------------------------------------------------------------------------------------------------------ ghost penalty
def ghost_upper(cells, faces, dofs, xnode_e, xi_nodes, n, nb, lam0, mu0, chunk=4096):
    """Unit ghost-penalty matrix (upper triplets: keys r * nb + c with r <= c, values; coalesced) of the physical-coordinate
    form above. faces (F, 3) = (owner, neighbour, axis), neighbour = owner + e_axis; dofs (E, 81) node-major xyz; xnode_e
    (E, 27, 3) physical node coordinates."""
    g, w = np.polynomial.legendre.leggauss(3)
    G2 = torch.tensor(np.array(list(product(g, g))), dtype=dt, device=dev)    # (9, 2) tangential points
    W2 = torch.tensor(np.prod(np.array(list(product(w, w))), 1), dtype=dt, device=dev)
    faces = torch.as_tensor(np.asarray(faces, np.int64), device=dev)
    keys, vals = [], []
    acc = None

    def merge(acc, keys, vals):                                                # coalesce as we go (bounded memory)
        k = torch.cat(keys + ([acc[0]] if acc is not None else []))
        v = torch.cat(vals + ([acc[1]] if acc is not None else []))
        U, inv = torch.unique(k, return_inverse=True)
        return U, torch.zeros(len(U), dtype=dt, device=dev).index_add_(0, inv, v)
    for ax in range(3):
        tang = [d for d in range(3) if d != ax]
        u_, v_ = (1, 2) if ax == 0 else ((2, 0) if ax == 1 else (0, 1))        # x_u x x_v points along +axis (identity)
        tp = torch.zeros((9, 3), dtype=dt, device=dev); tp[:, tang] = G2
        to, tn = tp.clone(), tp.clone(); to[:, ax] = 1.0; tn[:, ax] = -1.0      # owner face t_ax = +1, neighbour -1
        _, dNo, Ho = q2_basis(to, xi_nodes, second=True)                       # (9, 27, 3), (9, 27, 3, 3)
        _, dNn, Hn = q2_basis(tn, xi_nodes, second=True)
        fa = faces[faces[:, 2] == ax]
        for lo in range(0, len(fa), chunk):
            f = fa[lo:lo + chunk]
            sides = []
            for e, dN, H in ((f[:, 0], dNo, Ho), (f[:, 1], dNn, Hn)):
                xe = xnode_e[e]                                                  # (F, 27, 3)
                J = torch.einsum('fai,qaj->fqij', xe, dN)                         # (F, 9, 3, 3)
                Ji = torch.linalg.inv(J)
                gx = torch.einsum('qaj,fqji->fqai', dN, Ji)                       # physical gradients (F, 9, 27, 3)
                x2 = torch.einsum('fak,qaij->fqkij', xe, H)                       # d2x_k / dt_i dt_j (F, 9, 3, 3, 3)
                Hc = H[None] - torch.einsum('fqak,fqkij->fqaij', gx, x2)
                Hx = torch.einsum('fqIi,fqaIJ,fqJj->fqaij', Ji, Hc, Ji)          # J^-T (...) J^-1
                sides.append((J, gx, Hx))
            J = sides[0][0]
            nv = torch.linalg.cross(J[..., :, u_], J[..., :, v_])                 # (F, 9, 3)
            area = nv.norm(dim=-1); nh = nv / area[..., None]
            hx = (1.0 / n) * torch.sqrt(torch.abs((2 * n) ** 3 * torch.linalg.det(sides[0][0]))
                                        * torch.abs((2 * n) ** 3 * torch.linalg.det(sides[1][0]))) ** (1 / 3)
            rows = []
            for j in (1, 2):
                if j == 1:
                    vo = torch.einsum('fqai,fqi->fqa', sides[0][1], nh); vn = torch.einsum('fqai,fqi->fqa', sides[1][1], nh)
                else:
                    vo = torch.einsum('fqaij,fqi,fqj->fqa', sides[0][2], nh, nh)
                    vn = torch.einsum('fqaij,fqi,fqj->fqa', sides[1][2], nh, nh)
                sw = torch.sqrt(W2[None] * area * hx ** (2 * j - 1) * (lam0 + 2 * mu0))
                rows.append(torch.cat([vo, -vn], -1) * sw[..., None])             # (F, 9, 54)
            R = torch.cat(rows, 1)                                               # (F, 18, 54)
            S = R.transpose(1, 2) @ R                                            # (F, 54, 54) scalar face matrix
            d = torch.cat([dofs[f[:, 0]], dofs[f[:, 1]]], 1).reshape(len(f), 54, 3)  # (F, 54 nodes, 3 comps)
            for comp in range(3):
                r = d[:, :, comp][:, :, None].expand(-1, 54, 54)
                c = d[:, :, comp][:, None, :].expand(-1, 54, 54)
                up = r <= c
                keys.append((r * nb + c)[up]); vals.append(S[up])
            acc = merge(acc, keys, vals); keys, vals = [], []
    return acc


# ------------------------------------------------------------------------------------------------------ the cell
class MappedCell(TE.Cell):
    """teacher.Cell with the isoparametrically mapped stiffness and rigid bases (see the module docstring)."""

    def __init__(self, case, body_dir, map_spec, ports=('box', 'cut'), s=4, levels=1, log=print):
        super().__init__(case, body_dir, ports=ports, s=s, levels=levels, log=log)
        t0 = time.perf_counter()
        self.map_spec = dict(map_spec)
        phi = make_map(map_spec)
        ctx = json.loads((TE.packet_dir(case) / 'FRESH_CONTEXT.json').read_text())
        Emod = float(ctx['material']['E']); nu = float(ctx['material']['nu'])
        self.lam, self.mu = Emod * nu / ((1 + nu) * (1 - 2 * nu)), Emod / (2 * (1 + nu))
        lam0, mu0 = 0.3 / (1.3 * 0.4), 1 / 2.6                                  # ghost templates: E_0 = 1, nu_0 = 0.3
        self.xyz = node_xyz(self.nodes, self.n, phi)                             # (N, 3) physical node coordinates
        arr = {'NODES.npy': self.nodes, 'dofs.npy': self.dofs.cpu().numpy(), 'CELL_INDICES.npy': self.cells}
        self.xi_nodes = EM.local_coordinates(case, self.n, arr)[0]
        x_t = torch.as_tensor(self.xyz, dtype=dt, device=dev)
        self.xe = x_t[self.dofs[:, ::3] // 3]                                    # (E, 27, 3)
        self.Tt_up = torch.as_tensor(unit_templates(self.xi_nodes), dtype=dt, device=dev).reshape(45 * 125, 81, 81)[
            :, self.iu[0], self.iu[1]].contiguous()
        # ghost penalty in physical coordinates; pattern rebuilt from the element keys and these ghost keys
        faces = np.load(Path(body_dir) / case / 'GP_FACES.npy')
        key_g, val_g = ghost_upper(self.cells, faces, self.dofs, self.xe, self.xi_nodes, self.n, self.nb, lam0, mu0)
        self._rebuild_pattern(key_g, val_g)
        # rigid bases from the physical node coordinates
        onport = np.isin(self.nodes, self.port_node_ids)
        self.Q = rigid_basis_xyz(self.xyz[onport])
        self.Qall = rigid_basis_xyz(self.xyz)
        self.map_seconds = time.perf_counter() - t0
        log(json.dumps(dict(event='MAPPED_SETUP', case=case, map=self.map_spec, seconds=self.map_seconds)))

    def _rebuild_pattern(self, key_g, val_g):
        nb, iu = self.nb, self.iu
        r, c = self.dofs[:, iu[0]], self.dofs[:, iu[1]]
        key_e = torch.minimum(r, c) * nb + torch.maximum(r, c)
        U = torch.unique(torch.cat([key_e.reshape(-1), key_g]))
        self.pos_e = torch.searchsorted(U, key_e.reshape(-1)).int()
        del key_e
        self.base = torch.zeros(len(U), dtype=dt, device=dev)
        self.base.index_add_(0, torch.searchsorted(U, key_g), self.gamma * val_g)
        self.ghost_unit = (key_g, val_g)
        self.ru, self.cu = (U // nb).int(), (U % nb).int()
        del U
        self.crow = torch.cat([torch.zeros(1, dtype=torch.long, device=dev), torch.cumsum(torch.bincount(self.ru, minlength=nb), 0)])
        self.diag = torch.nonzero(self.ru == self.cu).squeeze(1)
        self.tperm = torch.argsort(self.cu.long() * nb + self.ru.long()).int()
        self.crow_t = torch.cat([torch.zeros(1, dtype=torch.long, device=dev), torch.cumsum(torch.bincount(self.cu, minlength=nb), 0)]).int()
        self.col_t = self.ru[self.tperm.long()]
        gc.collect(); torch.cuda.empty_cache()
        if len(self.diag) != nb:
            raise ValueError('DIAGONAL_INCOMPLETE')

    def assemble(self, taus=None):
        taus = self.taus0 if taus is None else taus
        self.taus = list(taus)
        self._free(); self.U = self.Ut = None; gc.collect(); torch.cuda.empty_cache()
        st = {}
        Mt, self.M = mapped_moments(self.cells, self.n, taus, self.normal, self.offset, self.s, self.levels, self.surface,
                                    self.xe, self.xi_nodes, self.lam, self.mu, geometric=True, stats=st)
        self.map_stats = dict(st, det_X_min=st['det_t_min'] * (2 * self.n) ** 3)
        Mt = Mt.reshape(len(Mt), -1)
        vals = self.base.clone()
        for lo in range(0, len(Mt), 1024):
            ke = Mt[lo:lo + 1024] @ self.Tt_up
            vals.index_add_(0, self.pos_e[lo * 3321:(lo + len(ke)) * 3321].long(), ke.reshape(-1))
        del Mt
        self.vals = vals
        self.U = torch.sparse_csr_tensor(self.crow.int(), self.cu, vals, size=(self.nb, self.nb))
        self.Ut = torch.sparse_csr_tensor(self.crow_t, self.col_t, vals[self.tperm.long()], size=(self.nb, self.nb))
        self.dK = vals[self.diag]
        self.K = self
        gc.collect(); torch.cuda.empty_cache()
        return self

    def _isotropic_only(self, *a, **k):
        raise NotImplementedError('MAPPED_CELL: this path uses the isotropic Cartesian templates')

    lean = assemble_deploy = dmoments = energy_density = sens = sens2 = _isotropic_only
