"""Direction A: co-rotated Jacobi preconditioner for the equilibrium correction (default off; new module, trainlib unchanged).

The correction of P1 (trainlib.tail_bounds / _cheb / smooth_tail / smooth_tail_T) uses the scalar diagonal D = diag(K_II).
On a mapped cell the diagonal of a nodal 3x3 block depends on the orientation of the Cartesian axes, so the correction is not
equivariant under rotations even when the network input is pulled back (V0R). Here D is replaced, on cells where
set_corot() was called, by the symmetric block-diagonal

    M_n = R_n diag(R_n^T B_n R_n) R_n^T,   B_n = nodal 3x3 diagonal block of K~,  R_n = nodal polar factor of dx/dX,

i.e. point Jacobi in the local (pulled-back) frame of each node. Properties:
  * identity map (R = I): M = D, the P1 correction (up to the rounding of R);
  * rigid rotation x = R X: K~ = R K R^T, M = R D R^T, and the power-estimate start vector is rotated by R, so the
    Chebyshev interval, every smoothing step and (with the rotation-invariant Q1 coarse span) the whole correction are
    conjugate to those of the reference cell: V0R with this preconditioner reproduces the identity map exactly;
  * M is SPD, so the smoothing is D-Chebyshev with a different SPD preconditioner: linearity, symmetry of S_hat,
    S_hat >= S~ and energy-norm non-expansion when the interval contains the spectrum all hold as in P1.
install() patches the four trainlib functions; cells without set_corot() call the original functions unchanged."""
import torch

import trainlib as TL

dev, dt = TL.dev, TL.dt
_ORIG = {}


def nodal_blocks(C):
    """B (N, 3, 3) fp64: nodal diagonal blocks of C.K (as a0_eval.write_data builds diag3)."""
    N = len(C.nodes)
    B = torch.zeros((N, 3, 3), dtype=dt, device=dev)
    ru, cu = C.ru.long(), C.cu.long()
    same = (ru // 3) == (cu // 3)
    r, c, v = ru[same], cu[same], C.vals[same]
    B.index_put_((r // 3, r % 3, c % 3), v, accumulate=True)
    off = r != c
    B.index_put_((c[off] // 3, c[off] % 3, r[off] % 3), v[off], accumulate=True)
    return B


def set_corot(C, R):
    """Use the co-rotated Jacobi preconditioner on cell C; R (N, 3, 3) nodal rotations (any float dtype)."""
    R = R.to(dt)
    B = nodal_blocks(C)
    d = torch.diagonal(R.transpose(1, 2) @ B @ R, dim1=1, dim2=2)              # (N, 3) local-frame diagonal
    C._cr_R = R
    C._cr_M = R @ torch.diag_embed(d) @ R.transpose(1, 2)
    C._cr_Minv = R @ torch.diag_embed(1 / d) @ R.transpose(1, 2)
    C._tail_bounds = None                                                       # interval depends on the preconditioner


def clear_corot(C):
    for k in ('_cr_R', '_cr_M', '_cr_Minv'):
        if hasattr(C, k):
            delattr(C, k)
    C._tail_bounds = None


def _on(C):
    return getattr(C, '_cr_Minv', None) is not None


def _blk(C, T, z):
    """Apply nodal 3x3 blocks T (N, 3, 3) to an interior vector z (ni, B); interior dofs are whole nodes."""
    full = torch.zeros((C.nb, z.shape[1]), dtype=z.dtype, device=z.device); full[C.I] = z
    out = torch.einsum('nij,njb->nib', T.to(z.dtype), full.reshape(-1, 3, z.shape[1])).reshape(C.nb, -1)
    return out[C.I]


def tail_bounds(C, alpha):
    if not _on(C):
        return _ORIG['tail_bounds'](C, alpha)
    t = getattr(C, '_tail_bounds', None)
    if t is not None and t[0] == alpha:
        return t[1], t[2]
    g = torch.Generator(device=dev); g.manual_seed(0)
    v = torch.randn((C.ni, 1), dtype=dt, device=dev, generator=g)
    v = _blk(C, C._cr_R, v)                                                     # same start in the local frame
    x = torch.zeros((C.nb, 1), dtype=dt, device=dev)
    lam = 1.0
    hold = C.hold64() if hasattr(C, 'hold64') else TL.contextlib.nullcontext()
    with torch.no_grad(), hold:
        for _ in range(40):                                                     # as trainlib (Euclidean steps are
            v = v / v.norm(); x[C.I] = v                                        # rotation invariant), M^-1 for D^-1
            w = _blk(C, C._cr_Minv, (C.K @ x)[C.I])
            lam = float((v * w).sum()); v = w
    lmax = 1.05 * lam
    C._tail_bounds = (alpha, lmax / alpha, lmax)
    return lmax / alpha, lmax


def _cheb(C, z0, b, k, alpha):
    if not _on(C):
        return _ORIG['_cheb'](C, z0, b, k, alpha)
    lmin, lmax = tail_bounds(C, alpha)
    theta, delta = (lmax + lmin) / 2, (lmax - lmin) / 2
    sigma = theta / delta; rho = 1 / sigma
    full = torch.zeros((C.nb, z0.shape[1]), dtype=dt, device=z0.device)

    def A(z):
        full.zero_(); full[C.I] = z
        return _blk(C, C._cr_Minv, TL.FI.rows(TL._Kc(C) @ full, C.I))
    z = z0.clone()
    d = (b - A(z)) / theta
    for i in range(k):
        z = z + d
        if i == k - 1:
            break
        rho_n = 1 / (2 * sigma - rho)
        d = rho_n * rho * d + (2 * rho_n / delta) * (b - A(z))
        rho = rho_n
    return z


def smooth_tail(C, u, k, alpha=30.0):
    if not _on(C):
        return _ORIG['smooth_tail'](C, u, k, alpha)
    lmin, lmax = tail_bounds(C, alpha)
    theta, delta = (lmax + lmin) / 2, (lmax - lmin) / 2
    sigma = theta / delta; rho = 1 / sigma
    x = u.to(dt)
    r = -TL.FI.rows(TL._KMat.apply(x, TL._Kc(C)), C.I)
    d = _blk(C, C._cr_Minv, r) / theta
    for i in range(k):
        x = x.index_add(0, C.I, d)
        if i == k - 1:
            break
        r = -TL.FI.rows(TL._KMat.apply(x, TL._Kc(C)), C.I)
        rho_n = 1 / (2 * sigma - rho)
        d = rho_n * rho * d + (2 * rho_n / delta) * _blk(C, C._cr_Minv, r)
        rho = rho_n
    return x.to(u.dtype)


@torch.no_grad()
def smooth_tail_T(C, y, k, alpha=30.0):
    """Adjoint of smooth_tail with M symmetric: w = M^-1 y_I, (T^T y)_I = M p_k(A) w, (T^T y)_P = y_P - K_PI s_k(A) w."""
    if not _on(C):
        return _ORIG['smooth_tail_T'](C, y, k, alpha)
    y = y.to(dt)
    w = _blk(C, C._cr_Minv, TL.FI.rows(y, C.I))
    zero = torch.zeros_like(w)
    out = y.clone()
    out[C.I] = _blk(C, C._cr_M, _cheb(C, w, zero, k, alpha))
    sw = _cheb(C, zero, w, k, alpha)
    full = torch.zeros_like(y); full[C.I] = sw
    out[C.P] = TL.FI.rows(y, C.P) - TL.FI.rows(TL._Kc(C) @ full, C.P)
    return out


def install():
    """Patch trainlib (module globals, so wrap() and wrap_T() pick the new functions up); idempotent."""
    if _ORIG:
        return
    for name, fn in (('tail_bounds', tail_bounds), ('_cheb', _cheb), ('smooth_tail', smooth_tail),
                     ('smooth_tail_T', smooth_tail_T)):
        _ORIG[name] = getattr(TL, name)
        setattr(TL, name, fn)
