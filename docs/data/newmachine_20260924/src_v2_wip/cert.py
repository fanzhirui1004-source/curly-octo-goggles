"""D1 residual certificate: a rigorous, label-free lower bound of the energy error of a learned field. Monitoring and
diagnostics only; never called inside a network forward (the network stays fully learned).

Field u (nb, B) on all DOFs, q = u_P its port values, u* the exact extension of q ((K u*)_I = 0, u*_P = q). Since
  min_{v_P = q} v^T K v = u^T K u - r_I^T K_II^-1 r_I,   r_I := (K u)_I,
the energy excess over the exact extension is  Delta(u) = u^T K u - q^T S q = r_I^T K_II^-1 r_I = e_I^T K_II e_I
(e = u - u*, e_P = 0, r_I = K_II e_I); with the variational readout S_hat = E_hat^T K E_hat, Delta = q^T (S_hat - S) q >= 0.
Galerkin: for any interior subspace W, Delta >= b^T G^+ b (G = W^T K_II W, b = W^T r_I), monotone in W; for one vector x,
Delta >= (x^T r_I)^2 / (x^T K_II x), with equality at x = K_II^-1 r_I.
  m = 0   Jacobi: x = D^-1 r_I, D = the 3x3 node diagonal blocks of K_II (interior DOFs are whole node-major xyz triples).
  m >= 1  block Jacobi-Krylov, per column: W = [w, (D^-1 K_II) w, ..., (D^-1 K_II)^m w], w = D^-1 r_I; a Lanczos basis with
          full reorthogonalization in the D-scaled variables (A = D^-1/2 K_II D^-1/2: W is D-orthonormal), near-dependent
          directions dropped (the column's Krylov space is then invariant), b^T G^+ b by eigh with the floor 1e-12 lambda_max.
          method='cg': the same space through m + 1 block-Jacobi PCG steps (O(1) vectors instead of m + 1).
  The reported bound is the one-vector bound of the Galerkin combination x = W G^+ b re-evaluated with a fresh K product
  (equal to b^T G^+ b in exact arithmetic): rigorous for the computed r_I whatever the rounding in the basis, G or CG.
mu_lower: the worst ratio mu = max S_hat / S >= mu(q) = e_hat / (e_hat - Delta) >= e_hat / (e_hat - Delta_lb), and
  eps(q) = Delta / q^T S q >= Delta_lb / (e_hat - Delta_lb)   (e_hat = u^T K u; e_hat - Delta_lb >= q^T S q > 0).
  For eps >> 1 the bound saturates near eff / (1 - eff), eff = Delta_lb / Delta (no upper bound of Delta is available);
  when q^T S q is known (unit-energy banks) Delta_lb / q^T S q is the sharper bound.
Everything in fp64 with the exact K (teacher.Cell: C @ x). Sparse K products of width B per call: 2 (m = 0), m + 3 (m >= 1).
"""
import time
import torch

dt = torch.float64
TINY = torch.finfo(dt).tiny


def node_blocks(C, chunk=1 << 24):
    """3x3 diagonal node blocks of K (N, 3, 3), fp64, from the upper CSR (ru, cu, vals) (= prep_data.netdata diag3)."""
    N = C.nb // 3
    D = torch.zeros((N, 3, 3), dtype=dt, device=C.vals.device)
    ru = C.ru if getattr(C, 'ru', None) is not None else \
        torch.repeat_interleave(torch.arange(C.nb, device=C.vals.device), C.crow[1:].long() - C.crow[:-1].long())
    for lo in range(0, len(C.vals), chunk):
        r, c = ru[lo:lo + chunk].long(), C.cu[lo:lo + chunk].long()
        s = (r // 3) == (c // 3)
        r, c, v = r[s], c[s], C.vals[lo:lo + chunk][s].to(dt)
        D.index_put_((r // 3, r % 3, c % 3), v, accumulate=True)
        o = r != c
        D.index_put_((c[o] // 3, c[o] % 3, r[o] % 3), v[o], accumulate=True)
    return D


class Cert:
    """Residual certificate of one cell. C: K products (C @ x, fp64; teacher.Cell), P / I: port / interior DOFs.
    diag3: optional (N, 3, 3) node blocks in NODES order (NETDATA 'diag3'); default: extracted from C (always the K in use)."""

    def __init__(self, C, P, I, diag3=None, floor=1e-12, gfloor=1e-12, tol=1e-10):
        self.C, self.P, self.I, self.nb = C, P, I, C.nb
        I3 = I.reshape(-1, 3)
        if I.numel() % 3 or not bool(((I3[:, 0] % 3 == 0) & (I3[:, 1] == I3[:, 0] + 1) & (I3[:, 2] == I3[:, 0] + 2)).all()):
            raise ValueError('INTERIOR_NOT_NODE_TRIPLES')
        self.inode = I3[:, 0] // 3
        D = torch.as_tensor(diag3, dtype=dt, device=I.device)[self.inode] if diag3 is not None else node_blocks(C)[self.inode]
        D = 0.5 * (D + D.transpose(1, 2)).to(I.device)
        lam, V = torch.linalg.eigh(D)
        fl = floor * lam[:, -1:].clamp_min(TINY)
        self.floored = int((lam < fl).sum())                             # eigenvalues raised to the floor (numerical safety)
        lam = torch.maximum(lam, fl)
        self.D = D
        self.Dmh = (V * lam[:, None, :] ** -0.5) @ V.transpose(1, 2)     # D^-1/2 (symmetric), floored
        self.Dinv = (V / lam[:, None, :]) @ V.transpose(1, 2)            # D^-1 = (D^-1/2)^2, floored
        self.gfloor, self.tol = gfloor, tol

    # ---------------------------------------------------------------- products
    def _blk(self, M, w):
        return torch.bmm(M, w.reshape(-1, 3, w.shape[1])).reshape(w.shape)

    def kii(self, w):
        """K_II w for interior vectors w (ni, B): (K [0; w])_I."""
        x = torch.zeros((self.nb, w.shape[1]), dtype=dt, device=w.device)
        x[self.I] = w
        return (self.C @ x)[self.I]

    def residual(self, u):
        """r_I = (K u)_I (fp64) of full fields u (nb, B)."""
        return (self.C @ u.to(dt).contiguous())[self.I]

    # ---------------------------------------------------------------- subspaces
    def _krylov(self, r, m):
        """Galerkin combination x = W G^+ b over the block Jacobi-Krylov space of dimension <= m + 1; also b^T G^+ b."""
        ni, B = r.shape
        rt = self._blk(self.Dmh, r)                                         # D^-1/2 r
        nr = rt.norm(dim=0)
        alive = nr > 0
        Y = torch.zeros((m + 1, ni, B), dtype=dt, device=r.device)
        Y[0] = torch.where(alive, rt / torch.where(alive, nr, 1), 0)
        G = torch.zeros((B, m + 1, m + 1), dtype=dt, device=r.device)
        for k in range(m + 1):
            Z = self._blk(self.Dmh, self.kii(self._blk(self.Dmh, Y[k])))   # A y_k
            g = (Y[:k + 1] * Z).sum(1).T
            G[:, :k + 1, k] = g; G[:, k, :k + 1] = g
            if k == m:
                break
            z0 = Z.norm(dim=0)
            for _ in range(2):                                              # full reorthogonalization (CGS2)
                Z = Z - (Y[:k + 1] * (Y[:k + 1] * Z).sum(1)[:, None, :]).sum(0)
            zn = Z.norm(dim=0)
            alive = alive & (zn > self.tol * z0)                            # near-dependent: the space is invariant, stop
            Y[k + 1] = torch.where(alive, Z / torch.where(alive, zn, 1), 0)
        b = (Y * rt).sum(1).T
        lam, V = torch.linalg.eigh(0.5 * (G + G.transpose(1, 2)))
        keep = lam > self.gfloor * lam[:, -1:]
        y = torch.where(keep, torch.einsum('bkj,bk->bj', V, b) / torch.where(keep, lam, 1), 0)
        c = torch.einsum('bkj,bj->bk', V, y)                               # G^+ b
        return self._blk(self.Dmh, (Y * c.T[:, None, :]).sum(0)), (b * c).sum(1)

    def _cg(self, r, m):
        """m + 1 block-Jacobi PCG steps on K_II x = r from x = 0 (x_{m+1} is the Galerkin solution on the same space)."""
        x = torch.zeros_like(r); s = r.clone(); z = self._blk(self.Dinv, s); p = z.clone(); rz = (s * z).sum(0)
        for k in range(m + 1):
            Ap = self.kii(p)
            pAp = (p * Ap).sum(0)
            a = torch.where(pAp > 0, rz / torch.where(pAp > 0, pAp, 1), 0)
            x = x + a * p
            if k == m:
                break
            s = s - a * Ap
            z = self._blk(self.Dinv, s)
            rz1 = (s * z).sum(0)
            p = z + torch.where(rz > 0, rz1 / torch.where(rz > 0, rz, 1), 0) * p
            rz = rz1
        return x

    def _rq(self, x, r):
        """(x^T r)^2 / (x^T K_II x), 0 where x = 0: the rigorous one-vector bound."""
        num, den = (x * r).sum(0), (x * self.kii(x)).sum(0)
        return torch.where(den > 0, num * num / torch.where(den > 0, den, 1), 0)

    @torch.no_grad()
    def bound(self, r, m=0, method='krylov'):
        """Delta_lb (B,) from the interior residual r (ni, B)."""
        if m == 0:
            return self._rq(self._blk(self.Dinv, r), r)
        x = self._cg(r, m) if method == 'cg' else self._krylov(r, m)[0]
        return self._rq(x, r)

    @torch.no_grad()
    def galerkin(self, r, W):
        """Galerkin bound over an arbitrary interior basis W (ni, k), shared by the columns of r (ni, B)."""
        W = W.to(dt)
        G = W.T @ self.kii(W)
        lam, V = torch.linalg.eigh(0.5 * (G + G.T))
        keep = lam > self.gfloor * lam[-1]
        Vk = V[:, keep] / torch.sqrt(lam[keep])[None, :]
        x = W @ (Vk @ (Vk.T @ (W.T @ r)))
        return self._rq(x, r)

    # ---------------------------------------------------------------- public
    @torch.no_grad()
    def certify(self, u, m=8, method='krylov', bs=None):
        """dict(e_hat, delta_lb, eps_lb, mu_lb, seconds) of full fields u (nb, B); bs: columns per block (memory)."""
        sync = u.is_cuda and torch.cuda.is_available()
        if sync:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        B = u.shape[1]; bs = bs or B
        eh, dl = [], []
        for j in range(0, B, bs):
            uj = u[:, j:j + bs].to(dt).contiguous()
            Ku = self.C @ uj
            eh.append((uj * Ku).sum(0))
            dl.append(self.bound(Ku[self.I], m, method))
        e_hat, dlb = torch.cat(eh), torch.cat(dl)
        den = torch.maximum(e_hat - dlb, (e_hat * 1e-15).clamp_min(TINY))  # >= q^T S q > 0 up to rounding
        eps = dlb / den
        if sync:
            torch.cuda.synchronize()
        return dict(e_hat=e_hat, delta_lb=dlb, eps_lb=eps, mu_lb=1 + eps, seconds=time.perf_counter() - t0)

    @torch.no_grad()
    def lower(self, u, m=0, method='krylov', bs=None):
        """Delta_lb (B,) fp64: rigorous lower bound of u^T K u - u_P^T S u_P."""
        B = u.shape[1]; bs = bs or B
        return torch.cat([self.bound(self.residual(u[:, j:j + bs]), m, method) for j in range(0, B, bs)])

    @torch.no_grad()
    def mu_lower(self, u, m=8, method='krylov', bs=None):
        """(mu_lb, eps_lb) (B,) each: lower bounds of the ratio S_hat / S in the direction u_P (hence of the worst ratio)
        and of the relative energy error (e_hat - q^T S q) / q^T S q."""
        c = self.certify(u, m, method, bs)
        return c['mu_lb'], c['eps_lb']


def from_geo(geo, **kw):
    return Cert(geo.C, geo.P, geo.I, **kw)


def lattice_lower(pairs, m=0, c_hat=None, **kw):
    """Lattice certificate: D = sum_c Delta_lb,c(u_c) <= C - C_hat over the learned cells c (exact cells contribute 0);
    pairs: [(cert_c, u_c)], u_c = the learned field of cell c at its lattice port values q_c = u_c[P] (nb_c, B).
    Why: with the exact lattice stiffness K = sum_c S_c (assembled) and the learned one K_hat = sum_c S_hat_c >= K,
      C = f^T K^-1 f = max_U 2 f^T U - U^T K U   (the minimum of the potential energy is -C / 2), and for any U
      2 f^T U - U^T K U = 2 f^T U - U^T K_hat U + sum_c q_c^T (S_hat_c - S_c) q_c = 2 f^T U - U^T K_hat U + sum_c Delta_c(u_c);
    at the learned solution K_hat U_hat = f the first two terms are C_hat = f^T U_hat, so C - C_hat >= sum_c Delta_c
    >= D (for an inexact U_hat, C >= 2 f^T U_hat - U_hat^T K_hat U_hat + D holds exactly). Exactly,
    C - C_hat = sum_c Delta_c + |U* - U_hat|_K^2 (cell energy excess + interface error), so D misses the interface part.
    Needs the variational readout (q^T S_hat_c q = u_c^T K_c u_c for the same field u_c). With c_hat: returns
    (D, D / (c_hat + D)), a lower bound of the relative compliance error (C - C_hat) / C, since C >= c_hat + D."""
    D = sum(cert.lower(u, m, **kw) for cert, u in pairs)
    if c_hat is None:
        return D
    c_hat = torch.as_tensor(c_hat, dtype=dt, device=D.device)
    return D, D / (c_hat + D)
