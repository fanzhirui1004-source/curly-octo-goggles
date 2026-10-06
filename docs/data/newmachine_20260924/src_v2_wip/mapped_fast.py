"""Direction A: the learned operator of a mapped cell with an explicit adjoint (no autograd graph), new module, default off.

Same operator as a0_budget.BudgetGeo (V0R: co-rotated network, rigid part in physical coordinates, port values exact; then
`cycles` correction cycles (trainlib.wrap with the A3 settings, co-rotated Jacobi when corot_smooth is installed and set
on the cell); then the weak-node patch if requested):
  E q = P( wrap^N( I_P q + Pi_I [ R ext(R^T (q - RP c)) + RA c ] ) ),  c = RP^+ q,   P = patch (I - E_W A_WW^-1 E_W^T K)
  E^T y = base^T( wrap_T^N( P^T y ) ),  P^T y = y - K E_W A_WW^-1 y_W,
  base^T y = y_P + qd + RP^+T (RA^T y_I - RP^T qd),  qd = R_port ext_T(R^T y_I)       (as fastnet.FastNet.field_T)
with ext / ext_T the frozen network map of fastnet.FastNet and its explicit adjoint. S_hat q = E^T (K E q), the K product
in float64. Peak memory of one application is that of FastNet (no stored graph), against ~15 GB per 16 columns through
autograd (Geo.s_hat_apply).
rigid64 (default on since 10-06, author's decision; rigid64=False reproduces the earlier runs): the rigid split c = RP^+ q, the rigid part RA c and the port values in float64 (the network still sees
the float32 deformation part qd). With float32 rigid parts the rounding of a large rigid component (slender, soft
structures: cell traces dominated by rigid motion) enters the energy through its cross term with the small deformation,
~1e-7 |q| / |q_def| relative, and can make q^T (S_hat - S~) q slightly negative (seen: twisted trimmed beam, -5e-4).
Usage: op = MappedFastOp(g, model, cycles=2, wrap=a0_eval._Wrap(model), patch=True, rigid64=True)  (g: a0_budget.BudgetGeo set to V0R,
       model caches built for g.case, corot set on g.C if wanted, weak_patch.setup(g.C, ...) done if patch);
       op.s_hat(q), op.field(q), op.apply(q) (= s_hat, the lattice operator interface)."""
import torch

import trainlib as TL
import fastnet as FN

dt, f32 = torch.float64, torch.float32


def _patch_solve(f, r):
    if f[0] == 'dense':
        return torch.cholesky_solve(r, f[1])
    s_ = f[2]
    return s_[:, None] * f[1].solve((s_[:, None] * r).contiguous())


@torch.no_grad()
def patch_apply(C, x):
    """weak_patch.apply without autograd: x + E_W A^-1 (-(K x)_W)."""
    if getattr(C, '_wp', None) is None:
        return x
    W, f = C._wp
    x = x.to(dt)
    r = -(TL._Kc(C) @ x)[W]
    return x.index_add(0, W, _patch_solve(f, r))


@torch.no_grad()
def patch_apply_T(C, y):
    """Adjoint: y - K E_W A^-1 y_W (K, A symmetric)."""
    if getattr(C, '_wp', None) is None:
        return y
    W, f = C._wp
    y = y.to(dt)
    z = torch.zeros_like(y)
    z[W] = _patch_solve(f, y[W])
    return y - TL._Kc(C) @ z


class MappedFastOp:
    def __init__(self, g, model, cycles, wrap, patch=False, rigid64=True):
        if getattr(g, 'variant', None) not in ('V0R', 'V0Rc'):
            raise ValueError('MAPPED_FAST_NEEDS_V0R')
        self.g, self.model, self.cycles, self.wrap, self.patch = g, model, int(cycles), wrap, bool(patch)
        self.C = g.C
        self.fast = FN.FastNet(model, g)                                     # frozen network map (ext / ext_T) on g.case
        self.RP, self.RA = g.RP.to(f32), g.RA.to(f32)
        self.RPpinv = g.RPpinv.to(f32)
        self.Rport, self.Rall = g.Rport.to(f32), g.Rall.to(f32)
        self.P = g.P
        self.rigid64 = bool(rigid64)
        if self.rigid64:
            self.RP64, self.RA64, self.RPpinv64 = g.RP.to(dt), g.RA.to(dt), g.RPpinv.to(dt)

    @staticmethod
    def _rot(R, x, transpose=False):
        k = x.shape[1]
        eq = 'nji,njk->nik' if transpose else 'nij,njk->nik'
        return torch.einsum(eq, R, x.reshape(-1, 3, k)).reshape(-1, k)

    @torch.no_grad()
    def field(self, q):
        if self.rigid64:
            q64 = q.to(dt)
            c = self.RPpinv64 @ q64
            qd = (q64 - self.RP64 @ c).to(f32)
            u = self._rot(self.Rall, self.fast.ext(self._rot(self.Rport, qd, transpose=True))).to(dt) + self.RA64 @ c
            u = u.index_copy(0, self.P, q64)
        else:
            q32 = q.to(f32)
            c = self.RPpinv @ q32
            qd = q32 - self.RP @ c
            u = self._rot(self.Rall, self.fast.ext(self._rot(self.Rport, qd, transpose=True))) + self.RA @ c
            u = u.index_copy(0, self.P, q32).to(dt)
        with TL._held(self.C):
            for _ in range(self.cycles):
                u = TL.wrap(self.C, u, self.wrap)
        if self.patch:
            u = patch_apply(self.C, u)
        return u

    @torch.no_grad()
    def field_T(self, y):
        y = y.to(dt)
        if self.patch:
            y = patch_apply_T(self.C, y)
        with TL._held(self.C):
            for _ in range(self.cycles):
                y = TL.wrap_T(self.C, y, self.wrap)
        if self.rigid64:
            yP = y[self.P]
            yI = y.index_fill(0, self.P, 0.0)
            qd = self._rot(self.Rport, self.fast.ext_T(self._rot(self.Rall, yI.to(f32), transpose=True))).to(dt)
            return yP + qd + self.RPpinv64.T @ (self.RA64.T @ yI - self.RP64.T @ qd)
        y = y.to(f32)
        yP = y[self.P]
        yI = y.index_fill(0, self.P, 0.0)
        qd = self._rot(self.Rport, self.fast.ext_T(self._rot(self.Rall, yI, transpose=True)))
        return (yP + qd + self.RPpinv.T @ (self.RA.T @ yI - self.RP.T @ qd)).to(dt)

    @torch.no_grad()
    def s_hat(self, q):
        u = self.field(q)
        with TL._held(self.C):
            Ku = self.C.K @ u
        return self.field_T(Ku)

    apply = s_hat
