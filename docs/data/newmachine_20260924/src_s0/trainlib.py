"""Step 1 training scaffold for a learned extension (architecture-agnostic).

A model is any torch.nn.Module with forward(geo, qd) -> field on all active DOFs (nb, B), float32, LINEAR in qd, where
qd (np, B) is the deformation part of the port data (rigid part removed). The wrapper enforces what must be exact:
  u = RA c + model(geo, qd) with u[P] overwritten by q      (exact port values, exact rigid motion; still linear)
Energy readout in fp64 with the exact K: e_hat(q) = u^T K u >= q^T S q; the banks are normalized to q^T S q = 1, so
the per-sample loss e_hat - 1 >= 0 is the relative energy error of the extension in that direction (= mu - 1).
Adversarial directions: block power iteration on the pencil (S_hat, S): q <- S^+ S_hat q, S-normalized, Rayleigh-Ritz.
"""
import json, time, gc, math
from pathlib import Path
import numpy as np
import torch
import teacher as TE
import ops as OP

dev, dt = TE.dev, TE.dt
CLASSES = ('force', 'macro', 'grf')
ALL_CLASSES = ('force', 'macro', 'grf', 'support', 'face')


class _Energy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, K):
        u64 = u.to(dt)
        Ku = K @ u64
        ctx.save_for_backward(Ku)
        ctx.in_dtype = u.dtype
        return (u64 * Ku).sum(0)

    @staticmethod
    def backward(ctx, g):
        (Ku,) = ctx.saved_tensors
        return (2 * Ku * g[None, :]).to(ctx.in_dtype), None


def energy(u, K):
    return _Energy.apply(u, K)


class _Sens(torch.autograd.Function):
    """s[c, b] = -u_b^T (dK/dtau_c) u_b from element moments derivatives (chunked, float32 products, float64 sums)."""

    @staticmethod
    def forward(ctx, u, dofs, Tm, dM, chunk):
        B = u.shape[1]
        s = torch.zeros((dM.shape[0], B), dtype=dt, device=dev)
        for lo in range(0, dofs.shape[0], chunk):
            ue = u[dofs[lo:lo + chunk]]                                              # c x 81 x B
            z = torch.einsum('mij,ejb->emib', Tm, ue)
            g = torch.einsum('emib,eib->emb', z, ue)
            s -= torch.einsum('cem,emb->cb', dM[:, lo:lo + chunk], g).to(dt)
        ctx.save_for_backward(u); ctx.dofs, ctx.Tm, ctx.dM, ctx.chunk = dofs, Tm, dM, chunk
        return s

    @staticmethod
    def backward(ctx, gs):
        (u,) = ctx.saved_tensors
        grad = torch.zeros_like(u)
        gs = gs.to(u.dtype)
        for lo in range(0, ctx.dofs.shape[0], ctx.chunk):
            dd = ctx.dofs[lo:lo + ctx.chunk]
            ue = u[dd]
            W = torch.einsum('cb,cem->emb', gs, ctx.dM[:, lo:lo + ctx.chunk])
            z = torch.einsum('mij,ejb->emib', ctx.Tm, ue)
            v = torch.einsum('emb,emib->eib', W, z)
            grad.index_add_(0, dd.reshape(-1), (-2 * v).reshape(-1, u.shape[1]))
        return grad, None, None, None, None


class _Sens2(torch.autograd.Function):
    """Same as _Sens, reassociated: per element chunk A_ce = sum_m dM_cem Tm_m (8 x 81 x 81, float32), then
    s[c, b] = -sum_e u_eb^T A_ce u_eb (8 instead of 125 contractions per field; A rebuilt per chunk, not stored)."""

    @staticmethod
    def forward(ctx, u, dofs, Tm, dM, chunk):
        B = u.shape[1]
        s = torch.zeros((dM.shape[0], B), dtype=dt, device=dev)
        for lo in range(0, dofs.shape[0], chunk):
            A = torch.einsum('cem,mij->ceij', dM[:, lo:lo + chunk], Tm)
            ue = u[dofs[lo:lo + chunk]]                                              # c x 81 x B
            z = torch.einsum('ceij,ejb->ceib', A, ue)
            s -= (z * ue[None]).sum((1, 2)).to(dt)
        ctx.save_for_backward(u); ctx.dofs, ctx.Tm, ctx.dM, ctx.chunk = dofs, Tm, dM, chunk
        return s

    @staticmethod
    def backward(ctx, gs):
        (u,) = ctx.saved_tensors
        grad = torch.zeros_like(u)
        gs = gs.to(u.dtype)
        for lo in range(0, ctx.dofs.shape[0], ctx.chunk):
            dd = ctx.dofs[lo:lo + ctx.chunk]
            A = torch.einsum('cem,mij->ceij', ctx.dM[:, lo:lo + ctx.chunk], ctx.Tm)
            ue = u[dd]
            z = torch.einsum('ceij,ejb->ceib', A, ue)                                 # A symmetric
            v = torch.einsum('cb,ceib->eib', gs, z)
            grad.index_add_(0, dd.reshape(-1), (-2 * v).reshape(-1, u.shape[1]))
        return grad, None, None, None, None


class Geo:
    """Everything one geometry contributes to training: exact K, ports, rigid split, banks, network input data."""

    def __init__(self, case, body_dir, data_root, neumann=True, log=print, cell=None, load_banks=True):
        t0 = time.perf_counter()
        self.case = case
        C = cell if cell is not None else TE.Cell(case, body_dir, log=lambda s_: None)
        if cell is None:
            C.assemble()
        if neumann:
            C.factor(neumann=True, interior=False, fp32_neumann=True)   # only an approximate S^+ (adversarial search)
        self.C = C
        d = Path(data_root) / case
        self.nd = dict(np.load(d / 'NETDATA.npz'))
        ports = json.loads((d / 'PORTS.json').read_text())
        if not np.array_equal(np.asarray(ports['port_node_ids']), C.port_node_ids):
            raise ValueError('PORT_ORDER')
        self.classes = [c for c in ALL_CLASSES if (d / f'train_{c}.npy').exists()]
        self.banks = {s: {c: torch.as_tensor(np.load(d / f'{s}_{c}.npy'), device=dev).T.contiguous() for c in self.classes}
                      for s in ('train', 'val', 'test')} if load_banks else None
        g = np.stack(np.unravel_index(C.port_node_ids, (2 * C.n + 1,) * 3), 1) / (2 * C.n)
        ctr = g.mean(0)
        self.RP = OP.rigid_raw(C.port_node_ids, C.n, ctr); self.RA = OP.rigid_raw(C.nodes, C.n, ctr)
        self.RPpinv = torch.linalg.pinv(self.RP)
        self.P, self.I = C.P, C.I
        self.np_, self.nb = C.np_, C.nb
        self.adv = None
        self.sens = None
        if load_banks and (d / 'train_force_sens.npy').exists():
            self.sens = {s_: {c: torch.as_tensor(np.load(d / f'{s_}_{c}_sens.npy'), device=dev).T.contiguous() for c in self.classes
                              if (d / f'{s_}_{c}_sens.npy').exists()}
                         for s_ in ('train', 'val', 'test')}
            C.dmoments()
            self.dM32 = C.dM.to(torch.float32); self.Tm32 = C.Tm.to(torch.float32)
        self.setup_seconds = time.perf_counter() - t0
        log(json.dumps(dict(event='GEO', case=case, ports=self.np_, dofs=self.nb, seconds=self.setup_seconds)))

    def field(self, model, q):
        """Full extension (fp32 network part, rigid part and port values exact)."""
        q32 = q.to(torch.float32)
        c = (self.RPpinv.to(torch.float32) @ q32)
        qd = q32 - self.RP.to(torch.float32) @ c
        u = model(self, qd)
        u = u + self.RA.to(torch.float32) @ c
        u = u.index_copy(0, self.P, q32)                                  # exact port values
        return u

    def sens_hat(self, u, chunk=256):
        import os
        if os.environ.get('SENS_REASSOC') == '1':
            return _Sens2.apply(u, self.C.dofs, self.Tm32, self.dM32, chunk)
        return _Sens.apply(u, self.C.dofs, self.Tm32, self.dM32, chunk)

    def sample_with_sens(self, B, gen, mix):
        """Like sample(), plus exact sensitivities (8, B) where available (NaN columns for adversarial directions)."""
        names = [k for k, w in mix.items() if w > 0 and (k != 'adv' or self.adv is not None) and (k == 'adv' or k in self.classes)]
        w = np.asarray([mix[k] for k in names], float)
        counts = gen.multinomial(B, w / w.sum())
        cols, sc = [], []
        for i, k in enumerate(names):
            m = int(counts[i])
            if m == 0:
                continue
            bank = self.adv if k == 'adv' else self.banks['train'][k]
            j = torch.as_tensor(gen.integers(0, bank.shape[1], m), device=dev)
            cols.append(bank[:, j].to(torch.float32))
            if k == 'adv' or self.sens is None or k not in self.sens['train']:
                sc.append(torch.full((8, m), float('nan'), dtype=dt, device=dev))
            else:
                sc.append(self.sens['train'][k][:, j])
        q = torch.cat(cols, 1); s = torch.cat(sc, 1)
        sg = torch.as_tensor(gen.choice([-1.0, 1.0], q.shape[1]), dtype=torch.float32, device=dev)
        return q * sg[None, :], s                                               # sensitivities are even in q

    def sample(self, B, gen, mix):
        """Batch of B training directions: class mix (dict class -> weight, incl. 'adv'); gen: numpy Generator."""
        names = [k for k, w in mix.items() if w > 0 and (k != 'adv' or self.adv is not None) and (k == 'adv' or k in self.classes)]
        w = np.asarray([mix[k] for k in names], float)
        counts = gen.multinomial(B, w / w.sum())
        cols = []
        for i, k in enumerate(names):
            m = int(counts[i])
            if m == 0:
                continue
            bank = self.adv if k == 'adv' else self.banks['train'][k]
            j = torch.as_tensor(gen.integers(0, bank.shape[1], m), device=dev)
            cols.append(bank[:, j].to(torch.float32))
        q = torch.cat(cols, 1)
        s = torch.as_tensor(gen.choice([-1.0, 1.0], q.shape[1]), dtype=torch.float32, device=dev)
        return q * s[None, :]

    def s_hat_apply(self, model, q):
        """S_hat q = E_hat^T K E_hat q (fp64) through the autograd adjoint of the linear model."""
        qq = q.detach().to(torch.float32).requires_grad_(True)
        with torch.enable_grad():
            u = self.field(model, qq)
            e = energy(u, self.C.K)
            g = torch.autograd.grad(e.sum(), qq)[0]
        return 0.5 * g.to(dt)

    @torch.no_grad()
    def evaluate(self, model, split='val', chunk=32):
        out = {}
        for c in self.classes:
            Q = self.banks[split][c]
            errs = torch.cat([energy(self.field(model, Q[:, j:j + chunk]), self.C.K) - 1 for j in range(0, Q.shape[1], chunk)])
            e = errs.cpu().numpy()
            out[c] = dict(mean=float(e.mean()), p90=float(np.quantile(e, .9)), max=float(e.max()), min=float(e.min()))
            if self.sens is not None and c in self.sens[split]:
                S = self.sens[split][c]
                se = []
                for j in range(0, Q.shape[1], chunk):
                    sh = self.sens_hat(self.field(model, Q[:, j:j + chunk]))
                    s0 = S[:, j:j + chunk]
                    se.append((sh - s0).norm(dim=0) / s0.norm(dim=0))
                se = torch.cat(se).cpu().numpy()
                out[c].update(sens_mean=float(se.mean()), sens_p90=float(np.quantile(se, .9)), sens_max=float(se.max()))
        return out

    def adversarial(self, model, k=16, iters=8, gen=None, start=None):
        """Worst directions of S^-1 S_hat by block power iteration with S-orthonormalization (Rayleigh-Ritz)."""
        C = self.C
        X = start if start is not None else torch.randn((self.np_, k), dtype=dt, device=dev, generator=gen)
        X = X - C.Q @ (C.Q.T @ X)
        ritz = None
        for _ in range(iters):
            Y = self.s_hat_apply(model, X)                                # S_hat X
            X = C.neumann(Y)                                              # S^+ S_hat X
            # S-orthonormalize via the exact energy Gram matrix: G = X^T S X, with S X = S S^+ S_hat X_prev = Y_eq
            G = X.T @ (Y - C.Q @ (C.Q.T @ Y))
            G = 0.5 * (G + G.T)
            ev, V = torch.linalg.eigh(G)
            keep = ev > ev.max() * 1e-12
            X = X @ (V[:, keep] / torch.sqrt(ev[keep])[None, :])
            H = X.T @ self.s_hat_apply(model, X)
            H = 0.5 * (H + H.T)
            ritz, W = torch.linalg.eigh(H)
            X = X @ W
        order = torch.argsort(ritz, descending=True)
        return X[:, order], ritz[order]
