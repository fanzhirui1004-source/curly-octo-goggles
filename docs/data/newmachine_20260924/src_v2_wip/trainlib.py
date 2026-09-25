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
ALL_CLASSES += ('force_c', 'face_c', 'support_k', 'glued')         # C1 lattice-context classes (prep_geo2); absent files stay absent
ALL_CLASSES += ('support64',)       # prep_geo2 --out: the recomputed (finite) support bank under its own name; 'support' is untouched
SPLITS = ('train', 'val', 'test')


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

    def worst_ratio(self, model, k=8, tol=1e-3, max_iters=30, gen=None, start=None):
        """mu = top Ritz value of (S_hat, S) (worst direction of S^-1 S_hat): adversarial() one iteration at a time from start
        (np, k) or a random block (gen) until the top value changes by less than tol (relative) or max_iters; needs the
        Neumann factor. Returns (mu, iterations, X)."""
        X, prev, mu, it = start, None, float('nan'), 0
        for it in range(1, max_iters + 1):
            X, ritz = self.adversarial(model, k=k, iters=1, gen=gen, start=X)
            mu = float(ritz[0])
            if prev is not None and abs(mu - prev) <= tol * abs(mu):
                break
            prev = mu
        return mu, it, X

    @torch.no_grad()
    def bank_ritz(self, model, Q, F, top=8, floor=1e-3, chunk=16):
        """Factorization-free worst directions in the span of bank samples with known reactions ('bank-span Ritz'):
        Q (np, R) at unit exact energy, F = S Q. G_hat = U^T K U (U = field(Q)), G = Q^T F (exact, symmetrised; eigenvalues
        floored at floor lambda_max against near-dependent samples); the top generalized eigenvectors c of (G_hat, G) give
        X = Q c, rescaled to unit exact energy c^T G c = 1 with the unfloored G. Returns X (np, top) fp64 and its Rayleigh
        quotients X^T S_hat X / X^T S X (descending)."""
        Q = Q.to(dt)
        U = torch.cat([self.field(model, Q[:, j:j + chunk]) for j in range(0, Q.shape[1], chunk)], 1)
        Gh = gram(U, self.C.K); Gh = 0.5 * (Gh + Gh.T)
        G = Q.T @ F.to(dt); G = 0.5 * (G + G.T)
        ev, V = torch.linalg.eigh(G)
        W = V * ev.clamp_min(floor * ev.max()).rsqrt()[None, :]
        H = W.T @ Gh @ W
        lam, Z = torch.linalg.eigh(0.5 * (H + H.T))
        C = W @ Z[:, torch.argsort(lam, descending=True)[:top]]
        C = C / torch.sqrt((C * (G @ C)).sum(0).clamp_min(1e-300))[None, :]
        ritz = (C * (Gh @ C)).sum(0)
        o = torch.argsort(ritz, descending=True)
        return Q @ C[:, o], ritz[o]


# ---------------------------------------------------------------------------------------------------------------------
# Trainer v2 pieces (train3.py; sens_loss also train1.py). Additive: Geo and the functions above are unchanged.
# ---------------------------------------------------------------------------------------------------------------------
def sens_loss(sh, s0, kind='sq', delta=0.003):
    """Sensitivity loss of a batch, rho_b = |s_hat_b - s_b| / |s_b| over the 8 corners:
    'sq' mean rho^2 (train1 / train2);  'smoothl1' mean(sqrt(rho^2 + delta^2) - delta) (C2: gradient ~ rho / delta near 0)."""
    r2 = ((sh - s0) ** 2).sum(0) / (s0 ** 2).sum(0)
    if kind == 'sq':
        return r2.mean()
    if kind == 'smoothl1':
        return (torch.sqrt(r2 + delta ** 2) - delta).mean()
    raise ValueError(f'sens_loss {kind!r}')


class _Gram(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, K):
        u64 = u.to(dt)
        Ku = K @ u64
        ctx.save_for_backward(Ku)
        ctx.in_dtype = u.dtype
        return u64.T @ Ku

    @staticmethod
    def backward(ctx, g):
        (Ku,) = ctx.saved_tensors
        return (Ku @ (g + g.T)).to(ctx.in_dtype), None


def gram(u, K):
    """G_hat = u^T K u (B x B, fp64; the energies on the diagonal), differentiable in u: dG_hat -> K u (g + g^T)."""
    return _Gram.apply(u, K)


def tail_loss(Gh, G, tau=0.1, floor=1e-3):
    """C3 Ritz tail term tau logsumexp_i(log max(lambda_i, 1) / tau) (a soft max of log mu over the batch span):
    lambda = generalized eigenvalues of (G_hat, G), G_hat = U^T K U (differentiable), G = Q^T F the exact Gram (F = S Q),
    fp64, via eigh of W^T G_hat W, W = V diag(max(ev, floor ev_max))^-1/2 (G = V diag(ev) V^T). Returns (term, lambda)."""
    G = 0.5 * (G + G.T).detach()
    ev, V = torch.linalg.eigh(G)
    W = V * ev.clamp_min(floor * ev.max()).rsqrt()[None, :]
    H = W.T @ Gh @ W
    lam = torch.linalg.eigvalsh(0.5 * (H + H.T))
    return tau * torch.logsumexp(torch.log(lam.clamp_min(1.0)) / tau, 0), lam.detach()


def quota_counts(B, w, gen, systematic=False):
    """Fixed class quota of a batch: floor(w_c B) per class, the remainder r = B - sum floor(w_c B) drawn
    systematic=False (default, the original rule): without replacement with draw probabilities proportional to the fractional
      parts fr_c; the resulting inclusion probabilities are NOT fr_c (audit TRAINER-5: a 0.05 class in a 9-class mix gets ~12%
      less than its share w_c B on average);
    systematic=True: systematic (Madow) sampling, one uniform u = gen.random(): class c gets +1 when a point u + j (j = 0 ..
      r - 1) falls in its interval [F_{c-1}, F_c) of the cumulative fractional parts; the inclusion probability is exactly
      fr_c (each fr_c < 1), so every class gets E[n_c] = w_c B.
    w sums to 1."""
    x = np.asarray(w, float) * B
    n = np.floor(x).astype(np.int64)
    r = B - int(n.sum())
    if r > 0:
        fr = x - n
        if systematic:
            cf = np.cumsum(fr) * (r / fr.sum())
            cf[-1] = r                                                     # exactly r at the end (rounding)
            pts = gen.random() + np.arange(r)
            n[np.minimum(np.searchsorted(cf, pts, side='right'), len(x) - 1)] += 1
        else:
            n[gen.choice(len(x), size=r, replace=False, p=fr / fr.sum())] += 1
    return n


def conv_precision():
    """Convolution precision in force (INVARIANTS-2), for the output JSON of trainers and evaluation scripts:
    conv_tf32 = torch.backends.cudnn.allow_tf32 (True: cuDNN may use TF32 kernels, PyTorch's default; False: true fp32, set by
    env OPL_CONV_FP32=1 at `import models` or the trainers' cfg conv_fp32), and the env value."""
    import os
    return dict(conv_tf32=bool(torch.backends.cudnn.allow_tf32), OPL_CONV_FP32=os.environ.get('OPL_CONV_FP32'))


class HostBanks:
    """The banks of one geometry kept on the HOST (CPU, pageable): only a sampled batch goes to the device.
    q[split][cls]   (np, m) fp32 tensor (taken from a Geo / slot cache) or (m, np) np.memmap (data dir, one row per sample)
    s[split][cls]   (8, m) fp64 exact sensitivities;   F[split][cls]  exact port reactions S q, same layout as q
    keep[split][cls]  indices of the usable samples (None: all); sample j of a bank is column / row keep[j]."""

    def __init__(self, q, s=None, F=None):
        self.q, self.s, self.F = q, s or {}, F or {}
        self.keep = {sp: {c: None for c in d} for sp, d in q.items()}
        self.classes = [c for c in ALL_CLASSES if c in q.get('train', {})]

    @classmethod
    def from_geo(cls, geo):
        """Take geo.banks / geo.sens to the host (the Geo keeps none: geo.banks = geo.sens = None)."""
        q = {sp: {c: v.cpu() for c, v in d.items()} for sp, d in geo.banks.items()}
        s = {sp: {c: v.cpu() for c, v in d.items()} for sp, d in geo.sens.items()} if geo.sens is not None else {}
        geo.banks = geo.sens = None
        return cls(q, s)

    @classmethod
    def from_data(cls, d, classes=ALL_CLASSES, splits=SPLITS):
        """Banks of a data dir ({split}_{cls}.npy m x np, {split}_{cls}_sens.npy m x 8, {split}_{cls}_F.npy m x np):
        q and F memory-mapped (np.load mmap_mode='r'), sensitivities loaded."""
        d = Path(d)
        cl = [c for c in classes if (d / f'train_{c}.npy').exists()]
        ex = lambda n_: (d / n_).exists()
        q = {sp: {c: np.load(d / f'{sp}_{c}.npy', mmap_mode='r') for c in cl if ex(f'{sp}_{c}.npy')} for sp in splits}
        s = {sp: {c: torch.as_tensor(np.load(d / f'{sp}_{c}_sens.npy')).T.contiguous() for c in q[sp] if ex(f'{sp}_{c}_sens.npy')}
             for sp in splits}
        F = {sp: {c: np.load(d / f'{sp}_{c}_F.npy', mmap_mode='r') for c in q[sp] if ex(f'{sp}_{c}_F.npy')} for sp in splits}
        for sp in splits:
            for c in F[sp]:
                if F[sp][c].shape != q[sp][c].shape:
                    raise ValueError(f'F_SHAPE {d.name} {sp}/{c}: {F[sp][c].shape} vs {q[sp][c].shape}')
            for c in s[sp]:
                if s[sp][c].shape[1] != q[sp][c].shape[0]:
                    raise ValueError(f'SENS_SHAPE {d.name} {sp}/{c}')
        return cls(q, s, F)

    @staticmethod
    def _n(a):
        return a.shape[0] if isinstance(a, np.ndarray) else a.shape[1]

    @staticmethod
    def _take(a, idx):
        """Samples idx of a host bank as a contiguous (rows, k) host tensor."""
        if isinstance(a, np.ndarray):
            return torch.from_numpy(np.ascontiguousarray(np.asarray(a[idx]).T))
        return a[:, torch.as_tensor(idx, dtype=torch.long)]

    @staticmethod
    def _finite(a, chunk=64):
        if isinstance(a, np.ndarray):
            return np.concatenate([np.isfinite(a[lo:lo + chunk]).all(1) for lo in range(0, a.shape[0], chunk)] or [np.ones(0, bool)])
        return torch.isfinite(a).all(0).numpy()

    def m(self, sp, c):
        k = self.keep[sp][c]
        return self._n(self.q[sp][c]) if k is None else len(k)

    def has_F(self, sp, c):
        return c in self.F.get(sp, {})

    def get(self, sp, c, j, F=False):
        """Samples j of bank (sp, c) on the device: q (np, k) fp32, s (8, k) fp64 or None, F (np, k) fp64 or None."""
        k = self.keep[sp][c]
        i = np.asarray(j, dtype=np.int64) if k is None else k[np.asarray(j, dtype=np.int64)]
        q = self._take(self.q[sp][c], i).to(dev).to(torch.float32)
        s = self._take(self.s[sp][c], i).to(dev) if c in self.s.get(sp, {}) else None
        f = self._take(self.F[sp][c], i).to(dev).to(dt) if F and self.has_F(sp, c) else None
        return q, s, f

    def chunks(self, sp, c, chunk):
        m = self.m(sp, c)
        for lo in range(0, m, chunk):
            q, s, _ = self.get(sp, c, np.arange(lo, min(lo + chunk, m)))
            yield q, s

    def clean(self, case=None, log=None, min_keep=8, memo=None):
        """train2.clean_banks on the host: drop samples whose q, exact sensitivities or reactions are non-finite; a class left
        with fewer than min_keep samples in some split is removed. Returns the dropped counts. memo (dict, keyed by case):
        reuse an earlier result for the same banks (memory-mapped banks are then scanned once per run, not per load)."""
        if memo is not None and case in memo:
            keep, classes, dropped = memo[case]
            for c in [c for c in self.classes if c not in classes]:
                for d_ in (self.q, self.s, self.F, self.keep):
                    for sp in d_:
                        d_[sp].pop(c, None)
            self.classes = list(classes)
            for sp, d_ in keep.items():
                self.keep[sp].update(d_)
            return dropped
        dropped = {}
        for c in list(self.classes):
            sps = [sp for sp in self.q if c in self.q[sp]]
            for sp in sps:
                ok = self._finite(self.q[sp][c])
                if c in self.s.get(sp, {}):
                    ok &= self._finite(self.s[sp][c])
                if self.has_F(sp, c):
                    ok &= self._finite(self.F[sp][c])
                if not ok.all():
                    dropped[f'{sp}/{c}'] = int((~ok).sum())
                    self.keep[sp][c] = np.flatnonzero(ok)
            if min(self.m(sp, c) for sp in sps) < min_keep:
                self.classes.remove(c)
                for d_ in (self.q, self.s, self.F, self.keep):
                    for sp in d_:
                        d_[sp].pop(c, None)
                dropped[c] = 'class removed'
        if dropped and log is not None:
            log(dict(event='BANK_CLEAN', case=case, dropped=dropped))
        if memo is not None:
            memo[case] = ({sp: {c: k for c, k in d_.items() if k is not None} for sp, d_ in self.keep.items()}, list(self.classes), dropped)
        return dropped

    def sample(self, B, gen, mix, adv=None, quota=False, want_F=False, split='train', quota_systematic=False):
        """Batch of B directions like Geo.sample_with_sens (the same generator calls: class counts, indices per class,
        random signs), from the host banks and adv (np, k) host buffer: q (np, B) fp32 and s (8, B) fp64 on the device (NaN
        columns: adversarial / no labels), kinds (class per column), F (np, B) fp64 (NaN adversarial columns; None unless
        want_F and every non-adversarial column has reactions). quota: fixed class counts (quota_counts; quota_systematic: its
        systematic remainder)."""
        names = [k for k, w in mix.items() if w > 0 and (k != 'adv' or adv is not None) and (k == 'adv' or k in self.classes)]
        w = np.asarray([mix[k] for k in names], float)
        counts = quota_counts(B, w / w.sum(), gen, quota_systematic) if quota else gen.multinomial(B, w / w.sum())
        cols, sc, fc, kinds = [], [], [], []
        for i, k in enumerate(names):
            m_ = int(counts[i])
            if m_ == 0:
                continue
            nan = torch.full((8, m_), float('nan'), dtype=dt, device=dev)
            if k == 'adv':
                j = gen.integers(0, adv.shape[1], m_)
                cols.append(adv[:, torch.as_tensor(j)].to(dev).to(torch.float32)); sc.append(nan); fc.append('adv')
            else:
                j = gen.integers(0, self.m(split, k), m_)
                q, s, f = self.get(split, k, j, F=want_F)
                cols.append(q); sc.append(nan if s is None else s); fc.append(f)
            kinds += [k] * m_
        q = torch.cat(cols, 1); s = torch.cat(sc, 1)
        sg = torch.as_tensor(gen.choice([-1.0, 1.0], q.shape[1]), dtype=torch.float32, device=dev)
        F = None
        if want_F and all(f is not None for f in fc):
            F = torch.cat([torch.full((q.shape[0], c_.shape[1]), float('nan'), dtype=dt, device=dev) if isinstance(f, str) else f
                           for f, c_ in zip(fc, cols)], 1) * sg.to(dt)[None, :]
        return q * sg[None, :], s, kinds, F                                    # sensitivities are even in q
