"""Fused hyperedge kernel (fused_hyper.py) against sparse_layers.HyperFn.
  unit [n] [out.json]      Q2 hyperedges on an n^3 element grid (27 nodes each, random 70% kept), random a, b, W, X, dY:
                           forward and the four gradients (X, a, b, W) of both float32 paths against a float64 reference,
                           plus (on a GPU) the forward+backward time of each path. n=2 runs on CPU with TRITON_INTERPRET=1.
  e2e <ckpt> <out.json> [batch] [case] [data]   one training step of the checkpoint's model on its first case (or on
                           <case> in <data>), sparse path without and with FUSED (column blocks 128 and 512): loss, every
                           parameter gradient, step time and peak memory (like t_sparse.py)."""
import sys, json, time
import torch
import sparse_layers as SL
import fused_hyper as FH


def grid_hyperedges(n, keep=0.7, seed=0):
    """Q2 elements of an n^3 grid: node ids on the (2n+1)^3 lattice, a random subset of elements, nodes renumbered."""
    g = torch.Generator().manual_seed(seed)
    m = 2 * n + 1
    o = torch.stack(torch.meshgrid(*[torch.arange(3)] * 3, indexing='ij'), -1).reshape(27, 3)
    c = torch.stack(torch.meshgrid(*[torch.arange(n) * 2] * 3, indexing='ij'), -1).reshape(-1, 3)
    c = c[torch.rand(len(c), generator=g) < keep]
    ids = ((c[:, None] + o[None]) * torch.tensor([m * m, m, 1])).sum(-1)                  # E x 27
    uniq, hn = torch.unique(ids, return_inverse=True)
    return hn, len(uniq)


def problem(n, B, F, H, dev, seed=0):
    hn, N = grid_hyperedges(n, seed=seed)
    g = torch.Generator().manual_seed(seed + 1)
    E = hn.shape[0]
    deg = torch.bincount(hn.reshape(-1), minlength=N).float()
    r = lambda *s: torch.randn(*s, generator=g)
    return dict(hn=hn.to(dev), N=N, deg=deg.to(dev), X=r(N, B, F).to(dev), a=r(E, 27, H).to(dev) / 27 ** .5,
                b=r(E, 27, H).to(dev) / 27 ** .5, W=r(H, F, F).to(dev) / F ** .5, dY=r(N, B, F).to(dev))


def run(fn, p, P, dtype):
    X, a, b, W = [p[k].to(dtype).clone().requires_grad_(True) for k in ('X', 'a', 'b', 'W')]
    Y = fn(X, a, b, W, p['deg'].to(dtype), P)
    Y.backward(p['dY'].to(dtype))
    return dict(Y=Y.detach(), X=X.grad, a=a.grad, b=b.grad, W=W.grad)


def rel(u, v):
    return float((u.double() - v.double()).norm() / v.double().norm())


def unit(n=2, out=None):
    dev = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    H, F = 4, 8 if n <= 3 else 32
    B = 3 if n <= 3 else 16                                                                  # B*F: not a multiple of 16 when small
    p = problem(n, B, F, H, dev)
    P = SL.pattern(p['hn'], H, p['N'])
    ref = run(lambda *z: SL.HyperFn.apply(*z, 2048), p, P, torch.float64)
    cus = run(lambda *z: SL.HyperFn.apply(*z, 2048), p, P, torch.float32)
    rec = dict(n=n, E=int(p['hn'].shape[0]), N=p['N'], B=B, F=F, H=H, device=str(dev), cusparse={}, fused={})
    for blk in (16, 64, 128) if n <= 3 else (64, 128, 256, 512):
        fus = run(lambda *z: FH.hyper(*z, blk=blk), p, P, torch.float32)
        rec['fused'][blk] = {k: rel(fus[k], ref[k]) for k in ref}
    rec['cusparse'] = {k: rel(cus[k], ref[k]) for k in ref}
    worst = max(max(v.values()) for v in rec['fused'].values())
    rec['ok'] = bool(worst < 10 * max(max(rec['cusparse'].values()), 1e-6))
    if dev.type == 'cuda':
        def timed(fn, reps=20):
            X, a, b, W = [p[k].clone().requires_grad_(True) for k in ('X', 'a', 'b', 'W')]
            for _ in range(3):
                fn(X, a, b, W, p['deg'], P).backward(p['dY'])
            torch.cuda.synchronize(); t = time.perf_counter()
            for _ in range(reps):
                fn(X, a, b, W, p['deg'], P).backward(p['dY'])
            torch.cuda.synchronize(); return (time.perf_counter() - t) / reps
        rec['ms_fwd_bwd'] = dict(cusparse=1e3 * timed(lambda *z: SL.HyperFn.apply(*z, 2048)),
                                 **{f'fused_{blk}': 1e3 * timed(lambda *z, b_=blk: FH.hyper(*z, blk=b_)) for blk in (64, 128, 256, 512)})
    print(json.dumps(rec, indent=1), flush=True)
    if out:
        open(out, 'w').write(json.dumps(rec, indent=1))
    return rec


def e2e(ckpt, out, batch=None, case=None, data=None):
    import os
    import numpy as np
    import trainlib as TL
    import models as MD
    dev = TL.dev
    ck = torch.load(ckpt, map_location=dev, weights_only=False)
    cfg = ck['cfg']; B = int(batch) if batch else cfg['batch']
    case = case or cfg['cases'][0]
    geo = TL.Geo(case, cfg['body'], data or cfg['data'], neumann=False, log=lambda s_: None)
    import train2 as T2
    T2.clean_banks(geo, case, lambda d_: print(json.dumps(d_, default=str), flush=True))       # drop non-finite bank samples
    margs = dict(cfg.get('model_args', {}))
    mix = {k: v for k, v in (cfg.get('mix') or {}).items() if k in ('force', 'support', 'face', 'macro', 'grf')} or dict.fromkeys(('force', 'support', 'face', 'macro', 'grf'), .2)
    mix = {k: v for k, v in mix.items() if k in geo.classes}
    q, s0 = geo.sample_with_sens(B, np.random.default_rng(0), mix)
    ok = ~torch.isnan(s0[0])
    os.environ['SENS_REASSOC'] = '1'
    rec = dict(case=case, batch=B, elements=int(len(geo.C.cells)), dofs=int(geo.nb))

    def go(fused, blk=128):
        SL.FUSED, FH.BLK = fused, blk
        torch.manual_seed(0)
        m = MD.build(cfg['model'], [geo], **dict(margs, sparse=True)).to(dev)
        (MD.load_compat(m, ck['model']) if hasattr(MD, 'load_compat') else m.load_state_dict(ck['model'], strict=False)); m.train()

        def step():
            u = geo.field(m, q)
            loss = torch.log(TL.energy(u, geo.C.K).clamp_min(1e-12)).mean()
            sh = geo.sens_hat(u[:, ok])
            loss = loss + (((sh - s0[:, ok]) ** 2).sum(0) / (s0[:, ok] ** 2).sum(0)).mean()
            m.zero_grad(set_to_none=True); loss.backward()
            return loss
        loss = float(step()); torch.cuda.synchronize()
        grads = {k: p.grad.detach().clone() for k, p in m.named_parameters() if p.grad is not None}
        for _ in range(2):
            step()
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t = time.perf_counter()
        for _ in range(5):
            step()
        torch.cuda.synchronize()
        return loss, grads, (time.perf_counter() - t) / 5, torch.cuda.max_memory_allocated() / 2 ** 30

    l0, g0, t0, m0 = go(False)
    la, ga, _, _ = go(False)                                                             # run-to-run noise of the reference
    flat = lambda g: torch.cat([g[k].reshape(-1) for k in g0])
    rec['ref_repeat_grad_rel'] = float((flat(ga) - flat(g0)).norm() / flat(g0).norm()); del ga
    rec.update(loss_ref=l0, ref_repeat_loss_rel=abs(la - l0) / abs(l0), step_s_ref=t0, peak_GB_ref=m0, fused={})
    pers = {}
    for blk in (128, 512):
        l1, g1, t1, m1 = go(True, blk)
        per = pers[blk] = {k: float((g1[k] - g0[k]).norm() / g0[k].norm().clamp_min(1e-30)) for k in g0}
        rec['fused'][blk] = dict(loss=l1, loss_rel=abs(l1 - l0) / abs(l0), grad_rel_global=float((flat(g1) - flat(g0)).norm() / flat(g0).norm()),
                                 grad_rel_worst=max(per.values()), grad_rel_worst_param=max(per, key=per.get), missing=sorted(set(g0) ^ set(g1)),
                                 step_s=t1, speedup=t0 / t1, peak_GB=m1)
        del g1
    SL.FUSED = False
    print(json.dumps(rec, indent=1), flush=True)
    open(out, 'w').write(json.dumps(dict(rec, per_param=pers), indent=1))


if __name__ == '__main__':
    a = sys.argv[1:] or ['unit']
    if a[0] == 'unit':
        unit(int(a[1]) if len(a) > 1 else 2, a[2] if len(a) > 2 else None)
    else:
        e2e(*a[1:6])
