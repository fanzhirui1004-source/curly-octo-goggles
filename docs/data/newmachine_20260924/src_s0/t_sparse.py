"""Sparse training layers vs the einsum path: loss, all parameter gradients, step time, peak memory (same weights, same batch).
Usage: t_sparse.py <checkpoint.pt> <out.json> [batch]"""
import sys, json, time, os
import numpy as np
import torch
import trainlib as TL
import models as MD

dev = TL.dev
ck = torch.load(sys.argv[1], map_location='cuda:0', weights_only=False)
cfg = ck['cfg']; B = int(sys.argv[3]) if len(sys.argv) > 3 else cfg['batch']
geo = TL.Geo(cfg['cases'][0], cfg['body'], cfg['data'], neumann=False, log=lambda s_: None)
margs = dict(cfg.get('model_args', {}))
q, s0 = geo.sample_with_sens(B, np.random.default_rng(0), {k: v for k, v in cfg['mix'].items() if k != 'adv'})
ok = ~torch.isnan(s0[0])
os.environ['SENS_REASSOC'] = '1'
rec = dict(batch=B)


def run(sparse):
    torch.manual_seed(0)
    m = MD.build(cfg['model'], [geo], **dict(margs, sparse=sparse)).to(dev)
    m.load_state_dict(ck['model'], strict=False); m.train()

    def step():
        u = geo.field(m, q)
        e = TL.energy(u, geo.C.K)
        loss = torch.log(e.clamp_min(1e-12)).mean()
        sh = geo.sens_hat(u[:, ok])
        loss = loss + (((sh - s0[:, ok]) ** 2).sum(0) / (s0[:, ok] ** 2).sum(0)).mean()
        m.zero_grad(set_to_none=True); loss.backward()
        return loss
    loss = step()
    torch.cuda.synchronize()
    grads = {n: p.grad.detach().clone() for n, p in m.named_parameters() if p.grad is not None}
    for _ in range(2):
        step()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t = time.perf_counter()
    for _ in range(5):
        step()
    torch.cuda.synchronize()
    return float(loss), grads, (time.perf_counter() - t) / 5, torch.cuda.max_memory_allocated() / 2 ** 30


l0, g0, t0, m0 = run(False)
la, ga, _, _ = run(False)                                                  # the einsum path again (atomics: run-to-run noise)
flat0 = torch.cat([g.reshape(-1) for g in g0.values()])
rec['einsum_repeat_grad_rel_global'] = float((torch.cat([ga[n].reshape(-1) for n in g0]) - flat0).norm() / flat0.norm())
rec['einsum_repeat_loss_rel'] = abs(la - l0) / abs(l0)
del ga
l1, g1, t1, m1 = run(True)
lb, gb, _, _ = run(True)
rec['sparse_repeat_grad_rel_global'] = float((torch.cat([gb[n].reshape(-1) for n in g0]) - torch.cat([g1[n].reshape(-1) for n in g0])).norm() / flat0.norm())
del gb
rel = {n: float((g1[n] - g0[n]).norm() / g0[n].norm().clamp_min(1e-30)) for n in g0}
flat0 = torch.cat([g.reshape(-1) for g in g0.values()]); flat1 = torch.cat([g1[n].reshape(-1) for n in g0])
rec.update(loss_einsum=l0, loss_sparse=l1, loss_rel=abs(l1 - l0) / abs(l0), grad_rel_global=float((flat1 - flat0).norm() / flat0.norm()),
           grad_rel_worst=max(rel.values()), grad_rel_worst_param=max(rel, key=rel.get), missing=sorted(set(g0) ^ set(g1)),
           step_s_einsum=t0, step_s_sparse=t1, peak_GB_einsum=m0, peak_GB_sparse=m1)
print(json.dumps(rec, indent=1), flush=True)
open(sys.argv[2], 'w').write(json.dumps(dict(rec, per_param=rel), indent=1))
