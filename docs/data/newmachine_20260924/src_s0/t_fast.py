"""Check fastnet.FastNet against the autograd model: forward, adjoint identity, variational reaction, timings, memory.
Usage: t_fast.py <out_json> <checkpoint.pt> [<checkpoint.pt> ...]   (each checkpoint on its own first case)"""
import sys, json, time, gc
from pathlib import Path
import numpy as np
import torch
import teacher as TE
import trainlib as TL
import models as MD
import fastnet as FN

dev, dt, f32 = TE.dev, TE.dt, torch.float32


def tic():
    TE.sync(); return time.perf_counter()


def toc(t):
    TE.sync(); return time.perf_counter() - t


def rel(a, b):
    a, b = a.to(dt), b.to(dt)
    return ((a - b).norm(dim=0) / b.norm(dim=0)).max().item()


def timed(fn, reps=3):
    fn(); ts = []
    for _ in range(reps):
        t = tic(); fn(); ts.append(toc(t))
    return min(ts)


def main(out, ckpts):
    rec = {}
    for ck_path in ckpts:
        ck = torch.load(ck_path, map_location='cuda:0', weights_only=False)
        cfg = ck['cfg']; case = cfg['cases'][0]
        geo = TL.Geo(case, cfg['body'], cfg['data'], neumann=False, log=lambda s_: None)
        model = MD.build(cfg['model'], [geo], **cfg.get('model_args', {})).cuda()
        model.load_state_dict(ck['model'], strict=False); model.eval()
        r = dict(case=case, model=cfg['model'], step=ck['step'])
        t = tic(); fast = FN.FastNet(model, geo); r['freeze_s'] = toc(t)
        gen = torch.Generator(device=dev).manual_seed(0)
        Q = torch.cat([geo.banks['val'][c][:, :3] for c in geo.classes], 1).to(dt)          # real directions
        Qr = torch.randn((geo.np_, 4), dtype=dt, device=dev, generator=gen)
        Q = torch.cat([Q, Qr], 1)
        with torch.no_grad():
            u_ref = geo.field(model, Q)
        u_fast = fast.field(Q)
        r['field_rel'] = rel(u_fast, u_ref)
        V = torch.randn((geo.nb, Q.shape[1]), dtype=f32, device=dev, generator=gen)
        lhs = (fast.field(Q) * V).sum(0).to(dt)
        rhs = (Q.to(f32) * fast.field_T(V)).sum(0).to(dt)
        r['adjoint_identity_rel'] = ((lhs - rhs).abs() / (fast.field(Q).norm(dim=0) * V.norm(dim=0)).to(dt)).max().item()
        s_ref = geo.s_hat_apply(model, Q)
        s_fast = fast.s_hat(Q)
        r['s_hat_rel'] = rel(s_fast, s_ref)
        e_ref = (Q * s_ref).sum(0); e_fast = (Q * s_fast).sum(0)
        r['energy_rel'] = ((e_fast - e_ref).abs() / e_ref.abs()).max().item()
        # model-only adjoint vs autograd vjp of the model map
        qd = torch.randn((geo.np_, 5), dtype=f32, device=dev, generator=gen)
        y = torch.randn((geo.nb, 5), dtype=f32, device=dev, generator=gen)
        qq = qd.clone().requires_grad_(True)
        with torch.enable_grad():
            g_ref = torch.autograd.grad(model(geo, qq), qq, grad_outputs=y)[0]
        r['model_T_rel'] = rel(fast.ext_T(y), g_ref)
        print(json.dumps(dict(accuracy=r)), flush=True)
        for B in (1, 9, 16, 64):
            Qb = torch.randn((geo.np_, B), dtype=dt, device=dev, generator=gen)
            runs = [('field_fast', lambda: fast.field(Qb)), ('shat_fast', lambda: fast.s_hat(Qb))]
            if B <= 16:
                runs += [('field_autograd', lambda: _nograd(geo.field, model, Qb)), ('shat_autograd', lambda: geo.s_hat_apply(model, Qb))]
            for name, fn in runs:
                gc.collect(); torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(); m0 = torch.cuda.memory_allocated()
                try:
                    r[f'B{B}_{name}_s'] = timed(fn)
                    r[f'B{B}_{name}_extra_GB'] = (torch.cuda.max_memory_allocated() - m0) / 2 ** 30
                except torch.OutOfMemoryError:
                    r[f'B{B}_{name}_s'] = 'OOM'
            print(json.dumps({f'B{B}': {k: v for k, v in r.items() if k.startswith(f'B{B}_')}}), flush=True)
        r['sparse_nnz_total'] = int(sum(h.G._nnz() for h in fast.elem + getattr(fast, 'face', []) + getattr(fast, 'fringe', [])))
        rec[ck_path] = r
        print(json.dumps(r), flush=True)
        del fast, model, geo; gc.collect(); torch.cuda.empty_cache()
    Path(out).write_text(json.dumps(rec, indent=1))


def _nograd(f, *a):
    with torch.no_grad():
        return f(*a)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:])
