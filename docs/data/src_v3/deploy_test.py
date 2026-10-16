"""Deployment mode check: S_hat q (baseline, full CSR, float64 correction) vs lean(deploy=True) with float32 correction,
fused tail transpose and float32 coarse solve; per-cell device memory of the learned operator and time per application.
Usage: deploy_test.py <out.json> <body>:<case>[,...] [--ckpt ...] [--B 6]"""
import os, json, time, argparse, gc
import models as MD                                                    # noqa: F401
import torch
import teacher as TE
import trainlib as TL
import fastnet as FN
import bench_deploy as BD

dev, dt = TE.dev, TE.dt
G = lambda: torch.cuda.memory_allocated() / 1e9


def timed(fn, rep=5):
    fn(); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(rep):
        out = fn()
    torch.cuda.synchronize(); return out, 1000 * (time.perf_counter() - t) / rep


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('cases')
    ap.add_argument('--ckpt', default='/root/autodl-tmp/OPL/S1/V2/A3_2grid/best.pt'); ap.add_argument('--B', type=int, default=6)
    a = ap.parse_args()
    BD.TMP.mkdir(parents=True, exist_ok=True)
    out = {}
    for spec in a.cases.split(','):
        body, case = spec.split(':')
        gc.collect(); torch.cuda.empty_cache(); g0 = G()
        os.environ['OPL_TAILT_FUSED'] = '0'; os.environ['OPL_COARSE_FP32'] = '0'
        C = TE.Cell(case, body, log=lambda s_: None); C.assemble()
        h = BD.ModelHolder(a.ckpt); BD.netdata(C, body, BD.TMP / case)
        geo = TL.Geo(case, body, BD.TMP, neumann=False, log=lambda s_: None, cell=C, load_banks=False)
        f = FN.FastNet(h.add(geo), geo)
        q = torch.randn((C.np_, a.B), dtype=dt, device=dev, generator=torch.Generator(device=dev).manual_seed(0))
        s0, t0 = timed(lambda: f.s_hat(q))
        r = dict(ports=C.np_, baseline_GB=G() - g0, baseline_ms=t0)
        for k in ('_cV', '_cL', '_c_space', '_tail_bounds'):
            if hasattr(C, k):
                delattr(C, k)
        C.lean(deploy=True)
        os.environ['OPL_TAILT_FUSED'] = '1'; os.environ['OPL_COARSE_FP32'] = '1'
        gc.collect(); torch.cuda.empty_cache()
        s1, t1 = timed(lambda: f.s_hat(q))
        gc.collect(); torch.cuda.empty_cache()
        r.update(deploy_GB=G() - g0, deploy_ms=t1, rel=float((s1 - s0).norm() / s0.norm()),
                 energy_rel=float(((q * (s1 - s0)).sum(0).abs() / (q * s0).sum(0)).max()))
        out[case] = r; print(json.dumps(dict(case=case, **r)), flush=True)
        h.model.caches.pop(case, None); del f, geo, h, C, q, s0, s1; gc.collect(); torch.cuda.empty_cache()
    json.dump(out, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
