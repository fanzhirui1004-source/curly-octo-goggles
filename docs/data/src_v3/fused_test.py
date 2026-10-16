"""Step 1 (P2 memory): FastNet with CSR hyperedge layers vs the fused Triton layers (fastnet.FUSED), same network and
cell: device memory of the frozen network state, time of S_hat q (deployment mode) and relative difference.
Usage: fused_test.py <out.json> <body>:<case>[,...] [--ckpt ...] [--B 1,6,16]"""
import os, json, time, argparse, gc
os.environ['OPL_TAILT_FUSED'] = '1'; os.environ['OPL_COARSE_FP32'] = '1'
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
    ap.add_argument('--ckpt', default='/root/autodl-tmp/OPL/S1/V2/A3_2grid/best.pt'); ap.add_argument('--B', default='1,6,16')
    a = ap.parse_args()
    BD.TMP.mkdir(parents=True, exist_ok=True)
    out = {}
    for spec in a.cases.split(','):
        body, case = spec.split(':')
        C = TE.Cell(case, body, log=lambda s_: None); C.assemble()
        h = BD.ModelHolder(a.ckpt); BD.netdata(C, body, BD.TMP / case)
        geo = TL.Geo(case, body, BD.TMP, neumann=False, log=lambda s_: None, cell=C, load_banks=False)
        m = h.add(geo)
        C.lean(deploy=True); gc.collect(); torch.cuda.empty_cache()
        r = dict(ports=C.np_)
        nets = {}
        for name, fused in (('csr', False), ('fused', True)):
            FN.FUSED = fused
            gc.collect(); torch.cuda.empty_cache(); g0 = G()
            nets[name] = FN.FastNet(m, geo)
            gc.collect(); torch.cuda.empty_cache(); r[name + '_state_GB'] = G() - g0
        FN.FUSED = False
        for B in [int(b) for b in a.B.split(',')]:
            q = torch.randn((C.np_, B), dtype=dt, device=dev, generator=torch.Generator(device=dev).manual_seed(B))
            s0, r[f'csr_ms_B{B}'] = timed(lambda: nets['csr'].s_hat(q))
            s1, r[f'fused_ms_B{B}'] = timed(lambda: nets['fused'].s_hat(q))
            r[f'rel_B{B}'] = float((s1 - s0).norm() / s0.norm())
            u0 = nets['csr'].ext(q.to(torch.float32)); u1 = nets['fused'].ext(q.to(torch.float32))
            r[f'net_rel_B{B}'] = float((u1 - u0).norm() / u0.norm())
        out[case] = r; print(json.dumps(dict(case=case, **{k: (round(v, 4) if isinstance(v, float) and v > 1e-3 else v) for k, v in r.items()})), flush=True)
        h.model.caches.pop(case, None); del nets, geo, h, C, m; gc.collect(); torch.cuda.empty_cache()
    json.dump(out, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
