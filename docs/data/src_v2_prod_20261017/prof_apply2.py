"""P2 speed: S_hat q per batch width for three settings on the same lean cell: baseline, correction in float32
(lean(correction_fp32=True)), and float32 correction + fused tail transpose (OPL_TAILT_FUSED=1). Reports ms and the
relative change of S_hat q and of the energy q^T S_hat q against the baseline.
Usage: prof_apply2.py <out.json> <body>:<case> [--ckpt ...] [--B 1,6,16,64] [--rep 5]"""
import os, json, time, argparse
import models as MD                                                    # noqa: F401
import torch
import teacher as TE
import trainlib as TL
import fastnet as FN
import bench_deploy as BD

dev, dt = TE.dev, TE.dt


def timed(fn, rep):
    fn(); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(rep):
        out = fn()
    torch.cuda.synchronize(); return out, 1000 * (time.perf_counter() - t) / rep


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('spec')
    ap.add_argument('--ckpt', default='/root/autodl-tmp/OPL/S1/V2/A3_2grid/best.pt')
    ap.add_argument('--B', default='1,6,16,64'); ap.add_argument('--rep', type=int, default=5)
    a = ap.parse_args()
    body, case = a.spec.split(':')
    BD.TMP.mkdir(parents=True, exist_ok=True)
    C = TE.Cell(case, body, log=lambda s_: None); C.assemble()
    h = BD.ModelHolder(a.ckpt); BD.netdata(C, body, BD.TMP / case)
    geo = TL.Geo(case, body, BD.TMP, neumann=False, log=lambda s_: None, cell=C, load_banks=False)
    f = FN.FastNet(h.add(geo), geo)
    C.lean(correction_fp32=True); Kc = C._Kc
    f.s_hat(torch.zeros((C.np_, 1), dtype=dt, device=dev))
    res = dict(case=case, per_B={})
    for B in [int(b) for b in a.B.split(',')]:
        q = torch.randn((C.np_, B), dtype=dt, device=dev, generator=torch.Generator(device=dev).manual_seed(B))
        r = {}
        C._Kc = None; os.environ['OPL_TAILT_FUSED'] = '0'
        s0, r['base_ms'] = timed(lambda: f.s_hat(q), a.rep)
        C._Kc = Kc
        s1, r['fp32corr_ms'] = timed(lambda: f.s_hat(q), a.rep)
        os.environ['OPL_TAILT_FUSED'] = '1'
        s2, r['fp32corr_fusedT_ms'] = timed(lambda: f.s_hat(q), a.rep)
        os.environ['OPL_COARSE_FP32'] = '1'
        s4, r['all_fp32_ms'] = timed(lambda: f.s_hat(q), a.rep)
        os.environ['OPL_COARSE_FP32'] = '0'
        C._Kc = None
        s3, r['fusedT_only_ms'] = timed(lambda: f.s_hat(q), a.rep)
        os.environ['OPL_TAILT_FUSED'] = '0'
        e0 = (q * s0).sum(0)
        for k, s in (('fp32corr', s1), ('fp32corr_fusedT', s2), ('fusedT_only', s3), ('all_fp32', s4)):
            r[k + '_rel'] = float((s - s0).norm() / s0.norm())
            r[k + '_energy_rel'] = float(((q * s).sum(0) - e0).abs().max() / e0.abs().min())
        res['per_B'][B] = r
        print(json.dumps(dict(B=B, **{k: (round(v, 2) if k.endswith('ms') else v) for k, v in r.items()})), flush=True)
    json.dump(res, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
