"""P2 steps 3/4 check: deployment-mode variants of one learned cell operator against the float64 baseline (full CSR,
float64 correction). Variants (each on a freshly assembled cell, fused hyperedge network):
  deploy        lean(deploy=True): stored float32 K_e and ghost CSR in the correction, float64 ghost CSR, K_e from moments
  kem           + ke_moments: float32 correction K_e recomputed from the moments per chunk
  faces         + ghost_faces: ghost part by face templates (float32 correction and float64 K), no ghost matrix
  kem+faces     both
Reported: S_hat q relative difference and energy difference vs the baseline, float64 K x difference vs the assembled K,
float32 correction stiffness difference vs float64 K, per-cell device bytes (cell + correction caches, network state),
time per S_hat application (B columns), per K x (float64) and per correction stiffness product.
Usage: deploy_test2.py <out.json> <body>:<case>[,...] [--ckpt ...] [--B 16] [--variants deploy,kem,faces,kem+faces]"""
import os, json, time, argparse, gc
import models as MD                                                    # noqa: F401
import torch
import teacher as TE
import trainlib as TL
import fastnet as FN
import bench_deploy as BD
import stream_ops as SO

dev, dt = TE.dev, TE.dt


def timed(fn, rep=5):
    fn(); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(rep):
        out = fn()
    torch.cuda.synchronize(); return out, 1000 * (time.perf_counter() - t) / rep


def dev_bytes(roots):
    h, seen = [], set()
    for r in roots:
        SO._walk(r, h, seen)
    st = {}
    for o, k, t in h:
        for x in SO._parts(t):
            st[(x.data_ptr(), x.numel())] = x.numel() * x.element_size()
    return sum(st.values()) / 1e9


def build(case, body, h, flat=False):
    C = TE.Cell(case, body, log=lambda s_: None); C.assemble()
    BD.netdata(C, body, BD.TMP / case)
    geo = TL.Geo(case, body, BD.TMP, neumann=False, log=lambda s_: None, cell=C, load_banks=False)
    return C, geo, FN.FastNet(h.add(geo), geo)


def clear(C):
    for k in ('_cV', '_cL', '_c_space', '_tail_bounds', '_cL32', '_cV32', '_cV32t', '_cL32_keep'):
        if hasattr(C, k):
            delattr(C, k)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('cases')
    ap.add_argument('--ckpt', default='/root/autodl-tmp/OPL/S1/V2/A3_2grid/best.pt'); ap.add_argument('--B', type=int, default=16)
    ap.add_argument('--variants', default='deploy,kem,faces,kem+faces')
    a = ap.parse_args()
    FN.FUSED = True
    SO.MIN_BYTES = 0
    BD.TMP.mkdir(parents=True, exist_ok=True)
    out = {}
    for spec in a.cases.split(','):
        body, case = spec.split(':')
        h = BD.ModelHolder(a.ckpt)
        os.environ['OPL_TAILT_FUSED'] = '0'; os.environ['OPL_COARSE_FP32'] = '0'
        C, geo, f = build(case, body, h)
        g = torch.Generator(device=dev).manual_seed(0)
        q = torch.randn((C.np_, a.B), dtype=dt, device=dev, generator=g)
        x = torch.randn((C.nb, a.B), dtype=dt, device=dev, generator=g)
        s0, t0 = timed(lambda: f.s_hat(q))
        k0 = C.K @ x
        rec = dict(ports=C.np_, nb=C.nb, elements=int(len(C.cells)), baseline_ms=t0, variants={})
        h.model.caches.pop(case, None); del f, geo, C; gc.collect(); torch.cuda.empty_cache()
        for v in a.variants.split(','):
            os.environ['OPL_TAILT_FUSED'] = '0'; os.environ['OPL_COARSE_FP32'] = '0'
            C, geo, f = build(case, body, h)
            clear(C)
            t = time.perf_counter()
            C.lean(deploy=True, ke_moments='kem' in v, ghost_faces='faces' in v)
            torch.cuda.synchronize(); lean_s = time.perf_counter() - t
            os.environ['OPL_TAILT_FUSED'] = '1'; os.environ['OPL_COARSE_FP32'] = '1'
            gc.collect(); torch.cuda.empty_cache()
            s1, t1 = timed(lambda: f.s_hat(q))
            k1, tk = timed(lambda: C.K @ x)
            kc, tc = timed(lambda: C._Kc @ x)
            r = dict(lean_s=lean_s, ms=t1, K64_ms=tk, Kc_ms=tc,
                     rel=float((s1 - s0).norm() / s0.norm()),
                     energy_rel=float(((q * (s1 - s0)).sum(0).abs() / (q * s0).sum(0)).max()),
                     K64_rel=float((k1 - k0).norm() / k0.norm()), Kc_rel=float((kc - k0).norm() / k0.norm()),
                     cell_GB=dev_bytes([C]), net_GB=dev_bytes([C, f]) - dev_bytes([C]),
                     ghost_faces=getattr(getattr(C, 'GF64', None), 'faces', None))
            rec['variants'][v] = r
            print(json.dumps(dict(case=case, variant=v, **r)), flush=True)
            h.model.caches.pop(case, None); del f, geo, C, s1, k1, kc; gc.collect(); torch.cuda.empty_cache()
        out[case] = rec
        json.dump(out, open(a.out, 'w'), indent=1)
        del h; gc.collect(); torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
