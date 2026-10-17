"""P16 (OPL_V3_HCUDA): fastnet's hyperedge gather / scatter as CUDA kernels (hyperk.py, hyper_kernel.cu: one warp per
hyperedge) against fused_hyper's Triton kernels (S10 tiles) on a full plate cell's element, face and fringe layers,
B = 1, 2, 3, 4: gather bitwise (forward weights a, and b / deg as in the adjoint), scatter to rounding (b / deg with the
port mask as in the forward; a as in the adjoint) and on a collision-free subset (one atomic add per row), device time
per call (CUDA events; warm, back to back, and cold: a 256 MB buffer rewritten before each call).
usage: python t_hyperk.py BODY_DIR:CASE [--B 1,2,3,4] [--rep 50]"""
import os, sys, json, argparse
SRC = os.environ.get('SRC', '/root/autodl-tmp/OPL/src_v3'); sys.path.insert(0, SRC)
os.environ.setdefault('OPL_TAILT_FUSED', '1'); os.environ.setdefault('OPL_COARSE_FP32', '1')
os.environ.setdefault('OPL_COARSE_ELEM', '1'); os.environ.setdefault('OPL_TET_TRITON', '1')
import models as MD  # noqa
import torch
import teacher as TE, trainlib as TL, fastnet as FN, bench_deploy as BD, fastidx as FI
import fused_hyper as FH, hyperk as HK

ap = argparse.ArgumentParser(); ap.add_argument('spec'); ap.add_argument('--B', default='1,2,3,4')
ap.add_argument('--rep', type=int, default=50)
a = ap.parse_args()
FN.FUSED = True; FI.ON = True
body, case = a.spec.split(':')
h = BD.ModelHolder('/root/autodl-tmp/OPL/S1/V2/A3_2grid/best.pt'); BD.TMP.mkdir(parents=True, exist_ok=True)
C = TE.Cell(case, body, log=lambda s_: None, deploy=True); C.assemble_deploy()
BD.netdata(C, body, BD.TMP / case)
geo = TL.Geo(case, body, BD.TMP, neumann=False, log=lambda s_: None, cell=C, load_banks=False)
f = FN.FastNet(h.add(geo), geo)
buf = torch.zeros(64 * 2 ** 20, dtype=torch.float32, device='cuda')


def ev(fn, rep):
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    fn(); torch.cuda.synchronize()
    s.record()
    for _ in range(rep):
        fn()
    e.record(); torch.cuda.synchronize()
    return 1000 * s.elapsed_time(e) / rep                              # us


def cold(fn, rep):
    return ev(lambda: (buf.add_(1.0), fn()), rep) - ev(lambda: buf.add_(1.0), rep)


def both(fn, rep):
    return dict(warm=round(ev(fn, rep), 1), cold=round(cold(fn, rep), 1))


def with_hk(on, fn):
    old = HK.ON
    HK.ON = on
    try:
        return fn()
    finally:
        HK.ON = old


if f.elem[0].a is None:                                                 # (slot weights recomputed: build them once)
    f.keep_slots() if hasattr(f, 'keep_slots') else None
layers = dict(elem=f.elem[0], face=f.face[0], fringe=f.fringe[0] if len(f.fringe) else None)
F = f.elem[0].W.shape[-1]
res = dict(case=case, F=F, H=f.elem[0].H, lib_error=None)
try:
    HK._lib()
except Exception as ex:                                                 # noqa: BLE001
    res['lib_error'] = repr(ex)[:1500]
print(json.dumps(res), flush=True)
pm8 = f.pm8 if f.pm8 is not None else f.c.is_port.to(torch.uint8).contiguous()
g = torch.Generator(device='cuda'); g.manual_seed(0)
with torch.no_grad():
    for name, L in layers.items():
        if L is None or L.Eh == 0 or L.a is None:
            continue
        N, H, Eh = L.N, L.H, L.Eh
        hn_cpu = L.hn.cpu(); used = torch.zeros(N, dtype=torch.bool); keep = []
        for e in range(Eh):                                             # collision-free subset
            r_ = hn_cpu[e]
            if not used[r_].any():
                used[r_] = True; keep.append(e)
        keep = torch.tensor(keep, device='cuda')
        hn_cf, b_cf = L.hn[keep].contiguous(), L.b[keep].contiguous()
        for B in [int(x) for x in a.B.split(',')]:
            BF = B * F
            X = torch.randn((N, BF), dtype=torch.float32, device='cuda', generator=g)
            Zs = torch.randn((H, Eh, BF), dtype=torch.float32, device='cuda', generator=g)
            Zcf = Zs[:, keep].contiguous()
            r = dict(Eh=Eh, N=N, cf=int(len(keep)), usable=HK.usable(H, 27, BF, L.a, L.hn, on=True))
            for tag, w, dg in (('a', L.a, False), ('b', L.b, True)):
                g0 = with_hk(False, lambda: FH.gather(X, L.hn, w, L.deg, H, FH.BLK, dg))
                g1 = with_hk(True, lambda: FH.gather(X, L.hn, w, L.deg, H, FH.BLK, dg))
                r[f'gather_{tag}'] = dict(bit=bool(torch.equal(g0, g1)), rel=float((g1 - g0).abs().max() / g0.abs().max()),
                                          triton=with_hk(False, lambda: both(lambda: FH.gather(X, L.hn, w, L.deg, H, FH.BLK, dg), a.rep)),
                                          cuda=with_hk(True, lambda: both(lambda: FH.gather(X, L.hn, w, L.deg, H, FH.BLK, dg), a.rep)))
            for tag, w, dg, rm in (('b_rmask', L.b, True, pm8), ('a', L.a, False, None)):
                s0 = with_hk(False, lambda: FH.scatter(Zs, L.hn, w, L.deg, N, FH.BLK, dg, rmask=rm))
                s0b = with_hk(False, lambda: FH.scatter(Zs, L.hn, w, L.deg, N, FH.BLK, dg, rmask=rm))
                s1 = with_hk(True, lambda: FH.scatter(Zs, L.hn, w, L.deg, N, FH.BLK, dg, rmask=rm))
                nrm = s0.abs().max()
                r[f'scatter_{tag}'] = dict(rel=float((s1 - s0).abs().max() / nrm), rerun_triton=float((s0b - s0).abs().max() / nrm),
                                           triton=with_hk(False, lambda: both(lambda: FH.scatter(Zs, L.hn, w, L.deg, N, FH.BLK, dg, rmask=rm), a.rep)),
                                           cuda=with_hk(True, lambda: both(lambda: FH.scatter(Zs, L.hn, w, L.deg, N, FH.BLK, dg, rmask=rm), a.rep)))
            c0 = with_hk(False, lambda: FH.scatter(Zcf, hn_cf, b_cf, L.deg, N, FH.BLK, True))
            c1 = with_hk(True, lambda: FH.scatter(Zcf, hn_cf, b_cf, L.deg, N, FH.BLK, True))
            r['scatter_cf'] = dict(bit=bool(torch.equal(c0, c1)), rel=float((c1 - c0).abs().max() / c0.abs().max()),
                                   ulp_rows=int((c1 != c0).any(1).sum()))
            res[f'{name}_B{B}'] = r
            print(json.dumps({f'{name}_B{B}': r}), flush=True)
print(json.dumps(res), flush=True)
