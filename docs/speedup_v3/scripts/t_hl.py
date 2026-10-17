"""P18 (OPL_V3_HFUSE with the CUDA layer, hyperfused.layer -> hyperk.layer): one hyperedge layer in one kernel against
today's three steps (P16 gather, cuBLAS channel mix, P16 scatter) on a full plate cell's element, face and fringe
layers, B = 1, 2, 4, 16, forward (a; b / deg, port rows skipped) and adjoint (b / deg; a): largest relative difference
(to the result's largest value) next to the three-step path's own rerun difference (atomics), device time per call
(CUDA events, warm and cold: a 256 MB buffer rewritten before each call).
usage: python t_hl.py BODY_DIR:CASE [--rep 50]"""
import os, sys, json, argparse
SRC = os.environ.get('SRC', '/root/autodl-tmp/OPL/src_v3'); sys.path.insert(0, SRC)
os.environ.setdefault('OPL_TAILT_FUSED', '1'); os.environ.setdefault('OPL_COARSE_FP32', '1')
os.environ.setdefault('OPL_COARSE_ELEM', '1'); os.environ.setdefault('OPL_TET_TRITON', '1')
import models as MD  # noqa
import torch
import teacher as TE, trainlib as TL, fastnet as FN, bench_deploy as BD, fastidx as FI
import fused_hyper as FH, hyperk as HK, hyperfused as HF

ap = argparse.ArgumentParser(); ap.add_argument('spec'); ap.add_argument('--rep', type=int, default=50)
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


def both(fn, rep):
    return dict(warm=round(ev(fn, rep), 1), cold=round(ev(lambda: (buf.add_(1.0), fn()), rep) - ev(lambda: buf.add_(1.0), rep), 1))


def three(L, X, ga, sb, W, gdeg, sdeg, rmask):
    N, BF = X.shape
    B = BF // 32
    Z = FH.gather(X, L.hn, ga, L.deg, L.H, FH.BLK, gdeg)
    Z = torch.bmm(Z.view(L.H, L.Eh * B, 32), W).view(L.H, L.Eh, BF)
    return FH.scatter(Z, L.hn, sb, L.deg, N, FH.BLK, sdeg, rmask=rmask)


res = dict(case=case, lib_error=None)
try:
    HK._lib()
except Exception as ex:                                                 # noqa: BLE001
    res['lib_error'] = repr(ex)[:1500]
print(json.dumps(res), flush=True)
with f.slots(), torch.no_grad():
    layers = dict(elem=f.elem[0], face=f.face[0], fringe=f.fringe[0] if len(f.fringe) else None)
    pm8 = f.pm8 if f.pm8 is not None else f.c.is_port.to(torch.uint8).contiguous()
    g = torch.Generator(device='cuda'); g.manual_seed(0)
    for name, L in layers.items():
        if L is None or L.Eh == 0:
            continue
        for B in (1, 2, 4, 16):
            X = torch.randn((L.N, 32 * B), dtype=torch.float32, device='cuda', generator=g)
            r = dict(Eh=L.Eh)
            for tag, ga, sb, W, gd, sd, rm in (('fwd', L.a, L.b, L.W, False, True, pm8), ('adj', L.b, L.a, L.Wt, True, False, None)):
                y0 = three(L, X, ga, sb, W, gd, sd, rm)
                y0b = three(L, X, ga, sb, W, gd, sd, rm)
                y1 = HF.layer(X, L.hn, ga, sb, L.deg, W, torch.zeros_like(X), gd, sd, rmask=rm)
                nrm = y0.abs().max()
                r[tag] = dict(rel=float((y1 - y0).abs().max() / nrm), rerun=float((y0b - y0).abs().max() / nrm),
                              three_us=both(lambda: three(L, X, ga, sb, W, gd, sd, rm), a.rep),
                              cuda_us=both(lambda: HF.layer(X, L.hn, ga, sb, L.deg, W, torch.zeros_like(X), gd, sd, rmask=rm), a.rep))
            res[f'{name}_B{B}'] = r
            print(json.dumps({f'{name}_B{B}': r}), flush=True)
print(json.dumps(res), flush=True)
