"""Fused ghost-face product (ghostfused, OPL_V3_GFFUSE) vs GhostFaces.matvec on a real cell (float32 correction's gf):
time per product and relative difference, for B = 1, 2, 3; and the full float32 K product time with the switch on/off.
usage: python t_gf.py BODY_DIR:CASE"""
import os, sys, json, time
SRC = os.environ.get('SRC', '/root/autodl-tmp/OPL/src_v3'); sys.path.insert(0, SRC)
os.environ.setdefault('OPL_TAILT_FUSED', '1'); os.environ.setdefault('OPL_COARSE_FP32', '1')
os.environ.setdefault('OPL_COARSE_ELEM', '1'); os.environ.setdefault('OPL_TET_TRITON', '1')
import models as MD  # noqa
import torch
import teacher as TE, trainlib as TL, fastnet as FN, bench_deploy as BD, fastidx as FI
import ghostfused as GFU


def bench(fn, rep=50):
    fn(); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(rep):
        fn()
    torch.cuda.synchronize(); return 1e6 * (time.perf_counter() - t) / rep


FN.FUSED = True; FI.ON = True
body, case = sys.argv[1].split(':')
h = BD.ModelHolder('/root/autodl-tmp/OPL/S1/V2/A3_2grid/best.pt'); BD.TMP.mkdir(parents=True, exist_ok=True)
C = TE.Cell(case, body, log=lambda s_: None, deploy=True); C.assemble_deploy()
BD.netdata(C, body, BD.TMP / case)
geo = TL.Geo(case, body, BD.TMP, neumann=False, log=lambda s_: None, cell=C, load_banks=False)
f = FN.FastNet(h.add(geo), geo)
g = torch.Generator(device='cuda'); g.manual_seed(0)
with torch.no_grad():
    f.s_hat(torch.randn((C.np_, 1), dtype=torch.float64, device='cuda', generator=g))
gf = C._Kc.gf
r = dict(case=case, faces=[int(t.shape[1]) for t in gf.idxT])
for b in (1, 2, 3):
    x = torch.randn((C.nb, b), dtype=torch.float32, device='cuda', generator=g)
    GFU.ON = False
    y0 = gf.matvec(x, torch.zeros_like(x)); t0 = bench(lambda: gf.matvec(x, torch.zeros_like(x)))
    with TL._held(C):
        k0 = C._Kc.mm32(x); tk0 = bench(lambda: C._Kc.mm32(x))
    GFU.ON = True
    y1 = gf.matvec(x, torch.zeros_like(x)); t1 = bench(lambda: gf.matvec(x, torch.zeros_like(x)))
    with TL._held(C):
        k1 = C._Kc.mm32(x); tk1 = bench(lambda: C._Kc.mm32(x))
    best = None
    for FB in (16, 32, 64):
        for NW in (2, 4, 8):
            GFU.FB, GFU.NW = FB, NW
            try:
                tt = bench(lambda: gf.matvec(x, torch.zeros_like(x)))
            except Exception as ex:                                                       # noqa
                continue
            if best is None or tt < best[0]:
                best = (round(tt, 1), FB, NW)
    GFU.FB, GFU.NW = 32, 4
    r[f'B{b}'] = dict(gf_us=(round(t0, 1), round(t1, 1)), gf_best=best, gf_rel=float((y1 - y0).abs().max() / y0.abs().max()),
                      kc_us=(round(tk0, 1), round(tk1, 1)), kc_rel=float((k1 - k0).abs().max() / k0.abs().max()))
    print(json.dumps(r), flush=True)
