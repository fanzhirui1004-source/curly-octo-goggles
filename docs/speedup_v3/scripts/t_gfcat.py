"""Concatenated ghost-face DOF lists (teacher.GF_CAT / OPL_V3_GFCAT): the same plate cell built with and without, time of
the ghost product, the float32 K product (held) and S_hat for B = 1, 3, 16, and the relative differences.
usage: python t_gfcat.py BODY_DIR:CASE"""
import os, sys, json, time
SRC = os.environ.get('SRC', '/root/autodl-tmp/OPL/src_v3'); sys.path.insert(0, SRC)
os.environ.setdefault('OPL_TAILT_FUSED', '1'); os.environ.setdefault('OPL_COARSE_FP32', '1')
os.environ.setdefault('OPL_COARSE_ELEM', '1'); os.environ.setdefault('OPL_TET_TRITON', '1')
import models as MD  # noqa
import torch
import teacher as TE, trainlib as TL, fastnet as FN, bench_deploy as BD, fastidx as FI

FN.FUSED = True; FI.ON = True
body, case = sys.argv[1].split(':')
h = BD.ModelHolder('/root/autodl-tmp/OPL/S1/V2/A3_2grid/best.pt'); BD.TMP.mkdir(parents=True, exist_ok=True)


def build(cat):
    TE.GF_CAT = cat
    C = TE.Cell(case, body, log=lambda s_: None, deploy=True); C.assemble_deploy()
    BD.netdata(C, body, BD.TMP / case)
    geo = TL.Geo(case, body, BD.TMP, neumann=False, log=lambda s_: None, cell=C, load_banks=False)
    if h.model is not None:
        h.model.caches.pop(case, None)
    f = FN.FastNet(h.add(geo), geo)
    with torch.no_grad():
        f.s_hat(torch.zeros((C.np_, 1), dtype=torch.float64, device='cuda'))
    C.store64()
    return C, f


def timed(fn, rep=20):
    fn(); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(rep):
        out = fn()
    torch.cuda.synchronize(); return out, round(1e3 * (time.perf_counter() - t) / rep, 3)


C0, f0 = build(False)
C1, f1 = build(True)
g = torch.Generator(device='cuda'); g.manual_seed(0)
r = dict(case=case, faces=C1._Kc.gf.cuts)
with torch.no_grad():
    for B in (1, 3, 16):
        x = torch.randn((C0.nb, B), dtype=torch.float32, device='cuda', generator=g)
        q = torch.randn((C0.np_, B), dtype=torch.float64, device='cuda', generator=g)
        a0, tg0 = timed(lambda: C0._Kc.gf.matvec(x, torch.zeros_like(x)))
        a1, tg1 = timed(lambda: C1._Kc.gf.matvec(x, torch.zeros_like(x)))
        with TL._held(C0), TL._held(C1):
            k0, tk0 = timed(lambda: C0._Kc.mm32(x))
            k1, tk1 = timed(lambda: C1._Kc.mm32(x))
        s0, ts0 = timed(lambda: f0.s_hat(q), 10)
        s1, ts1 = timed(lambda: f1.s_hat(q), 10)
        r[f'B{B}'] = dict(gf_ms=(tg0, tg1), kc_ms=(tk0, tk1), shat_ms=(ts0, ts1),
                          gf_rel=float((a1 - a0).abs().max() / a0.abs().max()), shat_rel=float((s1 - s0).abs().max() / s0.abs().max()))
print(json.dumps(r), flush=True)
