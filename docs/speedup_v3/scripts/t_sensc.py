"""Item 3 (src_v3 OPL_V3_SENS_NOCUT): the reverse-mode sensitivity pass (moments_ad.cell_sens, skip_full, energy and
volume columns, as opt_design --sens ad) without the tetrahedra that no tau-dependent clip cut (their vertices are
constants) against the pass with them, on full and cut plate cells, row batches 2048 and 1024: tetrahedra handed to the
fp64 VJP kernel, seconds per cell, device peak above the base, bitwise equality / largest relative difference.
usage: python t_sensc.py BODY_DIR CASE [CASE ...]"""
import os, sys, json, time
SRC = os.environ.get('SRC', '/root/autodl-tmp/OPL/src_v3'); sys.path.insert(0, SRC)
os.environ.setdefault('OPL_TAILT_FUSED', '1'); os.environ.setdefault('OPL_COARSE_FP32', '1')
os.environ.setdefault('OPL_COARSE_ELEM', '1'); os.environ.setdefault('OPL_TET_TRITON', '1')
import models as MD  # noqa: F401
import numpy as np
import torch
import teacher as TE, trainlib as TL, fastnet as FN, bench_deploy as BD, fastidx as FI
import moments_ad as MA, fast_tet2 as FT2

FN.FUSED = True; FI.ON = True
body, cases = sys.argv[1], sys.argv[2:]
h = BD.ModelHolder('/root/autodl-tmp/OPL/S1/V2/A3_2grid/best.pt'); BD.TMP.mkdir(parents=True, exist_ok=True)
NT = [0]
_apply = FT2.TetMoments2.apply


def counting(tets, tref, tw):
    NT[0] += len(tets)
    return _apply(tets, tref, tw)


FT2.TetMoments2.apply = counting


def timed(fn):
    torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    b = torch.cuda.memory_allocated(); t = time.perf_counter(); NT[0] = 0
    out = fn()
    torch.cuda.synchronize()
    return out, round(time.perf_counter() - t, 3), round((torch.cuda.max_memory_allocated() - b) / 2 ** 20, 1), NT[0]


res = {}
for case in cases:
    C = TE.Cell(case, body, log=lambda s_: None, deploy=True); C.assemble_deploy()
    BD.netdata(C, body, BD.TMP / case)
    geo = TL.Geo(case, body, BD.TMP, neumann=False, log=lambda s_: None, cell=C, load_banks=False)
    f = FN.FastNet(h.add(geo), geo)
    g = torch.Generator(device='cuda'); g.manual_seed(0)
    q = torch.randn((C.np_, 1), dtype=torch.float64, device='cuda', generator=g)
    with torch.no_grad():
        u = f.field(q).to(torch.float64)
        gd = C.energy_density(u)
        e0 = torch.zeros((gd.shape[0], gd.shape[1], 1), dtype=gd.dtype, device=gd.device); e0[:, 0, 0] = 1.0
        G = torch.cat([gd, e0], 2)
    del f
    r = dict(kind=C.kind, elements=int(len(C.cells)))
    for bt in (2048, 1024):
        out = {}
        for tag, on in (('all', False), ('cut', True)):
            MA.NOCUT = on
            MA.cell_sens(C, G, batch=bt, skip_full=True)                    # warm
            ts, pk = [], 0
            for _ in range(3):
                s, t_, p_, nt = timed(lambda: MA.cell_sens(C, G, batch=bt, skip_full=True))
                ts.append(t_); pk = max(pk, p_)
            out[tag] = (s, dict(s=round(float(np.median(ts)), 3), peak_MiB=pk, tets=nt))
        a, b = out['all'][0], out['cut'][0]
        r[f'batch_{bt}'] = dict(all=out['all'][1], cut=out['cut'][1], equal=bool(torch.equal(a, b)),
                                rel=float((b - a).abs().max() / a.abs().max()))
    MA.NOCUT = True
    res[case] = r
    print(json.dumps({case: r}), flush=True)
    del C, geo, G, gd, u
print(json.dumps(res), flush=True)
