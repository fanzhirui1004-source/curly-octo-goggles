"""Front-end profile of plate cells (the per-cell steps of opt_design.analyse_fast): synchronised wall time of every step
for each cell, and a cProfile of the steps (top functions by cumulative time) over the cells after the first.
usage: python t_prep.py BODY_DIR CASE1,CASE2,... [--top 40]"""
import os, sys, json, time, argparse, cProfile, pstats, io, gc
SRC = os.environ.get('SRC', '/root/autodl-tmp/OPL/src_v3'); sys.path.insert(0, SRC)
os.environ.setdefault('OPL_TAILT_FUSED', '1'); os.environ.setdefault('OPL_COARSE_FP32', '1')
os.environ.setdefault('OPL_COARSE_ELEM', '1'); os.environ.setdefault('OPL_TET_TRITON', '1')
import models as MD  # noqa
import numpy as np
import torch
import teacher as TE, trainlib as TL, fastnet as FN, bench_deploy as BD, fastidx as FI, evalnet as EN
import stream_ops as SO

ap = argparse.ArgumentParser(); ap.add_argument('body'); ap.add_argument('cases'); ap.add_argument('--top', type=int, default=40)
a = ap.parse_args()
FN.FUSED = True; FI.ON = True
SO.init()
h = BD.ModelHolder('/root/autodl-tmp/OPL/S1/V2/A3_2grid/best.pt'); BD.TMP.mkdir(parents=True, exist_ok=True)
dev, dt = TE.dev, TE.dt
prof = cProfile.Profile()
rows = []
keep = []
for j, case in enumerate(a.cases.split(',')):
    st = {}
    if j == 1:
        prof.enable()

    def mark(k, t0):
        torch.cuda.synchronize(); st[k] = round(time.perf_counter() - t0, 4); return time.perf_counter()
    t = time.perf_counter()
    C = TE.Cell(case, a.body, log=lambda s_: None, deploy=True); t = mark('ctor', t)
    C.assemble_deploy(C.taus0); t = mark('assemble_deploy', t)
    BD.netdata(C, a.body, BD.TMP / case); t = mark('netdata', t)
    geo = TL.Geo(case, a.body, BD.TMP, neumann=False, log=lambda s_: None, cell=C, load_banks=False); t = mark('geo', t)
    h.model.caches.pop(case, None) if h.model is not None else None
    model = h.add(geo); t = mark('model_add', t)
    op = EN.FastOp(FN.FastNet(model, geo)); t = mark('fastnet', t)
    with torch.no_grad():
        op.apply(torch.zeros((C.np_, 1), dtype=dt, device=dev))
    t = mark('warmup', t)
    r_, c_, v_ = C._kpp_cache
    C._kpp_cache = (r_.int().cpu(), c_.int().cpu(), v_.cpu()); C.diag3 = None
    t = mark('kpp_host', t)
    FN.slim_deploy(op, geo, C, model); t = mark('slim_deploy', t)
    so = SO.StreamedOp(op, C, [op.fast, geo, C]); t = mark('stream_op', t)
    TE._collect() if hasattr(TE, '_collect') else (gc.collect(), torch.cuda.empty_cache()); t = mark('gc_empty', t)   # as the drivers
    st['total'] = round(sum(v for v in st.values()), 3)
    rows.append(dict(case=case, elements=int(len(C.cells)), **st))
    print(json.dumps(rows[-1]), flush=True)
    keep.append(so)
    del geo, op
prof.disable()
s = io.StringIO(); pstats.Stats(prof, stream=s).sort_stats('cumulative').print_stats(a.top)
print(s.getvalue()[-12000:], flush=True)
s = io.StringIO(); pstats.Stats(prof, stream=s).sort_stats('tottime').print_stats(a.top)
print(s.getvalue()[-12000:], flush=True)
