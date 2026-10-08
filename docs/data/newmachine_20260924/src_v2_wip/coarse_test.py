"""OPL_COARSE_ELEM check: coarse Galerkin factor by probing (coarse_setup) vs element/face sums, on a deployment cell
(Cell(deploy=True).assemble_deploy()); factor and prolongation differences, setup times, S_hat q with each.
Usage: coarse_test.py <body>:<case>[,...] [--ckpt ...] [--space Q1_17]"""
import os, sys, json, time, argparse, gc
import models as MD  # noqa
import torch
import teacher as TE, trainlib as TL, fastnet as FN, bench_deploy as BD, fastidx as FI
dev, dt = TE.dev, TE.dt
ap = argparse.ArgumentParser(); ap.add_argument('cases'); ap.add_argument('--ckpt', default='/root/autodl-tmp/OPL/S1/V2/A3_2grid/best.pt')
ap.add_argument('--space', default='Q1_17'); a = ap.parse_args()
FI.ON = True; FN.FUSED = True; os.environ['OPL_TAILT_FUSED'] = '1'; os.environ['OPL_COARSE_FP32'] = '1'
BD.TMP.mkdir(parents=True, exist_ok=True)
KEYS = ('_cV', '_cL', '_c_space', '_tail_bounds', '_cL32', '_cV32', '_cV32t')
def clear(C):
    for k in KEYS:
        setattr(C, k, None)
def T():
    torch.cuda.synchronize(); return time.perf_counter()
for spec in a.cases.split(','):
    body, case = spec.split(':')
    C = TE.Cell(case, body, log=lambda s_: None, deploy=True); C.assemble_deploy()
    r = {}
    os.environ['OPL_COARSE_ELEM'] = '0'; clear(C)
    t = T(); TL.coarse_setup(C, a.space); r['probe_s'] = T() - t
    L0, V0 = C._cL.clone(), C._cV.to_dense() if C._cV.shape[0] * C._cV.shape[1] < 3e8 else None
    os.environ['OPL_COARSE_ELEM'] = '1'; clear(C)
    t = T(); TL.coarse_setup(C, a.space); r['elem_s'] = T() - t
    r['L_rel'] = float((C._cL - L0).norm() / L0.norm()); r['nc'] = int(L0.shape[0])
    if V0 is not None:
        r['V_rel'] = float((C._cV.to_dense() - V0).norm() / V0.norm())
    del L0, V0
    h = BD.ModelHolder(a.ckpt); BD.netdata(C, body, BD.TMP / case)
    geo = TL.Geo(case, body, BD.TMP, neumann=False, log=lambda s_: None, cell=C, load_banks=False)
    f = FN.FastNet(h.add(geo), geo)
    q = torch.randn((C.np_, 16), dtype=dt, device=dev, generator=torch.Generator(device=dev).manual_seed(0))
    S = {}
    for mode in ('0', '1'):
        os.environ['OPL_COARSE_ELEM'] = mode; clear(C)
        t = T(); S[mode] = f.s_hat(q); r['first_apply_' + mode] = T() - t
    r['shat_rel'] = float((S['1'] - S['0']).norm() / S['0'].norm())
    print(json.dumps(dict(case=case, **r)), flush=True)
    h.model.caches.pop(case, None); del f, geo, h, C, S; gc.collect(); torch.cuda.empty_cache()
