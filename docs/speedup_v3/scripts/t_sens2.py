"""moments_ad.cell_sens with the reduced volume-column kernel (fast_tet2._bwd2_e0, OPL_V3_E0FAST) on vs off: bitwise
comparison in deterministic mode (index_put / index_add without atomics), a rerun for reference, and the time per cell.
usage: python t_sens2.py BODY_DIR:CASE"""
import os, sys, json, time
SRC = os.environ.get('SRC', '/root/autodl-tmp/OPL/src_v3'); sys.path.insert(0, SRC)
os.environ.setdefault('OPL_TAILT_FUSED', '1'); os.environ.setdefault('OPL_COARSE_FP32', '1')
os.environ.setdefault('OPL_COARSE_ELEM', '1'); os.environ.setdefault('OPL_TET_TRITON', '1')
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import models as MD  # noqa
import torch
import teacher as TE, trainlib as TL, fastnet as FN, bench_deploy as BD, fastidx as FI
import moments_ad as MA

FN.FUSED = True; FI.ON = True
body, case = sys.argv[1].split(':')
h = BD.ModelHolder('/root/autodl-tmp/OPL/S1/V2/A3_2grid/best.pt'); BD.TMP.mkdir(parents=True, exist_ok=True)
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
r = dict(case=case)


def run(fast, det):
    MA.E0_FAST = fast
    torch.use_deterministic_algorithms(det, warn_only=True)
    torch.cuda.synchronize(); t = time.perf_counter()
    s = MA.cell_sens(C, G, batch=2048, skip_full=True)
    torch.cuda.synchronize(); dt_ = time.perf_counter() - t
    torch.use_deterministic_algorithms(False)
    return s, round(1e3 * dt_, 1)


run(False, False); run(True, False)
a, ta = run(False, True); b, tb = run(True, True); a2, ta2 = run(False, True)
r['det'] = dict(bitwise_fast=bool(torch.equal(a, b)), bitwise_rerun=bool(torch.equal(a, a2)),
                rel_fast=float((a - b).abs().max() / a.abs().max()), ms=(ta, tb, ta2))
c, tc = run(False, False); d, td = run(True, False)
r['nondet'] = dict(rel_fast=float((c - d).abs().max() / c.abs().max()), ms=(tc, td),
                   per_col_rel=[float((c[:, k] - d[:, k]).abs().max() / c[:, k].abs().max()) for k in range(c.shape[1])])
print(json.dumps(r), flush=True)
