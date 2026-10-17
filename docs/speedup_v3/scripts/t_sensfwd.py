"""Check of P6a (moments_ad.moments_vjp need_m=False: no clipped-tetrahedron moments in the forward pass of cell_sens,
OPL_V3_SENS_NOFWD): cell_sens of one full plate cell with and without it: bitwise equal (torch.equal) and the time.
usage: python t_sensfwd.py BODY_DIR:CASE [--rep 3]"""
import os, sys, json, time, argparse
SRC = os.environ.get('SRC', '/root/autodl-tmp/OPL/src_v3'); sys.path.insert(0, SRC)
os.environ.setdefault('OPL_TAILT_FUSED', '1'); os.environ.setdefault('OPL_COARSE_FP32', '1')
os.environ.setdefault('OPL_COARSE_ELEM', '1'); os.environ['OPL_TET_TRITON'] = '1'
import models as MD  # noqa
import torch
import teacher as TE, trainlib as TL, fastnet as FN, bench_deploy as BD, fastidx as FI
import moments_ad as MA

ap = argparse.ArgumentParser(); ap.add_argument('spec'); ap.add_argument('--rep', type=int, default=3)
a = ap.parse_args()
FN.FUSED = True; FI.ON = True
body, case = a.spec.split(':')
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


def T(fn, rep):
    fn(); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(rep):
        out = fn()
    torch.cuda.synchronize(); return out, round(1e3 * (time.perf_counter() - t) / rep, 1)


res = dict(case=case)
MA.NOFWD = False
s0, res['ms_with_fwd'] = T(lambda: MA.cell_sens(C, G, batch=2048, skip_full=True).detach().clone(), a.rep)
MA.NOFWD = True
s1, res['ms_no_fwd'] = T(lambda: MA.cell_sens(C, G, batch=2048, skip_full=True).detach().clone(), a.rep)
s1b = MA.cell_sens(C, G, batch=2048, skip_full=True).detach().clone()
MA.NOFWD = False
s0b = MA.cell_sens(C, G, batch=2048, skip_full=True).detach().clone()
res.update(equal=bool(torch.equal(s0, s1)), max_abs_diff=float((s0 - s1).abs().max()), scale=float(s0.abs().max()),
           with_fwd_rerun_equal=bool(torch.equal(s0, s0b)), no_fwd_rerun_equal=bool(torch.equal(s1, s1b)),
           rerun_max_abs_diff=float((s0 - s0b).abs().max()))
print(json.dumps(res), flush=True)
