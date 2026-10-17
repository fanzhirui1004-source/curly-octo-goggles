"""Sensitivity phase of one plate cell (opt_design.analyse_fast --sens ad): time of the field, energy density and
moments_ad.cell_sens (batch 2048, skip_full), and a torch.profiler summary of cell_sens (CUDA kernel time vs wall, top
kernels and top CPU ops) to see whether it is GPU- or host-bound.
usage: python t_sens.py BODY_DIR:CASE"""
import os, sys, json, time
SRC = os.environ.get('SRC', '/root/autodl-tmp/OPL/src_v3'); sys.path.insert(0, SRC)
os.environ.setdefault('OPL_TAILT_FUSED', '1'); os.environ.setdefault('OPL_COARSE_FP32', '1')
os.environ.setdefault('OPL_COARSE_ELEM', '1'); os.environ.setdefault('OPL_TET_TRITON', '1')
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
r = dict(case=case, elements=int(len(C.cells)))


def T(fn, rep=3):
    fn(); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(rep):
        out = fn()
    torch.cuda.synchronize(); return out, round(1e3 * (time.perf_counter() - t) / rep, 1)


with torch.no_grad():
    u, r['field_ms'] = T(lambda: f.field(q).to(torch.float64))
    gd, r['energy_ms'] = T(lambda: C.energy_density(u))
    e0 = torch.zeros((gd.shape[0], gd.shape[1], 1), dtype=gd.dtype, device=gd.device); e0[:, 0, 0] = 1.0
    G = torch.cat([gd, e0], 2)
s, r['cell_sens_ms'] = T(lambda: MA.cell_sens(C, G, batch=2048, skip_full=True))
for b in (1024, 4096):
    _, r[f'cell_sens_ms_batch{b}'] = T(lambda: MA.cell_sens(C, G, batch=b, skip_full=True), rep=2)
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
    MA.cell_sens(C, G, batch=2048, skip_full=True); torch.cuda.synchronize()
ka = p.key_averages()
dev_rows = [e for e in ka if e.device_type == torch.autograd.DeviceType.CUDA]
r['kernel_ms'] = round(sum(e.self_device_time_total for e in dev_rows) / 1e3, 1)
r['launches'] = sum(e.count for e in dev_rows)
r['top_kernels'] = [(e.key[:70], round(e.self_device_time_total / 1e3, 2), e.count) for e in sorted(dev_rows, key=lambda e: -e.self_device_time_total)[:12]]
cpu = [e for e in ka if e.device_type == torch.autograd.DeviceType.CPU]
r['top_cpu'] = [(e.key[:60], round(e.self_cpu_time_total / 1e3, 1), e.count) for e in sorted(cpu, key=lambda e: -e.self_cpu_time_total)[:12]]
print(json.dumps(r), flush=True)
