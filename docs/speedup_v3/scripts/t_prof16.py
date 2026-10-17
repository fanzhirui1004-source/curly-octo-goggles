"""Where the time of a wide S_hat goes (the coarse A Z applies 16 columns per call): torch.profiler with input shapes on a
full plate cell, aten operators grouped by input shape and sorted by their CUDA time (self), at B = 16 and B = 1.
usage: python t_prof16.py BODY_DIR:CASE [--B 16,1] [--top 30]"""
import os, sys, json, argparse
SRC = os.environ.get('SRC', '/root/autodl-tmp/OPL/src_v3'); sys.path.insert(0, SRC)
os.environ['OPL_TAILT_FUSED'] = '1'; os.environ['OPL_COARSE_FP32'] = '1'; os.environ['OPL_COARSE_ELEM'] = '1'
os.environ['OPL_TET_TRITON'] = '1'
import models as MD  # noqa
import torch
import teacher as TE, trainlib as TL, fastnet as FN, bench_deploy as BD, fastidx as FI

ap = argparse.ArgumentParser(); ap.add_argument('spec'); ap.add_argument('--B', default='16,1'); ap.add_argument('--top', type=int, default=30)
a = ap.parse_args()
FN.FUSED = True; FI.ON = True
body, case = a.spec.split(':')
h = BD.ModelHolder('/root/autodl-tmp/OPL/S1/V2/A3_2grid/best.pt'); BD.TMP.mkdir(parents=True, exist_ok=True)
C = TE.Cell(case, body, log=lambda s_: None, deploy=True); C.assemble_deploy()
BD.netdata(C, body, BD.TMP / case)
geo = TL.Geo(case, body, BD.TMP, neumann=False, log=lambda s_: None, cell=C, load_banks=False)
f = FN.FastNet(h.add(geo), geo)
with torch.no_grad():
    f.s_hat(torch.zeros((C.np_, 1), dtype=torch.float64, device='cuda'))
if hasattr(C, 'store64'):
    C.store64(allow=True)
g = torch.Generator(device='cuda'); g.manual_seed(0)
for B in [int(x) for x in a.B.split(',')]:
    q = torch.randn((C.np_, B), dtype=torch.float64, device='cuda', generator=g)
    with torch.no_grad():
        f.s_hat(q); torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                                    record_shapes=True) as p:
            for _ in range(3):
                f.s_hat(q)
            torch.cuda.synchronize()
    rows = []
    for e in p.key_averages(group_by_input_shape=True):
        if e.device_type == torch.autograd.DeviceType.CPU and e.self_device_time_total > 0:
            rows.append((e.self_device_time_total / 3000, e.count // 3, e.key, str(e.input_shapes)[:160]))
    rows.sort(key=lambda r: -r[0])
    cnt = [(e.count // 3, e.cpu_time_total / 3000, e.key, str(e.input_shapes)[:160]) for e in p.key_averages(group_by_input_shape=True)
           if e.device_type == torch.autograd.DeviceType.CPU and e.key.startswith('aten::') and e.key not in
           ('aten::view', 'aten::as_strided', 'aten::reshape', 'aten::slice', 'aten::select', 'aten::t', 'aten::transpose',
            'aten::empty', 'aten::empty_strided', 'aten::_reshape_alias', 'aten::alias', 'aten::unsqueeze', 'aten::permute',
            'aten::detach', 'aten::resize_', 'aten::expand', 'aten::squeeze', 'aten::_unsafe_view', 'aten::lift_fresh')]
    cnt.sort(key=lambda r: -r[0])
    print(json.dumps(dict(B=B, total_ms=round(sum(r[0] for r in rows), 2),
                          top=[dict(ms=round(r[0], 3), calls=r[1], op=r[2], shapes=r[3]) for r in rows[:a.top]],
                          by_count=[dict(calls=r[0], cpu_ms=round(r[1], 3), op=r[2], shapes=r[3]) for r in cnt[:a.top]])), flush=True)
