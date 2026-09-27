import os, json, sys
import models as MD  # noqa
import torch, teacher as TE, polyref_torch_fast as PT
body, case = sys.argv[1].split(':')
C = TE.Cell(case, body, log=lambda s_: None, deploy=True)
PT.cell_moments(C.cells, C.n, C.taus0, C.normal, C.offset, C.s, levels=C.levels)
torch.cuda.synchronize()
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU]) as p:
    PT.cell_moments(C.cells, C.n, C.taus0, C.normal, C.offset, C.s, levels=C.levels); torch.cuda.synchronize()
agg = {}
for e in p.events():
    if e.device_type == torch.autograd.DeviceType.CUDA:
        t, n = agg.get(e.name[:100], (0.0, 0)); agg[e.name[:100]] = (t + e.device_time_total / 1000, n + 1)
print(json.dumps(dict(gpu_ms=sum(v[0] for v in agg.values()), kernels=sum(v[1] for v in agg.values()))))
for k, (t, n) in sorted(agg.items(), key=lambda z: -z[1][0])[:14]:
    print(json.dumps(dict(ms=round(t, 1), n=n, name=k)))
cpu = sorted(((e.key, e.cpu_time_total / 1000, e.count) for e in p.key_averages()), key=lambda z: -z[1])[:14]
for k, t, n in cpu:
    print(json.dumps(dict(cpu_ms=round(t, 1), n=n, op=k[:80])))
print(json.dumps(dict(levels=C.levels, s=C.s, E=len(C.cells))))
