import os, json, sys
import models as MD  # noqa
import torch, teacher as TE, moments_ad as MA, fastidx as FI
dev, dt = TE.dev, TE.dt
FI.ON = True
body, case = sys.argv[1].split(':')
C = TE.Cell(case, body, log=lambda s_: None, deploy=True); C.assemble_deploy()
u = torch.randn((C.nb, 3), dtype=dt, device=dev)
g = C.energy_density(u).sum(2, keepdim=True)
MA.cell_sens(C, g, batch=2048, skip_full=True); torch.cuda.synchronize()
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU], record_shapes=True) as p:
    MA.cell_sens(C, g, batch=2048, skip_full=True); torch.cuda.synchronize()
agg = {}
for e in p.events():
    if e.device_type == torch.autograd.DeviceType.CUDA:
        t, n = agg.get(e.name[:100], (0.0, 0)); agg[e.name[:100]] = (t + e.device_time_total / 1000, n + 1)
print(json.dumps(dict(gpu_ms=sum(v[0] for v in agg.values()), kernels=sum(v[1] for v in agg.values()))))
for k, (t, n) in sorted(agg.items(), key=lambda z: -z[1][0])[:16]:
    print(json.dumps(dict(ms=round(t, 1), n=n, name=k)))

ka = p.key_averages(group_by_input_shape=True)
rows = sorted(((e.key, str(e.input_shapes)[:120], e.device_time_total / 1000, e.count) for e in ka if e.device_time_total > 0 and e.key.startswith('aten::')), key=lambda z: -z[2])[:25]
for k, sh, t, n in rows:
    print(json.dumps(dict(op=k, ms=round(t, 1), n=n, shapes=sh)))
