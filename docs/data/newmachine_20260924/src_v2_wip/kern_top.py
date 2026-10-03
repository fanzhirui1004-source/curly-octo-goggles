"""CUDA kernels of one S_hat application (deployment cell, B columns) by self GPU time, grouped by name, with the
enclosing Python-level segment (record_function ranges over field / K / field_T)."""
import os, json, sys
import models as MD  # noqa
import torch, teacher as TE, trainlib as TL, fastnet as FN, bench_deploy as BD, fastidx as FI
dev, dt, f32 = TE.dev, TE.dt, torch.float32
FN.FUSED = True; os.environ['OPL_COARSE_FP32'] = '1'; os.environ['OPL_TAILT_FUSED'] = '1'; os.environ['OPL_TET_TRITON'] = '1'; FI.ON = True
body, case = sys.argv[1].split(':'); B = int(sys.argv[2])
BD.TMP.mkdir(parents=True, exist_ok=True)
C = TE.Cell(case, body, log=lambda s_: None, deploy=True); C.assemble_deploy()
h = BD.ModelHolder('/root/autodl-tmp/OPL/S1/V2/A3_2grid/best.pt'); BD.netdata(C, body, BD.TMP / case)
geo = TL.Geo(case, body, BD.TMP, neumann=False, log=lambda s_: None, cell=C, load_banks=False)
f = FN.FastNet(h.add(geo), geo)
q = torch.randn((C.np_, B), dtype=dt, device=dev)
f.s_hat(q); torch.cuda.synchronize()
seg = {}
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as p:
    f.s_hat(q); torch.cuda.synchronize()
agg = {}
for e in p.events():
    if e.device_type == torch.autograd.DeviceType.CUDA:
        k = e.name[:110]
        t, n = agg.get(k, (0.0, 0)); agg[k] = (t + e.device_time_total / 1000, n + 1)
tot = sum(v[0] for v in agg.values())
print(json.dumps(dict(total_ms=tot, kernels=sum(v[1] for v in agg.values()))))
for k, (t, n) in sorted(agg.items(), key=lambda z: -z[1][0])[:28]:
    print(json.dumps(dict(ms=round(t, 2), n=n, us_per=round(1000 * t / n, 1), name=k)))
