"""Where the copies of one S_hat application come from: torch profiler with stacks, device time of aten::copy_ /
aten::clone / aten::contiguous grouped by the innermost source lines of this code base. Usage: prof_copy.py <body>:<case> [--B 3]"""
import os, sys, json, argparse, collections
import models as MD                                                    # noqa: F401
import torch
import teacher as TE, trainlib as TL, fastnet as FN, bench_deploy as BD
dev, dt = TE.dev, TE.dt
ap = argparse.ArgumentParser(); ap.add_argument('spec'); ap.add_argument('--B', type=int, default=3); a = ap.parse_args()
FN.FUSED = True; os.environ['OPL_TAILT_FUSED'] = '1'; os.environ['OPL_COARSE_FP32'] = '1'
body, case = a.spec.split(':')
BD.TMP.mkdir(parents=True, exist_ok=True)
C = TE.Cell(case, body, log=lambda s_: None, deploy=True); C.assemble_deploy()
h = BD.ModelHolder('/root/autodl-tmp/OPL/S1/V2/A3_2grid/best.pt'); BD.netdata(C, body, BD.TMP / case)
geo = TL.Geo(case, body, BD.TMP, neumann=False, log=lambda s_: None, cell=C, load_banks=False)
f = FN.FastNet(h.add(geo), geo)
q = torch.randn((C.np_, a.B), dtype=dt, device=dev)
for _ in range(2):
    f.s_hat(q)
torch.cuda.synchronize()
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], with_stack=True) as p:
    f.s_hat(q); torch.cuda.synchronize()
agg = collections.defaultdict(lambda: [0.0, 0])
for e in p.events():
    if e.name in ('aten::copy_', 'aten::clone', 'aten::contiguous', 'aten::index_add', 'aten::where', 'aten::take') and e.device_time_total > 0:
        st = [s for s in (e.stack or []) if 'src_v2' in s][:2]
        k = (e.name, ' <- '.join(s.split('src_v2/')[-1] for s in st))
        agg[k][0] += e.device_time_total / 1000; agg[k][1] += 1
for (n, s), (t, c) in sorted(agg.items(), key=lambda x: -x[1][0])[:25]:
    print(f'{t:7.3f} ms {c:4d}  {n:18s} {s}')
