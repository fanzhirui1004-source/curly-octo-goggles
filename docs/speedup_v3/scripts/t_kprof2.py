"""Kernel time of one full plate cell's S_hat attributed to the Python call site (torch.profiler with_stack): for B = 1 and
16, the device time per (operator, call site) group, the top groups, so that every large kernel is traced to its line.
usage: python t_kprof2.py BODY_DIR:CASE [--B 1,16] [--top 40] [--depth 4]"""
import os, sys, json, time, argparse
SRC = os.environ.get('SRC', '/root/autodl-tmp/OPL/src_v3'); sys.path.insert(0, SRC)
os.environ['OPL_TAILT_FUSED'] = '1'; os.environ['OPL_COARSE_FP32'] = '1'; os.environ['OPL_COARSE_ELEM'] = '1'
os.environ['OPL_TET_TRITON'] = '1'
import models as MD  # noqa
import torch
import teacher as TE, trainlib as TL, fastnet as FN, bench_deploy as BD, fastidx as FI
from torch.profiler import profile, ProfilerActivity

dev = 'cuda'
ap = argparse.ArgumentParser(); ap.add_argument('spec'); ap.add_argument('--B', default='1,16'); ap.add_argument('--top', type=int, default=40)
ap.add_argument('--depth', type=int, default=4)
a = ap.parse_args()
FN.FUSED = True; FI.ON = True
body, case = a.spec.split(':')
h = BD.ModelHolder('/root/autodl-tmp/OPL/S1/V2/A3_2grid/best.pt'); BD.TMP.mkdir(parents=True, exist_ok=True)
C = TE.Cell(case, body, log=lambda s_: None, deploy=True); C.assemble_deploy()
BD.netdata(C, body, BD.TMP / case)
geo = TL.Geo(case, body, BD.TMP, neumann=False, log=lambda s_: None, cell=C, load_banks=False)
f = FN.FastNet(h.add(geo), geo)
with torch.no_grad():
    f.s_hat(torch.zeros((C.np_, 1), dtype=torch.float64, device=dev))
if hasattr(C, 'store64'):
    C.store64(allow=True)
g = torch.Generator(device=dev); g.manual_seed(0)
SRCP = os.path.realpath(SRC)


def site(stack):
    """The innermost frames of src_v3 files in an event's stack (file:line function)."""
    out = []
    for fr in stack:
        if SRCP in fr or 'src_v3' in fr:
            out.append(fr.split('/')[-1][:70])
        if len(out) >= a.depth:
            break
    return ' < '.join(out) if out else '?'


res = dict(case=case)
with torch.no_grad():
    for B in [int(x) for x in a.B.split(',')]:
        q = torch.randn((C.np_, B), dtype=torch.float64, device=dev, generator=g)
        f.s_hat(q); torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True) as p:
            for _ in range(3):
                f.s_hat(q)
            torch.cuda.synchronize()
        groups = {}
        tot = 0.0
        for e in p.events():
            if e.device_time_total <= 0 or e.device_type != torch.autograd.DeviceType.CPU:
                continue
            kids = getattr(e, 'cpu_children', [])
            if any(k.device_time_total > 0 for k in kids):                   # count a kernel at its innermost CPU op
                continue
            key = (e.name[:40], site(e.stack or []))
            gr = groups.setdefault(key, [0.0, 0])
            gr[0] += e.device_time_total / 1e3 / 3; gr[1] += 1
            tot += e.device_time_total / 1e3 / 3
        top = sorted(groups.items(), key=lambda kv: -kv[1][0])[:a.top]
        rows = [e for e in p.key_averages() if e.device_type == torch.autograd.DeviceType.CUDA]
        kern = sum(e.self_device_time_total for e in rows) / 1e3 / 3
        ktop = [dict(k=e.key[:70], ms=round(e.self_device_time_total / 1e3 / 3, 3), n=e.count // 3)
                for e in sorted(rows, key=lambda e: -e.self_device_time_total)[:a.top]]
        res[f'B{B}'] = dict(kernel_ms=round(kern, 2), attributed_ms=round(tot, 2),
                            top=[dict(op=k[0], site=k[1], ms=round(v[0], 3), n=v[1] // 3) for k, v in top], kernels=ktop)
        print(json.dumps({f'B{B}': dict(kernel_ms=round(kern, 2), attributed_to_aten_ops_ms=round(tot, 2))}), flush=True)
        for r in res[f'B{B}']['top']:
            print('   %8.3f ms %4d  %-28s %s' % (r['ms'], r['n'], r['op'], r['site']), flush=True)
        print('   -- kernels by name', flush=True)
        for r in ktop[:25]:
            print('   %8.3f ms %4d  %s' % (r['ms'], r['n'], r['k']), flush=True)
print(json.dumps(res), flush=True)
