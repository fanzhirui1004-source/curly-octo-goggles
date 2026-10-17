"""Kernel-level profile of one full plate cell's S_hat (src_v3 defaults): device time per kernel (name, calls, ms), grouped
into categories, the sum of kernel time against the wall time (one stream: the GPU's idle share), for B = 1 (the PCG
matvec) and B = 16 (the coarse A Z chunks); also with two cells on two streams through CUDA graphs (the solve's case).
usage: python t_kprof.py BODY_DIR:CASE[,CASE2] [--B 1,16] [--top 40]"""
import os, sys, json, time, argparse, re
SRC = os.environ.get('SRC', '/root/autodl-tmp/OPL/src_v3'); sys.path.insert(0, SRC)
os.environ['OPL_TAILT_FUSED'] = '1'; os.environ['OPL_COARSE_FP32'] = '1'; os.environ['OPL_COARSE_ELEM'] = '1'
os.environ['OPL_TET_TRITON'] = '1'
import models as MD  # noqa
import torch
import teacher as TE, trainlib as TL, fastnet as FN, bench_deploy as BD, fastidx as FI
from torch.profiler import profile, ProfilerActivity

dev = 'cuda'
ap = argparse.ArgumentParser(); ap.add_argument('spec'); ap.add_argument('--B', default='1,16'); ap.add_argument('--top', type=int, default=40)
a = ap.parse_args()
FN.FUSED = True; FI.ON = True
body, cases = a.spec.split(':')
h = BD.ModelHolder('/root/autodl-tmp/OPL/S1/V2/A3_2grid/best.pt'); BD.TMP.mkdir(parents=True, exist_ok=True)


def cell(case):
    C = TE.Cell(case, body, log=lambda s_: None, deploy=True); C.assemble_deploy()
    BD.netdata(C, body, BD.TMP / case)
    geo = TL.Geo(case, body, BD.TMP, neumann=False, log=lambda s_: None, cell=C, load_banks=False)
    f = FN.FastNet(h.add(geo), geo)
    with torch.no_grad():
        f.s_hat(torch.zeros((C.np_, 1), dtype=torch.float64, device=dev))
    if hasattr(C, 'store64'):
        C.store64(allow=True)
    return C, f


CATS = [('ke_symv', r'symv|_ke|kmat|KMat|_kp|ke64|elem_prod'), ('hyper', r'gather|scatter|_hyper|htile'),
        ('cheb', r'cheb'), ('coarse', r'symm_map|csr_sub|coarse|cblk|_tiles|symtile'), ('conv', r'conv|cudnn|implicit|xmma|winograd|fft|dgrad|fprop|cutlass.*conv'),
        ('gemm', r'gemm|sgemm|dgemm|cutlass|cublas|Kernel2|splitK|reduce_kernel.*gemm'), ('net_fused', r'mask_split|conv_in|conv_out|clip|assign'),
        ('sparse', r'csrmm|spmm|cusparse|csr'), ('index', r'index|Index|gather_|take'), ('elementwise', r'elementwise|vectorized|unrolled|where|fill'),
        ('reduce', r'reduce|sum|norm'), ('copy', r'[Cc]opy|memcpy|Memcpy|memset|Memset|cat')]


def cat_of(name):
    for c, rx in CATS:
        if re.search(rx, name):
            return c
    return 'other'


def prof(fn, reps=3):
    fn(); torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
        t = time.perf_counter()
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
        wall = (time.perf_counter() - t) / reps
    rows = [e for e in p.key_averages() if e.device_type == torch.autograd.DeviceType.CUDA]
    kern = sum(e.self_device_time_total for e in rows) / 1e3 / reps
    cats = {}
    for e in rows:
        c = cat_of(e.key)
        ms, n = cats.get(c, (0.0, 0))
        cats[c] = (ms + e.self_device_time_total / 1e3 / reps, n + e.count // reps)
    top = [dict(k=e.key[:90], ms=round(e.self_device_time_total / 1e3 / reps, 3), n=e.count // reps, cat=cat_of(e.key))
           for e in sorted(rows, key=lambda e: -e.self_device_time_total)[:a.top]]
    return dict(wall_ms=round(1e3 * wall, 2), kernel_ms=round(kern, 2), launches=sum(e.count for e in rows) // reps,
                cats={k: dict(ms=round(v[0], 2), n=v[1]) for k, v in sorted(cats.items(), key=lambda kv: -kv[1][0])}, top=top)


cells = [cell(c) for c in cases.split(',')]
C, f = cells[0]
g = torch.Generator(device=dev); g.manual_seed(0)
res = dict(case=C.case if hasattr(C, 'case') else cases.split(',')[0])
with torch.no_grad():
    for B in [int(x) for x in a.B.split(',')]:
        q = torch.randn((C.np_, B), dtype=torch.float64, device=dev, generator=g)
        r = prof(lambda: f.s_hat(q))
        res[f'B{B}'] = r
        print(json.dumps({f'B{B}': {k: v for k, v in r.items() if k != 'top'}}), flush=True)
        for t in r['top'][:25]:
            print('   %8.3f ms %5d  %-12s %s' % (t['ms'], t['n'], t['cat'], t['k']), flush=True)
print(json.dumps(res), flush=True)
