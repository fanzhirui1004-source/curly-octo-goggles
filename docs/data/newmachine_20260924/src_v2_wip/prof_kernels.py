"""Where does the GPU time of one training step go, by kernel family? (torch.profiler over 3 steps after 3 warm-up steps,
the train1/train3 loss: log-energy + sensitivity, sparse q-path; optionally with the fused hyperedge kernel)
Families: sparse (cuSPARSE spmm / sddmm), conv (the U-Net 3D convolutions), gemm (bmm / gemm / gemv), triton (the fused hyperedge kernels), index (gather /
scatter / index_add / index_select), reduce, elementwise, copy (memcpy / memset / cat), other. Also the kernel launch count.
Usage: prof_kernels.py <ckpt> <out.json> [batch] [case] [data] [--fused]"""
import sys, json, os, re, time
from collections import defaultdict
import numpy as np
import torch
import trainlib as TL
import models as MD
import sparse_layers as SL

argv = [x for x in sys.argv[1:] if x != '--fused']
fused = '--fused' in sys.argv
ckpt, out = argv[0], argv[1]
dev = TL.dev
ck = torch.load(ckpt, map_location=dev, weights_only=False)
cfg = ck['cfg']
B = int(argv[2]) if len(argv) > 2 else cfg.get('batch', 16)
case = argv[3] if len(argv) > 3 else cfg['cases'][0]
geo = TL.Geo(case, cfg['body'], argv[4] if len(argv) > 4 else cfg['data'], neumann=False, log=lambda s_: None)
import train2 as T2
T2.clean_banks(geo, case, lambda d_: print(json.dumps(d_, default=str), flush=True))              # drop non-finite bank samples
m = MD.build(cfg['model'], [geo], **dict(cfg.get('model_args', {}), sparse=True)).to(dev)
(MD.load_compat(m, ck['model']) if hasattr(MD, 'load_compat') else m.load_state_dict(ck['model'], strict=False)); m.train()
SL.FUSED = fused
os.environ['SENS_REASSOC'] = '1'
mix = {k: v for k, v in (cfg.get('mix') or {}).items() if k in ('force', 'support', 'face', 'macro', 'grf')} or \
    dict.fromkeys(('force', 'support', 'face', 'macro', 'grf'), .2)
mix = {k: v for k, v in mix.items() if k in geo.classes}
q, s0 = geo.sample_with_sens(B, np.random.default_rng(0), mix)
ok = ~torch.isnan(s0[0])


def step():
    u = geo.field(m, q)
    loss = torch.log(TL.energy(u, geo.C.K).clamp_min(1e-12)).mean()
    sh = geo.sens_hat(u[:, ok])
    loss = loss + (((sh - s0[:, ok]) ** 2).sum(0) / (s0[:, ok] ** 2).sum(0)).mean()
    m.zero_grad(set_to_none=True); loss.backward()


for _ in range(3):
    step()
torch.cuda.synchronize(); t = time.perf_counter()
for _ in range(3):
    step()
torch.cuda.synchronize(); wall = (time.perf_counter() - t) / 3

RULES = [('triton', r'^_(gather|scatter|backward)|triton'), ('sparse', r'csr|spmm|sddmm|cusparse|sparse'),
         ('conv', r'conv|cudnn|winograd|implicit|fft'),
         ('gemm', r'gemm|gemv|cutlass|sm\d+_xmma|ampere|hopper|blackwell|bmm|matmul'),
         ('index', r'index|gather|scatter|embedding|take'), ('reduce', r'reduce|sum|norm|mean|softmax|max|min'),
         ('copy', r'memcpy|memset|copy|cat|fill'), ('elementwise', r'elementwise|vectorized|unrolled|pointwise|foreach')]


def family(name):
    n = name.lower()
    for f, rx in RULES:
        if re.search(rx, n):
            return f
    return 'other'


with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU]) as prof:
    for _ in range(3):
        step()
    torch.cuda.synchronize()
fam, top, launches = defaultdict(float), [], 0
for e in prof.key_averages():
    ct = getattr(e, 'device_time_total', None)
    if ct is None:
        ct = getattr(e, 'cuda_time_total', 0.0)
    if e.device_type == torch.autograd.DeviceType.CUDA and ct > 0:
        fam[family(e.key)] += ct / 3e3                                                     # ms per step
        top.append((ct / 3e3, e.count / 3, e.key[:120]))
        launches += e.count / 3
top.sort(reverse=True)
tot = sum(fam.values())
rec = dict(ckpt=ckpt, case=case, batch=B, fused=fused, dofs=int(geo.nb), elements=int(len(geo.C.cells)), wall_ms=1e3 * wall,
           gpu_ms=tot, launches_per_step=launches, families_ms={k: round(v, 2) for k, v in sorted(fam.items(), key=lambda x: -x[1])},
           families_share={k: round(v / tot, 3) for k, v in sorted(fam.items(), key=lambda x: -x[1])},
           top=[dict(ms=round(a, 3), calls=b, name=c) for a, b, c in top[:40]])
print(json.dumps({k: v for k, v in rec.items() if k != 'top'}, indent=1), flush=True)
for r in rec['top'][:25]:
    print(r, flush=True)
open(out, 'w').write(json.dumps(rec, indent=1))
