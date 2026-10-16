"""Which matmuls dominate a sparse training step? (profiler grouped by input shapes)"""
import sys, os
import numpy as np
import torch
from torch.profiler import profile, ProfilerActivity
import trainlib as TL
import models as MD
os.environ['SENS_REASSOC'] = '1'
ck = torch.load(sys.argv[1], map_location='cuda:0', weights_only=False); cfg = ck['cfg']
geo = TL.Geo(cfg['cases'][0], cfg['body'], cfg['data'], neumann=False, log=lambda s_: None)
q, s0 = geo.sample_with_sens(16, np.random.default_rng(0), {k: v for k, v in cfg['mix'].items() if k != 'adv'})
ok = ~torch.isnan(s0[0])
m = MD.build(cfg['model'], [geo], **dict(cfg.get('model_args', {}), sparse=True)).cuda(); m.load_state_dict(ck['model'], strict=False)
def step():
    u = geo.field(m, q); e = TL.energy(u, geo.C.K)
    loss = torch.log(e).mean() + (((geo.sens_hat(u[:, ok]) - s0[:, ok]) ** 2).sum(0) / (s0[:, ok] ** 2).sum(0)).mean()
    m.zero_grad(set_to_none=True); loss.backward()
for _ in range(2): step()
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU], record_shapes=True, with_stack=False) as prof:
    step(); torch.cuda.synchronize()
tab = prof.key_averages(group_by_input_shape=True)
rows = sorted([e for e in tab if e.key in ('aten::bmm', 'aten::addmm', 'aten::mm', 'aten::matmul', 'aten::einsum', 'aten::convolution', 'aten::cudnn_convolution')], key=lambda e: -e.device_time_total)
with open(sys.argv[2], 'w') as f:
    for e in rows[:25]:
        f.write('%-14s %8.2f ms  calls %3d  shapes %s\n' % (e.key, e.device_time_total / 1e3, e.count, str(e.input_shapes)[:160]))
