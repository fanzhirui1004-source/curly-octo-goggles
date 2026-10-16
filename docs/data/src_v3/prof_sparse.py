"""torch.profiler table of one training step (sparse vs einsum hyperedge layers). Usage: prof_sparse.py <ckpt> <out.txt> [batch]"""
import sys, os
import numpy as np
import torch
from torch.profiler import profile, ProfilerActivity
import trainlib as TL
import models as MD

os.environ['SENS_REASSOC'] = '1'
ck = torch.load(sys.argv[1], map_location='cuda:0', weights_only=False)
cfg = ck['cfg']; B = int(sys.argv[3]) if len(sys.argv) > 3 else 16
geo = TL.Geo(cfg['cases'][0], cfg['body'], cfg['data'], neumann=False, log=lambda s_: None)
q, s0 = geo.sample_with_sens(B, np.random.default_rng(0), {k: v for k, v in cfg['mix'].items() if k != 'adv'})
ok = ~torch.isnan(s0[0])
out = open(sys.argv[2], 'w')
for sparse in (True, False):
    m = MD.build(cfg['model'], [geo], **dict(cfg.get('model_args', {}), sparse=sparse)).cuda()
    m.load_state_dict(ck['model'], strict=False); m.train()

    def step():
        u = geo.field(m, q); e = TL.energy(u, geo.C.K)
        loss = torch.log(e).mean() + (((geo.sens_hat(u[:, ok]) - s0[:, ok]) ** 2).sum(0) / (s0[:, ok] ** 2).sum(0)).mean()
        m.zero_grad(set_to_none=True); loss.backward()
    for _ in range(2):
        step()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
        for _ in range(3):
            step()
        torch.cuda.synchronize()
    out.write(f'==== sparse={sparse}\n' + prof.key_averages().table(sort_by='cuda_time_total', row_limit=25) + '\n')
    del m; torch.cuda.empty_cache()
out.close()
