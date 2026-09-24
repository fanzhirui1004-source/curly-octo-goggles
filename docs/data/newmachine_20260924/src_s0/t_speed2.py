"""Speed / equality of the remaining options on one geometry (sparse layers on): torch.compile of the geometry path,
cudnn.benchmark, and 2 x 8 directions vs 1 x 16. Usage: t_speed2.py <ckpt> <out.json>"""
import sys, json, time, os
import numpy as np
import torch
import trainlib as TL
import models as MD
os.environ['SENS_REASSOC'] = '1'
ck = torch.load(sys.argv[1], map_location='cuda:0', weights_only=False); cfg = ck['cfg']
geo = TL.Geo(cfg['cases'][0], cfg['body'], cfg['data'], neumann=False, log=lambda s_: None)
mix = {k: v for k, v in cfg['mix'].items() if k != 'adv'}
q16, s16 = geo.sample_with_sens(16, np.random.default_rng(0), mix)
rec = {}


def make():
    torch.manual_seed(0)
    m = MD.build(cfg['model'], [geo], **dict(cfg.get('model_args', {}), sparse=True)).cuda()
    m.load_state_dict(ck['model'], strict=False); m.train(); return m


def loss_of(m, q, s0):
    ok = ~torch.isnan(s0[0])
    u = geo.field(m, q); e = TL.energy(u, geo.C.K)
    return torch.log(e).mean() + (((geo.sens_hat(u[:, ok]) - s0[:, ok]) ** 2).sum(0) / (s0[:, ok] ** 2).sum(0)).mean()


def timed(fn, reps=6):
    for _ in range(2): fn()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(reps): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / reps


def grads(m):
    return torch.cat([p.grad.reshape(-1) for p in m.parameters() if p.grad is not None])


m = make()
def step16():
    m.zero_grad(set_to_none=True); loss_of(m, q16, s16).backward()
torch.backends.cudnn.benchmark = False
rec['base_B16_s'] = timed(step16)
step16(); g_ref = grads(m).clone(); l_ref = float(loss_of(m, q16, s16))
torch.backends.cudnn.benchmark = True
rec['cudnn_benchmark_B16_s'] = timed(step16)
step16(); rec['cudnn_benchmark_grad_rel'] = float((grads(m) - g_ref).norm() / g_ref.norm())
def step2x8():
    m.zero_grad(set_to_none=True)
    for sl in (slice(0, 8), slice(8, 16)):
        (loss_of(m, q16[:, sl], s16[:, sl]) / 2).backward()
rec['two_by_8_s'] = timed(step2x8)
# torch.compile of the geometry path
try:
    m2 = make()
    m2.geometry = torch.compile(m2.geometry, dynamic=False)
    def stepc():
        m2.zero_grad(set_to_none=True); loss_of(m2, q16, s16).backward()
    t = time.perf_counter(); stepc(); torch.cuda.synchronize(); rec['compile_first_step_s'] = time.perf_counter() - t
    rec['compile_geometry_B16_s'] = timed(stepc)
    stepc(); rec['compile_grad_rel'] = float((grads(m2) - g_ref).norm() / g_ref.norm())
    rec['compile_loss_rel'] = abs(float(loss_of(m2, q16, s16)) - l_ref) / l_ref
except Exception as e:
    rec['compile_error'] = repr(e)[:300]
print(json.dumps(rec, indent=1), flush=True)
open(sys.argv[2], 'w').write(json.dumps(rec, indent=1))
