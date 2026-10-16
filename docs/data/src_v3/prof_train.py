"""Where does a training step of train1 spend its time? (MGNO2 on one geometry, the step-1 training path)
Usage: prof_train.py <checkpoint.pt> <out.json> [batch]"""
import sys, json, time
import numpy as np
import torch
import trainlib as TL
import models as MD

dev, dt = TL.dev, TL.dt


def sync():
    torch.cuda.synchronize()


def timeit(fn, reps=10, warm=2):
    for _ in range(warm):
        fn()
    sync(); t = time.perf_counter()
    for _ in range(reps):
        fn()
    sync(); return (time.perf_counter() - t) / reps


ck = torch.load(sys.argv[1], map_location='cuda:0', weights_only=False)
cfg = ck['cfg']; B = int(sys.argv[3]) if len(sys.argv) > 3 else cfg['batch']
geo = TL.Geo(cfg['cases'][0], cfg['body'], cfg['data'], log=lambda s_: None)
model = MD.build(cfg['model'], [geo], **cfg.get('model_args', {})).to(dev)
model.load_state_dict(ck['model'], strict=False); model.train()
gen = np.random.default_rng(0)
mix = {k: v for k, v in cfg['mix'].items() if k != 'adv'}
q, s0 = geo.sample_with_sens(B, gen, mix)
K = geo.C.K
r = dict(case=cfg['cases'][0], model=cfg['model'], batch=B, elements=int(len(geo.C.cells)), dofs=geo.nb)

r['sample'] = timeit(lambda: geo.sample_with_sens(B, gen, mix))
with torch.no_grad():
    r['model_fwd_nograd'] = timeit(lambda: geo.field(model, q))
    r['geometry_path'] = timeit(lambda: model.geometry(geo.case))


def fwd_only():
    u = geo.field(model, q); return u


r['model_fwd_grad'] = timeit(fwd_only)


def fwd_bwd_model():
    u = geo.field(model, q)
    (u.float() ** 2).sum().backward()
    model.zero_grad(set_to_none=True)


r['model_fwd_bwd'] = timeit(fwd_bwd_model)
with torch.no_grad():
    u = geo.field(model, q)
u64 = u.to(dt)
r['K_matvec_fp64'] = timeit(lambda: K @ u64)
u32 = u.float()
try:
    U32 = torch.sparse_csr_tensor(geo.C.U.crow_indices(), geo.C.U.col_indices(), geo.C.U.values().float(), size=geo.C.U.shape)
    Ut32 = torch.sparse_csr_tensor(geo.C.Ut.crow_indices(), geo.C.Ut.col_indices(), geo.C.Ut.values().float(), size=geo.C.Ut.shape)
    d32 = geo.C.dK.float()
    r['K_matvec_fp32'] = timeit(lambda: U32 @ u32 + Ut32 @ u32 - d32[:, None] * u32)
    del U32, Ut32
except Exception as e:
    r['K_matvec_fp32'] = repr(e)[:100]


def energy_fb():
    uu = u.detach().clone().requires_grad_(True)
    e = TL.energy(uu, K); torch.log(e).mean().backward()


r['energy_fwd_bwd'] = timeit(energy_fb)
ok = ~torch.isnan(s0[0])


def sens_fb():
    uu = u.detach().clone().requires_grad_(True)
    sh = geo.sens_hat(uu[:, ok])
    (((sh - s0[:, ok]) ** 2).sum(0) / (s0[:, ok] ** 2).sum(0)).mean().backward()


r['sens_fwd_bwd'] = timeit(sens_fb, reps=5)


def sens2_fb():
    uu = u.detach().clone().requires_grad_(True)
    sh = TL._Sens2.apply(uu[:, ok], geo.C.dofs, geo.Tm32, geo.dM32, 256)
    (((sh - s0[:, ok]) ** 2).sum(0) / (s0[:, ok] ** 2).sum(0)).mean().backward()


r['sens2_fwd_bwd'] = timeit(sens2_fb, reps=5)
# equality of the reassociated sensitivity loss (value and gradient) on this batch
g1u = u.detach().clone().requires_grad_(True); g2u = u.detach().clone().requires_grad_(True)
s1 = TL._Sens.apply(g1u[:, ok], geo.C.dofs, geo.Tm32, geo.dM32, 256)
s2 = TL._Sens2.apply(g2u[:, ok], geo.C.dofs, geo.Tm32, geo.dM32, 256)
w = torch.randn_like(s1)
(s1 * w).sum().backward(); (s2 * w).sum().backward()
r['sens2_value_rel'] = float(((s2 - s1).norm(dim=0) / s1.norm(dim=0)).max())
r['sens2_grad_rel'] = float((g2u.grad - g1u.grad).norm() / g1u.grad.norm())


def full_step():
    uu = geo.field(model, q)
    e = TL.energy(uu, K)
    loss = torch.log(e.clamp_min(1e-12)).mean()
    sh = geo.sens_hat(uu[:, ok])
    loss = loss + (((sh - s0[:, ok]) ** 2).sum(0) / (s0[:, ok] ** 2).sum(0)).mean()
    loss.backward(); model.zero_grad(set_to_none=True)


r['full_step'] = timeit(full_step, reps=5)
geo.adv = None
t = time.perf_counter()
X, ritz = geo.adversarial(model, k=16, iters=6, gen=torch.Generator(device=dev).manual_seed(0)); sync()
r['adversarial_call'] = time.perf_counter() - t
model.eval(); t = time.perf_counter(); geo.evaluate(model, 'val'); sync(); r['evaluate_call'] = time.perf_counter() - t
steps_per_adv, steps_per_eval = cfg.get('adv_every', 500), cfg['eval_every']
r['per_step_amortized'] = dict(adversarial=r['adversarial_call'] / steps_per_adv, evaluate=r['evaluate_call'] / steps_per_eval)
r['peak_GB'] = torch.cuda.max_memory_allocated() / 2 ** 30
print(json.dumps(r, indent=1), flush=True)
open(sys.argv[2], 'w').write(json.dumps(r, indent=1))
