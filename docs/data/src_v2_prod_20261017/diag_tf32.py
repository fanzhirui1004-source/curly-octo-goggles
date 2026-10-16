"""Do the U-Net convolutions run in TF32? PyTorch leaves torch.backends.cudnn.allow_tf32 = True by default and the code never
sets it, so cuDNN may pick TF32 tensor-core kernels for the float32 conv3d / conv_transpose3d (the float32 matmuls stay
exact: torch.backends.cuda.matmul.allow_tf32 defaults to False). On one geometry with a trained checkpoint:
  kernels     conv kernel names of one forward and whether they carry 'tf32'
  field       model field on 8 val directions per class with cuDNN TF32 on (the default: how every model so far was trained
              and evaluated) vs off: relative field difference, energy excess e_hat - 1 both ways (the variational bound holds
              either way: u is an admissible field)
  symmetry    fastnet.s_hat: max over 8 pairs of |y^T S_hat x - x^T S_hat y| / |y^T S_hat x| (forward conv3d vs adjoint
              conv_transpose3d: a TF32 forward and adjoint are not exact transposes), both ways
  time        no-grad forward of 16 directions and fastnet.s_hat of 16, both ways
Restores the default before exiting. Usage: diag_tf32.py <ckpt> <out.json> <case> [data]"""
import sys, json, time
import numpy as np
import torch
import trainlib as TL
import models as MD
import fastnet as FN
import train2 as T2

ckpt, out, case = sys.argv[1:4]
data = sys.argv[4] if len(sys.argv) > 4 else None
dev = TL.dev
ck = torch.load(ckpt, map_location=dev, weights_only=False)
cfg = ck['cfg']
geo = TL.Geo(case, cfg['body'], data or cfg['data'], neumann=False, log=lambda s_: None)
T2.clean_banks(geo, case, lambda d_: print(json.dumps(d_, default=str), flush=True))
m = MD.build(cfg['model'], [geo], **dict(cfg.get('model_args', {}), sparse=True)).to(dev)
(MD.load_compat(m, ck['model']) if hasattr(MD, 'load_compat') else m.load_state_dict(ck['model'], strict=False)); m.eval()
default = torch.backends.cudnn.allow_tf32
rec = dict(ckpt=ckpt, case=case, cudnn_allow_tf32_default=default, matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
           matmul_precision=torch.get_float32_matmul_precision(), cudnn_benchmark=torch.backends.cudnn.benchmark)
sp = 'val' if 'val' in geo.banks else sorted(geo.banks)[0]
classes = [c for c in geo.classes if geo.banks[sp][c].shape[1] >= 8]
Q = torch.cat([geo.banks[sp][c][:, :8] for c in classes], 1)
rec['classes'] = classes


def timed(fn, reps=5):
    fn(); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / reps


with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
    with torch.no_grad():
        geo.field(m, Q[:, :16])
    torch.cuda.synchronize()
names = sorted({e.key for e in prof.key_averages() if e.device_type == torch.autograd.DeviceType.CUDA})
conv = [n for n in names if any(k in n.lower() for k in ('conv', 'cudnn', 'implicit', 'winograd', 'xmma', 'fprop', 'dgrad'))]
rec['conv_kernels'] = [n[:140] for n in conv]
rec['tf32_kernels'] = [n[:140] for n in names if 'tf32' in n.lower()]

res = {}
try:
    for tf32 in (True, False):
        torch.backends.cudnn.allow_tf32 = tf32
        with torch.no_grad():
            u = geo.field(m, Q)
            e = (TL.energy(u, geo.C.K) - 1).cpu().numpy()
        fast = FN.FastNet(m, geo)
        x, y = Q[:, :8], Q[:, 8:16]
        a = (y * fast.s_hat(x)).sum(0); b = (x * fast.s_hat(y)).sum(0)
        res[tf32] = dict(u=u, e=e, asym=float(((a - b).abs() / a.abs()).max()),
                         t_field=timed(lambda: torch.no_grad()(geo.field)(m, Q[:, :16])), t_shat=timed(lambda: fast.s_hat(Q[:, :16])))
        del fast
finally:
    torch.backends.cudnn.allow_tf32 = default
per_class = lambda e: {c: float(e[8 * i:8 * i + 8].mean()) for i, c in enumerate(classes)}
rec.update(field_rel_tf32_vs_fp32=float((res[True]['u'] - res[False]['u']).norm() / res[False]['u'].norm()),
           excess_tf32=per_class(res[True]['e']), excess_fp32=per_class(res[False]['e']),
           excess_abs_diff_max=float(np.abs(res[True]['e'] - res[False]['e']).max()),
           excess_rel_diff_max=float((np.abs(res[True]['e'] - res[False]['e']) / np.abs(res[False]['e'])).max()),
           shat_asym_tf32=res[True]['asym'], shat_asym_fp32=res[False]['asym'],
           t_field16_tf32_ms=1e3 * res[True]['t_field'], t_field16_fp32_ms=1e3 * res[False]['t_field'],
           t_shat16_tf32_ms=1e3 * res[True]['t_shat'], t_shat16_fp32_ms=1e3 * res[False]['t_shat'])
print(json.dumps(rec, indent=1), flush=True)
open(out, 'w').write(json.dumps(rec, indent=1))
