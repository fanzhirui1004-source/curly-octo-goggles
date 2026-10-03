"""Is the fp32 difference between fastnet and the autograd model rounding? Reference: the autograd model in float64.
Usage: t_fast64.py <out_json> <checkpoint.pt> [...]"""
import sys, json, copy, gc
from pathlib import Path
import torch
import trainlib as TL
import models as MD
import fastnet as FN

dev, f32, f64 = TL.dev, torch.float32, torch.float64


def to64(model, case):
    m = copy.deepcopy(model).double()
    c = m.caches[case]
    for k, v in list(vars(c).items()):
        if torch.is_tensor(v) and v.dtype == f32:
            setattr(c, k, v.double())
    for t in c.trans:
        t['w'] = t['w'].double()
    return m


def rel(a, b):
    a, b = a.to(f64), b.to(f64)
    return [float(x) for x in ((a - b).norm(dim=0) / b.norm(dim=0))]


rec = {}
for ck_path in sys.argv[2:]:
    ck = torch.load(ck_path, map_location='cuda:0', weights_only=False)
    cfg = ck['cfg']; case = cfg['cases'][0]
    geo = TL.Geo(case, cfg['body'], cfg['data'], neumann=False, log=lambda s_: None)
    model = MD.build(cfg['model'], [geo], **cfg.get('model_args', {})).cuda()
    model.load_state_dict(ck['model'], strict=False); model.eval()
    fast = FN.FastNet(model, geo)
    gen = torch.Generator(device=dev).manual_seed(0)
    Qb = torch.cat([geo.banks['val'][c][:, :2] for c in geo.classes], 1)
    Qr = torch.randn((geo.np_, 3), dtype=f32, device=dev, generator=gen)
    y = torch.randn((geo.nb, 3), dtype=f32, device=dev, generator=gen)
    m64 = to64(model, case)
    torch.set_default_dtype(f64)
    with torch.no_grad():
        u64b, u64r = m64(geo, Qb.double()), m64(geo, Qr.double())
    qq = Qr.double().clone().requires_grad_(True)
    with torch.enable_grad():
        g64 = torch.autograd.grad(m64(geo, qq), qq, grad_outputs=y.double())[0]
    torch.set_default_dtype(f32)
    with torch.no_grad():
        ub32, ur32 = model(geo, Qb), model(geo, Qr)
    qq = Qr.clone().requires_grad_(True)
    with torch.enable_grad():
        g32 = torch.autograd.grad(model(geo, qq), qq, grad_outputs=y)[0]
    r = dict(case=case, classes=geo.classes,
             bank_fast_vs64=rel(fast.ext(Qb), u64b), bank_autograd32_vs64=rel(ub32, u64b),
             noise_fast_vs64=rel(fast.ext(Qr), u64r), noise_autograd32_vs64=rel(ur32, u64r),
             adj_fast_vs64=rel(fast.ext_T(y), g64), adj_autograd32_vs64=rel(g32, g64))
    rec[ck_path] = r
    print(json.dumps(r), flush=True)
    del fast, model, m64, geo; gc.collect(); torch.cuda.empty_cache()
Path(sys.argv[1]).write_text(json.dumps(rec, indent=1))
