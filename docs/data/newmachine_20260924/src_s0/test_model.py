"""Unit tests for a model: linearity in q, exact ports, exact rigid modes, adjoint consistency, time and memory.
Usage: test_model.py <model> <model_args_json|-> <body> <data> <case> [...]"""
import json, sys, time, gc
import torch
import trainlib as TL
import models as MD

name, margs, body, data = sys.argv[1:5]
margs = {} if margs == '-' else json.loads(margs)
for case in sys.argv[5:]:
    torch.cuda.reset_peak_memory_stats()
    geo = TL.Geo(case, body, data, neumann=False, log=lambda s_: None)
    model = MD.build(name, [geo], **margs).cuda()
    g = torch.Generator(device='cuda').manual_seed(0)
    q1 = geo.banks['val']['force'][:, :4].float(); q2 = geo.banks['val']['grf'][:, :4].float()
    with torch.no_grad():
        u1, u2 = geo.field(model, q1), geo.field(model, q2)
        u12 = geo.field(model, 0.7 * q1 - 1.3 * q2)
        lin = float((u12 - (0.7 * u1 - 1.3 * u2)).norm() / u12.norm())
        port = float((u1[geo.P] - q1).norm() / q1.norm())
        R = geo.C.Q.float()
        uR = geo.field(model, R)
        eR = TL.energy(uR, geo.C.K)
        e1 = TL.energy(u1, geo.C.K)
    # adjoint: <S_hat q1, q2> = <q1, S_hat q2>
    a = float((geo.s_hat_apply(model, q1) * q2.double()).sum()); b = float((q1.double() * geo.s_hat_apply(model, q2)).sum())
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(3):
        with torch.no_grad():
            geo.field(model, geo.banks['val']['force'][:, :16].float())
    torch.cuda.synchronize(); tf = (time.perf_counter() - t) / 3
    torch.cuda.synchronize(); t = time.perf_counter()
    geo.s_hat_apply(model, geo.banks['val']['force'][:, :16].float()); torch.cuda.synchronize(); ta = time.perf_counter() - t
    print(json.dumps(dict(case=case, params=sum(p.numel() for p in model.parameters()), linearity=lin, port=port,
                          rigid_energy_max=float(eR.abs().max()), energy_force_init=e1.tolist(),
                          adjoint_sym=abs(a - b) / max(abs(a), abs(b)), forward16_s=tf, apply16_s=ta,
                          peak_gb=torch.cuda.max_memory_allocated() / 2 ** 30)), flush=True)
    geo.C._free(); del geo, model; gc.collect(); torch.cuda.empty_cache()
