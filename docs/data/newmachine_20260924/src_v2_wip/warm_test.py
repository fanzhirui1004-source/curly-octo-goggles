"""Setup (coarse probing + Chebyshev bounds) of a deployment cell with fastidx off / on (held float64 K_e, element gathers):
times and differences of the coarse factor, the prolongation and the bounds."""
import os, json, time, sys
import models as MD  # noqa
import torch, teacher as TE, trainlib as TL, fastidx as FI
dev, dt = TE.dev, TE.dt
def T():
    torch.cuda.synchronize(); return time.perf_counter()
KEYS = ('_cV', '_cL', '_c_space', '_tail_bounds', '_cL32', '_cV32', '_cV32t')
for spec in sys.argv[1].split(','):
    body, case = spec.split(':')
    C = TE.Cell(case, body, log=lambda s_: None, deploy=True); C.assemble_deploy()
    r, out = {}, {}
    for on in (False, True):
        FI.ON = on
        for k in KEYS:
            setattr(C, k, None)
        t = T(); TL.coarse_setup(C, 'Q1_17'); r[f'coarse_s_{on}'] = T() - t
        t = T(); b = TL.tail_bounds(C, 30.0); r[f'bounds_s_{on}'] = T() - t
        out[on] = (C._cL.clone(), C._cV.values().clone(), b)
    (L0, V0, b0), (L1, V1, b1) = out[False], out[True]
    r.update(L_rel=float((L1 - L0).norm() / L0.norm()), V_rel=float((V1 - V0).norm() / V0.norm()),
             bounds=[b0[1], b1[1]], bounds_rel=abs(b1[1] - b0[1]) / b0[1])
    print(json.dumps(dict(case=case, **r)), flush=True)
    del C, out; torch.cuda.empty_cache()
