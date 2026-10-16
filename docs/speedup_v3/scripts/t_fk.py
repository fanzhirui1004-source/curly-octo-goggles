"""Full-element sharing check: S_hat q with fullke on / off (same process, same cell) for every cell of a layout."""
import os, sys, json
SRC = os.environ.get('SRC', '/root/autodl-tmp/OPL/src_v3'); sys.path.insert(0, SRC)
os.environ['OPL_TAILT_FUSED'] = '1'; os.environ['OPL_COARSE_FP32'] = '1'; os.environ['OPL_COARSE_ELEM'] = '1'; os.environ['OPL_TET_TRITON'] = '1'
import models as MD  # noqa
import torch
import teacher as TE, trainlib as TL, fastnet as FN, bench_deploy as BD, fastidx as FI, fullke as FK
FN.FUSED = True; FI.ON = True
body = sys.argv[1]; L = json.load(open(sys.argv[2]))
h = BD.ModelHolder('/root/autodl-tmp/OPL/S1/V2/A3_2grid/best.pt'); BD.TMP.mkdir(parents=True, exist_ok=True)
for c in L['cells']:
    case = c['case']
    C = TE.Cell(case, body, log=lambda s_: None, deploy=True); C.assemble_deploy()
    BD.netdata(C, body, BD.TMP / case)
    geo = TL.Geo(case, body, BD.TMP, neumann=False, log=lambda s_: None, cell=C, load_banks=False)
    if h.model is not None:
        h.model.caches.pop(case, None)
    f = FN.FastNet(h.add(geo), geo)
    g = torch.Generator(device='cuda'); g.manual_seed(0)
    q = torch.randn((C.np_, 16), dtype=torch.float64, device='cuda', generator=g)
    out = {}
    for on in (False, True):
        FK.ON = on
        C._fk = None
        with torch.no_grad():
            f.s_hat(q[:, :1])
            out[on] = {B: f.s_hat(q[:, :B].contiguous()) for B in (1, 3, 16)}
            x = torch.randn((C.nb, 3), dtype=torch.float64, device='cuda', generator=torch.Generator(device='cuda').manual_seed(1))
            out[on]['K64'] = C.K @ x
            out[on]['Kc'] = C._Kc @ x
    fk = FK.setup(C)
    r = {str(k): float((out[True][k] - out[False][k]).norm() / out[False][k].norm()) for k in out[False]}
    print(json.dumps(dict(case=case, f64=(fk['f64']['n_full'], fk['f64'].get('patterns'), fk['f64'].get('mode_count')),
                          f32=(fk['f32']['n_full'], fk['f32'].get('patterns'), fk['f32'].get('mode_count')), rel=r)), flush=True)
    h.model.caches.pop(case, None)
    del C, geo, f; torch.cuda.empty_cache()
