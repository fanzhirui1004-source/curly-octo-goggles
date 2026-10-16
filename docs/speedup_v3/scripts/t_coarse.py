"""Check of the src_v3 coarse solve (packed inverse + matrix-free prolongation) against the src_v2 float32 path on one
deployment cell: builds the cell twice (OPL_COARSE_PACK=0 / 1), compares _coarse_solve on seeded vectors and times it."""
import os, sys, time
SRC = os.environ.get('SRC', '/root/autodl-tmp/OPL/src_v3'); sys.path.insert(0, SRC)
os.environ['OPL_TAILT_FUSED'] = '1'; os.environ['OPL_COARSE_FP32'] = '1'; os.environ['OPL_COARSE_ELEM'] = '1'; os.environ['OPL_TET_TRITON'] = '1'
import models as MD  # noqa
import torch
import teacher as TE, trainlib as TL, fastnet as FN, bench_deploy as BD, fastidx as FI
FN.FUSED = True; FI.ON = True
body, case = sys.argv[1].split(':')
h = BD.ModelHolder('/root/autodl-tmp/OPL/S1/V2/A3_2grid/best.pt')
BD.TMP.mkdir(parents=True, exist_ok=True)
outs = {}
for mode in ('0', '1'):
    os.environ['OPL_COARSE_PACK'] = mode
    C = TE.Cell(case, body, log=lambda s_: None, deploy=True); C.assemble_deploy()
    BD.netdata(C, body, BD.TMP / case)
    geo = TL.Geo(case, body, BD.TMP, neumann=False, log=lambda s_: None, cell=C, load_banks=False)
    if h.model is not None:
        h.model.caches.pop(case, None)
    f = FN.FastNet(h.add(geo), geo)
    with torch.no_grad():
        f.s_hat(torch.zeros((C.np_, 1), dtype=torch.float64, device='cuda'))
    g = torch.Generator(device='cuda'); g.manual_seed(1)
    r = torch.randn((C.ni, 3), dtype=torch.float64, device='cuda', generator=g)
    y = TL._coarse_solve(C, r)
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(20):
        TL._coarse_solve(C, r)
    torch.cuda.synchronize()
    print('mode', mode, 'coarse_solve ms %.3f' % ((time.perf_counter() - t) / 20 * 1e3), 'inv_err', getattr(C, '_c_inv_err', None),
          'tri_err', getattr(C, '_c_tri_err', None), 'packed', getattr(C, '_cAp32', None) is not None, flush=True)
    q = torch.randn((C.np_, 3), dtype=torch.float64, device='cuda', generator=g)
    with torch.no_grad():
        outs[mode] = (y, f.s_hat(q))
    del C, geo, f; torch.cuda.empty_cache()
y0, s0 = outs['0']; y1, s1 = outs['1']
print('coarse_solve rel diff %.2e' % float((y1 - y0).norm() / y0.norm()))
print('S_hat rel diff %.2e' % float((s1 - s0).norm() / s0.norm()))
