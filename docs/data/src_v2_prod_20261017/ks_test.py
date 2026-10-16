"""OPL_KE_SYMV / OPL_GF_TRITON check: packed-upper Triton element product and fused ghost-face kernel vs the float32
bmm / template-GEMM path on a deploy cell (same _Kfp32), held, B in {1,3,6,12}; relative difference against the float64
product and timing."""
import sys, json, time, os
os.environ['OPL_KE_SYMV'] = '1'
import models as MD  # noqa
import torch, teacher as TE
import ke_symv as KS
body, case = sys.argv[1].split(':')
C = TE.Cell(case, body, log=lambda s_: None, deploy=True); C.assemble_deploy()
Kc = C._Kc
assert Kc.symv, 'symv not active'


def run(sym, gf, x, rep=20):
    Kc.symv, KS.GF_ON = sym, gf
    with Kc.hold():
        y = Kc @ x; torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(rep):
            y = Kc @ x
        torch.cuda.synchronize(); ms = 1000 * (time.perf_counter() - t) / rep
    return y, ms


out = []
for B in (1, 3, 6, 12):
    x = torch.randn((C.nb, B), dtype=torch.float64, device='cuda', generator=torch.Generator(device='cuda').manual_seed(B))
    y64 = C.K @ x
    n = float(y64.norm())
    y0, t0 = run(False, False, x)
    r = dict(case=case, B=B, base_ms=round(t0, 3), base_vs64=float((y0 - y64).norm()) / n)
    for tag, s, g in (('symv', True, False), ('gf', False, True), ('both', True, True)):
        y, t = run(s, g, x)
        r[tag + '_ms'] = round(t, 3); r[tag + '_vs64'] = float((y - y64).norm()) / n; r[tag + '_vs_base'] = float((y - y0).norm()) / n
    print(json.dumps(r), flush=True); out.append(r)
json.dump(out, open(sys.argv[2], 'w'), indent=1)
