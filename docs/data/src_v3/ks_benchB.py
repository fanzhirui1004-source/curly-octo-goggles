"""Per-switch timing against the original paths over column counts B (deploy cell, held element matrices as in the
correction): float32 element product (OPL_KE_SYMV), float64 K product (OPL_KE_SYMV64), coarse solve (OPL_COARSE_INV).
Writes <out.json> with per-B times and 'enable': the switches that are faster at B = 16 and 32 (the lattice widths)."""
import sys, json, time, os
os.environ['OPL_KE_SYMV'] = '1'
import models as MD  # noqa
import torch, teacher as TE
body, case = sys.argv[1].split(':'); out = sys.argv[2]
C = TE.Cell(case, body, log=lambda s_: None, deploy=True); C.assemble_deploy()
Kc = C._Kc


def tm(fn, rep=10):
    fn(); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(rep):
        fn()
    torch.cuda.synchronize(); return 1000 * (time.perf_counter() - t) / rep


res = {}
nc = 9000
A = torch.randn(nc, nc, dtype=torch.float64, device='cuda'); A = A @ A.T / nc + torch.eye(nc, dtype=torch.float64, device='cuda')
L32 = torch.linalg.cholesky(A).float(); Ai32 = torch.cholesky_inverse(L32); del A
for B in (1, 3, 6, 12, 16, 32, 64):
    x32 = torch.randn((C.nb, B), dtype=torch.float32, device='cuda'); x64 = x32.double()
    r = {}
    with Kc.hold():
        Kc.symv = False; r['f32_dense'] = tm(lambda: Kc @ x32)
        Kc.symv = True; r['f32_symv'] = tm(lambda: Kc @ x32)
    os.environ['OPL_KE_SYMV64'] = '0'; r['f64_dense'] = tm(lambda: C.K @ x64, 5)
    os.environ['OPL_KE_SYMV64'] = '1'; r['f64_symv'] = tm(lambda: C.K @ x64, 5)
    z = torch.randn((nc, B), device='cuda')
    r['coarse_chol'] = tm(lambda: torch.cholesky_solve(z, L32)); r['coarse_inv'] = tm(lambda: Ai32 @ z)
    res[B] = {k: round(v, 3) for k, v in r.items()}
    print(json.dumps({'B': B, **res[B]}), flush=True)
win = lambda a, b: all(res[B][a] < res[B][b] for B in (16, 32))
enable = dict(OPL_KE_SYMV=int(win('f32_symv', 'f32_dense')), OPL_KE_SYMV64=int(win('f64_symv', 'f64_dense')),
              OPL_COARSE_INV=int(win('coarse_inv', 'coarse_chol')))
print(json.dumps(dict(enable=enable)), flush=True)
json.dump(dict(case=case, per_B=res, enable=enable), open(out, 'w'), indent=1)
