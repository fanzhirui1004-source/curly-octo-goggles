"""Coarse solve A_c^-1 (float32 Cholesky solve, current) vs explicit float32 inverse (cholesky_inverse in float64, cast):
time per call and deviation from the float64 solve, B in {3,6,12}; nc from a deploy cell after one S_hat application."""
import os, sys, json, time
import models as MD                                                    # noqa: F401
import torch
import teacher as TE, trainlib as TL, fastnet as FN, bench_deploy as BD
dev, dt, f32 = TE.dev, TE.dt, torch.float32
FN.FUSED = True; os.environ['OPL_TAILT_FUSED'] = '1'; os.environ['OPL_COARSE_FP32'] = '1'
body, case = sys.argv[1].split(':')
BD.TMP.mkdir(parents=True, exist_ok=True)
C = TE.Cell(case, body, log=lambda s_: None, deploy=True); C.assemble_deploy()
h = BD.ModelHolder('/root/autodl-tmp/OPL/S1/V2/A3_2grid/best.pt'); BD.netdata(C, body, BD.TMP / case)
geo = TL.Geo(case, body, BD.TMP, neumann=False, log=lambda s_: None, cell=C, load_banks=False)
f = FN.FastNet(h.add(geo), geo)
os.environ['OPL_COARSE_FP32'] = '0'                                   # keep the float64 factor for the reference
f.s_hat(torch.randn((C.np_, 1), dtype=dt, device=dev))
L64 = C._cL; nc = L64.shape[0]
t = time.perf_counter(); Ainv = torch.cholesky_inverse(L64); torch.cuda.synchronize(); tinv = time.perf_counter() - t
Ainv32 = Ainv.to(f32); L32 = L64.to(f32)
A = L64 @ L64.T
print('tf32', torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, flush=True)
for _ in range(2):
    t = time.perf_counter(); Ai32 = torch.cholesky_inverse(L32); torch.cuda.synchronize(); t32 = time.perf_counter() - t
for _ in range(2):
    t = time.perf_counter(); Li = torch.linalg.solve_triangular(L32, torch.eye(nc, device=dev, dtype=f32), upper=False)
    At = Li.T @ Li; torch.cuda.synchronize(); ttri = time.perf_counter() - t
print(json.dumps(dict(fp32_cholesky_inverse_s=t32, fp32_trinv_s=ttri, fp32_ci_vs64=float((Ai32.double() - Ainv).norm() / Ainv.norm()),
                      fp32_tri_vs64=float((At.double() - Ainv).norm() / Ainv.norm()), cast_vs64=float((Ainv32.double() - Ainv).norm() / Ainv.norm()))), flush=True)
print(json.dumps(dict(case=case, nc=nc, inv_seconds=tinv, cond_est=float(torch.linalg.cond(A)))), flush=True)


def tm(fn, rep=50):
    fn(); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(rep):
        y = fn()
    torch.cuda.synchronize(); return y, 1000 * (time.perf_counter() - t) / rep


for B in (3, 6, 12):
    z = torch.randn((nc, B), dtype=dt, device=dev)
    ref = torch.cholesky_solve(z, L64)
    z32 = z.to(f32)
    ys, ts = tm(lambda: torch.cholesky_solve(z32, L32))
    yi, ti = tm(lambda: Ainv32 @ z32)
    yc = Ai32 @ z32; yt = At @ z32
    n = float(ref.norm())
    print(json.dumps(dict(B=B, solve32_ms=ts, inv32_ms=ti, solve32_err=float((ys.double() - ref).norm()) / n,
                          inv32_err=float((yi.double() - ref).norm()) / n,
                          solve32_Aerr=float((A @ ys.double() - z).norm() / z.norm()),
                          inv32_Aerr=float((A @ yi.double() - z).norm() / z.norm()),
                          ci32_err=float((yc.double() - ref).norm()) / n, tri32_err=float((yt.double() - ref).norm()) / n,
                          ci32_Aerr=float((A @ yc.double() - z).norm() / z.norm()), tri32_Aerr=float((A @ yt.double() - z).norm() / z.norm()))), flush=True)
