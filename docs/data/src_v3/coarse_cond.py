"""Conditioning of the Jacobi-scaled cell coarse Galerkin matrix (diagnosis of the k = 4 design of case A, 2026-10-03):
for each <body>:<case>, coarse_setup by probing (OPL_COARSE_ELEM=0), A_s = L L^T, eigenvalues (fp64), the smallest squared
diagonal of L, the shift chosen by _chol_jitter, and the float32 round trip ||A_s^-1 b - fp32 solve|| / ||A_s^-1 b|| for
random b (the precision of the coarse solve of the timed route).
Usage: coarse_cond.py <body>:<case>[,...] [--space Q1_17]"""
import os, sys, json, argparse
import models as MD  # noqa
import torch
import teacher as TE, trainlib as TL
dev, dt = TE.dev, TE.dt
ap = argparse.ArgumentParser(); ap.add_argument('cases'); ap.add_argument('--space', default='Q1_17'); a = ap.parse_args()
os.environ['OPL_COARSE_ELEM'] = '0'
for spec in a.cases.split(','):
    body, case = spec.split(':')
    C = TE.Cell(case, body, log=lambda s_: None, deploy=True); C.assemble_deploy()
    TL.coarse_setup(C, a.space)
    L = C._cL; As = L @ L.T
    ev = torch.linalg.eigvalsh(As)
    b = torch.randn((L.shape[0], 8), dtype=dt, device=dev, generator=torch.Generator(device=dev).manual_seed(0))
    x64 = torch.cholesky_solve(b, L)
    L32 = L.to(torch.float32); x32 = torch.cholesky_solve(b.to(torch.float32), L32).to(dt)
    print(json.dumps(dict(case=case, nc=int(L.shape[0]), shift=C._c_shift, lam_min=float(ev[0]), lam_max=float(ev[-1]),
                          cond=float(ev[-1] / ev[0]), n_below_1e6=int((ev < 1e-6).sum()), n_below_1e8=int((ev < 1e-8).sum()),
                          diagL2_min=float((torch.diagonal(L) ** 2).min()),
                          fp32_solve_rel=float((x32 - x64).norm() / x64.norm()))), flush=True)
    del C, L, As, ev; torch.cuda.empty_cache()
