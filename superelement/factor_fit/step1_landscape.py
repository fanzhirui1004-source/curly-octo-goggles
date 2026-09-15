#!/usr/bin/env python3
"""Step 1, measurement pass: what does the free-complete-factor landscape look like?

For a free upper-triangular student R_hat against the teacher R*, the exact
log-det divergence has a closed form with no spectrum and no probes:

    D(R_hat) = ||R_hat R*^-1||_F^2 - 2 sum_i log R_hat_ii + 2 sum_i log R*_ii - d
    dD/dR_hat = 2 R_hat A^-1 - 2 diag(1/R_hat_ii),        A = R*^T R*

D is convex in A_hat = R_hat^T R_hat and vanishes only at A_hat = A, so the
objective is not the difficulty. The difficulty, if any, is the factor
parameterization. This measures that before an optimizer is chosen:

  0  evaluator check: D(R*) must be 0
  1  where the usual initialisations actually start
  2  the curvature spread the parameterization imposes, raw and log-diagonal
"""
import argparse, json, math, time
from pathlib import Path
import numpy as np
import torch


def load_packed(path, device, dtype=torch.float64):
    """Row-packed upper triangle -> dense upper triangular."""
    p = np.load(path, mmap_mode='r')
    n = p.shape[0]; d = int((math.isqrt(8*n+1)-1)//2)
    if d*(d+1)//2 != n: raise ValueError('PACKED_TRIANGLE_SHAPE')
    R = torch.zeros((d, d), dtype=dtype, device=device); off = 0
    for i in range(d):
        R[i, i:] = torch.from_numpy(np.asarray(p[off:off+d-i], dtype=np.float64)).to(device=device, dtype=dtype)
        off += d-i
    return R


def divergence(Rhat, Rstar, logdet_A):
    """D = ||R_hat R*^-1||_F^2 - 2 sum log R_hat_ii + logdet A - d."""
    W = torch.linalg.solve_triangular(Rstar, Rhat, upper=True, left=False)
    tr = W.square().sum()
    return float(tr - 2.0*Rhat.diagonal().log().sum() + logdet_A - Rhat.shape[0]), float(tr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--factor', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    dev = torch.device(a.device); t0 = time.time()
    Rstar = load_packed(a.factor, dev); d = Rstar.shape[0]
    diag = Rstar.diagonal()
    logdet_A = float(2.0*diag.log().sum())
    print('d %d   loaded in %.1f s' % (d, time.time()-t0))
    print('teacher diagonal: min %.6e  max %.6e  ratio %.3e' % (float(diag.min()), float(diag.max()), float(diag.max()/diag.min())))

    out = dict(schema='STEP1_LANDSCAPE_V1', d=d, logdet_A=logdet_A,
               teacher_diag=dict(min=float(diag.min()), max=float(diag.max()), ratio=float(diag.max()/diag.min())))

    # 0. the evaluator must return exactly zero on the teacher itself
    D0, tr0 = divergence(Rstar, Rstar, logdet_A)
    print('\n0  D(R*) = %.6e   (trace term %.6f, should be d = %d)' % (D0, tr0, d))
    out['self_divergence'] = D0; out['self_trace'] = tr0

    # 1. where the usual initialisations start
    X = torch.linalg.solve_triangular(Rstar, torch.eye(d, dtype=torch.float64, device=dev), upper=True)  # R*^-1
    trAinv = float(X.square().sum())                 # tr(A^-1) = ||R*^-1||_F^2
    diagAinv = X.square().sum(dim=1)                 # (A^-1)_jj = ||row j of R*^-1||^2
    del X
    cstar = math.sqrt(d/trAinv)                      # argmin over R_hat = c I
    print('\n1  tr(A^-1) = %.6e   best scalar init c* = sqrt(d/tr(A^-1)) = %.6e' % (trAinv, cstar))
    inits = dict()
    for name, R in (('0.01_I', 0.01*torch.eye(d, dtype=torch.float64, device=dev)),
                    ('c_star_I', cstar*torch.eye(d, dtype=torch.float64, device=dev)),
                    ('diag_teacher', torch.diag(diag.clone())),
                    ('diag_sqrt_Aii', torch.diag(torch.linalg.vector_norm(Rstar, dim=0)))):
        D, tr = divergence(R, Rstar, logdet_A); del R
        inits[name] = dict(D=D, D_per_mode=D/d, trace=tr)
        print('   %-14s D = %14.6e   D/d = %12.6e' % (name, D, D/d))
    out['inits'] = inits; out['tr_Ainv'] = trAinv; out['c_star'] = cstar

    # 2. curvature the parameterization imposes, at the optimum
    #    raw   : d2D/dR_ij^2 = 2 (A^-1)_jj  (+ 2/R_ii^2 on the diagonal)
    #    logdiag: R_ii = exp(u_i) -> d2D/du_i^2 = 2 R_ii^2 (A^-1)_ii + 2
    hraw_off = 2.0*diagAinv
    hraw_dia = 2.0*diagAinv + 2.0/diag.square()
    hlog_dia = 2.0*diag.square()*diagAinv + 2.0
    def spread(v, name):
        v = v.double(); lo, hi = float(v.min()), float(v.max())
        r = dict(min=lo, max=hi, ratio=hi/lo, median=float(v.median()))
        print('   %-26s min %.4e  max %.4e  spread %.3e' % (name, lo, hi, hi/lo))
        return r
    print('\n2  curvature at the optimum')
    out['curvature'] = dict(
        raw_offdiagonal=spread(hraw_off, 'raw, off-diagonal cols'),
        raw_diagonal=spread(hraw_dia, 'raw, diagonal'),
        logdiag_diagonal=spread(hlog_dia, 'log-diagonal'))
    out['curvature']['raw_overall_spread'] = max(out['curvature']['raw_offdiagonal']['max'], out['curvature']['raw_diagonal']['max']) / \
                                             min(out['curvature']['raw_offdiagonal']['min'], out['curvature']['raw_diagonal']['min'])
    print('   raw parameterization overall spread: %.3e' % out['curvature']['raw_overall_spread'])
    print('   log-diagonal reduces the diagonal spread by %.3e x'
          % (out['curvature']['raw_diagonal']['ratio']/out['curvature']['logdiag_diagonal']['ratio']))

    out['seconds'] = time.time()-t0
    Path(a.out).write_text(json.dumps(out, indent=1))
    print('\nwritten', a.out, '(%.1f s)' % out['seconds'])


if __name__ == '__main__': main()
