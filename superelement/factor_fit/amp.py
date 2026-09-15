#!/usr/bin/env python3
"""Why the H-matrix construction failed: the gate is not an entry-wise accuracy on A.

Acceptance is  ||R*^-T Ahat R*^-1 - I|| <= tau, i.e. a relative error in the A-NORM:
    Ahat = A + E   accepted  iff  ||A^-1/2 E A^-1/2|| <= tau.
Any compression whose tolerance is relative to ||A|| (SVD truncation, banding, thresholding)
controls ||E||/||A||, and those two differ by exactly cond(A).  Measure that factor.
"""
import json, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, '/root/cutfem_neural_a_20260910/source_14301bc56')
sys.path.insert(0, '/root/cutfem_neural_a_20260910/superelement_v0')
import v1_scaled as V
from stage_cutfem_neural_a.elimination_reference import load_upper_factor

recs = json.load(open('/root/autodl-tmp/CUTFEM_SUPERELEMENT_LABELS/V1_LABELS.json'))
rec = [r for r in recs if int(r['seat']) == 328][0]
rec.setdefault('reference', '/root/autodl-tmp/CUTFEM_SUPERELEMENT_LABELS/REFERENCE_0328')
label = V.Label(rec, 0.2, 0.03, 10.0)
g = label.to_gpu(need_A=False, need_Z=False, z_dtype=torch.float32)
d = label.d
R = load_upper_factor(Path(rec['reference']) / 'R_UPPER.npy', d, V.DEV)
A = R.T @ R
ev = torch.linalg.eigvalsh(A)
lmin, lmax = float(ev[0]), float(ev[-1])
print('seat 328  d %d' % d)
print('lam_min(A) = %.4e   lam_max(A) = %.4e   cond = %.4e' % (lmin, lmax, lmax/lmin), flush=True)
print('||A||_F = %.4e' % float(A.norm()), flush=True)
print()
print('entry-wise budget implied by the spectral gate (worst-case alignment):')
for tau, name in ((0.03, '+-3%'), (0.10, '+-10%')):
    print('   %-6s  ||E||_2 <= tau*lam_min = %.3e   = %.2e relative to lam_max'
          % (name, tau*lmin, tau*lmin/lmax), flush=True)
print()
# the amplification actually realised by the tol=1e-3 H-matrix run
H = json.load(open('/root/autodl-tmp/NEURAL_SCHUR/HMAT_ECONOMY.json'))
r = [x for x in H['rows'] if x['tol'] == 1e-3][0]
print('the H-matrix point at blockwise tol 1e-3 reported eps_op = %.4e with %d params (%.1f%% of dense)'
      % (r['eps_op'], r['params'], 100*r['frac_of_dense']), flush=True)
print('predicted amplification of an entry-relative tolerance: tol * lam_max / lam_min = %.3e'
      % (1e-3 * lmax / lmin), flush=True)
print('   (the measured eps_op sits inside this envelope, so no extra pathology is needed to explain it)')
print()
# the multiplicative alternative, on the same seat: Rhat = R(I+Delta) needs NO conditioning slack
print('multiplicative check: perturb the FACTOR, Rhat = (I+Delta) R with ||Delta|| = delta')
torch.manual_seed(0)
for delta in (1e-3, 1e-2, 3e-2):
    Dm = torch.randn(d, d, dtype=torch.float64, device=V.DEV)
    Dm *= delta / float(torch.linalg.matrix_norm(Dm, 2))
    X = (torch.eye(d, dtype=torch.float64, device=V.DEV) + Dm)
    Hm = X.T @ X                       # R^-T Rhat^T Rhat R^-1 = X^T X exactly
    mu = torch.linalg.eigvalsh(0.5*(Hm+Hm.T))
    print('   ||Delta|| = %.0e  ->  eps_op = %.4e   (bound 2*delta+delta^2 = %.4e)'
          % (delta, float((mu-1).abs().max()), 2*delta+delta*delta), flush=True)
    del Dm, X, Hm, mu; torch.cuda.empty_cache()
Path('/root/autodl-tmp/NEURAL_SCHUR/AMPLIFICATION.json').write_text(json.dumps(
    dict(seat=328, d=int(d), lam_min=lmin, lam_max=lmax, cond=lmax/lmin,
         budget_3pct_abs=0.03*lmin, budget_3pct_rel=0.03*lmin/lmax,
         budget_10pct_abs=0.10*lmin, budget_10pct_rel=0.10*lmin/lmax), indent=1))
print('\nwritten AMPLIFICATION.json')
