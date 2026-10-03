#!/usr/bin/env python3
"""Independent full-spectrum audit of what E1-D actually produced.

The floor is a divergence.  The gate is eps_op = max_i |mu_i - 1|.  The hierarchical run showed
on this same teacher that those come apart badly: at blockwise tol 1e-3 it reached D/d = 1.27e-05,
clearing the +-3% NECESSARY condition by 36x, while eps_op = 0.229 missed the +-10% GATE by 2.3x.
A mean over 12792 modes cannot see a handful of bad ones.

So a descended floor decides nothing on its own.  This builds A_hat = T^T D T explicitly from each
arm's saved coefficients, with D set to its closed-form optimum for that T, and runs the evaluator
on the real whitened spectrum.  That is the fourth row of the table in CLAIM_SCOPE section 0: an
audit of one concrete operator, which can prove that operator passes or fails and nothing wider.
"""
import json, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, '/root/cutfem_neural_a_20260910/source_14301bc56')
sys.path.insert(0, '/root/cutfem_neural_a_20260910/superelement_v0')
sys.path.insert(0, '/root/autodl-tmp/NEURAL_SCHUR')
import v1_scaled as V
from stage_cutfem_neural_a.elimination_reference import load_upper_factor
from backend import QuotientOperator, multiscale_layers, checkerboard_layers, spatial_blocks
from fit import optimal_D_coefficients
from evaluator import whitened_spectrum, divergence

NEC3, NEC10 = 4.5921e-4, 5.3605e-3


def main():
    t0 = time.time()
    recs = json.load(open('/root/autodl-tmp/CUTFEM_SUPERELEMENT_LABELS/V1_LABELS.json'))
    rec = [r for r in recs if int(r['seat']) == 328][0]
    rec.setdefault('reference', '/root/autodl-tmp/CUTFEM_SUPERELEMENT_LABELS/REFERENCE_0328')
    label = V.Label(rec, 0.2, 0.03, 10.0)
    g = label.to_gpu(need_A=False, need_Z=False, z_dtype=torch.float32)
    d = label.d
    R = load_upper_factor(Path(rec['reference']) / 'R_UPPER.npy', d, V.DEV)
    order = g['data'].quotient.order.cpu().numpy(); posn = order[6:]
    pts = label.ijk[posn // 3].astype(float) / 64.0
    blocks = spatial_blocks(pts, 64)
    print('seat 328  d %d\n' % d, flush=True)
    print('%-14s %-9s %-12s %-12s %-8s %-8s %-11s %-11s %s'
          % ('arm', 'params', 'floor D/d', 'audited D/d', 'n>3%', 'n>10%', 'mu_min', 'mu_max', 'eps_op'), flush=True)

    E = json.load(open('/root/autodl-tmp/NEURAL_SCHUR/E1D.json'))
    rows = {}
    for name, fn, cap in (('median-plane', multiscale_layers, 220_000),
                          ('checkerboard', checkerboard_layers, 98_500)):
        layers, _ = fn(pts, levels=4, radius0=0.035, growth=2.0,
                       max_pairs_per_level=cap, device=V.DEV, seed=0)
        op = QuotientOperator(d, layers, blocks)
        coeff = torch.load('/root/autodl-tmp/NEURAL_SCHUR/E1D_%s.pt' % name).to(V.DEV)
        coeff = optimal_D_coefficients(op, coeff, R)      # D at its closed-form optimum for this T
        Ahat = op.dense_A(coeff)
        mu = whitened_spectrum(Ahat, R)
        del Ahat; torch.cuda.empty_cache()
        eps = float(np.abs(mu - 1).max())
        div = divergence(mu) / d
        o3 = int((np.abs(mu - 1) > 0.03).sum()); o10 = int((np.abs(mu - 1) > 0.10).sum())
        print('%-14s %-9d %-12.4e %-12.4e %-8d %-8d %-11.4e %-11.4e %.4e %s'
              % (name, E['arms'][name]['params'], E['arms'][name]['best'], div, o3, o10,
                 mu.min(), mu.max(), eps,
                 'PASSES +-3%' if eps <= 0.03 else 'PASSES +-10%' if eps <= 0.10 else 'fails'), flush=True)
        rows[name] = dict(params=E['arms'][name]['params'], floor=E['arms'][name]['best'],
                          audited_divergence_per_d=div, eps_op=eps, n_outside_3pct=o3,
                          n_outside_10pct=o10, mu_min=float(mu.min()), mu_max=float(mu.max()),
                          coverage=E['arms'][name]['coverage'])
        del op; torch.cuda.empty_cache()

    print('\nnecessary conditions: +-3%% needs D/d <= %.4e ; +-10%% needs D/d <= %.4e' % (NEC3, NEC10), flush=True)
    for n, r in rows.items():
        print('  %-14s divergence clears +-3%% necessary: %-5s | gate eps_op %.4e: %s'
              % (n, str(r['audited_divergence_per_d'] <= NEC3),
                 r['eps_op'], 'PASS' if r['eps_op'] <= 0.03 else 'FAIL'), flush=True)
    Path('/root/autodl-tmp/NEURAL_SCHUR/E1D_AUDIT.json').write_text(
        json.dumps(dict(seat=328, d=int(d), nec3=NEC3, nec10=NEC10, arms=rows,
                        seconds=time.time()-t0), indent=1))
    print('\nwritten E1D_AUDIT.json (%.0f s)' % (time.time()-t0), flush=True)


if __name__ == '__main__':
    main()
