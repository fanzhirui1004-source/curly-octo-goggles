"""E1-B: the oracle floor for the REAL teacher, before any training.

This is the cheapest decisive test in the plan.  For a given T and block partition the best
block-diagonal D reaches exactly

    floor(T) = sum_b logdet C_bb - logdet C,     C = T A^-1 T^T = (T R*^-1)(T R*^-1)^T

and nothing in the class can go below it.  So a structure can be rejected, or shown promising,
without training it.

Reference points on this seat, all measured earlier:
  certifying threshold for the +-3% gate    D/d = 3.449e-8
  free DENSE Cholesky factor                D/d = 2.11e-5   (passes +-10%, fails +-3%)
  teacher factor truncated to radius 0.2    D/d = 0.030-0.032
  old backend at R7 step 1000               D/d = 0.0350
"""
from __future__ import annotations

import json, sys, time
import numpy as np
import torch

sys.path.insert(0, '/root/cutfem_neural_a_20260910/source_14301bc56')
sys.path.insert(0, '/root/cutfem_neural_a_20260910/superelement_v0')
import v1_scaled as V

from backend import QuotientOperator, multiscale_layers, spatial_blocks
from oracle import oracle_floor
from fit import fit_T
import evaluator as EV


def main():
    t0 = time.time()
    recs = json.load(open('/root/autodl-tmp/CUTFEM_SUPERELEMENT_LABELS/V1_LABELS.json'))
    rec = [r for r in recs if int(r['seat']) == 328][0]
    label = V.Label(rec, 0.2, 0.03, 10.0)
    g = label.to_gpu(need_A=False, need_Z=False, z_dtype=torch.float32)
    d = label.d
    Rstar = g['Rstar']
    order = g['data'].quotient.order.cpu().numpy()
    pos = order[6:]
    pts = label.ijk[pos // 3].astype(float) / 64.0
    comp = (pos % 3).astype(float)[:, None] * 1e-6           # keep the 3 components distinguishable
    pts = np.concatenate([pts, comp], axis=1)[:, :3]
    thr = EV.sufficient_divergence_threshold(d, 0.03)
    print('seat 328  d %d  certifying threshold D/d %.3e  (%.0f s)' % (d, thr, time.time()-t0), flush=True)
    print('reference: dense free factor 2.11e-5 | truncated teacher 0.030-0.032 | old backend 0.0350\n', flush=True)

    print('%-8s %-7s %-10s %-14s %-14s %s' % ('levels', 'block', 'pairs', 'floor at T=I', 'floor reached', 'seconds'))
    rows = []
    for levels, radius0, block, steps in ((0, 0.0, 8, 0), (0, 0.0, 24, 0), (0, 0.0, 64, 0),
                                          (3, 0.05, 24, 300), (5, 0.05, 24, 300)):
        ts = time.time()
        layers, meta = (multiscale_layers(pts, levels=levels, radius0=radius0, growth=2.0,
                                          max_pairs_per_level=300000, device=V.DEV, seed=0)
                        if levels else ([], []))
        op = QuotientOperator(d, layers, spatial_blocks(pts, block))
        c = op.identity_coefficients(device=V.DEV, dtype=torch.float64)
        f0 = oracle_floor(op, c, Rstar)
        f1 = f0
        if steps and op.n_layer_coeff:
            torch.manual_seed(0)
            c = c + 0.01 * torch.cat([torch.randn(op.n_layer_coeff, dtype=torch.float64, device=V.DEV),
                                      torch.zeros(op.n_block_coeff, dtype=torch.float64, device=V.DEV)])
            c, _ = fit_T(op, c, Rstar, steps=steps, lr=5e-3, verbose=False)
            f1 = oracle_floor(op, c, Rstar)
        pairs = sum(m['pairs'] for m in meta)
        print('%-8d %-7d %-10d %-14.4e %-14.4e %.0f' % (levels, block, pairs, f0, f1, time.time()-ts), flush=True)
        rows.append(dict(levels=levels, block=block, pairs=pairs, floor_identity=float(f0),
                         floor_reached=float(f1), seconds=time.time()-ts))
        del op, c; torch.cuda.empty_cache()
    json.dump(dict(seat=328, d=int(d), threshold=thr, rows=rows),
              open('/root/autodl-tmp/NEURAL_SCHUR/E1B.json', 'w'), indent=1)
    print('\nwritten E1B.json (%.0f s total)' % (time.time()-t0))


if __name__ == '__main__':
    main()
