"""E0-E (corrected) and E0-D.

E0-E  screen structures by the floor they can REACH, not the floor at T = I.  The first version
      held every lifting coefficient at zero, so T was the identity in all cases and the
      comparison was vacuous.
E0-D  cost and memory at the real interface size.  Cost only; no accuracy claim.
"""
from __future__ import annotations

import json, sys, time
import numpy as np
import torch

from backend import QuotientOperator, multiscale_layers, spatial_blocks
from oracle import oracle_floor
from fit import fit_T, optimal_D_coefficients
import evaluator as EV

DT = torch.float64


def e0e():
    print('\nE0-E  screen structures by the floor each can reach')
    d = 300
    rng = np.random.default_rng(7)
    pts = rng.random((d, 3))
    X = torch.as_tensor(pts, dtype=DT)
    A = torch.exp(-torch.cdist(X, X) / 0.25) + 0.05 * torch.eye(d, dtype=DT)
    Rstar = torch.linalg.cholesky(A, upper=True)
    thr = EV.sufficient_divergence_threshold(d, 0.03)
    print('  target: exponential-correlation SPD, d %d, cond %.2e' % (d, float(torch.linalg.cond(A))))
    print('  a floor above %.3e cannot certify the +-3%% gate by divergence alone\n' % thr)
    print('  levels  block  pairs   floor at T=I    floor reached   eps_op after   state')
    rows = []
    for levels, block in ((0, 6), (1, 6), (2, 6), (3, 6), (3, 20), (0, 20)):
        layers, meta = (multiscale_layers(pts, levels=levels, radius0=0.15, growth=2.0, seed=1)
                        if levels else ([], []))
        op = QuotientOperator(d, layers, spatial_blocks(pts, block))
        c0 = op.identity_coefficients(dtype=DT)
        f0 = oracle_floor(op, c0, Rstar)
        if op.n_layer_coeff:
            torch.manual_seed(5)
            c0 = c0 + 0.02 * torch.cat([torch.randn(op.n_layer_coeff, dtype=DT),
                                        torch.zeros(op.n_block_coeff, dtype=DT)])
            c1, _ = fit_T(op, c0, Rstar, steps=1500, lr=5e-3, verbose=False)
        else:
            c1 = c0
        f1 = oracle_floor(op, c1, Rstar)
        c1 = optimal_D_coefficients(op, c1, Rstar)
        mu = EV.whitened_spectrum(op.dense_A(c1), Rstar)
        r = EV.assess(mu, d, 0.03)
        pairs = sum(m['pairs'] for m in meta)
        print('  %-7d %-6d %-7d %.4e      %.4e     %.4e     %s'
              % (levels, block, pairs, f0, f1, r['eps_op'], r['state']), flush=True)
        rows.append(dict(levels=levels, block=block, pairs=pairs, floor_identity=f0,
                         floor_reached=f1, eps_op=r['eps_op'], state=r['state']))
    print('\n  reading: the floor at T=I is the same for every layer schedule because T=I there;')
    print('  only the floor REACHED separates structures.')
    return rows


def e0d():
    print('\nE0-D  cost at the real interface size (cost only, no accuracy claim)')
    import sys as _s
    _s.path.insert(0, '/root/cutfem_neural_a_20260910/source_14301bc56')
    _s.path.insert(0, '/root/cutfem_neural_a_20260910/superelement_v0')
    import v1_scaled as V
    recs = json.load(open('/root/autodl-tmp/CUTFEM_SUPERELEMENT_LABELS/V1_LABELS.json'))
    rec = [r for r in recs if int(r['seat']) == 328][0]
    label = V.Label(rec, 0.2, 0.03, 10.0)
    g = label.to_gpu(need_A=False, need_Z=False, z_dtype=torch.float32)
    d, q = label.d, label.q
    order = g['data'].quotient.order.cpu().numpy()
    ijk = label.ijk                                   # (nodes, 3) lattice indices
    pos = order[6:]                                   # quotient coord j -> physical dof pos[j]
    pts = ijk[pos // 3].astype(float) / 64.0          # quotient coord -> node position
    print('  d %d  q %d  quotient coordinates mapped to %d distinct nodes'
          % (d, q, len(np.unique(pos // 3))))
    dev = V.DEV
    rows = []
    for levels, radius0, block, cap in ((3, 0.05, 8, 400000), (5, 0.05, 8, 400000), (5, 0.05, 24, 400000)):
        t0 = time.time()
        layers, meta = multiscale_layers(pts, levels=levels, radius0=radius0, growth=2.0,
                                         max_pairs_per_level=cap, device=dev, seed=0)
        op = QuotientOperator(d, layers, spatial_blocks(pts, block))
        t_compile = time.time() - t0
        c = op.identity_coefficients(device=dev, dtype=torch.float64).requires_grad_(True)
        z = torch.randn(d, 1, dtype=torch.float64, device=dev)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        for _ in range(5): y = op.apply_A(c, z)
        torch.cuda.synchronize(); t_apply = (time.time() - t0) / 5
        t0 = time.time()
        loss = (z * op.apply_A(c, z)).sum() + op.logdet_A(c)
        loss.backward(); torch.cuda.synchronize(); t_bwd = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 2**30
        nparam = op.n_layer_coeff + op.n_block_coeff
        print('  levels %d block %2d  pairs %-8d params %-9d compile %.2fs  apply %.4fs  fwd+bwd %.3fs  peak %.2f GiB'
              % (levels, block, sum(m['pairs'] for m in meta), nparam, t_compile, t_apply, t_bwd, peak), flush=True)
        rows.append(dict(levels=levels, block=block, pairs=sum(m['pairs'] for m in meta),
                         params=nparam, t_compile=t_compile, t_apply=t_apply,
                         t_fwd_bwd=t_bwd, peak_gib=peak))
        del op, c, z, y, loss; torch.cuda.empty_cache()
    print('\n  reference: production PARDISO condenses this cell in 31.85 s on 64 cores.')
    return rows


if __name__ == '__main__':
    out = dict(e0e=e0e(), e0d=e0d())
    open('/root/autodl-tmp/NEURAL_SCHUR/E0DE.json', 'w').write(json.dumps(out, indent=1))
    print('\nwritten E0DE.json')
