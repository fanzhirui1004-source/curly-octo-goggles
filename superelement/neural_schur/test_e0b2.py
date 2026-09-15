"""E0-B, second attempt: is the stall in the structure or in the optimisation?

Joint Adam reached D/d 3.206e-5 with the oracle floor for its own T at 3.198e-5, so D was optimal
for that T and T had stopped moving.  This compares three routes on the same target and the same
initialisation.
"""
from __future__ import annotations

import sys
import numpy as np
import torch

from backend import QuotientOperator, multiscale_layers, spatial_blocks
from oracle import oracle_floor
from fit import fit_T, optimal_D_coefficients
import evaluator as EV

DT = torch.float64
FAILS = []


def check(name, ok, detail=''):
    print('  %-46s %s   %s' % (name, 'ok' if ok else 'FAIL', detail), flush=True)
    if not ok: FAILS.append(name)


def main():
    d = 150
    rng = np.random.default_rng(3)
    pts = rng.random((d, 3))
    layers, meta = multiscale_layers(pts, levels=3, radius0=0.25, growth=1.6, seed=3)
    op = QuotientOperator(d, layers, spatial_blocks(pts, 6))
    print('d %d  layer pairs %s  blocks %d' % (d, [m['pairs'] for m in meta], len(op.blocks)))

    torch.manual_seed(11)
    c_star = op.identity_coefficients(dtype=DT) + 0.30 * torch.randn(op.n_layer_coeff + op.n_block_coeff, dtype=DT)
    A = op.dense_A(c_star)
    Rstar = torch.linalg.cholesky(A, upper=True)
    torch.manual_seed(2026)
    c0 = op.identity_coefficients(dtype=DT) + 0.05 * torch.randn(c_star.shape, dtype=DT)

    print('\n  the target is expressible by construction, so a zero floor exists somewhere')
    print('  floor at the TARGET coefficients      : D/d %.6e' % oracle_floor(op, c_star, Rstar))
    print('  floor at the student START            : D/d %.6e' % oracle_floor(op, c0, Rstar))

    print('\n  route 1: descend floor(T), D implicit and always optimal')
    c1, _ = fit_T(op, c0, Rstar, steps=4000, lr=3e-3, log_every=1000)
    f1 = oracle_floor(op, c1, Rstar)
    c1 = optimal_D_coefficients(op, c1, Rstar)
    mu1 = EV.whitened_spectrum(op.dense_A(c1), Rstar)
    r1 = EV.assess(mu1, d, 0.03, 'floor-descent')
    print('   ', EV.report(r1))
    print('    floor reached %.6e ; realised D/d %.6e ; agreement %.2e'
          % (f1, r1['D_per_mode'], abs(f1 - r1['D_per_mode'])))

    check('closed-form D reproduces the floor', abs(f1 - r1['D_per_mode']) < 1e-9,
          'diff %.2e' % abs(f1 - r1['D_per_mode']))
    check('floor descent beats joint Adam (3.206e-5)', r1['D_per_mode'] < 3.206e-5,
          'D/d %.3e' % r1['D_per_mode'])
    check('floor descent reaches the target gate', r1['state'] == EV.PASS,
          'eps_op %.3e' % r1['eps_op'])

    print('\n  route 2: same, then a short joint polish on (T, D)')
    c2 = c1.clone().requires_grad_(True)
    Ainv = torch.linalg.inv(A); ldA = float(torch.linalg.slogdet(A)[1])
    opt = torch.optim.Adam([c2], lr=3e-4)
    for step in range(1, 2001):
        opt.zero_grad()
        Ah = op.dense_A(c2)
        ((Ainv * Ah).sum() - op.logdet_A(c2) + ldA - d).div(d).backward()
        opt.step()
    mu2 = EV.whitened_spectrum(op.dense_A(c2.detach()), Rstar)
    r2 = EV.assess(mu2, d, 0.03, 'floor+polish')
    print('   ', EV.report(r2))
    check('polish does not make it worse', r2['eps_op'] <= r1['eps_op'] * 1.05,
          'eps %.3e -> %.3e' % (r1['eps_op'], r2['eps_op']))


if __name__ == '__main__':
    main()
    print('\n%s' % ('passed' if not FAILS else 'FAILURES: %s' % FAILS))
    sys.exit(1 if FAILS else 0)
