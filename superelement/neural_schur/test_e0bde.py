"""E0-B, E0-D, E0-E.

E0-B  a target this backend can express, relearned from an independent initialisation
E0-D  cost and memory at the real interface size, with no claim about accuracy
E0-E  the oracle floor, which screens a structure before any training
"""
from __future__ import annotations

import json, sys, time
import numpy as np
import torch

from backend import QuotientOperator, multiscale_layers, spatial_blocks
from oracle import oracle_floor, block_offdiagonal_mass
import evaluator as EV

DT = torch.float64
FAILS = []


def check(name, ok, detail=''):
    print('  %-50s %s   %s' % (name, 'ok' if ok else 'FAIL', detail), flush=True)
    if not ok:
        FAILS.append(name)


def build(d, levels, block, radius0, growth, seed, device=None):
    rng = np.random.default_rng(seed)
    pts = rng.random((d, 3))
    layers, meta = multiscale_layers(pts, levels=levels, radius0=radius0, growth=growth,
                                     seed=seed, device=device)
    return QuotientOperator(d, layers, spatial_blocks(pts, block)), pts, meta


def e0b():
    print('\nE0-B  relearn a target this backend can express')
    d = 150
    op, pts, meta = build(d, levels=3, block=6, radius0=0.25, growth=1.6, seed=3)
    print('  d %d  layer pairs %s  blocks %d  coefficients %d'
          % (d, [m['pairs'] for m in meta], len(op.blocks), op.n_layer_coeff + op.n_block_coeff))

    # target built from the SAME backend, so expressibility is not in question
    torch.manual_seed(11)
    c_star = op.identity_coefficients(dtype=DT)
    c_star = c_star + 0.30 * torch.randn(c_star.shape, dtype=DT)
    A = op.dense_A(c_star)
    Rstar = torch.linalg.cholesky(A, upper=True)

    # student starts elsewhere; the target coefficients are never touched
    torch.manual_seed(2026)
    c = (op.identity_coefficients(dtype=DT) + 0.05 * torch.randn(c_star.shape, dtype=DT)).requires_grad_(True)
    mu0 = EV.whitened_spectrum(op.dense_A(c.detach()), Rstar)
    r0 = EV.assess(mu0, d, 0.03, 'start')
    print('   ', EV.report(r0))
    floor = oracle_floor(op, c.detach(), Rstar)
    print('    oracle floor for the student T at start: D/d %.3e' % floor)

    opt = torch.optim.Adam([c], lr=3e-3)
    Ainv = torch.linalg.inv(A)
    logdet_A = float(torch.linalg.slogdet(A)[1])
    best = None
    for step in range(1, 4001):
        opt.zero_grad()
        Ah = op.dense_A(c)
        D = (Ainv * Ah).sum() - op.logdet_A(c) + logdet_A - d
        (D / d).backward()
        opt.step()
        if step % 1000 == 0 or step == 1:
            with torch.no_grad():
                mu = EV.whitened_spectrum(op.dense_A(c), Rstar)
            r = EV.assess(mu, d, 0.03, 'step %d' % step)
            print('   ', EV.report(r))
            best = r
    check('E0-B reaches the target gate', best['state'] == EV.PASS,
          'eps_op %.3e' % best['eps_op'])
    check('E0-B beats the oracle floor is impossible', best['D_per_mode'] >= oracle_floor(op, c.detach(), Rstar) - 1e-12,
          'D/d %.3e vs floor %.3e' % (best['D_per_mode'], oracle_floor(op, c.detach(), Rstar)))
    return best


def e0e():
    print('\nE0-E  the oracle floor discriminates between structures, before training')
    d = 300
    rng = np.random.default_rng(7)
    pts = rng.random((d, 3))
    # a target with real cross-scale structure: a dense SPD matrix with decaying correlations
    X = torch.as_tensor(pts, dtype=DT)
    dist = torch.cdist(X, X)
    A = torch.exp(-dist / 0.25) + 0.05 * torch.eye(d, dtype=DT)
    Rstar = torch.linalg.cholesky(A, upper=True)
    print('  target: exponential-correlation SPD, d %d, cond %.2e'
          % (d, float(torch.linalg.cond(A))))
    rows = []
    for levels, radius0, block in ((0, 0.0, 6), (1, 0.15, 6), (2, 0.15, 6), (3, 0.15, 6), (3, 0.15, 20)):
        layers, meta = multiscale_layers(pts, levels=levels, radius0=radius0, growth=2.0, seed=1) if levels else ([], [])
        op = QuotientOperator(d, layers, spatial_blocks(pts, block))
        c = op.identity_coefficients(dtype=DT)
        f = oracle_floor(op, c, Rstar)
        off = block_offdiagonal_mass(op, c, Rstar)
        rows.append((levels, block, sum(m['pairs'] for m in meta), f, off))
        print('    levels %d  block %2d  pairs %5d  ->  oracle floor D/d %.4e   off-block mass %.4f'
              % rows[-1], flush=True)
    check('floor is non-negative everywhere', all(r[3] >= -1e-12 for r in rows))
    check('larger blocks lower the floor', rows[-1][3] < rows[-2][3],
          '%.3e < %.3e' % (rows[-1][3], rows[-2][3]))
    thr = EV.sufficient_divergence_threshold(d, 0.03)
    print('    (a floor above %.3e already means this structure cannot certify the +-3%% gate'
          ' by divergence alone)' % thr)
    return rows


if __name__ == '__main__':
    t0 = time.time()
    e0b()
    e0e()
    print('\nelapsed %.0f s' % (time.time() - t0))
    print('%s' % ('E0-B/E checks passed' if not FAILS else 'FAILURES: %s' % FAILS))
    sys.exit(1 if FAILS else 0)
