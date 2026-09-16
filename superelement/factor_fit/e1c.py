#!/usr/bin/env python3
"""E1-C: does descending floor(T) over a real multiscale T beat the fixed-T candidates?

The audit's action 2.  Everything measured so far fixed T and optimised D in closed form, which
can only produce upper bounds on the class (F_class = inf_T F(T) <= F(T)).  The block-diagonal
result says "tuning D alone is not enough"; the banded sweep says "the teacher's own truncated
factor is a bad T".  Neither touches a T that is allowed to move.  So: fix a parameter budget,
let a legal multiscale T move, and see where floor(T) actually goes.

Two anchors on the same budget, so the descent is judged against something:
  - floor at T = I                       (pure block diagonal; the closed-form conditional optimum)
  - floor at T = R* truncated to a band   (the best fixed candidate found in PARAM_ECONOMY)

An exact structural shortcut makes this affordable.  T is a product of unit-triangular lifting
layers, so det T = 1 exactly, so for G = T R*^-1

    logdet G = logdet R*^-1 = -logdet R*        -- a CONSTANT, independent of the coefficients.

The dense d x d slogdet in floor_objective is therefore pure waste per step; dropping it leaves
only sparse applies and 64 x 64 block determinants.  Verified numerically below before use.

This measures what a descent REACHES.  A failure to reach the gate is not a proof that the class
cannot -- see docs/CLAIM_SCOPE_20260916.md.
"""
import json, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, '/root/cutfem_neural_a_20260910/source_14301bc56')
sys.path.insert(0, '/root/cutfem_neural_a_20260910/superelement_v0')
sys.path.insert(0, '/root/autodl-tmp/NEURAL_SCHUR')
import v1_scaled as V
from stage_cutfem_neural_a.elimination_reference import load_upper_factor
from backend import LiftingLayer, QuotientOperator, multiscale_layers, spatial_blocks
from fit import block_groups, floor_objective

NEC3, NEC10 = 4.5921e-4, 5.3605e-3


def floor_fast(op, coeff, Rinv, logdet_const, groups):
    """floor(T) with det(T) = 1 skipping the dense slogdet and blocks gathered by size."""
    return floor_objective(op, coeff, Rinv, groups=groups, logdet_G=logdet_const)


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
    Rinv = torch.linalg.solve_triangular(R, torch.eye(d, dtype=torch.float64, device=V.DEV), upper=True)
    logdet_Rinv = -float(torch.log(R.diagonal().abs()).sum())
    print('seat 328  d %d  dense %.3e entries' % (d, d*(d+1)/2), flush=True)

    layers, meta = multiscale_layers(pts, levels=4, radius0=0.035, growth=2.0,
                                     max_pairs_per_level=220_000, device=V.DEV, seed=0)
    blocks = spatial_blocks(pts, 64)
    op = QuotientOperator(d, layers, blocks)
    n_par = op.n_layer_coeff + op.n_block_coeff
    print('layers: ' + ' | '.join('lv%d r=%.3f pairs=%d' % (m['level'], m['radius'], m['pairs']) for m in meta), flush=True)
    print('parameters: %d lifting + %d block = %d  (%.2f%% of dense)\n'
          % (op.n_layer_coeff, op.n_block_coeff, n_par, 100*n_par/(d*(d+1)/2)), flush=True)

    coeff = op.identity_coefficients(device=V.DEV)
    # verify the det(T) = 1 shortcut once, on a non-trivial T
    probe = coeff.clone(); probe[:op.n_layer_coeff] = 0.05 * torch.randn(op.n_layer_coeff, device=V.DEV, dtype=torch.float64)
    Gp = op.apply_T(probe, Rinv)
    ld_meas = float(torch.linalg.slogdet(Gp)[1])
    print('det(T)=1 shortcut: measured logdet G = %.10f, constant -logdet R* = %.10f, diff %.2e'
          % (ld_meas, logdet_Rinv, abs(ld_meas - logdet_Rinv)), flush=True)
    assert abs(ld_meas - logdet_Rinv) < 1e-6 * max(1.0, abs(logdet_Rinv)), 'DET_NOT_ONE'
    del Gp, probe; torch.cuda.empty_cache()

    groups = block_groups(op, V.DEV)
    ld = torch.tensor(logdet_Rinv, dtype=torch.float64, device=V.DEV)
    print('block groups by size: ' + ', '.join('%d blocks of %d' % (g.shape[0], n) for n, g in groups), flush=True)
    f0 = float(floor_fast(op, coeff, Rinv, ld, groups))
    print('\nanchor  T = I (pure block diagonal, 64):            floor D/d = %.6e' % f0, flush=True)
    print('anchor  T = R* truncated (PARAM_ECONOMY best point): floor D/d = %.6e   at %d params'
          % (2.9519e-2, 4_931_495), flush=True)
    print('gates   +-3%%  %.4e   +-10%%  %.4e\n' % (NEC3, NEC10), flush=True)

    tail = coeff[op.n_layer_coeff:].detach()

    def descend(lr, steps, log_every, k0=None, tag=''):
        k = (coeff[:op.n_layer_coeff].clone() if k0 is None else k0.clone()).requires_grad_(True)
        opt = torch.optim.Adam([k], lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
        h, best = [], float('inf')
        for step in range(1, steps + 1):
            opt.zero_grad()
            f = floor_fast(op, torch.cat([k, tail]), Rinv, ld, groups)
            f.backward(); opt.step(); sched.step()
            fv = f.detach().item()
            best = min(best, fv)
            if step % log_every == 0 or step in (1, 10):
                h.append((step, fv))
                print('  %slr %.0e  step %-5d  floor D/d %.6e   (x%.3f from T=I)   %.0f s'
                      % (tag, lr, step, fv, f0 / max(fv, 1e-300), time.time() - t0), flush=True)
        return k.detach(), best, h

    print('--- learning-rate probe, 120 steps each ---', flush=True)
    probe = {}
    for lr in (3e-3, 1e-2, 3e-2):
        _, b, _ = descend(lr, 120, 40, tag='probe ')
        probe[lr] = b
    lr_best = min(probe, key=probe.get)
    print('\nprobe best: ' + '  '.join('%.0e -> %.4e' % (l, b) for l, b in probe.items())
          + '   picking %.0e\n' % lr_best, flush=True)

    STEPS = 2500
    print('--- long descent, %d steps at lr %.0e ---' % (STEPS, lr_best), flush=True)
    kfin, best, hist = descend(lr_best, STEPS, 100)
    k = kfin
    print('\nbest floor reached: %.6e   = %.2fx above the +-10%% necessary line, %.2fx above +-3%%'
          % (best, best / NEC10, best / NEC3), flush=True)
    Path('/root/autodl-tmp/NEURAL_SCHUR/E1C.json').write_text(json.dumps(
        dict(seat=328, d=int(d), params=int(n_par), frac_of_dense=n_par/(d*(d+1)/2),
             layers=meta, floor_T_identity=f0, floor_best=best, history=hist,
             lr_probe={('%.0e' % l): b for l, b in probe.items()}, lr_used=lr_best,
             nec3=NEC3, nec10=NEC10, steps=STEPS, seconds=time.time()-t0), indent=1))
    torch.save(torch.cat([k, tail]).cpu(), '/root/autodl-tmp/NEURAL_SCHUR/E1C_coeff.pt')
    print('written E1C.json + E1C_coeff.pt (%.0f s)' % (time.time()-t0), flush=True)


if __name__ == '__main__':
    main()
