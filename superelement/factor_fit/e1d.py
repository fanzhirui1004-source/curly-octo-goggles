#!/usr/bin/env python3
"""E1-D: change ONE thing -- the lifting bipartition rule -- and hold everything else fixed.

E1-C descended floor(T) over a 4-level multiscale T and moved it only from 3.355e-01 to about
2.84e-01.  The reason turned out to be in the schedule, not the descent: multiscale_layers splits
by a MEDIAN PLANE and couples across it within radius r, so for small r only a slab around that
plane has any pair.  Measured on this seat: level 0 touched 5.8% of coordinates and 2278 of 12792
(17.8%) appeared in no pair at any level, which pins their rows of T to the identity exactly.

checkerboard_layers colours by the parity of a coarse cell index of side r instead, giving 100%
coverage at every level.  This script runs both rules at a MATCHED coefficient count under an
identical optimiser, init and step budget, so the difference is attributable to the rule.

Both arms report a floor: an upper bound on what the class can do, achieved by a particular T
that a particular descent found.  Neither can exclude anything -- see docs/CLAIM_SCOPE_20260916.md.
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
from fit import block_groups, floor_objective

NEC3, NEC10 = 4.5921e-4, 5.3605e-3
STEPS, LR = 1200, 1e-2


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
    ld = torch.tensor(-float(torch.log(R.diagonal().abs()).sum()), dtype=torch.float64, device=V.DEV)
    blocks = spatial_blocks(pts, 64)
    print('seat 328  d %d   dense %.3e entries' % (d, d*(d+1)/2), flush=True)
    print('identical for both arms: Adam lr %.0e, cosine to 0 over %d steps, T init = I, blocks of 64'
          % (LR, STEPS), flush=True)
    print('anchors: T = I -> 3.355179e-01   |   R* banded @4.93e6 params -> 2.9519e-02   |'
          '   gates +-3%% %.4e  +-10%% %.4e\n' % (NEC3, NEC10), flush=True)

    arms = [('median-plane', multiscale_layers, 220_000),
            ('checkerboard', checkerboard_layers, 98_500)]
    out = {}
    for name, fn, cap in arms:
        layers, meta = fn(pts, levels=4, radius0=0.035, growth=2.0,
                          max_pairs_per_level=cap, device=V.DEV, seed=0)
        op = QuotientOperator(d, layers, blocks)
        touched = np.zeros(d, bool)
        for L in layers:
            pr = L.pairs.cpu().numpy(); touched[pr[:, 0]] = True; touched[pr[:, 1]] = True
        groups = block_groups(op, V.DEV)
        coeff = op.identity_coefficients(device=V.DEV)
        tail = coeff[op.n_layer_coeff:].detach()
        print('--- %s: %d lifting + %d block = %d params (%.2f%% of dense), coverage %.1f%% ---'
              % (name, op.n_layer_coeff, op.n_block_coeff, op.n_layer_coeff + op.n_block_coeff,
                 100*(op.n_layer_coeff + op.n_block_coeff)/(d*(d+1)/2), 100*touched.mean()), flush=True)
        k = coeff[:op.n_layer_coeff].clone().requires_grad_(True)
        opt = torch.optim.Adam([k], lr=LR)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
        hist, best = [], float('inf')
        for step in range(1, STEPS + 1):
            opt.zero_grad()
            f = floor_objective(op, torch.cat([k, tail]), Rinv, groups=groups, logdet_G=ld)
            f.backward(); opt.step(); sched.step()
            fv = f.detach().item()
            best = min(best, fv)
            if step % 100 == 0 or step in (1, 10, 50):
                hist.append((step, fv))
                print('   step %-5d  floor D/d %.6e   (x%.3f from T=I)   %.0f s'
                      % (step, fv, 3.355179e-01 / max(fv, 1e-300), time.time()-t0), flush=True)
        print('   best %.6e   = x%.3f from T=I,  %.1fx above +-10%%,  %.1fx above +-3%%\n'
              % (best, 3.355179e-01/best, best/NEC10, best/NEC3), flush=True)
        out[name] = dict(params=int(op.n_layer_coeff + op.n_block_coeff),
                         lifting=int(op.n_layer_coeff), coverage=float(touched.mean()),
                         layers=meta, best=best, history=hist)
        torch.save(torch.cat([k.detach(), tail]).cpu(),
                   '/root/autodl-tmp/NEURAL_SCHUR/E1D_%s.pt' % name)
        del op, k, opt, groups; torch.cuda.empty_cache()

    a, b = out['median-plane']['best'], out['checkerboard']['best']
    print('median-plane %.6e   vs   checkerboard %.6e   -> %.2fx' % (a, b, a / b), flush=True)
    Path('/root/autodl-tmp/NEURAL_SCHUR/E1D.json').write_text(json.dumps(
        dict(seat=328, d=int(d), steps=STEPS, lr=LR, floor_T_identity=3.355179e-01,
             nec3=NEC3, nec10=NEC10, arms=out, seconds=time.time()-t0), indent=1))
    print('\nwritten E1D.json (%.0f s)' % (time.time()-t0), flush=True)


if __name__ == '__main__':
    main()
