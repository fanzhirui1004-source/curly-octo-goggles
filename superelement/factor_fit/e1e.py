#!/usr/bin/env python3
"""E1-E: T^T D T at a parameter budget matched to the hierarchical construction.

E1-D ran the multiscale backend at 720k parameters and the hierarchical construction at 12.23e6.
A 17x budget gap, so "hierarchical is 254x better on the gate" compared two different questions.
This gives T^T D T the same budget the hierarchical construction needed to pass +-3%.

Checkerboard bipartition (100% coverage at every level, unlike the median-plane rule), six levels,
capped so the lifting coefficients land near 11.9e6.  Everything else is as in E1-D: T init = I,
D eliminated in closed form, Adam with a cosine schedule, and a full-spectrum audit at the end --
because the floor and the gate come apart by ~100x on this teacher, the floor decides nothing.
"""
import json, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, '/root/cutfem_neural_a_20260910/source_14301bc56')
sys.path.insert(0, '/root/cutfem_neural_a_20260910/superelement_v0')
sys.path.insert(0, '/root/autodl-tmp/NEURAL_SCHUR')
import v1_scaled as V
from stage_cutfem_neural_a.elimination_reference import load_upper_factor
from backend import QuotientOperator, checkerboard_layers, spatial_blocks
from fit import block_groups, floor_objective, optimal_D_coefficients
from evaluator import whitened_spectrum

NEC3, NEC10 = 4.5921e-4, 5.3605e-3
STEPS, LR = 600, 1e-2
TARGET = 11_900_000


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
    dense = d * (d + 1) // 2
    print('seat 328  d %d   dense %.3e' % (d, dense), flush=True)
    print('reference to match: hierarchical factor construction passes +-3%% at 12.23e6 params '
          '(14.95%% of dense), with no fitting\n', flush=True)

    layers, meta = checkerboard_layers(pts, levels=6, radius0=0.035, growth=1.6,
                                       max_pairs_per_level=2_800_000, device=V.DEV, seed=0)
    blocks = spatial_blocks(pts, 64)
    op = QuotientOperator(d, layers, blocks)
    touched = np.zeros(d, bool)
    for L in layers:
        pr = L.pairs.cpu().numpy(); touched[pr[:, 0]] = True; touched[pr[:, 1]] = True
    n_par = op.n_layer_coeff + op.n_block_coeff
    print('layers: ' + ' | '.join('lv%d r=%.3f n=%d c=%d' % (m['level'], m['radius'], m['pairs'], L.chunk)
                                  for m, L in zip(meta, layers)), flush=True)
    print('parameters: %d lifting + %d block = %d  (%.2f%% of dense), coverage %.1f%%\n'
          % (op.n_layer_coeff, op.n_block_coeff, n_par, 100*n_par/dense, 100*touched.mean()), flush=True)

    groups = block_groups(op, V.DEV)
    coeff = op.identity_coefficients(device=V.DEV)
    tail = coeff[op.n_layer_coeff:].detach()
    k = coeff[:op.n_layer_coeff].clone().requires_grad_(True)
    opt = torch.optim.Adam([k], lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
    f0 = float(floor_objective(op, torch.cat([k, tail]), Rinv, groups=groups, logdet_G=ld).detach())
    print('floor at T = I: %.6e\n' % f0, flush=True)
    hist, best = [], float('inf')
    for step in range(1, STEPS + 1):
        opt.zero_grad()
        f = floor_objective(op, torch.cat([k, tail]), Rinv, groups=groups, logdet_G=ld)
        f.backward(); opt.step(); sched.step()
        fv = f.detach().item(); best = min(best, fv)
        if step % 25 == 0 or step in (1, 5, 10):
            hist.append((step, fv))
            print('   step %-5d  floor D/d %.6e   (x%.3f from T=I)   %.0f s'
                  % (step, fv, f0/max(fv, 1e-300), time.time()-t0), flush=True)
    print('\nbest floor %.6e  (x%.3f from T=I)\n' % (best, f0/best), flush=True)

    coeff = optimal_D_coefficients(op, torch.cat([k.detach(), tail]), R)
    Ahat = op.dense_A(coeff)
    mu = whitened_spectrum(Ahat, R)
    del Ahat; torch.cuda.empty_cache()
    eps = float(np.abs(mu - 1).max())
    o3 = int((np.abs(mu - 1) > 0.03).sum()); o10 = int((np.abs(mu - 1) > 0.10).sum())
    print('AUDIT  eps_op %.4e   mu in [%.4e, %.4e]   %d outside +-3%%   %d outside +-10%%   -> %s'
          % (eps, mu.min(), mu.max(), o3, o10,
             'PASSES +-3%' if eps <= 0.03 else 'PASSES +-10%' if eps <= 0.10 else 'FAILS'), flush=True)
    print('\nat a matched budget: T^T D T eps_op %.4e  vs  hierarchical eps_op 2.597e-02 (passing)'
          % eps, flush=True)
    Path('/root/autodl-tmp/NEURAL_SCHUR/E1E.json').write_text(json.dumps(
        dict(seat=328, d=int(d), params=int(n_par), frac_of_dense=n_par/dense,
             lifting=int(op.n_layer_coeff), coverage=float(touched.mean()), layers=meta,
             floor_T_identity=f0, floor_best=best, history=hist, eps_op=eps,
             n_outside_3pct=o3, n_outside_10pct=o10, mu_min=float(mu.min()), mu_max=float(mu.max()),
             steps=STEPS, lr=LR, seconds=time.time()-t0), indent=1))
    torch.save(coeff.cpu(), '/root/autodl-tmp/NEURAL_SCHUR/E1E_coeff.pt')
    print('\nwritten E1E.json (%.0f s)' % (time.time()-t0), flush=True)


if __name__ == '__main__':
    main()
