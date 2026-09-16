#!/usr/bin/env python3
"""E1-F (v2): descend the gate, with the two flaws of v1 fixed.

v1 is void.  Two things were wrong with it, and only one was obvious.

 (1) The penalty was relu(mu_max - (1+tau))^2 with mu_max starting near 1e5.  Adam normalises
     gradient MAGNITUDE but not DIRECTION, so that term did not merely dominate the scale, it
     dominated the direction: the divergence term was effectively absent, and the arm sat at
     Div/d = 3.1e4 while the plain-divergence arm reached 1.4e3.
     Fix: penalise with the SAME function the divergence uses, f(mu) = mu - log mu - 1, divided by
     d.  Then the objective is the mean of f over all modes with the two extremes counted w extra
     times -- a reweighting of one objective, not a collision of two.

 (2) mu_min was taken by power iteration on cI - H with c = 1.05 mu_max.  With mu_max ~ 1e5 the
     gap between c - mu_min and c - mu_2 is ~1e-5 of c, so that iteration converges essentially not
     at all; the values it reported (78, 46) were not the smallest eigenvalue.
     Fix: H = M^T M with M = C T R*^-1, so H^-1 = M^-1 M^-T and M^-1 = R* T^-1 C^-1 is available
     exactly -- each lifting layer has disjoint row and column sets, so K^2 = 0 and
     (I + K)^-1 = I - K.  Power iteration on H^-1 converges at the same rate as the top one and
     needs no solve against H.  backend.apply_T_inverse and its transpose are checked against a
     dense inverse to 1e-10, and the whole scheme against eigvalsh, in test_lift.py.

Verdict at the end is a full eigvalsh, not the iteration.
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
from evaluator import whitened_spectrum

STEPS, LR, WEIGHT, ITERS = 800, 1e-2, 300.0, 20


def f_div(mu):
    return mu - torch.log(mu) - 1.0


def build_M(op, coeff, Rinv):
    ks, cs = op.split(coeff)
    G = op.apply_T(coeff, Rinv)
    chols = op._chol_blocks(cs)
    M = torch.empty_like(G)
    logdet_D = 0.0
    tr = 0.0
    for idx, Cb in zip(op.blocks, chols):
        i = idx.to(G.device)
        Mb = Cb @ G[i]
        M[i] = Mb
        tr = tr + Mb.pow(2).sum()
        logdet_D = logdet_D + 2.0 * torch.log(Cb.diagonal()).sum()
    return M, tr, logdet_D, chols


def extremes(op, coeff, chols, R, M, v_hi, v_lo, iters=ITERS):
    """mu_max by power iteration on H, mu_min by power iteration on H^-1 = M^-1 M^-T."""
    with torch.no_grad():
        for _ in range(iters):
            v_hi = M.T @ (M @ v_hi); v_hi = v_hi / v_hi.norm()

        def Minv(v):                                   # M^-1 = R T^-1 C^-1
            w = torch.empty_like(v)
            for idx, Cb in zip(op.blocks, chols):
                i = idx.to(v.device)
                w[i] = torch.linalg.solve_triangular(Cb, v[i], upper=True)
            return R @ op.apply_T_inverse(coeff, w)

        def MinvT(v):                                  # M^-T = C^-T T^-T R^T
            w = op.apply_T_inverse_transpose(coeff, R.T @ v)
            out = torch.empty_like(w)
            for idx, Cb in zip(op.blocks, chols):
                i = idx.to(v.device)
                out[i] = torch.linalg.solve_triangular(Cb.T, w[i], upper=False)
            return out

        for _ in range(iters):
            v_lo = Minv(MinvT(v_lo)); v_lo = v_lo / v_lo.norm()
    mu_hi = (M @ v_hi).pow(2).sum()          # Hellmann-Feynman, eigenvector detached
    mu_lo = (M @ v_lo).pow(2).sum()
    return mu_hi, mu_lo, v_hi.detach(), v_lo.detach()


def main():
    t0 = time.time()
    recs = json.load(open('/root/autodl-tmp/CUTFEM_SUPERELEMENT_LABELS/V1_LABELS.json'))
    rec = [r for r in recs if int(r['seat']) == 328][0]
    rec.setdefault('reference', '/root/autodl-tmp/CUTFEM_SUPERELEMENT_LABELS/REFERENCE_0328')
    label = V.Label(rec, 0.2, 0.03, 10.0)
    g = label.to_gpu(need_A=False, need_Z=False, z_dtype=torch.float32)
    d = label.d
    R = load_upper_factor(Path(rec['reference']) / 'R_UPPER.npy', d, V.DEV)
    order = g['data'].quotient.order.cpu().numpy()
    pts = label.ijk[order[6:] // 3].astype(float) / 64.0
    Rinv = torch.linalg.solve_triangular(R, torch.eye(d, dtype=torch.float64, device=V.DEV), upper=True)
    LDR = float(torch.log(R.diagonal().abs()).sum())
    layers, meta = checkerboard_layers(pts, levels=4, radius0=0.035, growth=2.0,
                                       max_pairs_per_level=98_500, device=V.DEV, seed=0)
    op = QuotientOperator(d, layers, spatial_blocks(pts, 64))
    n_par = op.n_layer_coeff + op.n_block_coeff
    print('seat 328  d %d   %d params (%.2f%% of dense)   %d steps  lr %.0e  weight %.0f  iters %d\n'
          % (d, n_par, 100*n_par/(d*(d+1)/2), STEPS, LR, WEIGHT, ITERS), flush=True)

    c_id = op.identity_coefficients(device=V.DEV)
    Mi, tri, ldi, chi = build_M(op, c_id, Rinv)
    div_id = float(tri - ldi + 2.0 * LDR - d) / d
    mu_id = whitened_spectrum(op.dense_A(c_id), R)
    ref = float((mu_id - np.log(mu_id) - 1).sum()) / d
    print('divergence identity: closed form %.8e  vs full spectrum %.8e  (rel %.2e)'
          % (div_id, ref, abs(div_id - ref)/max(ref, 1e-300)), flush=True)
    assert abs(div_id - ref) <= 1e-8 * max(abs(ref), 1.0), 'DIVERGENCE_IDENTITY_FAILED'
    # and the extremes, against the same full spectrum
    vh = torch.randn(d, 1, dtype=torch.float64, device=V.DEV); vh /= vh.norm()
    vl = torch.randn(d, 1, dtype=torch.float64, device=V.DEV); vl /= vl.norm()
    mh, ml, _, _ = extremes(op, c_id, chi, R, Mi, vh, vl, iters=200)
    print('extremes at T=I,D=I: iteration (%.6e, %.6e)  vs eigvalsh (%.6e, %.6e)'
          % (float(ml), float(mh), mu_id.min(), mu_id.max()), flush=True)
    assert abs(float(mh)/mu_id.max() - 1) < 1e-3 and abs(float(ml)/mu_id.min() - 1) < 1e-2, 'EXTREMES_NOT_CONVERGED'
    del Mi; torch.cuda.empty_cache()

    out = {}
    for arm in ('divergence', 'gate'):
        coeff = op.identity_coefficients(device=V.DEV).clone().requires_grad_(True)
        opt = torch.optim.Adam([coeff], lr=LR)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
        v_hi = torch.randn(d, 1, dtype=torch.float64, device=V.DEV); v_hi /= v_hi.norm()
        v_lo = torch.randn(d, 1, dtype=torch.float64, device=V.DEV); v_lo /= v_lo.norm()
        print('\n--- arm %s ---' % arm, flush=True)
        hist = []
        for step in range(1, STEPS + 1):
            opt.zero_grad()
            M, tr, ldD, chols = build_M(op, coeff, Rinv)
            div = (tr - ldD + 2.0 * LDR - d) / d
            mu_hi, mu_lo, v_hi, v_lo = extremes(op, coeff, chols, R, M, v_hi, v_lo)
            loss = div + (WEIGHT * (f_div(mu_hi) + f_div(mu_lo)) / d if arm == 'gate' else 0.0)
            loss.backward(); opt.step(); sched.step()
            del M
            if step % 100 == 0 or step in (1, 10, 50):
                hist.append((step, float(div), float(mu_lo), float(mu_hi)))
                print('   step %-5d  Div/d %.5e   mu_lo %.4e  mu_hi %.4e   %.0f s'
                      % (step, float(div), float(mu_lo), float(mu_hi), time.time()-t0), flush=True)
            torch.cuda.empty_cache()
        mu = whitened_spectrum(op.dense_A(coeff.detach()), R)
        eps = float(np.abs(mu - 1).max())
        o3 = int((np.abs(mu - 1) > 0.03).sum()); o10 = int((np.abs(mu - 1) > 0.10).sum())
        divf = float((mu - np.log(mu) - 1).sum()) / d
        print('   AUDIT  Div/d %.5e  eps_op %.5e  mu [%.4e, %.4e]  %d out +-3%%  %d out +-10%%  -> %s'
              % (divf, eps, mu.min(), mu.max(), o3, o10,
                 'PASSES +-3%' if eps <= 0.03 else 'PASSES +-10%' if eps <= 0.10 else 'FAILS'), flush=True)
        out[arm] = dict(divergence_per_d=divf, eps_op=eps, mu_min=float(mu.min()), mu_max=float(mu.max()),
                        n_outside_3pct=o3, n_outside_10pct=o10, history=hist)
        torch.save(coeff.detach().cpu(), '/root/autodl-tmp/NEURAL_SCHUR/E1F2_%s.pt' % arm)
        del coeff, opt; torch.cuda.empty_cache()

    a, b = out['divergence'], out['gate']
    print('\ndivergence-directed  Div/d %.5e  eps_op %.5e' % (a['divergence_per_d'], a['eps_op']), flush=True)
    print('gate-directed        Div/d %.5e  eps_op %.5e   -> gate %.2fx %s'
          % (b['divergence_per_d'], b['eps_op'], a['eps_op']/b['eps_op'],
             'better' if b['eps_op'] < a['eps_op'] else 'WORSE'), flush=True)
    Path('/root/autodl-tmp/NEURAL_SCHUR/E1F2.json').write_text(json.dumps(
        dict(seat=328, d=int(d), params=int(n_par), steps=STEPS, lr=LR, weight=WEIGHT,
             iters=ITERS, layers=meta, arms=out, seconds=time.time()-t0), indent=1))
    print('\nwritten E1F2.json (%.0f s)' % (time.time()-t0), flush=True)


if __name__ == '__main__':
    main()
