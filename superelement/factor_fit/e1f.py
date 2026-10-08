#!/usr/bin/env python3
"""E1-F: descend the GATE, not the floor.

Measured on this teacher: at one hierarchical tolerance, 3 modes out of 12792 lie outside +-10%
and eps_op = 0.229, while D/d = 1.27e-05 clears the +-3% necessary line by 36x.  The divergence is
an average and cannot see the modes that decide acceptance.  Everything fitted here so far has
minimised that average.

So: same budget, same schedule, same init, two objectives.

    arm  floor      L = Div/d
    arm  gate       L = Div/d + w * [ relu(mu_max - (1+tau))^2 + relu((1-tau) - mu_min)^2 ]

Both carry D as a trainable block-Cholesky (rather than eliminating it in closed form) so the two
arms differ only in the loss.

Everything needed is cheap and exact.  With C = G G^T, G = T R*^-1, D_b = C_b^T C_b and det T = 1:

    tr(A^-1 Ahat) = sum_b ||C_b G_b||_F^2                         (no C_bb ever formed)
    logdet(A^-1 Ahat) = sum_b logdet D_b - logdet A
    Div = sum_b ||C_b G_b||_F^2 - 2 sum_b sum log diag(C_b) + 2 logdet R* - d

and H = G^T D G = M^T M with M_b = C_b G_b, so the whitened eigenvalues are the squared singular
values of M.  mu_max comes from power iteration on H (matvec = M^T (M v)); mu_min from power
iteration on cI - H with c > mu_max, so it needs matvecs too and never a solve.  Both vectors are
warm-started across steps, since the operator moves slowly.  Gradients use the Hellmann-Feynman
form: with v detached, mu = ||M v||^2 differentiates correctly through M.

The verdict at the end is a full eigvalsh, not the power iteration.
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

TAU, STEPS, LR, WEIGHT = 0.03, 1200, 1e-2, 3.0
POWER_ITERS = 12


def build_M(op, coeff, Rinv):
    """M with M_b = C_b G_b, so H = M^T M is the whitened operator exactly."""
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
    return M, tr, logdet_D


def extremes(M, v_hi, v_lo, iters=POWER_ITERS):
    """mu_max and mu_min of H = M^T M by matvec only; vectors are updated in place, detached."""
    with torch.no_grad():
        for _ in range(iters):
            v_hi = M.T @ (M @ v_hi); v_hi = v_hi / v_hi.norm()
        hi = float((M @ v_hi).pow(2).sum())
        c = 1.05 * hi + 1e-12
        for _ in range(iters):
            v_lo = c * v_lo - M.T @ (M @ v_lo); v_lo = v_lo / v_lo.norm()
    mu_hi = (M @ v_hi).pow(2).sum()          # differentiable, v detached: Hellmann-Feynman
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
    order = g['data'].quotient.order.cpu().numpy(); posn = order[6:]
    pts = label.ijk[posn // 3].astype(float) / 64.0
    Rinv = torch.linalg.solve_triangular(R, torch.eye(d, dtype=torch.float64, device=V.DEV), upper=True)
    LDR = float(torch.log(R.diagonal().abs()).sum())
    layers, meta = checkerboard_layers(pts, levels=4, radius0=0.035, growth=2.0,
                                       max_pairs_per_level=98_500, device=V.DEV, seed=0)
    op = QuotientOperator(d, layers, spatial_blocks(pts, 64))
    n_par = op.n_layer_coeff + op.n_block_coeff
    print('seat 328  d %d   %d params (%.2f%% of dense)   tau %.2f   %d steps   lr %.0e   weight %.1f\n'
          % (d, n_par, 100*n_par/(d*(d+1)/2), TAU, STEPS, LR, WEIGHT), flush=True)

    # the divergence identity is asserted before either arm runs
    c_id = op.identity_coefficients(device=V.DEV)
    Mi, tri, ldi = build_M(op, c_id, Rinv)
    div_id = float(tri - ldi + 2.0 * LDR - d) / d
    mu_id = whitened_spectrum(op.dense_A(c_id), R)
    ref = float((mu_id - np.log(mu_id) - 1).sum()) / d
    print('divergence identity check at T=I, D=I:  closed form %.8e  vs  full spectrum %.8e  (rel %.2e)'
          % (div_id, ref, abs(div_id - ref) / max(ref, 1e-300)), flush=True)
    assert abs(div_id - ref) <= 1e-8 * max(abs(ref), 1.0), 'DIVERGENCE_IDENTITY_FAILED'
    del Mi; torch.cuda.empty_cache()

    out = {}
    for arm in ('floor', 'gate'):
        coeff = op.identity_coefficients(device=V.DEV).clone().requires_grad_(True)
        opt = torch.optim.Adam([coeff], lr=LR)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
        v_hi = torch.randn(d, dtype=torch.float64, device=V.DEV); v_hi /= v_hi.norm()
        v_lo = torch.randn(d, dtype=torch.float64, device=V.DEV); v_lo /= v_lo.norm()
        print('--- arm %s ---' % arm, flush=True)
        hist = []
        for step in range(1, STEPS + 1):
            opt.zero_grad()
            M, tr, ldD = build_M(op, coeff, Rinv)
            div = (tr - ldD + 2.0 * LDR - d) / d
            if arm == 'gate':
                mu_hi, mu_lo, v_hi, v_lo = extremes(M, v_hi, v_lo)
                pen = torch.relu(mu_hi - (1 + TAU)).pow(2) + torch.relu((1 - TAU) - mu_lo).pow(2)
                loss = div + WEIGHT * pen
            else:
                loss = div
                mu_hi = mu_lo = torch.tensor(float('nan'))
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
        div_f = float((mu - np.log(mu) - 1).sum()) / d
        print('   AUDIT  Div/d %.5e   eps_op %.5e   mu in [%.4e, %.4e]   %d out +-3%%   %d out +-10%%  -> %s\n'
              % (div_f, eps, mu.min(), mu.max(), o3, o10,
                 'PASSES +-3%' if eps <= 0.03 else 'PASSES +-10%' if eps <= 0.10 else 'FAILS'), flush=True)
        out[arm] = dict(divergence_per_d=div_f, eps_op=eps, mu_min=float(mu.min()),
                        mu_max=float(mu.max()), n_outside_3pct=o3, n_outside_10pct=o10, history=hist)
        torch.save(coeff.detach().cpu(), '/root/autodl-tmp/NEURAL_SCHUR/E1F_%s.pt' % arm)
        del coeff, opt; torch.cuda.empty_cache()

    print('floor-directed eps_op %.5e   vs   gate-directed eps_op %.5e   -> %.2fx'
          % (out['floor']['eps_op'], out['gate']['eps_op'], out['floor']['eps_op']/out['gate']['eps_op']), flush=True)
    print('floor-directed Div/d  %.5e   vs   gate-directed Div/d  %.5e'
          % (out['floor']['divergence_per_d'], out['gate']['divergence_per_d']), flush=True)
    Path('/root/autodl-tmp/NEURAL_SCHUR/E1F.json').write_text(json.dumps(
        dict(seat=328, d=int(d), params=int(n_par), tau=TAU, steps=STEPS, lr=LR, weight=WEIGHT,
             power_iters=POWER_ITERS, layers=meta, arms=out, seconds=time.time()-t0), indent=1))
    print('\nwritten E1F.json (%.0f s)' % (time.time()-t0), flush=True)


if __name__ == '__main__':
    main()
