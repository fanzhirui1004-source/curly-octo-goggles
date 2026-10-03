#!/usr/bin/env python3
"""Three diagnostics after eps_op ranked the two stage-1 arms backwards.

The assembled two-cell test gave 25.4% worst sensitivity error for the chol arm at
eps_op 215.72 and 542% for the sqrt arm at eps_op 48.03.  eps_op, D_per_mode and
e_A all rank those two the wrong way round.  The reason is structural: eps_op is
max|mu-1|, which is unbounded above and capped at 1 below, while compliance is
f^T K^-1 f and lives on the soft end.

1. Re-score every run with the inversion-symmetric quantity
       g = max_i max(mu_i, 1/mu_i)
   equivalently the Thompson metric max(log mu_max, -log mu_min), which is what a
   two-sided Loewner sandwich (1+eps)^-1 A <= A_hat <= (1+eps) A actually asserts.

2. The indefiniteness of the predicted symmetric root cannot be the cause: with
   M_hat = V L V^T, A_hat = M_hat^2 = V L^2 V^T, so flipping the sign of any
   eigenvalue leaves A_hat untouched.  Verified numerically here rather than
   asserted, and the negative count is reported as a measure of how far from the
   positive definite target branch the network landed.

3. Oracle: which entries of the true factor control the soft end?  Perturb one
   magnitude stratum at a time by a fixed RELATIVE amount and watch mu_min and
   mu_max separately.  A loss can only be reweighted toward the soft end once this
   says where the soft end is controlled from.
"""
import argparse, gc, json, time
from pathlib import Path
import numpy as np
import torch

DEV = 'cuda:0'
F64 = torch.float64


def unpack(path, d):
    p = np.load(path, mmap_mode='r', allow_pickle=False)
    if p.shape != (d * (d + 1) // 2,) or p.dtype != np.float64:
        raise ValueError(f'FP64_UPPER_REQUIRED {path} {p.shape} {p.dtype}')
    host = np.zeros((d, d), dtype=np.float64)
    off = 0
    for row in range(d):
        host[row, row:] = p[off:off + d - row]
        off += d - row
    return torch.from_numpy(host).to(DEV)


def score(mu):
    mu = np.asarray(mu, dtype=np.float64)
    mn, mx = float(mu.min()), float(mu.max())
    return dict(mu_min=mn, mu_max=mx,
                eps_op=max(abs(mn - 1), abs(mx - 1)),
                over_stiff_factor=mx,
                under_stiff_factor=1.0 / mn if mn > 0 else float('inf'),
                g=max(mx, 1.0 / mn) if mn > 0 else float('inf'),
                thompson=max(np.log(mx), -np.log(mn)) if mn > 0 else float('inf'),
                driven_by='soft' if (mn > 0 and 1.0 / mn > mx) else 'stiff')


def spectrum_of(Ahat, Rstar):
    X = torch.linalg.solve_triangular(Rstar, Ahat, upper=True, left=False)
    W = torch.linalg.solve_triangular(Rstar, X.T.contiguous(), upper=True, left=False)
    del X
    W = 0.5 * (W + W.T)
    mu = torch.linalg.eigvalsh(W)
    del W
    gc.collect(); torch.cuda.empty_cache()
    return mu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--d', type=int, default=12822)
    ap.add_argument('--reference', type=Path,
                    default=Path('/root/autodl-tmp/CUTFEM_SUPERELEMENT_LABELS/REFERENCE_0253'))
    ap.add_argument('--chol-eigs', type=Path, required=True)
    ap.add_argument('--sqrt-eigs', type=Path, required=True)
    ap.add_argument('--sqrt-factor', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--delta', type=float, default=0.01)
    ap.add_argument('--strata', type=int, default=10)
    ap.add_argument('--seed', type=int, default=20260918)
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    d = a.d
    t0 = time.time()
    report = dict(d=d, delta=a.delta, strata=a.strata, seed=a.seed,
                  scope='seat 0253 stage-1 arms plus a magnitude-stratified oracle on the true factor')

    # ---- 1. re-score -----------------------------------------------------
    rescore = {}
    for head, path in (('chol', a.chol_eigs), ('sqrt', a.sqrt_eigs)):
        rescore[head] = score(np.load(path, allow_pickle=False))
    measured = dict(chol=dict(worst_compliance_rel=0.2028, worst_sensitivity_rel=0.2543),
                    sqrt=dict(worst_compliance_rel=4.517, worst_sensitivity_rel=5.424))
    for head in rescore:
        rescore[head].update(measured[head])
    report['rescore'] = rescore
    for head, r in rescore.items():
        print('%-5s eps_op %10.4f   g %10.2f   1/mu_min %10.2f   mu_max %9.3f   driven_by %-5s'
              '   sensitivity %7.3f' % (head, r['eps_op'], r['g'], r['under_stiff_factor'],
                                        r['mu_max'], r['driven_by'], r['worst_sensitivity_rel']), flush=True)
    order_eps = sorted(rescore, key=lambda h: rescore[h]['eps_op'])
    order_g = sorted(rescore, key=lambda h: rescore[h]['g'])
    order_true = sorted(rescore, key=lambda h: rescore[h]['worst_sensitivity_rel'])
    report['ranking'] = dict(by_eps_op=order_eps, by_g=order_g, by_measured_sensitivity=order_true,
                             eps_op_agrees=order_eps == order_true, g_agrees=order_g == order_true)
    print('\nranking by eps_op %s   by g %s   by measured sensitivity %s'
          % (order_eps, order_g, order_true), flush=True)

    Rstar = unpack(a.reference / 'R_UPPER.npy', d)

    # ---- 2. the indefinite root cannot matter ----------------------------
    U = unpack(a.sqrt_factor, d)
    M = U + torch.triu(U, 1).T
    del U
    gc.collect(); torch.cuda.empty_cache()
    lam, V = torch.linalg.eigh(M)
    negatives = int((lam < 0).sum())
    root = dict(negative_eigenvalues=negatives, total=int(len(lam)),
                most_negative=float(lam.min()), largest=float(lam.max()),
                negative_mass_fraction=float(lam[lam < 0].square().sum() / lam.square().sum()))
    Aflip = (V * lam.abs()) @ V.T
    Asame = M @ M
    root['A_hat_unchanged_by_sign_flip'] = float((Aflip @ Aflip - Asame).norm() / Asame.norm())
    del Aflip, V, lam
    gc.collect(); torch.cuda.empty_cache()
    root['conclusion'] = ('A_hat = M^2 = V L^2 V^T is invariant under the sign of any eigenvalue, '
                          'so indefiniteness of the predicted root cannot explain the soft end; '
                          'it measures how far from the positive definite target branch the fit landed')
    report['predicted_root'] = root
    print('\npredicted root: %d of %d eigenvalues negative, most negative %.4g, '
          'negative energy fraction %.4f' % (negatives, root['total'], root['most_negative'],
                                             root['negative_mass_fraction']), flush=True)
    print('   A_hat unchanged by sign flip: relative difference %.3e'
          % root['A_hat_unchanged_by_sign_flip'], flush=True)
    del M, Asame
    gc.collect(); torch.cuda.empty_cache()

    # ---- 3. which entries control the soft end? --------------------------
    strict = torch.triu(torch.ones((d, d), dtype=torch.bool, device=DEV), 1)
    absR = Rstar.abs()
    vals = absR[strict]
    sample = vals[torch.randperm(vals.numel(), device=DEV)[:4_000_000]]
    edges = torch.quantile(sample.float(), torch.linspace(0, 1, a.strata + 1, device=DEV)).double()
    edges[0] = -1.0
    edges[-1] = float(vals.max()) * 1.000001
    del sample, vals
    gc.collect(); torch.cuda.empty_cache()
    report['stratum_edges'] = [float(v) for v in edges.cpu()]

    gen = torch.Generator(device=DEV).manual_seed(a.seed)
    signs = (torch.randint(0, 2, (d, d), generator=gen, device=DEV, dtype=torch.int8) * 2 - 1).double()

    rows = []
    cases = [('diagonal_only', None)] + [(f'stratum_{k}', k) for k in range(a.strata)]
    for name, k in cases:
        if k is None:
            mask = torch.eye(d, dtype=torch.bool, device=DEV)
        else:
            mask = strict & (absR > edges[k]) & (absR <= edges[k + 1])
        count = int(mask.sum())
        Rp = torch.where(mask, Rstar * (1.0 + a.delta * signs), Rstar)
        if k is None and bool((Rp.diagonal() <= 0).any()):
            raise ValueError('PERTURBED_PIVOT_NONPOSITIVE')
        del mask
        gc.collect(); torch.cuda.empty_cache()
        mu = spectrum_of(Rp.T @ Rp, Rstar)
        del Rp
        gc.collect(); torch.cuda.empty_cache()
        s = score(mu.cpu().numpy())
        del mu
        gc.collect(); torch.cuda.empty_cache()
        s.update(name=name, entries=count,
                 entry_fraction=count / (d * (d + 1) // 2),
                 magnitude_low=float(edges[k]) if k is not None else None,
                 magnitude_high=float(edges[k + 1]) if k is not None else None)
        rows.append(s)
        print('   %-14s entries %11d   mu_min %.6e   mu_max %10.4f   eps_op %9.4f   g %9.2f   %s'
              % (name, count, s['mu_min'], s['mu_max'], s['eps_op'], s['g'], s['driven_by']), flush=True)
        report['oracle'] = rows
        report['seconds'] = time.time() - t0
        (a.output / 'SOFTEND.json').write_text(json.dumps(report, indent=1))

    report['status'] = 'SOFTEND_DIAGNOSTICS_COMPLETE'
    report['seconds'] = time.time() - t0
    (a.output / 'SOFTEND.json').write_text(json.dumps(report, indent=1))
    print('\ndone (%.0f s)' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
