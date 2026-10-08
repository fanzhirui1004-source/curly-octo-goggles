#!/usr/bin/env python3
"""How much of the damage is the diagonal, and how bad is the network's diagonal?

The magnitude-stratified oracle says a 1% relative error on the 12822 diagonal
entries alone costs g = 2.43, while the same relative error on 65.8 million small
off-diagonal entries costs nothing measurable.  So the next question is causal, not
correlational: swap the diagonals between the prediction and the truth and see which
way g moves.

    H1 = predicted factor with the TRUE diagonal      -> how much of g is not the diagonal
    H2 = true factor with the PREDICTED diagonal      -> how much of g is the diagonal

and, separately, the network's relative pivot error as a function of pivot size,
because the loss normalises the diagonal bucket by one global RMS, which suppresses
the smallest pivots both in the residual and again through the exp parameterisation.
"""
import argparse, gc, json, time
from pathlib import Path
import numpy as np
import torch

DEV = 'cuda:0'


def unpack(path, d):
    p = np.load(path, mmap_mode='r', allow_pickle=False)
    if p.shape != (d * (d + 1) // 2,) or p.dtype != np.float64:
        raise ValueError(f'FP64_UPPER_REQUIRED {path}')
    host = np.zeros((d, d), dtype=np.float64)
    off = 0
    for row in range(d):
        host[row, row:] = p[off:off + d - row]
        off += d - row
    return torch.from_numpy(host).to(DEV)


def score(mu):
    mn, mx = float(mu.min()), float(mu.max())
    return dict(mu_min=mn, mu_max=mx, eps_op=max(abs(mn - 1), abs(mx - 1)),
                under_stiff_factor=1.0 / mn if mn > 0 else float('inf'),
                g=max(mx, 1.0 / mn) if mn > 0 else float('inf'))


def spectrum(Ahat, Rstar):
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
    ap.add_argument('--chol-factor', type=Path, required=True)
    ap.add_argument('--sqrt-factor', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    d = a.d
    t0 = time.time()
    Rstar = unpack(a.reference / 'R_UPPER.npy', d)
    pivot = Rstar.diagonal().clone()
    report = dict(d=d, reference=str(a.reference),
                  pivot_min=float(pivot.min()), pivot_max=float(pivot.max()),
                  pivot_log_range=[float(pivot.log().min()), float(pivot.log().max())],
                  scope='seat 0253; diagonal swap between prediction and truth')
    arms = {}

    for head, path in (('chol', a.chol_factor), ('sqrt', a.sqrt_factor)):
        U = unpack(path, d)
        if head == 'chol':
            P = U
        else:
            P = U + torch.triu(U, 1).T
            del U
        gc.collect(); torch.cuda.empty_cache()
        pred = P.diagonal().clone()
        rel = (pred / pivot - 1.0).abs()
        order = torch.argsort(pivot)
        deciles = []
        step = d // 10
        for k in range(10):
            ids = order[k * step:(k + 1) * step if k < 9 else d]
            deciles.append(dict(decile=k,
                                pivot_low=float(pivot[ids].min()), pivot_high=float(pivot[ids].max()),
                                median_relative_error=float(rel[ids].median()),
                                worst_relative_error=float(rel[ids].max())))
        row = dict(head=head, factor=str(path),
                   pivot_median_relative_error=float(rel.median()),
                   pivot_worst_relative_error=float(rel.max()),
                   pivot_relative_error_by_pivot_decile=deciles,
                   predicted_pivot_min=float(pred.min()), predicted_pivot_max=float(pred.max()))
        print(f'\n{head}: pivot relative error median {row["pivot_median_relative_error"]:.4g} '
              f'worst {row["pivot_worst_relative_error"]:.4g}', flush=True)
        print('   by true-pivot decile (smallest first): ' +
              ' '.join(f'{v["median_relative_error"]:.3g}' for v in deciles), flush=True)

        def build(mat):
            return mat.T @ mat if head == 'chol' else mat @ mat

        row['as_predicted'] = score(spectrum(build(P), Rstar))
        H1 = P.clone()
        H1.diagonal().copy_(pivot)
        row['true_diagonal_predicted_offdiagonal'] = score(spectrum(build(H1), Rstar))
        del H1
        gc.collect(); torch.cuda.empty_cache()
        H2 = Rstar.clone() if head == 'chol' else (Rstar + torch.triu(Rstar, 1).T)
        H2.diagonal().copy_(pred)
        row['predicted_diagonal_true_offdiagonal'] = score(spectrum(build(H2), Rstar))
        del H2, P
        gc.collect(); torch.cuda.empty_cache()
        for name in ('as_predicted', 'true_diagonal_predicted_offdiagonal',
                     'predicted_diagonal_true_offdiagonal'):
            s = row[name]
            print('   %-42s g %12.2f   1/mu_min %12.2f   mu_max %10.3f'
                  % (name, s['g'], s['under_stiff_factor'], s['mu_max']), flush=True)
        arms[head] = row
        report['arms'] = arms
        report['seconds'] = time.time() - t0
        (a.output / 'PIVOTS.json').write_text(json.dumps(report, indent=1))

    report['status'] = 'PIVOT_SPLIT_COMPLETE'
    report['seconds'] = time.time() - t0
    (a.output / 'PIVOTS.json').write_text(json.dumps(report, indent=1))
    print('\ndone (%.0f s)' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
