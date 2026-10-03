#!/usr/bin/env python3
"""E3 - is the single-signed over-stiffness concentrated near the rigid-body space?

Jiang/Zhan/Zhang/Wang (arXiv 2608.02036) state a regularization-nullspace principle:
if the regulariser's nullspace does not contain the ENTIRE physics nullspace there is
an irreducible floor at every regularisation strength, and its signature is a
SINGLE-SIGNED bias. Codex measures exactly that: negative c18_signed_error on all 8
seats, c6_mu_max < 1 on all 8, 8-70x too stiff.

Test. For the whitened spectrum H = R*^-T Ahat R*^-1 with H v = mu v, the physical
direction is x = R*^-1 v, and
        x^T Ahat x / x^T A x = mu          (the ratio the gate reads)
        x^T A x / x^T x = 1/|R*^-1 v|^2    (the TRUE stiffness of that direction)
So plotting mu against the true stiffness says whether the over-stiffening lives on
the soft (near-rigid) end. If it does, the floor has a structural cause and a cheap
fix; if mu is uncorrelated with stiffness, the hypothesis is dead and we stop
pursuing it.
"""
import json, gc, time
from pathlib import Path
import numpy as np, torch
OUT = Path('/root/autodl-tmp/CLAUDE_NULLSPACE_20260917'); OUT.mkdir(exist_ok=True)
RUN = Path('/root/autodl-tmp/CUTFEM_M4_MULTI_20260917/RUN_TRAIN32_ACCEL_V1')
REF = Path('/root/autodl-tmp/CUTFEM_SUPERELEMENT_LABELS'); DEV = 'cuda:0'


def unpack(path, d):
    p = np.load(path, mmap_mode='r', allow_pickle=False)
    R = torch.zeros((d, d), dtype=torch.float64, device=DEV); off = 0
    for row in range(d):
        n = d - row
        R[row, row:] = torch.from_numpy(np.ascontiguousarray(p[off:off + n])).to(DEV); off += n
    return R


def main():
    def dim(p):
        n = np.load(p, mmap_mode='r', allow_pickle=False).shape[0]
        return int((-1 + (1 + 8 * n) ** .5) / 2 + .5)
    seat = 253; d = dim(REF / f'REFERENCE_{seat:04d}' / 'R_UPPER.npy')
    Rs = unpack(REF / f'REFERENCE_{seat:04d}' / 'R_UPPER.npy', d)
    out = dict(seat=seat, d=d, arms={})
    for arm in ('baseline', 'relative'):
        t = time.time()
        Rh = unpack(RUN / arm / f'EVAL_040000_{seat:04d}' / 'R_PRED_UPPER.npy', d)
        Mt = torch.linalg.solve_triangular(Rs.T, Rh.T, upper=False); M = Mt.T.contiguous(); del Mt
        W = M.T @ M; del M, Rh; gc.collect(); torch.cuda.empty_cache()
        mu, V = torch.linalg.eigh((W + W.T) * .5); del W; gc.collect(); torch.cuda.empty_cache()
        # true stiffness of each whitened direction: 1/|R*^-1 v|^2
        X = torch.linalg.solve_triangular(Rs, V, upper=True)
        stiff = 1.0 / (X * X).sum(0); del X, V; gc.collect(); torch.cuda.empty_cache()
        lm, ls = torch.log(mu.clamp_min(1e-300)), torch.log(stiff.clamp_min(1e-300))
        # rank correlation, and the profile by stiffness decile
        rk = lambda t: torch.argsort(torch.argsort(t)).double()
        a, b = rk(lm), rk(ls)
        spearman = float(((a - a.mean()) * (b - b.mean())).sum() /
                         (a.std(unbiased=False) * b.std(unbiased=False) * len(a)))
        order = torch.argsort(stiff)           # softest first
        deciles = []
        for k in range(10):
            sl = order[k * d // 10:(k + 1) * d // 10]
            deciles.append(dict(decile=k,
                                stiffness_median=float(stiff[sl].median()),
                                mu_median=float(mu[sl].median()),
                                mu_max=float(mu[sl].max()),
                                frac_above_1p1=float((mu[sl] > 1.1).double().mean()),
                                frac_below_0p9=float((mu[sl] < 0.9).double().mean())))
        top = torch.topk(mu, 20)
        worst = [dict(mu=float(mu[i]), stiffness=float(stiff[i]),
                      stiffness_percentile=float((stiff < stiff[i]).double().mean()))
                 for i in top.indices.tolist()]
        out['arms'][arm] = dict(spearman_log_mu_vs_log_stiffness=spearman,
                                deciles=deciles, worst20=worst,
                                mu_min=float(mu.min()), mu_max=float(mu.max()),
                                stiffness_min=float(stiff.min()), stiffness_max=float(stiff.max()),
                                seconds=time.time() - t)
        print(f'\n=== {arm} ===  Spearman(log mu, log true stiffness) = {spearman:+.4f}', flush=True)
        print(f"{'decile(soft->stiff)':>21}{'true stiffness':>17}{'median mu':>12}{'max mu':>12}{'%>1.1':>9}{'%<0.9':>9}")
        for r in deciles:
            print(f"{r['decile']:>21}{r['stiffness_median']:>17.4g}{r['mu_median']:>12.4g}"
                  f"{r['mu_max']:>12.4g}{r['frac_above_1p1']:>9.1%}{r['frac_below_0p9']:>9.1%}", flush=True)
        print('  20 stiffest-predicted modes sit at true-stiffness percentiles: '
              + ', '.join(f"{w['stiffness_percentile']:.0%}" for w in worst), flush=True)
        del mu, stiff; gc.collect(); torch.cuda.empty_cache()
        (OUT / 'RESULT.json').write_text(json.dumps(out, indent=1))
    print('\ndone', flush=True)


if __name__ == '__main__':
    main()
