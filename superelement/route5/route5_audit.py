#!/usr/bin/env python3
"""Audit of the route-5 arm: is it converged, and where does its 25% bias live?

Three questions, all cheap:
  1. training curve -- is the diagonal bucket still falling at 20000 steps?
  2. pivot calibration -- diag(L_hat)/diag(L_*) by decile of the true pivot;
     a global scale bias shows up as a constant ratio != 1.
  3. diagonal swap in the same style as pivots.py, on the inverse target, using the
     cheap spectrum route nu = eig((L R_*^T)^T (L R_*^T)), mu = 1/nu.
"""
import json, gc, sys
from pathlib import Path
import numpy as np, torch

DEV = 'cuda:0'
R5 = Path('/root/autodl-tmp/CLAUDE_ROUTE5_20260918')
REF = Path('/root/autodl-tmp/CUTFEM_SUPERELEMENT_LABELS/REFERENCE_0253')
d = 12822


def unpack(path):
    p = np.load(path, mmap_mode='r', allow_pickle=False)
    host = np.zeros((d, d)); off = 0
    for row in range(d):
        host[row, row:] = p[off:off + d - row]; off += d - row
    return torch.from_numpy(host).to(DEV)


def score_inverse(L, Rstar):
    Q = L @ Rstar.T
    W = Q.T @ Q; del Q
    W = 0.5 * (W + W.T)
    nu = torch.linalg.eigvalsh(W); del W
    gc.collect(); torch.cuda.empty_cache()
    mu_min, mu_max = 1 / float(nu[-1]), 1 / float(nu[0])
    return dict(mu_min=mu_min, mu_max=mu_max, g=max(mu_max, 1 / mu_min),
                eps_op=max(abs(mu_min - 1), abs(mu_max - 1)))


out = {}
# ---- 1. curve
rows = [json.loads(l) for l in (R5 / 'S1_INVERSE' / 'TRAIN.jsonl').read_text().splitlines()]
def avg(seg, k): return float(np.mean([r['buckets'][k] for r in seg]))
segs = {'steps 1-200': rows[:200], 'steps 5000-6000': rows[4999:6000],
        'steps 10000-11000': rows[9999:11000], 'steps 15000-16000': rows[14999:16000],
        'steps 19000-20000': rows[18999:20000]}
curve = {n: dict(diagonal=avg(s, 0), same_patch=avg(s, 1), cross_patch=avg(s, 2),
                 total=float(np.mean([r['loss'] for r in s]))) for n, s in segs.items()}
out['curve'] = curve
print('bucket losses (mean over window):')
print('%-20s %10s %10s %10s %10s' % ('', 'diag', 'same', 'cross', 'total'))
for n, v in curve.items():
    print('%-20s %10.5f %10.5f %10.5f %10.5f' % (n, v['diagonal'], v['same_patch'], v['cross_patch'], v['total']))
cond = json.load(open(R5 / 'S1_INVERSE' / 'CONDITIONING.json'))
out['conditioning'] = cond['values']
print('conditioning:', {k: round(v, 5) for k, v in cond['values'].items()})

# ---- 2. pivot calibration
Lhat = unpack(R5 / 'EVAL' / 'L_PRED_UPPER_0253.npy')
Lstar = unpack(REF / 'G_UPPER.npy')
Rstar = unpack(REF / 'R_UPPER.npy')
ph, ps = Lhat.diagonal(), Lstar.diagonal()
ratio = ph / ps
order = torch.argsort(ps)
dec = []
for k in range(10):
    ids = order[k * (d // 10):(k + 1) * (d // 10) if k < 9 else d]
    dec.append(dict(k=k, true_pivot_lo=float(ps[ids].min()), true_pivot_hi=float(ps[ids].max()),
                    ratio_median=float(ratio[ids].median()),
                    ratio_iqr=[float(ratio[ids].quantile(.25)), float(ratio[ids].quantile(.75))]))
out['pivot_ratio_overall'] = dict(median=float(ratio.median()), mean_log=float(ratio.log().mean()),
                                  q10=float(ratio.quantile(.1)), q90=float(ratio.quantile(.9)))
out['pivot_ratio_by_true_pivot_decile'] = dec
print('\npredicted/true pivot ratio: median %.4f  exp(mean log) %.4f  q10 %.4f  q90 %.4f'
      % (out['pivot_ratio_overall']['median'], np.exp(out['pivot_ratio_overall']['mean_log']),
         out['pivot_ratio_overall']['q10'], out['pivot_ratio_overall']['q90']))
print('by true-pivot decile (small pivots first): ' + ' '.join('%.3f' % v['ratio_median'] for v in dec))
# off-diagonal scale: ratio of Frobenius norms of strict upper parts, by magnitude decile of L*
su = torch.triu(torch.ones(d, d, dtype=torch.bool, device=DEV), 1)
oh, os_ = Lhat[su], Lstar[su]
absq = os_.abs()
edges = torch.quantile(absq[torch.randperm(absq.numel(), device=DEV)[:4_000_000]].float(),
                       torch.linspace(0, 1, 11, device=DEV)).double()
offd = []
for k in range(10):
    m = (absq > edges[k]) & (absq <= edges[k + 1]) if k < 9 else (absq > edges[k])
    a, b = oh[m], os_[m]
    # least-squares scale of predicted vs true within the stratum, and relative residual
    s = float((a @ b) / (b @ b))
    rel = float((a - s * b).norm() / b.norm())
    offd.append(dict(k=k, ls_scale=s, residual_after_scale=rel,
                     plain_relative_error=float((a - b).norm() / b.norm())))
out['offdiagonal_by_magnitude_decile'] = offd
print('off-diagonal LS scale by |L*| decile (small first): ' + ' '.join('%.3f' % v['ls_scale'] for v in offd))
print('off-diagonal rel error after scale:              ' + ' '.join('%.3f' % v['residual_after_scale'] for v in offd))
del su, oh, os_, absq

# ---- 3. diagonal swap
out['as_predicted'] = score_inverse(Lhat, Rstar)
H = Lhat.clone(); H.diagonal().copy_(ps); out['true_diag_pred_offdiag'] = score_inverse(H, Rstar); del H
H = Lstar.clone(); H.diagonal().copy_(ph); out['pred_diag_true_offdiag'] = score_inverse(H, Rstar); del H
# global rescale of the predicted factor by the median pivot ratio
H = Lhat / float(ratio.median()); out['pred_rescaled_by_median_pivot_ratio'] = score_inverse(H, Rstar); del H
gc.collect(); torch.cuda.empty_cache()
print('\n%-36s %10s %10s %10s' % ('', 'g', '1/mu_min', 'mu_max'))
for k in ('as_predicted', 'true_diag_pred_offdiag', 'pred_diag_true_offdiag', 'pred_rescaled_by_median_pivot_ratio'):
    s = out[k]; print('%-36s %10.3f %10.3f %10.3f' % (k, s['g'], 1 / s['mu_min'], s['mu_max']))
(R5 / 'AUDIT.json').write_text(json.dumps(out, indent=1))
print('\nwritten AUDIT.json')
