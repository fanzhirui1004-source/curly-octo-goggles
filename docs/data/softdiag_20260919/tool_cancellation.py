"""Is the cancellation statistical (many small contributions of both signs) or a few big ones?"""
import json, glob, os
import numpy as np
E = '/root/autodl-tmp/CLAUDE_EQUI_20260918'
print(f'{"seat":>7} {"load":<11} {"g":>7} {"w(mu>1)":>8} {"w(mu<1)":>8} {"soft%":>7} {"stiff%":>7} {"net%":>7} '
      f'{"n90soft":>7} {"n90stif":>7} {"top1soft%":>9} {"top1stif%":>9} {"n_eff":>7}')
for p in sorted(glob.glob(f'{E}/SOFTDIAG/SEAT_*/SOFT_DIAG.json'), key=lambda q: -json.load(open(q))['g']):
    d = json.load(open(p)); seat = d['seat']
    z = np.load(os.path.join(os.path.dirname(p), 'SPECTRUM.npz'))
    mu = z['mu'].astype(np.float64); W = z['weights'].astype(np.float64); W /= W.sum(axis=0, keepdims=True)
    for j, nm in enumerate(d['load_names']):
        w = W[:, j]; contrib = w * (1.0 / mu - 1.0)
        soft = contrib[mu < 1]; stiff = contrib[mu > 1]
        def n90(v):
            v = np.sort(np.abs(v))[::-1]; c = np.cumsum(v)
            return int(np.searchsorted(c, 0.9 * c[-1]) + 1) if len(v) else 0
        n_eff = float(1.0 / (w * w).sum())
        print(f'{seat:>7} {nm:<11} {d["g"]:>7.2f} {w[mu>1].sum():>8.3f} {w[mu<1].sum():>8.3f} '
              f'{100*soft.sum():>7.2f} {100*stiff.sum():>7.2f} {100*contrib.sum():>7.2f} '
              f'{n90(soft):>7d} {n90(stiff):>7d} {100*np.abs(soft).max():>9.3f} {100*np.abs(stiff).max():>9.3f} {n_eff:>7.0f}')
