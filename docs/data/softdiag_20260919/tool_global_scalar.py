"""Would one global scalar per cell buy accuracy?  M_hat -> s M_hat sends mu -> mu/s^2, so
chat/c -> s^2 (chat/c).  A single s cannot zero every load, but it can centre them.  Two choices
are BC-free (usable at prediction time): s^2 = exp(mean log mu) and exp(median log mu) over the
whole spectrum, or over the k softest A* modes.  Reported against the per-seat oracle."""
import json, glob, os
import numpy as np
E = '/root/autodl-tmp/CLAUDE_EQUI_20260918'
print(f'{"seat":>7} {"g":>8} {"now%":>7} {"oracle%":>8} {"s_or":>7} {"meanlog%":>9} {"medlog%":>9} '
      f'{"bias_w%":>8} {"halfvar%":>9} {"floor%":>7}')
tot = {}
for p in sorted(glob.glob(f'{E}/SOFTDIAG/SEAT_*/SOFT_DIAG.json'), key=lambda q: -json.load(open(q))['g']):
    d = json.load(open(p)); seat = d['seat']
    z = np.load(os.path.join(os.path.dirname(p), 'SPECTRUM.npz'))
    mu = z['mu'].astype(np.float64); W = z['weights'].astype(np.float64)
    W = W / W.sum(axis=0, keepdims=True)
    l = np.log(mu)
    base = (W / mu[:, None]).sum(axis=0)              # chat/c per load at s = 1
    def worst(s2):
        return float(np.abs(s2 * base - 1.0).max())
    grid = np.exp(np.linspace(-0.5, 0.5, 20001))
    oracle_s2 = float(grid[np.argmin([worst(v) for v in grid])])
    bias = (W * l[:, None]).sum(axis=0); half = 0.5 * (W * l[:, None] ** 2).sum(axis=0)
    row = dict(g=d['g'], now=worst(1.0), oracle=worst(oracle_s2), s_or=np.sqrt(oracle_s2),
               mean=worst(float(np.exp(l.mean()))), med=worst(float(np.exp(np.median(l)))),
               bias=float(np.abs(bias).max()), half=float(half.max()),
               floor=float(np.abs(half - half.mean() + 0 * half).max()))
    # the irreducible part: after ANY global scalar, the spread across loads remains
    row['floor'] = float(.5 * (np.abs(base.max() - base.min())))
    tot[seat] = row
    print(f'{seat:>7} {row["g"]:>8.3f} {100*row["now"]:>7.3f} {100*row["oracle"]:>8.3f} {row["s_or"]:>7.4f} '
          f'{100*row["mean"]:>9.3f} {100*row["med"]:>9.3f} {100*row["bias"]:>8.3f} {100*row["half"]:>9.3f} '
          f'{100*row["floor"]:>7.3f}')
print('\nworst over seats: now %.3f%%  oracle %.3f%%  exp(mean log mu) %.3f%%  exp(median log mu) %.3f%%  '
      'irreducible spread %.3f%%' % (100 * max(r['now'] for r in tot.values()),
      100 * max(r['oracle'] for r in tot.values()), 100 * max(r['mean'] for r in tot.values()),
      100 * max(r['med'] for r in tot.values()), 100 * max(r['floor'] for r in tot.values())))
print('worst energy-weighted halfvar (the no-cancellation error) %.3f%%'
      % (100 * max(r['half'] for r in tot.values())))
