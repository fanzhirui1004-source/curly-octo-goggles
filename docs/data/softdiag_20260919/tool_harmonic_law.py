"""Aggregate the soft_diag spectra: which functional actually governs the physics error."""
import json, glob, os
import numpy as np

E = '/root/autodl-tmp/CLAUDE_EQUI_20260918'
rows = []
for p in sorted(glob.glob(f'{E}/SOFTDIAG/SEAT_*/SOFT_DIAG.json')):
    d = json.load(open(p))
    seat = d['seat']
    z = np.load(os.path.join(os.path.dirname(p), 'SPECTRUM.npz'))
    mu = z['mu'].astype(np.float64); W = z['weights'].astype(np.float64)   # (d, nloads)
    W = W / W.sum(axis=0, keepdims=True)
    acc = json.load(open(f'{E}/ACCEPT_MULTI/SEAT_{seat}/ASSEMBLE_TWO.json'))
    names = d['load_names']
    inv = 1.0 / mu
    for j, nm in enumerate(names):
        w = W[:, j]
        pred = float((w * inv).sum()) - 1.0
        soft = float((w * (inv - 1.0))[mu < 1.0].sum())        # over-soft: positive
        stiff = float((w * (inv - 1.0))[mu > 1.0].sum())       # over-stiff: negative
        e, pr = acc['exact'][nm], acc['predicted'][nm]
        meas_c = pr['compliance'] / e['compliance'] - 1.0
        fe = e['sens']['A'] / e['compliance']; fp = pr['sens']['A'] / pr['compliance']
        split = fp / fe - 1.0
        live = w >= 1e-3
        rows.append(dict(
            seat=seat, load=nm, g=d['g'], inv_mu_min=1.0 / d['mu_min'], mu_max=d['mu_max'],
            pred=pred, meas_c=meas_c, gap=pred - meas_c,
            soft=soft, stiff=stiff, cancel=abs(soft + stiff) / (abs(soft) + abs(stiff) + 1e-300),
            w_at_mu_min=float(w[int(np.argmin(mu))]), w_at_mu_max=float(w[int(np.argmax(mu))]),
            split=split, sens_A=pr['sens']['A'] / e['sens']['A'] - 1.0,
            n_live=int(live.sum()), g_live=float(np.maximum(mu[live], inv[live]).max()) if live.any() else np.nan,
            soft_live=float(np.maximum(inv[live], 0).max()) if live.any() else np.nan,
            e16=d['soft']['16']['energy_share'][j], e1024=d['soft']['1024']['energy_share'][j],
            gs16=d['soft']['16']['g_soft'], gs1024=d['soft']['1024']['g_soft'],
            part_min=d['soft32_participation_min'], A_cond=d['A_cond'],
            frac3=d['frac_within_3pct']))

print(f'{"seat":>7} {"load":<11} {"g":>8} {"1/mumin":>8} {"pred%":>8} {"meas%":>8} {"gap%":>7} '
      f'{"soft%":>8} {"stiff%":>8} {"cancel":>7} {"w@min":>9} {"split%":>8} {"sensA%":>8}')
for r in sorted(rows, key=lambda r: (-r['g'], r['load'])):
    print(f'{r["seat"]:>7} {r["load"]:<11} {r["g"]:>8.3f} {r["inv_mu_min"]:>8.3f} {100*r["pred"]:>8.3f} '
          f'{100*r["meas_c"]:>8.3f} {100*r["gap"]:>7.3f} {100*r["soft"]:>8.2f} {100*r["stiff"]:>8.2f} '
          f'{r["cancel"]:>7.3f} {r["w_at_mu_min"]:>9.2e} {100*r["split"]:>8.3f} {100*r["sens_A"]:>8.3f}')

# per-seat worst, and the rank correlations that matter
import itertools
seats = sorted({r['seat'] for r in rows})
print('\n== per seat: what predicts the worst measured compliance / sensitivity? ==')
print(f'{"seat":>7} {"g":>8} {"1/mumin":>8} {"g_live":>8} {"softlive":>8} {"gs16":>8} {"maxpred%":>9} '
      f'{"maxmeas%":>9} {"maxsplit%":>9} {"maxsensA%":>9} {"partmin":>8} {"frac3":>6}')
per = {}
for s in seats:
    rs = [r for r in rows if r['seat'] == s]
    per[s] = dict(g=rs[0]['g'], inv_mu_min=rs[0]['inv_mu_min'], g_live=rs[0]['g_live'],
                  soft_live=rs[0]['soft_live'], gs16=rs[0]['gs16'],
                  maxpred=max(abs(r['pred']) for r in rs), maxmeas=max(abs(r['meas_c']) for r in rs),
                  maxsplit=max(abs(r['split']) for r in rs), maxsens=max(abs(r['sens_A']) for r in rs),
                  part_min=rs[0]['part_min'], frac3=rs[0]['frac3'])
    v = per[s]
    print(f'{s:>7} {v["g"]:>8.3f} {v["inv_mu_min"]:>8.3f} {v["g_live"]:>8.3f} {v["soft_live"]:>8.3f} '
          f'{v["gs16"]:>8.3f} {100*v["maxpred"]:>9.3f} {100*v["maxmeas"]:>9.3f} {100*v["maxsplit"]:>9.3f} '
          f'{100*v["maxsens"]:>9.3f} {v["part_min"]:>8.2f} {v["frac3"]:>6.3f}')

def spear(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])

y = [per[s]['maxmeas'] for s in seats]
ys = [per[s]['maxsens'] for s in seats]
print('\n== Spearman vs worst measured compliance / worst sens_A ==')
for key in ('g', 'inv_mu_min', 'g_live', 'soft_live', 'gs16', 'maxpred', 'maxsplit', 'part_min', 'frac3'):
    x = [per[s][key] for s in seats]
    print(f'{key:>12}  compliance {spear(x, y):+.3f}   sensitivity {spear(x, ys):+.3f}')
print('\npredictor accuracy: max |pred - meas| over all seats/loads =',
      f'{max(abs(r["gap"]) for r in rows):.4f}')
