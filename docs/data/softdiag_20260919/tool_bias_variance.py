"""Is the compliance error a second-order residue of a near-cancellation?"""
import json, glob, os
import numpy as np
E = '/root/autodl-tmp/CLAUDE_EQUI_20260918'
print(f'{"seat":>7} {"load":<11} {"g":>8} {"bias":>8} {"halfvar":>8} {"2nd%":>8} {"exact%":>8} {"meas%":>8} '
      f'{"rmslog_w":>8} {"rmslog_u":>8} {"maxlog_u":>8}')
per = {}
for p in sorted(glob.glob(f'{E}/SOFTDIAG/SEAT_*/SOFT_DIAG.json')):
    d = json.load(open(p)); seat = d['seat']
    z = np.load(os.path.join(os.path.dirname(p), 'SPECTRUM.npz'))
    mu = z['mu'].astype(np.float64); W = z['weights'].astype(np.float64)
    W = W / W.sum(axis=0, keepdims=True)
    l = np.log(mu)
    acc = json.load(open(f'{E}/ACCEPT_MULTI/SEAT_{seat}/ASSEMBLE_TWO.json'))
    rms_u = float(np.sqrt((l * l).mean())); max_u = float(np.abs(l).max())
    rows = []
    for j, nm in enumerate(d['load_names']):
        w = W[:, j]
        bias = float((w * l).sum()); half = float(0.5 * (w * l * l).sum())
        exact = float((w / mu).sum()) - 1.0
        meas = acc['predicted'][nm]['compliance'] / acc['exact'][nm]['compliance'] - 1.0
        rms_w = float(np.sqrt((w * l * l).sum()))
        print(f'{seat:>7} {nm:<11} {d["g"]:>8.3f} {bias:>8.4f} {half:>8.4f} {100*(-bias+half):>8.3f} '
              f'{100*exact:>8.3f} {100*meas:>8.3f} {rms_w:>8.4f} {rms_u:>8.4f} {max_u:>8.3f}')
        rows.append((abs(meas), rms_w, half, abs(bias)))
    per[seat] = dict(g=d['g'], rms_u=rms_u, max_u=max_u,
                     maxmeas=max(r[0] for r in rows), maxrmsw=max(r[1] for r in rows),
                     maxhalf=max(r[2] for r in rows), maxbias=max(r[3] for r in rows))

def spear(a, b):
    ra = np.argsort(np.argsort(np.asarray(a, float))).astype(float)
    rb = np.argsort(np.argsort(np.asarray(b, float))).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])

seats = sorted(per)
y = [per[s]['maxmeas'] for s in seats]
print('\n== Spearman vs worst measured compliance ==')
for k in ('g', 'rms_u', 'max_u', 'maxrmsw', 'maxhalf', 'maxbias'):
    print(f'{k:>10}  {spear([per[s][k] for s in seats], y):+.3f}')
print('\n== g vs the unweighted log spread (is g just a proxy for overall fit quality?) ==')
print('spearman(g, rms_log_mu_unweighted) =',
      f"{spear([per[s]['g'] for s in seats], [per[s]['rms_u'] for s in seats]):+.3f}")
print('spearman(rms_log_unweighted, rms_log_energy_weighted) =',
      f"{spear([per[s]['rms_u'] for s in seats], [per[s]['maxrmsw'] for s in seats]):+.3f}")
print('\n== per seat ==')
print(f'{"seat":>7} {"g":>8} {"rmslog_u":>9} {"maxrmslog_w":>12} {"maxbias":>8} {"maxhalfvar%":>12} {"maxmeas%":>9}')
for s in seats:
    v = per[s]
    print(f'{s:>7} {v["g"]:>8.3f} {v["rms_u"]:>9.4f} {v["maxrmsw"]:>12.4f} {v["maxbias"]:>8.4f} '
          f'{100*v["maxhalf"]:>12.3f} {100*v["maxmeas"]:>9.3f}')
