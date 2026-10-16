"""Compare the exact CPU sensitivities of route (a) (and (b) if present) with each other and with NICE's (validation)."""
import json, sys
import numpy as np
V = sys.argv[1]
a = json.load(open(f'{V}/a_hlat221a.json'))
r = a['runs'][0]
print('route (a): compliance', [round(c, 6) for c in r['compliance'][:3]], 'max rel residual %.1e' % max(r['rel_residual']),
      'stages', {k: round(v, 1) for k, v in a['stages_s'].items()}, 'total %.1f' % a['total_s'], 'peak %.1f GiB' % a['process_peak_rss_GiB'])
sa = {c: np.array(v['s']) for c, v in a['sens'].items()}
print('route (a) sens seconds per cell', {c: round(v['seconds'], 1) for c, v in a['sens'].items()})
try:
    z = np.load(f'{V}/nice_sens_hlat221a.npz', allow_pickle=True)
    order = [str(x) for x in z['order']]
    sn = z['sens_ad_it0'][:, :, 0]
    errs = {c: float(np.linalg.norm(sn[i] - sa[c]) / np.linalg.norm(sa[c])) for i, c in enumerate(order) if c in sa}
    allr = float(np.linalg.norm(np.stack([sn[i] for i, c in enumerate(order)]) - np.stack([sa[c] for c in order])) /
                 np.linalg.norm(np.stack([sa[c] for c in order])))
    print('NICE vs exact (a): per-cell rel', {c: '%.2e' % e for c, e in errs.items()}, 'all %.2e' % allr)
    print('NICE compliance', [round(c, 6) for c in z['compliance_it0']])
except FileNotFoundError:
    print('no NICE file')
try:
    b = json.load(open(f'{V}/b_hlat221a.json'))
    sb = {c: np.array(v['s']) for c, v in b['sens'].items()}
    print('route (b): total %.1f (w/o sens %.1f)' % (b['total_s'], b['total_without_sens_s']), 'recovery %.1f sens %.1f' % (b['recovery_s'], b['sens_s']),
          'energy check', b['energy_check'], 'ref rel diff', b.get('reference', {}).get('rel_diff'))
    print('(a) vs (b) per-cell rel', {c: '%.1e' % (np.linalg.norm(sa[c] - sb[c]) / np.linalg.norm(sa[c])) for c in sa if c in sb})
    print('interior residuals', {c: '%.1e' % v['interior_rel_residual'] for c, v in b['recovery'].items()})
except FileNotFoundError:
    print('no (b) file yet')
