"""Fine-tune summary: single-cell class means on the 16 G twins (A3 zero-shot vs A3G) and the 80 P validation cells
(A3 vs A3G), and lattice compliance / sensitivity maxima (P twin hlat222, G glat222) for A3 and A3G."""
import json, sys
import numpy as np
D = sys.argv[1]
J = lambda f: json.load(open(f'{D}/{f}'))
cls = ['force', 'force_c', 'face', 'face_c', 'support', 'support_k', 'macro', 'grf']


def means(f, cases=None):
    pg = J(f)['per_geo']; cases = cases or list(pg)
    return {c: 100 * float(np.nanmean([pg[g]['0'].get(c, np.nan) for g in cases])) for c in cls}, \
           {c: 100 * float(np.nanmax([pg[g]['0'].get(c, np.nan) for g in cases])) for c in cls}


out = {}
for tag, f in (('G_A3', 'gval_A3.json'), ('G_A3G', 'gval_A3G.json'), ('P_A3', 'newval_A3_2grid.json'), ('P_A3G', 'newval_A3G.json')):
    try:
        m, x = means(f); out[tag] = dict(mean=m, max=x)
    except Exception as e:
        out[tag] = repr(e)


def lat(f, model):
    try:
        L = next(iter(J(f)['lattices'].values())); a = L[model]; se = np.asarray(a['sens_rel_err'])
        return dict(compliance_max_pct=100 * max(a['compliance_rel_err']), sens_max_pct=100 * float(se.max()), iterations=a['iterations'],
                    exact_iterations=L['exact']['iterations'])
    except Exception as e:
        return repr(e)


out['lat_P_A3'] = lat('lat_hetero222_deploy.json', 'A3'); out['lat_P_A3G'] = lat('lat_hetero222_A3G.json', 'A3G')
out['lat_G_A3'] = lat('lat_hetero_g222.json', 'A3'); out['lat_G_A3G'] = lat('lat_hetero_g222_A3G.json', 'A3G')
print(json.dumps(out, indent=1))
for t in ('G_A3', 'G_A3G', 'P_A3', 'P_A3G'):
    if isinstance(out[t], dict):
        print('%-6s mean ' % t + ' '.join('%s=%.3f' % (c, out[t]['mean'][c]) for c in cls))
