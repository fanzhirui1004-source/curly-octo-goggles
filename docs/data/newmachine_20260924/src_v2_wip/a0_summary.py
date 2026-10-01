"""Summary of a0_eval.py results: per map and variant, the class-mean energy excess averaged over cells (equal weights,
as in the G-cell report) and its maximum over cells; worst-direction lower bounds; physical rigid-mode energy; map
statistics; errors. Separate rows for full and cut cells.
Usage: a0_summary.py <RESULTS.jsonl> <out.json> [<out.md>]"""
import json, sys
from collections import defaultdict
import numpy as np

src, out = sys.argv[1], sys.argv[2]
md = sys.argv[3] if len(sys.argv) > 3 else None
R = [json.loads(l) for l in open(src) if l.strip()]
errors = [dict(case=r['case'], map=r['map'], error=r['error']) for r in R if 'error' in r]
R = [r for r in R if 'error' not in r]
maps = list(dict.fromkeys(r['map'] for r in R))
variants = ['V0', 'V0R', 'V0ref', 'Conly']
classes = ['force_c', 'face_c', 'force', 'macro', 'grf']
kind = lambda c: 'full' if c.endswith('_full') else 'cut'
S = {}
for m in maps:
    rows = [r for r in R if r['map'] == m]
    S[m] = dict(cells=len(rows), det_X_min=min(r['map_stats']['det_X_min'] for r in rows),
                cond_J_max=max(r['map_stats']['cond_J_max'] for r in rows), variants={})
    for v in variants:
        vv = [r for r in rows if v in r.get('variants', {})]
        if not vv:
            continue
        d = {}
        for grp in ('all', 'full', 'cut'):
            g = [r for r in vv if grp == 'all' or kind(r['case']) == grp]
            if not g:
                continue
            d[grp] = {c: dict(mean=float(np.mean([r['variants'][v][c]['mean'] for r in g])),
                              max_cell=float(np.max([r['variants'][v][c]['mean'] for r in g])),
                              max_dir=float(np.max([r['variants'][v][c]['max'] for r in g])),
                              min_dir=float(np.min([r['variants'][v][c]['min'] for r in g])))
                      for c in classes if all(c in r['variants'][v] for r in g)}
        w = [r['variants'][v].get('worst_mu') for r in vv if r['variants'][v].get('worst_mu') is not None]
        d['worst_mu_minus1'] = dict(median=float(np.median(w)) - 1, max=float(np.max(w)) - 1) if w else None
        d['rigid_energy_rel_max'] = float(np.max([r['variants'][v]['rigid_energy_rel'] for r in vv]))
        S[m]['variants'][v] = d
json.dump(dict(maps=S, errors=errors, n=len(R)), open(out, 'w'), indent=1)
if md:
    L = ['| map | cells | min det | max cond | V0 force_c | V0R force_c | Conly force_c | V0 face_c | V0R face_c | V0 max cell (force_c) | V0R max cell | V0 worst mu-1 (median / max) | V0R worst | V0ref rigid energy |',
         '|' + '---|' * 14]
    p = lambda x: f'{100 * x:.4f}%'
    for m in maps:
        s = S[m]; V = s['variants']
        g = lambda v, c, k='mean': p(V[v]['all'][c][k]) if v in V and c in V[v]['all'] else '-'
        wm = lambda v: (f"{100 * V[v]['worst_mu_minus1']['median']:.3f}% / {100 * V[v]['worst_mu_minus1']['max']:.3f}%"
                        if v in V and V[v]['worst_mu_minus1'] else '-')
        L.append(f"| {m} | {s['cells']} | {s['det_X_min']:.3f} | {s['cond_J_max']:.3f} | {g('V0', 'force_c')} | {g('V0R', 'force_c')} | "
                 f"{g('Conly', 'force_c')} | {g('V0', 'face_c')} | {g('V0R', 'face_c')} | {g('V0', 'force_c', 'max_cell')} | "
                 f"{g('V0R', 'force_c', 'max_cell')} | {wm('V0')} | {wm('V0R')} | "
                 f"{V['V0ref']['rigid_energy_rel_max']:.1e} |" if 'V0ref' in V else '')
    if errors:
        L += ['', f'errors: {len(errors)}'] + [f"- {e['case']} {e['map']}: {e['error']}" for e in errors]
    open(md, 'w').write('\n'.join(L) + '\n')
print(json.dumps(dict(n=len(R), errors=len(errors), maps=len(maps))))
