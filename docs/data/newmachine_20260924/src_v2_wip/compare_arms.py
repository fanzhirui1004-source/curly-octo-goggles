"""Paired comparison of step-2 continuation arms against the shared control (review section 4, stage 4 criteria).
For each arm, the last EVAL of its train.log (the selected weights: EMA when enabled) and, when present, the rotated-view
evaluation views_<arm>.json (eval_views.py). Per validation family and class: arm / control of the family mean energy
excess, the family p90 and the family sensitivity mean; 'wins' = families where the arm is lower. Also the per-geometry
max, mu on the fixed geometries, the certificate flag rate and the train-probe means (generalization gap).
Decision rule of the review: an arm is better when at least 3 of the 4 validation families improve in the same direction
(energy classes pooled), with no class mean worse by more than 5-10%.
Usage: compare_arms.py <dir with arm subdirs> <out.json> [control=c_ctrl] [arm ...]"""
import sys, json
from pathlib import Path
import numpy as np


def last_eval(d):
    L = [json.loads(l) for l in open(d / 'train.log')]
    ev = [x for x in L if x.get('event') == 'EVAL']
    if not ev:
        return None
    e = ev[-1]
    w = e.get('weights', {})
    sel = w.get(e.get('select', 'ema')) or w.get('ema') or w.get('raw')
    return dict(step=e['step'], sel=sel, top=e)


def fam_table(sel):
    out = {}
    for fam, v in sel['val_family'].items():
        out[fam] = {c: dict(mean=x['mean'], p90=x['p90'], sens=x.get('sens_mean')) for c, x in v.items() if isinstance(x, dict)}
    return out


def main(root, out, control='c_ctrl', arms=None):
    root = Path(root)
    arms = arms or [p.name for p in sorted(root.iterdir()) if p.is_dir() and p.name.startswith('c_') and p.name != control]
    base = last_eval(root / control)
    rec = dict(control=control, control_step=base['step'], arms={})
    bf = fam_table(base['sel'])
    classes = sorted({c for f in bf.values() for c in f})
    rec['control_val_mean'] = base['sel']['val_mean']; rec['control_probe_mean'] = base['sel'].get('train_geo_mean')
    for a in arms:
        if not (root / a / 'train.log').exists():
            continue
        e = last_eval(root / a)
        if e is None:
            continue
        af = fam_table(e['sel'])
        r = dict(step=e['step'], val_mean=e['sel']['val_mean'], probe_mean=e['sel'].get('train_geo_mean'), per_class={})
        pooled_wins = 0
        for fam in bf:
            ratios = [af[fam][c]['mean'] / bf[fam][c]['mean'] for c in classes if c in af.get(fam, {}) and bf[fam][c]['mean'] > 0]
            pooled_wins += int(np.exp(np.mean(np.log(ratios))) < 1) if ratios else 0
        for c in classes:
            rr = [af[f][c]['mean'] / bf[f][c]['mean'] for f in bf if c in af.get(f, {})]
            rp = [af[f][c]['p90'] / bf[f][c]['p90'] for f in bf if c in af.get(f, {})]
            rs = [af[f][c]['sens'] / bf[f][c]['sens'] for f in bf if c in af.get(f, {}) and bf[f][c]['sens']]
            r['per_class'][c] = dict(mean_ratio=float(np.mean(rr)), p90_ratio=float(np.mean(rp)), sens_ratio=float(np.mean(rs)) if rs else None,
                                     wins=int(sum(x < 1 for x in rr)), families=len(rr),
                                     val_mean_ratio=e['sel']['val_mean'][c] / base['sel']['val_mean'][c])
        r['pooled_family_wins'] = pooled_wins
        r['worst_class_mean_ratio'] = max(v['val_mean_ratio'] for v in r['per_class'].values())
        mu_b, mu_a = base['sel'].get('mu', {}), e['sel'].get('mu', {})
        r['mu'] = {g: (mu_b[g]['mu'], mu_a[g]['mu']) for g in mu_a if g in mu_b}
        cb, ca = base['sel'].get('cert', {}), e['sel'].get('cert', {})
        r['cert_flagged'] = (cb.get('flagged'), ca.get('flagged'))
        gmax = lambda s: {g: max(v[c]['max'] for c in v if isinstance(v[c], dict) and 'max' in v[c]) for g, v in s['val'].items()}
        mb, ma = gmax(base['sel']), gmax(e['sel'])
        r['geo_max_ratio_median'] = float(np.median([ma[g] / mb[g] for g in ma if g in mb]))
        r['better'] = bool(pooled_wins >= 3 and r['worst_class_mean_ratio'] <= 1.10)
        vb, va = root / f'views_{control}.json', root / f'views_{a}.json'
        if vb.exists() and va.exists():
            jb, ja = json.loads(vb.read_text()), json.loads(va.read_text())
            r['views'] = dict(identity={c: (jb['mean'][str(jb['views'][0])][c], ja['mean'][str(ja['views'][0])][c]) for c in ja['mean_over_rotated']},
                              rotated={c: (jb['mean_over_rotated'][c], ja['mean_over_rotated'][c]) for c in ja['mean_over_rotated']})
        rec['arms'][a] = r
        print(json.dumps({a: dict(better=r['better'], pooled_family_wins=pooled_wins, worst_class_mean_ratio=round(r['worst_class_mean_ratio'], 3),
                                  val_mean={c: round(v, 4) for c, v in r['val_mean'].items()},
                                  mean_ratio={c: round(v['mean_ratio'], 3) for c, v in r['per_class'].items()},
                                  p90_ratio={c: round(v['p90_ratio'], 3) for c, v in r['per_class'].items()},
                                  mu=r['mu'], views=r.get('views'))}), flush=True)
    Path(out).write_text(json.dumps(rec, indent=1))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], *(sys.argv[3:4] or ['c_ctrl']), sys.argv[4:] or None)
