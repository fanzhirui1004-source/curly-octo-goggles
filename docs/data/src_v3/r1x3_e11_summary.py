"""Revision round 1, E11 (I-26): NICE in cube view 17 against view 0 on the 80 validation geometries. Reads the view-17
output of eval_views_pd.py and the view-0 record (R1/ACC/newval3_A3.json when present, else the archived
newval2_A3_2grid.json); per class: mean over geometries of the per-geometry mean error in each view, their ratio, the
median / max of the per-geometry ratio, pooled per-direction p95 / max when both records have per-direction data.
Usage: r1x3_e11_summary.py <view17.json> <out.json> [view0.json ...]  (first existing view-0 file is used)"""
import json, sys
from pathlib import Path
import numpy as np

v17p, outp = sys.argv[1], sys.argv[2]
cands = sys.argv[3:] or ['/root/autodl-tmp/OPL/S1/V2/R1/ACC/newval3_A3.json', '/root/autodl-tmp/OPL/S1/V2/newval2_A3_2grid.json']
v0p = next(p for p in cands if Path(p).exists())
a, b = json.loads(Path(v0p).read_text()), json.loads(Path(v17p).read_text())
g0, g17 = a['per_geo'], b['per_geo']
cases = sorted(set(g0) & set(g17))
classes = sorted(set.intersection(*[set(g0[c]['0']) & set(g17[c]['17']) for c in cases]))
out = dict(view0_source=v0p, view17_source=v17p, n_geometries=len(cases), per_class={})
for k in classes:
    e0 = np.asarray([g0[c]['0'][k] for c in cases], float); e17 = np.asarray([g17[c]['17'][k] for c in cases], float)
    ok = np.isfinite(e0) & np.isfinite(e17) & (e0 > 0)
    r = e17[ok] / e0[ok]
    row = dict(mean_view0=float(np.nanmean(e0)), mean_view17=float(np.nanmean(e17)), ratio_of_means=float(np.nanmean(e17) / np.nanmean(e0)),
               per_geo_ratio_median=float(np.median(r)), per_geo_ratio_max=float(r.max()), per_geo_ratio_min=float(r.min()),
               max_view0=float(np.nanmax(e0)), max_view17=float(np.nanmax(e17)), worst_case_view17=cases[int(np.nanargmax(e17))])
    if 'per_dir' in a and 'per_dir' in b:
        d0 = np.concatenate([np.asarray(a['per_dir'][c]['0'][k], float) for c in cases if k in a['per_dir'].get(c, {}).get('0', {})])
        d17 = np.concatenate([np.asarray(b['per_dir'][c]['17'][k], float) for c in cases if k in b['per_dir'].get(c, {}).get('17', {})])
        row.update(pooled_p95_view0=float(np.nanquantile(d0, 0.95)), pooled_p95_view17=float(np.nanquantile(d17, 0.95)),
                   pooled_max_view0=float(np.nanmax(d0)), pooled_max_view17=float(np.nanmax(d17)))
    out['per_class'][k] = row
m0 = np.mean([out['per_class'][k]['mean_view0'] for k in classes]); m17 = np.mean([out['per_class'][k]['mean_view17'] for k in classes])
out['all_classes'] = dict(mean_view0=m0, mean_view17=m17, ratio=m17 / m0)
Path(outp).write_text(json.dumps(out, indent=1))
print(json.dumps(out['all_classes']), json.dumps({k: round(v['ratio_of_means'], 3) for k, v in out['per_class'].items()}))
