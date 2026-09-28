"""Revision: summary of lat_cond_cpu.py outputs (route (b)). Usage: r1_cond_summary.py <dir> -> <dir>/hostcond_summary.json"""
import json, glob, sys, numpy as np
from pathlib import Path
H = sys.argv[1]; out = {}
agg = lambda v: None if not [x for x in v if x is not None] else dict(median=float(np.median([x for x in v if x is not None])), min=float(min(x for x in v if x is not None)), n=len([x for x in v if x is not None]))
import re
keys = sorted({(m.group(1) or '_block', m.group(2)) for f in glob.glob(f'{H}/latcond*_rep*.json')
               for m in [re.match(r'latcond(_pardiso)?_(.+)_rep\d+\.json$', Path(f).name)] if m})
for sv, L in keys:
    pre = 'latcond_pardiso' if sv == '_pardiso' else 'latcond'
    rs = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(f'{H}/{pre}_{L}_rep*.json'))]
    ok = [r for r in rs if 'compliance' in r]
    o = dict(reps=len(rs), completed=len(ok), skipped=[r.get('skipped') for r in rs if 'skipped' in r],
             free_retained=rs[0].get('free_retained'), nnz_upper=rs[0].get('nnz_upper'), csr_GiB=rs[0].get('csr_GiB'),
             dense_S_total_GiB=rs[0].get('dense_S_total_GiB', rs[0].get('dense_S_total_GiB_predicted')),
             predicted_GiB=rs[0].get('predicted_GiB'), interface=rs[0].get('interface'), solver=rs[0].get('solver'),
             n_shared=rs[0].get('n_shared'), interface_dense_GiB=rs[0].get('interface_dense_GiB'), held_factor_GiB=rs[0].get('held_factor_GiB'))
    for k in ('front_end_s', 'schur_s', 'lattice_s', 'structure_s', 'indices_s', 'values_s', 'analysis_s', 'factor_s', 'solve_s',
              'condensed_s', 'total_s', 'process_peak_rss_GiB', 'assembly_peak_rss_GiB', 'factor_solve_peak_rss_GiB', 'throttled_s',
              'private_elimination_s', 'interface_factor_solve_s', 'back_substitution_s', 'block_peak_rss_GiB'):
        o[k] = agg([r.get(k) for r in ok])
    if ok:
        if 'pardiso' in ok[0]:
            o['pardiso_GiB'] = ok[0]['pardiso']['peak_mem_GiB']; o['nnz_factor'] = ok[0]['pardiso']['nnz_factor']
        o['schur_pardiso_GiB_max'] = max(c['schur']['pardiso']['peak_mem_GiB']['total'] for c in ok[0]['cells'].values())
        o['rel_residual_max'] = max((max(r['rel_residual']) for r in ok if r.get('rel_residual')), default=None)
        o['compliance'] = ok[0]['compliance']; o['reference'] = ok[0].get('reference')
    out[f'{sv.strip("_")}:{L}'] = o
Path(f'{H}/hostcond_summary.json').write_text(json.dumps(out, indent=1, default=float)); print('ok')
