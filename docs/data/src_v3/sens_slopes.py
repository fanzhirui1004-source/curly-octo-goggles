"""Log-slopes d ln(obs)/d eps of the design-sensitivity observables, teacher (exact elements) vs geometry
(polyhedral elements): one-sided at h = 1e-4, central at h = 1e-4 and 1e-3, relative error of the geometry slope."""
import json, sys
from pathlib import Path
import numpy as np
X = Path('/root/autodl-tmp/CLAUDE_TAKEOVER_20260923/SENS_01')
TAG = {'-1/1000': 'm1000', '-1/10000': 'm10000', '1/10000': 'p10000', '1/1000': 'p1000'}
rows = {}
for base in sys.argv[1:]:
    rec = {}
    for side in ('teacher', 'geometry'):
        obs = {}
        for e, t in [('0', None), *TAG.items()]:
            f = X / 'obs' / f'{base}_{side}_{t or "0"}' / 'OBSERVABLES.json'
            if f.exists():
                obs[e] = json.loads(f.read_text())
        rec[side] = obs
    def series(o):
        return {k: np.array([v['log_det'], v['trace_inverse'], v['softest'], v['stiffest'], *v['compliance']]) for k, v in o.items()}
    names = ['log_det', 'trace_inverse', 'softest', 'stiffest'] + [f'compliance_{k}' for k in range(8)]
    out = dict(observables=names)
    for side in ('teacher', 'geometry'):
        s = series(rec[side])
        if not all(k in s for k in ('0', '-1/10000', '1/10000')):
            out[side] = dict(missing=sorted({'0', '-1/10000', '1/10000'} - set(s)))
            continue
        lg = {k: np.concatenate([v[:1], np.log(np.abs(v[1:]))]) for k, v in s.items()}  # log_det is already a log
        h1, h3 = 1e-4, 1e-3
        out[side] = dict(left_1e4=((lg['0'] - lg['-1/10000']) / h1).tolist(), right_1e4=((lg['1/10000'] - lg['0']) / h1).tolist(),
                         central_1e4=((lg['1/10000'] - lg['-1/10000']) / (2 * h1)).tolist(), admitted_eps=sorted(s))
        if '-1/1000' in s and '1/1000' in s:
            out[side]['central_1e3'] = ((lg['1/1000'] - lg['-1/1000']) / (2 * h3)).tolist()
    if 'central_1e4' in out.get('teacher', {}) and 'central_1e4' in out.get('geometry', {}):
        tc, gc = np.array(out['teacher']['central_1e4']), np.array(out['geometry']['central_1e4'])
        out['relative_slope_error_central_1e4'] = ((gc - tc) / np.abs(tc)).tolist()
        if 'central_1e3' in out['teacher'] and 'central_1e3' in out['geometry']:
            tc3, gc3 = np.array(out['teacher']['central_1e3']), np.array(out['geometry']['central_1e3'])
            out['relative_slope_error_central_1e3'] = ((gc3 - tc3) / np.abs(tc3)).tolist()
        out['max_abs_relative_slope_error_1e4'] = float(np.max(np.abs(out['relative_slope_error_central_1e4'])))
        out['within_3pct_1e4'] = bool(out['max_abs_relative_slope_error_1e4'] <= 0.03)
    rows[base] = out
(X / 'SLOPES.json').write_text(json.dumps(rows, indent=2))
for b, o in rows.items():
    print(b, 'max slope err 1e-4:', o.get('max_abs_relative_slope_error_1e4'))
    if 'relative_slope_error_central_1e4' in o:
        for nm, tv, gv, er in zip(o['observables'], o['teacher']['central_1e4'], o['geometry']['central_1e4'], o['relative_slope_error_central_1e4']):
            print('  %-14s teacher %+10.4f geometry %+10.4f rel %+.4f' % (nm, tv, gv, er))
