"""Revision (E1): median / min over the repetitions of bench_cpu2.py (Table 5) and lat_direct_cpu2.py (Table 6) outputs.
Usage: r1_host_summary.py <host_dir>[,<host_dir2>...] <out.json> [--one <host1_dir>[,...]]
(records from several machines are merged; every row carries the machine name(s) of its records)
Reads <host_dir>/bench_cpu2_chol_rep*.json, bench_cpu2_luref.json, iparm_tuned.json, lat2_<layout>_chol_rep*.json,
lat2_<layout>_lu.json. Times in s, memory in GiB (2**30 bytes); PARDISO memory = iparm(16) + iparm(17) (kB / 2**20)."""
import sys, json, glob
from pathlib import Path
import numpy as np


def agg(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return None if not v else dict(median=float(np.median(v)), min=float(np.min(v)), max=float(np.max(v)), n=len(v))


def G(H, pat):
    return sorted(f for h in H.split(',') for f in glob.glob(f'{h}/{pat}'))


def mach(recs):
    return sorted({r.get('machine') or (r.get('env') or {}).get('host') for r in recs} - {None})


def get(d, *ks):
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def table5(H):
    reps = [json.loads(Path(f).read_text()) for f in G(H, 'bench_cpu2_chol_rep*.json')]
    out = {}
    cases = sorted({c for r in reps for c in r['per_case']})
    for c in cases:
        rs = [r['per_case'][c] for r in reps if c in r['per_case'] and 'error' not in r['per_case'][c]]
        if not rs:
            out[c] = dict(error=[r['per_case'][c].get('error') for r in reps if c in r['per_case']]); continue
        o = dict(machine=mach([r for r in reps if c in r['per_case']]), reps=len(rs), dofs=rs[0]['dofs'], ports=rs[0]['ports'], interior=rs[0]['interior'],
                 topology_s=get(rs[0], 'topology', 'seconds'))
        for k in ('setup_s', 'assembly_s', 'analysis_s', 'factor_s', 'factor_peak_rss_GiB', 'query_peak_rss_GiB',
                  'process_peak_rss_GiB', 'throttled_s'):
            o[k] = agg([r.get(k) for r in rs])
        o['analysis_factor_s'] = agg([r['analysis_s'] + r['factor_s'] for r in rs])
        o['front_end_host_s'] = agg([r['setup_s'] + r['assembly_s'] for r in rs])
        o['pardiso_GiB'] = rs[0]['pardiso']['peak_mem_GiB']; o['nnz_factor'] = rs[0]['pardiso']['nnz_factor']
        for B in ('1', '16', '64'):
            o[f'query_B{B}_s'] = agg([get(r, 'queries', B, 'query_s', 'median') for r in rs])
            o[f'query_B{B}_min_s'] = agg([get(r, 'queries', B, 'query_s', 'min') for r in rs])
            o[f'solve_only_B{B}_s'] = agg([get(r, 'queries', B, 'solve_only_s', 'median') for r in rs])
            o[f'rel_residual_B{B}_max'] = max(get(r, 'queries', B, 'rel_residual_max') or 0 for r in rs)
            o[f'factor_read_B{B}_GBps'] = agg([get(r, 'queries', B, 'factor_read_GBps') for r in rs])
        o['schur_s'] = agg([get(r, 'explicit_S_schur', 'seconds') for r in rs])
        o['schur_GiB'] = get(rs[0], 'explicit_S_schur', 'peak_mem_GiB')
        o['schur_peak_rss_GiB'] = agg([get(r, 'explicit_S_schur', 'peak_rss_GiB') for r in rs])
        o['schur_verify_rel_err_max'] = max(get(r, 'explicit_S_schur', 'verify_rel_err') or 0 for r in rs)
        o['schur_failed_variants'] = get(rs[0], 'explicit_S_schur', 'failed_variants')
        o['blocked_s'] = agg([get(r, 'explicit_S_blocked', 'seconds') for r in rs])
        o['blocked_plus_factor_s'] = agg([get(r, 'explicit_S_blocked', 'plus_factor_s') for r in rs])
        o['blocked_extrapolated'] = any(get(r, 'explicit_S_blocked', 'extrapolated') for r in rs)
        o['dense_S_GiB'] = get(rs[0], 'explicit_S_schur', 'dense_GiB') or get(rs[0], 'explicit_S_blocked', 'dense_GiB')
        if o['schur_s'] and o['blocked_plus_factor_s']:
            o['explicit_S_faster'] = 'schur' if o['schur_s']['median'] <= o['blocked_plus_factor_s']['median'] else 'blocked'
        o['loadavg_start'] = [r.get('loadavg_start') for r in rs]
        out[c] = o
    for lu in G(H, 'bench_cpu2_luref.json'):
        L = json.loads(Path(lu).read_text())
        for c, r in L['per_case'].items():
            if 'error' in r:
                out.setdefault(c, {})['luref_error'] = r['error']; continue
            out.setdefault(c, {})['luref'] = dict(
                factor_s=r['factor_s'], pardiso_GiB=r['pardiso']['peak_mem_GiB'], nnz_factor=r['pardiso']['nnz_factor'],
                hash_mode=r['pardiso']['hash_mode'], default_iparm=r['pardiso']['iparm_1based'],
                query_pypardiso_s={B: get(r, 'queries_pypardiso', B, 'query_s') for B in ('1', '16', '64')},
                query_direct_s={B: get(r, 'queries_direct', B, 'query_s') for B in ('1', '16', '64')},
                wrapper_check_s=r['wrapper_check_s'], wrapper_copies_s=r['wrapper_copies_s'],
                single_vector_overhead=r['single_vector_overhead'])
    tp = G(H, 'iparm_tuned.json')
    return dict(cells=out, iparm=[json.loads(Path(f).read_text()) for f in tp], n_rep_files=len(reps))


def table6(H):
    out = {}
    for lay in sorted({Path(f).name.split('_')[1] for f in G(H, 'lat2_*.json')}):
        o = {}
        for tag in ('chol', 'lu'):
            fs = G(H, f'lat2_{lay}_{tag}*.json')
            recs = [json.loads(Path(f).read_text()) for f in fs]
            runs = [(rc, r) for rc in recs for r in rc.get('runs', [])]
            if not runs:
                continue
            done = [(rc, r) for rc, r in runs if 'factor_s' in r]
            t = dict(machine=mach(recs), files=len(fs), global_dofs=recs[0].get('global_dofs'), nnz_upper=recs[0].get('nnz_upper'),
                     cells_s=agg([rc.get('cells_s') for rc in recs]), global_assembly_s=agg([rc.get('global_assembly_s') for rc in recs]),
                     predicted_GiB=[r['predicted_GiB'] for _, r in runs], skipped=[r.get('skipped') for _, r in runs if 'skipped' in r],
                     analysis_s=agg([r['analysis_s'] for _, r in runs]))
            if done:
                t.update(factor_s=agg([r['factor_s'] for _, r in done]), solve_s=agg([r['solve_s'] for _, r in done]),
                         pardiso_phases_s=agg([r['analysis_s'] + r['factor_s'] + r['solve_s'] for _, r in done]),
                         total_s=agg([rc['cells_s'] + rc['lattice_s'] + rc['global_assembly_s'] + rc.get('scaling_s', 0) +
                                      r['analysis_s'] + r['factor_s'] + r['solve_s'] for rc, r in done]),
                         pardiso_GiB=done[0][1]['pardiso']['peak_mem_GiB'], nnz_factor=done[0][1]['pardiso']['nnz_factor'],
                         peak_rss_GiB=agg([r.get('process_peak_rss_GiB') for _, r in done]),
                         factor_peak_rss_GiB=agg([r.get('factor_peak_rss_GiB') for _, r in done]),
                         rel_residual_max=max(max(r['rel_residual']) for _, r in done),
                         compliance=done[0][1]['compliance'], loadavg_start=[rc.get('loadavg_start') for rc in recs],
                         throttled_s=agg([r.get('throttled_s') for _, r in done]))
            o[tag] = t
        if 'compliance' in o.get('chol', {}) and 'compliance' in o.get('lu', {}):
            a, b = np.array(o['chol']['compliance']), np.array(o['lu']['compliance'])
            o['chol_vs_lu_compliance_rel'] = float(np.max(np.abs(a / b - 1)))
        out[lay] = o
    return out


def single_core(D):
    out = {}
    for f in G(D, 'lat1_*_chol*.json'):
        d = json.loads(Path(f).read_text())
        for r in d.get('runs', []):
            out[Path(f).stem] = dict(machine=mach([d]), layout=d.get('layout'), global_dofs=d.get('global_dofs'),
                                     threads=(d.get('env') or {}).get('mkl_max_threads'), cells_s=d.get('cells_s'),
                                     global_assembly_s=d.get('global_assembly_s'), **{k: r.get(k) for k in (
                                         'analysis_s', 'factor_s', 'solve_s', 'predicted_GiB', 'skipped', 'rel_residual',
                                         'compliance', 'process_peak_rss_GiB', 'throttled_s')}, iparm=r.get('iparm_requested'))
    for f in G(D, 'scipy_*.json'):
        if '.child' in f or 'selftest' in f:
            continue
        d = json.loads(Path(f).read_text())
        out[Path(f).stem] = dict(machine=mach([d]), layout=d.get('layout'), global_dofs=d.get('global_dofs'), nnz_full=d.get('nnz_full'),
                                 cells_s=d.get('cells_s'), solves={v: {k: c.get(k) for k in (
                                     'status', 'detail', 'factor_s', 'solve_s', 'seconds', 'nnz_L', 'nnz_U', 'LU_GiB', 'peak_rss_GiB',
                                     'child_peak_rss_GiB', 'as_limit_GiB', 'wall_limit_s', 'returncode', 'rel_residual', 'reference',
                                     'splu_kwargs', 'solver')} for v, c in d.get('solves', {}).items()})
    return out


def main(argv):
    H, out = argv[0], argv[1]
    one = argv[argv.index('--one') + 1] if '--one' in argv else None
    env = G(H, 'env_host*.json')
    rec = dict(env=[json.loads(Path(e).read_text()) for e in env if e.endswith('.json')], table5=table5(H), table6=table6(H),
               single_core=single_core(one) if one else None,
               units='seconds; GiB = 2**30 bytes; PARDISO kB / 2**20 = GiB')
    Path(out).write_text(json.dumps(rec, indent=1, default=float))
    print('summary written', out)


if __name__ == '__main__':
    main(sys.argv[1:])
