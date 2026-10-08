"""Facts of the exact checks of the plate designs (Section 6.11, Table ST27), from the records in this directory.

The records were produced by exact_check_cpu.py (Schur-complement route, PCG to 1e-10) on a rented CPU machine and
brought back without the port vectors (exact_cpu/*/q); the complete result archive exact_results.tgz (md5
6fdb6436f6904062aa1ea32857a55d8f) is kept in the author's file storage. Checked designs (minimal set agreed 2026-09-30):
plateB1 k = 0, 22 and plateB2 k = 0, 29 (B2 k = 0 shares the condensed matrices of B1 k = 0); hevalH_z k = 0 (Hom-z) and
xstartH_z k = 11 (X-z), compliance only.  Validity criteria: runbook EXACT_CHECK_RUNBOOK_CN.md section 7.
Usage: python3 exact_plates_facts.py   (writes EXACT_PLATES.json next to this file and prints a summary)"""
import glob, json, os

H = os.path.dirname(os.path.abspath(__file__))
F = json.load(open(os.path.join(H, '..', 'FACTS_6_11.json')))
NAME = {('plateB1', 0): 'B1:0', ('plateB1', 22): 'B1:22', ('plateB2', 0): 'B2:0', ('plateB2', 29): 'B2:29',
        ('hevalH_z', 0): 'Hom-z', ('xstartH_z', 11): 'X-z'}
out = {'designs': {}, 'T': {}, 'derived': {}}
for f in sorted(glob.glob(os.path.join(H, 'runs', '*', 'check_exact_*.json'))):
    run, k = f.split(os.sep)[-2], int(f[-8:-5])
    j = json.load(open(f)); c = j['cpu']
    d = dict(run=run, k=k, C_exact=j['C_exact'], C_hat=j['C_hat'], surrogate_err=j['surrogate_err'],
             grad_rel_err=j['grad_rel_err'], grad_cos=j['grad_cos'], comp_err_rel_to_max=j['comp_err_rel_to_max'],
             sign_agreement=j['sign_agreement'], n_vars=len(j['g_exact']) if j.get('g_exact') else None,
             exact_pcg=j['exact_pcg'], exact_true_residual=j['exact_true_residual'],
             pcg_converged=c['pcg_converged'], energy_T_vs_K_rel_max=c['energy_T_vs_K_rel_max'],
             V_exact_equals_V_nice=None if c['V_exact_cells'] is None else c['V_exact_cells'] == c['V_nice'], tau_packet_vs_history_max=c['tau_packet_vs_history_max'],
             body_vs_history_mismatches=len(c['body_vs_history']['mismatches']), distinct_cells=c['distinct_cells'],
             free_dofs=c['lattice']['free_dofs'], coarse=c['precond']['coarse'])
    # volume and the T-versus-K energy check are evaluated in the sensitivity phase only (None for compliance-only designs)
    d['valid'] = bool(d['pcg_converged'] and d['exact_true_residual'] < 1e-9 and d['V_exact_equals_V_nice'] is not False
                      and d['tau_packet_vs_history_max'] <= 5e-13 and d['body_vs_history_mismatches'] == 0)
    out['designs'][NAME[(run, k)]] = d
T = [json.load(open(f)) for f in glob.glob(os.path.join(H, 'exact_cpu', 'jobs', 'T_*.out.json'))]
out['T'] = dict(jobs=len(T), routes=sorted({t['route'] for t in T}), verify_cols=sorted({t['verify_cols'] for t in T}),
                verify_rel_err_max=max(t['verify_rel_err'] for t in T), asym_rel_max=max(t['asym_rel'] for t in T),
                failed_variants=sum(len(t['failed_variants']) for t in T), ports_range=[min(t['ports'] for t in T), max(t['ports'] for t in T)])
D = out['designs']; ex = lambda n: D[n]['C_exact']; nh = lambda n: D[n]['C_hat']
out['derived'] = {
    'homog_error_initial_vs_exact': {'y': F['Hmacro_y']['C0_macro'] / ex('B1:0') - 1, 'z': F['Hmacro_z']['C0_macro'] / ex('B2:0') - 1},
    'homog_error_initial_vs_nice': {'y': F['comparison']['y']['homog_prediction_error_initial'], 'z': F['comparison']['z']['homog_prediction_error_initial']},
    'Homz_vs_B2': {'exact': ex('Hom-z') / ex('B2:29') - 1, 'nice': nh('Hom-z') / nh('B2:29') - 1},
    'Xz_vs_B2': {'exact': ex('X-z') / ex('B2:29') - 1, 'nice': nh('X-z') / nh('B2:29') - 1},
    'Xz_vs_Homz': {'exact': ex('X-z') / ex('Hom-z') - 1, 'nice': nh('X-z') / nh('Hom-z') - 1},
    'reduction_B1_exact': 1 - ex('B1:22') / ex('B1:0'), 'reduction_B2_exact': 1 - ex('B2:29') / ex('B2:0'),
    'reduction_B1_nice': 1 - nh('B1:22') / nh('B1:0'), 'reduction_B2_nice': 1 - nh('B2:29') / nh('B2:0'),
    'surrogate_err_range': [min(d['surrogate_err'] for d in D.values()), max(d['surrogate_err'] for d in D.values())],
}
json.dump(out, open(os.path.join(H, 'EXACT_PLATES.json'), 'w'), indent=1)
for n, d in D.items():
    g = '' if d['grad_rel_err'] is None else (f" grad {100*d['grad_rel_err']:.3f}% cos {d['grad_cos']:.7f} comp p95/max "
                                             f"{100*d['comp_err_rel_to_max']['p95']:.3f}/{100*d['comp_err_rel_to_max']['max']:.3f}% sign {d['sign_agreement']} n {d['n_vars']}")
    print(f"{n:6s} C_exact {d['C_exact']:.6f} C_hat {d['C_hat']:.6f} err {100*d['surrogate_err']:+.4f}% pcg {d['exact_pcg']} res {d['exact_true_residual']:.1e} "
          f"ETK {d['energy_T_vs_K_rel_max']} valid {d['valid']}{g}")
print('T', out['T'])
for k, v in out['derived'].items():
    print(k, v)
