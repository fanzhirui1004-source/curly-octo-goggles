"""Section 6.11 fact sheet: every number used in the Section 6.11 text, tables and figures, read from the archived run
records in this directory (histories, checks, homogenisation law and macro runs).  Writes FACTS_6_11.json and
FACTS_6_11.md next to this script.  Usage: python3 facts_6_11.py"""
import json
from pathlib import Path
import numpy as np

D = Path(__file__).resolve().parent


def hist(p):
    return [json.loads(l) for l in open(D / p)]


def base(c):
    return c.rsplit('_o', 1)[0]


def run_summary(H, meta=None):
    tv0, tvf = np.array(H[0]['tv']), np.array(H[-1]['tv'])
    it = [h['times']['iter_s'] for h in H]
    pert = [dict(k=h['k'], **{k: p[k] for k in ('attempt', 'eps', 'vertices', 'failed', 'regenerated')})
            for h in H for p in h.get('body_perturb', [])]
    # each record is the perturbation applied after a failed generation attempt; the last record of an iteration is the
    # perturbation of the design that was analysed
    applied = [dict(k=h['k'], eps=h['body_perturb'][-1]['eps'], n_vertices=len(h['body_perturb'][-1]['vertices']),
                    attempts=len(h['body_perturb']) + 1) for h in H if h.get('body_perturb')]
    # discrete switches: per cell (by base case) change of the fingerprint between consecutive iterations
    sw = []
    for a, b in zip(H[:-1], H[1:]):
        fa = {base(c): v for c, v in a['fps'].items()}; fb = {base(c): v for c, v in b['fps'].items()}
        n = {key: sum(fa[c].get(key) != fb[c].get(key) for c in fa if c in fb) for key in ('active', 'faces', 'ports', 'cut_nodes', 'weak', 'el_fringe', 'gp_fringe', 'shift')}
        sw.append(dict(k=b['k'], **n))
    out = dict(iterations=len(H), last_k=H[-1]['k'], C0=H[0]['C'], C_final=H[-1]['C'], C_ratio=H[-1]['C'] / H[0]['C'],
               V0=H[0]['V'], V_final=H[-1]['V'], Vstar=H[-1]['V'] / H[-1]['V_rel'], V_rel_final=H[-1]['V_rel'],
               tau_min_final=H[-1]['tau_min'], tau_max_final=H[-1]['tau_max'], span_max_final=H[-1]['span_max'],
               grad_max_final=H[-1]['grad_max'], dx_final=H[-1]['dx'],
               pcg=[h['pcg'] for h in H], true_residual_range=[min(h['true_residual'] for h in H), max(h['true_residual'] for h in H)],
               Ut_rho_rel_absmax=max(abs(h['Ut_rho_rel']) for h in H),
               iter_s_mean=float(np.mean(it)), iter_s_min=min(it), iter_s_max=max(it), iter_s_total=float(np.sum(it)),
               phase_means={k: float(np.mean([h['times'].get(k, 0.0) for h in H])) for k in ('prep_s', 'precond_s', 'solve_s', 'sens_s', 'bodies_s', 'mma_s')},
               gpu_peak_gb_max=max(h.get('gpu_peak_gb', 0) for h in H), host_peak_gb_max=max(h.get('host_peak_gb', 0) for h in H),
               body_perturbations=pert, body_perturbations_applied=applied, C_trace=[h['C'] for h in H], V_rel_trace=[h['V_rel'] for h in H],
               switches_per_iteration=sw,
               n_vertices=len(tv0), tau0_range=[float(tv0.min()), float(tv0.max())])
    if meta:
        out['n_free'] = int(len(meta['fixed']) - sum(meta['fixed'])); out['n_fixed'] = int(sum(meta['fixed']))
    return out


F = {}
# ---------------------------------------------------------------- case A (2x2x2), NICE and exact twin, exact checks
A = hist('optA/optA/history.jsonl'); Ax = hist('optA/optAx/history.jsonl')
F['A'] = run_summary(A, json.load(open(D / 'optA/optA/meta.json')))
F['A_exact_twin'] = run_summary(Ax)
F['A_checks'] = {}
for k in (0, 12, 23):
    c = json.load(open(D / f'optA/optA/check_{k:03d}.json'))
    F['A_checks'][k] = {x: c[x] for x in ('C_exact', 'C_hat', 'surrogate_err', 'grad_rel_err', 'grad_cos', 'comp_err_rel_to_max', 'sign_agreement', 'kkt_exact', 'exact_pcg', 'exact_true_residual')}
F['A_final_designs'] = dict(C_exact_of_NICE_design=F['A_checks'][23]['C_exact'], C_exact_of_exact_design=Ax[-1]['C'],
                            rel_diff=(F['A_checks'][23]['C_exact'] - Ax[-1]['C']) / Ax[-1]['C'],
                            tau_maxabs_diff=float(np.abs(np.array(A[-1]['tv']) - np.array(Ax[-1]['tv'])).max()),
                            tau_rms_diff=float(np.sqrt(((np.array(A[-1]['tv']) - np.array(Ax[-1]['tv'])) ** 2).mean())),
                            exact_iter_s_mean=float(np.mean([h['times']['iter_s'] for h in Ax])), nice_iter_s_mean=float(np.mean([h['times']['iter_s'] for h in A])))
# per-iteration NICE vs exact-twin compliance (same iteration index; trajectories differ slightly)
F['A_twin_trace'] = [dict(k=a['k'], C_nice=a['C'], C_exact_twin=b['C']) for a, b in zip(A, Ax)]
# ---------------------------------------------------------------- route validation on hlat222 (iteration 0)
pf, pa, pw, ps = (hist(f'optA/{r}/history.jsonl') for r in ('pilot222f', 'pilot222a', 'pilot222w', 'pilot222'))
F['route_validation'] = dict(
    slow_vs_fast=[dict(k=a['k'], C_slow=a['C'], C_fast=b['C'], pcg_slow=a['pcg'], pcg_fast=b['pcg'], iter_s_slow=a['times']['iter_s'], iter_s_fast=b['times']['iter_s'],
                       s_vertex_rel=float(np.linalg.norm(np.array(b['s_vertex']) - np.array(a['s_vertex'])) / np.linalg.norm(np.array(a['s_vertex']))))
                  for a, b in zip(ps, pf)],
    fd_vs_ad_k0=dict(C_fd=pf[0]['C'], C_ad=pa[0]['C'], s_vertex_rel=float(np.linalg.norm(np.array(pa[0]['s_vertex']) - np.array(pf[0]['s_vertex'])) / np.linalg.norm(np.array(pf[0]['s_vertex']))),
                     sens_s_fd=pf[0]['times']['sens_s'], sens_s_ad=pa[0]['times']['sens_s']),
    cold_vs_warm=[dict(k=a['k'], C_cold=a['C'], C_warm=b['C'], pcg_cold=a['pcg'], pcg_warm=b['pcg'], solve_cold=a['times']['solve_s'], solve_warm=b['times']['solve_s'],
                       warm_hit=b['times'].get('warm_hit'), warm_scale=b['times'].get('warm_scale')) for a, b in zip(pf, pw)])
# ---------------------------------------------------------------- plates B1 / B2, homogenisation, cross-start
P = D / 'plates'
for tag, r in (('B1', 'plateB1'), ('B2', 'plateB2'), ('XH_y', 'xstartH_y'), ('XH_z', 'xstartH_z')):
    F[tag] = run_summary(hist(f'plates/{r}/history.jsonl'), json.load(open(P / r / 'meta.json')))
for tag, r in (('Hfine_y', 'hevalH_y'), ('Hfine_z', 'hevalH_z')):
    h = hist(f'plates/{r}/history.jsonl')[0]
    F[tag] = dict(C_fine=h['C'], V_fine=h['V'], pcg=h['pcg'], true_residual=h['true_residual'])
law = json.load(open(D / 'homog/homog_cells.json'))
F['homog_law'] = dict(taus=[c['tau'] for c in law['cells']], rho=[c['rho'] for c in law['cells']],
                      C11=[c['CH'][0][0] for c in law['cells']], C12=[c['CH'][0][1] for c in law['cells']], C44=[c['CH'][3][3] for c in law['cells']],
                      symmetry_max={k: max(c['symmetry'][k] for c in law['cells']) for k in law['cells'][0]['symmetry']},
                      equilibrium_residual_max=max(c['equilibrium_residual'] for c in law['cells']),
                      dofs=[c['dofs'] for c in law['cells']], seconds=[c['seconds'] for c in law['cells']])
for tag in ('y', 'z'):
    H = [json.loads(l) for l in open(D / f'homog/H_{tag}/history.jsonl')]
    meta = json.load(open(D / f'homog/H_{tag}/meta.json'))
    F[f'Hmacro_{tag}'] = dict(iterations=len(H), C0_macro=H[0]['C'], C_final_macro=H[-1]['C'], V_rel_final=H[-1]['V_rel'],
                              seconds_per_iteration=float(np.mean([h['seconds'] for h in H])), meta={k: v for k, v in meta.items() if not isinstance(v, (list, dict))})
Bmap = {'y': 'B1', 'z': 'B2'}
F['comparison'] = {}
for tag in ('y', 'z'):
    B, Hf, X, Hm = F[Bmap[tag]], F[f'Hfine_{tag}'], F[f'XH_{tag}'], F[f'Hmacro_{tag}']
    F['comparison'][tag] = dict(homog_prediction_error_initial=(Hm['C0_macro'] - B['C0']) / B['C0'],
                                C_B_final=B['C_final'], C_H_fine=Hf['C_fine'], C_X_final=X['C_final'],
                                H_vs_B=(Hf['C_fine'] - B['C_final']) / B['C_final'], X_vs_H=(X['C_final'] - Hf['C_fine']) / Hf['C_fine'],
                                X_vs_B=(X['C_final'] - B['C_final']) / B['C_final'], V_B=B['V_final'], V_H=Hf['V_fine'], V_X=X['V_final'])
    L = json.load(open(D / f'homog/H_{tag}/final_layout.json'))
    Bm = json.load(open(P / f'plate{Bmap[tag]}' / 'meta.json')); vid = np.array(Bm['vid'])
    cB = np.array(hist(f'plates/plate{Bmap[tag]}/history.jsonl')[-1]['tv'])[vid]
    cH = np.array([[float(x) for x in c['tau_corners']] for c in L['cells']])
    F['comparison'][tag].update(corner_corr=float(np.corrcoef(cB.ravel(), cH.ravel())[0, 1]), corner_rms_diff=float(np.sqrt(((cB - cH) ** 2).mean())))
# ---------------------------------------------------------------- storage (step 6)
F['storage'] = {}
for n in ('fp32store_compare.txt', 'packstore_compare.txt'):
    p = D / 'scale' / n
    if p.exists():
        F['storage'][n] = p.read_text()
(D / 'FACTS_6_11.json').write_text(json.dumps(F, indent=1, default=float))


def pct(x):
    return f'{100 * x:.3g}%'


L = ['# Section 6.11 fact sheet (generated by facts_6_11.py; do not edit by hand)', '']
a = F['A']
L += [f"## Case A (2x2x2, hlat222): NICE run", f"- iterations {a['iterations']} (last k {a['last_k']}), C {a['C0']:.5f} -> {a['C_final']:.5f} (ratio {a['C_ratio']:.4f}), V {a['V0']:.5f} -> {a['V_final']:.5f}, V* {a['Vstar']:.5f}",
      f"- free/fixed vertices {a.get('n_free')}/{a.get('n_fixed')}; final tau range {a['tau_min_final']:.4f}-{a['tau_max_final']:.4f}, span {a['span_max_final']:.4f}, grad {a['grad_max_final']:.4f}",
      f"- per iteration {a['iter_s_mean']:.0f} s mean ({a['iter_s_min']:.0f}-{a['iter_s_max']:.0f}), total {a['iter_s_total']:.0f} s; PCG {min(a['pcg'])}-{max(a['pcg'])}; recomputed residual {a['true_residual_range'][0]:.2e}-{a['true_residual_range'][1]:.2e}; |U^T rho|/C max {a['Ut_rho_rel_absmax']:.2e}",
      f"- GPU peak {a['gpu_peak_gb_max']:.1f} GiB, host peak {a['host_peak_gb_max']:.1f} GiB; body perturbations applied (k, eps, vertices, generation attempts): {[(p['k'], p['eps'], p['n_vertices'], p['attempts']) for p in a['body_perturbations_applied']]}"]
x = F['A_exact_twin']
L += [f"- exact twin: iterations {x['iterations']}, C {x['C0']:.5f} -> {x['C_final']:.5f}, per iteration {x['iter_s_mean']:.0f} s mean"]
fd = F['A_final_designs']
L += [f"- final designs, exact compliance: NICE design {fd['C_exact_of_NICE_design']:.6f}, exact design {fd['C_exact_of_exact_design']:.6f}, rel diff {fd['rel_diff']:.2e}; corner tau max|diff| {fd['tau_maxabs_diff']:.4f}, rms {fd['tau_rms_diff']:.4f}"]
for k, c in F['A_checks'].items():
    L += [f"- exact check k={k}: C_exact {c['C_exact']:.5f}, C_hat {c['C_hat']:.5f}, surrogate err {pct(c['surrogate_err'])}, grad rel err {pct(c['grad_rel_err'])}, cos {c['grad_cos']:.7f}, "
          f"component err / max|g|: median {c['comp_err_rel_to_max']['median']:.2e} p95 {c['comp_err_rel_to_max']['p95']:.2e} max {c['comp_err_rel_to_max']['max']:.2e}, sign {c['sign_agreement']:.3f}, KKT (volume multiplier only) {c['kkt_exact']:.3f}"]
for tag in ('B1', 'B2', 'XH_y', 'XH_z'):
    b = F[tag]
    L += ['', f"## {tag}", f"- iterations {b['iterations']}, C {b['C0']:.3f} -> {b['C_final']:.3f} (ratio {b['C_ratio']:.4f}), V_final {b['V_final']:.5f} (V* {b['Vstar']:.5f}), final tau {b['tau_min_final']:.3f}-{b['tau_max_final']:.3f}, span {b['span_max_final']:.3f}, grad {b['grad_max_final']:.3f}",
          f"- per iteration {b['iter_s_mean']:.0f} s mean ({b['iter_s_min']:.0f}-{b['iter_s_max']:.0f}); phases {({k: round(v) for k, v in b['phase_means'].items()})}; PCG {min(b['pcg'])}-{max(b['pcg'])}; residual {b['true_residual_range'][0]:.1e}-{b['true_residual_range'][1]:.1e}; |U^T rho|/C max {b['Ut_rho_rel_absmax']:.1e}; GPU {b['gpu_peak_gb_max']:.1f} GiB host {b['host_peak_gb_max']:.1f} GiB",
          f"- body perturbations applied (k, eps, vertices, generation attempts) {[(p['k'], p['eps'], p['n_vertices'], p['attempts']) for p in b['body_perturbations_applied']]}"]
L += ['', '## Homogenisation']
h = F['homog_law']
L += [f"- law: {len(h['taus'])} thicknesses {h['taus'][0]}-{h['taus'][-1]}, rho {h['rho'][0]:.4f}-{h['rho'][-1]:.4f}, C11 {h['C11'][0]:.5f}-{h['C11'][-1]:.5f}, C12 {h['C12'][0]:.5f}-{h['C12'][-1]:.5f}, C44 {h['C44'][0]:.5f}-{h['C44'][-1]:.5f}; symmetry max {h['symmetry_max']}; equilibrium residual max {h['equilibrium_residual_max']:.1e}"]
for tag in ('y', 'z'):
    m, c = F[f'Hmacro_{tag}'], F['comparison'][tag]
    L += [f"- {tag}: macro iterations {m['iterations']}, macro C {m['C0_macro']:.3f} -> {m['C_final_macro']:.3f}; homog prediction error at the initial design {pct(c['homog_prediction_error_initial'])}",
          f"  fine-scale (NICE) C: B {c['C_B_final']:.3f} (V {c['V_B']:.4f}), H {c['C_H_fine']:.3f} (V {c['V_H']:.4f}), cross-start {c['C_X_final']:.3f} (V {c['V_X']:.4f}); H vs B {pct(c['H_vs_B'])}, X vs H {pct(c['X_vs_H'])}, X vs B {pct(c['X_vs_B'])}; corner corr B/H {c['corner_corr']:.3f}, rms {c['corner_rms_diff']:.3f}"]
rv = F['route_validation']
L += ['', '## Route validation (hlat222)'] + [f"- slow vs fast k={r['k']}: C {r['C_slow']:.6f}/{r['C_fast']:.6f}, pcg {r['pcg_slow']}/{r['pcg_fast']}, iter s {r['iter_s_slow']:.0f}/{r['iter_s_fast']:.0f}, s_vertex rel {r['s_vertex_rel']:.1e}" for r in rv['slow_vs_fast']]
f2 = rv['fd_vs_ad_k0']
L += [f"- fd vs ad k=0: C {f2['C_fd']:.7f}/{f2['C_ad']:.7f}, s_vertex rel {f2['s_vertex_rel']:.1e}, sens s {f2['sens_s_fd']:.0f}/{f2['sens_s_ad']:.0f}"]
L += [f"- cold vs warm k={r['k']}: C {r['C_cold']:.7f}/{r['C_warm']:.7f}, pcg {r['pcg_cold']}/{r['pcg_warm']}, solve s {r['solve_cold']:.0f}/{r['solve_warm']:.0f}, hit {r['warm_hit']}, scale {r['warm_scale']}" for r in rv['cold_vs_warm']]
L += ['', '## Storage tests (raw)'] + [f'### {k}\n```\n{v}```' for k, v in F['storage'].items()]
(D / 'FACTS_6_11.md').write_text('\n'.join(L) + '\n')
print('\n'.join(L[:60]))
