"""Fact sheet of the plate supported on its cut (Section 5.10): every number used in the text, Table 6, Figures 12b and
13a,b and Supplementary Note S9.3, read from the archived run records in this directory: cplateN (NICE optimisation),
HC_x (homogenised macroscale optimisation), hevalHC (homogenisation design analysed with NICE), check_exact_*.json
(exact condensation).  Writes FACTS_CPLATE.json next to this script.  Usage: python3 facts_cplate.py"""
import json
from pathlib import Path
import numpy as np

D = Path(__file__).resolve().parent
KEYS = ('active', 'faces', 'ports', 'cut_nodes', 'weak', 'el_fringe', 'gp_fringe', 'shift')


def hist(p):
    return [json.loads(l) for l in open(D / p)]


def base(c):
    return c.rsplit('_o', 1)[0]


def check(run, k):
    f = D / run / f'check_exact_{k:03d}.json'
    if not f.exists():
        return None
    c = json.loads(f.read_text())
    return {x: c[x] for x in ('C_exact', 'C_hat', 'surrogate_err', 'grad_rel_err', 'grad_cos', 'comp_err_rel_to_max',
                              'sign_agreement', 'exact_pcg', 'exact_true_residual')} | dict(distinct_cells=c['cpu']['distinct_cells'])


F = {}
H, meta = hist('cplateN/history.jsonl'), json.load(open(D / 'cplateN/meta.json'))
it = [h['times']['iter_s'] for h in H]
Vstar = H[-1]['V'] / H[-1]['V_rel']
sw = []
for a, b in zip(H[:-1], H[1:]):
    fa = {base(c): v for c, v in a['fps'].items()}; fb = {base(c): v for c, v in b['fps'].items()}
    n = {key: sum(fa[c].get(key) != fb[c].get(key) for c in fa if c in fb) for key in KEYS}
    n['any'] = sum(any(fa[c].get(key) != fb[c].get(key) for key in KEYS) for c in fa if c in fb)
    sw.append(dict(k=b['k'], **n))
F['switches'] = {key: [int(np.min([s[key] for s in sw])), float(np.median([s[key] for s in sw])), int(np.max([s[key] for s in sw]))]
                 for key in KEYS + ('any',)}
# geometry-generation fallback: the last record of an iteration is the perturbation of the analysed design; the change of
# a parameter is |tau| |eps| / (1 + eps) unless the vertex was clipped to a bound
pert = []
for h in H:
    if h['body_perturb']:
        last = h['body_perturb'][-1]; tv = np.array(h['tv']); vs = last['vertices']; eps = last['eps']
        clipped = [v for v in vs if tv[v] in (0.18, 0.69)]
        pert.append(dict(k=h['k'], eps_tried=[p['eps'] for p in h['body_perturb']], eps_applied=eps, n_vertices=len(vs),
                         failed_first=h['body_perturb'][0]['failed'], failed_each=[p['failed'] for p in h['body_perturb']],
                         regenerated=len(last['regenerated']), attempts=len(h['body_perturb']) + 1,
                         dtau_max=float(max(abs(tv[v]) * abs(eps) / (1 + eps) for v in vs)), clipped=clipped))
F['NICE'] = dict(iterations=len(H), last_k=H[-1]['k'], C_trace=[h['C'] for h in H], V_rel_trace=[h['V_rel'] for h in H],
                 C0=H[0]['C'], C_final=H[-1]['C'], C_peak=max(h['C'] for h in H), k_peak=int(np.argmax([h['C'] for h in H])),
                 V0=H[0]['V'], Vstar=Vstar, V_rel_final=H[-1]['V_rel'], tau_final=[H[-1]['tau_min'], H[-1]['tau_max']],
                 span_final=H[-1]['span_max'], grad_final=H[-1]['grad_max'], span_max_all=max(h['span_max'] for h in H),
                 grad_max_all=max(h['grad_max'] for h in H), dx_last=[h['dx'] for h in H[-4:]],
                 rel_change_last=[abs(b['C'] - a['C']) / abs(a['C']) for a, b in zip(H[-4:-1], H[-3:])],
                 pcg=[h['pcg'] for h in H], true_residual=[min(h['true_residual'] for h in H), max(h['true_residual'] for h in H)],
                 Ut_rho_rel_absmax=max(abs(h['Ut_rho_rel']) for h in H),
                 iter_s_mean=float(np.mean(it)), iter_s_min=min(it), iter_s_max=max(it), iter_s_total=float(np.sum(it)),
                 k_iter_s_max=int(np.argmax(it)),
                 phase_means={k: float(np.mean([h['times'].get(k, 0.0) for h in H])) for k in ('bodies_s', 'prep_s', 'setup_s', 'precond_s', 'solve_s', 'sens_s', 'mma_s')},
                 gpu_peak_gb=max(h['gpu_peak_gb'] for h in H), host_peak_gb=max(h['host_peak_gb'] for h in H),
                 mma_ymax=[h['mma']['y_max'] for h in H], perturbations=pert,
                 n_free=int(len(meta['fixed']) - sum(meta['fixed'])), n_fixed=int(sum(meta['fixed'])), n_vertices=len(meta['fixed']),
                 cells=meta['cells'], span_pairs=meta['span_pairs'], grad_stencils=meta['grad_stencils'], args=meta['args'])
G = hist('HC_x/history.jsonl'); gm = json.load(open(D / 'HC_x/meta.json'))
F['Hmacro'] = dict(iterations=len(G), C0=G[0]['C'], C_final=G[-1]['C'], C_peak=max(g['C'] for g in G), V_rel_final=G[-1]['V_rel'],
                   dx_last=[g['dx'] for g in G[-4:]], rel_change_last=[abs(b['C'] - a['C']) / a['C'] for a, b in zip(G[-4:-1], G[-3:])],
                   seconds_per_iteration=float(np.mean([g['seconds'] for g in G])), elements=gm['elements'], nodes=gm['nodes'],
                   cut_section=gm['cut_section'], pen=gm['args']['pen'], C_trace=[g['C'] for g in G])
E = hist('hevalHC/history.jsonl')[0]
F['Hfine'] = dict(C=E['C'], V=E['V'], V_over_Vstar=E['V'] / Vstar, pcg=E['pcg'], true_residual=E['true_residual'],
                  Ut_rho_rel=E['Ut_rho_rel'], span=E['span_max'], grad=E['grad_max'], tau=[E['tau_min'], E['tau_max']],
                  iter_s=E['times']['iter_s'], perturbations=E['body_perturb'])
tN, tH = np.array(H[-1]['tv']), np.array(E['tv'])
vid = np.array(meta['vid'])
F['compare'] = dict(H_vs_N_nice=E['C'] / H[-1]['C'] - 1, tau_maxabs=float(np.abs(tN - tH).max()),
                    tau_rms=float(np.sqrt(((tN - tH) ** 2).mean())), corner_corr=float(np.corrcoef(tN[vid].ravel(), tH[vid].ravel())[0, 1]),
                    macro_vs_nice_start=G[0]['C'] / H[0]['C'] - 1)
F['checks'] = {'N0': check('cplateN', 0), f"N{H[-1]['k']}": check('cplateN', H[-1]['k']), 'H': check('hevalHC', 0)}
c0, cN, cH = F['checks']['N0'], F['checks'][f"N{H[-1]['k']}"], F['checks']['H']
if c0:
    F['compare']['macro_vs_exact_start'] = G[0]['C'] / c0['C_exact'] - 1
if cN and cH:
    F['compare']['H_vs_N_exact'] = cH['C_exact'] / cN['C_exact'] - 1
    F['compare']['macro_final_vs_exact_H'] = G[-1]['C'] / cH['C_exact'] - 1
# route check: the 24-cell plate of the scale demonstration (former B1 settings) rerun for three iterations with the four
# implementation options of the timed route of Table 5 (t5val), against the record without them (X6_opt/plates/plateB1)
a = [json.loads(l) for l in open(D.parent / 'X6_opt/plates/plateB1/history.jsonl')][:3]
b = hist('t5val/history.jsonl')
F['route_t5'] = [dict(k=x['k'], C=x['C'], C_rel=abs(y['C'] / x['C'] - 1),
                      s_vertex_rel=float(np.linalg.norm(np.array(y['s_vertex']) - x['s_vertex']) / np.linalg.norm(x['s_vertex'])),
                      pcg=[x['pcg'], y['pcg']], iter_s=[x['times']['iter_s'], y['times']['iter_s']],
                      prep_s=[x['times']['prep_s'], y['times']['prep_s']], solve_s=[x['times']['solve_s'], y['times']['solve_s']],
                      sens_s=[x['times']['sens_s'], y['times']['sens_s']]) for x, y in zip(a, b)]
(D / 'FACTS_CPLATE.json').write_text(json.dumps(F, indent=1, default=float))
print(json.dumps({k: v for k, v in F.items() if k != 'NICE'}, indent=1, default=float)[:4000])
N = F['NICE']; print({k: N[k] for k in N if k not in ('C_trace', 'V_rel_trace', 'pcg', 'args', 'mma_ymax')})
print('pcg', min(N['pcg']), max(N['pcg']), 'ymax', max(N['mma_ymax']))
