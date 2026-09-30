"""Supplementary Note S9 (design optimisation records): Markdown tables ST21-ST25 and the derived numbers quoted in the
text, printed to stdout.  Every value is read from FACTS_6_11.json (written by facts_6_11.py) or computed here from the
archived run records in this directory (history.jsonl, meta.json and the .log file of each run, the homogenisation
records and the plate layout); the MMA constants are read from mma.py, the fallback perturbations from opt_design.py and
the options of the timed run of Table 5 from docs/paper_p1/evidence/d5_off.json.  Statements in the captions that
depend on the records (coinciding designs, cold starts, reused geometry, fallback settings, failing cells) are computed
here as well.  Typed in by hand: labels and wording, and the training limits quoted from Table ST02 (0.47 and
[0.1752, 0.6993]).
Usage: python3 tables_s9.py > tables_s9.md"""
import json
import re
import sys
from pathlib import Path
import numpy as np

D = Path(__file__).resolve().parent
SRC = D.parents[3] / 'data' / 'newmachine_20260924' / 'src_v2_wip'
F = json.loads((D / 'FACTS_6_11.json').read_text())
MINUS = '\u2212'


def hist(p):
    return [json.loads(l) for l in open(D / p)]


def meta(p):
    return json.loads((D / p).read_text())


# ------------------------------------------------------------------------------------------------ number formatting
def fx(x, d=3):
    """Fixed point with thousands separators and a typographic minus."""
    s = f'{x:,.{d}f}'
    return s.replace('-', MINUS)


def sc(x, d=1):
    """Scientific notation as in Table ST17b (e.g. 2.6e-04), typographic minus on the mantissa."""
    s = f'{x:.{d}e}'
    return (MINUS + s[1:]) if s.startswith('-') else s


def pc(x, d=3):
    """Fraction -> percent with d decimals."""
    return fx(100 * x, d)


def rng(a, b, f):
    return f'{f(a)}–{f(b)}'


def table(head, rows):
    out = ['| ' + ' | '.join(head) + ' |', '| ' + ' | '.join('---' for _ in head) + ' |']
    out += ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows]
    return '\n'.join(out)


def base(c):
    return c.rsplit('_o', 1)[0]


def stop_reason(ctrace, dx_last, xtol=1e-3, ftol=1e-4, fixed_budget=None):
    """Stopping rule of opt_design.py / homog_macro.py applied to the recorded objective trace and last step."""
    if fixed_budget:
        return f'fixed budget of {fixed_budget} iterations'
    ch = [abs(ctrace[i] - ctrace[i - 1]) / abs(ctrace[i - 1]) for i in range(1, len(ctrace))]
    if dx_last < xtol:
        return '\\(\\max\\lvert\\Delta\\tau\\rvert<10^{-3}\\)'
    if len(ch) >= 3 and all(c < ftol for c in ch[-3:]):
        return 'objective change \\(<10^{-4}\\) three times in a row'
    return 'maximum iterations'


# ------------------------------------------------------------------------------------------------ archived records
RUN = {'A': 'optA/optA', 'A_exact_twin': 'optA/optAx', 'B1': 'plates/plateB1', 'B2': 'plates/plateB2',
       'XH_y': 'plates/xstartH_y', 'XH_z': 'plates/xstartH_z'}
H = {t: hist(f'{r}/history.jsonl') for t, r in RUN.items()}
M = {t: meta(f'{r}/meta.json') for t, r in RUN.items()}
HE = {t: hist(f'plates/hevalH_{t}/history.jsonl')[0] for t in ('y', 'z')}          # NICE evaluation of the H designs
HM = {t: hist(f'homog/H_{t}/history.jsonl') for t in ('y', 'z')}                   # macroscale MMA runs
HMmeta = {t: meta(f'homog/H_{t}/meta.json') for t in ('y', 'z')}
LAYOUT = meta('homog/plate841.json')
KIND = {c['case']: (c['kind'], c['retained']) for c in LAYOUT['cells']}
sys.path.insert(0, str(SRC))
import mma as MMA                                                                  # noqa: E402  (constants only)
MMA_SRC = (SRC / 'mma.py').read_text()
for pat in (r'c = 1000\.0 \* np\.ones\(m\)', r'd = np\.ones\(m\)', r'a = np\.zeros\(m\)', r'a0=1\.0'):
    assert re.search(pat, MMA_SRC), pat                                             # MMA problem constants as stated
ARGS = {t: M[t]['args'] for t in M}
VSTAR_B = F['B1']['Vstar']
P = []                                                                              # printed blocks
NAMES = {'A': 'A, NICE', 'A_exact_twin': 'A, exact twin', 'B1': 'B1', 'B2': 'B2', 'XH_y': 'X-\\(y\\)', 'XH_z': 'X-\\(z\\)'}
LOG = {t: f'{r.rsplit("/", 1)[0]}/{r.rsplit("/", 1)[1]}.log' for t, r in RUN.items()}


def log_events(t):
    """JSON event lines of a run log (tracebacks and warnings skipped); each event gets the design iteration 'k_run' at
    which the process stood (0 at the first start, the RESUME iteration after a resume)."""
    ev, k = [], 0
    for line in open(D / LOG[t]):
        if not line.startswith('{'):
            continue
        e = json.loads(line)
        if e['event'] == 'RESUME':
            k = e['k']
        elif e['event'] == 'ITER':
            k = e['k'] + 1
        e['k_run'] = k
        ev.append(e)
    return ev


EV = {t: log_events(t) for t in RUN}
OD_SRC = (SRC / 'opt_design.py').read_text()
FB_EPS = [float(x) for x in re.search(r'eps = \[([^\]]+)\]\[attempt\]', OD_SRC).group(1).split(',')]   # ST21 fallback
N_FB = len(FB_EPS)


def fallback_from(t):
    """First design iteration from which the vertex fallback of ST21 was enabled (a START with --body-retry equal to the
    length of its perturbation list), or None.  Checked against the events: no cell-local BODY_RETRY after it, and every
    BODY_PERTURB after it."""
    starts = [e for e in EV[t] if e['event'] == 'START']
    on = [e for e in starts if e['args'].get('body_retry') == N_FB]
    if not on:
        return None
    i0 = EV[t].index(on[0])
    assert all(e['event'] != 'BODY_RETRY' for e in EV[t][i0:]), t
    assert all(EV[t].index(e) > i0 for e in EV[t] if e['event'] == 'BODY_PERTURB'), t
    return on[0]['k_run']


def cell_local(t):
    """Earlier cell-local retries (BODY_RETRY events: the failing cell's own corners scaled by 1 + eps) per iteration."""
    out = {}
    for e in EV[t]:
        if e['event'] == 'BODY_RETRY':
            k = int(e['case'].rsplit('_o', 1)[1])
            assert not e['ok'], e
            out.setdefault(k, []).append((base(e['case']), e['eps']))
    return out


def eps_s(e, d=0):
    return ('+' if e > 0 else MINUS) + sc(abs(e), d)


def tex_eps(e):
    """1e-4 -> 10^{-4}, -3e-3 -> -3\\times10^{-3} (LaTeX)."""
    ex = int(np.floor(np.log10(abs(e)) + 1e-12)); m = abs(e) / 10 ** ex
    return ('-' if e < 0 else '') + ('' if np.isclose(m, 1) else f'{m:g}\\times') + f'10^{{{ex}}}'


def its(ks):
    return ('iteration ' if len(ks) == 1 else 'iterations ') + klist(ks)


def klist(ks):
    ks = [str(k) for k in ks]
    return ks[0] if len(ks) == 1 else ', '.join(ks[:-1]) + ' and ' + ks[-1]


RUNNAME = {'A': 'case A', 'A_exact_twin': 'the exact twin', 'B1': 'B1', 'B2': 'B2', 'XH_y': 'X-\\(y\\)', 'XH_z': 'X-\\(z\\)'}


def by_iter(d):
    """{run: iteration} -> 'iteration 16 of case A and iteration 5 of B1 and B2' (runs with equal iterations grouped)."""
    grp = {}
    for t, k in d.items():
        grp.setdefault(k, []).append(RUNNAME[t])
    return klist([f'iteration {k} of {klist(v)}' for k, v in grp.items()])


COLD = {t: [h['k'] for h in H[t] if 'warm_hit' not in h['times']] for t in RUN}            # PCG started from zero
RESUMED = {t: sorted({e['k'] for e in EV[t] if e['event'] == 'RESUME'}) for t in RUN}
GEOM_REUSED = {t: [h['k'] for h in H[t] if h['times']['bodies_s'] < 1.0] for t in RUN}     # geometry generated beforehand
GEOM_PART = {t: [h['k'] for h in H[t] if h['k'] in RESUMED[t] and h['k'] not in GEOM_REUSED[t]
                 and h['bodies_new'] < len(h['fps'])] for t in RUN}                         # partly generated before a resume
for t in RUN:
    if t != 'A_exact_twin':                                                        # the twin has no warm start at all
        assert set(COLD[t]) - {0} <= set(RESUMED[t]), (t, COLD[t], RESUMED[t])


def out(s=''):
    P.append(s)


# ================================================================================================ ST21 settings
a, b = ARGS['A'], ARGS['B1']
move_abs = a['move'] * (a['tmax'] - a['tmin'])


def span_pairs_free(t):
    """Ordered corner pairs of all cells (as opt_design.build_constraint_structure) and those with a free vertex (the
    only ones opt_design.constraints passes to MMA; pairs of two fixed vertices are constant and dropped)."""
    fixed, pairs = M[t]['fixed'], set()
    for row in M[t]['vid']:
        pairs |= {(row[i], row[j]) for i in range(8) for j in range(8) if i != j and row[i] != row[j]}
    assert len(pairs) == M[t]['span_pairs']
    return len(pairs), sum(not (fixed[p] and fixed[q]) for p, q in pairs)


def stencils_free(t):
    fixed, st = M[t]['fixed'], set()
    for row in M[t]['vid']:
        st |= {(row[i], row[i ^ 4], row[i ^ 2], row[i ^ 1]) for i in range(8)}
    assert len(st) == M[t]['grad_stencils']
    return len(st), sum(not all(fixed[v] for v in s) for s in st)


SPA, SPB = span_pairs_free('A'), span_pairs_free('B1')
GSA, GSB = stencils_free('A'), stencils_free('B1')
assert GSA[0] == GSA[1] and GSB[0] == GSB[1]                                        # every stencil has a free vertex
# interior-point loop of mma.subsolv: barrier parameter from 1 down by 0.1 to EPSIMIN, Newton steps to 0.9 of it, <= 200
m_ip = re.search(r'epsi = ([0-9.]+)\n', MMA_SRC); m_in = re.search(r'resmax > ([0-9.]+) \* epsi and ittt < ([0-9]+)', MMA_SRC)
assert m_ip and m_in and re.search(r'epsi \*= 0\.1', MMA_SRC)
# options of the timed run of Table 5 (Table ST17d, 2x2x2) that the optimisation runs did not use
D5 = json.loads((D.parents[2] / 'evidence' / 'd5_off.json').read_text())['args']
OPTS = [('tet_triton', 'OPL_TET_TRITON', 'fused per-tetrahedron moment kernels'),
        ('coarse_tpl', 'OPL_COARSE_ELEM', 'template-based coarse Galerkin matrices'),
        ('sparse_coarse', 'OPL_COARSE_SPARSE', 'sparse coarse space'),
        ('fastidx', 'OPL_FASTIDX', 'element gathers')]
for key, env, _ in OPTS:
    assert D5[key] is True, key
    for t in RUN:
        if t != 'A_exact_twin':
            assert env not in M[t]['env'], (t, env)
assert 'FI.ON' not in OD_SRC                                                         # opt_design never enables fastidx
assert D5['sens_obj'] == 'sum' and D5['sens'] == 'ad'                               # one reverse pass, summed compliance of 3 loads
OPT_LIST = ', '.join(o[2] for o in OPTS)
FBF = {t: fallback_from(t) for t in RUN}
fb_late = [t for t in RUN if FBF[t]]
assert all(FBF[t] == 0 for t in RUN if t not in fb_late)
cold_extra = {t: [k for k in COLD[t] if k] for t in RUN if t != 'A_exact_twin'}
assert all(len(v) <= 1 for v in cold_extra.values())
cold_txt = by_iter({t: v[0] for t, v in cold_extra.items() if v})
out('### Table ST21. Optimiser, constraints and analysis settings')
out()
rows = [
    ('Design variables', f"Corner thickness parameters at the lattice vertices \\(\\boldsymbol\\tau_g\\), each shared by the cells "
     f"meeting there; \\(\\boldsymbol\\tau_m=\\boldsymbol\\tau_m(\\boldsymbol\\tau_g)\\) copies the vertex values to the corners of cell \\(m\\). "
     f"Case A: {M['A']['vertices']} vertices, {M['A']['free']} free; plates: {M['B1']['vertices']} vertices, {M['B1']['free']} free"),
    ('Fixed vertices', f"The vertices in the plane of the loaded face keep their initial values ({sum(M['A']['fixed'])} in case A, "
     f"{sum(M['B1']['fixed'])} in the plates), so that the thickness field on that face, the material part of the face and its consistent nodal load do not change with the design (Appendix H: no load-derivative term)"),
    ('Objective', 'Compliance \\(\\widehat C=f_g^T\\bar U\\) under one consistent face traction of unit resultant, divided by its value at the first iteration'),
    ('Gradient', 'Field-based estimate \\(\\sum_m(\\partial\\boldsymbol\\tau_m/\\partial\\boldsymbol\\tau_g)^T\\widetilde{\\boldsymbol s}_m\\), '
     '\\(\\widetilde s_c\\) from Eq. (13) of every cell; reverse-mode differentiation of the moment integrals at the fixed recovered field. '
     'The complete surrogate derivative of Eq. (14) is not used. Exact twin: exact sensitivities from the exact field with the moment derivatives of Eq. (H.6)'),
    ('Volume constraint', f"\\(V/V^*-1\\le0\\); \\(V\\) = sum of the zeroth element moments of all cells (material volume of the discrete model), "
     f"its derivative from the same reverse pass (exact twin: Eq. (H.6)). \\(V^*={a['vfrac']}\\,V(\\boldsymbol\\tau^0)\\): "
     f"{fx(F['A']['Vstar'], 5)} (case A), {fx(VSTAR_B, 5)} (plates); cross-starts: the same absolute \\(V^*={fx(ARGS['XH_y']['vstar'], 5)}\\)"),
    ('Corner-span constraint', f"\\((\\tau_a-\\tau_b)/{a['span']}-1\\le0\\) for every ordered pair of distinct corner vertices of every cell "
     f"({SPA[0]} ordered pairs, {SPA[1]} of them involving a free vertex, in case A; {SPB[0]} and {SPB[1]} in the plates); pairs of two fixed vertices "
     "are constant and are not passed to the optimiser; training limit 0.47 (Table ST02)"),
    ('Gradient-norm constraint', f"\\(\\lvert\\nabla\\tau\\rvert^2/{a['grad']}^2-1\\le0\\) at every corner of every cell, from the three edge differences "
     f"({GSA[1]} corner stencils in case A, {GSB[1]} in the plates, each with a free vertex); for a trilinear field the largest "
     f"gradient norm over the cell is attained at a corner; training limit 0.47"),
    ('Bounds', f"\\({a['tmin']}\\le\\tau\\le{a['tmax']}\\); training range [0.1752, 0.6993] (Table ST02)"),
    ('Optimiser', f"Method of moving asymptotes, one update per analysis, no line search and no conservativeness test (no GCMMA inner loop); "
     f"subproblem solved by a primal–dual interior-point method, the barrier parameter reduced tenfold from {float(m_ip.group(1)):g} to "
     f"\\(10^{{{int(np.log10(MMA.EPSIMIN))}}}\\) (at each value, Newton steps until the largest residual is below {m_in.group(1)} of it, "
     f"at most {m_in.group(2)}); \\(a_0=1\\), \\(a_i=0\\), \\(c_i=1000\\), \\(d_i=1\\)"),
    ('Asymptotes', f"Initially \\(x\\pm{MMA.ASYINIT}(x_{{\\max}}-x_{{\\min}})\\) (first two iterations), then widened by {MMA.ASYINCR} "
     f"where consecutive steps have the same sign and narrowed by {MMA.ASYDECR} where they alternate, kept between 0.01 and 10 variable ranges from \\(x\\); "
     f"subproblem bounds at {MMA.ALBEFA} of the distance to the asymptotes"),
    ('Move limit', f"{a['move']} of the variable range, {fx(move_abs, 4)} per iteration"),
    ('Stopping rule', f"\\(\\max\\lvert\\Delta\\tau\\rvert<10^{{{int(np.log10(a['xtol']))}}}\\), or relative objective change below "
     f"\\(10^{{{int(np.log10(a['ftol']))}}}\\) in three consecutive iterations, or {a['maxit']} iterations; cross-starts: {ARGS['XH_y']['stop_after']} iterations"),
    ('Geometry', 'Every iteration regenerates the geometry of every cell from its current corner parameters with the geometry generator used for all cells of this study, '
     f"in parallel processes on the host ({a['workers']} for case A, {b['workers']} for the plates), and rebuilds every learned operator"),
    ('Lattice solve', f"Preconditioned conjugate gradients with the balanced two-level preconditioner of Supplementary Note S6.1 to a recursive relative residual of "
     f"\\(10^{{{int(np.log10(a['tol']))}}}\\) (at most {fx(a['maxit_pcg'], 0)} iterations); network and correction in single precision, stiffness actions of the condensed "
     f"product in double precision: the arithmetic and preconditioner of the timed route of Table 5, but without its implementation options ({OPT_LIST}; "
     f"record `evidence/d5_off.json`), so that the phase times are not comparable with Table ST17d. Exact twin: dense exact condensed matrices of every cell, "
     f"assembled solve to \\(10^{{{int(np.log10(ARGS['A_exact_twin']['exact_tol']))}}}\\)"),
    ('Warm start', 'From the previous iteration\'s solution, matched coordinate by coordinate on absolute grid position, displacement component and private cut-band flag; '
     'unmatched coordinates start at zero; the start is scaled by the energy-optimal factor \\(f_g^TX_0/(X_0^T\\widehat{\\mathbb K}X_0)\\); '
     f'the stopping criterion, relative to \\(\\lVert f_g\\rVert\\), is unchanged. At iteration 0 of every run, and at {cold_txt}, the solve started from zero. '
     'Exact twin: cold start'),
    ('Operator placement', f"Case A: all cells on the GPU; plates: cells kept on the GPU while its allocated memory stayed below {b['resident_gb']:.0f} GiB, "
     'the others streamed from host memory as in Supplementary Note S5'),
    ('Geometry-generation fallback', 'If generation fails for some cells, the free vertices of those cells are multiplied by \\(1+\\epsilon\\), '
     f"\\(\\epsilon={','.join(tex_eps(e) for e in FB_EPS)}\\) "
     'in turn (clipped to the bounds), every cell sharing them is regenerated, and the perturbed design is analysed and continued from. '
     f"In use from the start in {klist([RUNNAME[t] for t in RUN if t not in fb_late])}, and from {by_iter({t: FBF[t] for t in fb_late})}, "
     "where an earlier, cell-local scheme had been tried first (Table ST24a)"),
]
out(table(['Item', 'Setting'], rows))
out()
ALLH = list(H.values()) + [[HE['y']], [HE['z']]]
smax = max(h['span_max'] for Hr in ALLH for h in Hr); gmax = max(h['grad_max'] for Hr in ALLH for h in Hr)
tmin_ = min(h['tau_min'] for Hr in ALLH for h in Hr); tmax_ = max(h['tau_max'] for Hr in ALLH for h in Hr)
out(f"Over every design analysed on the fine scale (all iterations of all runs of Tables ST22 and ST23), the largest corner span is {smax:.4f}, the largest "
    f"gradient norm {gmax:.4f} and the corner parameters lie in [{tmin_:.4f}, {tmax_:.4f}]. "
    f"Records in `docs/paper_p1/review_r1/results/X6_opt`: `meta.json` of each run (arguments at the start of the run) and the run logs "
    f"{', '.join('`' + LOG[t] + '`' for t in fb_late)} (arguments from {by_iter({t: FBF[t] for t in fb_late})} onward); "
    "constants of `mma.py`; perturbations of the fallback in `opt_design.py`.")
out()

# ================================================================================================ ST22 case A
A, X = F['A'], F['A_exact_twin']
out('### Table ST22. Case A: NICE optimisation, exact twin and exact checks')
out()
out('#### ST22a. Iteration history')
out()
pert_k = {p['k'] for p in A['body_perturbations_applied']}
rows = []
for i, h in enumerate(H['A']):
    tw = F['A_twin_trace'][i]['C_exact_twin'] if i < len(F['A_twin_trace']) else None
    twv = X['V_rel_trace'][i] if i < len(X['V_rel_trace']) else None
    rows.append((f"{h['k']}{'*' if h['k'] in pert_k else ''}", fx(h['C'], 5), fx(h['V_rel'], 5), h['pcg'], sc(h['true_residual']),
                 sc(h['Ut_rho_rel']), fx(h['times']['iter_s'], 1), '—' if tw is None else fx(tw, 5),
                 '—' if twv is None else fx(twv, 5), '—' if tw is None else pc((h['C'] - tw) / tw, 3)))
out(table(['Iteration', '\\(\\widehat C\\) (NICE)', '\\(V/V^*\\)', 'PCG iterations', 'Recomputed residual', '\\(\\bar U^T\\rho/\\widehat C\\)',
           'Time (s)', '\\(C\\), exact twin', '\\(V/V^*\\), exact twin', 'NICE vs twin (%)'], rows))
out()
# NICE and twin designs per iteration: where do they coincide, and why
TVA = [np.asarray(h['tv']) for h in H['A']]; TVX = [np.asarray(h['tv']) for h in H['A_exact_twin']]
DTV = [float(np.abs(p_ - q_).max()) for p_, q_ in zip(TVA, TVX)]
SAME = [k for k, d in enumerate(DTV) if d < 1e-12]
assert SAME == list(range(len(SAME))) and len(SAME) > 1                            # 0 .. k_same, contiguous
FREE_A = ~np.asarray(M['A']['fixed'])
for k in SAME[1:]:                                                                  # steps k-1 -> k into the common design
    for tv_ in (TVA, TVX):
        d0, d1 = tv_[k - 1][FREE_A], tv_[k][FREE_A]
        lim = np.isclose(np.abs(d1 - d0), move_abs, atol=1e-6) | np.isclose(d1, a['tmin'], atol=1e-6) | np.isclose(d1, a['tmax'], atol=1e-6)
        assert lim.all(), k                                                         # every free parameter at the move limit or a bound
    assert H['A'][k - 1]['V_rel'] > 1                                               # volume bound violated before the step
k_div = SAME[-1] + 1
out(f"\\* Design perturbed by the geometry-generation fallback (Table ST24a). NICE vs twin: \\((\\widehat C-C_{{\\rm twin}})/C_{{\\rm twin}}\\) at the same iteration index. "
    f"The designs coincide (to {sc(max(DTV[:k_div]), 0)}) at iterations 0–{SAME[-1]}, where every free parameter moves by the move limit or to the lower bound while the "
    f"volume bound is violated; there the column is the surrogate error of the same design. From iteration {k_div} the corner parameters differ, by "
    f"{sc(DTV[k_div])} at iteration {k_div} and by at most {sc(max(DTV[k_div:]))} over the remaining iterations. "
    f"The solve started from zero at {its(COLD['A'])}. "
    "Recomputed residual: \\(\\lVert f_g-\\widehat{\\mathbb K}\\bar U\\rVert/\\lVert f_g\\rVert\\); time: complete design iteration including geometry generation, "
    f"except at {its(GEOM_REUSED['A'])}, whose geometry was generated beforehand.")
out()
out('#### ST22b. Exact checks of the NICE run')
out()
rows = []
for k in ('0', '12', '23'):
    c = F['A_checks'][k]
    e = c['comp_err_rel_to_max']
    rows.append((k, fx(c['C_exact'], 5), fx(c['C_hat'], 5), pc(c['surrogate_err'], 4), pc(c['grad_rel_err'], 3), f"{c['grad_cos']:.7f}",
                 f"{sc(e['median'])} / {sc(e['p95'])} / {sc(e['max'])}", f"{c['sign_agreement']:.3f}", f"{c['kkt_exact']:.3f}",
                 f"{c['exact_pcg']} / {sc(c['exact_true_residual'])}"))
out(table(['Iteration', 'Exact \\(C\\)', '\\(\\widehat C\\)', 'Surrogate error (%)', 'Gradient error (%)', 'Cosine',
           'Component error / \\(\\max\\lvert g\\rvert\\): median / 95th percentile / max', 'Sign agreement', 'KKT residual, volume multiplier only',
           'Exact solve: PCG iterations / recomputed residual'], rows))
out()
out(f"Over the {M['A']['free']} free vertex parameters. Surrogate error \\((\\widehat C-C)/C\\); gradient error \\(\\lVert\\widetilde s_g-s_g\\rVert/\\lVert s_g\\rVert\\); "
    "component error \\(\\lvert\\widetilde s_{g,i}-s_{g,i}\\rvert/\\max_j\\lvert s_{g,j}\\rvert\\); KKT residual \\(\\lVert s_g+\\lambda\\nabla V\\rVert/\\lVert s_g\\rVert\\) "
    "over the parameters strictly inside the bounds, with the volume multiplier \\(\\lambda\\) fitted by least squares and no other constraint.")
out()
out('#### ST22c. Final designs and cost')
out()
fd = F['A_final_designs']
tw_ph = {k: float(np.mean([h['times'].get(k, 0.0) for h in H['A_exact_twin']])) for k in ('make_T_s', 'setup_s', 'solve_s', 'sens_s', 'bodies_s', 'mma_s')}
ni_ph = A['phase_means']
ni_setup = float(np.mean([h['times'].get('setup_s', 0.0) for h in H['A']]))
rows = [
    ('Iterations; stopping rule', f"{A['iterations']}; {stop_reason(A['C_trace'], A['dx_final'])}", f"{X['iterations']}; {stop_reason(X['C_trace'], X['dx_final'])}"),
    ('Compliance of the run, first → last iteration', f"{fx(A['C0'], 5)} → {fx(A['C_final'], 5)} (\\(\\widehat C\\))", f"{fx(X['C0'], 5)} → {fx(X['C_final'], 5)} (\\(C\\))"),
    ('Exact compliance of the final design', fx(fd['C_exact_of_NICE_design'], 6), fx(fd['C_exact_of_exact_design'], 6)),
    ('Final \\(V/V^*\\)', fx(A['V_rel_final'], 6), fx(X['V_rel_final'], 6)),
    ('Final \\(\\tau\\) range; largest corner span; largest gradient norm', f"{fx(A['tau_min_final'], 4)}–{fx(A['tau_max_final'], 4)}; {fx(A['span_max_final'], 4)}; {fx(A['grad_max_final'], 4)}",
     f"{fx(X['tau_min_final'], 4)}–{fx(X['tau_max_final'], 4)}; {fx(X['span_max_final'], 4)}; {fx(X['grad_max_final'], 4)}"),
    ('PCG iterations; recomputed residual', f"{min(A['pcg'])}–{max(A['pcg'])}; {rng(*A['true_residual_range'], sc)}", f"{min(X['pcg'])}–{max(X['pcg'])}; {rng(*X['true_residual_range'], sc)}"),
    ('Time per iteration, mean (range) (s); sum over all iterations (s)', f"{fx(A['iter_s_mean'], 0)} ({fx(A['iter_s_min'], 0)}–{fx(A['iter_s_max'], 0)}); {fx(A['iter_s_total'], 0)}",
     f"{fx(X['iter_s_mean'], 0)} ({fx(X['iter_s_min'], 0)}–{fx(X['iter_s_max'], 0)}); {fx(X['iter_s_total'], 0)}"),
    ('Mean phases (s)', f"geometry {fx(ni_ph['bodies_s'], 1)}, front end {fx(ni_ph['prep_s'], 1)}, lattice and \\(K_{{PP}}\\) assembly {fx(ni_setup, 1)}, "
     f"preconditioner {fx(ni_ph['precond_s'], 1)}, PCG {fx(ni_ph['solve_s'], 1)}, sensitivities and volume gradient {fx(ni_ph['sens_s'], 1)}, MMA {fx(ni_ph['mma_s'], 3)}",
     f"geometry {fx(tw_ph['bodies_s'], 1)}, cell setup with moment derivatives {fx(tw_ph['setup_s'], 1)}, dense exact condensation {fx(tw_ph['make_T_s'], 1)}, "
     f"PCG {fx(tw_ph['solve_s'], 1)}, exact sensitivities {fx(tw_ph['sens_s'], 1)}, MMA {fx(tw_ph['mma_s'], 3)}"),
    ('Peak memory, GPU / host (GiB)', f"{fx(A['gpu_peak_gb_max'], 1)} / {fx(A['host_peak_gb_max'], 1)}", f"{fx(X['gpu_peak_gb_max'], 1)} / {fx(X['host_peak_gb_max'], 1)}"),
]
out(table(['', 'NICE run', 'Exact twin'], rows))
out()
out(f"Final designs: exact compliance of the NICE design relative to that of the twin's design {sc(fd['rel_diff'], 2)}; corner parameters differ by at most "
    f"{fx(fd['tau_maxabs_diff'], 4)} (root mean square {fx(fd['tau_rms_diff'], 4)}). Mean phases over all iterations; the geometry phase is zero at "
    f"{its(GEOM_REUSED['A'])} of the NICE run and at {its(GEOM_REUSED['A_exact_twin'])} of the twin, where the geometry had been generated beforehand. "
    "The phases of the NICE run are not comparable with Table ST17d, whose timed run used implementation options that these runs did not use (Table ST21).")
out()

# ================================================================================================ ST23 plates and homogenisation
out('### Table ST23. Plates: NICE optimisation, homogenisation designs and cross-starts')
out()
out('#### ST23a. Runs')
out()
cols = ['B1', 'B2', 'H_y', 'H_z', 'XH_y', 'XH_z']


def col_nice(t, budget=None):
    r = F[t]
    ph = r['phase_means']
    return dict(
        start='uniform 0.40' if t in ('B1', 'B2') else f"Hom-\\({t[-1]}\\) design",
        it=f"{r['iterations']}; {stop_reason(r['C_trace'], r['dx_final'], fixed_budget=budget)}",
        C=f"{fx(r['C0'], 3)} → {fx(r['C_final'], 3)}", ratio=f"{r['C_ratio']:.4f}", Cfine=fx(r['C_final'], 3),
        V=fx(r['V_final'] / VSTAR_B, 5), tau=f"{fx(r['tau_min_final'], 3)}–{fx(r['tau_max_final'], 3)}",
        sg=f"{fx(r['span_max_final'], 3)} / {fx(r['grad_max_final'], 3)}", dx=fx(r['dx_final'], 4),
        time=f"{fx(r['iter_s_mean'], 0)} ({fx(r['iter_s_min'], 0)}–{fx(r['iter_s_max'], 0)})", total=fx(r['iter_s_total'], 0),
        ph=f"{fx(ph['bodies_s'], 0)} / {fx(ph['prep_s'], 0)} / {fx(ph['precond_s'], 0)} / {fx(ph['solve_s'], 0)} / {fx(ph['sens_s'], 0)}",
        pcg=f"{min(r['pcg'])}–{max(r['pcg'])}", res=rng(*r['true_residual_range'], sc), urho=sc(r['Ut_rho_rel_absmax']),
        mem=f"{fx(r['gpu_peak_gb_max'], 1)} / {fx(r['host_peak_gb_max'], 1)}",
        pert=', '.join(f"iteration {p['k']}" for p in r['body_perturbations_applied']) or 'none')


def col_h(t):
    m, e, hm = F[f'Hmacro_{t}'], HE[t], HM[t]
    ch = [h['C'] for h in hm]
    tt = e['times']
    return dict(
        start='uniform 0.40 (macroscale)', it=f"{m['iterations']} (macroscale); {stop_reason(ch, hm[-1]['dx'])}",
        C=f"{fx(m['C0_macro'], 3)} → {fx(m['C_final_macro'], 3)} (macroscale)", ratio=f"{m['C_final_macro'] / m['C0_macro']:.4f} (macroscale)",
        Cfine=fx(F[f'Hfine_{t}']['C_fine'], 3), V=fx(F[f'Hfine_{t}']['V_fine'] / VSTAR_B, 5),
        tau=f"{fx(e['tau_min'], 3)}–{fx(e['tau_max'], 3)}", sg=f"{fx(e['span_max'], 3)} / {fx(e['grad_max'], 3)}", dx=f"{fx(hm[-1]['dx'], 4)} (macroscale)",
        time=f"fine-scale evaluation {fx(tt['iter_s'], 0)}", total='—',
        ph=f"{fx(tt['bodies_s'], 0)} / {fx(tt['prep_s'], 0)} / {fx(tt['precond_s'], 0)} / {fx(tt['solve_s'], 0)} / {fx(tt['sens_s'], 0)} (evaluation)",
        pcg=f"{e['pcg']} (evaluation)", res=f"{sc(e['true_residual'])} (evaluation)", urho=sc(abs(e['Ut_rho_rel'])),
        mem=f"{fx(e['gpu_peak_gb'], 1)} / {fx(e['host_peak_gb'], 1)} (evaluation)", pert='none')


C = {'B1': col_nice('B1'), 'B2': col_nice('B2'), 'H_y': col_h('y'), 'H_z': col_h('z'),
     'XH_y': col_nice('XH_y', ARGS['XH_y']['stop_after']), 'XH_z': col_nice('XH_z', ARGS['XH_z']['stop_after'])}
labels = [('start', 'Start design'), ('it', 'Iterations; stopping rule'), ('C', 'Compliance, first → last iteration'),
          ('ratio', 'Last / first'), ('Cfine', 'Fine-scale NICE compliance of the final design'), ('V', 'Final fine-scale \\(V/V^*\\)'),
          ('tau', 'Final \\(\\tau\\) range'), ('sg', 'Largest corner span / gradient norm at the end'), ('dx', 'Last \\(\\max\\lvert\\Delta\\tau\\rvert\\)'),
          ('time', 'Time per iteration, mean (range) (s)'), ('total', 'Sum of iteration times (s)'),
          ('ph', 'Mean phases (s): geometry / front end / preconditioner / PCG / sensitivities'), ('pcg', 'PCG iterations'),
          ('res', 'Recomputed residual'), ('urho', '\\(\\max\\lvert\\bar U^T\\rho\\rvert/\\widehat C\\)'), ('mem', 'Peak memory, GPU / host (GiB)'),
          ('pert', 'Geometry-generation fallback applied')]
out(table([''] + ['B1', 'B2', 'Hom-\\(y\\)', 'Hom-\\(z\\)', 'X-\\(y\\)', 'X-\\(z\\)'], [[lab] + [C[c][key] for c in cols] for key, lab in labels]))
out()
assert all(GEOM_REUSED[t] == [0] for t in ('B1', 'B2', 'XH_y', 'XH_z'))
out(f"B1, B2: NICE optimisation from the uniform design, in-plane (B1) and out-of-plane (B2) load. Hom-\\(y\\), Hom-\\(z\\): optimisation of the homogenised macroscale model "
    f"for the same loads ({HMmeta['y']['elements']:,} Q1 elements, {HMmeta['y']['nodes']:,} nodes), final design evaluated once with NICE on the fine scale. "
    f"X-\\(y\\), X-\\(z\\): NICE optimisation continued from Hom-\\(y\\), Hom-\\(z\\) under the absolute \\(V^*={fx(VSTAR_B, 5)}\\) of B1 and B2. "
    "\\(V/V^*\\) relative to that bound. Mean phases over all iterations; the geometry phase is zero at iteration 0, whose geometry was generated beforehand. "
    "Every fine-scale value is a NICE value; the exact verification is pending (S9.6).")
out()
out('#### ST23b. Homogenised law of the uniform-thickness cell')
out()
L = F['homog_law']
rows = [(f"{t:.2f}", f"{r:.4f}", f"{c11:.5f}", f"{c12:.5f}", f"{c44:.5f}", f"{n:,}") for t, r, c11, c12, c44, n in
        zip(L['taus'], L['rho'], L['C11'], L['C12'], L['C44'], L['dofs'])]
out(table(['\\(\\tau\\)', '\\(\\rho\\)', '\\(C^H_{11}\\)', '\\(C^H_{12}\\)', '\\(C^H_{44}\\)', 'DOFs'], rows))
out()
s = L['symmetry_max']
out(f"\\(E_Y=1\\), \\(\\nu=0.3\\), unit cell of volume 1, Voigt notation with engineering shear strains; \\(C^H_{{11}}\\), \\(C^H_{{12}}\\), \\(C^H_{{44}}\\) are the entry values of the computed tensor "
    f"(the macroscale model averages the three symmetric entries of each). Largest relative spreads over the twelve thicknesses: \\(C^H_{{11}},C^H_{{22}},C^H_{{33}}\\) {sc(s['C11_spread'])}, "
    f"\\(C^H_{{12}},C^H_{{13}},C^H_{{23}}\\) {sc(s['C12_spread'])}, \\(C^H_{{44}},C^H_{{55}},C^H_{{66}}\\) {sc(s['C44_spread'])}; largest normal–shear or shear–shear coupling "
    f"{sc(s['coupling_max'])} of \\(C^H_{{11}}\\); largest relative equilibrium residual of the periodic fluctuations {sc(L['equilibrium_residual_max'])}.")
out()
out('#### ST23c. Homogenisation against the fine-scale NICE optimisation')
out()
cmp_ = F['comparison']
V0_macro = {t: HM[t][0]['V'] for t in ('y', 'z')}
rows = []
for t, bb in (('y', 'B1'), ('z', 'B2')):
    c = cmp_[t]
    rows.append((f"\\({t}\\) ({bb})", f"{fx(F[f'Hmacro_{t}']['C0_macro'], 3)} / {fx(F[bb]['C0'], 3)}", pc(c['homog_prediction_error_initial'], 1),
                 f"{fx(c['C_B_final'], 3)} / {fx(c['C_H_fine'], 3)} / {fx(c['C_X_final'], 3)}",
                 f"{fx(c['V_B'] / VSTAR_B, 5)} / {fx(c['V_H'] / VSTAR_B, 5)} / {fx(c['V_X'] / VSTAR_B, 5)}",
                 pc(c['H_vs_B'], 2), pc(c['X_vs_H'], 2), pc(c['X_vs_B'], 3), f"{c['corner_corr']:.3f}", fx(c['corner_rms_diff'], 3)))
out(table(['Load', 'Initial design: macroscale / NICE compliance', 'Macroscale prediction error (%)', 'Final fine-scale NICE compliance: B / Hom / X',
           'Final \\(V/V^*\\): B / Hom / X', 'Hom vs B (%)', 'X vs Hom (%)', 'X vs B (%)', 'Corner correlation, B and Hom', 'Corner RMS difference, B and Hom'], rows))
out()
dv0 = max(abs(V0_macro[t] - F['B1']['V0']) / F['B1']['V0'] for t in ('y', 'z'))
out(f"Prediction error: \\((C_{{\\rm macro}}-\\widehat C)/\\widehat C\\) at the uniform design \\(\\tau=0.40\\). B: B1 or B2; Hom: Hom-\\(y\\) or Hom-\\(z\\); X: X-\\(y\\) or X-\\(z\\). Relative differences of the final compliances as stated, e.g. Hom vs B \\((\\widehat C_{{\\rm Hom}}-\\widehat C_B)/\\widehat C_B\\). "
    f"Corner correlation and RMS difference over the {len(LAYOUT['cells'])}\\(\\times\\)8 cell-corner parameters. The macroscale volume of the uniform design, "
    f"{fx(V0_macro['y'], 6)}, agrees with the fine-scale material volume {fx(F['B1']['V0'], 6)} to {sc(dv0)}.")
out()

# ================================================================================================ ST24 perturbations and switches
out('### Table ST24. Geometry-generation fallback and discrete switches')
out()
out('#### ST24a. Applied perturbations')
out()
rows = []


def kind_of(t, case, k):
    bc = base(case)
    if bc in KIND:
        kd, rv = KIND[bc]
        return f"{bc.split('_')[-1]} ({'cut, ' + format(rv, '.3f') if kd == 'CUT' else 'uncut'})"
    prev = [h for h in H[t] if h['k'] == k][0]['fps']
    cn = [v['cut_nodes'] for c, v in prev.items() if base(c) == bc][0]
    return f"{bc.split('_')[-1]} ({'cut' if cn else 'uncut'})"


LOCAL = {t: cell_local(t) for t in RUN}
DTAU = {}
for t in ('A', 'A_exact_twin', 'B1', 'B2', 'XH_y', 'XH_z'):
    recs = F[t]['body_perturbations']
    assert not (set(LOCAL[t]) - {p['k'] for p in recs}), t                         # cell-local attempts only where the fallback followed
    if not recs:
        rows.append((NAMES[t], '—', '—', '—', '—', '—', '—', '—', '—'))
        continue
    for k in sorted({p['k'] for p in recs}):
        rk = [p for p in recs if p['k'] == k]
        ap = [p for p in F[t]['body_perturbations_applied'] if p['k'] == k][0]
        assert rk[-1]['eps'] == ap['eps'] and [p['eps'] for p in rk] == FB_EPS[:len(rk)]
        h = [h for h in H[t] if h['k'] == k][0]
        tv = np.asarray(h['tv'])
        vs = np.asarray(rk[-1]['vertices'])
        # change by the applied perturbation, tau - tau/(1 + eps); at a vertex clipped to a bound the MMA value is not
        # recorded, so only the bound |b - b/(1 + eps)| of its change is known
        clip = np.isclose(tv[vs], ARGS[t]['tmin'], rtol=0, atol=1e-12) | np.isclose(tv[vs], ARGS[t]['tmax'], rtol=0, atol=1e-12)
        d = np.abs(tv[vs] - tv[vs] / (1 + ap['eps']))
        dt = sc(float(d[~clip].max()))
        if clip.any():
            dt += f" (≤ {sc(float(d[clip].max()))} at {int(clip.sum())} vertices clipped to {fx(float(tv[vs][clip][0]), 2)})"
        DTAU[(t, k)] = (float(d[~clip].max()), float(d[clip].max()) if clip.any() else None)
        # failing cells: those of the design returned by MMA, then those failing only after a perturbation
        seen, parts = set(), []
        for i, p in enumerate(rk):
            new = [c for c in p['failed'] if base(c) not in seen]
            seen |= {base(c) for c in p['failed']}
            if new:
                lab = ', '.join(kind_of(t, c, k) for c in new)
                parts.append(lab if i == 0 else f"after {eps_s(rk[i - 1]['eps'])} also {lab}")
        loc = LOCAL[t].get(k)
        loc_s = '—' if not loc else f"{', '.join(eps_s(e) for _, e in loc)} (cell {', '.join(sorted({c.split('_')[-1] for c, _ in loc}))})"
        rows.append((NAMES[t], k, '; '.join(parts), loc_s, ', '.join(eps_s(p['eps']) for p in rk), eps_s(ap['eps']),
                     f"{ap['n_vertices']} (of {M[t]['free']})", len(rk[-1]['regenerated']), f"{ap['attempts']}; {dt}"))
        assert ap['attempts'] == len(rk) + 1
out(table(['Case', 'Iteration', 'Failing cells', 'Earlier cell-local attempts, all failed', 'Fallback: \\(\\epsilon\\) tried, in order', 'Applied \\(\\epsilon\\)',
           'Perturbed vertices', 'Cells regenerated', 'Fallback generation attempts; largest \\(\\lvert\\Delta\\tau\\rvert\\)'], rows))
out()
out("Failing cells by layout index (retained volume fraction of cut cells): the cells whose generation failed for the design returned by MMA and, where stated, "
    "in addition after a perturbation. A failure means that the geometry generator could not certify the material patches of an element (Appendix A.1). "
    "Earlier cell-local attempts: before the fallback of Table ST21 was in use, the eight corner parameters of the failing cell alone were scaled by \\(1+\\epsilon\\); "
    "these attempts are not counted in the last column. Fallback: each tried \\(\\epsilon\\) is applied to the design returned by MMA, not accumulated, to the free "
    "vertices of the cells that failed at the preceding attempt; the last one tried is the applied one. Fallback generation attempts: the design returned by MMA and "
    "each perturbation. Largest \\(\\lvert\\Delta\\tau\\rvert\\): largest change of a corner parameter by the applied perturbation; at a vertex clipped to a bound the "
    "value returned by MMA is not recorded, and only an upper bound of its change is given.")
out()
out('#### ST24b. Cells whose discrete description changed between consecutive iterations')
out()
KEYS = [('active', 'Active elements'), ('faces', 'Ghost faces'), ('ports', 'Retained coordinates'), ('cut_nodes', 'Cut-band nodes'),
        ('weak', 'Weak-support nodes'), ('el_fringe', 'Element fringe hyperedges'), ('gp_fringe', 'Face fringe hyperedges'), ('shift', 'Coarse-factor shift')]


def any_changes(Hr):
    n = []
    for a_, b_ in zip(Hr[:-1], Hr[1:]):
        fa = {base(c): v for c, v in a_['fps'].items()}; fb = {base(c): v for c, v in b_['fps'].items()}
        n.append(sum(any(fa[c].get(k) != fb[c].get(k) for k, _ in KEYS) for c in fa if c in fb))
    return n


def mmm(v):
    return f"{int(np.min(v))} / {int(np.median(v)) if float(np.median(v)).is_integer() else float(np.median(v))} / {int(np.max(v))}"


rows = []
for t in ('A', 'A_exact_twin', 'B1', 'B2', 'XH_y', 'XH_z'):
    sw = F[t]['switches_per_iteration']
    ncell = len(H[t][0]['fps'])
    recorded = set(next(iter(H[t][0]['fps'].values())).keys())
    r = [f"{NAMES[t]} ({ncell} cells, {len(sw)} steps)"]
    for key, _ in KEYS:
        r.append(mmm([s_[key] for s_ in sw]) if key in recorded else 'not recorded')
    r.append(mmm(any_changes(H[t])))
    rows.append(r)
out(table(['Case'] + [lab for _, lab in KEYS] + ['Any of these'], rows))
out()
out("Minimum / median / maximum over the steps between consecutive iterations of the number of cells whose count changed: active elements, ghost-penalty faces, "
    "retained coordinates, nodes flagged by the network's binary node features for cut-band membership and weak support, element and face fringe hyperedges, "
    "and the diagonal shift of the coarse factorisation. A change that leaves a count unchanged is not detected, so the numbers are lower bounds. "
    "The exact twin recorded active elements and retained coordinates only.")
out()

# ================================================================================================ ST25 route checks
rv = F['route_validation']
out('### Table ST25. Route checks on the \\(2\\times2\\times2\\) block (case A settings, iterations 0–2)')
out()
rows = []
for r in rv['slow_vs_fast']:
    rows.append(('Deployed route vs double-precision correction route', r['k'], f"{fx(r['C_fast'], 6)} / {fx(r['C_slow'], 6)}",
                 sc(abs(r['C_fast'] - r['C_slow']) / r['C_slow']), f"{r['pcg_fast']} / {r['pcg_slow']}", sc(r['s_vertex_rel']),
                 f"iteration {fx(r['iter_s_fast'], 0)} / {fx(r['iter_s_slow'], 0)}"))
f2 = rv['fd_vs_ad_k0']
rows.append(('Reverse mode vs central moment differences', 0, f"{fx(f2['C_ad'], 7)} / {fx(f2['C_fd'], 7)}", sc(abs(f2['C_ad'] - f2['C_fd']) / f2['C_fd']),
             '—', sc(f2['s_vertex_rel']), f"sensitivities {fx(f2['sens_s_ad'], 0)} / {fx(f2['sens_s_fd'], 0)}"))
for r in rv['cold_vs_warm']:
    extra = '' if r['warm_hit'] is None else f"; matched {r['warm_hit']:.5f}, scale {r['warm_scale']:.3f}"
    rows.append(('Warm vs cold start', r['k'], f"{fx(r['C_warm'], 7)} / {fx(r['C_cold'], 7)}", sc(abs(r['C_warm'] - r['C_cold']) / r['C_cold']),
                 f"{r['pcg_warm']} / {r['pcg_cold']}", '—', f"PCG {fx(r['solve_warm'], 0)} / {fx(r['solve_cold'], 0)}{extra}"))
out(table(['Comparison (first / second)', 'Iteration', 'Compliance', 'Relative difference', 'PCG iterations', 'Vertex-gradient difference', 'Time (s)'], rows))
out()
PIL = {r: hist(f'optA/{r}/history.jsonl') for r in ('pilot222', 'pilot222f', 'pilot222a', 'pilot222w')}
host_slow = max(h['host_peak_gb'] for h in PIL['pilot222']); host_fast = max(h['host_peak_gb'] for h in PIL['pilot222f'])
# the warm-start run used reverse-mode sensitivities: its sensitivity phase takes the time of the reverse-mode run, not that of
# the central differences, and its vertex gradient at iteration 0 (cold in both) is that of the reverse-mode run
sv = {r: np.asarray(PIL[r][0]['s_vertex']) for r in PIL}
ts = {r: float(np.mean([h['times']['sens_s'] for h in PIL[r]])) for r in PIL}
assert abs(ts['pilot222w'] - ts['pilot222a']) < 0.1 * ts['pilot222a'] and ts['pilot222f'] > 1.8 * ts['pilot222w']
assert np.linalg.norm(sv['pilot222w'] - sv['pilot222a']) < np.linalg.norm(sv['pilot222w'] - sv['pilot222f'])
out("Each comparison runs the case A optimisation for up to three iterations with both settings. Deployed route: the implementation of the optimisation runs "
    "(Table ST21), network and correction in single precision, cold start, sensitivities from central moment differences (Eq. (H.6)) unless stated; "
    "the warm-start run uses reverse-mode sensitivities. Double-precision correction route: the implementation of the accuracy results of Sections 6.2–6.9, "
    f"with the correction in double precision and operator state held in host memory between applications (peak host memory {fx(host_slow, 1)} GiB, against "
    f"{fx(host_fast, 1)} GiB for the deployed route); the two routes therefore differ in implementation, correction precision and operator placement, and the time "
    "of the second is not a cost baseline. Vertex-gradient difference: \\(\\lVert\\widetilde s_{g,1}-\\widetilde s_{g,2}\\rVert/\\lVert\\widetilde s_{g,2}\\rVert\\) "
    "over all vertices. Matched: fraction of free retained coordinates found in the previous solution; scale: energy-optimal factor of the warm start. At iteration 0 the "
    "warm-start run has no previous solution.")
out()

# ================================================================================================ derived numbers quoted in the text
out('<!-- derived numbers for the text -->')
yA = [(h['k'], h['mma']['y_max']) for h in H['A'] if h['mma']['y_max'] > 1e-6]
yB = {t: [(h['k'], h['mma']['y_max']) for h in H[t] if h['mma']['y_max'] > 1e-6] for t in ('B1', 'B2', 'XH_y', 'XH_z')}
out(f"<!-- MMA elastic variables > 0: case A iterations {[k for k, _ in yA]} (max {max(v for _, v in yA):.2f}); "
    + '; '.join(f"{t} {[k for k, _ in v]}" + (f" (max {max(x for _, x in v):.2f})" if v else '') for t, v in yB.items()) + ' -->')
out(f"<!-- time ratio twin/NICE per iteration {X['iter_s_mean'] / A['iter_s_mean']:.2f}; total {X['iter_s_total'] / A['iter_s_total']:.2f} -->")
st = F['storage']['packstore_compare.txt']
g0, g1 = (float(x) for x in re.findall(r'stream_GB ([0-9.]+)', st)[:2])
out(f"<!-- packed storage: streamed state {g0:.4f} -> {g1:.4f} GB = {g0 * 1e9 / 2**30:.2f} -> {g1 * 1e9 / 2**30:.2f} GiB, reduction {100 * (1 - g1 / g0):.1f}% -->")
pk0, pk1 = (float(x) for x in re.findall(r' peak ([0-9.]+)', st)[:2])
tt0, tt1 = (float(x) for x in re.findall(r' total ([0-9.]+)', st)[:2])
pcg0, pcg1 = (int(x) for x in re.findall(r' pcg ([0-9]+)', st)[:2])
cm = re.search(r'bitwise equal (\w+) max rel ([0-9.e-]+)', st).group(2)
sr = re.search(r'sens bitwise equal \w+ rel ([0-9.e-]+)', st).group(1)
rr = re.search(r'run-to-run \(unpacked twice\): compliance max rel ([0-9.e-]+) sens rel ([0-9.e-]+)', st).groups()
out(f"<!-- packed storage: peak GPU {pk0} -> {pk1} GB (+{(pk1 - pk0) * 1e9 / 2**30:.2f} GiB), iteration {tt0} -> {tt1} s, PCG {pcg0}/{pcg1}; "
    f"compliance max rel change {float(cm):.1e}, sensitivities {float(sr):.1e}; two unpacked evaluations: {float(rr[0]):.1e}, {float(rr[1]):.1e} -->")
for t in ('A', 'B1', 'B2', 'XH_y', 'XH_z'):
    out(f"<!-- {t}: first-iteration C {F[t]['C0']:.6g}, max C {max(F[t]['C_trace']):.6g} at k={int(np.argmax(F[t]['C_trace']))} -->")
dd = [(r['k'], (r['C_nice'] - r['C_exact_twin']) / r['C_exact_twin']) for r in F['A_twin_trace']]
pk = {p['k'] for p in A['body_perturbations_applied']}
un = [v for k, v in dd if k not in pk]
out(f"<!-- NICE vs twin, same iteration, unperturbed iterations: {100 * max(un):.4f}% to {100 * min(un):.4f}%; perturbed: "
    f"{[(k, round(100 * v, 4)) for k, v in dd if k in pk]} -->")
un_same = [v for k, v in dd if k in SAME]; un_div = [v for k, v in dd if k >= k_div and k not in pk]
out(f"<!-- NICE vs twin on the common design (iterations {SAME[0]}-{SAME[-1]}): {100 * max(un_same):.4f}% to {100 * min(un_same):.4f}% "
    f"(iteration by iteration {[round(100 * v, 4) for v in un_same]}); after the paths separate (from iteration {k_div}, unperturbed): "
    f"{100 * max(un_div):.4f}% to {100 * min(un_div):.4f}%; design difference from iteration {k_div}: {DTV[k_div]:.2e} to max {max(DTV[k_div:]):.2e} -->")
out(f"<!-- cold starts (no warm_hit): {COLD} ; resumed iterations {RESUMED} ; geometry generated beforehand (bodies_s < 1 s) {GEOM_REUSED}; "
    f"partly generated before a resume {GEOM_PART} -->")
out(f"<!-- fallback of ST21 enabled from iteration {FBF}; earlier cell-local attempts {LOCAL} -->")
out(f"<!-- largest parameter change by the applied fallback perturbations (unclipped, clipped bound): {DTAU} -->")
out(f"<!-- span pairs case A {SPA}, plates {SPB}; gradient stencils {GSA}, {GSB} -->")
out(f"<!-- route checks: host memory peak, double-precision correction route {host_slow:.2f} GiB, deployed {host_fast:.2f} GiB; "
    f"sensitivity phase means (s) {ts} -->")
out(f"<!-- options of the timed run of Table 5 (evidence/d5_off.json) not used by the optimisation runs: {[o[0] for o in OPTS]}; its sensitivity "
    f"objective '{D5['sens_obj']}' (summed compliance of the three loads) -->")
out(f"<!-- geometry phase means with / without iteration 0: case A {A['phase_means']['bodies_s']:.1f} / "
    f"{np.mean([h['times']['bodies_s'] for h in H['A'][1:]]):.1f} s -->")
cw = rv['cold_vs_warm']
out(f"<!-- warm start PCG reduction: {[round(100 * (1 - r['pcg_warm'] / r['pcg_cold']), 1) for r in cw[1:]]}%; PCG time reduction "
    f"{[round(100 * (1 - r['solve_warm'] / r['solve_cold']), 1) for r in cw[1:]]}% -->")
for t in ('B1', 'B2'):
    out(f"<!-- {t}: compliance change first->last {100 * (F[t]['C_ratio'] - 1):.2f}%; sum of iteration times {F[t]['iter_s_total'] / 3600:.2f} h -->")
out(f"<!-- case A: compliance change first->last {100 * (A['C_ratio'] - 1):.2f}% (NICE), twin {100 * (X['C_ratio'] - 1):.2f}%; initial V/V* {A['V_rel_trace'][0]:.3f} -->")
out(f"<!-- case A initial corner range {A['tau0_range'][0]:.4f}-{A['tau0_range'][1]:.4f} -->")
spread = {t: (max(v) - min(v)) / min(v) for t, v in (('y', [cmp_['y'][k] for k in ('C_B_final', 'C_H_fine', 'C_X_final')]),
                                                       ('z', [cmp_['z'][k] for k in ('C_B_final', 'C_H_fine', 'C_X_final')]))}
out(f"<!-- spread of the three final fine-scale NICE compliances per load: y {100 * spread['y']:.2f}%, z {100 * spread['z']:.2f}% -->")
nv = LAYOUT['normal']
out(f"<!-- plate cut normal angle (internal) {np.degrees(np.arctan2(nv[1], nv[0])):.2f} deg; unrotated {90 - np.degrees(np.arctan2(nv[1], nv[0])):.2f} deg; "
    f"cells {len(LAYOUT['cells'])}, cut {sum(c['kind'] == 'CUT' for c in LAYOUT['cells'])}, retained {sorted({round(c['retained'], 3) for c in LAYOUT['cells'] if c['kind'] == 'CUT'})} -->")
nev = sum(len(F[t]['body_perturbations_applied']) for t in ('A', 'A_exact_twin', 'B1', 'B2', 'XH_y', 'XH_z'))
out(f"<!-- fallback events over all runs: {nev} -->")
print('\n'.join(P))
