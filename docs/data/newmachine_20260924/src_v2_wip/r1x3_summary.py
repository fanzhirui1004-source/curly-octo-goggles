"""Revision round 1, X3 summary of E3 / E4 / E14 (reads the outputs of r1_pairs.py, r1_lat.py, r1_psd.py; CPU only).
Applies the pre-registered gradient decision rule for Section 6.11 (DECISION below) and writes <dir>/SUMMARY_X3.json.

Measures (consistent / gate loads only, unless stated):
  e_t = ||s_tilde - s|| / ||s||, e_c = ||C_hat' - s|| / ||s||, d = ||C_hat' - s_tilde|| / ||s||, cosine, max component
  error; pairs over the test cell's 8 corners, lattices over all cell corners and over the shared lattice vertices.
  C_hat' = D_act (action-energy central difference at fixed q_hat) at the primary step H0; step study over the steps run;
  the extension route D_ext and the field-energy route D_fld as cross-checks; the re-solve check (pairs).
STEP RULE: D_act at H0 = 1e-3 is 'verified' for a configuration when its difference to both neighbouring steps is
  <= max(0.25 e_c, 2e-4) relative to ||s|| and D_act agrees with D_ext within the same tolerance; otherwise the most
  stable step (smallest difference to its neighbours) is used and flagged; with no step meeting the tolerance, C_hat'
  is 'unverified' for that configuration.
DECISION (Section 6.11 gradient), primary evidence = the lattices, vertex-aggregated, consistent loads; pairs support:
  A  C_hat' unverified on any lattice                                  -> s_tilde (complete derivative not resolved)
  B  max e_c <= 0.5 max e_t on the lattices AND on the pairs           -> C_hat_,c (consistent and more accurate)
  C  otherwise                                                         -> s_tilde; report max d as the objective /
     gradient inconsistency; optimiser without line search (OC / MMA); stopping tests with tolerance >= 2 max d
  (if max e_t <= 1% and max d <= 1%, the choice is immaterial at the paper's accuracy: s_tilde, with d reported).
Usage: r1x3_summary.py <dir>  (pair_*.json, lat64*.json, lat32*.json, psd.json)
"""
import json, sys, glob
from pathlib import Path
import numpy as np

H0 = '0.001'


def load(p):
    return json.loads(Path(p).read_text())


def rel(a, b, n):
    return np.linalg.norm(np.asarray(a) - np.asarray(b), axis=0) / n


def gm(ex, ap):
    ex, ap = np.asarray(ex, float), np.asarray(ap, float)
    n = np.linalg.norm(ex, axis=0)
    cos = (ex * ap).sum(0) / (n * np.linalg.norm(ap, axis=0))
    return rel(ap, ex, n), cos


def step_check(D, S, S_ti, steps):
    """-> (chosen step key, verified flag, per-step info). D: dict key -> (nvar, nload); S exact."""
    n = np.linalg.norm(S, axis=0)
    info = {}
    keys = [k for k in steps if k in D]
    for i, k in enumerate(keys):
        e_c = rel(D[k], S, n).max()
        nb = [rel(D[k], D[keys[j]], n).max() for j in (i - 1, i + 1) if 0 <= j < len(keys)]
        info[k] = dict(e_c=float(e_c), nb_diff=float(max(nb)) if nb else None)
    def ok(k):
        tol = max(0.25 * info[k]['e_c'], 2e-4)
        return info[k]['nb_diff'] is not None and info[k]['nb_diff'] <= tol
    if H0 in info and ok(H0):
        return H0, True, info
    good = [k for k in keys if ok(k)]
    if good:
        return min(good, key=lambda k: info[k]['nb_diff']), True, info
    return (H0 if H0 in info else (keys[0] if keys else None)), False, info


def pairs(d):
    out = {}
    for f in sorted(glob.glob(str(Path(d) / 'pair_*.json'))):
        r = load(f)
        z = np.load(f[:-5] + '.npz')
        cols = [str(c) for c in z['cols']] if 'cols' in z else []
        steps = sorted({k[6:] for k in z.files if k.startswith('D_act_') and not k.startswith('D_act_fixb')}, key=float, reverse=True)
        for conf, c in r['configs'].items():
            jj = [j for j, col in enumerate(cols) if col.split(':')[0] == conf]
            g = np.asarray(c['gate_mask'], bool)
            jg = [jj[i] for i in range(len(jj)) if g[i]]
            key = f"{r['case']}/{conf}"
            ins = c['instr']
            row = dict(gate_compliance_max=c['gate']['gate_compliance_max'], gate_sens_max=c['gate']['gate_sens_max'],
                       Ut_rho_rel_max=float(np.max(np.abs(np.asarray(ins['Ut_rho_rel'])[g]))),
                       bound_dual_rel_max=float(np.max(np.asarray(ins['bound_dual_rel'])[g])),
                       err_C_bar_max=float(np.max(np.abs(np.asarray(ins['err_C_bar'])[g]))),
                       err_J_max=float(np.max(np.abs(np.asarray(ins['err_J'])[g]))),
                       omega_rel_max=float(np.max(np.abs(np.asarray(ins['omega_rel'])[g]))),
                       identity_resid_rel_max=float(np.max(np.abs(np.asarray(ins['identity_resid_rel'])[g]))),
                       true_residual_max=float(np.max(np.asarray(ins['true_residual'])[g])))
            if jg and steps:
                S, St = z['S_ex'][:, jg], z['S_tilde'][:, jg]
                D = {k: z[f'D_act_{k}'][:, jg] for k in steps}
                hk, ver, info = step_check(D, S, St, steps)
                e_t, cos_t = gm(S, St)
                e_c, cos_c = gm(S, D[hk])
                n = np.linalg.norm(S, axis=0)
                dd = rel(D[hk], St, n)
                ext = rel(z[f'D_ext_{hk}'][:, jg], D[hk], n)
                fld = rel(z[f'D_fld_{hk}'][:, jg], D[hk], n)
                row.update(step=hk, verified=bool(ver and ext.max() <= max(0.25 * e_c.max(), 2e-4)), step_info=info,
                           e_t=float(e_t.max()), e_c=float(e_c.max()), d=float(dd.max()), cos_t=float(cos_t.min()),
                           cos_c=float(cos_c.min()), act_vs_ext=float(ext.max()), act_vs_fld=float(fld.max()))
                if f'D_act_fixb_{hk}' in z.files:
                    row['b_term'] = float(rel(z[f'D_act_fixb_{hk}'][:, jg], D[hk], n).max())
            out[key] = row
        if r.get('resolve'):
            rs = []
            for x in r['resolve']:
                st = np.asarray(x['s_tilde'])
                rs.append(dict(config=x['config'], corner=x['corner'], step=x['step'], s=np.asarray(x['s_exact']).tolist(),
                               ext_resolve=(np.asarray(x['dC_hat_resolve']) - st).tolist(),
                               ext_resolve_J=(np.asarray(x['dJ_resolve']) - st).tolist(),
                               ext_fixed_q=(np.asarray(x['D_act_fixed_q']) - st).tolist(),
                               true_res_max=x.get('true_res_max')))
            out[r['case'] + '/resolve'] = rs
        rb = r.get('rebuilds', {})
        out[r['case'] + '/meta'] = dict(reproducibility=r.get('reproducibility'), rebuilds=rb,
                                        switches=sum(1 for s in r.get('switches', []) if s.get('any_switch')))
    return out


def lattices(d, pattern):
    out = {}
    for f in sorted(glob.glob(str(Path(d) / pattern))):
        r = load(f)
        z = np.load(f[:-5] + '.npz') if Path(f[:-5] + '.npz').exists() else None
        for LN, rec in r['lattices'].items():
            m = [k for k in rec if isinstance(rec[k], dict) and 'final' in rec[k]]
            if not m:
                continue
            L = rec[m[0]]
            nc = len(L['final']['C_bar']) - r['args']['n_random']
            fin, du = L['final'], L['dual']
            row = dict(file=Path(f).name, gate_compliance_max=L['gate_compliance_max'], gate_sens_max=L['gate_sens_max'],
                       iterations=L['iterations'], true_residual=fin['true_residual'], Ut_rho_rel=fin['Ut_rho_rel'],
                       err_C_bar=fin['err_C_bar'], err_J=fin['err_J'], bound_dual_rel=du['bound_dual_rel'],
                       omega_rel=du['omega_rel'], identity_resid_rel=du['identity_resid_rel'],
                       snaps=[dict(level=s['level'], it=s['iterations'], true_res=max(s['true_residual']),
                                   err_C_bar=max(abs(v) for v in s['err_C_bar'][:nc]), err_J=max(abs(v) for v in s['err_J'][:nc]),
                                   Ut_rho_rel=max(abs(v) for v in s['Ut_rho_rel'][:nc])) for s in L['snaps']],
                       grad_s_tilde=dict(cellcorner_rel=L['grad_s_tilde']['cellcorner']['rel'], vertex_rel=L['grad_s_tilde']['vertex']['rel'],
                                         vertex_cos=L['grad_s_tilde']['vertex']['cos'], vertex_comp_max=L['grad_s_tilde']['vertex']['comp_max']),
                       env=dict((k, r['env'].get(k)) for k in ('OPL_COARSE_FP32', 'OPL_TAILT_FUSED', 'correction', 'OPL_CONV_FP32')))
            if z is not None and 'grad_complete' in rec:
                S = z[f'{LN}__S']; St = z[f'{LN}__S_tilde']; vid = z[f'{LN}__vid']; nv = rec['vertices']['n']
                def agg(G):
                    o = np.zeros((nv, G.shape[2]))
                    for i in range(G.shape[0]):
                        for c in range(8):
                            o[vid[i, c]] += G[i, c]
                    return o
                steps = sorted([k for k in rec['grad_complete']], key=float, reverse=True)
                Sv, Stv = agg(S)[:, :nc], agg(St)[:, :nc]
                Dv = {k: agg(z[f'{LN}__{k}'])[:, :nc] for k in steps}
                hk, ver, info = step_check(Dv, Sv, Stv, steps)
                e_t, cos_t = gm(Sv, Stv); e_c, cos_c = gm(Sv, Dv[hk])
                n = np.linalg.norm(Sv, axis=0)
                row.update(step=hk, verified=ver, step_info=info, e_t=float(e_t.max()), e_c=float(e_c.max()),
                           d=float(rel(Dv[hk], Stv, n).max()), cos_t=float(cos_t.min()), cos_c=float(cos_c.min()),
                           cellcorner=dict(e_t=float(gm(S.reshape(-1, S.shape[2])[:, :nc], St.reshape(-1, S.shape[2])[:, :nc])[0].max()),
                                           e_c=float(gm(S.reshape(-1, S.shape[2])[:, :nc], z[f'{LN}__{hk}'].reshape(-1, S.shape[2])[:, :nc])[0].max())))
            out[f'{Path(f).stem}/{LN}'] = row
        if 'deriv' in r:
            for case, dv in r['deriv'].items():
                out[f'{Path(f).stem}/{case}/meta'] = dict(reproducibility=dv['reproducibility'], rebuilds=dv['rebuilds'],
                                                         switches=sum(1 for s in dv['switches'] if s.get('any_switch')))
    return out


def decide(P, L64):
    lat = [v for k, v in L64.items() if isinstance(v, dict) and 'e_c' in v]
    pr = [v for k, v in P.items() if isinstance(v, dict) and 'e_c' in v]
    if not lat or any(not v['verified'] for v in lat):
        return dict(rule='A', gradient='s_tilde', why='complete derivative not verified on every lattice (or not run)')
    et_l, ec_l = max(v['e_t'] for v in lat), max(v['e_c'] for v in lat)
    et_p = max((v['e_t'] for v in pr if v.get('verified')), default=float('nan'))
    ec_p = max((v['e_c'] for v in pr if v.get('verified')), default=float('nan'))
    dmax = max([v['d'] for v in lat] + [v['d'] for v in pr if v.get('verified')])
    base = dict(e_t_lat=et_l, e_c_lat=ec_l, e_t_pairs=et_p, e_c_pairs=ec_p, d_max=dmax,
                pairs_unverified=[k for k, v in P.items() if isinstance(v, dict) and 'e_c' in v and not v['verified']])
    if ec_l <= 0.5 * et_l and (np.isnan(ec_p) or ec_p <= 0.5 * et_p):
        return dict(base, rule='B', gradient='C_hat_c')
    imm = et_l <= 0.01 and dmax <= 0.01 and (np.isnan(et_p) or et_p <= 0.01)
    return dict(base, rule='C', gradient='s_tilde', immaterial=bool(imm),
                stopping_tolerance_min=2 * dmax)


def main(d):
    P = pairs(d)
    L64 = lattices(d, 'lat64*.json')
    L32 = lattices(d, 'lat32*.json')
    psd = load(Path(d) / 'psd.json') if (Path(d) / 'psd.json').exists() else None
    ps = None
    if psd:
        ps = {c: dict(partial=v['partial'], K_e_min=v['K_e_partial']['q0'], fd_min=v['worst_fd'], ad_min=v.get('worst_ad'),
                      fd_n_below_1e8=max(r['n_below']['1e-8'] for r in v['fd']),
                      ad_vs_fd=max(v['ad_vs_fd_rel']) if v.get('ad_vs_fd_rel') else None) for c, v in psd['cells'].items()}
    out = dict(pairs=P, lat64=L64, lat32=L32, psd=ps, decision=decide(P, L64))
    Path(d, 'SUMMARY_X3.json').write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out['decision'], default=float))


if __name__ == '__main__':
    main(sys.argv[1])
