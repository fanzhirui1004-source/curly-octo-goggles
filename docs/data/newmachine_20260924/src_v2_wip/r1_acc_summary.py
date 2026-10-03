"""Revision round 1, accuracy chain (E2, E5, E6): summary of the outputs in <R1 dir> (CPU only, numpy).
Reads the pre-registration <R1>/PREREG_R1ACC.json (written before any result existed), <R1>/valmeta.json, the per-direction
single-cell evaluations <R1>/newval3_{A3,BW,C,CW}.json, the historical newval2_*.json and gate_*.json in <V2 dir>, the C+W
pair gates <R1>/gate_CW_*.json and the held-out pair gates <R1>/ho_gate_{A3,BW,CW}_*.json.
Writes <R1>/R1_ACC_SUMMARY.json and a plain-text digest <R1>/R1_ACC_SUMMARY.txt. Missing inputs are reported, not fatal.
Usage: r1_acc_summary.py <R1 dir> [<V2 dir>]"""
import json, sys, glob, os, time
from pathlib import Path
import numpy as np

CLASSES = ('force_c', 'force', 'support', 'face', 'face_c', 'support_k', 'glued', 'macro', 'grf')
ARMS = dict(A3='NICE (A3_2grid)', BW='NICE-post (B+W: v2L1 + W at evaluation)', C='Uncorrected (A0_ctrl)',
            CW='C+W (A0_ctrl + W at evaluation)')
HIST = dict(A3='newval2_A3_2grid.json', BW='newval2_B2grid.json', C='newval2_A0_ctrl.json')
PAIR_CELLS = ('fresh_val_2000_full', 'fresh_val_2001_full', 'fresh_val_2005_d1_v0', 'fresh_val_2006_d0_v1',
              'fresh_val_2003_d1_v1', 'fresh_val_2004_d0_v2', 'fresh_val_2002_d0_v0', 'fresh_val_2010_d0_v0')


def load(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception as e:                                               # noqa: BLE001
        return None


def strat(meta, c):
    m = meta[c]
    if m['kind'] == 'FULL':
        return 'FULL'
    v = m['vol']
    return 'light' if v > 2 / 3 else ('moderate' if v > 1 / 3 else 'heavy')


def boot_ratio(num, den, n=20000, seed=0):
    """ratio of means mean(num)/mean(den), paired bootstrap over geometries: point, 95% CI; plus geometric-mean ratio."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    ok = np.isfinite(num) & np.isfinite(den)
    num, den = num[ok], den[ok]
    if num.size < 3:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, num.size, size=(n, num.size))
    r = num[idx].mean(1) / den[idx].mean(1)
    pos = (num > 0) & (den > 0)
    lr = np.log(num[pos] / den[pos])
    gi = rng.integers(0, lr.size, size=(n, lr.size))
    g = np.exp(lr[gi].mean(1))
    return dict(n=int(num.size), ratio=float(num.mean() / den.mean()), ci95=[float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))],
                geo_ratio=float(np.exp(lr.mean())), geo_ci95=[float(np.percentile(g, 2.5)), float(np.percentile(g, 97.5))],
                median_ratio=float(np.median(num[pos] / den[pos])), share_num_worse=float((num > den).mean()))


def dstats(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if x.size == 0:
        return None
    return dict(n=int(x.size), mean=float(x.mean()), median=float(np.median(x)), p90=float(np.percentile(x, 90)),
                p95=float(np.percentile(x, 95)), p99=float(np.percentile(x, 99)), max=float(x.max()))


def single_cell(R, V2, meta, out, txt):
    nv = {a: load(R / f'newval3_{a}.json') for a in ARMS}
    out['single_cell_files'] = {a: (str(R / f'newval3_{a}.json') if nv[a] else None) for a in ARMS}
    cases = sorted(meta)
    held = [c for c in cases if not meta[c]['sel']]
    # E2a: ratios (improvement factors: mean(other) / mean(A3) etc.)
    pairs = dict(CW_over_A3=('CW', 'A3'), BW_over_A3=('BW', 'A3'), BW_over_CW=('BW', 'CW'), C_over_CW=('C', 'CW'), C_over_A3=('C', 'A3'))
    e2 = {}
    for pn, (u, d) in pairs.items():
        if not (nv[u] and nv[d]):
            continue
        e2[pn] = {}
        for cls in CLASSES:
            for sub, cs in (('all80', cases), ('held60', held)):
                cc = [c for c in cs if c in nv[u]['per_geo'] and c in nv[d]['per_geo'] and cls in nv[u]['per_geo'][c]['0'] and cls in nv[d]['per_geo'][c]['0']]
                e2[pn][f'{cls}/{sub}'] = boot_ratio([nv[u]['per_geo'][c]['0'][cls] for c in cc], [nv[d]['per_geo'][c]['0'][cls] for c in cc])
            if cls == 'force_c':
                for st in ('FULL', 'light', 'moderate', 'heavy'):
                    cc = [c for c in cases if strat(meta, c) == st and c in nv[u]['per_geo'] and c in nv[d]['per_geo']]
                    e2[pn][f'force_c/{st}'] = boot_ratio([nv[u]['per_geo'][c]['0']['force_c'] for c in cc], [nv[d]['per_geo'][c]['0']['force_c'] for c in cc])
    out['E2a_ratios'] = e2
    # E5: per-direction statistics
    e5 = {}
    for a, d in nv.items():
        if not d:
            continue
        e5[a] = {}
        for cls in CLASSES:
            for sub, cs in (('all80', cases), ('held60', held)):
                cc = [c for c in cs if c in d.get('per_dir', {}) and cls in d['per_dir'][c]['0']]
                if not cc:
                    continue
                pooled = np.concatenate([np.asarray(d['per_dir'][c]['0'][cls], float) for c in cc])
                gmax = {c: d['dir_stats'][c]['0'][cls].get('max', np.nan) for c in cc}
                gmean = {c: d['per_geo'][c]['0'][cls] for c in cc}
                ndir = [d['dir_stats'][c]['0'][cls]['n'] for c in cc]
                top = sorted(cc, key=lambda c: -np.nan_to_num(gmax[c], nan=-1))[:5]
                e5[a][f'{cls}/{sub}'] = dict(pooled_directions=dstats(pooled), geometry_means=dstats(list(gmean.values())),
                                             geometry_max=dstats(list(gmax.values())),
                                             max_over_mean_per_geometry=dstats([gmax[c] / gmean[c] for c in cc if gmean[c] > 0]),
                                             directions_per_geometry=dict(min=int(min(ndir)), max=int(max(ndir)), total=int(sum(ndir))),
                                             worst5_by_direction_max=[dict(case=c, stratum=strat(meta, c), sel=meta[c]['sel'],
                                                                           max=gmax[c], mean=gmean[c]) for c in top])
    out['E5_per_direction'] = e5
    # reproduction of the archived means (same protocol): max relative difference per arm
    rep = {}
    for a, f in HIST.items():
        h = load(Path(V2) / f)
        if not (h and nv.get(a)):
            continue
        diffs = [abs(nv[a]['per_geo'][c]['0'][cls] / h['per_geo'][c]['0'][cls] - 1) for c in nv[a]['per_geo'] if c in h['per_geo']
                 for cls in nv[a]['per_geo'][c]['0'] if cls in h['per_geo'][c]['0'] and h['per_geo'][c]['0'][cls] not in (0,)
                 and np.isfinite(h['per_geo'][c]['0'][cls])]
        rep[a] = dict(archived=f, n=len(diffs), max_rel_diff=float(max(diffs)) if diffs else None,
                      median_rel_diff=float(np.median(diffs)) if diffs else None,
                      ckpt_step_new=nv[a].get('ckpt_step'), ckpt_step_archived=h.get('ckpt_step'))
    out['reproduction_vs_archived'] = rep
    for a, r in rep.items():
        txt.append(f"reproduction {a} vs {r['archived']}: n={r['n']} max rel diff {r['max_rel_diff']} (ckpt step {r['ckpt_step_new']} / {r['ckpt_step_archived']})")
    for a in e5:
        for sub in ('all80', 'held60'):
            k = f'force_c/{sub}'
            if k in e5[a]:
                p = e5[a][k]['pooled_directions']; gm = e5[a][k]['geometry_max']
                txt.append(f"E5 {a} force_c {sub}: directions n={p['n']} mean={100*p['mean']:.4f}% p95={100*p['p95']:.4f}% p99={100*p['p99']:.4f}% "
                           f"max={100*p['max']:.4f}%; per-geometry max: median={100*gm['median']:.4f}% max={100*gm['max']:.4f}%")
    return nv


def gates(R, V2, out, txt):
    def row(p):
        d = load(p)
        if d is None:
            return dict(file=str(p), status='missing')
        if not d.get('results'):
            return dict(file=str(p), status=d.get('r1_status', 'no results'), error=(d.get('r1_error') or {}).get('message'))
        r = d['results'][0]; t = r['test']
        es = np.asarray(r.get('energy_share', []), float)
        return dict(file=str(p), status=d.get('r1_status', 'ok (archived)'), gate_compliance_max=t['gate_compliance_max'],
                    gate_sens_max=t['gate_sens_max'], cut_compliance_max=t.get('cut_compliance_max'), cut_sens_max=t.get('cut_sens_max'),
                    pcg_iterations=t.get('pcg_iterations'), test_energy_share_min=float(es[0].min()) if es.size else None,
                    nbr_energy_share_min=float(es[1].min()) if es.ndim == 2 and es.shape[0] > 1 else None,
                    flags=r.get('r1_flags'), seconds=r.get('seconds'),
                    nonfinite=int((~np.isfinite(np.asarray(t['compliance_rel_err'], float))).sum()))
    # E2b: existing pair configurations, A3 / B+W (archived) and C+W (new)
    e2b = {}
    for c in PAIR_CELLS:
        for conf in 'xy':
            e2b[f'{c}/{conf}'] = dict(A3=row(Path(V2) / f'gate_A3_2grid_{c}_{conf}.json'), BW=row(Path(V2) / f'gate_B2grid_{c}_{conf}.json'),
                                      C=row(Path(V2) / f'gate_A0_ctrl_{c}_{conf}.json'), CW=row(R / f'gate_CW_{c}_{conf}.json'))
    out['E2b_pairs'] = e2b
    txt.append('E2b pairs (compliance max / sensitivity max, %): cell/conf  A3 | B+W | C+W')
    f = lambda r: (f"{100*r['gate_compliance_max']:.4f}/{100*r['gate_sens_max']:.3f}" if 'gate_compliance_max' in r else r['status'])
    for k, v in e2b.items():
        txt.append(f"  {k}: {f(v['A3'])} | {f(v['BW'])} | {f(v['CW'])}")
    # E6: held-out pairs
    e6 = {}
    for p in sorted(glob.glob(str(R / 'ho_gate_*.json'))):
        n = Path(p).stem[len('ho_gate_'):]
        arm, rest = n.split('_', 1)
        e6.setdefault(rest, {})[arm] = row(p)
    out['E6_heldout_pairs'] = e6
    txt.append('E6 held-out pairs (compliance max / sensitivity max, %; flags): cell_conf  A3 | B+W | C+W')
    for k, v in e6.items():
        txt.append(f"  {k}: " + ' | '.join(f(v[a]) + ('' if not v[a].get('flags') else
                   (' [PCG cap]' if v[a]['flags']['pcg_at_cap'] else '') + (' [neg share]' if v[a]['flags']['negative_energy_share'] else '')
                   + (' [nonfinite]' if v[a]['flags']['nonfinite_compliance'] else '')) if a in v else '-' for a in ('A3', 'BW', 'CW')))
    agg = {}
    for arm in ('A3', 'BW', 'CW'):
        ok = [v[arm] for v in e6.values() if arm in v and 'gate_compliance_max' in v[arm] and v[arm]['nonfinite'] == 0
              and not (v[arm].get('flags') or {}).get('pcg_at_cap') and not (v[arm].get('flags') or {}).get('negative_energy_share')]
        bad = [k for k, v in e6.items() if arm in v and v[arm] not in ok]
        if ok or bad:
            agg[arm] = dict(n_wellposed=len(ok), n_excluded_or_failed=len(bad), excluded_or_failed=bad,
                            compliance_max=max((r['gate_compliance_max'] for r in ok), default=float('nan')),
                            sens_max=max((r['gate_sens_max'] for r in ok), default=float('nan')),
                            cut_compliance_max=max(((r['cut_compliance_max'] or 0) for r in ok), default=float('nan')),
                            cut_sens_max=max(((r['cut_sens_max'] or 0) for r in ok), default=float('nan')))
    out['E6_aggregate'] = agg
    for a, v in agg.items():
        txt.append(f"E6 {a}: {v['n_wellposed']} well-posed configs, compliance max {100*v['compliance_max']:.4f}%, sensitivity max {100*v['sens_max']:.3f}%, "
                   f"excluded/failed: {v['excluded_or_failed']}")


def main():
    R = Path(sys.argv[1]); V2 = Path(sys.argv[2]) if len(sys.argv) > 2 else R.parent
    out, txt = dict(generated=time.strftime('%Y-%m-%d %H:%M:%S'), r1_dir=str(R), v2_dir=str(V2)), []
    pre = load(R / 'PREREG_R1ACC.json'); meta = load(R / 'valmeta.json')
    out['prereg'] = pre
    if meta is None:
        out['error'] = 'valmeta.json missing'
    else:
        try:
            nv = single_cell(R, V2, meta, out, txt)
        except Exception as e:                                           # noqa: BLE001
            out['single_cell_error'] = repr(e); nv = {}
        # pre-registered rule
        try:
            r = out.get('E2a_ratios', {}).get('CW_over_A3', {}).get('force_c/all80')
            if r:
                F = r['ratio']
                out['PREREG_RULE_RESULT'] = dict(rule=pre and pre.get('rule_E2'), statistic='mean_force_c(C+W) / mean_force_c(A3), 80 geometries, view 0',
                                                 value=F, ci95=r['ci95'], threshold=1.1, downgrade_contribution_ii=bool(F <= 1.1),
                                                 held60=out['E2a_ratios']['CW_over_A3'].get('force_c/held60'),
                                                 BW_over_A3_same_run=out['E2a_ratios'].get('BW_over_A3', {}).get('force_c/all80'),
                                                 BW_over_CW_continuation_share=out['E2a_ratios'].get('BW_over_CW', {}).get('force_c/all80'))
                txt.insert(0, f"PRE-REGISTERED RULE (E2): F = mean force_c(C+W)/mean force_c(A3) over 80 geometries = {F:.4f} "
                              f"[95% CI {r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}] -> downgrade contribution (ii): {F <= 1.1}")
        except Exception as e:                                           # noqa: BLE001
            out['prereg_eval_error'] = repr(e)
    try:
        gates(R, V2, out, txt)
    except Exception as e:                                               # noqa: BLE001
        out['gates_error'] = repr(e)
    (R / 'R1_ACC_SUMMARY.json').write_text(json.dumps(out, indent=1, default=float))
    (R / 'R1_ACC_SUMMARY.txt').write_text('\n'.join(txt) + '\n')
    print('\n'.join(txt))


if __name__ == '__main__':
    main()
