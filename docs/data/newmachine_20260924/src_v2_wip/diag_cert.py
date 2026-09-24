"""A5: can the residual certificate (cert.py, D1) find bad cells without labels? Monitoring only: the certificate never
enters the network forward.

For every val-bank direction q (unit exact energy q^T S q = 1) of a (network, geometry) pair, u = E_hat q (network field,
exact rigid part and port values):
  Delta_true   = e_hat - 1, e_hat = u^T K u                       (= eps_true, the relative energy error)
  Delta_lb(m)  = cert.Cert.lower(u, m) <= Delta_true               (m = 0 Jacobi, m >= 1 block Jacobi-Krylov)
  eps_lb(m)    = Delta_lb / (e_hat - Delta_lb) <= Delta_true / q^T S q   (as cert.Cert.mu_lower; rigorous lower bound)
Parts:
  own       step-1 networks on their own geometry (the gate-passing runs of PROGRESS / MORNING_REPORT_20260924_CN.md:
            r1 mgno_r1_e, FULL mgno_full_c, r2 mgno2_r2_d2)
  zeroshot  the zero-shot pairs of PROGRESS_20260924_CN.md (zeroshot.sh / zs_eval.py: ZEROSHOT_mgno2_r2 with mgno2_r2_c,
            ZEROSHOT_mgno_full, ZEROSHOT_mgno_r1; the r1 row of the table has two targets, so 7 pairs)
  step2     a step-2 checkpoint on the val geometries of SPLIT.json
Report per (part, network, geometry, class) = one 'case': Delta_true mean / p90 / max; per m the efficiency
Delta_lb / Delta_true (mean / median / p10), eps_lb mean / max, bound violations, within-case Spearman. Per m: Spearman
(Delta_lb, Delta_true) pooled over all directions and over case means, and the flagging rule flag = eps_lb > t at case
level (case-mean eps_lb): feasible range max(own-geometry) < t <= min(cases with true mean > hi), chosen t = geometric
midpoint (or the grid value with the fewest errors when infeasible), confusion at case and direction level, and the
review's criteria (every case with true error > hi flagged, no own-geometry case flagged, pooled Spearman >= 0.7).
Since eps_lb <= eps_true, a flag is never raised on a direction whose true error is below t.
Raw per-direction arrays: <out>.npz.  The helpers (load_geo, net_for, field_fn, spearman) are shared with diag_mu.py,
lat_full.py and diag_fringe.py.
Usage: diag_cert.py <out.json> [--parts own,zeroshot,step2] [--ms 0,8,16] [--step2 <ckpt>] ... (--help)
"""
import json, sys, time, gc, argparse
from pathlib import Path
import numpy as np
import torch
import trainlib as TL
import models as MD

dev, dt = TL.dev, TL.dt
S1 = '/root/autodl-tmp/OPL/S1'
BODY, DATA = '/root/autodl-tmp/OPL/S0', '/root/autodl-tmp/OPL/S2/data'
SLOTS, SPLIT = '/root/autodl-tmp/OPL/S2/slots', '/root/autodl-tmp/OPL/S2/SPLIT.json'
R1, R2, FULL = 'fresh_train_0020_cover01_r1', 'fresh_train_0020_cover01_r2', 'fresh_train_0020_full'
RUNS = dict(r1='mgno_r1_e', full='mgno_full_c', r2='mgno2_r2_d2', r2zs='mgno2_r2_c')      # configs mg_r1e / mg_fullc / mg2_r2d2 / mg2_r2c
OWN = (('r1', R1), ('full', FULL), ('r2', R2))
ZS = (('r2zs', 'fresh_train_0031_cover01_r2'), ('r2zs', 'fresh_train_0031_cover01_r1'), ('r2zs', 'fresh_development_0000_cover01_r1'),
      ('full', 'fresh_train_0031_full'), ('full', 'fresh_development_0002_full'),
      ('r1', 'fresh_development_0000_cover01_r1'), ('r1', 'fresh_train_0031_cover01_r1'))


# ------------------------------------------------------------------------------------------ shared helpers
def sync():
    if dev.type == 'cuda':
        torch.cuda.synchronize()


def free_mem():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def family_full(case):
    """The family FULL parent: fresh_<split>_<id>_... -> fresh_<split>_<id>_full (a FULL case maps to itself)."""
    return '_'.join(case.split('_')[:3]) + '_full'


def load_geo(case, body=BODY, data=DATA, slots=SLOTS, log=print):
    """trainlib.Geo on dev: from the step-2 slot cache when present (as train2.Slot), else built (as zs_eval); non-finite
    bank samples dropped (train2.clean_banks: the 5 geometries with NaN support samples)."""
    import train2 as T2
    f = Path(slots) / f'{case}.pt' if slots else None
    if f is not None and f.exists():
        g = torch.load(f, map_location='cpu', weights_only=False)
        T2.move(g, dev); g.C.K = g.C
    else:
        g = TL.Geo(case, body, data, neumann=False, log=lambda s_: None)
    T2.clean_banks(g, case, log)
    return g


def load_ckpt(path):
    return torch.load(path, map_location=dev, weights_only=False)


def net_for(ck, geo):
    """The checkpoint's network built on geo (eval mode) and the number of missing keys (strict=False as evalnet)."""
    cfg = ck['cfg']
    m = MD.build(cfg['model'], [geo], **cfg.get('model_args', {})).to(dev)
    res = m.load_state_dict(ck['model'], strict=False)
    return m.eval(), len(res.missing_keys)


def field_fn(model, geo, fast=True):
    """q (np, k) -> u = E_hat q (nb, k) fp64, through fastnet (default) or trainlib.Geo.field."""
    if fast:
        import fastnet as FN
        f = FN.FastNet(model, geo)
        return lambda q: f.field(q).to(dt)

    def fld(q):
        with torch.no_grad():
            return geo.field(model, q).to(dt)
    return fld


def spearman(a, b):
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.ptp(a[ok]) == 0 or np.ptp(b[ok]) == 0:
        return float('nan')
    from scipy.stats import spearmanr
    return float(spearmanr(a[ok], b[ok])[0])


def eps_from(lb, ehat):
    """cert.Cert.mu_lower's eps_lb = Delta_lb / (e_hat - Delta_lb) (denominator >= q^T S q > 0; guarded)."""
    return lb / np.maximum(ehat - lb, 1e-300)


# ------------------------------------------------------------------------------------------ A5
@torch.no_grad()
def cert_case(geo, field, ct, ms=(0, 8, 16), split='val', chunk=16, classes=None):
    """Per class of geo.banks[split]: raw arrays dtrue = e_hat - 1 and lb<m> = ct.lower(u, m), plus seconds per m."""
    out = {}
    for c in classes or geo.classes:
        Q = geo.banks[split][c]
        acc = {k: [] for k in ['dtrue'] + [f'lb{m}' for m in ms]}
        sec = {m: 0.0 for m in ms}
        for j in range(0, Q.shape[1], chunk):
            u = field(Q[:, j:j + chunk].to(dt))
            acc['dtrue'].append((u * (geo.C @ u)).sum(0) - 1)
            for m in ms:
                sync(); t = time.perf_counter()
                acc[f'lb{m}'].append(ct.lower(u, m).to(dt).reshape(-1))
                sync(); sec[m] += time.perf_counter() - t
            del u
        out[c] = {k: torch.cat(v).cpu().numpy() for k, v in acc.items()}
        out[c]['seconds'] = {str(m): s for m, s in sec.items()}
    return out


def summarize(raw, ms):
    """Case statistics from the raw arrays of one class."""
    d = raw['dtrue']; e = 1 + d
    r = dict(n=int(len(d)), true_mean=float(d.mean()), true_p90=float(np.quantile(d, .9)), true_max=float(d.max()),
             true_min=float(d.min()), m={})
    ok = d > 1e-12
    for m in ms:
        lb = raw[f'lb{m}']; eps = eps_from(lb, e)
        eff = lb[ok] / d[ok]
        r['m'][str(m)] = dict(lb_mean=float(lb.mean()), eps_mean=float(eps.mean()), eps_max=float(eps.max()),
                              eff_mean=float(eff.mean()) if ok.any() else float('nan'),
                              eff_median=float(np.median(eff)) if ok.any() else float('nan'),
                              eff_p10=float(np.quantile(eff, .1)) if ok.any() else float('nan'),
                              violations=int((lb > d * (1 + 1e-6) + 1e-9).sum()), neg=int((lb < -1e-12).sum()),
                              spearman=spearman(lb, d), seconds=raw.get('seconds', {}).get(str(m)))
    return r


def flag_rule(rows, m, hi=0.3, good=0.05, grid=None):
    """Case-level rule flag = (case-mean eps_lb > t) for one m. rows: case summaries with 'part', 'true_mean'."""
    m = str(m)
    eps = np.asarray([r['m'][m]['eps_mean'] for r in rows]); tru = np.asarray([r['true_mean'] for r in rows])
    own = np.asarray([r['part'] == 'own' for r in rows]); bad = tru > hi
    lo = float(eps[own].max()) if own.any() else 0.0
    up = float(eps[bad].min()) if bad.any() else float('inf')
    feasible = lo < up
    if feasible and own.any() and bad.any():
        t = float(np.sqrt(max(lo, 1e-6 * up) * up))
    elif feasible and bad.any():
        t = up / 2
    elif feasible and own.any():
        t = max(2 * lo, 1e-3)
    elif feasible:
        t = 0.1                                                                   # nothing to calibrate on: the trainer's cert_flag
    else:                                                                         # fewest misses + false alarms on a grid
        grid = np.geomspace(1e-4, 1.0, 81) if grid is None else np.asarray(grid)
        err = [int((bad & (eps <= g_)).sum() + (own & (eps > g_)).sum()) for g_ in grid]
        t = float(grid[int(np.argmin(err))])
    fl = eps > t
    return dict(t=t, feasible=bool(feasible), t_low=lo, t_high=up, hi=hi,
                bad_cases=int(bad.sum()), bad_flagged=int((fl & bad).sum()), own_cases=int(own.sum()),
                own_flagged=int((fl & own).sum()), good_cases=int((tru < good).sum()), good_flagged=int((fl & (tru < good)).sum()),
                missed=[r['key'] for r, f_, b_ in zip(rows, fl, bad) if b_ and not f_],
                false_alarms=[r['key'] for r, f_, o_ in zip(rows, fl, own) if o_ and f_])


def direction_rates(raws, m, t, hi=0.3, good=0.05):
    """Direction-level rule flag = eps_lb > t: hit rate among directions with Delta_true > hi, alarm rate below good."""
    d = np.concatenate([r['dtrue'] for r in raws]); lb = np.concatenate([r[f'lb{m}'] for r in raws])
    fl = eps_from(lb, 1 + d) > t
    b, g_ = d > hi, d < good
    return dict(n=int(len(d)), bad=int(b.sum()), hit_rate=float(fl[b].mean()) if b.any() else float('nan'),
                good=int(g_.sum()), alarm_rate=float(fl[g_].mean()) if g_.any() else float('nan'),
                flagged_true_below_t=int((fl & (d < t)).sum()))


def analyse(rows, raws, ms, hi=0.3, good=0.05, fixed=(0.01, 0.02, 0.05, 0.1, 0.2)):
    """Per m: pooled Spearman (directions, case means; overall and per part), the case-level rule, the direction rates,
    fixed-threshold table, efficiency of the force / support classes, and the review's pass criteria."""
    out = {}
    for m in ms:
        sm = str(m)
        d = np.concatenate([r['dtrue'] for r in raws]); lb = np.concatenate([r[f'lb{m}'] for r in raws])
        a = dict(spearman_directions=spearman(lb, d),
                 spearman_cases=spearman([r['m'][sm]['eps_mean'] for r in rows], [r['true_mean'] for r in rows]),
                 spearman_within_case_median=float(np.nanmedian([r['m'][sm]['spearman'] for r in rows])))
        for part in sorted({r['part'] for r in rows}):
            idx = [i for i, r in enumerate(rows) if r['part'] == part]
            a[f'spearman_directions_{part}'] = spearman(np.concatenate([raws[i][f'lb{m}'] for i in idx]),
                                                        np.concatenate([raws[i]['dtrue'] for i in idx]))
        rule = flag_rule(rows, m, hi, good)
        a['rule'] = rule
        a['directions_at_t'] = direction_rates(raws, m, rule['t'], hi, good)
        a['fixed_t'] = {}
        for t in fixed:
            fl = np.asarray([r['m'][sm]['eps_mean'] > t for r in rows]); tru = np.asarray([r['true_mean'] for r in rows])
            own = np.asarray([r['part'] == 'own' for r in rows])
            a['fixed_t'][str(t)] = dict(bad_flagged=int((fl & (tru > hi)).sum()), bad=int((tru > hi).sum()),
                                        own_flagged=int((fl & own).sum()), directions=direction_rates(raws, m, t, hi, good))
        eff = [r['m'][sm]['eff_median'] for r in rows if r['cls'] in ('force', 'support')]
        a['eff_force_support_median'] = float(np.nanmedian(eff)) if eff else float('nan')
        a['criteria'] = dict(all_bad_flagged=rule['bad_flagged'] == rule['bad_cases'], no_own_flagged=rule['own_flagged'] == 0,
                             spearman_ge_0p7=bool(a['spearman_directions'] >= 0.7))
        a['criteria']['pass'] = all(a['criteria'].values())
        out[sm] = a
    passing = [m for m in ms if out[str(m)]['criteria']['pass']]
    best = passing[0] if passing else max(ms, key=lambda m_: np.nan_to_num(out[str(m_)]['spearman_directions'], nan=-1))
    dec = dict(best_m=best, passes=bool(passing), t=out[str(best)]['rule']['t'],
               learned_head_needed=bool(out[str(best)]['eff_force_support_median'] < 0.2))   # D1: learned certificate head only if < 0.2
    return out, dec


def jobs(args):
    """(part, tag, checkpoint path, case) in evaluation order, grouped by case so that each geometry loads once."""
    s1 = Path(args.s1)
    runs = dict(r1=args.r1_run, full=args.full_run, r2=args.r2_run, r2zs=args.zs_r2_run)
    J = []
    if 'own' in args.parts:
        own = dict(r1=args.r1_case, full=args.full_case, r2=args.r2_case)
        J += [('own', runs[k], str(s1 / runs[k] / 'best.pt'), own[k]) for k, _ in OWN]
    if 'zeroshot' in args.parts:
        pairs = [p.split(':') for p in args.zs.split(',')] if args.zs else [(runs[k], c) for k, c in ZS]
        J += [('zeroshot', r, str(s1 / r / 'best.pt'), c) for r, c in pairs]
    if 'step2' in args.parts:
        split = json.loads(Path(args.split).read_text())
        cases = args.val_cases.split(',') if args.val_cases else split['val'][:args.val_max]
        J += [('step2', Path(args.step2).parent.name + '/' + Path(args.step2).stem, args.step2, c) for c in cases]
    order = list(dict.fromkeys(c for *_, c in J))
    return [j for c in order for j in J if j[3] == c]


def parser():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('out')
    ap.add_argument('--parts', default='own,zeroshot,step2')
    ap.add_argument('--ms', default='0,8,16')
    ap.add_argument('--s1', default=S1)
    ap.add_argument('--r1-run', default=RUNS['r1']); ap.add_argument('--full-run', default=RUNS['full'])
    ap.add_argument('--r2-run', default=RUNS['r2']); ap.add_argument('--zs-r2-run', default=RUNS['r2zs'])
    ap.add_argument('--r1-case', default=R1); ap.add_argument('--full-case', default=FULL); ap.add_argument('--r2-case', default=R2)
    ap.add_argument('--zs', default='', help='run:case,run:case,... (default: the zero-shot pairs of PROGRESS_20260924_CN.md)')
    ap.add_argument('--step2', default=S1 + '/s2_full/last.pt')
    ap.add_argument('--split', default=SPLIT); ap.add_argument('--val-max', type=int, default=20); ap.add_argument('--val-cases', default='')
    ap.add_argument('--body', default=BODY); ap.add_argument('--data', default=DATA); ap.add_argument('--slots', default=SLOTS)
    ap.add_argument('--hi', type=float, default=0.3); ap.add_argument('--good', type=float, default=0.05)
    ap.add_argument('--chunk', type=int, default=16); ap.add_argument('--no-fast', action='store_true')
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    import cert as CE
    ms = tuple(int(x) for x in args.ms.split(','))
    rec = dict(args=vars(args), cases=[]); rows, raws, npz = [], [], {}
    log = lambda d: print(json.dumps(d), flush=True)
    cks, geo, ct, gcase = {}, None, None, None
    for part, tag, ckp, case in jobs(args):
        t0 = time.perf_counter()
        try:
            if case != gcase:
                geo = ct = None; free_mem()                                       # one geometry alive at a time
                geo, gcase = load_geo(case, args.body, args.data, args.slots, log), case
                ct = CE.from_geo(geo)
            if ckp not in cks:
                cks = {ckp: load_ckpt(ckp)}                                       # one checkpoint alive at a time
            model, miss = net_for(cks[ckp], geo)
            raw = cert_case(geo, field_fn(model, geo, not args.no_fast), ct, ms, 'val', args.chunk)
        except Exception as e:                                                    # a missing run or geometry must not stop the rest
            log(dict(event='FAIL', part=part, tag=tag, case=case, error=repr(e)[:300])); continue
        for c, rw in raw.items():
            key = f'{part}|{tag}|{case}|{c}'
            r = dict(key=key, part=part, tag=tag, case=case, cls=c, ckpt=ckp, missing_keys=miss, **summarize(rw, ms))
            rows.append(r); raws.append(rw)
            npz.update({f'{key}|{k}': v for k, v in rw.items() if k != 'seconds'})
            log(dict(key=key, true_mean=r['true_mean'], **{f'eff{m}': r['m'][str(m)]['eff_median'] for m in ms},
                     **{f'eps{m}': r['m'][str(m)]['eps_mean'] for m in ms}))
        del model; free_mem()
        rec['cases'] = rows
        Path(args.out).write_text(json.dumps(rec, indent=1))
        log(dict(event='CASE_DONE', part=part, tag=tag, case=case, seconds=time.perf_counter() - t0))
    if rows:
        rec['analysis'], rec['decision'] = analyse(rows, raws, ms, args.hi, args.good)
        rec['analysis_hi0p5'], rec['decision_hi0p5'] = analyse(rows, raws, ms, max(args.hi, 0.5), args.good)   # review: 30-50%
        log(dict(event='DECISION', **rec['decision']))
    Path(args.out).write_text(json.dumps(rec, indent=1))
    np.savez(Path(args.out).with_suffix('.npz'), **npz)
    return rec


if __name__ == '__main__':
    main()
