"""Local CPU tests of the audit-v2 trainer changes (train3.py / trainlib.py / models.py B1 counters; fixture.py, no GPU).
Small and fast (target < 3 min wall on 4 cores): every train3 run is a few steps on the fixture geometry and its aliases,
run in subprocesses (the same thread count everywhere) so the pre-change code can run next to the new one.
 a  defaults unchanged: the pre-change train3 / trainlib / models (a copy given by --orig DIR, default ../../orig_t3) and the
    new code on the same default-style config: STEP losses, grad norms and EVAL scores bit-identical
 b  eval_views = [0, 5] (real oh.py): per-view scores logged and in best.pt / last.pt; the identity score and the training
    losses equal the eval_views = [0] run exactly; the score is the mean of the view scores
 c  ema_debias math on a toy (e_t / (1 - d^t) == the normalised weighted average of the iterates; t = 0 -> current weights)
 d  strict_mix: MIX_ABSENT raises for a class no training geometry has (default config); strict_mix raises for a class that
    one training geometry lacks; MIX_MISSING logged per geometry without strict_mix (in run b); support64 is a known class
    (HostBanks.from_data reads it; accepted in mix / score_classes; absent from the slot banks -> MIX_ABSENT)
 e  resume: a stop/resume run (O_h training views, quota + systematic remainder, debiased EMA, select_min_step) gives the
    same STEP losses and EVAL scores as the uninterrupted run (CPU determinism); negative control: resume_rng = false (the
    old reseed-from-step resume) diverges after the stop
 f  sat_stats knee counting on a toy (fractions above T = knee A and above A, peak), A <= 0 / knee >= 1 guards, bnd_knee
    in the state dict, an old state dict without it keeps the constructor knee
 g  GATE-6 score with a non-finite geometry = inf; quota_systematic shares on a rare-class mix
Usage: python3 t_train3_v2.py [--orig DIR]
"""
import os, sys, json, time, copy, shutil, tempfile, subprocess
from pathlib import Path
os.environ['OPL_DEV'] = 'cpu'
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fixture as FX
import trainlib as TL
import models as MD
import train3 as T3

FAIL, T0 = [], time.perf_counter()
CKPT = FX.FIX / 's2v1_snap_10000.pt'
ORIG = Path(sys.argv[sys.argv.index('--orig') + 1]) if '--orig' in sys.argv else HERE.parent.parent / 'orig_t3'
THREADS = os.environ.get('T3_THREADS', '1')
DRIVER = r'''
import os, sys, json
os.environ['OPL_DEV'] = 'cpu'
sys.path.insert(0, sys.argv[1])
import torch
torch.set_num_threads(int(os.environ.get('T3_THREADS', '1')))
import train3 as T3
T3.main(json.loads(open(sys.argv[2]).read()))
'''


def check(name, ok, **info):
    print(json.dumps(dict(test=name, ok=bool(ok), **info), default=str), flush=True)
    if not ok:
        FAIL.append(name)


def ev(L, k):
    return [r for r in L if r.get('event') == k]


def read_log(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def launch(src, cfg, tmp, name):
    """Start one train3 run in a subprocess; returns (Popen, out dir)."""
    f = tmp / f'{name}.json'
    f.write_text(json.dumps(cfg))
    return subprocess.Popen([sys.executable, str(tmp / 'driver.py'), str(src), str(f)], stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE, env=dict(os.environ, T3_THREADS=THREADS, PYTHONWARNINGS='ignore'))


def wait(procs):
    errs = {}
    for n, p in procs.items():
        _, e = p.communicate()
        if p.returncode:
            errs[n] = e.decode()[-1500:]
    return errs


def fixture_tree(tmp):
    """Slot cache + data dir: 3 training geometries (the real one and 2 aliases; force_c only on alias a) and 1 val."""
    g = FX.small()
    REAL = g.case
    train = [REAL, 'fresh_train_0011_alias_a', 'fresh_train_0012_alias_b']
    val = ['fresh_train_0013_alias_v']
    slots, data = tmp / 'slots', tmp / 'data'
    slots.mkdir(parents=True, exist_ok=True)
    VCL, NV = ('force', 'face'), 8                                          # small val banks: evaluation dominates CPU time
    for c in train + val:
        x = copy.copy(g); x.case = c
        for a in ('fix_ustar_val', 'fix_react_val'):
            x.__dict__.pop(a, None)
        if c in val:
            x.banks = dict(g.banks, val={cl: g.banks['val'][cl][:, :NV].contiguous() for cl in VCL})
            x.sens = dict(g.sens, val={cl: g.sens['val'][cl][:, :NV].contiguous() for cl in VCL})
        torch.save(x, slots / f'{c}.pt')
        d = data / c; d.mkdir(parents=True, exist_ok=True)
        (d / 'DONE.json').write_text(json.dumps(dict(case=c, dofs=g.nb)))
        for cl in g.classes + (['force_c'] if c == train[1] else []):
            src = 'force' if cl == 'force_c' else cl
            for sp_ in TL.SPLITS:
                n_ = NV if sp_ == 'val' else 16
                if sp_ == 'val' and cl not in VCL:
                    continue
                np.save(d / f'{sp_}_{cl}.npy', g.banks['val'][src][:, :n_].T.contiguous().numpy())
                np.save(d / f'{sp_}_{cl}_sens.npy', g.sens['val'][src][:, :n_].T.contiguous().numpy())
    split = dict(train=train, val=val, test=[], val_families=['fresh_train_0013'])
    (tmp / 'SPLIT.json').write_text(json.dumps(split))
    ck = FX.checkpoint(CKPT.name)
    margs = dict(ck['cfg']['model_args']); margs['sparse'] = False
    base = dict(split=str(tmp / 'SPLIT.json'), train_key='train', body=str(tmp / 'nobody'), data=str(data), slot_cache=str(slots),
                model='mgno2', model_args=margs, init=str(CKPT), batch=2, geos_per_step=1, lr=1e-4, pct_start=0.1, seed=0,
                sens_w=1.0, packets=str(tmp / 'nopackets'), val_max=1, probe_max=0, mu_geos=0, cert=False, pool=2, pool_dofs=0,
                val_chunk=16, log_every=1, eval_weights='sel', mix={'force': .3, 'support': .15, 'face': .2, 'macro': .15, 'grf': .2})
    return g, train, base


def steps_of(L):
    return [(r['step'], r['loss'], r['grad_norm']) for r in ev(L, 'STEP')]


def main():
    tmp = Path(tempfile.mkdtemp(prefix='t3v2_', dir=str(HERE.parent.parent)))
    try:
        (tmp / 'driver.py').write_text(DRIVER)
        g, train, base = fixture_tree(tmp)
        # --- runs in parallel: a (orig / new), b (views), e (uninterrupted / stopped)
        orig = tmp / 'orig_src'
        orig.mkdir()
        for f in HERE.glob('*.py'):
            shutil.copy(f, orig / f.name)
        for f in ('train3.py', 'trainlib.py', 'models.py'):
            shutil.copy(ORIG / f, orig / f)
        for f in HERE.glob('*.npz'):
            shutil.copy(f, orig / f.name)
        cfgA = dict(base, steps=6, swap_every=3, eval_every=3)
        cfgB = dict(cfgA, eval_views=[0, 5], bank_source='data', mix=dict(cfgA['mix'], force_c=.1))
        cfgB0 = dict(cfgB, eval_views=[0])
        cfgE = dict(base, steps=8, swap_every=2, eval_every=4, ckpt_every=4, oh=True, quota=True, quota_systematic=True, ema=0.9,
                    ema_debias=True, select_min_step=5)
        t = time.perf_counter()
        P = dict(a_orig=launch(orig, dict(cfgA, out=str(tmp / 'a_orig')), tmp, 'a_orig'),
                 a_new=launch(HERE, dict(cfgA, out=str(tmp / 'a_new')), tmp, 'a_new'),
                 b_views=launch(HERE, dict(cfgB, out=str(tmp / 'b_views')), tmp, 'b_views'),
                 b_id=launch(HERE, dict(cfgB0, out=str(tmp / 'b_id')), tmp, 'b_id'))
        errs = wait(P)
        P = dict(e_full=launch(HERE, dict(cfgE, out=str(tmp / 'e_full')), tmp, 'e_full'),
                 e_stop=launch(HERE, dict(cfgE, out=str(tmp / 'e_res'), stop_after=4), tmp, 'e_stop'))
        # --- in-process checks meanwhile
        t_ema(); t_sat(); t_score_quota(); t_strict(base, tmp)
        errs.update(wait(P))
        shutil.copytree(tmp / 'e_res', tmp / 'e_old')                          # negative control: the old reseeding resume
        P = dict(e_res=launch(HERE, dict(cfgE, out=str(tmp / 'e_res'), resume=True), tmp, 'e_res'),
                 e_old=launch(HERE, dict(cfgE, out=str(tmp / 'e_old'), resume=True, resume_rng=False), tmp, 'e_old'))
        errs.update(wait(P))
        check('runs_completed', not errs, errors=errs, seconds=round(time.perf_counter() - t, 1))
        # --- a: defaults unchanged
        La, Lo = read_log(tmp / 'a_new' / 'train.log'), read_log(tmp / 'a_orig' / 'train.log')
        sa, so = steps_of(La), steps_of(Lo)
        ea, eo = ev(La, 'EVAL'), ev(Lo, 'EVAL')
        same_eval = len(ea) == len(eo) == 2 and all(x['score'] == y['score'] and x['weights']['raw']['val'] == y['weights']['raw']['val']
                                                     for x, y in zip(ea, eo))
        bn = torch.load(tmp / 'a_new' / 'best.pt', map_location='cpu', weights_only=False)
        bo = torch.load(tmp / 'a_orig' / 'best.pt', map_location='cpu', weights_only=False)
        same_w = bn['step'] == bo['step'] and all(torch.equal(bn['model'][k], bo['model'][k]) for k in bo['model'])
        check('a_defaults_identical_to_pre_change', len(sa) == 6 and sa == so and same_eval and same_w,
              steps_new=sa, steps_orig=so, scores=[(x['score'], y['score']) for x, y in zip(ea, eo)], best_step=bn['step'],
              new_keys=sorted(set(ea[0]) - set(eo[0])) if ea and eo else None)
        # --- b: views
        Lb, Li = read_log(tmp / 'b_views' / 'train.log'), read_log(tmp / 'b_id' / 'train.log')
        eb, ei = ev(Lb, 'EVAL'), ev(Li, 'EVAL')
        ok = len(eb) == len(ei) == 2
        rows = []
        for x, y in zip(eb, ei):
            sv = x['score_views']
            ok &= set(sv) == {'0', '5'} and x['score_identity'] == y['score'] == sv['0'] and \
                abs(x['score'] - (sv['0'] + sv['5']) / 2) < 1e-15 and x['score_worst_view'] == max(sv.values()) and \
                x['eval_views'] == [0, 5] and 'views' in x['weights']['raw'] and sv['5'] != sv['0']
            rows.append(dict(step=x['step'], identity=sv['0'], view5=sv['5'], score=x['score'], score_views0_run=y['score']))
        bb = torch.load(tmp / 'b_views' / 'last.pt', map_location='cpu', weights_only=False)
        ok &= bb.get('eval_views') == [0, 5] and set(bb.get('score_views', {})) == {'0', '5'} and 'conv' in bb
        ok &= steps_of(Lb) == steps_of(Li)
        mm = ev(Lb, 'MIX_MISSING')
        check('b_eval_views', ok, evals=rows, train_losses_equal=steps_of(Lb) == steps_of(Li))
        check('d_mix_missing_logged', mm and all(r['missing'] == ['force_c'] for r in mm) and
              {r['case'] for r in mm} == {train[0], train[2]}, logged=[(r['case'], r['missing']) for r in mm])
        # --- e: resume
        Lf, Lr = read_log(tmp / 'e_full' / 'train.log'), read_log(tmp / 'e_res' / 'train.log')
        sf, sr = steps_of(Lf), steps_of(Lr)
        rs = ev(Lr, 'RESUME')
        ef, er = ev(Lf, 'EVAL'), ev(Lr, 'EVAL')
        views_f = [(r['case'], r['k']) for r in ev(Lf, 'VIEW')]
        ok = rs and rs[0]['rng_restored'] and sr == sf and len(sf) == 8 and [x['score'] for x in er] == [x['score'] for x in ef] and \
            len(views_f) >= 4 and not ev(Lf, 'NONFINITE_LOSS')
        bf = torch.load(tmp / 'e_full' / 'best.pt', map_location='cpu', weights_only=False)
        br = torch.load(tmp / 'e_res' / 'best.pt', map_location='cpu', weights_only=False)
        ok &= bf['step'] == br['step'] == 8 and bf['score'] == br['score'] and [x['selectable'] for x in ef] == [False, True]
        so_ = steps_of(read_log(tmp / 'e_old' / 'train.log'))
        ok &= len(so_) == 8 and so_[:4] == sf[:4] and so_[4:] != sf[4:]
        check('e_resume_matches_uninterrupted', ok, losses_old_reseed_resume=[s_[1] for s_ in so_], losses_full=[s_[1] for s_ in sf], losses_resumed=[s_[1] for s_ in sr],
              resume=rs[0] if rs else None, views=views_f, eval_scores=[(x['step'], x['score'], x['selectable']) for x in ef],
              best_step=bf['step'])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(json.dumps(dict(summary='t_train3_v2', failed=FAIL, seconds=round(time.perf_counter() - T0, 1))), flush=True)
    sys.exit(1 if FAIL else 0)


# ------------------------------------------------------------------------------------------------ c EMA debias
def t_ema():
    torch.manual_seed(0)
    m = torch.nn.Linear(3, 2)
    d = 0.9
    e_plain, e_deb = T3.EMA(m, d), T3.EMA(m, d, debias=True)
    w0 = [p.detach().clone() for p in m.parameters()]
    cur_ok = all(torch.equal(a, b) for a, b in zip(e_deb.model_state(m).values(), m.state_dict().values()))
    its = []
    for t in range(5):
        with torch.no_grad():
            for p in m.parameters():
                p.add_(torch.randn_like(p))
        its.append([p.detach().clone() for p in m.parameters()])
        e_plain.update(m); e_deb.update(m)
    wts = np.array([(1 - d) * d ** (4 - s) for s in range(5)])
    ref = [sum(float(w) * it[i] for w, it in zip(wts / wts.sum(), its)) for i in range(2)]
    ref_plain = [d ** 5 * w0[i] + sum(float(w) * it[i] for w, it in zip(wts, its)) for i in range(2)]
    deb = e_deb.weights()
    err = max(float((a - b).abs().max()) for a, b in zip(deb, ref))
    errp = max(float((a - b).abs().max()) for a, b in zip(e_plain.weights(), ref_plain))
    sd = e_deb.state_dict()
    e2 = T3.EMA(m, d, debias=True); e2.load_state_dict(sd)
    rt = e2.t == 5 and all(torch.equal(a, b) for a, b in zip(e2.weights(), deb))
    try:
        T3.EMA(m, d).load_state_dict(sd); mode_guard = False
    except ValueError:
        mode_guard = True
    with e_deb.applied(m):
        inside = [p.detach().clone() for p in m.parameters()]
    back = all(torch.equal(p, q) for p, q in zip(m.parameters(), its[-1]))
    check('c_ema_debias', err < 1e-6 and errp < 1e-6 and cur_ok and rt and mode_guard and back and
          all(torch.allclose(a, b) for a, b in zip(inside, deb)), debias_err=err, plain_err=errp,
          init_weight_share_plain=d ** 5)


# ------------------------------------------------------------------------------------------------ f sat_stats knee
def t_sat():
    m = MD.MGNO.__new__(MD.MGNO)
    torch.nn.Module.__init__(m)
    m.bounded, m.bound_knee, m.levels, m._rec, m._sat, m.track_sat = True, 0.5, 1, None, {}, True
    m.register_buffer('bnd_ab', torch.tensor([[2.0], [float('inf')]]))
    m.register_buffer('bnd_rw', torch.tensor([[4.0, float('inf')]]))
    m.register_buffer('bnd_knee', torch.tensor(0.5, dtype=torch.float64))
    m._refresh_bounds()
    # x: (E=1, 27, 2, L=1, H=4): group alpha (A = 2, T = 1) gets |x| = 0.5, 1.5, 1.5, 3 on slot 0 (others 0); beta (inf)
    x = torch.zeros(1, 27, 2, 1, 4)
    x[0, 0, 0, 0] = torch.tensor([0.5, -1.5, 1.5, -3.0])
    x[0, :, 1, 0] = 100.0
    y = m._clip('ab', x, (1, 1, 2, 1, 1))
    m._transfer(torch.tensor([[0.0, 0.0], [10.0, 0.0], [1.5, 0.0]]), 0)     # softplus: .69, 10.00005, 1.70 vs A = 4, T = 2
    s = m.sat_stats()
    n = 27 * 4
    ok = abs(s['ab'][0][0] - 1 / n) < 1e-7 and abs(s['knee']['ab'][0][0] - 3 / n) < 1e-7 and s['ab'][1][0] == 0 and \
        s['knee']['ab'][1][0] == 0 and abs(s['rw'][0][0] - 1 / 3) < 1e-6 and abs(s['knee']['rw'][0][0] - 1 / 3) < 1e-6 and \
        abs(s['peak'] - float(torch.nn.functional.softplus(torch.tensor(10.0))) / 4) < 1e-6 and \
        abs(float(y[0, 0, 0, 0, 1]) + (1 + 1 * np.tanh(0.5))) < 1e-6 and abs(s['max'] - 1 / 3) < 1e-6
    # knee in the middle (2 > |x| > 1): counted by 'knee' only; the old counter saw 1 of 3 compressed values
    guards = []
    for bad in (dict(ab=torch.tensor([[0.0], [1.0]])), dict(ab=torch.tensor([[float('nan')], [1.0]])), dict(knee=1.0)):
        try:
            m.set_bounds(bad); guards.append(False)
        except ValueError:
            guards.append(True)
    try:
        MD._check_knee(-0.1); guards.append(False)
    except ValueError:
        guards.append(True)
    # a real MGNO2: knee buffer in the state dict, old dicts (no bnd_knee) keep the constructor knee, new ones carry theirs
    g = FX.small()
    mb = FX.model_for(g, CKPT.name, sparse=False, bounded=True, bound_knee=0.5)
    sd = mb.state_dict()
    old = {k: v for k, v in sd.items() if k != 'bnd_knee'}
    m2 = FX.model_for(g, CKPT.name, sparse=False, bounded=True, bound_knee=0.25)
    m2.load_state_dict(old)
    keep_ctor = m2.bound_knee == 0.25
    kept = MD.load_compat(m2, old)['kept']
    import warnings
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')
        m2.load_state_dict(sd)
    carried = m2.bound_knee == 0.5 and float(m2.bnd_knee) == 0.5
    mb.set_bounds(dict(bounds=dict(ab=torch.full((2, 8), 3.0)), knee=0.3))
    fk = mb.bound_knee == 0.3 and float(mb.bnd_knee) == 0.3
    check('f_sat_stats_knee', ok and all(guards) and 'bnd_knee' in sd and keep_ctor and 'bnd_knee' in kept and carried and fk,
          sat=dict(ab=s['ab'], knee_ab=s['knee']['ab'], rw=s['rw'], knee_rw=s['knee']['rw'], peak=s['peak'], max=s['max'],
                   knee_max=s['knee_max']), guards=guards, old_dict_keeps_ctor_knee=keep_ctor, loaded_knee_carried=carried)


# ------------------------------------------------------------------------------------------------ g score / quota
def t_score_quota():
    fam = lambda c: c.rsplit('_', 1)[0]
    ok_r = dict(mean=0.1, p90=0.2, max=0.3, sens_mean=0.01, sens_p90=0.02)
    per = {'f1_a': dict(force=ok_r), 'f1_b': dict(force=dict(ok_r, mean=float('nan'))), 'f2_a': dict(force=ok_r)}
    sc, parts = T3.score(T3.families(per, fam), ['force'])
    per2 = {'f1_a': dict(force=ok_r), 'f2_a': dict(force=ok_r)}
    sc2, _ = T3.score(T3.families(per2, fam), ['force'])
    sc3, p3 = T3.score(T3.families(per2, fam), ['force'], sens_p90=True)
    ok = sc == float('inf') and parts['f1']['total'] == float('inf') and abs(sc2 - (0.1 + 0.01 + 0.1)) < 1e-12 and \
        abs(sc3 - (sc2 + 0.01)) < 1e-12
    w = np.array([.2, .15, .15, .15, .1, .1, .05, .05, .05])
    g1, g2 = np.random.default_rng(0), np.random.default_rng(0)
    a = np.mean([TL.quota_counts(16, w, g1) for _ in range(20000)], 0) / (w * 16)
    b = np.mean([TL.quota_counts(16, w, g2, systematic=True) for _ in range(20000)], 0) / (w * 16)
    check('g_score_nonfinite_and_quota', ok and abs(b - 1).max() < 0.03 and a[-1] < 0.93,
          score_with_nan_geo=sc, score_finite=sc2, score_sens_p90=sc3, share_ratio_old=a.round(3).tolist(),
          share_ratio_systematic=b.round(3).tolist())


# ------------------------------------------------------------------------------------------------ d strict mix
def t_strict(base, tmp):
    res = {}
    for name, cfg in (('absent', dict(base, mix=dict(base['mix'], glued=.1))),
                      ('strict', dict(base, mix=dict(base['mix'], force_c=.1), bank_source='data', strict_mix=True))):
        cfg = dict(cfg, out=str(tmp / f'd_{name}'), steps=2, eval_every=2)
        try:
            T3.main(cfg) if False else _quiet(T3.main, cfg)
            res[name] = 'no error'
        except ValueError as e:
            res[name] = str(e)[:160]
    check('d_strict_mix_raises', res['absent'].startswith('MIX_ABSENT') and 'glued' in res['absent'] and
          res['strict'].startswith('MIX_CLASSES_MISSING') and 'force_c' in res['strict'], errors=res)
    # support64 (prep_geo2 --out): a known class (after the old ones), read from a data dir, accepted in mix / score_classes
    d = tmp / 's64' / 'fresh_train_0010_cover01_r1'
    d.mkdir(parents=True)
    src = Path(base['data']) / 'fresh_train_0010_cover01_r1'
    for sp_ in TL.SPLITS:
        for c, f in (('force', 'force'), ('support64', 'support')):
            for suf in ('', '_sens'):
                if (src / f'{sp_}_{f}{suf}.npy').exists():
                    shutil.copy(src / f'{sp_}_{f}{suf}.npy', d / f'{sp_}_{c}{suf}.npy')
    hb = TL.HostBanks.from_data(d)
    order_ok = list(TL.ALL_CLASSES[:9]) == ['force', 'macro', 'grf', 'support', 'face', 'force_c', 'face_c', 'support_k', 'glued'] \
        and TL.ALL_CLASSES[-1] == 'support64'
    try:
        _quiet(T3.main, dict(base, out=str(tmp / 'd_s64'), steps=2, eval_every=2, mix=dict(base['mix'], support64=.1),
                             score_classes=['force', 'support64']))
        e64 = 'no error'
    except ValueError as e:
        e64 = str(e)[:160]
    check('d_support64_class', hb.classes == ['force', 'support64'] and order_ok and e64.startswith('MIX_ABSENT') and
          'support64' in e64, classes=hb.classes, slot_source_error=e64)


def _quiet(fn, *a):
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a)


if __name__ == '__main__':
    main()
