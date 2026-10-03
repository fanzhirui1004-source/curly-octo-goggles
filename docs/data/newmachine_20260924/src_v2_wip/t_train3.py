"""Local CPU tests of the trainer v2 (train3.py) and its trainlib pieces on the fixture (fixture.py; no GPU, no cuDSS).

 1 quota_counts: sums, floor / floor + 1 per class, seeded, remainder frequencies ~ fractional parts
 2 HostBanks sampling == Geo.sample_with_sens (device banks) for the same generator (incl. the adversarial buffer); the
   default-config train3 loss and its gradients on that batch == train2's loss op for op; HostBanks.clean ==
   train2.clean_banks (sample drop and class removal), data-dir banks (memory-mapped) == slot banks
 3 sens_loss: 'sq' == the train2 expression; 'smoothl1' value and gradcheck (fp64) at moderate and tiny rho
 4 gram / tail_loss: gradcheck (fp64 toy), eigenvalues == scipy generalized eigh, floor keeps a singular G finite
 5 Geo.bank_ritz with the fixture's exact reactions (fix_react_val as F): unit exact energy, Ritz value == e_hat of the
   returned direction, top >= every single sample
 6 Geo.worst_ratio (mu) with a scipy factor stand-in (teacher.SPDSolver -> splu, as t_diag_b.py)
 7 Slot view bookkeeping: shared attributes re-pointed after a (simulated) device move, own ones kept
 8 train3.main end to end on a fake split (the small geometry and aliases of it as train / val geometries, slot cache in a
   temp dir, data dir with the fixture banks and F): run A (data banks, F, quota, EMA, smooth-L1, tail loss, O_h stand-in,
   adversarial first visit with the scipy factor, bank-span Ritz on later visits, mu, certificate, probes, snapshots,
   stop_after 20) -> run B (resume to 30, DONE) -> run C (train2-like defaults, slot banks, DOF-capped pool with EVICT);
   oh = true without oh.py fails loudly
Remote (GPU, not run here): the device-side checks with explicit pass criteria, see remote().
Usage: OPL_DEV=cpu python3 t_train3.py   (T3_REAL_OH=1: also a short run with the real oh.py if importable)
       python3 t_train3.py --remote [--split S] [--slots D] [--data D] [--ckpt C] [--out D]   (remote GPU)
"""
import os
import sys
REMOTE = '--remote' in sys.argv
if not REMOTE:
    os.environ['OPL_DEV'] = 'cpu'
import json, time, copy, types, shutil, tempfile, contextlib, io
from pathlib import Path
import numpy as np
import torch
import scipy.sparse as sp
import scipy.linalg as sla
from scipy.sparse.linalg import splu

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fixture as FX

for _ in range(0 if REMOTE else 120):                                                 # the fixture may still be downloading
    if (FX.FIX / 'geo_small.pt').exists() and (FX.FIX / 's2v1_snap_10000.pt').exists():
        break
    print('waiting for the fixture ...', flush=True); time.sleep(30)
import teacher as TE
import trainlib as TL
import train2 as T2
import train3 as T3

dt = torch.float64
FAIL, T0 = [], time.perf_counter()
CKPT = FX.FIX / 's2v1_snap_10000.pt'
SCR = Path(os.environ.get('T3_TMP', HERE.parent.parent))


def check(name, ok, **info):
    print(json.dumps(dict(test=name, ok=bool(ok), **{k: (round(v, 10) if isinstance(v, float) else v) for k, v in info.items()}),
                     default=str), flush=True)
    if not ok:
        FAIL.append(name)


class SciSPD:
    """teacher.SPDSolver interface on scipy's sparse LU (fp64); fdt = fp32 rounds the solution (fp32 Neumann factor)."""

    def __init__(self, crow, col, vals, n, w=16, threads=16, fdt=None):
        self.fdt = vals.dtype if fdt is None else fdt
        U = sp.csr_matrix((vals.double().cpu().numpy(), col.long().cpu().numpy(), crow.long().cpu().numpy()), shape=(n, n))
        self.lu = splu((U + U.T - sp.diags(U.diagonal())).tocsc())

    def solve(self, r):
        return torch.as_tensor(self.lu.solve(r.double().cpu().numpy())).to(self.fdt).to(r.dtype)

    def free(self):
        self.lu = None


if not REMOTE:
    TE.SPDSolver = SciSPD                                                             # test-only stand-ins (no cuDSS here)
    TE.sync = lambda: None
    torch.set_num_threads(int(os.environ.get('T3_THREADS', '2')))                     # other CPU jobs: oversubscribed OpenMP barriers


def nan_equal(a, b):
    return bool(torch.equal(torch.isnan(a), torch.isnan(b)) and torch.equal(torch.nan_to_num(a), torch.nan_to_num(b)))


def clone_banks(d):
    return {sp_: {c: v.clone() for c, v in x.items()} for sp_, x in d.items()}


# ------------------------------------------------------------------------------------------------ 1 quota
def t_quota():
    w = np.array([.25, .15, .2, .1, .15, .15])
    ok, fr = True, np.zeros(len(w))
    for s in range(400):
        n = TL.quota_counts(16, w, np.random.default_rng(s))
        ok &= n.sum() == 16 and bool(((n == np.floor(w * 16)) | (n == np.floor(w * 16) + 1)).all())
        fr += n - np.floor(w * 16)
    same = np.array_equal(TL.quota_counts(16, w, np.random.default_rng(3)), TL.quota_counts(16, w, np.random.default_rng(3)))
    x = w * 16 - np.floor(w * 16)
    check('quota_counts', ok and same and np.abs(fr / 400 - x).max() < 0.08, remainder_freq=(fr / 400).round(3).tolist(),
          fractional=x.round(3).tolist())


# ------------------------------------------------------------------------------------------------ 2 sampling / loss / clean
def train2_loss(geo, model, q, s0, sw=1.0):
    """train2.main's loss lines, verbatim."""
    u = geo.field(model, q)
    e = TL.energy(u, geo.C.K)
    loss = torch.log(e.clamp_min(1e-12)).mean()
    ok = ~torch.isnan(s0[0]); ls = None
    if sw > 0 and ok.any():
        sh = geo.sens_hat(u[:, ok])
        ls = (((sh - s0[:, ok]) ** 2).sum(0) / (s0[:, ok] ** 2).sum(0)).mean()
        loss = loss + sw * ls
    return loss, ls


def t_sampling_loss(g, model):
    mix = {'force': .25, 'support': .15, 'face': .2, 'macro': .1, 'grf': .15, 'adv': .15}
    g.adv = torch.randn((g.np_, 5), generator=torch.Generator().manual_seed(3)).to(torch.float32)
    hb = TL.HostBanks(clone_banks(g.banks), clone_banks(g.sens))
    ok = True
    for s in range(5):
        q1, s1 = g.sample_with_sens(8, np.random.default_rng(s), mix)
        q2, s2, kinds, F = hb.sample(8, np.random.default_rng(s), mix, adv=g.adv)
        ok &= torch.equal(q1, q2) and nan_equal(s1, s2) and F is None and len(kinds) == 8
    check('host_sampling_equals_device', ok)
    q1, s1 = g.sample_with_sens(6, np.random.default_rng(11), mix)
    q2, s2, kinds, _ = hb.sample(6, np.random.default_rng(11), mix, adv=g.adv)
    nt = torch.get_num_threads()
    torch.set_num_threads(1)                    # multi-threaded CPU reductions differ run to run (~1e-6) even for identical code
    try:
        model.zero_grad(set_to_none=True)
        l1, ls1 = train2_loss(g, model, q1, s1); l1.backward()
        g1 = [p.grad.clone() for p in model.parameters()]
        model.zero_grad(set_to_none=True)
        l2, ls2, tl, e, lam = T3.batch_loss(g, model, q2, s2, kinds, None, {})
        l2.backward()
        g2 = [p.grad.clone() for p in model.parameters()]
        model.zero_grad(set_to_none=True)
    finally:
        torch.set_num_threads(nt)
    check('default_loss_equals_train2', float(l1) == float(l2) and float(ls1) == float(ls2) and tl is None and
          all(torch.equal(a, b) for a, b in zip(g1, g2)), loss_train2=float(l1), loss_train3=float(l2), adv_cols=kinds.count('adv'))
    # clean: sample drop + class removal, same as train2.clean_banks
    gd = copy.copy(g); gd.banks = clone_banks(g.banks); gd.sens = clone_banks(g.sens); gd.classes = list(g.classes)
    gd.banks['train']['support'][5, 3] = float('nan'); gd.sens['val']['face'][2, 7] = float('inf')
    gd.banks['test']['grf'][:, :10] = float('nan')                                  # grf test keeps 6 < 8 -> class removed
    hb2 = TL.HostBanks(clone_banks(gd.banks), clone_banks(gd.sens))
    rec = []
    T2.clean_banks(gd, 'x', rec.append)
    d2 = hb2.clean('x')
    mix2 = {k: v for k, v in mix.items() if k != 'adv'}
    same = all(torch.equal(gd.sample_with_sens(8, np.random.default_rng(s), mix2)[0],
                           hb2.sample(8, np.random.default_rng(s), mix2)[0]) for s in range(4))
    check('clean_equals_train2', rec and rec[0]['dropped'] == d2 and gd.classes == hb2.classes and same, dropped=d2)
    g.adv = None
    return hb


def write_data(d, g, F_classes=('force', 'macro', 'grf', 'support', 'face'), extra=None, nan_F=None):
    """Data-dir banks from the fixture: every split = the val bank (16 samples; the fixture's exact reactions belong to it)."""
    d.mkdir(parents=True, exist_ok=True)
    (d / 'DONE.json').write_text(json.dumps(dict(case=d.name, dofs=g.nb)))
    for c in g.classes:
        for name in [c] + ([extra] if extra and c == 'force' else []):
            for sp_ in TL.SPLITS:
                np.save(d / f'{sp_}_{name}.npy', g.banks['val'][c].T.contiguous().numpy())
                np.save(d / f'{sp_}_{name}_sens.npy', g.sens['val'][c].T.contiguous().numpy())
                if c in F_classes:
                    F = g.fix_react_val[c].T.contiguous().to(torch.float32).numpy()
                    if nan_F == (sp_, name):
                        F = F.copy(); F[3, 11] = np.nan
                    np.save(d / f'{sp_}_{name}_F.npy', F)


def t_data_banks(g, tmp):
    d = tmp / 'dcheck' / g.case
    write_data(d, g, F_classes=('force', 'macro', 'grf', 'support'), extra='force_c', nan_F=('train', 'support'))
    hb = TL.HostBanks.from_data(d)
    dropped = hb.clean('x')
    hs = TL.HostBanks({sp_: {c: g.banks['val'][c].clone() for c in g.classes} for sp_ in TL.SPLITS},
                      {sp_: {c: g.sens['val'][c].clone() for c in g.classes} for sp_ in TL.SPLITS})
    mix = {'force': .3, 'macro': .2, 'grf': .2, 'face': .3}
    same = all(torch.equal(hb.sample(8, np.random.default_rng(s), mix)[0], hs.sample(8, np.random.default_rng(s), mix)[0])
               for s in range(4))
    q, s0, kinds, F = hb.sample(8, np.random.default_rng(0), {'force': .5, 'macro': .5}, want_F=True)
    qn, _, _, Fn = hb.sample(8, np.random.default_rng(0), {'force': .5, 'face': .5}, want_F=True)   # face has no F
    unit = float(((q.to(dt) * F).sum(0) - 1).abs().max())
    check('data_banks_mmap', isinstance(hb.q['train']['force'], np.memmap) and 'force_c' in hb.classes and same and
          dropped == {'train/support': 1} and hb.m('train', 'support') == 15 and Fn is None and unit < 1e-6,
          classes=hb.classes, dropped=dropped, unit_energy_err=unit)


# ------------------------------------------------------------------------------------------------ 3 / 4 loss pieces
def t_sens_loss():
    torch.manual_seed(0)
    s0 = torch.randn(8, 6, dtype=dt)
    sh = (s0 + 0.05 * torch.randn(8, 6, dtype=dt)).requires_grad_(True)
    ref = (((sh - s0) ** 2).sum(0) / (s0 ** 2).sum(0)).mean()
    rho = (sh - s0).norm(dim=0) / s0.norm(dim=0)
    v = TL.sens_loss(sh, s0, 'smoothl1', 0.003)
    ok = float(TL.sens_loss(sh, s0, 'sq')) == float(ref) and abs(float(v) - float((torch.sqrt(rho ** 2 + 0.003 ** 2) - 0.003).mean())) < 1e-15
    g1 = torch.autograd.gradcheck(lambda x: TL.sens_loss(x, s0, 'smoothl1', 0.003), (sh,), eps=1e-7, atol=1e-6)
    tiny = (s0 + 1e-5 * torch.randn(8, 6, dtype=dt)).requires_grad_(True)
    g2 = torch.autograd.gradcheck(lambda x: TL.sens_loss(x, s0, 'smoothl1', 0.003), (tiny,), eps=1e-9, atol=1e-6)
    # gradient share near convergence (rho = 1e-3): smooth-L1 / sq = 1 / (2 sqrt(rho^2 + delta^2))
    x = (s0 * (1 + 1e-3)).requires_grad_(True)
    ga = torch.autograd.grad(TL.sens_loss(x, s0, 'smoothl1'), x)[0].norm(); gb = torch.autograd.grad(TL.sens_loss(x, s0, 'sq'), x)[0].norm()
    check('sens_loss', ok and g1 and g2 and abs(float(ga / gb) - 1 / (2 * np.sqrt(1e-6 + 9e-6))) < 1e-3 * float(ga / gb),
          grad_ratio_at_rho_1e3=float(ga / gb))


def t_tail():
    torch.manual_seed(1)
    n, B = 14, 5
    A = torch.randn(n, n, dtype=dt); K = A @ A.T + n * torch.eye(n, dtype=dt)
    A2 = torch.randn(n, n, dtype=dt); K0 = 0.5 * K + 0.05 * A2 @ A2.T                    # an "exact" energy below K on U
    U = torch.randn(n, B, dtype=dt, requires_grad=True)
    G = (U.detach().T @ K0 @ U.detach())
    g1 = torch.autograd.gradcheck(lambda x: TL.gram(x, K), (U,))
    g2 = torch.autograd.gradcheck(lambda x: TL.tail_loss(TL.gram(x, K), G, tau=0.1)[0], (U,), eps=1e-6, atol=1e-5)
    Gh = TL.gram(U, K).detach()
    lam = TL.tail_loss(Gh, G)[1]
    ref = sla.eigh(Gh.numpy(), G.numpy(), eigvals_only=True)
    ev_err = float(np.abs(np.sort(lam.numpy()) - np.sort(ref)).max() / ref.max())
    Gs = G.clone(); Gs[:, -1] = Gs[:, 0]; Gs[-1, :] = Gs[0, :]                         # duplicated column: singular G
    t_s, lam_s = TL.tail_loss(Gh, Gs)
    term = TL.tail_loss(Gh, G, tau=0.1)[0]
    check('tail_loss', g1 and g2 and ev_err < 1e-10 and bool(torch.isfinite(lam_s).all()) and bool(torch.isfinite(t_s)) and
          float(term) >= float(np.log(ref.max())) - 1e-12, eig_err=ev_err, lam=ref.round(4).tolist(), term=float(term))


# ------------------------------------------------------------------------------------------------ 5 / 6 bank-span Ritz, mu
@torch.no_grad()
def e_hat(g, model, X, chunk=16):
    return torch.cat([TL.energy(g.field(model, X[:, j:j + chunk].to(torch.float32)), g.C.K) for j in range(0, X.shape[1], chunk)])


def t_bank_ritz(g, model):
    cl = ('force', 'support', 'face')
    Q = torch.cat([g.banks['val'][c] for c in cl], 1)
    F = torch.cat([g.fix_react_val[c] for c in cl], 1)
    unit = float(((Q.to(dt) * F).sum(0) - 1).abs().max())
    t = time.perf_counter()
    X, r = g.bank_ritz(model, Q, F, top=4)
    sec = time.perf_counter() - t
    C = torch.linalg.lstsq(Q.to(dt), X).solution
    ex = (C * (Q.to(dt).T @ F @ C)).sum(0)                                             # exact energies X^T S X = c^T Q^T F c
    eh = e_hat(g, model, X)
    es = e_hat(g, model, Q)
    ok = float((ex - 1).abs().max()) < 1e-6 and float(((eh - r) / r).abs().max()) < 1e-3 and float(r[0]) >= float(es.max()) * (1 - 1e-3) \
        and bool((r[:-1] >= r[1:]).all())
    check('bank_ritz', ok, bank_unit_energy_err=unit, exact_energy_err=float((ex - 1).abs().max()),
          ritz=r.tolist(), e_hat_of_X=eh.tolist(), max_single=float(es.max()), seconds=sec)
    return float(r[0])


def t_mu(g, model, r_bank):
    t = time.perf_counter()
    g.C.factor(neumann=True, interior=False, fp32_neumann=True)
    try:
        mu, it, X = g.worst_ratio(model, k=2, tol=1e-3, max_iters=4, gen=torch.Generator().manual_seed(7))
        mu1 = g.worst_ratio(model, k=2, tol=1e-3, max_iters=1, gen=torch.Generator().manual_seed(7))[0]
    finally:
        g.C._free(); g.C.sol_N = None
    check('worst_ratio_mu', np.isfinite(mu) and mu >= 1 - 1e-9 and 1 <= it <= 4 and mu >= mu1 * (1 - 1e-6), mu=mu, iters=it,
          mu_after_1_iter=mu1, bank_ritz_top=r_bank, seconds=time.perf_counter() - t)


# ------------------------------------------------------------------------------------------------ 7 view bookkeeping
def fake_oh():
    m = types.ModuleType('oh')
    m.ELEMS = [np.eye(3, dtype=int) for _ in range(48)]
    m.made, m.dropped = [], []

    def view(model, geo, k):
        v = copy.copy(geo)
        v.case, v.nd, v.oh_R = f'{geo.case}@oh{k}', dict(geo.nd), m.ELEMS[k]
        model.add_geo(v); m.made.append(v.case)
        return v

    def drop(model, v):
        model.caches.pop(v.case, None); m.dropped.append(v.case)
    m.view, m.drop = view, drop
    return m


def t_view_move(g, model, tmp):
    OH = fake_oh()
    geo = copy.copy(g)
    hb = TL.HostBanks(clone_banks(g.banks), clone_banks(g.sens))
    s = T3.Slot(g.case, dict(), model, lambda d_: None, host=(geo, hb))
    s.set_view(OH, model, 5)

    moved = set()

    def clone_move(obj, device, seen=None):                  # a "device move": new tensors, except those already moved
        for k, v in list(vars(obj).items()):
            if torch.is_tensor(v) and id(v) not in moved:
                v = v.clone(); moved.add(id(v)); setattr(obj, k, v)
        return obj
    real = T3.move
    T3.move = clone_move
    try:
        old = s.view.RP
        s._move(T3.CPU)
    finally:
        T3.move = real
    ok = s.view.RP is s.geo.RP and s.view.RP is not old and s.view.case.endswith('@oh5') and s.view.nd is not s.geo.nd and \
        s.view.C is s.geo.C and 'RP' in s._shared and 'case' not in s._shared
    s.drop(model, OH)
    gone = g.case not in model.caches
    model.add_geo(g)                                                                    # the other tests keep using this model
    check('slot_view_repoint', ok and OH.dropped == OH.made and gone, shared=len(s._shared))


# ------------------------------------------------------------------------------------------------ 8 end to end
def read_log(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def run(cfg, quiet=True):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf) if quiet else contextlib.nullcontext():
        T3.main(cfg)


def t_end_to_end(g, tmp):
    REAL = g.case
    train = [REAL, 'fresh_train_0011_alias_a', 'fresh_train_0012_alias_b', 'fresh_train_0012_alias_c']
    val = ['fresh_train_0013_alias_v', REAL]
    slots, data = tmp / 'slots', tmp / 'data'
    slots.mkdir(parents=True, exist_ok=True)
    for c in sorted(set(train + val)):
        x = copy.copy(g); x.case = c
        for a in ('fix_ustar_val', 'fix_react_val'):
            x.__dict__.pop(a, None)
        torch.save(x, slots / f'{c}.pt')
        write_data(data / c, g, extra='force_c' if c == train[1] else None, nan_F=('train', 'support') if c == train[1] else None)
    split = dict(train=train, val=val, test=[], train_families=['fresh_train_0010', 'fresh_train_0011', 'fresh_train_0012'],
                 val_families=['fresh_train_0013', 'fresh_train_0010'])
    (tmp / 'SPLIT.json').write_text(json.dumps(split))
    ck = FX.checkpoint(CKPT.name)
    margs = dict(ck['cfg']['model_args']); margs['sparse'] = False
    base = dict(split=str(tmp / 'SPLIT.json'), train_key='train', body=str(tmp / 'nobody'), data=str(data), slot_cache=str(slots),
                model='mgno2', model_args=margs, init=str(CKPT), batch=4, geos_per_step=1, lr=1e-3, pct_start=0.1, seed=0,
                sens_w=1.0, packets=str(tmp / 'nopackets'))
    # ---- run A: every v2 option, stopped after step 20
    outA = tmp / 'runA'
    cfgA = dict(base, out=str(outA), steps=30, val_max=2, probe_max=1, pool=2, pool_dofs=0, swap_every=5, adv_on_load=True,
                adv_k=4, adv_iters=2, adv_buffer=8, adv_ritz_R=16, adv_ritz_top=4, eval_every=10, ckpt_every=10, stop_after=20,
                mix={'force': .2, 'support': .15, 'face': .15, 'macro': .1, 'grf': .1, 'force_c': .15, 'adv': .15},
                bank_source='data', quota=True, ema=0.9, sens_loss='smoothl1', tail_w=0.3, oh=True, mu_geos=1, mu_k=2, mu_iters=2,
                cert_m=2, val_chunk=16, log_every=5)
    OH = fake_oh(); sys.modules['oh'] = OH
    t = time.perf_counter(); run(cfgA); tA = time.perf_counter() - t
    LA = read_log(outA / 'train.log')
    ev = lambda L, k: [r for r in L if r.get('event') == k]
    advs = ev(LA, 'ADV')
    steps = ev(LA, 'STEP')
    evals = ev(LA, 'EVAL')
    ok = bool(ev(LA, 'WARN')) and bool(ev(LA, 'STOP')) and not ev(LA, 'DONE') and not ev(LA, 'NONFINITE_LOSS') \
        and not ev(LA, 'ADV_FAIL') and not ev(LA, 'MU_FAIL') and not ev(LA, 'PREFETCH_FAIL')
    check('runA_events', ok, seconds=tA, events=sorted({r['event'] for r in LA}))
    first = [r for r in advs if r['visit'] == 'first']; later = [r for r in advs if r['visit'] == 'later']
    check('runA_adversarial', len(first) == 4 and later and all('bank_ritz_top' in r for r in later) and all(r['size'] <= 8 for r in later),
          first=[r['ritz_top'][:1] for r in first], later=[(r.get('bank_ritz_top', [None])[:1], r.get('added')) for r in later])
    builds = ev(LA, 'BUILD')
    check('runA_prefetch_views', any(r['prefetched'] for r in builds) and len(ev(LA, 'VIEW')) >= 6 and
          len(OH.made) - len(OH.dropped) == 2 and any('force_c' in r['classes'] for r in builds) and
          any(r['case'] == train[1] and r['classes'] and 'support' in r['F'] for r in builds) and bool(ev(LA, 'BANK_CLEAN')),
          prefetched=[r['prefetched'] for r in builds], views=len(OH.made), dropped=len(OH.dropped))
    quota = {k: (int(np.floor(v / sum(cfgA['mix'].values()) * 4)), int(np.floor(v / sum(cfgA['mix'].values()) * 4)) + 1)
             for k, v in cfgA['mix'].items()}
    check('runA_steps', [r['step'] for r in steps] == [5, 10, 15, 20] and all(np.isfinite(r['tail']) and r['lam_max'] >= 1 - 1e-3 for r in steps)
          and all(sum(r['counts'].values()) == 4 and all(quota[k][0] <= n <= quota[k][1] for k, n in r['counts'].items()) for r in steps)
          and all('view' in r for r in steps), steps=[{k: r[k] for k in ('step', 'loss', 'sens_loss', 'tail', 'lam_max', 'counts')} for r in steps])
    e10 = evals[0]
    w = e10['weights']
    fam_ok = set(w['ema']['val_family']) == {'fresh_train_0013', 'fresh_train_0010'} and w['ema']['mu'] and 'cert' in w['ema']
    sc, parts = T3.score(w['ema']['val_family'], [c for c in cfgA['mix'] if c != 'adv'], split['val_families'])
    check('runA_eval', [r['step'] for r in evals] == [10, 20] and e10['select'] == 'ema' and 'raw' in w and fam_ok and
          abs(sc - e10['score']) < 1e-12 and set(e10['since_visit']) == set(ev(LA, 'PROBES')[0]['cases']) and
          w['raw']['score'] != w['ema']['score'],
          score=e10['score'], score_raw=w['raw']['score'], parts=e10['score_parts'], mu=w['ema']['mu'], cert=w['ema']['cert'],
          since=e10['since_visit'], val_mean=e10['val_mean'])
    snaps = sorted(p.name for p in outA.glob('snap_*.pt'))
    b = torch.load(outA / 'best.pt', map_location='cpu', weights_only=False)
    ckA = torch.load(outA / 'ckpt.pt', map_location='cpu', weights_only=False)
    diff = max(float((b['model'][k] - b['model_raw'][k]).abs().max()) for k in b['model'] if b['model'][k].is_floating_point())
    check('runA_checkpoints', snaps == ['snap_10.pt', 'snap_20.pt'] and b['weights'] == 'ema' and diff > 0 and ckA['step'] == 20 and
          ckA['ema'] is not None, snaps=snaps, ema_minus_raw=diff)
    # ---- run B: resume to 30
    t = time.perf_counter(); run(dict(cfgA, resume=True, stop_after=None)); tB = time.perf_counter() - t
    LB = read_log(outA / 'train.log')[len(LA):]
    rs = ev(LB, 'RESUME')
    check('runB_resume', rs and rs[0]['step'] == 20 and rs[0]['ema'] and [r['step'] for r in ev(LB, 'EVAL')] == [30] and
          bool(ev(LB, 'DONE')) and (outA / 'snap_30.pt').exists() and
          torch.load(outA / 'ckpt.pt', map_location='cpu', weights_only=False)['step'] == 30, seconds=tB, qi=rs[0]['qi'] if rs else None)
    # ---- run C: train2-like defaults (slot banks, no adversarial / EMA / quota / O_h), DOF-capped pool -> EVICT
    outC = tmp / 'runC'
    cfgC = dict(base, out=str(outC), steps=8, val_max=1, probe_max=0, pool=2, pool_dofs=20000, swap_every=4, eval_every=8,
                mix={'force': .25, 'support': .15, 'face': .2, 'macro': .1, 'grf': .15, 'adv': .15}, mu_geos=0, cert=False)
    del sys.modules['oh']
    t = time.perf_counter(); run(cfgC); tC = time.perf_counter() - t
    LC = read_log(outC / 'train.log')
    bc = torch.load(outC / 'best.pt', map_location='cpu', weights_only=False)
    eC = ev(LC, 'EVAL')
    check('runC_defaults', bool(ev(LC, 'EVICT')) and bool(ev(LC, 'DONE')) and eC and list(eC[0]['weights']) == ['raw'] and
          bc['weights'] == 'raw' and 'model_raw' not in bc and not ev(LC, 'ADV') and not ev(LC, 'VIEW') and
          all(r['F'] == [] for r in ev(LC, 'BUILD')), seconds=tC, evict=len(ev(LC, 'EVICT')), score=eC[0]['score'] if eC else None)
    # ---- oh = true without oh.py: fail loudly
    sys.modules['oh'] = None
    try:
        run(dict(cfgC, out=str(tmp / 'runD'), oh=True))
        loud = False
    except ImportError as e:
        loud = 'oh.py' in str(e)
    finally:
        del sys.modules['oh']
    check('oh_missing_fails_loudly', loud)
    if os.environ.get('T3_REAL_OH') == '1':
        try:
            import oh as REAL_OH  # noqa: F401
            run(dict(cfgA, out=str(tmp / 'runE'), steps=6, stop_after=None, eval_every=6, swap_every=3, probe_max=0, mu_geos=0))
            LE = read_log(tmp / 'runE' / 'train.log')
            check('real_oh_run', bool(ev(LE, 'DONE')) and len(ev(LE, 'VIEW')) >= 3 and not ev(LE, 'NONFINITE_LOSS'))
        except ImportError:
            check('real_oh_run', True, skipped='oh.py not importable')


def t_batch_options(g, model, tmp):
    """quota counts in a real batch and the tail term on the train path (F from the data dir)."""
    hb = TL.HostBanks.from_data(tmp / 'data' / 'fresh_train_0011_alias_a')
    hb.clean()
    mix = {'force': .2, 'support': .15, 'face': .15, 'macro': .1, 'grf': .1, 'force_c': .15, 'adv': .15}
    adv = torch.randn((g.np_, 3), generator=torch.Generator().manual_seed(0)).to(torch.float32)
    q, s0, kinds, F = hb.sample(16, np.random.default_rng(4), mix, adv=adv, quota=True, want_F=True)
    cnt = {k: kinds.count(k) for k in mix}
    base = {k: int(np.floor(mix[k] / sum(mix.values()) * 16)) for k in mix}
    cfg = dict(tail_w=0.3, sens_loss='smoothl1')
    model.zero_grad(set_to_none=True)
    loss, ls, tl, e, lam = T3.batch_loss(g, model, q, s0, kinds, F, cfg)
    loss.backward()
    gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9))
    model.zero_grad(set_to_none=True)
    e0 = TL.energy(g.field(model, q), g.C.K).detach()
    adv_m = [k == 'adv' for k in kinds]
    check('batch_quota_tail', all(base[k] <= cnt[k] <= base[k] + 1 for k in mix) and sum(cnt.values()) == 16 and tl is not None and
          np.isfinite(float(tl)) and np.isfinite(gn) and bool(torch.isnan(F[:, torch.as_tensor(adv_m)]).all()) and
          float(((e - e0) / e0).abs().max()) < 1e-12 and float(lam.max()) >= float(e0[~torch.as_tensor(adv_m)].max()) * (1 - 1e-2),
          counts=cnt, tail=float(tl), lam_max=float(lam.max()), grad_norm=gn)


def main():
    tmp = Path(tempfile.mkdtemp(prefix='t_train3_', dir=str(SCR)))
    try:
        t_quota(); t_sens_loss(); t_tail()
        g = FX.small()
        model = FX.model_for(g, CKPT.name, sparse=False)
        t_sampling_loss(g, model)
        t_data_banks(g, tmp)
        r = t_bank_ritz(g, model)
        t_mu(g, model, r)
        t_view_move(g, model, tmp)
        t_end_to_end(g, tmp)
        t_batch_options(g, model, tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(json.dumps(dict(summary='t_train3', failed=FAIL, seconds=time.perf_counter() - T0)), flush=True)
    sys.exit(1 if FAIL else 0)


# ------------------------------------------------------------------------------------------------ remote (GPU)
def remote():
    """GPU checks on the remote (run after deployment; not run locally). Pass criteria:
     R1 Slot device round trip on the smallest training geometry (with an O_h view when oh.py imports): after to(cpu) and
        to(cuda) every tensor of the geo, its cell, the view and both caches is on cuda and the view's shared attributes are
        the geo's; after drop torch.cuda.memory_allocated() <= the level before the slot + 64 MB
     R2 HostBanks.sample == Geo.sample_with_sens with device banks (bitwise, 3 seeds, with an adversarial buffer)
     R3 default batch_loss == train2's loss on a fixed batch (relative 1e-5; CUDA atomics are not bitwise reproducible)
     R4 train3.main on the 4 smallest training geometries (pool 2, swap every 10, 60 steps, adv_on_load with the real fp32
        Neumann factor, EMA, quota, O_h if importable, mu on 1 val geometry, 2 probes, eval at 30 and 60): DONE, no
        NONFINITE / ADV_FAIL / MU_FAIL / PREFETCH_FAIL, >= 3 prefetched BUILDs, >= 1 later ADV visit, mu finite >= 1,
        peak GPU < 30 GB; prints mean step time, swap time, BUILD seconds prefetched vs not
     R5 only when the data dir has F files (after prep_geo2 --F-old / new classes): R4 with bank_source 'data', tail_w 0.3
        and bank-span Ritz: tail and lam_max finite in STEP lines, later ADV visits carry bank_ritz_top"""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--remote', action='store_true')
    ap.add_argument('--split', default='/root/autodl-tmp/OPL/S2/SPLIT.json')
    ap.add_argument('--slots', default='/root/autodl-tmp/OPL/S2/slots')
    ap.add_argument('--data', default='/root/autodl-tmp/OPL/S2/data')
    ap.add_argument('--body', default='/root/autodl-tmp/OPL/S0')
    ap.add_argument('--ckpt', default='/root/autodl-tmp/OPL/S1/s2_full/last.pt')
    ap.add_argument('--out', default='/root/autodl-tmp/OPL/S1/t_train3_remote')
    a = ap.parse_args()
    import models as MD
    dev = TL.dev
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    split = json.loads(Path(a.split).read_text())
    dofs = lambda c: json.loads((Path(a.data) / c / 'DONE.json').read_text())['dofs']
    have = lambda c: (Path(a.slots) / f'{c}.pt').exists()
    tr = sorted([c for c in split['train'] if have(c)], key=dofs)[:4]
    va = sorted([c for c in split['val'] if have(c)], key=dofs)[:2]
    try:
        import oh as OH
    except ImportError:
        OH = None
    ck = torch.load(a.ckpt, map_location='cpu', weights_only=False)
    margs = dict(ck['cfg']['model_args'])
    cfg0 = dict(slot_cache=a.slots, data=a.data, body=a.body)
    g0, hb0 = T3.load_host(tr[0], cfg0, print)
    T3.move(g0, dev); g0.C.K = g0.C
    model = MD.build(ck['cfg']['model'], [g0], **margs).to(dev)
    model.load_state_dict(ck['model'], strict=False)
    model.caches.clear(); del g0, hb0; T3._empty()
    # R1
    base = torch.cuda.memory_allocated()
    s = T3.Slot(tr[0], cfg0, model, print)
    if OH is not None:
        s.set_view(OH, model, 5)
    s.to('cpu', model); s.to(dev, model)
    ts = [v for o in (s.geo, s.geo.C, s.cache) + ((s.view, s.vcache) if s.view is not None else ()) for v in vars(o).values()
          if torch.is_tensor(v)]
    ok = all(t.is_cuda for t in ts) and all(getattr(s.view, k) is getattr(s.geo, k) for k in s._shared) and s.geo.banks is None
    s.drop(model, OH); T3._empty()
    extra = (torch.cuda.memory_allocated() - base) / 2 ** 20
    check('R1_slot_round_trip', ok and extra <= 64, tensors=len(ts), view=OH is not None, extra_MB=extra)
    # R2 / R3
    gd = torch.load(Path(a.slots) / f'{tr[0]}.pt', map_location='cpu', weights_only=False)
    T3.move(gd, dev); gd.C.K = gd.C
    T2.clean_banks(gd, tr[0], print)
    _, hb = T3.load_host(tr[0], cfg0, print)
    model.add_geo(gd)
    mix = {'force': .25, 'support': .15, 'face': .2, 'macro': .1, 'grf': .15, 'adv': .15}
    gd.adv = torch.randn((gd.np_, 8), device=dev, generator=torch.Generator(device=dev).manual_seed(0))
    ok = all(torch.equal(gd.sample_with_sens(16, np.random.default_rng(s_), mix)[0],
                         hb.sample(16, np.random.default_rng(s_), mix, adv=gd.adv.cpu())[0]) for s_ in range(3))
    check('R2_host_sampling', ok)
    q1, s1 = gd.sample_with_sens(16, np.random.default_rng(5), mix)
    q2, s2, kinds, _ = hb.sample(16, np.random.default_rng(5), mix, adv=gd.adv.cpu())
    l1 = float(train2_loss(gd, model, q1, s1)[0]); l2 = float(T3.batch_loss(gd, model, q2, s2, kinds, None, {})[0])
    check('R3_default_loss', abs(l1 - l2) <= 1e-5 * abs(l1), train2=l1, train3=l2)
    model.caches.clear(); del gd, hb; T3._empty()
    # R4 / R5
    (out / 'SPLIT.json').write_text(json.dumps(dict(train=tr, val=va, test=[], train_families=[], val_families=[])))
    has_F = any((Path(a.data) / c).glob('train_*_F.npy') for c in tr)
    base_cfg = dict(split=str(out / 'SPLIT.json'), body=a.body, data=a.data, slot_cache=a.slots, model=ck['cfg']['model'],
                    model_args=margs, init=a.ckpt, steps=60, batch=16, lr=3e-4, pool=2, pool_dofs=700000, swap_every=10,
                    adv_on_load=True, adv_k=16, adv_iters=4, eval_every=30, val_max=2, probe_max=2, mu_geos=1, ema=0.999,
                    quota=True, oh=OH is not None, seed=0, sens_w=1.0, ckpt_every=30,
                    mix={'force': .25, 'support': .15, 'face': .2, 'macro': .1, 'grf': .15, 'adv': .15})
    runs = [('R4_run', dict(base_cfg, out=str(out / 'r4')))]
    if has_F:
        runs.append(('R5_run_F', dict(base_cfg, out=str(out / 'r5'), bank_source='data', tail_w=0.3)))
    else:
        check('R5_run_F', True, skipped='no F files in the data dir')
    for name, cfg in runs:
        shutil.rmtree(cfg['out'], ignore_errors=True)
        torch.cuda.reset_peak_memory_stats()
        T3.main(cfg)
        L = read_log(Path(cfg['out']) / 'train.log')
        ev = lambda k: [r for r in L if r.get('event') == k]
        bad = [r['event'] for r in L if r.get('event') in ('NONFINITE_LOSS', 'NONFINITE_GRAD', 'ADV_FAIL', 'MU_FAIL', 'PREFETCH_FAIL')]
        mus = [m['mu'] for r in ev('EVAL') for m in r['weights']['raw']['mu'].values()]
        later = [r for r in ev('ADV') if r['visit'] == 'later']
        steps = ev('STEP')
        builds = ev('BUILD')
        ok = bool(ev('DONE')) and not bad and sum(r['prefetched'] for r in builds) >= 3 and later and mus and \
            all(np.isfinite(m) and m >= 1 - 1e-6 for m in mus) and max(r['gpu_GB'] for r in steps) < 30
        if name == 'R5_run_F':
            ok = ok and all(np.isfinite(r.get('tail', np.nan)) for r in steps) and all('bank_ritz_top' in r for r in later)
        d = ev('DONE')[0]['time'] if ev('DONE') else {}
        check(name, ok, bad=bad, mu=mus, later_adv=len(later), step_s_mean=d.get('step', 0) / 60, swap_s=d.get('swap'),
              adv_s=d.get('adv'), eval_s=d.get('eval'), peak_GB=max([r['gpu_GB'] for r in steps] or [0]),
              build_s_prefetched=[round(r['seconds'], 2) for r in builds if r['prefetched']],
              build_s_sync=[round(r['seconds'], 2) for r in builds if not r['prefetched']])
    print(json.dumps(dict(event='REMOTE_DONE', failed=FAIL)), flush=True)
    sys.exit(1 if FAIL else 0)


if __name__ == '__main__':
    remote() if REMOTE else main()
