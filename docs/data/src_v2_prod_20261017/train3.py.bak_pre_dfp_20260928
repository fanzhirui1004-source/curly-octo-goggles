"""Step-2 trainer v2 (review C4 + C2 + C3 options): one network over many geometries, loss = mean log e_hat + sens_w x sensitivity
term (+ tail_w x Ritz tail term). train2.py is untouched; config keys that are not given keep train2 behaviour, except
swap_every (default 100 here, 250 in train2) and the adversarial search (a per-geometry buffer kept across visits, below).

Memory. Banks, exact sensitivities and reactions F stay on the HOST (trainlib.HostBanks, pageable): a batch is indexed on the
host and only the batch goes to the device. A pool of `pool` geometries (summed DOFs <= pool_dofs incl. the incoming one; older
members leave early, EVICT) is resident on the device; one enters every swap_every steps and the oldest leaves (blocking copy
to the host, then released, as train2). A background thread torch.load()s the next geometry of the order from the slot cache to
the host while training (prefetch), so a swap pays only the device copy; validation prefetches the next validation geometry.
Adversarial directions (adv_on_load): host buffer per geometry (<= adv_buffer vectors) persisting across visits. First visit:
block power iteration with the fp32 Neumann factor (Geo.adversarial, k = adv_k, adv_iters, started from force / face bank
samples), factor freed. Later visits: the buffer is re-scored with the current network (e_hat of every vector, unit exact
energy) and, when the geometry has reaction banks F, refreshed WITHOUT any factorization by bank-span Ritz (Geo.bank_ritz on
adv_ritz_R train samples: top adv_ritz_top generalized eigenvectors of (U^T K U, Q^T F)); the adv_buffer worst by current
Rayleigh quotient are kept. Adversarial columns carry NaN sensitivities (and no F). The buffers are in ckpt.pt up to
ckpt_adv_mb (resume_rng, below); a buffer not stored is searched again on its next first visit.
O_h augmentation (oh = true, needs oh.py): every entry of a geometry into the pool draws k uniformly from the 48 cube-group
elements (seeded) and trains on oh.view(model, geo, k), whose field is returned in the ORIGINAL frame (energies and
sensitivities with the original K and dM); validation, mu and the certificate use the identity (eval_views below adds
rotated views to the validation and the selection score).
Loss options: sens_loss 'sq' (mean rho^2, train2) | 'smoothl1' (mean sqrt(rho^2 + sens_delta^2) - sens_delta), rho =
|s_hat - s| / |s|; tail_w > 0: + tail_w tau logsumexp(log max(lambda, 1) / tau) over the generalized eigenvalues of the batch
Grams (U^T K U, Q^T F) of the non-adversarial columns, used when every one of them has F (trainlib.tail_loss).
Batch classes: multinomial counts as train2, or quota = true: floor(mix_c B) + the remainder drawn by the seeded generator
(quota_systematic = true: systematic remainder, trainlib.quota_counts).
EMA (ema = decay, 0 = off): shadow parameters initialised at the starting weights, updated after every optimizer step;
validation reports raw and EMA weights, checkpoints hold both, selection uses the EMA when enabled.
Validation (every eval_every, identity view): per geometry and class on the val banks energy excess e_hat - 1 (mean / p90 /
max), sensitivity relative error (mean / p90), certificate eps_lb (cert.py if importable: mean / max / fraction > cert_flag);
per-family aggregates; mu (top Ritz value of (S_hat, S), fp32 Neumann factor, block power iteration from a fixed start, until the
relative change < mu_tol or mu_iters) on mu_geos; probe_max training geometries (fixed, stratified over (family, DOF tercile)) with their steps since the
last visit. Selection score (see score()): mean over val families of (energy-class mean excess + sensitivity mean + 0.5 x p90
excess). last.pt, snap_<step>.pt at every eval, best.pt on the score (checkpoint 'model' = the selected weights, 'model_raw' =
raw when EMA is on).
Usage: train3.py <config.json>
config (train2): split, train_key, val_max, probe_max, body, data, out, slot_cache, model, model_args, init, steps, batch,
        geos_per_step, lr, pct_start, final_div, clip, pool, pool_dofs, swap_every, adv_on_load, adv_k, adv_iters, mix,
        eval_every, seed, sens_w, ckpt_every, resume
       (v2)  bank_source 'slot' (banks of the slot cache, 5 old classes) | 'data' (data dir, memory-mapped: new classes, F),
        prefetch (true), adv_buffer (32), adv_ritz_R (64), adv_ritz_top (8), adv_floor (1e-3), oh (false), quota (false),
        ema (0), sens_loss ('sq'), sens_delta (0.003), tail_w (0), tail_tau (0.1), tail_floor (1e-3), val_chunk (8),
        mu_geos (5: evenly spaced val geometries | list of cases), mu_k (8), mu_iters (30), mu_tol (1e-3), mu_seed (7),
        mu_start ('bank': fixed soft-class val samples | 'random'), cert (true), cert_m (8), cert_flag (0.1),
        score_classes (non-adversarial mix classes), probe_seed (seed), log_every (50: STEP lines),
        packets (FRESH_CONTEXT dirs, family ids), stop_after (end this process after that step without DONE, as if killed)
       (audit v2, 2026-09-25; every default keeps the earlier behaviour)
        eval_views ([0]): M3 multi-view selection. Each val and probe geometry is also evaluated under these O_h views (oh.view
          -> eval_geo -> oh.drop; energies / sensitivities in the original frame, as eval_views.py); per view the family score
          (score()); the selection score = the mean over the listed views of the view scores (eval_views_agg 'max': the
          worst view). The identity (view 0) is always evaluated and gives score_identity, mu and the certificate (identity
          only, for cost); EVAL, best.pt and last.pt carry eval_views, score_views {k: score}, score_identity and
          score_worst_view. Needs oh.py when a view is not 0 (with or without oh = true for training).
        score_sens (false): + 0.5 x the sensitivity p90 in each family total (score()); score_classes may list any class of
          trainlib.ALL_CLASSES (new ones included; unknown names raise; classes absent from every val geometry are logged).
        ema_debias (false): M4 bias-corrected EMA, shadow started at zero and divided by 1 - d^t (t = EMA updates so far), so
          the warm-start weights never enter the average (the plain EMA still holds d^t of them: 22% at 5k steps for
          d = 0.9997). select_min_step (0): evaluations before this step are logged and snapshotted but never become best.pt
          (the last evaluation always can).
        strict_mix (false): M5 raise when a training geometry lacks a mix class (default: MIX_MISSING logged once per
          geometry; HostBanks.sample then renormalises over the classes it has). Always (a guard): a mix class absent from
          EVERY training geometry raises at startup (slot / data-dir scan; strict_mix scans every geometry).
        quota_systematic (false): TRAINER-5 systematic remainder of quota (exact shares for rare classes).
        resume_rng (false; set true for v2): TRAINER-7 ckpt.pt holds the generator states (numpy gen / agen / ogen, torch tgen / global / cuda,
          python random), the pool members with their O_h view, and the adversarial buffers (up to ckpt_adv_mb, 256 MB); a
          resume restores them (a stop/resume run then follows the uninterrupted one), and best = min(ckpt best, best.pt
          score). False, or a checkpoint without them: the earlier reseed-from-step rebuild.
        Always (guards): non-finite validation metrics count as +inf in the score (GATE-6, VAL_NONFINITE) and best.pt is only
          taken from finite scores; val_max < len(split val) logs WARN (TRAINER-9); the conv precision (cudnn.allow_tf32)
          is in MODEL / EVAL / checkpoints (INVARIANTS-2); an 'adv' class dropped after ADV_FAIL is logged.
"""
import json, sys, time, gc, threading, random
from contextlib import contextmanager, nullcontext
from pathlib import Path
import numpy as np
import torch
import trainlib as TL
import models as MD
import train2 as T2

dev, dt = TL.dev, TL.dt
CPU = torch.device('cpu')
move = T2.move


def _empty():
    gc.collect()
    if dev.type == 'cuda':
        torch.cuda.empty_cache()


def _sync():
    if dev.type == 'cuda':
        torch.cuda.synchronize()


def _peak_gb():
    return torch.cuda.max_memory_allocated() / 2 ** 30 if dev.type == 'cuda' else 0.0


# --------------------------------------------------------------------------------------------------------- geometries
_CLEAN = {}                                                                     # data-dir banks: finiteness scanned once per run


def load_host(case, cfg, log, build=True):
    """(Geo on the host without banks, HostBanks). Slot cache when present (torch.load to the CPU), else built with the teacher
    (train2.build_geo) and cached like train2 (banks included, train2-compatible). Banks by bank_source: 'slot' the Geo's own,
    'data' the data dir (memory-mapped). Non-finite samples dropped (BANK_CLEAN, train2.clean_banks semantics incl. F)."""
    cache = Path(cfg['slot_cache']) / f'{case}.pt' if cfg.get('slot_cache') else None
    if cache is not None and cache.exists():
        g = torch.load(cache, map_location='cpu', weights_only=False)
    elif not build:
        raise FileNotFoundError(str(cache))
    else:
        g = T2.build_geo(case, cfg)
        move(g, 'cpu')
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save(g, cache)
    g.C.K = g.C
    hb = TL.HostBanks.from_geo(g) if g.banks is not None else None
    if cfg.get('bank_source', 'slot') == 'data':
        hb = TL.HostBanks.from_data(Path(cfg['data']) / case)
    elif cfg.get('bank_source', 'slot') != 'slot':
        raise ValueError(f"bank_source {cfg['bank_source']!r}")
    if hb is None:
        raise ValueError(f'NO_BANKS {case}')
    hb.clean(case, log, memo=_CLEAN if cfg.get('bank_source', 'slot') == 'data' else None)
    g.classes = list(hb.classes)
    return g, hb


class Prefetch:
    """One background load (slot cache -> host) of the geometry that is needed next; take() joins it (None if another case
    was prefetched, the cache is missing or the load failed: the caller then loads synchronously)."""

    def __init__(self, cfg, log, on=True):
        self.cfg, self.log, self.on = cfg, log, on and bool(cfg.get('slot_cache'))
        self.case = self.th = self.res = None

    def start(self, case):
        if not self.on or case == self.case:
            return
        self.cancel()
        if not (Path(self.cfg['slot_cache']) / f'{case}.pt').exists():
            return
        self.case = case

        def run():
            try:
                self.res = load_host(case, self.cfg, self.log, build=False)
            except BaseException as e:                                         # reported by take()
                self.res = e
        self.th = threading.Thread(target=run, daemon=True, name=f'prefetch:{case}')
        self.th.start()

    def take(self, case):
        if self.th is None or case != self.case:
            return None
        self.th.join()
        r = self.res
        self.case = self.th = self.res = None
        if isinstance(r, BaseException):
            self.log(dict(event='PREFETCH_FAIL', case=case, error=repr(r)[:200]))
            return None
        return r

    def cancel(self):
        if self.th is not None:
            self.th.join()
        self.case = self.th = self.res = None
        gc.collect()


class Slot:
    """One geometry: the slimmed trainlib.Geo (device or host), its HostBanks (always host), the model cache and, while in the
    training pool with oh = true, an O_h view (tgeo: the geometry the network trains on)."""

    def __init__(self, case, cfg, model, log, host=None):
        t0 = time.perf_counter()
        self.case = case
        g, hb = host if host is not None else load_host(case, cfg, log)
        move(g, dev); g.C.K = g.C
        self.geo, self.banks = g, hb
        self.view = self.vcache = self.k = None
        self._shared = []
        model.add_geo(g)
        self.cache = model.caches[case]
        self.where = dev
        self.build_seconds = time.perf_counter() - t0
        log(dict(event='BUILD', case=case, seconds=self.build_seconds, dofs=g.nb, ports=g.np_, prefetched=host is not None,
                 classes=hb.classes, F=[c for c in hb.classes if hb.has_F('train', c)]))

    @property
    def tgeo(self):
        return self.geo if self.view is None else self.view

    def set_view(self, OH, model, k):
        self.view = OH.view(model, self.geo, k)
        self.vcache = model.caches[self.view.case]
        self.k = k
        own = vars(self.view)
        self._shared = [a for a, v in vars(self.geo).items() if own.get(a, None) is v]   # re-pointed after every move

    def _move(self, device):
        move(self.geo, device); move(self.cache, device)
        if self.view is not None:
            for a in self._shared:
                setattr(self.view, a, getattr(self.geo, a))
            move(self.view, device); move(self.vcache, device)

    def drop(self, model, OH=None):
        """Forget this geometry (reloaded from the slot cache when needed again): tensors to the host first, so a stale
        reference cannot keep them on the device (train2 semantics)."""
        self._move(CPU)
        for c in (self.cache, self.vcache):
            if c is not None and getattr(model, '_cur', None) is c:
                model._cur = None
        model.caches.pop(self.case, None)
        if self.view is not None:
            OH.drop(model, self.view)
            model.caches.pop(self.view.case, None)
        self.geo = self.cache = self.view = self.vcache = self.banks = None
        self.where = None
        _empty()

    def to(self, device, model):
        device = torch.device(device)
        if self.where == device:
            return
        self._move(device)
        keys = [(self.case, self.cache)] + ([(self.view.case, self.vcache)] if self.view is not None else [])
        for key, c in keys:
            if device == CPU:
                model.caches.pop(key, None)
            else:
                model.caches[key] = c
        self.where = device
        _sync(); _empty()


# --------------------------------------------------------------------------------------------------------- loss
def batch_loss(geo, model, q, s0, kinds, F, cfg):
    """loss = mean log e_hat + sens_w x sens term (+ tail_w x Ritz tail term); default config: train2's loss op for op.
    Returns (loss, sens term or None, tail term or None, e (B,), lambda or None)."""
    sw, tw = cfg.get('sens_w', 1.0), cfg.get('tail_w', 0.0)
    u = geo.field(model, q)
    Gh = None
    if tw > 0 and F is not None:
        Gh = TL.gram(u, geo.C.K)
        e = torch.diagonal(Gh)
    else:
        e = TL.energy(u, geo.C.K)
    loss = torch.log(e.clamp_min(1e-12)).mean()
    ok = ~torch.isnan(s0[0]); ls = tail = lam = None
    if sw > 0 and ok.any():
        sh = geo.sens_hat(u[:, ok])
        ls = TL.sens_loss(sh, s0[:, ok], cfg.get('sens_loss', 'sq'), cfg.get('sens_delta', 0.003))
        loss = loss + sw * ls
    if Gh is not None:
        m = torch.as_tensor([k != 'adv' for k in kinds], device=u.device)
        if int(m.sum()) >= 2:
            G = q[:, m].to(dt).T @ F[:, m]
            tail, lam = TL.tail_loss(Gh[m][:, m], G, cfg.get('tail_tau', 0.1), cfg.get('tail_floor', 1e-3))
            loss = loss + tw * tail
    return loss, ls, tail, e, lam


class EMA:
    """Exponential moving average of the parameters (decay d, e <- e + (1 - d)(p - e)), initialised at the starting weights.
    debias=True (cfg ema_debias): the shadow starts at zero and the weights are e / (1 - d^t) after t updates (Adam's bias
    correction): a normalised average of the trained iterates only, without the starting (warm-start) weights; before the
    first update the model's current weights are used."""

    def __init__(self, model, decay, debias=False):
        self.decay, self.debias, self.t = decay, bool(debias), 0
        self.names = [n for n, _ in model.named_parameters()]
        self.p = [torch.zeros_like(p.detach()) if self.debias else p.detach().clone() for p in model.parameters()]

    @torch.no_grad()
    def update(self, model):
        torch._foreach_lerp_(self.p, [p.detach() for p in model.parameters()], 1.0 - self.decay)
        self.t += 1

    def weights(self):
        """The averaged weights (None: debiased and not updated yet = use the current weights)."""
        if not self.debias:
            return self.p
        if self.t == 0:
            return None
        c = 1.0 - self.decay ** self.t
        return [e / c for e in self.p]

    @contextmanager
    def applied(self, model):
        """Temporarily run the model with the EMA weights."""
        ps = list(model.parameters())
        keep = [p.detach().clone() for p in ps]
        ew = self.weights()
        with torch.no_grad():
            for p, e in zip(ps, ew if ew is not None else keep):
                p.copy_(e)
        try:
            yield
        finally:
            with torch.no_grad():
                for p, k in zip(ps, keep):
                    p.copy_(k)

    def model_state(self, model):
        sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
        ew = self.weights()
        for n, e in zip(self.names, ew if ew is not None else []):
            sd[n] = e.detach().clone()
        return sd

    def state_dict(self):
        return dict(decay=self.decay, p=[e.cpu() for e in self.p], t=self.t, debias=self.debias)

    def load_state_dict(self, d):
        if bool(d.get('debias', False)) != self.debias:
            raise ValueError(f"EMA: checkpoint debias={d.get('debias', False)} but cfg ema_debias={self.debias} (resume with the "
                             f"same EMA mode)")
        self.p = [e.to(p.device) for e, p in zip(d['p'], self.p)]
        self.t = int(d.get('t', 0))


# --------------------------------------------------------------------------------------------------------- validation
@torch.no_grad()
def eval_geo(g, hb, model, chunk=8, cert=None, cert_m=8, cert_flag=0.1):
    """Val banks of one geometry per class: energy excess e_hat - 1 (banks at unit exact energy) mean / p90 / max, sensitivity
    relative error |s_hat - s| / |s| mean / p90, and with a certificate eps_lb (cert.mu_lower; a rigorous lower bound of the
    excess) mean / max / fraction > cert_flag."""
    out = {}
    for c in hb.classes:
        if c not in hb.q.get('val', {}):
            continue
        e, se, ce = [], [], []
        for q, s0 in hb.chunks('val', c, chunk):
            u = g.field(model, q)
            e.append(TL.energy(u, g.C.K) - 1)
            if s0 is not None and hasattr(g, 'dM32'):
                sh = g.sens_hat(u)
                se.append((sh - s0).norm(dim=0) / s0.norm(dim=0))
            if cert is not None:
                ce.append(cert.mu_lower(u, cert_m)[1])
        e = torch.cat(e).cpu().numpy()
        r = dict(mean=float(e.mean()), p90=float(np.quantile(e, .9)), max=float(e.max()))
        if se:
            se = torch.cat(se).cpu().numpy()
            r.update(sens_mean=float(se.mean()), sens_p90=float(np.quantile(se, .9)))
        if ce:
            ce = torch.cat(ce).cpu().numpy()
            r.update(cert_mean=float(ce.mean()), cert_max=float(ce.max()), cert_flag=float((ce > cert_flag).mean()))
        out[c] = r
    return out


def family_resolver(packets):
    """case -> family id: FRESH_CONTEXT case.family_id (as make_split.py), else the name prefix fresh_<split>_<id>."""
    memo = {}

    def fam(case):
        if case not in memo:
            try:
                f_ = Path(packets) / case / 'FRESH_CONTEXT.json'
                if not f_.exists():
                    f_ = TL.TE.packet_dir(case) / 'FRESH_CONTEXT.json'              # OPL_PACKETS_EXTRA (unset: the same path)
                memo[case] = json.loads(f_.read_text())['case']['family_id']
            except (OSError, KeyError, ValueError, TypeError):
                memo[case] = '_'.join(case.split('_')[:3])
        return memo[case]
    return fam


def _nanmean(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return float(np.mean(v)) if v else float('nan')


def _geo_mean(v):
    """families(): mean over geometries; None = absent (skipped); a non-finite value (a diverged geometry) makes the result
    +inf instead of being dropped (GATE-6); empty -> NaN (absent). Equal to _nanmean when every value is finite."""
    v = [x for x in v if x is not None]
    if not v:
        return float('nan')
    return float(np.mean(v)) if all(np.isfinite(x) for x in v) else float('inf')


def _sel_mean(v):
    """score(): NaN = absent (families() turns diverged values into +inf) and skipped; +inf propagates."""
    v = [x for x in v if x is not None and not np.isnan(x)]
    return float(np.mean(v)) if v else float('nan')


def families(per_geo, fam):
    """Per family and class: mean over its geometries of the class mean / p90 excess and sensitivity mean / p90, max of the
    class max, number of geometries. A non-finite geometry metric counts as +inf (GATE-6)."""
    out = {}
    for case, r in per_geo.items():
        out.setdefault(fam(case), {}).setdefault('_cases', []).append(case)
    for f, d in out.items():
        cases = d.pop('_cases')
        for c in sorted({c for case in cases for c in per_geo[case]}):
            rs = [per_geo[case][c] for case in cases if c in per_geo[case]]
            mx = [r['max'] for r in rs]
            d[c] = dict(mean=_geo_mean([r['mean'] for r in rs]), p90=_geo_mean([r['p90'] for r in rs]),
                        max=max(mx) if all(np.isfinite(x) for x in mx) else float('inf'),
                        sens_mean=_geo_mean([r.get('sens_mean') for r in rs]), sens_p90=_geo_mean([r.get('sens_p90') for r in rs]),
                        n=len(rs))
        d['n_geo'] = len(cases)
    return out


def score(fam_agg, classes, val_families=None, sens_p90=False):
    """Selection score (lower is better) = mean over val families f of  E_f + S_f + 0.5 P_f (+ 0.5 S90_f)  with
         E_f = mean over the energy classes c of the family's class-mean excess (mean over its geometries of mean e_hat - 1),
         S_f = mean over those classes with labels of the family's mean relative sensitivity error,
         P_f = mean over the energy classes of the family's p90 excess,
         S90_f (sens_p90 = cfg score_sens) = mean over those classes with labels of the family's sensitivity p90.
    Families weigh equally whatever their size (4 val families: a family-level decision rule); classes absent from a family
    are skipped. A diverged (non-finite) metric makes its family total and the score +inf (GATE-6).
    Returns (score, parts {family: dict(energy, sens, p90, total[, sens_p90])})."""
    fs = [f for f in (val_families or sorted(fam_agg)) if f in fam_agg] or sorted(fam_agg)
    parts = {}
    for f in fs:
        d = fam_agg[f]
        E = _sel_mean([d[c]['mean'] for c in classes if c in d])
        S = _sel_mean([d[c]['sens_mean'] for c in classes if c in d])
        P = _sel_mean([d[c]['p90'] for c in classes if c in d])
        parts[f] = dict(energy=E, sens=S, p90=P, total=E + (0.0 if np.isnan(S) else S) + 0.5 * P)
        if sens_p90:
            S9 = _sel_mean([d[c]['sens_p90'] for c in classes if c in d])
            parts[f].update(sens_p90=S9, total=parts[f]['total'] + (0.0 if np.isnan(S9) else 0.5 * S9))
    return _sel_mean([p['total'] for p in parts.values()]), parts


def stratified_probes(cases, n, fam, dofs, seed):
    """n training geometries spread over the strata (family, DOF tercile of the training set): strata in a seeded random
    order, one seeded random member per stratum per round, until n are picked."""
    if n <= 0 or not cases:
        return []
    d = np.asarray([dofs(c) for c in cases], float)
    t1, t2 = np.quantile(d, [1 / 3, 2 / 3])
    strata = {}
    for c, x in zip(cases, d):
        strata.setdefault((fam(c), int(x > t1) + int(x > t2)), []).append(c)
    rng = np.random.default_rng(seed)
    keys = sorted(strata)
    keys = [keys[i] for i in rng.permutation(len(keys))]
    for k in keys:
        strata[k] = [strata[k][i] for i in rng.permutation(len(strata[k]))]
    picks = []
    while len(picks) < min(n, len(cases)):
        for k in keys:
            if strata[k] and len(picks) < n:
                picks.append(strata[k].pop(0))
    return picks


# --------------------------------------------------------------------------------------------------------- main
def main(cfg):
    if cfg.get('conv_fp32'):                                                   # true fp32 convolutions (no TF32)
        MD.set_conv_fp32(True)
    out = Path(cfg['out']); out.mkdir(parents=True, exist_ok=True)
    log_f = open(out / 'train.log', 'a')
    lock = threading.Lock()

    def log(d):
        s = json.dumps(d)
        with lock:
            print(s, flush=True); log_f.write(s + '\n'); log_f.flush()

    OH = CE = None
    if cfg.get('oh'):
        try:
            import oh as OH
        except ImportError as e:
            raise ImportError('cfg oh = true needs oh.py (O_h views); not importable') from e
    if cfg.get('cert', True):
        try:
            import cert as CE
        except ImportError:
            log(dict(event='CERT_OFF', reason='cert.py not importable'))
    seed = cfg.get('seed', 0)
    torch.manual_seed(seed)
    gen = np.random.default_rng(seed)
    tgen = torch.Generator(device=dev).manual_seed(seed)
    agen = np.random.default_rng([seed, 1])                                    # adversarial starts / bank-span samples
    ogen = np.random.default_rng([seed, 48])                                   # O_h view per entry
    split = json.loads(Path(cfg['split']).read_text())
    key = cfg.get('train_key', 'train')
    train_cases = split['curve'][key.split(':')[1]]['geometries'] if key.startswith('curve:') else split['train']
    val_cases = split['val'][:cfg.get('val_max', 8)]
    fam = family_resolver(cfg.get('packets', TL.TE.ROOT / 'packets'))
    log(dict(event='SPLIT', train=len(train_cases), val=len(val_cases)))
    if len(val_cases) < len(split['val']):                                     # TRAINER-9: val is whole families in order
        log(dict(event='WARN', msg=f"val_max {cfg.get('val_max', 8)} < {len(split['val'])} val geometries: only the first "
                                   f"{len(val_cases)} (families {sorted({fam(c) for c in val_cases})}) enter validation"))
    views = [int(k) for k in cfg.get('eval_views', [0])]                       # M3 multi-view selection
    if not views or len(set(views)) != len(views):
        raise ValueError(f'eval_views {views}: a non-empty list of distinct O_h element indices')
    view_agg = cfg.get('eval_views_agg', 'mean')
    if view_agg not in ('mean', 'max'):
        raise ValueError(f'eval_views_agg {view_agg!r}')
    OHV = OH
    if any(k != 0 for k in views) and OHV is None:
        try:
            import oh as OHV
        except ImportError as e:
            raise ImportError('cfg eval_views with a non-identity view needs oh.py (O_h views); not importable') from e
    if OHV is not None and not all(0 <= k < len(OHV.ELEMS) for k in views):
        raise ValueError(f'eval_views {views}: indices of oh.ELEMS (0 .. {len(OHV.ELEMS) - 1})')
    mix = dict(cfg['mix'])
    want = [c for c, w in mix.items() if w > 0 and c != 'adv']
    bad = [c for c in want + list(cfg.get('score_classes') or []) if c not in TL.ALL_CLASSES]
    if bad:
        raise ValueError(f'unknown class names {bad} in mix / score_classes (trainlib.ALL_CLASSES: {list(TL.ALL_CLASSES)})')
    if mix.get('adv', 0) > 0 and not cfg.get('adv_on_load'):
        log(dict(event='MIX_MISSING', missing=['adv'], reason='adv_on_load is off: the adv share is never sampled'))

    def train_classes(case):
        """Classes a training geometry will have before cleaning: its slot cache banks (memory-mapped load) when present
        (bank_source 'slot'), else the data dir files (the build path and bank_source 'data')."""
        if cfg.get('bank_source', 'slot') == 'slot' and cfg.get('slot_cache'):
            f = Path(cfg['slot_cache']) / f'{case}.pt'
            if f.exists():
                try:
                    g_ = torch.load(f, map_location='cpu', weights_only=False, mmap=True)
                except RuntimeError:                                           # not a zip-format file: full load
                    g_ = torch.load(f, map_location='cpu', weights_only=False)
                return set((g_.banks or {}).get('train', {}))
        d_ = Path(cfg['data']) / case
        return {c for c in TL.ALL_CLASSES if (d_ / f'train_{c}.npy').exists()}
    # M5: a mix class that NO training geometry has would silently never be trained (e.g. new classes with bank_source 'slot')
    seen, per_case = set(), {}
    for case in train_cases:
        if set(want) <= seen and not cfg.get('strict_mix'):
            break                                                               # every class found: the guard is satisfied
        per_case[case] = train_classes(case)
        seen |= per_case[case]
    absent = [c for c in want if c not in seen]
    if absent:
        log(dict(event='MIX_ABSENT', classes=absent, bank_source=cfg.get('bank_source', 'slot'), scanned=len(per_case)))
        raise ValueError(f"MIX_ABSENT: mix classes {absent} are in no training geometry (bank_source "
                         f"{cfg.get('bank_source', 'slot')!r}); they would never be trained")
    if cfg.get('strict_mix'):
        miss = {c_: [c for c in want if c not in cl] for c_, cl in per_case.items()}
        miss = {k: v for k, v in miss.items() if v}
        if miss:
            log(dict(event='MIX_MISSING', missing=miss, strict=True))
            raise ValueError(f'MIX_CLASSES_MISSING (strict_mix) {miss}')
    if cfg.get('init') and cfg['lr'] > 3e-4:
        log(dict(event='WARN', msg=f"warm start (init) with lr {cfg['lr']} > 3e-4: re-warming a converged model to 1e-3 wrecked "
                                   f"it in step 1 (force 2.4% -> 10%)"))
    # the model needs one geometry at construction (caches only); the rest are added on the fly
    first, _ = load_host(train_cases[0], cfg, lambda d_: None)
    move(first, dev); first.C.K = first.C
    model = MD.build(cfg['model'], [first], **cfg.get('model_args', {})).to(dev)
    conv = TL.conv_precision()
    del first; model.caches.clear(); _empty()
    if cfg.get('init'):
        ck = torch.load(cfg['init'], map_location=dev, weights_only=False)
        ma = cfg.get('model_args', {})
        if hasattr(MD, 'load_compat') and any(ma.get(k) for k in ('feat_v2', 'bounded', 'fringe_soft', 'b3')):
            log(dict(event='INIT', ckpt=cfg['init'], compat=MD.load_compat(model, ck['model'])))
        else:
            res = model.load_state_dict(ck['model'], strict=False)
            log(dict(event='INIT', ckpt=cfg['init'], missing=len(res.missing_keys)))
    log(dict(event='MODEL', params=sum(p.numel() for p in model.parameters()), **conv))
    opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    steps = cfg['steps']
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg['lr'], total_steps=steps, pct_start=cfg.get('pct_start', 0.05),
                                                anneal_strategy='cos', final_div_factor=cfg.get('final_div', 100))
    ema = EMA(model, cfg['ema'], cfg.get('ema_debias', False)) if cfg.get('ema', 0) > 0 else None
    slots = {}                                                                  # live slots (pool members, validation)
    tpre, vpre = Prefetch(cfg, log, cfg.get('prefetch', True)), Prefetch(cfg, log, cfg.get('prefetch', True))

    def get(case, pre=None, train=False, view=None):
        if case in slots:
            s_ = slots[case]; s_.to(dev, model)
        else:
            s_ = slots[case] = Slot(case, cfg, model, log, host=pre.take(case) if pre is not None else None)
        if train and OH is not None and s_.view is None:
            s_.set_view(OH, model, int(ogen.integers(len(OH.ELEMS))) if view is None else int(view))
        return s_

    mix_logged = set()

    def check_mix(s_):
        """M5: mix classes this training geometry lacks (after cleaning): logged once per geometry; strict_mix raises."""
        miss = [c for c in want if c not in s_.banks.classes]
        if miss and s_.case not in mix_logged:
            mix_logged.add(s_.case)
            log(dict(event='MIX_MISSING', case=s_.case, missing=miss, classes=list(s_.banks.classes)))
            if cfg.get('strict_mix'):
                raise ValueError(f'MIX_CLASSES_MISSING (strict_mix) {s_.case}: {miss}')

    def drop(s_):
        s_.drop(model, OH); slots.pop(s_.case, None)

    ndofs = {}

    def dofs(case):
        if case not in ndofs:
            ndofs[case] = json.loads((Path(cfg['data']) / case / 'DONE.json').read_text())['dofs']
        return ndofs[case]

    # ---------------------------------------------------------------- adversarial host buffers
    advbuf = {}                                                                 # case -> (np, k) fp32 host
    A = dict(k=cfg.get('adv_k', 16), iters=cfg.get('adv_iters', 4), buf=cfg.get('adv_buffer', 32), R=cfg.get('adv_ritz_R', 64),
             top=cfg.get('adv_ritz_top', 8), floor=cfg.get('adv_floor', 1e-3))

    def bank_cols(hb, classes, n, F=False):
        """n train samples spread uniformly over classes (agen; distinct samples within a class while the bank allows):
        q (np, n) fp32 [, F (np, n) fp64] on the device."""
        pick = agen.integers(0, len(classes), n)
        qs, fs = [], []
        for i, c in enumerate(classes):
            m_, mc = int((pick == i).sum()), hb.m('train', c)
            if m_:
                q, _, f = hb.get('train', c, agen.choice(mc, size=m_, replace=m_ > mc), F=F)
                qs.append(q); fs.append(f)
        return torch.cat(qs, 1), (torch.cat(fs, 1) if F else None)

    @torch.no_grad()
    def rescore(g, X, chunk=16):
        return torch.cat([TL.energy(g.field(model, X[:, j:j + chunk].to(dev)), g.C.K) for j in range(0, X.shape[1], chunk)])

    def adv_entry(s_):
        g, case, t = s_.tgeo, s_.case, time.perf_counter()
        model.eval()
        try:
            if case not in advbuf:
                cl = [c for c in ('force', 'face', 'force_c', 'face_c') if c in s_.banks.classes]
                start = bank_cols(s_.banks, cl, A['k'])[0].to(dt) if cl else None
                C = g.C
                try:
                    C.factor(neumann=True, interior=False, fp32_neumann=True)
                    X, ritz = g.adversarial(model, k=A['k'], iters=A['iters'], gen=tgen, start=start)
                finally:
                    C._free() if hasattr(C, '_free') else None
                    C.sol_N = None
                    _empty()
                advbuf[case] = X.to(torch.float32).cpu()
                log(dict(event='ADV', case=case, visit='first', ritz_top=ritz[:3].tolist(), seconds=time.perf_counter() - t))
                return
            X = advbuf[case]
            sc = rescore(g, X).cpu()
            fcl = [c for c in s_.banks.classes if s_.banks.has_F('train', c)]
            rec = dict(event='ADV', case=case, visit='later', buffer_top=float(sc.max()))
            if fcl and A['R'] > 0:
                Q, F = bank_cols(s_.banks, fcl, A['R'], F=True)
                Xn, rn = g.bank_ritz(model, Q, F, top=A['top'], floor=A['floor'])
                Xn, rn = Xn.to(torch.float32).cpu(), rn.cpu()
                Xu = X / X.norm(dim=0).clamp_min(1e-30)
                cos = (Xu.T.to(dt) @ (Xn / Xn.norm(dim=0).clamp_min(1e-30)).to(dt)).abs().max(0).values
                new = cos < 0.98                                                # drop near-copies of buffered directions
                allX, alls = torch.cat([X, Xn[:, new]], 1), torch.cat([sc, rn[new]])
                keep = torch.argsort(alls, descending=True)[:A['buf']]
                advbuf[case] = allX[:, keep].contiguous()
                rec.update(bank_ritz_top=rn[:3].tolist(), added=int((keep >= X.shape[1]).sum()), size=len(keep))
            rec['seconds'] = time.perf_counter() - t
            log(rec)
        except Exception as e:
            log(dict(event='ADV_FAIL', case=case, error=repr(e)[:200],
                     adv_dropped=case not in advbuf and mix.get('adv', 0) > 0))   # no buffer: 'adv' leaves this geometry's mix
        finally:
            model.train()

    # ---------------------------------------------------------------- pool
    cap = cfg.get('pool_dofs', 0)
    P = cfg.get('pool', 4)
    visits = {}                                                                 # case -> dict(n, enter, last)

    def leave(s_, step):
        visits[s_.case]['last'] = step
        drop(s_)

    def admit(case, step, view=None, rebuild=False, exact=False):
        """Add a geometry to the pool, first evicting the oldest members while the pool would exceed the DOF cap.
        rebuild (resume): a member the run had at the checkpoint, not counted as a new visit; exact (restored state): its
        view is the stored one (none drawn) and, when its adversarial buffer was restored, the entry search is not repeated."""
        while cap and pool and sum(dofs(p_.case) for p_ in pool) + dofs(case) > cap:
            o = pool.pop(0); leave(o, step); log(dict(event='EVICT', case=o.case, incoming=case))
        s_ = get(case, pre=tpre, train=True, view=view)
        check_mix(s_)
        v = visits.setdefault(case, dict(n=0))
        v.update(n=v['n'] + (0 if rebuild else 1), enter=v.get('enter', step) if rebuild else step, last=None)
        if s_.view is not None:
            log(dict(event='VIEW', case=case, k=s_.k, step=step))
        if cfg.get('adv_on_load') and not (exact and case in advbuf):
            ta = time.perf_counter(); adv_entry(s_); tstat['adv'] += time.perf_counter() - ta
        pool.append(s_)

    order = list(train_cases); gen.shuffle(order)
    qi, start = 0, 1
    ck_path = out / 'ckpt.pt'
    best = None

    def rng_state():
        """Every generator the trajectory draws from (TRAINER-7), for ckpt.pt."""
        d = dict(gen=gen.bit_generator.state, agen=agen.bit_generator.state, ogen=ogen.bit_generator.state,
                 tgen=tgen.get_state(), torch=torch.get_rng_state(), py=random.getstate())
        if dev.type == 'cuda':
            d['cuda'] = torch.cuda.get_rng_state(dev)
        return d

    def adv_for_ckpt():
        """Adversarial host buffers for ckpt.pt within ckpt_adv_mb: pool members first, then by the latest visit."""
        capb = float(cfg.get('ckpt_adv_mb', 256)) * 2 ** 20
        live = [p_.case for p_ in pool]
        rest = sorted((c for c in advbuf if c not in live), key=lambda c: -(visits.get(c, {}).get('last') or 0))
        keep, tot = {}, 0
        for c in [c for c in live if c in advbuf] + rest:
            nb_ = advbuf[c].numel() * advbuf[c].element_size()
            if tot + nb_ <= capb:
                keep[c] = advbuf[c]; tot += nb_
        if len(keep) < len(advbuf):
            log(dict(event='ADV_CKPT', stored=len(keep), dropped=len(advbuf) - len(keep), MB=tot / 2 ** 20))
        return keep
    pool_spec = None
    if cfg.get('resume') and ck_path.exists():
        ck = torch.load(ck_path, map_location=dev, weights_only=False)
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['opt']); sched.load_state_dict(ck['sched'])
        start, qi, best = ck['step'] + 1, ck['qi'], ck.get('best')
        if best is not None and not np.isfinite(best):
            best = None
        bp = out / 'best.pt'                                                    # TRAINER-7: best.pt may be newer than ckpt.pt
        if bp.exists():
            try:
                bs = torch.load(bp, map_location='cpu', weights_only=False).get('score')
                if bs is not None and np.isfinite(bs) and (best is None or bs < best):
                    log(dict(event='RESUME_BEST', ckpt_best=best, best_pt=bs)); best = float(bs)
            except Exception as e:
                log(dict(event='WARN', msg=f'best.pt unreadable on resume: {e!r}'[:200]))
        ema_reset = False
        if ema is not None and ck.get('ema') is not None:
            ema.load_state_dict(ck['ema'])
        elif ema is not None:                                                   # EMA turned on at resume: start it here
            ema = EMA(model, cfg['ema'], cfg.get('ema_debias', False)); ema_reset = True
        visits.update(ck.get('visits', {}))
        rng = ck.get('rng') if cfg.get('resume_rng', False) else None
        if rng is not None:                                                     # the exact continuation
            gen.bit_generator.state = rng['gen']; agen.bit_generator.state = rng['agen']; ogen.bit_generator.state = rng['ogen']
            tgen.set_state(rng['tgen'].to(tgen.get_state().device) if torch.is_tensor(rng['tgen']) else rng['tgen'])
            torch.set_rng_state(rng['torch'].cpu()); random.setstate(rng['py'])
            if dev.type == 'cuda' and rng.get('cuda') is not None:
                torch.cuda.set_rng_state(rng['cuda'].cpu(), dev)
            advbuf.update({c: x.cpu() for c, x in (ck.get('advbuf') or {}).items()})
            pool_spec = ck.get('pool')
        else:
            gen = np.random.default_rng([seed, ck['step']])
            agen = np.random.default_rng([seed, 1, ck['step']]); ogen = np.random.default_rng([seed, 48, ck['step']])
        log(dict(event='RESUME', step=ck['step'], qi=qi, ema=ema is not None and ck.get('ema') is not None, ema_reset=ema_reset,
                 rng_restored=rng is not None, adv_restored=len(advbuf), pool=pool_spec, best=best))
    pool = []
    tstat = dict(swap=0.0, step=0.0, adv=0.0, eval=0.0)
    if pool_spec is not None:                                                   # the members (and views) at the checkpoint
        for case, k in pool_spec:
            admit(case, start - 1, view=k, rebuild=True, exact=True)
    else:
        resumed = start > 1
        qi = max(qi - P, 0)                                                     # rebuild the pool the run had at the checkpoint
        for _ in range(min(P, len(order))):
            admit(order[qi % len(order)], start - 1, rebuild=resumed); qi += 1
    if len(order) > len(pool):
        tpre.start(order[qi % len(order)])
    probe_cases = stratified_probes(train_cases, cfg.get('probe_max', 0), fam, dofs, cfg.get('probe_seed', seed))
    log(dict(event='PROBES', cases=probe_cases))
    mg = cfg.get('mu_geos', 5)
    if isinstance(mg, (list, tuple)):
        mu_cases = list(mg)
    else:                                                                       # evenly spaced over the (family-ordered) val list
        n_mu = min(int(mg), len(val_cases))
        mu_cases = [val_cases[i] for i in sorted(set(np.linspace(0, len(val_cases) - 1, n_mu).round().astype(int).tolist()))] if n_mu else []
    log(dict(event='MU_GEOS', cases=mu_cases))
    mix = dict(cfg['mix']); B = cfg['batch']
    score_classes = cfg.get('score_classes') or [c for c, w in mix.items() if c != 'adv' and w > 0]
    want_F = cfg.get('tail_w', 0.0) > 0
    wsets = [('raw', None)] + ([('ema', ema)] if ema is not None else [])
    sel = wsets[-1][0]
    # evaluation cost (monitoring only: neither changes the trajectory or the selected checkpoint): eval_weights 'sel' scores
    # only the selected weights at intermediate evals (all weight sets at the last one); cert_every k computes the
    # certificate on every k-th eval (and the last one). Defaults keep the full evaluation every time.
    eval_w, cert_every = cfg.get('eval_weights', 'all'), max(1, int(cfg.get('cert_every', 1)))
    if eval_w not in ('all', 'sel'):
        raise ValueError(f'eval_weights {eval_w!r}')
    score_sens = bool(cfg.get('score_sens', False))
    min_step = int(cfg.get('select_min_step', 0))
    if min_step > steps:
        log(dict(event='WARN', msg=f'select_min_step {min_step} > steps {steps}: only the last evaluation can become best.pt'))
    log(dict(event='SELECTION', score_classes=score_classes, eval_views=views, eval_views_agg=view_agg, score_sens=score_sens,
             select_min_step=min_step, weights=sel, ema_debias=bool(cfg.get('ema_debias', False))))
    rot = [k for k in views if k != 0]
    score_absent_logged = []

    def mu_start(hb, k):
        """Fixed start block of the mu iteration: k val samples of the soft classes (seeded by mu_seed, the same every eval)."""
        cl = [c for c in ('force', 'support', 'face', 'force_c', 'face_c', 'glued', 'support_k', 'support64') if c in hb.q.get('val', {})]
        if not cl:
            return None
        r, cols = np.random.default_rng(cfg.get('mu_seed', 7)), []
        for i, c in enumerate(cl):
            n_c, m_c = len(range(i, k, len(cl))), hb.m('val', c)
            if n_c:
                cols.append(hb.get('val', c, r.choice(m_c, size=n_c, replace=n_c > m_c))[0])
        return torch.cat(cols, 1).to(dt)

    def evaluate(step):
        for p_ in pool:                                                        # pool members wait on the host
            p_.to(CPU, model)
        live = {p_.case for p_ in pool}

        def release(s_):
            s_.to(CPU, model) if s_.case in live else drop(s_)
        last = step == steps
        ws = wsets if (eval_w == 'all' or last) else [w_ for w_ in wsets if w_[0] == sel]
        use_cert = CE is not None and (last or (step // cfg['eval_every']) % cert_every == 0)
        res = {w: dict(val={}, train_geo={}, mu={}, **({'views': {str(k): dict(val={}, train_geo={}) for k in rot}} if rot else {}))
               for w, _ in ws}
        model.eval()
        todo = [('val', c) for c in val_cases] + [('train_geo', c) for c in probe_cases]
        for i, (kind, case) in enumerate(todo):
            s_ = get(case, pre=vpre)
            if i + 1 < len(todo) and todo[i + 1][1] not in slots:          # next one loads on the host meanwhile
                vpre.start(todo[i + 1][1])
            g = s_.geo                                                      # identity view
            cer = CE.from_geo(g) if use_cert and kind == 'val' else None
            for w, e_ in ws:
                with (e_.applied(model) if e_ is not None else nullcontext()):
                    res[w][kind][case] = eval_geo(g, s_.banks, model, cfg.get('val_chunk', 8), cer, cfg.get('cert_m', 8),
                                                  cfg.get('cert_flag', 0.1))
            del cer
            for k in rot:                                                   # M3: rotated views (no certificate / mu: cost)
                reuse = s_.view is not None and s_.k == k                   # a pool member training on this very view
                v = s_.view if reuse else OHV.view(model, g, k)
                try:
                    for w, e_ in ws:
                        with (e_.applied(model) if e_ is not None else nullcontext()):
                            r_ = eval_geo(v, s_.banks, model, cfg.get('val_chunk', 8))
                        res[w]['views'][str(k)][kind][case] = {c: {m: x[m] for m in ('mean', 'p90', 'max', 'sens_mean', 'sens_p90')
                                                                   if m in x} for c, x in r_.items()}
                finally:
                    if not reuse:
                        OHV.drop(model, v)
            if kind == 'val' and case in mu_cases:
                t = time.perf_counter()
                try:
                    g.C.factor(neumann=True, interior=False, fp32_neumann=True)
                    X0 = mu_start(s_.banks, cfg.get('mu_k', 8)) if cfg.get('mu_start', 'bank') == 'bank' else None
                    for w, e_ in ws:
                        with (e_.applied(model) if e_ is not None else nullcontext()):
                            mu, it, _ = g.worst_ratio(model, k=cfg.get('mu_k', 8), tol=cfg.get('mu_tol', 1e-3),
                                                      max_iters=cfg.get('mu_iters', 30), start=X0,
                                                      gen=torch.Generator(device=dev).manual_seed(cfg.get('mu_seed', 7)))
                        res[w]['mu'][case] = dict(mu=mu, iters=it, seconds=time.perf_counter() - t)
                except Exception as e:
                    log(dict(event='MU_FAIL', case=case, error=repr(e)[:200]))
                finally:
                    g.C._free() if hasattr(g.C, '_free') else None
                    g.C.sol_N = None
                    _empty()
            release(s_)
        vpre.cancel()
        model.train()
        for p_ in pool:
            p_.to(dev, model)
        for w in res:
            r = res[w]
            r['val_mean'] = {c: _nanmean([v[c]['mean'] for v in r['val'].values() if c in v])
                             for c in sorted({c for v in r['val'].values() for c in v})}
            r['train_geo_mean'] = {c: _nanmean([v[c]['mean'] for v in r['train_geo'].values() if c in v])
                                   for c in sorted({c for v in r['train_geo'].values() for c in v})}
            r['val_family'] = families(r['val'], fam)
            r['train_geo_family'] = families(r['train_geo'], fam) if r['train_geo'] else {}
            r['score'], r['score_parts'] = score(r['val_family'], score_classes, split.get('val_families'), score_sens)
            r['score_identity'] = r['score']
            per_view = {}
            for k in views:
                if k == 0:
                    per_view['0'] = r['score_identity']
                    continue
                pv = r['views'][str(k)]
                pv['val_mean'] = {c: _nanmean([v[c]['mean'] for v in pv['val'].values() if c in v])
                                  for c in sorted({c for v in pv['val'].values() for c in v})}
                pv['train_geo_mean'] = {c: _nanmean([v[c]['mean'] for v in pv['train_geo'].values() if c in v])
                                        for c in sorted({c for v in pv['train_geo'].values() for c in v})}
                pv['val_family'] = families(pv['val'], fam)
                pv['score'], pv['score_parts'] = score(pv['val_family'], score_classes, split.get('val_families'), score_sens)
                per_view[str(k)] = pv['score']
            r['score_views'] = per_view
            vs = np.asarray(list(per_view.values()), float)
            r['score_worst_view'] = float(vs.max())                            # NaN / inf propagate: not selectable
            if views != [0]:
                r['score'] = float(vs.mean()) if view_agg == 'mean' else r['score_worst_view']
            gone = [c for c in score_classes if c not in r['val_mean']]
            if gone and not score_absent_logged:
                score_absent_logged.append(True)
                log(dict(event='SCORE_CLASS_ABSENT', classes=gone, msg='score classes in no val geometry: not in the score'))
            bad = sorted({(case, c) for case, v in r['val'].items() for c, x in v.items()
                          if not all(np.isfinite(x[m]) for m in ('mean', 'p90', 'max', 'sens_mean', 'sens_p90') if m in x)})
            if bad:
                log(dict(event='VAL_NONFINITE', step=step, weights=w, entries=[list(b_) for b_ in bad][:50]))
            cm = [v[c]['cert_mean'] for v in r['val'].values() for c in v if 'cert_mean' in v[c]]
            if cm:
                r['cert'] = dict(mean=_nanmean(cm), max=max(v[c]['cert_max'] for v in r['val'].values() for c in v if 'cert_max' in v[c]),
                                 flagged=_nanmean([v[c]['cert_flag'] for v in r['val'].values() for c in v if 'cert_flag' in v[c]]))
        since = {c: (0 if c in live else (step - visits[c]['last']) if visits.get(c, {}).get('last') is not None else None)
                 for c in probe_cases}
        return res, since

    t0 = time.perf_counter()
    for step in range(start, steps + 1):
        if step % cfg.get('swap_every', 100) == 0 and len(order) > len(pool):
            ts = time.perf_counter()
            if len(pool) >= P:
                leave(pool.pop(0), step)
            admit(order[qi % len(order)], step); qi += 1
            tpre.start(order[qi % len(order)])
            tstat['swap'] += time.perf_counter() - ts
        ts = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        G = min(cfg.get('geos_per_step', 1), len(pool))
        picks = gen.choice(len(pool), size=G, replace=False)
        loss_sum, ls_sum, tl_sum, e_all, skipped, lam_max, counts = 0.0, 0.0, None, [], 0, None, {}
        for pi in picks:
            s = pool[int(pi)]; geo = s.tgeo
            q, s0, kinds, F = s.banks.sample(B, gen, mix, adv=advbuf.get(s.case) if cfg.get('adv_on_load') else None,
                                             quota=cfg.get('quota', False), want_F=want_F,
                                             quota_systematic=cfg.get('quota_systematic', False))
            loss, ls, tl, e, lam = batch_loss(geo, model, q, s0, kinds, F, cfg)
            if not bool(torch.isfinite(loss)):                                 # never let one bad batch poison the weights
                log(dict(event='NONFINITE_LOSS', step=step, geo=s.case)); skipped += 1
                del loss, e
                continue
            (loss / G).backward()                                              # one graph alive at a time
            loss_sum += float(loss) / G; ls_sum += (0.0 if ls is None else float(ls)) / G; e_all.append(e.detach())
            if tl is not None:
                tl_sum = (tl_sum or 0.0) + float(tl) / G; lam_max = max(lam_max or 0.0, float(lam.max()))
            for k in kinds:
                counts[k] = counts.get(k, 0) + 1
            del loss, e
        gn = torch.tensor(float('nan'))
        if skipped < G:
            e = torch.cat(e_all)
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get('clip', 1.0))
            if bool(torch.isfinite(gn)):
                opt.step()
                if ema is not None:
                    ema.update(model)
            else:
                log(dict(event='NONFINITE_GRAD', step=step, geo=s.case))
        sched.step()
        tstat['step'] += time.perf_counter() - ts
        if step % cfg.get('log_every', 50) == 0 and skipped < G:
            rec = dict(event='STEP', step=step, geo=s.case, loss=loss_sum, sens_loss=ls_sum, e_mean=float((e - 1).mean()),
                       grad_norm=float(gn), lr=sched.get_last_lr()[0], counts=counts, s=time.perf_counter() - t0,
                       swap_s=tstat['swap'], step_s=tstat['step'], gpu_GB=_peak_gb())
            if tl_sum is not None:
                rec.update(tail=tl_sum, lam_max=lam_max)
            if s.view is not None:
                rec['view'] = s.k
            if getattr(model, 'bounded', False) and hasattr(model, 'sat_stats'):
                st_ = model.sat_stats()
                rec['sat_max'] = st_['max']
                if 'knee_max' in st_:                                          # M7: compression starts at the knee
                    rec.update(sat_knee_max=st_['knee_max'], sat_peak=st_['peak'])
            log(rec)
        if step % cfg['eval_every'] == 0 or step == steps:
            te = time.perf_counter()
            res, since = evaluate(step)
            sc = res[sel]['score']
            tstat['eval'] += time.perf_counter() - te
            eligible = step >= min_step or step == steps                       # M4: early (warm-start dominated) evals
            if not np.isfinite(sc):
                log(dict(event='VAL_NONFINITE', step=step, score=sc, msg='non-finite selection score: never best.pt'))
            vinfo = dict(eval_views=views, score_identity=res[sel]['score_identity'], score_views=res[sel]['score_views'],
                         score_worst_view=res[sel]['score_worst_view'])
            log(dict(event='EVAL', step=step, select=sel, score=sc, score_parts=res[sel]['score_parts'],
                     val_mean=res[sel]['val_mean'], train_geo_mean=res[sel]['train_geo_mean'], since_visit=since,
                     weights=res, s=time.perf_counter() - t0, eval_s=tstat['eval'], selectable=eligible, **vinfo,
                     conv_tf32=conv['conv_tf32']))
            rec = dict(model=ema.model_state(model) if ema is not None else model.state_dict(), cfg=cfg, step=step, weights=sel,
                       score=sc, **vinfo, conv=conv)
            if ema is not None:
                rec.update(model_raw=model.state_dict(), score_raw=res['raw']['score'] if 'raw' in res else None,
                           ema_t=ema.t, ema_debias=ema.debias)
            torch.save(rec, out / 'last.pt'); torch.save(rec, out / f'snap_{step}.pt')
            if eligible and np.isfinite(sc) and (best is None or sc < best):    # GATE-6: only finite scores
                best = sc
                torch.save(rec, out / 'best.pt')
        if cfg.get('ckpt_every') and step % cfg['ckpt_every'] == 0:
            ckd = dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), step=step, qi=qi, best=best,
                       cfg=cfg, ema=ema.state_dict() if ema is not None else None, visits=visits, conv=conv)
            if cfg.get('resume_rng', False):                                    # TRAINER-7: the exact continuation state
                ckd.update(rng=rng_state(), pool=[(p_.case, p_.k) for p_ in pool], advbuf=adv_for_ckpt())
            torch.save(ckd, out / 'ckpt.tmp')
            (out / 'ckpt.tmp').replace(ck_path)
        if cfg.get('stop_after') and step >= cfg['stop_after'] and step < steps:
            log(dict(event='STOP', step=step)); tpre.cancel(); log_f.close()
            return
    tpre.cancel()
    log(dict(event='DONE', seconds=time.perf_counter() - t0, best_score=best, time=tstat))
    log_f.close()


if __name__ == '__main__':
    main(json.loads(Path(sys.argv[1]).read_text()))
