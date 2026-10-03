"""Step-2 trainer: one network over many geometries (same loss as train1: log e_hat + sens_w * relative sensitivity error).

A pool of `pool` geometries is resident on the GPU; the others stay on disk (slot cache) and are swapped in one at a time
every `swap_every` steps. A geometry leaving the pool is dropped from memory (the container has a 90 GiB host limit and one
slot is 1.5-4.3 GB, so the run cannot park every geometry on the host); validation loads each geometry, evaluates it and
drops it again. On entry a geometry optionally gets fresh adversarial directions (fp32 Neumann
factor, block power iteration, factor freed afterwards). Validation: bank-level errors on the validation geometries.
Usage: train2.py <config.json>
config: split (make_split.py json), train_key ('curve:50' | 'train'), val_max, body, data, out, model, model_args, steps,
        batch, lr, pct_start, mix, pool, swap_every, adv_on_load, adv_k, eval_every, seed, sens_w, init,
        ckpt_every (model + optimizer + scheduler state to out/ckpt.pt), resume (continue from out/ckpt.pt),
        pool_dofs (cap on the summed DOFs of the pool incl. the incoming geometry; older members leave early to make room,
        so two 400k-DOF geometries are never resident together; exposure per geometry is unchanged in simulation)
"""
import json, sys, time, gc, math
from pathlib import Path
import numpy as np
import torch
import trainlib as TL
import models as MD

dev, dt = TL.dev, TL.dt
KEEP_CELL = ('U', 'Ut', 'dK', 'dofs', 'P', 'I', 'Q', 'Tm', 'n', 'nb', 'np_', 'ni', 'case', 'cells', 'nodes',
             'port_node_ids', 'port_is_box', 'port_is_cut', 'is_box', 'is_cut', 'taus', 'normal', 'offset', 's', 'levels',
             'crow', 'cu', 'ru', 'vals', 'diag', 'K', 'sol_I', 'sol_N', 'fp32', 'Qall')


def move(obj, device, seen=None):
    """Move every tensor reachable through attributes / dicts / lists of obj to device (in place)."""
    seen = set() if seen is None else seen
    if id(obj) in seen:
        return obj
    seen.add(id(obj))
    if torch.is_tensor(obj):
        return obj.to(device)          # blocking: a non_blocking copy to the host lands in the never-released pinned cache
    if isinstance(obj, dict):
        for k in list(obj):
            obj[k] = move(obj[k], device, seen)
        return obj
    if isinstance(obj, list):
        for i in range(len(obj)):
            obj[i] = move(obj[i], device, seen)
        return obj
    if hasattr(obj, '__dict__') and not isinstance(obj, (torch.nn.Module, type)):
        for k, v in list(vars(obj).items()):
            if torch.is_tensor(v) or isinstance(v, (dict, list)) or (hasattr(v, '__dict__') and type(v).__module__ in ('teacher', 'models', 'types')):
                setattr(obj, k, move(v, device, seen))
    return obj


def build_geo(case, cfg):
    g = TL.Geo(case, cfg['body'], cfg['data'], neumann=False, log=lambda s_: None)
    C = g.C
    for k in list(vars(C)):
        if k not in KEEP_CELL:
            delattr(C, k)
    C.K = C
    return g


def clean_banks(g, case, log, min_keep=8):
    """Drop q samples whose direction or exact sensitivities are non-finite (the support-spring solve failed for most
    faces on 5 of the 206 geometries); a class left with fewer than min_keep samples in some split is removed."""
    dropped = {}
    for c in list(g.classes):
        for s_ in list(g.banks):
            ok = torch.isfinite(g.banks[s_][c]).all(0)
            has_s = g.sens is not None and c in g.sens.get(s_, {})
            if has_s:
                ok &= torch.isfinite(g.sens[s_][c]).all(0)
            if not bool(ok.all()):
                dropped[f'{s_}/{c}'] = int((~ok).sum())
                g.banks[s_][c] = g.banks[s_][c][:, ok].contiguous()
                if has_s:
                    g.sens[s_][c] = g.sens[s_][c][:, ok].contiguous()
        if min(g.banks[s_][c].shape[1] for s_ in g.banks) < min_keep:
            g.classes.remove(c)
            for s_ in g.banks:
                g.banks[s_].pop(c, None)
                if g.sens is not None:
                    g.sens.get(s_, {}).pop(c, None)
            dropped[c] = 'class removed'
    if dropped:
        log(dict(event='BANK_CLEAN', case=case, dropped=dropped))


class Slot:
    """One geometry: a trainlib.Geo slimmed to what training needs, plus the model cache; lives on cpu or cuda."""

    def __init__(self, case, cfg, model, log):
        t0 = time.perf_counter()
        self.case = case
        cache = Path(cfg['slot_cache']) / f'{case}.pt' if cfg.get('slot_cache') else None
        if cache is not None and cache.exists():
            g = torch.load(cache, map_location='cpu', weights_only=False)
            move(g, 'cuda'); g.C.K = g.C
        else:
            g = build_geo(case, cfg)
            if cache is not None:
                cache.parent.mkdir(parents=True, exist_ok=True)
                move(g, 'cpu'); torch.save(g, cache); move(g, 'cuda'); g.C.K = g.C
        clean_banks(g, case, log)
        self.geo = g
        model.add_geo(g)
        self.cache = model.caches[case]
        self.where = 'cuda'
        self.build_seconds = time.perf_counter() - t0
        log(dict(event='BUILD', case=case, seconds=self.build_seconds, dofs=g.nb, ports=g.np_))

    def drop(self, model):
        """Forget this geometry entirely (it is reloaded from the slot cache when needed again). The tensors go to the host
        first, so a stale reference (a loop variable, the model's current-cache pointer) cannot keep them on the GPU."""
        move(self.geo, 'cpu'); move(self.cache, 'cpu')
        if getattr(model, '_cur', None) is self.cache:
            model._cur = None
        model.caches.pop(self.case, None)
        self.geo = self.cache = None
        self.where = None
        gc.collect(); torch.cuda.empty_cache()

    def to(self, device, model):
        if self.where == device:
            return
        move(self.geo, device); move(self.cache, device)
        if device == 'cpu':
            model.caches.pop(self.case, None)
        else:
            model.caches[self.case] = self.cache
        self.where = device
        torch.cuda.synchronize(); gc.collect(); torch.cuda.empty_cache()


def adversarial_on_load(slot, model, cfg, tgen, log):
    g = slot.geo; C = g.C
    t = time.perf_counter()
    try:
        C.factor(neumann=True, interior=False, fp32_neumann=True)
        model.eval()
        X, ritz = g.adversarial(model, k=cfg.get('adv_k', 16), iters=cfg.get('adv_iters', 4), gen=tgen)
        model.train()
        g.adv = X.to(torch.float32)
        log(dict(event='ADV', case=slot.case, ritz_top=ritz[:3].tolist(), seconds=time.perf_counter() - t))
    except Exception as e:
        log(dict(event='ADV_FAIL', case=slot.case, error=repr(e)[:200]))
    finally:
        C._free() if hasattr(C, '_free') else None
        C.sol_N = None
        gc.collect(); torch.cuda.empty_cache()


@torch.no_grad()
def validate(cases, get, release, model, chunk=8):
    model.eval()
    out = {}
    for case in cases:
        s = get(case)
        g = s.geo
        r = {}
        for c in g.classes:
            Q = g.banks['val'][c]
            e = torch.cat([TL.energy(g.field(model, Q[:, j:j + chunk]), g.C.K) - 1 for j in range(0, Q.shape[1], chunk)])
            r[c] = float(e.mean())
        out[s.case] = r
        release(s)
    model.train()
    return out


def main(cfg):
    out = Path(cfg['out']); out.mkdir(parents=True, exist_ok=True)
    log_f = open(out / 'train.log', 'a')

    def log(d):
        s = json.dumps(d); print(s, flush=True); log_f.write(s + '\n'); log_f.flush()

    torch.manual_seed(cfg.get('seed', 0))
    gen = np.random.default_rng(cfg.get('seed', 0))
    tgen = torch.Generator(device=dev).manual_seed(cfg.get('seed', 0))
    split = json.loads(Path(cfg['split']).read_text())
    key = cfg.get('train_key', 'train')
    train_cases = split['curve'][key.split(':')[1]]['geometries'] if key.startswith('curve:') else split['train']
    val_cases = split['val'][:cfg.get('val_max', 8)]
    log(dict(event='SPLIT', train=len(train_cases), val=len(val_cases)))
    # the model needs one geometry at construction; the rest are added on the fly
    first = TL.Geo(train_cases[0], cfg['body'], cfg['data'], neumann=False, log=lambda s_: None)
    model = MD.build(cfg['model'], [first], **cfg.get('model_args', {})).to(dev)
    del first; model.caches.clear(); gc.collect(); torch.cuda.empty_cache()
    if cfg.get('init'):
        ck = torch.load(cfg['init'], map_location=dev, weights_only=False)
        res = model.load_state_dict(ck['model'], strict=False)
        log(dict(event='INIT', ckpt=cfg['init'], missing=len(res.missing_keys)))
    log(dict(event='MODEL', params=sum(p.numel() for p in model.parameters())))
    opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    steps = cfg['steps']
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg['lr'], total_steps=steps, pct_start=cfg.get('pct_start', 0.05),
                                                anneal_strategy='cos', final_div_factor=cfg.get('final_div', 100))
    slots = {}                                                                  # live slots only (pool members)

    def get(case):
        if case not in slots:
            slots[case] = Slot(case, cfg, model, log)
        else:
            slots[case].to('cuda', model)
        return slots[case]

    def drop(s):
        s.drop(model); slots.pop(s.case, None)

    ndofs = {}

    def dofs(case):
        if case not in ndofs:
            ndofs[case] = json.loads((Path(cfg['data']) / case / 'DONE.json').read_text())['dofs']
        return ndofs[case]

    cap = cfg.get('pool_dofs', 0)

    def admit(case):
        """Add a geometry to the pool, first evicting the oldest members while the pool would exceed the DOF cap."""
        while cap and pool and sum(dofs(p_.case) for p_ in pool) + dofs(case) > cap:
            o = pool.pop(0); drop(o); log(dict(event='EVICT', case=o.case, incoming=case))
        s_ = get(case)
        if cfg.get('adv_on_load'):
            adversarial_on_load(s_, model, cfg, tgen, log)
        pool.append(s_)

    order = list(train_cases); gen.shuffle(order)
    qi, start = 0, 1
    ck_path = out / 'ckpt.pt'
    if cfg.get('resume') and ck_path.exists():
        ck = torch.load(ck_path, map_location=dev, weights_only=False)
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['opt']); sched.load_state_dict(ck['sched'])
        start, qi, best = ck['step'] + 1, ck['qi'], ck.get('best')
        gen = np.random.default_rng([cfg.get('seed', 0), ck['step']])
        log(dict(event='RESUME', step=ck['step'], qi=qi))
    else:
        best = None
    qi0 = max(qi - cfg.get('pool', 4), 0)                                      # rebuild the pool the run had at the checkpoint
    qi = qi0
    pool = []
    for _ in range(min(cfg.get('pool', 4), len(order))):
        admit(order[qi % len(order)]); qi += 1
    probe_cases = order[:cfg.get('probe_max', 0)]                              # training geometries, unseen q (generalization gap)
    mix = dict(cfg['mix']); B = cfg['batch']; sw = cfg.get('sens_w', 1.0)
    t0 = time.perf_counter(); tstat = dict(swap=0.0, step=0.0)
    for step in range(start, steps + 1):
        if step % cfg.get('swap_every', 250) == 0 and len(order) > len(pool):
            ts = time.perf_counter()
            if len(pool) >= cfg.get('pool', 4):
                old = pool.pop(0); drop(old)
            admit(order[qi % len(order)]); qi += 1
            tstat['swap'] += time.perf_counter() - ts
        ts = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        G = min(cfg.get('geos_per_step', 1), len(pool))
        picks = gen.choice(len(pool), size=G, replace=False)
        loss_sum, ls_sum, e_all, skipped = 0.0, 0.0, [], 0
        for pi in picks:
            s = pool[int(pi)]; geo = s.geo
            q, s0 = geo.sample_with_sens(B, gen, mix)
            u = geo.field(model, q)
            e = TL.energy(u, geo.C.K)
            loss = torch.log(e.clamp_min(1e-12)).mean()
            ok = ~torch.isnan(s0[0]); ls = None
            if sw > 0 and ok.any():
                sh = geo.sens_hat(u[:, ok])
                ls = (((sh - s0[:, ok]) ** 2).sum(0) / (s0[:, ok] ** 2).sum(0)).mean()
                loss = loss + sw * ls
            if not bool(torch.isfinite(loss)):                                 # never let one bad batch poison the weights
                log(dict(event='NONFINITE_LOSS', step=step, geo=s.case)); skipped += 1
                del u, e, loss
                continue
            (loss / G).backward()                                              # one graph alive at a time
            loss_sum += float(loss) / G; ls_sum += (0.0 if ls is None else float(ls)) / G; e_all.append(e.detach())
            del u, e, loss
        gn = torch.tensor(float('nan'))
        if skipped < G:
            loss, ls, e = torch.tensor(loss_sum), ls_sum, torch.cat(e_all)
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get('clip', 1.0))
            if bool(torch.isfinite(gn)):
                opt.step()
            else:
                log(dict(event='NONFINITE_GRAD', step=step, geo=s.case))
        sched.step()
        tstat['step'] += time.perf_counter() - ts
        if step % 50 == 0 and skipped < G:
            log(dict(event='STEP', step=step, geo=s.case, loss=float(loss), sens_loss=float(ls),
                     e_mean=float((e - 1).mean()), grad_norm=float(gn), lr=sched.get_last_lr()[0], s=time.perf_counter() - t0,
                     swap_s=tstat['swap'], step_s=tstat['step'], gpu_GB=torch.cuda.max_memory_allocated() / 2 ** 30))
        if step % cfg['eval_every'] == 0 or step == steps:
            for p_ in pool:                                                    # pool members wait on the host (<= 13 GB)
                p_.to('cpu', model)
            live = {p_.case for p_ in pool}

            def release(s_):                                                   # pool members back to the host, others dropped
                s_.to('cpu', model) if s_.case in live else drop(s_)
            val = validate(val_cases, get, release, model)
            probe = validate(probe_cases, get, release, model) if probe_cases else {}
            for p_ in pool:
                p_.to('cuda', model)
            per_class = {c: float(np.mean([v[c] for v in val.values() if c in v])) for c in mix if c != 'adv'}
            per_class_probe = {c: float(np.mean([v[c] for v in probe.values() if c in v])) for c in mix if c != 'adv'} if probe else {}
            score = max(per_class.values())
            log(dict(event='EVAL', step=step, val_mean=per_class, train_geo_mean=per_class_probe, val=val, train_geo=probe,
                     score=score, s=time.perf_counter() - t0))
            out.mkdir(parents=True, exist_ok=True)
            torch.save(dict(model=model.state_dict(), cfg=cfg, step=step), out / 'last.pt')
            if best is None or score < best:
                best = score
                torch.save(dict(model=model.state_dict(), cfg=cfg, step=step), out / 'best.pt')
        if cfg.get('ckpt_every') and step % cfg['ckpt_every'] == 0:
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), step=step, qi=qi,
                            best=best, cfg=cfg), out / 'ckpt.tmp')
            (out / 'ckpt.tmp').replace(ck_path)
    log(dict(event='DONE', seconds=time.perf_counter() - t0, best_score=best))


if __name__ == '__main__':
    main(json.loads(Path(sys.argv[1]).read_text()))
