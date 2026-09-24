"""Step-2 trainer: one network over many geometries (same loss as train1: log e_hat + sens_w * relative sensitivity error).

A pool of `pool` geometries is resident on the GPU; the others wait in host memory (built once, then moved) and are swapped
in one at a time every `swap_every` steps. On entry a geometry optionally gets fresh adversarial directions (fp32 Neumann
factor, block power iteration, factor freed afterwards). Validation: bank-level errors on the validation geometries.
Usage: train2.py <config.json>
config: split (make_split.py json), train_key ('curve:50' | 'train'), val_max, body, data, out, model, model_args, steps,
        batch, lr, pct_start, mix, pool, swap_every, adv_on_load, adv_k, eval_every, seed, sens_w, init
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
        return obj.to(device, non_blocking=True)
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
        self.geo = g
        model.add_geo(g)
        self.cache = model.caches[case]
        self.where = 'cuda'
        self.build_seconds = time.perf_counter() - t0
        log(dict(event='BUILD', case=case, seconds=self.build_seconds, dofs=g.nb, ports=g.np_))

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
def validate(slots, model, chunk=8):
    model.eval()
    out = {}
    for s in slots:
        s.to('cuda', model)
        g = s.geo
        r = {}
        for c in g.classes:
            Q = g.banks['val'][c]
            e = torch.cat([TL.energy(g.field(model, Q[:, j:j + chunk]), g.C.K) - 1 for j in range(0, Q.shape[1], chunk)])
            r[c] = float(e.mean())
        out[s.case] = r
        s.to('cpu', model)
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
    slots = {}

    def get(case):
        if case not in slots:
            slots[case] = Slot(case, cfg, model, log)
        else:
            slots[case].to('cuda', model)
        return slots[case]

    order = list(train_cases); gen.shuffle(order)
    qi = 0
    pool = []
    for _ in range(min(cfg.get('pool', 4), len(order))):
        s = get(order[qi % len(order)]); qi += 1
        if cfg.get('adv_on_load'):
            adversarial_on_load(s, model, cfg, tgen, log)
        pool.append(s)
    for c in val_cases:                                                       # build validation slots, park on the host
        get(c).to('cpu', model)
    probe_cases = order[:cfg.get('probe_max', 0)]                              # training geometries, unseen q (generalization gap)
    mix = dict(cfg['mix']); B = cfg['batch']; sw = cfg.get('sens_w', 1.0)
    t0 = time.perf_counter(); best = None; tstat = dict(swap=0.0, step=0.0)
    for step in range(1, steps + 1):
        if step % cfg.get('swap_every', 250) == 0 and len(order) > len(pool):
            ts = time.perf_counter()
            old = pool.pop(0); old.to('cpu', model)
            s = get(order[qi % len(order)]); qi += 1
            if cfg.get('adv_on_load'):
                adversarial_on_load(s, model, cfg, tgen, log)
            pool.append(s)
            tstat['swap'] += time.perf_counter() - ts
        ts = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        G = min(cfg.get('geos_per_step', 1), len(pool))
        picks = gen.choice(len(pool), size=G, replace=False)
        loss_sum, ls_sum, e_all = 0.0, 0.0, []
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
            (loss / G).backward()                                              # one graph alive at a time
            loss_sum += float(loss) / G; ls_sum += (0.0 if ls is None else float(ls)) / G; e_all.append(e.detach())
            del u, e, loss
        loss, ls, e = torch.tensor(loss_sum), ls_sum, torch.cat(e_all)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get('clip', 1.0))
        opt.step(); sched.step()
        tstat['step'] += time.perf_counter() - ts
        if step % 50 == 0:
            log(dict(event='STEP', step=step, geo=s.case, loss=float(loss), sens_loss=float(ls),
                     e_mean=float((e - 1).mean()), grad_norm=float(gn), lr=sched.get_last_lr()[0], s=time.perf_counter() - t0,
                     swap_s=tstat['swap'], step_s=tstat['step'], gpu_GB=torch.cuda.max_memory_allocated() / 2 ** 30))
        if step % cfg['eval_every'] == 0 or step == steps:
            for p_ in pool:
                p_.to('cpu', model)
            val = validate([slots[c] for c in val_cases], model)
            probe = validate([get(c) if c in slots else get(c) for c in probe_cases], model) if probe_cases else {}
            for p_ in pool:
                p_.to('cuda', model)
            per_class = {c: float(np.mean([v[c] for v in val.values() if c in v])) for c in mix if c != 'adv'}
            per_class_probe = {c: float(np.mean([v[c] for v in probe.values() if c in v])) for c in mix if c != 'adv'} if probe else {}
            score = max(per_class.values())
            log(dict(event='EVAL', step=step, val_mean=per_class, train_geo_mean=per_class_probe, val=val, train_geo=probe,
                     score=score, s=time.perf_counter() - t0))
            torch.save(dict(model=model.state_dict(), cfg=cfg, step=step), out / 'last.pt')
            if best is None or score < best:
                best = score
                torch.save(dict(model=model.state_dict(), cfg=cfg, step=step), out / 'best.pt')
    log(dict(event='DONE', seconds=time.perf_counter() - t0, best_score=best))


if __name__ == '__main__':
    main(json.loads(Path(sys.argv[1]).read_text()))
