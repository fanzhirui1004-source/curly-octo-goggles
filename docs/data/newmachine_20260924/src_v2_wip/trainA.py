"""Direction A: fine-tune the P1 network (A3) on mapped cells with stretch-conditioned inputs (new script; runs on the GPU).

Model: A3 + a_ucond.attach() (zero-initialised stretch input, so step 0 is A3 exactly). Deployed extension: V0R with the
co-rotated Jacobi preconditioner (corot_smooth) and det-normalised diag3 (a_ucond.prepare); the correction (A3's
tail(8) - Q1_17 - tail(8)) is inside the loss, as in P1 (Eq. (18) without the sensitivity term):
    loss = mean_j log(q_j^T S_hat q_j),  q_j at unit exact energy (bank directions of mapped training cells).
Data: a0_eval data-only runs (RESULTS.jsonl gives each (case, map) spec; data/<map>/<case>). A pool of resident geometries
is swapped every --swap steps. Validation: fixed (case, map) pairs from other a0_eval run directories, assembled at each
evaluation and released. fp32 network, fp64 energies, no TF32 (OPL_CONV_FP32=1).
Usage: trainA.py <out> <ckpt> <body> --train runs/gen1 [--val runs/full16:id,strx2,strx0.5:case1,case2 ...]
       [--steps 3000] [--lr 1e-4] [--lr_u 1e-3] [--batch 16] [--pool 3] [--swap 100] [--eval_every 1000] [--id_frac 0.25]"""
import argparse, gc, json, math, sys, time
from pathlib import Path
import numpy as np
import torch

import models as MD                                                       # first: applies OPL_CONV_FP32
import trainlib as TL
import a0_eval as AE
import mapped_cell as MC
import corot_smooth as CR
import a_ucond as AU

dev, dt = AE.dev, AE.dt


def load_pairs(run_dir):
    last = {}
    for r in map(json.loads, (Path(run_dir) / 'RESULTS.jsonl').read_text().splitlines()):
        last[(r['case'], r['map'])] = r
    return [(c, m, r['spec']) for (c, m), r in last.items() if 'error' not in r]


class Slot:
    """One resident mapped geometry ready for the deployed extension."""

    def __init__(self, model, wrap, body, run_dir, case, mname, spec, det_norm=True):
        self.C = MC.MappedCell(case, body, spec, log=lambda s_: None); self.C.assemble()
        self.g = AE.MappedGeo(case, body, Path(run_dir) / 'data' / mname, neumann=False, log=lambda s_: None, cell=self.C)
        self.g.case = f'{case}@{mname}@{Path(run_dir).name}'
        AU.prepare(self.g, self.C, det_norm)
        model.add_geo(self.g); AU.set_ufeat(model, self.g, self.C)
        self.g.set_variant('V0R', wrap)
        CR.set_corot(self.C, AE.nodal_rotations(self.C))
        gc.collect(); torch.cuda.empty_cache()
        with torch.no_grad():                                # coarse factor and smoothing interval outside any graph
            TL.coarse_setup(self.C, wrap.coarse_space); TL.tail_bounds(self.C, wrap.smooth_alpha)
        torch.cuda.empty_cache()
        self.case, self.map, self.model = case, mname, model

    def release(self):
        self.model.caches.pop(self.g.case, None)
        self.C._free(); del self.g, self.C


def energies(slot, model, Q, chunk=16, grad=False):
    out = []
    for j in range(0, Q.shape[1], chunk):
        out.append(TL.energy(slot.g.field(model, Q[:, j:j + chunk]), slot.C.K))
    return torch.cat(out)


def evaluate(model, wrap, body, val, classes=('force_c', 'face_c'), log=print):
    model.eval(); res = []
    for run_dir, case, mname, spec in val:
        s = None
        try:
            s = Slot(model, wrap, body, run_dir, case, mname, spec)
            r = dict(case=case, map=mname, run=Path(run_dir).name)
            with torch.no_grad():
                for cls in classes:
                    e = (energies(s, model, s.g.banks['val'][cls]) - 1).cpu().numpy()
                    r[cls] = dict(mean=float(e.mean()), max=float(e.max()), min=float(e.min()))
            res.append(r)
        except Exception as ex:
            res.append(dict(case=case, map=mname, error=repr(ex)[:300]))
        finally:
            if s is not None:
                s.release()
            gc.collect(); torch.cuda.empty_cache()
    model.train()
    ok = [r for r in res if 'error' not in r]
    score = float(np.mean([math.log1p(r['force_c']['mean']) for r in ok])) if ok else float('nan')
    return score, res


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('ckpt'); ap.add_argument('body')
    ap.add_argument('--train', action='append', required=True); ap.add_argument('--val', action='append', default=[])
    ap.add_argument('--steps', type=int, default=3000); ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--lr_u', type=float, default=1e-3); ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--pool', type=int, default=3); ap.add_argument('--swap', type=int, default=100)
    ap.add_argument('--eval_every', type=int, default=1000); ap.add_argument('--id_frac', type=float, default=0.25)
    ap.add_argument('--classes', default='force,macro,grf,force_c,face_c'); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--clip', type=float, default=1.0)
    ap.add_argument('--accum', type=int, default=1)                          # micro-batches per step (memory)
    ap.add_argument('--freeze_base', type=int, default=0)                    # train only u_in: similarity maps stay A3 exactly
    ap.add_argument('--l2sp', type=float, default=0.0)                       # lam * sum (theta - theta_A3)^2 over A3 parameters
    ap.add_argument('--ema', type=float, default=0.0)                        # EMA of the weights for evaluation / saving
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    logf = open(out / 'train.log', 'a')

    def log(obj):
        s = obj if isinstance(obj, str) else json.dumps(obj)
        print(s, flush=True); logf.write(s + '\n'); logf.flush()
    rng = np.random.default_rng(a.seed); torch.manual_seed(a.seed)
    pairs = [(rd, c, m, sp) for rd in a.train for (c, m, sp) in load_pairs(rd)]
    ids = [p for p in pairs if p[2] == 'id']; maps = [p for p in pairs if p[2] != 'id']
    val = []
    for v in a.val:                                                                 # run_dir:maps:cases
        rd, ms, cs = v.split(':')
        specs = {(c, m): sp for (c, m, sp) in load_pairs(rd)}
        val += [(rd, c, m, specs[(c, m)]) for m in ms.split(',') for c in cs.split(',') if (c, m) in specs]
    log(dict(event='SETUP', train_pairs=len(pairs), id_pairs=len(ids), val_pairs=len(val), args=vars(a)))
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False); cfg = ck['cfg']
    first = Slot.__new__(Slot)                                                       # build the model on a first geometry
    p0 = ids[0] if ids else pairs[0]
    C0 = MC.MappedCell(p0[1], a.body, p0[3], log=lambda s_: None); C0.assemble()
    g0 = AE.MappedGeo(p0[1], a.body, Path(p0[0]) / 'data' / p0[2], neumann=False, log=lambda s_: None, cell=C0)
    model = MD.build(cfg['model'], [g0], **dict(cfg.get('model_args', {}))).to(dev)
    MD.load_compat(model, ck['model']); AU.attach(model)
    model.caches.pop(g0.case, None); C0._free(); del g0, C0, first
    wrap = AE._Wrap(model)
    CR.install()
    u_params = list(model.u_in.parameters()); u_ids = {id(p) for p in u_params}
    base = [p for p in model.parameters() if id(p) not in u_ids]
    if a.freeze_base:
        for p in base:
            p.requires_grad_(False)
        opt = torch.optim.AdamW([dict(params=u_params, lr=a.lr_u)], weight_decay=0.0)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr_u], total_steps=a.steps, pct_start=0.05)
    else:
        opt = torch.optim.AdamW([dict(params=base, lr=a.lr), dict(params=u_params, lr=a.lr_u)], weight_decay=0.0)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr, a.lr_u], total_steps=a.steps, pct_start=0.05)
    classes = a.classes.split(',')
    theta0 = [p.detach().clone() for p in base] if a.l2sp > 0 else None
    ema = [p.detach().clone() for p in model.parameters()] if a.ema > 0 else None

    def swap_ema():                                                           # exchange live and EMA weights in place
        if ema is not None:
            with torch.no_grad():
                for p, e in zip(model.parameters(), ema):
                    t = p.detach().clone(); p.copy_(e); e.copy_(t)

    def pick():
        src = ids if (ids and rng.random() < a.id_frac) else maps
        return src[rng.integers(len(src))]

    best = float('inf')
    if val:
        score, res = evaluate(model, wrap, a.body, val, log=log)
        log(dict(event='EVAL', step=0, score=score, res=res)); best = score
        torch.save(dict(model=model.state_dict(), cfg=cfg, step=0, score=score, u=True), out / 'best.pt')
    pool = []
    model.train()
    t0 = time.perf_counter(); hist = []
    for step in range(1, a.steps + 1):
        if len(pool) < a.pool or (step % a.swap == 0):
            if len(pool) >= a.pool:
                pool.pop(int(rng.integers(len(pool)))).release(); gc.collect(); torch.cuda.empty_cache()
            while len(pool) < a.pool:
                p = pick()
                try:
                    pool.append(Slot(model, wrap, a.body, *p))
                except Exception as ex:
                    log(dict(event='SLOT_FAIL', pair=p[1:3], err=repr(ex)[:200]))
        s = pool[step % len(pool)]
        cls = [c for c in classes if c in s.g.classes]
        cs = rng.choice(cls, a.batch)
        cols = []
        for c in cls:
            k = int((cs == c).sum())
            if k:
                Qc = s.g.banks['train'][c]; cols.append(Qc[:, torch.as_tensor(rng.choice(Qc.shape[1], k, replace=False), device=dev)])
        Q = torch.cat(cols, 1)
        opt.zero_grad(set_to_none=True)
        loss = 0.0
        for Qm in torch.tensor_split(Q, a.accum, dim=1):                       # same mean over the batch, less memory
            e = TL.energy(s.g.field(model, Qm), s.C.K)
            lm = torch.log(e.clamp_min(1e-12)).sum() / Q.shape[1]
            lm.backward(); loss = loss + lm.detach()
        if theta0 is not None:
            reg = a.l2sp * sum(((p - p0) ** 2).sum() for p, p0 in zip(base, theta0))
            reg.backward(); loss = loss + reg.detach()
        gn = float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], a.clip))
        opt.step(); sched.step()
        if ema is not None:
            with torch.no_grad():
                for p, e_ in zip(model.parameters(), ema):
                    e_.mul_(a.ema).add_(p.detach(), alpha=1 - a.ema)
        hist.append(float(loss.detach()))
        if step % 50 == 0:
            log(dict(event='STEP', step=step, loss=float(np.mean(hist[-50:])), gn=gn, map=s.map, case=s.case,
                     s_per_step=(time.perf_counter() - t0) / step, mem_gb=torch.cuda.max_memory_allocated() / 2 ** 30,
                     u_norm=float(model.u_in.weight.norm())))
        if val and (step % a.eval_every == 0 or step == a.steps):
            for sl in pool:
                sl.release()
            pool = []; gc.collect(); torch.cuda.empty_cache()
            swap_ema()
            score, res = evaluate(model, wrap, a.body, val, log=log)
            log(dict(event='EVAL', step=step, score=score, res=res, ema=a.ema))
            torch.save(dict(model=model.state_dict(), cfg=cfg, step=step, score=score, u=True), out / 'last.pt')
            if score < best:
                best = score; torch.save(dict(model=model.state_dict(), cfg=cfg, step=step, score=score, u=True), out / 'best.pt')
            swap_ema()
    log(dict(event='DONE', best=best, seconds=time.perf_counter() - t0))


if __name__ == '__main__':
    main(sys.argv[1:])
