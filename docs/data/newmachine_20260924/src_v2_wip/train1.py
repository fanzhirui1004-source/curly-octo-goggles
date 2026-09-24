"""Step 1 trainer: one or several fixed geometries, energy loss, adversarial directions, checkpoints.

loss = mean over the batch of log(e_hat(q)) with the banks at unit exact energy (log e_hat ~ e_hat - 1 near the
solution; robust while e_hat >> 1 early). Evaluation: validation banks per class (mean / p90 / max of e_hat - 1) and the
block-power estimate of the worst direction (largest eigenvalue of S^-1 S_hat, reported).
Usage: train1.py <config.json>
config: {"cases": [...], "body": ..., "data": ..., "out": ..., "model": "<name>", "model_args": {...}, "steps": N,
         "batch": B, "lr": x, "mix": {"force": .3, "macro": .2, "grf": .3, "adv": .2}, "adv_start": N, "adv_every": N,
         "adv_k": 16, "eval_every": N, "seed": s}
        sens_loss: 'sq' (default, relative squared error) | 'smoothl1' (mean sqrt(rho^2 + sens_delta^2) - sens_delta, sens_delta
        default 0.003; trainlib.sens_loss)
"""
import json, sys, time, gc, math
from pathlib import Path
import numpy as np
import torch
import trainlib as TL
import models as MD

dev, dt = TL.dev, TL.dt


def main(cfg):
    out = Path(cfg['out']); out.mkdir(parents=True, exist_ok=True)
    log_f = open(out / 'train.log', 'a')

    def log(d):
        s = json.dumps(d); print(s, flush=True); log_f.write(s + '\n'); log_f.flush()

    torch.manual_seed(cfg.get('seed', 0))
    gen = np.random.default_rng(cfg.get('seed', 0))
    tgen = torch.Generator(device=dev).manual_seed(cfg.get('seed', 0))
    geos = [TL.Geo(c, cfg['body'], cfg['data'], log=lambda s_: print(s_, flush=True)) for c in cfg['cases']]
    if cfg.get('sens_w', 0) > 0 and any(g_.sens is None for g_ in geos):
        raise ValueError('SENS_LABELS_MISSING')                               # never fall back silently
    model = MD.build(cfg['model'], geos, **cfg.get('model_args', {})).to(dev)
    if cfg.get('init'):                                                       # warm start from a checkpoint
        ck = torch.load(cfg['init'], map_location=dev, weights_only=False)
        res = model.load_state_dict(ck['model'], strict=False)
        log(dict(event='INIT', ckpt=cfg['init'], step=ck.get('step'), missing=len(res.missing_keys), unexpected=len(res.unexpected_keys)))
    nparam = sum(p.numel() for p in model.parameters())
    log(dict(event='MODEL', name=cfg['model'], params=nparam, args=cfg.get('model_args', {})))
    opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    steps = cfg['steps']
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg['lr'], total_steps=steps, pct_start=cfg.get('pct_start', 0.05),
                                                anneal_strategy='cos', final_div_factor=cfg.get('final_div', 100))
    mix = dict(cfg['mix']); B = cfg['batch']
    t0 = time.perf_counter(); best = None
    for step in range(1, steps + 1):
        gi = step % len(geos)
        geo = geos[gi]
        sw = cfg.get('sens_w', 0.0)
        if sw > 0 and geo.sens is not None:
            q, s0 = geo.sample_with_sens(B, gen, mix)
        else:
            q, s0 = geo.sample(B, gen, mix), None
        u = geo.field(model, q)
        e = TL.energy(u, geo.C.K)
        loss = torch.log(e.clamp_min(1e-12)).mean()
        ls = None
        if s0 is not None:
            ok = ~torch.isnan(s0[0])
            if ok.any():
                sh = geo.sens_hat(u[:, ok])
                ls = TL.sens_loss(sh, s0[:, ok], cfg.get('sens_loss', 'sq'), cfg.get('sens_delta', 0.003))
                loss = loss + sw * ls
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get('clip', 1.0))
        opt.step(); sched.step()
        if step % 50 == 0:
            log(dict(event='STEP', step=step, geo=geo.case, loss=float(loss), sens_loss=None if ls is None else float(ls), e_mean=float((e - 1).mean()),
                     e_max=float((e - 1).max()), grad_norm=float(gn), lr=sched.get_last_lr()[0], s=time.perf_counter() - t0))
        if cfg.get('adv_start') and step >= cfg['adv_start'] and step % cfg['adv_every'] == 0:
            for g_ in geos:
                X, ritz = g_.adversarial(model, k=cfg.get('adv_k', 16), iters=cfg.get('adv_iters', 6), gen=tgen,
                                         start=None if g_.adv is None else g_.adv[:, :cfg.get('adv_k', 16)].to(dt))
                Xn = X.to(torch.float32)
                g_.adv = Xn if g_.adv is None else torch.cat([Xn, g_.adv], 1)[:, :cfg.get('adv_buffer', 256)]
                log(dict(event='ADV', step=step, geo=g_.case, ritz_top=ritz[:4].tolist(), ritz_min=float(ritz[-1])))
        if step % cfg['eval_every'] == 0 or step == steps:
            model.eval()
            ev = {g_.case: g_.evaluate(model, 'val') for g_ in geos}
            worst = {}
            for g_ in geos:
                _, ritz = g_.adversarial(model, k=8, iters=10, gen=torch.Generator(device=dev).manual_seed(7))
                worst[g_.case] = float(ritz[0])
            model.train()
            score = max(v[c]['mean'] for v in ev.values() for c in v)
            log(dict(event='EVAL', step=step, val=ev, worst_ratio=worst, score=score, s=time.perf_counter() - t0))
            torch.save(dict(model=model.state_dict(), cfg=cfg, step=step), out / 'last.pt')
            if best is None or score < best:
                best = score
                torch.save(dict(model=model.state_dict(), cfg=cfg, step=step), out / 'best.pt')
    log(dict(event='DONE', seconds=time.perf_counter() - t0, best_score=best))


if __name__ == '__main__':
    main(json.loads(Path(sys.argv[1]).read_text()))
