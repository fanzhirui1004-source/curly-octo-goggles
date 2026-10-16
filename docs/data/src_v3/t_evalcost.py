"""GPU check of the train3 evaluation-cost switches (eval_weights 'sel', cert_every) against the default full evaluation.
Two short runs on the 4 smallest training geometries (pool 2, swap every 10, 60 steps, EMA, certificate, mu on 1 val
geometry, 2 probes, evals at 20 / 40 / 60), identical except for the switches. Pass criteria:
  A (defaults): every EVAL scores raw and ema, every val class carries the certificate
  B (sel, cert_every 2): EVAL 20 = ema only without certificate, EVAL 40 = ema only with it, EVAL 60 (last) = raw and ema with it
  trajectory: STEP losses and ema scores of A and B agree to 1e-4 relative (the switches touch evaluation only; CUDA atomics
  are not bitwise reproducible); prints the eval seconds of both runs.
  --fused: run C = A with the fused hyperedge kernel (sparse_layers.FUSED): losses and scores within 1e-3 of A, step seconds.
Usage: t_evalcost.py [--out D] [--ckpt C] [--fused]"""
import sys, json, shutil, argparse
from pathlib import Path
import numpy as np
import torch
import train3 as T3

ap = argparse.ArgumentParser()
ap.add_argument('--split', default='/root/autodl-tmp/OPL/S2/SPLIT.json')
ap.add_argument('--slots', default='/root/autodl-tmp/OPL/S2/slots')
ap.add_argument('--data', default='/root/autodl-tmp/OPL/S2/data')
ap.add_argument('--body', default='/root/autodl-tmp/OPL/S0')
ap.add_argument('--ckpt', default='/root/autodl-tmp/OPL/S1/s2_full/snap_60000.pt')
ap.add_argument('--out', default='/root/autodl-tmp/OPL/S1/V2/t_evalcost')
ap.add_argument('--fused', action='store_true')
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
split = json.loads(Path(a.split).read_text())
dofs = lambda c: json.loads((Path(a.data) / c / 'DONE.json').read_text())['dofs']
have = lambda c: (Path(a.slots) / f'{c}.pt').exists()
tr = sorted([c for c in split['train'] if have(c)], key=dofs)[:4]
va = sorted([c for c in split['val'] if have(c)], key=dofs)[:2]
(out / 'SPLIT.json').write_text(json.dumps(dict(train=tr, val=va, test=[], train_families=[], val_families=[])))
ck = torch.load(a.ckpt, map_location='cpu', weights_only=False)
base = dict(split=str(out / 'SPLIT.json'), body=a.body, data=a.data, slot_cache=a.slots, model=ck['cfg']['model'],
            model_args=dict(ck['cfg']['model_args']), init=a.ckpt, steps=60, batch=16, lr=3e-4, pool=2, pool_dofs=700000,
            swap_every=10, adv_on_load=True, adv_k=8, adv_iters=4, eval_every=20, val_max=2, probe_max=2, mu_geos=1, ema=0.999,
            quota=True, cert=True, seed=0, sens_w=1.0, log_every=10,
            mix={'force': .25, 'support': .15, 'face': .2, 'macro': .1, 'grf': .15, 'adv': .15})
del ck
FAIL = []


def check(name, ok, **info):
    print(json.dumps(dict(test=name, ok=bool(ok), **info), default=str), flush=True)
    if not ok:
        FAIL.append(name)


def run(name, **kw):
    cfg = dict(base, out=str(out / name), **kw)
    shutil.rmtree(cfg['out'], ignore_errors=True)
    T3.main(cfg)
    return [json.loads(l) for l in open(Path(cfg['out']) / 'train.log')]


def evals(L):
    return {r['step']: r for r in L if r.get('event') == 'EVAL'}


def has_cert(r, w):
    return any('cert_mean' in v for g in r['weights'][w]['val'].values() for v in g.values())


LA = run('A')
LB = run('B', eval_weights='sel', cert_every=2)
EA, EB = evals(LA), evals(LB)
check('A_full_every_eval', sorted(EA) == [20, 40, 60] and all(set(r['weights']) == {'raw', 'ema'} and has_cert(r, 'raw') and has_cert(r, 'ema')
                                                                for r in EA.values()))
okB = sorted(EB) == [20, 40, 60] and set(EB[20]['weights']) == {'ema'} and not has_cert(EB[20], 'ema') and \
    set(EB[40]['weights']) == {'ema'} and has_cert(EB[40], 'ema') and set(EB[60]['weights']) == {'raw', 'ema'} and \
    has_cert(EB[60], 'raw') and has_cert(EB[60], 'ema')
check('B_switches', okB, weights={s: sorted(r['weights']) for s, r in EB.items()}, cert={s: has_cert(r, 'ema') for s, r in EB.items()})


def same(name, L1, L2, E1, E2, tol):
    s1 = {r['step']: r['loss'] for r in L1 if r.get('event') == 'STEP'}
    s2 = {r['step']: r['loss'] for r in L2 if r.get('event') == 'STEP'}
    rl = max(abs(s1[s] - s2[s]) / abs(s1[s]) for s in s1 if s in s2)
    rs = max(abs(E1[s]['score'] - E2[s]['score']) / abs(E1[s]['score']) for s in E1 if s in E2)
    best = lambda E: min(E, key=lambda s: E[s]['score'])
    check(name, rl <= tol and rs <= tol, step_loss_rel_max=rl, score_rel_max=rs, best=(best(E1), best(E2)),
          score={s: (round(E1[s]['score'], 6), round(E2[s]['score'], 6)) for s in E1 if s in E2})


same('same_trajectory_AB', LA, LB, EA, EB, 1e-4)
per = lambda E: [round(E[s]['eval_s'] - E.get(s - 20, {}).get('eval_s', 0.0), 1) for s in sorted(E)]
done = lambda L: [r for r in L if r.get('event') == 'DONE'][0]['time']
rec = dict(eval_s_per_eval_A=per(EA), eval_s_per_eval_B=per(EB), step_s_A=done(LA)['step'] / 60, step_s_B=done(LB)['step'] / 60)
if a.fused:
    import sparse_layers as SL
    SL.FUSED = True
    LC = run('C')
    SL.FUSED = False
    EC = evals(LC)
    same('fused_trajectory_AC', LA, LC, EA, EC, 1e-3)
    rec.update(step_s_C_fused=done(LC)['step'] / 60, eval_s_per_eval_C=per(EC))
check('timing', True, **rec)
print(json.dumps(dict(event='DONE', failed=FAIL)), flush=True)
sys.exit(1 if FAIL else 0)
