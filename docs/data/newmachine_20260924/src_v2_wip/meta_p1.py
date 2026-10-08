"""P1 training metadata (host only, no GPU): per run directory under S1/V2 -- checkpoint sha256, config (model, model_args,
steps, lr, batch, split, init, seed, resume), parameter count (floating tensors of the saved model state), log-derived wall
time, device memory and the training / evaluation curves (subsampled); per split -- training / validation geometry counts
and the realised bank sizes (columns per class and split) of every geometry.
Usage: meta_p1.py <out.json> <run>[,<run>...] [--root /root/autodl-tmp/OPL/S1/V2] [--data /root/autodl-tmp/OPL/S2/data]"""
import sys, json, hashlib, argparse
from pathlib import Path
import numpy as np


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


def curves(log):
    steps, evals, t_first, t_last, gpu = [], [], None, None, 0.0
    for line in open(log, errors='replace'):
        if not line.startswith('{'):
            continue
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if d.get('event') == 'STEP':
            steps.append((d['step'], d.get('loss'), d.get('e_mean'), d.get('s')))
            gpu = max(gpu, float(d.get('gpu_GB') or 0))
        elif d.get('event') == 'EVAL':
            evals.append(dict(step=d['step'], score=d.get('score'), val_mean=d.get('val_mean')))
    sub = steps[::max(1, len(steps) // 300)]
    wall = None
    s_vals = [s[3] for s in steps if s[3] is not None]
    if s_vals:
        wall = float(max(s_vals))
    return dict(n_step_records=len(steps), curve=[dict(step=a, loss=b, e_mean=c, s=e) for a, b, c, e in sub], evals=evals,
                max_step_seconds_field=wall, gpu_GB_max=gpu)


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('runs')
    ap.add_argument('--root', default='/root/autodl-tmp/OPL/S1/V2'); ap.add_argument('--data', default='/root/autodl-tmp/OPL/S2/data')
    a = ap.parse_args(argv)
    import torch
    root = Path(a.root)
    rec = dict(runs={}, splits={})
    for run in a.runs.split(','):
        ck = root / run / 'best.pt'
        r = dict(ckpt=str(ck))
        if ck.exists():
            c = torch.load(ck, map_location='cpu', weights_only=False)
            cfg = c.get('cfg', {})
            r['sha256'] = sha(ck)
            r['cfg'] = {k: cfg.get(k) for k in ('model', 'model_args', 'steps', 'lr', 'batch', 'B', 'split', 'init', 'seed', 'resume',
                                                'resume_rng', 'eval_every', 'select_min_step', 'eval_views', 'val_max', 'mix', 'sens_w',
                                                'augment', 'views') if k in cfg}
            r['cfg_keys'] = sorted(cfg.keys())
            sd = c['model']
            r['params'] = int(sum(v.numel() for v in sd.values() if torch.is_tensor(v) and v.is_floating_point()))
            r['step'] = c.get('step'); r['score'] = c.get('score'); r['conv'] = c.get('conv')
            sp = cfg.get('split')
            if sp and sp not in rec['splits'] and Path(sp).exists():
                rec['splits'][sp] = None
        log = root / f'{run}.log'
        if log.exists():
            r['log'] = curves(log)
        rec['runs'][run] = r
        print(json.dumps(dict(run=run, params=r.get('params'), step=r.get('step'), sha=r.get('sha256', '')[:12],
                              wall=r.get('log', {}).get('max_step_seconds_field'))), flush=True)
    for sp in list(rec['splits']):
        s = json.loads(Path(sp).read_text())
        info = {k: len(v) for k, v in s.items() if isinstance(v, list)}
        banks = {}
        for k in ('train', 'val', 'val_s3'):
            for case in s.get(k, []):
                d = Path(a.data) / case
                if not d.exists():
                    continue
                banks[case] = {f.stem: int(np.load(f, mmap_mode='r').shape[0]) for f in sorted(d.glob('*.npy'))
                               if f.stem.split('_')[0] in ('train', 'val', 'test') and not f.stem.endswith('_sens')}
        agg = {}
        for case, b in banks.items():
            for k_, v in b.items():
                agg.setdefault(k_, []).append(v)
        rec['splits'][sp] = dict(lists=info, bank_columns_summary={k_: dict(geos=len(v), min=min(v), median=float(np.median(v)), max=max(v))
                                                                   for k_, v in sorted(agg.items())}, per_case=banks)
    Path(a.out).write_text(json.dumps(rec, indent=1, default=str))


if __name__ == '__main__':
    main(sys.argv[1:])
