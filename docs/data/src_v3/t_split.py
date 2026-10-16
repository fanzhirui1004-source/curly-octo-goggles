"""coarse_split on real geometries: split statistics per level, and the same checkpoint with grid vs split coarse levels
(field difference and energy excess on validation directions). Usage: t_split.py <ckpt> <out.json> <case>[,<case>...]
[--classes force_c,force] [--m 16] [--body S0] [--data S2/data_v2]   (OPL_DEV=cpu runs it without a GPU)"""
import sys, json, time, argparse
import os
if os.environ.get('OPL_DEV') == 'cpu':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''                               # CPU run must never touch the training GPU
from pathlib import Path
import torch
import models as MD
import trainlib as TL
if os.environ.get('OPL_DEV') == 'cpu':                                       # polyref defaults to device='cuda'
    import inspect, polyref_torch_fast as _PT
    TL.TE.sync = lambda: None
    for _f in vars(_PT).values():
        if inspect.isfunction(_f) and _f.__defaults__ and 'cuda' in _f.__defaults__:
            _f.__defaults__ = tuple('cpu' if d == 'cuda' else d for d in _f.__defaults__)


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt'); ap.add_argument('out'); ap.add_argument('cases')
    ap.add_argument('--classes', default='force_c,force'); ap.add_argument('--m', type=int, default=16)
    ap.add_argument('--body', default='/root/autodl-tmp/OPL/S0'); ap.add_argument('--data', default='/root/autodl-tmp/OPL/S2/data_v2')
    a = ap.parse_args(argv)
    dev = TL.dev
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    cfg = ck['cfg']
    rec = dict(ckpt=a.ckpt, per_case={})
    for case in a.cases.split(','):
        t0 = time.perf_counter()
        geo = TL.Geo(case, a.body, a.data, neumann=False, log=lambda s_: None)
        models = {}
        for split in (False, True):
            m = MD.build(cfg['model'], [geo], **dict(cfg.get('model_args', {}), sparse=False, coarse_split=split)).to(dev)
            MD.load_compat(m, ck['model']); m.eval(); models[split] = m
        cg, cs = models[False].caches[case], models[True].caches[case]
        lv = []
        for l in range(len(cg.trans)):
            tg, ts = cg.trans[l], cs.trans[l]
            lv.append(dict(grid_nodes=int(tg['n_dst']), split_nodes=int(ts['n_dst']), split_extra=int(ts['split_extra']),
                           edges=int(len(ts['e_src']))))
        r = dict(levels=lv, setup_s=time.perf_counter() - t0)
        for cls in [c for c in a.classes.split(',') if c in geo.classes]:
            Q = geo.banks['val'][cls][:, :a.m]
            with torch.no_grad():
                ug, us = geo.field(models[False], Q).double(), geo.field(models[True], Q).double()
            e_g = (TL.energy(ug, geo.C.K) - 1); e_s = (TL.energy(us, geo.C.K) - 1)
            r[cls] = dict(field_rel_diff=float((us - ug).norm() / ug.norm()), excess_grid=float(e_g.mean()), excess_split=float(e_s.mean()))
        rec['per_case'][case] = r
        print(json.dumps(dict(case=case, **{k: v for k, v in r.items()})), flush=True)
        Path(a.out).write_text(json.dumps(rec, indent=1))


if __name__ == '__main__':
    main(sys.argv[1:])
