"""B1 calibration: soft-clip bounds for the geometry-produced q-path coefficients of a trained MGNO / MGNO2.
Runs the geometry path only (no q) over training geometries and records p50 / p99 / p99.9 / max of |coef| per group:
  ab, fab, xab   group = [alpha / beta, layer]  (the value after the 0.2 scale, before any clip)
  rw             group = [level, restrict / prolong]  (softplus(r), i.e. the transfer weight before the + 1e-3)
and writes A = mult x p99.9 per group (mult_rw for rw; models.MGNO.set_bounds format, model_args bounded=True) to
<out.pt>, the stats as json next to it (<out>.json). Quantiles from a log10 histogram with 0.001-decade bins (upper bin
edge: conservative, <= 0.23% high), max exact. No q, no K: the checkpoint's model on the device of OPL_DEV (cuda:0).
Usage: calib_b1.py <ckpt> <split.json> <out.pt> [--max-geos N] [--mult 2.5] [--mult-rw M] [--slots DIR] [--data DIR]
                  [--key train]
  geometries: split[key] (an evenly spaced subset of N with --max-geos); network inputs from the slot cache (--slots, default
  the checkpoint's cfg slot_cache, else /root/autodl-tmp/OPL/S2/slots: pickled trainlib.Geo), or with --data DIR from
  DIR/<case>/NETDATA.npz (identical inputs: Geo.nd is that file; far less I/O). A missing slot falls back to cfg data.
"""
import argparse, json, sys, time, gc
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import models as MD

LO, HI, BW = -12.0, 6.0, 1e-3                                                        # log10 |coef| histogram
NB = int(round((HI - LO) / BW))
QS = (('p50', 0.5), ('p99', 0.99), ('p999', 0.999))


def _groups(rec):
    """Raw capture of one geometry() call -> {name: [(group index tuple, values 1-D)]}."""
    out = {}
    for k in ('ab', 'fab', 'xab'):
        if k in rec:
            x = rec[k]                                                                   # Eh x 27 x 2 x L x H
            out[k] = [((j, l), x[:, :, j, l, :].reshape(-1)) for j in range(2) for l in range(x.shape[3])]
    if 'rw' in rec:
        out['rw'] = [((l, j), rec['rw'][l][j].reshape(-1)) for l in range(len(rec['rw'])) for j in range(2)]
    return out


class _Hist:
    def __init__(self, shape):
        self.shape = shape
        G = int(np.prod(shape))
        self.cnt = torch.zeros((G, NB), dtype=torch.float64)
        self.mx = torch.zeros(G, dtype=torch.float64)

    def add(self, idx, v):
        g = int(np.ravel_multi_index(idx, self.shape))
        a = v.abs().double()
        b = ((torch.log10(a.clamp_min(1e-300)) - LO) / BW).floor().clamp(0, NB - 1).long()
        self.cnt[g] += torch.bincount(b, minlength=NB).double().cpu()
        self.mx[g] = max(float(self.mx[g]), float(a.max()) if a.numel() else 0.0)

    def quant(self):
        cum = self.cnt.cumsum(1)
        tot = cum[:, -1:]
        out = {}
        for name, p in QS:
            i = (cum < p * tot).sum(1).clamp_max(NB - 1)                                 # first bin reaching the quantile
            out[name] = torch.minimum(10 ** (LO + (i.double() + 1) * BW), self.mx).reshape(self.shape)
        out['max'] = self.mx.reshape(self.shape)
        out['count'] = tot.reshape(self.shape)
        return out


@torch.no_grad()
def calibrate(model, geos, mult=2.5, log=print, mult_rw=None):
    """geos: iterable of objects with .case and .nd (trainlib.Geo or a light stand-in); each one is added to the model
    (unless already there), its geometry path run once, and dropped again if it was added here.
    Returns (bounds {name: tensor}, stats {name: {p50, p99, p999, max, count, A}}, per_geo {case: {name: max |coef|}})."""
    if not model.bounded:
        raise ValueError('calibrate needs a model built with bounded=True (the raw values are captured in its clip)')
    hist, per_geo = {}, {}
    try:
        for g in geos:
            t = time.perf_counter()
            had = g.case in model.caches
            model.add_geo(g)
            model._rec = {}
            model.geometry(g.case)
            rec, model._rec = model._rec, None
            pg = {}
            for name, grp in _groups(rec).items():
                if name not in hist:
                    hist[name] = _Hist(tuple(getattr(model, 'bnd_' + name).shape))
                for idx, v in grp:
                    hist[name].add(idx, v)
                pg[name] = max(float(v.abs().max()) for _, v in grp)
            per_geo[g.case] = pg
            if not had:
                model.drop_geo(g.case)
            del rec
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            log(json.dumps(dict(event='CALIB_GEO', case=g.case, seconds=time.perf_counter() - t, max=pg)))
    finally:
        model._rec = None
    bounds, stats = {}, {}
    for name, h in hist.items():
        q = h.quant()
        mu = mult if name != 'rw' or mult_rw is None else mult_rw
        bounds[name] = torch.where(q['count'] > 0, mu * q['p999'], torch.full_like(q['p999'], float('inf'))).float()   # empty group: identity
        stats[name] = {k: v.tolist() for k, v in q.items()}
        stats[name]['A'] = bounds[name].tolist()
    return bounds, stats, per_geo


def _geo_iter(cases, slots, data, fallback, log):
    for case in cases:
        f = Path(slots) / f'{case}.pt' if slots else None
        if data is None and f is not None and f.exists():
            g = torch.load(f, map_location='cpu', weights_only=False)
            yield SimpleNamespace(case=case, nd=g.nd)
            del g
        else:
            d = Path(data or fallback) / case / 'NETDATA.npz'
            if data is None:
                log(json.dumps(dict(event='CALIB_NO_SLOT', case=case, netdata=str(d))))
            yield SimpleNamespace(case=case, nd=dict(np.load(d)))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt'); ap.add_argument('split'); ap.add_argument('out')
    ap.add_argument('--max-geos', type=int, default=0); ap.add_argument('--mult', type=float, default=2.5)
    ap.add_argument('--mult-rw', type=float, default=None)                             # transfer weights (default: --mult)
    ap.add_argument('--slots', default=None); ap.add_argument('--data', default=None); ap.add_argument('--key', default='train')
    a = ap.parse_args(argv)
    ck = torch.load(a.ckpt, map_location='cpu', weights_only=False)
    cfg = ck['cfg']
    args = dict(cfg.get('model_args', {}))
    args.pop('bounds', None); args['bounded'] = True
    model = MD.build(cfg['model'], [], **args).to(MD.dev)
    info = MD.load_compat(model, ck['model'])
    model.eval()
    cases = json.loads(Path(a.split).read_text())[a.key]
    if a.max_geos and a.max_geos < len(cases):
        cases = [cases[i] for i in np.unique(np.linspace(0, len(cases) - 1, a.max_geos).round().astype(int))]
    slots = a.slots or cfg.get('slot_cache') or '/root/autodl-tmp/OPL/S2/slots'
    log = lambda s_: print(s_, flush=True)
    log(json.dumps(dict(event='CALIB', ckpt=a.ckpt, model=cfg['model'], geos=len(cases), mult=a.mult, compat=info,
                        source=('netdata:' + a.data) if a.data else ('slots:' + slots))))
    t0 = time.perf_counter()
    bounds, stats, per_geo = calibrate(model, _geo_iter(cases, slots, a.data, cfg.get('data', '/root/autodl-tmp/OPL/S2/data'), log),
                                       mult=a.mult, log=log, mult_rw=a.mult_rw)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(bounds=bounds, mult=a.mult, mult_rw=a.mult_rw, ckpt=a.ckpt, cases=cases, stats=stats), out)
    rec = dict(ckpt=a.ckpt, split=a.split, key=a.key, mult=a.mult, mult_rw=a.mult_rw, cases=cases, seconds=time.perf_counter() - t0,
               groups={'ab/fab/xab': '[alpha/beta, layer]', 'rw': '[level, restrict/prolong], softplus(r)'},
               stats=stats, per_geo_max=per_geo)
    out.with_suffix('.json').write_text(json.dumps(rec, indent=1))
    log(json.dumps(dict(event='CALIB_DONE', out=str(out), seconds=rec['seconds'],
                        A_range={k: [float(v.min()), float(v.max())] for k, v in bounds.items()})))
    return bounds, stats


if __name__ == '__main__':
    main()
