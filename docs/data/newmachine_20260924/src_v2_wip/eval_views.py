"""Bank-level errors of a step-2 checkpoint on the validation geometries under fixed cube-group views (oh.view): the
orientation sensitivity that the identity-only validation of train2 / train3 cannot see. Energies with the original K
(unit exact energy banks, so e_hat - 1 is the relative energy error); views returned in the original frame.
Usage: eval_views.py <ckpt> <out.json> [--views 0,5,17,29,38,46] [--val-max 20] [--split SPLIT.json] [--chunk 16]
       (default views: identity and five fixed non-identity elements of oh.ELEMS)"""
import sys, json, time, gc, argparse
from pathlib import Path
import numpy as np
import torch
import trainlib as TL
import models as MD
import train2 as T2
import oh

dev = TL.dev


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt'); ap.add_argument('out')
    ap.add_argument('--views', default='0,5,17,29,38,46'); ap.add_argument('--val-max', type=int, default=20)
    ap.add_argument('--split', default='/root/autodl-tmp/OPL/S2/SPLIT.json'); ap.add_argument('--chunk', type=int, default=16)
    ap.add_argument('--slots', default='/root/autodl-tmp/OPL/S2/slots')
    a = ap.parse_args(argv)
    ks = [int(k) for k in a.views.split(',')]
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    cfg = ck['cfg']
    cases = json.loads(Path(a.split).read_text())['val'][:a.val_max]
    model, rec, t0 = None, dict(ckpt=a.ckpt, views=ks, per_geo={}), time.perf_counter()
    for case in cases:
        g = torch.load(Path(a.slots) / f'{case}.pt', map_location='cpu', weights_only=False)
        T2.move(g, dev); g.C.K = g.C
        T2.clean_banks(g, case, lambda d_: None)
        if model is None:
            model = MD.build(cfg['model'], [g], **cfg.get('model_args', {})).to(dev)
            (MD.load_compat(model, ck['model']) if hasattr(MD, 'load_compat') else model.load_state_dict(ck['model'], strict=False))
            model.eval()
        else:
            model.add_geo(g)
        r = {}
        with torch.no_grad():
            for k in ks:
                v = g if k == 0 else oh.view(model, g, k)
                r[k] = {}
                for c in g.classes:
                    Q = g.banks['val'][c]
                    e = torch.cat([TL.energy(v.field(model, Q[:, j:j + a.chunk]), g.C.K) - 1 for j in range(0, Q.shape[1], a.chunk)])
                    r[k][c] = float(e.mean())
                if k != 0:
                    oh.drop(model, v)
        rec['per_geo'][case] = r
        print(json.dumps(dict(case=case, **{str(k): {c: round(x, 4) for c, x in r[k].items()} for k in ks})), flush=True)
        model.caches.pop(case, None); T2.move(g, 'cpu'); del g; gc.collect(); torch.cuda.empty_cache()
    cls = sorted({c for r in rec['per_geo'].values() for c in r[ks[0]]})
    rec['mean'] = {str(k): {c: float(np.mean([r[k][c] for r in rec['per_geo'].values() if c in r[k]])) for c in cls} for k in ks}
    rec['mean_over_rotated'] = {c: float(np.mean([rec['mean'][str(k)][c] for k in ks if k != 0])) for c in cls}
    rec['seconds'] = time.perf_counter() - t0
    Path(a.out).write_text(json.dumps(rec))
    print(json.dumps(dict(event='DONE', identity=rec['mean'][str(ks[0])], rotated=rec['mean_over_rotated'], seconds=rec['seconds'])), flush=True)


if __name__ == '__main__':
    main(sys.argv[1:])
