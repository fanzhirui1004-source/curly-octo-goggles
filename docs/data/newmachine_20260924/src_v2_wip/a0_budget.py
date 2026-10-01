"""Direction A, step 0b: correction budget on mapped cells (new script, default paths unchanged; runs on the GPU server).

How much of the out-of-distribution error (stretch, shear, twist) the deployed correction removes at fixed network weights,
and how it splits between the network and the correction. Same banks as an a0_eval run directory (<data>/<map>/<case>).
Fields (all V0R, co-rotated Jacobi preconditioner unless --corot 0):
  net        network extension only (no correction)
  c1, c2, c4 the A3 cycle tail(8) - Q1_17 coarse - tail(8) applied 1, 2, 4 times (c1 = deployed NICE)
  z1, z4     zero interior + rigid part (Conly) with 1 and 4 cycles
Every field is a linear extension read out in energy form, so S_hat >= S~ (min excess reported); repeating a cycle whose
smoothing interval contains the spectrum cannot increase the energy error.
Usage: a0_budget.py <out_dir> <ckpt> <body_dir> <data_dir> <cases (comma)> <maps.json> --maps id,strx2 [--corot 1]
       [--fields net,c1,c2,c4,z1,z4] [--classes force_c,face_c,grf]"""
import argparse, gc, json, sys, time
from pathlib import Path
import numpy as np
import torch

import models as MD                                                       # first: applies OPL_CONV_FP32
import trainlib as TL
import a0_eval as AE
import mapped_cell as MC
import corot_smooth as CR

dev, dt = AE.dev, AE.dt


class _NoCorr:
    smooth_k, coarse_space, smooth_alpha = 0, None, 30.0


class BudgetGeo(AE.MappedGeo):
    """MappedGeo whose field applies the correction cycle `cycles` times to the uncorrected extension (fp64 between cycles)."""

    def set_budget(self, base, cycles, wrap):
        self.set_variant(base, _NoCorr())
        self.cycles, self.cyc_wrap = cycles, wrap

    def field(self, model, q):
        u = super().field(model, q).to(dt)                                  # rigid split, (co-rotated) network, no correction
        for _ in range(self.cycles):
            u = TL.wrap(self.C, u, self.cyc_wrap)
        return u


FIELDS = {'net': ('V0R', 0), 'c1': ('V0R', 1), 'c2': ('V0R', 2), 'c4': ('V0R', 4), 'c8': ('V0R', 8),
          'z1': ('Conly', 1), 'z4': ('Conly', 4)}


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('ckpt'); ap.add_argument('body'); ap.add_argument('data'); ap.add_argument('cases')
    ap.add_argument('mapsjson'); ap.add_argument('--maps', default='id')
    ap.add_argument('--corot', type=int, default=1); ap.add_argument('--fields', default='net,c1,c2,c4,z1,z4')
    ap.add_argument('--classes', default='force_c,face_c,grf')
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    specs = {m['name']: m['spec'] for m in json.loads(Path(a.mapsjson).read_text())}
    res_path = out / 'BUDGET.jsonl'
    done = set()
    if res_path.exists():
        done = {(r['case'], r['map']) for r in map(json.loads, res_path.read_text().splitlines()) if 'error' not in r}
    classes = a.classes.split(',')
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False); cfg = ck['cfg']
    model = None
    if a.corot:
        CR.install()
    for mname in a.maps.split(','):
        for case in [c for c in a.cases.split(',') if c]:
            if (case, mname) in done:
                continue
            rec = dict(case=case, map=mname, corot=a.corot, fields={})
            t0 = time.perf_counter(); C = g = None
            try:
                C = MC.MappedCell(case, a.body, specs[mname], log=lambda s_: None); C.assemble()
                rec['map_stats'] = C.map_stats
                g = BudgetGeo(case, a.body, Path(a.data) / mname, neumann=False, log=lambda s_: None, cell=C)
                g.case = f'{case}@{mname}'
                if model is None:
                    model = MD.build(cfg['model'], [g], **dict(cfg.get('model_args', {}))).to(dev)
                    MD.load_compat(model, ck['model']); model.eval()
                else:
                    model.add_geo(g)
                wrap = AE._Wrap(model)
                if a.corot:
                    CR.set_corot(C, AE.nodal_rotations(C))
                for fname in a.fields.split(','):
                    base, cyc = FIELDS[fname]
                    t = time.perf_counter()
                    g.set_budget(base, cyc, wrap)
                    r = {}
                    with torch.no_grad():
                        for cls in classes:
                            Q = g.banks['val'][cls]
                            e = torch.cat([TL.energy(g.field(model, Q[:, j:j + 16]), C.K) - 1
                                           for j in range(0, Q.shape[1], 16)]).cpu().numpy()
                            r[cls] = dict(mean=float(e.mean()), p90=float(np.quantile(e, .9)), max=float(e.max()), min=float(e.min()))
                    r['seconds'] = time.perf_counter() - t
                    rec['fields'][fname] = r
                rec['tail_bounds'] = list(C._tail_bounds or ())
                model.caches.pop(g.case, None)
            except Exception as e:
                import traceback
                rec['error'] = repr(e)[:400]; rec['trace'] = traceback.format_exc()[-2000:]
                try:
                    model.caches.pop(g.case, None)
                except Exception:
                    pass
            rec['seconds'] = time.perf_counter() - t0
            with open(res_path, 'a') as f:
                f.write(json.dumps(rec) + '\n')
            print(json.dumps(dict(case=case, map=mname, s=round(rec['seconds'], 1), err=rec.get('error'),
                                  fc={k: round(100 * v['force_c']['mean'], 4) for k, v in rec['fields'].items() if 'force_c' in v})),
                  flush=True)
            try:
                C._free()
            except Exception:
                pass
            del C, g
            gc.collect(); torch.cuda.empty_cache()
    print('DONE', flush=True)


if __name__ == '__main__':
    main(sys.argv[1:])
