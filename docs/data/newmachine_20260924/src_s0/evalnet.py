"""Lattice gate for a trained model: the test cell uses the learned extension (variational readout), the neighbour is
exact. Configurations x and y; gate loads and cut-surface loads (reported).
Usage: evalnet.py <checkpoint.pt> <out.json> [<case> ...]   (default: the checkpoint's cases)
"""
import json, sys, time, gc
from pathlib import Path
import numpy as np
import torch
import lattice3 as LT
import trainlib as TL
import models as MD
import ops as OP

FULL = 'fresh_train_0020_full'


class NetOp:
    """Operator interface for a learned extension on the lattice cell (field and variational reaction)."""

    def __init__(self, geo, model):
        self.geo, self.model = geo, model

    def field(self, q):
        with torch.no_grad():
            return self.geo.field(self.model, q).to(TL.dt)

    def apply(self, q):
        return self.geo.s_hat_apply(self.model, q)


def main(ckpt, out, cases):
    ck = torch.load(ckpt, map_location='cuda:0', weights_only=False)
    cfg = ck['cfg']
    cases = cases or cfg['cases']
    rec = dict(ckpt=str(ckpt), step=ck['step'], results=[])
    for case in cases:
        C, _ = LT.prepared(case, cfg['body'])                          # one teacher cell shared with the lattice
        geo = TL.Geo(case, cfg['body'], cfg['data'], neumann=False, log=lambda s_: None, cell=C, load_banks=False)
        model = MD.build(cfg['model'], [geo], **cfg.get('model_args', {})).cuda()
        missing = model.load_state_dict(ck['model'], strict=False)
        model.eval()
        for conf in ('x', 'y'):
            t0 = time.perf_counter()
            lat = LT.build(case, FULL, conf, cfg['body'])
            lat.reference()
            exact_nbr = OP.ExactOp(lat.cells[1]['cell'], lat.cells[1]['T'])
            op = NetOp(geo, model)
            # the lattice cell object must be the same geometry: reuse the lattice's teacher cell for sensitivities
            res = lat.evaluate([op, exact_nbr], maxit=400)
            cmp_ = lat.compare(res)
            cmp_.update(case=case, config=conf, seconds=time.perf_counter() - t0)
            rec['results'].append(cmp_)
            print(json.dumps(dict(case=case, config=conf, gate_compliance_max=cmp_['gate_compliance_max'],
                                  gate_sens_max=cmp_['gate_sens_max'], gate_pass=cmp_['gate_pass'],
                                  cut_compliance_max=cmp_.get('cut_compliance_max'), cut_sens_max=cmp_.get('cut_sens_max'),
                                  pcg=cmp_['pcg_iterations'])), flush=True)
            del lat; gc.collect(); torch.cuda.empty_cache()
        del geo, model; gc.collect(); torch.cuda.empty_cache()
    Path(out).write_text(json.dumps(rec, indent=1))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
