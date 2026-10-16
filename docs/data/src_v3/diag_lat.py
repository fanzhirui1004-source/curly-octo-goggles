"""Where does a lattice sensitivity error of the test cell come from? For every load of the lattice gate:
  q_ref  = test-cell port displacement of the exact lattice solution, q_net = the same with the learned operator;
  total     |s(E_hat q_net) - s(E q_ref)|  (what the gate measures)
  extension |s(E_hat q_ref) - s(E q_ref)|  (learned field at the exact port data: extension error only)
  interface |s(E q_net)     - s(E q_ref)|  (exact field at the lattice solution of the learned operator: reaction error only)
all divided by |s(E q_ref)|; plus the relative port-displacement error |q_net - q_ref| / |q_ref| and the energy shares.
Usage: diag_lat.py <checkpoint.pt> <out.json> <case> <config> [<config> ...]   (traction-consistent loads)"""
import json, sys, os
from pathlib import Path
import numpy as np
import torch
import lattice3 as LT
import trainlib as TL
import models as MD
import ops as OP
import fastnet as FN

FULL = 'fresh_train_0020_full'


def main(ckpt, out, case, configs):
    os.environ['LAT_LOADS'] = 'consistent'
    ck = torch.load(ckpt, map_location='cuda:0', weights_only=False)
    cfg = ck['cfg']
    C, _ = LT.prepared(case, cfg['body'])
    geo = TL.Geo(case, cfg['body'], cfg['data'], neumann=False, log=lambda s_: None, cell=C, load_banks=False)
    model = MD.build(cfg['model'], [geo], **cfg.get('model_args', {})).cuda()
    model.load_state_dict(ck['model'], strict=False); model.eval()
    fast = FN.FastNet(model, geo)

    class Op:
        def field(self, q):
            return fast.field(q).to(TL.dt)

        def apply(self, q):
            return fast.s_hat(q)

    rec = dict(ckpt=str(ckpt), case=case, results=[])
    for conf in configs:
        lat = LT.build(case, FULL, conf, cfg['body'])
        ref = lat.reference()
        res = lat.evaluate([Op(), OP.ExactOp(lat.cells[1]['cell'], lat.cells[1]['T'])], maxit=400)
        q_ref, q_net = lat.gather(ref['U'], 0), lat.gather(res['U'], 0)
        C.factor(neumann=False, fp32=True)
        s_ref = C.sens(C.extend(q_ref))
        s_ext = C.sens(Op().field(q_ref))
        s_int = C.sens(C.extend(q_net))
        s_tot = C.sens(Op().field(q_net))
        C._free()
        nrm = s_ref.norm(dim=0)
        r = dict(config=conf, loads=lat.labels, gate=lat.gate.tolist(),
                 total=((s_tot - s_ref).norm(dim=0) / nrm).tolist(),
                 extension=((s_ext - s_ref).norm(dim=0) / nrm).tolist(),
                 interface=((s_int - s_ref).norm(dim=0) / nrm).tolist(),
                 q_rel=((q_net - q_ref).norm(dim=0) / q_ref.norm(dim=0)).tolist(),
                 energy_share_test=(ref['energy'][0] / ref['compliance']).tolist(),
                 compliance_rel=np.abs(res['compliance'] / ref['compliance'] - 1).tolist())
        rec['results'].append(r)
        for j, lab in enumerate(r['loads']):
            print(json.dumps(dict(config=conf, load=lab, total=round(r['total'][j], 5), extension=round(r['extension'][j], 5),
                                  interface=round(r['interface'][j], 5), q_rel=round(r['q_rel'][j], 5),
                                  share=round(r['energy_share_test'][j], 4))), flush=True)
        del lat
    Path(out).write_text(json.dumps(rec, indent=1))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:])
