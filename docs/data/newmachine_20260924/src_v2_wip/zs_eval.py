"""Bank-level evaluation of a trained model on (possibly unseen) geometries through fastnet (low memory):
per class, the energy error e_hat - 1 (unit exact energy) and the relative sensitivity error of the 8-corner vector.
Usage: zs_eval.py <out.json> <checkpoint.pt> <body> <data> <case> [<case> ...]"""
import sys, json, gc
import numpy as np
import torch
import trainlib as TL
import models as MD
import fastnet as FN

dt = TL.dt
out, ck_path, body, data = sys.argv[1:5]
ck = torch.load(ck_path, map_location='cuda:0', weights_only=False)
cfg = ck['cfg']
rec = dict(ckpt=ck_path, trained_on=cfg['cases'], results={})
for case in sys.argv[5:]:
    geo = TL.Geo(case, body, data, neumann=False, log=lambda s_: None)
    model = MD.build(cfg['model'], [geo], **cfg.get('model_args', {})).cuda()
    model.load_state_dict(ck['model'], strict=False); model.eval()
    fast = FN.FastNet(model, geo)
    C = geo.C
    r = {}
    for c in geo.classes:
        Q = geo.banks['val'][c]
        es, ss = [], []
        for j in range(0, Q.shape[1], 16):
            u = fast.field(Q[:, j:j + 16].to(dt)).to(dt)
            es.append(((u * (C @ u)).sum(0) - 1).cpu())
            if geo.sens is not None and c in geo.sens['val']:
                s0 = geo.sens['val'][c][:, j:j + 16]
                ss.append(((C.sens2(u) - s0).norm(dim=0) / s0.norm(dim=0)).cpu())
        e = torch.cat(es).numpy()
        r[c] = dict(energy_mean=float(e.mean()), energy_p90=float(np.quantile(e, .9)), energy_max=float(e.max()))
        if ss:
            s = torch.cat(ss).numpy()
            r[c].update(sens_mean=float(s.mean()), sens_p90=float(np.quantile(s, .9)), sens_max=float(s.max()))
    rec['results'][case] = r
    print(json.dumps({case: r}), flush=True)
    del fast, model, geo, C; gc.collect(); torch.cuda.empty_cache()
open(out, 'w').write(json.dumps(rec, indent=1))
