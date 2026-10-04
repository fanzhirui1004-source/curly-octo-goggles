"""Test of mapped_fast.MappedFastOp against the autograd adjoint (Geo.s_hat_apply of a0_budget.BudgetGeo): relative
difference of S_hat Q on 16 random port columns, symmetry x^T S_hat y - y^T S_hat x, timings (B = 1, 16, 64) and peak
allocated memory of one 16-column application, per (cell, map, field).
Usage: t_mapped_fast.py <out.jsonl> <ckpt> <case@body>[,...] <maps.json[,...]> --maps strx2,twist30 --fields c1,c2w"""
import argparse, gc, json, sys, time
from pathlib import Path
import torch

import models as MD                                                       # first: applies OPL_CONV_FP32
import a0_eval as AE
import a0_budget as AB
import mapped_cell as MC
import corot_smooth as CR
import weak_patch as WP
import mapped_fast as MF
from cost_mapped import timed, peak

dev, dt = AE.dev, AE.dt


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('ckpt'); ap.add_argument('cases'); ap.add_argument('mapsjson')
    ap.add_argument('--maps', default='strx2,twist30'); ap.add_argument('--fields', default='c1,c2w')
    ap.add_argument('--work', default='/root/autodl-tmp/OPL/A0/work_mfast')
    a = ap.parse_args(argv)
    specs = {}
    for f in a.mapsjson.split(','):
        specs.update({m['name']: m['spec'] for m in json.loads(Path(f).read_text())})
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False); cfg = ck['cfg']
    model = None
    CR.install()
    for item in a.cases.split(','):
        case, body = item.split('@', 1)
        for mname in a.maps.split(','):
            C = MC.MappedCell(case, body, specs[mname], log=lambda s_: None); C.assemble()
            d = Path(a.work) / mname
            AE.write_data(C, d / case, {}, (), body)
            g = AB.BudgetGeo(case, body, d, neumann=False, log=lambda s_: None, cell=C, load_banks=False)
            g.case = f'{case}@{mname}'
            if model is None:
                model = MD.build(cfg['model'], [g], **dict(cfg.get('model_args', {}))).to(dev)
                MD.load_compat(model, ck['model']); model.eval()
            else:
                model.add_geo(g)
            wrap = AE._Wrap(model)
            CR.set_corot(C, AE.nodal_rotations(C))
            gen = torch.Generator(device=dev).manual_seed(0)
            Q = torch.randn((C.np_, 64), dtype=dt, device=dev, generator=gen)
            Q = Q - C.Q @ (C.Q.T @ Q); Q = Q / Q.norm(dim=0)
            for fname in a.fields.split(','):
                base, cyc = AB.FIELDS[fname]
                wp = fname.endswith('w')
                if wp and getattr(C, '_wp', 'unset') == 'unset':
                    WP.setup(C, g.nd, dil=2, seed='cutweakbox')
                g.set_budget(base, cyc, wrap, wp=wp)
                rec = dict(case=case, map=mname, field=fname)
                S_ag = g.s_hat_apply(model, Q[:, :16])
                op = MF.MappedFastOp(g, model, cyc, wrap, patch=wp)
                S_mf = op.s_hat(Q[:, :16])
                rec['rel_diff'] = float((S_mf - S_ag).norm() / S_ag.norm())
                x, y = Q[:, 16:17], Q[:, 17:18]
                rec['asym'] = float(abs((x * op.s_hat(y)).sum() - (y * op.s_hat(x)).sum()) / abs((x * op.s_hat(x)).sum()))
                for B in (1, 16, 64):
                    _, rec[f'mf_B{B}_s'] = timed(lambda: op.s_hat(Q[:, :B]), reps=3, warm=1)
                for B in (1, 16):
                    _, rec[f'ag_B{B}_s'] = timed(lambda: g.s_hat_apply(model, Q[:, :B]), reps=3, warm=1)
                _, rec['mf_B16_peak_GB'] = peak(lambda: op.s_hat(Q[:, :16]))
                _, rec['ag_B16_peak_GB'] = peak(lambda: g.s_hat_apply(model, Q[:, :16]))
                with open(a.out, 'a') as f:
                    f.write(json.dumps(rec) + '\n')
                print(json.dumps(rec), flush=True)
                del op
            model.caches.pop(g.case, None)
            WP.free(C); C._free(); del C, g; gc.collect(); torch.cuda.empty_cache()
    print('DONE', flush=True)


if __name__ == '__main__':
    main(sys.argv[1:])
