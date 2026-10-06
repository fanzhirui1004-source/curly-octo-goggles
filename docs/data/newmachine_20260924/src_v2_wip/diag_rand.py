"""Direction A: where the random-load error of the corrected learned operator lives, and which patch / cycle variant removes
it (new script; runs on the GPU server). One mapped cell, explicit-adjoint operator (mapped_fast, float64 rigid split).
Random port loads (--qmode): 'gauss' = 64 Gaussian port vectors with the rigid part removed; 'force' = the free-floating cell's
response q = S~^+ f to 64 Gaussian equilibrated port forces f (the cell counterpart of the lattice's random loads, which are
forces: their traces are dominated by soft modes, where the relative error is largest; Gaussian displacements are dominated by
stiff high-frequency modes).
Per variant (cycles N, weak-patch seed and radius, or no patch):
  ratio      q^T S_hat q / q^T S~ q - 1 over the 64 vectors (mean, p90, max)
  where      error field du = E_hat q - E q (8 vectors), its energy du^T K du split by node class: weak nodes (diag3 norm below
             1% of the median, as weak_patch), within 2 grid steps of a weak node, box-face nodes, box edges (on >= 2 box faces),
             other; and the share inside the patch
  patch      patch nodes / interior DOFs
Variants: name=cycles:seed:dil, seed in {cutweakbox, weak, weakport, none}.
Usage: diag_rand.py <out.jsonl> <ckpt> <case@body>[,...] <maps.json[,...]> --maps strz0.5
       [--variants c2w=2:cutweakbox:2,c2w3=2:cutweakbox:3,c2wall=2:weak:2,c4w=4:cutweakbox:2,c2=2:none:0]"""
import argparse, gc, json, sys
from pathlib import Path
import numpy as np
import torch

import models as MD                                                       # first: applies OPL_CONV_FP32
import a0_eval as AE
import a0_budget as AB
import mapped_cell as MC
import corot_smooth as CR
import weak_patch as WP
import mapped_fast as MF

dev, dt = AE.dev, AE.dt


def node_classes(C, nd):
    n = C.n; M = 2 * n + 1
    g = np.stack(np.unravel_index(C.nodes, (M,) * 3), 1)
    onface = ((g == 0) | (g == 2 * n)).sum(1)
    weak = np.asarray(nd['weak'], bool)
    vol = torch.zeros((1, 1, M, M, M), dtype=torch.float32, device=dev)
    gt = torch.as_tensor(g, device=dev)
    w = gt[torch.as_tensor(weak, device=dev)]
    vol[0, 0, w[:, 0], w[:, 1], w[:, 2]] = 1
    vol = torch.nn.functional.max_pool3d(vol, 5, stride=1, padding=2)
    near = (vol[0, 0, gt[:, 0], gt[:, 1], gt[:, 2]] > 0).cpu().numpy() & ~weak
    return dict(weak=weak, near_weak=near, edge=(onface >= 2) & ~weak & ~near, face=(onface == 1) & ~weak & ~near,
                other=(onface == 0) & ~weak & ~near)


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('ckpt'); ap.add_argument('cases'); ap.add_argument('mapsjson')
    ap.add_argument('--maps', default='strz0.5')
    ap.add_argument('--variants', default='c2w=2:cutweakbox:2,c2w3=2:cutweakbox:3,c2wall=2:weak:2,c4w=4:cutweakbox:2,c2=2:none:0')
    ap.add_argument('--work', default='/root/autodl-tmp/OPL/A0/work_diag'); ap.add_argument('--qmode', default='gauss,force')
    a = ap.parse_args(argv)
    specs = {}
    for f in a.mapsjson.split(','):
        specs.update({m['name']: m['spec'] for m in json.loads(Path(f).read_text())})
    variants = [(v.split('=')[0], *v.split('=')[1].split(':')) for v in a.variants.split(',')]
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
            cls = node_classes(C, g.nd)
            clsT = {k: torch.as_tensor(np.repeat(v, 3), device=dev) for k, v in cls.items()}
            C.factor(neumann=True, fp32=False)
            gen = torch.Generator(device=dev).manual_seed(0)
            QS = {}
            for qm in a.qmode.split(','):
                X = torch.randn((C.np_, 64), dtype=dt, device=dev, generator=gen)
                X = X - C.Q @ (C.Q.T @ X)
                if qm == 'force':
                    X = C.neumann(X); X = X - C.Q @ (C.Q.T @ X)
                X = X / X.norm(dim=0)
                QS[qm] = (X, (X * C.apply(X)).sum(0), C.extend(X[:, :8]))
            for (vname, cyc, seed, dil), qm in [(v, m) for v in variants for m in QS]:
                Q, e_ex, U_ex = QS[qm]
                cyc, dil = int(cyc), int(dil)
                WP.free(C); C._wp = None
                rec = dict(case=case, map=mname, variant=vname, cycles=cyc, seed=seed, dil=dil, qmode=qm)
                if seed != 'none':
                    rec['patch'] = WP.setup(C, g.nd, dil=dil, seed=seed)
                g.set_budget('V0R', cyc, wrap, wp=seed != 'none')
                op = MF.MappedFastOp(g, model, cyc, wrap, patch=seed != 'none' and C._wp is not None, rigid64=True)
                e_hat = torch.cat([(Q[:, j:j + 16] * op.s_hat(Q[:, j:j + 16])).sum(0) for j in range(0, 64, 16)])
                r = (e_hat / e_ex - 1).cpu().numpy()
                rec['ratio'] = dict(mean=float(r.mean()), p90=float(np.quantile(r, .9)), max=float(r.max()), min=float(r.min()))
                du = op.field(Q[:, :8]) - U_ex
                en = (du * (C.K @ du)).sum(1)                                          # per DOF, summed over 8 vectors
                tot = float(en.sum())
                rec['where'] = {k: float(en[m].sum() / tot) for k, m in clsT.items()}
                if C._wp is not None:
                    inp = torch.zeros(C.nb, dtype=torch.bool, device=dev); inp[C._wp[0]] = True
                    rec['where']['in_patch'] = float(en[inp].sum() / tot)
                with open(a.out, 'a') as f:
                    f.write(json.dumps(rec) + '\n')
                print(json.dumps(rec), flush=True)
                del op, du, en
                torch.cuda.empty_cache()
            model.caches.pop(g.case, None)
            WP.free(C); C._free(); del C, g; gc.collect(); torch.cuda.empty_cache()
    print('DONE', flush=True)


if __name__ == '__main__':
    main(sys.argv[1:])
