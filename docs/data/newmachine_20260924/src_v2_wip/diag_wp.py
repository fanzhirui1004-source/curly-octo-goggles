"""Direction A: where is the error energy of the worst directions? (diagnostic for weak_patch.py; GPU server)
For saved top Ritz vectors X_<case>_<map>.npy (a0_worst --detail), computes the error field d = F x - E x of the deployed
extension (V0R + co-rotated Jacobi) and reports the fraction of its nodal energy (d_n^T (K d)_n) captured by candidate
patches: nodes within k grid steps (Chebyshev) of (a) weak retained nodes, (b) all retained cut-band nodes, (c) weak nodes,
and the patch sizes (interior DOFs). Usage: diag_wp.py <worst_dir> <ckpt> <body> <data_dir> <case> <map>"""
import json, sys
from pathlib import Path
import numpy as np
import torch

import models as MD
import a0_eval as AE
import mapped_cell as MC
import corot_smooth as CR

dev, dt = AE.dev, AE.dt


def dilate(C, sel, k):
    M = 2 * C.n + 1
    g = torch.as_tensor(np.stack(np.unravel_index(C.nodes, (M,) * 3), 1), device=dev)
    vol = torch.zeros((1, 1, M, M, M), dtype=torch.float32, device=dev)
    w = g[torch.as_tensor(sel, device=dev)]
    vol[0, 0, w[:, 0], w[:, 1], w[:, 2]] = 1
    if k > 0:
        vol = torch.nn.functional.max_pool3d(vol, 2 * k + 1, stride=1, padding=k)
    return (vol[0, 0, g[:, 0], g[:, 1], g[:, 2]] > 0).cpu().numpy()


def main(argv):
    wd, ckp, body, data, case, mname = argv[:6]
    specs = {m['name']: m['spec'] for m in json.loads(Path('src/a0_maps.json').read_text())}
    C = MC.MappedCell(case, body, specs[mname], log=lambda s_: None); C.assemble()
    g = AE.MappedGeo(case, body, Path(data) / mname, neumann=False, log=lambda s_: None, cell=C)
    ck = torch.load(ckp, map_location=dev, weights_only=False)
    model = MD.build(ck['cfg']['model'], [g], **dict(ck['cfg'].get('model_args', {}))).to(dev)
    MD.load_compat(model, ck['model']); model.eval()
    g.set_variant('V0R', AE._Wrap(model)); CR.install(); CR.set_corot(C, AE.nodal_rotations(C))
    X = torch.as_tensor(np.load(Path(wd) / f'X_{case}_{mname}.npy'), device=dev)
    C.factor(neumann=False)
    with torch.no_grad():
        U = g.field(model, X).to(dt); E = C.extend(X)
    D = U - E; en = (D * (C.K @ D)).reshape(-1, 3, X.shape[1]).sum(1).cpu().numpy()      # (N, m)
    nd = g.nd; weak = np.asarray(nd['weak'], bool); port = np.asarray(nd['is_port'], bool); cut = np.asarray(nd['is_cut'], bool)
    interior_nodes = ~port
    out = dict(case=case, map=mname, err_energy=en.sum(0).tolist(), weak_int=int((weak & ~port).sum()))
    for name, seed in (('weakport', weak & port), ('cutport', cut & port), ('weak', weak)):
        for k in (0, 1, 2, 3, 4, 6):
            s = dilate(C, seed, k)
            out[f'{name}_k{k}'] = dict(frac=(en[s].sum(0) / en.sum(0)).round(3).tolist(), int_dofs=int(3 * (s & interior_nodes).sum()))
    # error energy on interior nodes sorted: how many nodes hold 50/80/90%?
    e1 = np.sort(en[:, 0])[::-1]; c = np.cumsum(e1) / e1.sum()
    out['nodes_for_50_80_90'] = [int(np.searchsorted(c, q) + 1) for q in (.5, .8, .9)]
    print(json.dumps(out), flush=True)


if __name__ == '__main__':
    main(sys.argv[1:])
