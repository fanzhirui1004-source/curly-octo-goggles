"""Direction A: equivariance test of V0R under a rigid rotation, with the P1 scalar-Jacobi correction and with the co-rotated
Jacobi preconditioner of corot_smooth.py (new script, default paths unchanged; runs on the GPU server).

One cell, maps 'id' and 'rot30' (a0_maps.json). The rot30 banks are the identity banks rotated node by node (q -> R q), so
both maps see the same physical directions (energy is invariant, unit exact energy is kept). For each class:
  V0R      : as in a0_eval (P1 correction)                      -> differences = smoother + rounding
  V0R+cr   : co-rotated Jacobi preconditioner on both cells    -> differences should be at fp32 rounding (~1e-7)
and the identity check V0R+cr(id) - V0R(id) (R = I up to rounding: should be at rounding too).
Usage: t_corot.py <out_dir> <ckpt> <body_dir> <case> <maps.json> [--classes force_c,face_c,grf]"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch

import models as MD                                                       # first: applies OPL_CONV_FP32
import trainlib as TL
import a0_eval as AE
import mapped_cell as MC
import corot_smooth as CR

dev, dt = AE.dev, AE.dt


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('ckpt'); ap.add_argument('body'); ap.add_argument('case'); ap.add_argument('maps')
    ap.add_argument('--classes', default='force_c,face_c,grf'); ap.add_argument('--val', type=int, default=64)
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    maps = {m['name']: m['spec'] for m in json.loads(Path(a.maps).read_text())}
    classes = a.classes.split(',')
    splits = (('train', 8), ('val', a.val), ('test', 8)); total = sum(m for _, m in splits)
    rep = dict(case=a.case, classes=classes)
    t0 = time.perf_counter()

    C0 = MC.MappedCell(a.case, a.body, maps['id'], log=lambda s_: None); C0.assemble()
    C0.factor(neumann=True, interior=False); raw, _ = AE.raw_banks(C0, a.case, 'id', total, classes); C0._free()
    C0.factor(neumann=False); banks0 = AE.finish_banks(C0, raw, {c: {} for c in classes}); del raw; C0._free()

    C1 = MC.MappedCell(a.case, a.body, maps['rot30'], log=lambda s_: None); C1.assemble()
    R0, R1 = AE.nodal_rotations(C0), AE.nodal_rotations(C1)
    rep['R0_minus_I'] = float((R0 - torch.eye(3, dtype=dt, device=dev)).abs().max())
    rep['R1_spread'] = float((R1 - R1[0]).abs().max())
    onport = torch.as_tensor(np.isin(C1.nodes, C1.port_node_ids), device=dev)
    Rp = R1[onport]
    banks1 = {c: torch.einsum('nij,njk->nik', Rp, q.reshape(-1, 3, q.shape[1])).reshape(q.shape) for c, q in banks0.items()}

    geos = []
    for name, C, banks in (('id', C0, banks0), ('rot30', C1, banks1)):
        AE.write_data(C, out / 'data' / name / a.case, banks, splits, a.body)
        g = AE.MappedGeo(a.case, a.body, out / 'data' / name, neumann=False, log=lambda s_: None, cell=C)
        g.case = f'{a.case}@{name}'; geos.append(g)
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False); cfg = ck['cfg']
    model = MD.build(cfg['model'], [geos[0]], **dict(cfg.get('model_args', {}))).to(dev)
    MD.load_compat(model, ck['model']); model.eval(); model.add_geo(geos[1])
    wrap = AE._Wrap(model)

    def energies(g):
        g.set_variant('V0R', wrap)
        with torch.no_grad():
            return {c: torch.cat([TL.energy(g.field(model, g.banks['val'][c][:, j:j + 16]), g.C.K) - 1
                                  for j in range(0, g.banks['val'][c].shape[1], 16)]).cpu().numpy() for c in g.classes}

    E = {('id', 'V0R'): energies(geos[0]), ('rot30', 'V0R'): energies(geos[1])}
    CR.install()
    CR.set_corot(C0, R0); CR.set_corot(C1, R1)
    E[('id', 'V0R+cr')] = energies(geos[0]); E[('rot30', 'V0R+cr')] = energies(geos[1])
    rep['tail_bounds'] = {'id': list(C0._tail_bounds), 'rot30': list(C1._tail_bounds)}

    rep['classes_out'] = {}
    for c in geos[0].classes:
        r = {f'{m}/{v}_mean_pct': 100 * float(E[(m, v)][c].mean()) for (m, v) in E}
        r['min_excess'] = float(min(E[k][c].min() for k in E))
        r['V0R: max|e_rot - e_id|'] = float(np.abs(E[('rot30', 'V0R')][c] - E[('id', 'V0R')][c]).max())
        r['V0R+cr: max|e_rot - e_id|'] = float(np.abs(E[('rot30', 'V0R+cr')][c] - E[('id', 'V0R+cr')][c]).max())
        r['id: max|e_cr - e_P1|'] = float(np.abs(E[('id', 'V0R+cr')][c] - E[('id', 'V0R')][c]).max())
        rep['classes_out'][c] = r
    rep['seconds'] = time.perf_counter() - t0
    rep['gpu_peak_gb'] = torch.cuda.max_memory_allocated() / 2 ** 30
    (out / 'T_COROT.json').write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1), flush=True)


if __name__ == '__main__':
    main(sys.argv[1:])
