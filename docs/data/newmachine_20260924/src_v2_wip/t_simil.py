"""Direction A: invariance test of V0Rc under a SIMILARITY map x = s R X (new script, default paths unchanged; GPU server).
V0Rc = V0R with the co-rotated Jacobi preconditioner (corot_smooth.py); network inputs written plainly (P1 diag3) or with
diag3 / det(J)^(1/3) (a0_eval.write_data(det_norm=True)). K~ = s R K R^T, so with unit-energy banks q -> R q / sqrt(s) the
energy excess must coincide with the identity map when every input is invariant.

One cell, maps 'id' and 'rot30' (a0_maps.json). The rot30 banks are the identity banks rotated node by node (q -> R q), so
both maps see the same physical directions (energy is invariant, unit exact energy is kept). For each class:
  V0R      : as in a0_eval (P1 correction)                      -> differences = smoother + rounding
  V0R+cr   : co-rotated Jacobi preconditioner on both cells    -> differences should be at fp32 rounding (~1e-7)
and the identity check V0R+cr(id) - V0R(id) (R = I up to rounding: should be at rounding too).
Usage: t_simil.py <out_dir> <ckpt> <body_dir> <case> <maps.json> [--s 1.5] [--classes force_c,face_c,grf]"""
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
    ap.add_argument('--s', type=float, default=1.5)
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

    spec1 = dict(kind='affine', A=(a.s * np.asarray(maps['rot30']['A'])).tolist())
    C1 = MC.MappedCell(a.case, a.body, spec1, log=lambda s_: None); C1.assemble()
    R0, R1 = AE.nodal_rotations(C0), AE.nodal_rotations(C1)
    rep['R0_minus_I'] = float((R0 - torch.eye(3, dtype=dt, device=dev)).abs().max())
    rep['R1_spread'] = float((R1 - R1[0]).abs().max())
    onport = torch.as_tensor(np.isin(C1.nodes, C1.port_node_ids), device=dev)
    Rp = R1[onport]
    banks1 = {c: torch.einsum('nij,njk->nik', Rp, q.reshape(-1, 3, q.shape[1])).reshape(q.shape) / a.s ** 0.5 for c, q in banks0.items()}

    geos = []
    for name, C, banks, dn in (('id', C0, banks0, False), ('sim', C1, banks1, False), ('simdn', C1, banks1, True)):
        AE.write_data(C, out / 'data' / name / a.case, banks, splits, a.body, det_norm=dn)
        g = AE.MappedGeo(a.case, a.body, out / 'data' / name, neumann=False, log=lambda s_: None, cell=C)
        g.case = f'{a.case}@{name}'; geos.append(g)
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False); cfg = ck['cfg']
    model = MD.build(cfg['model'], [geos[0]], **dict(cfg.get('model_args', {}))).to(dev)
    MD.load_compat(model, ck['model']); model.eval(); model.add_geo(geos[1]); model.add_geo(geos[2])
    wrap = AE._Wrap(model)
    rep['model_args'] = {k: (v if isinstance(v, (int, float, str, bool)) else str(v)) for k, v in dict(cfg.get('model_args', {})).items()}

    def energies(g):
        g.set_variant('V0R', wrap)
        with torch.no_grad():
            return {c: torch.cat([TL.energy(g.field(model, g.banks['val'][c][:, j:j + 16]), g.C.K) - 1
                                  for j in range(0, g.banks['val'][c].shape[1], 16)]).cpu().numpy() for c in g.classes}

    CR.install()
    CR.set_corot(C0, R0); CR.set_corot(C1, R1)
    E = {('id', 'V0Rc'): energies(geos[0]), ('sim', 'V0Rc'): energies(geos[1]), ('simdn', 'V0Rc'): energies(geos[2])}
    rep['tail_bounds'] = {'id': list(C0._tail_bounds), 'sim': list(C1._tail_bounds)}
    rep['classes_out'] = {}
    for c in geos[0].classes:
        r = {f'{m}_mean_pct': 100 * float(E[(m, 'V0Rc')][c].mean()) for (m, _) in E}
        r['min_excess'] = float(min(E[k][c].min() for k in E))
        r['plain: max|e_sim - e_id|'] = float(np.abs(E[('sim', 'V0Rc')][c] - E[('id', 'V0Rc')][c]).max())
        r['detnorm: max|e_sim - e_id|'] = float(np.abs(E[('simdn', 'V0Rc')][c] - E[('id', 'V0Rc')][c]).max())
        rep['classes_out'][c] = r
    rep['seconds'] = time.perf_counter() - t0
    rep['gpu_peak_gb'] = torch.cuda.max_memory_allocated() / 2 ** 30
    (out / 'T_SIMIL.json').write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1), flush=True)


if __name__ == '__main__':
    main(sys.argv[1:])
