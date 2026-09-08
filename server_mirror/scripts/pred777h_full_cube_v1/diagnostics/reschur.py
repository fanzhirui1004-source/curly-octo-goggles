#!/usr/bin/env python3
"""Re-run the fixed port and the Tet10 Schur on an alternative volume mesh of a produced label's geometry, and
compare the operator with the label's own and with a reference label.

    reschur.py <label dir> <mesh path> [<reference label dir>] [--out DIR]

Reports lambda_max, the top mode's localisation (mass on its four heaviest carrier nodes), the uniform-strain
stiffness relative differences, and the spectrum ladder against the reference on common support.
"""
from __future__ import annotations

import argparse, json, sys, time
from importlib import import_module
from pathlib import Path

import numpy as np

ROOT = Path("/root/autodl-tmp/cut_control_tpms_v1_full_cube_single_cell_v1")
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(Path(__file__).resolve().parent))
P = "pred777h_full_cube_single_cell_mesh_graph_fixed_port_schur_v1"
fcb = import_module(f"{P}.full_cube_backend"); snap = import_module(f"{P}.mesher_neutral_snapshot")
pipe = import_module(f"{P}.sheet_label_pipeline")
from label_error import load_label, uniform_strain_stiffness


def top_mode(S):
    w, U = np.linalg.eigh(S); v = U[:, -1]; n = S.shape[0] // 3
    part = np.sort(np.linalg.norm(v.reshape(n, 3), axis=1) ** 2)[::-1]
    return float(w[-1]), float(part[:4].sum()), float(part[:16].sum()), int((part > 0.01).sum())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("label"); ap.add_argument("mesh"); ap.add_argument("reference", nargs="?")
    ap.add_argument("--out"); ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    lab_dir = Path(a.label); r = json.loads((lab_dir / "LABEL_RECEIPT.json").read_text())
    _, spec, geom = snap.load_geometry_manifest(Path(r["geometry_manifest"])); cn = int(r["resolution"]["carrier_n"])
    charts, *_ = fcb.compile_full_cube_geometry_inputs(spec)
    layout = pipe.build_carrier_layout(charts, spec, carrier_n=cn)
    t0 = time.perf_counter()
    stage = pipe.compile_port(Path(a.mesh), geom, spec, layout, carrier_n=cn)
    sch = pipe.schur_label(stage, workers=a.workers)
    S = sch.schur_active; xyz = np.asarray(stage.port.carrier_coordinates)[stage.active]
    ids = [str(k) for k, keep in zip(stage.port.active_global_carrier_ids, stage.active) if keep]
    own = load_label(lab_dir)
    lam, t4, t16, n1 = top_mode(S); lam0, t40, t160, n10 = top_mode(own["S"])
    out = {"label": str(lab_dir), "mesh": str(a.mesh), "q": int(S.shape[0]), "q_label": own["q"],
           "rigid_residual": sch.rigid_residual, "seconds": time.perf_counter() - t0,
           "lambda_max": lam, "lambda_max_label": lam0, "top4": t4, "top4_label": t40, "nodes_above_1pct": n1, "nodes_above_1pct_label": n10}
    print(f"q {S.shape[0]} (label {own['q']}), rigid residual {sch.rigid_residual:.1e}, {out['seconds']:.0f} s")
    print(f"lambda_max  remeshed {lam:.5g}   label {lam0:.5g}   ratio {lam / lam0:.3f}")
    print(f"top mode on 4 nodes  remeshed {t4:.3f} ({n1} nodes > 1%)   label {t40:.3f} ({n10} nodes > 1%)")
    same = (len(ids) == len(own["ids"])) and all(x == y for x, y in zip(ids, own["ids"]))
    if same:
        C1 = uniform_strain_stiffness(S, xyz); C0 = uniform_strain_stiffness(own["S"], own["xyz"])
        rel = C1 / C0 - 1; out["uniform_strain_rel_vs_label"] = rel.tolist()
        print(f"uniform strain vs label: max |rel| {np.abs(rel).max():.2e}   ({', '.join(f'{v:+.1e}' for v in rel)})")
    if a.reference:
        ref = load_label(Path(a.reference))
        ib = {k: i for i, k in enumerate(ref["ids"])}; common = [k for k in ids if k in ib]
        ka = np.asarray([ids.index(k) for k in common]); kb = np.asarray([ib[k] for k in common])
        da = np.repeat(3 * ka, 3) + np.tile(np.arange(3), len(ka)); db = np.repeat(3 * kb, 3) + np.tile(np.arange(3), len(kb))
        Cr = uniform_strain_stiffness(ref["S"][np.ix_(db, db)], ref["xyz"][kb]); C1 = uniform_strain_stiffness(S[np.ix_(da, da)], xyz[ka])
        C0 = uniform_strain_stiffness(own["S"][np.ix_(da, da)], xyz[ka]) if same else None
        lamr, t4r, _, n1r = top_mode(ref["S"])
        print(f"reference: lambda_max {lamr:.5g} (remeshed/ref {lam / lamr:.3f}, label/ref {lam0 / lamr:.3f}), top4 {t4r:.3f} ({n1r} nodes)")
        print(f"uniform strain vs reference: remeshed max |rel| {np.abs(C1 / Cr - 1).max():.2e}" + (f", label max |rel| {np.abs(C0 / Cr - 1).max():.2e}" if C0 is not None else ""))
        out["lambda_max_reference"] = lamr; out["uniform_strain_rel_vs_reference"] = (C1 / Cr - 1).tolist()
        if C0 is not None: out["uniform_strain_rel_label_vs_reference"] = (C0 / Cr - 1).tolist()
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        np.save(Path(a.out) / "S.npy", S); json.dump(out, open(Path(a.out) / "RESCHUR.json", "w"), indent=1, default=float)
    return 0


if __name__ == "__main__":
    sys.exit(main())
