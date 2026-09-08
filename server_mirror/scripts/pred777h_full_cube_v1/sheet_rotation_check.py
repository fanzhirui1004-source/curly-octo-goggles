#!/usr/bin/env python3
"""Rotation action check for the sheet route: the label of a cube-symmetry image of a cell equals the transformed label
of the original cell up to mesh noise.  Only vertical cuts are generated in production; the other orientations come
from the 48 cube symmetries (24 proper rotations + 24 improper), so this is the premise of the augmentation.

For a symmetry R about the cell centre: tau'(x) = tau(R^-1 x) (corner permutation), cut plane (n, d) -> (R n, d').
The rotated cell is built and labelled independently (its own marching cubes, remesh, Gmsh, Schur); the prediction is
T S T^T with T the vector-dof transform of the carrier-node permutation induced by R on the active carrier set.
"""
from __future__ import annotations
import argparse, json, sys, tempfile, time
from fractions import Fraction
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
P = "pred777h_full_cube_single_cell_mesh_graph_fixed_port_schur_v1"
from importlib import import_module
fcb = import_module(f"{P}.full_cube_backend"); snap = import_module(f"{P}.mesher_neutral_snapshot"); sss = import_module(f"{P}.sheet_solid_surface")
pipe = import_module(f"{P}.sheet_label_pipeline"); t10 = import_module(f"{P}.tet10_label"); rot = import_module(f"{P}.rotations"); ct = import_module(f"{P}.carrier_triangulation")


def fp(v: Fraction) -> dict:
    return {"numerator": v.numerator, "denominator": v.denominator}


def rotated_manifest(payload: dict, R: np.ndarray, out: Path) -> Path:
    """Manifest of the symmetry image: the cell's eight corner values (taken from the parent lattice at the cell's
    placement) permuted, written as a single-cell field at the same placement; the world cut plane transformed about
    the cell centre (exact rationals)."""
    tf = payload["global_thickness_field"]; shape = [int(v) for v in tf["cell_shape"]]
    origin = [Fraction(v["numerator"], v["denominator"]) for v in tf["origin"]]; size = [Fraction(v["numerator"], v["denominator"]) for v in tf["cell_size"]]
    t = [Fraction(v["numerator"], v["denominator"]) for v in payload["cell_placement"]["translation"]]
    base = [int((t[k] - origin[k]) / size[k]) for k in range(3)]          # lattice index of the cell's lower corner
    lat = lambda i, j, k: ((base[0] + i) * (shape[1] + 1) + (base[1] + j)) * (shape[2] + 1) + (base[2] + k)
    vals = tf["vertex_values"]; index = lambda i, j, k: (i * 2 + j) * 2 + k
    Ri = np.rint(np.asarray(R, dtype=float)).astype(int); new_vals = [None] * 8
    for i in range(2):
        for j in range(2):
            for k in range(2):
                x = np.array([i, j, k]) - 0.5; xi = Ri.T @ x + 0.5   # tau'(x) = tau(R^-1 x), R^-1 = R^T
                new_vals[index(i, j, k)] = vals[lat(int(round(xi[0])), int(round(xi[1])), int(round(xi[2])))]
    new = json.loads(json.dumps(payload))
    new["global_thickness_field"] = {"field_id": tf["field_id"] + "_ROT", "origin": [fp(v) for v in t], "cell_size": [fp(v) for v in size], "cell_shape": [1, 1, 1], "vertex_values": new_vals}
    new["cell_id"] += "_rot"
    if payload.get("cut_plane"):
        c = payload["cut_plane"]["raw_coefficients"]; n = [Fraction(c[k]["numerator"], c[k]["denominator"]) for k in "abc"]; d = Fraction(c["d"]["numerator"], c["d"]["denominator"])
        centre = [t[k] + Fraction(1, 2) for k in range(3)]
        nn = [sum(Fraction(int(Ri[r, cc])) * n[cc] for cc in range(3)) for r in range(3)]
        dd = d - sum(n[cc] * centre[cc] for cc in range(3)) + sum(nn[r] * centre[r] for r in range(3))
        new["cut_plane"]["raw_coefficients"] = {"a": fp(nn[0]), "b": fp(nn[1]), "c": fp(nn[2]), "d": fp(dd)}
    out.mkdir(parents=True, exist_ok=True); (out / "geometry_material_manifest.json").write_text(json.dumps(new, indent=1)); return out / "geometry_material_manifest.json"


def label(manifest: Path, a, workdir: Path):
    _, spec, geom = snap.load_geometry_manifest(manifest); charts, *_ = fcb.compile_full_cube_geometry_inputs(spec)
    layout = pipe.build_carrier_layout(charts, spec, carrier_n=a.carrier_n, triangulation=a.carrier_triangulation)
    surf = sss.build_sheet_solid_surface(geom, layout, n_per_unit=a.n_per_unit, remesh_size=a.remesh_size, carrier_n=a.carrier_n)
    workdir.mkdir(parents=True, exist_ok=True); sss.mesh_sheet_solid(surf, workdir / "mesh.mesh", interior_size_max=a.size_max, algorithm3d=1, threads=a.threads)
    stage = pipe.compile_port(workdir / "mesh.mesh", geom, spec, layout, carrier_n=a.carrier_n); sch = pipe.schur_label(stage, workers=a.workers)
    coords = np.asarray(stage.port.carrier_coordinates)[stage.active]
    norms = t10.build_carrier_norms(layout, stage.port, wavelength_cutoff=4.0 / a.carrier_n)
    return sch.schur_active, coords, stage.active, norms, layout, stage.port


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--geometry-manifest", type=Path, required=True); ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--n-per-unit", type=int, default=48); ap.add_argument("--remesh-size", type=float, default=0.03); ap.add_argument("--size-max", type=float, default=0.05)
    ap.add_argument("--carrier-n", type=int, default=32); ap.add_argument("--workers", type=int, default=8); ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--symmetries", default="R05,M01", help="comma list: Rxx = proper rotation index, Mxx = improper (reflection) index")
    ap.add_argument("--carrier-triangulation", choices=("frozen", "unionjack"), default=pipe.CARRIER_TRIANGULATION, help="frozen: the vendored fan by node id; unionjack: the cube-symmetric parity diagonal")
    a = ap.parse_args(); out = a.output_dir; out.mkdir(parents=True, exist_ok=True)
    payload = json.loads(a.geometry_manifest.read_text()); rep = {"manifest": str(a.geometry_manifest), "results": {}}
    proper = rot.proper_cubic_rotations()
    improper = [np.diag([-1, 1, 1]).astype(np.int8) @ r.matrix for r in proper]          # 24 reflections (det -1)
    S0, C0, act0, norms0, layout0, port0 = label(a.geometry_manifest, a, out / "original")
    print(f"original label: q = {S0.shape[0]}", flush=True)
    for tag in a.symmetries.split(","):
        R = np.eye(3, dtype=np.int8) if tag == "ID" else ((-np.eye(3)).astype(np.int8) if tag == "INV" else (proper[int(tag[1:])].matrix if tag[0] == "R" else improper[int(tag[1:])]))
        man = rotated_manifest(payload, R, out / f"cell_{tag}")
        S1, C1, act1, norms1, layout1, port1 = label(man, a, out / f"label_{tag}")
        # permutation of active carrier nodes induced by R (by coordinates rounded to 1e-9; box-face lattice, cut-face crossings and merged nodes alike)
        key = lambda p: tuple(np.round(np.asarray(p) * 1e9).astype(np.int64).tolist())      # exact coordinates: cut-face nodes are off the lattice
        idx1 = {key(p): i for i, p in enumerate(C1)}; perm = np.empty(len(C0), dtype=np.int64); missing = 0
        for i, p in enumerate(C0):
            k = key(rot.rotate_points(p[None], R)[0]); perm[i] = idx1.get(k, -1); missing += perm[i] < 0
        res = {"symmetry": tag, "det": int(round(np.linalg.det(R))), "matrix": R.astype(int).tolist(), "q_original": int(S0.shape[0]), "q_rotated": int(S1.shape[0]), "active_set_mismatch": int(missing + (len(C1) - (len(C0) - missing)))}
        # rows of the original that vanish (no support) and are absent in the rotated set, and vice versa: harmless if their rows are zero
        rown0 = np.linalg.norm(S0.reshape(len(C0), 3, -1), axis=(1, 2)); only0 = np.where(perm < 0)[0]
        matched1 = set(perm[perm >= 0].tolist()); only1 = np.array([i for i in range(len(C1)) if i not in matched1], dtype=int)
        rown1 = np.linalg.norm(S1.reshape(len(C1), 3, -1), axis=(1, 2))
        res.update({"only_original_nodes": int(len(only0)), "only_original_max_row_norm_rel": float(rown0[only0].max() / rown0.max()) if len(only0) else 0.0,
                    "only_rotated_nodes": int(len(only1)), "only_rotated_max_row_norm_rel": float(rown1[only1].max() / rown1.max()) if len(only1) else 0.0})
        # compare on the common nodes (intersection), in the rotated cell's carrier norms restricted to them
        keep0 = np.where(perm >= 0)[0]; keep1 = perm[keep0]
        T = rot.vector_dof_transform(np.arange(len(keep0)), R)                    # rotation of the vector components only
        S0c = S0[np.ix_(np.repeat(keep0 * 3, 3) + np.tile(np.arange(3), len(keep0)), np.repeat(keep0 * 3, 3) + np.tile(np.arange(3), len(keep0)))]
        S1c = S1[np.ix_(np.repeat(keep1 * 3, 3) + np.tile(np.arange(3), len(keep1)), np.repeat(keep1 * 3, 3) + np.tile(np.arange(3), len(keep1)))]
        S_pred = rot.rotate_schur(S0c, T)
        cidx1 = np.where(act1)[0]; Mc = norms1.mass_scalar[np.ix_(cidx1[keep1], cidx1[keep1])]; Lc = norms1.laplacian_scalar[np.ix_(cidx1[keep1], cidx1[keep1])]
        w, Vm = np.linalg.eigh(Mc); Mih = Vm @ np.diag(w ** -0.5) @ Vm.T; mu, U = np.linalg.eigh(Mih @ Lc @ Mih); kmax = 2 * np.pi / (4.0 / a.carrier_n); Uk = Mih @ U[:, mu <= kmax ** 2]
        normsc = t10.CarrierNorms(Mc, Lc, np.kron(Mih, np.eye(3)), np.kron(Uk, np.eye(3)), mu[mu <= kmax ** 2], 4.0 / a.carrier_n)
        cmp = t10.compare_operators(S_pred, S1c, normsc)
        res.update({k: float(v) for k, v in cmp.items()}); res["common_nodes"] = int(len(keep0))
        rep["results"][tag] = res; print(tag, json.dumps(res, default=float), flush=True)
    (out / "SHEET_ROTATION_CHECK.json").write_text(json.dumps(rep, indent=1, default=float)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
