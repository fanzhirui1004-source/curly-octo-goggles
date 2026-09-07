#!/usr/bin/env python3
"""What KIND is every cut-face carrier node?  Lattice coordinates, and whether an off-line node is the exact centroid
of the polygon it subdivides."""
from __future__ import annotations
import json, sys, tempfile
from fractions import Fraction
from importlib import import_module
from pathlib import Path
import numpy as np

ROOT = Path("/root/autodl-tmp/cut_control_tpms_v1_full_cube_single_cell_v1"); sys.path.insert(0, str(ROOT / "src"))
P = "pred777h_full_cube_single_cell_mesh_graph_fixed_port_schur_v1"
fcb = import_module(f"{P}.full_cube_backend"); snap = import_module(f"{P}.mesher_neutral_snapshot"); pipe = import_module(f"{P}.sheet_label_pipeline")
CN = 32


def frac(x): return {"numerator": Fraction(x).numerator, "denominator": Fraction(x).denominator}


def build(base, a, b, t):
    man = json.loads(Path(base).read_text())
    man["cut_plane"]["raw_coefficients"] = {"a": frac(a), "b": frac(b), "c": frac(0), "d": frac(Fraction(a) + Fraction(b) - Fraction(t))}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh: json.dump(man, fh); p = fh.name
    _, spec, geom = snap.load_geometry_manifest(Path(p)); Path(p).unlink()
    charts, *_ = fcb.compile_full_cube_geometry_inputs(spec)
    return pipe.build_carrier_layout(charts, spec, carrier_n=CN)


def report(base, a, b, t, label):
    lay = build(base, a, b, t); tr = lay.local_trace("cut_0")
    V = np.asarray(tr.node_coordinates, float); T = np.asarray(tr.triangles, int)
    lat = np.abs(V * CN - np.round(V * CN)) < 1e-7                  # which coordinates sit on the 1/32 lattice
    k = lat.sum(axis=1)
    print(f"\n=== {label}: a/b {a}/{b}, t {float(t):.6g} | {len(V)} nodes, {len(T)} triangles ===")
    print(f"  nodes with 3 lattice coords (a background grid NODE): {int((k == 3).sum())}")
    print(f"  nodes with 2 (a background grid LINE crossing):        {int((k == 2).sum())}")
    print(f"  nodes with 1 (a background grid PLANE crossing):       {int((k == 1).sum())}")
    print(f"  nodes with 0:                                          {int((k == 0).sum())}")
    # which single coordinate is on the lattice for the k == 1 nodes
    if (k == 1).any():
        which = {0: 0, 1: 0, 2: 0}
        for row in lat[k == 1]: which[int(np.flatnonzero(row)[0])] += 1
        print(f"     of the k==1 nodes, the lattice coordinate is x:{which[0]} y:{which[1]} z:{which[2]}")
        idx = np.flatnonzero(k == 1)[:3]
        for i in idx: print(f"     example {V[i].round(6).tolist()}")
    # is any node the centroid of its incident triangles' other vertices?
    cen = 0
    for i in np.flatnonzero(k <= 1):
        star = T[(T == i).any(axis=1)]
        ring = sorted({int(v) for f in star for v in f if v != i})
        if len(ring) >= 3 and np.linalg.norm(V[ring].mean(axis=0) - V[i]) < 1e-9: cen += 1
    print(f"  of the {int((k <= 1).sum())} nodes with at most one lattice coordinate, {cen} are the exact centroid of their ring (fan centres)")
    return V, T, k


if __name__ == "__main__":
    base = sys.argv[1]
    report(base, 1, 1, Fraction(1, 2), "45 degrees, half the cell")
    report(base, 2, 1, Fraction(1, 8), "26.6 degrees, a corner")
