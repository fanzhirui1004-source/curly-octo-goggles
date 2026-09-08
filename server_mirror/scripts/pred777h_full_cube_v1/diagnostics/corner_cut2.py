#!/usr/bin/env python3
"""What the cut-face carrier does when the plane only shaves a corner off the cell.

The plane a x + b y <= d (rational, vertical) with t = a + b - d > 0 removes the corner (1, 1, *).  For a shrinking t
this reports the cut face's own carrier: how many nodes, WHICH background grid lines they sit on (the index set), what
the merge pass does, and the shape of the triangles.  A cut-face carrier node is by construction the crossing of the
plane with a grid LINE, so the identity of every node is a member of the fixed index set; what changes with t is which
members are active, and that is what this measures."""
from __future__ import annotations
import json, sys, tempfile
from fractions import Fraction
from importlib import import_module
from pathlib import Path
import numpy as np

ROOT = Path("/root/autodl-tmp/cut_control_tpms_v1_full_cube_single_cell_v1"); sys.path.insert(0, str(ROOT / "src"))
P = "pred777h_full_cube_single_cell_mesh_graph_fixed_port_schur_v1"
fcb = import_module(f"{P}.full_cube_backend"); snap = import_module(f"{P}.mesher_neutral_snapshot")
pipe = import_module(f"{P}.sheet_label_pipeline")
CN = 32; H = 1.0 / CN


def frac(x): return {"numerator": Fraction(x).numerator, "denominator": Fraction(x).denominator}


def identity(p, tol=1e-7):
    """(axis of the grid line, the two lattice indices) or None if the point is not on a background grid line."""
    on = [k for k in range(3) if abs(p[k] * CN - round(p[k] * CN)) < tol]
    if len(on) < 2: return None
    if len(on) == 3: on = on[:2]
    axis = [k for k in range(3) if k not in on][0]
    return (axis, tuple(int(round(p[k] * CN)) for k in on))


def probe(base_manifest, a, b, t, out):
    d = Fraction(a) + Fraction(b) - Fraction(t)
    man = json.loads(Path(base_manifest).read_text())
    man["cut_plane"]["raw_coefficients"] = {"a": frac(a), "b": frac(b), "c": frac(0), "d": frac(d)}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(man, fh); path = fh.name
    _, spec, geom = snap.load_geometry_manifest(Path(path))
    deg = pipe.cut_contains_a_cell_edge(spec)
    charts, *_ = fcb.compile_full_cube_geometry_inputs(spec)
    layout = pipe.build_carrier_layout(charts, spec, carrier_n=CN)
    tr = layout.local_trace("cut_0"); V = np.asarray(tr.node_coordinates, float); T = np.asarray(tr.triangles, int)
    lat = np.abs(V * CN - np.round(V * CN)) < 1e-7; kk = lat.sum(axis=1)
    n3, n2, nc = int((kk == 3).sum()), int((kk == 2).sum()), int((kk <= 1).sum())
    Pt = V[T] if len(T) else np.zeros((0, 3, 3))
    area = 0.5 * np.linalg.norm(np.cross(Pt[:, 1] - Pt[:, 0], Pt[:, 2] - Pt[:, 0]), axis=1).sum() if len(T) else 0.0
    ar = []
    for q in Pt:
        e = max(np.linalg.norm(q[i] - q[j]) for i, j in ((0, 1), (1, 2), (0, 2)))
        A = 0.5 * np.linalg.norm(np.cross(q[1] - q[0], q[2] - q[0])); ar.append(e * e / max(A, 1e-300))
    mg = (layout.diagnostics.get("cut_carrier_merge") or {})
    nrm = float(np.hypot(float(a), float(b)))
    width = float(t) * nrm / (float(a) * float(b)) if a and b else float("nan")   # in-plane length of the cut face's horizontal edge
    dist = float(t) / nrm
    print(f"a/b {a}/{b}  t {float(t):<10.5g} depth {dist:<9.5g} width {width:<9.5g} ({width / H:>7.3f} h) | "
          f"nodes {len(V):>5} = grid-node {n3:>4} + line-crossing {n2:>4} + fan-centroid {nc:>4} | tris {len(T):>5} area {area:<9.5g} "
          f"max aspect {max(ar) if ar else float('nan'):>8.3g} merged {mg.get('merged_nodes')} degen {deg is not None}")
    Path(path).unlink()
    return {"a": a, "b": b, "t": float(t), "width_h": width / H, "nodes": len(V), "grid_nodes": n3, "line_crossings": n2,
            "fan_centroids": nc, "tris": len(T), "area": area, "max_aspect": max(ar) if ar else None,
            "merged": mg.get("merged_nodes"), "degenerate": deg is not None}


if __name__ == "__main__":
    base = sys.argv[1]; out = []
    print(f"carrier spacing h = {H}; merge radius h/3 = {H/3:.6f}\n")
    for a, b in ((1, 1), (2, 1), (8, 1)):
        for t in (Fraction(1, 8), Fraction(1, 32), Fraction(1, 64), Fraction(1, 200), Fraction(1, 1000), Fraction(1, 10000)):
            try: out.append(probe(base, a, b, t, out))
            except Exception as e: print(f"a/b {a}/{b}  t {float(t):<9.5g} FAILED {type(e).__name__}: {str(e)[:120]}")
        print()
    json.dump(out, open("/root/autodl-tmp/_claude_diag/tonight/CORNER_CUT.json", "w"), indent=1, default=float)
