#!/usr/bin/env python3
"""Are the needle apexes the carrier support points?"""
import json, sys
from importlib import import_module
from pathlib import Path
import numpy as np
ROOT = Path("/root/autodl-tmp/cut_control_tpms_v1_full_cube_single_cell_v1"); sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
P = "pred777h_full_cube_single_cell_mesh_graph_fixed_port_schur_v1"
fcb = import_module(f"{P}.full_cube_backend"); snap = import_module(f"{P}.mesher_neutral_snapshot"); sss = import_module(f"{P}.sheet_solid_surface"); pipe = import_module(f"{P}.sheet_label_pipeline")
from cap_needles import tri_min_angle
d = Path(sys.argv[1]); r = json.loads((d / "LABEL_RECEIPT.json").read_text())
_, spec, geom = snap.load_geometry_manifest(Path(r["geometry_manifest"])); cn = int(r["resolution"]["carrier_n"])
charts, *_ = fcb.compile_full_cube_geometry_inputs(spec); layout = pipe.build_carrier_layout(charts, spec, carrier_n=cn)
ports = [n for n, _, _ in sss.clip_planes(geom)]
_, _, support = sss.carrier_active_set(geom, layout, ports, carrier_n=cn)
S = np.asarray([p for face in support.values() for p in face.values()]).reshape(-1, 3)
lines = (d / "SHEET_SOLID_SURFACE.off").read_text().split("\n"); nv, nf, _ = map(int, lines[1].split())
V = np.asarray([[float(x) for x in l.split()[:3]] for l in lines[2:2 + nv]]); F = np.asarray([[int(x) for x in l.split()[1:4]] for l in lines[2 + nv:2 + nv + nf]])
onplane = np.zeros(len(V), bool)
for k in range(3): onplane |= (np.abs(V[:, k]) < 1e-9) | (np.abs(V[:, k] - 1) < 1e-9)
Fc = F[onplane[F].all(axis=1)]; ang = tri_min_angle(V[Fc]); needles = np.flatnonzero(ang < 1.0)
hits = 0; depths = []
for i in needles:
    f = Fc[i]; p = V[f]; e = [np.linalg.norm(p[1] - p[2]), np.linalg.norm(p[0] - p[2]), np.linalg.norm(p[0] - p[1])]; apex = V[f[int(np.argmax(e))]]
    dist = np.linalg.norm(S - apex, axis=1).min(); hits += dist < 1e-9
    depths.append(float(-sss.sheet_level(geom, apex[None])[0]))
print(f"support points {len(S)}; needles {len(needles)}; apex is a support point: {hits}")
print("level depth (-lev) at the needle apexes:", np.round(sorted(depths)[:10], 5))
allS = -sss.sheet_level(geom, S); print(f"level depth of all support points: p05 {np.percentile(allS, 5):.4f} median {np.median(allS):.4f}; below 0.01: {(allS < 0.01).sum()}, below 0.02: {(allS < 0.02).sum()}")
