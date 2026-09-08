#!/usr/bin/env python3
"""Combine per-mesh S_*.npy files (from run_tet10_label_gates.py single-mesh runs) into the two gates."""
import argparse, json, sys, glob, os
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
P = "pred777h_full_cube_single_cell_mesh_graph_fixed_port_schur_v1"
from importlib import import_module
adapter = import_module(f"{P}.cgal_mesh3_adapter"); fpa = import_module(f"{P}.fixed_port_adapter"); fcb = import_module(f"{P}.full_cube_backend")
snap = import_module(f"{P}.mesher_neutral_snapshot"); t10 = import_module(f"{P}.tet10_label")
ap = argparse.ArgumentParser(); ap.add_argument("--case-id", required=True); ap.add_argument("--geometry-manifest", type=Path, required=True)
ap.add_argument("--any-mesh", type=Path, required=True, help="one mesh of the geometry, to rebuild the carrier norms"); ap.add_argument("--dir", type=Path, required=True)
ap.add_argument("--reference-tag", default="A1"); ap.add_argument("--wavelength-cutoff", type=float, default=0.25)
ap.add_argument("--repro-threshold", type=float, default=0.005); ap.add_argument("--conv-threshold", type=float, default=0.03)
a = ap.parse_args()
_, spec, geom = snap.load_geometry_manifest(a.geometry_manifest)
charts, *_ = fcb.compile_full_cube_geometry_inputs(spec); layout = fcb.build_full_cube_trace_layout(spec, charts=charts)
mp = {"domain_contract": "COHERENT_TRIANGULATED_CLOSED_VOLUME_BOUNDING_POLYHEDRAL_COMPLEX", "optimization_enabled": False, "semantic_plane_tolerance": 1e-10}
art = adapter.mesh_artifact_from_cgal_medit(a.any_mesh, geometry=geom, mesher_version="5.4", mesher_parameters=mp, plane_tolerance=1e-10, source_geometry_hash=spec.spec_hash)
port = fpa.compile_fixed_port(art, spec, layout); norms = t10.build_carrier_norms(layout, port, wavelength_cutoff=a.wavelength_cutoff)
ops = {}
for f in sorted(glob.glob(str(a.dir / "**" / "S_*.npy"), recursive=True)):
    el, tag = os.path.basename(f)[2:-4].split("_", 1); ops.setdefault(tag, {})[el] = np.load(f)
tags = [t for t in ops if t != a.reference_tag]
report = {"case_id": a.case_id, "q": int(port.q), "lowfreq_mode_count": int(norms.lowfreq_vector.shape[1]), "tags": tags, "gates": {}}
for el in ("TET4", "TET10"):
    pair = {}
    for i in range(len(tags)):
        for j in range(i + 1, len(tags)):
            if el in ops[tags[i]] and el in ops[tags[j]]: pair[f"{tags[i]}|{tags[j]}"] = t10.compare_operators(ops[tags[i]][el], ops[tags[j]][el], norms)
    g = {"pairwise": pair}
    for key in ("lowfreq_frobenius", "whitened_frobenius", "raw_frobenius"): g["max_" + key] = max((v[key] for v in pair.values()), default=None)
    g["reproducibility_status"] = None if g["max_lowfreq_frobenius"] is None else ("PASS" if g["max_lowfreq_frobenius"] <= a.repro_threshold else "FAIL")
    if a.reference_tag in ops and el in ops[a.reference_tag]:
        conv = {t: t10.compare_operators(ops[t][el], ops[a.reference_tag][el], norms) for t in tags if el in ops[t]}
        g["convergence_vs_reference"] = conv; g["max_lowfreq_vs_reference"] = max(v["lowfreq_frobenius"] for v in conv.values()) if conv else None
        g["convergence_status"] = None if g["max_lowfreq_vs_reference"] is None else ("PASS" if g["max_lowfreq_vs_reference"] <= a.conv_threshold else "FAIL")
    if el == "TET10":
        g["tet4_vs_tet10_same_mesh"] = {t: t10.compare_operators(ops[t]["TET4"], ops[t]["TET10"], norms) for t in ops if "TET4" in ops[t] and "TET10" in ops[t]}
    report["gates"][el] = g
(a.dir / "TET10_LABEL_GATES.json").write_text(json.dumps(report, indent=1, default=float))
lines = [f"# Tet10 label gates: {a.case_id}", "", f"- q = {report['q']}, low-frequency vector modes (wavelength >= {a.wavelength_cutoff}) = {report['lowfreq_mode_count']}", f"- meshes: {tags}", ""]
fmt = lambda v: "n/a" if v is None else f"{v:.4f}"
for el, g in report["gates"].items():
    if g["max_lowfreq_frobenius"] is None and "convergence_status" not in g:
        continue
    lines += [f"## {el}", "", f"- reproducibility max pairwise lowfreq / whitened / raw: {fmt(g['max_lowfreq_frobenius'])} / {fmt(g['max_whitened_frobenius'])} / {fmt(g['max_raw_frobenius'])} -> **{g['reproducibility_status']}**"]
    if "convergence_status" in g: lines.append(f"- convergence vs {a.reference_tag} max lowfreq: {fmt(g['max_lowfreq_vs_reference'])} -> **{g['convergence_status']}**")
    for t, v in g.get("tet4_vs_tet10_same_mesh", {}).items(): lines.append(f"- {t}: Tet4 vs Tet10 lowfreq {v['lowfreq_frobenius']:.4f} (energy p50 {v['lowfreq_energy_ratio_p50']:+.4f}, p95abs {v['lowfreq_energy_ratio_p95abs']:.4f})")
    lines.append("")
(a.dir / "TET10_LABEL_GATES.md").write_text("\n".join(lines)); print("\n".join(lines))
