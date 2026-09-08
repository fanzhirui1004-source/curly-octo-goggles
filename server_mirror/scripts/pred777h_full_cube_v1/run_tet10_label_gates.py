#!/usr/bin/env python3
"""Tet10 label gates: mesher reproducibility and A0->A1 convergence in carrier-weighted norms."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
P = "pred777h_full_cube_single_cell_mesh_graph_fixed_port_schur_v1"
from importlib import import_module
adapter = import_module(f"{P}.cgal_mesh3_adapter"); fpa = import_module(f"{P}.fixed_port_adapter"); fcb = import_module(f"{P}.full_cube_backend")
snap = import_module(f"{P}.mesher_neutral_snapshot"); t10 = import_module(f"{P}.tet10_label")

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-id", required=True); ap.add_argument("--geometry-manifest", type=Path, required=True)
    ap.add_argument("--mesh", action="append", required=True, help="tag=path/to/mesh.mesh (reproducibility set)")
    ap.add_argument("--reference-mesh", help="tag=path (finer level for the convergence gate)")
    ap.add_argument("--output-dir", type=Path, required=True); ap.add_argument("--elements", default="TET4,TET10")
    ap.add_argument("--wavelength-cutoff", type=float, default=0.25, help="keep carrier modes with wavelength >= this (4H for H=1/16)")
    ap.add_argument("--repro-threshold", type=float, default=0.005); ap.add_argument("--conv-threshold", type=float, default=0.03)
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--pardiso-above", type=int, default=0, help="internal dof above which MKL Pardiso replaces SuperLU (default 0: always; SuperLU fallback if pypardiso is missing)")
    a = ap.parse_args(); out = a.output_dir; out.mkdir(parents=True, exist_ok=True)
    _, spec, geom = snap.load_geometry_manifest(a.geometry_manifest)
    charts, *_ = fcb.compile_full_cube_geometry_inputs(spec); layout = fcb.build_full_cube_trace_layout(spec, charts=charts)
    mp = {"domain_contract": "COHERENT_TRIANGULATED_CLOSED_VOLUME_BOUNDING_POLYHEDRAL_COMPLEX", "optimization_enabled": False, "semantic_plane_tolerance": 1e-10}
    elements = [e.strip().upper() for e in a.elements.split(",")]
    entries = [m.split("=", 1) for m in a.mesh]; ref = a.reference_mesh.split("=", 1) if a.reference_mesh else None
    ops: dict[str, dict[str, np.ndarray]] = {}; meta: dict[str, dict] = {}; norms = None; port0 = None
    for tag, path in entries + ([ref] if ref else []):
        t0 = time.perf_counter(); rec = {"path": path}
        try:
            art = adapter.mesh_artifact_from_cgal_medit(Path(path), geometry=geom, mesher_version="5.4", mesher_parameters=dict(mp, tag=tag), plane_tolerance=1e-10, source_geometry_hash=spec.spec_hash)
            port = fpa.compile_fixed_port(art, spec, layout)
            if port0 is None:
                port0 = port; norms = t10.build_carrier_norms(layout, port, wavelength_cutoff=a.wavelength_cutoff)
            elif tuple(port.active_global_carrier_ids) != tuple(port0.active_global_carrier_ids):
                raise ValueError("carrier identity differs between meshes of the same geometry")
            rec.update({"tet4_nodes": int(art.node_count), "tets": int(len(art.tet_connectivity)), "q": int(port.q),
                        "nonmanifold_boundary_edge_count": int(adapter.audit_cgal_mesh_artifact(art, geom)["topology"]["nonmanifold_boundary_edge_count"])})
            ops[tag] = {}
            for el in elements:
                try:
                    S, timing = t10.dense_fixed_port_schur(art, port, element=el, workers=a.workers, column_chunk_size=a.chunk, pardiso_above=a.pardiso_above)
                    np.save(out / f"S_{el}_{tag}.npy", S); ops[tag][el] = S; rec[f"{el}_timing"] = timing
                except Exception as e:
                    rec[f"{el}_error"] = f"{type(e).__name__}: {str(e)[:200]}"
            rec["status"] = "OK"
        except Exception as e:
            rec["status"] = "ERROR"; rec["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        rec["seconds"] = round(time.perf_counter() - t0, 1); meta[tag] = rec; print(tag, json.dumps({k: v for k, v in rec.items() if "timing" not in k}), flush=True)
    report = {"case_id": a.case_id, "wavelength_cutoff": a.wavelength_cutoff, "lowfreq_mode_count": int(norms.lowfreq_vector.shape[1]) if norms else None,
              "q": int(port0.q) if port0 else None, "meshes": meta, "gates": {}}
    tags = [t for t, _ in entries if t in ops]
    for el in elements:
        # reproducibility: all pairs among the set
        pair = {}
        for i in range(len(tags)):
            for j in range(i + 1, len(tags)):
                if el in ops[tags[i]] and el in ops[tags[j]]:
                    pair[f"{tags[i]}|{tags[j]}"] = t10.compare_operators(ops[tags[i]][el], ops[tags[j]][el], norms)
        worst = max((v["lowfreq_frobenius"] for v in pair.values()), default=None)
        worst_w = max((v["whitened_frobenius"] for v in pair.values()), default=None)
        worst_raw = max((v["raw_frobenius"] for v in pair.values()), default=None)
        g = {"pairwise": pair, "max_lowfreq": worst, "max_whitened": worst_w, "max_raw": worst_raw,
             "reproducibility_status": None if worst is None else ("PASS" if worst <= a.repro_threshold else "FAIL")}
        if ref and ref[0] in ops and el in ops[ref[0]]:
            conv = {t: t10.compare_operators(ops[t][el], ops[ref[0]][el], norms) for t in tags if el in ops[t]}
            cw = max(v["lowfreq_frobenius"] for v in conv.values()) if conv else None
            g["convergence_vs_reference"] = conv; g["max_lowfreq_vs_reference"] = cw
            g["convergence_status"] = None if cw is None else ("PASS" if cw <= a.conv_threshold else "FAIL")
        if el == "TET10":
            g["tet4_vs_tet10_same_mesh"] = {t: t10.compare_operators(ops[t]["TET4"], ops[t]["TET10"], norms) for t in tags if "TET4" in ops[t] and "TET10" in ops[t]}
        report["gates"][el] = g
    (out / "TET10_LABEL_GATES.json").write_text(json.dumps(report, indent=1, default=float))
    lines = [f"# Tet10 label gates: {a.case_id}", "", f"- q = {report['q']}, low-frequency modes (wavelength >= {a.wavelength_cutoff}) = {report['lowfreq_mode_count']}", ""]
    for el, g in report["gates"].items():
        lines += [f"## {el}", "", f"- reproducibility (max pairwise, lowfreq / whitened / raw): {g['max_lowfreq']} / {g['max_whitened']} / {g['max_raw']} -> **{g['reproducibility_status']}**"]
        if "convergence_status" in g: lines.append(f"- convergence vs reference (max lowfreq): {g['max_lowfreq_vs_reference']} -> **{g['convergence_status']}**")
        if "tet4_vs_tet10_same_mesh" in g:
            for t, v in g["tet4_vs_tet10_same_mesh"].items(): lines.append(f"- {t}: Tet4 vs Tet10 lowfreq {v['lowfreq_frobenius']:.4f}, energy-ratio p50 {v['lowfreq_energy_ratio_p50']:+.4f}, p95abs {v['lowfreq_energy_ratio_p95abs']:.4f}")
        lines.append("")
    (out / "TET10_LABEL_GATES.md").write_text("\n".join(lines)); print("\n".join(lines)); return 0

if __name__ == "__main__":
    raise SystemExit(main())
