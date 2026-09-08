#!/usr/bin/env python3
"""Generate a boundary-conforming fixed-port mesh and verify the selection property of P."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
P = "pred777h_full_cube_single_cell_mesh_graph_fixed_port_schur_v1"
from importlib import import_module
adapter = import_module(f"{P}.cgal_mesh3_adapter"); fpa = import_module(f"{P}.fixed_port_adapter"); fcb = import_module(f"{P}.full_cube_backend")
snap = import_module(f"{P}.mesher_neutral_snapshot"); cpm = import_module(f"{P}.conforming_port_mesh")

ap = argparse.ArgumentParser()
ap.add_argument("--case-id", required=True); ap.add_argument("--off", type=Path, required=True); ap.add_argument("--geometry-manifest", type=Path, required=True)
ap.add_argument("--output-dir", type=Path, required=True); ap.add_argument("--size-max", type=float, default=0.05); ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--algorithm3d", type=int, default=10); ap.add_argument("--optimize-passes", type=int, default=3); ap.add_argument("--port-refine", type=int, default=1); ap.add_argument("--no-netgen", action="store_true"); ap.add_argument("--cavity-remesh", type=float, default=None); ap.add_argument("--no-cavity-smooth", action="store_true")
a = ap.parse_args(); a.output_dir.mkdir(parents=True, exist_ok=True)
_, spec, geom = snap.load_geometry_manifest(a.geometry_manifest)
charts, *_ = fcb.compile_full_cube_geometry_inputs(spec); layout = fcb.build_full_cube_trace_layout(spec, charts=charts)
active = tuple(sorted(str(c.source_id) for c in charts.charts))
res = cpm.build_conforming_port_mesh(off_path=a.off, layout=layout, geometry=geom, active_ports=active, output_path=a.output_dir / "mesh.mesh",
                                     interior_size_max=a.size_max, algorithm3d=a.algorithm3d, optimize_passes=a.optimize_passes, seed=a.seed, port_refine_levels=a.port_refine, netgen_optimize=not a.no_netgen, cavity_remesh_size=a.cavity_remesh, cavity_smooth=not a.no_cavity_smooth)
rep = {k: (str(v) if isinstance(v, Path) else v) for k, v in res.__dict__.items()}; rep["active_ports"] = active
mp = {"domain_contract": "COHERENT_TRIANGULATED_CLOSED_VOLUME_BOUNDING_POLYHEDRAL_COMPLEX", "optimization_enabled": False, "semantic_plane_tolerance": 1e-10, "mesher": "gmsh_conforming_port"}
t = time.perf_counter()
art = adapter.mesh_artifact_from_cgal_medit(res.mesh_path, geometry=geom, mesher_version="gmsh-4.15", mesher_parameters=mp, plane_tolerance=1e-10, source_geometry_hash=spec.spec_hash)
port = fpa.compile_fixed_port(art, spec, layout)
S = port.scalar_fine_to_carrier.tocsr()
w = S.data; k = a.port_refine
dyadic = np.allclose(w * (2 ** k), np.round(w * (2 ** k)), atol=1e-12)
sel = dyadic and float(port.certificates["maximum_affine_reproduction_error"]) < 1e-12
topo = adapter.audit_cgal_mesh_artifact(art, geom)["topology"]
rep.update({"fine_port_nodes": int(S.shape[0]), "carrier_scalar_nodes": int(S.shape[1]), "P_is_exact_nested": bool(sel), "P_weights_dyadic": bool(dyadic), "affine_reproduction_error": float(port.certificates["maximum_affine_reproduction_error"]), "P_nnz": int(S.nnz),
            "max_offdiag_weight": float(np.max(np.abs(S.data[S.data < 0.999999])) if np.any(S.data < 0.999999) else 0.0), "q": int(port.q),
            "zero_support_columns": int(port.certificates["zero_support_active_carrier_column_count"]),
            "nonmanifold_boundary_edge_count": int(topo["nonmanifold_boundary_edge_count"]), "boundary_component_count": int(topo["boundary_component_count"]),
            "adapter_seconds": round(time.perf_counter() - t, 1)})
(a.output_dir / "CONFORMING_MESH_REPORT.json").write_text(json.dumps(rep, indent=1, default=float)); print(json.dumps(rep, indent=1, default=float))
