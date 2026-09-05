#!/usr/bin/env python3
"""Production label command: geometry manifest + coherent polyhedral OFF -> boundary-conforming mesh -> Tet10 exact
fixed-port Schur -> create-only label directory with SHA-256 receipts.  Solver: MKL Pardiso (default; SuperLU fallback below --pardiso-above internal
DOFs, Pardiso (MKL) above.  No gate is applied here; gates run on the six-case panel (run_tet10_label_gates.py)."""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np, scipy.sparse as sp
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
P = "pred777h_full_cube_single_cell_mesh_graph_fixed_port_schur_v1"
from importlib import import_module
adapter = import_module(f"{P}.cgal_mesh3_adapter"); fpa = import_module(f"{P}.fixed_port_adapter"); fcb = import_module(f"{P}.full_cube_backend")
snap = import_module(f"{P}.mesher_neutral_snapshot"); t10 = import_module(f"{P}.tet10_label"); cpm = import_module(f"{P}.conforming_port_mesh"); tcc = import_module(f"{P}.teacher_contract_calibration")
import_module(f"{P}.frozen_backend").activate_frozen_backend()
from cctpms.fem.tet10 import assemble_global_tet10_stiffness

STRATUM_PRESETS = {  # label resolution per population stratum (protocol decision, step 1/2 evidence)
    "full": {"port_refine": 1, "size_max": 0.05}, "thin": {"port_refine": 1, "size_max": 0.05}, "near_empty": {"port_refine": 1, "size_max": 0.05},
}

def sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def git_head() -> str:
    try: return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception: return "unknown"

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-id", required=True); ap.add_argument("--geometry-manifest", type=Path, required=True); ap.add_argument("--off", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True); ap.add_argument("--stratum", choices=sorted(STRATUM_PRESETS), default=None)
    ap.add_argument("--port-refine", type=int); ap.add_argument("--size-max", type=float); ap.add_argument("--algorithm3d", type=int, default=10)
    ap.add_argument("--pardiso-above", type=int, default=0, help="internal dof above which MKL Pardiso is used (default 0: always, SuperLU fallback if pypardiso is missing)"); ap.add_argument("--workers", type=int, default=8); ap.add_argument("--chunk", type=int, default=1024)
    a = ap.parse_args()
    preset = STRATUM_PRESETS[a.stratum] if a.stratum else {}
    k = a.port_refine if a.port_refine is not None else preset.get("port_refine", 2); size = a.size_max if a.size_max is not None else preset.get("size_max", 0.035)
    out = a.output_dir.resolve()
    if out.exists(): raise FileExistsError(f"create-only label directory exists: {out}")
    out.mkdir(parents=True); t_all = time.perf_counter(); receipt = {"schema": "pred777h_tet10_conforming_fixed_port_label_v1", "case_id": a.case_id, "code_commit": git_head(),
        "inputs": {"geometry_manifest": {"path": str(a.geometry_manifest), "sha256": sha(a.geometry_manifest)}, "surface_off": {"path": str(a.off), "sha256": sha(a.off)}},
        "resolution": {"stratum": a.stratum, "port_refine_levels": k, "interior_size_max": size, "algorithm3d": a.algorithm3d}, "element": "TET10_P2_straight_sided", "material": {"E": 1.0, "nu": 0.3}}
    _, spec, geom = snap.load_geometry_manifest(a.geometry_manifest); charts, *_ = fcb.compile_full_cube_geometry_inputs(spec); layout = fcb.build_full_cube_trace_layout(spec, charts=charts)
    active = tuple(sorted(str(c.source_id) for c in charts.charts))
    t0 = time.perf_counter()
    mesh = cpm.build_conforming_port_mesh(off_path=a.off, layout=layout, geometry=geom, active_ports=active, output_path=out / "mesh.mesh", interior_size_max=size, algorithm3d=a.algorithm3d, optimize_passes=3, port_refine_levels=k)
    receipt["mesh"] = {kk: (str(v) if isinstance(v, Path) else v) for kk, v in mesh.__dict__.items()}; receipt["mesh"]["sha256"] = sha(out / "mesh.mesh"); receipt["timing"] = {"mesh": time.perf_counter() - t0}
    mp = {"domain_contract": "COHERENT_TRIANGULATED_CLOSED_VOLUME_BOUNDING_POLYHEDRAL_COMPLEX", "optimization_enabled": False, "semantic_plane_tolerance": 1e-10, "mesher": "gmsh_conforming_port", "port_refine_levels": k, "interior_size_max": size}
    t0 = time.perf_counter()
    art = adapter.mesh_artifact_from_cgal_medit(out / "mesh.mesh", geometry=geom, mesher_version="gmsh-4.15", mesher_parameters=mp, plane_tolerance=1e-10, source_geometry_hash=spec.spec_hash)
    port = fpa.compile_fixed_port(art, spec, layout); topo = adapter.audit_cgal_mesh_artifact(art, geom)["topology"]
    Sc = port.scalar_fine_to_carrier.tocsr(); dyadic = bool(np.allclose(Sc.data * 2 ** k, np.round(Sc.data * 2 ** k), atol=1e-12))
    receipt["fixed_port"] = {"q": int(port.q), "carrier_scalar_nodes": int(Sc.shape[1]), "fine_port_nodes": int(Sc.shape[0]), "P_weights_dyadic": dyadic,
        "affine_reproduction_error": float(port.certificates["maximum_affine_reproduction_error"]), "zero_support_columns": int(port.certificates["zero_support_active_carrier_column_count"]),
        "layout_semantic_signature": layout.semantic_signature, "active_ports": list(active), "deterministic_replay_sha256": port.certificates["deterministic_replay_sha256"]}
    receipt["topology"] = {kk: (int(v) if isinstance(v, (int, np.integer)) else v) for kk, v in topo.items() if isinstance(v, (int, np.integer, float, str, bool))}
    if not dyadic or receipt["fixed_port"]["affine_reproduction_error"] > 1e-12 or receipt["fixed_port"]["zero_support_columns"] != 0:
        raise RuntimeError("conforming fixed port is not exactly nested")
    receipt["timing"]["adapter_and_port"] = time.perf_counter() - t0
    t0 = time.perf_counter(); promo = t10.promote_mesh_artifact_to_tet10(art, port)
    K = assemble_global_tet10_stiffness(promo.node_coordinates, promo.elements, E=1.0, nu=0.3, workers=a.workers).tocsr(); n10 = len(promo.node_coordinates)
    vec = sp.kron(promo.scalar_prolongation, sp.eye(3, format="csr"), format="csr")
    constraint, _ = tcc.build_fixed_carrier_constraint(node_count=n10, port_node_indices=promo.port_node_indices, vector_prolongation=vec)
    Kc = (constraint.T @ K @ constraint).tocsr(); Kc = (0.5 * (Kc + Kc.T)).tocsr(); q = port.q; ni = Kc.shape[0] - q
    receipt["tet10"] = {"nodes": int(n10), "elements": int(len(promo.elements)), "fine_dof": int(3 * n10), "internal_dof": int(ni)}; receipt["timing"]["tet10_assembly"] = time.perf_counter() - t0
    Kqq = Kc[:q, :q].toarray(); Kqi = Kc[:q, q:].tocsr(); Kii = Kc[q:, q:].tocsr(); t0 = time.perf_counter()
    if ni > a.pardiso_above and t10.pardiso_available():
        import pypardiso
        solver = pypardiso.PyPardisoSolver(); solver.set_iparm(1, 1); solver.set_iparm(2, 3); solver.factorize(Kii)
        S = Kqq.copy(); B = Kqi.T.tocsc()
        for c0 in range(0, q, a.chunk): S[:, c0:c0 + a.chunk] -= Kqi @ solver.solve(Kii, B[:, c0:c0 + a.chunk].toarray())
        solver.free_memory(everything=True); receipt["solver"] = "mkl_pardiso"
    else:
        from scipy.sparse.linalg import splu
        lu = splu(Kii.tocsc(), permc_spec="COLAMD"); S = Kqq.copy(); B = Kqi.T.tocsc()
        for c0 in range(0, q, a.chunk): S[:, c0:c0 + a.chunk] -= Kqi @ lu.solve(B[:, c0:c0 + a.chunk].toarray())
        receipt["solver"] = "superlu"
    S = 0.5 * (S + S.T); receipt["timing"]["schur"] = time.perf_counter() - t0
    rig = port.rigid_basis; rr = float(np.linalg.norm(S @ rig) / np.linalg.norm(S)); ev = np.linalg.eigvalsh(S)
    receipt["schur"] = {"q": int(q), "rigid_residual_relative": rr, "min_eigenvalue": float(ev[0]), "max_eigenvalue": float(ev[-1]), "symmetric": True}
    np.savez(out / "FIXED_PORT_SCHUR_TET10.npz", schur=S, carrier_ids=np.asarray(port.active_global_carrier_ids), carrier_coordinates=port.carrier_coordinates, rigid_basis=rig, active_ports=np.asarray(active))
    receipt["outputs"] = {"schur_npz": {"path": "FIXED_PORT_SCHUR_TET10.npz", "sha256": sha(out / "FIXED_PORT_SCHUR_TET10.npz")}, "mesh": {"path": "mesh.mesh", "sha256": receipt["mesh"]["sha256"]}}
    receipt["timing"]["total"] = time.perf_counter() - t_all; receipt["status"] = "PASS" if rr < 1e-8 and ev[0] > -1e-8 * ev[-1] else "FAIL"
    (out / "LABEL_RECEIPT.json").write_text(json.dumps(receipt, indent=1, default=float))
    md = [f"# Tet10 conforming fixed-port label: {a.case_id}", "", f"- status: **{receipt['status']}**", f"- resolution: k={k}, interior size {size}, stratum {a.stratum}",
          f"- mesh: {mesh.node_count} Tet4 nodes / {mesh.tet_count} tets, Tet10 {n10} nodes ({3*n10} dof), min dihedral {mesh.min_dihedral_degrees:.2f} deg, material volume {mesh.material_volume:.6f}",
          f"- fixed port: q={q}, P dyadic={dyadic}, affine error {receipt['fixed_port']['affine_reproduction_error']:.1e}, zero-support columns {receipt['fixed_port']['zero_support_columns']}",
          f"- Schur: rigid residual {rr:.1e}, eig [{ev[0]:.3e}, {ev[-1]:.3e}], solver {receipt['solver']}", f"- timing: mesh {receipt['timing']['mesh']:.0f}s, assembly {receipt['timing']['tet10_assembly']:.0f}s, schur {receipt['timing']['schur']:.0f}s, total {receipt['timing']['total']:.0f}s"]
    (out / "LABEL_RECEIPT.md").write_text("\n".join(md) + "\n"); print("\n".join(md)); return 0 if receipt["status"] == "PASS" else 1

if __name__ == "__main__":
    raise SystemExit(main())
