#!/usr/bin/env python3
"""Production command for one sheet-TPMS fixed-port label (true geometry, no port collars).

Geometry -> carrier-conforming surface -> Gmsh Tet4 -> Tet10 -> fixed-port Schur (MKL Pardiso), written to a create-only
directory together with the carrier norms the training loss needs and a SHA-256 receipt.

Fail-closed: an empty cell produces no label (status EMPTY); a mesh whose port support escapes the geometric active set,
a rigid-body residual above 1e-10, or a negative eigenvalue below -1e-8 * lambda_max is an error.

Reproducibility: the geometry (surface and mesh) is byte-identical across repeats and across thread counts. The Schur
operator is byte-identical at a fixed thread count and agrees to 2e-14 across thread counts (MKL summation order); set
MKL_CBWR=COMPATIBLE for byte-identical operators across machines. The receipt records the execution settings.
"""
from __future__ import annotations

import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
P = "pred777h_full_cube_single_cell_mesh_graph_fixed_port_schur_v1"
from importlib import import_module  # noqa: E402

fcb = import_module(f"{P}.full_cube_backend"); snap = import_module(f"{P}.mesher_neutral_snapshot")
sss = import_module(f"{P}.sheet_solid_surface"); pipe = import_module(f"{P}.sheet_label_pipeline")
SheetSolidContract = import_module(f"{P}.sheet_contract").SheetSolidContract

# surface grid, sheet remesh target, interior element size
RESOLUTION_PRESETS = {
    "reference": {"n_per_unit": 96, "remesh_size": 0.015, "size_max": 0.03},
    "production": {"n_per_unit": 64, "remesh_size": 0.02, "size_max": 0.04},
    "fast": {"n_per_unit": 48, "remesh_size": 0.03, "size_max": 0.05},
}
CARRIER_N = 32          # carrier P1 grid per face (decision of record, 2026-09-03)
EMPTY_VOLUME = 1.0e-6   # a cell below this material volume carries no label


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-id", required=True)
    ap.add_argument("--geometry-manifest", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--resolution", choices=sorted(RESOLUTION_PRESETS), default="production")
    ap.add_argument("--n-per-unit", type=int); ap.add_argument("--remesh-size", type=float); ap.add_argument("--size-max", type=float)
    ap.add_argument("--carrier-n", type=int, default=CARRIER_N)
    ap.add_argument("--algorithm3d", type=int, default=10, help="Gmsh 3D algorithm: 10 HXT (algorithm of record, deterministic at the pinned thread count), 1 legacy Delaunay")
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--sliver-smooth", action="store_true", help="boundary-locked sliver perturbation after Gmsh (opt-in diagnostic: it raised lambda_max on pop_uncut_0658, see sheet_contract)")
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--memory-budget-dir", type=Path, help="directory of the cooperative memory budget shared by the jobs of this machine (see MemoryBudget)")
    ap.add_argument("--memory-budget-gb", type=float, default=110.0, help="budget in GB for the Schur stages in flight (a 128 GiB cgroup: 110)")
    a = ap.parse_args()

    preset = dict(RESOLUTION_PRESETS[a.resolution])
    for key in ("n_per_unit", "remesh_size", "size_max"):
        if getattr(a, key) is not None:
            preset[key] = getattr(a, key)
    out = a.output_dir
    out.mkdir(parents=True, exist_ok=False)      # create-only: never overwrite a published label
    t_all = time.perf_counter()
    receipt: dict = {"case_id": a.case_id, "geometry_manifest": str(a.geometry_manifest), "git_head": git_head(),
                     "resolution": {"preset": a.resolution, **preset, "carrier_n": a.carrier_n, "algorithm3d": a.algorithm3d},
                     "execution": {"workers": a.workers, "threads": a.threads, "column_chunk": a.chunk,
                                   "omp_num_threads": os.environ.get("OMP_NUM_THREADS"), "mkl_cbwr": os.environ.get("MKL_CBWR")},
                     "contract": SheetSolidContract().as_dict(), "timing": {}}

    _, spec, geom = snap.load_geometry_manifest(a.geometry_manifest)
    degenerate = pipe.cut_contains_a_cell_edge(spec)
    if degenerate is not None:
        receipt["status"] = "GEOMETRY_DEGENERATE"; receipt["geometry_degeneracy"] = degenerate
        (out / "LABEL_RECEIPT.json").write_text(json.dumps(receipt, indent=1, default=float))
        (out / "LABEL_RECEIPT.md").write_text(f"# Sheet label: {a.case_id}\n\n- status: **GEOMETRY_DEGENERATE**: {degenerate['reason']}\n- remedy: {degenerate['remedy']}\n")
        print(f"{a.case_id}: GEOMETRY_DEGENERATE ({degenerate['reason']})"); return 0
    charts, *_ = fcb.compile_full_cube_geometry_inputs(spec)
    layout = pipe.build_carrier_layout(charts, spec, carrier_n=a.carrier_n)
    receipt["carrier_triangulation"] = pipe.CARRIER_TRIANGULATION
    receipt["cut_carrier_merge"] = layout.diagnostics.get("cut_carrier_merge")
    receipt["cut_face_trace"] = pipe.write_cut_face_trace(out / "CUT_FACE_TRACE.npz", layout, spec)
    receipt["cut_carrier_residual_strip"] = pipe.cc.cut_carrier_gate(receipt["cut_carrier_merge"])   # diagnostic only (harmless, see cut_carrier)
    receipt["geometry"] = {"spec_hash": spec.spec_hash, "tau_corners": [float(t) for t in geom.tau_corners],
                           "cut_plane": None if geom.cut_plane is None else [float(v) for v in geom.cut_plane]}

    # ---- empty-cell screen (exact level set, no meshing) ----
    t0 = time.perf_counter(); n = 160; xs = (np.arange(n) + 0.5) / n
    grid = np.stack(np.meshgrid(xs, xs, xs, indexing="ij"), axis=-1).reshape(-1, 3)
    inside = sss.sheet_level(geom, grid) < 0.0
    for _, nn, dd in sss.clip_planes(geom):
        inside &= (grid @ nn - dd) <= 0.0
    volume = float(inside.mean()); receipt["geometry"]["material_volume_estimate"] = volume
    receipt["timing"]["empty_screen"] = time.perf_counter() - t0
    if volume < EMPTY_VOLUME:
        receipt["status"] = "EMPTY"
        (out / "LABEL_RECEIPT.json").write_text(json.dumps(receipt, indent=1, default=float))
        (out / "LABEL_RECEIPT.md").write_text(f"# Sheet label: {a.case_id}\n\n- status: **EMPTY** (material volume {volume:.2e}); no operator is produced.\n")
        print(f"{a.case_id}: EMPTY (volume {volume:.2e})", flush=True)
        return 0

    # ---- surface and volume mesh ----
    t0 = time.perf_counter()
    surf = sss.build_sheet_solid_surface(geom, layout, n_per_unit=preset["n_per_unit"], remesh_size=preset["remesh_size"], carrier_n=a.carrier_n)
    receipt["surface"] = {k: v for k, v in surf.report.items() if k not in ("caps", "non_manifold_examples", "surface_quality_by_label", "remesh_attempts")}
    if surf.report["watertight"]["non_two_manifold_edges"]:
        raise SystemExit(f"{a.case_id}: surface is not watertight ({surf.report['watertight']})")
    if surf.report["self_intersecting_triangles"]:
        raise SystemExit(f"{a.case_id}: surface self-intersects ({surf.report.get('self_intersection_examples')})")
    sss.write_off(out / "SHEET_SOLID_SURFACE.off", surf.V, surf.F)
    try:
        mesh = sss.mesh_sheet_solid(surf, out / "mesh.mesh", interior_size_max=preset["size_max"], algorithm3d=a.algorithm3d, threads=a.threads, sliver_smooth=a.sliver_smooth)
    except RuntimeError as exc:
        raise SystemExit(f"{a.case_id}: MESH_FAIL {exc}")
    receipt["mesh"] = {k: (str(v) if isinstance(v, Path) else v) for k, v in mesh.__dict__.items()}
    receipt["timing"]["surface_and_mesh"] = time.perf_counter() - t0

    # ---- fixed port on the geometric active set (P is never modified) ----
    try:
        stage = pipe.compile_port(out / "mesh.mesh", geom, spec, layout, carrier_n=a.carrier_n)
    except RuntimeError as exc:
        raise SystemExit(f"{a.case_id}: {exc}")
    port, active, check = stage.port, stage.active, stage.check
    receipt["fixed_port"] = stage.summary; receipt["timing"]["fixed_port"] = stage.seconds

    # ---- Tet10 fixed-port Schur with the operator gates, under the machine's memory budget ----
    budget = None
    if a.memory_budget_dir is not None:
        tet10_dof = 3 * (int(receipt["mesh"]["node_count"]) + int(round(1.2 * int(receipt["mesh"]["tet_count"]))))
        budget = pipe.MemoryBudget(a.memory_budget_dir, a.memory_budget_gb); need = budget.estimate_gb(tet10_dof)
        receipt["timing"]["memory_budget_wait"] = budget.reserve(need); receipt["execution"]["memory_reserved_gb"] = need
    try:
        sch = pipe.schur_label(stage, workers=a.workers, column_chunk_size=a.chunk)
    except RuntimeError as exc:
        if budget is not None:
            budget.release()
        raise SystemExit(f"{a.case_id}: {exc}")
    finally:
        if budget is not None:
            budget.release()
    S_active, rigid, eig, rigid_residual = sch.schur_active, sch.rigid_active, sch.eigenvalues, sch.rigid_residual
    receipt["schur_timing"] = sch.timing; receipt["timing"]["schur"] = sch.seconds
    receipt["schur"] = {"q_active": int(S_active.shape[0]), "rigid_residual_relative": rigid_residual,
                        "min_eigenvalue": float(eig[0]), "max_eigenvalue": float(eig[-1]), "inactive_block_max": sch.inactive_block_max,
                        "top_mode_mass_on_4_nodes": sch.top_mode_mass_on_4_nodes, "top_mode_nodes_above_1pct": sch.top_mode_nodes_above_1pct}

    # ---- support mask and numerical rank (decision of record 2026-09-05: the null space is a lower bound, no gap) ----
    # a carrier node the mesh trace never loads has an exactly zero row; the rank at two declared thresholds is shipped
    # because there is no spectral gap and any single threshold is a convention (see STEP6 report, section 8)
    n_nodes = S_active.shape[0] // 3
    row_norm = np.linalg.norm(S_active.reshape(n_nodes, 3, -1), axis=(1, 2)); support_mask = row_norm > 1e-12 * max(float(row_norm.max()), 1e-300)
    rank_1e14 = int((eig > 1e-14 * eig[-1]).sum()); rank_1e10 = int((eig > 1e-10 * eig[-1]).sum())
    receipt["schur"]["support"] = {"supported_nodes": int(support_mask.sum()), "unsupported_nodes": int((~support_mask).sum()),
                                  "rank_at_1e-14": rank_1e14, "rank_at_1e-10": rank_1e10, "null_dim_lower_bound": int(6 + 3 * (~support_mask).sum()),
                                  "null_dim_at_1e-14": int(S_active.shape[0] - rank_1e14), "null_dim_at_1e-10": int(S_active.shape[0] - rank_1e10)}
    # ---- carrier norms for the training loss ----
    mass, lap = pipe.carrier_norms(layout, port, active)
    ids = np.asarray(port.active_global_carrier_ids)[active]
    coords = np.asarray(port.carrier_coordinates)[active]
    membership = np.asarray([",".join(m) for m in port.carrier_port_membership], dtype=object)[active].astype(str)
    # storage (decision of record 2026-09-04): the symmetric operator is stored as its upper triangle in float32
    # (q(q+1)/2 entries; 64 MB at q = 5643 instead of 255 MB), the gates above were evaluated in float64.
    # pipe.load_schur() rebuilds the full symmetric float64 matrix.
    iu = np.triu_indices(S_active.shape[0])
    from scipy.sparse import coo_matrix
    mass_c = coo_matrix(mass); lap_c = coo_matrix(lap)       # P1 matrices are sparse (about 7 entries per row): stored as COO
    np.savez(out / "FIXED_PORT_SCHUR_TET10.npz", schur_upper_f32=S_active[iu].astype(np.float32), schur_format=np.asarray(["upper_triangle_float32"]),
             q=np.asarray([S_active.shape[0]]), carrier_ids=ids, carrier_coordinates=coords,
             carrier_port_membership=membership, rigid_basis=rigid, active_ports=np.asarray(list(port.active_global_port_ids)),
             carrier_mass_coo=np.vstack([mass_c.row, mass_c.col]).astype(np.int32), carrier_mass_values=mass_c.data,
             carrier_laplacian_coo=np.vstack([lap_c.row, lap_c.col]).astype(np.int32), carrier_laplacian_values=lap_c.data,
             carrier_scalar_count=np.asarray([mass.shape[0]]), material_volume=np.asarray([mesh.material_volume]),
             support_mask=support_mask, numerical_rank=np.asarray([rank_1e14, rank_1e10]), rank_thresholds=np.asarray([1e-14, 1e-10]))
    receipt["schur"]["storage"] = {"format": "upper_triangle_float32", "entries": int(len(iu[0])), "bytes": int(4 * len(iu[0])),
                                   "float32_rounding_relative": float(np.abs(S_active[iu] - S_active[iu].astype(np.float32)).max() / np.abs(S_active).max())}

    receipt["outputs"] = {name: {"path": name, "sha256": sha256(out / name)} for name in ("mesh.mesh", "SHEET_SOLID_SURFACE.off", "FIXED_PORT_SCHUR_TET10.npz")}
    receipt["timing"]["total"] = time.perf_counter() - t_all
    receipt["status"] = "PASS"
    (out / "LABEL_RECEIPT.json").write_text(json.dumps(receipt, indent=1, default=float))
    md = [f"# Sheet label: {a.case_id}", "", f"- status: **PASS**",
          f"- geometry: rho = {volume:.5f}, tau in [{min(receipt['geometry']['tau_corners']):.4f}, {max(receipt['geometry']['tau_corners']):.4f}], cut = {receipt['geometry']['cut_plane']}",
          f"- resolution: {a.resolution} (surface 1/{preset['n_per_unit']}, remesh {preset['remesh_size']}, interior {preset['size_max']}, carrier 1/{a.carrier_n})",
          f"- mesh: {mesh.node_count} Tet4 nodes / {mesh.tet_count} tets, min dihedral {mesh.min_dihedral_degrees:.2f} deg, volume {mesh.material_volume:.5f}",
          f"- fixed port: {receipt['fixed_port']['active_carrier_nodes']} active carrier nodes (q = {receipt['fixed_port']['q_active']}), "
          f"partition of unity {receipt['fixed_port']['partition_of_unity_error']:.1e}, affine {receipt['fixed_port']['affine_reproduction_error']:.1e}, "
          f"support outside active {check['support_outside_active']}, active without support {check['active_without_support']}",
          f"- Schur: rigid residual {rigid_residual:.1e}, eigenvalues [{eig[0]:.3e}, {eig[-1]:.3e}]",
          f"- timing: surface+mesh {receipt['timing']['surface_and_mesh']:.0f}s, port {receipt['timing']['fixed_port']:.0f}s, schur {receipt['timing']['schur']:.0f}s, total {receipt['timing']['total']:.0f}s",
          f"- storage: upper triangle float32, {receipt['schur']['storage']['bytes'] / 1e6:.1f} MB (float32 rounding {receipt['schur']['storage']['float32_rounding_relative']:.1e} relative)",
          "", "Contents: `FIXED_PORT_SCHUR_TET10.npz` (schur_upper_f32 + q, carrier ids/coordinates/port membership, rigid basis, carrier mass and",
          "Laplace-Beltrami matrices of the active complex as COO, material volume; pipe.load_schur() / pipe.load_carrier_norms() rebuild them), `mesh.mesh`, `SHEET_SOLID_SURFACE.off`."]
    (out / "LABEL_RECEIPT.md").write_text("\n".join(md) + "\n")
    print("\n".join(md), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
