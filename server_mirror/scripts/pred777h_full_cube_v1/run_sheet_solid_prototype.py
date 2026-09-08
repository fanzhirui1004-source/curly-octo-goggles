#!/usr/bin/env python3
"""Prototype: true sheet-TPMS solid (no collars) -> carrier-conforming surface -> Gmsh Tet4 -> Tet10 fixed-port Schur."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
P = "pred777h_full_cube_single_cell_mesh_graph_fixed_port_schur_v1"
from importlib import import_module
fcb = import_module(f"{P}.full_cube_backend"); pipe = import_module(f"{P}.sheet_label_pipeline")
snap = import_module(f"{P}.mesher_neutral_snapshot"); sss = import_module(f"{P}.sheet_solid_surface")

def build_layout(spec, charts, carrier_n: int):
    """Carrier trace layout with a (carrier_n x carrier_n) P1 grid per face (the frozen contract uses 16)."""
    return pipe.build_carrier_layout(charts, spec, carrier_n=carrier_n)

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-id", required=True); ap.add_argument("--geometry-manifest", type=Path, required=True); ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--n-per-unit", type=int, default=48); ap.add_argument("--remesh-size", type=float, default=0.02); ap.add_argument("--size-max", type=float, default=0.05); ap.add_argument("--algorithm3d", type=int, default=10)
    ap.add_argument("--no-netgen", action="store_true"); ap.add_argument("--no-smooth", action="store_true", help="ignored (kept for old panel scripts)"); ap.add_argument("--cap-max-area", type=float, default=None)
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--threads", type=int, default=8); ap.add_argument("--skip-schur", action="store_true"); ap.add_argument("--carrier-n", type=int, default=32); ap.add_argument("--no-carrier-edges", action="store_true", help="constrain only carrier nodes (not edges) in the cap triangulation")
    a = ap.parse_args(); out = a.output_dir; out.mkdir(parents=True, exist_ok=True); rep: dict = {"case_id": a.case_id}
    _, spec, geom = snap.load_geometry_manifest(a.geometry_manifest)
    charts, *_ = fcb.compile_full_cube_geometry_inputs(spec); layout = build_layout(spec, charts, a.carrier_n); rep["carrier_n"] = a.carrier_n
    t0 = time.perf_counter()
    surf = sss.build_sheet_solid_surface(geom, layout, n_per_unit=a.n_per_unit, remesh_size=a.remesh_size, carrier_n=a.carrier_n, cap_max_area=a.cap_max_area, constrain_carrier_edges=not a.no_carrier_edges)
    sss.write_off(out / "SHEET_SOLID_SURFACE.off", surf.V, surf.F); rep["surface"] = surf.report; rep["surface"]["seconds_total"] = time.perf_counter() - t0
    print("surface:", json.dumps({k: v for k, v in surf.report.items() if k != "caps"}, default=float)); print("caps:", json.dumps(surf.report["caps"], default=float), flush=True)
    if surf.report["watertight"]["non_two_manifold_edges"] != 0:
        (out / "SHEET_PROTOTYPE_REPORT.json").write_text(json.dumps(rep, indent=1, default=float)); print("surface not watertight, stop"); return 1
    m = sss.mesh_sheet_solid(surf, out / "mesh.mesh", interior_size_max=a.size_max, algorithm3d=a.algorithm3d, netgen_optimize=not a.no_netgen, threads=a.threads)
    rep["mesh"] = {k: (str(v) if isinstance(v, Path) else v) for k, v in m.__dict__.items()}; print("mesh:", json.dumps(rep["mesh"], default=float), flush=True)
    stage = pipe.compile_port(out / "mesh.mesh", geom, spec, layout, carrier_n=a.carrier_n); rep["active_set"] = stage.check; print("active_set:", stage.check, flush=True)
    support = stage.active; rep["fixed_port"] = {**stage.summary, "seconds": stage.seconds}; print("fixed_port:", json.dumps(rep["fixed_port"], default=float), flush=True)
    np.save(out / "carrier_support.npy", support)
    if not a.skip_schur:
        sch = pipe.schur_label(stage, workers=a.workers)
        np.save(out / "S_TET10_supported.npy", sch.schur_active)
        rep["schur"] = {"timing": sch.timing, "q_supported": int(sch.schur_active.shape[0]), "unsupported_block_max": sch.inactive_block_max,
                        "min_eig_supported": float(sch.eigenvalues[0]), "max_eig_supported": float(sch.eigenvalues[-1]), "rigid_residual_supported": sch.rigid_residual}
        print("schur:", json.dumps(rep["schur"], default=float), flush=True)
    (out / "SHEET_PROTOTYPE_REPORT.json").write_text(json.dumps(rep, indent=1, default=float)); return 0

if __name__ == "__main__":
    raise SystemExit(main())
