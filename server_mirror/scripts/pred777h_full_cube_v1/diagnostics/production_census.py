#!/usr/bin/env python3
"""Full-set census of the 2009-cell production run.

Pass 1 (this file) reads only receipts: guards, null space, the sliver top-mode diagnostic, mesh and surface
quality, geometry coverage, storage and throughput.  The null-space and top-mode sweeps used to need an
eigendecomposition per cell; the receipt now carries both, so the full set costs a JSON parse.
"""
from __future__ import annotations
import glob, json, collections
from pathlib import Path
import numpy as np

L = Path("/root/autodl-tmp/_claude_diag/production/labels")
POP = Path("/root/autodl-tmp/_claude_diag/population/cells")

def q(v, ps=(0, 1, 5, 50, 95, 99, 100)):
    a = np.asarray([x for x in v if x is not None and np.isfinite(x)], dtype=float)
    if a.size == 0: return None
    return {f"p{p}": float(np.percentile(a, p)) for p in ps} | {"n": int(a.size), "mean": float(a.mean())}

rows, status = [], collections.Counter()
heads = collections.Counter()
for f in sorted(glob.glob(str(L / "*" / "LABEL_RECEIPT.json"))):
    d = json.loads(Path(f).read_text())
    status[d.get("status", "?")] += 1
    heads[d.get("git_head", "?")[:7]] += 1
    rows.append(d)

allcells = sorted(p.name for p in POP.iterdir() if p.is_dir())
noreceipt = [c for c in allcells if not (L / c / "LABEL_RECEIPT.json").exists()]

P = [d for d in rows if d.get("status") == "PASS"]
# Two flags, not one.  has_cut_plane: the cell's geometry carries a world cut plane at all.  cut_is_port: the cut
# face actually carries material, so 'cut_0' is a load-bearing port.  A cell can have the first without the second,
# and conflating them puts cut cells into the uncut density fit and hides the thin cut faces STEP8 measured.
has_cut = [d for d in P if (d.get("geometry", {}) or {}).get("cut_face") is not None]
cut_port = [d for d in P if "cut_0" in d["fixed_port"]["active_ports"]]
uncut = [d for d in P if (d.get("geometry", {}) or {}).get("cut_face") is None]
cut = has_cut

def g(d, *path, default=None):
    o = d
    for k in path:
        if not isinstance(o, dict) or k not in o: return default
        o = o[k]
    return o

out = {
    "population": {"cells": len(allcells), "receipts": len(rows), "status": dict(status),
                   "unproducible": noreceipt, "git_head": dict(heads),
                   "pass_has_cut_plane": len(has_cut), "pass_cut_is_load_bearing_port": len(cut_port),
                   "pass_cut_plane_no_material": len(has_cut) - len(cut_port), "pass_truly_uncut": len(uncut)},

    # --- the operator gates, over every PASS label -----------------------------------------------------------
    "guards": {
        "rigid_residual_relative": q([g(d, "schur", "rigid_residual_relative") for d in P]),
        "min_eig_over_max_eig": q([g(d, "schur", "min_eigenvalue", default=0.0) / g(d, "schur", "max_eigenvalue", default=1.0) for d in P]),
        "inactive_block_max": q([g(d, "schur", "inactive_block_max") for d in P]),
        "partition_of_unity_error": q([g(d, "fixed_port", "partition_of_unity_error") for d in P]),
        "affine_reproduction_error": q([g(d, "fixed_port", "affine_reproduction_error") for d in P]),
        "float32_rounding_relative": q([g(d, "schur", "storage", "float32_rounding_relative") for d in P]),
        "worst_rigid_residual": max((g(d, "schur", "rigid_residual_relative", default=0.0), d["case_id"]) for d in P),
        "worst_min_eig_ratio": min((g(d, "schur", "min_eigenvalue", default=0.0) / g(d, "schur", "max_eigenvalue", default=1.0), d["case_id"]) for d in P),
    },

    # --- null space: is it exactly the combinatorial lower bound, everywhere? --------------------------------
    "nullspace": {
        "checked": sum(1 for d in P if g(d, "schur", "support", "null_dim_at_1e-14") is not None),
        "equal_to_lower_bound": sum(1 for d in P if g(d, "schur", "support", "null_dim_at_1e-14") == g(d, "schur", "support", "null_dim_lower_bound")),
        "excess": collections.Counter(
            (g(d, "schur", "support", "null_dim_at_1e-14") or 0) - (g(d, "schur", "support", "null_dim_lower_bound") or 0) for d in P),
        "threshold_disagreement": sum(1 for d in P if g(d, "schur", "support", "null_dim_at_1e-14") != g(d, "schur", "support", "null_dim_at_1e-10")),
        "unsupported_nodes": q([g(d, "schur", "support", "unsupported_nodes") for d in P]),
        "unsupported_fraction": q([g(d, "schur", "support", "unsupported_nodes", default=0) / max(1, g(d, "fixed_port", "active_carrier_nodes", default=1)) for d in P]),
        "support_outside_active": sum(g(d, "fixed_port", "active_set_check", "support_outside_active", default=0) for d in P),
    },

    # --- the sliver artefact, by the top-mode diagnostic of record -------------------------------------------
    "top_mode": {
        "mass_on_4_nodes": q([g(d, "schur", "top_mode_mass_on_4_nodes") for d in P]),
        "ge_0.99": sum(1 for d in P if (g(d, "schur", "top_mode_mass_on_4_nodes") or 0) >= 0.99),
        "ge_0.95": sum(1 for d in P if (g(d, "schur", "top_mode_mass_on_4_nodes") or 0) >= 0.95),
        "ge_0.90": sum(1 for d in P if (g(d, "schur", "top_mode_mass_on_4_nodes") or 0) >= 0.90),
        "worst": sorted(((g(d, "schur", "top_mode_mass_on_4_nodes") or 0), d["case_id"]) for d in P)[-8:],
        "nodes_above_1pct": q([g(d, "schur", "top_mode_nodes_above_1pct") for d in P]),
    },

    # --- surface and mesh quality ------------------------------------------------------------------------------
    "surface": {
        "non_two_manifold_edges_nonzero": sum(1 for d in P if g(d, "surface", "watertight", "non_two_manifold_edges", default=0)),
        "self_intersecting_triangles_nonzero": sum(1 for d in P if g(d, "surface", "self_intersecting_triangles", default=0)),
        "cap_needles_remaining_nonzero": sum(1 for d in P if g(d, "surface", "cap_needles", "remaining", default=0)),
        "cap_needle_apex_collapsed": q([g(d, "surface", "cap_needles", "apex_collapsed") for d in P]),
        "cap_min_angle_degrees": q([g(d, "surface", "cap_needles", "min_angle_degrees") for d in P]),
        "components_dropped_nonzero": sum(1 for d in P if g(d, "surface", "components", "dropped", default=0)),
        "components_gt1": sum(1 for d in P if (g(d, "surface", "components", "components") or 1) > 1),
        "quality_min": q([g(d, "surface", "surface_quality", "min") for d in P]),
        "triangles": q([g(d, "surface", "surface_quality", "triangles") for d in P]),
    },
    "mesh": {
        "min_dihedral_degrees": q([g(d, "mesh", "min_dihedral_degrees") for d in P]),
        "below_1deg": sum(1 for d in P if (g(d, "mesh", "min_dihedral_degrees") or 99) < 1.0),
        "tets_below_5deg": q([g(d, "mesh", "tets_below_5deg") for d in P]),
        "nonmanifold_facets_nonzero": sum(1 for d in P if g(d, "mesh", "nonmanifold_facets", default=0)),
        "node_count": q([g(d, "mesh", "node_count") for d in P]),
        "tet_count": q([g(d, "mesh", "tet_count") for d in P]),
        "volume_vs_surface_rel": q([abs(g(d, "mesh", "material_volume", default=0) - g(d, "surface", "volume", default=0)) / max(1e-12, g(d, "surface", "volume", default=1)) for d in P]),
    },

    # --- what the dataset actually covers -----------------------------------------------------------------------
    "coverage": {
        "material_volume": q([g(d, "mesh", "material_volume") for d in P]),
        "material_volume_cut": q([g(d, "mesh", "material_volume") for d in cut]),
        "material_volume_uncut": q([g(d, "mesh", "material_volume") for d in uncut]),
        "q_active": q([g(d, "schur", "q_active") for d in P]),
        "active_carrier_nodes": q([g(d, "fixed_port", "active_carrier_nodes") for d in P]),
        "max_eigenvalue": q([g(d, "schur", "max_eigenvalue") for d in P]),
        "cut_face_min_width_over_h": q([g(d, "geometry", "cut_face", "min_width_over_carrier_spacing") for d in cut]),
        "cut_face_below_one_h": sum(1 for d in cut if (g(d, "geometry", "cut_face", "min_width_over_carrier_spacing") or 99) < 1.0),
        "cut_face_area": q([g(d, "geometry", "cut_face", "area") for d in cut]),
        "empty_cells": sum(1 for d in rows if d.get("status") == "EMPTY"),
    },

    # --- cost -----------------------------------------------------------------------------------------------------
    "cost": {
        "seconds_total": q([g(d, "timing", "total") for d in P]),
        "cpu_seconds_sum": float(sum(g(d, "timing", "total", default=0.0) for d in rows)),
        "memory_budget_wait": q([g(d, "timing", "memory_budget_wait") for d in P]),
        "schur_seconds": q([g(d, "schur_timing", "schur_total") for d in P]),
        "fine_dof": q([g(d, "schur_timing", "fine_dof") for d in P]),
        "operator_bytes_sum_gb": float(sum(g(d, "schur", "storage", "bytes", default=0) for d in P)) / 2**30,
        "operator_bytes": q([g(d, "schur", "storage", "bytes") for d in P]),
    },
}

Path("/root/autodl-tmp/_claude_diag/production/PRODUCTION_CENSUS.json").write_text(json.dumps(out, indent=2, default=str))
print(json.dumps(out, indent=2, default=str))
