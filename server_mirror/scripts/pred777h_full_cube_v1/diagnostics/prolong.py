#!/usr/bin/env python3
"""The prolongation itself: why the label operator has more null modes than 6 + 3 * unsupported.

S = P' S_bb P with P = scalar (x) I3.  This rebuilds `scalar` for a produced label from its SAVED mesh (no remeshing)
and measures its null space directly: how many active carrier columns the boundary trace never reaches, and how many
SURVIVING columns are linearly dependent on the trace.  Each dependency costs three null modes in S.
"""
from __future__ import annotations

import json, sys
from importlib import import_module
from pathlib import Path

import numpy as np

ROOT = Path("/root/autodl-tmp/cut_control_tpms_v1_full_cube_single_cell_v1")
sys.path.insert(0, str(ROOT / "src"))
P = "pred777h_full_cube_single_cell_mesh_graph_fixed_port_schur_v1"
fcb = import_module(f"{P}.full_cube_backend"); snap = import_module(f"{P}.mesher_neutral_snapshot")
sss = import_module(f"{P}.sheet_solid_surface"); pipe = import_module(f"{P}.sheet_label_pipeline")

T = Path("/root/autodl-tmp/_claude_diag/tonight")


def analyse(case: str) -> dict:
    d = T / "B" / case
    r = json.loads((d / "LABEL_RECEIPT.json").read_text())
    man = Path(r["geometry_manifest"]); carrier_n = int(r["resolution"]["carrier_n"])
    _, spec, geom = snap.load_geometry_manifest(man)
    charts, *_ = fcb.compile_full_cube_geometry_inputs(spec)
    layout = pipe.build_carrier_layout(charts, spec, carrier_n=carrier_n)
    stage = pipe.compile_port(d / "mesh.mesh", geom, spec, layout, carrier_n=carrier_n)
    scalar = stage.port.scalar_fine_to_carrier.tocsc()
    active = np.asarray(stage.active, dtype=bool)
    A = scalar[:, np.flatnonzero(active)].toarray()          # n_fine x n_active
    nfine, nact = A.shape
    nnz = (np.abs(A) > 0).sum(axis=0)
    coords = np.asarray(stage.port.carrier_coordinates, dtype=float)[active]
    fine_xyz = np.asarray(stage.artifact.mesh_nodes[stage.port.fine_port_node_ids], dtype=float) \
        if hasattr(stage.port, "fine_port_node_ids") else None

    s = np.linalg.svd(A, compute_uv=False)
    smax = s[0] if len(s) else 1.0
    tiers = {e: int((s <= 10.0 ** (-e) * smax).sum()) for e in (14, 12, 10, 8, 6, 4)}
    U_, s_, Vt = np.linalg.svd(A, full_matrices=True)
    nullA = Vt[len(s_) - tiers[10]:].T if tiers[10] else np.zeros((nact, 0))
    if tiers[10]:
        nullA = Vt[nact - tiers[10]:, :].T                    # n_active x dim
    zero_cols = np.flatnonzero(nnz == 0)
    # the dependencies among SURVIVING columns: null space modulo the zero-column indicators
    keep = np.flatnonzero(nnz > 0)
    dep = 0; groups = []
    if nullA.shape[1]:
        Nk = nullA[keep, :]
        sk = np.linalg.svd(Nk, compute_uv=False) if Nk.size else np.zeros(0)
        dep = int((sk > 1e-8 * max(sk[0], 1e-300)).sum()) if sk.size else 0
        if dep:
            Uk, sk2, _ = np.linalg.svd(Nk, full_matrices=False)
            B = Uk[:, :dep]
            Pi = B @ B.T                                     # projector onto the dependency space, on surviving cols
            part = np.diag(Pi)
            invol = np.flatnonzero(part > 1e-6)
            # try to name each dependency as a pair of columns whose trace restrictions are proportional
            for a_ in range(len(invol)):
                for b_ in range(a_ + 1, len(invol)):
                    i, j = keep[invol[a_]], keep[invol[b_]]
                    ca, cb = A[:, i], A[:, j]
                    na, nb = np.linalg.norm(ca), np.linalg.norm(cb)
                    if na == 0 or nb == 0: continue
                    cos = abs(float(ca @ cb) / (na * nb))
                    if cos > 1 - 1e-10:
                        rows = np.flatnonzero((np.abs(ca) > 0) | (np.abs(cb) > 0))
                        groups.append({"columns": [int(i), int(j)], "xyz": [coords[i].round(6).tolist(), coords[j].round(6).tolist()],
                                       "nnz": [int(nnz[i]), int(nnz[j])], "cos": cos,
                                       "shared_trace_rows": int(len(rows)),
                                       "weights": [[round(float(ca[t]), 6), round(float(cb[t]), 6)] for t in rows[:6]],
                                       "trace_xyz": [fine_xyz[t].round(6).tolist() for t in rows[:6]] if fine_xyz is not None else None})
    hist = {int(k): int(v) for k, v in zip(*np.unique(np.minimum(nnz, 8), return_counts=True))}
    out = {"case": case, "fine_trace_nodes": int(nfine), "active_carrier_columns": int(nact),
           "zero_support_columns": int(len(zero_cols)), "singular_value_tiers": tiers,
           "dim_null_scalar": tiers[10], "dependencies_among_supported_columns": dep,
           "support_size_histogram_capped_at_8": hist, "pairs": groups[:12],
           "predicted_null_dim_of_S": 3 * tiers[10] + 3, "q_active": int(3 * nact)}
    print(f"\n=== {case} ===")
    print(f"  fine boundary-trace vertices {nfine}, active carrier scalar columns {nact}, q = {3*nact}")
    print(f"  columns the trace never reaches (zero support)   {len(zero_cols)}")
    print(f"  support size histogram (trace vertices per column, capped at 8): {hist}")
    print(f"  singular values of the prolongation below 10^-e * s_max: " +
          ", ".join(f"1e-{e}:{tiers[e]}" for e in (14, 12, 10, 8, 6, 4)))
    print(f"  dim null(scalar) = {tiers[10]}  =  {len(zero_cols)} zero columns + {dep} dependencies among supported columns")
    print(f"  => predicted dim Null(S) = 3 * {tiers[10]} + 3 = {3 * tiers[10] + 3}")
    for g in groups[:6]:
        print(f"    dependent pair {g['xyz'][0]} and {g['xyz'][1]}: supports {g['nnz']}, "
              f"{g['shared_trace_rows']} trace vertices, cos {g['cos']:.12f}")
        for wt, xy in zip(g["weights"], (g["trace_xyz"] or [])):
            print(f"        trace vertex {xy}  weights {wt}")
    sys.stdout.flush()
    return out


if __name__ == "__main__":
    cases = sys.argv[1:] or ["pop_cut_0236"]
    rows = [analyse(c) for c in cases]
    json.dump(rows, open(T / "PROLONGATION_NULL.json", "w"), indent=1, default=float)
    print("\nwritten", T / "PROLONGATION_NULL.json")
