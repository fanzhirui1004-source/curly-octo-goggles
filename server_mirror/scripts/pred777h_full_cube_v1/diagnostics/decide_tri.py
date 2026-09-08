#!/usr/bin/env python3
"""Does a carrier NODE's own P1 patch hold material?  Decided on the patch triangles, not their bounding boxes.

For each carrier trace triangle: 4-way midpoint subdivision, with two rigorous tests per sub-triangle.

  PROOF OF MATERIAL   a sample point inside the sub-triangle with lev = |phi| - tau <= 0 (and on the retained side).
  PROOF OF EMPTINESS  min|phi| > max tau over the sub-triangle's axis-aligned bounding box, both exact in closed form
                      (phi is separable, tau is trilinear so its box extrema are at the corners).

A triangle is empty only if EVERY sub-triangle is proven empty; undecided sub-triangles are subdivided.  Whatever is
still undecided at the depth limit counts as material (the safe side, so the active set stays a superset).
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
DEPTH = 7
BARY = np.asarray([(1/3, 1/3, 1/3), (0.5, 0.25, 0.25), (0.25, 0.5, 0.25), (0.25, 0.25, 0.5),
                   (0.6, 0.2, 0.2), (0.2, 0.6, 0.2), (0.2, 0.2, 0.6), (1/3, 1/3, 1/3)])


def cos_range(a, b):
    ca, cb = np.cos(2 * np.pi * a), np.cos(2 * np.pi * b)
    lo = np.minimum(ca, cb); hi = np.maximum(ca, cb)
    full = (b - a) >= 1.0
    hi = np.where(full | (np.floor(b) >= np.ceil(a)), 1.0, hi)
    lo = np.where(full | (np.floor(b - 0.5) >= np.ceil(a - 0.5)), -1.0, lo)
    return lo, hi


def decide(geometry, planes, V, depth=DEPTH):
    """V: (n, 3, 3) triangles.  Returns (has_material, undecided) booleans of length n."""
    n = len(V)
    mat = np.zeros(n, bool)
    owner = np.arange(n); W = V.copy()
    for _ in range(depth + 1):
        if not len(owner): break
        lo = W.min(axis=1); hi = W.max(axis=1)
        plo = np.zeros(len(W)); phh = np.zeros(len(W))
        for k in range(3):
            a, b = cos_range(lo[:, k], hi[:, k]); plo += a; phh += b
        straddle = (plo <= 0) & (phh >= 0)
        min_abs = np.where(straddle, 0.0, np.minimum(np.abs(plo), np.abs(phh)))
        tmax = np.full(len(W), -np.inf)
        for c in range(8):
            sel = np.array([[c & 1, (c >> 1) & 1, (c >> 2) & 1]], dtype=bool)
            X = np.where(sel, hi, lo)
            tmax = np.maximum(tmax, np.abs(np.cos(2 * np.pi * X).sum(axis=1)) - sss.sheet_level(geometry, X))
        proven_empty = min_abs > tmax
        witness = np.zeros(len(W), bool)
        for bc in BARY:
            X = np.einsum("c,ncj->nj", bc, W)
            lev = sss.sheet_level(geometry, X); ins = np.ones(len(X), bool)
            for _, nrm, d in planes: ins &= (X @ nrm - d) <= 1e-12
            witness |= ins & (lev <= 0.0)
        np.logical_or.at(mat, owner[witness], True)
        keep = ~proven_empty & ~witness & ~mat[owner]
        if not keep.any(): break
        owner = owner[keep]; W = W[keep]
        A, B, C = W[:, 0], W[:, 1], W[:, 2]
        AB, BC, CA = 0.5 * (A + B), 0.5 * (B + C), 0.5 * (C + A)
        W = np.concatenate([np.stack([A, AB, CA], 1), np.stack([AB, B, BC], 1),
                            np.stack([CA, BC, C], 1), np.stack([AB, BC, CA], 1)])
        owner = np.tile(owner, 4)
    undec = np.zeros(n, bool); np.logical_or.at(undec, owner, True)
    return mat, undec & ~mat


def analyse(case: str) -> dict:
    d = T / case if "/" in case else T / "B" / case
    r = json.loads((d / "LABEL_RECEIPT.json").read_text())
    man = Path(r["geometry_manifest"]); cn = int(r["resolution"]["carrier_n"])
    _, spec, geom = snap.load_geometry_manifest(man)
    charts, *_ = fcb.compile_full_cube_geometry_inputs(spec)
    layout = pipe.build_carrier_layout(charts, spec, carrier_n=cn)
    stage = pipe.compile_port(d / "mesh.mesh", geom, spec, layout, carrier_n=cn)
    active = np.asarray(stage.active, bool)
    Sc = stage.port.scalar_fine_to_carrier.tocsc()
    support = np.asarray((Sc != 0).sum(axis=0)).ravel() > 0
    keys = list(layout.global_scalar_node_keys); g2c = dict(stage.port.global_to_compact)
    lay2c = np.array([g2c.get(str(k), -1) for k in keys]); planes = sss.clip_planes(geom)
    shell, _all, per_face = sss.carrier_shell(layout, tuple(stage.port.active_global_port_ids))
    has = np.zeros(len(keys), bool); und = np.zeros(len(keys), bool)
    for src, tris in per_face:
        mat, undec = decide(geom, planes, shell[tris])
        for t in np.flatnonzero(mat): has[tris[t]] = True
        for t in np.flatnonzero(undec): und[tris[t]] = True
    hv = np.zeros(len(active), bool); uv = np.zeros(len(active), bool); ok = lay2c >= 0
    hv[lay2c[ok]] = has[np.flatnonzero(ok)]; uv[lay2c[ok]] = und[np.flatnonzero(ok)]
    keep = active & (hv | uv); drop = active & ~hv & ~uv
    out = {"case": case, "active": int(active.sum()), "q": int(3 * active.sum()),
           "patch_holds_material": int((active & hv).sum()), "undecided": int((active & uv & ~hv).sum()),
           "patch_proven_empty": int(drop.sum()), "proven_empty_but_meshed": int((drop & support).sum()),
           "zero_support": int((active & ~support).sum()),
           "zero_support_with_material": int((active & hv & ~support).sum()),
           "q_new": int(3 * keep.sum())}
    print(f"\n=== {case} ===")
    print(f"  active {out['active']} (q {out['q']}): patch PROVEN to hold material {out['patch_holds_material']}, "
          f"undecided {out['undecided']}, PROVEN EMPTY {out['patch_proven_empty']}")
    print(f"  zero-support columns {out['zero_support']}, of which the patch provably HAS material: "
          f"{out['zero_support_with_material']}  <- the mesh missed real material")
    print(f"  dropping only the proven-empty: q {out['q']} -> {out['q_new']} ({100*(out['q_new']/out['q']-1):+.1f} %); "
          f"proven-empty yet meshed (must be 0): {out['proven_empty_but_meshed']}")
    sys.stdout.flush()
    return out


if __name__ == "__main__":
    rows = [analyse(c) for c in (sys.argv[1:] or ["pop_cut_0136"])]
    json.dump(rows, open(T / "DECIDE_TRI.json", "w"), indent=1, default=float)
    print("\nwritten", T / "DECIDE_TRI.json")
