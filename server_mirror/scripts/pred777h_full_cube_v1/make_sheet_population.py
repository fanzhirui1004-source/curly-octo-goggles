#!/usr/bin/env python3
"""Sheet-TPMS label population: space-filling random design over graded thickness and vertical cuts (2026-09-04).

Design of record (discussion of 2026-09-04):
- corners tau_1..tau_8 in [TAU_MIN, TAU_MAX] = [0.1755, 0.8775] (rho 0.10-0.50 by C = 1.755 rho, Hao 2023);
- corner field = mean + linear gradient g.(x - c) + higher trilinear modes; |g| = DTAU_MAX * u^2 (u uniform, so
  small gradients dominate), direction uniform on the sphere, 10 % of the cells forced into |g| in [0.85, 1] DTAU_MAX,
  higher modes up to 20 % of |g|; a draw with any corner outside the range or a corner spread above DTAU_MAX is
  redrawn (rejection);
- DTAU_MAX = 0.47 (= the largest in-cell spread a density filter of radius 1.5 cells can produce);
- cut cells: vertical plane n = (cos theta, sin theta, 0), theta uniform in [0, 45 deg] (the other orientations are
  reached by the 48 cube symmetries), offset uniform over the range of n.x on the cell, retained side n.x <= c;
- Sobol sequence (scrambled, fixed seed) over (mean, u, dir1, dir2, m1, m2, m3[, theta, c]);
- fixed anchors: uniform rho 0.1/0.2/0.3/0.4/0.5, axis and diagonal top-gradient cells, centre cuts at 0 and 45 deg.
Writes <out>/<case>/geometry_material_manifest.json (frozen manifest format), <out>/POPULATION.jsonl and
<out>/COVERAGE.json (bin occupancy over mean x gradient x theta x offset).
"""
from __future__ import annotations

import argparse, json, sys
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.stats import qmc

TAU_MIN, TAU_MAX = 0.1755, 0.8775
DTAU_MAX = 0.47


def frac(x: float, den: int = 10000) -> Fraction:
    return Fraction(x).limit_denominator(den)


def fp(v: Fraction) -> dict:
    return {"numerator": v.numerator, "denominator": v.denominator}


def write_case(out_root: Path, cid: str, corners, plane, meta: dict) -> dict:
    vals = [frac(c) for c in corners]
    payload = {"cell_id": cid, "parent_world_cut_id": "NO_CUT" if plane is None else f"CUT_{cid}",
               "global_thickness_field": {"field_id": f"POP_{cid}", "origin": [fp(Fraction(0))] * 3, "cell_size": [fp(Fraction(1))] * 3,
                                          "cell_shape": [1, 1, 1], "vertex_values": [fp(v) for v in vals]},
               "cell_placement": {"translation": [fp(Fraction(0))] * 3},
               "cut_plane": None if plane is None else {"plane_id": "cut_0", "semantic_id": "cut_0", "inside_relation": "a*x+b*y+c*z<=d",
                                                        "raw_coefficients": {k: fp(frac(v)) for k, v in zip("abcd", plane)}}}
    d = out_root / cid; d.mkdir(parents=True, exist_ok=True)
    (d / "geometry_material_manifest.json").write_text(json.dumps(payload, indent=1))
    rec = {"case": cid, "tau_corners": [float(v) for v in vals], "cut_plane": None if plane is None else [float(frac(v)) for v in plane], **meta}
    return rec


def corners_from(mean: float, g: np.ndarray, modes: np.ndarray) -> np.ndarray:
    out = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                x = np.array([i, j, k]) - 0.5
                out.append(mean + g @ x + modes[0] * x[0] * x[1] * 4 + modes[1] * x[1] * x[2] * 4 + modes[2] * x[0] * x[2] * 4)
    return np.asarray(out)


def draw_field(row: np.ndarray, force_top: bool) -> tuple[np.ndarray, dict] | None:
    mean = TAU_MIN + row[0] * (TAU_MAX - TAU_MIN)
    u = row[1]; mag = DTAU_MAX * (0.85 + 0.15 * u if force_top else u * u)
    cosphi = 2 * row[2] - 1; sinphi = np.sqrt(max(0.0, 1 - cosphi * cosphi)); az = 2 * np.pi * row[3]
    direction = np.array([sinphi * np.cos(az), sinphi * np.sin(az), cosphi])
    g = mag * direction
    modes = 0.2 * mag * (2 * row[4:7] - 1)
    corners = corners_from(mean, g, modes)
    if corners.min() < TAU_MIN or corners.max() > TAU_MAX or corners.max() - corners.min() > DTAU_MAX:
        return None
    return corners, {"tau_mean": float(mean), "gradient": g.tolist(), "gradient_magnitude": float(mag), "higher_modes": modes.tolist(),
                     "corner_span": float(corners.max() - corners.min()), "forced_top_gradient": bool(force_top)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True); ap.add_argument("--n-uncut", type=int, default=700); ap.add_argument("--n-cut", type=int, default=1300)
    ap.add_argument("--seed", type=int, default=20260904); ap.add_argument("--prefix", default="pop")
    a = ap.parse_args(); out = a.out; out.mkdir(parents=True, exist_ok=True); records = []
    # anchors
    for k, rho in enumerate((0.1, 0.2, 0.3, 0.4, 0.5)):
        tau = min(max(1.755 * rho, TAU_MIN), TAU_MAX)
        records.append(write_case(out, f"{a.prefix}_anchor_rho{int(rho * 100):02d}", [tau] * 8, None, {"kind": "anchor_uniform", "tau_mean": tau, "gradient_magnitude": 0.0, "corner_span": 0.0}))
    for name, direction in (("axis_x", np.array([1.0, 0, 0])), ("diag", np.array([1.0, 1.0, 1.0]) / np.sqrt(3))):
        g = DTAU_MAX * direction; mean = 0.5 * (TAU_MIN + TAU_MAX); corners = np.clip(corners_from(mean, g, np.zeros(3)), TAU_MIN, TAU_MAX)
        records.append(write_case(out, f"{a.prefix}_anchor_grad_{name}", corners, None, {"kind": "anchor_gradient", "tau_mean": mean, "gradient_magnitude": float(DTAU_MAX), "corner_span": float(corners.max() - corners.min())}))
    # the centred 45 deg plane x + y = 1 CONTAINS two vertical cell edges, which the exact chart compiler refuses
    # (measured 2026-09-04; see sheet_label_pipeline.cut_contains_a_cell_edge), so the anchor is offset off it.
    for name, (na, nb), off in (("theta00", (1.0, 0.0), 0.5), ("theta45", (1.0, 1.0), 0.53)):
        records.append(write_case(out, f"{a.prefix}_anchor_cut_{name}", [0.4] * 8, (na, nb, 0.0, off * (na + nb)), {"kind": "anchor_cut", "tau_mean": 0.4, "gradient_magnitude": 0.0, "corner_span": 0.0, "theta_deg": 45.0 if nb else 0.0, "offset_fraction": off}))
    # Sobol designs
    for kind, n, dims in (("uncut", a.n_uncut, 7), ("cut", a.n_cut, 9)):
        sampler = qmc.Sobol(d=dims, scramble=True, seed=a.seed + (1 if kind == "cut" else 0))
        made = 0; tried = 0
        while made < n:
            rows = sampler.random(256)
            for row in rows:
                if made >= n:
                    break
                tried += 1
                force_top = (tried % 10 == 0)
                drawn = draw_field(row[:7], force_top)
                if drawn is None:
                    continue
                corners, meta = drawn; meta["kind"] = kind
                plane = None
                if kind == "cut":
                    theta = row[7] * np.pi / 4; na, nb = float(np.cos(theta)), float(np.sin(theta)); rng = na + nb
                    c = row[8] * rng; plane = (na, nb, 0.0, c); meta.update({"theta_deg": float(np.degrees(theta)), "offset_fraction": float(row[8])})
                records.append(write_case(out, f"{a.prefix}_{kind}_{made:04d}", corners, plane, meta)); made += 1
        print(f"{kind}: {made} cells from {tried} draws (rejection rate {1 - made / tried:.2f})")
    (out / "POPULATION.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")
    # coverage: mean (5 bins) x gradient (3 bins) x theta (3 bins) x offset (4 bins), cut cells; mean x gradient for uncut
    mean_edges = np.linspace(TAU_MIN, TAU_MAX, 6); grad_edges = np.array([0, 0.1, 0.3, DTAU_MAX + 1e-9]); theta_edges = np.array([0, 15, 30, 45.01]); off_edges = np.linspace(0, 1, 5)
    cov = {"cut": {}, "uncut": {}}
    for r in records:
        if r.get("kind") not in ("cut", "uncut"):
            continue
        im = int(np.clip(np.searchsorted(mean_edges, r["tau_mean"], side="right") - 1, 0, 4)); ig = int(np.clip(np.searchsorted(grad_edges, r["gradient_magnitude"], side="right") - 1, 0, 2))
        if r["kind"] == "cut":
            it = int(np.clip(np.searchsorted(theta_edges, r["theta_deg"], side="right") - 1, 0, 2)); io = int(np.clip(np.searchsorted(off_edges, r["offset_fraction"], side="right") - 1, 0, 3))
            key = f"mean{im}_grad{ig}_theta{it}_off{io}"
        else:
            key = f"mean{im}_grad{ig}"
        cov[r["kind"]][key] = cov[r["kind"]].get(key, 0) + 1
    summary = {kind: {"bins": len(v), "min_occupancy": min(v.values()) if v else 0, "max_occupancy": max(v.values()) if v else 0, "cells": sum(v.values())} for kind, v in cov.items()}
    (out / "COVERAGE.json").write_text(json.dumps({"summary": summary, "bins": cov, "edges": {"mean": mean_edges.tolist(), "gradient": grad_edges.tolist(), "theta_deg": theta_edges.tolist(), "offset_fraction": off_edges.tolist()}}, indent=1))
    print(json.dumps(summary)); print(len(records), "cells written to", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
