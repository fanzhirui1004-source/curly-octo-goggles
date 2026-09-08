#!/usr/bin/env python3
"""Near-degenerate stress population for the sheet surface pipeline (robustness census, 2026-09-04).

Six thickness fields (uniform rho 0.1 / 0.23 / 0.5 and three graded fields) x {uncut, vertical cuts at 0/15/30/45 deg
x 12 offsets across the cell} plus 24 random graded cells (cut and uncut) = 342 manifests, written as
<out>/<case>/geometry_material_manifest.json in the frozen manifest format read by mesher_neutral_snapshot.
"""
import sys, json, numpy as np
from fractions import Fraction
from pathlib import Path
out_root = Path(sys.argv[1]); out_root.mkdir(parents=True, exist_ok=True)
def frac(x, den=10000): return Fraction(x).limit_denominator(den)
def fp(v): return {"numerator": v.numerator, "denominator": v.denominator}
def write_case(cid, corners, plane):
    """corners: 8 floats ordered (i,j,k) for i,j,k in {0,1}; plane: (a,b,c,d) floats or None (retained side a x + b y + c z <= d)."""
    vals = [frac(c) for c in corners]
    payload = {"cell_id": cid, "parent_world_cut_id": "NO_CUT" if plane is None else f"CUT_{cid}",
               "global_thickness_field": {"field_id": f"STRESS_{cid}", "origin": [fp(Fraction(0))] * 3, "cell_size": [fp(Fraction(1))] * 3, "cell_shape": [1, 1, 1], "vertex_values": [fp(v) for v in vals]},
               "cell_placement": {"translation": [fp(Fraction(0))] * 3},
               "cut_plane": None if plane is None else {"plane_id": "cut_0", "semantic_id": "cut_0", "inside_relation": "a*x+b*y+c*z<=d", "raw_coefficients": {k: fp(frac(v)) for k, v in zip("abcd", plane)}}}
    d = out_root / cid; d.mkdir(exist_ok=True); (d / "geometry_material_manifest.json").write_text(json.dumps(payload, indent=1)); return cid
TAU_MIN, TAU_MAX = 0.1755, 0.8775
rng = np.random.default_rng(20260903); cases = []
def graded(tau_bar, gvec, hi_scale):
    corners = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                x = np.array([i, j, k]) - 0.5; v = tau_bar + gvec @ x
                v += hi_scale * (rng.uniform(-1, 1) * x[0] * x[1] + rng.uniform(-1, 1) * x[1] * x[2] + rng.uniform(-1, 1) * x[0] * x[2])
                corners.append(float(np.clip(v, TAU_MIN, TAU_MAX)))
    return corners
fields = {"thin": [TAU_MIN] * 8, "mid": [0.4] * 8, "thick": [TAU_MAX] * 8,
          "thin_grad": graded(0.30, np.array([0.25, 0.0, 0.0]), 0.05), "mid_grad_diag": graded(0.5, np.array([0.27, 0.27, 0.27]), 0.08), "thick_grad": graded(0.72, np.array([0.0, -0.3, 0.2]), 0.06)}
angles = {"a00": (1.0, 0.0), "a15": (1.0, 0.2679), "a30": (1.0, 0.5774), "a45": (1.0, 1.0)}
for fname, corners in fields.items():
    cases.append(write_case(f"{fname}_uncut", corners, None))
    for an, (a, b) in angles.items():
        for m, frac_d in enumerate(np.linspace(0.04, 0.96, 12)):
            cases.append(write_case(f"{fname}_{an}_d{m:02d}", corners, (a, b, 0.0, frac_d * (a + b))))
# random graded cells (sampling scheme: mean, gradient magnitude ~ u^2, random direction, higher modes)
for r in range(24):
    tau_bar = rng.uniform(TAU_MIN + 0.05, TAU_MAX - 0.05); mag = 0.47 * rng.uniform() ** 2; v = rng.normal(size=3); v /= np.linalg.norm(v)
    corners = graded(tau_bar, mag * v, 0.2 * mag)
    a, b = np.cos(rng.uniform(0, np.pi / 4)), np.sin(rng.uniform(0, np.pi / 4)); frac_d = rng.uniform(0.05, 0.95)
    cases.append(write_case(f"rand{r:02d}_cut", corners, (a, b, 0.0, frac_d * (a + b))))
    cases.append(write_case(f"rand{r:02d}_uncut", corners, None))
(out_root / "CASES.txt").write_text("\n".join(cases) + "\n"); print(len(cases), "cases")
