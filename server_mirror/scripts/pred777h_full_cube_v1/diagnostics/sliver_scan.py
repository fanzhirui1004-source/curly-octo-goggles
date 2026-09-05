#!/usr/bin/env python3
"""How many produced labels carry a sliver artefact in the operator?

Evidence (top_mode.py): in the bad labels ALL of lambda_max's eigenvector sits on the four corners of ONE carrier
square, and lambda_max is up to 11x its converged value.  That is a single near-degenerate tetrahedron, not a port
stiffness.  This scans the production sample and counts them, by the participation of the top mode.
"""
from __future__ import annotations
import glob, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from label_error import load_label

T = Path("/root/autodl-tmp/_claude_diag/tonight")


def top_mode(S, iters=300):
    n = S.shape[0]
    v = np.ones(n) / np.sqrt(n)
    lam = 0.0
    for _ in range(iters):
        w = S @ v; nw = np.linalg.norm(w)
        if nw == 0: return 0.0, v
        v = w / nw
        lam = float(v @ (S @ v))
    return lam, v


rows = []
files = sorted(glob.glob(str(T / "B" / "*" / "LABEL_RECEIPT.json")))
print(f"scanning {len(files)} labels", flush=True)
for f in files:
    d = Path(f).parent; r = json.loads(Path(f).read_text())
    if r.get("status") != "PASS": continue
    try: lab = load_label(d)
    except Exception: continue
    S = lab["S"]; n = S.shape[0] // 3
    lam, v = top_mode(S)
    part = np.linalg.norm(v.reshape(n, 3), axis=1) ** 2
    order = np.argsort(-part)
    m = lab["receipt"]["mesh"]
    rows.append({"case": d.name, "lambda_max": lam, "top4": float(part[order[:4]].sum()),
                 "top16": float(part[order[:16]].sum()), "nodes_above_1pct": int((part > 0.01).sum()),
                 "min_dihedral": float(m["min_dihedral_degrees"]), "tets_below_5deg": int(m["tets_below_5deg"]),
                 "cut": r["geometry"]["cut_plane"] is not None, "volume": float(m["material_volume"])})
    if len(rows) % 25 == 0: print(f"  {len(rows)} done", flush=True)

t4 = np.asarray([x["top4"] for x in rows]); md = np.asarray([x["min_dihedral"] for x in rows])
print(f"\nlabels scanned {len(rows)}")
for thr in (0.5, 0.7, 0.9, 0.95, 0.99):
    print(f"  top mode with >= {thr:.0%} of its mass on 4 carrier nodes (one carrier square): "
          f"{int((t4 >= thr).sum())}  ({100 * (t4 >= thr).mean():.1f} %)")
print(f"  min dihedral: med {np.median(md):.3f} deg, p05 {np.percentile(md, 5):.4f}, min {md.min():.4f}")
print(f"  correlation(top4, -log min_dihedral) = {np.corrcoef(t4, -np.log(np.maximum(md, 1e-6)))[0,1]:.3f}")
worst = sorted(rows, key=lambda x: -x["top4"])[:10]
print("\n  worst by localisation:")
for x in worst:
    print(f"    {x['case']:<20} lambda_max {x['lambda_max']:.5g}  top4 {x['top4']:.3f}  "
          f"nodes>1% {x['nodes_above_1pct']:>3}  minDihed {x['min_dihedral']:.4f}")
json.dump(rows, open(T / "SLIVER_SCAN.json", "w"), indent=1, default=float)
print("\nwritten", T / "SLIVER_SCAN.json")
