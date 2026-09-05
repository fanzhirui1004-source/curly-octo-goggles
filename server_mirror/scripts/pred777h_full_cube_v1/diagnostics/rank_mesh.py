#!/usr/bin/env python3
"""Is the operator's RANK a property of the geometry, or of the mesh?

q (the port dimension) is mesh-independent by contract.  The rank is not obviously so: an exact null mode appears when
two active carrier columns see the SAME trace vertices with the same weights, which a finer mesh can break.  Track A
produced the same cells at three mesh resolutions, so this is directly measurable.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from label_error import load_label

A = Path("/root/autodl-tmp/_claude_diag/tonight/A"); T = Path("/root/autodl-tmp/_claude_diag/tonight")
cells = [c.strip() for c in (T / "trackA.txt").read_text().split() if c.strip()]
print(f"{'cell':<20}{'preset':<12}{'q':>7}{'zero':>7}{'tets':>9}" + "".join(f"{('null<1e%d'%-e):>11}" for e in (14, 12, 10, 8)) + f"{'rank':>7}{'rank/q':>9}")
rows = []
for c in cells:
    for preset in ("fast", "production", "reference"):
        d = A / f"A1_{preset}_{c}"
        if not (d / "FIXED_PORT_SCHUR_TET10.npz").exists(): continue
        try: lab = load_label(d)
        except Exception: continue
        S = lab["S"]; n = S.shape[0] // 3
        rown = np.linalg.norm(S.reshape(n, 3, -1), axis=(1, 2)); nz = int((rown <= 1e-12 * rown.max()).sum())
        w = np.linalg.eigvalsh(S); wmax = w[-1]
        cnt = {e: int((w <= 10.0 ** (-e) * wmax).sum()) for e in (14, 12, 10, 8)}
        tets = int(lab["receipt"]["mesh"]["tet_count"])
        rows.append({"cell": c, "preset": preset, "q": lab["q"], "zero_rows": nz, "tets": tets, "null_counts": cnt,
                     "rank": lab["q"] - cnt[14]})
        print(f"{c:<20}{preset:<12}{lab['q']:>7}{nz:>7}{tets:>9}" + "".join(f"{cnt[e]:>11}" for e in (14, 12, 10, 8))
              + f"{lab['q'] - cnt[14]:>7}{(lab['q'] - cnt[14]) / lab['q']:>9.4f}", flush=True)
byc = {}
for r in rows: byc.setdefault(r["cell"], []).append(r)
print("\nper cell, across mesh resolutions:")
for c, rs in byc.items():
    if len(rs) < 2: continue
    q = {r["q"] for r in rs}; z = {r["zero_rows"] for r in rs}; k = {r["null_counts"][14] for r in rs}
    print(f"  {c:<20} q {'STABLE' if len(q)==1 else 'VARIES ' + str(sorted(q)):<28} zero-rows {'STABLE' if len(z)==1 else 'VARIES ' + str(sorted(z)):<26} null-dim {'STABLE' if len(k)==1 else 'VARIES ' + str(sorted(k))}")
json.dump(rows, open(T / "RANK_VS_MESH.json", "w"), indent=1, default=float)
print("written", T / "RANK_VS_MESH.json")
