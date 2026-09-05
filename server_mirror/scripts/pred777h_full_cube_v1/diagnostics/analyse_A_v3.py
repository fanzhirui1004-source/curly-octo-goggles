#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from label_error3 import compare_v3, KS
T = Path(os.environ.get("TONIGHT_DIR", "/root/autodl-tmp/_claude_diag/tonight")); A = T / "A"
cells = [c.strip() for c in (T / "trackA.txt").read_text().split() if c.strip()]
have = lambda p: (A / p / "FIXED_PORT_SCHUR_TET10.npz").exists()
print("A1 v3: relative strain-energy bound of S_h against S_ref on the k SOFTEST interface modes of the reference (mass-matrix norm)")
print(f"{'cell':<17}{'preset':<11}{'common':>7}" + "".join(f"{'k=%d' % k:>9}" for k in KS) + f"{'uniform':>10}")
rows = {}
for c in cells:
    ref = f"A1_reference_{c}"
    if not have(ref): continue
    for preset in ("fast", "production"):
        tag = f"A1_{preset}_{c}"
        if not have(tag): continue
        r = compare_v3(A / tag, A / ref); rows.setdefault(c, {})[preset] = r
        print(f"{c:<17}{preset:<11}{r['support']['common_supported']:>7}" + "".join(f"{e['bound']:>9.4f}" for e in r["ladder"]) + f"{r['uniform_strain_max_abs_relative']:>10.2e}", flush=True)
json.dump(rows, open(T / "TRACK_A_ANALYSIS_V3.json", "w"), indent=1, default=float); print("written")
