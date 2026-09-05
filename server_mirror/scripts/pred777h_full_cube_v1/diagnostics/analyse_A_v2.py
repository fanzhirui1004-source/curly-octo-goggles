#!/usr/bin/env python3
"""A1 with the revised metric (see label_error2): common support, and the error bound as a function of a declared
energy floor rather than one arbitrary threshold."""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from label_error2 import compare_v2, FLOORS

A = Path("/root/autodl-tmp/_claude_diag/tonight/A"); T = Path("/root/autodl-tmp/_claude_diag/tonight")
cells = [c.strip() for c in (T / "trackA.txt").read_text().split() if c.strip()]
have = lambda p: (A / p / "FIXED_PORT_SCHUR_TET10.npz").exists()

print("=" * 124)
print("A1  fine-mesh convergence, carrier 1/32 fixed, on the carrier nodes BOTH meshes load")
print("    bound = max|lambda - 1| over the reference modes carrying at least `floor` of lambda_max")
print("=" * 124)
head = f"{'cell':<17}{'preset':<11}{'common':>7}{'ref only':>9}{'h only':>7}"
head += "".join(f"{('%.0e' % f):>11}" for f in FLOORS) + f"{'uniform':>10}"
print(head)
print(f"{'':<17}{'':<11}{'nodes':>7}{'':>9}{'':>7}" + "".join(f"{'bound':>11}" for _ in FLOORS) + f"{'strain':>10}")
rows = {}
for c in cells:
    ref = f"A1_reference_{c}"
    if not have(ref):
        print(f"{c:<17}reference missing"); continue
    for preset in ("fast", "production"):
        tag = f"A1_{preset}_{c}"
        if not have(tag):
            print(f"{c:<17}{preset:<11}missing"); continue
        r = compare_v2(A / tag, A / ref); rows.setdefault(c, {})[preset] = r
        if "ladder" not in r:
            print(f"{c:<17}{preset:<11}{r.get('error')}"); continue
        s = r["support"]
        line = f"{c:<17}{preset:<11}{s['common_supported']:>7}{s['supported_in_ref_only']:>9}{s['supported_in_h_only']:>7}"
        line += "".join(f"{(('%.3f' % e['bound']) if 'bound' in e else '-'):>11}" for e in r["ladder"])
        line += f"{r['uniform_strain_max_abs_relative']:>10.2e}"
        print(line, flush=True)

first = next((v for d in rows.values() for v in d.values() if "ladder" in v), None)
if first:
    print("\nenergy share retained at each floor (from the first cell, the others are within a few tenths of a percent):")
    print("   " + "  ".join(f"{('%.0e' % e['floor'])}: {100 * e.get('energy_share', float('nan')):.2f}%" for e in first["ladder"]))
print("\nreading: the bound is the worst relative strain-energy error over the retained subspace, so it grows as the")
print("floor drops and softer directions enter.  The uniform-strain column is the same comparison on the six")
print("homogenisation load cases, which is what an engineer would quote.")
json.dump(rows, open(T / "TRACK_A_ANALYSIS_V2.json", "w"), indent=1, default=float)
print("\nwritten", T / "TRACK_A_ANALYSIS_V2.json")
