#!/usr/bin/env python3
"""Track A: the single-cell label error, from the labels produced tonight.

A1  fine-mesh convergence, carrier fixed: fast and production against reference, per cell, by the generalized
    eigenvalues of (S_h, S_ref) on the reference's range.  Load-free and sharp.
A2  carrier resolution: 16 and 64 against 32, compared through the six uniform-strain apparent stiffnesses, because
    the port SPACE changes and only fields that belong to every carrier (the affine ones) are comparable.
A3  reproducibility: the same settings at a different thread count.
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from label_error import compare, compare_across_carriers, load_label, uniform_strain_stiffness

A = Path("/root/autodl-tmp/_claude_diag/tonight/A")
T = Path("/root/autodl-tmp/_claude_diag/tonight")
cells = [c.strip() for c in (T / "trackA.txt").read_text().split() if c.strip()]
cellsA2 = [c.strip() for c in (T / "trackA2.txt").read_text().split() if c.strip()]
cellsA3 = [c.strip() for c in (T / "trackA3.txt").read_text().split() if c.strip()]
have = lambda p: (A / p / "LABEL_RECEIPT.json").exists() and (A / p / "FIXED_PORT_SCHUR_TET10.npz").exists()
status = lambda p: json.loads((A / p / "LABEL_RECEIPT.json").read_text()).get("status") if (A / p / "LABEL_RECEIPT.json").exists() else "-"

out: dict = {"A1": {}, "A2": {}, "A3": {}, "missing": []}
print("=" * 118)
print("A1  fine-mesh convergence (carrier 1/32 fixed): generalized eigenvalues of S_h relative to S_reference")
print("    lambda in [lo, hi] bounds the relative strain energy over EVERY port displacement; bound = max|lambda-1|")
print("=" * 118)
print(f"{'cell':<17}{'preset':<12}{'q':>7}{'modes':>7}{'lambda_lo':>11}{'lambda_hi':>11}{'bound':>9}{'uniform strain max':>20}{'Frob rel':>10}")
for c in cells:
    ref = f"A1_reference_{c}"
    if not have(ref):
        out["missing"].append((ref, status(ref))); print(f"{c:<17}reference missing ({status(ref)})"); continue
    for preset in ("fast", "production"):
        tag = f"A1_{preset}_{c}"
        if not have(tag):
            out["missing"].append((tag, status(tag))); print(f"{c:<17}{preset:<12}missing ({status(tag)})"); continue
        r = compare(A / tag, A / ref); sp = r.get("spectrum", {})
        if "lambda_min" not in sp:
            print(f"{c:<17}{preset:<12}{sp.get('error', 'no spectrum')}"); continue
        out["A1"].setdefault(c, {})[preset] = r
        print(f"{c:<17}{preset:<12}{r['q_h']:>7}{sp['kept_modes']:>7}{sp['lambda_min']:>11.5f}{sp['lambda_max']:>11.5f}"
              f"{sp['energy_error_bound']:>9.4f}{r['uniform_strain_max_abs_relative']:>20.2e}{r['frobenius_relative']:>10.2e}")
        if r["alignment"]["only_a"] or r["alignment"]["only_b"]:
            print(f"{'':<17}{'':<12}active-set differs: only_h {r['alignment']['only_a']}, only_ref {r['alignment']['only_b']}")

print()
print("=" * 118)
print("A2  carrier resolution (production mesh fixed): the six uniform-strain apparent stiffnesses against carrier 1/32")
print("    the port space itself changes, so only affine fields, which belong to every carrier, are comparable")
print("=" * 118)
print(f"{'cell':<17}{'carrier':<9}{'q':>7}{'q/q32':>7}{'Exx':>10}{'Eyy':>10}{'Ezz':>10}{'Gyz':>10}{'Gxz':>10}{'Gxy':>10}{'max|rel|':>10}")
for c in cellsA2:
    base = f"A1_production_{c}"
    if not have(base):
        out["missing"].append((base, status(base))); print(f"{c:<17}carrier-32 baseline missing"); continue
    b = load_label(A / base); Cb = uniform_strain_stiffness(b["S"], b["xyz"])
    print(f"{c:<17}{'32':<9}{b['q']:>7}{1.0:>7.2f}" + "".join(f"{v:>10.4g}" for v in Cb) + f"{0.0:>10.1e}")
    for n in (16, 64):
        tag = f"A2_c{n}_{c}"
        if not have(tag):
            out["missing"].append((tag, status(tag))); print(f"{c:<17}{str(n):<9}missing ({status(tag)})"); continue
        r = compare_across_carriers(A / tag, A / base); out["A2"].setdefault(c, {})[n] = r
        rel = np.asarray(r["uniform_strain_relative"], float)
        print(f"{c:<17}{str(n):<9}{r['q_h']:>7}{r['q_h'] / max(b['q'], 1):>7.2f}" + "".join(f"{v:>10.4g}" for v in r["uniform_strain_h"]) + f"{r['uniform_strain_max_abs_relative']:>10.2e}")

print()
print("=" * 118)
print("A3  reproducibility: the same geometry and settings at a different thread count")
print("=" * 118)
for c in cellsA3:
    a, b = f"A1_production_{c}", f"A3_t12_{c}"
    if not (have(a) and have(b)):
        out["missing"].append((b, status(b))); print(f"{c:<17}missing ({status(b)})"); continue
    r = compare(A / b, A / a); sp = r.get("spectrum", {}); out["A3"][c] = r
    print(f"{c:<17}q {r['q_h']:>6} vs {r['q_ref']:<6} lambda [{sp.get('lambda_min', float('nan')):.9f}, {sp.get('lambda_max', float('nan')):.9f}]"
          f"  Frobenius rel {r['frobenius_relative']:.2e}  uniform strain max {r['uniform_strain_max_abs_relative']:.2e}")

if out["missing"]:
    print(f"\nmissing or failed: {len(out['missing'])}")
    for tag, st in out["missing"][:20]: print("   ", tag, st)
json.dump(out, open(T / "TRACK_A_ANALYSIS.json", "w"), indent=1, default=float)
print("\nwritten", T / "TRACK_A_ANALYSIS.json")
