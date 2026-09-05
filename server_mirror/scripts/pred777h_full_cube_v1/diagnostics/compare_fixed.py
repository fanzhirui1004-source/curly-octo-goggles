#!/usr/bin/env python3
"""The repaired pipeline against the old labels: cap needles, slivers, lambda_max, the top mode's localisation, and
the uniform-strain stiffnesses (physics) per cell and tier.  Usage: compare_fixed.py tag:old_label_dir[:reference_dir] ..."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from label_error import load_label, uniform_strain_stiffness

import os
T = Path("/root/autodl-tmp/_claude_diag/tonight"); W = Path(os.environ.get("FIXED_DIR", str(T / "fixed")))


def top_mode(S):
    w, U = np.linalg.eigh(S); n = S.shape[0] // 3
    part = np.sort(np.linalg.norm(U[:, -1].reshape(n, 3), axis=1) ** 2)[::-1]
    return float(w[-1]), float(part[:4].sum()), int((part > 0.01).sum())


def common(a, b):
    ib = {k: i for i, k in enumerate(b["ids"])}; keys = [k for k in a["ids"] if k in ib]
    ka = np.asarray([a["ids"].index(k) for k in keys]); kb = np.asarray([ib[k] for k in keys])
    da = np.repeat(3 * ka, 3) + np.tile(np.arange(3), len(ka)); db = np.repeat(3 * kb, 3) + np.tile(np.arange(3), len(kb))
    return da, db, ka, kb


print(f"{'tag':<11}{'status':<8}{'q':>6}{'needles':>8}{'capMinAng':>10}{'dih0':>8}{'dih1':>8}{'<5deg':>7}{'lmax':>9}{'lmax old':>9}{'top4':>7}{'top4 old':>9}{'vs old':>9}{'vs ref':>9}{'old vs ref':>11}")
for arg in sys.argv[1:]:
    parts = arg.split(":"); tag, old = parts[0], Path(parts[1]); ref = Path(parts[2]) if len(parts) > 2 else None
    d = W / tag
    if not (d / "LABEL_RECEIPT.json").exists():
        print(f"{tag:<11}missing"); continue
    r = json.loads((d / "LABEL_RECEIPT.json").read_text())
    if r.get("status") != "PASS" or not (d / "FIXED_PORT_SCHUR_TET10.npz").exists():
        print(f"{tag:<11}{r.get('status')}"); continue
    cn = r["surface"].get("cap_needles", {}); ss = r["mesh"].get("sliver_smooth") or {}
    new = load_label(d); o = load_label(old)
    lam, t4, n1 = top_mode(new["S"]); lam0, t40, _ = top_mode(o["S"])
    da, db, ka, kb = common(new, o)
    Cn = uniform_strain_stiffness(new["S"][np.ix_(da, da)], new["xyz"][ka]); Co = uniform_strain_stiffness(o["S"][np.ix_(db, db)], o["xyz"][kb])
    vs_old = float(np.abs(Cn / Co - 1).max())
    vs_ref = old_vs_ref = float("nan")
    if ref is not None and (ref / "FIXED_PORT_SCHUR_TET10.npz").exists():
        rf = load_label(ref); da2, db2, ka2, kb2 = common(new, rf)
        Cr = uniform_strain_stiffness(rf["S"][np.ix_(db2, db2)], rf["xyz"][kb2]); Cn2 = uniform_strain_stiffness(new["S"][np.ix_(da2, da2)], new["xyz"][ka2])
        vs_ref = float(np.abs(Cn2 / Cr - 1).max())
        da3, db3, ka3, kb3 = common(o, rf); Cr3 = uniform_strain_stiffness(rf["S"][np.ix_(db3, db3)], rf["xyz"][kb3]); Co3 = uniform_strain_stiffness(o["S"][np.ix_(da3, da3)], o["xyz"][ka3])
        old_vs_ref = float(np.abs(Co3 / Cr3 - 1).max()); lamr = top_mode(rf["S"])[0]
    print(f"{tag:<11}{'PASS':<8}{new['q']:>6}{cn.get('remaining', -1):>8}{cn.get('min_angle_degrees', float('nan')):>10.3f}"
          f"{(ss.get('before') or {}).get('min_dihedral_degrees', float('nan')):>8.3f}{r['mesh']['min_dihedral_degrees']:>8.3f}{r['mesh']['tets_below_5deg']:>7}"
          f"{lam:>9.4f}{lam0:>9.4f}{t4:>7.3f}{t40:>9.3f}{vs_old:>9.1e}{vs_ref:>9.1e}{old_vs_ref:>11.1e}"
          + (f"   lmax/ref {lam / lamr:.3f} (old {lam0 / lamr:.3f})" if ref is not None and not np.isnan(vs_ref) else ""))
