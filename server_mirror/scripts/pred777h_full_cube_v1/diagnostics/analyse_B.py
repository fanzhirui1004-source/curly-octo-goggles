#!/usr/bin/env python3
"""Track B: the statistics the production plan and the training-target decisions need, plus track C, the external
stiffness benchmark.

B  guards, zero rows, operator scale, storage and throughput over the 300-cell stratified production sample.
C  the KUBC apparent stiffness of the uncut cells against the material volume fraction.  Our operator under an affine
   port displacement is the kinematic-uniform-boundary-condition estimate, an UPPER bound on the effective modulus;
   published sheet-TPMS moduli are usually periodic homogenisation, so the level is not directly comparable but the
   POWER-LAW EXPONENT is: a sheet TPMS is stretch dominated and should give an exponent near 1, not the 2 of a
   bending-dominated network.
"""
from __future__ import annotations

import glob, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from label_error import load_label, uniform_strain_stiffness

T = Path("/root/autodl-tmp/_claude_diag/tonight"); B = T / "B"
rows = []
for f in sorted(glob.glob(str(B / "*" / "LABEL_RECEIPT.json"))):
    d = json.loads(Path(f).read_text()); case = Path(f).parent.name
    rows.append({"case": case, "status": d.get("status"), "receipt": d, "dir": Path(f).parent})
done = [r for r in rows if r["status"] == "PASS"]
print(f"track B: {len(rows)} receipts, PASS {len(done)}, "
      f"other {sorted({r['status'] for r in rows if r['status'] != 'PASS'})}")

def col(key, where=lambda r: r["receipt"]):
    out = []
    for r in done:
        d = where(r)
        for k in key.split("."):
            d = (d or {}).get(k) if isinstance(d, dict) else None
        if d is not None: out.append(float(d))
    return np.asarray(out)

def line(name, v, fmt="%.4g"):
    if not len(v): print(f"  {name:<34} (none)"); return
    print(f"  {name:<34} min {fmt % v.min():>12}  med {fmt % np.median(v):>12}  p95 {fmt % np.percentile(v, 95):>12}  max {fmt % v.max():>12}")

print("\n=== guards ===")
line("rigid-body residual (relative)", col("schur.rigid_residual_relative"), "%.2e")
line("min eigenvalue", col("schur.min_eigenvalue"), "%.2e")
line("min eigenvalue / max", np.abs(col("schur.min_eigenvalue")) / np.maximum(col("schur.max_eigenvalue"), 1e-300), "%.2e")
line("inactive block max", col("schur.inactive_block_max"), "%.2e")
line("nonmanifold volume facets", col("mesh.nonmanifold_facets"), "%.0f")
line("min dihedral (deg)", col("mesh.min_dihedral_degrees"), "%.4f")
line("tets below 5 deg", col("mesh.tets_below_5deg"), "%.0f")
line("surface min quality", col("surface.surface_quality.min"), "%.2e")
alg = col("mesh.algorithm3d"); print(f"  {'algorithm3d values':<34} {sorted(set(int(x) for x in alg))}")

print("\n=== operator scale and size ===")
q = col("schur.q_active"); line("q_active", q, "%.0f")
emax = col("schur.max_eigenvalue"); line("max eigenvalue (operator scale)", emax, "%.4g")
print(f"  {'operator scale dynamic range':<34} max/min = {emax.max() / max(emax.min(), 1e-300):.1f}")
vol = col("mesh.material_volume"); line("material volume", vol, "%.4f")

print("\n=== zero rows (the training-target decision) ===")
zr = []
for r in done:
    npz = r["dir"] / "FIXED_PORT_SCHUR_TET10.npz"
    if not npz.exists(): continue
    try:
        lab = load_label(r["dir"])
    except Exception:
        continue
    S = lab["S"]; n = S.shape[0] // 3
    rown = np.linalg.norm(S.reshape(n, 3, -1), axis=(1, 2))
    zero = rown <= 1e-12 * rown.max()
    zr.append({"case": r["case"], "nodes": n, "zero": int(zero.sum()), "frac": float(zero.mean()),
               "scale": float(rown.max()), "cut": r["receipt"]["geometry"]["cut_plane"] is not None})
if zr:
    fr = np.asarray([z["frac"] for z in zr]); line("zero-row fraction of active nodes", fr, "%.4f")
    cutf = np.asarray([z["frac"] for z in zr if z["cut"]]); unc = np.asarray([z["frac"] for z in zr if not z["cut"]])
    line("  cut cells", cutf, "%.4f"); line("  uncut cells", unc, "%.4f")
    print(f"  {'labels inspected':<34} {len(zr)} (cut {int(sum(z['cut'] for z in zr))}, uncut {int(sum(not z['cut'] for z in zr))})")

print("\n=== storage and throughput ===")
mb = col("schur.storage.bytes") / 1e6; line("label operator (MB)", mb, "%.1f")
tt = col("timing.total"); line("wall time per cell (s)", tt, "%.0f")
for k in ("timing.surface_and_mesh", "timing.fixed_port", "timing.schur"):
    line(k.split(".")[-1] + " (s)", col(k), "%.0f")
if len(mb):
    print(f"  {'projected for 1949 OK cells':<34} storage {mb.mean() * 1949 / 1000:.0f} GB, "
          f"machine time {tt.mean() * 1949 / 3600:.0f} core-batches, wall at 8 concurrent {tt.mean() * 1949 / 8 / 3600:.1f} h")
    tet = col("mesh.tet_count"); dof = 3 * col("mesh.node_count")
    if len(tet): print(f"  {'memory coefficient (GB per 1000 dof)':<34} {np.median(mb / 1000 / (dof / 1000)):.4f} (operator only)")

print("\n=== C: external stiffness benchmark, uncut cells ===")
pts = []
for r in done:
    if r["receipt"]["geometry"]["cut_plane"] is not None: continue
    try:
        lab = load_label(r["dir"])
    except Exception:
        continue
    C = uniform_strain_stiffness(lab["S"], lab["xyz"])
    v = r["receipt"]["mesh"]["material_volume"]
    pts.append({"case": r["case"], "rho": float(v), "E": float(np.mean(C[:3])), "G": float(np.mean(C[3:])),
                "tau_mean": float(np.mean(r["receipt"]["geometry"]["tau_corners"]))})
if len(pts) >= 6:
    rho = np.asarray([p["rho"] for p in pts]); E = np.asarray([p["E"] for p in pts]); G = np.asarray([p["G"] for p in pts])
    keep = (rho > 1e-6) & (E > 0)
    for name, Y in (("normal (Exx,Eyy,Ezz mean)", E), ("shear (Gyz,Gxz,Gxy mean)", G)):
        k = keep & (Y > 0)
        a, b = np.polyfit(np.log(rho[k]), np.log(Y[k]), 1)
        resid = np.log(Y[k]) - (a * np.log(rho[k]) + b)
        print(f"  {name:<28} exponent {a:.3f}   coefficient {np.exp(b):.4g}   R2 {1 - resid.var() / np.log(Y[k]).var():.4f}   n {int(k.sum())}")
    print(f"  {'volume fraction range':<28} {rho.min():.3f} to {rho.max():.3f}")
    print("  reading: a sheet TPMS is stretch dominated, so the exponent should sit near 1; near 2 would mean")
    print("  bending dominated.  The COEFFICIENT is a KUBC (affine port displacement) value, an upper bound on the")
    print("  periodic effective modulus, so it is not directly comparable to published periodic data.")
else:
    print(f"  only {len(pts)} uncut labels so far, need at least 6 for the fit")
json.dump({"zero_rows": zr, "benchmark": pts}, open(T / "TRACK_B_ANALYSIS.json", "w"), indent=1, default=float)
print("\nwritten", T / "TRACK_B_ANALYSIS.json")
