"""Two different questions were being answered by one flag.

  has_cut_plane  : the cell's geometry carries a world cut plane at all
  cut_is_a_port  : 'cut_0' is in active_ports, i.e. the cut face actually carries material

A cell can have a cut plane whose face touches no material.  The first census used the second flag for both, which
(a) put 52 cut cells into the 'uncut' power-law fit and (b) reported the cut-face width statistics over the wrong
subset, hiding the thin faces STEP8 measured.
"""
import json, glob
from pathlib import Path
import numpy as np
L = Path("/root/autodl-tmp/_claude_diag/production/labels")
R = {r["case"]: r for r in json.loads(Path("/root/autodl-tmp/_claude_diag/production/STIFFNESS_FULL.json").read_text()) if "C" in r}
rows = []
for f in sorted(glob.glob(str(L / "*" / "LABEL_RECEIPT.json"))):
    d = json.loads(Path(f).read_text())
    if d.get("status") != "PASS": continue
    cf = (d.get("geometry", {}) or {}).get("cut_face")
    rows.append({"case": d["case_id"], "has_cut": cf is not None,
                 "cut_port": "cut_0" in d["fixed_port"]["active_ports"],
                 "width_h": (cf or {}).get("min_width_over_carrier_spacing"),
                 "vol": d["mesh"]["material_volume"], "C": np.array(R[d["case_id"]]["C"])})
n = len(rows)
hc = sum(r["has_cut"] for r in rows); cp = sum(r["cut_port"] for r in rows)
print("PASS %d :  has a cut plane %d,  cut is a load-bearing port %d,  cut plane but no material on it %d,  truly uncut %d"
      % (n, hc, cp, hc - cp, n - hc))

w = np.array([r["width_h"] for r in rows if r["has_cut"]], float)
print()
print("=== cut-face minimum width, over ALL PASS cells that have a cut plane (n=%d)" % len(w))
print("   p0=%.4g p1=%.4g p5=%.4g p50=%.4g p95=%.4g p100=%.4g   (units: carrier spacings)"
      % tuple(np.percentile(w, [0, 1, 5, 50, 95, 100])))
print("   below one carrier spacing: %d ; below 0.5: %d" % (int((w < 1).sum()), int((w < 0.5).sum())))

print()
print("=== density power law, on TRULY UNCUT cells only")
u = [r for r in rows if not r["has_cut"]]
Cn = np.array([r["C"][:3].mean() for r in u]); rho = np.array([r["vol"] for r in u])
p = np.polyfit(np.log(rho), np.log(Cn), 1)
pred = np.exp(np.polyval(p, np.log(rho)))
r2 = 1 - ((np.log(Cn) - np.log(pred)) ** 2).sum() / ((np.log(Cn) - np.log(Cn).mean()) ** 2).sum()
print("   n=%d  C_normal = %.4g * rho^%.4f   R^2(log)=%.5f   rho in [%.4g, %.4g]"
      % (len(u), np.exp(p[1]), p[0], r2, rho.min(), rho.max()))
hi = rho > np.median(rho)
ph = np.polyfit(np.log(rho[hi]), np.log(Cn[hi]), 1)
print("   dense half (rho > %.3f): exponent %.4f prefactor %.4g" % (np.median(rho), ph[0], np.exp(ph[1])))
# and for comparison, the 52 that have a cut plane carrying no material
q = [r for r in rows if r["has_cut"] and not r["cut_port"]]
if q:
    Cq = np.array([r["C"][:3].mean() for r in q]); rq = np.array([r["vol"] for r in q])
    print("   (the 52 cut-plane-no-material cells sit at rho in [%.4g, %.4g]; they were wrongly in the previous fit)"
          % (rq.min(), rq.max()))
json.dump({"pass": n, "has_cut_plane": hc, "cut_is_port": cp, "cut_plane_no_material": hc - cp, "truly_uncut": n - hc,
           "cut_width_h": {f"p{k}": float(np.percentile(w, k)) for k in (0, 1, 5, 50, 95, 100)},
           "cut_width_below_1h": int((w < 1).sum()),
           "power_law": {"prefactor": float(np.exp(p[1])), "exponent": float(p[0]), "r2_log": float(r2),
                         "n": len(u), "rho_min": float(rho.min()), "rho_max": float(rho.max()),
                         "dense_half_exponent": float(ph[0])}},
          open("/root/autodl-tmp/_claude_diag/production/SPLIT_FIX.json", "w"), indent=2)
