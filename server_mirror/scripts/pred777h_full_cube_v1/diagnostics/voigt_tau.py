import json, glob
from pathlib import Path
import numpy as np
L = Path("/root/autodl-tmp/_claude_diag/production/labels")
R = json.loads(Path("/root/autodl-tmp/_claude_diag/production/STIFFNESS_FULL.json").read_text())
ok = {r["case"]: r for r in R if "C" in r}
C1111_SOLID = 1.0 * (1 - 0.3) / ((1 + 0.3) * (1 - 2 * 0.3))     # E=1, nu=0.3
print("solid C_1111 (E=1, nu=0.3) = %.6f" % C1111_SOLID)

rows = []
for f in sorted(glob.glob(str(L / "*" / "LABEL_RECEIPT.json"))):
    d = json.loads(Path(f).read_text())
    if d.get("status") != "PASS": continue
    c = d["case_id"]
    if c not in ok: continue
    rows.append((c, np.asarray(d["geometry"]["tau_corners"], float), np.array(ok[c]["C"]),
                 ok[c]["volume"], ok[c]["cut"]))

print()
print("=== Voigt bound check: C_ii <= rho * C_1111_solid, for every cell and every normal direction")
viol = 0; ratios = []
for c, tau, C, vol, cut in rows:
    b = vol * C1111_SOLID
    r = C[:3].max() / b
    ratios.append(r)
    if r > 1.0: viol += 1; print("   VIOLATION %-16s rho=%.4g C_max=%.4g bound=%.4g ratio=%.4f" % (c, vol, C[:3].max(), b, r))
ratios = np.array(ratios)
print("  violations: %d / %d" % (viol, len(rows)))
print("  C_max / (rho * C_solid): p5=%.4f p50=%.4f p95=%.4f p100=%.4f" % tuple(np.percentile(ratios, [5, 50, 95, 100])))
print("  (the contract's reference line C = 1.755 rho would sit at ratio %.3f -- above the Voigt bound)"
      % (1.755 * 0.228 / (0.228 * C1111_SOLID)))

print()
print("=== what the thickness field actually covers")
tau = np.array([r[1] for r in rows])
spread = tau.max(axis=1) - tau.min(axis=1)
print("  tau corner value : p0=%.4g p5=%.4g p50=%.4g p95=%.4g p100=%.4g" % tuple(np.percentile(tau, [0, 5, 50, 95, 100])))
print("  per-cell spread (max-min over the 8 corners):")
print("     p0=%.4g p25=%.4g p50=%.4g p75=%.4g p95=%.4g p100=%.4g" % tuple(np.percentile(spread, [0, 25, 50, 75, 95, 100])))
print("  cells with spread < 0.01 (effectively uniform tau): %d / %d" % (int((spread < 0.01).sum()), len(spread)))
print("  cells with spread < 0.05: %d" % int((spread < 0.05).sum()))
uncut = np.array([not r[4] for r in rows])
print("  uncut cells with spread < 0.01: %d / %d" % (int((spread[uncut] < 0.01).sum()), int(uncut.sum())))
Cn = np.array([r[2][:3] for r in rows])
aniso = Cn.max(axis=1) / Cn.min(axis=1)
m = spread > 0.05
if m.sum():
    print("  among cells with spread > 0.05: anisotropy p50=%.4g p95=%.4g p100=%.4g (n=%d)" % (
        *np.percentile(aniso[m], [50, 95, 100]), int(m.sum())))
