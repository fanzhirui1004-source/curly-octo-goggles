import json
from pathlib import Path
import numpy as np
R = json.loads(Path("/root/autodl-tmp/_claude_diag/production/STIFFNESS_FULL.json").read_text())
ok = [r for r in R if "C" in r]
C = np.array([r["C"] for r in ok])            # (n, 6): xx yy zz yz xz xy
vol = np.array([r["volume"] for r in ok])
cut = np.array([r["cut"] for r in ok])
lmax = np.array([r["lmax"] for r in ok])
NAMES = ["xx", "yy", "zz", "yz", "xz", "xy"]

print("cells: %d (cut %d, uncut %d)" % (len(ok), int(cut.sum()), int((~cut).sum())))
print("negative stiffness entries (must be 0):", int((C < 0).sum()),
      " min over all =", float(C.min()))
print()
print("=== the six KUBC apparent stiffnesses, whole population")
for i, n in enumerate(NAMES):
    a = C[:, i]
    print("  C_%-3s p0=%.4g p5=%.4g p50=%.4g p95=%.4g p100=%.4g" % (
        n, *[np.percentile(a, p) for p in (0, 5, 50, 95, 100)]))
print()
print("=== uncut cells: the density power law  (KUBC is an upper bound; the EXPONENT is the physics)")
u = ~cut
Cn = C[u][:, :3].mean(axis=1)      # mean normal stiffness
rho = vol[u]
m = (Cn > 0) & (rho > 0)
p = np.polyfit(np.log(rho[m]), np.log(Cn[m]), 1)
pred = np.exp(np.polyval(p, np.log(rho[m])))
r2 = 1 - ((np.log(Cn[m]) - np.log(pred)) ** 2).sum() / ((np.log(Cn[m]) - np.log(Cn[m]).mean()) ** 2).sum()
print("  n=%d   C_normal = %.4g * rho^%.4f     R^2(log) = %.5f" % (m.sum(), np.exp(p[1]), p[0], r2))
print("  a sheet TPMS is stretch dominated: exponent near 1 expected, 2 would mean bending dominated")
print("  reference in the contract: C = 1.755 rho (Hao 2023 Table 2.2, periodic homogenisation)")
print("  rho range covered: %.4g .. %.4g" % (rho.min(), rho.max()))
# restricted to the dense half, where a KUBC upper bound is tightest
hi = m & (rho > np.median(rho))
ph = np.polyfit(np.log(rho[hi]), np.log(Cn[hi]), 1)
print("  dense half only (rho > %.3f): exponent %.4f, prefactor %.4g" % (np.median(rho), ph[0], np.exp(ph[1])))
print()
print("=== shear vs normal")
sh = C[:, 3:].mean(axis=1); no = C[:, :3].mean(axis=1)
r = sh / np.maximum(no, 1e-300)
print("  mean shear / mean normal: p5=%.4g p50=%.4g p95=%.4g" % tuple(np.percentile(r, [5, 50, 95])))
print()
print("=== anisotropy of the uncut cells (they carry a trilinear tau, so they are NOT isotropic)")
n3 = C[u][:, :3]
aniso = n3.max(axis=1) / np.maximum(n3.min(axis=1), 1e-300)
print("  max/min normal stiffness: p50=%.4g p95=%.4g p100=%.4g" % tuple(np.percentile(aniso, [50, 95, 100])))
print()
print("=== does lambda_max track the stiffness, or the mesh?  (the STEP8 question, restated)")
def sp(a, b):
    return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])
print("  spearman(lambda_max, C_normal_mean) = %+.3f" % sp(lmax, no))
print("  spearman(lambda_max, volume)        = %+.3f" % sp(lmax, vol))
print("  spearman(C_normal_mean, volume)     = %+.3f" % sp(no, vol))
out = {"n": len(ok), "power_law": {"prefactor": float(np.exp(p[1])), "exponent": float(p[0]), "r2_log": float(r2),
                                   "n_uncut": int(m.sum()), "rho_min": float(rho.min()), "rho_max": float(rho.max()),
                                   "dense_half_exponent": float(ph[0])},
       "negative_entries": int((C < 0).sum()),
       "shear_over_normal_p50": float(np.percentile(r, 50)),
       "anisotropy_p50": float(np.percentile(aniso, 50)),
       "C_percentiles": {n: [float(np.percentile(C[:, i], q)) for q in (0, 5, 50, 95, 100)] for i, n in enumerate(NAMES)}}
Path("/root/autodl-tmp/_claude_diag/production/STIFFNESS_SUMMARY.json").write_text(json.dumps(out, indent=2))
