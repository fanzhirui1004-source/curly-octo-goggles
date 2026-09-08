import json, glob
from pathlib import Path
import numpy as np
L = Path("/root/autodl-tmp/_claude_diag/production/labels")
P = []
for f in sorted(glob.glob(str(L / "*" / "LABEL_RECEIPT.json"))):
    d = json.loads(Path(f).read_text())
    if d.get("status") == "PASS": P.append(d)
tm = np.array([d["schur"]["top_mode_mass_on_4_nodes"] for d in P])
lm = np.array([d["schur"]["max_eigenvalue"] for d in P])
vol = np.array([d["mesh"]["material_volume"] for d in P])
med = np.median(lm)
# the STEP8 signature is a LOCALISED top mode together with an inflated lambda_max that the volume does not explain.
# regress log lambda_max on log volume, then look at the residual: that is "inflated for its density".
c = np.polyfit(np.log(vol), np.log(lm), 1)
resid = np.log(lm) - np.polyval(c, np.log(vol))
s = resid.std()
print("log lambda_max ~ %.4f log rho + %.4f ; residual sd = %.4f" % (c[0], c[1], s))
for tmin, k in ((0.99, 3), (0.99, 2), (0.95, 3), (0.95, 2)):
    m = (tm >= tmin) & (resid > k * s)
    print("  flag: top_mode >= %.2f AND lambda_max residual > %d sd  ->  %d cells (%.3f%%)" % (
        tmin, k, int(m.sum()), 100 * m.mean()))
m = (tm >= 0.99) & (resid > 2 * s)
print("  the flagged cells:", [(P[i]["case_id"], round(float(tm[i]), 4), round(float(resid[i] / s), 2),
                                round(float(P[i]["mesh"]["min_dihedral_degrees"]), 4)) for i in np.where(m)[0]])
print()
print("  residual > 3sd alone (regardless of top mode): %d" % int((resid > 3 * s).sum()))
print("  top_mode >= 0.99 alone: %d" % int((tm >= 0.99).sum()))
