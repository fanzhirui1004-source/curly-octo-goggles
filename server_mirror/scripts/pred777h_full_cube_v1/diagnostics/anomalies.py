import glob, json, collections
from pathlib import Path
import numpy as np
L = Path("/root/autodl-tmp/_claude_diag/production/labels")
rows = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(L / "*" / "LABEL_RECEIPT.json")))]
P = [d for d in rows if d.get("status") == "PASS"]

print("=== null-space excess vs surface components (lower bound assumes ONE component: 6 + 3*n_unsupported)")
tab = collections.Counter()
odd = []
for d in P:
    s = d["schur"]["support"]
    e = s["null_dim_at_1e-14"] - s["null_dim_lower_bound"]
    c = d["surface"]["components"]["components"]
    tab[(e, c)] += 1
    if e != 0:
        odd.append((e, c, d["case_id"], s["null_dim_at_1e-14"], s["null_dim_lower_bound"],
                    s["null_dim_at_1e-10"], d["schur"]["max_eigenvalue"], d["mesh"]["min_dihedral_degrees"]))
for k in sorted(tab): print("  excess=%+d components=%d : %d" % (k[0], k[1], tab[k]))
print("  cells with excess != 0:")
for r in sorted(odd): print("   excess=%+d comp=%d %-16s null14=%d lb=%d null10=%d lmax=%.4g mindih=%.4g" % r)

print()
print("=== top mode >= 0.95: what else is true of them?")
bad = sorted(P, key=lambda d: -d["schur"]["top_mode_mass_on_4_nodes"])[:20]
lmax_med = float(np.median([d["schur"]["max_eigenvalue"] for d in P]))
print("  population median lambda_max = %.5g" % lmax_med)
for d in bad:
    print("   %-16s tm=%.5f lmax=%.4g (=%.1fx med) mindih=%.4g tets<5=%d capneedle_rem=%d capmin=%.4g vol=%.4g" % (
        d["case_id"], d["schur"]["top_mode_mass_on_4_nodes"], d["schur"]["max_eigenvalue"],
        d["schur"]["max_eigenvalue"] / lmax_med, d["mesh"]["min_dihedral_degrees"], d["mesh"]["tets_below_5deg"],
        d["surface"]["cap_needles"]["remaining"], d["surface"]["cap_needles"]["min_angle_degrees"],
        d["mesh"]["material_volume"]))

print()
print("=== correlations over the full set")
tm = np.array([d["schur"]["top_mode_mass_on_4_nodes"] for d in P])
lm = np.array([d["schur"]["max_eigenvalue"] for d in P])
md = np.array([d["mesh"]["min_dihedral_degrees"] for d in P])
cn = np.array([d["surface"]["cap_needles"]["remaining"] for d in P])
ca = np.array([d["surface"]["cap_needles"]["min_angle_degrees"] for d in P])
vol = np.array([d["mesh"]["material_volume"] for d in P])
def sp(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])
print("  spearman(top_mode, lambda_max)      = %+.3f" % sp(tm, lm))
print("  spearman(top_mode, min_dihedral)    = %+.3f" % sp(tm, md))
print("  spearman(lambda_max, min_dihedral)  = %+.3f" % sp(lm, md))
print("  spearman(lambda_max, cap_min_angle) = %+.3f" % sp(lm, ca))
print("  spearman(lambda_max, volume)        = %+.3f" % sp(lm, vol))
print("  cap needles remaining: nonzero=%d, distribution p50=%g p95=%g max=%g" % (
    int((cn > 0).sum()), float(np.percentile(cn, 50)), float(np.percentile(cn, 95)), float(cn.max())))
print("  lambda_max above 3x median: %d cells; above 10x: %d" % (int((lm > 3*lmax_med).sum()), int((lm > 10*lmax_med).sum())))
print("  of those above 3x median, median volume = %.4g (population %.4g)" % (
    float(np.median(vol[lm > 3*lmax_med])), float(np.median(vol))))
