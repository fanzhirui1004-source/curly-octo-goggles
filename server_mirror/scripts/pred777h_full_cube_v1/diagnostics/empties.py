import json, glob
from pathlib import Path
import numpy as np
L = Path("/root/autodl-tmp/_claude_diag/production/labels")
E, P, D = [], [], []
for f in sorted(glob.glob(str(L / "*" / "LABEL_RECEIPT.json"))):
    d = json.loads(Path(f).read_text())
    (E if d.get("status") == "EMPTY" else P if d.get("status") == "PASS" else D).append(d)
def w(d):
    return (d.get("geometry", {}).get("cut_face") or {}).get("min_width_over_carrier_spacing")
def v(d):
    return (d.get("geometry", {}) or {}).get("material_volume_estimate")
ew = [w(d) for d in E if w(d) is not None]
print("EMPTY: %d  (with a cut face: %d)" % (len(E), len(ew)))
if ew:
    print("  their cut-face min width / carrier spacing: p0=%.4g p50=%.4g p100=%.4g" % tuple(np.percentile(ew, [0, 50, 100])))
    print("  how many below one carrier spacing: %d" % int((np.asarray(ew) < 1.0).sum()))
print("  their material volume estimate: p0=%.4g p50=%.4g p100=%.4g" % tuple(
    np.percentile([x for x in (v(d) for d in E) if x is not None], [0, 50, 100])))
pw = [w(d) for d in P if w(d) is not None]
print("PASS cut cells: %d, min width/h p0=%.4g, below 1h: %d" % (len(pw), min(pw), int((np.asarray(pw) < 1).sum())))
print("GEOMETRY_DEGENERATE: %s" % [(d["case_id"], w(d), v(d)) for d in D])
print()
print("=== the thin cut faces STEP8 listed: where did they land?")
THIN = sorted([(w(d), d["case_id"], d["status"]) for d in E + P + D if w(d) is not None])[:15]
for t in THIN: print("   width/h=%.4g  %-16s %s" % t)
