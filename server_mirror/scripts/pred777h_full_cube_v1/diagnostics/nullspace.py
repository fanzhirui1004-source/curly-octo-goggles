#!/usr/bin/env python3
"""The label operator's null space: is it exactly 6 * components + 3 * unsupported nodes, or is there a third family?

The geometric active carrier set is deliberately a superset of any mesh's support (mesh independence of q), so an
active node the material never reaches gives an exact zero row AND zero column.  With S = [[S_ss, 0], [0, 0]] the
null space is Null(S_ss) + 3 * n_zero, and Null(S_ss) is 6 per connected piece of retained material that touches a
port.  This checks that count numerically against the eigenvalues.
"""
from __future__ import annotations
import glob, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from label_error import load_label

T = Path("/root/autodl-tmp/_claude_diag/tonight")
dirs = sorted(glob.glob(str(T / "B" / "*" / "LABEL_RECEIPT.json")))
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 40      # a full eigendecomposition per cell: sample, do not sweep
if len(dirs) > LIMIT:
    step = len(dirs) / LIMIT; dirs = [dirs[int(i * step)] for i in range(LIMIT)]
print(f"sampling {len(dirs)} labels for the eigen-decomposition", flush=True)
print(f"{'case':<20}{'cut':>4}{'nodes':>6}{'zero':>6}{'comps':>6}{'pred':>7}" + "".join(f"{('<1e%d' % -e):>8}" for e in (14, 12, 10, 8, 6, 4)) + f"{'gap@pred':>10}")
print("  pred = 6*components + 3*unsupported.  The columns count eigenvalues below that fraction of lambda_max:")
print("  a clean null space shows the same count in every column; a staircase means near-null modes with tiny support.")
rows = []; mismatch = 0
for f in dirs:
    d = Path(f).parent; r = json.loads(Path(f).read_text())
    if r.get("status") != "PASS": continue
    try: lab = load_label(d)
    except Exception: continue
    S = lab["S"]; n = S.shape[0] // 3
    rown = np.linalg.norm(S.reshape(n, 3, -1), axis=(1, 2))
    zero = rown <= 1e-12 * rown.max(); nz = int(zero.sum())
    comps = int(((r.get("surface") or {}).get("components") or {}).get("components", 1))
    dropped = int(((r.get("surface") or {}).get("components") or {}).get("dropped", 0))
    kept = max(comps - dropped, 1)
    w = np.linalg.eigvalsh(S); wmax = w[-1]
    counts = {e: int((w <= 10.0 ** (-e) * wmax).sum()) for e in (14, 12, 10, 8, 6, 4)}
    predicted = 6 * kept + 3 * nz
    gap = float(w[predicted] / wmax) if predicted < len(w) else float("nan")
    ok = counts[10] == predicted
    mismatch += (not ok)
    rows.append({"case": d.name, "cut": r["geometry"]["cut_plane"] is not None, "nodes": n, "zero": nz,
                 "components": comps, "kept_components": kept, "predicted": predicted, "counts": counts,
                 "match": bool(ok), "eigenvalue_at_predicted_index_over_max": gap})
    print(f"{d.name:<20}{'Y' if rows[-1]['cut'] else 'N':>4}{n:>6}{nz:>6}{kept:>6}{predicted:>7}" + "".join(f"{counts[e]:>8}" for e in (14, 12, 10, 8, 6, 4)) + f"{gap:>10.1e}", flush=True)
if rows:
    z = np.asarray([r["zero"] / r["nodes"] for r in rows]); g = np.asarray([r["eigenvalue_at_predicted_index_over_max"] for r in rows])
    cut = np.asarray([r["cut"] for r in rows])
    print(f"\ncells {len(rows)}, null-space formula matches {len(rows) - mismatch}/{len(rows)}")
    print(f"zero-row fraction: med {np.median(z):.3f} p95 {np.percentile(z, 95):.3f} max {z.max():.3f}")
    if cut.any(): print(f"  cut cells   med {np.median(z[cut]):.3f} max {z[cut].max():.3f}")
    if (~cut).any(): print(f"  uncut cells med {np.median(z[~cut]):.3f} max {z[~cut].max():.3f}")
    print(f"components > 1: {sum(1 for r in rows if r['kept_components'] > 1)} cells")
    print(f"eigenvalue just above the predicted null dimension, over lambda_max: min {np.nanmin(g):.2e} med {np.nanmedian(g):.2e} max {np.nanmax(g):.2e}")
    ex = np.asarray([r["counts"][10] - r["predicted"] for r in rows])
    print(f"measured (at 1e-10) minus predicted: min {ex.min()} med {int(np.median(ex))} max {ex.max()}; cells with extra modes {int((ex > 0).sum())}")
    stair = np.asarray([r["counts"][4] - r["counts"][14] for r in rows])
    print(f"staircase width (count below 1e-4 minus below 1e-14): min {stair.min()} med {int(np.median(stair))} max {stair.max()}")
    print("  a wide staircase means the operator has a continuum of nearly-null modes, so any threshold is arbitrary")
json.dump(rows, open(T / "NULLSPACE_CHECK.json", "w"), indent=1, default=float)
print("written", T / "NULLSPACE_CHECK.json")
