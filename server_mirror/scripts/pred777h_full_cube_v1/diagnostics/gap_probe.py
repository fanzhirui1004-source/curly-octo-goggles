"""Where is the spectral gap in the SHIPPED operator?

The receipt's rank_at_1e-14 / null_dim_at_1e-14 are computed inside the pipeline on the float64 operator.  The file
we ship is float32 (float32_rounding_relative ~ 3.3e-8), so a consumer reading the npz cannot resolve a null space at
1e-14.  This finds the threshold that actually separates the rigid modes from the elastic ones in the shipped file.
"""
import json
from pathlib import Path
import numpy as np
L = Path("/root/autodl-tmp/_claude_diag/production/labels")
CASES = ["pop_cut_0473", "pop_cut_0995", "pop_cut_0915", "pop_cut_1142", "pop_uncut_0447", "pop_cut_0355"]
for case in CASES:
    z = np.load(L / case / "FIXED_PORT_SCHUR_TET10.npz")
    q = int(z["q"][0]); S = np.zeros((q, q)); S[np.triu_indices(q)] = z["schur_upper_f32"]
    S = S + S.T - np.diag(np.diag(S))
    sup = np.asarray(z["support_mask"]).astype(bool)
    d = json.loads((L / case / "LABEL_RECEIPT.json").read_text())
    comps = d["surface"]["components"]["components"]
    Ss = S[np.repeat(sup, 3)][:, np.repeat(sup, 3)]
    w = np.linalg.eigvalsh(Ss); lmax = w.max()
    r = np.sort(np.abs(w)) / lmax
    counts = {f"{t:g}": int((np.abs(w) < t * lmax).sum()) for t in (1e-14, 1e-12, 1e-10, 1e-8, 1e-7, 1e-6, 1e-5)}
    # the largest ratio gap in the bottom 40 eigenvalues: that is the real separation
    b = r[:40]; nz = b[b > 0]
    ratios = nz[1:] / nz[:-1]
    k = int(np.argmax(ratios))
    print("%-16s comps=%d expect_rigid=%d  gap after mode %d (ratio %.3g): %.3g -> %.3g" % (
        case, comps, 6 * comps, k + 1, ratios[k], nz[k], nz[k + 1]))
    print("   null dim at relative threshold:", counts)
