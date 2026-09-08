"""The 16 cells whose null dim differs from 6 + 3*n_unsupported: what is the extra family?"""
import json
from pathlib import Path
import numpy as np
L = Path("/root/autodl-tmp/_claude_diag/production/labels")
CASES = ["pop_cut_1142", "pop_cut_0915", "pop_cut_1240", "pop_cut_0346", "pop_cut_0495",
         "pop_cut_0945", "pop_cut_1017", "pop_cut_1062", "pop_uncut_0447", "pop_cut_0995"]
for case in CASES:
    z = np.load(L / case / "FIXED_PORT_SCHUR_TET10.npz")
    q = int(z["q"][0]); S = np.zeros((q, q)); S[np.triu_indices(q)] = z["schur_upper_f32"]
    S = S + S.T - np.diag(np.diag(S))
    sup = np.asarray(z["support_mask"]).astype(bool)
    d = json.loads((L / case / "LABEL_RECEIPT.json").read_text())
    lb = d["schur"]["support"]["null_dim_lower_bound"]
    # restrict to the SUPPORTED columns: there the null space should be exactly 6 per component
    keep = np.repeat(sup, 3)
    Ss = S[np.ix_(keep, keep)]
    w = np.linalg.eigvalsh(Ss)
    lmax = w.max()
    nz = int((w < 1e-14 * lmax).sum())
    comps = d["surface"]["components"]["components"]
    # how much of each near-null vector lives on one carrier node?
    w2, V = np.linalg.eigh(Ss)
    extra = nz - 6 * comps
    msg = ""
    if extra > 0:
        # the extra modes beyond 6*components: how localised are they?
        idx = np.arange(6 * comps, nz)
        for k in idx[:3]:
            v = V[:, k]; m = (v.reshape(-1, 3) ** 2).sum(axis=1)
            o = np.argsort(m)[::-1]
            msg += " mode%d: top-node mass %.4f, top-3 %.4f;" % (k, m[o[0]], m[o[:3]].sum())
    print("%-16s comps=%d supported_null=%d  6*comps=%d  extra=%+d  lb=%d %s" % (
        case, comps, nz, 6 * comps, extra, lb, msg))
