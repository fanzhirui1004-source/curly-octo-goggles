#!/usr/bin/env python3
"""Track C over the FULL production set: the six KUBC apparent stiffnesses of every PASS label.

C[i] = u_i' S u_i with u_i the affine port displacement of the i-th unit strain.  Affine fields lie exactly in every
carrier space, so this is the one quantity comparable across meshes, carrier resolutions AND against an independent
mesher.  Also the quantity the CGAL reference will be compared on.
"""
from __future__ import annotations
import glob, json, os
from pathlib import Path
from multiprocessing import Pool
import numpy as np

L = Path("/root/autodl-tmp/_claude_diag/production/labels")
PAIRS = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))


def one(directory):
    try:
        z = np.load(Path(directory) / "FIXED_PORT_SCHUR_TET10.npz")
        q = int(z["q"][0])
        S = np.zeros((q, q), dtype=np.float32)
        S[np.triu_indices(q)] = z["schur_upper_f32"]
        xyz = np.asarray(z["carrier_coordinates"], dtype=np.float64)
        x = xyz - xyz.mean(axis=0)
        out = []
        for i, j in PAIRS:
            u = np.zeros_like(x)
            if i == j:
                u[:, i] = x[:, i]
            else:
                u[:, i] += 0.5 * x[:, j]; u[:, j] += 0.5 * x[:, i]
            v = u.ravel().astype(np.float32)
            # S is the upper triangle only: v'Sv = 2 v'Uv - v'diag(U)v
            Uv = S @ v
            full = 2.0 * float(np.dot(v.astype(np.float64), Uv.astype(np.float64))) \
                   - float(np.dot((v * v).astype(np.float64), np.diag(S).astype(np.float64)))
            out.append(full)
        d = json.loads((Path(directory) / "LABEL_RECEIPT.json").read_text())
        return {"case": Path(directory).name, "C": out, "q": q,
                "volume": float(d["mesh"]["material_volume"]),
                "cut": "cut_0" in d["fixed_port"]["active_ports"],
                "lmax": float(d["schur"]["max_eigenvalue"])}
    except Exception as exc:  # a label that cannot be read is a finding, not a crash
        return {"case": Path(directory).name, "error": repr(exc)[:200]}


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    dirs = []
    for f in sorted(glob.glob(str(L / "*" / "LABEL_RECEIPT.json"))):
        d = json.loads(Path(f).read_text())
        if d.get("status") == "PASS":
            dirs.append(str(Path(f).parent))
    print("cells:", len(dirs), flush=True)
    res = []
    with Pool(6) as pool:
        for i, r in enumerate(pool.imap_unordered(one, dirs, chunksize=4)):
            res.append(r)
            if (i + 1) % 100 == 0:
                print("  %d/%d" % (i + 1, len(dirs)), flush=True)
    Path("/root/autodl-tmp/_claude_diag/production/STIFFNESS_FULL.json").write_text(json.dumps(res))
    ok = [r for r in res if "C" in r]
    print("done: %d ok, %d failed" % (len(ok), len(res) - len(ok)), flush=True)
