"""Unit ghost-penalty matrix (upper entries, coalesced) on the CPU, same construction as box_encode.ghost_faces_gpu:
sum over faces of F_f^T F_f with the fixed 135-DOF face templates. Writes <body>/<case>/GP_UPPER.npz (keys = r*nb + c, r <= c).
Usage: gp_cpu.py <body_dir> [--check] <case> [<case> ...]"""
import sys, json, time
from pathlib import Path
import numpy as np


def gp_upper(body, case, n=32):
    d = Path(body) / case
    faces = np.load(d / 'GP_FACES.npy').astype(np.int64)
    cells = np.load(d / 'CELL_INDICES.npy').astype(np.int64)
    nodes = np.load(d / 'NODES.npy').astype(np.int64)
    tpl = np.load(Path(body) / f'GP_TEMPLATES_n{n}.npz')
    canon = tpl['canonical'].astype(np.float64); offs = tpl['offsets'].astype(np.int64)
    tk = np.transpose(canon, (0, 2, 1)) @ canon                                   # 3 x 135 x 135
    M = 2 * n + 1
    nb = 3 * len(nodes)
    g = 2 * cells[faces[:, 0]][:, None, :] + offs[faces[:, 2]]
    local = np.searchsorted(nodes, (g[..., 0] * M + g[..., 1]) * M + g[..., 2])
    dofs = (3 * local[:, :, None] + np.arange(3)).reshape(len(faces), -1)
    w = dofs.shape[1]
    iu = np.triu_indices(w)
    keys, vals = [], []
    for lo in range(0, len(faces), 2048):
        dd = dofs[lo:lo + 2048]
        r, c = dd[:, iu[0]], dd[:, iu[1]]
        k = np.minimum(r, c) * nb + np.maximum(r, c)
        v = tk[faces[lo:lo + 2048, 2]][:, iu[0], iu[1]]
        keys.append(k.reshape(-1)); vals.append(v.reshape(-1))
    keys = np.concatenate(keys); vals = np.concatenate(vals)
    order = np.argsort(keys, kind='stable')
    keys, vals = keys[order], vals[order]
    start = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
    return keys[start], np.add.reduceat(vals, start)


if __name__ == '__main__':
    body = sys.argv[1]
    args = sys.argv[2:]
    check = '--check' in args
    for case in [a for a in args if a != '--check']:
        t0 = time.perf_counter()
        k, v = gp_upper(body, case)
        rec = dict(case=case, entries=int(len(k)), seconds=time.perf_counter() - t0)
        if check:
            import torch, box_encode as BX
            nb = 3 * len(np.load(Path(body) / case / 'NODES.npy'))
            G = BX.ghost_faces_gpu(body, case, 32, nb)
            gi, gv = G.indices(), G.values(); up = gi[0] <= gi[1]
            kg = (gi[0][up] * nb + gi[1][up]).cpu().numpy(); vg = gv[up].cpu().numpy()
            o = np.argsort(kg); kg, vg = kg[o], vg[o]
            rec.update(same_keys=bool(np.array_equal(kg, k)), max_val_diff=float(np.abs(vg - v).max()) if np.array_equal(kg, k) else None)
        else:
            np.savez(Path(body) / case / 'GP_UPPER.npz', keys=k, vals=v)
        print(json.dumps(rec), flush=True)
