import numpy as np
def face_weights(C, axis, value, taus, normal, offset, ppe=6):
    """consistent nodal weights W (N_nodes,) and quadrature (points, weights, node ids, shape values) on box face."""
    n = C.n; h = 1.0 / (n * ppe)
    s = np.arange(0, 1, h) + h / 2
    A, B = np.meshgrid(s, s, indexing='ij')
    P = np.zeros((A.size, 3)); o = [d for d in range(3) if d != axis]
    P[:, axis] = value; P[:, o[0]] = A.ravel(); P[:, o[1]] = B.ravel()
    f = np.cos(2 * np.pi * P).sum(1)
    cube = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], float)
    # tau trilinear: corner order as in CUBE of element_polyref (lexicographic 0/1) -- check against packet ordering
    w8 = np.where(cube[:, None, :].astype(bool), P[None], 1 - P[None]).prod(-1)
    tau = np.asarray(taus) @ w8
    keep = np.abs(f) <= tau
    if normal is not None:
        keep &= (P @ np.asarray(normal)) <= offset
    P = P[keep]
    c = np.minimum(np.floor(P * n).astype(np.int64), n - 1)
    key = lambda a: (a[:, 0] * n + a[:, 1]) * n + a[:, 2]
    ck = key(C.cells); order = np.argsort(ck); cks = ck[order]
    pk = key(c); loc = np.searchsorted(cks, pk); loc = np.minimum(loc, len(cks) - 1)
    ok = cks[loc] == pk
    P, c = P[ok], c[ok]
    xi = P * n - c
    L = np.stack([2 * (xi - .5) * (xi - 1), -4 * xi * (xi - 1), 2 * xi * (xi - .5)], -1)
    M1 = 2 * n + 1
    nid = C.z['node_ids'].astype(np.int64)
    W = np.zeros(C.N)
    for oo in np.ndindex(3, 3, 3):
        g = 2 * c + np.asarray(oo)
        ids = (g[:, 0] * M1 + g[:, 1]) * M1 + g[:, 2]
        li = np.searchsorted(nid, ids); good = (li < len(nid))
        li = np.minimum(li, len(nid) - 1); good &= nid[li] == ids
        val = h * h * L[:, 0, oo[0]] * L[:, 1, oo[1]] * L[:, 2, oo[2]]
        np.add.at(W, li[good], val[good])
    return W, P
def rigid(xyz):
    c = xyz - xyz.mean(0); N = len(xyz)
    R = np.zeros((N, 3, 6))
    for a in range(3):
        R[:, a, a] = 1
        e = np.zeros(3); e[a] = 1
        R[:, :, 3 + a] = np.cross(e[None], c)
    Q, _ = np.linalg.qr(R.reshape(-1, 6)); return Q
