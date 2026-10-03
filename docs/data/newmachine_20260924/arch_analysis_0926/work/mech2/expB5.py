"""0021: partition-of-unity enriched coarse spaces: u(x) = sum_v N_v(x) [a_v + B_v (x - x_v)] (Q1 PU x linear, 12 dof/node)
vs Q1 (3 dof/node) and Q2. Also 'rigid-enriched' PU: a_v + w_v x (x - x_v) (6 dof/node: translation + infinitesimal rotation)."""
import json, time, numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spl, scipy.linalg as sl
from common import *
case = 'fresh_train_0021_cover01_r1'; t0 = time.time()
C = cell.RealCell(case, gamma=1e-4); K = C.K.tocsr(); I, P = C.I, C.P
f = np.load('fields_%s.npz' % case); Xc, Xm = f['Xc'], f['Xm']
KII = K[I][:, I].tocsr()
xyz = C.xyz
EUc = (Xc * (K @ Xc)).sum(0); EUm = (Xm * (K @ Xm)).sum(0)
def pu(ne, kind):
    nn = ne + 1; e = np.minimum(np.floor(xyz * ne).astype(int), ne - 1); t = xyz * ne - e
    cols = []; rows = []; vals = []
    ncomp = {'Q1': 3, 'rigid': 6, 'linear': 12}[kind]
    for a in range(2):
        for b in range(2):
            for c in range(2):
                v = ((e[:, 0] + a) * nn + e[:, 1] + b) * nn + e[:, 2] + c
                w = (t[:, 0] if a else 1 - t[:, 0]) * (t[:, 1] if b else 1 - t[:, 1]) * (t[:, 2] if c else 1 - t[:, 2])
                dx = (xyz - (np.stack([e[:, 0] + a, e[:, 1] + b, e[:, 2] + c], 1) / ne)) * ne    # local, O(1)
                funcs = []   # list of (component d, value) per enrichment function
                for d in range(3): funcs.append([(d, np.ones(len(xyz)))])
                if kind == 'rigid':
                    for r in range(3):   # rotation about axis r: u = e_r x dx
                        e_ = np.zeros(3); e_[r] = 1; u = np.cross(e_[None], dx)
                        funcs.append([(d, u[:, d]) for d in range(3)])
                if kind == 'linear':
                    for d in range(3):
                        for k in range(3): funcs.append([(d, dx[:, k])])
                for j, fn in enumerate(funcs):
                    for d, val in fn:
                        rows.append(3 * np.arange(len(xyz)) + d); cols.append(v * ncomp + j); vals.append(w * val)
    Pd = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(3 * len(xyz), nn ** 3 * ncomp))
    PI = Pd[I]; used = np.flatnonzero(np.abs(PI).sum(0).A1 > 0)
    return PI[:, used].tocsc()
def best(PI, X, EU):
    KP = (KII @ PI).tocsc(); A = (PI.T @ KP).tocsc(); A = (A + A.T) / 2
    rhs = KP.T @ X[I]
    if A.shape[0] <= 12000:
        w, V = sl.eigh(A.toarray()); keep = w > 1e-12 * w.max(); c = V[:, keep] @ ((V[:, keep].T @ rhs) / w[keep][:, None])
    else:
        lu = spl.splu((A + sp.eye(A.shape[0]) * 1e-13 * A.diagonal().max()).tocsc()); c = lu.solve(rhs)
    E = X[I] - PI @ c
    return float(np.median((E * (KII @ E)).sum(0) / EU))
for ne, kind in [(8, 'Q1'), (8, 'rigid'), (8, 'linear'), (16, 'Q1'), (16, 'rigid'), (16, 'linear')]:
    PI = pu(ne, kind)
    print('PU %-6s on %2d^3 elements (%d dof): best energy excess consistent %.4f macro %.4f  (%ds)' % (kind, ne, PI.shape[1], best(PI, Xc, EUc), best(PI, Xm, EUm), time.time() - t0), flush=True)
print('done')
