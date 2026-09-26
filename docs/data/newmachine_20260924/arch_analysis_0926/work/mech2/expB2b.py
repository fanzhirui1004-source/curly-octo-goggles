"""0021: can fixed polynomial coarse spaces (as in the U-Net transfers) represent the exact soft fields?
Energy-norm best approximation of the interior field (ports exact) from span(P) for Q1 (trilinear) and Q2 coarse spaces."""
import json, time, numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spl
from common import *
case = sys.argv[1] if len(sys.argv) > 1 else 'fresh_train_0021_cover01_r1'
t0 = time.time()
C = cell.RealCell(case, gamma=1e-4); K = C.K.tocsr(); I, P = C.I, C.P
f = np.load('fields_%s.npz' % case); Xc, Xn, Xm = f['Xc'], f['Xn'], f['Xm']
KII = K[I][:, I].tocsr()
xyz = C.xyz
def shape1(t):   # t in [0,1] local -> linear weights for 2 nodes
    return np.stack([1 - t, t], -1)
def shape2(t):
    s = 2 * t - 1
    return np.stack([s * (s - 1) / 2, 1 - s * s, s * (s + 1) / 2], -1)
def prolong(order, ne):
    """nodes of a tensor Lagrange space of given order on ne^3 coarse elements covering [0,1]^3"""
    nn = order * ne + 1
    e = np.minimum(np.floor(xyz * ne).astype(int), ne - 1); t = xyz * ne - e
    W = [shape1(t[:, d]) if order == 1 else shape2(t[:, d]) for d in range(3)]
    rows, cols, vals = [], [], []
    for a in range(order + 1):
        for b in range(order + 1):
            for c in range(order + 1):
                idx = ((order * e[:, 0] + a) * nn + order * e[:, 1] + b) * nn + order * e[:, 2] + c
                w = W[0][:, a] * W[1][:, b] * W[2][:, c]
                rows.append(np.arange(len(xyz))); cols.append(idx); vals.append(w)
    Pn = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(len(xyz), nn ** 3))
    Pn.eliminate_zeros()
    Pd = sp.kron(Pn, sp.eye(3), format='csr')          # node-major xyz dofs
    PI = Pd[I]
    used = np.flatnonzero(np.abs(PI).sum(0).A1 > 0)
    return PI[:, used].tocsc()
def best(PI, X):
    """relative energy error of best approximation of interior part, ports exact: min ||u_I - PI c||_KII (lifting = q with 0 interior;
    energy of residual measured with the full K: ||u - u0 - P c||_K = ||u_I - PI c||_KII + cross terms -> use exact identity below)"""
    A = (PI.T @ KII @ PI).tocsc()
    # target: Galerkin projection of the exact interior field: u_I solves KII u_I = -KIP q; best c: A c = PI^T KII u_I
    rhs = PI.T @ (KII @ X[I])
    lu = spl.splu(A + sp.eye(A.shape[0]) * 1e-14 * A.diagonal().max())
    c = lu.solve(rhs)
    E = X[I] - PI @ c
    en = (X * (K @ X)).sum(0)
    return (E * (KII @ E)).sum(0) / en      # = energy excess of the best extension in u0 + span(P)
out = {}
d = KII.diagonal(); Dinv = sp.diags(1 / d)
lmax = spl.eigsh(sp.diags(d ** -.5) @ KII @ sp.diags(d ** -.5), k=1, which='LA', return_eigenvectors=False)[0]
om = 4 / 3 / lmax
Xv = None
for name, order, ne, ns in [('Q1_17', 1, 16, 1), ('Q1_17', 1, 16, 2), ('Q1_17', 1, 16, 4), ('Q1_9', 1, 8, 1), ('Q1_9', 1, 8, 2), ('Q1_9', 1, 8, 4), ('Q1_33', 1, 32, 1), ('Q1_33', 1, 32, 2)]:
    PI = prolong(order, ne).tocsr()
    for _ in range(ns):
        PI = (PI - om * (Dinv @ (KII @ PI))).tocsr()
        PI.data[np.abs(PI.data) < 1e-12 * np.abs(PI.data).max()] = 0; PI.eliminate_zeros()
    r = {}
    for fn, X in (('consistent', Xc), ('macro', Xm)):
        e = best(PI.tocsc(), X); r[fn] = [float(np.median(e)), float(e.min()), float(e.max())]
    out['%s_smoothed%d' % (name, ns)] = dict(ncoarse=PI.shape[1], nnzP=PI.nnz, **r)
    print('%s smoothed x%d' % (name, ns), PI.shape[1], PI.nnz, json.dumps(r), round(time.time() - t0), flush=True)
json.dump(out, open('expB2b_%s.json' % case, 'w'), indent=1)
print('done')
