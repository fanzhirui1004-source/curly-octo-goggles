import json, time, numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spl, scipy.linalg as sl
from common import *
case = 'fresh_train_0021_cover01_r1'; t0 = time.time()
C = cell.RealCell(case, gamma=1e-4); K = C.K.tocsr(); I, P = C.I, C.P
f = np.load('fields_%s.npz' % case); Xc, Xm = f['Xc'], f['Xm']
KII = K[I][:, I].tocsr(); d = KII.diagonal(); Dinv = sp.diags(1 / d)
src = open('expB2.py').read(); g = dict(np=np, sp=sp, xyz=C.xyz, I=I); exec(src[src.index('def shape1'):src.index('def best')], g); prolong = g['prolong']
lmax = spl.eigsh(sp.diags(d ** -.5) @ KII @ sp.diags(d ** -.5), k=1, which='LA', return_eigenvectors=False)[0]
om = 4 / 3 / lmax
EUc = (Xc * (K @ Xc)).sum(0); EUm = (Xm * (K @ Xm)).sum(0)
base_c = np.median((Xc[I] * (KII @ Xc[I])).sum(0) / EUc); base_m = np.median((Xm[I] * (KII @ Xm[I])).sum(0) / EUm)
print('c=0 baseline: consistent %.3f macro %.3f' % (base_c, base_m), flush=True)
def best_dense(PI, X, EU):
    A = (PI.T @ (KII @ PI)).toarray(); A = (A + A.T) / 2
    w, V = sl.eigh(A); keep = w > 1e-12 * w.max()
    rhs = PI.T @ (KII @ X[I]); c = V[:, keep] @ ((V[:, keep].T @ rhs) / w[keep][:, None])
    E = X[I] - PI @ c
    return np.median((E * (KII @ E)).sum(0) / EU), int((~keep).sum())
out = {}
for name, order, ne, ns in [('Q1_17', 1, 16, 0), ('Q2_17', 2, 8, 0), ('Q1_9', 1, 8, 0), ('Q1_17', 1, 16, 1), ('Q1_9', 1, 8, 1), ('Q1_9', 1, 8, 2), ('Q1_17', 1, 16, 2), ('Q2_9(el 8h)', 2, 4, 0), ('Q2_9(el 8h)', 2, 4, 1)]:
    PI = prolong(order, ne).tocsr()
    for _ in range(ns):
        PI = (PI - om * (Dinv @ (KII @ PI))).tocsr()
    ec, drop = best_dense(PI, Xc, EUc); em, _ = best_dense(PI, Xm, EUm)
    out['%s_s%d' % (name, ns)] = dict(n=PI.shape[1], consistent=float(ec), macro=float(em), dropped=drop)
    print('%-12s smooth x%d  n=%5d  best energy excess: consistent %.4f  macro %.4f  (dropped %d null dirs)  %ds' % (name, ns, PI.shape[1], ec, em, drop, time.time() - t0), flush=True)
json.dump(out, open('expB2c.json', 'w'), indent=1)
print('done')
