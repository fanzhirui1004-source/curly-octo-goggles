import json, time, numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spl, scipy.linalg as sl
from common import *
case = 'fresh_train_0021_cover01_r1'; t0 = time.time()
C = cell.RealCell(case, gamma=1e-4); K = C.K.tocsr(); I, P = C.I, C.P
f = np.load('fields_%s.npz' % case); Xc, Xm = f['Xc'][:, :8], f['Xm']
KII = K[I][:, I].tocsr(); d = KII.diagonal(); Dinv = sp.diags(1 / d)
src = open('expB2.py').read(); g = dict(np=np, sp=sp, xyz=C.xyz, I=I); exec(src[src.index('def shape1'):src.index('def best')], g); prolong = g['prolong']
lmax = spl.eigsh(sp.diags(d ** -.5) @ KII @ sp.diags(d ** -.5), k=1, which='LA', return_eigenvectors=False)[0]
EUc = (Xc * (K @ Xc)).sum(0); EUm = (Xm * (K @ Xm)).sum(0)
def best_dense(PI, X, EU, cut):
    KP = KII @ PI
    A = (PI.T @ KP).toarray(); A = (A + A.T) / 2
    w, V = sl.eigh(A); keep = w > cut * w.max()
    rhs = KP.T @ X[I]; c = V[:, keep] @ ((V[:, keep].T @ rhs) / w[keep][:, None])
    E = X[I] - PI @ c
    return np.median((E * (KII @ E)).sum(0) / EU), int((~keep).sum()), float(w[keep].min() / w.max())
PI0 = prolong(1, 8).tocsr()
for om_fac in (4 / 3, 2 / 3, 1 / 3):
    om = om_fac / lmax
    PI = (PI0 - om * (Dinv @ (KII @ PI0))).tocsr()
    for cut in (1e-12, 1e-9, 1e-6):
        ec, drop, cond = best_dense(PI, Xc, EUc, cut); em, _, _ = best_dense(PI, Xm, EUm, cut)
        print('Q1_9 smoothed omega=%.2f/lmax cut %.0e: consistent %.4f macro %.4f dropped %d min kept w %.1e' % (om_fac, cut, ec, em, drop, cond), flush=True)
# sanity: unsmoothed with same routine
ec, drop, cond = best_dense(PI0, Xc, EUc, 1e-12); print('Q1_9 unsmoothed', ec, drop, cond)
# sanity 2: span(P0) + span(P_s) must be at least as good as P0
PI = (PI0 - (4 / 3 / lmax) * (Dinv @ (KII @ PI0))).tocsr()
ec, drop, cond = best_dense(sp.hstack([PI0, PI]).tocsr(), Xc, EUc, 1e-12); print('Q1_9 union(P0,Ps)', ec, drop, cond)
