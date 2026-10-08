"""0021: error types -> (i) energy amplification kappa = excess/delta^2, (ii) sensitivity-error / energy-excess ratio,
(iii) decay under the Jacobi-Chebyshev [lmax/30, lmax] smoother on K_II (ports held), k = 1..32."""
import json, time, numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spl
from common import *
import importlib
case = 'fresh_train_0021_cover01_r1'
t0 = time.time()
C = cell.RealCell(case, gamma=1e-4); K = C.K.tocsr(); I, P = C.I, C.P
f = np.load('fields_%s.npz' % case); Xc, Xm = f['Xc'][:, :8], f['Xm']
KII = K[I][:, I].tocsr(); d = KII.diagonal(); Dinv = 1 / d
lmax = spl.eigsh(sp.diags(d ** -.5) @ KII @ sp.diags(d ** -.5), k=1, which='LA', return_eigenvectors=False)[0] * 1.01
a_, b_ = lmax / 30, lmax
def cheb(e, k):
    if k == 0: return e
    th, de = (b_ + a_) / 2, (b_ - a_) / 2; sig = th / de; rho = 1 / sig
    r = -(KII @ e); dx = (Dinv[:, None] * r) / th; x = e + dx
    for _ in range(1, k):
        rn = 1 / (2 * sig - rho); r = -(KII @ x)
        dx = rn * rho * dx + 2 * rn / de * (Dinv[:, None] * r); rho = rn; x = x + dx
    return x
src = open('expB2.py').read(); ns = {}
exec(src[src.index('def shape1'):src.index('def best')], dict(np=np, sp=sp, xyz=C.xyz, I=I), ns)
g = dict(np=np, sp=sp, xyz=C.xyz, I=I); exec(src[src.index('def shape1'):src.index('def best')], g)
prolong = g['prolong']
def best_resid(PI, XI):
    A = (PI.T @ KII @ PI).tocsc(); lu = spl.splu((A + sp.eye(A.shape[0]) * 1e-14 * A.diagonal().max()).tocsc()); c = lu.solve(PI.T @ (KII @ XI)); return XI - PI @ c
dM = np.load(B + '/dM_%s.npy' % case); dKs = [np.einsum('em,mab->eab', dM[c_], C.Tm) for c_ in range(8)]; del dM
def sens(X):
    ue = X[C.dofs]; return np.stack([-np.einsum('eaj,eab,ebj->j', ue, dKc, ue, optimize=True) for dKc in dKs])   # (8,L)
rng = np.random.default_rng(3)
xyz = C.xyz
def planewave_field(kmax, L):
    E = np.zeros((C.N, 3, L))
    for j in range(L):
        for c_ in range(3):
            for _ in range(24):
                kv = rng.standard_normal(3); kv *= (0.5 + (kmax - 0.5) * rng.random()) / np.linalg.norm(kv)
                E[:, c_, j] += rng.standard_normal() * np.cos(2 * np.pi * xyz @ kv + 2 * np.pi * rng.random())
    return E.reshape(-1, L)[I]
out = {}
for base, X in (('consistent', Xc), ('macro', Xm)):
    L = X.shape[1]; XI = X[I]; EU = (X * (K @ X)).sum(0); s0 = sens(X)
    errs = {'smooth planewave k<=2': planewave_field(2, L), 'planewave k<=8': planewave_field(8, L), 'white noise': rng.standard_normal(XI.shape)}
    PI = prolong(1, 32); errs['Q1_33 locking residual'] = best_resid(PI, XI)
    PI = prolong(2, 8); errs['Q2_17 residual (long-wave)'] = best_resid(PI, XI)
    PI = prolong(1, 16); errs['Q1_17 residual'] = best_resid(PI, XI)
    for name, E in errs.items():
        E = E / np.linalg.norm(E, axis=0) * np.linalg.norm(XI, axis=0) * 0.01          # delta = 1% (L2, interior)
        ex = (E * (KII @ E)).sum(0) / EU
        kap = ex / 1e-4
        # sensitivity error at the scale where energy excess = 5%
        Es = E * np.sqrt(0.05 / ex)[None]
        Xh = X.copy(); Xh[I] += Es
        ds = np.linalg.norm(sens(Xh) - s0, axis=0) / np.linalg.norm(s0, axis=0)
        dec = []
        for k in (1, 2, 4, 8, 16, 32):
            Ek = cheb(Es, k); dec.append(float(np.median((Ek * (KII @ Ek)).sum(0) / EU / 0.05)))
        # sensitivity after 8 sweeps
        Xh8 = X.copy(); Xh8[I] += cheb(Es, 8); ds8 = np.linalg.norm(sens(Xh8) - s0, axis=0) / np.linalg.norm(s0, axis=0)
        r = dict(kappa=float(np.median(kap)), sens_err_at_5pct=float(np.median(ds)), sens_err_after8=float(np.median(ds8)), energy_left=dec)
        out[base + '|' + name] = r
        print('%-10s %-28s kappa %10.1f | sens err at 5%% energy excess: %.3f (after 8 sweeps %.4f) | energy left k=1,2,4,8,16,32: %s' %
              (base, name, r['kappa'], r['sens_err_at_5pct'], r['sens_err_after8'], ' '.join('%.3f' % v for v in dec)), flush=True)
json.dump(out, open('expB4.json', 'w'), indent=1)
print('done', time.time() - t0)
