import sys, time, json
sys.path.append('pylib')
import numpy as np, scipy.sparse as sp, pypardiso, cell, loads
case = sys.argv[1]
C = cell.RealCell(case)
z = C.z; taus = z['taus']; normal = z['normal']; offset = float(z['offset'])
# check tau corner ordering via a material test on quadrature: taus as in packet (CUBE order lexicographic)
F = []; names = []
Wf = {}
for ax in range(3):
    for val in (0.0, 1.0):
        W, P = loads.face_weights(C, ax, val, taus, normal, offset); Wf[(ax, val)] = W
print('face material areas', {k: round(float(v.sum()), 4) for k, v in Wf.items()}, flush=True)
Nn = C.N
def nodal(tr):   # tr: dict (ax,val)->(3,) traction vector (uniform) or callable
    f = np.zeros((Nn, 3))
    for k, t in tr.items():
        f += Wf[k][:, None] * np.asarray(t)[None]
    return f.ravel()
e = np.eye(3)
for a in range(3):
    if Wf[(a, 0.0)].sum() > 1e-6 and Wf[(a, 1.0)].sum() > 1e-6:
        for d in range(3):
            F.append(nodal({(a, 1.0): e[d], (a, 0.0): -e[d]})); names.append('macro_ax%d_dir%d' % (a, d))
# smooth random tractions (degree<=2 Legendre in the face coordinates) on all faces
rng = np.random.default_rng(1)
xyz = C.xyz
for r in range(12):
    f = np.zeros((Nn, 3))
    for ax in range(3):
        o = [d for d in range(3) if d != ax]
        for val in (0.0, 1.0):
            W = Wf[(ax, val)]
            if W.sum() < 1e-9: continue
            u, v = 2 * xyz[:, o[0]] - 1, 2 * xyz[:, o[1]] - 1
            basis = np.stack([np.ones_like(u), u, v, u * v, 1.5 * u * u - .5, 1.5 * v * v - .5], 1)
            coef = rng.standard_normal((6, 3)) / np.array([1, 1, 1, 1.5, 1.5, 1.5])[:, None]
            f += W[:, None] * (basis @ coef)
    F.append(f.ravel()); names.append('smooth_%d' % r)
F = np.stack(F, 1)
Qr = loads.rigid(xyz)
F = F - Qr @ (Qr.T @ F)                               # self-equilibrated
nonport = ~np.repeat(z['is_port'], 3)
print('load on non-port dofs (rel):', float(np.abs(F[nonport]).max() / np.abs(F).max()), flush=True)
# 3-2-1 support on nodes of full elements far apart
full = np.flatnonzero(C.vf > 0.999)
cand = np.unique(C.en[full].ravel())
xa = xyz[cand]
A_ = cand[np.argmin(xa.sum(1))]; B_ = cand[np.argmax(np.linalg.norm(xa - xyz[A_], axis=1))]
d2 = np.linalg.norm(np.cross(xa - xyz[A_], xyz[B_] - xyz[A_]), axis=1); C_ = cand[np.argmax(d2)]
fix = [3 * A_, 3 * A_ + 1, 3 * A_ + 2, 3 * B_ + 1, 3 * B_ + 2, 3 * C_ + 2]
keep = np.setdiff1d(np.arange(C.nb), fix)
t = time.time()
Kr = C.K[keep][:, keep]; U = sp.triu(Kr, format='csr'); U.sort_indices()
del Kr
ps = pypardiso.PyPardisoSolver(mtype=-2); ps.factorize(U)
print('factor', time.time() - t, flush=True)
X = np.zeros_like(F)
X[keep] = ps.solve(U, np.ascontiguousarray(F[keep]))
ps.free_memory(everything=True)
X = X - Qr @ (Qr.T @ X)
res = C.K @ X - F
print('neumann residual rel', float(np.linalg.norm(res) / np.linalg.norm(F)), flush=True)
en = (F * X).sum(0)
np.savez('neu_%s.npz' % case, X=X, F=F, names=np.array(names), energy=en)
print('energies', dict(zip(names, np.round(en, 8).tolist())))
