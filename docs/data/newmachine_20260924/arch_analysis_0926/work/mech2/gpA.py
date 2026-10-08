"""Exp A (cell 0010, heavy cut 5.5%): how much of the soft response is ghost-penalty controlled?
For gamma in {1e-5,1e-4,1e-3}: (1) lowest non-rigid eigenvalues of S and the GP share of their extension energy;
(2) compliance of 'force'-class-like NODAL port loads vs traction-CONSISTENT loads; its gamma dependence."""
import sys, json, numpy as np, scipy.linalg as sl
sys.path.insert(0, '../mechanics'); sys.path.insert(0, '../mechanics/lib')
import cell, loads, scipy.sparse as sp, scipy.sparse.linalg as spl
sys.path.append('../mechanics/pylib')
import pypardiso
case = 'fresh_train_0010_cover01_r1'
rng = np.random.default_rng(0)
out = {}
C0 = cell.RealCell(case, gamma=1e-4)
z = C0.z; P, I = C0.P, C0.I
pn = P[::3] // 3
xyz = C0.xyz
isbox = z['is_box']
# nodal plane-wave loads on box-port nodes (as prep_data 'force'), k in [0.5, 8]
def planewave(X, m=48):
    d = rng.standard_normal((3, m, 3)); d /= np.linalg.norm(d, axis=-1, keepdims=True)
    mag = 0.5 * 16 ** rng.random((3, m)); beta = 2 * rng.random(); amp = mag ** (-beta) * rng.standard_normal((3, m))
    ph = 2 * np.pi * rng.random((3, m))
    return np.stack([(np.cos(2 * np.pi * (X @ (d[c] * mag[c][:, None]).T) + ph[c]) * amp[c]).sum(1) for c in range(3)], 1)
nL = 24
Fn = np.zeros((C0.N, 3, nL))
for j in range(nL):
    f = planewave(xyz); f[~(isbox & z['is_port'])] = 0; Fn[:, :, j] = f
Fn = Fn.reshape(-1, nL)
# consistent smooth tractions on box faces (lattice3-type quadrature)
taus = z['taus']; normal = z['normal']; offset = float(z['offset'])
Wf = {}
for ax in range(3):
    for val in (0.0, 1.0):
        Wf[(ax, val)] = loads.face_weights(C0, ax, val, taus, normal, offset)[0]
Fc = np.zeros((C0.N, 3, nL))
for j in range(nL):
    for (ax, val), W in Wf.items():
        if W.sum() < 1e-9: continue
        o = [d for d in range(3) if d != ax]; u, v = 2 * xyz[:, o[0]] - 1, 2 * xyz[:, o[1]] - 1
        B = np.stack([np.ones_like(u), u, v, u * v, 1.5 * u * u - .5, 1.5 * v * v - .5], 1)
        Fc[:, :, j] += W[:, None] * (B @ rng.standard_normal((6, 3)))
Fc = Fc.reshape(-1, nL)
Qr = loads.rigid(xyz)
Fn -= Qr @ (Qr.T @ Fn); Fc -= Qr @ (Qr.T @ Fc)
print('face areas', {k: round(float(v.sum()), 5) for k, v in Wf.items()})
print('nodal load on weak nodes share', float((Fn.reshape(-1,3,nL)[z['weak']]**2).sum()/(Fn**2).sum()),
      'consistent load on weak nodes share', float((Fc.reshape(-1,3,nL)[z['weak']]**2).sum()/(Fc**2).sum()))
vf = C0.vf
for gamma in [1e-5, 1e-4, 1e-3, 1e-2]:
    C = cell.RealCell(case, gamma=gamma)
    K = C.K.tocsr(); G = C.G.tocsr()
    # Neumann: 3-2-1 support on far-apart nodes of well-filled elements, then rigid projection
    full = np.flatnonzero(C.vf > 0.9); cand = np.unique(C.en[full].ravel()); xa = xyz[cand]
    A_ = cand[np.argmin(xa.sum(1))]; B_ = cand[np.argmax(np.linalg.norm(xa - xyz[A_], axis=1))]
    d2 = np.linalg.norm(np.cross(xa - xyz[A_], xyz[B_] - xyz[A_]), axis=1); C_ = cand[np.argmax(d2)]
    fix = [3 * A_, 3 * A_ + 1, 3 * A_ + 2, 3 * B_ + 1, 3 * B_ + 2, 3 * C_ + 2]
    keep = np.setdiff1d(np.arange(C.nb), fix)
    Kr = K[keep][:, keep]; Ur = sp.triu(Kr, format='csr'); Ur.sort_indices()
    ps = pypardiso.PyPardisoSolver(mtype=-2); ps.factorize(Ur)
    r = {}
    for name, F in (('nodal', Fn), ('consistent', Fc)):
        X = np.zeros_like(F); X[keep] = ps.solve(Ur, np.ascontiguousarray(F[keep])); X -= Qr @ (Qr.T @ X)
        res = np.linalg.norm(K @ X - F) / np.linalg.norm(F)
        comp = (F * X).sum(0)
        gp = gamma * (X * (G @ X)).sum(0) / comp
        ee = np.einsum('eaj,eab,ebj->ej', X[C.dofs], C.Ke, X[C.dofs])
        sl01 = ee[vf < 0.1].sum(0) / ee.sum(0)
        r[name] = dict(res=float(res), compliance_median=float(np.median(comp)), gp_share_median=float(np.median(gp)), gp_share_max=float(gp.max()),
                       vf01_energy_share_median=float(np.median(sl01)))
        out.setdefault(name + '_comp', {})[gamma] = comp.tolist()
    ps.free_memory(everything=True)
    KII = K[I][:, I].toarray(); KIP = K[I][:, P].toarray()
    Xe = -np.linalg.solve(KII, KIP)
    S = K[P][:, P].toarray() + KIP.T @ Xe; S = (S + S.T) / 2
    del KIP
    w, V = np.linalg.eigh(S); del S
    lo = slice(6, 26)
    U = np.zeros((C.nb, 20)); U[P] = V[:, lo]; U[I] = Xe @ V[:, lo]
    del V, Xe
    gps = gamma * (U * (G @ U)).sum(0) / (U * (K @ U)).sum(0)
    r['S_low'] = w[6:26].tolist(); r['S_low_gp_share'] = gps.tolist(); r['S_max'] = float(w[-1])
    r['n_S_below_1e-6'] = int((w[6:] < 1e-6).sum()); r['n_S_below_1e-7'] = int((w[6:] < 1e-7).sum())
    out[gamma] = r
    print(gamma, json.dumps({k: (v if not isinstance(v, list) else [float('%.3g' % x) for x in v[:12]]) for k, v in r.items()}), flush=True)
    del K, G, KII, U
g = [1e-5, 1e-4, 1e-3, 1e-2]
for name in ('nodal', 'consistent'):
    c = np.array([out[name + '_comp'][x] for x in g])
    print(name, 'compliance ratio gamma 1e-5/1e-4 median %.3f  1e-3/1e-4 %.3f 1e-2/1e-4 %.3f' % tuple(np.median(c[i] / c[1]) for i in (0, 2, 3)))
json.dump(out, open('gpA_%s.json' % case, 'w'), default=float)
