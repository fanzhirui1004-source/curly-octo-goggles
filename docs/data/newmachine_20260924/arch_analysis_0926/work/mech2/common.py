import sys, os, numpy as np, scipy.sparse as sp
B = '/tmp/claude-0/-home-user-curly-octo-goggles/76f53c89-e2a9-52bb-8817-9460fd780b76/scratchpad/pack/work/mechanics'
sys.path.insert(0, B); sys.path.insert(0, B + '/lib'); sys.path.append(B + '/pylib')
import cell, loads, pypardiso
rng = np.random.default_rng(0)
def face_loads(C, nL, deg_lo=0, only=None, rng=rng):
    z = C.z; xyz = C.xyz; F = np.zeros((C.N, 3, nL)); Wf = {}
    for ax in range(3):
        for val in (0.0, 1.0):
            Wf[(ax, val)] = loads.face_weights(C, ax, val, z['taus'], z['normal'], float(z['offset']))[0]
    for j in range(nL):
        for (ax, val), W in Wf.items():
            if W.sum() < 1e-9 or (only is not None and (ax, val) != only): continue
            o = [d for d in range(3) if d != ax]; u, v = 2 * xyz[:, o[0]] - 1, 2 * xyz[:, o[1]] - 1
            Bs = [np.ones_like(u), u, v, u * v, 1.5 * u * u - .5, 1.5 * v * v - .5, u * (1.5 * v * v - .5), v * (1.5 * u * u - .5),
                  2.5 * u ** 3 - 1.5 * u, 2.5 * v ** 3 - 1.5 * v]
            Bm = np.stack(Bs[deg_lo:], 1)
            F[:, :, j] += W[:, None] * (Bm @ rng.standard_normal((Bm.shape[1], 3)))
    return F.reshape(-1, nL), Wf
def nodal_planewave(C, nL, rng=rng, m=48):
    z = C.z; X = C.xyz; F = np.zeros((C.N, 3, nL)); box = z['is_box'] & z['is_port']
    for j in range(nL):
        d = rng.standard_normal((3, m, 3)); d /= np.linalg.norm(d, axis=-1, keepdims=True)
        mag = 0.5 * 16 ** rng.random((3, m)); beta = 2 * rng.random(); amp = mag ** (-beta) * rng.standard_normal((3, m))
        ph = 2 * np.pi * rng.random((3, m))
        f = np.stack([(np.cos(2 * np.pi * (X @ (d[c] * mag[c][:, None]).T) + ph[c]) * amp[c]).sum(1) for c in range(3)], 1)
        f[~box] = 0; F[:, :, j] = f
    return F.reshape(-1, nL)
class Neu:
    def __init__(self, C):
        K = C.K.tocsr(); xyz = C.xyz
        full = np.flatnonzero(C.vf > 0.9); cand = np.unique(C.en[full].ravel()); xa = xyz[cand]
        A_ = cand[np.argmin(xa.sum(1))]; B_ = cand[np.argmax(np.linalg.norm(xa - xyz[A_], axis=1))]
        d2 = np.linalg.norm(np.cross(xa - xyz[A_], xyz[B_] - xyz[A_]), axis=1); C_ = cand[np.argmax(d2)]
        fix = [3 * A_, 3 * A_ + 1, 3 * A_ + 2, 3 * B_ + 1, 3 * B_ + 2, 3 * C_ + 2]
        self.keep = np.setdiff1d(np.arange(C.nb), fix)
        Kr = K[self.keep][:, self.keep]; self.U = sp.triu(Kr, format='csr'); self.U.sort_indices()
        self.ps = pypardiso.PyPardisoSolver(mtype=-2); self.ps.factorize(self.U)
        self.Qr = loads.rigid(xyz); self.K = K
    def equilibrate(self, F, wdof):
        """remove resultant force/moment with a correction supported on dofs with weight wdof (port-supported, consistent)"""
        R = self.Qr; WR = wdof[:, None] * R
        return F - WR @ np.linalg.solve(R.T @ WR, R.T @ F)
    def solve(self, F, wdof=None):
        if wdof is not None: F = self.equilibrate(F, wdof)
        F = F - self.Qr @ (self.Qr.T @ F)
        X = np.zeros_like(F); X[self.keep] = self.ps.solve(self.U, np.ascontiguousarray(F[self.keep]))
        return X - self.Qr @ (self.Qr.T @ X), F
    def free(self): self.ps.free_memory(everything=True)
class Dir:
    """interior solves with ports clamped"""
    def __init__(self, C):
        K = C.K.tocsr(); self.I = C.I; self.P = C.P
        self.KII = K[C.I][:, C.I].tocsr(); self.KIP = K[C.I][:, C.P].tocsr()
        self.U = sp.triu(self.KII, format='csr'); self.U.sort_indices()
        self.ps = pypardiso.PyPardisoSolver(mtype=2); self.ps.factorize(self.U)
    def solve(self, b): return self.ps.solve(self.U, np.ascontiguousarray(b))
    def extend(self, Q): return -self.solve(self.KIP @ Q)
def elem_energy(C, X):
    return np.einsum('eaj,eab,ebj->ej', X[C.dofs], C.Ke, X[C.dofs])
