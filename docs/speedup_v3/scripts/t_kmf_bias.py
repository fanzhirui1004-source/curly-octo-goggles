# (local CPU check, numpy float32; run: python t_kmf_bias.py)
# Systematic vs random fp32 error of three element-product forms on many cut elements (numpy float32 arithmetic):
# energy u^T K u of smooth fields, relative to the float64 template value.
import sys, types, numpy as np
sys.modules['scipy'] = types.ModuleType('scipy'); so = types.ModuleType('scipy.optimize'); so.nnls = None
sys.modules['scipy.optimize'] = so
sys.path.insert(0, '/home/user/curly-octo-goggles/docs/data/src_v3')
import element_moments as EM
from itertools import product
f32 = np.float32
rng = np.random.default_rng(3)
n = 10; lam, mu = 0.5769230769230769, 0.38461538461538464         # E = 1, nu = 0.3
loc = np.array(list(product((-1, 0, 1), repeat=3)))                 # tensor order nodes
T = EM.pattern_operators(loc, lam, mu, n)[1].reshape(125, 81 * 81)
iu = np.triu_indices(81)
cells = np.array(list(product(range(n), repeat=3))); E = len(cells)
# random cut moments: element = part of the box below a random plane (Monte-Carlo-free: product of 1-D boxes)
M = np.zeros((E, 125))
for e in range(E):
    lo = rng.uniform(-1, 0.6, 3); hi = np.minimum(lo + rng.uniform(0.2, 2.0, 3), 1)
    m1 = [np.array([(hi[d] ** (a + 1) - lo[d] ** (a + 1)) / (a + 1) for a in range(5)]) / (2 * n) for d in range(3)]
    M[e] = np.einsum('a,b,c->abc', *m1).reshape(125)
# smooth field: u(x) = quadratic, evaluated at element nodes (cell coords + local)
def field(k):
    g = (2 * cells[:, None, :] + 1 + loc[None]) / (2 * n)            # node coords in [0,1]
    x, y, z = g[..., 0], g[..., 1], g[..., 2]
    if k == 0: U = np.stack([x * y, y * z + x, z * x * x], -1)
    elif k == 1: U = np.stack([np.sin(3 * x) * y, np.cos(2 * z), x + y * z * z], -1)
    else: U = rng.standard_normal((E, 27, 3))
    return U.reshape(E, 81)
Lg = lambda k, x: {-1: x * (x - 1) / 2, 0: 1 - x * x, 1: x * (x + 1) / 2}[k]
dLg = lambda k, x: {-1: x - 0.5, 0: -2 * x, 1: x + 0.5}[k]
def mf32(pts, U):
    V1 = np.vander(pts, 5, increasing=True).T; V1i = np.linalg.inv(V1)
    Vi3 = np.kron(np.kron(V1i, V1i), V1i)
    w = (M @ Vi3.T) * (2 * n) ** 2                                    # float64
    wl, wm = (w * lam).astype(f32), (w * mu).astype(f32)
    Lt = np.array([[Lg(k, q) for k in (-1, 0, 1)] for q in pts]).astype(f32)
    Dt = np.array([[dLg(k, q) for k in (-1, 0, 1)] for q in pts]).astype(f32)
    X = U.astype(f32).reshape(E, 3, 3, 3, 3)                          # e, i0, i1, i2, c
    # gradients at points (q0, q1, q2): d/dxi_0 = D L L, etc. (float32 einsums)
    g0 = np.einsum('pa,qb,rc,eabck->epqrk', Dt, Lt, Lt, X, dtype=f32, optimize=True)
    g1 = np.einsum('pa,qb,rc,eabck->epqrk', Lt, Dt, Lt, X, dtype=f32, optimize=True)
    g2 = np.einsum('pa,qb,rc,eabck->epqrk', Lt, Lt, Dt, X, dtype=f32, optimize=True)
    tr = g0[..., 0] + g1[..., 1] + g2[..., 2]
    WL, WM = wl.reshape(E, 5, 5, 5), wm.reshape(E, 5, 5, 5)
    G = [g0, g1, g2]
    en = np.zeros(E, dtype=f32)
    for c in range(3):
        for d in range(3):
            s = WM * (G[d][..., c] + G[c][..., d]) + (WL * tr if c == d else 0)
            en += np.einsum('epqr,epqr->e', s, G[d][..., c], dtype=f32)
    return en                                                          # per element u^T K u (float32 sums)
def packed32(U):
    P = (M.astype(f32) @ T.astype(f32)).astype(f32)                   # E x 6561 (float32 GEMM)
    K = P.reshape(E, 81, 81)
    return np.einsum('ei,eij,ej->e', U.astype(f32), K, U.astype(f32), dtype=f32)
gauss = np.polynomial.legendre.leggauss(5)[0]; equi = np.array([-1, -0.5, 0, 0.5, 1.0])
for k in range(3):
    U = field(k)
    ref = np.einsum('ei,eij,ej->e', U, (M @ T).reshape(E, 81, 81), U)   # float64
    tot = ref.sum()
    for name, en in (('packed32', packed32(U)), ('mf_gauss', mf32(gauss, U)), ('mf_equi', mf32(equi, U))):
        err = en.astype(np.float64) - ref
        print('field %d %-9s total rel err %+.2e   per-element rms rel %.2e   bias/rms %.2f' % (
            k, name, err.sum() / tot, np.sqrt((err ** 2).mean()) / np.abs(ref).mean(), err.mean() / np.sqrt((err ** 2).mean())))
