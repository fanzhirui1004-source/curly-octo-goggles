"""2D plane-strain Q2 strip (length 1, thickness t), both ends clamped = 'ports'. h = 1/32 along x, t/h through thickness.
(1) Rayleigh contrast of soft (transverse, bending) vs stiff (axial) port data;
(2) energy excess of generic vs Kirchhoff-consistent field errors of equal L2 size delta (the 'locking' amplification);
(3) Jacobi-Chebyshev ([lmax/30,lmax]) decay of each error type (what a few exact-K sweeps can and cannot fix)."""
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spl, scipy.linalg as sl
E_, nu = 1.0, 0.3
lam = E_ * nu / ((1 + nu) * (1 - 2 * nu)); mu = E_ / (2 * (1 + nu))
Cm = np.array([[lam + 2 * mu, lam, 0], [lam, lam + 2 * mu, 0], [0, 0, mu]])
g = np.array([-np.sqrt(.6), 0, np.sqrt(.6)]); wg = np.array([5, 8, 5]) / 9
def L1(x): return np.array([x * (x - 1) / 2, 1 - x * x, x * (x + 1) / 2])
def dL1(x): return np.array([x - .5, -2 * x, x + .5])
def ke(hx, hy):
    K = np.zeros((18, 18))
    for i, a in enumerate(g):
        for j, b in enumerate(g):
            Nx, Ny, dNx, dNy = L1(a), L1(b), dL1(a) * 2 / hx, dL1(b) * 2 / hy
            dx = np.outer(dNx, Ny).ravel(); dy = np.outer(Nx, dNy).ravel()   # node order (ix, iy)
            Bm = np.zeros((3, 18)); Bm[0, 0::2] = dx; Bm[1, 1::2] = dy; Bm[2, 0::2] = dy; Bm[2, 1::2] = dx
            K += wg[i] * wg[j] * Bm.T @ Cm @ Bm * hx * hy / 4
    return K
def strip(t, nx=32, ny=None):
    hx = 1 / nx; ny = ny or max(1, int(round(t / hx))); hy = t / ny
    Nx, Ny = 2 * nx + 1, 2 * ny + 1; nid = lambda i, j: i * Ny + j
    Ke = ke(hx, hy); rows = []; cols = []; vals = []
    for ex in range(nx):
        for ey in range(ny):
            nodes = [nid(2 * ex + a, 2 * ey + b) for a in range(3) for b in range(3)]
            d = np.ravel([[2 * n, 2 * n + 1] for n in nodes])
            rows.append(np.repeat(d, 18)); cols.append(np.tile(d, 18)); vals.append(Ke.ravel())
    nd = 2 * Nx * Ny
    K = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nd, nd))
    X = np.array([[i * hx / 2, j * hy / 2 - t / 2] for i in range(Nx) for j in range(Ny)])
    port = (np.isclose(X[:, 0], 0) | np.isclose(X[:, 0], 1)); P = np.flatnonzero(np.repeat(port, 2)); I = np.flatnonzero(~np.repeat(port, 2))
    return K, X, P, I, ny
def run(t):
    K, X, P, I, ny = strip(t)
    KII = K[I][:, I].tocsc(); KIP = K[I][:, P]; lu = spl.splu(KII)
    x, y = X[:, 0], X[:, 1]; right = np.isclose(x, 1)
    def ext(qfull):
        u = qfull.copy(); u[I] = -lu.solve(KIP @ qfull[P]); return u
    q_soft = np.zeros(2 * len(X)); q_soft[1::2][right] = 1.0          # transverse end offset, no end rotation (S-bend)
    q_stiff = np.zeros(2 * len(X)); q_stiff[0::2][right] = 1.0        # axial stretch
    u_s, u_a = ext(q_soft), ext(q_stiff)
    Rq = lambda u: (u @ (K @ u)) / (u[P] @ u[P])
    RI = lambda u: (u @ (K @ u)) / (u[I] @ u[I])
    # error fields, interior only (ports exact), same L2 size delta*||u_I||
    xi = x[None, :] if False else x
    bump = np.sin(np.pi * x)                                          # vanishes at both ports
    w1 = bump ** 2 * np.sin(2 * np.pi * x)                            # smooth transverse shape, zero value & slope at ends
    dw1 = np.gradient(w1, x) if False else (2 * bump * np.pi * np.cos(np.pi * x) * np.sin(2 * np.pi * x) + bump ** 2 * 2 * np.pi * np.cos(2 * np.pi * x))
    def field(ux, uy):
        e = np.zeros(2 * len(X)); e[0::2] = ux; e[1::2] = uy; e[P] = 0; return e
    rng = np.random.default_rng(1)
    errs = {
        'kirchhoff (v=w, u=-y w\')': field(-y * dw1, w1),                   # inextensional, shear-free: pure bending
        'transverse only (v=w, u=0)': field(0 * x, w1),                      # violates shear constraint (gamma = w')
        'axial smooth (u=w)': field(w1, 0 * x),                              # membrane strain w'
        'thickness-profile (u=w*(y/t)^2)': field(w1 * (2 * y / t) ** 2, 0 * x),
        'nodal white noise': field(rng.standard_normal(len(X)), rng.standard_normal(len(X))),
    }
    out = dict(t=t, t_over_h=t * 32, ny=ny, R_soft=Rq(u_s), R_stiff=Rq(u_a), contrast=Rq(u_a) / Rq(u_s))
    # Chebyshev on D^-1 KII
    d = KII.diagonal(); Dm = sp.diags(1 / d)
    lmax = spl.eigsh(sp.diags(d ** -.5) @ KII @ sp.diags(d ** -.5), k=1, which='LA', return_eigenvectors=False)[0] * 1.02
    a, b = lmax / 30, lmax
    def cheb(e, k):
        # error propagation: e_k = p_k(D^-1 K) e_0 with Chebyshev p_k on [a,b], p_k(0)=1 (three-term recurrence)
        th, de = (b + a) / 2, (b - a) / 2; sig = th / de; rho = 1 / sig
        r = -(KII @ e) ; dx = (Dm @ r) / th; x_ = e + dx
        for _ in range(1, k):
            rn = 1 / (2 * sig - rho); r = -(KII @ x_)
            dx = rn * rho * dx + 2 * rn / de * (Dm @ r); rho = rn; x_ = x_ + dx
        return x_ if k > 0 else e
    for base, u in (('soft', u_s), ('stiff', u_a)):
        EU = u @ (K @ u)
        for name, e in errs.items():
            eI = e[I] / np.linalg.norm(e[I]) * np.linalg.norm(u[I]) * 0.01        # delta = 1% L2
            ex = eI @ (KII @ eI) / EU
            row = [ex / 1e-4]
            for k in (1, 2, 4, 8, 32):
                ek = cheb(eI, k); row.append((ek @ (KII @ ek) / EU) / ex)
            out[f'{base}|{name}'] = row
    return out
for t in (1.5 / 32, 2.5 / 32, 4 / 32):
    o = run(t)
    print('t/L=%.3f (t/h=%.1f, %d Q2 el through t): R_soft=%.3e R_stiff=%.3e contrast stiff/soft=%.0f' % (o['t'], o['t_over_h'], o['ny'], o['R_soft'], o['R_stiff'], o['contrast']))
    for k, v in o.items():
        if '|' in k: print('   %-40s kappa=energy_excess/delta^2 = %9.1f | energy left after k=1,2,4,8,32 sweeps: %s' % (k, v[0], ' '.join('%.3f' % x for x in v[1:])))
