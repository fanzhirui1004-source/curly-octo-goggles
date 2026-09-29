"""Section 6.11: the method of moving asymptotes (MMA), standard form, without the GCMMA inner loop. New script.

Svanberg, K. (1987). The method of moving asymptotes - a new method for structural optimization. IJNME 24, 359-373.
Svanberg, K. (2007). MMA and GCMMA - two methods for nonlinear optimization (subproblem and interior-point solver).

Problem form (Svanberg 2007, Eq. 1.1):
    minimise  f0(x) + a0 z + sum_i (c_i y_i + d_i y_i^2 / 2)
    s.t.      f_i(x) - a_i z - y_i <= 0,   xmin <= x <= xmax,   y >= 0, z >= 0.
For a standard nonlinear programme use a0 = 1, a = 0, d = 1 and large c (the artificial y_i vanish at a feasible optimum).

mma_update() performs ONE outer iteration: moving asymptotes from the last two iterates, the convex separable approximation
at xval (one function and gradient evaluation per outer iteration, no line search, no conservativeness test), and the
primal-dual interior-point solution of the subproblem (subsolv).  All arrays are float64 numpy.
    x_new, state = mma_update(xval, f0val, df0dx, fval, dfdx, xmin, xmax, state, move=0.05)
state carries the iteration counter, the two previous iterates and the asymptotes (a dict; pass None the first time).
Usage (self-test):  python3 mma.py --selftest
"""
import numpy as np

ASYINIT, ASYINCR, ASYDECR = 0.5, 1.2, 0.7
ALBEFA, RAA0, EPSIMIN = 0.1, 1e-5, 1e-7


def mma_update(xval, f0val, df0dx, fval, dfdx, xmin, xmax, state=None, move=0.05, a0=1.0, a=None, c=None, d=None):
    """One MMA iteration. fval (m,), dfdx (m, n). Returns (x_new, state, info)."""
    xval = np.asarray(xval, float); n = xval.size
    fval = np.atleast_1d(np.asarray(fval, float)); m = fval.size
    dfdx = np.asarray(dfdx, float).reshape(m, n); df0dx = np.asarray(df0dx, float).reshape(n)
    xmin = np.broadcast_to(np.asarray(xmin, float), (n,)).copy(); xmax = np.broadcast_to(np.asarray(xmax, float), (n,)).copy()
    a = np.zeros(m) if a is None else np.asarray(a, float)
    c = 1000.0 * np.ones(m) if c is None else np.asarray(c, float)
    d = np.ones(m) if d is None else np.asarray(d, float)
    st = dict(it=0, xold1=xval.copy(), xold2=xval.copy(), low=None, upp=None) if state is None else dict(state)
    it = st['it'] + 1
    xrange = xmax - xmin
    if it <= 2 or st['low'] is None:
        low = xval - ASYINIT * xrange
        upp = xval + ASYINIT * xrange
    else:
        zzz = (xval - st['xold1']) * (st['xold1'] - st['xold2'])
        factor = np.ones(n); factor[zzz > 0] = ASYINCR; factor[zzz < 0] = ASYDECR
        low = xval - factor * (st['xold1'] - st['low'])
        upp = xval + factor * (st['upp'] - st['xold1'])
        low = np.clip(low, xval - 10 * xrange, xval - 0.01 * xrange)
        upp = np.clip(upp, xval + 0.01 * xrange, xval + 10 * xrange)
    alfa = np.maximum.reduce([low + ALBEFA * (xval - low), xval - move * xrange, xmin])
    beta = np.minimum.reduce([upp - ALBEFA * (upp - xval), xval + move * xrange, xmax])
    xmamiinv = 1.0 / np.maximum(xrange, 1e-5)
    ux1 = upp - xval; ux2 = ux1 ** 2; xl1 = xval - low; xl2 = xl1 ** 2
    p0 = np.maximum(df0dx, 0); q0 = np.maximum(-df0dx, 0)
    pq0 = 0.001 * (p0 + q0) + RAA0 * xmamiinv
    p0 = (p0 + pq0) * ux2; q0 = (q0 + pq0) * xl2
    P = np.maximum(dfdx, 0); Q = np.maximum(-dfdx, 0)
    PQ = 0.001 * (P + Q) + RAA0 * xmamiinv[None, :]
    P = (P + PQ) * ux2[None, :]; Q = (Q + PQ) * xl2[None, :]
    b = P @ (1 / ux1) + Q @ (1 / xl1) - fval
    x, y, z, lam, info = subsolv(m, n, low, upp, alfa, beta, p0, q0, P, Q, a0, a, b, c, d)
    st.update(it=it, xold2=st['xold1'].copy(), xold1=xval.copy(), low=low, upp=upp)
    info.update(it=it, max_dx=float(np.abs(x - xval).max()), y_max=float(y.max()) if m else 0.0, z=float(z),
                lam=lam)
    return x, st, info


def subsolv(m, n, low, upp, alfa, beta, p0, q0, P, Q, a0, a, b, c, d, epsimin=EPSIMIN):
    """Primal-dual Newton interior-point solution of the MMA subproblem (Svanberg 2007, Sec. 5)."""
    een, eem = np.ones(n), np.ones(m)
    epsi = 1.0
    x = 0.5 * (alfa + beta); y = eem.copy(); z = 1.0; lam = eem.copy()
    xsi = np.maximum(een / (x - alfa), een); eta = np.maximum(een / (beta - x), een)
    mu = np.maximum(eem, 0.5 * c); zet = 1.0; s = eem.copy()
    total = 0

    def residual(x, y, z, lam, xsi, eta, mu, zet, s, epsi):
        ux1 = upp - x; xl1 = x - low
        plam = p0 + P.T @ lam; qlam = q0 + Q.T @ lam
        gvec = P @ (1 / ux1) + Q @ (1 / xl1)
        dpsidx = plam / ux1 ** 2 - qlam / xl1 ** 2
        r = np.concatenate([dpsidx - xsi + eta, c + d * y - mu - lam, [a0 - zet - a @ lam],
                            gvec - a * z - y + s - b, xsi * (x - alfa) - epsi, eta * (beta - x) - epsi,
                            mu * y - epsi, [zet * z - epsi], lam * s - epsi])
        return r

    while epsi > epsimin:
        res = residual(x, y, z, lam, xsi, eta, mu, zet, s, epsi)
        resnorm = np.linalg.norm(res); resmax = np.abs(res).max()
        ittt = 0
        while resmax > 0.9 * epsi and ittt < 200:
            ittt += 1; total += 1
            ux1 = upp - x; xl1 = x - low; ux2 = ux1 ** 2; xl2 = xl1 ** 2; ux3 = ux1 * ux2; xl3 = xl1 * xl2
            plam = p0 + P.T @ lam; qlam = q0 + Q.T @ lam
            gvec = P @ (1 / ux1) + Q @ (1 / xl1)
            GG = P / ux2[None, :] - Q / xl2[None, :]
            dpsidx = plam / ux2 - qlam / xl2
            delx = dpsidx - epsi / (x - alfa) + epsi / (beta - x)
            dely = c + d * y - lam - epsi / y
            delz = a0 - a @ lam - epsi / z
            dellam = gvec - a * z - y - b + epsi / lam
            diagx = 2 * (plam / ux3 + qlam / xl3) + xsi / (x - alfa) + eta / (beta - x)
            diagy = d + mu / y
            diaglamyi = s / lam + 1 / diagy
            if m < n:
                blam = dellam + dely / diagy - GG @ (delx / diagx)
                Alam = np.diag(diaglamyi) + (GG / diagx[None, :]) @ GG.T
                AA = np.block([[Alam, a[:, None]], [a[None, :], np.array([[-zet / z]])]])
                sol = np.linalg.solve(AA, np.concatenate([blam, [delz]]))
                dlam, dz = sol[:m], sol[m]
                dx = -delx / diagx - (GG.T @ dlam) / diagx
            else:
                dellamyi = dellam + dely / diagy
                Axx = np.diag(diagx) + (GG.T / diaglamyi[None, :]) @ GG
                azz = zet / z + a @ (a / diaglamyi)
                axz = -GG.T @ (a / diaglamyi)
                bx = delx + GG.T @ (dellamyi / diaglamyi)
                bz = delz - a @ (dellamyi / diaglamyi)
                AA = np.block([[Axx, axz[:, None]], [axz[None, :], np.array([[azz]])]])
                sol = np.linalg.solve(AA, -np.concatenate([bx, [bz]]))
                dx, dz = sol[:n], sol[n]
                dlam = (GG @ dx) / diaglamyi - dz * (a / diaglamyi) + dellamyi / diaglamyi
            dy = -dely / diagy + dlam / diagy
            dxsi = -xsi + epsi / (x - alfa) - (xsi * dx) / (x - alfa)
            deta = -eta + epsi / (beta - x) + (eta * dx) / (beta - x)
            dmu = -mu + epsi / y - (mu * dy) / y
            dzet = -zet + epsi / z - zet * dz / z
            ds = -s + epsi / lam - (s * dlam) / lam
            xx = np.concatenate([y, [z], lam, xsi, eta, mu, [zet], s])
            dxx = np.concatenate([dy, [dz], dlam, dxsi, deta, dmu, [dzet], ds])
            stmxx = np.max(-1.01 * dxx / xx)
            stmalfa = np.max(-1.01 * dx / (x - alfa)); stmbeta = np.max(1.01 * dx / (beta - x))
            steg = 1.0 / max(stmalfa, stmbeta, stmxx, 1.0)
            old = (x, y, z, lam, xsi, eta, mu, zet, s)
            itto = 0; resnew = 2 * resnorm
            while resnew > resnorm and itto < 50:
                itto += 1
                x, y, z, lam, xsi, eta, mu, zet, s = (o + steg * dd for o, dd in
                                                      zip(old, (dx, dy, dz, dlam, dxsi, deta, dmu, dzet, ds)))
                res = residual(x, y, z, lam, xsi, eta, mu, zet, s, epsi)
                resnew = np.linalg.norm(res)
                steg /= 2
            resnorm = resnew; resmax = np.abs(res).max()
        epsi *= 0.1
    return x, y, z, lam, dict(newton_steps=total)


# ---------------------------------------------------------------------------------------------------------- self-test
def _selftest():
    ok = True
    # (1) Svanberg (2007) toy problem: min x1^2+x2^2+x3^2, two ball constraints radius 3, bounds [0, 5], start (4, 3, 2).
    x = np.array([4.0, 3.0, 2.0]); st = None
    for k in range(60):
        f0, g0 = x @ x, 2 * x
        f = np.array([(x[0] - 5) ** 2 + (x[1] - 2) ** 2 + (x[2] - 1) ** 2 - 9, (x[0] - 3) ** 2 + (x[1] - 4) ** 2 + (x[2] - 3) ** 2 - 9])
        g = np.array([[2 * (x[0] - 5), 2 * (x[1] - 2), 2 * (x[2] - 1)], [2 * (x[0] - 3), 2 * (x[1] - 4), 2 * (x[2] - 3)]])
        xn, st, info = mma_update(x, f0, g0, f, g, 0.0, 5.0, st, move=1.0)
        if np.abs(xn - x).max() < 1e-7:
            x = xn; break
        x = xn
    ref = np.array([2.0175, 1.7800, 1.2375])
    e1 = np.abs(x - ref).max(); ok &= e1 < 2e-3
    print(f'toy problem: x = {np.round(x, 4)} (reference {ref}), |dx| = {e1:.1e}, iterations {k + 1}')
    # (2) compliance-like separable problem: min sum c_i / x_i s.t. sum x_i <= V; optimum x_i = V sqrt(c_i) / sum sqrt(c).
    rng = np.random.default_rng(0); n = 40; cc = rng.uniform(0.5, 3.0, n); V = 0.3 * n
    x = np.full(n, 0.3); st = None
    for k in range(200):
        f0, g0 = (cc / x).sum(), -cc / x ** 2
        f, g = np.array([x.sum() / V - 1]), np.ones((1, n)) / V
        xn, st, info = mma_update(x, f0, g0, f, g, 0.05, 1.0, st, move=0.1)
        conv = np.abs(xn - x).max() < 1e-6; x = xn
        if conv:
            break
    xs = V * np.sqrt(cc) / np.sqrt(cc).sum()
    e2 = np.abs(x - xs).max() / xs.max(); ok &= e2 < 1e-3
    print(f'separable compliance problem: relative error {e2:.1e} after {k + 1} iterations, volume {x.sum() / V:.6f}')
    # (3) many linear pair constraints (m > n branch): min sum c_i / x_i, sum x <= V, |x_i - x_{i+1}| <= 0.05.
    rows = []
    for i in range(n - 1):
        r = np.zeros(n); r[i], r[i + 1] = 1, -1; rows += [r, -r]
    A = np.array(rows) / 0.05
    x = np.full(n, 0.3); st = None
    for k in range(300):
        f0, g0 = (cc / x).sum(), -cc / x ** 2
        f = np.concatenate([[x.sum() / V - 1], A @ x - 1]); g = np.vstack([np.ones((1, n)) / V, A])
        xn, st, info = mma_update(x, f0, g0, f, g, 0.05, 1.0, st, move=0.1)
        conv = np.abs(xn - x).max() < 1e-7; x = xn
        if conv:
            break
    viol = max(x.sum() / V - 1, (A @ x - 1).max()); ok &= viol < 1e-4
    print(f'pair-constrained problem (m = {A.shape[0] + 1} > n = {n}): max constraint {viol:.1e}, iterations {k + 1}, '
          f'objective {(cc / x).sum():.6f}')
    print('SELFTEST', 'PASS' if ok else 'FAIL')
    return ok


if __name__ == '__main__':
    import sys
    if '--selftest' in sys.argv:
        sys.exit(0 if _selftest() else 1)
    print(__doc__)
