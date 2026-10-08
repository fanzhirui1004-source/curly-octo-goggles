"""Toy lattice optimisation with NODAL (shared-corner) thickness variables: how do per-cell sensitivity errors translate
into design regret (exact compliance of the design found with erroneous gradients / exact-gradient optimum - 1)?
2D plane-stress Q4 'cells', cell stiffness factor E_c = tau_bar_c^m (tau_bar = mean of its 4 corner taus), volume = sum tau_bar.
Cantilever (clamped left edge, unit tip load at mid right), OC update on corner taus, bounds [tau_lo, tau_hi]."""
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spl, json
nx, ny, m = 40, 20, 1.5
tau_lo, tau_hi, vfrac = 0.18, 0.8, 0.40
def ke_q4(nu=0.3):
    k = np.array([1/2-nu/6, 1/8+nu/8, -1/4-nu/12, -1/8+3*nu/8, -1/4+nu/12, -1/8-nu/8, nu/6, 1/8-3*nu/8])
    KE = 1/(1-nu**2)*np.array([[k[0],k[1],k[2],k[3],k[4],k[5],k[6],k[7]],[k[1],k[0],k[7],k[6],k[5],k[4],k[3],k[2]],
        [k[2],k[7],k[0],k[5],k[6],k[3],k[4],k[1]],[k[3],k[6],k[5],k[0],k[7],k[2],k[1],k[4]],[k[4],k[5],k[6],k[7],k[0],k[1],k[2],k[3]],
        [k[5],k[4],k[3],k[2],k[1],k[0],k[7],k[6]],[k[6],k[3],k[4],k[1],k[2],k[7],k[0],k[5]],[k[7],k[2],k[1],k[4],k[3],k[6],k[5],k[0]]])
    return KE
KE = ke_q4()
nel = nx*ny; nnode = (nx+1)*(ny+1)
elx, ely = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij'); elx = elx.ravel(); ely = ely.ravel()
n1 = (ny+1)*elx+ely; n2 = (ny+1)*(elx+1)+ely
cn = np.stack([n1+1, n2+1, n2, n1], 1)            # 4 corner nodes (shared)
edof = np.stack([2*n1+2, 2*n1+3, 2*n2+2, 2*n2+3, 2*n2, 2*n2+1, 2*n1, 2*n1+1], 1)
iK = np.kron(edof, np.ones((8, 1))).ravel(); jK = np.kron(edof, np.ones((1, 8))).ravel()
fixed = np.arange(2*(ny+1)); free = np.setdiff1d(np.arange(2*nnode), fixed)
F = np.zeros(2*nnode); F[2*(nx*(ny+1)+ny//2)+1] = -1.0
def solve(t):
    tb = t[cn].mean(1); E = tb**m
    K = sp.csc_matrix((np.kron(E, KE.ravel()), (iK, jK)), shape=(2*nnode,)*2)
    u = np.zeros(2*nnode); u[free] = spl.spsolve(K[free][:, free], F[free])
    ce = np.einsum('ei,ij,ej->e', u[edof], KE, u[edof])
    C = float(E @ ce)
    dcdtb = -m*tb**(m-1)*ce                          # per cell
    s_cell = np.repeat(dcdtb[:, None]/4, 4, 1)      # cell x 4 corners
    return C, s_cell, E*ce/C                         # energy share per cell
def assemble(s_cell):
    g = np.zeros(nnode); np.add.at(g, cn.ravel(), s_cell.ravel()); return g
dV = assemble(np.full((nel, 4), 0.25))
def optimise(err_fn, iters=150, seed=0):
    rng = np.random.default_rng(seed)
    t = np.full(nnode, 0.5); t *= (vfrac*nel/((t[cn].mean(1)).sum()))
    Vt = vfrac*nel
    for it in range(iters):
        C, s, w = solve(t)
        s_hat = err_fn(s, w, rng)
        g = assemble(s_hat)
        l1, l2 = 1e-9, 1e9
        while (l2-l1)/(l1+l2) > 1e-6:
            lm = 0.5*(l1+l2)
            tn = np.clip(t*np.sqrt(np.maximum(-g, 1e-30)/(lm*dV)), np.maximum(tau_lo, t-0.05), np.minimum(tau_hi, t+0.05))
            if tn[cn].mean(1).sum() > Vt: l1 = lm
            else: l2 = lm
        t = tn
    C, s, w = solve(t)
    return t, C, w
exact = lambda s, w, r: s
t0, C0, w0 = optimise(exact)
print('exact-gradient optimum C = %.5f; energy-share quantiles (cells): ' % C0, np.quantile(w0, [.1, .5, .9, .99]).round(5), 'min %.2e' % w0.min())
res = dict(C0=C0)
def mk(rule, d):
    def f(s, w, rng):
        if rule == 'uniform':      dl = np.full(len(w), d)
        elif rule == 'lowshare':   dl = np.where(w < 1e-3, 2*d, d)          # like the gate: 2x worse on 0.1%-share cells
        elif rule == 'share_inv':  dl = np.minimum(d*np.sqrt(np.median(w)/w), 1.0)   # rel err grows as share falls
        elif rule == 'boundary':   # 'cut cells' along top/bottom/right boundary rows: 5x worse
            dl = np.full(len(w), d); dl[(ely == 0) | (ely == ny-1) | (elx == nx-1)] = 5*d
        xi = rng.standard_normal(s.shape); xi /= np.linalg.norm(xi, axis=1, keepdims=True)
        return s + dl[:, None]*np.linalg.norm(s, axis=1, keepdims=True)*xi
    return f
def mkbias(rule, d):
    def f(s, w, rng):   # systematic, same sign every iteration: cells look stiffer -> |s| over/under-estimated by type
        dl = np.full(len(w), 0.0)
        if rule == 'bias_boundary': dl[(ely == 0) | (ely == ny-1) | (elx == nx-1)] = d
        if rule == 'bias_all': dl[:] = d
        return s*(1+dl[:, None])
    return f
rows = []
for name, fn in [('uniform 3%', mk('uniform', .03)), ('uniform 6%', mk('uniform', .06)), ('uniform 15%', mk('uniform', .15)),
                 ('lowshare 3%/6%', mk('lowshare', .03)), ('share_inv 3% at median share', mk('share_inv', .03)),
                 ('boundary cells 5x (3%->15%)', mk('boundary', .03)),
                 ('bias all +10%', mkbias('bias_all', .10)), ('bias boundary +15%', mkbias('bias_boundary', .15)),
                 ('bias boundary -15%', mkbias('bias_boundary', -.15)), ('bias boundary +35%', mkbias('bias_boundary', .35))]:
    regs = []
    for seed in range(3):
        t, C, w = optimise(fn, seed=seed)
        regs.append(C/C0-1)
    rows.append((name, float(np.mean(regs)), float(np.max(regs))))
    print('%-32s regret mean %.4f%%  max %.4f%%   (design dist %.3f)' % (name, 100*np.mean(regs), 100*np.max(regs), np.linalg.norm(t-t0)/np.linalg.norm(t0)))
res['rows'] = rows
# second-order prediction: 0.5*m/(m+1)*sum w delta^2 (separable, interior variables only)
print('2nd-order separable prediction for uniform 3%%: %.4f%%' % (100*0.5*m/(m+1)*0.03**2))
json.dump(res, open('regret_toy.json', 'w'), indent=1)
