"""Clean verification, with modes isolated exactly and NO eigensolver in the loop.

Construction: R_hat = diag(e^s) @ R_star  =>  M = R_hat R_star^-1 = diag(e^s),
so H = M^T M = diag(e^{2s}) and mu_i = e^{2 s_i} exactly. Every quantity below is
then closed form, so a finite difference is exact to round-off.
"""
import numpy as np
np.random.seed(7)
d = 64
X = np.random.randn(d, d); A = X@X.T + d*np.eye(d)
Rs = np.linalg.cholesky(A).T
logdetRs = np.sum(np.log(np.diag(Rs)))

def quantities(s):
    """s: log-scale per mode. Returns (relative loss, D) with mu = e^{2s}."""
    Rh = np.diag(np.exp(s)) @ Rs
    M  = Rh @ np.linalg.inv(Rs)                       # == diag(e^s)
    L_rel = np.linalg.norm(M - np.eye(d),'fro')**2/d  # exactly Codex's relative loss
    trH   = np.linalg.norm(M,'fro')**2                # what the probes estimate
    logdetH = 2*(np.sum(np.log(np.diag(Rh))) - logdetRs)   # from PREDICTED log pivots
    D = (trH - logdetH - d)/d
    return L_rel, D

print("=== reproduce Codex LOSS_IDENTITY.json, one mode off, rest exact ===")
print(f"{'mu':>10}{'rel MSE(1 mode)':>18}{'D*d (1 mode)':>15}{'d(relMSE)/dlogR':>18}{'d(D*d)/dlogR':>15}")
h = 1e-6
for mu in (1.0, 1e-2, 1e-6, 1e-12):
    s = np.zeros(d); s[0] = 0.5*np.log(mu)
    L0, D0 = quantities(s)
    sp = s.copy(); sp[0] += h; sm = s.copy(); sm[0] -= h
    Lp, Dp = quantities(sp); Lm, Dm = quantities(sm)
    # isolate the single mode's contribution (the other d-1 modes are exact => 0)
    print(f"{mu:>10.0e}{L0*d:>18.10f}{D0*d:>15.6f}{(Lp-Lm)/(2*h)*d:>18.3e}{(Dp-Dm)/(2*h)*d:>15.6f}")

print("\nCodex's published table for comparison:")
print("   mu      relative_factor_MSE          D     dMSE/dlogfactor   dD/dlogfactor")
for mu, m, D_, gm, gD in [(1.0,0.0,0.0,0.0,0.0),(1e-2,0.81,3.6151701859880907,-0.18,-1.98),
                          (1e-6,0.998001,12.815511557964273,-0.001998,-2.0),
                          (1e-12,0.9999980000009999,26.631021115929546,-1.999998e-06,-2.0)]:
    print(f"{mu:>8.0e}{m:>22.10f}{D_:>14.6f}{gm:>18.3e}{gD:>15.6f}")

print("\n=== consequence: pressure ratio on a soft mode ===")
for mu in (1e-2, 1e-4, 1e-6, 1e-12):
    gm = abs(2*(mu-np.sqrt(mu)))    # d/dt (sqrt(mu)e^t-1)^2 at t=0
    gD = abs(2*(mu-1))
    print(f"  mu={mu:>8.0e}   |d(relMSE)|={gm:.3e}   |d(D)|={gD:.4f}   ratio={gD/gm:.3e}")
