"""Why the relative loss HEDGES TOWARD SOFTNESS, and by how much.

Observation to explain: in the measured A/B, closed_logdet_gap is POSITIVE on every
baseline row (+2263..+4718, prediction too stiff) and NEGATIVE on every relative row
(-10651..-25205, prediction too soft). The relative loss flipped the global bias.

Model: the network must commit one log-scale u for a mode whose correct scale it
knows only up to noise eps ~ N(0, sigma^2). mu = e^{2(u - eps)}. Minimise E[loss].
"""
import numpy as np
from scipy.optimize import minimize_scalar
rng = np.random.default_rng(0)

def optimum(loss, sigma, n=4_000_000):
    eps = rng.normal(0, sigma, n)
    f = lambda u: loss(np.exp(2*(u-eps))).mean()
    return minimize_scalar(f, bounds=(-6*sigma-1, 6*sigma+1), method='bounded',
                           options=dict(xatol=1e-7)).x

relative  = lambda mu: (np.sqrt(mu) - 1)**2          # Codex's relative auxiliary
stein     = lambda mu: mu - np.log(mu) - 1           # D, the necessary-condition metric
jeffreys  = lambda mu: mu + 1/mu - 2                 # symmetrised

print(f"{'sigma':>7}{'relative u*':>14}{'predicted':>12}{'D u*':>10}{'predicted':>12}{'Jeffreys u*':>14}{'predicted':>12}")
for sigma in (0.1, 0.2, 0.3, 0.5):
    ur, ud, uj = (optimum(l, sigma) for l in (relative, stein, jeffreys))
    print(f"{sigma:>7.2f}{ur:>14.5f}{-1.5*sigma**2:>12.5f}{ud:>10.5f}{-sigma**2:>12.5f}{uj:>14.5f}{0.0:>12.5f}")

print("\n=> relative hedges soft by 1.5*sigma^2, D by sigma^2, Jeffreys not at all.")
print("   A soft mode also costs the relative loss AT MOST 1, however wrong it is,")
print("   so once a mode has collapsed there is no gradient left to recover it:")
for mu in (1e-2, 1e-6, 1e-12, 1e-16):
    print(f"     mu={mu:>7.0e}   relative={relative(mu):.6f} (capped at 1)   D={stein(mu):8.3f} (unbounded)")
