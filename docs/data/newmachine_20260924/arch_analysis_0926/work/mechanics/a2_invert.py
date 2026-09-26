"""Invert measured Chebyshev decay curves (energy excess vs k) for the energy-weighted spectral measure of the network
error over x = lambda / lmax of D^-1 K_II. p_k(x) = T_k((1+a-2x)/(1-a)) / T_k((1+a)/(1-a)), a = 1/30 (same for any lmax)."""
import numpy as np
from scipy.optimize import nnls
a = 1 / 30
def T(k, y):
    y = np.asarray(y, float)
    return np.where(np.abs(y) <= 1, np.cos(k * np.arccos(np.clip(y, -1, 1))), np.cosh(k * np.arccosh(np.maximum(np.abs(y), 1))) * np.sign(y) ** k)
def p(k, x):
    return T(k, (1 + a - 2 * x) / (1 - a)) / T(k, (1 + a) / (1 - a))
ks = np.array([0, 1, 2, 4, 8, 16, 32])
curves = {
 '2010 heavy force_c': [0.3495, 0.1255, 0.05904, 0.03568, 0.002052, 0.0001795, 3.33e-05],
 '2010 heavy force':   [0.1951, 0.07479, 0.03830, 0.02206, 0.002277, 0.000394, 7.42e-05],
 '2003 medium force_c': [0.135, 0.099, 0.079, 0.061, 0.047, 0.038, 0.0295],
 '2003 medium force':   [0.072, 0.051, 0.040, 0.031, 0.023, 0.019, 0.0145],
 '2006 medium force_c': [0.036, 0.026, 0.020, 0.0145, 0.0104, 0.0081, 0.0061],
 '2006 medium force':   [0.022, 0.014, 0.010, 0.0072, 0.0046, 0.0035, 0.0025],
}
# bin edges in x
edges = np.r_[0, np.logspace(-5, 0, 26)]
xs = [np.linspace(edges[i], edges[i + 1], 200)[1:-1] if i else np.logspace(-8, -5, 200) for i in range(len(edges) - 1)]
A = np.array([[np.mean(p(k, x) ** 2) for x in xs] for k in ks])
# reference: where p_k^2 < 0.1 (i.e. component damped 10x in energy)
for k in [1, 2, 4, 8, 16, 32]:
    xx = np.logspace(-6, 0, 20001); v = p(k, xx) ** 2
    print('k=%2d: energy damping >=10x for x >= %.2e ; >= 2x for x >= %.2e' % (k, xx[np.argmax(v < 0.1)], xx[np.argmax(v < 0.5)]))
for name, c in curves.items():
    c = np.array(c); r = c / c[0]
    W = 1 / np.maximum(r, 1e-12)
    w, res = nnls(A * W[:, None], r * W)
    fit = A @ w
    cum = np.cumsum(w)
    def below(xc):
        return sum(w[i] for i in range(len(w)) if edges[i + 1] <= xc + 1e-15)
    print('\n%s: total=%.3f  share(x<1e-3)=%.3f  share(x<1e-2)=%.3f share(x<1/30)=%.3f  share(x<0.1)=%.3f' % (name, w.sum(), below(1e-3), below(1e-2), below(10**(np.log10(1/30))), below(0.1)))
    print('  fit/obs:', np.round(fit / r, 3).tolist())
    print('  mass per bin (x_hi, share):', [(float('%.1e' % edges[i + 1]), round(float(w[i]), 3)) for i in range(len(w)) if w[i] > 0.005])
