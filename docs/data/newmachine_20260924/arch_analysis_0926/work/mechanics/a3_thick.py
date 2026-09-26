import numpy as np
rng = np.random.default_rng(0)
phi = lambda x: np.cos(2*np.pi*x).sum(-1)
grad = lambda x: -2*np.pi*np.sin(2*np.pi*x)
# points on phi=0: random lines, bisection
P = []
while len(P) < 20000:
    x0 = rng.random(3); d = rng.standard_normal(3); d /= np.linalg.norm(d)
    ts = np.linspace(-0.3, 0.3, 61); v = phi(x0 + ts[:, None]*d)
    idx = np.flatnonzero(np.sign(v[:-1]) != np.sign(v[1:]))
    for i in idx:
        a, b = ts[i], ts[i+1]
        for _ in range(50):
            m = (a+b)/2
            if np.sign(phi(x0+m*d)) == np.sign(phi(x0+a*d)): a = m
            else: b = m
        P.append(x0 + a*d)
P = np.array(P) % 1.0
g = grad(P); gn = np.linalg.norm(g, axis=1); nrm = g/gn[:, None]
# weight by area: line sampling biases by |cos| to direction; use uniform-area resampling via weights 1/|d.n| is messy; instead report quantiles of the point cloud and of |grad| directly
def thick(tau):
    t = np.zeros(len(P))
    for s in (+1, -1):
        a = np.zeros(len(P)); b = np.full(len(P), 0.3)
        for _ in range(60):
            m = (a+b)/2; inside = np.abs(phi(P + s*m[:, None]*nrm)) <= tau
            a = np.where(inside, m, a); b = np.where(inside, b, m)
        t += a
    return t
h = 1/32
print('|grad phi| on surface quantiles (0,5,25,50,75,95,100%):', np.quantile(gn, [0,.05,.25,.5,.75,.95,1]).round(2), ' 2pi=', round(2*np.pi,2), ' 2pi*sqrt3=', round(2*np.pi*3**.5,2))
for tau in [0.1755, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6994, 0.8775]:
    t = thick(tau)
    q = np.quantile(t/h, [0.05, .25, .5, .75, .95])
    print('tau=%.4f  wall thickness / h (Q2 element, h=1/32): p5 %.2f p25 %.2f p50 %.2f p75 %.2f p95 %.2f | node layers (h/2) median %.1f | linear estimate 2tau/|grad| median %.2f h' % (tau, *q, 2*np.median(t)/h, np.median(2*tau/gn)/h))
