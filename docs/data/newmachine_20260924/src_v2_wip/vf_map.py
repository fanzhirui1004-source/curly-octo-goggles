"""Volume fraction of the sheets |f| <= tau (Monte Carlo, 4M points) for P and G, and the tau map P -> G with equal
volume fraction (monotone interpolation). Writes vf_map.json next to the output path given."""
import json, sys
import numpy as np
import surfaces as SF
rng = np.random.default_rng(0)
X = rng.random((4_000_000, 3))
fP, fG = np.abs(SF.f_np(X, 'P')), np.abs(SF.f_np(X, 'G'))
tau = np.linspace(0.0, 1.5, 1501)
vP = np.searchsorted(np.sort(fP), tau) / len(fP)
vG = np.searchsorted(np.sort(fG), tau) / len(fG)
def p2g(t):
    v = np.interp(t, tau, vP)
    return float(np.interp(v, vG, tau))
lo, hi = 0.1755, 0.6983                                            # P training contract (corner values)
rec = dict(points=len(X), tau=tau[::10].tolist(), vf_P=vP[::10].tolist(), vf_G=vG[::10].tolist(),
           P_range=[lo, hi], VF_range=[float(np.interp(lo, tau, vP)), float(np.interp(hi, tau, vP))],
           G_range=[p2g(lo), p2g(hi)], examples={f'{t:.4f}': p2g(t) for t in (0.1755, 0.25, 0.3583, 0.45, 0.55, 0.6983)})
json.dump(rec, open(sys.argv[1], 'w'), indent=1)
print(json.dumps({k: rec[k] for k in ('P_range', 'VF_range', 'G_range', 'examples')}))
