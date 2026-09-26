import numpy as np, scipy.sparse as sp, scipy.linalg as sl
import importlib.util
spec = importlib.util.spec_from_file_location('ts', 'toy_strip.py'); src = open('toy_strip.py').read().split('for t in (1.5')[0]
ns = {}; exec(src, ns)
for nx, tl in [(32, 1.5), (16, 1.5), (8, 1.5), (32, 2.5), (16, 2.5), (8, 2.5)]:
    K, X, P, I, ny = ns['strip'](tl / 32 * 1.0, nx=nx) if False else ns['strip'](tl / 32, nx=nx)
    # strip() uses hx=1/nx over length 1: rescale so that h=1/32 always -> length = nx/32
    KII = K[I][:, I].toarray(); d = np.diag(KII)
    w = sl.eigh(KII, np.diag(d), eigvals_only=True)
    print('span %2d h (L=%.3f), t=%.1f h: x_min=lam_min/lam_max=%.2e  x_2=%.2e x_5=%.2e | #modes with x<1e-2: %d of %d | sweeps for Chebyshev(1/30) to damp x_min 2x ~ n/a' % (nx, nx / 32, tl, w[0] / w[-1], w[1] / w[-1], w[4] / w[-1], (w / w[-1] < 1e-2).sum(), len(w)))
