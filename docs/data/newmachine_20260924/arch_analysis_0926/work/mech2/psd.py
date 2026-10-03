import numpy as np
from common import *
C = cell.RealCell('fresh_train_0021_cover01_r1', gamma=1e-4)
dM = np.load(B + '/dM_fresh_train_0021_cover01_r1.npy')
vf = C.vf; rng = np.random.default_rng(0)
idx = rng.choice(np.flatnonzero((vf > 1e-3) & (vf < 0.999)), 300, replace=False)
neg = []
for c in range(8):
    dK = np.einsum('em,mab->eab', dM[c][idx], C.Tm)
    w = np.linalg.eigvalsh(dK)
    mx = np.abs(w).max(1); mx[mx == 0] = 1
    neg.append(w.min(1) / mx)
neg = np.array(neg)
print('min eig / max|eig| of dK_e/dtau_c over 300 cut elements x 8 corners: quantiles', np.quantile(neg, [0, .01, .1, .5]).round(4))
dMs = dM.sum(0)[idx]; dK = np.einsum('em,mab->eab', dMs, C.Tm); w = np.linalg.eigvalsh(dK)
print('sum over corners (uniform thickening): min/max', np.quantile(w.min(1) / np.abs(w).max(1), [0, .01, .1, .5]).round(4))
