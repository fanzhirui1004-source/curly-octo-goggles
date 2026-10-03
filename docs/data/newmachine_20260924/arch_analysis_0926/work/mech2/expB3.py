"""0021: Saint-Venant decay. Loads on the y=0 face only: (a) per-face SELF-EQUILIBRATED tractions (Legendre degree 1..3 fluctuations,
resultant force and moment removed on the face itself), (b) net traction on y=0 balanced on y=1 (transmitted load).
Energy fraction vs distance from the loaded face; exact sensitivity density vs distance."""
import json, time, numpy as np
from common import *
case = 'fresh_train_0021_cover01_r1'
C = cell.RealCell(case, gamma=1e-4); z = C.z; xyz = C.xyz
Wf = {k: loads.face_weights(C, k[0], k[1], z['taus'], z['normal'], float(z['offset']))[0] for k in [(1, 0.0), (1, 1.0)]}
W0, W1 = Wf[(1, 0.0)], Wf[(1, 1.0)]
Qr = loads.rigid(xyz)
u, v = 2 * xyz[:, 0] - 1, 2 * xyz[:, 2] - 1
rng = np.random.default_rng(5)
def selfeq(W, deg_min, nL):
    Bs = [np.ones_like(u), u, v, u * v, 1.5 * u * u - .5, 1.5 * v * v - .5, u * (1.5 * v * v - .5), v * (1.5 * u * u - .5), 2.5 * u ** 3 - 1.5 * u, 2.5 * v ** 3 - 1.5 * v]
    deg = [0, 1, 1, 2, 2, 2, 3, 3, 3, 3]
    Bm = np.stack([b for b, d in zip(Bs, deg) if d >= deg_min], 1)
    F = np.zeros((C.N, 3, nL))
    for j in range(nL): F[:, :, j] = W[:, None] * (Bm @ rng.standard_normal((Bm.shape[1], 3)))
    F = F.reshape(-1, nL); wd = np.repeat(W, 3)
    WR = wd[:, None] * Qr
    return F - WR @ np.linalg.solve(Qr.T @ WR, Qr.T @ F)     # equilibrated on the face itself
sets = {'selfeq_deg>=1': selfeq(W0, 1, 8), 'selfeq_deg>=2': selfeq(W0, 2, 8), 'selfeq_deg>=3': selfeq(W0, 3, 8)}
T = np.zeros((C.N, 3, 3)); 
for d in range(3): T[:, d, d] = W0 / W0.sum() - W1 / W1.sum()
sets['transmitted_y0_to_y1'] = T.reshape(-1, 3)
N = Neu(C)
cen_y = (C.cells[:, 1] + 0.5)          # element index distance from y=0 face, in h
out = {}
for name, F in sets.items():
    X, F2 = N.solve(F)
    assert np.abs(F2 - F).max() < 1e-10 * np.abs(F).max()
    ee = elem_energy(C, X); ee = ee / ee.sum(0)
    cum = [float(np.median(ee[cen_y < d].sum(0))) for d in (1, 2, 3, 4, 6, 8, 12, 16, 24)]
    gp = float(np.median(1e-4 * (X * (C.G @ X)).sum(0) / (X * (C.K @ X)).sum(0)))
    out[name] = dict(cum_energy_within_d_h=dict(zip([1, 2, 3, 4, 6, 8, 12, 16, 24], cum)), gp_share=gp)
    print(name, 'GP share %.4f' % gp, 'energy fraction within d elements of the loaded face (d=1,2,3,4,6,8,12,16,24):', np.round(cum, 3).tolist(), flush=True)
print('element fraction within d:', [round(float((cen_y < d).mean()), 3) for d in (1, 2, 3, 4, 6, 8, 12, 16, 24)])
json.dump(out, open('expB3.json', 'w'), indent=1)
N.free()
