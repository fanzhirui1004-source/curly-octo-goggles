"""0021 light cut (76%): exact fields for consistent / nodal / macro port data; GP share; vf<0.1 share; port-layer share;
Rayleigh quotients; exact sensitivity density location; gamma dependence."""
import json, time, numpy as np
from common import *
case = sys.argv[1] if len(sys.argv) > 1 else 'fresh_train_0021_cover01_r1'
t0 = time.time()
C = cell.RealCell(case, gamma=1e-4); z = C.z; vf = C.vf
print('assembled', C.nb, time.time() - t0, flush=True)
nL = 16
Fc, Wf = face_loads(C, nL); Fn = nodal_planewave(C, nL)
print('face areas', {str(k): round(float(v.sum()), 4) for k, v in Wf.items()}, flush=True)
import os
if os.path.exists('fields_%s.npz' % case):
    f = np.load('fields_%s.npz' % case); Xc, Xn, Xm = f['Xc'], f['Xn'], f['Xm']
else:
    raise SystemExit('run fields first')
K = C.K; G = C.G
port = z['is_port']; box = z['is_box']
# distance of element centre to nearest box face that has port material (in h units)
cen = (C.cells + 0.5) / C.n
faces_with = [(ax, val) for (ax, val), W in Wf.items() if W.sum() > 1e-6]
dist = np.min([np.abs(cen[:, ax] - val) for ax, val in faces_with], axis=0) * C.n
touch_port = port[C.en].any(1)
res = {}
dM = np.load(B + '/dM_%s.npy' % case)       # (8, E, 125)
dKs = [np.einsum('em,mab->eab', dM[c], C.Tm) for c in range(8)]
for name, X in (('consistent', Xc), ('nodal', Xn), ('macro', Xm)):
    en = (X * (K @ X)).sum(0); gp = 1e-4 * (X * (G @ X)).sum(0) / en
    q = X[C.P]; R = en / (q * q).sum(0)
    ee = elem_energy(C, X); ee = ee / ee.sum(0)
    # exact sensitivity density per element and corner: -u_e^T dK_e u_e
    ue = X[C.dofs]                                   # (E,81,L)
    sd = np.stack([-np.einsum('eaj,eab,ebj->ej', ue, dKc, ue, optimize=True) for dKc in dKs])   # (8,E,L)
    s = sd.sum(1)                                    # (8,L)
    sa = np.abs(sd).sum(0); sa = sa / sa.sum(0)       # per element share of |density|
    grp = dict(vf01=vf < 0.1, vf01_05=(vf >= 0.1) & (vf < 0.5), vf05_999=(vf >= 0.5) & (vf < 0.999), full=vf >= 0.999,
               touch_port=touch_port, within2h=dist < 2, within4h=dist < 4)
    r = dict(R_median=float(np.median(R)), R_q=np.quantile(R, [0, .5, 1]).tolist(), gp_share_median=float(np.median(gp)),
             energy_share={k: float(np.median(ee[m].sum(0))) for k, m in grp.items()},
             sens_abs_share={k: float(np.median(sa[m].sum(0))) for k, m in grp.items()},
             sens_over_energy=float(np.median(np.linalg.norm(s, axis=0) / en)))
    res[name] = r
    print(name, json.dumps(r), flush=True)
res['element_fraction'] = {k: float(m.mean()) for k, m in dict(vf01=vf < 0.1, vf01_05=(vf >= 0.1) & (vf < 0.5), vf05_999=(vf >= 0.5) & (vf < 0.999), full=vf >= 0.999, touch_port=touch_port, within2h=dist < 2, within4h=dist < 4).items()}
print('element fractions', res['element_fraction'])
# relative sensitivity of element stiffness: ||dK_e/dtau|| / ||K_e|| (Frobenius, summed over corners) vs vf
dKn = np.sqrt(sum(dKc ** 2 for dKc in dKs).sum((1, 2)))
Kn = np.sqrt((C.Ke ** 2).sum((1, 2)))
ratio = dKn / np.maximum(Kn, 1e-300)
bins = [0, 1e-3, 0.01, 0.05, 0.1, 0.3, 0.5, 0.8, 0.999, 1.01]
res['dK_over_K_by_vf'] = [(bins[i], bins[i + 1], int(((vf >= bins[i]) & (vf < bins[i + 1])).sum()), float(np.median(ratio[(vf >= bins[i]) & (vf < bins[i + 1])])) if ((vf >= bins[i]) & (vf < bins[i + 1])).any() else None) for i in range(len(bins) - 1)]
print('median ||dK_e/dtau|| / ||K_e|| by vf bin:', res['dK_over_K_by_vf'], flush=True)
json.dump(res, open('expB1_%s.json' % case, 'w'), indent=1)
print('done', time.time() - t0)
