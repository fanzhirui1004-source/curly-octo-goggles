"""CPU checks of homog_macro.py --clamp cut on the plate layout: (1) the clamped plane section has the analytic area (length
of the line n . X = b inside the lattice box times the plate depth), split into meshed and unmeshed parts; (2) compliance
converges as the penalty factor grows (1e2 ... 1e6); (3) adjoint gradient against central differences at a graded design.
Usage: python t_homog_cut.py <layout.json> <homog_cells.json> [--load y,min] [--load-dir x]
"""
import sys, json, time, argparse
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import homog_macro as HM

ap = argparse.ArgumentParser(); ap.add_argument('layout'); ap.add_argument('cells')
ap.add_argument('--load', default='y,min'); ap.add_argument('--load-dir', default='x'); ap.add_argument('--m', type=int, default=6)
A = ap.parse_args()
L = json.loads(Path(A.layout).read_text()); mat = HM.Material.from_json(A.cells)
shape, n, b = L['shape'], np.asarray(L['normal'], float), float(L['b_global'])
out = {}

# (1) analytic section length of {n . x = b} in [0, X] x [0, Y] (n_z = 0), times the depth Z
X, Y, Z = shape
pts = []
for x in (0.0, X):
    y = (b - n[0] * x) / n[1]
    if 0 <= y <= Y: pts.append((x, y))
for y in (0.0, Y):
    x = (b - n[1] * y) / n[0]
    if 0 <= x <= X: pts.append((x, y))
pts = np.unique(np.round(np.asarray(pts), 12), axis=0)
area_exact = float(np.linalg.norm(pts[0] - pts[-1])) * Z
t = time.perf_counter()
M = HM.Macro(shape, L['cells'], n, b, A.m, ('cut', ''), A.load.split(','), A.load_dir, mat, pen=1e4)
out['build_s'] = time.perf_counter() - t
out['section'] = dict(exact=area_exact, meshed=M.pen_area, unmeshed=M.pen_missed,
                      rel_err=(M.pen_area + M.pen_missed - area_exact) / area_exact, unmeshed_share=M.pen_missed / area_exact)
assert abs(out['section']['rel_err']) < 1e-9, out['section']

# (2) penalty convergence at the uniform start design
tv = np.zeros(M.nv)
for k, c in enumerate(L['cells']):
    tv[M.vid[k]] = c['tau_corners']
Cs = {}
for pen in (1e2, 1e3, 1e4, 1e5, 1e6):
    Mp = HM.Macro(shape, L['cells'], n, b, A.m, ('cut', ''), A.load.split(','), A.load_dir, mat, pen=pen)
    t = time.perf_counter(); C, g, V, dV, u = Mp.solve(tv); Cs[pen] = dict(C=C, solve_s=time.perf_counter() - t, gnorm=float(np.linalg.norm(g)))
out['penalty'] = {f'{k:.0e}': v for k, v in Cs.items()}
out['penalty_rel_change_1e4_1e5'] = abs(Cs[1e5]['C'] - Cs[1e4]['C']) / Cs[1e5]['C']
out['penalty_rel_change_1e5_1e6'] = abs(Cs[1e6]['C'] - Cs[1e5]['C']) / Cs[1e6]['C']

# (3) adjoint gradient vs central differences at a graded design
rng = np.random.default_rng(3)
tg = np.clip(tv + 0.08 * rng.uniform(-1, 1, M.nv), 0.2, 0.65)
C, g, V, dV, _ = M.solve(tg)
errs = []
for v in rng.choice(M.nv, 6, replace=False):
    h = 1e-6; tp = tg.copy(); tp[v] += h; tm = tg.copy(); tm[v] -= h
    errs.append(abs((M.solve(tp, False)[0] - M.solve(tm, False)[0]) / (2 * h) - g[v]) / np.abs(g).max())
out['adjoint_vs_fd_max'] = float(max(errs))
assert out['adjoint_vs_fd_max'] < 1e-5
out['elements'], out['nodes'] = len(M.els), M.nnode
print(json.dumps(out, indent=1, default=float))
print('T_HOMOG_CUT PASS')
