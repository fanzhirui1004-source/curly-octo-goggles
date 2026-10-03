"""Discretisation check of the macroscale (homogenised) plate model of Section 5.10 (Codex review item 1).
Re-evaluates, with the unchanged homog_macro.Macro, the compliance of the cut-clamped plate (clamp on the cut plane by
penalty, unit traction in x on the face y = min) at the uniform start (tau = 0.40) for Q1 meshes of m = 6 (as in the
paper) and 12 elements per cell and axis, each with penalty factors 1e6 (paper) and 1e8.
Read-only use of src_v2_wip/homog_macro.py and of the archived layout / cell tensors.  CPU only.
Usage: python3 homog_conv.py   (writes homog_conv.json next to this file)"""
import json, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[5]
sys.path.insert(0, str(REPO / 'docs/data/newmachine_20260924/src_v2_wip'))
import homog_macro as HM
try:                                                   # MKL PARDISO for the linear solve when available (same system, faster than SuperLU)
    import pypardiso
    HM.spla.spsolve = lambda A, b: pypardiso.spsolve(A.tocsr(), np.asarray(b))
    SOLVER = 'pypardiso'
except ImportError:
    SOLVER = 'scipy.spsolve'

FIN = HERE.parent
L = json.loads((FIN / 'plate841.json').read_text())
mat = HM.Material.from_json(REPO / 'docs/paper_p1/review_r1/results/X6_opt/homog/homog_cells.json')
meta = json.loads((FIN / 'HC_x/meta.json').read_text())['args']


def designs(M):
    tv0 = np.zeros(M.nv)
    for n, c in enumerate(L['cells']):
        tv0[M.vid[n]] = c['tau_corners']
    out = {'uniform_0.40': tv0, 'homog_final': np.load(FIN / 'HC_x/final_tv.npy')}
    nl = FIN / 'cplateN/final_layout.json'
    if nl.exists():
        LN = json.loads(nl.read_text()); tvN = np.zeros(M.nv)
        for n, c in enumerate(LN['cells']):
            tvN[M.vid[n]] = c['tau_corners']
        out['nice_final'] = tvN
    return out


rows = []
CASES = [(6, 1e6), (6, 1e8), (12, 1e6), (12, 1e8)]
for m, pen in CASES:
    t = time.perf_counter()
    M = HM.Macro(L['shape'], L['cells'], L['normal'], L['b_global'], m, ('cut', ''), meta['load'].split(','), meta['load_dir'], mat, pen=pen)
    rec = dict(m=m, pen=pen, solver=SOLVER, elements=len(M.els), nodes=M.nnode)
    for name, tv in list(designs(M).items())[:1]:                      # uniform start only
        C, g, V, dV, _ = M.solve(tv, want_grad=False) if 'want_grad' in HM.Macro.solve.__code__.co_varnames else M.solve(tv)
        rec[name] = dict(C=float(C), V=float(V))
    rec['seconds'] = time.perf_counter() - t
    rows.append(rec); print(json.dumps(rec), flush=True)
(HERE / 'homog_conv.json').write_text(json.dumps(dict(paper_m6_uniform_C=55.19222070779435, exact_uniform_C=75.599,
                                                      exact_homog_final_C=80.198, exact_nice_final_C=78.287, rows=rows), indent=1))
