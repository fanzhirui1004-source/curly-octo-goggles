"""Step 4: do the old learned operators (E1: M_q, E2: F_bg; 60k steps) also fail on the hard boundary directions found
by the solver work on the moderate case fresh_train_0013_d0_v1 (a training case of both)?

All directions live in the teacher-whitened quotient frame (REFERENCE_RQ, RIGID_Q of targets/<case>_v1), which is the
frame of both our witnesses and Codex's spectra. For a direction z the learned operator's
  compliance ratio  r_c(z) = z^T C z / z^T z      (C = whitened learned compliance; teacher: 1)
  stiffness ratio   r_s(z) = z^T C^-1 z / z^T z   (comparable to mu; teacher: 1)
and its place in the arm's own full spectrum (percentile of |ln mu|). Also: overlap of the arm's own extreme
eigenvectors with z, and whether they sit on the same boundary coordinates (top-coordinate sets of B^T R^-1 z).
Usage: attr_e12.py E1|E2 <output json>"""
import json, sys, time
from pathlib import Path
import numpy as np
from scipy import linalg
from threadpoolctl import threadpool_limits

arm, out = sys.argv[1], Path(sys.argv[2])
R = Path('/root/autodl-tmp/CUTFEM_FRESH_GP_20260921')
T = Path('/root/autodl-tmp/CLAUDE_TAKEOVER_20260923')
case = 'fresh_train_0013_d0_v1'
src = R / ('e1_mq_spectrum_admission_source_04' if arm == 'E1' else 'e2_owned_spectrum_source_01')
sys.path.insert(0, str(src))
from research.fresh_gp.algebra import RigidComplement
from research.fresh_gp.compile_target import unpack_upper
from research.fresh_gp.numerics import gram_rows
bg = R / 'targets' / (case + '_v1')
rq = np.load(bg / 'REFERENCE_RQ.npy'); d = rq.shape[0]
rec = dict(arm=arm, case=case, d=d)
t0 = time.perf_counter()

# frame check: Codex's RigidComplement vs the frozen run_mechanics.Quotient used for our witnesses
sys.path.insert(0, str(R / 'diagnostics' / 'MECHANICS_NETWORK_20260922_01' / 'source_v1'))
import importlib.util, torch
spec = importlib.util.spec_from_file_location('rm', R / 'diagnostics' / 'MECHANICS_NETWORK_20260922_01' / 'source_v1' / 'run_mechanics.py')
rm = importlib.util.module_from_spec(spec); spec.loader.exec_module(rm)
rigid = np.load(bg / 'RIGID_Q.npy')
hq = RigidComplement(rigid); quo = rm.Quotient(rigid, torch.device('cpu'))
v = np.random.default_rng(0).standard_normal((rigid.shape[0], 3))
ours = quo.reduce(torch.as_tensor(v)).numpy(); theirs = hq.left(v, transpose=True)[6:]
rec['frame_check_relative'] = float(np.linalg.norm(ours - theirs) / np.linalg.norm(ours))

with threadpool_limits(16):
    if arm == 'E1':
        h = unpack_upper(R / 'predictions' / f'E1_SELECTED_TRAIN_060000_{case}_01' / 'RAW_H_UPPER.npy', rigid.shape[0])
        hq.left(h, transpose=True, overwrite=True); hq.right(h, overwrite=True)
        m = np.ascontiguousarray(h[6:, 6:]); del h
        x = rq @ m; del m
    else:
        from research.fresh_gp.evaluate import frames
        meta = json.loads((bg / 'RESULT.json').read_text())
        j, w, _ = frames(bg)
        h = unpack_upper(R / 'predictions' / f'E2_SELECTED_TRAIN_060000_{case}_01' / 'RAW_H_UPPER.npy', meta['m'])
        m = j @ h @ j.T; del h, j
        x = w @ m; del w, m
    C = gram_rows(x); del x
    C = (C + C.T) / 2
    rec['build_seconds'] = time.perf_counter() - t0
    L = linalg.cho_factor(C, lower=True, check_finite=False)

    ev = R / 'assessments' / f'{arm}_SELECTED_TRAIN_060000_{case}_SPECTRUM_01'
    mu = np.load(ev / 'MU.npy'); ext = np.load(ev / 'EIGEN_BEFORE_LOGDET.npz')['extreme_vectors']
    lmu = np.sort(np.abs(np.log(mu)))
    rec['arm_spectrum'] = dict(mu_min=float(mu.min()), mu_max=float(mu.max()),
                               abs_log_mu_quantiles={q: float(np.quantile(lmu, q)) for q in (.5, .9, .99, .999)})

    dirs = {
        'solver_hard_original64': np.load(T / 'CROSS_CASE_02' / case / 'WITNESS_WHITE_064.npy'),
        'solver_hard_split64': np.load(T / 'CROSS_CASE_03_SPLIT0p1' / case / 'WITNESS_WHITE_064.npy'),
        'solver_pair64_witness': np.load(T / 'CROSS_CASE_04_PAIR0p9' / case / 'WITNESS_WHITE_064.npy'),
        'geometry_elements_res8_witness': np.load(T / 'ENCODE_GPU_01' / (case + '_polyref8') / 'WITNESS_WHITE.npy'),
        'arm_extreme_compliance_max': ext[:, 1:2], 'arm_extreme_compliance_min': ext[:, 0:1],
    }
    rng = np.random.default_rng(20260923)
    for k in range(32):
        dirs[f'random_{k:02d}'] = rng.standard_normal((d, 1))

    def rank(val):
        return float(np.searchsorted(lmu, abs(np.log(val))) / len(lmu))

    def coords(z):  # boundary coordinates carrying the direction: B^T R^-1 z, top 24 by magnitude
        qv = hq.left(np.r_[np.zeros(6), linalg.solve_triangular(rq, z.ravel(), lower=False)], transpose=False)
        return set(np.argsort(-np.abs(qv))[:24].tolist()), qv

    hard_set, hard_q = coords(dirs['solver_hard_original64'])
    rows = {}
    for name, z in dirs.items():
        z = np.asarray(z, dtype=np.float64).reshape(d, -1)[:, :1]
        nz = float(z[:, 0] @ z[:, 0])
        rc = float(z[:, 0] @ (C @ z[:, 0]) / nz)
        rs = float(z[:, 0] @ linalg.cho_solve(L, z[:, 0], check_finite=False) / nz)
        s, qv = coords(z)
        rows[name] = dict(compliance_ratio=rc, stiffness_ratio=rs, abs_log_stiffness_rank_in_arm_spectrum=rank(rs),
                          cosine_with_solver_hard=float(abs(z[:, 0] @ dirs['solver_hard_original64'].ravel()) /
                                                        np.sqrt(nz * float(dirs['solver_hard_original64'].ravel() @ dirs['solver_hard_original64'].ravel()))),
                          top24_boundary_coordinate_overlap_with_solver_hard=len(s & hard_set),
                          top24_energy_fraction=float(np.sort(qv ** 2)[-24:].sum() / (qv ** 2).sum()))
    rec['directions'] = rows
    rand = np.array([rows[f'random_{k:02d}']['stiffness_ratio'] for k in range(32)])
    rec['random_stiffness_ratio'] = dict(min=float(rand.min()), median=float(np.median(rand)), max=float(rand.max()))
rec['seconds'] = time.perf_counter() - t0
out.write_text(json.dumps(rec, indent=2))
print(json.dumps({k: rec[k] for k in ('frame_check_relative', 'arm_spectrum', 'random_stiffness_ratio')}))
for n, r in rows.items():
    if not n.startswith('random_'):
        print('%-34s r_c %.4g  r_s %.4g  rank %.4f  cos %.3f  top24 overlap %d  top24 frac %.3f' % (
            n, r['compliance_ratio'], r['stiffness_ratio'], r['abs_log_stiffness_rank_in_arm_spectrum'],
            r['cosine_with_solver_hard'], r['top24_boundary_coordinate_overlap_with_solver_hard'], r['top24_energy_fraction']))
