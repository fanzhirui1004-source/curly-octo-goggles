"""Local CPU checks of the prep_geo2 audit fixes (docs/AUDIT_V2_20260925_CN.md DATA-1..9); no GPU, no cuDSS, < 2 min.

 a  DATA-1  seeds: sha256 case seeds differ across the audit's colliding names (legacy: all 1936036181) and are
            reproducible; per-class generators are independent of which other classes drew first
 b  DATA-2  traction equilibration on the fixture geometry (fresh_train_0010_cover01_r1, cut cell, real Traction
            quadrature): Q^T f and the direct resultant force / moment of f = A t vanish, the correction is a rigid-type
            traction h^2 (a + b x (x - x_c)); force_c / face_c (Neumann solve replaced by the identity: no factor on this
            CPU) report rigid_frac_max ~1e-15 vs rigid_frac_raw_median ~0.3
 c  DATA-5/8/6/3  one() end to end with --out on a synthetic full-cube cell (n = 2, graph-Laplacian stiffness, scipy
            factors as teacher.SPDSolver, parent = the same cube): the source tree stays byte-identical (sha256 + mtime
            + inode link counts only grow), old files are hard links, stale pilot v2 banks are not inherited, support64
            is written, F-old goes to the out tree, DONE2 carries seeds / rigid_frac / cut_qp / dropped / bc_counts /
            residuals; a rerun is a no-op (no Cell built)
 d  DATA-4/9  per-class failure isolation (face_c and glued raise): force_c and support_k saved, DONE2 records the
            failures, status partial; main() exit status 1 only when nothing was produced; in-place --fix-support writes
            the new bank before renaming the old one (q, _sens, _F -> *.nan_bak.npy) and leaves it untouched on failure
 e  DATA-3  UpperSym (symmetric matvec from an upper CSR) == scipy on the fixture K; iterative refinement of an fp32
            Cholesky solve on an SPD system (kappa ~1e5) reaches the fp64 level; glued with a real fp32 factor refines
            and is accepted, glued with an inaccurate solver is rejected (offset skipped); per-chunk BC mix recorded
Not covered here (GPU): cuDSS factors, the memory fallbacks, real FULL parents (family_full / TE.Cell on packets).
Usage: python3 t_prep2.py
"""
import os
import sys
import types
os.environ['OPL_DEV'] = 'cpu'
os.environ['BANK_SPLITS'] = '8,4,4'                                                  # before prep_data is imported
import json, time, hashlib, tempfile, contextlib, io
from pathlib import Path
import numpy as np
import torch
import scipy.sparse as sp
from scipy.sparse.linalg import splu

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
try:
    import element_polyref  # noqa: F401
except ImportError:                                                                   # only CUBE is used by prep_geo2
    from itertools import product
    element_polyref = types.ModuleType('element_polyref')
    element_polyref.CUBE = np.array(list(product((0, 1), repeat=3)))
    sys.modules['element_polyref'] = element_polyref
import fixture as FX
import teacher as TE

torch.set_num_threads(int(os.environ.get('T3_THREADS', '2')))
dt = torch.float64
FAIL, T0 = [], time.perf_counter()


def check(name, ok, **info):
    print(json.dumps(dict(test=name, ok=bool(ok), t=round(time.perf_counter() - T0, 1),
                          **{k: (float(f'{v:.4g}') if isinstance(v, float) else v) for k, v in info.items()}), default=str), flush=True)
    if not ok:
        FAIL.append(name)


class SciSPD:
    """teacher.SPDSolver interface on scipy's sparse LU (fp64); fdt = fp32 rounds the solution."""

    def __init__(self, crow, col, vals, n, w=16, threads=16, fdt=None):
        self.fdt = vals.dtype if fdt is None else fdt
        U = sp.csr_matrix((vals.double().cpu().numpy(), col.long().cpu().numpy(), crow.long().cpu().numpy()), shape=(n, n))
        self.lu = splu((U + U.T - sp.diags(U.diagonal())).tocsc())

    def solve(self, r):
        return torch.as_tensor(self.lu.solve(r.double().cpu().numpy())).to(self.fdt).to(r.dtype)

    def free(self):
        self.lu = None


class Dense32:
    """A genuinely fp32 factor (dense Cholesky in fp32) of an SPD upper CSR: fp32 rounding in the factor and the solve."""

    def __init__(self, crow, col, vals, n, **kw):
        U = sp.csr_matrix((vals.double().numpy(), col.long().numpy(), crow.long().numpy()), shape=(n, n))
        A = torch.as_tensor((U + U.T - sp.diags(U.diagonal())).toarray(), dtype=torch.float32)
        self.L = torch.linalg.cholesky(A)

    def solve(self, r):
        return torch.cholesky_solve(r.to(torch.float32), self.L).to(r.dtype)

    def free(self):
        self.L = None


class Sloppy(SciSPD):
    """An inaccurate solver (0.9 x the exact solution): refinement cannot reach the tolerance in 3 steps."""

    def solve(self, r):
        return 0.9 * super().solve(r)


TE.SPDSolver = SciSPD                                                                 # before prep_geo2 captures it
TE.sync = lambda: None
import prep_geo as PG
import prep_geo2 as P2

RealCell = TE.Cell


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def snapshot(d):
    return {f.name: (sha(f), f.stat().st_mtime_ns, f.stat().st_ino) for f in sorted(Path(d).iterdir()) if f.is_file()}


# ------------------------------------------------------------------------------------------------ synthetic cell
def synth_cell(case, n=2, tau=3.5, seed=0):
    """A teacher.Cell on the full cube (every element active, material everywhere), stiffness = element graph Laplacian x I3
    (PSD, translations only in the null space: SPD once ports / pins / springs / a clamped face are fixed)."""
    C = object.__new__(RealCell)
    M = 2 * n + 1
    C.case, C.n, C.log = case, n, (lambda s_: None)
    C.nodes = np.arange(M ** 3)
    g = np.stack(np.unravel_index(C.nodes, (M,) * 3), 1)
    C.cells = np.array(list(np.ndindex(n, n, n)))
    C.taus = C.taus0 = [tau] * 8
    C.normal = C.offset = None
    C.is_box = ((g == 0) | (g == 2 * n)).any(1)
    C.is_cut = np.zeros(len(C.nodes), bool)
    en = np.stack([(((2 * c[0] + o[0]) * M + 2 * c[1] + o[1]) * M + 2 * c[2] + o[2]) for c in C.cells for o in np.ndindex(3, 3, 3)])
    en = en.reshape(len(C.cells), 27)
    dofs = (3 * en[:, :, None] + np.arange(3)[None, None]).reshape(len(C.cells), 81)
    C.dofs = torch.as_tensor(dofs)
    Ke = np.kron(27 * np.eye(27) - np.ones((27, 27)), np.eye(3))
    nb = 3 * len(C.nodes); C.nb = nb
    rr = np.repeat(dofs, 81, 1).reshape(-1); cc = np.tile(dofs, (1, 81)).reshape(-1)
    K = sp.coo_matrix((np.tile(Ke.reshape(-1), len(C.cells)), (rr, cc)), shape=(nb, nb)).tocsr()
    U = sp.triu(K).tocsr(); U.sort_indices()
    C.crow = torch.as_tensor(U.indptr, dtype=torch.long); C.cu = torch.as_tensor(U.indices, dtype=torch.int32)
    C.vals = torch.as_tensor(U.data, dtype=dt)
    FX._rebuild_K(C)
    C.diag = torch.nonzero(C.ru == C.cu).squeeze(1); C.dK = C.vals[C.diag]
    onport = C.is_box | C.is_cut
    C.port_node_ids = C.nodes[onport]; C.port_is_box = C.is_box[onport]; C.port_is_cut = C.is_cut[onport]
    pm = torch.as_tensor(np.repeat(onport, 3))
    C.P = torch.nonzero(pm).squeeze(1); C.I = torch.nonzero(~pm).squeeze(1)
    C.np_, C.ni = len(C.P), len(C.I)
    C.Q = TE.rigid_basis(C.port_node_ids, n); C.Qall = TE.rigid_basis(C.nodes, n)
    C.sol_I = C.sol_N = None
    gg = torch.Generator().manual_seed(seed)
    Tm = torch.randn((125, 81, 81), generator=gg, dtype=dt) * 1e-2
    C.Tm = Tm + Tm.transpose(1, 2)
    C.dM = torch.randn((8, len(C.cells), 125), generator=gg, dtype=dt)
    C.assemble = lambda taus=None: C                                                    # no moments on this CPU
    C.dmoments = lambda rel=1e-5: C.dM
    return C


N_CELLS = [0]


def fake_cell(case, body, log=None, **kw):
    N_CELLS[0] += 1
    return synth_cell(case)


def write_source(d, np_, nan_support=True, seed=1):
    """A source data/<case> dir like prep_geo + a pilot prep_geo2 run left it (old v2 record, stale force_c bank)."""
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for s_, m in P2.SPLITS:
        np.save(d / f'{s_}_force.npy', rng.standard_normal((m, np_)).astype(np.float32))
        np.save(d / f'{s_}_force_sens.npy', rng.standard_normal((m, 8)))
        a = rng.standard_normal((m, np_)).astype(np.float32)
        if nan_support:
            a[0, :5] = np.nan
        np.save(d / f'{s_}_support.npy', a); np.save(d / f'{s_}_support_sens.npy', rng.standard_normal((m, 8)))
        np.save(d / f'{s_}_support_F.npy', rng.standard_normal((m, np_)).astype(np.float32))
        for suf in P2.SUFFIXES:                                                         # stale pilot bank of a v2 class
            np.save(d / f'{s_}_force_c{suf}.npy', rng.standard_normal((m, np_ if suf != '_sens' else 8)).astype(np.float32))
    np.savez(d / 'NETDATA.npz', x=rng.standard_normal(10))
    (d / 'DONE.json').write_text(json.dumps(dict(case=d.name)))
    (d / 'DONE2.json').write_text(json.dumps(dict(case=d.name, banks={'force_c': dict(count=16)}, seconds={})))


def quiet(fn, *a, **k):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = fn(*a, **k)
    return r, [json.loads(l_) for l_ in buf.getvalue().splitlines() if l_.startswith('{')]


# ------------------------------------------------------------------------------------------------ a seeds
names = ['fresh_train_0020_cover01_r2', 'fresh_train_0020_full', 'fresh_development_0002_full', 'fresh_train_0031_cover01_r1_rotgeneral']
leg = {P2.case_seed(c, legacy=True) for c in names}
new = [P2.case_seed(c) for c in names]
gA, gB = P2.ClassGens(names[0]), P2.ClassGens(names[0])
gA('force_c'); torch.rand(100, generator=gA('force_c'))                                # another class draws first in A only
xa, xb = torch.rand(5, generator=gA('glued')), torch.rand(5, generator=gB('glued'))
xf = torch.rand(5, generator=P2.ClassGens(names[0])('force_c'))
gl = P2.ClassGens(names[0], legacy=True)
check('a_seeds', leg == {1936036181} and len(set(new)) == 4 and new == [P2.case_seed(c) for c in names]
      and torch.equal(xa, xb) and not torch.equal(xa, xf) and gl('force_c') is gl('glued')
      and len({P2.class_seed(names[0], c) for c in P2.NEW}) == 4 and max(P2.class_seed(c, k_) for c in names for k_ in P2.NEW) < 2 ** 32,
      legacy=sorted(leg), new=new)

# ------------------------------------------------------------------------------------------------ b traction equilibration (fixture)
geo = FX.small(); FC = geo.C
faces = P2.box_tractions(FC)
cut = P2.Traction(FC, cut=True)
gen = torch.Generator().manual_seed(3)
Xp = torch.as_tensor(np.stack(np.unravel_index(FC.port_node_ids, (2 * FC.n + 1,) * 3), 1) / (2 * FC.n), dtype=dt)


def resultants(f):
    """Direct resultant force and moment about the port centroid of nodal forces f (np, k); relative to sum |f_i| |x_i - c|."""
    fn = f.reshape(-1, 3, f.shape[1]); xc = Xp - Xp.mean(0)
    Fr = fn.sum(0); Mr = torch.cross(xc[:, :, None].expand_as(fn), fn, dim=1).sum(0)
    scale = fn.norm(dim=1).sum(0)
    return float((Fr.norm(dim=0) / scale).max()), float((Mr.norm(dim=0) / scale).max())


keys = list(faces)
Gs = [faces[k].rigid_map(FC.Q) for k in keys]
Gc = cut.rigid_map(FC.Q)
k = 6
ts = [faces[k_].random(k, gen) for k_ in keys]
tcut = cut.random(k, gen)
on = torch.tensor([1., 0, 1, 0, 0, 1], dtype=dt)
M0 = sum(P2._gram(G) for G in Gs)
Ms = M0[None] + on[:, None, None] * P2._gram(Gc)[None]
parts = [(G, t, None) for G, t in zip(Gs, ts)] + [(Gc, tcut, on)]
te = P2.equilibrate(parts, Ms)
f_raw = sum(faces[k_].forces(t) for k_, t in zip(keys, ts)) + cut.forces(tcut) * on[None]
f = sum(faces[k_].forces(t) for k_, t in zip(keys, te[:-1])) + cut.forces(te[-1]) * on[None]
fr_raw, fr = P2._frac(FC.Q, f_raw), P2._frac(FC.Q, f)
Fd, Md = resultants(f)
Fd0, Md0 = resultants(f_raw)
# the correction on one face is h^2 x a rigid-type traction a + b x (x - x_c)
T0f, dtc = faces[keys[0]], (ts[0] - te[0])[:, :, 0]
Xq = T0f.X - T0f.X.mean(0)
R = torch.zeros((T0f.m, 3, 6), dtype=dt)
for a_ in range(3):
    R[:, a_, a_] = 1
    e = torch.zeros(3, dtype=dt); e[a_] = 1
    R[:, :, 3 + a_] = torch.cross(e[None].expand_as(Xq), Xq, dim=1)
coef = torch.linalg.lstsq(R.reshape(-1, 6), dtc.reshape(-1, 1)).solution
fit = float((R.reshape(-1, 6) @ coef - dtc.reshape(-1, 1)).norm() / dtc.norm())
check('b_equilibrate_fixture', float(fr.max()) < 1e-10 and float(fr_raw.median()) > 1e-2 and Fd < 1e-10 and Md < 1e-10 and fit < 1e-8,
      faces=len(keys), cut_qp=cut.m, cut_dropped=cut.dropped, rigid_frac_raw_median=float(fr_raw.median()),
      rigid_frac_max=float(fr.max()), resultant_force_rel=Fd, resultant_moment_rel=Md, raw_force_rel=Fd0, raw_moment_rel=Md0,
      correction_rigid_fit_rel=fit)
FC.neumann = lambda F: F                                                              # no factor here: the identity stands in
qf, inf_f = P2.force_c(FC, faces, 16, torch.Generator().manual_seed(4), cut_frac=0.5, chunk=8)
proj = float(((qf - (qf - FC.Q @ (FC.Q.T @ qf))).norm(dim=0) / qf.norm(dim=0)).max())  # what C.neumann would still remove
qa, inf_a = P2.force_c(FC, faces, 16, torch.Generator().manual_seed(4), cut_frac=0.5, chunk=8)
check('b_force_c', inf_f['rigid_frac_max'] < 1e-10 and inf_f['rigid_frac_raw_median'] > 1e-2 and proj < 1e-10
      and inf_f['cut_qp'] == cut.m and 'cut' in inf_f['traction_dropped'] and torch.equal(qf, qa) and inf_f['cut_loaded'] > 0,
      **{k_: v_ for k_, v_ in inf_f.items() if k_ != 'traction_dropped'}, dropped=inf_f['traction_dropped'], neumann_projection_rel=proj)
qfc, inf_fc = P2.face_c(FC, faces, 12, torch.Generator().manual_seed(5), chunk=8)
check('b_face_c', inf_fc['rigid_frac_max'] < 1e-10 and inf_fc['rigid_frac_raw_median'] > 1e-2 and qfc.shape[1] == 12,
      **inf_fc)
del FC.neumann

# ------------------------------------------------------------------------------------------------ e refinement / matvec
x = torch.randn((geo.C.nb, 3), dtype=dt, generator=torch.Generator().manual_seed(6))
Ks = FX.scipy_K(geo.C)
mv = P2.UpperSym(geo.C.crow, geo.C.cu, geo.C.vals)
y_ref = torch.as_tensor(Ks @ x.numpy())
mv_err = max(float((mv(x) - y_ref).norm() / y_ref.norm()),
             float((P2.UpperSym(geo.C.crow, geo.C.cu, geo.C.vals, host=True)(x) - y_ref).norm() / y_ref.norm()))
n_ = 400                                                                              # 1D Laplacian + shift: kappa ~ 1e5
main_d = np.full(n_, 2.0 + 1e-4); off = np.full(n_ - 1, -1.0)
Uu = sp.diags([main_d, off], [0, 1], format='csr'); Uu.sort_indices()
crow, col, vals = torch.as_tensor(Uu.indptr), torch.as_tensor(Uu.indices, dtype=torch.int32), torch.as_tensor(Uu.data)
S32 = Dense32(crow, col, vals, n_)
b = torch.randn((n_, 4), dtype=dt, generator=torch.Generator().manual_seed(7))
A_ = P2.UpperSym(crow, col, vals)
y0 = S32.solve(b)
rel0 = float(((b - A_(y0)).norm(0) / b.norm(0)).max())
y, rel, it, _ = P2.refine(A_, S32.solve, b, y0)
yx = torch.linalg.solve(torch.as_tensor((Uu + Uu.T - sp.diags(Uu.diagonal())).toarray()), b)
check('e_refine', mv_err < 1e-13 and rel0 > 1e-8 and float(rel.max()) < 1e-10 and 1 <= it <= P2.REFINE_STEPS
      and float((y - yx).norm() / yx.norm()) < 1e-8,
      upper_matvec_vs_scipy=mv_err, resid_fp32=rel0, resid_refined=float(rel.max()), steps=it,
      err_fp32=float((y0 - yx).norm() / yx.norm()), err_refined=float((y - yx).norm() / yx.norm()))

# ------------------------------------------------------------------------------------------------ e glued on the synthetic cube
TE.Cell = fake_cell
P2.family_full = lambda case, packets: 'synth_parent'
Ct = synth_cell('fresh_train_9999_cover01_r1')
_spd_glued0 = P2._spd_glued
P2._spd_glued = lambda crow, col, vals, n: (Dense32(crow, col, vals, n), 'fp32')
(qg, ig), ev = quiet(P2.glued, Ct, 'body', 64, torch.Generator().manual_seed(8), chunk=4)
grp = [g_ for o in ig['offsets'] for g_ in o['groups']]
bc = ig['bc_counts']
check('e_glued_fp32_refined', qg is not None and qg.shape[1] == 64 and ig['resid_max'] < P2.RESID_TOL
      and ig['refine_steps_max'] >= 1 and sum(v['clamped'] + v['springs'] for v in bc.values()) == 64
      and sum(v['clamped'] for v in bc.values()) > 0 and sum(v['springs'] for v in bc.values()) > 0
      and sorted(bc) == ['+x', '+y', '+z', '-z'],
      bc_counts=bc, resid_max=ig['resid_max'], precisions=ig['precisions'], refine_steps_max=ig['refine_steps_max'],
      groups=len(grp), resid_unrefined_max=max(g_['resid_unrefined_max'] for g_ in grp))
P2._spd_glued = lambda crow, col, vals, n: (Sloppy(crow, col, vals, n), 'fp32')
(qs, is_), ev = quiet(P2.glued, Ct, 'body', 16, torch.Generator().manual_seed(8), chunk=4)
check('e_glued_rejected', qs is None and 'skipped' in is_ and all('residual' in o.get('skipped', '') for o in is_['offsets'])
      and sum(e_['event'] == 'GLUED_OFFSET_REJECTED' for e_ in ev) == 4,
      skipped=is_.get('skipped'), offset0=is_['offsets'][0].get('skipped'))
P2._spd_glued = _spd_glued0
(qg64, ig64), _ = quiet(P2.glued, Ct, 'body', 16, torch.Generator().manual_seed(8), chunk=4)
check('e_glued_fp64', qg64 is not None and ig64['resid_max'] < 1e-10 and ig64['refine_steps_max'] == 0, resid_max=ig64['resid_max'])

# ------------------------------------------------------------------------------------------------ c --out end to end
TMP = Path(tempfile.mkdtemp(prefix='t_prep2_', dir=os.environ.get('T3_TMP', str(HERE.parent.parent))))
case = 'fresh_train_9999_cover01_r1'
SRC, OUT = TMP / 'data', TMP / 'data_v2'
write_source(SRC / case, Ct.np_)
before = snapshot(SRC / case)
r1, ev1 = quiet(P2.one, 'body', SRC, case, list(P2.NEW), F_old=True, fix_support=True, out=OUT, log=print)
after = snapshot(SRC / case)
d = OUT / case
rec = json.loads((d / 'DONE2.json').read_text())
B = rec['banks']
linked = all(os.stat(SRC / case / n_).st_ino == os.stat(d / n_).st_ino for n_ in ('NETDATA.npz', 'DONE.json', 'train_force.npy', 'train_support.npy'))
new_files_ok = all((d / f'{s_}_{c}{suf}.npy').exists() for s_, _ in P2.SPLITS for c in list(P2.NEW) + ['support64'] for suf in P2.SUFFIXES)
not_inherited = os.stat(SRC / case / 'train_force_c.npy').st_ino != os.stat(d / 'train_force_c.npy').st_ino
fold = (d / 'train_force_F.npy').exists() and not (SRC / case / 'train_force_F.npy').exists() \
    and os.stat(d / 'train_support_F.npy').st_ino != os.stat(SRC / case / 'train_support_F.npy').st_ino
s64 = np.load(d / 'train_support64.npy')
check('c_out_source_untouched', before == after and sorted(before) == sorted(after), files=len(before))
check('c_out_tree', r1['status'] == 'done' and linked and new_files_ok and not_inherited and fold and np.isfinite(s64).all()
      and not (d / 'train_support.nan_bak.npy').exists() and all(B[c]['version'] == P2.VERSION for c in list(P2.NEW) + ['support64'])
      and B['force_c']['rigid_frac_max'] < 1e-10 and B['face_c']['rigid_frac_max'] < 1e-10 and 'cut_qp' in B['force_c']
      and 'traction_dropped' in B['force_c'] and 'bc_counts' in B['glued'] and B['glued']['resid_max'] < 1e-10
      and rec['seed']['scheme'] == 'sha256' and B['glued']['seed'] == P2.class_seed(case, 'glued') and rec['links']['linked'] > 0,
      produced=r1['produced'], links=rec['links'], force_c_rigid_frac=[B['force_c']['rigid_frac_raw_median'], B['force_c']['rigid_frac_max']],
      glued_bc=B['glued']['bc_counts'], glued_resid=B['glued']['resid_max'])
N_CELLS[0] = 0
r2, ev2 = quiet(P2.one, 'body', SRC, case, list(P2.NEW), fix_support=True, out=OUT, log=print)
check('c_rerun_idempotent', r2['status'] == 'skipped' and N_CELLS[0] == 0 and snapshot(SRC / case) == before,
      status=r2['status'], cells_built=N_CELLS[0])
N_CELLS[0] = 0
before_out = snapshot(d)
r3, _ = quiet(P2.one, 'body', SRC, case, ['support_k'], out=OUT, log=print, force=True)
after_out = snapshot(d)
check('c_force_one_class', r3['produced'] == ['support_k'] and N_CELLS[0] == 1
      and after_out['train_support_k.npy'][0] == before_out['train_support_k.npy'][0]              # same seed: same bank
      and after_out['train_force_c.npy'] == before_out['train_force_c.npy'] and snapshot(SRC / case) == before,
      produced=r3['produced'])

# ------------------------------------------------------------------------------------------------ d failure isolation
face_c0, glued0 = P2.face_c, P2.glued


def boom(msg):
    def f(*a, **k):
        raise RuntimeError(msg)
    return f


P2.face_c, P2.glued = boom('INJECTED_FACE_C'), boom('INJECTED_GLUED')
OUT2 = TMP / 'data_v2b'
r4, ev4 = quiet(P2.one, 'body', SRC, case, list(P2.NEW), out=OUT2, log=print)
B4 = json.loads((OUT2 / case / 'DONE2.json').read_text())['banks']
check('d_failure_isolation', r4['status'] == 'partial' and sorted(r4['produced']) == ['force_c', 'support_k']
      and 'INJECTED_FACE_C' in B4['face_c']['failed'] and 'INJECTED_GLUED' in B4['glued']['failed']
      and (OUT2 / case / 'train_force_c.npy').exists() and not (OUT2 / case / 'train_face_c.npy').exists()
      and any(e_['event'] == 'PARTIAL2' for e_ in ev4) and not any(e_['event'] == 'FAILED2' for e_ in ev4),
      produced=r4['produced'], failed=r4['failed'])
P2.face_c, P2.glued = face_c0, glued0
r5, ev5 = quiet(P2.one, 'body', SRC, case, list(P2.NEW), out=OUT2, log=print)      # rerun redoes only the failed classes
check('d_rerun_failed_only', sorted(r5['produced']) == ['face_c', 'glued'], produced=r5['produced'])
force_c0 = P2.force_c
P2.force_c = boom('INJECTED_FORCE_C')
rc_fail, ev6 = quiet(P2.main, ['body', str(SRC), case, '--out', str(TMP / 'data_v2c'), '--classes', 'force_c'])
rc_part, ev7 = quiet(P2.main, ['body', str(SRC), case, '--out', str(TMP / 'data_v2d'), '--classes', 'force_c,support_k'])
P2.force_c = force_c0
check('d_main_exit', rc_fail == 1 and rc_part == 0 and any(e_['event'] == 'FAILED2' for e_ in ev6)
      and not any(e_['event'] == 'FAILED2' for e_ in ev7), rc_nothing=rc_fail, rc_partial=rc_part)
# in-place --fix-support: new files staged first, old q / _sens / _F -> *.nan_bak.npy only after; a failure touches nothing
IP = TMP / 'inplace'
write_source(IP / case, Ct.np_)
h0 = {n_: sha(IP / case / n_) for n_ in ('train_support.npy', 'train_support_sens.npy', 'train_support_F.npy')}
norm0 = PG.normalize
PG.normalize = boom('INJECTED_NORMALIZE')
try:
    quiet(P2.one, 'body', IP, case, [], fix_support=True, log=print)
    r8 = 'no raise'
except RuntimeError as e:                                                             # nothing produced: one() raises (FAILED2)
    r8 = repr(e)
PG.normalize = norm0
untouched = all(sha(IP / case / n_) == h for n_, h in h0.items()) and not list((IP / case).glob('*nan_bak*')) \
    and not list((IP / case).glob('.*')) and 'NOTHING_PRODUCED' in r8
r9, _ = quiet(P2.one, 'body', IP, case, [], fix_support=True, log=print)
bak_ok = all(sha(IP / case / n_.replace('.npy', '.nan_bak.npy')) == h for n_, h in h0.items())
fixed = np.isfinite(np.load(IP / case / 'train_support.npy')).all() and (IP / case / 'train_support_F.npy').exists()
check('d_fix_support_inplace', untouched and bak_ok and fixed and r9['produced'] == ['support'], after_failure_untouched=untouched,
      backups_are_originals=bak_ok, fixed_finite=bool(fixed))
# stale pilot bank in place + a failing class: retired to *.stale_bak.npy, DONE2 says failed
P2.face_c = boom('INJECTED_FACE_C')
for s_, m in P2.SPLITS:
    for suf in P2.SUFFIXES:
        np.save(IP / case / f'{s_}_face_c{suf}.npy', np.zeros((m, 3), np.float32))
try:
    quiet(P2.one, 'body', IP, case, ['face_c'], log=print)
except RuntimeError:
    pass
P2.face_c = face_c0
check('d_stale_retired', not (IP / case / 'train_face_c.npy').exists() and (IP / case / 'train_face_c.stale_bak.npy').exists()
      and 'failed' in json.loads((IP / case / 'DONE2.json').read_text())['banks']['face_c'])

import shutil
shutil.rmtree(TMP, ignore_errors=True)
print(json.dumps(dict(event='T_PREP2_DONE', failed=FAIL, seconds=round(time.perf_counter() - T0, 1))), flush=True)
sys.exit(1 if FAIL else 0)
