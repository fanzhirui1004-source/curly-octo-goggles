"""Section 6.11: thickness optimisation of a lattice with NICE, standard MMA and field-based sensitivities. New script.

Design variables: the corner thickness parameters at the lattice vertices (shared by the cells meeting there), except the
vertices of the load face, which are held fixed so that the consistent face load does not depend on the design (loads
applied through a non-design layer, review item I-50).  Objective: compliance C_hat = f^T U_bar of one consistent load
column (--load-dir) of the NICE lattice solve.  Gradient: the field-based sensitivities s_tilde (Eq. 13) of every cell,
summed over the cells at each vertex (pre-registered rule A of E3).  Constraints (scaled, <= 0):
  volume      V(tau) / V* - 1, V = sum over cells of the zeroth element moments (material volume of the discrete model),
              dV/dtau_c from the central-difference moment derivatives; V* = --vfrac x V(initial design)
  span        (tau_a - tau_b) / --span - 1 for every ordered pair of corners of every cell (training contract <= 0.47)
  gradient    |grad tau|^2 / --grad^2 - 1 at every corner of every cell (edge differences; trilinear field: the maximum
              over the cell is attained at a corner; training contract <= 0.47)
  bounds      --tmin <= tau <= --tmax (training range [0.1752, 0.6993])
Every iteration regenerates the geometry of every cell: new packets (FRESH_CONTEXT of the base case with the new corners)
and new bodies by fast_prep4 in the frozen CPU environment (--workers in parallel), then the NICE lattice solve as in
r1_lat.py (learned branch; --park / --resident / --deploy as there).  One MMA update per iteration (mma.py; no line
search).  Stopping: max |d tau| < --xtol, or relative objective change < --ftol for three consecutive iterations, or
--maxit.  Per iteration one JSON line in <root>/history.jsonl: compliance, volume, constraint maxima, PCG iterations,
recomputed residual and signed residual work, per-cell switch fingerprints (active elements, ghost faces, ports, weak nodes,
fringe hyperedges, coarse shift, Chebyshev endpoint), phase times and memory; state in <root>/state.npz (resume by rerun).
Bodies of every iteration are kept (<root>/body) for the exact checks (opt_check.py).
Usage: opt_design.py <root> <layout.json> --model A3=<ckpt> [--clamp y,min] [--load y,max] [--load-dir y] [--vfrac 0.8]
       [--move 0.05] [--maxit 60] [--tmin 0.18] [--tmax 0.69] [--span 0.45] [--grad 0.45] [--tol 1e-6]
       [--workers 16] [--park] [--resident 2] [--deploy] [--max-cols 16] [--stop-after K]
--exact: the twin run, same MMA, constraints and regeneration, with the exact model: dense exact condensed matrices per cell
  (make_T_gpu.py, one cell per process, cached as <body>/<case>_portview/T64.npy), the assembled solve by PCG to
  --exact-tol, and exact sensitivities -u^T K_,c u from the exact field (host interior factor, as the reference of r1_lat.py).
--clamp cut: (opt-in) clamp every cut-band DOF of every cell instead of a lattice face (the lattice bonded to a wall
  along the cut plane; lat_multi.MultiLattice clamp=('cut', '')); applies to the NICE, --exact and --check analyses alike.
--table5 (with --fast): the four implementation options of the timed route of Table 5 (lat_scale.py --coarse-tpl
  --tet-triton --sparse-coarse --fastidx): coarse Galerkin matrix by element/face templates, fused per-tetrahedron moment
  kernels, sparse coarse space, element gathers.
--check k1,k2,...: verify iterations of an existing run (<root>/history.jsonl) with the exact model; writes
  <root>/check_<k>.json: exact compliance and the surrogate error, the vertex-gradient error, cosine, per-variable
  percentiles and sign agreement of s_tilde on the free vertices, and a KKT residual with the exact gradient.
"""
import json, time, os, sys, argparse, contextlib, subprocess, shutil, resource
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('root'); ap.add_argument('layout'); ap.add_argument('--model', required=True)
ap.add_argument('--clamp', default='y,min'); ap.add_argument('--load', default='y,max'); ap.add_argument('--load-dir', default='y')
ap.add_argument('--vfrac', type=float, default=0.8); ap.add_argument('--move', type=float, default=0.05)
ap.add_argument('--maxit', type=int, default=60); ap.add_argument('--stop-after', type=int, default=0,
                                                                  help='end this run after K iterations (pilot / scale runs)')
ap.add_argument('--tmin', type=float, default=0.18); ap.add_argument('--tmax', type=float, default=0.69)
ap.add_argument('--span', type=float, default=0.45); ap.add_argument('--grad', type=float, default=0.45)
ap.add_argument('--xtol', type=float, default=1e-3); ap.add_argument('--ftol', type=float, default=1e-4)
ap.add_argument('--prec', default='bnn:kpp:q1r'); ap.add_argument('--tol', type=float, default=1e-6)
ap.add_argument('--maxit-pcg', type=int, default=3000); ap.add_argument('--max-cols', type=int, default=16)
ap.add_argument('--workers', type=int, default=16)
ap.add_argument('--park', action='store_true'); ap.add_argument('--resident', type=int, default=0)
ap.add_argument('--deploy', action='store_true'); ap.add_argument('--lean', action='store_true')
ap.add_argument('--frozen', default='/root/autodl-tmp/CUTFEM_DEPENDENCIES_20260924/run_frozen_python.sh')
ap.add_argument('--frozen-cwd', default='/root/autodl-tmp/CUTFEM_DEPENDENCIES_20260924/root/autodl-tmp/CLAUDE_TAKEOVER_20260923/COVER_G/src')
ap.add_argument('--templates', default='/root/autodl-tmp/OPL/S4/body/GP_TEMPLATES_n32.npz')
ap.add_argument('--exact', action='store_true', help='optimise with exact condensation (twin run; no network)')
ap.add_argument('--fast', action='store_true', help='deployment route of Table 5 (lat_scale.py): deploy cells, fused network, '
                'single-precision correction, streamed operators; sensitivities by central moment differences')
ap.add_argument('--resident-gb', type=float, default=12.0, help='--fast: keep cells on the device while allocated memory < this')
ap.add_argument('--sens', default='fd', choices=['fd', 'ad'],
                help="--fast: 'fd' central-difference moment derivatives (Cell.dmoments); 'ad' reverse mode through the moment "
                     "integrator (moments_ad.cell_sens, as the timed route of Table 5), one pass for compliance and volume")
ap.add_argument('--ad-batch', type=int, default=512, help="--sens ad: element rows per reverse pass")
ap.add_argument('--body-retry', type=int, default=0,
                help='N > 0: if body generation fails for some cells (e.g. an exact local-support certificate left unresolved '
                     'where the cut plane and the sheet surface nearly touch), the free design vertices of those cells are '
                     'scaled by (1 + eps), eps = 1e-4, -1e-4, 1e-3, -1e-3, 3e-3, -3e-3 (first N), all cells sharing them are '
                     'regenerated, and the perturbed design is analysed and continued from (recorded as body_perturb)')
ap.add_argument('--t-cpu-fallback', action='store_true',
                help='--exact: if the GPU dense condensation of a cell fails, build it on the host (make_T_cpu.py, PARDISO)')
ap.add_argument('--vstar', type=float, default=0.0,
                help='> 0: absolute volume bound V* (material volume of the discrete model) instead of --vfrac x V(initial design), '
                     'e.g. to continue from another design under the bound of an earlier run')
ap.add_argument('--warm', action='store_true',
                help='--fast: start PCG from the previous design iteration\'s solution, matched DOF by DOF on (absolute grid '
                     'position, component, cut-port flag) and scaled by the energy-optimal factor; unmatched DOFs start at 0 '
                     '(the stopping rule, relative to |f|, is unchanged)')
ap.add_argument('--table5', action='store_true', help='--fast: OPL_COARSE_ELEM, OPL_TET_TRITON, OPL_COARSE_SPARSE, fastidx.ON '
                '(the implementation options of the timed route of Table 5, as lat_scale.py)')
ap.add_argument('--check', default='', help='comma list of iterations of an existing run to verify with the exact model')
ap.add_argument('--exact-tol', type=float, default=1e-10)
A = ap.parse_args()
if A.deploy or A.fast:                                                   # as r1_lat.py --deploy / lat_scale.py (before any correction)
    os.environ['OPL_TAILT_FUSED'] = '1'; os.environ['OPL_COARSE_FP32'] = '1'
if A.table5:
    if not A.fast or A.exact or A.check:
        raise SystemExit('--table5 applies to the --fast NICE route only')
    os.environ['OPL_COARSE_ELEM'] = '1'; os.environ['OPL_TET_TRITON'] = '1'; os.environ['OPL_COARSE_SPARSE'] = '1'
    # coarse factor of the float32 correction: a Cholesky factor is accepted only if its smallest pivot^2 is >= 1e-7 of the
    # unit (Jacobi-scaled) diagonal, otherwise the next diagonal shift is tried (trainlib._chol_jitter; Appendix F.2)
    os.environ.setdefault('OPL_COARSE_PIVOT_FLOOR', '1e-7')
    for _k in [k for k in os.environ.get('OPL_T5_SKIP', '').split(',') if k]:   # diagnostic only (default: none skipped)
        if _k in ('COARSE_ELEM', 'TET_TRITON', 'COARSE_SPARSE'):
            os.environ['OPL_' + _k] = '0'
CLAMP = tuple((A.clamp + ',').split(',')[:2])                            # 'y,min' -> ('y', 'min'); 'cut' -> ('cut', '')
ROOT = Path(A.root); ROOT.mkdir(parents=True, exist_ok=True)
for d in ('packets', 'body', 'tmp'):
    (ROOT / d).mkdir(exist_ok=True)
os.environ['OPL_PACKETS_EXTRA'] = ':'.join([str(ROOT / 'packets')] + [p for p in os.environ.get('OPL_PACKETS_EXTRA', '').split(':') if p])

import torch                                                            # noqa: E402
import models as MD                                                     # noqa: E402,F401  first: applies OPL_CONV_FP32
import teacher as TE                                                    # noqa: E402
import trainlib as TL                                                   # noqa: E402
import fastnet as FN                                                    # noqa: E402
import evalnet as EN                                                    # noqa: E402
import bench_deploy as BD                                               # noqa: E402
import lat_multi as LM                                                  # noqa: E402
import lat_precond as PR                                                # noqa: E402
import lat_hetero as LH                                                 # noqa: E402
import r1x3_common as RC                                                # noqa: E402
import mma as MMA                                                       # noqa: E402
if A.table5:
    import fastidx as FI                                                # noqa: E402
    FI.ON = 'FI' not in os.environ.get('OPL_T5_SKIP', '').split(',')

dev, dt = TE.dev, TE.dt
BODY, TMP = ROOT / 'body', ROOT / 'tmp'
HIST, STATE = ROOT / 'history.jsonl', ROOT / 'state.npz'
log = lambda d: print(json.dumps(RC.tojson(d), default=float), flush=True)
AX = {'x': 0, 'y': 1, 'z': 2}


def packet_root(case):
    for r in [p for p in os.environ.get('OPL_PACKETS_EXTRA', '').split(':') if p] + ['/root/autodl-tmp/CUTFEM_FRESH_GP_20260921/packets']:
        if (Path(r) / case / 'FRESH_CONTEXT.json').exists():
            return Path(r) / case
    raise FileNotFoundError(case)


# ------------------------------------------------------------------------------------------------ geometry per iteration
def write_packets(base_cases, corners, k):
    cases = []
    for base, tc in zip(base_cases, corners):
        case = f'{base}_o{k:03d}'
        d = ROOT / 'packets' / case
        if not (d / 'FRESH_CONTEXT.json').exists():
            pk = packet_root(base)
            cx = json.loads((pk / 'FRESH_CONTEXT.json').read_text())
            cx['case']['tau_corners'] = [format(float(x), '.12f') for x in tc]
            cx['case']['case_id'] = case
            cx.setdefault('provenance', {})['opt_design_of'] = base
            cx['provenance']['opt_iteration'] = k
            d.mkdir(parents=True, exist_ok=True)
            (d / 'FRESH_CONTEXT.json').write_text(json.dumps(cx, indent=1))
            shutil.copy(pk / 'SAMPLE.json', d / 'SAMPLE.json')
        cases.append(case)
    return cases


class BodyFail(RuntimeError):
    def __init__(self, bad):
        super().__init__(f'BODY_FAIL {bad}'); self.bad = bad


def make_bodies(cases):
    """fast_prep4 for every case without PREP.json, --workers in parallel; raises on any failure."""
    if not (BODY / Path(A.templates).name).exists() and Path(A.templates).exists():
        shutil.copy(A.templates, BODY / Path(A.templates).name)
    todo = [c for c in cases if not (BODY / c / 'PREP.json').exists()]
    if todo:
        src = Path(__file__).resolve().parent / 'fast_prep4.py'
        one = ROOT / 'body_one.sh'
        one.write_text('#!/bin/bash\n[ -f $1/$2/PREP.json ] && exit 0\n'
                       f'cd {A.frozen_cwd} && OMP_NUM_THREADS=1 timeout 1800 {A.frozen} {src} 1 $1 $2 > $1/$2.log 2>&1 || touch $1/$2.failed\n')
        one.chmod(0o755)
        (ROOT / 'todo.txt').write_text('\n'.join(todo) + '\n')
        subprocess.run(['bash', '-c', f'xargs -a {ROOT}/todo.txt -P {A.workers} -I{{}} {one} {BODY} {{}}'], check=False,
                       env=dict(os.environ))
    bad = [c for c in cases if not (BODY / c / 'PREP.json').exists()]
    if bad:
        raise BodyFail(bad)
    return len(todo)


# ------------------------------------------------------------------------------------------------ constraints
def build_constraint_structure(vid):
    """Unique span pairs (va, vb) and gradient stencils (vc, vx, vy, vz) over all cells (vertex ids)."""
    pairs, stencils = set(), set()
    for row in vid:
        for a in range(8):
            for b in range(8):
                if a != b and row[a] != row[b]:
                    pairs.add((int(row[a]), int(row[b])))
            stencils.add((int(row[a]), int(row[a ^ 4]), int(row[a ^ 2]), int(row[a ^ 1])))
    return sorted(pairs), sorted(stencils)


def constraints(tv, free, V, dV, Vstar, pairs, stencils):
    """Scaled constraint values (m,) and gradients (m, n_free) at the vertex field tv."""
    nv = len(tv); col = np.full(nv, -1); col[free] = np.arange(len(free))
    f, G = [V / Vstar - 1.0], [dV[free] / Vstar]
    for va, vb in pairs:
        g = np.zeros(len(free))
        if col[va] >= 0:
            g[col[va]] += 1 / A.span
        if col[vb] >= 0:
            g[col[vb]] -= 1 / A.span
        if not g.any():
            continue
        f.append((tv[va] - tv[vb]) / A.span - 1.0); G.append(g)
    for vc, vx, vy, vz in stencils:
        dd = np.array([tv[vx] - tv[vc], tv[vy] - tv[vc], tv[vz] - tv[vc]])
        g = np.zeros(len(free))
        for vn, di in zip((vx, vy, vz), dd):
            if col[vn] >= 0:
                g[col[vn]] += 2 * di / A.grad ** 2
            if col[vc] >= 0:
                g[col[vc]] -= 2 * di / A.grad ** 2
        if not g.any():
            continue
        f.append((dd ** 2).sum() / A.grad ** 2 - 1.0); G.append(g)
    return np.asarray(f), np.asarray(G)


# ------------------------------------------------------------------------------------------------ one analysis
def analyse(cases, positions, h):
    """NICE lattice solve of the current design: compliance, s_tilde per cell (ncell x 8), cell volumes and dV (ncell x 8),
    solve scalars and per-cell switch fingerprints."""
    T = {}
    t = time.perf_counter()
    Cs, lay, vol, dvol = [], {}, [], []
    for case, p in zip(cases, positions):
        C = TE.Cell(case, str(BODY), log=lambda s_: None); C.assemble()
        C.dmoments()
        vol.append(float(C.M[:, 0].sum())); dvol.append(C.dM[:, :, 0].sum(1).cpu().numpy())
        Cs.append(C); lay[tuple(p)] = LM.from_teacher(C)
        if A.park:
            LH._move(C, torch.device('cpu')); RC.free()
    la = A.load.split(',')
    lat = LM.MultiLattice(lay, clamp=CLAMP, load=(la[0], la[1]), loads='consistent', n_random=0, device=dev,
                          max_cols=A.max_cols, log=lambda s_: None)
    order = [g.case for g in lat.geoms]
    Cmap = {C.case: C for C in Cs}
    F = lat.F[:, ['xyz'.index(A.load_dir)]].contiguous()
    kpp = lat.assemble_kpp() if A.park else None
    T['setup_s'] = time.perf_counter() - t
    t = time.perf_counter()
    lops, fps = [], {}
    for j, c in enumerate(order):
        C = Cmap[c]
        if A.park:
            LH._move(C, dev)
        BD.netdata(C, str(BODY), TMP / c)
        geo = TL.Geo(c, str(BODY), TMP, neumann=False, log=lambda s_: None, cell=C, load_banks=False)
        model = h.add(geo)
        op = EN.FastOp(FN.FastNet(model, geo))
        if A.deploy and not getattr(C, '_lean', False):
            C.lean(deploy=True); RC.free()
        elif A.lean and not getattr(C, '_lean', False):
            C.lean(); RC.free()
        with torch.no_grad():
            op.apply(torch.zeros((C.np_, 1), dtype=dt, device=dev))
        cc = model.caches.get(c)
        faces = np.load(BODY / c / 'GP_FACES.npy')
        fps[c] = dict(active=int(len(C.cells)), faces=int(len(faces)), ports=int(C.np_), cut_nodes=int(C.is_cut.sum()),
                      weak=int(geo.nd['weak'].sum()),
                      el_fringe=int(cc.el_fringe.sum()) if cc is not None and getattr(cc, 'el_fringe', None) is not None else None,
                      gp_fringe=int(cc.gp_fringe.sum()) if cc is not None and getattr(cc, 'gp_fringe', None) is not None else None,
                      shift=float(getattr(C, '_c_shift', 0.0) or 0.0),
                      b=float(C._tail_bounds[2]) if getattr(C, '_tail_bounds', None) else None)
        if A.park:
            op = LH.ParkedOp(op, C, resident=j < A.resident)
            if not op.resident:
                LH._move(C, torch.device('cpu')); RC.free()
        lops.append(op)
    T['prep_s'] = time.perf_counter() - t
    t = time.perf_counter()
    fac = PR.Factory(lat, lops, shared={} if kpp is None else {'kpp_triplets': kpp, 'kpp_triplets_s': 0.0},
                     kpp_backend='auto', log=lambda d: None)
    pc, pst, _ = fac.build(A.prec)
    T['precond_s'] = time.perf_counter() - t
    t = time.perf_counter()
    r = PR.pcg(lat, lops, pc, F=F, tol=A.tol, maxit=A.maxit_pcg)
    fac.free(); del pc
    X = r['X']
    with torch.no_grad():
        rho = F - lat.matvec(lops, X)
        Chat = float((F * X).sum()); Ur = float((X * rho).sum())
        tres = float(rho.norm() / F.norm())
    T['solve_s'] = time.perf_counter() - t
    t = time.perf_counter()
    S = np.zeros((len(order), 8))
    for i, c in enumerate(order):
        C = Cmap[c]
        ctx = lops[i].active() if A.park else contextlib.nullcontext()
        with ctx:
            if A.park:
                LH._move(C, dev, min_bytes=0)
            with torch.no_grad():
                q = lat.gather(X, i)
                u = lops[i].field(q).to(dt)
                S[i] = C.sens(u)[:, 0].cpu().numpy()
                del u
        if A.park and not lops[i].resident:
            LH._move(C, torch.device('cpu')); RC.free()
    T['sens_s'] = time.perf_counter() - t
    pos_order = [tuple(lat.positions[i]) for i in range(len(order))]
    idx = [cases.index(c) for c in order]
    out = dict(C=Chat, S=S, order=order, pos=pos_order, vol=np.asarray(vol)[idx], dvol=np.stack(dvol)[idx],
               pcg=int(r['iterations']), true_residual=tres, Ut_rho_rel=Ur / Chat, fps=[fps[c] for c in order], times=T)
    for c in order:
        h.model.caches.pop(c, None)
    del lops, lat, X, rho, F
    for C in Cs:
        C._free()
    del Cs, Cmap
    RC.free()
    return out


_WARM = {}


def _dof_keys(lat):
    """One int64 key per free lattice DOF: absolute grid position, component, private (cut-port) flag."""
    g = lat.gpos[lat.free].astype(np.int64); c = lat.gcomp[lat.free].astype(np.int64)
    M = np.int64(1 << 16)
    return (((g[:, 0] * M + g[:, 1]) * M + g[:, 2]) * 3 + c) * 2 + np.asarray(lat.priv, np.int64)


def analyse_fast(cases, positions, h):
    """--fast: the deployment route of Table 5 (lat_scale.py) for the current design; same outputs as analyse()."""
    import gc
    import stream_ops as SO
    import lat_scale as LS
    FN.FUSED = True
    if not getattr(analyse_fast, '_init', False):
        SO.init(); analyse_fast._init = True
    T = {}
    t = time.perf_counter()
    Cs, ops, kpp_host, fps = {}, {}, {}, {}
    for case in cases:
        C = TE.Cell(case, str(BODY), log=lambda s_: None, deploy=True)
        C.assemble_deploy(C.taus0)
        BD.netdata(C, str(BODY), TMP / case)
        geo = TL.Geo(case, str(BODY), TMP, neumann=False, log=lambda s_: None, cell=C, load_banks=False)
        if h.model is not None:
            h.model.caches.pop(case, None)
        model = h.add(geo)
        op = EN.FastOp(FN.FastNet(model, geo))
        with torch.no_grad():
            op.apply(torch.zeros((C.np_, 1), dtype=dt, device=dev))
        cc = model.caches.get(case)
        fps[case] = dict(active=int(len(C.cells)), faces=int(len(np.load(BODY / case / 'GP_FACES.npy'))), ports=int(C.np_),
                         cut_nodes=int(C.is_cut.sum()), weak=int(geo.nd['weak'].sum()),
                         el_fringe=int(cc.el_fringe.sum()) if cc is not None and getattr(cc, 'el_fringe', None) is not None else None,
                         gp_fringe=int(cc.gp_fringe.sum()) if cc is not None and getattr(cc, 'gp_fringe', None) is not None else None,
                         shift=float(getattr(C, '_c_shift', 0.0) or 0.0),
                         b=float(C._tail_bounds[2]) if getattr(C, '_tail_bounds', None) else None)
        kpp_host[case] = tuple(x.cpu() for x in C._kpp_cache)
        C._kpp_cache = kpp_host[case]; C.diag3 = None
        if torch.cuda.memory_allocated() / 2 ** 30 < A.resident_gb:
            ops[case] = LS._Resident(op)
        else:
            ops[case] = SO.StreamedOp(op, C, [op.fast, geo, C])
        Cs[case] = C
        del geo, op
        gc.collect(); torch.cuda.empty_cache()
    T['prep_s'] = time.perf_counter() - t
    t = time.perf_counter()
    lay = {}
    for case, p in zip(cases, positions):
        g = LM.from_teacher(Cs[case])
        g._kpp_fn = (lambda cs: (lambda: tuple(x.to(dev) for x in kpp_host[cs])))(case)
        lay[tuple(p)] = g
    la = A.load.split(',')
    lat = LM.MultiLattice(lay, clamp=CLAMP, load=(la[0], la[1]), loads='consistent', n_random=0, device=dev,
                          max_cols=A.max_cols, log=lambda s_: None)
    order = [g.case for g in lat.geoms]
    olist = [ops[c] for c in order]
    SO.link([o for o in olist if isinstance(o, SO.StreamedOp)])
    F = lat.F[:, ['xyz'.index(A.load_dir)]].contiguous()
    for g in lat.geoms:
        g._kpp = None
    kpp = lat.assemble_kpp()
    for g in lat.geoms:
        g._kpp = None
    T['setup_s'] = time.perf_counter() - t
    t = time.perf_counter()
    fac = PR.Factory(lat, olist, shared={'kpp_triplets': kpp, 'kpp_triplets_s': 0.0}, kpp_backend='auto', log=lambda d: None)
    pc, _, _ = fac.build(A.prec)
    del kpp
    T['precond_s'] = time.perf_counter() - t
    t = time.perf_counter()
    X0 = None
    if A.warm:
        keys = _dof_keys(lat)
        if 'keys' in _WARM:
            pk, px = _WARM['keys'], _WARM['X']
            pos = np.clip(np.searchsorted(pk, keys), 0, len(pk) - 1)
            hit = pk[pos] == keys
            x0 = np.zeros(len(keys)); x0[hit] = px[pos[hit]]
            X0 = torch.as_tensor(x0[:, None], dtype=dt, device=dev)
            with torch.no_grad():
                den = float((X0 * lat.matvec(olist, X0)).sum())
                scale = float((F * X0).sum()) / den if den > 0 else 0.0
            X0 *= scale
            T['warm_hit'] = float(hit.mean()); T['warm_scale'] = scale
    r = PR.pcg(lat, olist, pc, F=F, tol=A.tol, maxit=A.maxit_pcg, X0=X0)
    LS._free_prec(fac, pc); del fac, pc
    X = r['X']
    if A.warm:
        o_ = np.argsort(keys)
        _WARM['keys'], _WARM['X'] = keys[o_], X[:, 0].detach().cpu().numpy()[o_]
    with torch.no_grad():
        rho = F - lat.matvec(olist, X)
        Chat = float((F * X).sum()); Ur = float((X * rho).sum())
        tres = float(rho.norm() / F.norm())
    T['solve_s'] = time.perf_counter() - t
    t = time.perf_counter()
    S = np.zeros((len(order), 8)); vol = np.zeros(len(order)); dvol = np.zeros((len(order), 8))
    for i, c in enumerate(order):
        C, o = Cs[c], olist[i]
        with o.active():
            with torch.no_grad():
                u = o.field(lat.gather(X, i)).to(dt)
            vol[i] = float(C.M[:, 0].sum())
            if A.sens == 'ad':
                import moments_ad as MA
                with torch.no_grad():
                    g = C.energy_density(u)
                    e0 = torch.zeros((g.shape[0], g.shape[1], 1), dtype=g.dtype, device=g.device); e0[:, 0, 0] = 1.0
                    g = torch.cat([g, e0], 2); del e0
                r_ = MA.cell_sens(C, g, batch=A.ad_batch, skip_full=True).detach().cpu().numpy()   # (8, 2): -u^T dK u, -dV
                S[i] = r_[:, 0]; dvol[i] = -r_[:, 1]
                del g
            else:
                with torch.no_grad():
                    C.dmoments()
                    S[i] = C.sens(u)[:, 0].cpu().numpy()
                    dvol[i] = C.dM[:, :, 0].sum(1).cpu().numpy()
                    C.dM = None
            del u
    SO.park_all()
    T['sens_s'] = time.perf_counter() - t
    out = dict(C=Chat, S=S, order=order, vol=vol, dvol=dvol, pcg=int(r['iterations']), true_residual=tres, Ut_rho_rel=Ur / Chat,
               fps=[fps[c] for c in order], times=T, streamed=int(sum(isinstance(o, SO.StreamedOp) for o in olist)))
    for o in olist:
        o.release()
    for c in order:
        h.model.caches.pop(c, None)
    del olist, ops, lat, X, rho, F, Cs
    gc.collect(); torch.cuda.empty_cache()
    return out


def make_T(cases):
    """Dense exact condensed matrix per cell (make_T_gpu.py, one process per cell), cached next to the body."""
    t = time.perf_counter(); n = 0
    src = Path(__file__).resolve().parent / 'make_T_gpu.py'
    for c in cases:
        if (BODY / f'{c}_portview' / 'T64.npy').exists():
            continue
        r = subprocess.run([sys.executable, '-u', str(src), str(BODY), c], capture_output=True, text=True,
                           cwd=str(src.parent), env=dict(os.environ))
        if r.returncode or not (BODY / f'{c}_portview' / 'T64.npy').exists():
            (BODY / f'{c}.make_T_gpu.err').write_text(r.stdout + '\n' + r.stderr)
            if not A.t_cpu_fallback:
                raise RuntimeError(f'MAKE_T_FAIL {c}: {r.stderr[-400:]}')
            cpu = Path(__file__).resolve().parent / 'make_T_cpu.py'
            r = subprocess.run([sys.executable, '-u', str(cpu), str(BODY), c], capture_output=True, text=True,
                               cwd=str(cpu.parent), env=dict(os.environ, OPL_DEV='cpu'))
            print(json.dumps(dict(event='MAKE_T_CPU_FALLBACK', case=c, rc=r.returncode,
                                  ok=(BODY / f'{c}_portview' / 'T64.npy').exists())), flush=True)
            if r.returncode or not (BODY / f'{c}_portview' / 'T64.npy').exists():
                raise RuntimeError(f'MAKE_T_FAIL {c} (gpu and cpu): {r.stderr[-400:]}')
        n += 1
    return n, time.perf_counter() - t


def analyse_exact(cases, positions):
    """Exact counterpart of analyse(): compliance, exact sensitivities per cell, volumes, solve scalars."""
    T = {}
    nT, T['make_T_s'] = make_T(cases)
    t = time.perf_counter()
    Cs, lay, vol, dvol = [], {}, [], []
    for case, p in zip(cases, positions):
        C = TE.Cell(case, str(BODY), log=lambda s_: None); C.assemble()
        C.dmoments()
        vol.append(float(C.M[:, 0].sum())); dvol.append(C.dM[:, :, 0].sum(1).cpu().numpy())
        Cs.append(C); lay[tuple(p)] = LM.from_teacher(C)
        LH._move(C, torch.device('cpu')); RC.free()
    la = A.load.split(',')
    lat = LM.MultiLattice(lay, clamp=CLAMP, load=(la[0], la[1]), loads='consistent', n_random=0, device=dev,
                          max_cols=A.max_cols, log=lambda s_: None)
    order = [g.case for g in lat.geoms]
    Cmap = {C.case: C for C in Cs}
    F = lat.F[:, ['xyz'.index(A.load_dir)]].contiguous()
    kpp = lat.assemble_kpp()
    T['setup_s'] = time.perf_counter() - t
    t = time.perf_counter()
    ops = [LH.DenseExactOp(Cmap[c], str(BODY)) for c in order]
    fac = PR.Factory(lat, ops, shared={'kpp_triplets': kpp, 'kpp_triplets_s': 0.0}, kpp_backend='auto', log=lambda d: None)
    pc, _, _ = fac.build(A.prec)
    r = PR.pcg(lat, ops, pc, F=F, tol=A.exact_tol, maxit=A.maxit_pcg)
    fac.free(); del pc
    X = r['X']
    with torch.no_grad():
        rho = F - lat.matvec(ops, X)
        Cex = float((F * X).sum()); tres = float(rho.norm() / F.norm())
    del ops; RC.free()
    T['solve_s'] = time.perf_counter() - t
    t = time.perf_counter()
    S = np.zeros((len(order), 8))
    for i, c in enumerate(order):
        C = Cmap[c]
        LH._move(C, dev, min_bytes=0)
        LH.exact_op_host(C)
        with torch.no_grad():
            u = C.extend(lat.gather(X, i).to(dt))
            S[i] = C.sens(u)[:, 0].cpu().numpy()
        del u
        C._free(); LH._move(C, torch.device('cpu')); RC.free()
    T['sens_s'] = time.perf_counter() - t
    idx = [cases.index(c) for c in order]
    out = dict(C=Cex, S=S, order=order, vol=np.asarray(vol)[idx], dvol=np.stack(dvol)[idx], pcg=int(r['iterations']),
               true_residual=tres, Ut_rho_rel=0.0, fps=[dict(active=int(len(Cmap[c].cells)), ports=int(Cmap[c].np_)) for c in order],
               times=dict(T, make_T_new=nT))
    del lat, X, rho, F
    for C in Cs:
        C._free()
    RC.free()
    return out


def check(iters):
    """Exact verification of iterations of an existing run."""
    L = json.loads(Path(A.layout).read_text())
    base = [c['case'] for c in L['cells']]
    positions = [tuple(int(v) for v in c['position']) for c in L['cells']]
    meta = json.loads((ROOT / 'meta.json').read_text())
    vid, fixed = np.asarray(meta['vid']), np.asarray(meta['fixed'], bool)
    nv = len(fixed); free = np.flatnonzero(~fixed)
    hist = {}
    for line in HIST.read_text().splitlines():
        d = json.loads(line); hist[int(d['k'])] = d
    for k in iters:
        d = hist[k]
        cases = d['cases']
        res = analyse_exact(cases, positions)
        m_of = {c: i for i, c in enumerate(cases)}
        perm = [m_of[c] for c in res['order']]
        S = np.zeros((len(base), 8)); S[perm] = res['S']
        dVc = np.zeros((len(base), 8)); dVc[perm] = res['dvol']
        g_ex = RC.aggregate(S[:, :, None], vid, nv)[:, 0][free]
        dV = RC.aggregate(dVc[:, :, None], vid, nv)[:, 0][free]
        g_ti = np.asarray(d['s_vertex'], float)[free]
        err = np.abs(g_ti - g_ex) / np.abs(g_ex).max()
        tv = np.asarray(d['tv'], float)[free]
        inner = (tv > A.tmin + 1e-6) & (tv < A.tmax - 1e-6)
        gi, vi = g_ex[inner], dV[inner]
        lam = -float(gi @ vi / (vi @ vi)) if inner.any() else 0.0
        kkt = float(np.linalg.norm(gi + lam * vi) / np.linalg.norm(gi)) if inner.any() else float('nan')
        out = dict(k=k, C_exact=res['C'], C_hat=d['C'], surrogate_err=(d['C'] - res['C']) / res['C'],
                   grad_rel_err=float(np.linalg.norm(g_ti - g_ex) / np.linalg.norm(g_ex)),
                   grad_cos=float(g_ti @ g_ex / (np.linalg.norm(g_ti) * np.linalg.norm(g_ex))),
                   comp_err_rel_to_max=dict(median=float(np.median(err)), p95=float(np.percentile(err, 95)), max=float(err.max())),
                   sign_agreement=float(np.mean(np.sign(g_ti) == np.sign(g_ex))), kkt_exact=kkt, kkt_lambda=lam,
                   exact_pcg=res['pcg'], exact_true_residual=res['true_residual'], times=res['times'], g_exact=g_ex, g_tilde=g_ti)
        (ROOT / f'check_{k:03d}.json').write_text(json.dumps(RC.tojson(out), default=float, indent=1))
        log({k_: v for k_, v in out.items() if k_ not in ('g_exact', 'g_tilde')})


# ------------------------------------------------------------------------------------------------ main loop
def main():
    L = json.loads(Path(A.layout).read_text())
    base = [c['case'] for c in L['cells']]
    positions = [tuple(int(v) for v in c['position']) for c in L['cells']]
    taus0 = [[float(x) for x in c['tau_corners']] for c in L['cells']]
    vid, nv, tv0, dmax, vkeys = RC.vertex_map(positions, taus0)
    if dmax > 1e-9:
        raise ValueError(f'INCONSISTENT_SHARED_TAU {dmax}')
    la = A.load.split(',')
    ax = AX[la[0]]
    coord = np.asarray(vkeys)[:, ax]
    plane = coord.max() if la[1] == 'max' else coord.min()
    fixed = coord == plane
    free = np.flatnonzero(~fixed)
    pairs, stencils = build_constraint_structure(vid)
    name, h = (None, None) if A.exact else LH.holder_for(A.model)
    if STATE.exists():
        z = np.load(STATE, allow_pickle=True)
        k0, tv, Vstar, C0 = int(z['k']), z['tv'], float(z['Vstar']), float(z['C0'])
        mstate = z['mstate'].item(); fhist = list(z['fhist'])
        log(dict(event='RESUME', k=k0))
    else:
        k0, tv, Vstar, C0, mstate, fhist = 0, tv0.copy(), None, None, None, []
    meta = dict(event='START', layout=A.layout, cells=len(base), vertices=nv, free=len(free), fixed=int(fixed.sum()),
                span_pairs=len(pairs), grad_stencils=len(stencils), args=vars(A), env=RC.env_flags())
    log(meta)
    if k0 == 0:
        (ROOT / 'meta.json').write_text(json.dumps(RC.tojson(dict(meta, vid=vid, vkeys=vkeys, fixed=fixed, tv0=tv0)), default=float))
    k = k0; nrun = 0
    while True:
        t_it = time.perf_counter()
        corners = [tv[vid[m]] for m in range(len(base))]
        cases = write_packets(base, corners, k)
        t = time.perf_counter(); tv_mma, perturb = tv.copy(), []
        for attempt in range(A.body_retry + 1):
            try:
                nb = make_bodies(cases); break
            except BodyFail as e:
                if attempt == A.body_retry:
                    raise
                eps = [1e-4, -1e-4, 1e-3, -1e-3, 3e-3, -3e-3][attempt]
                vs = sorted({int(v) for c in e.bad for v in vid[cases.index(c)] if not fixed[v]})
                tv = tv_mma.copy(); tv[vs] = np.clip(tv_mma[vs] * (1 + eps), A.tmin, A.tmax)
                aff = [m for m in range(len(base)) if set(int(v) for v in vid[m]) & set(vs)]
                for m in aff:
                    c = cases[m]
                    shutil.rmtree(ROOT / 'packets' / c, ignore_errors=True); shutil.rmtree(BODY / c, ignore_errors=True)
                    if (BODY / f'{c}.log').exists():
                        (BODY / f'{c}.log').rename(BODY / f'{c}.log.attempt{attempt}')
                    if (BODY / f'{c}.failed').exists():
                        (BODY / f'{c}.failed').unlink()
                corners = [tv[vid[m]] for m in range(len(base))]
                cases = write_packets(base, corners, k)
                perturb.append(dict(attempt=attempt, eps=eps, failed=e.bad, vertices=vs, regenerated=[cases[m] for m in aff]))
                log(dict(event='BODY_PERTURB', k=k, **perturb[-1]))
        t_body = time.perf_counter() - t
        torch.cuda.reset_peak_memory_stats()
        res = analyse_exact(cases, positions) if A.exact else (analyse_fast(cases, positions, h) if A.fast else analyse(cases, positions, h))
        # cell arrays are in lattice order; map to the layout order of vid
        m_of = {c: i for i, c in enumerate(cases)}
        perm = [m_of[c] for c in res['order']]
        S = np.zeros((len(base), 8)); Vc = np.zeros(len(base)); dVc = np.zeros((len(base), 8))
        S[perm] = res['S']; Vc[perm] = res['vol']; dVc[perm] = res['dvol']
        gv = RC.aggregate(S[:, :, None], vid, nv)[:, 0]
        dVv = RC.aggregate(dVc[:, :, None], vid, nv)[:, 0]
        V = float(Vc.sum())
        if Vstar is None:
            Vstar, C0 = (A.vstar if A.vstar > 0 else A.vfrac * V), res['C']
        fval, dfdx = constraints(tv, free, V, dVv, Vstar, pairs, stencils)
        f0 = res['C'] / C0
        fhist.append(f0)
        t = time.perf_counter()
        xnew, mstate, info = MMA.mma_update(tv[free], f0, gv[free] / C0, fval, dfdx, A.tmin, A.tmax, mstate, move=A.move)
        t_mma = time.perf_counter() - t
        dx = float(np.abs(xnew - tv[free]).max())
        rec = dict(event='ITER', k=k, C=res['C'], f0=f0, V=V, V_rel=V / Vstar, g_max=float(fval.max()),
                   span_max=float(max((tv[a] - tv[b] for a, b in pairs), default=0.0)),
                   grad_max=float(np.sqrt(max(sum((tv[n] - tv[c]) ** 2 for n in (x_, y_, z_)) for c, x_, y_, z_ in stencils))),
                   tau_min=float(tv.min()), tau_max=float(tv.max()), dx=dx, grad_norm=float(np.linalg.norm(gv[free] / C0)),
                   pcg=res['pcg'], true_residual=res['true_residual'], Ut_rho_rel=res['Ut_rho_rel'],
                   bodies_new=nb, times=dict(res['times'], bodies_s=t_body, mma_s=t_mma, iter_s=time.perf_counter() - t_it),
                   gpu_peak_gb=torch.cuda.max_memory_allocated() / 2 ** 30,
                   host_peak_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20,
                   mma=dict(newton=info['newton_steps'], y_max=info['y_max'], z=info['z']),
                   cases=cases, fps=dict(zip(res['order'], res['fps'])), body_perturb=perturb)
        with open(HIST, 'a') as fh:
            fh.write(json.dumps(RC.tojson(dict(rec, tv=tv, s_vertex=gv, s_cell=S)), default=float) + '\n')
        log({k_: v for k_, v in rec.items() if k_ not in ('cases', 'fps')})
        if A.table5 and (res['pcg'] >= A.maxit_pcg or res['true_residual'] > 1e-2):   # final route: never continue from
            log(dict(event='PCG_NOT_CONVERGED', k=k, pcg=res['pcg'], true_residual=res['true_residual']))   # a failed solve
            raise SystemExit('PCG_NOT_CONVERGED')
        tv = tv.copy(); tv[free] = xnew
        k += 1; nrun += 1
        np.savez(STATE, k=k, tv=tv, Vstar=Vstar, C0=C0, mstate=np.array(mstate, dtype=object), fhist=np.asarray(fhist))
        small_f = len(fhist) >= 4 and all(abs(fhist[-i] - fhist[-i - 1]) / abs(fhist[-i - 1]) < A.ftol for i in (1, 2, 3))
        if dx < A.xtol or small_f or k >= A.maxit:
            log(dict(event='CONVERGED' if (dx < A.xtol or small_f) else 'MAXIT', k=k, dx=dx))
            break
        if A.stop_after and nrun >= A.stop_after:
            log(dict(event='STOP_AFTER', k=k)); break


if __name__ == '__main__':
    if A.check:
        check([int(x) for x in A.check.split(',') if x])
    else:
        main()
