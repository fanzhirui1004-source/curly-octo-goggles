"""Exact verification of design iterations of opt_design.py runs on a CPU host (no GPU): the counterpart of
opt_design.py --check for lattices whose dense exact condensed cell matrices T do not fit next to the device (plates B1/B2,
the homogenisation designs and their NICE continuations of Section 6.11: 65-93 GB of T per design in fp64).
NEW script; opt_design.py, make_T_cpu.py, lat_hetero.py, lat_multi.py, lat_precond.py are used unchanged.

Per requested design (iteration k of a run directory written by opt_design.py: history.jsonl, meta.json, packets/, body/):
  plan      cases of iteration k (history.jsonl), layout positions, meta.json (vid, fixed; clamp, load face, load column,
            tmin / tmax and max_cols from the run's arguments); one fingerprint per cell: sha1 of the body arrays that
            teacher.Cell reads (NODES, CELL_INDICES, dofs, GP_FACES, BOX_NODES, CUT_NODES), the ghost templates, and the
            packet quantities that enter K (n, E, nu, surface, tau corners, plane, gamma). Equal fingerprints = byte-identical
            inputs = the same K and T (the 16 uncut cells and the four pairs of cut cells of the uniform plate start; B1 and
            B2 at k = 0): T is built and held once, and those cells share ONE operator object, which lat_multi applies to all
            of them in one call. --no-dedupe: one T per cell.
  cellinfo  per distinct cell, a worker process (the CPU environment of make_T_cpu.py): teacher.Cell + assemble() ->
            port node ids / flags, the upper K_PP triplets (lat_multi.from_teacher) and the consistent face-load weights of
            all six box faces (lattice3.face_traction_weights through from_teacher) -> <work>/cellinfo/<fp>.npz
  T         per distinct cell without a valid <body>/<case>_portview/T64.npy (shape, dtype, readable, BOX_NODES = ports):
              --t-route make_T_cpu (default): make_T_cpu.py unchanged (columns S e_j = (K E e_j)_P in blocks of --block
                through the interior factor of diag_sens.cpu_factor (pypardiso), symmetrised), run in a worker that first
                sets box_encode.dev to the host: without GP_UPPER.npz (the runs used OPL_GP_CACHE=0) teacher.Cell assembles
                the ghost penalty with box_encode.ghost_faces_gpu, whose device is otherwise cuda:0;
              --t-route schur: MKL PARDISO Cholesky with the Schur-complement option (iparm(36) = 1) on the Jacobi-scaled
                cell matrix with the retained DOFs marked in perm (bench_cpu2.explicit_schur, the explicit-S column of
                Table ST18: 11-166 s per cell there, columns equal to C.apply to 1e-14), unscaled and symmetrised; before
                T64.npy is written, --verify-cols columns are compared with C.apply (diag_sens.cpu_factor, as
                make_T_cpu.py --check) and T is refused above --verify-tol. Same files as make_T_cpu.py.
  solve     (this process) the lattice of the run (lat_multi.MultiLattice: same clamp, load face, consistent load column),
            one host operator per distinct cell, q -> T q in fp64 (the product of lat_hetero.DenseExactOp), PCG
            (lat_precond.pcg) from zero with the run's preconditioner (--prec, default: the run's, bnn:kpp:q1r) to the
            recursive relative residual --tol (1e-10). Fine level 'kpp' = K_PP^-1: --fine pardiso (default) factorises the
            Jacobi-scaled K_PP by MKL PARDISO Cholesky (pardiso_direct; the action of lat_precond.SparseSPD, which uses cuDSS
            on the GPU) and hands it to lat_precond.Factory; --fine factory leaves it to lat_precond (SuperLU / CHOLMOD).
            Exact compliance f^T U, recomputed residual, port vectors q_c and energies q_c^T T_c q_c per cell.
  sens      per distinct cell a worker: teacher.Cell + assemble(); --sens fd (default, as opt_design.analyse_exact):
            dmoments() (central differences of the moments) -> dV and s = C.sens(u); --sens ad: moments_ad.cell_sens
            (reverse mode, the NICE route of the runs); the interior factor diag_sens.cpu_factor and u = C.extend(q) for all
            cells of the group at once; the energy u^T K u is compared with q^T T q of the solve (consistency of T and u).
  check     aggregation to the vertices (r1x3_common.aggregate) and the quantities of opt_design.check(), written to
            <run_dir>/check_exact_<k:03d>.json with the keys of check_<k:03d>.json plus a record 'cpu' (settings, groups,
            times and peak memory per phase, consistency checks, per-cell values). --no-sens: compliance only (the
            gradient keys are null).
Phases are resumable: cellinfo, T, the solve (<work>/<run>_k<k>/solve.json + q/), the sensitivities (sens/<fp>.json)
and the check are skipped when their outputs exist (--force: redo the solve, sensitivities and check).
--delete-T removes each T64.npy after the last solve of this invocation that needs it (bounds the disk to one design).
Memory: the solve holds every distinct T of a design (65-93 GB for the plates), refused if MemAvailable is smaller than
that plus --solve-margin-gib; workers start only while MemAvailable >= --min-free-gib (and at most --*-jobs at a time);
a worker that fails (e.g. killed for memory) is retried once alone.
Usage: exact_check_cpu.py <run_dir>:<iters> [<run_dir>:<iters> ...] [--work DIR] [--t-route make_T_cpu|schur]
         [--threads N] [--t-jobs N] [--t-threads N] [--cell-jobs N] [--cell-threads N] [--sens-jobs N] [--sens-threads N]
         [--tol 1e-10] [--maxit 3000] [--prec SPEC] [--fine pardiso|factory] [--sens fd|ad] [--no-sens] [--delete-T]
         [--phases cellinfo,T,solve,sens,check] [--iparm-file iparm_tuned.json] [--dry-run] [--force]
       <iters>: comma list of iteration numbers, 'last' (last recorded k) and 'mid' (last // 2)
       exact_check_cpu.py --self-test
Environment as the CPU scripts (env_cpu.sh): PYPARDISO_MKL_RT and LD_LIBRARY_PATH for libmkl_rt, PYTHONPATH with pypardiso;
this script sets OPL_DEV=cpu, CUDA_VISIBLE_DEVICES= and OPL_GP_CACHE=0 for itself and every worker, and MKL_NUM_THREADS /
OMP_NUM_THREADS per worker from --*-threads.
"""
import os
import sys
import json
import time
import argparse
import hashlib
import re
import subprocess
import gc
from fractions import Fraction
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
GIB = 2 ** 30
FP_FILES = ('NODES.npy', 'CELL_INDICES.npy', 'dofs.npy', 'GP_FACES.npy', 'BOX_NODES.npy', 'CUT_NODES.npy')
PHASES = ('cellinfo', 'T', 'solve', 'sens', 'check')
AXES = {'x': 0, 'y': 1, 'z': 2}
CPU_ENV = dict(OPL_DEV='cpu', CUDA_VISIBLE_DEVICES='', OPL_GP_CACHE='0')


# ================================================================================================ small helpers
def log(d):
    print(json.dumps(tojson(dict(t=time.strftime('%H:%M:%S'), **d)), default=float), flush=True)


def tojson(x):
    """r1x3_common.tojson without its imports (numpy / torch values to JSON types)."""
    if isinstance(x, dict):
        return {str(k): tojson(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [tojson(v) for v in x]
    if isinstance(x, np.ndarray):
        return tojson(x.tolist())
    if 'torch' in sys.modules and sys.modules['torch'].is_tensor(x):
        return tojson(x.detach().cpu().numpy())
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, Path):
        return str(x)
    return x


def write_json(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + '.part')
    tmp.write_text(json.dumps(tojson(obj), default=float, indent=1))
    os.replace(tmp, path)


def aggregate(G, vid, nvert):
    """r1x3_common.aggregate: G (ncell, 8, nload) -> (nvert, nload), sum over the cell corners at each vertex."""
    out = np.zeros((nvert, G.shape[2]))
    for i in range(G.shape[0]):
        for c in range(8):
            out[vid[i, c]] += G[i, c]
    return out


def meminfo_gib(key='MemAvailable'):
    try:
        for line in open('/proc/meminfo'):
            if line.startswith(key + ':'):
                return int(line.split()[1]) / 2 ** 20
    except OSError:
        pass
    return float('inf')


def _read_int(path):
    try:
        v = Path(path).read_text().strip()
    except OSError:
        return None
    return None if v in ('', 'max') or int(v) >= 1 << 60 else int(v)


def _stat_key(path, key):
    try:
        for line in open(path):
            k_, _, v = line.partition(' ')
            if k_ == key:
                return int(v)
    except OSError:
        pass
    return None


def avail_gib():
    """Memory this process tree can still take: min(MemAvailable, container limit - anonymous memory in use) (cgroup v2:
    memory.high or memory.max and memory.stat anon; v1: memory.limit_in_bytes and total_rss), as pardiso_direct.avail_gib
    does; /proc/meminfo alone shows the host inside a container."""
    m = meminfo_gib()
    lim = _read_int('/sys/fs/cgroup/memory.high') or _read_int('/sys/fs/cgroup/memory.max')
    if lim is not None:
        anon = _stat_key('/sys/fs/cgroup/memory.stat', 'anon')
    else:
        lim = _read_int('/sys/fs/cgroup/memory/memory.limit_in_bytes')
        anon = _stat_key('/sys/fs/cgroup/memory/memory.stat', 'total_rss')
    if lim is None or anon is None:
        return m
    return min(m, (lim - anon) / GIB)


def status_gib(key):
    try:
        for line in open('/proc/self/status'):
            if line.startswith(key + ':'):
                return int(line.split()[1]) / 2 ** 20
    except OSError:
        pass
    return float('nan')


def reset_peak():
    """Reset the process high-water mark (VmHWM) where the kernel allows it."""
    try:
        with open('/proc/self/clear_refs', 'w') as f:
            f.write('5')
        return True
    except OSError:
        return False


class Phase:
    """Context: wall seconds and peak resident memory of this process over the block (VmHWM after a reset; the
    lifetime peak where a reset is not allowed, flagged)."""

    def __init__(self, name, rec):
        self.name, self.rec = name, rec

    def __enter__(self):
        self.reset = reset_peak()
        self.t = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.rec[self.name] = dict(seconds=time.perf_counter() - self.t, peak_rss_gib=status_gib('VmHWM'),
                                   peak_is_lifetime=not self.reset, rss_gib=status_gib('VmRSS'),
                                   mem_available_gib=avail_gib())
        return False


def ncpu():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


# ================================================================================================ specs, runs, plan
def parse_spec(s):
    """'<run_dir>:<iters>' -> (Path, [tokens]); tokens: non-negative integers, 'last', 'mid'."""
    if ':' not in s:
        raise argparse.ArgumentTypeError(f'expected <run_dir>:<iters>, got {s!r}')
    run, it = s.rsplit(':', 1)
    toks = [t.strip() for t in it.split(',') if t.strip()]
    if not run or not toks:
        raise argparse.ArgumentTypeError(f'empty run directory or iteration list in {s!r}')
    for t in toks:
        if t not in ('last', 'mid') and not re.fullmatch(r'\d+', t):
            raise argparse.ArgumentTypeError(f'bad iteration {t!r} in {s!r}')
    return Path(run), toks


def resolve_iters(toks, ks):
    """Iteration tokens -> sorted unique recorded iterations ('last' = max k, 'mid' = max k // 2)."""
    ks = sorted(int(k) for k in ks)
    if not ks:
        raise ValueError('EMPTY_HISTORY')
    last, out = ks[-1], []
    for t in toks:
        k = last if t == 'last' else (last // 2 if t == 'mid' else int(t))
        if k not in ks:
            raise ValueError(f'ITERATION_NOT_RECORDED {k} (recorded {ks[0]}..{last})')
        if k not in out:
            out.append(k)
    return out


def load_run(run, layout_override=None):
    """meta.json, history.jsonl (by k), layout (positions, base cases) and the run arguments that define the check."""
    run = Path(run).resolve()
    meta = json.loads((run / 'meta.json').read_text())
    hist = {}
    for line in (run / 'history.jsonl').read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            hist[int(d['k'])] = d
    cands = [Path(layout_override)] if layout_override else []
    cands += [run / 'layout.json']
    for p in (meta.get('layout'), meta.get('args', {}).get('layout')):
        if p:
            cands.append(Path(p))
    lp = next((p for p in cands if p.exists()), None)
    if lp is None:
        raise FileNotFoundError(f'LAYOUT_NOT_FOUND for {run}: tried {[str(c) for c in cands]}')
    L = json.loads(lp.read_text())
    args = meta.get('args', {})
    need = ('clamp', 'load', 'load_dir')
    if any(k not in args for k in need):
        raise ValueError(f'META_ARGS_INCOMPLETE {run}: {need}')
    cfg = dict(clamp=tuple(args['clamp'].split(',')), load=tuple(args['load'].split(',')), load_dir=args['load_dir'],
               tmin=float(args.get('tmin', 0.18)), tmax=float(args.get('tmax', 0.69)), max_cols=args.get('max_cols', 16),
               prec=args.get('prec', 'bnn:kpp:q1r'), nice_tol=args.get('tol'), sens_route_of_run=args.get('sens'))
    return dict(run=run, name=run.name, meta=meta, hist=hist, layout_path=lp,
                base=[c['case'] for c in L['cells']], positions=[tuple(int(v) for v in c['position']) for c in L['cells']],
                vid=np.asarray(meta['vid'], np.int64), fixed=np.asarray(meta['fixed'], bool), cfg=cfg)


_SHA_CACHE = {}


def _file_sha1(p):
    p = Path(p).resolve()
    st = p.stat()
    key = (str(p), st.st_size, st.st_mtime_ns)
    if key not in _SHA_CACHE:
        h = hashlib.sha1()
        with open(p, 'rb') as f:
            for chunk in iter(lambda: f.read(1 << 22), b''):
                h.update(chunk)
        _SHA_CACHE[key] = h.hexdigest()
    return _SHA_CACHE[key]


def cell_key(packets, case):
    """The packet quantities teacher.Cell uses (same parsing as teacher.Cell.__init__)."""
    ctx = json.loads((Path(packets) / case / 'FRESH_CONTEXT.json').read_text())
    smp = json.loads((Path(packets) / case / 'SAMPLE.json').read_text())
    cs = ctx['case']
    normal = None if cs.get('normal') is None else [float(Fraction(v)) for v in cs['normal']]
    return dict(n=int(ctx['n']), E=float(ctx['material']['E']), nu=float(ctx['material']['nu']),
                surface=cs.get('surface', 'P') or 'P', tau=[float(Fraction(v)) for v in cs['tau_corners']], normal=normal,
                offset=None if normal is None else float(Fraction(cs['offset'])), gamma=float(smp['gp']['gamma']))


def fingerprint(body, packets, case):
    """sha1 over everything teacher.Cell reads for K and the port set (see module docstring); 20 hex digits."""
    key = cell_key(packets, case)
    h = hashlib.sha1(json.dumps(key, sort_keys=True).encode())
    for fn in FP_FILES:
        h.update(fn.encode()); h.update(_file_sha1(Path(body) / case / fn).encode())
    h.update(_file_sha1(Path(body) / f'GP_TEMPLATES_n{key["n"]}.npz').encode())
    return h.hexdigest()[:20]


def make_plan(R, k, dedupe=True):
    d = R['hist'][k]
    cases = list(d['cases'])
    if len(cases) != len(R['positions']):
        raise ValueError(f'CASES_VS_LAYOUT {len(cases)} != {len(R["positions"])}')
    if len(set(cases)) != len(cases):
        raise ValueError('DUPLICATE_CASE_NAMES')
    body, packets = R['run'] / 'body', R['run'] / 'packets'
    fps = [fingerprint(body, packets, c) if dedupe else f'{R["name"]}.{c}' for c in cases]
    groups = {}
    for i, f in enumerate(fps):
        groups.setdefault(f, []).append(i)
    ports = {c: int(v['ports']) for c, v in d.get('fps', {}).items()}
    return dict(run=R['run'], name=R['name'], k=k, cases=cases, positions=R['positions'], fps=fps, groups=groups,
                rep={f: cases[m[0]] for f, m in groups.items()}, body=body, packets=packets, ports_recorded=ports,
                tag=f'{R["name"]}_k{k:03d}')


# ================================================================================================ worker processes
def worker_env(threads, packets=None):
    env = dict(os.environ)
    env.update(CPU_ENV)
    env.update(MKL_NUM_THREADS=str(threads), OMP_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads),
               PYTHONUNBUFFERED='1')
    if packets is not None:
        extra = [str(packets)] + [p for p in os.environ.get('OPL_PACKETS_EXTRA', '').split(':') if p and p != str(packets)]
        env['OPL_PACKETS_EXTRA'] = ':'.join(extra)
    return env


def run_pool(jobs, parallel, min_free_gib, tag, retry=True, poll=2.0):
    """jobs: list of dict(name, cmd, env, log). At most `parallel` at a time; a job starts only while MemAvailable >=
    min_free_gib (unless nothing else runs). Per job: exit code, wall seconds, peak RSS of the child (wait4 rusage).
    Failed jobs are retried once, one at a time. Returns {name: record}."""
    pending, running, res = list(jobs), {}, {}
    while pending or running:
        while pending and len(running) < max(1, parallel) and (not running or avail_gib() >= min_free_gib):
            j = pending.pop(0)
            fh = open(j['log'], 'ab')
            p = subprocess.Popen(j['cmd'], env=j['env'], cwd=str(HERE), stdout=fh, stderr=subprocess.STDOUT)
            running[p.pid] = (j, p, fh, time.perf_counter())
            log(dict(event='JOB_START', phase=tag, job=j['name'], running=len(running), pending=len(pending),
                     mem_available_gib=round(avail_gib(), 1)))
        pid, status, ru = os.wait4(-1, os.WNOHANG)
        if pid == 0:
            time.sleep(poll)
            continue
        if pid not in running:
            continue
        j, p, fh, t0 = running.pop(pid)
        fh.close()
        p.returncode = rc = os.waitstatus_to_exitcode(status)
        res[j['name']] = dict(rc=rc, seconds=time.perf_counter() - t0, peak_rss_gib=ru.ru_maxrss / 2 ** 20)
        log(dict(event='JOB_END', phase=tag, job=j['name'], **res[j['name']]))
    bad = [j for j in jobs if res[j['name']]['rc'] != 0]
    if bad and retry:
        log(dict(event='JOB_RETRY', phase=tag, jobs=[j['name'] for j in bad]))
        again = run_pool(bad, 1, 0.0, tag + '_retry', retry=False, poll=poll)
        for n_, r_ in again.items():
            res[n_] = dict(r_, retried=True, first=res[n_])
    return res


def self_cmd(kind, job_path):
    return [sys.executable, '-u', str(Path(__file__).resolve()), '--worker', kind, '--job', str(job_path)]


def _cpu_prologue():
    """The CPU environment of make_T_cpu.py / bench_cpu2.py: diag_sens first (hides the GPU when OPL_DEV=cpu, host
    defaults of polyref, no-op sync), then box_encode.dev on the host (see the module docstring)."""
    if os.environ.get('OPL_DEV') != 'cpu':
        raise RuntimeError('WORKER_NEEDS_OPL_DEV_CPU')
    import diag_sens as DS                                                              # noqa: F401  first
    import trainlib as TL
    import box_encode as BX
    BX.dev = TL.dev
    return DS


def _done(job, rec):
    rec = dict(rec, peak_rss_gib=status_gib('VmHWM'), host=os.uname().nodename)
    write_json(job['out'], rec)
    print(json.dumps(tojson(rec), default=float), flush=True)


def worker_cellinfo(job):
    """teacher.Cell + assemble(): ports, K_PP upper triplets and face-load weights of the six faces (lat_multi.from_teacher)."""
    _cpu_prologue()
    import torch
    import teacher as TE
    import lat_multi as LM
    t0 = time.perf_counter()
    C = TE.Cell(job['case'], job['body'], log=lambda s_: None)
    t1 = time.perf_counter(); C.assemble(); t2 = time.perf_counter()
    G = LM.from_teacher(C)
    with torch.no_grad():
        r, c, v = G.kpp()
    arr = dict(port_node_ids=np.asarray(C.port_node_ids), port_is_box=np.asarray(C.port_is_box),
               port_is_cut=np.asarray(C.port_is_cut), kpp_r=r.cpu().numpy(), kpp_c=c.cpu().numpy(), kpp_v=v.cpu().numpy())
    t3 = time.perf_counter()
    for ax in range(3):
        for val in (0, 1):
            w, out = G.face_w_fn(ax, float(val))
            arr[f'face_w_{ax}{val}'] = np.asarray(w, np.float64); arr[f'face_out_{ax}{val}'] = np.asarray(out, np.float64)
    t4 = time.perf_counter()
    tmp = Path(job['npz']).with_name(Path(job['npz']).name + '.part.npz')
    np.savez(tmp, **arr)
    os.replace(tmp, job['npz'])
    _done(job, dict(case=job['case'], n=int(C.n), normal=C.normal, offset=C.offset, dofs=int(C.nb), ports=int(C.np_),
                    interior=int(C.ni), elements=int(len(C.cells)), kpp_nnz_upper=int(len(arr['kpp_v'])),
                    setup_s=t1 - t0, assemble_s=t2 - t1, kpp_s=t3 - t2, face_s=t4 - t3))


def _portview_files(body, case, port_node_ids):
    """The portview directory of lattice3.dense_T / make_T_cpu.py (node-list symlinks, BOX_NODES.npy = retained nodes)."""
    pd = Path(body) / (case + '_portview')
    pd.mkdir(exist_ok=True)
    for fn in ('NODES.npy', 'CELL_INDICES.npy', 'dofs.npy', 'GP_FACES.npy'):
        if not (pd / fn).exists():
            os.symlink(Path(body) / case / fn, pd / fn)
    np.save(pd / 'BOX_NODES.npy', port_node_ids)
    return pd


def symmetrise_inplace(S, block=4096):
    """S <- (S + S^T) / 2 in blocks (no second n x n array); returns max |S - S^T| / max |S| before."""
    n = S.shape[0]
    amax, dmax = 0.0, 0.0
    for i0 in range(0, n, block):
        i1 = min(n, i0 + block)
        for j0 in range(i0, n, block):
            j1 = min(n, j0 + block)
            a = S[i0:i1, j0:j1]
            b = S[j0:j1, i0:i1]
            amax = max(amax, float(np.abs(a).max()), float(np.abs(b).max()))
            dmax = max(dmax, float(np.abs(a - b.T).max()))
            m = (a + b.T) * 0.5
            S[i0:i1, j0:j1] = m
            S[j0:j1, i0:i1] = m.T
    return dmax / amax if amax > 0 else 0.0


def worker_schur(job):
    """Dense exact condensed matrix by PARDISO's Schur-complement option (bench_cpu2.explicit_schur), verified, saved as
    make_T_cpu.py saves it."""
    DS = _cpu_prologue()
    import torch
    import teacher as TE
    import pardiso_direct as PD
    import bench_cpu2 as BC
    rec = dict(case=job['case'], route='schur')
    t0 = time.perf_counter()
    C = TE.Cell(job['case'], job['body'], log=lambda s_: None); C.assemble()
    rec['setup_assemble_s'] = time.perf_counter() - t0
    U, s = BC.full_upper(C)
    P_ = C.P.cpu().numpy()
    ns = len(P_)
    perm = np.zeros(C.nb, np.int32); perm[P_] = 1
    ip = {int(k_): int(v_) for k_, v_ in json.loads(Path(job['iparm_file']).read_text())['iparm'].items()} \
        if job.get('iparm_file') else PD.tuned_iparm()
    ip[36] = 1
    x, tried = None, []
    for variant in (ip, {**ip, 24: 0}, {**ip, 24: 0, 2: 2}):
        H = PD.Pardiso(U, 2, variant, perm=perm)
        x = np.zeros(max(C.nb, ns * ns))
        try:
            ta = H.phase(11); tf = H.phase(22, x=x)
            rec.update(analysis_s=ta, factor_schur_s=tf, pardiso=H.info(),
                       used_iparm={str(k_): int(v_) for k_, v_ in variant.items()})
            H.release()
            break
        except PD.PardisoError as e:
            tried.append(dict(iparm={str(k_): int(v_) for k_, v_ in variant.items()}, error=str(e)))
            H.release(); x = None
    rec['failed_variants'] = tried
    del U
    if x is None:
        raise RuntimeError(f'SCHUR_FAILED {job["case"]}: {tried}')
    S = x[:ns * ns].reshape(ns, ns)
    sP = s[P_]
    S /= sP[:, None]; S /= sP[None, :]                                                  # S_K = S_A / (s_P s_P^T)
    t = time.perf_counter()
    rec['asym_rel'] = symmetrise_inplace(S)
    rec['symmetrise_s'] = time.perf_counter() - t
    if not np.isfinite(S).all():
        raise RuntimeError(f'SCHUR_NOT_FINITE {job["case"]}')
    nver = min(int(job.get('verify_cols', 16)), ns)
    if nver > 0:
        t = time.perf_counter()
        DS.cpu_factor(C)
        j = np.sort(np.random.default_rng(0).choice(ns, size=nver, replace=False))
        E = torch.zeros((ns, nver), dtype=TE.dt); E[torch.as_tensor(j), torch.arange(nver)] = 1
        with torch.no_grad():
            Sj = C.apply(E).cpu().numpy()
        C._free()
        rec['verify_cols'] = int(nver)
        rec['verify_rel_err'] = float(np.linalg.norm(S[:, j] - Sj) / np.linalg.norm(Sj))
        rec['verify_s'] = time.perf_counter() - t
        if not rec['verify_rel_err'] <= float(job.get('verify_tol', 1e-8)):
            raise RuntimeError(f'SCHUR_VERIFY_FAILED {job["case"]} rel {rec["verify_rel_err"]:.3e}')
    t = time.perf_counter()
    pd = _portview_files(job['body'], job['case'], C.port_node_ids)
    tmp = pd / 'T64.npy.part'
    with open(tmp, 'wb') as f:
        np.save(f, S)
    os.replace(tmp, pd / 'T64.npy')
    rec.update(write_s=time.perf_counter() - t, ports=int(ns), dofs=int(C.nb), interior=int(C.ni),
               T_gib=8 * ns * ns / GIB, seconds=time.perf_counter() - t0)
    C._free()
    _done(job, rec)


def worker_maket(job):
    """make_T_cpu.py unchanged, after the CPU prologue (box_encode.dev on the host)."""
    _cpu_prologue()
    import io
    import contextlib
    import make_T_cpu as MT
    t0 = time.perf_counter()
    buf, old = io.StringIO(), sys.argv
    sys.argv = ['make_T_cpu.py', str(job['body']), job['case'], '--block', str(int(job.get('block', 256)))]
    try:
        with contextlib.redirect_stdout(buf):
            MT.main()
    finally:
        sys.argv = old
    out = [json.loads(line) for line in buf.getvalue().splitlines() if line.strip().startswith('{')]
    print(buf.getvalue(), end='', flush=True)
    _done(job, dict(case=job['case'], route='make_T_cpu', make_T_cpu=out[-1] if out else None,
                    seconds=time.perf_counter() - t0))


def worker_sens(job):
    """Exact sensitivities of every cell of one fingerprint group: u = E q (interior factor), s = -u^T dK/dtau_c u."""
    DS = _cpu_prologue()
    import torch
    import teacher as TE
    rec = dict(case=job['case'], mode=job['mode'], members=[m['case'] for m in job['members']])
    t0 = time.perf_counter()
    C = TE.Cell(job['case'], job['body'], log=lambda s_: None); C.assemble()
    rec['setup_assemble_s'] = time.perf_counter() - t0
    vol = float(C.M[:, 0].sum())
    dvol = None
    if job['mode'] == 'fd':
        t = time.perf_counter()
        dmc = Path(job['dm_cache']) if job.get('dm_cache') else None
        if dmc is not None and dmc.exists():                                            # same fingerprint, earlier design
            C.dM = torch.from_numpy(np.load(dmc))
            rec['dmoments_cached'] = str(dmc)
        else:
            C.dmoments()
            if dmc is not None:
                tmp = dmc.with_name(dmc.name + '.part')
                with open(tmp, 'wb') as f:
                    np.save(f, C.dM.cpu().numpy())
                os.replace(tmp, dmc)
        dvol = C.dM[:, :, 0].sum(1).cpu().numpy()
        rec['dmoments_s'] = time.perf_counter() - t
    t = time.perf_counter()
    DS.cpu_factor(C)
    rec['factor_s'] = time.perf_counter() - t
    Q = np.concatenate([np.load(m['q']) for m in job['members']], 1)
    if Q.shape[0] != C.np_:
        raise ValueError(f'Q_ROWS {Q.shape[0]} != ports {C.np_}')
    t = time.perf_counter()
    with torch.no_grad():
        Qt = torch.from_numpy(Q).to(TE.dt)
        u = C.extend(Qt)
        C._free()                                                                       # interior factor no longer needed
        eK = (u * (C.K @ u)).sum(0).cpu().numpy()
        rec['extend_s'] = time.perf_counter() - t
        t = time.perf_counter()
        if job['mode'] == 'fd':
            S = C.sens(u).cpu().numpy()                                                 # (8, members)
        else:
            import moments_ad as MA
            g = C.energy_density(u)
            e0 = torch.zeros((g.shape[0], g.shape[1], 1), dtype=g.dtype, device=g.device); e0[:, 0, 0] = 1.0
            g = torch.cat([g, e0], 2); del e0
            r_ = MA.cell_sens(C, g, batch=int(job.get('ad_batch', 2048)), skip_full=True).detach().cpu().numpy()
            S, dvol = r_[:, :-1], -r_[:, -1]
            del g
    rec['sens_s'] = time.perf_counter() - t
    eT = np.asarray([m['energy_T'] for m in job['members']], float)
    rec.update(vol=vol, dvol=dvol, per_member={m['case']: dict(s=S[:, i], energy_K=float(eK[i]), energy_T=float(eT[i]),
                                                               energy_rel_diff=float((eK[i] - eT[i]) / eT[i]))
                                               for i, m in enumerate(job['members'])},
               energy_rel_diff_max=float(np.max(np.abs((eK - eT) / eT))), seconds=time.perf_counter() - t0)
    _done(job, rec)


WORKERS = dict(cellinfo=worker_cellinfo, schur=worker_schur, maket=worker_maket, sens=worker_sens)


# ================================================================================================ T files
def t_valid(pd, port_node_ids):
    """T64.npy present, complete (memory-mappable), float64, (n, n) with n = 3 x ports, and BOX_NODES.npy = the ports."""
    pd = Path(pd)
    try:
        if not np.array_equal(np.load(pd / 'BOX_NODES.npy'), port_node_ids):
            return False
        T = np.load(pd / 'T64.npy', mmap_mode='r')
        n = 3 * len(port_node_ids)
        return T.dtype == np.float64 and T.shape == (n, n)
    except (OSError, ValueError):
        return False


class TCache:
    """fingerprint -> portview directory holding a valid T64.npy (<work>/tcache.json), shared by designs and runs."""

    def __init__(self, path):
        self.path = Path(path)
        self.d = json.loads(self.path.read_text()) if self.path.exists() else {}

    def save(self):
        write_json(self.path, self.d)

    def find(self, fp, candidates, port_node_ids):
        """A valid T for fp: the cached entry, else any candidate portview (a member cell's own)."""
        e = self.d.get(fp)
        if e and t_valid(e['portview'], port_node_ids):
            return Path(e['portview'])
        for pd in candidates:
            if t_valid(pd, port_node_ids):
                self.d[fp] = dict(self.d.get(fp, {}), portview=str(pd), found=True)
                self.save()
                return Path(pd)
        return None

    def put(self, fp, pd, rec):
        self.d[fp] = dict(portview=str(pd), record=rec)
        self.save()

    def drop(self, fp):
        e = self.d.pop(fp, None)
        self.save()
        return e


# ================================================================================================ lattice solve
class PardisoKpp:
    """Fine level 'kpp' of lat_precond (B = K_PP^-1 on the free retained DOFs) with the factorisation by MKL PARDISO
    Cholesky (mtype 2, pardiso_direct) of the Jacobi-scaled upper triplets: the same scaling and action as
    lat_precond.SparseSPD. Handed to lat_precond.Factory through its `shared` dict."""

    def __init__(self, rows, cols, vals, n, iparm=None):
        import torch
        import scipy.sparse as sp
        import pardiso_direct as PD
        r, c, v = (x.cpu().numpy() for x in (rows, cols, vals))
        dg = np.zeros(n); d = r == c
        dg[r[d]] = v[d]
        if not (dg > 0).all():
            raise ValueError('KPP_DIAGONAL_NOT_POSITIVE')
        s = 1 / np.sqrt(dg)
        U = sp.csr_matrix((v * s[r] * s[c], (r, c)), shape=(n, n))
        U.sum_duplicates(); U.sort_indices()
        if (U.tocoo().row > U.tocoo().col).any():
            raise ValueError('KPP_NOT_UPPER')
        self.P = PD.Pardiso(U, 2, PD.tuned_iparm() if iparm is None else iparm)
        ta = self.P.phase(11); tf = self.P.phase(22)
        self.s = torch.as_tensor(s, dtype=torch.float64)
        self.setup = dict(kpp_triplets_s=0.0, kpp_assemble_s=0.0, kpp_factor_s=ta + tf)
        self.backend, self.nnz_upper, self.info = 'pardiso', int(U.nnz), self.P.info()

    def __call__(self, R):
        import torch
        x = (self.s[:, None] * R).cpu().numpy()
        y = self.P.solve(x[:, 0].copy() if x.shape[1] == 1 else x)
        y = torch.as_tensor(np.ascontiguousarray(y).reshape(x.shape), dtype=torch.float64, device=R.device)
        return self.s[:, None] * y

    def free(self):
        self.P.release()


class HostDenseOp:
    """lat_hetero.DenseExactOp's product on the host: port reactions T q of a cell (T dense fp64). One object per
    fingerprint group, so that lat_multi batches the cells that share it."""

    def __init__(self, T, fp):
        self.T, self.fp = T, fp

    def apply(self, q):
        return self.T @ q.to(self.T.dtype)


def geoms_from_cellinfo(cases, fps, infos):
    """lat_multi.CellGeom per cell from the cached cell information (K_PP triplets, face weights of all faces)."""
    import torch
    import lat_multi as LM
    geoms = []
    for c, f in zip(cases, fps):
        z, meta = infos[f]
        trip = (torch.from_numpy(z['kpp_r']), torch.from_numpy(z['kpp_c']), torch.from_numpy(z['kpp_v']))

        def face_w(axis, value, z=z):
            return z[f'face_w_{int(axis)}{int(round(value))}'], float(z[f'face_out_{int(axis)}{int(round(value))}'])
        geoms.append(LM.CellGeom(c, meta['n'], z['port_node_ids'], z['port_is_box'], z['port_is_cut'],
                                 kpp_fn=(lambda t=trip: t), face_w_fn=face_w, normal=meta.get('normal'),
                                 offset=meta.get('offset')))
    return geoms


def exact_lattice_solve(cases, positions, geoms, ops, cfg, tol, maxit, prec, fine, iparm=None, loads='consistent'):
    """The exact counterpart of the lattice solve of opt_design.analyse_exact on the host. ops[i]: operator of cell i
    (cells sharing an object are batched). Returns the compliance, solve record, lattice order and port vectors."""
    import torch
    import lat_multi as LM
    import lat_precond as PR
    T = {}
    t = time.perf_counter()
    lay = {tuple(p): g for p, g in zip(positions, geoms)}
    if len(lay) != len(geoms):
        raise ValueError('DUPLICATE_POSITIONS')
    lat = LM.MultiLattice(lay, clamp=tuple(cfg['clamp']), load=tuple(cfg['load']), loads=loads, n_random=0,
                          device='cpu', max_cols=cfg.get('max_cols'), log=lambda s_: None)
    order = [g.case for g in lat.geoms]
    op_of = dict(zip(cases, ops))
    lops = [op_of[c] for c in order]
    F = lat.F[:, ['xyz'.index(cfg['load_dir'])]].contiguous()
    kpp = lat.assemble_kpp()
    T['setup_s'] = time.perf_counter() - t
    t = time.perf_counter()
    shared = {'kpp_triplets': kpp, 'kpp_triplets_s': 0.0}
    if fine == 'pardiso' and 'kpp' in prec.split(':'):
        shared['kpp'] = PardisoKpp(*kpp, lat.nfree, iparm)
    fac = PR.Factory(lat, lops, shared=shared, kpp_backend='auto', log=lambda d_: None)
    pc, pst, pdesc = fac.build(prec)
    T['precond_s'] = time.perf_counter() - t
    t = time.perf_counter()
    r = PR.solve(lat, lops, pc, F=F, tol=tol, maxit=maxit)
    T['pcg_s'] = time.perf_counter() - t
    t = time.perf_counter()
    X = r['X']
    with torch.no_grad():
        rho = F - lat.matvec(lops, X)
        C = float((F * X).sum()); tres = float(rho.norm() / F.norm())
        Q = [lat.gather(X, i) for i in range(len(order))]
        Y = lat.apply_blocks(lops, Q)
        energy = [float((q * y).sum()) for q, y in zip(Q, Y)]
    T['post_s'] = time.perf_counter() - t
    T['solve_s'] = T['pcg_s'] + T['post_s']
    fine_obj = shared.get('kpp')                                    # PardisoKpp, or lat_precond.KppFine built by Factory
    rec = dict(C=C, pcg=int(r['iterations']), residual_recursive=float(r['residual']), true_residual=tres,
               energy_sum_rel=float((sum(energy) - C) / C), history=[float(h) for h in r.get('history', [])],
               precond_setup=pst, precond=pdesc,
               fine=dict(kind=fine, backend=getattr(fine_obj, 'backend', None), nnz_upper=getattr(fine_obj, 'nnz_upper', None),
                         pardiso=getattr(fine_obj, 'info', None)),
               lattice=dict(free_dofs=int(lat.nfree), glued_dofs=int(lat.N), cells=len(order),
                            face_load_outside=[float(x) for x in lat.face_load_outside]), times=T)
    fac.free()
    if fine_obj is not None and hasattr(fine_obj, 'free'):
        fine_obj.free()
    out = dict(rec=rec, order=order, Q=[q.cpu().numpy() for q in Q], energy=energy)
    del lat, lops, X, rho, F, Q, Y, pc, fac, shared, kpp
    gc.collect()
    return out


# ================================================================================================ check quantities
def check_record(k, rec_nice, S_lay, dV_lay, vid, nv, free, tmin, tmax, C_exact, pcg, tres, times):
    """The quantities of opt_design.check() (same formulas and keys) from the exact per-cell sensitivities S_lay and
    volume derivatives dV_lay (layout order, ncell x 8); S_lay None: compliance only (gradient keys null)."""
    d = rec_nice
    out = dict(k=k, C_exact=C_exact, C_hat=d['C'], surrogate_err=(d['C'] - C_exact) / C_exact)
    if S_lay is None:
        out.update(grad_rel_err=None, grad_cos=None, comp_err_rel_to_max=None, sign_agreement=None, kkt_exact=None,
                   kkt_lambda=None, exact_pcg=pcg, exact_true_residual=tres, times=times, g_exact=None,
                   g_tilde=np.asarray(d['s_vertex'], float)[free])
        return out
    g_ex = aggregate(S_lay[:, :, None], vid, nv)[:, 0][free]
    dV = aggregate(dV_lay[:, :, None], vid, nv)[:, 0][free]
    g_ti = np.asarray(d['s_vertex'], float)[free]
    err = np.abs(g_ti - g_ex) / np.abs(g_ex).max()
    tv = np.asarray(d['tv'], float)[free]
    inner = (tv > tmin + 1e-6) & (tv < tmax - 1e-6)
    gi, vi = g_ex[inner], dV[inner]
    lam = -float(gi @ vi / (vi @ vi)) if inner.any() else 0.0
    kkt = float(np.linalg.norm(gi + lam * vi) / np.linalg.norm(gi)) if inner.any() else float('nan')
    out.update(grad_rel_err=float(np.linalg.norm(g_ti - g_ex) / np.linalg.norm(g_ex)),
               grad_cos=float(g_ti @ g_ex / (np.linalg.norm(g_ti) * np.linalg.norm(g_ex))),
               comp_err_rel_to_max=dict(median=float(np.median(err)), p95=float(np.percentile(err, 95)), max=float(err.max())),
               sign_agreement=float(np.mean(np.sign(g_ti) == np.sign(g_ex))), kkt_exact=kkt, kkt_lambda=lam,
               exact_pcg=pcg, exact_true_residual=tres, times=times, g_exact=g_ex, g_tilde=g_ti)
    return out


# ================================================================================================ orchestration
class Runner:
    def __init__(self, a):
        self.a = a
        self.work = Path(a.work).resolve()
        for sub in ('cellinfo', 'jobs', 'logs', 'dmoments'):
            (self.work / sub).mkdir(parents=True, exist_ok=True)
        self.tcache = TCache(self.work / 'tcache.json')
        self.infos = {}

    # ---------------------------------------------------------------- cellinfo
    def ensure_cellinfo(self, P):
        todo, jobs = [], []
        for f, rep in P['rep'].items():
            npz = self.work / 'cellinfo' / f'{f}.npz'
            if npz.exists() and (self.work / 'cellinfo' / f'{f}.json').exists():
                continue
            job = dict(case=rep, body=str(P['body']), npz=str(npz), out=str(self.work / 'cellinfo' / f'{f}.json'))
            jp = self.work / 'jobs' / f'cellinfo_{f}.json'; write_json(jp, job)
            jobs.append(dict(name=f'cellinfo_{f}', cmd=self_cmd('cellinfo', jp), env=worker_env(self.a.cell_threads, P['packets']),
                             log=self.work / 'logs' / f'cellinfo_{f}_{rep}.log'))
            todo.append(f)
        res = run_pool(jobs, self.a.cell_jobs, self.a.min_free_gib, 'cellinfo') if jobs else {}
        bad = [n_ for n_, r_ in res.items() if r_['rc'] != 0]
        if bad:
            raise RuntimeError(f'CELLINFO_FAILED {bad} (logs in {self.work / "logs"})')
        for f in P['rep']:
            if f not in self.infos:
                z = np.load(self.work / 'cellinfo' / f'{f}.npz')
                self.infos[f] = ({k_: z[k_] for k_ in z.files}, json.loads((self.work / 'cellinfo' / f'{f}.json').read_text()))
        return dict(new=len(todo), jobs=res)

    # ---------------------------------------------------------------- T
    def t_candidates(self, P, f):
        return [P['body'] / (P['cases'][i] + '_portview') for i in P['groups'][f]]

    def ensure_T(self, P):
        jobs, built = [], {}
        for f, rep in P['rep'].items():
            ports = self.infos[f][0]['port_node_ids']
            if self.tcache.find(f, self.t_candidates(P, f), ports) is not None:
                continue
            out = self.work / 'jobs' / f'T_{f}.out.json'
            job = dict(case=rep, body=str(P['body']), out=str(out), block=self.a.block, verify_cols=self.a.verify_cols,
                       verify_tol=self.a.verify_tol, iparm_file=str(Path(self.a.iparm_file).resolve()) if self.a.iparm_file else None)
            jp = self.work / 'jobs' / f'T_{f}.json'; write_json(jp, job)
            kind = 'schur' if self.a.t_route == 'schur' else 'maket'
            jobs.append(dict(name=f'T_{f}', cmd=self_cmd(kind, jp), env=worker_env(self.a.t_threads, P['packets']),
                             log=self.work / 'logs' / f'T_{f}_{rep}.log'))
            built[f'T_{f}'] = (f, P['body'] / (rep + '_portview'), out)
        res = run_pool(jobs, self.a.t_jobs, self.a.min_free_gib, 'T') if jobs else {}
        for name, (f, pd, out) in built.items():
            ports = self.infos[f][0]['port_node_ids']
            if res[name]['rc'] != 0 or not t_valid(pd, ports):
                raise RuntimeError(f'T_FAILED {name} rc={res[name]["rc"]} (log {self.work / "logs"})')
            rec = json.loads(Path(out).read_text()) if Path(out).exists() else {}
            self.tcache.put(f, pd, dict(rec, job=res[name]))
        return dict(new=len(jobs), jobs=res, seconds=sum(r_['seconds'] for r_ in res.values()))

    # ---------------------------------------------------------------- solve
    def solve(self, P, R, D, phases):
        import torch
        torch.set_num_threads(self.a.threads)
        need = 0
        for f in P['rep']:
            n = 3 * len(self.infos[f][0]['port_node_ids'])
            need += 8 * n * n
        avail = avail_gib()
        phases['solve_mem_plan'] = dict(T_distinct_gib=need / GIB, mem_available_gib=avail)
        if need / GIB + self.a.solve_margin_gib > avail and not self.a.no_mem_check:
            raise MemoryError(f'SOLVE_NEEDS {need / GIB:.1f} GiB of T + {self.a.solve_margin_gib} GiB margin, '
                              f'MemAvailable {avail:.1f} GiB (--no-mem-check to try anyway)')
        with Phase('load_T', phases):
            ops_by_fp = {}
            for f in P['rep']:
                ports = self.infos[f][0]['port_node_ids']
                pd = self.tcache.find(f, self.t_candidates(P, f), ports)
                if pd is None:
                    raise FileNotFoundError(f'T_MISSING {f} ({P["rep"][f]})')
                ops_by_fp[f] = HostDenseOp(torch.from_numpy(np.load(pd / 'T64.npy')), f)
        ops = [ops_by_fp[f] for f in P['fps']]
        geoms = geoms_from_cellinfo(P['cases'], P['fps'], self.infos)
        iparm = ({int(k_): int(v_) for k_, v_ in json.loads(Path(self.a.iparm_file).read_text())['iparm'].items()}
                 if self.a.iparm_file else None)
        with Phase('lattice_solve', phases):
            out = exact_lattice_solve(P['cases'], P['positions'], geoms, ops, R['cfg'], self.a.tol, self.a.maxit,
                                      self.a.prec or R['cfg']['prec'], self.a.fine, iparm)
        del ops, ops_by_fp, geoms
        gc.collect()
        (D / 'q').mkdir(exist_ok=True)
        for c, q in zip(out['order'], out['Q']):
            np.save(D / 'q' / f'{c}.npy', q)
        rec = dict(out['rec'], order=out['order'], energy_T=dict(zip(out['order'], out['energy'])))
        write_json(D / 'solve.json', rec)
        log(dict(event='SOLVED', design=P['tag'], C_exact=rec['C'], C_hat=R['hist'][P['k']]['C'], pcg=rec['pcg'],
                 true_residual=rec['true_residual'], energy_sum_rel=rec['energy_sum_rel'], times=rec['times']))
        return rec

    # ---------------------------------------------------------------- sensitivities
    def ensure_sens(self, P, D, solve_rec):
        (D / 'sens').mkdir(exist_ok=True)
        jobs = []
        for f, members in P['groups'].items():
            out = D / 'sens' / f'{f}.json'
            if out.exists():
                continue
            job = dict(case=P['rep'][f], body=str(P['body']), mode=self.a.sens, ad_batch=self.a.ad_batch, out=str(out),
                       dm_cache=str(self.work / 'dmoments' / f'{f}.npy') if self.a.dm_cache else None,
                       members=[dict(case=P['cases'][i], q=str(D / 'q' / f'{P["cases"][i]}.npy'),
                                     energy_T=solve_rec['energy_T'][P['cases'][i]]) for i in members])
            jp = D / 'sens' / f'job_{f}.json'; write_json(jp, job)
            jobs.append(dict(name=f'sens_{f}', cmd=self_cmd('sens', jp), env=worker_env(self.a.sens_threads, P['packets']),
                             log=self.work / 'logs' / f'sens_{P["tag"]}_{f}.log'))
        res = run_pool(jobs, self.a.sens_jobs, self.a.min_free_gib, 'sens') if jobs else {}
        bad = [n_ for n_, r_ in res.items() if r_['rc'] != 0]
        if bad:
            raise RuntimeError(f'SENS_FAILED {bad}')
        return dict(new=len(jobs), jobs=res, seconds=sum(r_['seconds'] for r_ in res.values()))

    # ---------------------------------------------------------------- one design
    def design(self, R, P, later_fps):
        a = self.a
        D = self.work / P['tag']
        D.mkdir(exist_ok=True)
        out_path = R['run'] / f'check_exact_{P["k"]:03d}.json'
        if out_path.exists() and not a.force and 'check' in a.phases:
            log(dict(event='SKIP_DONE', design=P['tag'], out=out_path))
            return
        write_json(D / 'plan.json', dict(run=P['run'], k=P['k'], cases=P['cases'], positions=P['positions'], fps=P['fps'],
                                         groups=P['groups'], rep=P['rep'], layout=R['layout_path'], cfg=R['cfg']))
        pf = D / 'phases.json'
        phases = json.loads(pf.read_text()) if pf.exists() else {}
        log(dict(event='DESIGN', design=P['tag'], cells=len(P['cases']), distinct=len(P['groups']), cfg=R['cfg']))
        if 'cellinfo' in a.phases or 'T' in a.phases or 'solve' in a.phases or 'sens' in a.phases:
            with Phase('cellinfo', phases):
                phases['cellinfo_jobs'] = self.ensure_cellinfo(P)
        if 'T' in a.phases:
            with Phase('T', phases):
                phases['T_jobs'] = self.ensure_T(P)
            write_json(pf, phases)
        solve_rec = None
        if 'solve' in a.phases:
            if (D / 'solve.json').exists() and not a.force:
                solve_rec = json.loads((D / 'solve.json').read_text())
            else:
                solve_rec = self.solve(P, R, D, phases)
            write_json(pf, phases)
            if a.delete_T:
                for f in P['rep']:
                    if f in later_fps:
                        continue
                    e = self.tcache.drop(f)
                    pds = {Path(e['portview'])} if e else set()
                    pds |= set(self.t_candidates(P, f))
                    for pd in pds:
                        if (pd / 'T64.npy').exists():
                            (pd / 'T64.npy').unlink()
                            log(dict(event='DELETE_T', fp=f, path=pd / 'T64.npy'))
        elif (D / 'solve.json').exists():
            solve_rec = json.loads((D / 'solve.json').read_text())
        if 'sens' in a.phases and not a.no_sens:
            if solve_rec is None:
                raise RuntimeError(f'SENS_NEEDS_SOLVE {P["tag"]}')
            if a.force:
                for p in (D / 'sens').glob('*.json') if (D / 'sens').exists() else []:
                    if not p.name.startswith('job_'):
                        p.unlink()
            with Phase('sens', phases):
                phases['sens_jobs'] = self.ensure_sens(P, D, solve_rec)
            write_json(pf, phases)
            if a.delete_T:
                for f in P['groups']:
                    dmc = self.work / 'dmoments' / f'{f}.npy'
                    if f not in later_fps and dmc.exists():
                        dmc.unlink()
        if 'check' in a.phases:
            if solve_rec is None:
                raise RuntimeError(f'CHECK_NEEDS_SOLVE {P["tag"]}')
            self.check(R, P, D, solve_rec, phases, out_path)

    def check(self, R, P, D, solve_rec, phases, out_path):
        a = self.a
        d = R['hist'][P['k']]
        vid, fixed = R['vid'], R['fixed']
        nv = len(fixed); free = np.flatnonzero(~fixed)
        if sorted(solve_rec['order']) != sorted(P['cases']):
            raise ValueError(f'SOLVE_CASES_DIFFER {P["tag"]}')
        S_lay = dV_lay = None                                                           # layout order (rows of vid)
        vol = np.full(len(P['cases']), np.nan)
        cells, emax = {}, 0.0
        if not a.no_sens:
            S_lay = np.zeros((len(P['cases']), 8)); dV_lay = np.zeros((len(P['cases']), 8))
            for f, members in P['groups'].items():
                s = json.loads((D / 'sens' / f'{f}.json').read_text())
                for i in members:
                    c = P['cases'][i]
                    pm = s['per_member'][c]
                    S_lay[i] = pm['s']; dV_lay[i] = s['dvol']; vol[i] = s['vol']
                    cells[c] = dict(fp=f, s=pm['s'], energy_rel_diff=pm['energy_rel_diff'])
                    emax = max(emax, abs(pm['energy_rel_diff']))
        times = dict(make_T_s=phases.get('T', {}).get('seconds'), make_T_new=phases.get('T_jobs', {}).get('new'),
                     cellinfo_s=phases.get('cellinfo', {}).get('seconds'),
                     load_T_s=phases.get('load_T', {}).get('seconds'),
                     setup_s=solve_rec['times'].get('setup_s'), precond_s=solve_rec['times'].get('precond_s'),
                     solve_s=solve_rec['times'].get('solve_s'), sens_s=phases.get('sens', {}).get('seconds'))
        out = check_record(P['k'], d, S_lay, dV_lay, vid, nv, free, R['cfg']['tmin'], R['cfg']['tmax'], solve_rec['C'],
                           solve_rec['pcg'], solve_rec['true_residual'], times)
        s_cell_nice = np.asarray(d.get('s_cell'), float) if d.get('s_cell') is not None else None
        out['cpu'] = dict(
            script='exact_check_cpu.py', run=str(R['run']), design=P['tag'], cases=P['cases'], layout=str(R['layout_path']),
            cfg=R['cfg'], tol=a.tol, prec=a.prec or R['cfg']['prec'], fine=a.fine, t_route=a.t_route, sens=None if a.no_sens else a.sens,
            distinct_cells=len(P['groups']), groups={f: [P['cases'][i] for i in m] for f, m in P['groups'].items()},
            pcg_residual_recursive=solve_rec['residual_recursive'], energy_sum_rel=solve_rec['energy_sum_rel'],
            energy_T_vs_K_rel_max=emax if not a.no_sens else None, lattice=solve_rec['lattice'],
            V_exact_cells=float(np.nansum(vol)) if not a.no_sens else None, V_nice=d.get('V'),
            s_cell_exact=S_lay, s_cell_nice=s_cell_nice, phases=phases, solve_fine=solve_rec.get('fine'),
            precond=solve_rec.get('precond'))
        write_json(out_path, out)
        write_json(D / out_path.name, out)
        log({k_: v for k_, v in out.items() if k_ not in ('g_exact', 'g_tilde', 'cpu')})


def plan_all(a):
    runs, designs = {}, []
    for run, toks in a.specs:
        key = str(run.resolve())
        if key not in runs:
            runs[key] = load_run(run, a.layout if len(a.specs) == 1 else None)
        R = runs[key]
        for k in resolve_iters(toks, R['hist'].keys()):
            designs.append((R, make_plan(R, k, dedupe=not a.no_dedupe)))
    return designs


def env_record():
    r = dict(python=sys.version.split()[0], numpy=np.__version__, host=os.uname().nodename, ncpu=ncpu(),
             mem_total_gib=meminfo_gib('MemTotal'), mem_available_gib=meminfo_gib(), avail_gib_cgroup=avail_gib(),
             env={k: os.environ.get(k) for k in ('MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'PYPARDISO_MKL_RT', 'LD_LIBRARY_PATH',
                                                 'PYTHONPATH', 'OPL_DEV', 'OPL_GP_CACHE', 'OPL_COARSE_SPARSE')})
    try:
        for line in open('/proc/cpuinfo'):
            if line.startswith('model name'):
                r['cpu_model'] = line.split(':', 1)[1].strip(); break
    except OSError:
        pass
    for mod in ('scipy', 'torch'):
        try:
            r[mod] = __import__(mod).__version__
        except Exception as e:                                                          # noqa: BLE001
            r[mod] = f'unavailable: {e!r}'[:120]
    try:
        import pardiso_direct as PD
        r['mkl'] = PD.mkl_version()
    except Exception as e:                                                              # noqa: BLE001
        r['mkl'] = f'unavailable: {e!r}'[:120]
    try:
        import importlib.metadata as im
        r['pypardiso'] = im.version('pypardiso')
    except Exception as e:                                                              # noqa: BLE001
        r['pypardiso'] = f'unavailable: {e!r}'[:120]
    return r


def build_parser():
    n = ncpu()
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('specs', nargs='*', type=parse_spec, help='<run_dir>:<iters>, iters = comma list of k / last / mid')
    ap.add_argument('--work', default=None, help='work directory (default: <parent of the first run>/exact_cpu)')
    ap.add_argument('--layout', default=None, help='layout JSON (single run only; default <run>/layout.json, then meta.json)')
    ap.add_argument('--phases', default=','.join(PHASES), type=lambda s: [p for p in s.split(',') if p])
    ap.add_argument('--t-route', default='make_T_cpu', choices=['make_T_cpu', 'schur'])
    ap.add_argument('--block', type=int, default=256, help='make_T_cpu.py --block (columns per interior solve)')
    ap.add_argument('--verify-cols', type=int, default=16, help='schur route: columns checked against C.apply (0: none)')
    ap.add_argument('--verify-tol', type=float, default=1e-8)
    ap.add_argument('--iparm-file', default=None, help='PARDISO settings {"iparm": {...}} (e.g. iparm_tuned.json of the CPU '
                    'runs); default pardiso_direct.tuned_iparm()')
    ap.add_argument('--threads', type=int, default=n, help='torch / MKL threads of this process (solve)')
    ap.add_argument('--t-jobs', type=int, default=1); ap.add_argument('--t-threads', type=int, default=min(n, 16))
    ap.add_argument('--cell-jobs', type=int, default=4); ap.add_argument('--cell-threads', type=int, default=max(1, min(8, n // 4)))
    ap.add_argument('--sens-jobs', type=int, default=4); ap.add_argument('--sens-threads', type=int, default=max(1, min(16, n // 4)))
    ap.add_argument('--min-free-gib', type=float, default=48.0, help='start another worker only while MemAvailable >= this')
    ap.add_argument('--solve-margin-gib', type=float, default=24.0, help='MemAvailable needed beyond the T of a design')
    ap.add_argument('--no-mem-check', action='store_true')
    ap.add_argument('--tol', type=float, default=1e-10); ap.add_argument('--maxit', type=int, default=3000)
    ap.add_argument('--prec', default=None, help="lattice preconditioner (default: the run's --prec, bnn:kpp:q1r)")
    ap.add_argument('--fine', default='pardiso', choices=['pardiso', 'factory'])
    ap.add_argument('--sens', default='fd', choices=['fd', 'ad']); ap.add_argument('--ad-batch', type=int, default=2048)
    ap.add_argument('--no-sens', action='store_true', help='exact compliance only (gradient keys null)')
    ap.add_argument('--no-dm-cache', dest='dm_cache', action='store_false',
                    help='do not keep dM/dtau per fingerprint in <work>/dmoments (reused when a later design has the same cell)')
    ap.add_argument('--no-dedupe', action='store_true')
    ap.add_argument('--delete-T', action='store_true', help='remove T64.npy after the last solve of this invocation that uses it')
    ap.add_argument('--dry-run', action='store_true', help='print the plan (cells, groups, T sizes, T present) and exit')
    ap.add_argument('--force', action='store_true', help='redo solve, sensitivities and check of the requested designs')
    ap.add_argument('--self-test', action='store_true')
    ap.add_argument('--make-t-check', default=None, metavar='BODY:CASES',
                    help='run make_T_cpu.py --check on existing T64.npy files (CPU prologue applied) and exit')
    ap.add_argument('--worker', default=None, choices=sorted(WORKERS)); ap.add_argument('--job', default=None)
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    if a.worker:
        WORKERS[a.worker](json.loads(Path(a.job).read_text()))
        return 0
    if a.self_test:
        return self_test()
    if a.make_t_check:
        return make_t_check(a.make_t_check)
    if not a.specs:
        build_parser().error('no <run_dir>:<iters> given')
    bad = [p for p in a.phases if p not in PHASES]
    if bad:
        build_parser().error(f'unknown phases {bad}')
    os.environ.update(CPU_ENV)
    os.environ['MKL_NUM_THREADS'] = os.environ['OMP_NUM_THREADS'] = str(a.threads)
    if a.work is None:
        a.work = str(a.specs[0][0].resolve().parent / 'exact_cpu')
    designs = plan_all(a)
    log(dict(event='PLAN', designs=[P['tag'] for _, P in designs], work=a.work))
    if a.dry_run:
        tc = TCache(Path(a.work) / 'tcache.json')                                        # read only (never saved here)
        seen = set()
        for R, P in designs:
            rows = []
            for f, m in P['groups'].items():
                c = P['rep'][f]
                pr = P['ports_recorded'].get(c)
                pd = P['body'] / (c + '_portview')
                have = (f in tc.d) or (pd / 'T64.npy').exists()
                rows.append(dict(fp=f, rep=c, cells=len(m), ports_recorded=pr, T_gib=None if pr is None else 8 * pr * pr / GIB,
                                 T_present=have, shared_with_earlier=f in seen))
            seen |= set(P['groups'])
            tot = sum(r_['T_gib'] or 0 for r_ in rows)
            log(dict(event='DRY', design=P['tag'], cells=len(P['cases']), distinct=len(rows), T_distinct_gib=tot,
                     T_to_build=sum(1 for r_ in rows if not r_['T_present'] and not r_['shared_with_earlier']),
                     C_hat=R['hist'][P['k']]['C'], cfg=R['cfg'], groups=rows))
        return 0
    Path(a.work).mkdir(parents=True, exist_ok=True)
    write_json(Path(a.work) / f'env_{time.strftime("%Y%m%d_%H%M%S")}.json', dict(env_record(), argv=sys.argv))
    rn = Runner(a)
    for j, (R, P) in enumerate(designs):
        later = set()
        for _, P2 in designs[j + 1:]:
            later |= set(P2['groups'])
        t = time.perf_counter()
        rn.design(R, P, later)
        log(dict(event='DESIGN_DONE', design=P['tag'], seconds=time.perf_counter() - t))
    return 0


def make_t_check(spec):
    """make_T_cpu.py --check (32 random columns of C.apply against an existing T64.npy) in the CPU environment of the
    workers; spec '<body_dir>:<case>[,<case>...]', packets taken from <body_dir>/../packets."""
    body, cases = spec.rsplit(':', 1)
    body = Path(body).resolve()
    os.environ.update(CPU_ENV)
    extra = [str(body.parent / 'packets')] + [q for q in os.environ.get('OPL_PACKETS_EXTRA', '').split(':') if q]
    os.environ['OPL_PACKETS_EXTRA'] = ':'.join(extra)
    _cpu_prologue()
    import make_T_cpu as MT
    old = sys.argv
    sys.argv = ['make_T_cpu.py', str(body), cases, '--check']
    try:
        MT.main()
    finally:
        sys.argv = old
    return 0



# ================================================================================================ self-test
def _extract(path, names, ns):
    """Define the named top-level functions of a source file in ns without importing the file (its imports may need a
    GPU); used to compare against the original code."""
    import ast
    tree = ast.parse(Path(path).read_text())
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if len(body) != len(names):
        raise LookupError(f'{path}: {names}')
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), 'exec'), ns)
    return [ns[n] for n in names]


def _toy_run(root, rng, ncell_xy=(3, 2), k=4):
    """A toy run directory (layout, meta.json, history.jsonl) with random sensitivities; returns what check() needs."""
    import types
    pos = [(i, j, 0) for i in range(ncell_xy[0]) for j in range(ncell_xy[1])]
    cube = [((c >> 2) & 1, (c >> 1) & 1, c & 1) for c in range(8)]                    # element_polyref.CUBE order
    keys, vid = {}, np.zeros((len(pos), 8), np.int64)
    for i, p_ in enumerate(pos):
        for c in range(8):
            v = tuple(a + b for a, b in zip(p_, cube[c]))
            vid[i, c] = keys.setdefault(v, len(keys))
    nv = len(keys)
    vk = np.asarray(list(keys))
    fixed = vk[:, 0] == vk[:, 0].max()
    cases = [f'toy_{i}{j}{z}_o{k:03d}' for i, j, z in pos]
    tv = rng.uniform(0.18, 0.69, nv); tv[:3] = 0.18; tv[3] = 0.69                      # some at the bounds
    rec = dict(k=k, C=1.2345, cases=cases, tv=tv.tolist(), s_vertex=(-rng.uniform(0.5, 2, nv)).tolist(),
               s_cell=np.zeros((len(pos), 8)).tolist())
    (root / 'history.jsonl').write_text(json.dumps(dict(rec, k=k - 1, C=1.3)) + '\n' + json.dumps(rec) + '\n')
    lay = dict(cells=[dict(case=c.rsplit('_o', 1)[0], position=list(p_)) for c, p_ in zip(cases, pos)])
    (root / 'layout.json').write_text(json.dumps(lay))
    (root / 'meta.json').write_text(json.dumps(dict(vid=vid.tolist(), fixed=fixed.tolist(), layout=str(root / 'layout.json'),
                                                    args=dict(clamp='x,min', load='x,max', load_dir='y', tmin=0.18, tmax=0.69))))
    S = -rng.uniform(0.1, 1.0, (len(pos), 8)); dV = rng.uniform(0.01, 0.1, (len(pos), 8))
    return types.SimpleNamespace(pos=pos, vid=vid, nv=nv, fixed=fixed, cases=cases, rec=rec, S=S, dV=dV, k=k)


def self_test():
    """Pure-Python parts (no data): spec parsing, aggregation and check quantities against the original opt_design.check /
    r1x3_common.aggregate code, fingerprints, symmetrisation, T-file validation, the worker pool; and, where torch,
    lat_multi and lat_precond import, the host lattice solve on synthetic cells against a dense solve (with the PARDISO
    fine level when libmkl_rt loads)."""
    import tempfile
    import types
    results, t00 = {}, time.perf_counter()

    def ok(name, **kw):
        results[name] = kw
        print(f'PASS {name} ' + json.dumps(tojson(kw), default=float)[:400], flush=True)

    rng = np.random.default_rng(0)
    # ------------------------------------------------------------ specs / iterations / parser
    assert parse_spec('/a/b:0,mid,last') == (Path('/a/b'), ['0', 'mid', 'last'])
    assert parse_spec('rel/run:12')[1] == ['12']
    for bad in ('nocolon', '/a:', '/a:x1', ':3'):
        try:
            parse_spec(bad)
        except argparse.ArgumentTypeError:
            continue
        raise AssertionError(bad)
    assert resolve_iters(['0', 'mid', 'last'], range(23)) == [0, 11, 22]
    assert resolve_iters(['last', '29', 'mid', '0'], range(30)) == [29, 14, 0]
    try:
        resolve_iters(['40'], range(30)); raise AssertionError('missing k accepted')
    except ValueError:
        pass
    a = build_parser().parse_args(['/r/plateB1:0,last', 'x/y:3', '--t-route', 'schur', '--phases', 'T,solve', '--delete-T'])
    assert a.specs[0] == (Path('/r/plateB1'), ['0', 'last']) and a.t_route == 'schur' and a.phases == ['T', 'solve'] and a.delete_T
    assert a.tol == 1e-10 and a.fine == 'pardiso' and a.sens == 'fd' and not a.no_dedupe
    ok('specs_parser')
    # ------------------------------------------------------------ aggregation and check() quantities vs the original code
    ns = dict(np=np)
    agg_ref, = _extract(HERE / 'r1x3_common.py', ['aggregate'], ns)
    G = rng.standard_normal((6, 8, 3)); vid = rng.integers(0, 17, (6, 8))
    assert np.array_equal(aggregate(G, vid, 17), agg_ref(G, vid, 17))
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        toy = _toy_run(root, rng)
        order = toy.cases[::-1]                                                          # lattice order != layout order
        idx = [toy.cases.index(c) for c in order]
        res = dict(C=1.2340, S=toy.S[idx], order=order, vol=np.ones(len(order)), dvol=toy.dV[idx], pcg=321,
                   true_residual=9e-11, times=dict(solve_s=1.0))
        torch_stub = sys.modules.get('torch') or types.SimpleNamespace(is_tensor=lambda x: False)
        rc_ns = dict(np=np, torch=torch_stub)
        _extract(HERE / 'r1x3_common.py', ['aggregate', 'tojson'], rc_ns)
        ns_c = dict(json=json, np=np, Path=Path, ROOT=root, HIST=root / 'history.jsonl',
                    A=types.SimpleNamespace(layout=str(root / 'layout.json'), tmin=0.18, tmax=0.69),
                    RC=types.SimpleNamespace(aggregate=rc_ns['aggregate'], tojson=rc_ns['tojson']),
                    analyse_exact=lambda cases, positions: res, log=lambda d_: None)
        check_ref, = _extract(HERE / 'opt_design.py', ['check'], ns_c)
        check_ref([toy.k])
        ref = json.loads((root / f'check_{toy.k:03d}.json').read_text())
        R = load_run(root)
        mine = check_record(toy.k, R['hist'][toy.k], toy.S, toy.dV, R['vid'], len(R['fixed']), np.flatnonzero(~R['fixed']),
                            R['cfg']['tmin'], R['cfg']['tmax'], res['C'], res['pcg'], res['true_residual'], res['times'])
        mine = json.loads(json.dumps(tojson(mine), default=float))
        assert set(ref) == set(mine), (set(ref) ^ set(mine))
        diff = 0.0
        for key in ref:
            a_, b_ = np.asarray(ref[key] if not isinstance(ref[key], dict) else list(ref[key].values()), dtype=object), \
                np.asarray(mine[key] if not isinstance(mine[key], dict) else list(mine[key].values()), dtype=object)
            if key == 'times':
                continue
            a_, b_ = np.asarray(a_, float), np.asarray(b_, float)
            diff = max(diff, float(np.max(np.abs(a_ - b_))) if a_.size else 0.0)
        assert diff == 0.0, diff
        assert R['cfg']['clamp'] == ('x', 'min') and R['cfg']['load_dir'] == 'y'
        # compliance-only record keeps the keys
        co = check_record(toy.k, R['hist'][toy.k], None, None, R['vid'], len(R['fixed']), np.flatnonzero(~R['fixed']),
                          0.18, 0.69, res['C'], 1, 1e-11, {})
        assert set(co) == set(ref)
        ok('check_quantities_vs_opt_design_check', keys=sorted(ref), max_abs_diff=diff, kkt=mine['kkt_exact'])
    # ------------------------------------------------------------ fingerprints
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        ctx = dict(n=4, material=dict(E=1.0, nu=0.3), case=dict(tau_corners=['0.4'] * 8, normal=['1', '0', '0'], offset='1',
                                                                   kind='FULL', case_id='x'))
        for c in ('a_o000', 'b_o000', 'c_o000'):
            (td / 'packets' / c).mkdir(parents=True); (td / 'body' / c).mkdir(parents=True)
            cx = json.loads(json.dumps(ctx)); cx['case']['case_id'] = c
            if c == 'c_o000':
                cx['case']['tau_corners'][3] = '0.400000000001'
            (td / 'packets' / c / 'FRESH_CONTEXT.json').write_text(json.dumps(cx))
            (td / 'packets' / c / 'SAMPLE.json').write_text(json.dumps(dict(gp=dict(gamma=1e-4))))
            for fn in FP_FILES:
                np.save(td / 'body' / c / fn, np.arange(10))
        np.savez(td / 'body' / 'GP_TEMPLATES_n4.npz', a=np.ones(3))
        fa, fb, fc = (fingerprint(td / 'body', td / 'packets', c) for c in ('a_o000', 'b_o000', 'c_o000'))
        assert fa == fb and fa != fc                                                     # case id ignored, tau not
        np.save(td / 'body' / 'b_o000' / 'dofs.npy', np.arange(11))
        assert fingerprint(td / 'body', td / 'packets', 'b_o000') != fa                  # body arrays enter
        ok('fingerprint')
    # ------------------------------------------------------------ symmetrisation, T validation
    S = rng.standard_normal((1000, 1000)); S0 = S.copy()
    asym = symmetrise_inplace(S, block=128)
    assert np.array_equal(S, (S0 + S0.T) * 0.5) and abs(asym - np.abs(S0 - S0.T).max() / np.abs(S0).max()) < 1e-15
    with tempfile.TemporaryDirectory() as td:
        pd = Path(td); ports = np.arange(5) * 7
        np.save(pd / 'BOX_NODES.npy', ports); np.save(pd / 'T64.npy', np.eye(15))
        assert t_valid(pd, ports) and not t_valid(pd, ports + 1)
        raw = (pd / 'T64.npy').read_bytes(); (pd / 'T64.npy').write_bytes(raw[:-8])
        assert not t_valid(pd, ports)                                                    # truncated file
        np.save(pd / 'T64.npy', np.eye(15, dtype=np.float32))
        assert not t_valid(pd, ports)
        tc = TCache(pd / 'tc.json'); np.save(pd / 'T64.npy', np.eye(15))
        assert tc.find('f', [pd], ports) == pd and json.loads((pd / 'tc.json').read_text())['f']['portview'] == str(pd)
    ok('symmetrise_and_T_validation', asym=asym)
    # ------------------------------------------------------------ worker pool
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        mk = lambda n_, code: dict(name=n_, cmd=[sys.executable, '-c', code], env=dict(os.environ), log=td / f'{n_}.log')
        res = run_pool([mk('ok1', 'pass'), mk('big', 'import numpy as n; a = n.ones(40_000_000); print(a.sum())'),
                        mk('bad', 'import sys; sys.exit(3)'), mk('ok2', 'print(1)')], 2, 0.0, 'selftest', poll=0.05)
        assert res['ok1']['rc'] == 0 and res['ok2']['rc'] == 0 and res['big']['rc'] == 0
        assert res['bad']['rc'] == 3 and res['bad'].get('retried') and res['bad']['first']['rc'] == 3
        assert res['big']['peak_rss_gib'] > 0.25 > res['ok1']['peak_rss_gib']
        ok('run_pool', big_peak_gib=res['big']['peak_rss_gib'], small_peak_gib=res['ok1']['peak_rss_gib'])
    # ------------------------------------------------------------ host lattice solve on synthetic cells
    try:
        import torch
        import t_lat_precond as TP
    except Exception as e:                                                              # noqa: BLE001
        print(f'SKIP synthetic lattice solve (torch / lat_multi / lat_precond not importable: {e!r})', flush=True)
        return 0
    kinds = [TP.synth_cell(4, s, 1.0, n_priv=2, n_box_cut=3) for s in (0, 1)]
    infos, cases, positions, fps = {}, [], [], []
    for f, (G, op, _) in zip(('fpA', 'fpB'), kinds):
        r_, c_, v_ = G.kpp()
        z = dict(port_node_ids=G.port_node_ids, port_is_box=~G.priv, port_is_cut=G.priv, kpp_r=r_.numpy(), kpp_c=c_.numpy(),
                 kpp_v=v_.numpy())
        for ax in range(3):
            for val in (0, 1):
                on = G.grid[:, ax] == val * 2 * G.n
                z[f'face_w_{ax}{val}'] = on.astype(float) * (1 + 0.1 * np.arange(len(on)) / len(on))
                z[f'face_out_{ax}{val}'] = np.float64(0.0)
        infos[f] = (z, dict(n=G.n, normal=None, offset=None))
    for p_ in np.ndindex(3, 2, 1):
        f = ('fpA', 'fpB')[sum(p_) % 2]
        cases.append(f'syn_{"".join(map(str, p_))}'); positions.append(p_); fps.append(f)

    class Counting(HostDenseOp):
        def apply(self, q):
            self.calls = getattr(self, 'calls', 0) + 1
            return super().apply(q)
    ops_fp = {f: Counting(kinds[j][1].S.clone(), f) for j, f in enumerate(('fpA', 'fpB'))}
    ops = [ops_fp[f] for f in fps]
    cfg = dict(clamp=('x', 'min'), load=('x', 'max'), load_dir='y', max_cols=16)
    import lat_multi as LM
    fines = ['factory']
    try:
        import pardiso_direct as PD
        PD.lib()
        fines.append('pardiso')
    except Exception as e:                                                              # noqa: BLE001
        print(f'NOTE PARDISO fine level not tested (libmkl_rt not loadable: {e!r})', flush=True)
    for fine in fines:
        for o in ops_fp.values():
            o.calls = 0
        geoms = geoms_from_cellinfo(cases, fps, infos)
        out = exact_lattice_solve(cases, positions, geoms, ops, cfg, 1e-12, 3000, 'bnn:kpp:q1r', fine)
        lat = LM.MultiLattice({tuple(p_): g for p_, g in zip(positions, geoms_from_cellinfo(cases, fps, infos))},
                              clamp=('x', 'min'), load=('x', 'max'), loads='consistent', n_random=0, device='cpu')
        A = TP.dense_A(lat, [ops_fp[f] for f in fps])
        F = lat.F[:, [1]]
        X = torch.linalg.solve(A, F)
        Cd = float((F * X).sum())
        qerr = max(float(np.linalg.norm(q - lat.gather(X, i).numpy()) / np.linalg.norm(lat.gather(X, i).numpy()))
                   for i, q in enumerate(out['Q']))
        eerr = max(abs(e - float((lat.gather(X, i) * (ops[i].T @ lat.gather(X, i))).sum())) / abs(e)
                   for i, e in enumerate(out['energy']))
        rel = abs(out['rec']['C'] - Cd) / Cd
        assert out['order'] == cases and rel < 1e-10 and qerr < 1e-8 and eerr < 1e-8, (rel, qerr, eerr)
        assert out['rec']['true_residual'] < 1e-11 and abs(out['rec']['energy_sum_rel']) < 1e-10
        assert out['rec']['fine']['backend'] == ('pardiso' if fine == 'pardiso' else out['rec']['fine']['backend'])
        results.setdefault('pcg', {})[fine] = out['rec']['pcg']
        ok(f'synthetic_lattice_solve_{fine}', C=out['rec']['C'], C_dense=Cd, rel=rel, q_rel=qerr, pcg=out['rec']['pcg'],
           free=out['rec']['lattice']['free_dofs'], backend=out['rec']['fine']['backend'],
           apply_calls={f: o.calls for f, o in ops_fp.items()})
    if len(results['pcg']) == 2:
        assert abs(results['pcg']['factory'] - results['pcg']['pardiso']) <= 2, results['pcg']
    print(f'ALL PASS in {time.perf_counter() - t00:.1f}s', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
