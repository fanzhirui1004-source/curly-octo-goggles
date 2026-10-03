"""Revision (Table 5 single-core column, part 2): the whole lattice solved by SciPy's SuperLU (scipy.sparse.linalg.splu,
the solver behind spsolve; sequential, single thread) on the full symmetric, UNSCALED global matrix in CSC format; factor once,
then solve all six loads (factor and solve timed separately). NEW script. Variants (--variants):
  colamd  splu defaults = spsolve's defaults: permc_spec='COLAMD', partial pivoting (diag_pivot_thresh default)
  mmd     permc_spec='MMD_AT_PLUS_A' (symmetric minimum-degree ordering of A^T + A) with diag_pivot_thresh=0 and
          SymmetricMode=True (SciPy's recommended setting for symmetric positive definite matrices; keeps the ordering)
Parent (needs the CPU torch environment): cells (host front end), lattice (clamp y = min, consistent y = max + 3 random loads,
as lat_direct_cpu.py), global upper matrix (lat_direct_cpu.global_matrix) -> full symmetric CSC, saved with F to
<out_dir>/<layout>_K.npz; the cells are released. Then a CHILD process (numpy / scipy only) is started with
RLIMIT_AS = --as-gib (address-space limit: allocations beyond it fail inside the solver -> MemoryError) and a wall limit
--wall-s enforced by the child itself (signal.alarm; SIGALRM's default action ends the child). A memory watchdog thread in
the child also ends it (exit code 3, record written first) when its RSS exceeds --watch-gib. The parent records the child's
return code, the failure mode (ok / MemoryError / other exception / wall limit / watchdog), wall time and the child's peak RSS
(RUSAGE_CHILDREN: the maximum over the children so far; a child that exits normally also records its own); on success compliance per load, relative residual, and the relative compliance difference to --ref.
Usage: lat_scipy_cpu.py <out.json> <layout.json> [--body /root/autodl-tmp/OPL/S4/body] [--as-gib 70] [--watch-gib 70]
       [--wall-s 7200] [--ref f1.json,...] [--keep-npz-dir d] [--variants colamd,mmd]
       lat_scipy_cpu.py --child <npz> <child_out.json> <wall_s> <watch_gib> <variant>
       lat_scipy_cpu.py --selftest
Threads: OMP_NUM_THREADS = MKL_NUM_THREADS = OPENBLAS_NUM_THREADS = 1 (set by the chain)."""
import os, sys, json, time, argparse, gc, resource, signal, subprocess, threading
from pathlib import Path
import numpy as np

GIB = 2 ** 30


def _rss_gib():
    for line in open('/proc/self/status'):
        if line.startswith('VmRSS:'):
            return int(line.split()[1]) / 2 ** 20
    return float('nan')


VARIANTS = dict(colamd=dict(permc_spec='COLAMD'),
                mmd=dict(permc_spec='MMD_AT_PLUS_A', diag_pivot_thresh=0.0, options=dict(SymmetricMode=True)))


def child(npz, out, wall_s, watch_gib, variant='colamd'):
    import scipy, scipy.sparse as sp, scipy.sparse.linalg as sla
    rec = dict(status='started', machine=__import__('platform').node(), scipy=scipy.__version__, pid=os.getpid(), rlimit_as=resource.getrlimit(resource.RLIMIT_AS),
               variant=variant, splu_kwargs=VARIANTS[variant],
               solver=f'SuperLU bundled with SciPy {scipy.__version__} (scipy.sparse.linalg.splu; sequential, single thread)')
    save = lambda: Path(out).write_text(json.dumps(rec, indent=1, default=float))
    t = time.perf_counter()
    z = np.load(npz)
    K = sp.csc_matrix((z['data'], z['indices'], z['indptr']), shape=tuple(z['shape']))
    B = z['B']; nF = int(z['nF'])
    rec.update(load_s=time.perf_counter() - t, n=int(K.shape[0]), nnz=int(K.nnz), rss_after_load_GiB=_rss_gib())
    save()

    def watchdog():
        while True:
            r_ = _rss_gib()
            if r_ > watch_gib:
                rec.update(status='watchdog', detail=f'RSS {r_:.1f} GiB > {watch_gib} GiB', seconds=time.perf_counter() - t0)
                save(); os._exit(3)
            time.sleep(1.0)
    t0 = time.perf_counter()
    threading.Thread(target=watchdog, daemon=True).start()
    rec['status'] = 'solving'; rec['wall_limit_s'] = wall_s; save()
    signal.alarm(int(wall_s))                                                       # default action: the child ends itself
    try:
        rec['status'] = 'factorising'; save()
        lu = sla.splu(K, **VARIANTS[variant])
        rec['factor_s'] = time.perf_counter() - t0
        rec['nnz_L'] = int(lu.L.nnz); rec['nnz_U'] = int(lu.U.nnz)
        rec['LU_GiB'] = (lu.L.nnz + lu.U.nnz) * 12 / GIB
        rec['status'] = 'solving'; save()
        t1 = time.perf_counter()
        X = lu.solve(B)
        rec['solve_s'] = time.perf_counter() - t1
        rec['seconds'] = time.perf_counter() - t0
        signal.alarm(0)
        X = np.asarray(X).reshape(B.shape)
        R = K @ X - B
        rec['rel_residual'] = [float(np.linalg.norm(R[:, j]) / np.linalg.norm(B[:, j])) for j in range(B.shape[1])]
        rec['compliance'] = [float(B[:nF, j] @ X[:nF, j]) for j in range(B.shape[1])]
        rec['status'] = 'ok'
    except MemoryError as e:
        rec.update(status='MemoryError', detail=repr(e)[:300], seconds=time.perf_counter() - t0)
    except Exception as e:                                                          # noqa: BLE001
        rec.update(status='exception', detail=repr(e)[:300], seconds=time.perf_counter() - t0)
    rec['peak_rss_GiB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20
    save()


def run_child(npz, cout, as_gib, wall_s, watch_gib, log, variant='colamd'):
    def limits():
        lim = int(as_gib * GIB)
        resource.setrlimit(resource.RLIMIT_AS, (lim, lim))
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    t = time.perf_counter()
    p = subprocess.Popen([sys.executable, '-u', os.path.abspath(__file__), '--child', str(npz), str(cout), str(wall_s),
                          str(watch_gib), variant], preexec_fn=limits, env=env)
    backstop = False
    try:
        rc = p.wait(timeout=wall_s + 1800)                                          # the child's own alarm ends it first
    except subprocess.TimeoutExpired:
        backstop = True
        p.kill(); rc = p.wait()
    el = time.perf_counter() - t
    ru = resource.getrusage(resource.RUSAGE_CHILDREN)
    try:
        c = json.loads(Path(cout).read_text())
    except Exception:                                                               # noqa: BLE001
        c = dict(status='no child record')
    if rc == -signal.SIGALRM:
        c['status'] = 'wall limit (child ended itself by SIGALRM)'
    elif rc not in (0, 3) and c.get('status') in ('solving', 'started', 'factorising'):
        c['status'] = f'child ended with return code {rc}'
    c.update(returncode=rc, parent_wall_s=el, parent_backstop_kill=backstop, child_peak_rss_GiB=ru.ru_maxrss / 2 ** 20,
             as_limit_GiB=as_gib, watchdog_GiB=watch_gib, wall_limit_s=wall_s)
    log(dict(event='CHILD', **{k: v for k, v in c.items() if k not in ('compliance', 'rel_residual')}))
    return c


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('layout')
    ap.add_argument('--body', default='/root/autodl-tmp/OPL/S4/body')
    ap.add_argument('--as-gib', type=float, default=70.0); ap.add_argument('--watch-gib', type=float, default=70.0)
    ap.add_argument('--wall-s', type=int, default=7200); ap.add_argument('--ref', default='')
    ap.add_argument('--n-random', type=int, default=3); ap.add_argument('--keep-npz-dir', default=None)
    ap.add_argument('--variants', default='colamd,mmd')
    a = ap.parse_args(argv)
    import diag_sens as DS                                               # noqa: F401  CPU environment
    import scipy.sparse as sp
    import trainlib as TL
    import teacher as TE
    import box_encode as BX
    import lat_multi as LM
    import lat_direct_cpu as LD
    BX.dev = TL.dev
    log = lambda d: print(json.dumps(d, default=float), flush=True)
    L = json.loads(Path(a.layout).read_text())
    rec = dict(layout=L['name'], machine=__import__('platform').node(), route='scipy.sparse.linalg.splu (SuperLU), factor once + solve 6 loads, full symmetric unscaled CSC, single thread',
               threads=dict(OMP=os.environ.get('OMP_NUM_THREADS'), MKL=os.environ.get('MKL_NUM_THREADS')),
               loadavg_start=list(os.getloadavg()), cells={})
    Cs, lay = [], {}
    t0 = time.perf_counter()
    for c in L['cells']:
        t = time.perf_counter(); C = TE.Cell(c['case'], a.body, log=lambda s_: None); ts = time.perf_counter() - t
        t = time.perf_counter(); C.assemble(); ta = time.perf_counter() - t
        rec['cells'][c['case']] = dict(setup_s=ts, assembly_s=ta, dofs=int(C.nb))
        Cs.append(C); lay[tuple(c['position'])] = LM.from_teacher(C)
    rec['cells_s'] = time.perf_counter() - t0
    lat = LM.MultiLattice(lay, clamp=('y', 'min'), load=('y', 'max'), loads='consistent', n_random=a.n_random,
                          device='cpu', log=lambda s_: None)
    F = lat.F.cpu().numpy().astype(np.float64)
    t = time.perf_counter()
    Ku, n, offs = LD.global_matrix(Cs, lat)
    for C in Cs:
        C._free()
    for G in lat.geoms:
        G.cell, G._kpp, G._kpp_fn, G.face_w_fn = None, None, None, None
    nF = F.shape[0]
    Cs = lay = lat = None; gc.collect()
    K = (Ku + sp.triu(Ku, 1).T).tocsc(); K.sort_indices()
    del Ku; gc.collect()
    rec['global_assembly_s'] = time.perf_counter() - t
    B = np.zeros((n, F.shape[1])); B[:nF] = F
    rec.update(global_dofs=int(n), nnz_full=int(K.nnz), matrix='full symmetric CSC, unscaled (no Jacobi scaling)')
    d = Path(a.keep_npz_dir or Path(a.out).parent); d.mkdir(parents=True, exist_ok=True)
    npz = d / f"{L['name']}_K.npz"
    t = time.perf_counter()
    np.savez(npz, data=K.data, indices=K.indices, indptr=K.indptr, shape=np.array(K.shape), B=B, nF=nF)
    rec['save_s'] = time.perf_counter() - t; rec['npz'] = str(npz); rec['npz_GiB'] = npz.stat().st_size / GIB
    del K, B; gc.collect()
    Path(a.out).write_text(json.dumps(rec, indent=1, default=float))
    rec['solves'] = {}
    for v in a.variants.split(','):
        c = run_child(npz, Path(a.out).with_suffix(f'.child_{v}.json'), a.as_gib, a.wall_s, a.watch_gib, log, v)
        if 'compliance' in c:
            for f in [x for x in a.ref.split(',') if x]:
                try:
                    dd = json.loads(Path(f).read_text())
                    rr = next(r for r in dd.get('runs', []) if 'compliance' in r)
                    c['reference'] = dict(file=f, compliance=rr['compliance'],
                                          rel_diff=[abs(x / y - 1) for x, y in zip(c['compliance'], rr['compliance'])])
                    break
                except Exception:                                                   # noqa: BLE001
                    continue
        rec['solves'][v] = c
        Path(a.out).write_text(json.dumps(rec, indent=1, default=float))
    rec['loadavg_end'] = list(os.getloadavg())
    Path(a.out).write_text(json.dumps(rec, indent=1, default=float))
    log(dict(event='DONE', **{v: (c.get('status'), c.get('factor_s'), c.get('solve_s'), c.get('child_peak_rss_GiB')) for v, c in rec['solves'].items()}))


def selftest(d):
    """Tiny 3D Laplacian (success path) and the same with a 1-second wall limit on a larger one (failure path)."""
    import scipy.sparse as sp
    out = {}
    for name, m, wall, asg, var in (('ok_colamd', 20, 600, 8.0, 'colamd'), ('ok_mmd', 20, 600, 8.0, 'mmd'),
                                     ('wall', 70, 1, 8.0, 'colamd'), ('as', 70, 600, 0.6, 'colamd')):
        L1 = sp.diags([-np.ones(m - 1), 2 * np.ones(m), -np.ones(m - 1)], [-1, 0, 1]); I1 = sp.identity(m)
        K = (sp.kron(sp.kron(L1, I1), I1) + sp.kron(sp.kron(I1, L1), I1) + sp.kron(sp.kron(I1, I1), L1)).tocsc()
        B = np.random.default_rng(0).standard_normal((K.shape[0], 2))
        npz = Path(d) / f'selftest_{name}.npz'
        np.savez(npz, data=K.data, indices=K.indices, indptr=K.indptr, shape=np.array(K.shape), B=B, nF=K.shape[0])
        c = run_child(npz, Path(d) / f'selftest_{name}.child.json', asg, wall, 50.0, lambda x: None, var)
        out[name] = {k: c.get(k) for k in ('status', 'returncode', 'factor_s', 'solve_s', 'rel_residual', 'child_peak_rss_GiB', 'detail')}
    print(json.dumps(out, default=float))


if __name__ == '__main__':
    if sys.argv[1:2] == ['--child']:
        child(sys.argv[2], sys.argv[3], int(sys.argv[4]), float(sys.argv[5]), sys.argv[6] if len(sys.argv) > 6 else 'colamd')
    elif sys.argv[1:2] == ['--selftest']:
        selftest(sys.argv[2] if len(sys.argv) > 2 else '.')
    else:
        main(sys.argv[1:])
