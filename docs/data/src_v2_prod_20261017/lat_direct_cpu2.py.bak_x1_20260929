"""Revision (E1) whole-lattice direct solve for Table 6, configured properly. NEW script; lat_direct_cpu.py is unchanged and
its helpers (global_matrix, cell handling, loads) are reused, so matrix, numbering, clamp and loads are identical.
One process = one repetition (the chain runs three): cells (setup + assembly, host) -> MultiLattice (clamp y = min,
consistent loads on y = max + 3 random) -> global upper matrix -> symmetric Jacobi scaling -> MKL PARDISO through direct
ctypes calls (pardiso_direct.Pardiso) with explicit iparm (--iparm-file, as tuned for Table 5), phases 11 / 22 / 33
(all load cases in one solve), each timed.
  mtype 2   real SPD Cholesky on the upper triangle (the configuration of the revision)
  mtype 11  real nonsymmetric LU with MKL defaults (iparm(1) = 0): the previously reported reference (run once)
Memory: the unscaled matrix is released before the factorisation (residuals are evaluated from the scaled matrix), the
CSR arrays are passed without copies. After the analysis the predicted PARDISO memory (iparm(16) + iparm(17), kB -> GiB)
is compared with what the container can still allocate: the numerical factorisation runs only if
    predicted + --margin-gib <= memory.high (or memory.max) - anonymous memory in use by the whole container
otherwise the run records 'skipped' with the prediction. Peak RSS per phase (VmHWM reset per phase), process lifetime peak.
Outputs per run: seconds per phase, PARDISO memory (kB and GiB), nnz(L), iparm, peak RSS, compliance F_j^T u_j per load,
relative residual ||K u - f|| / ||f|| per load (unscaled system), loadavg, container CPU throttling.
Usage: lat_direct_cpu2.py <out.json> <layout.json> [--body /root/autodl-tmp/OPL/S4/body] [--mtypes 2] [--iparm-file f]
       [--margin-gib 4] [--n-random 3] [--analysis-only]
--analysis-only (default off): stop after the analysis phase and record the predicted memory (no factorisation).
Environment as lat_direct_cpu.py: OPL_DEV=cpu, CUDA_VISIBLE_DEVICES=, MKL_NUM_THREADS=OMP_NUM_THREADS=16, PYPARDISO_MKL_RT."""
import diag_sens as DS                                                   # noqa: F401  first: CPU environment
import os, sys, json, time, argparse, gc
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import torch
import trainlib as TL
import teacher as TE
import box_encode as BX
import lat_multi as LM
import lat_direct_cpu as LD
import pardiso_direct as PD

BX.dev = TL.dev


def pardiso_run(A, s, b, F, mtype, iparm, margin, log, analysis_only=False):
    """A: scaled upper CSR (mtype 2) or scaled full CSR (mtype 11); s: scaling; b: unscaled right-hand sides (n, k)."""
    n = A.shape[0]
    r = dict(mtype=mtype, nnz_stored=int(A.nnz), iparm_requested=None if iparm is None else {str(k): v for k, v in iparm.items()},
             loadavg_start=PD.loadavg(), cpu_stat_start=PD.cpu_stat())
    P = PD.Pardiso(A, mtype, iparm)
    ok = PD.reset_peak()
    r['analysis_s'] = P.phase(11)
    r['analysis_peak_rss_GiB'] = PD.peak_rss_gib() if ok else None
    r['predicted_kB'] = P.mem_kB(); r['predicted_GiB'] = P.mem_GiB()
    av = PD.avail_gib()
    r['avail_before_factor_GiB'] = av
    r['cgroup_before_factor'] = PD.cgroup()
    log(dict(event='ANALYSIS', **{k: v for k, v in r.items() if k not in ('cpu_stat_start', 'cgroup_before_factor')}))
    if analysis_only:
        r['skipped'] = 'analysis only (--analysis-only): predicted memory recorded, no factorisation'
        P.release()
        return r
    need = r['predicted_GiB']['total'] + margin
    if need > av:
        r['skipped'] = f"predicted {r['predicted_GiB']['total']:.2f} GiB + margin {margin} GiB > available {av:.2f} GiB"
        P.release()
        return r
    ok = PD.reset_peak()
    r['factor_s'] = P.phase(22)
    r['factor_peak_rss_GiB'] = PD.peak_rss_gib() if ok else None
    r['pardiso'] = P.info()
    bs = np.asfortranarray(b * s[:, None])
    ok = PD.reset_peak()
    t = time.perf_counter(); x = P.solve(bs); r['solve_s'] = time.perf_counter() - t
    r['solve_peak_rss_GiB'] = PD.peak_rss_gib() if ok else None
    P.release()
    # residual of the unscaled system from the scaled matrix: K u - f = (A x - s f) / s, u = s x
    Ax = PD.sym_upper_matvec(A, x) if mtype == 2 else A @ x
    res = (Ax - bs) / s[:, None]
    u = x * s[:, None]
    r['rel_residual'] = [float(np.linalg.norm(res[:, j]) / np.linalg.norm(b[:, j])) for j in range(b.shape[1])]
    r['compliance'] = [float(F[:, j] @ u[:F.shape[0], j]) for j in range(F.shape[1])]
    r['loadavg_end'] = PD.loadavg(); cs = PD.cpu_stat()
    r['throttled_s'] = (cs.get('throttled_usec', 0) - r['cpu_stat_start'].get('throttled_usec', 0)) / 1e6
    r['process_peak_rss_GiB'] = PD.lifetime_peak_gib()
    return r


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('layout')
    ap.add_argument('--body', default='/root/autodl-tmp/OPL/S4/body')
    ap.add_argument('--mtypes', default='2'); ap.add_argument('--iparm-file', default=None)
    ap.add_argument('--margin-gib', type=float, default=4.0); ap.add_argument('--n-random', type=int, default=3)
    ap.add_argument('--analysis-only', action='store_true')                   # default off: predicted memory only
    a = ap.parse_args(argv)
    log = lambda d: print(json.dumps(d, default=float), flush=True)
    iparm = PD.tuned_iparm()
    if a.iparm_file and Path(a.iparm_file).exists():
        iparm = {int(k): int(v) for k, v in json.loads(Path(a.iparm_file).read_text())['iparm'].items()}
    L = json.loads(Path(a.layout).read_text())
    rec = dict(layout=L['name'], env=PD.env_record(), loadavg_start=PD.loadavg(), cells={})
    Cs, lay = [], {}
    PD.reset_peak()
    t0 = time.perf_counter()
    for c in L['cells']:
        t = time.perf_counter(); C = TE.Cell(c['case'], a.body, log=lambda s_: None); ts = time.perf_counter() - t
        t = time.perf_counter(); C.assemble(); ta = time.perf_counter() - t
        rec['cells'][c['case']] = dict(setup_s=ts, assembly_s=ta, dofs=int(C.nb), retained=int(C.np_), interior=int(C.ni))
        Cs.append(C); lay[tuple(c['position'])] = LM.from_teacher(C)
    rec['cells_s'] = time.perf_counter() - t0
    t = time.perf_counter()
    lat = LM.MultiLattice(lay, clamp=('y', 'min'), load=('y', 'max'), loads='consistent', n_random=a.n_random,
                          device='cpu', log=lambda s_: None)
    rec['lattice_s'] = time.perf_counter() - t
    assert [G.case for G in lat.geoms] == [C.case for C in Cs]
    F = lat.F.cpu().numpy().astype(np.float64)
    t = time.perf_counter()
    Ku, n, offs = LD.global_matrix(Cs, lat)
    rec['global_assembly_s'] = time.perf_counter() - t
    rec.update(global_dofs=int(n), free_retained=int(lat.nfree), interior=int(n - lat.nfree), nnz_upper=int(Ku.nnz),
               loads=lat.labels, front_end_peak_rss_GiB=PD.peak_rss_gib())
    log(dict(event='ASSEMBLED', **{k: v for k, v in rec.items() if k not in ('cells', 'env')}))
    for C in Cs:                                                            # release the cells: only K, F remain
        C._free()
    for G in lat.geoms:
        G.cell, G._kpp, G._kpp_fn, G.face_w_fn = None, None, None, None
    Cs = lay = lat = None; gc.collect()
    b = np.zeros((n, F.shape[1])); b[:F.shape[0]] = F
    d = Ku.diagonal()
    s = 1 / np.sqrt(np.where(d > 0, d, 1.0))
    Sd = sp.diags(s)
    t = time.perf_counter()
    Us = (Sd @ Ku @ Sd).tocsr(); Us.sort_indices()
    rec['scaling_s'] = time.perf_counter() - t
    del Ku; gc.collect()                                                    # residuals from the scaled matrix
    rec['rss_before_pardiso_GiB'] = PD.rss_gib()
    rec['runs'] = []
    for mt in (int(m) for m in a.mtypes.split(',')):
        if mt == 2:
            A, ip = Us, iparm
        else:
            A, ip = (Us + sp.triu(Us, 1).T).tocsr(), None                   # full matrix, MKL defaults (old configuration)
            A.sort_indices()
        r = pardiso_run(A, s, b, F, mt, ip, a.margin_gib, log, a.analysis_only)
        if mt != 2:
            del A; gc.collect()
        rec['runs'].append(r)
        log(dict(event='RUN', **{k: v for k, v in r.items() if k not in ('cpu_stat_start',)}))
        Path(a.out).write_text(json.dumps(rec, indent=1, default=float))
    rec['process_peak_rss_GiB'] = PD.lifetime_peak_gib()
    Path(a.out).write_text(json.dumps(rec, indent=1, default=float))
    log(dict(event='DONE'))


if __name__ == '__main__':
    main(sys.argv[1:])
