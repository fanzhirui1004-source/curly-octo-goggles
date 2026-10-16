"""Revision (E1) host baseline for Table 5: the conventional route on the host CPU, configured properly. NEW script; bench_cpu.py
(the originally reported LU / pypardiso configuration) is unchanged.
Per cell (one process = one repetition; the chain runs three):
  front end   topology seconds from PREP.json; teacher.Cell setup and assembly on the host (as bench_cpu.py)
  interior    K_II (symmetrically Jacobi scaled, upper triangle) -> MKL PARDISO real SPD Cholesky (mtype 2) through direct
              ctypes calls (pardiso_direct.Pardiso: zero-based CSR passed once, no per-call hashing / copies), explicit iparm
              (--iparm-file from --mode tune, else pardiso_direct.tuned_iparm()); analysis and factorisation timed separately
  query       S q = (K E q)_P for B = 1, 16, 64 random retained vectors: one warm-up, then --reps individually timed calls
              (median, min, all); the PARDISO solve phase alone (phase 33) timed the same way, relative residual of the
              scaled interior system, and the implied factor-read rate 2 * 8 * nnz(L) / t
  explicit S  (a) PARDISO Schur-complement option (iparm(36) = 1) on the full scaled cell matrix with perm marking the
              retained DOFs: analysis + factorisation + dense S in one call; verified against 32 columns S e_j from the
              query route; (b) the blocked route of bench_cpu.py (256 retained unit vectors per call, now through the direct
              Cholesky handle) until --s-budget seconds, then linear extrapolation in the number of columns
  memory      PARDISO iparm(15-17) in kB and GiB (kB / 2**20), process peak RSS per phase (VmHWM reset per phase), dense S
Modes:
  --mode main   the above
  --mode luref  the previously reported configuration for reference: pypardiso LU (mtype 11, MKL defaults, iparm(1) = 0) of
                the full K_II; queries through pypardiso.solve() (hash + index copies every call) AND through a direct
                phase-33 call on the same factor, plus the wrapper overhead alone (hash / compare + copies), so the share of
                the old single-vector time that was wrapper overhead is measured
  --mode tune   one cell: ordering iparm(2) in {2, 3} x factorisation iparm(24) in {0, 1} (analysis + factorisation time),
                then iparm(25) in {0, 1, 2} for the solve phase (B = 1 and 64); writes the chosen settings to --iparm-file
Usage: bench_cpu2.py <out.json> <case>[,...] [--mode main|luref|tune] [--iparm-file f.json] [--reps 3] [--s-budget 300]
       [--no-schur] [--no-blocked] [--body /root/autodl-tmp/OPL/S0]
Needs OPL_DEV=cpu (diag_sens environment), PYPARDISO_MKL_RT, MKL_NUM_THREADS = OMP_NUM_THREADS = container cores."""
import diag_sens as DS                                                   # noqa: F401  first: CPU environment
import os, sys, json, time, argparse, gc
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import torch
import trainlib as TL
import teacher as TE
import box_encode as BX
import pardiso_direct as PD

dt = TL.dt
BX.dev = TL.dev


def interior_upper(C):
    """Scaled upper CSR of K_II (as bench_cpu.factor) and the scaling vector sA (torch)."""
    pm = torch.zeros(C.nb, dtype=torch.bool); pm[C.P] = True
    new = torch.full((C.nb,), -1, dtype=torch.long); new[C.I] = torch.arange(C.ni)
    ru, cu = C.ru.long(), C.cu.long()
    sel = torch.nonzero(~pm[ru] & ~pm[cu]).squeeze(1)
    rA, cA, vA = new[ru[sel]], new[cu[sel]], C.vals[sel]
    sA = torch.zeros(C.ni, dtype=dt); sA[rA[rA == cA]] = 1 / torch.sqrt(vA[rA == cA])
    U = sp.csr_matrix(((vA * sA[rA] * sA[cA]).numpy(), (rA.numpy(), cA.numpy())), shape=(C.ni, C.ni))
    U.sum_duplicates(); U.sort_indices()
    return U, sA


def full_upper(C):
    """Scaled upper CSR of the whole cell matrix K (all DOFs) and its scaling vector (numpy)."""
    ru, cu, v = C.ru.long().numpy(), C.cu.long().numpy(), C.vals.numpy()
    d = v[C.diag.numpy()]
    s = 1 / np.sqrt(d)
    U = sp.csr_matrix((v * s[ru] * s[cu], (ru, cu)), shape=(C.nb, C.nb))
    U.sum_duplicates(); U.sort_indices()
    return U, s


class DirectSolve:
    """C.sol_I replacement: phase 33 on a Pardiso handle."""

    def __init__(self, P):
        self.P = P

    def solve(self, r):
        b = r.numpy()
        x = self.P.solve(b[:, 0] if b.shape[1] == 1 else b)
        return torch.from_numpy(np.ascontiguousarray(x).reshape(r.shape))

    def free(self):
        self.P.release()


class OldSolve:
    """bench_cpu.py's C.sol_I: pypardiso.solve(A, b) every call."""

    def __init__(self, ps, A):
        self.ps, self.A = ps, A

    def solve(self, r):
        return torch.as_tensor(self.ps.solve(self.A, np.ascontiguousarray(r.numpy())), dtype=dt).reshape(r.shape)

    def free(self):
        self.ps.free_memory(everything=True)


def timed(fn, reps=3, warm=1):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return dict(median=float(np.median(ts)), min=float(np.min(ts)), all=ts)


def phase_peak(fn):
    """Run fn with the process high-water mark reset; returns (result, seconds, peak RSS GiB)."""
    ok = PD.reset_peak()
    t = time.perf_counter(); out = fn(); el = time.perf_counter() - t
    return out, el, (PD.peak_rss_gib() if ok else float('nan'))


def front_end(case, body):
    r = dict(loadavg_start=PD.loadavg(), cpu_stat_start=PD.cpu_stat())
    prep = Path(body) / case / 'PREP.json'
    if prep.exists():
        p = json.loads(prep.read_text())
        r['topology'] = dict(seconds=p.get('total_seconds'), workers=p.get('workers'))
    t = time.perf_counter(); C = TE.Cell(case, body, log=lambda s_: None); r['setup_s'] = time.perf_counter() - t
    t = time.perf_counter(); C.assemble(); r['assembly_s'] = time.perf_counter() - t
    r.update(dofs=int(C.nb), ports=int(C.np_), interior=int(C.ni), elements=int(len(C.cells)), K_nnz_upper=int(len(C.vals)))
    return C, r


def queries(C, reps, P=None, U=None):
    """Timed S q for B = 1, 16, 64 (the full query) and, with a Pardiso handle P, the solve phase alone + residuals."""
    out = {}
    g = torch.Generator().manual_seed(0)
    for B in (1, 16, 64):
        q = torch.randn((C.np_, B), dtype=dt, generator=g)
        o = dict(query_s=timed(lambda: C.apply(q), reps))
        if P is not None:
            b = np.asfortranarray(np.random.default_rng(B).standard_normal((C.ni, B)))
            bb = b[:, 0].copy() if B == 1 else b
            xb = np.empty_like(bb, order='F')
            o['solve_only_s'] = timed(lambda: P.solve(bb, xb), reps)
            x = P.solve(bb).reshape(C.ni, B)
            res = PD.sym_upper_matvec(U, x) - b
            o['rel_residual_max'] = float(max(np.linalg.norm(res[:, j]) / np.linalg.norm(b[:, j]) for j in range(B)))
            nnzL = int(P.iparm[17])
            if nnzL > 0:
                o['factor_read_GBps'] = 2 * 8 * nnzL / o['solve_only_s']['median'] / 1e9       # L and L^T values once each
        out[str(B)] = o
    return out


def explicit_blocked(C, budget):
    npt = C.np_; blk = 256; done = 0
    t = time.perf_counter()
    while done < npt and time.perf_counter() - t < budget:
        m = min(blk, npt - done)
        E = torch.zeros((npt, m), dtype=dt); E[done + torch.arange(m), torch.arange(m)] = 1
        _ = C.apply(E)
        done += m
    el = time.perf_counter() - t
    return dict(columns_done=done, columns=npt, seconds_done=el, seconds=el * npt / done, extrapolated=bool(done < npt),
                block=blk, dense_GiB=8 * npt ** 2 / PD.GIB)


def explicit_schur(C, iparm, verify=32):
    """Dense S by PARDISO's Schur-complement option on the full scaled cell matrix; returns record (S verified, then freed)."""
    U, s = full_upper(C)
    P_ = C.P.numpy()
    perm = np.zeros(C.nb, np.int32); perm[P_] = 1
    ns = len(P_)
    ip = dict(iparm); ip[36] = 1
    r = dict(schur_rows=ns, iparm36=1)
    tried = []
    for variant in (ip, {**ip, 24: 0}, {**ip, 24: 0, 2: 2}):
        H = PD.Pardiso(U, 2, variant, perm=perm)
        x = np.zeros(max(C.nb, ns * ns))
        try:
            ok = PD.reset_peak()
            t = time.perf_counter(); H.phase(11); ta = time.perf_counter() - t
            t = time.perf_counter(); H.phase(22, x=x); tf = time.perf_counter() - t
            r.update(analysis_s=ta, factor_schur_s=tf, seconds=ta + tf, peak_rss_GiB=PD.peak_rss_gib() if ok else None,
                     used_iparm={str(k): int(v) for k, v in variant.items()}, **H.info())
            break
        except PD.PardisoError as e:
            tried.append(dict(iparm={str(k): int(v) for k, v in variant.items()}, error=str(e)))
            H.release(); x = None
            continue
    r['failed_variants'] = tried
    if 'seconds' not in r:
        return r
    H.release()
    S = x[:ns * ns].reshape(ns, ns)
    sP = s[P_]
    S /= sP[:, None]; S /= sP[None, :]                                               # unscale: S_K = S_A / (s_P s_P^T)
    r['symmetry_rel'] = float(np.abs(S - S.T).max() / np.abs(S).max())
    rng = np.random.default_rng(1)
    j = np.sort(rng.choice(ns, size=min(verify, ns), replace=False))
    E = torch.zeros((ns, len(j)), dtype=dt); E[torch.as_tensor(j), torch.arange(len(j))] = 1
    Sj = C.apply(E).numpy()
    r['verify_columns'] = int(len(j))
    r['verify_rel_err'] = float(np.linalg.norm(S[:, j] - Sj) / np.linalg.norm(Sj))
    r['dense_GiB'] = 8 * ns * ns / PD.GIB
    del S, x, U; gc.collect()
    return r


def run_main(C, r, iparm, a):
    U, sA = interior_upper(C)
    r['K_II_nnz_upper'] = int(U.nnz)
    P = PD.Pardiso(U, 2, iparm)
    ok = PD.reset_peak()
    r['analysis_s'] = P.phase(11)
    r['factor_s'] = P.phase(22)
    r['factor_peak_rss_GiB'] = PD.peak_rss_gib() if ok else None
    r['pardiso'] = P.info()
    C._free(); C.sA, C.sol_I, C.fp32 = sA, DirectSolve(P), False
    r['queries'], _, r['query_peak_rss_GiB'] = phase_peak(lambda: queries(C, a.reps, P, U))
    if not a.no_blocked:
        r['explicit_S_blocked'], _, pk = phase_peak(lambda: explicit_blocked(C, a.s_budget))
        r['explicit_S_blocked']['peak_rss_GiB'] = pk
        r['explicit_S_blocked']['plus_factor_s'] = r['explicit_S_blocked']['seconds'] + r['analysis_s'] + r['factor_s']
    if not a.no_schur:
        r['explicit_S_schur'] = explicit_schur(C, iparm)
    C._free()
    del U


def run_luref(C, r, a):
    import pypardiso
    U, sA = interior_upper(C)
    A = (U + sp.triu(U, 1).T).tocsr()                                                # full K_II, as bench_cpu.py
    del U
    ps = pypardiso.PyPardisoSolver()
    ok = PD.reset_peak()
    t = time.perf_counter(); ps.factorize(A); r['factor_s'] = time.perf_counter() - t
    r['factor_peak_rss_GiB'] = PD.peak_rss_gib() if ok else None
    ip = ps.get_iparms()
    r['pardiso'] = dict(iparm_1based={str(k): int(v) for k, v in ip.items() if v != 0},
                        peak_mem_kB=dict(peak_analysis=int(ip[15]), permanent=int(ip[16]), factor_solve=int(ip[17])),
                        peak_mem_GiB=dict(peak_analysis=ip[15] / 2 ** 20, permanent=ip[16] / 2 ** 20,
                                          factor_solve=ip[17] / 2 ** 20, total=(ip[16] + ip[17]) / 2 ** 20),
                        nnz_factor=int(ip[18]), hash_mode=bool(A.nnz > ps.size_limit_storage))
    r['K_II_nnz_full'] = int(A.nnz)
    C._free(); C.sA, C.fp32 = sA, False
    C.sol_I = OldSolve(ps, A)
    r['queries_pypardiso'] = queries(C, a.reps)
    # direct phase 33 on the SAME factor (pypardiso's handle and iparm, one-based arrays built once)
    H = PD.Pardiso(A, 11, None); H.pt = ps.pt; H.iparm = ps.iparm; H.released = True
    C.sol_I = DirectSolve(H)
    r['queries_direct'] = queries(C, a.reps)
    # the wrapper overhead alone: pypardiso's factorisation check + index copies per call
    b1 = np.ones((A.shape[0], 1))
    r['wrapper_check_s'] = timed(lambda: ps._is_already_factorized(A), a.reps)
    r['wrapper_copies_s'] = timed(lambda: (A.indptr.astype(np.int32) + 1, A.indices.astype(np.int32) + 1), a.reps)
    r['wrapper_check_b_s'] = timed(lambda: ps._check_b(A, b1), a.reps)
    o, d = r['queries_pypardiso']['1']['query_s']['median'], r['queries_direct']['1']['query_s']['median']
    r['single_vector_overhead'] = dict(old_s=o, direct_s=d, overhead_s=o - d, overhead_share=(o - d) / o)
    ps.free_memory(everything=True)
    C.sol_I = None
    del A, ps


def run_tune(C, r, a):
    U, sA = interior_upper(C)
    combos = []
    for ordering in (2, 3):
        for two in (0, 1):
            ip = PD.tuned_iparm(ordering=ordering, two_level=two, par_solve=1)
            P = PD.Pardiso(U, 2, ip)
            try:
                ta = P.phase(11); tf = P.phase(22)
                combos.append(dict(ordering=ordering, two_level=two, analysis_s=ta, factor_s=tf, total_s=ta + tf,
                                   nnz_factor=int(P.iparm[17]), mem_GiB=P.mem_GiB()))
            except PD.PardisoError as e:
                combos.append(dict(ordering=ordering, two_level=two, error=str(e)))
            P.release()
            print(json.dumps(dict(tune=combos[-1]), default=float), flush=True)
    good = [c for c in combos if 'error' not in c]
    best = min(good, key=lambda c: c['total_s'])
    solves = []
    for ps_ in (0, 1, 2):
        ip = PD.tuned_iparm(ordering=best['ordering'], two_level=best['two_level'], par_solve=ps_)
        P = PD.Pardiso(U, 2, ip)
        try:
            P.phase(11); P.phase(22)
            o = dict(par_solve=ps_)
            for B in (1, 64):
                b = np.asfortranarray(np.random.default_rng(B).standard_normal((C.ni, B)))
                bb = b[:, 0].copy() if B == 1 else b
                xb = np.empty_like(bb, order='F')
                o[f'B{B}'] = timed(lambda: P.solve(bb, xb), a.reps)
                x = P.solve(bb).reshape(C.ni, B)
                o[f'B{B}_rel_residual'] = float(np.linalg.norm(PD.sym_upper_matvec(U, x) - b) / np.linalg.norm(b))
            solves.append(o)
        except PD.PardisoError as e:
            solves.append(dict(par_solve=ps_, error=str(e)))
        P.release()
        print(json.dumps(dict(tune=solves[-1]), default=float), flush=True)
    goods = [s for s in solves if 'error' not in s]
    bs = min(goods, key=lambda s: (s['B1']['median'], s['B64']['median']))
    chosen = PD.tuned_iparm(ordering=best['ordering'], two_level=best['two_level'], par_solve=bs['par_solve'])
    r['tuning'] = dict(factor=combos, solve=solves, chosen={str(k): v for k, v in chosen.items()})
    if a.iparm_file:
        Path(a.iparm_file).write_text(json.dumps(dict(iparm={str(k): v for k, v in chosen.items()}, tuned_on=C.case,
                                                      tuning=r['tuning']), indent=1, default=float))


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('cases')
    ap.add_argument('--mode', default='main', choices=['main', 'luref', 'tune'])
    ap.add_argument('--iparm-file', default=None); ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--s-budget', type=float, default=300.0)
    ap.add_argument('--no-schur', action='store_true'); ap.add_argument('--no-blocked', action='store_true')
    ap.add_argument('--body', default='/root/autodl-tmp/OPL/S0')
    a = ap.parse_args(argv)
    iparm = PD.tuned_iparm()
    if a.mode == 'main' and a.iparm_file and Path(a.iparm_file).exists():
        iparm = {int(k): int(v) for k, v in json.loads(Path(a.iparm_file).read_text())['iparm'].items()}
    rec = dict(mode=a.mode, machine=__import__('platform').node(), env=PD.env_record(), iparm_requested={str(k): v for k, v in iparm.items()} if a.mode == 'main' else None,
               reps_within=a.reps, s_budget=a.s_budget, per_case={})
    if Path(a.out).exists():
        rec = json.loads(Path(a.out).read_text())
    for case in a.cases.split(','):
        if case in rec['per_case']:
            continue
        C, r = front_end(case, a.body)
        try:
            {'main': lambda: run_main(C, r, iparm, a), 'luref': lambda: run_luref(C, r, a), 'tune': lambda: run_tune(C, r, a)}[a.mode]()
        except Exception as e:                                                      # noqa: BLE001  record, go on
            import traceback
            r['error'] = traceback.format_exc()[-2000:]
        r['loadavg_end'] = PD.loadavg(); r['cpu_stat_end'] = PD.cpu_stat()
        r['throttled_s'] = (r['cpu_stat_end'].get('throttled_usec', 0) - r['cpu_stat_start'].get('throttled_usec', 0)) / 1e6
        r['cpu_s'] = (r['cpu_stat_end'].get('usage_usec', 0) - r['cpu_stat_start'].get('usage_usec', 0)) / 1e6
        r['process_peak_rss_GiB'] = PD.lifetime_peak_gib()
        try:
            C._free()
        except Exception:                                                           # noqa: BLE001
            pass
        C = None; gc.collect()
        rec['per_case'][case] = r
        print(json.dumps(dict(case=case, **{k: v for k, v in r.items() if k in ('setup_s', 'assembly_s', 'analysis_s', 'factor_s', 'error')}),
                         default=float), flush=True)
        Path(a.out).write_text(json.dumps(rec, indent=1, default=float))


if __name__ == '__main__':
    main(sys.argv[1:])
