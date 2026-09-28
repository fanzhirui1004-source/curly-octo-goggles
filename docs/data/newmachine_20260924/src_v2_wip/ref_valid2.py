"""Revision (E7): CutFEM reference refinement to n = 56 and 64. NEW wrapper; ref_valid.py is imported and run UNCHANGED (same
clamp / load faces chosen by pick_faces at n = 32, unit consistent traction in x, y, z, compliance and 8-corner
sensitivities, vs_finest, and the n = 32 studies of gamma, derivative step and integration), except for the linear solve:
  ref_valid.solve (pypardiso LU, MKL defaults, full matrix) is replaced by a Cholesky solve (mtype 2) of the symmetrically
  Jacobi-scaled upper triangle of K restricted to the free DOFs, through direct PARDISO calls (pardiso_direct.Pardiso,
  explicit iparm), which halves the factor memory; the solution is the same to solver precision (relative residual of the
  unscaled system recorded per solve).
Memory guard: after the analysis phase the predicted PARDISO memory (iparm(16) + iparm(17)) must satisfy
    predicted + --headroom-gib <= memory.high - anonymous memory of the whole container,
otherwise the solve raises MemoryError('SKIP_PREDICTED ...'), which ref_valid records as that n's 'error' (the sweep goes on).
--retry-errors removes such error entries first (for a later run with less headroom).
Order: --stage sweep runs only the resolution sweeps (the n = 32 studies are deferred with placeholders), --stage studies then
runs the studies; --stage all does both, sweeps of every case first.
--seed <file> copies a case's existing record (e.g. ref_valid_h1.json) into the output before extending it.
Every solve is logged (case, n, sizes, phase times, PARDISO memory in GiB, residuals, RSS) to <out>_solves.jsonl.
Usage: ref_valid2.py <out.json> <case>[,...] [--ns 24,32,40,48,56,64] [--stage all|sweep|studies] [--headroom-gib 25]
       [--seed file.json] [--retry-errors] [ref_valid.py options]
Needs OPL_DEV=cpu, PYPARDISO_MKL_RT, OPL_PACKETS_EXTRA including the <case>_n<n> packet root; bodies <case>_n<n> built first."""
import ref_valid as RV                                                   # imports diag_sens first (CPU environment)
import sys, json, time, argparse, gc
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import torch
import pardiso_direct as PD

A = dict(headroom=25.0, log=None, iparm=PD.tuned_iparm(ordering=3, two_level=0, par_solve=0))
PLACE = '__deferred_by_ref_valid2__'


def solve(C, F, fr):
    t0 = time.perf_counter()
    ru, cu, v = C.ru.long().numpy(), C.cu.long().numpy(), C.vals.numpy()
    fr = np.asarray(fr)
    m = np.full(C.nb, -1, np.int64); m[fr] = np.arange(len(fr))              # fr ascending: upper stays upper
    r_, c_ = m[ru], m[cu]
    ok = (r_ >= 0) & (c_ >= 0)
    nf = len(fr)
    U = sp.csr_matrix((v[ok], (r_[ok], c_[ok])), shape=(nf, nf))
    del r_, c_, ok
    U.sum_duplicates()
    s = 1 / np.sqrt(U.diagonal())
    U = (sp.diags(s) @ U @ sp.diags(s)).tocsr(); U.sort_indices()
    rec = dict(case=C.case, n=int(C.n), dofs=int(C.nb), free=int(nf), nnz_upper=int(U.nnz), build_s=time.perf_counter() - t0)
    P = PD.Pardiso(U, 2, A['iparm'])
    PD.reset_peak()
    rec['analysis_s'] = P.phase(11)
    rec['predicted_GiB'] = P.mem_GiB()
    av = PD.avail_gib()
    rec['avail_GiB'] = av; rec['rss_GiB'] = PD.rss_gib()
    if rec['predicted_GiB']['total'] + A['headroom'] > av:
        P.release()
        rec['skipped'] = f"SKIP_PREDICTED {rec['predicted_GiB']['total']:.2f} GiB + headroom {A['headroom']} GiB > available {av:.2f} GiB"
        _log(rec)
        raise MemoryError(rec['skipped'])
    rec['factor_s'] = P.phase(22)
    b = np.asfortranarray(F[fr] * s[:, None])
    t = time.perf_counter(); x = P.solve(b); rec['solve_s'] = time.perf_counter() - t
    rec['pardiso'] = P.info(); rec['peak_rss_GiB'] = PD.peak_rss_gib()
    P.release()
    res = (PD.sym_upper_matvec(U, x) - b) / s[:, None]
    rec['rel_residual'] = [float(np.linalg.norm(res[:, j]) / np.linalg.norm(F[fr][:, j])) for j in range(F.shape[1])]
    _log(rec)
    u = np.zeros_like(F); u[fr] = x * s[:, None]
    return torch.as_tensor(u, dtype=RV.dt)


def _log(rec):
    if A['log']:
        with open(A['log'], 'a') as f:
            f.write(json.dumps(rec, default=float) + '\n')
    print(json.dumps(dict(solve={k: rec[k] for k in ('case', 'n', 'free', 'analysis_s', 'factor_s', 'solve_s', 'skipped')
                                 if k in rec}), default=float), flush=True)


RV.solve = solve                                                         # respond() and the direct fd check use it


def edit(out, fn):
    p = Path(out)
    rec = json.loads(p.read_text()) if p.exists() else dict(setup='clamp one box face, unit consistent traction (x, y, z) on another (per case: faces); host PARDISO', per_case={})
    fn(rec)
    p.write_text(json.dumps(rec, indent=1))


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('cases')
    ap.add_argument('--ns', default='24,32,40,48,56,64'); ap.add_argument('--stage', default='all', choices=['all', 'sweep', 'studies'])
    ap.add_argument('--headroom-gib', type=float, default=25.0); ap.add_argument('--seed', default=None)
    ap.add_argument('--retry-errors', action='store_true')
    a, rest = ap.parse_known_args(argv)
    A['headroom'] = a.headroom_gib
    A['log'] = str(Path(a.out).with_suffix('')) + '_solves.jsonl'
    cases = a.cases.split(',')

    def prep(rec):
        rec.setdefault('r1', dict(solver='PARDISO Cholesky (mtype 2), scaled upper triangle, direct calls',
                                  iparm={str(k): v for k, v in A['iparm'].items()}, env=PD.env_record()))
        if a.seed:
            src = json.loads(Path(a.seed).read_text())['per_case']
            for c in cases:
                if c in src and c not in rec['per_case']:
                    rec['per_case'][c] = src[c]
                    rec['per_case'][c]['seeded_from'] = a.seed
        for c in cases:
            r = rec['per_case'].get(c)
            if r and a.retry_errors:
                for n in list(r.get('n', {})):
                    if 'error' in r['n'][n] or 'missing' in r['n'][n]:
                        del r['n'][n]
    edit(a.out, prep)

    def placeholders(on):
        def f(rec):
            for c in cases:
                r = rec['per_case'].setdefault(c, {})
                for k in ('gamma', 'fd', 'integ'):
                    if on and k not in r:
                        r[k] = PLACE
                    if not on and r.get(k) == PLACE:
                        del r[k]
        edit(a.out, f)
    base = [a.out, None, '--ns', a.ns] + rest
    if a.stage in ('all', 'sweep'):
        placeholders(True)
        for c in cases:
            base[1] = c; RV.main(list(base)); gc.collect()
        placeholders(False)
    if a.stage in ('all', 'studies'):
        for c in cases:
            base[1] = c; RV.main(list(base)); gc.collect()


if __name__ == '__main__':
    main(sys.argv[1:])
