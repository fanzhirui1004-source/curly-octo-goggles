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
Body-load fallback: when ref_valid.pick_faces returns a load face without material (a heavily cut cell whose material touches
only one box face, e.g. H2 = fresh_val_2010_d0_v0), the clamp is kept on the face with the largest weight and the load
becomes a unit uniform BODY force in x, y, z with consistent Q2 nodal weights f_i = int N_i dV computed exactly from the
element moments (C.M, monomials up to degree 2 per axis on the reference cube); recorded as load face [-1, 0.0]. The load
vector is fixed at the base design, as for the face loads, so s_c = -u^T K_,c u and the direct fd check stay consistent.
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

_pick0, _loads0 = RV.pick_faces, RV.loads
L1D = np.array([[0.0, -0.5, 0.5], [1.0, 0.0, -1.0], [0.0, 0.5, 0.5]])      # Q2 Lagrange at -1, 0, 1: coeffs of 1, xi, xi^2


def pick_faces(C):
    (cl, ld), w = _pick0(C)
    if w[tuple(ld)] > 0:
        return (cl, ld), w
    cl = max(RV.FACES, key=lambda f: w[f])
    return (cl, (-1, 0.0)), w


def body_weights(C):
    """Consistent nodal weights of a unit body force: f_k = sum_e int_e N_j(e,k) dV (reference-cube moments, common scale)."""
    dofs = C.dofs.long().numpy()
    assert np.all(dofs[:, 1::3] == dofs[:, 0::3] + 1) and np.all(dofs[:, 2::3] == dofs[:, 0::3] + 2)
    k = dofs[:, 0::3] // 3                                                      # E x 27 local node index
    M1 = 2 * C.n + 1
    g = np.stack(np.unravel_index(np.asarray(C.nodes)[k], (M1,) * 3), -1)          # E x 27 x 3 grid coordinates
    xi = g - 2 * np.asarray(C.cells)[:, None, :] - 1
    assert xi.min() >= -1 and xi.max() <= 1
    M5 = C.M.numpy().reshape(-1, 5, 5, 5)[:, :3, :3, :3]
    cx, cy, cz = L1D[xi[..., 0] + 1], L1D[xi[..., 1] + 1], L1D[xi[..., 2] + 1]    # E x 27 x 3 each
    f = np.einsum('ejp,ejq,ejr,epqr->ej', cx, cy, cz, M5)
    assert np.allclose(f.sum(1), C.M.numpy()[:, 0], rtol=1e-9, atol=1e-14)
    w = np.zeros(len(C.nodes))
    np.add.at(w, k.reshape(-1), f.reshape(-1))
    return w


def loads(C, load=(2, 1.0)):
    if int(load[0]) != -1:
        return _loads0(C, load)
    w = body_weights(C)
    w = w / w.sum()
    F = np.zeros((C.nb, 3))
    for d in range(3):
        F[d::3, d] = w
    return F


RV.pick_faces, RV.loads = pick_faces, loads


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
