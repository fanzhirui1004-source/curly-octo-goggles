"""P2: how much does an element-level stiffness error move the condensed boundary operator? CPU only.

Rebuild the body stiffness from the archived per-element matrices (checked against the archived assembled
K), perturb every cut element's 81x81 matrix in a controlled way, keep the ghost penalty exact, apply the
same trace compiler P, condense exactly with a sparse direct solve, and compare with the teacher through
the original quotient and teacher factor R: mu = eig(R^-T B S' B^T R^-1).

Perturbation families (fixed seed, every element independently):
  spectral : K_e' = K_e^{1/2} (I + eps E) K_e^{1/2},  E symmetric, ||E||_2 = 1  (Loewner-bounded; theory: mu in [1-eps,1+eps])
  entry    : K_e' = K_e o (1 + eps Z),  Z symmetric N(0,1)  (per-entry relative)
  frob     : K_e' = K_e + eps ||K_e||_F Z / ||Z||_F  (Frobenius-relative, the usual regression error)
Teacher-side only. No network, no training, no change to any archived file.
"""
import argparse, hashlib, json, sys, tarfile, time
from pathlib import Path
import numpy as np
from scipy import linalg, sparse
from scipy.sparse.linalg import splu

ROOT = Path('/root/autodl-tmp/CUTFEM_FRESH_GP_20260921')


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 23), b''):
            h.update(b)
    return h.hexdigest()


def csr_read(d, prefix, n):
    return sparse.csr_matrix(tuple(np.load(d / f'{prefix}_{k}.npy', mmap_mode='r') for k in ('data', 'indices', 'indptr')), shape=(n, n))


def symmetric(u):
    return (u + u.T - sparse.diags(u.diagonal())).tocsr()


def extract(case, work):
    src = ROOT / 'source_archives' / case
    inv = json.loads((src / 'INVENTORY.json').read_text())
    g = next(s for s in inv['stages'] if s.endswith('_G'))
    o = next(s for s in inv['stages'] if s.endswith('_O'))
    body = [f'{g}/body/{n}' for n in ('canonical_K.npy', 'canonical_ids.npy', 'reflections.npy', 'dofs.npy', 'NODES.npy',
                                      'K_data.npy', 'K_indices.npy', 'K_indptr.npy', 'CELL_INDICES.npy', 'V_data.npy')]
    ghost = [f'{o}/ghost/{n}' for n in ('K_data.npy', 'K_indices.npy', 'K_indptr.npy', 'global_support.npy')]
    with tarfile.open(src / 'SOURCE.tar', 'r:') as t:
        for name in body + ghost:
            dest = work / name
            if dest.exists():
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            with t.extractfile(t.getmember(name)) as s, open(dest, 'wb') as f:
                f.write(s.read())
            if sha(dest) != inv['files'][name]['sha256']:
                raise ValueError('ARCHIVE_SHA:' + name)
    return work / g / 'body', work / o / 'ghost'


def assemble(Ke, dofs, nb):
    r = np.repeat(dofs, 81, axis=1).ravel()
    c = np.tile(dofs, (1, 81)).ravel()
    return sparse.csr_matrix((Ke.ravel(), (r, c)), shape=(nb, nb))


def moment_model(bdir, n, lam, mu_):
    """Exact 125-moment form of every element matrix (see element_moments.py): K_e = sum_m M_em T_m."""
    import element_moments as EM
    arrays = {k: np.load(bdir / k) for k in ('NODES.npy', 'dofs.npy', 'CELL_INDICES.npy', 'V_data.npy')}
    xi = EM.local_coordinates(None, n, arrays)
    Vf = arrays['V_data.npy']; V = np.einsum('eki,ekj->eij', Vf, Vf)
    keys, inverse = np.unique(xi.reshape(len(V), -1), axis=0, return_inverse=True); inverse = inverse.ravel()
    M = np.empty((len(V), 125)); T = {}
    for k in range(len(keys)):
        Lv, T[k] = EM.pattern_operators(keys[k].reshape(27, 3), lam, mu_, n)
        sel = np.flatnonzero(inverse == k)
        M[sel] = np.linalg.lstsq(Lv, V[sel].reshape(len(sel), -1).T, rcond=None)[0].T
    return M, T, inverse


def from_moments(M, T, inverse):
    K = np.empty((len(M), 81, 81))
    for k, Tk in T.items():
        sel = np.flatnonzero(inverse == k)
        K[sel] = np.einsum('em,mij->eij', M[sel], Tk)
    return K


def nnls_moments(M, p, eps, rng):
    import element_moments as EM
    from scipy.optimize import nnls
    from itertools import product as prod
    x = EM.gll(int(p)); pts = np.array(list(prod(x, x, x)))
    mono = np.array([pts[:, 0] ** i * pts[:, 1] ** j * pts[:, 2] ** k for i, j, k in prod(range(5), repeat=3)])
    out = np.empty_like(M)
    for e in range(len(M)):
        w, _ = nnls(mono, M[e] / M[e, 0], maxiter=50 * len(pts))
        if eps:
            w = w * (1 + eps * rng.standard_normal(len(w)))
        out[e] = mono @ w * M[e, 0]
    return out


def perturb(Ke, family, eps, rng):
    out = np.empty_like(Ke)
    for i, K in enumerate(Ke):
        Z = rng.standard_normal((81, 81)); Z = (Z + Z.T) / np.sqrt(2)
        if family == 'spectral':
            w, V = np.linalg.eigh(K)
            R = (V * np.sqrt(np.clip(w, 0, None))) @ V.T
            E = Z / np.linalg.norm(Z, 2)
            out[i] = R @ (np.eye(81) + eps * E) @ R
        elif family == 'entry':
            out[i] = K * (1 + eps * Z)
        elif family == 'frob':
            out[i] = K + eps * np.linalg.norm(K) * Z / np.linalg.norm(Z)
        else:
            raise ValueError(family)
        out[i] = (out[i] + out[i].T) / 2
    return out


def main(a):
    case = a.case
    out = Path(a.output); out.mkdir(parents=True, exist_ok=False)
    work = Path(a.work); work.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    bdir, gdir = extract(case, work)
    packet = ROOT / 'packets' / case
    sample = json.loads((packet / 'SAMPLE.json').read_text())
    m = sample['full_trace_dimension']; gamma = float(sample['gp']['gamma'])
    nodes = np.load(bdir / 'NODES.npy'); nb = 3 * len(nodes)
    Ke = np.load(bdir / 'canonical_K.npy'); ids = np.load(bdir / 'canonical_ids.npy'); refl = np.load(bdir / 'reflections.npy')
    dofs = np.load(bdir / 'dofs.npy')
    if not (np.array_equal(ids, np.arange(len(Ke))) and not refl.any()):
        raise ValueError('CANONICAL_VIEW_NOT_IDENTITY: element map needs reflections; not handled here')
    upper = csr_read(bdir, 'K', nb)
    Kbody = symmetric(upper)
    Kre = assemble(Ke, dofs, nb)
    rebuild = float(sparse.linalg.norm(Kre - Kbody) / sparse.linalg.norm(Kbody))
    support = np.load(gdir / 'global_support.npy'); gu = csr_read(gdir, 'K', len(support))
    counts = np.zeros(nb, dtype=np.int64); counts[support] = np.diff(gu.indptr)
    emb = sparse.csr_matrix((gu.data, support[gu.indices], np.r_[0, np.cumsum(counts)]), shape=(nb, nb))
    G = symmetric(emb)
    P = sparse.load_npz(packet / 'ORIGINAL_FROM_TRACE_FREE.npz').tocsr()
    target = ROOT / 'targets' / (case + '_v1')
    R = np.load(target / 'REFERENCE_RQ.npy'); rig = np.load(target / 'RIGID_Q.npy')
    d = R.shape[0]
    # Householder quotient identical to run_mechanics.Quotient, in numpy.
    a_ = np.array(rig, copy=True); refl_v = []
    for k in range(6):
        v = a_[k:, k].copy(); al = -np.copysign(np.linalg.norm(v), v[0]); v[0] -= al; v /= np.linalg.norm(v)
        a_[k:, k:] -= 2 * np.outer(v, v @ a_[k:, k:]); refl_v.append(v)
    def reduce(X):
        Y = X.copy()
        for k in range(6):
            v = refl_v[k]; Y[k:] -= 2 * np.outer(v, v @ Y[k:])
        return Y[6:]
    cutmask = None
    rec = dict(case=case, elements=int(len(Ke)), body_dofs=nb, q=m, physical=d, gamma=gamma,
               element_rebuild_relative=rebuild, seed=a.seed, solver=a.solver, families={})
    print(json.dumps(dict(event='LOADED', rebuild=rebuild, elements=len(Ke), q=m, seconds=time.perf_counter() - t0)), flush=True)
    if rebuild > 1e-12:
        raise ValueError('ELEMENT_REBUILD_MISMATCH')

    def spectrum(Kb):
        K = (Kb + gamma * G).tocsr()
        Kc = (P.T @ K @ P).tocsr()
        Kc = symmetric(sparse.triu(Kc, format='csr'))
        D = Kc[:m, :m].toarray(); C = Kc[m:, :m].tocsc(); A = Kc[m:, m:].tocsc()
        s = 1 / np.sqrt(A.diagonal())
        if a.solver == 'splu':
            As = (sparse.diags(s) @ A @ sparse.diags(s)).tocsc()
            f = splu(As, permc_spec='MMD_AT_PLUS_A', diag_pivot_thresh=0., options={'SymmetricMode': True})
            X = s[:, None] * f.solve(s[:, None] * C.toarray())
            S = D - C.T @ X
        else:
            # Same scaled SPD factorisation Codex used for the native qualification; C^T A^-1 C in column blocks,
            # two steps of iterative refinement per block as in validate_saved_native.
            from stage_cutfem_solver.pardiso import PardisoSPD
            upper = sparse.triu(A, format='csr'); upper.data *= np.repeat(s, np.diff(upper.indptr)) * s[upper.indices]
            Ar = A.tocsr(); Cr = C.tocsc(); S = D.copy()
            with PardisoSPD(upper, threads=a.threads) as f:
                if f.iparm[13] or f.iparm[29]:
                    raise ValueError('NATIVE_PERTURBED_OR_NONPOSITIVE_PIVOT')
                for lo in range(0, m, a.block):
                    hi = min(m, lo + a.block)
                    rhs = Cr[:, lo:hi].toarray()
                    X = s[:, None] * f.solve(s[:, None] * rhs)
                    for _ in range(2):
                        X += s[:, None] * f.solve(s[:, None] * (rhs - Ar @ X))
                    S[:, lo:hi] -= Cr.T @ X
        S = (S + S.T) / 2
        Wq = reduce(reduce(S).T)
        W = linalg.solve_triangular(R.T, linalg.solve_triangular(R.T, Wq, lower=True).T, lower=True)
        W = (W + W.T) / 2
        mu = linalg.eigvalsh(W, driver='evd')
        return mu

    ctxf = json.loads((packet / 'FRESH_CONTEXT.json').read_text())
    E_, nu_ = float(ctxf['material']['E']), float(ctxf['material']['nu'])
    Mom, Tm, orde = moment_model(bdir, int(ctxf['n']), E_ * nu_ / ((1 + nu_) * (1 - 2 * nu_)), E_ / (2 * (1 + nu_)))
    mrel = float(np.linalg.norm(from_moments(Mom, Tm, orde) - Ke) / np.linalg.norm(Ke))
    rec['moment_form_relative'] = mrel
    print(json.dumps(dict(event='MOMENT_FORM', relative=mrel)), flush=True)
    if mrel > 1e-10:
        raise ValueError('MOMENT_FORM_MISMATCH')
    # Element classes: macro-plane cut (plane changes sign over the cell box), TPMS-only cut, full.
    cells = np.load(bdir / 'CELL_INDICES.npy'); nn = int(ctxf['n'])
    corners = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)])
    if ctxf['case'].get('normal') is not None:
        nrm = np.array([float(v) for v in ctxf['case']['normal']]); off = float(ctxf['case']['offset'])
        sv = ((cells[:, None, :] + corners[None]) / nn) @ nrm - off
        plane_cut = (sv.min(1) < 0) & (sv.max(1) > 0)
    else:
        plane_cut = np.zeros(len(cells), bool)
    full = Mom[:, 0] * nn ** 3 > 1 - 1e-12
    vfrac = Mom[:, 0] * nn ** 3
    classes = dict(plane=plane_cut, tpms=~plane_cut & ~full, full=full & ~plane_cut, all=np.ones(len(cells), bool),
                   small=vfrac < 1e-3, large=vfrac >= 1e-3)
    rec['classes'] = {k: int(v.sum()) for k, v in classes.items()}
    subset = classes[a.subset]
    print(json.dumps(dict(event='CLASSES', subset=a.subset, **rec['classes'])), flush=True)
    t = time.perf_counter(); mu0 = spectrum(Kbody)
    rec['exact'] = dict(mu_min=float(mu0.min()), mu_max=float(mu0.max()), seconds=time.perf_counter() - t)
    print(json.dumps(dict(event='EXACT', **rec['exact'])), flush=True)
    rng_master = np.random.default_rng(a.seed)
    for fam in a.families:
        for eps in a.eps:
            rng = np.random.default_rng(rng_master.integers(1 << 62))
            t = time.perf_counter()
            if fam == 'moment':  # relative error on every one of the 125 moments; rigid kernel stays exact
                Kp = from_moments(Mom * (1 + eps * rng.standard_normal(Mom.shape)), Tm, orde)
            elif fam == 'moment_vol':  # error eps on the volume-normalised moments M/M000 (network-like)
                Kp = from_moments(Mom + eps * Mom[:, :1] * rng.standard_normal(Mom.shape), Tm, orde)
            elif fam == 'replace':  # moments from a file (e.g. POLYREF_S4.npz), same element order
                Kp = from_moments(np.load(a.replace_moments)['moments'], Tm, orde)
            elif fam.startswith('nnls'):  # nonnegative fixed-grid (GLL order in the name) measure, weights * (1+eps z)
                Kp = from_moments(nnls_moments(Mom, int(fam[4:]), eps, rng), Tm, orde)
            else:
                Kp = perturb(Ke, fam, eps, rng)
            Kp = np.where(subset[:, None, None], Kp, Ke)
            try:
                mu = spectrum(assemble(Kp, dofs, nb))
            except (ValueError, RuntimeError) as exc:  # perturbed interior block singular or not positive definite
                r = dict(eps=eps, error=str(exc), seconds=time.perf_counter() - t)
                rec['families'].setdefault(fam, []).append(r)
                print(json.dumps(dict(event='VARIANT_FAILED', family=fam, **r)), flush=True)
                continue
            neg = int((mu <= 0).sum())
            pos = mu[mu > 0]
            r = dict(eps=eps, mu_min=float(mu.min()), mu_max=float(mu.max()), nonpositive=neg,
                     outside_work=int(((mu < .9) | (mu > 1.1)).sum()), outside_target=int(((mu < .97) | (mu > 1.03)).sum()),
                     D_over_d=float(np.mean(pos - 1 - np.log(pos))) if neg == 0 else None,
                     within_theory_band=bool(mu.min() >= 1 - eps - 1e-9 and mu.max() <= 1 + eps + 1e-9),
                     seconds=time.perf_counter() - t)
            rec['families'].setdefault(fam, []).append(r)
            print(json.dumps(dict(event='VARIANT', family=fam, **r)), flush=True)
            np.save(out / f'MU_{fam}_{eps:g}_{a.subset}.npy', mu)
            if fam == 'replace':
                break
    rec['seconds'] = time.perf_counter() - t0
    (out / 'RESULT.json').write_text(json.dumps(rec, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--case', required=True); p.add_argument('--output', required=True); p.add_argument('--work', required=True)
    p.add_argument('--seed', type=int, default=20260923)
    p.add_argument('--families', nargs='+', default=['spectral', 'entry', 'frob'])
    p.add_argument('--eps', type=float, nargs='+', default=[0.01, 0.03, 0.10])
    p.add_argument('--solver', choices=['splu', 'pardiso'], default='splu')
    p.add_argument('--subset', choices=['all', 'plane', 'tpms', 'full', 'small', 'large'], default='all')
    p.add_argument('--replace-moments', default=None)
    p.add_argument('--threads', type=int, default=6)
    p.add_argument('--block', type=int, default=1024)
    a = p.parse_args()
    main(a)  # thread count set by OMP/MKL/OPENBLAS_NUM_THREADS in the launcher
