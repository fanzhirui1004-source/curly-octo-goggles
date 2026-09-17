#!/usr/bin/env python3
"""Does a per-parameter relative error on the collar's core material map 1:1 to eps_op?

This is the claim that decides whether the 23,016-number route is worth pursuing.
WHERE_THE_ERROR_IS measured eps_op = 104 * eps for the dense factor's 8.2e7 entries;
FIRST_PRINCIPLES asserts (without full measurement) that for a LOCAL MULTIPLICATIVE
parameterisation the mapping is 1:1, because the Schur complement is Loewner-monotone
and positively homogeneous in the element energies. If that holds, the collar route
needs a few percent per number instead of 3e-4 - a ~100x looser target on 3,500x
fewer numbers.

Construction (all artefacts from CUTFEM_B1B2_ROUND2_20260913, nothing re-derived):
  K_COLLAR1.npz        62,088 x 62,088 assembled collar + coarse core, TRUE material,
                       zero learned parameters (ASSEMBLY.json: parameters = 0)
  ENERGY_COEFFICIENTS  (1096, 21, 24, 24): core cell c's stiffness is
                       sum_p theta[c,p] * E[c,p], scattered to CORE_DOFS[c]
  CORE_DOFS            (1096, 24)
  trace 12,798 dofs, internal 49,290

Perturb theta by independent relative noise eps, re-assemble, re-condense, whiten
against the teacher's factor, report eps_op(eps).

SELF-TEST: at eps = 0 the spectrum must reproduce the recorded zero-learning result
mu in [1.000, 1.855] (REVIEW_20260916 table). mu >= 1 is a theorem here - the collar
space is a subspace of the fine space carrying the same energy - so mu_min < 1 means
the dof ordering or the assembly is wrong and the run is void.
"""
import json, time, sys
from pathlib import Path
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spl
import torch

D = Path('/root/autodl-tmp/CUTFEM_B1B2_ROUND2_20260913')
OUT = Path('/root/autodl-tmp/CLAUDE_AMP_20260917'); OUT.mkdir(exist_ok=True)
sys.path.insert(0, '/root/cutfem_neural_a_20260910/source_14301bc56')
sys.path.insert(0, '/root/cutfem_neural_a_20260910/superelement_v0')
sys.path.insert(0, '/root/autodl-tmp/NEURAL_SCHUR')
import v1_scaled as V
from stage_cutfem_neural_a.elimination_reference import load_upper_factor
DEV = V.DEV; F64 = torch.float64
NT = 12798


def condense(K, nt):
    """S = K_tt - K_ti K_ii^-1 K_it, interior eliminated by one sparse factorisation."""
    K = K.tocsc()
    Ktt = K[:nt, :nt].toarray()
    Kti = K[:nt, nt:].tocsc()
    Kii = K[nt:, nt:].tocsc()
    t = time.time(); lu = spl.splu(Kii); fact = time.time() - t
    X = np.empty((Kii.shape[0], nt))
    B = Kti.T.toarray()                      # (ni, nt)
    step = 2048
    t = time.time()
    for j in range(0, nt, step):
        X[:, j:j + step] = lu.solve(B[:, j:j + step])
    solve = time.time() - t
    S = Ktt - B.T @ X
    return 0.5 * (S + S.T), fact, solve


def to_quotient(St, quotient):
    """A = B S B^T, the exact inverse of assemble.py::dense_S.

    Probed API (RigidQuotient): `lift` is (d,n)->(q,n); `project(x) = lift(self(x))`
    is (q,n)->(q,n), an orthogonal projection, NOT the reduction. The reduction is
    `quotient(x)` itself, (q,n)->(d,n). So apply the reduction twice."""
    left = quotient(St)                              # (q,q) -> (d,q)
    A = quotient(left.T.contiguous())                # (q,d) -> (d,d)
    return 0.5 * (A + A.T)


def check_quotient(quotient, Rs, d):
    """Instant round trip: A -> S -> A must be the identity on the quotient."""
    A = Rs.T @ Rs
    left = quotient.lift(A.T.contiguous())
    S = quotient.lift(left.T.contiguous()); S = 0.5 * (S + S.T)
    back = to_quotient(S, quotient)
    rel = float((back - A).norm() / A.norm())
    print(f'  quotient round trip A->S->A relative error {rel:.3e} '
          f'{"OK" if rel < 1e-12 else "FAIL"}', flush=True)
    if rel >= 1e-12:
        raise RuntimeError('QUOTIENT_ROUND_TRIP_FAILED')
    return rel


def spectrum(S, quotient, Rs, d):
    St = torch.as_tensor(S, dtype=F64, device=DEV)
    A = to_quotient(St, quotient); del St
    C = torch.linalg.solve_triangular(Rs.T, A, upper=False); del A
    C = torch.linalg.solve_triangular(Rs.T, C.T.contiguous(), upper=False)
    mu = torch.linalg.eigvalsh(0.5 * (C + C.T)); del C
    return float(mu.min()), float(mu.max())


def main():
    t0 = time.time()
    print('loading artefacts...', flush=True)
    K0 = sp.load_npz(D / 'B_ZERO_R1' / 'K_COLLAR1.npz').tocsr()
    E = np.load(D / 'B_CORE_COMPILE_R1' / 'ENERGY_COEFFICIENTS.npy')      # (1096,21,24,24)
    CD = np.load(D / 'B_CORE_COMPILE_R1' / 'CORE_DOFS.npy')               # (1096,24)
    print(f'  K {K0.shape} nnz {K0.nnz}   E {E.shape}   CORE_DOFS {CD.shape}', flush=True)
    assert K0.shape[0] == 62088 and E.shape[:2] == (1096, 21)

    recs = json.load(open('/root/autodl-tmp/CUTFEM_SUPERELEMENT_LABELS/V1_LABELS.json'))
    rec = [r for r in recs if int(r['seat']) == 328][0]
    rec.setdefault('reference', '/root/autodl-tmp/CUTFEM_SUPERELEMENT_LABELS/REFERENCE_0328')
    label = V.Label(rec, 0.2, 0.03, 10.0)
    g = label.to_gpu(need_A=False, need_Z=False, z_dtype=torch.float32)
    d, q = label.d, label.q
    quotient = g['data'].quotient
    Rs = load_upper_factor(Path(rec['reference']) / 'R_UPPER.npy', d, DEV)
    print(f'  teacher d={d} q={q}', flush=True)
    check_quotient(quotient, Rs, d)

    # theta_true: the coefficients already baked into K0. The perturbation is applied as
    # a DELTA, so theta_true itself is never needed - only its relative size, which is 1.
    rng = np.random.default_rng(20260917)
    out = dict(seat=328, core_cells=int(E.shape[0]), parameters=int(E.shape[0] * E.shape[1]),
               rows=[])
    for eps in (0.0, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1):
        t = time.time()
        K = K0.copy()
        if eps > 0:
            noise = rng.normal(0.0, eps, size=E.shape[:2])                # (1096,21)
            blocks = np.einsum('cp,cpij->cij', noise, E)                  # (1096,24,24)
            r = np.repeat(CD[:, :, None], 24, axis=2).ravel()
            c = np.repeat(CD[:, None, :], 24, axis=1).ravel()
            K = (K + sp.coo_matrix((blocks.ravel(), (r, c)), shape=K.shape)).tocsr()
        cache = OUT / 'S_eps0.npy'
        if eps == 0.0 and cache.exists():
            S = np.load(cache); fa = so = 0.0
            print('  reusing cached eps=0 condensation', flush=True)
        else:
            S, fa, so = condense(K, NT)
            if eps == 0.0:
                np.save(cache, S)
        mn, mx = spectrum(S, quotient, Rs, d)
        row = dict(entry_relative_noise=eps, mu_min=mn, mu_max=mx,
                   eps_op=max(abs(mn - 1), abs(mx - 1)),
                   factor_seconds=fa, solve_seconds=so, seconds=time.time() - t)
        out['rows'].append(row)
        print(f'  eps={eps:<7.0e} mu=[{mn:.6f},{mx:.6f}]  eps_op={row["eps_op"]:.6f}'
              f'  ({row["seconds"]:.0f}s)', flush=True)
        if eps == 0.0:
            ok = abs(mn - 1.0) < 5e-3 and abs(mx - 1.855) < 0.05
            print(f'  SELF-TEST vs recorded zero-learning mu=[1.000,1.855]: '
                  f'{"PASS" if ok else "FAIL"}', flush=True)
            out['self_test_pass'] = bool(ok)
            if not ok:
                print('  ABORTING: assembly or ordering convention is wrong; '
                      'perturbation results would be meaningless.', flush=True)
                (OUT / 'RESULT.json').write_text(json.dumps(out, indent=1)); return
        (OUT / 'RESULT.json').write_text(json.dumps(out, indent=1))
    print(f'\ndone ({time.time()-t0:.0f} s)', flush=True)


if __name__ == '__main__':
    main()
