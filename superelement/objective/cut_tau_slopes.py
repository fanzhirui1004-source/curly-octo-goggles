"""Gate 2, the measurement: one-sided tau slopes of a CUT cell's response, teacher side only.

Reads the frozen base operator and the admitted perturbed operators produced by
`cut_tau_geometry.py`, and reports, for each observable, the left and right one-sided log-slopes,
their ratio, the central difference and the Richardson ratio.  The 0328 box-cell result this is
compared against: the FREE cell's compliance had one-sided log-slopes -6.09 and -21.36 at h = 1e-4,
a factor of 3.5, while the clamped-platen observable was smooth with a Richardson ratio of 0.999997.

Observables, all in the trace space the teacher exports:

  full_trace_inverse    trace(S^+) over the non-rigid subspace -- the 0328 analogue, and the one
                        that sees a near-mechanism wherever it is;
  condensed_inverse     trace(T^+) with T = S_BB - S_BC S_CC^-1 S_CB, the cut surface free.  This
                        is what a lattice reads from the cell, because the assembly glues on kind-0
                        and leaves kind-1 free;
  box_load_<k>          c(f) = f^T T^+ f for fixed loads supported on the box faces;
  full_load_<k>         c(f) = f^T S^+ f for fixed loads that also touch the cut coordinates;
  softest_nonrigid      the smallest non-rigid eigenvalue of S, so a near-mechanism appearing or
                        disappearing is visible directly rather than inferred.

Every load is projected onto the complement of the operator's own rigid nullspace before use, and
the same load vectors are used at every epsilon.
"""
from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path

import numpy as np


def unpack_upper(path, q):
    v = np.load(path, mmap_mode='r')
    if v.shape != (q * (q + 1) // 2,):
        raise ValueError(f'PACKED_UPPER_SHAPE {path} {v.shape} for q={q}')
    M = np.zeros((q, q))
    M[np.triu_indices(q)] = np.asarray(v, dtype=np.float64)
    M += M.T
    M[np.diag_indices(q)] *= 0.5
    return M


def pinv_quadratic(M, loads, drop=6):
    """f^T M^+ f for each load, and trace(M^+), both over the non-rigid subspace."""
    w, V = np.linalg.eigh(M)
    keep = w[drop:]
    if np.any(keep <= 0):
        raise ValueError(f'NON_POSITIVE_NONRIGID_SPECTRUM min={keep.min():.3e}')
    U = V[:, drop:]
    trace_inverse = float((1.0 / keep).sum())
    out = []
    for f in loads:
        c = U.T @ f
        out.append(float((c * c / keep).sum()))
    return trace_inverse, out, float(keep[0]), float(w[5]), float(keep[-1])


def condense(S, nb):
    from scipy.linalg import cho_factor, cho_solve
    Scc = np.array(S[nb:, nb:], order='F')
    Scb = np.array(S[nb:, :nb])
    cf = cho_factor(Scc, lower=True, check_finite=False)
    T = np.array(S[:nb, :nb]) - Scb.T @ cho_solve(cf, Scb, check_finite=False)
    return 0.5 * (T + T.T)


def slopes(values, epsilons):
    """One-sided and central log-slopes at each h for which both signs exist."""
    by = {round(e, 12): v for e, v in zip(epsilons, values)}
    base = by[0.0]
    out = {}
    lefts = sorted([e for e in by if e < 0], key=abs)
    rights = sorted([e for e in by if e > 0], key=abs)
    for e in lefts + rights:
        out[f'one_sided_h={abs(e):g}_{"left" if e < 0 else "right"}'] = (
            (np.log(by[e]) - np.log(base)) / e)
    for h in sorted({abs(e) for e in lefts} & {abs(e) for e in rights}):
        left = (np.log(base) - np.log(by[-h])) / h
        right = (np.log(by[h]) - np.log(base)) / h
        out[f'central_h={h:g}'] = 0.5 * (left + right)
        out[f'one_sided_ratio_h={h:g}'] = max(abs(left), abs(right)) / max(abs(min(abs(left), abs(right))), 1e-300)
        out[f'kink_relative_h={h:g}'] = abs(left - right) / max(abs(left) + abs(right), 1e-300)
    hs = sorted({abs(e) for e in lefts} & {abs(e) for e in rights})
    if len(hs) >= 2:
        fine, coarse = hs[0], hs[1]
        out['richardson_central_ratio'] = out[f'central_h={fine:g}'] / out[f'central_h={coarse:g}']
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seat', type=int, required=True)
    ap.add_argument('--geometry', type=Path, required=True, help='the GEO_<seat> directory')
    ap.add_argument('--numeric-root', type=Path, required=True, help='directory holding NUM_<seat>_<label> dirs')
    ap.add_argument('--manifest', type=Path, default=Path('/root/autodl-tmp/CLAUDE_LABELS_20260917/V2_LABELS.json'))
    ap.add_argument('--loads', type=int, default=8)
    ap.add_argument('--seed', type=int, default=20260920)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()

    row = [r for r in json.loads(a.manifest.read_text()) if int(r['seat']) == a.seat][0]
    sample = json.loads((Path(row['packet']) / 'SAMPLE.json').read_text())
    q = int(sample['full_trace_dimension'])
    nb = q - 3 * int(sample['additional_scalar_cut_coordinates'])
    selected = json.loads((a.geometry / 'SELECTED.json').read_text())['selected']

    cases = [(0.0, Path(row['packet']) / 'S_UPPER.npy', 'frozen packet')]
    for entry in selected:
        label = Path(entry['directory']).name
        path = a.numeric_root / f'NUM_{a.seat}_{label}' / 'operator' / 'full_operator' / 'S_UPPER.npy'
        if not path.exists():
            raise ValueError(f'MISSING_OPERATOR {path}')
        cases.append((float(Fraction(entry['epsilon'])), path, label))
    cases.sort(key=lambda c: c[0])

    rng = np.random.default_rng(a.seed)
    box_loads = [np.concatenate([rng.standard_normal(nb), np.zeros(q - nb)]) for _ in range(a.loads)]
    full_loads = [rng.standard_normal(q) for _ in range(a.loads)]

    series = {}
    per_case = []
    for eps, path, label in cases:
        S = unpack_upper(path, q)
        # project the loads onto the complement of this operator's own rigid nullspace
        w, V = np.linalg.eigh(S)
        rigid = V[:, :6]
        bl = [f - rigid @ (rigid.T @ f) for f in box_loads]
        fl = [f - rigid @ (rigid.T @ f) for f in full_loads]
        tr_full, c_full, soft, rigid_leak, stiff = pinv_quadratic(S, fl)
        T = condense(S, nb)
        wT, VT = np.linalg.eigh(T)
        rigidT = VT[:, :6]
        blT = [f[:nb] - rigidT @ (rigidT.T @ f[:nb]) for f in bl]
        tr_cond, c_box, softT, rigid_leakT, stiffT = pinv_quadratic(T, blT)
        record = dict(epsilon=eps, label=label, source=str(path),
                      full_trace_inverse=tr_full, condensed_inverse=tr_cond,
                      softest_nonrigid=soft, largest=stiff, rigid_residual=rigid_leak,
                      condensed_softest_nonrigid=softT, condensed_rigid_residual=rigid_leakT,
                      condensed_kappa=stiffT / softT, full_kappa=stiff / soft)
        for i, v in enumerate(c_box):
            record[f'box_load_{i}'] = v
        for i, v in enumerate(c_full):
            record[f'full_load_{i}'] = v
        per_case.append(record)
        del S, T
        print(json.dumps({k: record[k] for k in
                          ('epsilon', 'label', 'full_trace_inverse', 'condensed_inverse',
                           'softest_nonrigid', 'condensed_softest_nonrigid')}), flush=True)

    epsilons = [r['epsilon'] for r in per_case]
    names = ['full_trace_inverse', 'condensed_inverse', 'softest_nonrigid',
             'condensed_softest_nonrigid'] + \
            [f'box_load_{i}' for i in range(a.loads)] + [f'full_load_{i}' for i in range(a.loads)]
    for name in names:
        series[name] = slopes([r[name] for r in per_case], epsilons)

    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(dict(
        seat=a.seat, q=q, box_coordinates=nb, cut_coordinates=q - nb,
        epsilons=epsilons, cases=per_case, slopes=series,
        scope='gate 2 teacher side: one-sided tau slopes of a cut cell, no network anywhere',
        reference='0328 box cell, free: one-sided -6.09 vs -21.36 at h=1e-4; clamped platen '
                  'Richardson 0.999997',
        loads='fixed vectors, seed %d, projected onto each operator own non-rigid subspace' % a.seed,
    ), indent=1) + '\n')
    print(json.dumps(dict(phase='slopes_written', output=str(a.output))), flush=True)


if __name__ == '__main__':
    main()
