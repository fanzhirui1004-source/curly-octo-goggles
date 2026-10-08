"""1.6b: does the teacher's NUMERIC body stiffness commute with the cube group?

1.6a settled the combinatorial half (the active cells and nodes of the rotated geometry are the
group image of the original's).  The numeric half factors exactly, because the teacher's body
stiffness is assembled cell by cell from

    local_stiffness(moments, n) = tensordot(moments, coefficients(n))          [stage_cutfem_q2.space]

with `moments` the 125 monomial moments of the cell's solid region in natural coordinates.  So
body equivariance is the conjunction of two statements that can be tested separately:

  B1  ALGEBRA.  For any moments, local_stiffness of the ROTATED moments equals the rotated local
      stiffness: local(g . M) = (P_g (x) Q_g) local(M) (P_g (x) Q_g)^T, with P_g the permutation
      of the 27 Q2 nodes and Q_g the 3x3 signed permutation.  This tests the teacher's
      `coefficients(n)` tensor alone, needs no geometry, and is exact arithmetic on random
      moments -- a failure here would be a defect in the element itself.

  B2  QUADRATURE.  For the real geometry, the moments the quadrature produces for the rotated
      cell are the rotated moments of the original cell.  This tests the integration over the
      TPMS solid region, which is where a level-set/octree rule could break symmetry.

Under g the natural coordinate transforms as x'_i = sum_j Q[i,j] x_j = s_i x_{sigma(i)}, so a
monomial moment maps as M'[a] = (prod_i s_i^{a_i}) M[b] with b_{sigma(i)} = a_i.  B1 and B2
together give the assembled body operator's equivariance, since the scatter is combinatorial and
1.6a already showed the cell and node sets are the group image.

    python witness_b.py <CASE.json> [g ...]
"""
import json, os, sys, time
from fractions import Fraction
from pathlib import Path
import numpy as np

sys.path.insert(0, '/root/autodl-tmp/CLAUDE_EQUI_20260918/src')
from superelement.equi.cubic_group import Q_ALL, CORNER_PERM, DET, map_int_positions

from stage_cutfem_graded.contract import from_case
from stage_cutfem_q2.space import local_stiffness, OFFSETS
from stage_cutfem_q2.quadrature import axis_cell_moments

N = 32
POWERS = np.array([(a, b, c) for a in range(5) for b in range(5) for c in range(5)], dtype=np.int64)
POWER_INDEX = {tuple(p): i for i, p in enumerate(POWERS)}


def signed_permutation(g):
    """sigma, s with Q[i, sigma[i]] = s[i]; every element of O_h is a signed permutation."""
    Q = np.asarray(Q_ALL[g], dtype=np.int64)
    sigma = np.empty(3, dtype=np.int64); s = np.empty(3, dtype=np.int64)
    for i in range(3):
        j = int(np.flatnonzero(Q[i] != 0)[0]); sigma[i] = j; s[i] = int(Q[i, j])
    if not np.array_equal(Q, np.eye(3, dtype=np.int64)[sigma] * s[:, None]):
        raise ValueError('NOT_A_SIGNED_PERMUTATION')
    return sigma, s


def moment_map(g):
    """index_source[k] and sign[k] with M_rot[k] = sign[k] * M[index_source[k]]."""
    sigma, s = signed_permutation(g)
    src = np.empty(len(POWERS), dtype=np.int64); sign = np.empty(len(POWERS), dtype=np.int64)
    for k, a in enumerate(POWERS):
        b = np.zeros(3, dtype=np.int64)
        for i in range(3):
            b[sigma[i]] = a[i]
        src[k] = POWER_INDEX[tuple(b)]
        sign[k] = int(np.prod([s[i] ** int(a[i]) for i in range(3)]))
    return src, sign


def node_permutation_27(g):
    """The Q2 cell's 27 nodes: offset o (in {0,1,2}^3) goes to Q (o - 1) + 1."""
    off = np.asarray(OFFSETS, dtype=np.int64)
    if off.shape != (27, 3) or off.min() < 0 or off.max() > 2:
        raise ValueError(f'UNEXPECTED_Q2_OFFSETS {off.shape}')
    Q = np.asarray(Q_ALL[g], dtype=np.int64)
    image = (off - 1) @ Q.T + 1
    key = {tuple(v): i for i, v in enumerate(off)}
    return np.array([key[tuple(v)] for v in image], dtype=np.int64)     # perm[i] = index of image of node i


def conjugate(local, g):
    """(P (x) Q) local (P (x) Q)^T for the 81 x 81 Q2 local stiffness."""
    perm = node_permutation_27(g); Q = np.asarray(Q_ALL[g], dtype=np.float64)
    T = np.zeros((81, 81))
    for i in range(27):
        T[3 * perm[i]:3 * perm[i] + 3, 3 * i:3 * i + 3] = Q
    return T @ local @ T.T


def b1_algebra(elements, trials=6, seed=20260919):
    """Random valid moments: local(g . M) must equal the conjugated local(M)."""
    rng = np.random.default_rng(seed)
    worst = {}
    for g in elements:
        src, sign = moment_map(g)
        w = 0.0
        for _ in range(trials):
            # a positive-measure moment vector: integrate a random positive density on the cell
            pts = rng.uniform(-1, 1, size=(4096, 3)); wts = rng.uniform(.2, 1., size=4096) / 4096 * 8
            M = np.einsum('i,ia,ib,ic->abc', wts, *[np.stack([pts[:, d] ** k for k in range(5)], axis=1)
                                                    for d in range(3)], optimize=True).ravel()
            got = local_stiffness(sign * M[src], N)
            want = conjugate(local_stiffness(M, N), g)
            w = max(w, float(np.abs(got - want).max() / max(np.abs(want).max(), 1e-300)))
        worst[g] = w
    return worst


def permuted_case(case, g):
    row = dict(case)
    tau = list(case['tau_corners']); out = [None] * 8
    for c in range(8):
        out[CORNER_PERM[g][c]] = tau[c]
    row['tau_corners'] = out
    a = np.array([Fraction(v) for v in case['normal']], dtype=object)
    Qa = [sum(Fraction(int(Q_ALL[g][i, j])) * a[j] for j in range(3)) for i in range(3)]
    d = Fraction(case['offset']) + sum(Qa[i] - a[i] for i in range(3)) / 2
    row['normal'] = [str(v) for v in Qa]; row['offset'] = str(d)
    row['case_id'] = case['case_id'] + f'_g{g:02d}'
    corners = [(x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1)]
    row['_plane_inert'] = all(d - sum(Qa[i] * Fraction(p[i]) for i in range(3)) >= 0 for p in corners)
    return row


def b2_quadrature(case, elements, cells, level, order, axis):
    """The real integrals: the rotated geometry's moments at the image cell vs the rotated moments.

    `axis_cell_moments` integrates along one axis, so the same solid region integrated along x, y
    or z is three different floating-point paths to the same number.  That spread is measured
    first and separately: it is an intrinsic bound on the teacher's quadrature anisotropy that
    needs no rotation at all, and any rotation discrepancy below it is arithmetic, not asymmetry.
    """
    c0 = from_case(case, N)
    base, live = {}, []
    for ijk in cells:
        try:
            m = axis_cell_moments(c0, tuple(int(v) for v in ijk), level, order, axis)
        except Exception as error:
            continue
        m = np.asarray(m, dtype=np.float64)
        if m.shape != (125,) or not np.all(np.isfinite(m)) or m[0] <= 0:
            continue
        base[tuple(int(v) for v in ijk)] = m; live.append(tuple(int(v) for v in ijk))
    out = dict(level=level, order=order, axis=axis, probed_cells=len(cells), live_cells=len(live))
    if not live:
        out['failed'] = 'NO_LIVE_CELL_AMONG_THE_PROBES'; return out
    spread = []
    for ijk in live[:16]:
        ms = []
        for ax in range(3):
            try:
                ms.append(np.asarray(axis_cell_moments(c0, ijk, level, order, ax), dtype=np.float64))
            except Exception:
                ms = []; break
        if len(ms) == 3:
            scale = max(abs(ms[0][0]), 1e-300)
            spread.append(max(float(np.abs(ms[i] - ms[j]).max() / scale) for i in range(3) for j in range(i + 1, 3)))
    out['axis_spread'] = dict(cells=len(spread), median=float(np.median(spread)) if spread else None,
                              worst=float(max(spread)) if spread else None)
    print(json.dumps(dict(phase='B2_axis_spread', **out['axis_spread'])), flush=True)
    out['elements'] = {}
    for g in elements:
        row = permuted_case(case, g)
        if not row.pop('_plane_inert'):
            out['elements'][str(g)] = dict(skipped='rotated plane would cut the unit box'); continue
        cg = from_case(row, N)
        src, sign = moment_map(g)
        worst = 0.0; worst_cell = None; n_ok = 0; missing = 0
        for ijk in live:
            img = tuple(int(v) for v in map_int_positions(np.asarray(ijk, dtype=np.int64)[None, :], g, N - 1)[0])
            try:
                got = np.asarray(axis_cell_moments(cg, img, level, order, axis), dtype=np.float64)
            except Exception:
                missing += 1; continue
            want = sign * base[ijk][src]
            rel = float(np.abs(got - want).max() / max(abs(want[0]), 1e-300))
            if rel > worst:
                worst, worst_cell = rel, dict(cell=list(ijk), image=list(img))
            n_ok += 1
        out['elements'][str(g)] = dict(proper=bool(DET[g] > 0), cells=n_ok, unavailable=missing,
                                       worst_relative=worst, worst_cell=worst_cell)
        print(json.dumps({str(g): out['elements'][str(g)]}), flush=True)
    rows = [v for v in out['elements'].values() if 'skipped' not in v and v.get('cells')]
    out['worst_relative'] = max((v['worst_relative'] for v in rows), default=None)
    return out


def main():
    case = json.loads(Path(sys.argv[1]).read_text())
    elements = [int(x) for x in sys.argv[2:]] or [1, 5, 13, 25, 36, 43]
    report = dict(case=case['case_id'], n=N, elements=elements,
                  scope='1.6b: the teacher body stiffness under the cube group, algebra (B1) and quadrature (B2)')
    t0 = time.time()
    report['B1_algebra'] = {str(g): v for g, v in b1_algebra(list(range(48))).items()}
    report['B1_worst'] = max(report['B1_algebra'].values())
    report['B1_pass'] = bool(report['B1_worst'] < 1e-11)
    print(json.dumps(dict(phase='B1', elements=48, worst=report['B1_worst'], pass_=report['B1_pass'])), flush=True)

    rng = np.random.default_rng(7)
    probe = np.unique(rng.integers(4, N - 4, size=(400, 3)), axis=0)[:120]
    report['B2'] = b2_quadrature(case, elements, probe, level=int(os.environ.get('WB_LEVEL', 8)),
                                 order=int(os.environ.get('WB_ORDER', 8)), axis=int(os.environ.get('WB_AXIS', 2)))
    spread = (report['B2'].get('axis_spread') or {}).get('worst')
    worst = report['B2'].get('worst_relative')
    report['B2_pass'] = bool(worst is not None and spread is not None and worst <= max(1e-11, 10 * spread))
    report['B2_interpretation'] = ('the rotation discrepancy is within ten times the quadrature\'s own '
                                   'axis-to-axis spread, so it is arithmetic rather than asymmetry'
                                   if report['B2_pass'] else
                                   'the rotation discrepancy exceeds the quadrature\'s intrinsic spread')
    report['body_numeric_is_equivariant'] = bool(report['B1_pass'] and report['B2_pass'])
    report['seconds'] = time.time() - t0
    print(json.dumps(report, indent=1), flush=True)
    out = os.environ.get('WITNESS_B_OUT')
    if out:
        Path(out).write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
