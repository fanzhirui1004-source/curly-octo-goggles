"""1.6e: WHY the raw point-value basis is ill-conditioned (cond 1.1e7 against the teacher's 5) and
which covariant construction is not.  Facts first, on the same seat:

  1. the raw point functionals AT THE TEACHER'S OWN selected points: condition number;
  2. per cut cell: polygon area, picks, and where the tiny pivots of the greedy come from
     (sliver polygons or lattice clustering);
  3. candidate lattices of degree 2, 3, 4, 6 on the centroid fan: span, raw cond, tie margins,
     AND the unit-pivot residual basis in the greedy order (the teacher's normalisation applied
     to our covariant pick order): its condition number.

    python witness_e.py <seat>
"""
import json, os, sys, time
from fractions import Fraction
from pathlib import Path
import numpy as np
import scipy.sparse as sp
sys.path.insert(0, '/root')
from _wd import (case_from_seat, compile_one, dense_rows, greedy, subspace_residual, N,
                 polytope, canonical, box_planes, exact_basis_row)


def lattice(tri, k):
    a, b, c = tri
    return [tuple((i * a[d] + j * b[d] + (k - i - j) * c[d]) / k for d in range(3)) for i in range(k + 1) for j in range(k + 1 - i)]


def tri_area(tri):
    a, b, c = [np.array([float(x) for x in v]) for v in tri]
    return float(np.linalg.norm(np.cross(b - a, c - a)) / 2)


def candidates(got, degree, mode='fan'):
    c = got['contract']; n = c.n
    positions = {int(node): i for i, node in enumerate(got['nodes'])}
    normal, offset = c.rational_plane()
    points, cell_of, area_of = {}, {}, {}
    for patch in got['topo']['patches']:
        if patch['tag'] != 'MACRO_CUT_FACE':
            continue
        ijk = tuple(int(v) for v in patch['parent'])
        ambient = polytope(canonical(box_planes(ijk, n) + [(normal, offset, 'MACRO_CUT_FACE')]))
        poly = [tuple(Fraction(x) for x in v) for v in
                [polygon for _, _, tag, polygon in ambient['faces'] if 'MACRO_CUT_FACE' in tag.split('|')][0]]
        m = len(poly)
        centroid = tuple(sum(v[d] for v in poly) / m for d in range(3))
        tris = [(centroid, poly[i], poly[(i + 1) % m]) for i in range(m)]
        area_of[ijk] = sum(tri_area(t) for t in tris)
        if mode == 'maxtri':                                  # covariant only if the max is unique
            tris = [max(tris, key=tri_area)]
        for tri in tris:
            for p in lattice(tri, degree):
                if p in points:
                    continue
                points[p] = {int(k): float(v) for k, v in exact_basis_row(p, ijk, n, positions).items()}
                cell_of[p] = ijk
    keys = sorted(points)
    return keys, [points[k] for k in keys], [cell_of[k] for k in keys], area_of


def rows_at_points(got, pts):
    """Raw Q2 point rows at given exact points, evaluated in an active cut cell containing them."""
    c = got['contract']; n = c.n
    positions = {int(node): i for i, node in enumerate(got['nodes'])}
    cells = [tuple(int(v) for v in patch['parent']) for patch in got['topo']['patches'] if patch['tag'] == 'MACRO_CUT_FACE']
    rows = []
    for p in pts:
        found = None
        for ijk in cells:
            if all(Fraction(ijk[d], n) <= p[d] <= Fraction(ijk[d] + 1, n) for d in range(3)):
                found = ijk; break
        if found is None:
            raise ValueError('TEACHER_POINT_OUTSIDE_CUT_CELLS')
        rows.append({int(k): float(v) for k, v in exact_basis_row(p, found, n, positions).items()})
    return rows


def unit_pivot_residuals(Z, sel):
    """Eliminate the picks sequentially (teacher style), normalise each residual to unit max |coef|."""
    basis = []
    for i in sel:
        r = Z[i].copy()
        for b in basis:
            piv = int(np.argmax(np.abs(b)))
            r -= r[piv] * b                                  # b has +-1 at its pivot: kill that column
        basis.append(r / r[np.argmax(np.abs(r))])
    return np.array(basis)


def evaluate(label, rows, cells, area_of, T1d, cols_of, boxcol, kind1, keys=None):
    t0 = time.perf_counter()
    A = dense_rows(rows, cols_of, T1d.shape[1])
    Z = A.copy(); Z[:, boxcol] = 0.0
    sel, margins, pivnorm, left = greedy(Z.copy(), kind1)
    O1 = Z[sel]
    fwd, _ = subspace_residual(T1d, O1); rev, _ = subspace_residual(O1, T1d)
    so = np.linalg.svd(O1, compute_uv=False)
    out = dict(label=label, candidates=len(rows), picked=len(sel), left=left, span=dict(ours_onto_teacher=fwd, teacher_onto_ours=rev),
               raw_cond=float(so[0] / so[-1]), pivot_last_over_first=float(pivnorm[-1]),
               margin_min=float(min(margins)), margins_below_1e6=int(sum(m < 1e-6 for m in margins)),
               margins_below_1e3=int(sum(m < 1e-3 for m in margins)))
    if len(sel) == kind1:
        R = unit_pivot_residuals(Z, sel)
        sr = np.linalg.svd(R, compute_uv=False)
        out['unit_pivot_residual_cond'] = float(sr[0] / sr[-1])
        out['unit_pivot_residual_nnz'] = dict(mean=float((np.abs(R) > 1e-12).sum(1).mean()), max=int((np.abs(R) > 1e-12).sum(1).max()))
    # where do the small pivots come from
    order = np.argsort(pivnorm)[:12]
    h2 = 1.0 / (N * N)
    picks_in = {}
    for i in sel:
        picks_in[cells[i]] = picks_in.get(cells[i], 0) + 1
    out['smallest_pivots'] = [dict(pivot=float(pivnorm[j]), step=int(j), cell_area_over_h2=round(area_of[cells[sel[j]]] / h2, 4),
                                   picks_in_cell=picks_in[cells[sel[j]]]) for j in order]
    areas = np.array([area_of[c] for c in picks_in]) / h2
    out['cells_with_picks'] = len(picks_in)
    out['area_over_h2_quantiles'] = np.quantile(areas, [0, .05, .25, .5, .75, 1]).round(4).tolist()
    out['seconds'] = round(time.perf_counter() - t0, 1)
    print(json.dumps(out), flush=True)
    return out


def main():
    seat = int(sys.argv[1])
    case = case_from_seat(seat)
    got = compile_one(case, 'base')
    L = got['L']; kind1 = int((got['kind'] == 1).sum())
    T1 = L[got['kind'] == 1].tocsr()
    box_nodes = set(map(int, L[got['kind'] == 0].indices))
    report = dict(seat=seat, scope=__doc__.split('\n')[0], kind1=kind1)
    # 1. raw rows at the teacher's selected points
    trows = rows_at_points(got, got['teacher_selected_points'])
    cols = sorted(set(k for r in trows for k in r) | set(map(int, T1.indices)))
    # the candidate sets may reach more columns; collect them all first
    variants = [('fan6', 6, 'fan'), ('fan4', 4, 'fan'), ('fan3', 3, 'fan'), ('fan2', 2, 'fan'), ('maxtri6', 6, 'maxtri')]
    cand = {}
    for name, deg, mode in variants:
        t0 = time.perf_counter()
        cand[name] = candidates(got, deg, mode)
        cols = sorted(set(cols) | set(k for r in cand[name][1] for k in r))
        print(json.dumps(dict(variant=name, candidates=len(cand[name][0]), seconds=round(time.perf_counter() - t0, 1))), flush=True)
    cols_of = {c: i for i, c in enumerate(cols)}
    T1d = np.asarray(T1[:, cols].todense())
    boxcol = np.array([cols_of[c] for c in cols if c in box_nodes], dtype=np.int64)
    st = np.linalg.svd(T1d, compute_uv=False)
    report['teacher_basis_cond'] = float(st[0] / st[-1])
    A = dense_rows(trows, cols_of, len(cols)); Z = A.copy(); Z[:, boxcol] = 0.0
    s = np.linalg.svd(Z, compute_uv=False)
    fwd, _ = subspace_residual(T1d, Z); rev, _ = subspace_residual(Z, T1d)
    report['teacher_selected_points_raw'] = dict(count=len(trows), cond=float(s[0] / s[-1]), smin=float(s[-1]), smax=float(s[0]),
                                                 span=dict(raw_onto_teacher=fwd, teacher_onto_raw=rev))
    print(json.dumps(dict(teacher_selected_points_raw=report['teacher_selected_points_raw'], teacher_basis_cond=report['teacher_basis_cond'])), flush=True)
    report['variants'] = {}
    for name, deg, mode in variants:
        keys, rows, cells, area_of = cand[name]
        report['variants'][name] = evaluate(name, rows, cells, area_of, T1d, cols_of, boxcol, kind1)
    out_path = os.environ.get('WITNESS_E_OUT', f'WITNESS_E_{seat}.json')
    Path(out_path).write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps(dict(wrote=out_path)), flush=True)


if __name__ == '__main__':
    main()
