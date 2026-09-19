"""1.6d: a covariant, geometric basis for the cut-surface trace, on our side, measured.

witness_c established that the teacher's cut-surface residual functionals span a SUBSPACE that is
exactly the group image of the base cell's (worst 6.5e-16), so the label is right up to a
congruence.  The teacher's own basis is not covariant by construction: `compile_trace` lays a
degree-6 lattice on the triangle of the polygon's first three hull vertices (`polygon[:3]`, hull
vertices sorted lexicographically), scans candidates in lexicographic point order and pivots on
the largest coefficient.  Vertex order and scan order both change under an axis permutation.

This witness measures a rule of OUR choosing that touches nothing frozen:

  candidates  for every active cut parent cell, the plane's polygon in that cell (the same
              `polytope(canonical(box_planes + plane))` the teacher uses), fanned from its vertex
              centroid, degree-6 lattice on every fan triangle.  The fan's triangle SET and the
              lattice are covariant regardless of vertex order.  Points are exact rationals; a
              point is one raw Q2 point-evaluation functional (exact_basis_row), so signed sum 1.
  selection   greedy max residual norm (pivoted QR) on the candidates with the box-node columns
              zeroed, until the kind-1 dimension the teacher reports is reached.  Covariant unless
              two candidates tie exactly; the tie margin at every step is recorded.
  coordinates the raw point functionals at the selected points, next to the box coordinates.

Measured, on one seat, for g in the cube group and one tau perturbation:
  * candidate point sets equal under g (mapped back exactly);
  * selected point sets equal under g, and the selection sequence;
  * the teacher's selected point set and basis rows under g, for contrast;
  * tau perturbation: does the candidate set / selection change at all (it should not: the plane
    and the active cut cells alone determine them);
  * conditioning: singular values of the teacher's kind-1 basis and of ours on the same subspace,
    and the condition number of the change of basis C (ours = C teacher), which bounds how far the
    operator's spectrum moves under the congruence.

    python witness_d.py <seat> [g ...]
"""
import json, os, sys, time
from fractions import Fraction
from pathlib import Path
import numpy as np
import scipy.sparse as sp

sys.path.insert(0, '/root/autodl-tmp/CLAUDE_EQUI_20260918/src')
from superelement.equi.cubic_group import Q_ALL, CORNER_PERM, DET, INVERSE, map_int_positions

from stage_cutfem_graded.contract import from_case
from stage_cutfem_graded.topology import compile_topology
from stage_cutfem_multiconstraint.complete_trace import compile_trace, polytope, canonical, box_planes
from stage_cutfem_q2.space import node_ids
from stage_cutfem_q2.trace import exact_basis_row, sextic_triangle_points

N = 32
MANIFEST = '/root/autodl-tmp/CLAUDE_LABELS_20260917/V2_LABELS.json'
WORKERS = int(os.environ.get('CUTFEM_ASSEMBLY_WORKERS', 1))
HALF = Fraction(1, 2)


def case_from_seat(seat):
    row = [r for r in json.load(open(MANIFEST)) if int(r['seat']) == seat][0]
    meta = json.loads((Path(row['trace_cache']).parent / 'INPUT.json').read_text())['metadata']
    geo = meta['geometry']
    plane = geo['cut_plane']
    return dict(case_id=f'seat{seat}', tau_corners=list(geo['tau_corners']),
                normal=[str(plane[0]), str(plane[1]), str(plane[2])], offset=str(plane[3]))


def permuted_case(case, g):
    row = dict(case)
    tau = list(case['tau_corners']); out = [None] * 8
    for c in range(8):
        out[CORNER_PERM[g][c]] = tau[c]
    row['tau_corners'] = out
    a = [Fraction(v) for v in case['normal']]
    Qa = [sum(Fraction(int(Q_ALL[g][i, j])) * a[j] for j in range(3)) for i in range(3)]
    d = Fraction(case['offset']) + sum(Qa[i] - a[i] for i in range(3)) / 2
    row['normal'] = [str(v) for v in Qa]; row['offset'] = str(d)
    row['case_id'] = case['case_id'] + f'_g{g:02d}'
    return row


def perturbed_case(case, corner, delta):
    row = dict(case)
    tau = list(case['tau_corners'])
    tau[corner] = str(Fraction(tau[corner]) + Fraction(delta))
    row['tau_corners'] = tau
    row['case_id'] = case['case_id'] + f'_tau{corner}'
    return row


def map_point_back(p, g):
    """p -> Q_{g^-1} (p - 1/2) + 1/2 in exact rationals."""
    Q = Q_ALL[int(INVERSE[g])]
    c = [p[i] - HALF for i in range(3)]
    return tuple(sum(Fraction(int(Q[i, j])) * c[j] for j in range(3)) + HALF for i in range(3))


def compile_one(case, label):
    t0 = time.perf_counter()
    c = from_case(case, N)
    topo = compile_topology(c, workers=WORKERS)
    cells = np.asarray(topo['active_parent_cells'], dtype=np.int64).reshape(-1, 3)
    nodes = np.unique(np.stack([node_ids(N, i) for i in cells]))
    compiled = compile_trace(dict(contract=c, nodes=nodes, topology=topo),
                             coordinate_convention='max_pivot_geometric_residuals_v2')
    L = compiled['L'][::3, ::3].tocsr()
    kind = (np.diff(L.indptr) > 1).astype(np.int64)
    exact = compiled['record']['exact']
    labels = exact['labels']
    teacher_points = [tuple(Fraction(x) for x in lab['point']) for lab in labels if lab['kind'] == 'cut_geometric_functional']
    selected = [i for i in exact['selected_rows']]
    box_count = sum(1 for lab in labels if lab['kind'] == 'box_coefficient')
    teacher_selected_points = [tuple(Fraction(x) for x in labels[i]['point']) for i in selected if i >= box_count]
    print(json.dumps(dict(label=label, status=topo['status'], active_cells=len(cells), nodes=len(nodes),
                          coordinates=int(L.shape[0]), kind1=int((kind == 1).sum()),
                          teacher_candidates=len(teacher_points), seconds=round(time.perf_counter() - t0, 1))), flush=True)
    return dict(L=L, kind=kind, nodes=nodes, cells=cells, contract=c, topo=topo,
                teacher_points=teacher_points, teacher_selected_points=teacher_selected_points)


def our_candidates(got):
    """Exact candidate points and their Q2 point-evaluation rows (local node index -> float)."""
    c = got['contract']; n = c.n
    positions = {int(node): i for i, node in enumerate(got['nodes'])}
    normal, offset = c.rational_plane()
    points = {}
    patches = 0
    for patch in got['topo']['patches']:
        if patch['tag'] != 'MACRO_CUT_FACE':
            continue
        patches += 1
        ijk = tuple(int(v) for v in patch['parent'])
        ambient = polytope(canonical(box_planes(ijk, n) + [(normal, offset, 'MACRO_CUT_FACE')]))
        polys = [polygon for _, _, tag, polygon in ambient['faces'] if 'MACRO_CUT_FACE' in tag.split('|')]
        if len(polys) != 1:
            raise ValueError('AMBIENT_PATCH')
        poly = [tuple(Fraction(x) for x in v) for v in polys[0]]
        m = len(poly)
        centroid = tuple(sum(v[d] for v in poly) / m for d in range(3))
        for i in range(m):
            tri = (centroid, poly[i], poly[(i + 1) % m])
            for p in sextic_triangle_points(tri):
                key = tuple(p)
                if key in points:
                    continue
                row = exact_basis_row(p, ijk, n, positions)
                points[key] = {int(k): float(v) for k, v in row.items()}
    keys = sorted(points)                                   # any fixed order; the rule must not depend on it
    return keys, [points[k] for k in keys], patches


def dense_rows(rows, cols_of, ncols):
    A = np.zeros((len(rows), ncols))
    for i, row in enumerate(rows):
        for k, v in row.items():
            A[i, cols_of[k]] = v
    return A


def greedy(Rm, target, tol=1e-10):
    """Pivoted QR by rows, in place.  Returns the pick sequence, the tie margin 1 - second/best at
    every step, the pivot norm relative to the first, and what is left after the last pick."""
    norms = np.sqrt((Rm * Rm).sum(1)); first = float(norms.max())
    sel, margins, pivnorm = [], [], []
    for _ in range(target):
        i = int(np.argmax(norms)); top = float(norms[i])
        if top <= tol * first:
            break
        second = float(np.partition(norms, -2)[-2])
        margins.append(1.0 - second / top); pivnorm.append(top / first)
        q = Rm[i] / top
        r = Rm @ q
        Rm -= np.outer(r, q)
        Rm[i] = 0.0
        norms = np.sqrt((Rm * Rm).sum(1))
        sel.append(i)
    return sel, margins, pivnorm, float(norms.max() / first)


def subspace_residual(A, B):
    """Worst relative residual of projecting each row of B onto the row space of A (dense rows)."""
    Q, _ = np.linalg.qr(A.T)
    R = B.T - Q @ (Q.T @ B.T)
    rel = np.linalg.norm(R, axis=0) / np.maximum(np.linalg.norm(B.T, axis=0), 1e-300)
    return float(rel.max()), int(Q.shape[1])


def analyse(got, label, base=None, g=None, col_map=None):
    """Run our rule on one compiled cell.  With base/col_map the rows are placed in the base node
    basis (col_map: local rotated column -> local base column) and points mapped back by g^-1."""
    t0 = time.perf_counter()
    keys, rows, patches = our_candidates(got)
    kind1 = int((got['kind'] == 1).sum())
    L = got['L']
    box_nodes = set(map(int, L[got['kind'] == 0].indices))
    if col_map is not None:
        rows = [{int(col_map[k]): v for k, v in r.items()} for r in rows]
        box_nodes = {int(col_map[k]) for k in box_nodes}
        keys_base = [map_point_back(k, g) for k in keys]
        T1 = (L[got['kind'] == 1] @ sp.csr_matrix((np.ones(len(col_map)), (np.arange(len(col_map)), col_map)),
                                                  shape=(len(col_map), len(base['nodes'])))).tocsr()
        tsel = [map_point_back(p, g) for p in got['teacher_selected_points']]
        tcand = [map_point_back(p, g) for p in got['teacher_points']]
    else:
        keys_base = keys
        T1 = L[got['kind'] == 1].tocsr()
        tsel = got['teacher_selected_points']; tcand = got['teacher_points']
    cols = sorted(set(k for r in rows for k in r) | set(map(int, T1.indices)))
    cols_of = {c: i for i, c in enumerate(cols)}
    A = dense_rows(rows, cols_of, len(cols))                # raw point functionals
    boxcol = np.array([cols_of[c] for c in cols if c in box_nodes], dtype=np.int64)
    Z = A.copy(); Z[:, boxcol] = 0.0                        # box columns zeroed: the V1 part
    sel, margins, pivnorm, left = greedy(Z.copy(), kind1)
    T1d = T1.toarray()[:, [c for c in cols]] if False else np.asarray(T1[:, cols].todense())
    O1 = Z[sel]
    fwd, rank_t = subspace_residual(T1d, O1); rev, rank_o = subspace_residual(O1, T1d)
    st = np.linalg.svd(T1d, compute_uv=False); so = np.linalg.svd(O1, compute_uv=False)
    # change of basis: O1 = C11 T1  (least squares, exact if the spans agree)
    X, *_ = np.linalg.lstsq(T1d.T, O1.T, rcond=None)
    C11 = X.T
    c11_res = float(np.linalg.norm(C11 @ T1d - O1) / np.linalg.norm(O1))
    C10 = A[sel][:, boxcol]
    nb = len(boxcol)
    C = np.zeros((nb + kind1, nb + kind1)); C[:nb, :nb] = np.eye(nb); C[nb:, :nb] = C10; C[nb:, nb:] = C11
    sc = np.linalg.svd(C, compute_uv=False); s11 = np.linalg.svd(C11, compute_uv=False)
    signed = A[sel].sum(1)
    out = dict(label=label, cut_patches=patches, candidates=len(keys), kind1=kind1, box_nodes=len(box_nodes),
               columns=len(cols), picked=len(sel), left_after_last_pick=left,
               margin_min=float(min(margins)), margin_median=float(np.median(margins)),
               margins_below_1e6=int(sum(m < 1e-6 for m in margins)), margins_below_1e3=int(sum(m < 1e-3 for m in margins)),
               pivot_norm_last_over_first=float(pivnorm[-1]),
               span_equals_teacher=dict(ours_onto_teacher=fwd, teacher_onto_ours=rev, rank_teacher=rank_t, rank_ours=rank_o),
               teacher_basis_singular=dict(max=float(st[0]), min=float(st[-1]), cond=float(st[0] / st[-1])),
               our_basis_singular=dict(max=float(so[0]), min=float(so[-1]), cond=float(so[0] / so[-1])),
               C11=dict(residual=c11_res, cond=float(s11[0] / s11[-1])), C_full_cond=float(sc[0] / sc[-1]),
               point_functional_signed_sum=dict(min=float(signed.min()), max=float(signed.max())),
               selected_nnz=dict(min=int((A[sel] != 0).sum(1).min()), max=int((A[sel] != 0).sum(1).max()),
                                 mean=float((A[sel] != 0).sum(1).mean())),
               seconds=round(time.perf_counter() - t0, 1))
    data = dict(candidates=set(keys_base), selected=[keys_base[i] for i in sel], margins=margins,
                teacher_selected=set(tsel), teacher_candidates=set(tcand), T1=T1d, cols=cols)
    print(json.dumps(dict(analysis=out)), flush=True)
    return out, data


def teacher_rows_set(got, col_map=None):
    """The teacher's kind-1 basis rows as hashable tuples in the base node basis."""
    L = got['L'][got['kind'] == 1].tocsr()
    rows = set()
    for i in range(L.shape[0]):
        s = slice(L.indptr[i], L.indptr[i + 1])
        idx = L.indices[s] if col_map is None else col_map[L.indices[s]]
        rows.add(tuple(sorted((int(c), round(float(v), 10)) for c, v in zip(idx, L.data[s]))))
    return rows


def main():
    seat = int(sys.argv[1])
    elements = [int(x) for x in sys.argv[2:]] or [5, 13]
    case = case_from_seat(seat)
    report = dict(seat=seat, n=N, elements=elements, case=case, scope=__doc__.split('\n')[0])
    t0 = time.time()
    base = compile_one(case, 'base')
    top = 2 * N
    base_out, base_data = analyse(base, 'base')
    report['base'] = base_out
    base_trows = teacher_rows_set(base)
    report['elements_detail'] = {}
    for g in elements:
        got = compile_one(permuted_case(case, g), f'g={g}')
        pos_g = np.column_stack(np.unravel_index(got['nodes'], (top + 1,) * 3))
        back = map_int_positions(pos_g, int(INVERSE[g]), top)
        ids_back = np.ravel_multi_index(back.T, (top + 1,) * 3)
        order = np.argsort(base['nodes'])
        where = np.searchsorted(base['nodes'][order], ids_back)
        ok = (where < len(order)) & (base['nodes'][order][np.minimum(where, len(order) - 1)] == ids_back)
        row = dict(proper=bool(DET[g] > 0), node_sets_match=bool(ok.all() and len(ids_back) == len(base['nodes'])))
        if not row['node_sets_match']:
            report['elements_detail'][str(g)] = row; continue
        col_map = order[where]
        out, data = analyse(got, f'g={g}', base=base, g=g, col_map=col_map)
        row['analysis'] = out
        row['our_candidates_match'] = bool(data['candidates'] == base_data['candidates'])
        row['our_candidates_symmetric_difference'] = len(data['candidates'] ^ base_data['candidates'])
        row['our_selected_set_match'] = bool(set(data['selected']) == set(base_data['selected']))
        row['our_selected_sequence_match'] = bool(data['selected'] == base_data['selected'])
        row['our_selected_symmetric_difference'] = len(set(data['selected']) ^ set(base_data['selected']))
        row['our_margins_max_abs_diff'] = float(np.max(np.abs(np.array(data['margins']) - np.array(base_data['margins'])))) \
            if len(data['margins']) == len(base_data['margins']) else None
        row['teacher_candidates_match'] = bool(data['teacher_candidates'] == base_data['teacher_candidates'])
        row['teacher_candidates_symmetric_difference'] = len(data['teacher_candidates'] ^ base_data['teacher_candidates'])
        row['teacher_selected_set_match'] = bool(data['teacher_selected'] == base_data['teacher_selected'])
        row['teacher_selected_symmetric_difference'] = len(data['teacher_selected'] ^ base_data['teacher_selected'])
        trows = teacher_rows_set(got, col_map)
        row['teacher_basis_rows_match'] = bool(trows == base_trows)
        row['teacher_basis_rows_in_common'] = len(trows & base_trows)
        # the operator congruence between our bases across g is then a permutation: check the
        # selected functionals coincide row by row in the base node basis
        report['elements_detail'][str(g)] = row
        print(json.dumps({str(g): {k: v for k, v in row.items() if k != 'analysis'}}), flush=True)
        del got, data
    # tau perturbation: one corner by 1/500 (about 1 % of tau); the plane is untouched
    pert = compile_one(perturbed_case(case, 0, '1/500'), 'tau0+1/500')
    same_nodes = bool(len(pert['nodes']) == len(base['nodes']) and np.array_equal(pert['nodes'], base['nodes']))
    row = dict(active_cells=[int(len(base['cells'])), int(len(pert['cells']))], node_sets_match=same_nodes,
               kind1=[int((base['kind'] == 1).sum()), int((pert['kind'] == 1).sum())])
    if same_nodes:
        out, data = analyse(pert, 'tau0+1/500')
        row['analysis'] = out
        row['our_candidates_match'] = bool(data['candidates'] == base_data['candidates'])
        row['our_selected_sequence_match'] = bool(data['selected'] == base_data['selected'])
        row['teacher_selected_set_match'] = bool(data['teacher_selected'] == base_data['teacher_selected'])
        row['teacher_basis_rows_match'] = bool(teacher_rows_set(pert) == base_trows)
    report['tau_perturbation'] = row
    print(json.dumps(dict(tau_perturbation={k: v for k, v in row.items() if k != 'analysis'})), flush=True)
    live = [v for v in report['elements_detail'].values() if 'analysis' in v]
    report['our_rule_covariant'] = bool(live) and all(v['our_candidates_match'] and v['our_selected_set_match'] for v in live)
    report['teacher_rows_covariant'] = bool(live) and all(v['teacher_basis_rows_match'] for v in live)
    report['seconds'] = time.time() - t0
    out_path = os.environ.get('WITNESS_D_OUT', f'WITNESS_D_{seat}.json')
    Path(out_path).write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps(dict(our_rule_covariant=report['our_rule_covariant'], teacher_rows_covariant=report['teacher_rows_covariant'],
                          seconds=report['seconds'], wrote=out_path)), flush=True)


if __name__ == '__main__':
    main()
