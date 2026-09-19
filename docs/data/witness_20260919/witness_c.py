"""1.6c: for a CUT cell, does the teacher pick the same residual SUBSPACE under the cube group?

The augmentation identity M_q(g . cell) = G M_q G^T needs the teacher's trace of the rotated
geometry to be the group image of the original's.  For a full cell that is forced: every
coordinate is one background node and the group maps nodes to nodes (1.6a checked the counts and
the cell and node sets).  A cut cell also carries residual functionals on the macro cut surface,
and those are chosen by `max_pivot_geometric_residuals_v2` -- by PIVOTING, which need not be
rotation covariant.

The decisive question is NOT whether the compiler picks the same functionals.  They are only a
basis for the cut surface's residual subspace, and a different basis gives a CONGRUENT operator,
so the label would be right up to a change of basis that we could undo ourselves.  The question is
whether it picks the same SUBSPACE:

  * same subspace, different basis -> re-expressing the cut-surface block in a covariant basis on
    our side repairs augmentation without touching the frozen teacher;
  * different subspace -> the teacher resolves a genuinely different residual space and no basis
    change can repair it.

So this compares row spaces.  The rotated geometry's nodes are mapped back into the base frame
exactly (integer positions, `map_int_positions`), the kind-1 rows of both traces are written in
that common node basis, and each row of one is projected onto the other's row space through an
economic QR.  The reported number is the worst relative projection residual, in both directions.

    python witness_c.py <seat> [g ...]
"""
import json, os, sys, time
from fractions import Fraction
from pathlib import Path
import numpy as np

sys.path.insert(0, '/root/autodl-tmp/CLAUDE_EQUI_20260918/src')
from superelement.equi.cubic_group import Q_ALL, CORNER_PERM, DET, INVERSE, map_int_positions

from stage_cutfem_graded.contract import from_case
from stage_cutfem_graded.topology import compile_topology
from stage_cutfem_multiconstraint.complete_trace import compile_trace
from stage_cutfem_q2.space import node_ids

N = 32
MANIFEST = '/root/autodl-tmp/CLAUDE_LABELS_20260917/V2_LABELS.json'
WORKERS = int(os.environ.get('CUTFEM_ASSEMBLY_WORKERS', 1))


def case_from_seat(seat):
    """The exact rational geometry the teacher was given, read back from the export metadata."""
    row = [r for r in json.load(open(MANIFEST)) if int(r['seat']) == seat][0]
    meta = json.loads((Path(row['trace_cache']).parent / 'INPUT.json').read_text())['metadata']
    geo = meta['geometry']
    plane = geo['cut_plane']
    if len(plane) != 4:
        raise ValueError(f'UNEXPECTED_CUT_PLANE {plane}')
    return dict(case_id=f'seat{seat}', tau_corners=list(geo['tau_corners']),
                normal=[str(plane[0]), str(plane[1]), str(plane[2])], offset=str(plane[3])), row


def permuted_case(case, g):
    """The same cell rotated by g, in exact rational arithmetic: corners permute, the plane maps
    a -> Q a, d -> d + (Q a - a) . (1/2, 1/2, 1/2), the unique choice that keeps the signed
    distance field invariant about the cell centre."""
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
    corners = [(x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1)]
    margins = [d - sum(Qa[i] * Fraction(p[i]) for i in range(3)) for p in corners]
    row['_cuts'] = any(m < 0 for m in margins) and any(m > 0 for m in margins)
    return row


def compile_one(case, label):
    t0 = time.perf_counter()
    c = from_case(case, N)
    topo = compile_topology(c, workers=WORKERS)
    cells = np.asarray(topo['active_parent_cells'], dtype=np.int64).reshape(-1, 3)
    nodes = np.unique(np.stack([node_ids(N, i) for i in cells]))
    compiled = compile_trace(dict(contract=c, nodes=nodes, topology=topo),
                             coordinate_convention='max_pivot_geometric_residuals_v2')
    Lv = compiled['L']
    L = Lv[::3, ::3].tocsr()                                  # the scalar CSR: coordinate x node
    kind = np.asarray(compiled['kind']).astype(np.int64) if 'kind' in compiled else None
    if kind is None:                                          # reconstruct: kind 0 has one node
        kind = (np.diff(L.indptr) > 1).astype(np.int64)
    print(json.dumps(dict(label=label, status=topo['status'], active_cells=len(cells), nodes=len(nodes),
                          q=int(Lv.shape[0]), coordinates=int(L.shape[0]),
                          kind1=int((kind == 1).sum()),
                          extra_cut=compiled['record']['additional_cut_trace_dimension'],
                          seconds=round(time.perf_counter() - t0, 1))), flush=True)
    return dict(L=L, kind=kind, nodes=nodes, cells=cells, q=int(Lv.shape[0]),
                extra=compiled['record']['additional_cut_trace_dimension'])


def subspace_residual(A, B):
    """Worst relative residual of projecting each row of B onto the row space of A.

    Economic QR of A^T, then r_j = || b_j - Q Q^T b_j || / || b_j ||.  Both are dense here because
    the supports are tiny (2..20 nodes) but the node basis is shared and large.
    """
    At = np.asarray(A.T.todense(), dtype=np.float64)
    Q, _ = np.linalg.qr(At)                                   # (nodes, rank<=m)
    Bt = np.asarray(B.T.todense(), dtype=np.float64)
    R = Bt - Q @ (Q.T @ Bt)
    num = np.linalg.norm(R, axis=0); den = np.linalg.norm(Bt, axis=0)
    rel = num / np.maximum(den, 1e-300)
    return float(rel.max()), float(np.median(rel)), int(Q.shape[1])


def main():
    seat = int(sys.argv[1])
    elements = [int(x) for x in sys.argv[2:]] or [1, 5, 13]
    case, row = case_from_seat(seat)
    report = dict(seat=seat, n=N, elements=elements, case=case,
                  scope='1.6c: is the cut cell residual SUBSPACE the group image, basis aside?')
    t0 = time.time()
    base = compile_one(case, 'base')
    top = 2 * N
    base_pos = np.column_stack(np.unravel_index(base['nodes'], (top + 1,) * 3))
    report['base'] = dict(coordinates=int(base['L'].shape[0]), kind1=int((base['kind'] == 1).sum()),
                          q=base['q'], extra_cut=base['extra'], nodes=len(base['nodes']))
    report['elements_detail'] = {}
    for g in elements:
        case_g = permuted_case(case, g)
        cuts = case_g.pop('_cuts')
        got = compile_one(case_g, f'g={g}')
        row_g = dict(proper=bool(DET[g] > 0), rotated_plane_still_cuts=bool(cuts),
                     coordinates=int(got['L'].shape[0]), kind1=int((got['kind'] == 1).sum()),
                     q=got['q'], extra_cut=got['extra'], nodes=len(got['nodes']),
                     counts_match=bool(got['q'] == base['q'] and got['extra'] == base['extra']
                                       and int((got['kind'] == 1).sum()) == int((base['kind'] == 1).sum())))
        # map the rotated geometry's nodes back into the base frame, exactly
        pos_g = np.column_stack(np.unravel_index(got['nodes'], (top + 1,) * 3))
        back = map_int_positions(pos_g, int(INVERSE[g]), top)
        ids_back = np.ravel_multi_index(back.T, (top + 1,) * 3)
        order = np.argsort(base['nodes'])
        where = np.searchsorted(base['nodes'][order], ids_back)
        ok = (where < len(order)) & (base['nodes'][order][np.minimum(where, len(order) - 1)] == ids_back)
        row_g['node_sets_match'] = bool(ok.all() and len(ids_back) == len(base['nodes']))
        if not row_g['node_sets_match']:
            row_g['unmapped_nodes'] = int((~ok).sum())
            report['elements_detail'][str(g)] = row_g
            print(json.dumps({str(g): row_g}), flush=True); continue
        col = order[where]                                    # rotated column -> base column
        # place the rotated rows in the base node basis
        import scipy.sparse as sp
        P = sp.csr_matrix((np.ones(len(col)), (np.arange(len(col)), col)), shape=(len(col), len(base['nodes'])))
        Lg = (got['L'] @ P).tocsr()
        k0b = base['L'][base['kind'] == 0]; k1b = base['L'][base['kind'] == 1]
        k0g = Lg[got['kind'] == 0]; k1g = Lg[got['kind'] == 1]
        # kind 0 is one node with one coefficient, so the sets must agree exactly
        nb = set(map(int, k0b.indices)); ng = set(map(int, k0g.indices))
        row_g['kind0_node_sets_match'] = bool(nb == ng)
        row_g['kind0_count'] = [int(k0b.shape[0]), int(k0g.shape[0])]
        if k1b.shape[0] and k1g.shape[0]:
            fwd = subspace_residual(k1b, k1g); rev = subspace_residual(k1g, k1b)
            row_g['subspace'] = dict(rotated_rows_onto_base_worst=fwd[0], rotated_rows_onto_base_median=fwd[1],
                                     base_rank=fwd[2], base_rows_onto_rotated_worst=rev[0],
                                     base_rows_onto_rotated_median=rev[1], rotated_rank=rev[2])
            row_g['subspaces_agree'] = bool(max(fwd[0], rev[0]) < 1e-8)
        report['elements_detail'][str(g)] = row_g
        print(json.dumps({str(g): row_g}), flush=True)
    live = [v for v in report['elements_detail'].values() if 'subspace' in v]
    report['worst_subspace_residual'] = max((max(v['subspace']['rotated_rows_onto_base_worst'],
                                                 v['subspace']['base_rows_onto_rotated_worst']) for v in live), default=None)
    report['cut_subspace_is_the_group_image'] = bool(live) and all(v.get('subspaces_agree') for v in live)
    report['interpretation'] = (
        'subspaces agree: the label is right up to a congruence, so a covariant basis on our side '
        'repairs augmentation without touching the teacher'
        if report['cut_subspace_is_the_group_image'] else
        'subspaces differ: the teacher resolves a different residual space and no basis change repairs it'
        if live else 'no element produced a comparable trace')
    report['seconds'] = time.time() - t0
    print(json.dumps(report, indent=1), flush=True)
    out = os.environ.get('WITNESS_C_OUT')
    if out:
        Path(out).write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
