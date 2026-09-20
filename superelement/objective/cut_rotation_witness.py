"""Is the teacher's CUT-cell trace operator covariant under the 48 cube symmetries?

This is the one measurement route A stands on, and it has never been made on a cut cell.

For a BOX cell the identity `M_q(g . cell) = G M_q G^T` holds to machine precision, which is why
48-fold augmentation is the only lever in this project measured to convert a fit into
generalisation.  For a CUT cell it fails, and the CUT trace audit located the reason in the
*coordinates*, not the operator: the teacher picks its cut-surface residual functionals by a
greedy pivot (`max_pivot_geometric_residuals_v2`), and the pivot order is not equivariant, so the
rotated cell's cut basis is a different basis of the same subspace (measured: subspace residual
1.6e-15, only 144 of 1436 basis rows shared).  A non-orthogonal change of basis does not commute
with the matrix square root, so `M_q` cannot be covariant even when the operator is.

The fix under test needs no new coordinate rule at all.  Write `q_c = C u`, where `u` is the vector
of background-node displacements on the cut support and `C` is the teacher's own functional matrix.
Pull the trace form back to `u`:

    S_tilde = (C (x) I_3)^T S (C (x) I_3),

reading `C` for the box rows too, where it is a single-node selection.  Under any change of the cut
basis `q'_c = V q_c` the teacher's blocks move as `C' = V C`, `S'_cc = V^-T S_cc V^-1`,
`S'_bc = S_bc V^-1`, so

    C'^T S'_cc C' = C^T V^T V^-T S_cc V^-1 V C = C^T S_cc C,      S'_bc C' = S_bc C,

i.e. **the pullback is exactly invariant to the pivot choice**.  It is therefore covariant if the
underlying CutFEM discretisation is, and the price is only size: the nodal support is 1.00-1.41x
the teacher's q (median 1.31x, dense storage median 1.71x, measured over all 184 cut seats).
It also keeps the cut surface loadable, which condensation does not: a surface traction `t` enters
as the nodal load `C^T t`, which lies in the range of `C^T` and so in the pullback's own range.

What is left unverified by that algebra is whether the CutFEM *discretisation* of a cut cell is
equivariant -- the cut-surface quadrature, the ghost penalty and the stabilisation are all built
from the geometry, and nothing forces them to commute with a rotation.  If they do not, no
coordinate change can rescue augmentation for cut cells and route A is dead.  So this wrapper runs
the pinned teacher on the seat and on its rotated images and measures

    ||G S_tilde G^T - S_tilde(g . cell)||_F / ||S_tilde||_F,      G = P_g (x) Q_g,

against the same residual for the teacher's own box block (which must already match, since the
box coordinates are permuted by g) as the control.

It changes nothing inside the teacher.  Admission differs from `cut_tau_geometry` in exactly one
way and for a stated reason: there the box trace had to be *identical* because tau moved by
1e-4 and the box coordinates could not move; here the geometry is rotated on purpose, so the box
trace must be the *permutation* of the base one, and that is what is required.

    geometry:  python -m superelement.objective.cut_rotation_witness geometry --seat 100000 \
                   --elements 1 5 13 --source <pinned teacher> --output DIR --source-sha <sha>
    numeric:   ... numeric --geometry DIR/G05 --output DIR/G05_NUMERIC --source ... --source-sha ...
    compare:   ... compare --root DIR --numeric-root DIR --output DIR/COMPARE --source ... --source-sha ...
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from fractions import Fraction
from pathlib import Path

import numpy as np

from superelement.equi.cubic_group import CORNER_PERM, DET, INVERSE, Q_ALL, map_int_positions
from superelement.objective.cut_tau_geometry import (case_from_seat, seat_row, sha, source_binding,
                                                     trace_cache, write)

N = 32


def permuted_case(base, g, identifier):
    """The rotated geometry, exactly: corners permuted, plane normal and offset mapped by Q_g.

    The Schwarz-P field is invariant about the cell centre, so `g . cell` is the cell whose tau
    corners are permuted by `CORNER_PERM[g]`; the macro plane `{y : a . y = d}` becomes
    `{y : (Q a) . y = d + sum(Q a - a) / 2}` because the map is `y -> Q (y - 1/2) + 1/2`.
    """
    row = dict(base)
    tau = list(base['tau_corners'])
    out = [None] * 8
    for corner in range(8):
        out[int(CORNER_PERM[g][corner])] = tau[corner]
    a = [Fraction(v) for v in base['normal']]
    Qa = [sum(Fraction(int(Q_ALL[g][i, j])) * a[j] for j in range(3)) for i in range(3)]
    offset = Fraction(base['offset']) + sum(Qa[i] - a[i] for i in range(3)) / 2
    row.update(case_id=identifier, tau_corners=out,
               normal=[str(v) for v in Qa], offset=str(offset))
    return row


def node_map(rotated_nodes, base_nodes, g):
    """Where each rotated-cell node sits in the base node ordering, or None if the sets differ."""
    top = 2 * N
    positions = np.column_stack(np.unravel_index(np.asarray(rotated_nodes), (top + 1,) * 3))
    back = map_int_positions(positions, int(INVERSE[g]), top)
    ids = np.ravel_multi_index(back.T, (top + 1,) * 3)
    order = np.argsort(base_nodes)
    where = np.searchsorted(base_nodes[order], ids)
    hit = (where < len(order)) & (base_nodes[order][np.minimum(where, len(order) - 1)] == ids)
    if not (hit.all() and len(ids) == len(base_nodes)):
        return None, int((~hit).sum())
    return order[where], 0


def compare_traces(base, candidate, g, base_nodes):
    """Admission for a ROTATED cell: the box coordinates must be the permutation of the base ones."""
    out = dict(element=int(g), proper=bool(DET[g] > 0))
    kb = np.asarray(base['kind']); kc = np.asarray(candidate['kind'])
    nb_base = int((kb == 0).sum()); nb_cand = int((kc == 0).sum())
    out['box_coordinates'] = dict(base=nb_base, candidate=nb_cand)
    out['cut_coordinates'] = dict(base=int(len(kb) - nb_base), candidate=int(len(kc) - nb_cand))
    mapping, missing = node_map(candidate['background_nodes'], base_nodes, g)
    out['node_sets_match'] = bool(mapping is not None)
    out['nodes_not_in_base'] = missing
    if mapping is None:
        out['admitted'] = False
        out['status'] = 'ROTATED_ACTIVE_NODE_SET_IS_NOT_THE_IMAGE_OF_THE_BASE_ONE'
        return out, None
    if nb_base != nb_cand:
        out['admitted'] = False
        out['status'] = 'ROTATED_BOX_COORDINATE_COUNT_CHANGED'
        return out, mapping
    # box rows carry one node each: compare the support as a SET after mapping back
    base_box = np.asarray(base['background_nodes'])[np.asarray(base['indices'])[:nb_base]]
    cand_box = np.asarray(base_nodes)[mapping[np.asarray(candidate['indices'])[:nb_cand]]]
    out['box_support_is_the_permuted_set'] = bool(np.array_equal(np.sort(base_box), np.sort(cand_box)))
    out['box_support_same_order'] = bool(np.array_equal(base_box, cand_box))
    out['admitted'] = out['box_support_is_the_permuted_set']
    out['status'] = ('ADMITTED_BOX_TRACE_IS_THE_PERMUTATION' if out['admitted']
                     else 'REJECTED_BOX_SUPPORT_IS_NOT_THE_PERMUTED_SET')
    out['admission_rule'] = ('the rotated cell must have the image active-node set, the same number '
                            'of box coordinates, and the box support must be the permuted set; the '
                            'cut rows are free to be a different basis of the rotated subspace, '
                            'which is exactly what this experiment measures')
    return out, mapping


def geometry(args):
    from stage_cutfem_graded.contract import from_case
    from stage_cutfem_graded.topology import compile_topology
    from stage_cutfem_multiconstraint.complete_trace import compile_trace
    from stage_cutfem_q2.space import node_ids
    from stage_cutfem_runtime.provenance import write_json
    base_case, row, _ = case_from_seat(args.seat)
    with np.load(row['trace_cache'], allow_pickle=False) as z:
        frozen = {k: z[k] for k in z.files}
    base_nodes = np.asarray(frozen['background_nodes'])
    attempts = []
    cases = [('BASE_REPLAY', 0)] + [(f'G{g:02d}', int(g)) for g in args.elements]
    for label, g in cases:
        out = args.output / label
        out.mkdir(exist_ok=False)
        tick = time.perf_counter()
        current = (dict(base_case, case_id=f'CUT_ROT_{int(args.seat)}_BASE') if label == 'BASE_REPLAY'
                   else permuted_case(base_case, g, f'CUT_ROT_{int(args.seat)}_G{g:02d}'))
        write(out / 'CASE.json', current)
        write(out / 'PROTOCOL.json', dict(json.loads((args.output / 'PROTOCOL.json').read_text()),
                                          phase='geometry_trace_only', element=g, label=label))
        c = from_case(current, N)
        topology = compile_topology(c, workers=args.workers)
        write_json(out / 'TOPOLOGY.json', topology)
        if topology['status'] != 'TRUE_TOPOLOGY_CERTIFIED' or topology['component_count'] != 1:
            result = dict(label=label, element=g, admitted=False, status='TOPOLOGY_FAILURE_RETAINED',
                          directory=str(out), seconds=time.perf_counter() - tick)
            write(out / 'RESULT.json', result); attempts.append(result)
            print(json.dumps({k: result[k] for k in ('label', 'element', 'status')}), flush=True)
            continue
        cells = np.asarray(topology['active_parent_cells'], dtype=np.int32)
        nodes = np.unique(np.stack([node_ids(N, i) for i in cells]))
        body = dict(contract=c, nodes=nodes, topology=topology)
        compiled = compile_trace(body, coordinate_convention='max_pivot_geometric_residuals_v2')
        write_json(out / 'COMPLETE_TRACE.json', compiled['record'])
        np.save(out / 'NODES.npy', nodes)
        cache = trace_cache(body, compiled)
        np.savez(out / 'TRACE_CACHE.npz', **cache)
        if label == 'BASE_REPLAY':
            exact = all(np.array_equal(np.asarray(frozen[k]), np.asarray(cache[k]))
                        for k in ('kind', 'background_nodes', 'indptr', 'indices', 'coefficients', 'order'))
            comparison = dict(element=0, admitted=bool(exact), status='BASE_REPLAY_EXACT' if exact else 'BASE_REPLAY_DIFFERS',
                              frozen_cache=row['trace_cache'], frozen_cache_sha256=sha(row['trace_cache']))
            if not exact:
                write(out / 'TRACE_COMPARISON.json', comparison)
                raise ValueError('ZERO_ROTATION_MUST_REPLAY_THE_FROZEN_CACHE_EXACTLY')
            mapping = np.arange(len(nodes))
        else:
            comparison, mapping = compare_traces(frozen, cache, g, base_nodes)
        if mapping is not None:
            np.save(out / 'NODE_MAP_TO_BASE.npy', mapping)
        comparison['candidate_sha256'] = sha(out / 'TRACE_CACHE.npz')
        write(out / 'TRACE_COMPARISON.json', comparison)
        result = dict(label=label, element=g, directory=str(out), admitted=bool(comparison['admitted']),
                      status=comparison['status'], q=int(3 * len(cache['kind'])),
                      body_dofs=int(3 * len(nodes)), active_cells=int(len(cells)),
                      seconds=time.perf_counter() - tick, trace_comparison=comparison)
        write(out / 'RESULT.json', result); attempts.append(result)
        print(json.dumps({k: result[k] for k in ('label', 'element', 'status', 'q', 'active_cells', 'seconds')}),
              flush=True)
    write(args.output / 'SELECTED.json', dict(seat=int(args.seat), elements=[int(g) for g in args.elements],
                                              attempts=attempts, numerical_labels_generated=0,
                                              status='GEOMETRY_SCREEN_COMPLETE_NUMERICAL_LABELS_PENDING'))


def numeric(args):
    from stage_cutfem_graded.contract import from_case
    from stage_cutfem_multiconstraint.body import build
    from stage_cutfem_gp.assembly import build as build_gp
    from stage_cutfem_preproduction.local_operator import export_local
    from stage_cutfem_runtime.record_io import write_record
    from stage_cutfem_runtime.resources import read_cgroup
    import os
    admission = json.loads((args.geometry / 'RESULT.json').read_text())
    if not admission['admitted']:
        raise ValueError('ONLY_ADMITTED_CANDIDATES_ENTER_NUMERICAL_PRODUCTION')
    comparison = json.loads((args.geometry / 'TRACE_COMPARISON.json').read_text())
    if sha(args.geometry / 'TRACE_CACHE.npz') != comparison['candidate_sha256']:
        raise ValueError('ADMITTED_TRACE_CACHE_CHANGED')
    case = json.loads((args.geometry / 'CASE.json').read_text())
    c = from_case(case, N)
    topology = json.loads((args.geometry / 'TOPOLOGY.json').read_text())
    if topology['contract'] != c.record():
        raise ValueError('GEOMETRY_CONTRACT_IDENTITY_CHANGED')
    tick = time.perf_counter()
    os.environ['CUTFEM_STAGE_OUTPUT'] = str(args.output)
    body = build(c, args.output / 'body', workers=args.workers, topology=topology)
    if not np.array_equal(body['nodes'], np.load(args.geometry / 'NODES.npy')):
        raise ValueError('BODY_ASSEMBLY_CHANGED_ADMITTED_ACTIVE_NODES')
    volume_seconds = time.perf_counter() - tick
    print(json.dumps(dict(phase='volume_complete', case=case['case_id'], seconds=volume_seconds)), flush=True)
    mark = time.perf_counter(); gp = build_gp(body, args.output / 'ghost', workers=args.workers)
    gp_seconds = time.perf_counter() - mark
    write_record(args.output / 'GP.json', gp['record'])
    mark = time.perf_counter()
    result = export_local(body, gp, args.output / 'operator', gamma=1e-4, method='native-schur',
                          trace_coordinate_convention='max_pivot_geometric_residuals_v2',
                          matrix_format='packed-upper', verification='production')
    result['body'] = {k: v for k, v in result['body'].items() if k != 'cells'}
    result.update(geometry=str(args.geometry), element=admission['element'], case_parameters=case,
                  volume_seconds=volume_seconds, gp_seconds=gp_seconds,
                  operator_seconds=time.perf_counter() - mark, total_seconds=time.perf_counter() - tick,
                  cgroup_end=read_cgroup(), spectrum_pending=True, diagnostic_only=True,
                  full_factor_network_training=False,
                  physical_geometry_error_separately_qualified=body['record']['geometry_error_separately_qualified'])
    # the export's own box trace must be the admitted one -- self-consistency, not a symmetry claim
    with np.load(args.output / 'operator/REGISTRY.npz') as registry:
        with np.load(args.geometry / 'TRACE_CACHE.npz', allow_pickle=False) as z:
            expected = {k: z[k] for k in z.files}
        kind = np.asarray(expected['kind']); nb = int((kind == 0).sum())
        indptr = np.asarray(expected['indptr'])
        if not np.array_equal(indptr[:nb + 1], np.arange(nb + 1)):
            raise ValueError('BOX_TRACE_ROWS_ARE_NOT_SINGLE_NODE')
        support = np.asarray(expected['background_nodes'])[np.asarray(expected['indices'])[:nb]]
        bbo = registry['box_boundary_original']
        if len(bbo) != 3 * nb or len(registry['boundary']) != 3 * len(kind):
            raise ValueError(f'EXPORT_TRACE_SHAPE_MISMATCH box={len(bbo)} expected={3 * nb} '
                             f'full={len(registry["boundary"])} expected={3 * len(kind)}')
        if not np.array_equal(registry['nodes'][bbo[::3] // 3], support):
            raise ValueError('EXPORT_BOX_TRACE_DOES_NOT_MATCH_ADMITTED_TRACE')
        result['export_trace_check'] = dict(box_coordinates=nb, cut_coordinates=int(len(kind) - nb),
                                            box_support_identical_in_order=True)
    write(args.output / 'RESULT.json', result)
    print(json.dumps(dict(phase='numeric_complete', case=case['case_id'], status=result['status'],
                          seconds=result['total_seconds'])), flush=True)


def unpack(path, q=None):
    """A packed-upper float64 file as a dense symmetric matrix."""
    v = np.load(path)
    if v.ndim != 1:
        raise ValueError(f'PACKED_UPPER_NOT_A_VECTOR {path} {v.shape}')
    n = int(round((np.sqrt(8 * v.size + 1) - 1) / 2))
    if n * (n + 1) // 2 != v.size:
        raise ValueError(f'PACKED_UPPER_LENGTH_NOT_TRIANGULAR {path} {v.size}')
    if q is not None and n != q:
        raise ValueError(f'PACKED_UPPER_DIMENSION {path} {n} expected {q}')
    M = np.zeros((n, n), dtype=np.float64)
    iu = np.triu_indices(n)
    M[iu] = v
    M.T[iu] = v
    return M


def pullback(cache, S):
    """S_tilde on the nodal dofs of the support, plus the support's global node ids.

    `C` is the trace basis over background nodes, one row per scalar coordinate, and the three
    components are interleaved, so `q = (C (x) I_3) u` and `S_tilde = (C (x) I_3)^T S (C (x) I_3)`.
    Only the columns `C` actually touches carry anything, so the matrix is returned on the support.
    """
    import scipy.sparse as sp
    indptr = np.asarray(cache['indptr']); indices = np.asarray(cache['indices'])
    data = np.asarray(cache['coefficients']); nodes = np.asarray(cache['background_nodes'])
    rows = len(indptr) - 1
    C = sp.csr_matrix((data, indices, indptr), shape=(rows, len(nodes)))
    used = np.flatnonzero(np.asarray(abs(C).sum(axis=0)).ravel() > 0)
    C = C[:, used].toarray()
    if S.shape[0] != 3 * rows:
        raise ValueError(f'TRACE_OPERATOR_DIMENSION {S.shape} for {rows} coordinate rows')
    M = np.zeros((3 * rows, 3 * len(used)), dtype=np.float64)
    for axis in range(3):
        M[axis::3, axis::3] = C
    return M.T @ S @ M, nodes[used]


def carry_back(St, support, g, base_support):
    """G^T S_tilde(g . cell) G in the BASE support's node order, with G = P_g (x) Q_g.

    The rotated cell's support is the image of the base one, so node `s` of the rotated support is
    the base node `g^-1(s)` and its three components carry `Q_{g^-1} = Q_g^T`.  Mapping the rotated
    pullback back this way turns the claim `S_tilde(g . cell) = G S_tilde G^T` into a direct
    comparison against the base pullback.
    """
    top = 2 * N
    gi = int(INVERSE[g])
    positions = np.column_stack(np.unravel_index(support, (top + 1,) * 3))
    image = np.ravel_multi_index(map_int_positions(positions, gi, top).T, (top + 1,) * 3)
    order = np.argsort(base_support)
    where = np.searchsorted(base_support[order], image)
    hit = (where < len(order)) & (base_support[order][np.minimum(where, len(order) - 1)] == image)
    if not (hit.all() and len(image) == len(base_support)):
        return None, int((~hit).sum())
    target = order[where]                                    # row i of `support` -> row target[i]
    Q = Q_ALL[gi].astype(np.float64)
    n = len(support)
    P = np.zeros((3 * n, 3 * n), dtype=np.float64)
    for i in range(n):
        P[3 * target[i]:3 * target[i] + 3, 3 * i:3 * i + 3] = Q
    return P @ St @ P.T, 0


def compare(args):
    rows = []
    base_dir = args.root / 'BASE_REPLAY'
    with np.load(base_dir / 'TRACE_CACHE.npz', allow_pickle=False) as z:
        base_cache = {k: z[k] for k in z.files}
    base_S = unpack(args.numeric_root / 'BASE_REPLAY_NUMERIC' / 'operator/full_operator/S_UPPER.npy',
                    3 * len(base_cache['kind']))
    base_St, base_support = pullback(base_cache, base_S)
    base_norm = float(np.linalg.norm(base_St))
    nb_base = int((np.asarray(base_cache['kind']) == 0).sum())
    header = dict(seat=int(args.seat), q_teacher=int(base_S.shape[0]), nodal_dofs=int(base_St.shape[0]),
                  support_nodes=int(len(base_support)), box_coordinates=nb_base,
                  pullback_frobenius=base_norm,
                  growth=round(base_St.shape[0] / base_S.shape[0], 4))
    print(json.dumps(header), flush=True)
    for directory in sorted(args.root.glob('G[0-9][0-9]')):
        label = directory.name
        result = json.loads((directory / 'RESULT.json').read_text())
        if not result['admitted']:
            rows.append(dict(label=label, element=result['element'], status=result['status']))
            continue
        numeric_dir = args.numeric_root / f'{label}_NUMERIC'
        path = numeric_dir / 'operator/full_operator/S_UPPER.npy'
        if not path.exists():
            rows.append(dict(label=label, element=result['element'], status='NUMERIC_MISSING'))
            continue
        g = int(result['element'])
        with np.load(directory / 'TRACE_CACHE.npz', allow_pickle=False) as z:
            cache = {k: z[k] for k in z.files}
        S = unpack(path, 3 * len(cache['kind']))
        St, support = pullback(cache, S)
        row = dict(label=label, element=g, proper=bool(DET[g] > 0),
                   q_teacher=int(S.shape[0]), nodal_dofs=int(St.shape[0]),
                   support_nodes=int(len(support)),
                   support_size_matches=bool(len(support) == len(base_support)))
        if St.shape != base_St.shape:
            row['status'] = 'SUPPORT_DIMENSION_DIFFERS'
            rows.append(row); print(json.dumps(row), flush=True); continue
        rotated, missing = carry_back(St, support, g, base_support)
        if rotated is None:
            row['status'] = 'SUPPORT_IS_NOT_THE_IMAGE_OF_THE_BASE_SUPPORT'
            row['support_nodes_not_in_base_image'] = missing
            rows.append(row); print(json.dumps(row), flush=True); continue
        # the claim: the pullback of the rotated cell is the rotation of the pullback
        delta = float(np.linalg.norm(rotated - base_St))
        row['covariance_residual_relative'] = delta / base_norm
        row['covariance_residual_absolute'] = delta
        row['frobenius_ratio'] = float(np.linalg.norm(St) / base_norm)
        # control: the box-coordinate block of the teacher's own operator, which g only permutes
        row['teacher_cut_coordinates'] = dict(base=int(len(base_cache['kind'])) - nb_base,
                                              rotated=int(len(cache['kind'])) - int((np.asarray(cache['kind']) == 0).sum()))
        row['status'] = 'MEASURED'
        rows.append(row)
        print(json.dumps(row), flush=True)
    write(args.output / 'COVARIANCE.json', dict(header=header, rows=rows,
                                                claim='S_tilde(g . cell) == (P_g (x) Q_g) S_tilde (P_g (x) Q_g)^T',
                                                basis_invariance='exact: C -> V C and S_cc -> V^-T S_cc V^-1'))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['geometry', 'numeric', 'compare'])
    p.add_argument('--seat', type=int)
    p.add_argument('--elements', type=int, nargs='+', default=[1, 5, 13])
    for key in ('source', 'output', 'geometry', 'root', 'numeric-root'):
        p.add_argument('--' + key, type=Path)
    p.add_argument('--source-sha', required=True)
    p.add_argument('--workers', type=int, default=16)
    args = p.parse_args()
    args.numeric_root = getattr(args, 'numeric_root', None) or args.root
    args.output.mkdir(parents=True, exist_ok=False)
    pins = {}
    if args.mode == 'geometry':
        _, row, _ = case_from_seat(args.seat)
        pins = dict(seat=int(args.seat), sample_sha256=sha(Path(row['packet']) / 'SAMPLE.json'),
                    trace_cache_sha256=sha(row['trace_cache']),
                    s_upper_sha256=sha(Path(row['packet']) / 'S_UPPER.npy'))
    protocol = dict(schema='CUT_ROTATION_WITNESS_V1', mode=args.mode,
                    purpose='is the teacher CUT-cell trace operator covariant in the nodal pullback',
                    elements=[int(g) for g in args.elements], n=N, body_space='Q2', E=1., nu=.3,
                    gamma=1e-4, GP_orders=[1, 2], internal_mechanics_dtype='float64',
                    label_encoding='float64 packed-upper',
                    coordinate_convention='max_pivot_geometric_residuals_v2',
                    admission='the rotated cell must carry the image node set and the permuted box support',
                    neural_training=False, validation_test_access=False, diagnostic_only=True,
                    original_teacher_and_labels_unchanged=True,
                    source_sha=args.source_sha, source_script_sha256=sha(__file__), pins=pins)
    write(args.output / 'PROTOCOL.json', protocol)
    import os
    if 'CUTFEM_EXECUTION_CONFIG' in os.environ:
        write(args.output / 'EXECUTION_CONTRACT.json',
              json.loads(Path(os.environ['CUTFEM_EXECUTION_CONFIG']).read_text()))
    started = time.perf_counter()
    try:
        if args.mode == 'compare':
            write(args.output / 'SOURCE_BINDING.json', dict(script_sha256=sha(__file__),
                                                            numpy=np.__version__, executable=sys.executable,
                                                            source_sha=args.source_sha,
                                                            note='pure post-processing of frozen outputs'))
            compare(args)
        else:
            write(args.output / 'SOURCE_BINDING.json', source_binding(args, pins))
            geometry(args) if args.mode == 'geometry' else numeric(args)
            modules = {str(Path(m.__file__).relative_to(args.source)): sha(m.__file__)
                       for m in list(sys.modules.values())
                       if getattr(m, '__file__', None) and Path(m.__file__).is_file()
                       and Path(m.__file__).is_relative_to(args.source)}
            write(args.output / 'EXECUTED_SOURCE_FILES.json', modules)
        write(args.output / 'COMPLETION.json', dict(exit_code=0, seconds=time.perf_counter() - started))
    except BaseException as exc:
        write(args.output / 'FAILURE.json', dict(error=str(exc), traceback=traceback.format_exc(),
                                                 seconds=time.perf_counter() - started))
        raise


if __name__ == '__main__':
    main()
