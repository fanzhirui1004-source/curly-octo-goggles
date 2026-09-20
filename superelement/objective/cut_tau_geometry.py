"""Gate 2, teacher side: does a CUT cell's response have a tau derivative at all?

The tau smoothness gate has only ever been run on seat 0328, which is a BOX cell (kind1 = 0,
cut_plane offset 2, i.e. no macro cut).  For a free cell it found the truth itself one-sided
inconsistent -- log-slopes -6.09 against -21.36 at h = 1e-4 -- because one CutFEM cell being born
moves the softest direction by 60 %; the clamped-platen observable was smooth.  A cut cell's cut
surface is free by construction, so that is exactly the configuration we have never tested, and
the contract has a sensitivity leg.  If the truth has no derivative there, no learned operator can
supply one, and the whole cut-cell programme needs re-scoping.  So this runs before anything is
trained and before any condensed label is built.

It calls the pinned teacher and changes nothing inside it.  Two guardrails in the 0328 wrapper
(`CLAUDE_TRACE_20260917/full_factor_geometry.py`, whose behaviour this file follows otherwise) are
box-only and are replaced here, not relaxed:

  * admission required the COMPLETE trace to be bit-identical across the perturbation.  For a cut
    cell that can never hold: 60-93 % of the trace is cut-surface residual functionals, and the cut
    surface is the intersection of a fixed plane with the material, which moves when tau moves.  So
    admission here requires the KIND-0 (box-face) part of the trace to be identical and records what
    the kind-1 part did.  That is the right rule for the object the lattice reads: the condensed
    operator T = S_BB - S_BC S_CC^-1 S_CB is invariant to any change of basis in the eliminated
    block, so it is comparable across epsilon even when the cut basis is not.
  * the numeric stage ended with `if np.any(expected['kind']): raise`, i.e. it refused cut seats
    outright.  Here the same nodal-support comparison is made over the box rows, and the cut rows'
    count and support are recorded beside it.

Both the old and the new comparison are written out, so nothing is hidden by the change.

    geometry:  python -m superelement.objective.cut_tau_geometry geometry --seat 100000 \
                   --source <pinned teacher> --output DIR --source-sha <sha>
    numeric:   ... numeric --geometry DIR/PATH_1_H0 --output DIR/PATH_1_NUMERIC --source ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from fractions import Fraction
from pathlib import Path

import numpy as np

COMMIT = '6624dc86706bbd8e3cc16b4b3adbfb043eff08ec'
EPSILONS = ('-0.001', '-0.0001', '0.0001', '0.001')
MANIFEST = Path('/root/autodl-tmp/CLAUDE_LABELS_20260917/V2_LABELS.json')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def seat_row(seat):
    rows = [r for r in json.loads(MANIFEST.read_text()) if int(r['seat']) == int(seat)]
    if len(rows) != 1:
        raise ValueError(f'SEAT_NOT_UNIQUE_IN_MANIFEST {seat}')
    return rows[0]


def case_from_seat(seat):
    """The seat's exact rational case, rebuilt from its frozen packet.

    The dataset's per-seat `active_*` run directories have been reclaimed, so the recipe file the
    0328 wrapper read is gone; SAMPLE.json carries the same rational tau corners and plane.
    """
    row = seat_row(seat)
    sample = json.loads((Path(row['packet']) / 'SAMPLE.json').read_text())
    geo = sample['geometry']
    plane = geo['cut_plane']                      # ["1", b, "0", offset], exact rationals
    case = dict(case_id=f'seat{int(seat)}',
                tau_corners=[str(Fraction(v)) for v in geo['tau_corners']],
                normal=[str(Fraction(v)) for v in plane[:3]],
                offset=str(Fraction(plane[3])),
                mother_field_id=sample['dataset_context']['mother_field_id'],
                split='train', diagnostic_only=True)
    return case, row, sample


def perturbed_case(base, epsilon, identifier):
    """All eight corners scaled by the exact rational 1 + epsilon, as the 0328 protocol did."""
    epsilon = Fraction(str(epsilon))
    if not 0 < abs(epsilon) <= Fraction(1, 1000):
        raise ValueError('UNREGISTERED_THICKNESS_PERTURBATION')
    row = dict(base)
    row.update(case_id=identifier,
               tau_corners=[str(Fraction(v) * (1 + epsilon)) for v in base['tau_corners']])
    return row


def trace_cache(body, compiled):
    """The frozen input adapter's deterministic calculation, byte-for-byte as the 0328 wrapper."""
    c, nodes = body['contract'], body['nodes']
    points = np.stack(np.unravel_index(nodes, (2 * c.n + 1,) * 3), axis=1) / (2 * c.n)
    centered = points - points.mean(axis=0)
    rigid = np.zeros((3 * len(points), 6), dtype=np.float64)
    for axis in range(3):
        rigid[axis::3, axis] = 1.
        rigid[:, 3 + axis] = np.cross(np.eye(3)[axis], centered).ravel()
    Lv = compiled['L']
    L = Lv[::3, ::3].tocsr()
    weights = abs(L)
    mass = np.asarray(weights.sum(axis=1)).ravel()
    if np.any(mass <= 0):
        raise ValueError('EMPTY_PHYSICAL_TRACE_FUNCTIONAL')
    centroids = (weights @ points) / mass[:, None]
    cell = np.floor(centroids * 1023).astype(np.int64)
    if np.any(cell < 0) or np.any(cell > 1023):
        raise ValueError('SUPPORT_OUTSIDE_ORIGINAL_BOX')
    keys = np.zeros(len(cell), dtype=np.int64)
    for bit in range(10):
        for axis in range(3):
            keys |= ((cell[:, axis] >> bit) & 1) << (3 * bit + axis)
    scalar = np.lexsort((np.arange(len(keys)), keys))
    order = (3 * scalar[:, None] + np.arange(3)).ravel()
    kind = np.zeros(L.shape[0], dtype=np.uint8)
    extra = compiled['record']['additional_cut_trace_dimension']
    if extra:
        kind[-extra:] = 1
    return dict(rigid=Lv @ rigid, order=order, support_centroid=centroids,
                kind=kind, background_nodes=nodes, indptr=L.indptr,
                indices=L.indices, coefficients=L.data,
                tau_corners=np.array(list(map(float, c.tau_corners))),
                cut_plane=np.array([*map(float, c.normal), float(c.offset)]))


def rows_of(cache, lo, hi):
    """The (indptr, global node ids, coefficients) of coordinate rows [lo, hi)."""
    indptr = np.asarray(cache['indptr'])
    a, b = int(indptr[lo]), int(indptr[hi])
    return (indptr[lo:hi + 1] - indptr[lo],
            np.asarray(cache['background_nodes'])[np.asarray(cache['indices'])[a:b]],
            np.asarray(cache['coefficients'])[a:b])


def compare(base, candidate):
    """Cut-aware admission: the box rows must be identical, the cut rows are recorded."""
    out = {}
    nb_base = int((np.asarray(base['kind']) == 0).sum())
    nb_cand = int((np.asarray(candidate['kind']) == 0).sum())
    out['box_coordinates'] = dict(base=nb_base, candidate=nb_cand)
    out['cut_coordinates'] = dict(base=int(len(base['kind']) - nb_base),
                                 candidate=int(len(candidate['kind']) - nb_cand))
    same_box = nb_base == nb_cand and all(
        np.array_equal(x, y) for x, y in zip(rows_of(base, 0, nb_base), rows_of(candidate, 0, nb_cand)))
    out['same_box_trace'] = bool(same_box)
    # the original, complete-trace rule, reported so the change of rule is visible
    def semantics(cache):
        return [np.asarray(cache['indptr']),
                np.asarray(cache['background_nodes'])[np.asarray(cache['indices'])],
                np.asarray(cache['coefficients']), np.asarray(cache['kind'])]
    out['same_complete_trace_original_rule'] = bool(
        all(np.array_equal(a, b) for a, b in zip(semantics(base), semantics(candidate))))
    out['same_order'] = bool(np.array_equal(np.asarray(base['order']), np.asarray(candidate['order'])))
    box_rigid = np.asarray(base['rigid'])[:3 * nb_base], np.asarray(candidate['rigid'])[:3 * nb_cand]
    out['box_rigid_absolute_frobenius_difference'] = (
        float(np.linalg.norm(box_rigid[0] - box_rigid[1])) if same_box else None)
    out['admitted'] = bool(same_box and out['box_rigid_absolute_frobenius_difference'] is not None
                           and out['box_rigid_absolute_frobenius_difference'] <= 1e-12)
    out['admission_rule'] = ('kind-0 rows identical and the box rigid trace unchanged; kind-1 rows '
                             'may change because the cut surface moves with tau, and the condensed '
                             'operator does not depend on their basis')
    return out


def source_binding(args, pins):
    """The pinned teacher's identity, checked on tracked content only.

    The 0328 wrapper required `git status --porcelain` to be empty outright.  The pinned tree now
    carries three UNTRACKED scripts left behind by the 2026-09-16 GP-tree profiling
    (`gp_build.py`, `gp_launch.py`, `gp_tree_profile.py`); no tracked file is modified, so the
    teacher's code is still exactly the pinned commit, and moving them out would mean writing inside
    the frozen tree.  So the requirement here is the one that carries the meaning - HEAD at the
    pinned commit and **no modified or staged tracked file** - and every untracked path is listed
    with its sha256 in the record instead of being silently tolerated.  A stray file that could
    shadow an import of the teacher's own packages is still refused.
    """
    actual = subprocess.check_output(['git', '-C', str(args.source), 'rev-parse', 'HEAD'], text=True).strip()
    tracked = subprocess.check_output(
        ['git', '-C', str(args.source), 'status', '--porcelain', '--untracked-files=no'], text=True)
    if actual != COMMIT or tracked.strip():
        raise ValueError(f'NUMERICAL_SOURCE_BINDING_CHANGED head={actual} tracked_dirty={tracked!r}')
    untracked = [line[3:] for line in subprocess.check_output(
        ['git', '-C', str(args.source), 'status', '--porcelain', '--untracked-files=all'],
        text=True).splitlines() if line.startswith('??')]
    if any(part.startswith('stage_cutfem') for path in untracked for part in Path(path).parts):
        raise ValueError(f'UNTRACKED_FILE_INSIDE_A_TEACHER_PACKAGE {untracked}')
    untracked_record = {path: sha(args.source / path) for path in untracked
                        if (args.source / path).is_file()}
    sys.path.insert(0, str(args.source))
    from stage_cutfem_runtime.resources import require_target, read_cgroup
    from stage_cutfem_runtime.config import validate_allocation, RUNTIME_ROOT
    from stage_cutfem_runtime.provenance import verify_files
    require_target()
    allocation = validate_allocation(read_cgroup())
    runtime = {}
    for manifest in sorted(RUNTIME_ROOT.glob('*/RUNTIME_MANIFEST.json')):
        record = json.loads(manifest.read_text())
        verify_files(manifest.parent, record['files'])
        runtime[manifest.parent.name] = dict(sha256=sha(manifest), files=len(record['files']))
    if not {'r13_pardiso_v1', 'r17_algoim_v2', 'r22_gmpy2_v1'} <= set(runtime):
        raise ValueError('PINNED_NATIVE_RUNTIMES_MISSING')
    return dict(numerical_source=str(args.source), numerical_commit=actual,
                tracked_content_clean=True, untracked_files=untracked_record,
                cleanliness_rule='HEAD at the pinned commit and no modified or staged tracked file; '
                                 'untracked paths listed above, none inside a teacher package',
                runtime=runtime, allocation=allocation,
                executable=sys.executable, numpy=np.__version__,
                wrapper_sha256=sha(__file__), source_sha=args.source_sha, pins=pins)


def geometry(args):
    from stage_cutfem_graded.contract import from_case
    from stage_cutfem_graded.topology import compile_topology
    from stage_cutfem_multiconstraint.complete_trace import compile_trace
    from stage_cutfem_q2.space import node_ids
    from stage_cutfem_runtime.provenance import write_json
    base_case, row, sample = case_from_seat(args.seat)
    with np.load(row['trace_cache'], allow_pickle=False) as z:
        frozen = {k: z[k] for k in z.files}
    selected, attempts = [], []
    cases = [('BASE_REPLAY', base_case, 0.0)] + [
        (f'PATH_{i + 1}', None, Fraction(e)) for i, e in enumerate(EPSILONS)]
    for label, row_case, initial in cases:
        for halving in range(1 if row_case is not None else 9):
            out = args.output / f'{label}_H{halving}'
            out.mkdir(exist_ok=False)
            tick = time.perf_counter()
            epsilon = None if row_case is not None else initial / (2 ** halving)
            current = row_case if row_case is not None else perturbed_case(
                base_case, epsilon, f'CUT_TAU_{int(args.seat)}_{label}_H{halving}')
            write(out / 'CASE.json', current)
            write(out / 'PROTOCOL.json', dict(json.loads((args.output / 'PROTOCOL.json').read_text()),
                                              phase='geometry_trace_only', initial_epsilon=str(initial),
                                              actual_epsilon=str(epsilon), halving=halving))
            c = from_case(current, 32)
            topology = compile_topology(c, workers=args.workers)
            write_json(out / 'TOPOLOGY.json', topology)
            if topology['status'] != 'TRUE_TOPOLOGY_CERTIFIED' or topology['component_count'] != 1:
                result = dict(label=label, halving=halving, epsilon=str(epsilon), admitted=False,
                              status='TOPOLOGY_FAILURE_RETAINED', directory=str(out),
                              seconds=time.perf_counter() - tick)
                write(out / 'RESULT.json', result); attempts.append(result)
                break
            cells = np.asarray(topology['active_parent_cells'], dtype=np.int32)
            nodes = np.unique(np.stack([node_ids(32, i) for i in cells]))
            body = dict(contract=c, nodes=nodes, topology=topology)
            compiled = compile_trace(body, coordinate_convention='max_pivot_geometric_residuals_v2')
            write_json(out / 'COMPLETE_TRACE.json', compiled['record'])
            np.save(out / 'NODES.npy', nodes)
            cache = trace_cache(body, compiled)
            np.savez(out / 'TRACE_CACHE.npz', **cache)
            comparison = compare(frozen, cache)
            comparison['candidate_sha256'] = sha(out / 'TRACE_CACHE.npz')
            comparison['frozen_cache'] = row['trace_cache']
            comparison['frozen_cache_sha256'] = sha(row['trace_cache'])
            write(out / 'TRACE_COMPARISON.json', comparison)
            result = dict(label=label, halving=halving, epsilon=str(epsilon), directory=str(out),
                          status='ADMITTED_SAME_BOX_TRACE' if comparison['admitted'] else 'REJECTED_BOX_TRACE_CHANGE',
                          admitted=comparison['admitted'], q=int(3 * len(cache['kind'])),
                          body_dofs=int(3 * len(nodes)), active_cells=int(len(cells)),
                          seconds=time.perf_counter() - tick, trace_comparison=comparison)
            write(out / 'RESULT.json', result); attempts.append(result)
            print(json.dumps({k: result[k] for k in
                              ('label', 'halving', 'epsilon', 'status', 'q', 'active_cells', 'seconds')}), flush=True)
            if row_case is not None:
                if not comparison['same_complete_trace_original_rule'] or not comparison['same_order']:
                    raise ValueError('ZERO_PERTURBATION_MUST_REPLAY_THE_FROZEN_CACHE_EXACTLY')
                break
            if comparison['admitted']:
                selected.append(result)
                break
    write(args.output / 'SELECTED.json', dict(seat=int(args.seat), selected=selected, attempts=attempts,
                                              maximum_new_labels=len(EPSILONS), numerical_labels_generated=0,
                                              status='GEOMETRY_SCREEN_COMPLETE_NUMERICAL_LABELS_PENDING'))


def numeric(args):
    from stage_cutfem_graded.contract import from_case
    from stage_cutfem_multiconstraint.body import build
    from stage_cutfem_gp.assembly import build as build_gp
    from stage_cutfem_preproduction.local_operator import export_local
    from stage_cutfem_runtime.record_io import write_record
    from stage_cutfem_runtime.resources import read_cgroup
    admission = json.loads((args.geometry / 'RESULT.json').read_text())
    if not admission['admitted'] or admission['label'] == 'BASE_REPLAY':
        raise ValueError('ONLY_ADMITTED_PERTURBED_CANDIDATES_ENTER_NUMERICAL_PRODUCTION')
    comparison = json.loads((args.geometry / 'TRACE_COMPARISON.json').read_text())
    if sha(args.geometry / 'TRACE_CACHE.npz') != comparison['candidate_sha256']:
        raise ValueError('ADMITTED_TRACE_CACHE_CHANGED')
    case = json.loads((args.geometry / 'CASE.json').read_text())
    c = from_case(case, 32)
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
    result.update(geometry=str(args.geometry), epsilon=admission['epsilon'], case_parameters=case,
                  volume_seconds=volume_seconds, gp_seconds=gp_seconds,
                  operator_seconds=time.perf_counter() - mark, total_seconds=time.perf_counter() - tick,
                  cgroup_end=read_cgroup(), spectrum_pending=True, diagnostic_only=True,
                  full_factor_network_training=False,
                  physical_geometry_error_separately_qualified=body['record']['geometry_error_separately_qualified'])
    # The box rows of the exported trace must be the admitted ones; the cut rows are recorded.
    with np.load(args.output / 'operator/REGISTRY.npz') as registry:
        with np.load(args.geometry / 'TRACE_CACHE.npz', allow_pickle=False) as z:
            expected = {k: z[k] for k in z.files}
        kind = np.asarray(expected['kind']); nb = int((kind == 0).sum())
        indptr = np.asarray(expected['indptr'])
        if not np.array_equal(indptr[:nb + 1], np.arange(nb + 1)):
            raise ValueError('BOX_TRACE_ROWS_ARE_NOT_SINGLE_NODE')
        support = np.asarray(expected['background_nodes'])[np.asarray(expected['indices'])[:nb]]
        # `boundary` is the FULL trace; the box part is `box_boundary_original`.  For a box-only
        # cell the two coincide, which is why the 0328 wrapper could index `boundary` directly.
        bbo = registry['box_boundary_original']
        if len(bbo) != 3 * nb or len(registry['boundary']) != 3 * len(kind):
            raise ValueError(f'EXPORT_TRACE_SHAPE_MISMATCH box={len(bbo)} expected={3 * nb} '
                             f'full={len(registry["boundary"])} expected={3 * len(kind)}')
        actual = registry['nodes'][bbo[::3] // 3]
        if not np.array_equal(actual, support):
            raise ValueError('EXPORT_BOX_TRACE_DOES_NOT_MATCH_ADMITTED_TRACE')
        result['export_trace_check'] = dict(box_coordinates=nb,
                                           cut_coordinates=int(len(kind) - nb),
                                           box_support_identical_in_order=True,
                                           rule='the box rows of the export, taken from '
                                                'box_boundary_original, match the admitted trace '
                                                'exactly and in order; the 0328 wrapper indexed '
                                                '`boundary` and refused any seat with kind-1 rows')
    write(args.output / 'RESULT.json', result)
    print(json.dumps(dict(phase='numeric_complete', case=case['case_id'], status=result['status'],
                          seconds=result['total_seconds'])), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['geometry', 'numeric'])
    p.add_argument('--seat', type=int)
    for key in ('source', 'output', 'geometry'):
        p.add_argument('--' + key, type=Path)
    p.add_argument('--source-sha', required=True)
    p.add_argument('--workers', type=int, default=16)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    pins = {}
    if args.mode == 'geometry':
        _, row, _ = case_from_seat(args.seat)
        pins = dict(seat=int(args.seat), sample_sha256=sha(Path(row['packet']) / 'SAMPLE.json'),
                    trace_cache_sha256=sha(row['trace_cache']),
                    s_upper_sha256=sha(Path(row['packet']) / 'S_UPPER.npy'))
    protocol = dict(schema='CUT_TAU_LOCAL_GEOMETRY_V1', mode=args.mode,
                    purpose='gate 2 teacher side: is a CUT cell response differentiable in tau',
                    initial_epsilons=EPSILONS, maximum_halvings=8,
                    maximum_new_numeric_labels=len(EPSILONS),
                    n=32, body_space='Q2', E=1., nu=.3, gamma=1e-4, GP_orders=[1, 2],
                    internal_mechanics_dtype='float64', label_encoding='float64 packed-upper',
                    coordinate_convention='max_pivot_geometric_residuals_v2',
                    admission='kind-0 rows identical; kind-1 rows may move with tau',
                    exact_rational_base_times_one_plus_epsilon=True,
                    neural_training=False, validation_test_access=False, diagnostic_only=True,
                    original_teacher_and_labels_unchanged=True,
                    source_sha=args.source_sha, source_script_sha256=sha(__file__), pins=pins)
    write(args.output / 'PROTOCOL.json', protocol)
    if 'CUTFEM_EXECUTION_CONFIG' in os.environ:
        write(args.output / 'EXECUTION_CONTRACT.json',
              json.loads(Path(os.environ['CUTFEM_EXECUTION_CONFIG']).read_text()))
    started = time.perf_counter()
    try:
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
