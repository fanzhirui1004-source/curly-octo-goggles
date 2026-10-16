"""Route 7: the frozen compile_trace ('max_pivot_geometric_residuals_v2' only) with its two exact eliminations and the
reverse elimination of the direct sum done by exact_trace (C++ with GMP rationals).

Everything else is the frozen code copied verbatim from stage_cutfem_multiconstraint.complete_trace and
.pivot_coordinates, including every exact certificate. Exact arithmetic makes the eliminated rows unique given the
pivot rule, so the result must equal the frozen one bit for bit; compare_trace.py checks that (coordinate digest,
P, inverse, L, boundary, box).
"""
import subprocess, time
from pathlib import Path
import numpy as np
from scipy import sparse
from stage_cutfem_q2.trace import exact_basis_row, sextic_triangle_points
from stage_cutfem_full_interface.full_trace import _matrix, _subtract
from stage_cutfem_full_interface.rational_polytope import polytope, box_planes
from stage_cutfem_runtime.provenance import sha256
from stage_cutfem_full_interface.contracts import digest
from stage_cutfem_full_interface.exact_numbers import BACKEND as EXACT_BACKEND, ONE, exact_row, exact
from stage_cutfem_multiconstraint.topology import canonical
from stage_cutfem_multiconstraint.trace import box_ports
import stage_cutfem_multiconstraint.complete_trace as frozen_complete_trace

EXE = Path(__file__).resolve().with_name('exact_trace')


def _run(mode, rows, ncol):
    text = [f'{mode} {ncol} {len(rows)}']
    for r in rows:
        text.append(' '.join([str(len(r))] + [f'{c} {v}' for c, v in r.items()]))
    out = subprocess.run([str(EXE)], input='\n'.join(text) + '\n', capture_output=True, text=True, check=True).stdout
    return out.split('\n')


def _row(line):
    t = line.split()
    return {int(t[1 + 2 * i]): exact(t[2 + 2 * i]) for i in range(int(t[0]))}


def eliminate(rows, ncol, timings, with_combinations):
    t = time.perf_counter()
    out = _run('max' if with_combinations else 'maxnc', rows, ncol)
    timings['cxx_max_pivot_and_direct_sum'] = time.perf_counter() - t
    t = time.perf_counter()
    n = int(out[0].split()[1])
    pairs = [tuple(map(int, line.split())) for line in out[1:1 + n]]
    selected, pivots = [p[0] for p in pairs], [p[1] for p in pairs]
    i = 1 + n
    assert out[i] == 'basis'
    basis = [_row(out[i + 1 + j]) for j in range(n)]
    i += 1 + n
    combinations = None
    if with_combinations:
        assert out[i] == 'combinations'
        combinations = [_row(out[i + 1 + j]) for j in range(n)]
        i += 1 + n
    assert out[i] == 'augmented'
    augmented = [_row(out[i + 1 + j]) for j in range(n)]
    timings['parse_cxx_output'] = time.perf_counter() - t
    return selected, basis, combinations, pivots, augmented


def first_pivot_selected(rows, ncol):
    out = _run('first', rows, ncol)
    n = int(out[0].split()[1])
    return [int(line.split()[0]) for line in out[1:1 + n]]


def direct_sum_from_augmented(rows, pivots, augmented, dimension):
    """Frozen direct_sum_with_pivots after its reverse elimination (done in C++), including its exact identity check."""
    if len(rows) != len(pivots) or len(set(pivots)) != len(pivots):
        raise ValueError('MAX_PIVOT_REGISTRY')
    positions = {pivot: i for i, pivot in enumerate(pivots)}
    for i, row in enumerate(rows):
        if row.get(pivots[i]) != 1:
            raise ValueError('MAX_PIVOT_EXPECTED_NORMALIZED_ECHELON')
        for column, value in row.items():
            target = positions.get(column)
            if target is not None and target < i and value:
                raise ValueError('MAX_PIVOT_EXPECTED_NORMALIZED_ECHELON')
    pivot_set = set(pivots)
    free = [column for column in range(dimension) if column not in pivot_set]
    free_positions = {column: j for j, column in enumerate(free)}
    q, null = [dict() for _ in range(dimension)], [dict() for _ in range(dimension)]
    for i, pivot in enumerate(pivots):
        q[pivot] = {column-dimension: value for column, value in augmented[i].items() if column >= dimension}
        null[pivot] = {free_positions[column]: -augmented[i][column]
                       for column in sorted(augmented[i]) if column in free_positions}
    for j, column in enumerate(free):
        null[column] = {j: ONE}
    for i, row in enumerate(rows):
        action, kernel_action = {}, {}
        for column, value in row.items():
            _subtract(action, q[column], -value)
            _subtract(kernel_action, null[column], -value)
        if action != {i: ONE} or kernel_action:
            raise ValueError('MAX_PIVOT_EXACT_DIRECT_SUM_IDENTITY')
    return q, null, pivots, free


def compile_trace(body, coordinate_convention='max_pivot_geometric_residuals_v2', with_combinations=True):
    """with_combinations=False skips only the combination rows of the audit record (the coordinates are unaffected);
    the record's digest is then taken without that field and labelled as such."""
    if coordinate_convention != 'max_pivot_geometric_residuals_v2':
        raise ValueError('fast path implements max_pivot_geometric_residuals_v2 only')
    start=time.perf_counter();c=body['contract'];nodes=body['nodes'];positions={int(node):i for i,node in enumerate(nodes)}
    phases = {}
    phase_start = start
    def finish_phase(name):
        nonlocal phase_start
        now = time.perf_counter()
        phases[name] = now-phase_start
        phase_start = now
    _,box=box_ports(body);rows=[{int(i):ONE} for i in box];labels=[dict(kind='box_coefficient',node=int(nodes[i])) for i in box]
    normal,offset=c.rational_plane();functionals={}
    for patch in body['topology']['patches']:
        if patch['tag']!='MACRO_CUT_FACE':continue
        index=patch['parent'];ambient=polytope(canonical(box_planes(index,c.n)+[(normal,offset,'MACRO_CUT_FACE')]))
        polygons=[polygon for _,_,tag,polygon in ambient['faces'] if 'MACRO_CUT_FACE' in tag.split('|')]
        if len(polygons)!=1:raise ValueError('COMPLETE_CUT_TRACE_AMBIENT_PATCH')
        for point in sextic_triangle_points(polygons[0]):
            row=exact_row(exact_basis_row(point,index,c.n,positions));key=tuple(point)
            if key in functionals and functionals[key]!=row:raise ValueError('EXACT_CUT_TRACE_SEAM_DISAGREEMENT')
            functionals[key]=row
    for key,row in sorted(functionals.items()):
        rows.append(row);labels.append(dict(kind='cut_geometric_functional',point=[str(x) for x in key]))
    finish_phase('geometric_functionals')
    inner = {}
    selected,basis,combinations,pivots,augmented=eliminate(rows,len(nodes),inner,with_combinations)
    finish_phase('geometric_residuals')
    q,null,pivots,free=direct_sum_from_augmented(basis,pivots,augmented,len(nodes))
    finish_phase('exact_direct_sum')
    old_selected=first_pivot_selected(rows,len(nodes))
    if selected != old_selected:
        raise ValueError('MAX_PIVOT_COORDINATE_CHANGE_ALTERED_INDEPENDENT_ROW_IDENTITIES')
    finish_phase('independent_historical_row_identities')
    coordinate_policy = 'COMPLETE_TRACE_PIVOT_V2.json'
    if selected[:len(box)]!=list(range(len(box))) or basis[:len(box)]!=rows[:len(box)]:raise ValueError('COMPLETE_TRACE_CHANGED_BOX_COORDINATES')
    for row in rows:
        action={}
        for column,value in row.items():_subtract(action,null[column],-value)
        if action:raise ValueError('COMPLETE_MACRO_TRACE_HAS_UNREPRESENTED_DIRECTION')
    finish_phase('all_geometric_trace_kernel_certificate')
    Q,N,L=_matrix(q,len(selected)),_matrix(null,len(free)),_matrix(basis,len(nodes))
    P=sparse.hstack((Q,N),format='csr');extract=sparse.csr_matrix((np.ones(len(free)),(np.arange(len(free)),free)),shape=(len(free),len(nodes)))
    inverse=sparse.vstack((L,extract),format='csr');identity=sparse.eye(len(nodes),format='csr')
    replay=float(sparse.linalg.norm(inverse@P-identity)/max(sparse.linalg.norm(P)*sparse.linalg.norm(inverse),1.))
    if replay>1e-12:raise ValueError('COMPLETE_TRACE_FLOATING_COORDINATE_REPLAY')
    finish_phase('floating_matrices_and_replay')
    def serialize(items):return [{str(k):str(v) for k,v in sorted(row.items())} for row in items]
    exact_record=dict(labels=labels,all_functionals=serialize(rows),selected_rows=selected,
        coordinate_functionals=serialize(basis),combinations=serialize(combinations) if with_combinations else None,
        Q=serialize(q),N=serialize(null),pivot_columns=pivots,free_columns=free)
    if not with_combinations:
        exact_record.pop('combinations')
    finish_phase('exact_record_serialization')
    coordinate_sha256=digest(exact_record)
    finish_phase('exact_semantic_digest')
    policy_dir = Path(frozen_complete_trace.__file__)
    record=dict(policy_sha256=sha256(policy_dir.with_name(coordinate_policy)),
        scalar_body_dimension=len(nodes),scalar_box_trace_dimension=len(box),scalar_complete_trace_dimension=len(selected),
        additional_cut_trace_dimension=len(selected)-len(box),scalar_trace_free_dimension=len(free),
        all_geometric_functionals=len(rows),exact_LQ_identity=True,exact_all_trace_N_zero=True,exact_invertible_direct_sum=True,
        box_coefficients_unchanged=True,cut_coordinates_are_residual_functionals=True,low_dimensional_projection=False,
        cut_free_traction_is_applied_only_at_global_solve=True,floating_inverse_replay=replay,
        P_norm_frobenius=float(sparse.linalg.norm(P)),inverse_norm_frobenius=float(sparse.linalg.norm(inverse)),
        P_nnz=P.nnz,inverse_nnz=inverse.nnz,coordinate_sha256=coordinate_sha256,exact=exact_record,seconds=time.perf_counter()-start)
    record.update(coordinate_convention=coordinate_convention,
                  historical_coordinate_policy_sha256=sha256(policy_dir.with_name('COMPLETE_TRACE_POLICY.json')),
                  exact_independent_geometric_row_identities_unchanged=True,
                  complete_trace_dimension_unchanged=True,
                  explicit_coordinate_migration_required=True)
    result = dict(P=sparse.kron(P,sparse.eye(3),format='csr'),inverse=sparse.kron(inverse,sparse.eye(3),format='csr'),
        L=sparse.kron(L,sparse.eye(3),format='csr'),boundary=np.arange(3*len(selected)),box=box,record=record)
    finish_phase('record_metadata_and_vector_expansion')
    record.update(seconds=time.perf_counter()-start,phase_seconds=phases,inner_seconds=inner,
                  coordinate_digest_covers_combinations=bool(with_combinations),
                  implementation='route7_cxx_gmp_exact_elimination_v1',
                  exact_arithmetic_backend=EXACT_BACKEND,
                  timing_and_implementation_excluded_from_coordinate_digest=True)
    return result
