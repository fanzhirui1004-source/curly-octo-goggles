"""Frozen GP face selection + template assembly on a locally recomputed second-batch geometry run;
compare with the producer's published GP context (faces, dof map, support)."""
import json, sys, time
from pathlib import Path
import numpy as np
case = sys.argv[1]
W = Path('/root/autodl-tmp/CLAUDE_TAKEOVER_20260923/COVER_G')
R = Path('/root/autodl-tmp/CUTFEM_FRESH_GP_20260921/packets') / case / 'CONTEXT'
from scipy import sparse
from stage_cutfem_gp import assembly
from stage_cutfem_runtime.record_io import read_record
from stage_cutfem_runtime.sparse_assembly import index_dtype


def load_body(run):
    # Same construction as stage_cutfem_multiconstraint.chain.load_body, without the runner's FILES.json
    # manifest (this geometry run was produced by calling the frozen trial directly and verified bitwise).
    result = json.loads((run / 'RESULT.json').read_text()); topology = json.loads((run / 'TOPOLOGY.json').read_text())
    if result['status'] != 'GEOMETRY_INTEGRATION_LOCAL_TRACE_QUALIFIED_COMPLETE_CHAIN_PENDING':
        raise ValueError('GEOMETRY_NOT_QUALIFIED')
    from stage_cutfem_graded.contract import from_case
    c = from_case(result['case'], result['n'])
    d = run / 'body'; record = read_record(d / 'BODY.json')
    nodes = np.load(d / 'NODES.npy', mmap_mode='r'); indices = np.load(d / 'CELL_INDICES.npy', mmap_mode='r'); cells = len(indices)
    mats = {}
    for name, width, dim in [('G', 81, 3 * len(nodes)), ('V', 27, len(nodes))]:
        data = np.load(d / (name + '_data.npy'), mmap_mode='r').ravel(); cols = np.load(d / (name + '_indices.npy'), mmap_mode='r').ravel()
        kind = index_dtype(dim, len(data))
        mats[name] = sparse.csr_matrix((data, cols, np.arange(cells * width + 1, dtype=kind) * width), shape=(cells * width, dim), copy=False)
    upper = sparse.csr_matrix(tuple(np.load(d / (n_ + '.npy'), mmap_mode='r') for n_ in ['K_data', 'K_indices', 'K_indptr']), shape=(3 * len(nodes),) * 2, copy=False)
    points = np.stack(np.unravel_index(nodes, (2 * c.n + 1,) * 3), axis=1) / (2 * c.n)
    return dict(contract=c, nodes=nodes, points=points, K_upper=upper, **mats, moments=np.load(d / 'cell_moments.npy', mmap_mode='r'),
                active=list(range(cells)), cell_indices=indices, topology=topology, record=record)


t = time.perf_counter(); body = load_body(W / 'runs' / (case + '_G')); t_load = time.perf_counter() - t
t = time.perf_counter(); faces, coverage = assembly.select_faces(body); t_sel = time.perf_counter() - t
faces = np.asarray(faces)
ref = np.load(R / 'GP_FACES.npy')
same = faces.shape == ref.shape and np.array_equal(faces, ref)
out = W / 'gp_check'; out.mkdir(exist_ok=True)
t_build = None; sup = None
if len(sys.argv) > 2 and sys.argv[2] == 'build':
    t = time.perf_counter(); ghost = assembly.build(body, out / case, workers=2); t_build = time.perf_counter() - t
    sup = np.load(out / case / 'global_support.npy') if (out / case / 'global_support.npy').exists() else None
res = dict(case=case, faces=int(len(faces)), faces_equal_published=bool(same), load_seconds=t_load, select_seconds=t_sel,
           build_seconds=t_build, support_equal_published=None if sup is None else bool(np.array_equal(sup, np.load(R / 'GP_GLOBAL_SUPPORT.npy'))),
           coverage={k: v for k, v in coverage.items() if not isinstance(v, (list, dict))})
print(json.dumps(res))
