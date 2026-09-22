import json
import numpy as np
from scipy import sparse
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

