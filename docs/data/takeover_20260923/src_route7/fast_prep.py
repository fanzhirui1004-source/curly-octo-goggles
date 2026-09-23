"""Route 7: geometry -> trace coordinates P and ghost faces without the frozen G stage's cell quadrature.

P and the ghost faces depend only on the certified topology (active parent cells, nodes, boundary and cut patches);
the frozen body builder derives nodes and active cells from compile_topology alone and spends the rest of its time
on algoim quadrature, which the GPU encoder replaces. So: frozen compile_topology (unchanged) -> the same nodes and
active cells as the frozen body -> fast_trace (C++ exact elimination, no audit combinations) -> fast_gp (parallel
full-cell test). The ghost root diagnostic uses the frozen body's volume fractions here only because this script
checks against existing cases; in deployment they come from the GPU cell integration.
Checks per case: nodes, active cells and topology equal to the frozen G run; P and inverse bitwise equal to
COVER_INPUTS; faces equal to the published GP_FACES. Usage: fast_prep.py <workers> <case> [<case> ...]
"""
import json, sys, time
from pathlib import Path
import numpy as np
from scipy import sparse
sys.path.insert(0, '/root/autodl-tmp/CLAUDE_TAKEOVER_20260923/xcase_src_11')
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gp_check_body import load_body
from stage_cutfem_graded.contract import from_case
from stage_cutfem_graded.topology import compile_topology
from stage_cutfem_q2.space import node_ids
import fast_trace, fast_gp

T = Path('/root/autodl-tmp/CLAUDE_TAKEOVER_20260923'); R = Path('/root/autodl-tmp/CUTFEM_FRESH_GP_20260921/packets')
out = T / 'R7_04'
workers = int(sys.argv[1]) if len(sys.argv) > 1 else 1


def same(a, b):
    a, b = a.tocsr(), b.tocsr(); a.sort_indices(); b.sort_indices()
    return bool(a.shape == b.shape and a.nnz == b.nnz and np.array_equal(a.indptr, b.indptr)
                and np.array_equal(a.indices, b.indices) and np.array_equal(a.data, b.data))


def main(case):
    run = T / 'COVER_G' / 'runs' / (case + '_G')
    result = json.loads((run / 'RESULT.json').read_text())
    frozen = load_body(run)
    row = dict(case=case, workers=workers)
    t0 = time.perf_counter()
    c = from_case(result['case'], result['n'])
    topo = compile_topology(c, workers)
    row['topology_seconds'] = time.perf_counter() - t0
    if topo['status'] != 'TRUE_TOPOLOGY_CERTIFIED' or topo['component_count'] != 1:
        raise ValueError('TOPOLOGY')
    t = time.perf_counter()
    indices = np.asarray(topo['active_parent_cells'], dtype=np.int32)
    nodes = np.unique(np.stack([node_ids(c.n, i) for i in indices]))
    body = dict(contract=c, nodes=nodes, topology=topo, cell_indices=indices, active=list(range(len(indices))),
                record=dict(component_count=topo['component_count'], topology_proof=topo['proof']),
                moments=frozen['moments'])
    row['nodes_seconds'] = time.perf_counter() - t
    row['same_nodes'] = bool(np.array_equal(nodes, np.asarray(frozen['nodes'])))
    row['same_active_cells'] = bool(np.array_equal(indices, np.asarray(frozen['cell_indices'])))
    row['same_topology'] = json.dumps(topo, sort_keys=True) == json.dumps(frozen['topology'], sort_keys=True)
    t = time.perf_counter()
    tr = fast_trace.compile_trace(body, with_combinations=False)
    row['trace_seconds'] = time.perf_counter() - t
    t = time.perf_counter()
    faces, _ = fast_gp.select_faces(body, workers=workers)
    row['faces_seconds'] = time.perf_counter() - t
    ci = T / 'COVER_INPUTS' / case
    row['same_P_bitwise'] = same(tr['P'], sparse.load_npz(ci / 'P.npz'))
    row['same_inverse_bitwise'] = same(tr['inverse'], sparse.load_npz(ci / 'Pinv.npz'))
    row['same_faces'] = bool(np.array_equal(np.asarray(faces), np.load(R / case / 'CONTEXT' / 'GP_FACES.npy')))
    row['total_seconds'] = time.perf_counter() - t0
    (out / (case + '.json')).write_text(json.dumps(row, indent=2))
    print(json.dumps(row), flush=True)


if __name__ == '__main__':   # compile_topology uses a spawn pool, which re-imports this file in every worker
    out.mkdir(exist_ok=True)
    for case in sys.argv[2:]:
        main(case)
