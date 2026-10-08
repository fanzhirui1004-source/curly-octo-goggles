"""First batch: rebuild the G-stage body from the archive, run the frozen compile_trace, compare its P with the
packet's ORIGINAL_FROM_TRACE_FREE (the operator's coordinates)."""
import json, sys, tarfile, time
from pathlib import Path
import numpy as np
from scipy import sparse
sys.path.insert(0, str(Path(__file__).resolve().parent))
case = sys.argv[1]
R = Path('/root/autodl-tmp/CUTFEM_FRESH_GP_20260921')
work = Path('/root/autodl-tmp/CLAUDE_TAKEOVER_20260923/TRACE_CHECK') / case
inv = json.loads((R / 'source_archives' / case / 'INVENTORY.json').read_text())
g = next(s for s in inv['stages'] if s.endswith('_G'))
want = [n for n in inv['files'] if n.startswith(g + '/') and (n.count('/') == 1 or n.startswith(g + '/body/') and n.split('/')[-1] in (
    'BODY.json', 'NODES.npy', 'CELL_INDICES.npy', 'cell_moments.npy', 'G_data.npy', 'G_indices.npy', 'V_data.npy', 'V_indices.npy',
    'K_data.npy', 'K_indices.npy', 'K_indptr.npy', 'TOPOLOGY.json') or n.startswith(g + '/body/bank/'))]
if not (work / 'done').exists():
    with tarfile.open(R / 'source_archives' / case / 'SOURCE.tar', 'r:') as t:
        for n in want:
            d = work / n; d.parent.mkdir(parents=True, exist_ok=True)
            with t.extractfile(t.getmember(n)) as s, open(d, 'wb') as f:
                f.write(s.read())
    (work / 'done').write_text('ok')
from gp_check_body import load_body
body = load_body(work / g)
from stage_cutfem_multiconstraint.complete_trace import compile_trace
t = time.perf_counter(); tr = compile_trace(body, coordinate_convention='max_pivot_geometric_residuals_v2'); sec = time.perf_counter() - t
Pp = sparse.load_npz(R / 'packets' / case / 'ORIGINAL_FROM_TRACE_FREE.npz').tocsr()
P = sparse.csr_matrix(tr['P'])
res = dict(case=case, seconds=sec, P_shape=list(P.shape), packet_shape=list(Pp.shape), P_nnz=int(P.nnz), packet_nnz=int(Pp.nnz))
if P.shape == Pp.shape:
    res['relative_difference'] = float(sparse.linalg.norm(P - Pp) / sparse.linalg.norm(Pp))
    # boundary columns might be permuted: compare column sets by hashing columns
    m = json.loads((R / 'packets' / case / 'SAMPLE.json').read_text())['full_trace_dimension']
    res['m'] = m
    res['interior_block_equal'] = bool((abs(P[:, m:] - Pp[:, m:])).max() == 0)
    trb = np.load(R / 'packets' / case / 'TRACE.npz')
    res['boundary_equal_packet_trace'] = bool(np.array_equal(np.asarray(tr['boundary']), trb['boundary']))
print(json.dumps(res))
