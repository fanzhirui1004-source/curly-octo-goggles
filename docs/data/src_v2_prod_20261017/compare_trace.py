"""Route 7 check: fast_trace (C++ exact elimination) against the frozen compile_trace outputs, bit for bit.

Against every case: P and inverse (the frozen outputs saved by cover_blocks.py in COVER_INPUTS/<case>) must be
exactly equal (same shape, same sparsity, zero difference). Where the frozen elimination's inputs and outputs were
saved (R7_01/<case>/ELIM_IO.pkl), also the selected rows, pivots, basis rows and combination rows, and the frozen
coordinate digest (which covers combinations) when the fast path is run with combinations.
Usage: compare_trace.py <case> <with_combinations 0|1>
"""
import json, pickle, sys, time
from pathlib import Path
import numpy as np
from scipy import sparse
sys.path.insert(0, '/root/autodl-tmp/CLAUDE_TAKEOVER_20260923/xcase_src_11')
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gp_check_body import load_body
import fast_trace

T = Path('/root/autodl-tmp/CLAUDE_TAKEOVER_20260923')
case, with_comb = sys.argv[1], bool(int(sys.argv[2]))
out = T / 'R7_02'; out.mkdir(exist_ok=True)
t0 = time.perf_counter()
body = load_body(T / 'COVER_G' / 'runs' / (case + '_G'))
load_seconds = time.perf_counter() - t0
t = time.perf_counter()
res = fast_trace.compile_trace(body, with_combinations=with_comb)
seconds = time.perf_counter() - t
rec = res['record']


def same(a, b):
    a, b = a.tocsr(), b.tocsr()
    if a.shape != b.shape:
        return False
    a.sort_indices(); b.sort_indices()
    return bool(a.nnz == b.nnz and np.array_equal(a.indptr, b.indptr) and np.array_equal(a.indices, b.indices)
                and np.array_equal(a.data, b.data))


ci = T / 'COVER_INPUTS' / case
row = dict(case=case, with_combinations=with_comb, load_body_seconds=load_seconds, fast_seconds=seconds,
           phase_seconds=rec['phase_seconds'], inner_seconds=rec['inner_seconds'],
           rows=rec['all_geometric_functionals'], selected=rec['scalar_complete_trace_dimension'],
           same_P_bitwise=same(res['P'], sparse.load_npz(ci / 'P.npz')),
           same_inverse_bitwise=same(res['inverse'], sparse.load_npz(ci / 'Pinv.npz')),
           coordinate_sha256=rec['coordinate_sha256'])
io = T / 'R7_01' / case / 'ELIM_IO.pkl'
if io.exists():
    frozen = pickle.loads(io.read_bytes())
    exact = rec['exact']
    row['same_selected'] = exact['selected_rows'] == frozen['selected']
    row['same_pivots'] = exact['pivot_columns'] == frozen['pivots']
    row['same_basis'] = exact['coordinate_functionals'] == [{str(k): v for k, v in sorted(r.items())} for r in frozen['basis']]
    if with_comb:
        row['same_combinations'] = exact['combinations'] == [{str(k): v for k, v in sorted(r.items())} for r in frozen['combinations']]
        row['same_coordinate_sha256'] = rec['coordinate_sha256'] == frozen['coordinate_sha256']
(out / f'{case}_comb{int(with_comb)}.json').write_text(json.dumps(row, indent=2))
print(json.dumps(row))
