"""Route 7 step 1: where does the frozen compile_trace spend its time, and how big is the exact elimination?

Runs the frozen compile_trace unchanged (from the private COVER_G/src copy) on one second-version case, with a
counting wrapper around the sparse row update used by the max-pivot elimination, and saves the elimination's input
rows and outputs so a faster implementation can be checked bit for bit offline.
Usage: r7_profile.py <case> <out_dir>
"""
import json, pickle, sys, time
from pathlib import Path
sys.path.insert(0, '/root/autodl-tmp/CLAUDE_TAKEOVER_20260923/xcase_src_11')
from gp_check_body import load_body
from stage_cutfem_multiconstraint import complete_trace as CT
from stage_cutfem_multiconstraint import pivot_coordinates as PC
from stage_cutfem_full_interface import full_trace as FT
from stage_cutfem_full_interface.exact_numbers import BACKEND

T = Path('/root/autodl-tmp/CLAUDE_TAKEOVER_20260923')
case, out = sys.argv[1], Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
t0 = time.perf_counter()
body = load_body(T / 'COVER_G' / 'runs' / (case + '_G'))
load_seconds = time.perf_counter() - t0
count = dict(calls=0, entries=0)
original_subtract = FT._subtract
def counting_subtract(row, basis, factor):
    count['calls'] += 1; count['entries'] += len(basis)
    return original_subtract(row, basis, factor)
saved = {}
original_elim = PC.geometric_residuals_max_pivot
def saving_elim(rows):
    saved['rows'] = [{int(k): str(v) for k, v in r.items()} for r in rows]
    t = time.perf_counter(); result = original_elim(rows); saved['elim_seconds'] = time.perf_counter() - t
    saved['result'] = result
    return result
PC._subtract = counting_subtract
PC.geometric_residuals_max_pivot = saving_elim
t = time.perf_counter()
res = CT.compile_trace(body, coordinate_convention='max_pivot_geometric_residuals_v2')
total = time.perf_counter() - t
rec = res['record']
selected, basis, combinations, pivots = saved['result']
def bits(x):
    return max(int(x.numerator).bit_length(), int(x.denominator).bit_length())
stats = dict(case=case, backend=BACKEND, load_body_seconds=load_seconds, compile_seconds=total, elim_seconds=saved['elim_seconds'],
             phase_seconds=rec['phase_seconds'], rows=len(saved['rows']), selected=len(selected), nodes=len(body['nodes']),
             row_nnz_in=sum(len(r) for r in saved['rows']), basis_nnz=sum(len(r) for r in basis),
             combination_nnz=sum(len(r) for r in combinations), subtract_calls=count['calls'], subtract_entries=count['entries'],
             basis_max_bits=max(bits(v) for r in basis for v in r.values()),
             combination_max_bits=max(bits(v) for r in combinations for v in r.values()),
             coordinate_sha256=rec['coordinate_sha256'])
(out / 'PROFILE.json').write_text(json.dumps(stats, indent=2))
with open(out / 'ELIM_IO.pkl', 'wb') as f:
    pickle.dump(dict(rows=saved['rows'], selected=selected, pivots=pivots,
                     basis=[{k: str(v) for k, v in r.items()} for r in basis],
                     combinations=[{k: str(v) for k, v in r.items()} for r in combinations],
                     n_nodes=len(body['nodes']), coordinate_sha256=rec['coordinate_sha256']), f)
print(json.dumps(stats))
