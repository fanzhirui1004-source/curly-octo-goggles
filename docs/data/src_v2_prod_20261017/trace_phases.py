"""Phase timing of the frozen compile_trace on one second-batch case (record['phase_seconds'])."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, '/root/autodl-tmp/CLAUDE_TAKEOVER_20260923/xcase_src_11')
from gp_check_body import load_body
from stage_cutfem_multiconstraint.complete_trace import compile_trace
case = sys.argv[1]
body = load_body(Path('/root/autodl-tmp/CLAUDE_TAKEOVER_20260923/COVER_G/runs') / (case + '_G'))
t = time.perf_counter(); tr = compile_trace(body, coordinate_convention='max_pivot_geometric_residuals_v2')
r = tr['record']
print(json.dumps(dict(case=case, seconds=time.perf_counter() - t, phases=r['phase_seconds'], backend=str(r['exact_arithmetic_backend']),
                      functionals=r['all_geometric_functionals'], complete=r['scalar_complete_trace_dimension'], box=r['scalar_box_trace_dimension'],
                      nodes=r['scalar_body_dimension'])))
