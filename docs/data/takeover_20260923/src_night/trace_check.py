"""Frozen compile_trace on a locally recomputed second-batch geometry; compare with the packet's TRACE.npz."""
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gp_check_body import load_body
case = sys.argv[1]
W = Path('/root/autodl-tmp/CLAUDE_TAKEOVER_20260923/COVER_G')
pk = Path('/root/autodl-tmp/CUTFEM_FRESH_GP_20260921/packets') / case
from stage_cutfem_multiconstraint.complete_trace import compile_trace
body = load_body(W / 'runs' / (case + '_G'))
t = time.perf_counter(); tr = compile_trace(body, coordinate_convention='max_pivot_geometric_residuals_v2'); sec = time.perf_counter() - t
ref = np.load(pk / 'TRACE.npz')
out = dict(case=case, seconds=sec, keys=sorted(tr.keys()) if isinstance(tr, dict) else str(type(tr)), ref_keys=list(ref.files))
for k in ref.files:
    if isinstance(tr, dict) and k in tr:
        a, b = np.asarray(tr[k]), ref[k]
        out[k] = dict(shape=list(a.shape), ref_shape=list(b.shape), equal=bool(a.shape == b.shape and np.array_equal(a, b)))
print(json.dumps(out, default=str)[:3000])
