"""Route 7: timing and equality of the closed-form full-sub-cube moments against the original batched integration."""
import json, sys, time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import polyref_torch as OLD
import polyref_torch_fast as NEW
ROOT = Path('/root/autodl-tmp/CUTFEM_FRESH_GP_20260921')
body = Path(sys.argv[1])
for case in sys.argv[2:]:
    ctx = json.loads((ROOT / 'packets' / case / 'FRESH_CONTEXT.json').read_text())
    n = int(ctx['n']); taus = [float(v) for v in ctx['case']['tau_corners']]
    normal = None if ctx['case'].get('normal') is None else [float(v) for v in ctx['case']['normal']]
    offset = None if normal is None else float(ctx['case']['offset'])
    cells = np.load(body / case / 'CELL_INDICES.npy')
    res = dict(case=case, cells=int(len(cells)))
    for name, mod in (('old', OLD), ('new', NEW), ('old2', OLD), ('new2', NEW)):
        torch.cuda.synchronize(); t = time.perf_counter()
        M = mod.cell_moments(cells, n, taus, normal, offset, 4, levels=1)
        torch.cuda.synchronize(); res[name + '_seconds'] = time.perf_counter() - t
        res[name] = M
    ref, new = res.pop('old'), res.pop('new'); res.pop('old2'); res.pop('new2')
    res['max_rel_diff'] = float((np.abs(new - ref).max(1) / np.abs(ref).max(1).clip(1e-300)).max())
    print(json.dumps(res), flush=True)
