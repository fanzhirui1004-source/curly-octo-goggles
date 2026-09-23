import json, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import polyref_prof2 as PP
ROOT = Path('/root/autodl-tmp/CUTFEM_FRESH_GP_20260921')
body = Path(sys.argv[1])
for case in sys.argv[2:]:
    ctx = json.loads((ROOT / 'packets' / case / 'FRESH_CONTEXT.json').read_text())
    n = int(ctx['n']); taus = [float(v) for v in ctx['case']['tau_corners']]
    normal = None if ctx['case'].get('normal') is None else [float(v) for v in ctx['case']['normal']]
    offset = None if normal is None else float(ctx['case']['offset'])
    cells = np.load(body / case / 'CELL_INDICES.npy')
    PP.cell_moments(cells, n, taus, normal, offset, 4, levels=1)      # warm-up
    PP.PROF.clear()
    torch.cuda.synchronize(); t = time.perf_counter()
    PP.cell_moments(cells, n, taus, normal, offset, 4, levels=1)
    torch.cuda.synchronize()
    print(json.dumps(dict(case=case, total=time.perf_counter() - t, **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in PP.PROF.items()})), flush=True)
