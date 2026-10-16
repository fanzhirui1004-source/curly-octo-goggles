"""gyr_prep (surface P) vs the existing fast_prep4 bodies: equality of cells, nodes, dofs, ghost faces, box / cut nodes."""
import json, sys
from pathlib import Path
import numpy as np
ref, new = Path(sys.argv[1]), Path(sys.argv[2])
for case in sys.argv[3:]:
    r = {}
    for f in ('CELL_INDICES', 'NODES', 'dofs', 'GP_FACES', 'BOX_NODES', 'CUT_NODES', 'CUT_CELLS'):
        a, b = np.load(ref / case / f'{f}.npy'), np.load(new / case / f'{f}.npy')
        eq = a.shape == b.shape and np.array_equal(a, b)
        r[f] = 'same' if eq else f'{a.shape}->{b.shape}'
        if not eq and a.ndim == 2 and f in ('CELL_INDICES', 'GP_FACES'):
            sa = {tuple(x) for x in a.tolist()}; sb = {tuple(x) for x in b.tolist()}
            r[f] += f' only_ref={len(sa - sb)} only_new={len(sb - sa)}'
        elif not eq and a.ndim == 1:
            r[f] += f' only_ref={len(np.setdiff1d(a, b))} only_new={len(np.setdiff1d(b, a))}'
    print(json.dumps(dict(case=case, **r)), flush=True)
