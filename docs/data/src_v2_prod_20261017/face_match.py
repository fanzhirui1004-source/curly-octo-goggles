"""Shared-face port consistency of a lattice layout: for each pair of face neighbours, the box nodes of cell A on its +d
face vs the box nodes of cell B on its -d face (same in-face grid coordinates); plus kept-component info from PREP.json."""
import sys, json
from pathlib import Path
import numpy as np
L = json.loads(Path(sys.argv[1]).read_text()); body = Path(sys.argv[2])
cells = {tuple(c['position']): c['case'] for c in L['cells']}
out = []
for p, case in cells.items():
    prep = json.loads((body / case / 'PREP.json').read_text())
    n = int(prep.get('n', 32)); n2 = 2 * n + 1
    for d in range(3):
        q = list(p); q[d] += 1; q = tuple(q)
        if q not in cells:
            continue
        def face(cs, side):
            b = np.load(body / cs / 'BOX_NODES.npy')
            g = np.stack(np.unravel_index(b, (n2,) * 3), 1)                  # BOX_NODES: grid node ids
            sel = g[:, d] == (n2 - 1 if side else 0)
            rest = np.delete(g[sel], d, 1)
            return set(map(tuple, rest))
        a, bb = face(case, 1), face(cells[q], 0)
        out.append(dict(a=case, b=cells[q], axis=d, a_nodes=len(a), b_nodes=len(bb), common=len(a & bb), only_a=len(a - bb), only_b=len(bb - a)))
for r in out:
    print(json.dumps(r))
