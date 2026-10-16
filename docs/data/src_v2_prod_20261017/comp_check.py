"""Per cell body: face-connected components of the active cells and whether each carries box nodes (a component without
ports would make K_II singular)."""
import sys, json
from itertools import product
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
body = Path(sys.argv[1])
for case in sys.argv[2:]:
    d = body / case
    cells = np.load(d / 'CELL_INDICES.npy').astype(np.int64); box = set(np.load(d / 'BOX_NODES.npy').tolist())
    n = json.loads((d / 'PREP.json').read_text())['n']; n2 = 2 * n + 1
    lookup = {tuple(x): k for k, x in enumerate(cells.tolist())}
    ei, ej = [], []
    for k, x in enumerate(cells.tolist()):
        for a in range(3):
            y = list(x); y[a] += 1
            j = lookup.get(tuple(y))
            if j is not None:
                ei.append(k); ej.append(j)
    nc, lab = connected_components(sparse.coo_matrix((np.ones(len(ei)), (ei, ej)), shape=(len(cells),) * 2), directed=False)
    offs = np.asarray(list(product(range(3), repeat=3)))
    ids = np.ravel_multi_index((2 * cells[:, None, :] + offs[None]).transpose(2, 0, 1), (n2,) * 3)
    ports = [bool(set(np.unique(ids[lab == c]).tolist()) & box) for c in range(nc)]
    print(json.dumps(dict(case=case, components=int(nc), sizes=np.bincount(lab).tolist(), without_ports=int(sum(not p for p in ports)))))
