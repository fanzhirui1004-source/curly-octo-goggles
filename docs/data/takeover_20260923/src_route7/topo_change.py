"""How often does the discrete topology (active cells, ghost faces, box nodes) change under a design step?"""
import json, sys
from pathlib import Path
import numpy as np
base = Path(sys.argv[1])
for case in sys.argv[2:]:
    b = base / case
    ref = {k: np.load(b / f) for k, f in (('cells', 'CELL_INDICES.npy'), ('faces', 'GP_FACES.npy'), ('box', 'BOX_NODES.npy'), ('nodes', 'NODES.npy'))}
    for d in sorted(base.glob(case + '_eps*')):
        cur = {k: np.load(d / f) for k, f in (('cells', 'CELL_INDICES.npy'), ('faces', 'GP_FACES.npy'), ('box', 'BOX_NODES.npy'), ('nodes', 'NODES.npy'))}
        rc = {tuple(x) for x in ref['cells']}; cc = {tuple(x) for x in cur['cells']}
        row = dict(case=case, eps=d.name.split('_eps')[1], cells_added=len(cc - rc), cells_removed=len(rc - cc),
                   nodes_added=int(len(np.setdiff1d(cur['nodes'], ref['nodes']))), nodes_removed=int(len(np.setdiff1d(ref['nodes'], cur['nodes']))),
                   box_nodes_added=int(len(np.setdiff1d(cur['box'], ref['box']))), box_nodes_removed=int(len(np.setdiff1d(ref['box'], cur['box']))),
                   faces_ref=int(len(ref['faces'])), faces_cur=int(len(cur['faces'])),
                   same_topology=bool(len(cc ^ rc) == 0 and np.array_equal(ref['box'], cur['box']) and ref['faces'].shape == cur['faces'].shape and np.array_equal(ref['faces'], cur['faces'])))
        print(json.dumps(row), flush=True)
