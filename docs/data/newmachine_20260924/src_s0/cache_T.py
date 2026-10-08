"""Precompute the dense fp64 port operators (box U cut band) into <body>/<case>_portview/T64.npy (cache for lattice3).
Uses only the node lists (no teacher Cell), so the peak memory is the Schur-mode factorization alone."""
import sys, os, gc, json
from pathlib import Path
import numpy as np
import torch
import schur_encoder as SE

body = Path(sys.argv[1])
for case in sys.argv[2:]:
    d = body / case
    nodes = np.load(d / 'NODES.npy')
    port = np.isin(nodes, np.union1d(np.load(d / 'BOX_NODES.npy'), np.load(d / 'CUT_NODES.npy')))
    ids = nodes[port]
    pd = body / (case + '_portview'); pd.mkdir(exist_ok=True)
    for fn in ('NODES.npy', 'CELL_INDICES.npy', 'dofs.npy', 'GP_FACES.npy'):
        if not (pd / fn).exists():
            os.symlink(d / fn, pd / fn)
    np.save(pd / 'BOX_NODES.npy', ids)
    if (pd / 'T64.npy').exists():
        print(json.dumps(dict(case=case, cached=True)), flush=True); continue
    enc = SE.SchurEncoder(case, str(body), precision='fp64', log=lambda s_: None, sub=case + '_portview', out_host=True)
    T, st = enc.update()
    T = T.T.contiguous().cpu()
    enc.free(); enc.Tbuf = None; del enc; gc.collect(); torch.cuda.empty_cache()
    np.save(pd / 'T64.npy', T.numpy())
    print(json.dumps(dict(case=case, ports=int(T.shape[0]), seconds=st)), flush=True)
    del T; gc.collect()
