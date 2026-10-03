"""Validate the lattice harness (exact operator ~ 0 error) and measure the zero-parameter baselines (graph lifting)."""
import json, sys, time, gc
import numpy as np
import torch
import lattice3 as LT
import ops as OP

body, out = sys.argv[1], sys.argv[2]
FULL = 'fresh_train_0020_full'
import os
tests = [(t, FULL) for t in os.environ.get('LAT_TESTS', 'fresh_train_0020_cover01_r1,fresh_train_0020_cover01_r2,' + FULL).split(',')]
rec = []
for test, nbr in tests:
    for cfg in ('x', 'y'):
        t0 = time.perf_counter()
        lat = LT.build(test, nbr, cfg, body)
        ref = lat.reference()
        row = dict(test=test, nbr=nbr, config=cfg, free=int(len(lat.free)), loads=lat.labels,
                   ref_compliance=ref['compliance'].tolist(), ref_seconds=ref['seconds'],
                   cut_load_outside=getattr(lat, 'cut_load_outside', None), cut_area=getattr(lat, 'cut_area', None))
        exact = [OP.ExactOp(cd['cell'], cd['T']) for cd in lat.cells]
        row['exact'] = lat.compare(lat.evaluate(exact))
        for w in ('volume', 'uniform'):
            C = lat.cells[0]['cell']
            gl = OP.GraphLift(C, weight=w)
            op = OP.ExtensionOp(C, gl.ext, gl.ext_T)
            res = lat.evaluate([op, exact[1]], maxit=400)
            row[f'graphlift_{w}'] = lat.compare(res)
            gl.sol.free(); del gl, op
            gc.collect(); torch.cuda.empty_cache()
        row['seconds'] = time.perf_counter() - t0
        print(json.dumps({k: (v if k not in ('loads', 'ref_compliance') else None) for k, v in row.items()}), flush=True)
        rec.append(row)
        del lat, exact; gc.collect(); torch.cuda.empty_cache()
    if test != FULL:
        LT._CACHE.pop(test, None); gc.collect(); torch.cuda.empty_cache()
        open(out, 'w').write(json.dumps(rec, indent=1))
print('DONE', flush=True)
