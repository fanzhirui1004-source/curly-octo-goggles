#!/usr/bin/env python3
"""How much of the domain does each multiscale lifting layer actually touch?

multiscale_layers bipartitions by a MEDIAN PLANE on one axis and couples pairs within radius r
across it.  For small r only a slab of thickness ~2r around that plane contains any pair at all,
so an early (local) layer may be touching a few percent of the coordinates rather than coupling
near neighbours everywhere -- which is what the docstring describes ("a coordinate-parity rule").
Measure the coverage instead of reading the code.
"""
import json, sys
import numpy as np
sys.path.insert(0, '/root/cutfem_neural_a_20260910/source_14301bc56')
sys.path.insert(0, '/root/cutfem_neural_a_20260910/superelement_v0')
sys.path.insert(0, '/root/autodl-tmp/NEURAL_SCHUR')
import v1_scaled as V
from backend import multiscale_layers, checkerboard_layers

recs = json.load(open('/root/autodl-tmp/CUTFEM_SUPERELEMENT_LABELS/V1_LABELS.json'))
rec = [r for r in recs if int(r['seat']) == 328][0]
rec.setdefault('reference', '/root/autodl-tmp/CUTFEM_SUPERELEMENT_LABELS/REFERENCE_0328')
label = V.Label(rec, 0.2, 0.03, 10.0)
g = label.to_gpu(need_A=False, need_Z=False, z_dtype=np.float32 and __import__('torch').float32)
d = label.d
order = g['data'].quotient.order.cpu().numpy(); posn = order[6:]
pts = label.ijk[posn // 3].astype(float) / 64.0

print('d = %d\n' % d)
for rule, fn in (('median plane (multiscale_layers)', multiscale_layers),
                 ('checkerboard (checkerboard_layers)', checkerboard_layers)):
  layers, meta = fn(pts, levels=4, radius0=0.035, growth=2.0,
                    max_pairs_per_level=220_000, device='cpu', seed=0)
  print('=== %s ===' % rule, flush=True)
  print('%-6s %-8s %-9s %-14s %-14s %-12s' %
      ('level', 'radius', 'pairs', 'coords touched', 'as a row', 'as a source'), flush=True)
  touched_any = np.zeros(d, bool)
  for L, m in zip(layers, meta):
    pr = L.pairs.numpy()
    rows, cols = np.unique(pr[:, 0]), np.unique(pr[:, 1])
    both = np.union1d(rows, cols)
    touched_any[both] = True
    print('%-6d %-8.3f %-9d %-14s %-14s %-12s'
          % (m['level'], m['radius'], m['pairs'],
             '%d (%.1f%%)' % (len(both), 100*len(both)/d),
             '%d (%.1f%%)' % (len(rows), 100*len(rows)/d),
             '%d (%.1f%%)' % (len(cols), 100*len(cols)/d)), flush=True)
  print('  union: %d of %d coordinates (%.1f%%) appear in some pair; %d never do; %d coefficients total\n'
        % (touched_any.sum(), d, 100*touched_any.sum()/d, int((~touched_any).sum()),
           sum(m['pairs'] for m in meta)), flush=True)
