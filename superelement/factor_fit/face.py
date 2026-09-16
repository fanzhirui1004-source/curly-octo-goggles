#!/usr/bin/env python3
"""Can two copies of one cell's Schur complement actually be glued?

BRIEFING section 3 records an assembly-consistency test (two cells, 2x2x2, 1x1x4) at 1e-12, but it
was run by a different agent on a different machine and neither its code nor its bodies are here;
COMPOSABILITY_R1 on this box is a no-GP 19-cell intra-body split, not a lattice.  Before writing
an assembly test for APPROXIMATE S, check the prerequisite on the data we do have: do the trace
dofs sit on the faces of the background grid, and does the +x face of a cell match the -x face of
its copy one period over?

Reads the label only.  No GPU, no solve.
"""
import json, sys
from collections import Counter
import numpy as np
sys.path.insert(0, '/root/cutfem_neural_a_20260910/source_14301bc56')
sys.path.insert(0, '/root/cutfem_neural_a_20260910/superelement_v0')
import v1_scaled as V

recs = json.load(open('/root/autodl-tmp/CUTFEM_SUPERELEMENT_LABELS/V1_LABELS.json'))
rec = [r for r in recs if int(r['seat']) == 328][0]
rec.setdefault('reference', '/root/autodl-tmp/CUTFEM_SUPERELEMENT_LABELS/REFERENCE_0328')
label = V.Label(rec, 0.2, 0.03, 10.0)
ijk = np.asarray(label.ijk)
print('label fields:', [a for a in dir(label) if not a.startswith('_')][:30])
print('d =', label.d, '  ijk shape', ijk.shape, ' dtype', ijk.dtype)
print('ijk range per axis: ', [(int(ijk[:, a].min()), int(ijk[:, a].max())) for a in range(ijk.shape[1])])
q = ijk.shape[0] * 3
print('nodes %d  -> trace dofs %d (3 per node)' % (ijk.shape[0], q))

for a, name in enumerate('xyz'):
    lo = int(ijk[:, a].min()); hi = int(ijk[:, a].max())
    n_lo = int((ijk[:, a] == lo).sum()); n_hi = int((ijk[:, a] == hi).sum())
    print('  axis %s: %d nodes at %d, %d nodes at %d' % (name, n_lo, lo, n_hi, hi))

# does the +x face match the -x face under the same (y, z)?
a = 0
lo, hi = int(ijk[:, a].min()), int(ijk[:, a].max())
face_lo = ijk[ijk[:, a] == lo]
face_hi = ijk[ijk[:, a] == hi]
key_lo = {(int(r[1]), int(r[2])) for r in face_lo}
key_hi = {(int(r[1]), int(r[2])) for r in face_hi}
print('\n-x face: %d nodes, %d distinct (y,z)' % (len(face_lo), len(key_lo)))
print('+x face: %d nodes, %d distinct (y,z)' % (len(face_hi), len(key_hi)))
print('(y,z) sets identical: %s   |intersection| = %d' % (key_lo == key_hi, len(key_lo & key_hi)))
if key_lo != key_hi:
    print('  only in -x: %d   only in +x: %d' % (len(key_lo - key_hi), len(key_hi - key_lo)))

# how many trace nodes are on a face at all, vs interior to the trace
on_face = np.zeros(len(ijk), bool)
for a in range(3):
    on_face |= (ijk[:, a] == ijk[:, a].min()) | (ijk[:, a] == ijk[:, a].max())
print('\nnodes on some outer face: %d of %d (%.1f%%)' % (on_face.sum(), len(ijk), 100*on_face.mean()))
print('nodes NOT on any outer face: %d  <- these cannot be shared with a neighbour'
      % int((~on_face).sum()))
c = Counter(int(((ijk[:, a] == ijk[:, a].min()) | (ijk[:, a] == ijk[:, a].max())).sum()) for a in range(3))
print('\nif the +x/-x faces match, a 1x1x2 assembly has %d trace dofs (shared face counted once)'
      % (2 * q - 3 * len(key_lo & key_hi)))
