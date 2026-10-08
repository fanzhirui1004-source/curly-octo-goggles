"""Compare a locally recomputed G stage with the producer's published CONTEXT for the same case."""
import hashlib, json, sys
from pathlib import Path
import numpy as np
R = Path('/root/autodl-tmp/CUTFEM_FRESH_GP_20260921')
W = Path('/root/autodl-tmp/CLAUDE_TAKEOVER_20260923/COVER_G')
case = sys.argv[1]
ctx = R / 'packets' / case / 'CONTEXT'
man = json.loads((R / 'packets' / case / 'FRESH_MANIFEST.json').read_text())
for n in ('CELL_INDICES.npy', 'CELL_MOMENTS27.npy', 'ACTIVE_NODE_IDS.npy'):
    if hashlib.sha256((ctx / n).read_bytes()).hexdigest() != man['CONTEXT/' + n]:
        raise ValueError('PACKET_CONTEXT_SHA:' + n)
body = W / 'runs' / (case + '_G') / 'body'
mine = dict(cells=np.load(body / 'CELL_INDICES.npy'), moments=np.load(body / 'cell_moments.npy'), nodes=np.load(body / 'NODES.npy'))
ref = dict(cells=np.load(ctx / 'CELL_INDICES.npy'), moments=np.load(ctx / 'CELL_MOMENTS27.npy'), nodes=np.load(ctx / 'ACTIVE_NODE_IDS.npy'))
res = dict(case=case, cells_equal=bool(np.array_equal(mine['cells'], ref['cells'])),
           nodes_equal=bool(np.array_equal(mine['nodes'], ref['nodes'])), elements=int(len(mine['cells'])))
if res['cells_equal']:
    d = np.abs(mine['moments'] - ref['moments']).max(axis=1) / np.abs(ref['moments']).max(axis=1)
    res.update(moments_bitwise_equal=bool(np.array_equal(mine['moments'], ref['moments'])),
               moment_relative_max=float(d.max()), moment_relative_median=float(np.median(d)))
res['status'] = 'MATCH' if res['cells_equal'] and res['nodes_equal'] and res.get('moment_relative_max', 1) < 1e-12 else 'MISMATCH'
(W / 'runs' / (case + '_G_VERIFY.json')).write_text(json.dumps(res, indent=2))
print(json.dumps(res))
