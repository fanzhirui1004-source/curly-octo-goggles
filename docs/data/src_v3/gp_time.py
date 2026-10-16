import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, '/root/autodl-tmp/CLAUDE_TAKEOVER_20260923/xcase_src_11')
from gp_check_body import load_body
from stage_cutfem_gp import assembly
case = sys.argv[1]
t = time.perf_counter(); body = load_body(Path('/root/autodl-tmp/CLAUDE_TAKEOVER_20260923/COVER_G/runs') / (case + '_G')); tl = time.perf_counter() - t
t = time.perf_counter(); faces, cov = assembly.select_faces(body); ts = time.perf_counter() - t
n = body['contract'].n; nodes = np.asarray(body['nodes'])
t = time.perf_counter()
positions = {int(v): i for i, v in enumerate(nodes)}
dofs = []
for owner, _, axis in faces:
    ijk = assembly.cell_index(body, owner); offsets, _, _ = assembly.stencil(axis)
    ids = np.ravel_multi_index((2 * ijk + offsets).T, (2 * n + 1,) * 3)
    local = np.array([positions[int(i)] for i in ids])
    dofs.append((3 * local[:, None] + np.arange(3)).ravel())
tloop = time.perf_counter() - t
# vectorized equivalent
t = time.perf_counter()
F = np.asarray(faces); order = np.argsort(nodes); sn = nodes[order]
cells = np.stack([assembly.cell_index(body, int(o)) for o in F[:, 0]])
offs = np.stack([assembly.stencil(a)[0] for a in range(3)])  # (3, 45, 3)
ids = np.ravel_multi_index((2 * cells[:, None, :] + offs[F[:, 2]]).transpose(2, 0, 1), (2 * n + 1,) * 3)
local = order[np.searchsorted(sn, ids)]
dv = (3 * local[:, :, None] + np.arange(3)).reshape(len(F), -1)
tvec = time.perf_counter() - t
print(json.dumps(dict(case=case, faces=len(faces), active=cov['active_cells'], full_certified=cov['full_certified_cells'],
                      interval_full_in_body='interval_full' in body, load=tl, select=ts, loop=tloop, vectorized=tvec,
                      equal=bool(np.array_equal(dv, np.asarray(dofs))))))
