"""Teacher-free encoder inputs from a recomputed geometry stage, as used in deployment: trace coordinates P (frozen
compile_trace), ghost penalty (frozen face selection, fixed templates, vectorized scatter), boundary / interior rigid
fields and interior points. No exact condensation and no probe check (those are evaluation, not preparation).
Usage: e2e_inputs.py <G run dir> <output dir>"""
import json, sys, time
from pathlib import Path
import numpy as np
from scipy import sparse
sys.path.insert(0, '/root/autodl-tmp/CLAUDE_TAKEOVER_20260923/xcase_src_11')
from gp_check_body import load_body
from stage_cutfem_multiconstraint.complete_trace import compile_trace
from stage_cutfem_gp import assembly
run, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
rec = {}; t0 = time.perf_counter()
t = time.perf_counter(); body = load_body(run); rec['load_seconds'] = time.perf_counter() - t
n = body['contract'].n; nodes = np.asarray(body['nodes']); nb = 3 * len(nodes)
t = time.perf_counter(); tr = compile_trace(body, coordinate_convention='max_pivot_geometric_residuals_v2'); rec['trace_seconds'] = time.perf_counter() - t
P = sparse.csr_matrix(tr['P']); Pinv = sparse.csr_matrix(tr['inverse']); m = len(tr['boundary'])
t = time.perf_counter()
faces, _ = assembly.select_faces(body); rec['face_select_seconds'] = time.perf_counter() - t
t = time.perf_counter()
F = np.asarray(faces); order = np.argsort(nodes); sn = nodes[order]
cells = np.stack([assembly.cell_index(body, int(o)) for o in F[:, 0]])
offs = np.stack([assembly.stencil(a)[0] for a in range(3)])
ids = np.ravel_multi_index((2 * cells[:, None, :] + offs[F[:, 2]]).transpose(2, 0, 1), (2 * n + 1,) * 3)
local = order[np.searchsorted(sn, ids)]
dofs = (3 * local[:, :, None] + np.arange(3)).reshape(len(F), -1)
canonical = np.stack([assembly.face_factor(a, 1 / n) for a in range(3)])
tk = canonical.transpose(0, 2, 1) @ canonical
G = sparse.csr_matrix((tk[F[:, 2]].ravel(), (np.repeat(dofs, 135, axis=1).ravel(), np.tile(dofs, (1, 135)).ravel())), shape=(nb, nb))
rec['ghost_scatter_seconds'] = time.perf_counter() - t
points = np.stack(np.unravel_index(nodes, (2 * n + 1,) * 3), axis=1) / (2 * n)
r = np.zeros((nb, 6))
for d in range(3):
    r[d::3, d] = 1
    r[:, d + 3] = np.cross(np.broadcast_to(np.eye(3)[d], points.shape), points).reshape(-1)
Rc = Pinv @ r
interior_nodes = Pinv[m:, :].indices[::3] // 3
t = time.perf_counter()
sparse.save_npz(out / 'P.npz', P); sparse.save_npz(out / 'GHOST.npz', G)
np.save(out / 'Q_RIGID.npy', Rc[:m]); np.save(out / 'INTERIOR_RIGID.npy', Rc[m:]); np.save(out / 'INTERIOR_POINTS.npy', points[interior_nodes])
rec['write_seconds'] = time.perf_counter() - t
rec.update(m=m, nb=nb, faces=int(len(F)), seconds=time.perf_counter() - t0)
(out / 'RESULT.json').write_text(json.dumps(rec, indent=2))
print(json.dumps(rec))
