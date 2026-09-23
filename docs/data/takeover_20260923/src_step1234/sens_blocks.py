"""Teacher side of the design-sensitivity check: exact compiled blocks A, C, D at tau * (1 + eps).

For one perturbed geometry-stage run: trace coordinates P from the frozen compile_trace, ghost faces from the frozen
selection with the fixed templates, exact body K from the recomputed geometry stage (same steps as cover_blocks.py).
Admission (same meaning as the 2026-09-20 tau gate): the complete trace P and its boundary numbering, the ghost matrix
and the boundary rigid subspace must equal the base case's; otherwise the operators are not on a common basis and the
perturbation is recorded as not admitted. Usage: sens_blocks.py <base_case> <perturbed_case_id>
"""
import json, sys, time
from pathlib import Path
import numpy as np
from scipy import sparse
sys.path.insert(0, '/root/autodl-tmp/CLAUDE_TAKEOVER_20260923/xcase_src_11')
from gp_check_body import load_body
from stage_cutfem_multiconstraint.complete_trace import compile_trace
from stage_cutfem_gp import assembly

T = Path('/root/autodl-tmp/CLAUDE_TAKEOVER_20260923')
base, pid = sys.argv[1], sys.argv[2]
bdir = T / 'COVER_INPUTS' / base
out = T / 'SENS_01' / 'blocks' / pid / 'compiled'
out.mkdir(parents=True, exist_ok=True)
pk = Path('/root/autodl-tmp/CUTFEM_FRESH_GP_20260921/packets') / base
rec = dict(base=base, case=pid)
t0 = time.perf_counter()
body = load_body(T / 'SENS_01' / 'runs' / (pid + '_G'))
n = body['contract'].n; nodes = np.asarray(body['nodes']); nb = 3 * len(nodes)
rec['active_cells'] = int(len(body['active'])); rec['nodes'] = int(len(nodes))
t = time.perf_counter(); tr = compile_trace(body, coordinate_convention='max_pivot_geometric_residuals_v2'); rec['trace_seconds'] = time.perf_counter() - t
P = sparse.csr_matrix(tr['P']); m = len(tr['boundary'])
P0 = sparse.load_npz(bdir / 'P.npz').tocsr()
same_P = P.shape == P0.shape and P.nnz == P0.nnz and abs(P - P0).max() == 0 if P.shape == P0.shape else False
rec.update(m=m, same_P=bool(same_P), base_m=json.loads((bdir / 'RESULT.json').read_text())['m'])
faces, _ = assembly.select_faces(body)
order = np.argsort(nodes); sn = nodes[order]
F = np.asarray(faces)
cells = np.stack([assembly.cell_index(body, int(o)) for o in F[:, 0]])
offs = np.stack([assembly.stencil(a)[0] for a in range(3)])
ids = np.ravel_multi_index((2 * cells[:, None, :] + offs[F[:, 2]]).transpose(2, 0, 1), (2 * n + 1,) * 3)
local = order[np.searchsorted(sn, ids)]
dofs = (3 * local[:, :, None] + np.arange(3)).reshape(len(F), -1)
canonical = np.stack([assembly.face_factor(a, 1 / n) for a in range(3)])
tk = canonical.transpose(0, 2, 1) @ canonical
G = sparse.csr_matrix((tk[F[:, 2]].ravel(), (np.repeat(dofs, 135, axis=1).ravel(), np.tile(dofs, (1, 135)).ravel())), shape=(nb, nb))
G0 = sparse.load_npz(bdir / 'GHOST.npz').tocsr()
rec['same_ghost'] = bool(G.shape == G0.shape and (abs(G - G0).max() == 0 if G.shape == G0.shape else False))
rec['faces'] = int(len(F))
gamma = float(json.loads((pk / 'SAMPLE.json').read_text())['gp']['gamma'])
upper = body['K_upper']
K = (upper + upper.T - sparse.diags(upper.diagonal())).tocsr() + gamma * G
Kc = (P.T @ K @ P).tocsr()
Kc = sparse.triu(Kc, format='csr'); Kc = (Kc + Kc.T - sparse.diags(Kc.diagonal())).tocsr()
points = np.stack(np.unravel_index(nodes, (2 * n + 1,) * 3), axis=1) / (2 * n)
r = np.zeros((nb, 6))
for dd in range(3):
    r[dd::3, dd] = 1
    r[:, dd + 3] = np.cross(np.broadcast_to(np.eye(3)[dd], points.shape), points).reshape(-1)
Rc = sparse.csr_matrix(tr['inverse']) @ r
Q0 = np.linalg.qr(np.load(bdir / 'Q_RIGID.npy'))[0]; Q1 = np.linalg.qr(Rc[:m])[0] if m == rec['base_m'] else None
rec['rigid_subspace_residual'] = None if Q1 is None else float(np.linalg.norm(Q0 - Q1 @ (Q1.T @ Q0)))
rec['admitted'] = bool(rec['same_P'] and rec['same_ghost'] and rec['rigid_subspace_residual'] is not None
                       and rec['rigid_subspace_residual'] <= 1e-12)
if rec['admitted']:
    for name, M in (('A', Kc[m:, m:]), ('C', Kc[m:, :m]), ('D', Kc[:m, :m])):
        sparse.save_npz(out / (name + '.npz'), M.tocsr())
    for f in ('Q_RIGID.npy', 'INTERIOR_RIGID.npy', 'INTERIOR_POINTS.npy', 'QUALIFICATION_PROBES.npz'):
        (out / f).write_bytes((bdir / f).read_bytes())
rec['seconds'] = time.perf_counter() - t0
(out.parent / 'SENS_BLOCKS.json').write_text(json.dumps(rec, indent=2))
print(json.dumps(rec))
