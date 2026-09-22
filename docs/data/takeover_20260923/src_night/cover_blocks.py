"""Second batch, no teacher artifacts: trace coordinates P (frozen compile_trace), ghost penalty from the frozen face
selection and fixed face templates, and the exact body K of the locally recomputed geometry stage. Condense exactly
(PARDISO) and compare with the packet's probes (teacher force S q). Save the pieces for the GPU encoder.

Interior coordinates, rigid fields and interior points follow prepare_assets_v3.compile_case.
"""
import json, sys, time
from pathlib import Path
import numpy as np
from scipy import sparse
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gp_check_body import load_body
from stage_cutfem_multiconstraint.complete_trace import compile_trace
from stage_cutfem_gp import assembly
from stage_cutfem_solver.pardiso import PardisoSPD

case = sys.argv[1]
W = Path('/root/autodl-tmp/CLAUDE_TAKEOVER_20260923/COVER_G')
pk = Path('/root/autodl-tmp/CUTFEM_FRESH_GP_20260921/packets') / case
out = Path('/root/autodl-tmp/CLAUDE_TAKEOVER_20260923/COVER_INPUTS') / case
out.mkdir(parents=True, exist_ok=True)
rec = dict(case=case)
t0 = time.perf_counter()
body = load_body(W / 'runs' / (case + '_G'))
n = body['contract'].n; nodes = np.asarray(body['nodes']); nb = 3 * len(nodes)
t = time.perf_counter(); tr = compile_trace(body, coordinate_convention='max_pivot_geometric_residuals_v2'); rec['trace_seconds'] = time.perf_counter() - t
P = sparse.csr_matrix(tr['P']); Pinv = sparse.csr_matrix(tr['inverse']); m = len(tr['boundary'])
rec['m'] = m; rec['nb'] = nb
# ghost penalty: frozen face selection, frozen stencil/templates, own scatter of the 135x135 template products
t = time.perf_counter()
faces, coverage = assembly.select_faces(body)
positions = {int(v): i for i, v in enumerate(nodes)}
dofs = []
for owner, _, axis in faces:
    ijk = assembly.cell_index(body, owner); offsets, _, _ = assembly.stencil(axis)
    ids = np.ravel_multi_index((2 * ijk + offsets).T, (2 * n + 1,) * 3)
    local = np.array([positions[int(i)] for i in ids])
    dofs.append((3 * local[:, None] + np.arange(3)).ravel())
dofs = np.asarray(dofs); axes = np.asarray([f[2] for f in faces])
canonical = np.stack([assembly.face_factor(a, 1 / n) for a in range(3)])
tk = canonical.transpose(0, 2, 1) @ canonical  # 3 x 135 x 135
rows = np.repeat(dofs, 135, axis=1).ravel(); cols = np.tile(dofs, (1, 135)).ravel(); vals = tk[axes].ravel()
G = sparse.csr_matrix((vals, (rows, cols)), shape=(nb, nb))
rec['ghost_seconds'] = time.perf_counter() - t; rec['faces'] = len(faces)
ref_faces = np.load(pk / 'CONTEXT' / 'GP_FACES.npy'); ref_sup = np.load(pk / 'CONTEXT' / 'GP_GLOBAL_SUPPORT.npy')
rec['faces_equal_published'] = bool(np.array_equal(np.asarray(faces), ref_faces))
rec['support_equal_published'] = bool(np.array_equal(np.unique(dofs), ref_sup))
rec['templates_equal_published'] = bool(np.allclose(canonical, np.load(pk / 'CONTEXT' / 'GP_FACTOR_TEMPLATES.npy'), rtol=0, atol=0))
gamma = float(json.loads((pk / 'SAMPLE.json').read_text())['gp']['gamma'])
upper = body['K_upper']
Kbody = (upper + upper.T - sparse.diags(upper.diagonal())).tocsr()
K = (Kbody + gamma * G).tocsr()
Kc = (P.T @ K @ P).tocsr()
Kc = sparse.triu(Kc, format='csr'); Kc = (Kc + Kc.T - sparse.diags(Kc.diagonal())).tocsr()
D = Kc[:m, :m].tocsr(); C = Kc[m:, :m].tocsr(); A = Kc[m:, m:].tocsr()
probes = np.load(pk / 'PROBES.npz'); q = probes['trace']; ref = probes['matrix_force']
t = time.perf_counter()
s = 1 / np.sqrt(A.diagonal()); up = sparse.triu(A, format='csr'); up.data *= np.repeat(s, np.diff(up.indptr)) * s[up.indices]
with PardisoSPD(up, threads=6) as f:
    rhs = -C @ q; z = s[:, None] * f.solve(s[:, None] * rhs)
    for _ in range(2): z += s[:, None] * f.solve(s[:, None] * (rhs - A @ z))
rec['solve_seconds'] = time.perf_counter() - t
action = D @ q + C.T @ z
rec['probe_force_relative'] = (np.linalg.norm(action - ref, axis=0) / np.linalg.norm(ref, axis=0)).tolist()
# interior bookkeeping as in prepare_assets_v3.compile_case
points = np.stack(np.unravel_index(nodes, (2 * n + 1,) * 3), axis=1) / (2 * n)
extract = Pinv[m:, :]
interior_nodes = extract.indices[::3] // 3
def rigid(p):
    r = np.zeros((3 * len(p), 6))
    for d in range(3): r[d::3, d] = 1
    for d in range(3): r[:, d + 3] = np.cross(np.broadcast_to(np.eye(3)[d], p.shape), p).reshape(-1)
    return r
Rc = Pinv @ rigid(points)
rec['rigid_relative'] = float(np.linalg.norm(Kc @ Rc) / (sparse.linalg.norm(Kc) * np.linalg.norm(Rc)))
sparse.save_npz(out / 'P.npz', P); sparse.save_npz(out / 'Pinv.npz', Pinv); sparse.save_npz(out / 'GHOST.npz', G.tocsr())
np.save(out / 'Q_RIGID.npy', Rc[:m]); np.save(out / 'INTERIOR_RIGID.npy', Rc[m:]); np.save(out / 'INTERIOR_POINTS.npy', points[interior_nodes])
np.savez(out / 'QUALIFICATION_PROBES.npz', q=q, z=z, force=action, reference_force=ref, energy=np.sum(q * action, axis=0))
for name, M in (('A', A), ('C', C), ('D', D)):
    sparse.save_npz(out / (name + '.npz'), M)
rec['seconds'] = time.perf_counter() - t0
(out / 'RESULT.json').write_text(json.dumps(rec, indent=2))
print(json.dumps(rec))
