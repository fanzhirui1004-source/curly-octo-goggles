"""Route 2, experiment D1: a two-sided compliance bracket for one CutFEM+GP cell with no teacher inside the bracket.

Discrete setting. K = sum_e G_e^T G_e (bulk cells; the body's 81-row element factors) + gamma sum_f F_f^T F_f
(ghost faces; fixed templates). For a self-equilibrated nodal load F the compliance c = F^T K^+ F satisfies
  lower:  c >= 2 F^T u - u^T K u                              for any nodal field u        (potential energy)
  upper:  c <= sum_j |M_j y_j|^2  (M_j = G_e or sqrt(gamma) F_f) for any element fields y_j with
          sum_j M_j^T M_j y_j = F                                                           (complementary energy)
The element forces M_j^T M_j y_j lie in range(K_j) by construction, so no pseudo-inverse is involved.

Admissible element fields from an approximate u (discrete flux-free equilibration):
 1. partition of unity phi_b = trilinear hats of a coarse grid aligned with the cells (spacing H cells);
 2. Galerkin correction of u in span{phi_b * rho_k on each connected component of patch b} (rho_k: 6 rigid fields),
    so the residual r = F - K u is orthogonal to every phi_b * rho_k (defect reported);
 3. per patch b (cells inside the support box of phi_b; ghost faces whose two cells are both inside), the local
    Neumann problem K_b w_b = phi_b * r, solvable because of step 2;
 4. y_j = u + sum_{b containing j} w_b (a broken field), which balances F exactly.
Reported per load and perturbation size: exact c, relative c_low and c_up, energy error of u before and after the
coarse correction, efficiency (c_up - c_low)/(c - c_low), orthogonality defect, timings.
Usage: dual_d1.py <case> <H> <out.json> [halo 0|1]
"""
import json, sys, time
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse import csgraph
sys.path.insert(0, '/root/autodl-tmp/CLAUDE_TAKEOVER_20260923/xcase_src_11')
from gp_check_body import load_body
from stage_cutfem_gp import assembly
from stage_cutfem_solver.pardiso import PardisoSPD

T = Path('/root/autodl-tmp/CLAUDE_TAKEOVER_20260923')
case, H, outp = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
HALO = int(sys.argv[4]) if len(sys.argv) > 4 else 0   # 1: add every ghost face touching a patch cell and the cell across it (zero weight)
EPS = (0.0, 0.01, 0.03, 0.1, 0.3)
rec = dict(case=case, H=H, halo=HALO, eps=EPS)
t0 = time.perf_counter()
body = load_body(T / 'COVER_G' / 'runs' / (case + '_G'))
n = body['contract'].n; nodes = np.asarray(body['nodes']); N = len(nodes); nb = 3 * N
points = np.asarray(body['points'])
Gm = body['G'].tocsr()
ncell = Gm.shape[0] // 81
cidx = np.asarray(body['cell_indices'])
cells = cidx if cidx.ndim == 2 else np.stack(np.unravel_index(cidx, (n,) * 3), axis=1)   # grid triple per active cell
rec['cell_indices_shape'] = list(cidx.shape)
# ghost faces: frozen selection and fixed templates (as in sens_blocks.py)
faces, _ = assembly.select_faces(body)
Fa = np.asarray(faces)
order = np.argsort(nodes); sn = nodes[order]
own = np.stack([assembly.cell_index(body, int(o)) for o in Fa[:, 0]])
nbr = np.stack([assembly.cell_index(body, int(o)) for o in Fa[:, 1]])
offs = np.stack([assembly.stencil(a)[0] for a in range(3)])
ids = np.ravel_multi_index((2 * own[:, None, :] + offs[Fa[:, 2]]).transpose(2, 0, 1), (2 * n + 1,) * 3)
fnode = order[np.searchsorted(sn, ids)]
fdofs = (3 * fnode[:, :, None] + np.arange(3)).reshape(len(Fa), -1)
canonical = np.stack([assembly.face_factor(a, 1 / n) for a in range(3)])
gamma = float(json.loads((Path('/root/autodl-tmp/CUTFEM_FRESH_GP_20260921/packets') / case / 'SAMPLE.json').read_text())['gp']['gamma'])
fac = canonical[Fa[:, 2]] * np.sqrt(gamma)
rf = fac.shape[1]
rows = (np.arange(len(Fa))[:, None, None] * rf + np.arange(rf)[None, :, None]).repeat(135, axis=2)
Fm = sparse.csr_matrix((fac.ravel(), (rows.ravel(), np.repeat(fdofs[:, None, :], rf, axis=1).ravel())), shape=(len(Fa) * rf, nb))
M = sparse.vstack([Gm, Fm]).tocsr()
K = (M.T @ M).tocsr()
upper = body['K_upper'].tocsr()
Kbody = (upper + upper.T - sparse.diags(upper.diagonal())).tocsr()
rec['body_factor_check'] = float(abs((Gm.T @ Gm).tocsr() - Kbody).max() / abs(Kbody).max())   # K_body = G^T G ?
# cell id of each face's two cells (row of the active-cell list)
flat = np.ravel_multi_index(cells.T, (n,) * 3); so = np.argsort(flat)
def row_of(tr):
    f = np.ravel_multi_index(tr.T, (n,) * 3); p = so[np.searchsorted(flat[so], f)]
    assert np.array_equal(flat[p], f); return p
fcell = np.stack([row_of(own), row_of(nbr)], axis=1)
rec.update(cells=int(ncell), faces=int(len(Fa)), nodes=int(N), gamma=gamma, setup_seconds=time.perf_counter() - t0)


def rigid(p):
    R = np.zeros((3 * len(p), 6))
    for d in range(3):
        R[d::3, d] = 1
    c = p - p.mean(0)
    R[0::3, 3], R[1::3, 3] = -c[:, 1], c[:, 0]
    R[1::3, 4], R[2::3, 4] = -c[:, 2], c[:, 1]
    R[0::3, 5], R[2::3, 5] = c[:, 2], -c[:, 0]
    return R


Rg = np.linalg.qr(rigid(points))[0]
# exact solves: a statically determinate support (3 + 2 + 1 dofs on three non-collinear nodes) removes exactly the
# rigid modes; for a self-equilibrated load its reactions vanish, so the pinned solution solves K u = F exactly
A_ = int(np.argmin(points[:, 0])); B_ = int(np.argmax(points[:, 0]))
ab = points[B_] - points[A_]
dist = np.linalg.norm(np.cross(points - points[A_], ab), axis=1); C_ = int(np.argmax(dist))
nrm = np.cross(ab, points[C_] - points[A_]); ax = int(np.argmax(np.abs(nrm)))
bperp = [d for d in range(3) if d != int(np.argmax(np.abs(ab)))]
pins = [3 * A_ + d for d in range(3)] + [3 * B_ + d for d in bperp] + [3 * C_ + ax]
free = np.setdiff1d(np.arange(nb), pins)
t = time.perf_counter()
Kf = K[free][:, free]
fact = PardisoSPD(sparse.triu(Kf, format='csr'), threads=8)
rec['exact_factor_seconds'] = time.perf_counter() - t


def solve_exact(Fn):
    x = np.zeros((nb,) + Fn.shape[1:])
    x[free] = fact.solve(Fn[free])
    return x


# loads: (a) box-node reactions of the 6 homogeneous strains, (b) 4 random box-node loads; all rigid-free
box = np.where(np.any((points <= 1e-12) | (points >= 1 - 1e-12), axis=1))[0]
bd = (3 * box[:, None] + np.arange(3)).ravel()
rng = np.random.default_rng(20260923)
names, Fs = [], []
for k, (i, j) in enumerate([(0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2)]):
    E = np.zeros((3, 3)); E[i, j] = E[j, i] = 1
    ua = ((points - 0.5) @ E).reshape(-1)
    Fn = np.zeros(nb); Fn[bd] = (K @ ua)[bd]
    names.append(f'strain_{k}'); Fs.append(Fn - Rg @ (Rg.T @ Fn))
for k in range(4):
    Fn = np.zeros(nb); Fn[bd] = rng.standard_normal(len(bd))
    names.append(f'random_{k}'); Fs.append(Fn - Rg @ (Rg.T @ Fn))
Fs = np.stack(Fs, axis=1)                                   # nb x nloads
t = time.perf_counter()
Ustar = solve_exact(Fs)
rec['exact_solve_seconds'] = time.perf_counter() - t
comp = np.einsum('ij,ij->j', Fs, Ustar)
rec['equilibrium_residual'] = float(np.linalg.norm(K @ Ustar - Fs) / np.linalg.norm(Fs))
# perturbed primal fields: exact + eps * (smooth response to a random body load, scaled to relative energy eps^2*c)
cols, Fcol, meta = [], [], []
for li in range(len(names)):
    for eps in EPS:
        if eps == 0:
            u = Ustar[:, li].copy()
        else:
            g = rng.standard_normal(nb); g -= Rg @ (Rg.T @ g)
            d = solve_exact(g[:, None])[:, 0]
            u = Ustar[:, li] + eps * np.sqrt(comp[li] / float(d @ (K @ d))) * d
        cols.append(u); Fcol.append(Fs[:, li]); meta.append((names[li], eps))
U0 = np.stack(cols, axis=1); FF = np.stack(Fcol, axis=1)
Us = np.stack([Ustar[:, names.index(m[0])] for m in meta], axis=1)
cs = np.array([comp[names.index(m[0])] for m in meta])

# partition of unity on the coarse grid, patches and connected components
t = time.perf_counter()
nv = n // H + 1
xs = points * n / H
base = np.floor(xs + 1e-12).astype(int)
pu_rows, pu_cols, pu_vals = [], [], []
for corner in range(8):
    v = base + np.array([(corner >> 0) & 1, (corner >> 1) & 1, (corner >> 2) & 1])
    w = np.prod(np.clip(1 - np.abs(xs - v), 0, None), axis=1)
    ok = (w > 0) & np.all(v <= nv - 1, axis=1)
    pu_rows.append(np.ravel_multi_index(v[ok].T, (nv,) * 3)); pu_cols.append(np.where(ok)[0]); pu_vals.append(w[ok])
PU = sparse.csr_matrix((np.concatenate(pu_vals), (np.concatenate(pu_rows), np.concatenate(pu_cols))), shape=(nv ** 3, N))
rec['pu_max_error'] = float(np.abs(np.asarray(PU.sum(0)).ravel() - 1).max())
vcell = cells // H
patch_cells = {}
for c in range(ncell):
    for corner in range(8):
        v = vcell[c] + np.array([(corner >> 0) & 1, (corner >> 1) & 1, (corner >> 2) & 1])
        if np.all(v < nv):
            patch_cells.setdefault(int(np.ravel_multi_index(v, (nv,) * 3)), []).append(c)
cell_patch = [[] for _ in range(ncell)]
plist = sorted(patch_cells)
for pi, b in enumerate(plist):
    for c in patch_cells[b]:
        cell_patch[c].append(pi)
face_of_patch = [[] for _ in plist]
halo_cells = [set() for _ in plist]
for f in range(len(Fa)):
    a_, b_ = fcell[f, 0], fcell[f, 1]
    pats = (set(cell_patch[a_]) | set(cell_patch[b_])) if HALO else (set(cell_patch[a_]) & set(cell_patch[b_]))
    for pi in pats:
        face_of_patch[pi].append(f)
        if HALO:
            halo_cells[pi].update((a_, b_))
nG = Gm.shape[0]
patches = []
Vcols = []
Vcols_count = []
full_rank = []                                             # components whose 6 weighted rigid fields are independent
for pi, b in enumerate(plist):
    cl = np.array(sorted(set(patch_cells[b]) | halo_cells[pi]))
    parts = [(cl[:, None] * 81 + np.arange(81)).ravel()]
    if face_of_patch[pi]:
        parts.append((np.array(face_of_patch[pi])[:, None] * rf + np.arange(rf)).ravel() + nG)
    rsel = np.concatenate(parts)
    Mb = M[rsel]
    dofs = np.unique(Mb.indices)
    Mb = Mb[:, dofs]
    dn = dofs // 3
    un = np.unique(dn)
    # components through shared nodes: rows (cells/faces) as hyperedges over node columns
    inc = sparse.csr_matrix((np.ones(Mb.nnz), (np.repeat(np.arange(Mb.shape[0]), np.diff(Mb.indptr)), np.searchsorted(un, dn[Mb.indices]))),
                            shape=(Mb.shape[0], len(un)))
    ncomp, lab = csgraph.connected_components((inc.T @ inc) > 0, directed=False)
    w = np.asarray(PU[b, un].todense()).ravel()
    wd = w[np.searchsorted(un, dn)]
    comps = []
    for k in range(ncomp):
        nodes_k = un[lab == k]
        sel = np.where(np.isin(dn, nodes_k))[0]
        Rn = rigid(points[nodes_k]); pos = np.searchsorted(nodes_k, dn[sel])
        Rdof = Rn[3 * pos + (dofs[sel] % 3)]
        # rigid space of the component (rank-revealing: a component with one or two nodes has fewer modes)
        Uq, sq, _ = np.linalg.svd(Rdof, full_matrices=False)
        Q = Uq[:, sq > 1e-10 * sq[0]]
        comps.append((sel, Q))
        # weighted rigid fields phi_b * rho_k on the component, orthonormalized with the same rank rule
        Wf = wd[sel, None] * Rdof
        if np.abs(Wf).max() > 0:
            Uw, sw, _ = np.linalg.svd(Wf, full_matrices=False)
            keep = sw > 1e-10 * sw[0]
            vcol = np.zeros((nb, int(keep.sum()))); vcol[dofs[sel]] = Uw[:, keep]
            if keep.sum() == 6:
                full_rank.append((float(sw[5] / sw[0]), len(Vcols_count), len(Vcols_count) + 6))
            Vcols.append(sparse.csr_matrix(vcol)); Vcols_count.extend(range(int(keep.sum())))
    patches.append((rsel, dofs, wd, comps))
V = sparse.hstack(Vcols).tocsr()
rec.update(patches=len(patches), coarse_dimension=int(V.shape[1]), patch_setup_seconds=time.perf_counter() - t)
t = time.perf_counter()
KH = (V.T @ (K @ V)).tocsr()
# KH = V^T K V is positive semidefinite: the partition of unity sums the weighted rigid fields back to the 6 global
# rigid modes, and weighted fields of different patches can be linearly dependent on small components. The right-hand
# sides are orthogonal to its kernel (loads and K u are rigid-free), so the singular system is consistent; solve it
# with a relative diagonal shift of 1e-12 and report the remaining residual of every coarse equation (which bounds the
# orthogonality defect handed to the local problems).
print(json.dumps(dict(stage='coarse', patches=len(patches), coarse_dimension=int(V.shape[1]))), flush=True)
dKH = KH.diagonal()
shift = 1e-12
cfact = PardisoSPD(sparse.triu(KH + sparse.diags(shift * dKH), format='csr'), threads=8)
R0 = FF - K @ U0
rhs = V.T @ R0
X = cfact.solve(rhs)
for _ in range(2):                                         # iterative refinement on the unshifted system
    X = X + cfact.solve(rhs - KH @ X)
rec['coarse_shift'] = shift
rec['coarse_residual_all_rows'] = float(np.linalg.norm(KH @ X - rhs) / max(np.linalg.norm(rhs), 1e-300))
U = U0 + V @ X
Rr = FF - K @ U
rec['coarse_seconds'] = time.perf_counter() - t
# local Neumann problems, all right-hand sides at once
t = time.perf_counter()
Y = M @ U                                                   # (elements+faces rows) x columns
defect = 0.0
rscale = np.maximum(np.linalg.norm(Rr, axis=0), 1e-300)    # orthogonality defect relative to each column's residual
local_res = 0.0
for rsel, dofs, wd, comps in patches:
    Mb = M[rsel][:, dofs]
    Kb = (Mb.T @ Mb).toarray()
    rb = wd[:, None] * Rr[dofs]
    Wb = np.zeros_like(rb)
    for sel, Q in comps:
        rs = rb[sel]
        defect = max(defect, float(np.max(np.linalg.norm(Q.T @ rs, axis=0) / rscale)))
        Kc = Kb[np.ix_(sel, sel)]
        s = np.trace(Kc) / len(sel)
        y = rs - Q @ (Q.T @ rs)
        x = np.linalg.solve(Kc + s * (Q @ Q.T), y)
        local_res = max(local_res, float(np.max(np.linalg.norm(Kc @ x - y, axis=0) / rscale)))
        Wb[sel] = x
    Y[rsel] += Mb @ Wb
rec['local_seconds'] = time.perf_counter() - t
rec['orthogonality_defect_max'] = defect
rec['local_equation_residual_max'] = local_res
c_up = np.einsum('ij,ij->j', Y, Y)
# direct validity check of the dual: the element forces M_j^T M_j y_j must sum to the load
eq = np.linalg.norm(M.T @ Y - FF, axis=0) / np.linalg.norm(FF, axis=0)
rec['dual_equilibrium_error_max'] = float(eq.max())
c_low = 2 * np.einsum('ij,ij->j', FF, U) - np.einsum('ij,ij->j', U, K @ U)
e0 = np.einsum('ij,ij->j', U0 - Us, K @ (U0 - Us)) / cs
e1 = np.einsum('ij,ij->j', U - Us, K @ (U - Us)) / cs
out = {}
for k, (name, eps) in enumerate(meta):
    out.setdefault(name, {})[str(eps)] = dict(
        rel_energy_error_before=float(e0[k]), rel_energy_error_after_coarse=float(e1[k]),
        c_low_rel=float((c_low[k] - cs[k]) / cs[k]), c_up_rel=float((c_up[k] - cs[k]) / cs[k]),
        efficiency=float((c_up[k] - c_low[k]) / (cs[k] - c_low[k])) if cs[k] - c_low[k] > 1e-14 * cs[k] else None,
        upper_valid=bool(c_up[k] >= cs[k] * (1 - 1e-10)), lower_valid=bool(c_low[k] <= cs[k] * (1 + 1e-10)))
rec['compliance'] = dict(zip(names, comp.tolist()))
rec['results'] = out
rec['seconds'] = time.perf_counter() - t0
outp.write_text(json.dumps(rec, indent=2))
for name in names:
    print(name, ' '.join('eps=%s low %+.2e up %+.2e eff %s' % (e, v['c_low_rel'], v['c_up_rel'],
          'n/a' if v['efficiency'] is None else '%.2f' % v['efficiency']) for e, v in out[name].items()), flush=True)
print(json.dumps({k: rec[k] for k in ('body_factor_check', 'pu_max_error', 'patches', 'coarse_dimension', 'orthogonality_defect_max', 'local_equation_residual_max', 'dual_equilibrium_error_max', 'coarse_residual_all_rows',
                                      'equilibrium_residual', 'exact_factor_seconds', 'coarse_seconds', 'local_seconds', 'seconds')}))
