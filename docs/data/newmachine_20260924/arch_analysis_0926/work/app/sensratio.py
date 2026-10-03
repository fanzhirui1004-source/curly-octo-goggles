# (a) exact small-cell encode/query cost on CPU; (b) is ||dC/dtau|| proportional to cell energy for loaded vs passenger states?
import sys, time, json, numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spl
sys.path.insert(0, '/tmp/claude-0/-home-user-curly-octo-goggles/76f53c89-e2a9-52bb-8817-9460fd780b76/scratchpad/pack/work/mechanics')
import cell
case = 'fresh_train_0010_cover01_r1'
t0 = time.perf_counter(); C = cell.RealCell(case); t_asm = time.perf_counter() - t0
K = C.K.tocsc(); P, I = C.P, C.I
print('elements', len(C.cells), 'dofs', C.nb, 'ports', len(P), 'interior', len(I), 'nnz', K.nnz, 'assemble_s %.2f' % t_asm)
KII = K[I][:, I].tocsc(); KIP = K[I][:, P].tocsc(); KPP = K[P][:, P].tocsc()
t0 = time.perf_counter(); lu = spl.splu(KII, permc_spec='MMD_AT_PLUS_A'); t_fac = time.perf_counter() - t0
print('splu factor K_II %.3f s, nnz(L+U) %d' % (t_fac, lu.L.nnz + lu.U.nnz))
rng = np.random.default_rng(0)
for B in (1, 16, 64):
    q = rng.standard_normal((len(P), B))
    t0 = time.perf_counter()
    for _ in range(3):
        y = KPP @ q - KIP.T @ lu.solve(KIP @ q)
    print('exact S q B=%d: %.4f s' % (B, (time.perf_counter() - t0) / 3))
# dense S cost (ports x ports) -- interior small
t0 = time.perf_counter(); X = lu.solve(KIP.toarray()); S = KPP.toarray() - KIP.T @ X; t_S = time.perf_counter() - t0
print('dense S build %.2f s, %.1f MB' % (t_S, S.nbytes / 2**20))
t0 = time.perf_counter()
for _ in range(20): y = S @ q[:, :1]
print('dense S q B=1 %.5f s' % ((time.perf_counter() - t0) / 20))
# ---- sensitivities
dM = np.load('/tmp/claude-0/-home-user-curly-octo-goggles/76f53c89-e2a9-52bb-8817-9460fd780b76/scratchpad/pack/work/mechanics/dM_%s.npy' % case)  # 8 x E x 125
Tm = C.Tm; dofs = C.dofs; taus = np.asarray(C.z['taus'], float)
def sens(u):
    ue = u[dofs]                               # E x 81 x B
    W = np.einsum('eib,mij,ejb->emb', ue, Tm, ue)   # E x 125 x B
    return -np.einsum('cem,emb->cb', dM, W)    # 8 x B
xyz = np.repeat(C.xyz, 3, 0).reshape(-1, 3, 3)[:, 0, :]  # per dof coords
comp = np.tile(np.arange(3), C.nb // 3)
xyzd = C.xyz[np.arange(C.nb) // 3]
def field_from_q(q):
    u = np.zeros((C.nb, q.shape[1])); u[P] = q; u[I] = -lu.solve(KIP @ q); return u
res = {}
# (1) loaded: 6 macro strain modes on all ports
E6 = []
for (a, b) in [(0,0),(1,1),(2,2),(0,1),(0,2),(1,2)]:
    e = np.zeros((3,3)); e[a,b] = e[b,a] = 1.0
    E6.append((xyzd @ e)[np.arange(C.nb), comp])
Um = np.stack(E6, 1); qm = Um[P]; um = field_from_q(qm)
# (2) passenger: prescribe the ports on face x=0 only (bending / shear / stretch of that face), all other ports free
face = np.flatnonzero(np.isclose(xyzd[:, 0], 0.0)); free = np.setdiff1d(np.arange(C.nb), face)
Kff = K[free][:, free].tocsc(); Kfb = K[free][:, face]
lu2 = spl.splu(Kff + 1e-12 * sp.identity(len(free), format='csc'))
y, z_ = xyzd[face, 1], xyzd[face, 2]; cf = comp[face]
G = []
for mode in range(6):
    g = np.zeros(len(face))
    if mode == 0: g[cf == 0] = (y[cf == 0] - .5) ** 2            # face bending out of plane (x-disp ~ y^2)
    if mode == 1: g[cf == 0] = (z_[cf == 0] - .5) ** 2
    if mode == 2: g[cf == 1] = (z_[cf == 1] - .5)                # in-plane shear strain of the face
    if mode == 3: g[cf == 2] = (y[cf == 2] - .5) ** 2            # in-plane curvature
    if mode == 4: g[cf == 0] = (y[cf == 0] - .5) * (z_[cf == 0] - .5)  # twist
    if mode == 5: g[cf == 1] = (y[cf == 1] - .5)                  # in-plane stretch
    G.append(g)
G = np.stack(G, 1)
up = np.zeros((C.nb, 6)); up[face] = G; up[free] = -lu2.solve(Kfb @ G)
for name, u in (('loaded_macro', um), ('passenger_face', up)):
    W = 0.5 * np.einsum('ib,ib->b', u, K @ u)
    s = sens(u)
    Sdot = (taus[:, None] * s).sum(0)          # Euler: d/dalpha C(alpha tau) direction
    kap = np.linalg.norm(s, axis=0) * taus.mean() / (2 * W)
    res[name] = dict(W=W.tolist(), s_norm=np.linalg.norm(s, axis=0).tolist(), kappa=kap.tolist(),
                     euler=(-Sdot / (2 * W)).tolist(), all_neg=bool((s <= 0).all()))
    print(name, 'kappa=||s|| tau/(2W):', np.round(kap, 3), ' -tau.s/(2W):', np.round(-Sdot / (2 * W), 3), ' all s<=0:', (s <= 0).all(), (s<=0).mean())
json.dump(dict(case=case, elements=int(len(C.cells)), dofs=int(C.nb), ports=int(len(P)), interior=int(len(I)),
               assemble_s=t_asm, factor_s=t_fac, dense_S_s=t_S, res=res), open('sensratio.json', 'w'), indent=1)
