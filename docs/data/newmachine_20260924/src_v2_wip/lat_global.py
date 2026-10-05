"""Direction A, step 2: lattices under a NON-AFFINE lattice-scale map (twisted beam, bent panel, Bezier), fine-scale exact vs
learned vs homogenised (new script; default paths unchanged; runs on the GPU server; P1 bodies are only read).

Cells: one cell kind (case) at the integer positions of an nx x ny x nz block; the cell at p is the MappedCell of the global
map (mapped_cell.make_global) restricted to p + [0, 1]^3 (make_map kind 'global'), so neighbouring cells are conforming.
Rotation classes: cells whose maps differ by a rigid motion x -> R x + t have S_p = T_R S T_R^T (T_R = blockdiag R over the port
nodes) for the exact operator (the mapped stiffness of a rigidly moved body) and for the learned operator (V0R + co-rotated
Jacobi is exactly equivariant under similarity maps; the weak patch and the correction are built from K~). twist: cells with
the same (i, j) differ by a rotation about the twist axis; bend: cells with the same k differ by a rotation about the bend
axis (and by a translation in y); otherwise every cell is its own class. R is fitted (Procrustes) on sample points and the
fit residual is checked; --check builds one non-representative cell and compares its operators with the rotated ones.
Operators: dense port matrices per class representative (exact S~, learned S_hat per field as in lat_mapped.py), applied
to the cells of the class through T_R. K_PP of a cell = the representative's K_PP rotated node block by node block.
Supports and loads: clamp face and load face (default z = min / z = max for twist, x = min / x = max for bend); lattice3
consistent weights of the reference face times the physical area factor det J |J^-T N| at each port node (1 when the face
is mapped rigidly, as for the end faces of twist and bend), unit total force in x, y, z; + n_random random loads.
--fast 1: the learned dense operators through mapped_fast.MappedFastOp (explicit adjoint, 64 columns per application).
Homogenised macro model (Q1 isoparametric on the mapped box, m elements per cell and axis, Richardson from the two finest m):
  homog_cell   per-cell tensor: periodic C^H of the AFFINE cell with A = J at the centre of the class representative,
               rotated by the class rotation (local homogenisation at cell resolution, re-solved per class)
  homog_push   reference C^H pushed forward by J(x) at every Gauss point
  homog_rot    reference C^H rotated by the polar factor of J(x) at every Gauss point
Usage: lat_global.py <out.json> <ckpt> <body_dir> <case> --map '<global map json>' --shape 2x2x4 [--clamp z,min]
       [--load z,max] [--fields c1,c2w] [--m 2,4] [--check 1] [--p1 homog_cells.json] [--work dir] [--fast 1]
       --layout <gen_cutglob.py layout json>: per-position cases (full and trimmed cells; <case> and <body_dir> as given
       are then ignored in favour of the layout's), rotation classes per (case, position class), no homogenisation
       lat_global.py --selftest     (CPU: macro solver on mapped meshes; Procrustes; K_PP block rotation)"""
import argparse, gc, json, sys, time
from pathlib import Path
import numpy as np

AX = {'x': 0, 'y': 1, 'z': 2}


# ------------------------------------------------------------------------------------------------ geometry helpers
def jac(phi, X, h=1e-5):
    """dx/dX (N, 3, 3) of the map phi at the points X (N, 3) by central differences."""
    X = np.asarray(X, float)
    J = np.zeros((len(X), 3, 3))
    for d in range(3):
        e = np.zeros(3); e[d] = h
        J[:, :, d] = (phi(X + e) - phi(X - e)) / (2 * h)
    return J


def polar_R(J):
    U, _, Vh = np.linalg.svd(J)
    return U @ Vh


def procrustes(a, b):
    """R, t, residual with b ~ a R^T + t (rows are points)."""
    ca, cb = a.mean(0), b.mean(0)
    H = (a - ca).T @ (b - cb)
    U, _, Vh = np.linalg.svd(H)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vh.T @ U.T))])
    R = Vh.T @ D @ U.T
    t = cb - ca @ R.T
    return R, t, float(np.abs(a @ R.T + t - b).max())


def area_factor(phi, X, axis):
    """Physical area element of the reference face x_axis = const per reference area at X: det J |J^-T e_axis|."""
    J = jac(phi, X)
    cof = np.linalg.det(J)[:, None, None] * np.transpose(np.linalg.inv(J), (0, 2, 1))
    return np.linalg.norm(cof[:, :, axis], axis=1)


def class_key(gspec, p):
    k = gspec['kind']
    if k == 'twist':
        ax = int(gspec.get('axis', 2))
        return tuple(int(p[d]) for d in range(3) if d != ax)
    if k == 'bend':
        return (int(p[2]),)
    if k == 'affine':
        return ()
    return tuple(int(v) for v in p)


def rotate_kpp(r, c, v, R, xp):
    """Upper triplets (r <= c) of a port stiffness on node-major xyz port DOFs -> those of T_R K T_R^T (xp: numpy or torch)."""
    import torch
    full_r = torch.cat([r, c[r != c]]); full_c = torch.cat([c, r[r != c]]); full_v = torch.cat([v, v[r != c]])
    na, nb_ = full_r // 3, full_c // 3
    nn = int(max(int(na.max()), int(nb_.max()))) + 1
    key, inv = torch.unique(na * nn + nb_, return_inverse=True)
    B = torch.zeros((len(key), 3, 3), dtype=v.dtype, device=v.device)
    B.index_put_((inv, full_r % 3, full_c % 3), full_v, accumulate=True)
    Rt = torch.as_tensor(R, dtype=v.dtype, device=v.device)
    B = Rt @ B @ Rt.T
    a_, b_ = key // nn, key % nn
    i_, j_ = torch.meshgrid(torch.arange(3, device=v.device), torch.arange(3, device=v.device), indexing='ij')
    rr = (3 * a_[:, None, None] + i_[None]).reshape(-1); cc = (3 * b_[:, None, None] + j_[None]).reshape(-1)
    vv = B.reshape(-1)
    up = (rr <= cc) & (vv != 0)
    return rr[up], cc[up], vv[up]


# ------------------------------------------------------------------------------------------------ macro model (CPU)
def macro_mapped(phi, shape, m, tensor_fn, clamp=('z', 'min'), load=('z', 'max')):
    """Q1 isoparametric macro model on the mapped lattice box [0, shape] (reference cell units), m elements per cell and
    axis; tensor_fn(Xg (q, 3) reference Gauss points, cell (q, 3) int) -> (q, 6, 6) Voigt tensors (engineering shear).
    Clamp face fixed; uniform traction of unit total force in x, y, z on the mapped load face (2 x 2 Gauss on each face
    quad with the physical area element). Returns M (3, 3) = F^T U (compliance matrix of the three loads), dofs."""
    import scipy.sparse as sp
    shape = np.asarray(shape, int); N = shape * m; M = N + 1
    gi = np.stack(np.unravel_index(np.arange(int(M.prod())), tuple(M)), 1)
    Xn = gi / m
    xn = phi(Xn)
    nid = lambda i, j, k: (i * M[1] + j) * M[2] + k
    I, J, K = [a.ravel() for a in np.meshgrid(np.arange(N[0]), np.arange(N[1]), np.arange(N[2]), indexing='ij')]
    loc = np.array([[(c >> 2) & 1, (c >> 1) & 1, c & 1] for c in range(8)])
    en = np.stack([nid(I + loc[a, 0], J + loc[a, 1], K + loc[a, 2]) for a in range(8)], 1)      # (E, 8)
    g = np.array([-1, 1]) / np.sqrt(3); gp = np.array([[(a + 1) / 2 for a in (u, v, w)] for u in g for v in g for w in g])
    Ke = np.zeros((len(en), 24, 24))
    for q in range(8):
        xi = gp[q]
        Nv = np.prod(np.where(loc == 1, xi, 1 - xi), 1)                                            # (8,)
        dN = np.zeros((8, 3))
        for d in range(3):
            o = [e for e in range(3) if e != d]
            dN[:, d] = (2 * loc[:, d] - 1) * np.where(loc[:, o[0]] == 1, xi[o[0]], 1 - xi[o[0]]) * \
                np.where(loc[:, o[1]] == 1, xi[o[1]], 1 - xi[o[1]])
        xe = xn[en]                                                                                # (E, 8, 3)
        Jq = np.einsum('eai,ad->eid', xe, dN)                                                       # dx/dxi (E, 3, 3)
        det = np.linalg.det(Jq)
        if (det <= 0).any():
            raise ValueError('MACRO_ELEMENT_INVERTED')
        G = np.einsum('ad,edi->eai', dN, np.linalg.inv(Jq))                                        # physical gradients
        Xg = np.einsum('a,ead->ed', Nv, Xn[en])
        cell = np.minimum(np.floor(Xg).astype(int), shape - 1)
        D = tensor_fn(Xg, cell)                                                                    # (E, 6, 6)
        B = np.zeros((len(en), 6, 24))
        for a in range(8):
            dx, dy, dz = G[:, a, 0], G[:, a, 1], G[:, a, 2]
            B[:, 0, 3 * a] = dx; B[:, 1, 3 * a + 1] = dy; B[:, 2, 3 * a + 2] = dz
            B[:, 3, 3 * a + 1] = dz; B[:, 3, 3 * a + 2] = dy
            B[:, 4, 3 * a] = dz; B[:, 4, 3 * a + 2] = dx
            B[:, 5, 3 * a] = dy; B[:, 5, 3 * a + 1] = dx
        Ke += np.einsum('eji,ejk,ekl->eil', B, D, B) * (det / 8)[:, None, None]
    ed = (3 * en[:, :, None] + np.arange(3)).reshape(len(en), 24)
    ndof = 3 * int(M.prod())
    Kg = sp.csr_matrix((Ke.ravel(), (np.repeat(ed, 24, 1).ravel(), np.tile(ed, (1, 24)).ravel())), shape=(ndof, ndof))
    ca, la = AX[clamp[0]], AX[load[0]]
    fixed_n = gi[:, ca] == (0 if clamp[1] == 'min' else N[ca])
    # load face quads: the two other axes
    o = [d for d in range(3) if d != la]
    lv = 0 if load[1] == 'min' else N[la]
    fw = np.zeros(len(gi))
    A_, B_ = [a.ravel() for a in np.meshgrid(np.arange(N[o[0]]), np.arange(N[o[1]]), indexing='ij')]
    corners = [(0, 0), (1, 0), (0, 1), (1, 1)]
    ids = []
    for (s, t) in corners:
        idx = np.zeros((len(A_), 3), int); idx[:, la] = lv; idx[:, o[0]] = A_ + s; idx[:, o[1]] = B_ + t
        ids.append(nid(idx[:, 0], idx[:, 1], idx[:, 2]))
    ids = np.stack(ids, 1)                                                                         # (F, 4)
    xf = xn[ids]
    for u in (g + 1) / 2:
        for v in (g + 1) / 2:
            Nf = np.array([(1 - u) * (1 - v), u * (1 - v), (1 - u) * v, u * v])
            du = np.array([-(1 - v), (1 - v), -v, v]); dv = np.array([-(1 - u), -u, (1 - u), u])
            xu = np.einsum('a,fad->fd', du, xf); xv = np.einsum('a,fad->fd', dv, xf)
            dA = np.linalg.norm(np.cross(xu, xv), axis=1) / 4
            np.add.at(fw, ids.ravel(), (Nf[None, :] * dA[:, None]).ravel())
    fw /= fw.sum()
    free = np.flatnonzero(np.repeat(~fixed_n, 3))
    Kf = Kg[free][:, free].tocsr()
    F = np.zeros((ndof, 3))
    for d in range(3):
        F[d::3, d] = fw
    Ff = F[free]
    try:
        import pypardiso
        U = pypardiso.spsolve(Kf, Ff)
    except ImportError:
        import scipy.sparse.linalg as spla
        U = spla.splu(Kf.tocsc()).solve(Ff)
    return Ff.T @ U, dict(dofs=len(free))


def _rotV(Cv, R):
    """Voigt tensor rotated by R (same as homog_mapped.rotate, numpy only, batched over R (q, 3, 3))."""
    VO = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
    C = np.zeros((3, 3, 3, 3))
    for a, (i, j) in enumerate(VO):
        for b, (k, l) in enumerate(VO):
            for (p, q) in {(i, j), (j, i)}:
                for (r, s) in {(k, l), (l, k)}:
                    C[p, q, r, s] = Cv[a][b]
    R = np.asarray(R)
    if R.ndim == 2:
        R = R[None]
    Cr = np.einsum('niI,njJ,nkK,nlL,IJKL->nijkl', R, R, R, R, C)
    return np.stack([np.array([[Cr[n, i, j, k, l] for (k, l) in VO] for (i, j) in VO]) for n in range(len(R))])


def _pushV(Cv, A):
    VO = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
    C = np.zeros((3, 3, 3, 3))
    for a, (i, j) in enumerate(VO):
        for b, (k, l) in enumerate(VO):
            for (p, q) in {(i, j), (j, i)}:
                for (r, s) in {(k, l), (l, k)}:
                    C[p, q, r, s] = Cv[a][b]
    A = np.asarray(A)
    Cr = np.einsum('niI,njJ,nkK,nlL,IJKL->nijkl', A, A, A, A, C) / np.linalg.det(A)[:, None, None, None, None]
    return np.stack([np.array([[Cr[n, i, j, k, l] for (k, l) in VO] for (i, j) in VO]) for n in range(len(A))])


def _maps_module():
    """make_map / make_global of mapped_cell without importing it (it needs the GPU stack)."""
    import ast, types
    src = (Path(__file__).parent / 'mapped_cell.py').read_text()
    fns = [n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name in ('make_map', 'make_global')]
    mod = types.ModuleType('mapped_cell_maps'); mod.np = np
    exec(compile(ast.Module(body=fns, type_ignores=[]), 'mapped_cell_maps', 'exec'), mod.__dict__)
    return mod


def selftest():
    import sys as _s
    _s.path.insert(0, str(Path(__file__).parent))
    import lat_mapped as LMP
    MM = _maps_module()
    E = 2.0; nu = 0.3
    lam, mu = E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))
    CH = np.zeros((6, 6)); CH[:3, :3] = lam; CH[np.arange(3), np.arange(3)] += 2 * mu; CH[3:, 3:] = mu * np.eye(3)
    shape = (2, 1, 3)
    # identity map == box solver
    Mi, _ = macro_mapped(lambda X: np.asarray(X, float), shape, 2, lambda X, c: np.repeat(CH[None], len(X), 0))
    cb, _ = LMP.macro_compliance(CH, shape, np.asarray(shape) * 2, clamp=('z', 'min'), load=('z', 'max'))
    print(json.dumps(dict(identity_vs_box=float(np.abs(np.diag(Mi) / cb - 1).max()))))
    assert np.abs(np.diag(Mi) / cb - 1).max() < 1e-10
    # rigid rotation of the body with the rotated tensor: M_rot = R M R^T
    R = polar_R(np.array([[[0.9, -0.3, 0.2], [0.35, 0.95, -0.1], [-0.15, 0.1, 1.0]]]))[0]
    phi = MM.make_global({'kind': 'affine', 'A': R.tolist(), 'center': [1, 0.5, 1.5]})
    Mr, _ = macro_mapped(phi, shape, 2, lambda X, c: np.repeat(_rotV(CH, R), len(X), 0))
    err = np.abs(Mr - R @ Mi @ R.T).max() / np.abs(Mi).max()
    print(json.dumps(dict(rotation_covariance=float(err))))
    assert err < 1e-9
    # affine stretch with the pushed-forward tensor of a stretched homogeneous material is the stretched problem:
    # diagonal A on the box with C pushed == box of size A shape with C pushed (box solver)
    A = np.diag([1.5, 1.0, 0.7])
    Mp, _ = macro_mapped(MM.make_global({'kind': 'affine', 'A': A.tolist()}), shape, 2,
                         lambda X, c: np.repeat(_pushV(CH, A[None]), len(X), 0))
    cp, _ = LMP.macro_compliance(_pushV(CH, A[None])[0], np.asarray(shape) * np.diag(A), np.asarray(shape) * 2,
                                 clamp=('z', 'min'), load=('z', 'max'))
    print(json.dumps(dict(affine_vs_box=float(np.abs(np.diag(Mp) / cp - 1).max()))))
    assert np.abs(np.diag(Mp) / cp - 1).max() < 1e-10
    # twisted beam: refinement converges
    tw = MM.make_global({'kind': 'twist', 'deg': 10, 'center': [1, 0.5, 0]})
    cs = [np.diag(macro_mapped(tw, shape, m, lambda X, c: np.repeat(CH[None], len(X), 0))[0]) for m in (1, 2, 4)]
    print(json.dumps(dict(twist_refinement=[c.tolist() for c in cs])))
    # Procrustes and class rotation on the twist map
    X = np.random.default_rng(0).uniform(0, 1, (50, 3))
    g = {'kind': 'twist', 'deg': 10, 'center': [1, 1, 0]}
    a = MM.make_map({'kind': 'global', 'map': g, 'offset': [0, 1, 0]})(X)
    b = MM.make_map({'kind': 'global', 'map': g, 'offset': [0, 1, 3]})(X)
    Rf, t, res = procrustes(a, b)
    ang = np.rad2deg(np.arctan2(Rf[1, 0], Rf[0, 0]))
    print(json.dumps(dict(procrustes_residual=res, angle=float(ang))))
    assert res < 1e-12 and abs(ang - 30) < 1e-9
    # K_PP block rotation: dense check
    import torch
    rng = np.random.default_rng(1)
    n = 5; Kd = rng.standard_normal((3 * n, 3 * n)); Kd = Kd @ Kd.T
    r, c = np.triu_indices(3 * n)
    rr, cc, vv = rotate_kpp(torch.as_tensor(r), torch.as_tensor(c), torch.as_tensor(Kd[r, c]), Rf, torch)
    K2 = np.zeros_like(Kd); K2[rr.numpy(), cc.numpy()] = vv.numpy(); K2 = K2 + np.triu(K2, 1).T
    T = np.kron(np.eye(n), Rf)
    err = np.abs(K2 - T @ Kd @ T.T).max()
    print(json.dumps(dict(kpp_rotation=float(err))))
    assert err < 1e-12
    print('SELFTEST_OK')


# ------------------------------------------------------------------------------------------------ fine scale (GPU)
class RotOp:
    """S_p q = T_R S T_R^T q for a dense representative S (T_R: R on every port node, node-major xyz)."""

    def __init__(self, S, R):
        import torch
        self.S, self.R = S, torch.as_tensor(R, dtype=S.dtype, device=S.device)

    def _rot(self, q, R):
        k = q.shape[1]
        return (R @ q.reshape(-1, 3, k)).reshape(-1, k)

    def apply(self, q):
        q = q.to(self.S.dtype)
        return self._rot(self.S @ self._rot(q, self.R.T), self.R)


def main(argv):
    if argv and argv[0] == '--selftest':
        selftest(); return
    import torch
    import models as MD                                                   # first: applies OPL_CONV_FP32
    import trainlib as TL
    import a0_eval as AE
    import a0_budget as AB
    import mapped_cell as MC
    import corot_smooth as CR
    import weak_patch as WP
    import homog_mapped as HM
    import lat_multi as LM
    import lat_precond as PR
    import lat_mapped as LMP
    dev, dt = AE.dev, AE.dt

    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('ckpt'); ap.add_argument('body'); ap.add_argument('case')
    ap.add_argument('--map', required=True); ap.add_argument('--shape', default='2x2x4')
    ap.add_argument('--clamp', default=''); ap.add_argument('--load', default='')
    ap.add_argument('--fields', default='c1,c2w'); ap.add_argument('--work', default='/root/autodl-tmp/OPL/A0/work_glob')
    ap.add_argument('--prec', default='bnn:kpp:q1r'); ap.add_argument('--tol', type=float, default=1e-10)
    ap.add_argument('--maxit', type=int, default=4000); ap.add_argument('--n-random', type=int, default=2)
    ap.add_argument('--m', default='2,4'); ap.add_argument('--p1', default=''); ap.add_argument('--check', type=int, default=1)
    ap.add_argument('--wp_dil', type=int, default=2); ap.add_argument('--wp_seed', default='cutweakbox')
    ap.add_argument('--chunk', type=int, default=16); ap.add_argument('--fast', type=int, default=0)
    ap.add_argument('--layout', default='')
    a = ap.parse_args(argv)
    log = lambda d: print(json.dumps(d, default=float), flush=True)
    G = json.loads(a.map); shape = tuple(int(v) for v in a.shape.split('x'))
    dflt = {'twist': (('z', 'min'), ('z', 'max')), 'bend': (('x', 'min'), ('x', 'max'))}.get(G['kind'], (('z', 'min'), ('z', 'max')))
    clamp = tuple(a.clamp.split(',')) if a.clamp else dflt[0]
    load = tuple(a.load.split(',')) if a.load else dflt[1]
    gphi = MC.make_global(G)
    spec_at = lambda p: {'kind': 'global', 'map': G, 'offset': [int(v) for v in p]}
    if a.layout:
        LY = json.loads(Path(a.layout).read_text())
        if tuple(LY['shape']) != shape:
            raise ValueError('LAYOUT_SHAPE')
        case_of = {tuple(c['position']): c['case'] for c in LY['cells']}
        a.body = LY['body']
    else:
        case_of = {(i, j, k): a.case for i in range(shape[0]) for j in range(shape[1]) for k in range(shape[2])}
    positions = sorted(case_of)
    classes = {}
    for p in positions:
        classes.setdefault((case_of[p],) + class_key(G, p), []).append(p)
    Xs = np.random.default_rng(0).uniform(0, 1, (64, 3))
    rot = {}                                                            # p -> (representative, R)
    for key, ps in classes.items():
        rep = ps[0]; xr = MC.make_map(spec_at(rep))(Xs)
        for p in ps:
            R, t, res = procrustes(xr, MC.make_map(spec_at(p))(Xs))
            if res > 1e-9:
                raise ValueError(f'CLASS_NOT_RIGID:{key}:{p}:{res}')
            rot[p] = (rep, R)
    res_ = dict(args=vars(a), map=G, shape=shape, clamp=clamp, load=load, conv=TL.conv_precision(),
                classes={str(k): [list(p) for p in v] for k, v in classes.items()}, reps={}, fields={})
    ref = None
    if a.p1:
        for c in json.loads(Path(a.p1).read_text())['cells']:
            if c['case'] == a.case:
                ref = np.asarray(c['CH'])
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False); cfg = ck['cfg']
    model = None
    CR.install()
    reps = sorted({v[0] for v in rot.values()})
    S = {}                                                              # (field, rep) -> dense operator (host fp64)
    geoms, CHrep = {}, {}
    for rep in reps:
        t0 = time.perf_counter()
        rr = res_['reps'][str(rep)] = {}
        C = MC.MappedCell(case_of[rep], a.body, spec_at(rep), log=lambda s_: None); C.assemble()
        rr.update(map_stats=C.map_stats, ports=int(C.np_), dofs=int(C.nb), setup_s=time.perf_counter() - t0)
        geom = LM.from_teacher(C)
        face = {}                                                       # consistent face weights depend on the case only
        for ax_, side in (clamp, load):
            face[(AX[ax_], 0.0 if side == 'min' else 1.0)] = geom.face_w_fn(AX[ax_], 0.0 if side == 'min' else 1.0)
        geoms[rep] = (dict(n=geom.n, port_node_ids=geom.port_node_ids, priv=geom.priv, face=face),
                      tuple(x.clone() for x in geom.kpp()))
        del geom                                                        # (holds the cell)
        # homogenised tensor of the affine cell with A = J at the representative's centre
        Jc = jac(gphi, (np.asarray(rep, float) + 0.5)[None])[0]
        rr['J_centre'] = Jc.tolist(); sv = np.linalg.svd(Jc, compute_uv=False); rr['kappa_centre'] = float(sv[0] / sv[-1])
        if not a.layout:
            Ca = MC.MappedCell(a.case, a.body, {'kind': 'affine', 'A': Jc.tolist()}, log=lambda s_: None); Ca.assemble()
            CHa, _ = HM.periodic_CH(Ca); CHrep[rep] = CHa / np.linalg.det(Jc)
            Ca._free(); del Ca; gc.collect(); torch.cuda.empty_cache()
        rr['case'] = case_of[rep]
        # exact
        t = time.perf_counter()
        C.factor(neumann=False, fp32=False)
        S[('exact', rep)] = LMP._dense(C.apply, C.np_, 512, dev, dt).cpu()
        C._free(); rr['exact_dense_s'] = time.perf_counter() - t
        # learned
        d = Path(a.work) / ('rep_' + '_'.join(map(str, rep)))
        AE.write_data(C, d / case_of[rep], {}, (), a.body)
        g = AB.BudgetGeo(case_of[rep], a.body, d, neumann=False, log=lambda s_: None, cell=C, load_banks=False)
        g.case = f'{case_of[rep]}@{rep}'
        if model is None:
            model = MD.build(cfg['model'], [g], **dict(cfg.get('model_args', {}))).to(dev)
            MD.load_compat(model, ck['model']); model.eval()
        else:
            model.add_geo(g)
        wrap = AE._Wrap(model)
        CR.set_corot(C, AE.nodal_rotations(C))
        for fname in [f for f in a.fields.split(',') if f]:
            base, cyc = AB.FIELDS[fname]
            if fname.endswith('w') and getattr(C, '_wp', 'unset') == 'unset':
                rr['weakpatch'] = WP.setup(C, g.nd, dil=a.wp_dil, seed=a.wp_seed)
            g.set_budget(base, cyc, wrap, wp=fname.endswith('w'))
            t = time.perf_counter()
            if a.fast:
                import mapped_fast as MF
                mop = MF.MappedFastOp(g, model, cyc, wrap, patch=fname.endswith('w'))
                S[(fname, rep)] = LMP._dense(mop.s_hat, C.np_, 64, dev, dt).cpu()
                del mop
            else:
                S[(fname, rep)] = LMP._dense(lambda E: g.s_hat_apply(model, E), C.np_, a.chunk, dev, dt).cpu()
            rr[f'{fname}_dense_s'] = time.perf_counter() - t
            torch.cuda.empty_cache()
        model.caches.pop(g.case, None)
        try:
            WP.free(C)
        except Exception:                                                # noqa: BLE001
            pass
        C._free(); del C, g; gc.collect(); torch.cuda.empty_cache()
        log(dict(event='REP', rep=rep, **{k: v for k, v in rr.items() if k != 'map_stats'}))
        Path(a.out).write_text(json.dumps(res_, indent=1, default=float))
    # ------------------------------------------------ check the rotation sharing on one non-representative cell
    if a.check:
        others = [p for p in positions if rot[p][0] != p]
        if others:
            p = others[-1]; rep, R = rot[p]
            C = MC.MappedCell(case_of[p], a.body, spec_at(p), log=lambda s_: None); C.assemble()
            C.factor(neumann=False, fp32=False)
            gen = torch.Generator(device=dev).manual_seed(1)
            Q = torch.randn((C.np_, 16), dtype=dt, device=dev, generator=gen)
            Se = C.apply(Q); C._free()
            Sr = RotOp(S[('exact', rep)].to(dev), R).apply(Q)
            chk = dict(cell=list(p), rep=list(rep), exact_rel=float((Se - Sr).norm() / Se.norm()))
            d = Path(a.work) / ('chk_' + '_'.join(map(str, p)))
            AE.write_data(C, d / case_of[p], {}, (), a.body)
            g = AB.BudgetGeo(case_of[p], a.body, d, neumann=False, log=lambda s_: None, cell=C, load_banks=False)
            g.case = f'{case_of[p]}@{p}'; model.add_geo(g)
            wrap = AE._Wrap(model)
            CR.set_corot(C, AE.nodal_rotations(C))
            for fname in [f for f in a.fields.split(',') if f]:
                base, cyc = AB.FIELDS[fname]
                if fname.endswith('w') and getattr(C, '_wp', 'unset') == 'unset':
                    WP.setup(C, g.nd, dil=a.wp_dil, seed=a.wp_seed)
                g.set_budget(base, cyc, wrap, wp=fname.endswith('w'))
                Sl = g.s_hat_apply(model, Q)
                Slr = RotOp(S[(fname, rep)].to(dev), R).apply(Q)
                chk[f'{fname}_rel'] = float((Sl - Slr).norm() / Sl.norm())
            model.caches.pop(g.case, None)
            try:
                WP.free(C)
            except Exception:                                            # noqa: BLE001
                pass
            C._free(); del C, g; gc.collect(); torch.cuda.empty_cache()
            res_['rotation_check'] = chk
            log(dict(event='CHECK', **chk))
    # ------------------------------------------------ the lattice
    layout = {}
    la_ = AX[load[0]]
    afac = []
    for p in positions:
        rep, R = rot[p]
        geom0, (r0, c0, v0) = geoms[rep]
        kr = rotate_kpp(r0, c0, v0, R, torch) if not np.allclose(R, np.eye(3)) else (r0, c0, v0)
        face = {}
        for key_, (w, out) in geom0['face'].items():
            w = np.asarray(w, float).copy()
            if key_[0] == la_:
                on = np.flatnonzero(w != 0)
                n_ = geom0['n']
                Xp = np.stack(np.unravel_index(geom0['port_node_ids'][on], (2 * n_ + 1,) * 3), 1) / (2 * n_) + np.asarray(p)
                af = area_factor(gphi, Xp, la_)
                w[on] *= af; afac.append(af)
            face[key_] = (w, out)
        gp_ = LM.CellGeom(case_of[p], geom0['n'], geom0['port_node_ids'], np.zeros(len(geom0['port_node_ids']), bool),
                          geom0['priv'], kpp_fn=(lambda kr=kr: kr), face_w_fn=(lambda ax_, val, f=face: f[(ax_, float(val))]))
        layout[p] = gp_
    if afac:
        afac = np.concatenate(afac)
        res_['load_area_factor'] = dict(min=float(afac.min()), max=float(afac.max()), mean=float(afac.mean()))
    lat = LM.MultiLattice(layout, clamp=clamp, load=load, loads='consistent', n_random=a.n_random, device=dev,
                          log=lambda s_: None)
    kpp_l = lat.assemble_kpp()
    res_['free_dofs'] = int(lat.nfree); res_['labels'] = lat.labels
    order = [tuple(int(v) for v in pp) for pp in lat.positions]
    Xref = None
    for fname in ['exact'] + [f for f in a.fields.split(',') if f]:
        dense = {rep: S[(fname, rep)].to(dev) for rep in reps}
        ops = [RotOp(dense[rot[p][0]], rot[p][1]) for p in order]
        fac = PR.Factory(lat, ops, shared={'kpp_triplets': kpp_l, 'kpp_triplets_s': 0.0}, kpp_backend='auto', log=lambda d_: None)
        pc, st, _ = fac.build(a.prec)
        r = PR.solve(lat, ops, pc, tol=a.tol, maxit=a.maxit)
        X = r['X']
        row = dict(iterations=r['iterations'], seconds=r['seconds'], true_residual=PR.true_residual(lat, ops, X),
                   compliance=(lat.F * X).sum(0).cpu().numpy().tolist())
        if fname == 'exact':
            Xref = X
        else:
            ce = np.asarray(res_['fields']['exact']['compliance'])
            row['compliance_rel_err'] = (np.asarray(row['compliance']) / ce - 1).tolist()
            row['solution_rel_err'] = float((X - Xref).norm() / Xref.norm())
            ex = sx = 0
            for i, p in enumerate(order):
                q = lat.gather(Xref, i)
                ex = ex + (q * ops[i].apply(q)).sum(0)
                sx = sx + (q.cpu() * RotOp(S[('exact', rot[p][0])], rot[p][1]).apply(q.cpu())).sum(0).to(dev)
            row['bound'] = (ex / sx - 1).cpu().numpy().tolist()
        res_['fields'][fname] = row
        log(dict(event='LAT', field=fname, it=row['iterations'], err=row.get('compliance_rel_err'), res=row['true_residual']))
        fac.free(); del pc, r, ops, dense; gc.collect(); torch.cuda.empty_cache()
        Path(a.out).write_text(json.dumps(res_, indent=1, default=float))
    # ------------------------------------------------ homogenised macro models
    if a.layout:
        print('DONE', flush=True)
        return
    ce = np.asarray(res_['fields']['exact']['compliance'][:3])
    ms = sorted(int(v) for v in a.m.split(','))
    tens = {'homog_cell': lambda X, c: np.stack([_rotV(CHrep[rot[tuple(cc)][0]], rot[tuple(cc)][1])[0] for cc in c])}
    if ref is not None:
        tens['homog_push'] = lambda X, c: _pushV(ref, jac(gphi, X))
        tens['homog_rot'] = lambda X, c: _rotV(ref, polar_R(jac(gphi, X)))
    for k, fn in tens.items():
        cs = []
        for m in ms:
            t = time.perf_counter()
            Mm, info = macro_mapped(gphi, shape, m, fn, clamp=clamp, load=load)
            cs.append(np.diag(Mm))
            res_['fields'][f'{k}_m{m}'] = dict(compliance=np.diag(Mm).tolist(), compliance_rel_err=(np.diag(Mm) / ce - 1).tolist(),
                                              M=Mm.tolist(), seconds=time.perf_counter() - t, **info)
            log(dict(event='MACRO', field=f'{k}_m{m}', err=(np.diag(Mm) / ce - 1).tolist()))
        if len(ms) >= 2 and ms[-1] == 2 * ms[-2]:
            c = cs[-1] + (cs[-1] - cs[-2]) / 3
            res_['fields'][f'{k}_rich'] = dict(compliance=c.tolist(), compliance_rel_err=(c / ce - 1).tolist())
            log(dict(event='MACRO', field=f'{k}_rich', err=(c / ce - 1).tolist()))
    Path(a.out).write_text(json.dumps(res_, indent=1, default=float))
    print('DONE', flush=True)


if __name__ == '__main__':
    main(sys.argv[1:])
