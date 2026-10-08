"""Direction A / P1 operator: weak-node patch correction (default off; new module, trainlib unchanged).

Observation (a0_worst, 16 validation cells, identity map): the worst directions of every cut cell (mu - 1 = 8-45%) are
displacements of about a dozen weakly supported cut-band nodes; ~80% of their error energy lies on weak nodes, which the
Jacobi-Chebyshev tail cannot reach (tiny diagonal, poor conditioning) and the Q1 coarse space does not resolve.
Correction: Galerkin minimisation of the energy over the interior DOFs W within `dil` grid steps of the weakly supported
RETAINED nodes (seed 'weakport'), of all retained cut-band nodes (seed 'cutport': diag_wp.py finds 100% of the worst-direction
error energy within 2 grid steps of them, 90% on 10-21 nodes), of those plus the weak box-face retained nodes (seed
'cutweakbox': the box-load worst directions of full cells sit on < 10 weak box nodes), or of all weak nodes ('weak', a large part of a thin-walled cell):

    x_W <- x_W + A_WW^-1 r_W,   r = -(K x)_I,   W = interior DOFs of the patch,

i.e. Eq. (14) of P1 with V = the coordinate injection of W. Linear in x, preserves the retained values, cannot increase the
A-energy error (exact subspace minimisation), so S <= S_hat(after) <= S_hat(before) for every input. A_WW is extracted from
the upper-triangular COO of K (C.ru, C.cu, C.vals); patches up to `dense_max` DOFs are factored densely (fp64 Cholesky),
larger ones with the sparse SPD solver of teacher.py (cuDSS, Jacobi-scaled upper CSR, fp64). free(C) releases it."""
import numpy as np
import torch

import trainlib as TL

dev, dt = TL.dev, TL.dt


def setup(C, nd, dil=2, seed='cutport', max_dofs=200000, dense_max=8000):
    """nd: NETDATA dict ('weak', 'is_port' over C.nodes). Stores C._wp = (W (global interior dof ids), L) or None."""
    n = C.n; M = 2 * n + 1
    grid = np.stack(np.unravel_index(C.nodes, (M,) * 3), 1)
    weak_nodes = np.asarray(nd['weak'], bool)
    port = np.asarray(nd['is_port'], bool)
    cutp = np.asarray(nd['is_cut'], bool) & port
    sel = {'weakport': weak_nodes & port, 'cutport': cutp, 'weak': weak_nodes.copy(),
           'cutweakbox': cutp | (weak_nodes & port & np.asarray(nd['is_box'], bool))}[seed]   # + weak box-face ports
    seed_nodes = int(sel.sum())
    if dil > 0 and sel.any():
        vol = torch.zeros((1, 1, M, M, M), dtype=torch.float32, device=dev)
        g = torch.as_tensor(grid, device=dev)
        w = g[torch.as_tensor(sel, device=dev)]
        vol[0, 0, w[:, 0], w[:, 1], w[:, 2]] = 1
        vol = torch.nn.functional.max_pool3d(vol, 2 * dil + 1, stride=1, padding=dil)
        sel = (vol[0, 0, g[:, 0], g[:, 1], g[:, 2]] > 0).cpu().numpy()
    node_ids = np.flatnonzero(sel)
    dofs = (3 * node_ids[:, None] + np.arange(3)[None]).reshape(-1)
    interior = np.zeros(C.nb, bool); interior[C.I.cpu().numpy()] = True
    W = dofs[interior[dofs]]
    info = dict(seed=seed, dil=dil, seed_nodes=seed_nodes, weak_nodes=int(weak_nodes.sum()), patch_nodes=int(sel.sum()), patch_dofs=int(len(W)))
    if len(W) == 0 or len(W) > max_dofs:
        C._wp = None; info['skipped'] = True
        return info
    loc = np.full(C.nb, -1, np.int64); loc[W] = np.arange(len(W))
    locT = torch.as_tensor(loc, device=dev)
    ru, cu, v = C.ru.long(), C.cu.long(), C.vals.to(dt)
    a, b = locT[ru], locT[cu]
    keep = (a >= 0) & (b >= 0)
    if len(W) <= dense_max:
        A = torch.zeros((len(W), len(W)), dtype=dt, device=dev)
        A.index_put_((a[keep], b[keep]), v[keep], accumulate=True)
        off = keep & (ru != cu)
        A.index_put_((b[off], a[off]), v[off], accumulate=True)
        A = 0.5 * (A + A.T)
        L, err = torch.linalg.cholesky_ex(A)
        if int(err) != 0:
            C._wp = None; info['skipped'] = 'cholesky'
            return info
        info['A_diag_ratio'] = float(A.diagonal().min() / A.diagonal().max()); del A
        C._wp = (torch.as_tensor(W, device=dev), ('dense', L))
    else:                                                 # sparse: rows of the upper COO stay sorted (loc is monotone)
        import teacher as TE
        rA, cA, vA = a[keep], b[keep], v[keep]
        d = vA[rA == cA]
        s_ = torch.zeros(len(W), dtype=dt, device=dev); s_[rA[rA == cA]] = 1 / torch.sqrt(d)
        crow = torch.cat([torch.zeros(1, dtype=torch.long, device=dev), torch.cumsum(torch.bincount(rA, minlength=len(W)), 0)])
        sol = TE.SPDSolver(crow.int(), cA.int(), (vA * s_[rA] * s_[cA]).contiguous(), len(W))
        info['A_diag_ratio'] = float(d.min() / d.max()); info['sparse'] = True
        C._wp = (torch.as_tensor(W, device=dev), ('sparse', sol, s_))
    return info


class _SparseSolve(torch.autograd.Function):
    """y = A^-1 r with A SPD (symmetric), so the adjoint is the same solve."""

    @staticmethod
    def forward(ctx, r, sol, s_):
        ctx.sol, ctx.s_ = sol, s_
        return s_[:, None] * sol.solve((s_[:, None] * r).contiguous())

    @staticmethod
    def backward(ctx, g):
        s_ = ctx.s_
        return s_[:, None] * ctx.sol.solve((s_[:, None] * g).contiguous()), None, None


def free(C):
    wp = getattr(C, '_wp', None)
    if wp is not None and wp[1][0] == 'sparse':
        try:
            wp[1][1].free()
        except Exception:
            pass
    C._wp = None


def apply(C, x):
    """x (nb, B) -> x with the patch correction (differentiable, fp64)."""
    if getattr(C, '_wp', None) is None:
        return x
    W, f = C._wp
    x = x.to(dt)
    r = -TL._KMat.apply(x, TL._Kc(C))[W]
    y = torch.cholesky_solve(r, f[1]) if f[0] == 'dense' else _SparseSolve.apply(r, f[1], f[2])
    return x.index_add(0, W, y)


def wrap_geo(g):
    """Make g.field (and hence s_hat_apply) apply the patch after the deployed correction."""
    orig = g.field

    def field(model, q):
        return apply(g.C, orig(model, q))
    g.field = field
    return g
