"""GPU encode: geometry -> reusable structured boundary operator (first-batch cases, for accuracy and timing).

Stages (each timed, all FP64 on one GPU):
  E1 elements   polyhedral reference moments (polyref_torch) -> K_e = sum_m M_m T_m -> body K (upper, scatter-add)
                -> + gamma * archived ghost penalty -> Kc = P^T K P with the packet's trace compiler P -> A, C, D.
                (--elements teacher uses the compiled teacher A, C, D instead, to time the hierarchy alone.)
  E2 hierarchy  4h/8h patches; per patch an orthonormal basis of all affine displacement fields (rigid6 + 6
                strains = the span of Codex's rigid6 -> strain-enriched spaces), batched SVD; P1, P2, P12 = P1^T P2
                (nested exactly); A1 = P1^T A P1, A2 = P12^T A1 P12; level-1 block factors (batched Cholesky);
                dense coarsest inverse-Cholesky; strong-pair fine blocks (theta), omega bounds.
  E3 design     PCG-Lanczos estimate of lambda_min(B A) -> Chebyshev interval a = safety * ritz_min and depth.
Then the fixed Chebyshev strain network (Codex's frozen core and recurrence, coefficients for interval [a,1])
is evaluated with the finite witness protocol against the teacher (REFERENCE_RQ).
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
from scipy import sparse
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = Path('/root/autodl-tmp/CUTFEM_FRESH_GP_20260921')
MN = ROOT / 'diagnostics' / 'MECHANICS_NETWORK_20260922_01'
HS = ROOT / 'diagnostics' / 'HIERARCHY_STRAIN_20260922_01'
dev = torch.device('cuda:0')
CPU_TRACE = False
GPU_EXPAND = False
TAU_EPS = '0'
COVER_INPUTS = None  # teacher-free inputs (P, ghost, interior bookkeeping) built by gp/cover_blocks.py
dt = torch.float64


class Timer:
    def __init__(self): self.rows = {}
    def __call__(self, name):
        timer = self
        class _T:
            def __enter__(s): torch.cuda.synchronize(); s.t = time.perf_counter()
            def __exit__(s, *a): torch.cuda.synchronize(); timer.rows[name] = time.perf_counter() - s.t
        return _T()


def to_torch_csr(M):
    M = sparse.csr_matrix(M)
    return torch.sparse_csr_tensor(torch.from_numpy(M.indptr.astype(np.int64)), torch.from_numpy(M.indices.astype(np.int64)),
                                   torch.from_numpy(M.data), size=M.shape, dtype=dt).to(dev)


def coo(rows, cols, vals, shape):
    return torch.sparse_coo_tensor(torch.stack([rows, cols]), vals, shape).coalesce()


def spmm(a, b):
    return torch.sparse.mm(a, b)


def elements_polyref(case, s, order, T, levels=0):
    import element_moments as EM, polyref_torch as PT
    ctx = json.loads((ROOT / 'packets' / case / 'FRESH_CONTEXT.json').read_text())
    n = int(ctx['n']); E = float(ctx['material']['E']); nu = float(ctx['material']['nu'])
    lam = E * nu / ((1 + nu) * (1 - 2 * nu)); mu_ = E / (2 * (1 + nu))
    from fractions import Fraction
    taus = [float(Fraction(v) * (1 + Fraction(TAU_EPS))) for v in ctx['case']['tau_corners']]
    normal = None if ctx['case'].get('normal') is None else [float(v) for v in ctx['case']['normal']]
    offset = None if normal is None else float(ctx['case']['offset'])
    arr = EM.members(case, ['NODES.npy', 'dofs.npy', 'CELL_INDICES.npy'])
    xi = EM.local_coordinates(case, n, arr)
    keys = np.unique(xi.reshape(len(xi), -1), axis=0)
    if len(keys) != 1:
        raise ValueError('ONE_NODE_ORDERING_EXPECTED')
    Tm = torch.tensor(EM.pattern_operators(keys[0].reshape(27, 3), lam, mu_, n)[1], dtype=dt, device=dev)  # 125x81x81
    with T('E1a_moments'):
        M = torch.tensor(PT.cell_moments(arr['CELL_INDICES.npy'], n, taus, normal, offset, s, rule_order=order, levels=levels), dtype=dt, device=dev)
    nb = 3 * len(arr['NODES.npy'])
    dofs = torch.as_tensor(arr['dofs.npy'], dtype=torch.long, device=dev)
    iu = torch.triu_indices(81, 81, device=dev)
    with T('E1bc_element_matrices_and_assembly'):
        # upper triangle of every element matrix, assembled in chunks (bounded memory), then symmetrized
        parts = []
        for lo in range(0, len(M), 2048):
            Ke = (M[lo:lo + 2048] @ Tm.reshape(125, -1)).reshape(-1, 81, 81)[:, iu[0], iu[1]]
            dd = dofs[lo:lo + 2048]
            r, c = dd[:, iu[0]], dd[:, iu[1]]
            lo_, hi_ = torch.minimum(r, c), torch.maximum(r, c)
            parts.append(coo(lo_.reshape(-1), hi_.reshape(-1), Ke.reshape(-1), (nb, nb)))
            del Ke
        U = parts[0]
        for q in parts[1:]:
            U = (U + q).coalesce()
        del parts
        i, v = U.indices(), U.values()
        off = i[0] != i[1]
        Kb = coo(torch.cat([i[0], i[1][off]]), torch.cat([i[1], i[0][off]]), torch.cat([v, v[off]]), (nb, nb))
        del U, i, v
    return Kb, nb, ctx


def ghost(case, nb):
    if COVER_INPUTS is not None:
        G = sparse.load_npz(COVER_INPUTS / 'GHOST.npz').tocoo()
        return coo(torch.as_tensor(G.row.astype(np.int64), device=dev), torch.as_tensor(G.col.astype(np.int64), device=dev),
                   torch.as_tensor(G.data, dtype=dt, device=dev), (nb, nb))
    src = ROOT / 'source_archives' / case
    inv = json.loads((src / 'INVENTORY.json').read_text())
    o = next(s for s in inv['stages'] if s.endswith('_O'))
    import tarfile, io
    arrs = {}
    with tarfile.open(src / 'SOURCE.tar', 'r:') as t:
        for n_ in ('K_data.npy', 'K_indices.npy', 'K_indptr.npy', 'global_support.npy'):
            arrs[n_] = np.load(io.BytesIO(t.extractfile(t.getmember(f'{o}/ghost/{n_}')).read()))
    sup = arrs['global_support.npy']
    gu = sparse.csr_matrix((arrs['K_data.npy'], arrs['K_indices.npy'], arrs['K_indptr.npy']), shape=(len(sup), len(sup))).tocoo()
    r, c, v = sup[gu.row], sup[gu.col], gu.data
    off = r != c
    rr = np.r_[r, c[off]]; cc = np.r_[c, r[off]]; vv = np.r_[v, v[off]]
    return coo(torch.as_tensor(rr, device=dev), torch.as_tensor(cc, device=dev), torch.as_tensor(vv, dtype=dt, device=dev), (nb, nb))


def blocks_from_polyref(case, s, order, T, levels=0):
    Kb, nb, ctx = elements_polyref(case, s, order, T, levels)
    G = ghost(case, nb)
    gamma = float(json.loads((ROOT / 'packets' / case / 'SAMPLE.json').read_text())['gp']['gamma'])
    if COVER_INPUTS is not None:
        P = sparse.load_npz(COVER_INPUTS / 'P.npz').tocsr(); m = json.loads((COVER_INPUTS / 'RESULT.json').read_text())['m']
    else:
        P = sparse.load_npz(ROOT / 'packets' / case / 'ORIGINAL_FROM_TRACE_FREE.npz').tocsr()
        m = json.loads((ROOT / 'packets' / case / 'SAMPLE.json').read_text())['full_trace_dimension']
    with T('E1d_trace_compile_PtKP'):
        K = (Kb + gamma * G).coalesce()
        del Kb, G
        Kc = None
        if GPU_EXPAND:
            try:
                Kc = ptkp_upper_gpu(K, P)
            except torch.OutOfMemoryError:  # very large cells: fall back to the CPU product below
                Kc = None; torch.cuda.empty_cache()
                T.rows['E1d_gpu_trace_fallback_cpu'] = 1.0
        if Kc is not None:
            del K
        elif CPU_TRACE or GPU_EXPAND:  # large cells: P^T K P with scipy (K assembled on the GPU), result back to the GPU
            ki, kv = K.indices().cpu().numpy(), K.values().cpu().numpy(); del K
            Ks = sparse.csr_matrix((kv, (ki[0], ki[1])), shape=(nb, nb))
            Kcs = (P.T @ Ks @ P).tocoo(); del Ks
            Kc = coo(torch.as_tensor(Kcs.row.astype(np.int64), device=dev), torch.as_tensor(Kcs.col.astype(np.int64), device=dev),
                     torch.as_tensor(Kcs.data, dtype=dt, device=dev), Kcs.shape)
        else:
            Pq = P.tocoo()
            Pc = coo(torch.as_tensor(Pq.row.astype(np.int64), device=dev), torch.as_tensor(Pq.col.astype(np.int64), device=dev),
                     torch.as_tensor(Pq.data, dtype=dt, device=dev), P.shape)
            Kc = spmm(Pc.t().coalesce(), spmm(K, Pc)).coalesce()
        # original compiler: upper triangle, then symmetric replay
        i, v = Kc.indices(), Kc.values()
        up = i[0] <= i[1]
        i, v = i[:, up], v[up]
        off = i[0] != i[1]
        Kc = coo(torch.cat([i[0], i[1][off]]), torch.cat([i[1], i[0][off]]), torch.cat([v, v[off]]), Kc.shape)
        i, v = Kc.indices(), Kc.values()
        def block(rsel, csel, r0, c0, shape):
            sel = rsel & csel
            return coo(i[0][sel] - r0, i[1][sel] - c0, v[sel], shape)
        n0 = Kc.shape[0] - m
        A = block(i[0] >= m, i[1] >= m, m, m, (n0, n0))
        C = block(i[0] >= m, i[1] < m, m, 0, (n0, m))
        D = block(i[0] < m, i[1] < m, 0, 0, (m, m))
    return A, C, D


def ptkp_upper_gpu(K, P, chunk=1 << 22):
    """Upper triangle of P^T K P by direct triplet expansion on the GPU, reading only the upper triangle of the
    symmetric K. An entry (i, j, v), i <= j, and the nonzeros P[i, a], P[j, b] give w = v P[i, a] P[j, b]:
    for i < j the pair (i, j), (j, i) of the full K adds w at (min(a, b), max(a, b)), doubled on the diagonal a == b;
    for i == j only a <= b is kept. (P is a permutation on full cells and has one nonzero in most rows otherwise.)
    Chunks are merged as they accumulate so the peak stays near one coalesced upper triangle."""
    ip = torch.as_tensor(P.indptr.astype(np.int64), device=dev); ix = torch.as_tensor(P.indices.astype(np.int64), device=dev)
    pv = torch.as_tensor(P.data, dtype=dt, device=dev); cnt = ip[1:] - ip[:-1]
    ki, kv = K.indices(), K.values()
    upper = torch.nonzero(ki[0] <= ki[1]).squeeze(1)
    shape = (P.shape[1], P.shape[1]); acc = None
    for s0 in range(0, upper.numel(), chunk):
        sel = upper[s0:s0 + chunk]
        i, j, v = ki[0, sel], ki[1, sel], kv[sel]
        ri, rj = cnt[i], cnt[j]; m = ri * rj
        e = torch.repeat_interleave(torch.arange(len(m), device=dev), m)
        t = torch.arange(len(e), device=dev) - (torch.cumsum(m, 0) - m)[e]
        pa = ip[i[e]] + t // rj[e]; pb = ip[j[e]] + t % rj[e]
        ca, cb = ix[pa], ix[pb]; w = v[e] * pv[pa] * pv[pb]
        off = i[e] != j[e]
        keep = off | (ca <= cb)
        w = torch.where(off & (ca == cb), 2 * w, w)
        lo, hi = torch.minimum(ca, cb), torch.maximum(ca, cb)
        part = coo(lo[keep], hi[keep], w[keep], shape)
        del e, t, pa, pb, ca, cb, w, off, keep, lo, hi
        acc = part if acc is None else coo(torch.cat([acc.indices()[0], part.indices()[0]]), torch.cat([acc.indices()[1], part.indices()[1]]),
                                           torch.cat([acc.values(), part.values()]), shape)
        del part
    return acc


def blocks_teacher(case, asset_root):
    comp = (MN / 'assets' / case if asset_root is None else Path(asset_root) / case) / 'compiled'
    out = []
    for name in ('A', 'C', 'D'):
        M = sparse.load_npz(comp / (name + '.npz')).tocoo()
        out.append(coo(torch.as_tensor(M.row.astype(np.int64), device=dev), torch.as_tensor(M.col.astype(np.int64), device=dev),
                       torch.as_tensor(M.data, dtype=dt, device=dev), M.shape))
    return out


def affine_basis(points, owner, tol=1e-10, degree=1):
    """Per patch: orthonormal basis of the polynomial displacement fields of the given degree (padded batched SVD):
    degree 1 = the 12 affine fields (rigid + strain), degree 2 adds the 18 quadratic fields (30 in total)."""
    order = torch.argsort(owner, stable=True)
    ids, counts = torch.unique_consecutive(owner[order], return_counts=True)
    npatch, width = len(ids), int(counts.max())
    start = torch.cumsum(counts, 0) - counts
    local = torch.arange(len(owner), device=dev) - torch.repeat_interleave(start, counts)
    patch_of = torch.repeat_interleave(torch.arange(npatch, device=dev), counts)
    node_of = torch.full((npatch, width), -1, dtype=torch.long, device=dev)
    node_of[patch_of, local] = order
    valid = node_of >= 0
    xyz = torch.zeros((npatch, width, 3), dtype=dt, device=dev)
    xyz[valid] = points[node_of[valid]]
    cnt = counts.to(dt)[:, None]
    center = xyz.sum(1) / cnt
    off = (xyz - center[:, None]) * valid[..., None]
    rms = torch.sqrt((off ** 2).sum((1, 2)) / counts.to(dt)).clamp_min(1e-300)
    u = off / rms[:, None, None]
    quad = [(e, f) for e in range(3) for f in range(e, 3)] if degree == 2 else []
    R = 12 + 3 * len(quad)
    H = torch.zeros((npatch, width, 3, R), dtype=dt, device=dev)
    for d in range(3):
        H[:, :, d, d] = valid.to(dt)                       # translations
    for d in range(3):
        for e in range(3):                                  # full displacement gradients e_d x_e (9)
            H[:, :, d, 3 + 3 * d + e] = u[:, :, e]
    for d in range(3):
        for k, (e, f) in enumerate(quad):                   # quadratic fields e_d x_e x_f (18)
            H[:, :, d, 12 + 6 * d + k] = u[:, :, e] * u[:, :, f] * valid.to(dt)
    H = H.reshape(npatch, 3 * width, R)
    # Same span as an SVD of H: eigen-decomposition of the 12x12 Gram, then one re-orthonormalization.
    lam, V = torch.linalg.eigh(H.transpose(1, 2) @ H)
    lam, V = lam.flip(-1), V.flip(-1)                      # descending, leading columns kept
    keep = lam > 1e-12 * lam[:, :1]  # eigenvalues of the Gram are only accurate to ~1e-16 lam_max
    U = H @ (V / torch.sqrt(lam.clamp_min(1e-300))[:, None, :]) * keep[:, None, :]
    G = U.transpose(1, 2) @ U + torch.diag_embed((~keep).to(dt))
    L = torch.linalg.cholesky(G)
    U = torch.linalg.solve_triangular(L, U.transpose(1, 2), upper=False).transpose(1, 2) * keep[:, None, :]
    rank = keep.sum(1)
    assert bool((keep == (torch.arange(keep.shape[1], device=dev)[None] < rank[:, None])).all()), 'kept columns must be a prefix'
    return U, keep, node_of, rank


def prolongation(points, owner, n_nodes, degree=1, extra=None, extra_k=0, extra_tol=1e-6):
    U, keep, node_of, rank = affine_basis(points, owner, degree=degree)
    if extra is not None and extra_k > 0:
        # route 1 oracle: add, per patch, the leading directions of the block restrictions of the given fine vectors
        # after removing their polynomial part; kept columns are packed to the left so keep stays a prefix
        npatch_, rows3_, R0 = U.shape
        width_ = node_of.shape[1]
        dof_ = (3 * node_of[:, :, None].clamp_min(0) + torch.arange(3, device=dev)).reshape(npatch_, rows3_)
        valid_ = (node_of[:, :, None] >= 0).expand(-1, -1, 3).reshape(npatch_, rows3_)
        S = extra[dof_] * valid_[..., None]                       # npatch x rows3 x m
        Uk = U * keep[:, None, :]
        S = S - Uk @ (Uk.transpose(1, 2) @ S)
        Us, sv, _ = torch.linalg.svd(S, full_matrices=False)
        k = min(extra_k, Us.shape[2])
        Us, sv = Us[:, :, :k], sv[:, :k]
        smax = float(sv.max()) if sv.numel() else 0.0
        ekeep = (sv > extra_tol * sv[:, :1]) & (sv > 1e-12 * smax)
        Uall = torch.cat([U, Us], dim=2)
        kall = torch.cat([keep, ekeep], dim=1)
        order_ = torch.argsort((~kall).to(torch.int8), dim=1, stable=True)
        U = torch.gather(Uall, 2, order_[:, None, :].expand(-1, rows3_, -1))
        keep = torch.gather(kall, 1, order_)
        rank = keep.sum(1)
        assert bool((keep == (torch.arange(keep.shape[1], device=dev)[None] < rank[:, None])).all())
    npatch, rows3, R = U.shape
    width = node_of.shape[1]
    colstart = torch.cumsum(rank, 0) - rank
    dof = (3 * node_of[:, :, None] + torch.arange(3, device=dev)).reshape(npatch, 3 * width)
    rowvalid = (node_of[:, :, None] >= 0).expand(-1, -1, 3).reshape(npatch, 3 * width)
    # columns kept per patch are the leading singular vectors
    colidx = colstart[:, None] + torch.arange(R, device=dev)[None]
    m = rowvalid[:, :, None] & keep[:, None, :]
    r = dof[:, :, None].expand(-1, -1, R)[m]; c = colidx[:, None, :].expand(-1, 3 * width, -1)[m]; v = U[m]
    ncol = int(rank.sum())
    P = coo(r, c, v, (3 * n_nodes, ncol))
    # block index lists of each patch's columns (for level-1 block Jacobi)
    bidx = torch.where(keep, colidx, torch.full_like(colidx, -1))
    # node-level rows of the patch basis (3 x 12, dropped columns zeroed) and the owning patch of each node
    Ukeep = U * keep[:, None, :]
    Unode = torch.zeros((n_nodes, 3, R), dtype=dt, device=dev)
    vn = node_of >= 0
    Unode[node_of[vn]] = Ukeep.reshape(npatch, width, 3, R)[vn]
    patch_node = torch.full((n_nodes,), -1, dtype=torch.long, device=dev)
    patch_node[node_of[vn]] = torch.arange(npatch, device=dev)[:, None].expand(-1, width)[vn]
    return P, bidx, rank, dict(Unode=Unode, patch=patch_node, colstart=colstart, rank=rank)


def slow_modes(Aop, Bop, n, m, rounds=4, degree=20, a_cut=0.05, seed=7):
    """m approximate slowest eigenpairs of B A (A-self-adjoint, spectrum in (0, 1]): Chebyshev filter that is bounded
    on [a_cut, 1] and grows below it, then A-orthonormalization and Rayleigh-Ritz (V^T A B A V y = theta V^T A V y)."""
    g = torch.Generator(device=dev).manual_seed(seed)
    V = torch.randn((n, m), dtype=dt, device=dev, generator=g)
    c, e = (1 + a_cut) / 2, (1 - a_cut) / 2
    theta = None
    for _ in range(rounds):
        Y0 = V; Y1 = (Bop(Aop(V)) - c * V) / e
        for _k in range(2, degree + 1):
            Y2 = 2 * (Bop(Aop(Y1)) - c * Y1) / e - Y0
            sc = Y2.norm(dim=0, keepdim=True).clamp_min(1e-300)
            Y0, Y1 = Y1 / sc, Y2 / sc
        V = -Y1 if False else Y1
        W = Aop(V); H2 = V.T @ W; H2 = (H2 + H2.T) / 2
        lam, Q = torch.linalg.eigh(H2)
        ok = lam > 1e-12 * lam.max()
        T = Q[:, ok] / torch.sqrt(lam[ok])[None]
        Vn, Wn = V @ T, W @ T                                      # A-orthonormal
        H1 = Wn.T @ Bop(Wn); H1 = (H1 + H1.T) / 2
        theta, Yv = torch.linalg.eigh(H1)
        V = Vn @ Yv
    return theta, V


def galerkin(A, info, chunk=1 << 18):
    """P^T A P for a patch-block-diagonal P, accumulated as 12x12 blocks per coupled patch pair."""
    Unode, patch, colstart, rank = info['Unode'], info['patch'], info['colstart'], info['rank']
    npatch = len(rank)
    i, v = A.indices(), A.values()
    ni, ci, nj, cj = i[0] // 3, i[0] % 3, i[1] // 3, i[1] % 3
    key = patch[ni] * npatch + patch[nj]
    ukeys, inv = torch.unique(key, return_inverse=True)
    R = Unode.shape[-1]
    blocks = torch.zeros((len(ukeys), R, R), dtype=dt, device=dev)
    for lo in range(0, len(v), chunk):
        sl = slice(lo, lo + chunk)
        contrib = Unode[ni[sl], ci[sl]][:, :, None] * Unode[nj[sl], cj[sl]][:, None, :] * v[sl][:, None, None]
        blocks.index_add_(0, inv[sl], contrib)
    pp, qq = ukeys // npatch, ukeys % npatch
    a = torch.arange(R, device=dev)
    m = (a[None, :, None] < rank[pp][:, None, None]) & (a[None, None, :] < rank[qq][:, None, None])
    rows = (colstart[pp][:, None, None] + a[None, :, None]).expand(-1, R, R)[m]
    cols = (colstart[qq][:, None, None] + a[None, None, :]).expand(-1, R, R)[m]
    n1 = int(rank.sum())
    return coo(rows, cols, blocks[m], (n1, n1))


def block_factors(Adense_blocks):
    L = torch.linalg.cholesky(Adense_blocks)
    eye = torch.eye(L.shape[-1], dtype=dt, device=dev).expand_as(L)
    return torch.linalg.solve_triangular(L.transpose(-1, -2), eye, upper=True)  # upper, F F^T = block^-1


def gather_blocks(Acoo, bidx):
    """Dense diagonal blocks A[b][:, b] for padded index lists, read from the sparse entries directly
    (padding rows/cols get an identity diagonal so the batched Cholesky is defined; their factors are zeroed)."""
    nblk, w = bidx.shape
    valid = bidx >= 0
    n = Acoo.shape[0]
    blk_of = torch.full((n,), -1, dtype=torch.long, device=dev); loc_of = torch.full((n,), -1, dtype=torch.long, device=dev)
    b_ids = torch.arange(nblk, device=dev)[:, None].expand(-1, w)
    l_ids = torch.arange(w, device=dev)[None].expand(nblk, -1)
    blk_of[bidx[valid]] = b_ids[valid]; loc_of[bidx[valid]] = l_ids[valid]
    i, v = Acoo.indices(), Acoo.values()
    same = blk_of[i[0]] == blk_of[i[1]]
    B = torch.zeros((nblk, w, w), dtype=dt, device=dev)
    B[blk_of[i[0][same]], loc_of[i[0][same]], loc_of[i[1][same]]] = v[same]
    B = B + torch.diag_embed((~valid).to(dt))
    return B, valid


def rowsum_bound(Acoo, bidx, F):
    """max row sum of |F_g^T A F_g| with F_g the global block-diagonal factor."""
    nblk, w = bidx.shape
    valid = bidx >= 0
    r = bidx[:, :, None].expand(-1, -1, w); c = bidx[:, None, :].expand(-1, w, -1)
    m = valid[:, :, None] & valid[:, None, :]
    Fg = coo(r[m], c[m], F[m], Acoo.shape)
    S = spmm(spmm(Fg.t().coalesce(), Acoo), Fg)
    S = S.coalesce()
    return float(torch.zeros(Acoo.shape[0], dtype=dt, device=dev).index_add_(0, S.indices()[0], S.values().abs()).max())


def pair_fine_blocks_gpu(A, theta, max_nodes=4):
    """GPU version of cross_case_chebyshev.pair_blocks: node-block strength, strong-edge clusters (label propagation),
    exact inverse-Cholesky factors of each cluster block, row-sum bound of F^T A F."""
    i, v = A.indices(), A.values()
    nn = A.shape[0] // 3
    ni, nj = i[0] // 3, i[1] // 3
    key = ni * nn + nj
    uk, inv = torch.unique(key, return_inverse=True)
    norm2 = torch.zeros(len(uk), dtype=dt, device=dev).index_add_(0, inv, v * v)
    bi, bj = uk // nn, uk % nn
    diag = torch.zeros(nn, dtype=dt, device=dev)
    on = bi == bj
    diag[bi[on]] = torch.sqrt(norm2[on])
    strength = torch.sqrt(norm2) / torch.sqrt(diag[bi] * diag[bj])
    strong = (bi != bj) & (strength >= theta)
    ei, ej = bi[strong], bj[strong]
    label = torch.arange(nn, device=dev)
    for _ in range(64):
        new = label.clone()
        new.scatter_reduce_(0, ei, label[ej], reduce='amin'); new.scatter_reduce_(0, ej, label[ei], reduce='amin')
        if torch.equal(new, label):
            break
        label = new
    _, cl = torch.unique(label, return_inverse=True)
    sizes = torch.bincount(cl)
    if int(sizes.max()) > max_nodes:
        raise ValueError(f'STRONG_CLUSTER_TOO_LARGE:{int(sizes.max())}')
    order = torch.argsort(cl, stable=True)
    start = torch.cumsum(sizes, 0) - sizes
    pos = torch.arange(nn, device=dev) - torch.repeat_interleave(start, sizes)
    nb_, w = len(sizes), 3 * int(sizes.max())
    bidx = torch.full((nb_, w), -1, dtype=torch.long, device=dev)
    nodes_sorted = order
    for d in range(3):
        bidx[cl[nodes_sorted], 3 * pos + d] = 3 * nodes_sorted + d
    # sort each row's real dofs ascending, keep right padding
    big = torch.where(bidx < 0, torch.full_like(bidx, 1 << 62), bidx)
    bidx = torch.sort(big, dim=1).values
    bidx = torch.where(bidx == (1 << 62), torch.full_like(bidx, -1), bidx)
    B, valid = gather_blocks(A, bidx)
    F = block_factors(B)
    F = torch.where(valid[:, :, None] & valid[:, None, :], F, torch.zeros_like(F))
    bound = rowsum_bound(A, bidx, F)
    rec = dict(theta=theta, blocks=int(nb_), nodes=int(nn), size_histogram=torch.bincount(sizes).tolist(),
               merged_nodes=int((sizes[cl] > 1).sum()), row_sum_bound=bound, omega0=1 / bound, backend='gpu')
    return bidx, F, bound, rec


def pair_fine_blocks(A, theta, max_nodes=4):
    import cross_case_chebyshev as X
    Ac = sparse.csr_matrix((A.values().cpu().numpy(), (A.indices()[0].cpu().numpy(), A.indices()[1].cpu().numpy())), shape=A.shape)
    idx, F, bound, rec = X.pair_blocks(Ac, theta, max_nodes)
    return torch.as_tensor(idx, device=dev), torch.tensor(F, dtype=dt, device=dev), bound, rec


class DirectModel(torch.nn.Module):
    """Boundary operator S q = D q + C^T z, z = -A^{-1} C q, with A factored once on the GPU (cuDSS sparse Cholesky of
    the Jacobi-scaled upper triangle). Same interface as the frozen MechanicsNetwork for the frozen evaluators.
    cuDSS (nvmath DirectSolver) fixes the right-hand-side width at planning; panels are solved in blocks of width w."""
    def __init__(self, A, C, D, Ub, Ui, T, width=8, threads=16, reorder='DEFAULT'):
        super().__init__()
        import os
        from nvmath.sparse.advanced import (DirectSolver, DirectSolverOptions, DirectSolverMatrixType,
                                            DirectSolverMatrixViewType, DirectSolverReorderingAlg)
        self.A, self.C, self.D, self.Ub, self.Ui = A, C, D, Ub, Ui
        self.Ct = C.t().coalesce(); self.width = width
        with T('E4a_direct_scaled_upper'):
            i, v = A.indices(), A.values()
            dg = torch.zeros(A.shape[0], dtype=dt, device=dev).index_add_(0, i[0][i[0] == i[1]], v[i[0] == i[1]])
            self.s = 1 / torch.sqrt(dg)
            up = i[0] <= i[1]
            U = torch.sparse_coo_tensor(i[:, up], v[up] * self.s[i[0][up]] * self.s[i[1][up]], A.shape).coalesce().to_sparse_csr()
            self.U = torch.sparse_csr_tensor(U.crow_indices().int(), U.col_indices().int(), U.values(), size=A.shape)
        lib = os.environ.get('CUDSS_MT')
        opts = DirectSolverOptions(sparse_system_type=DirectSolverMatrixType.SPD, sparse_system_view=DirectSolverMatrixViewType.UPPER,
                                   **(dict(multithreading_lib=lib) if lib else {}))
        self.b = torch.zeros((width, A.shape[0]), dtype=dt, device=dev).T  # column-major panel
        self.solver = DirectSolver(self.U, self.b, options=opts)
        if lib:
            self.solver.plan_config.host_nthreads = threads
        self.solver.plan_config.reordering_algorithm = getattr(DirectSolverReorderingAlg, reorder)
        # cuDSS allocates with its own cudaMalloc: hand back the blocks PyTorch's caching allocator still holds from assembly
        torch.cuda.empty_cache()
        self.device_free_before_plan_gb = torch.cuda.mem_get_info()[0] / 2**30
        with T('E4b_direct_plan'):
            self.solver.plan()
        with T('E4c_direct_factor'):
            self.solver.factorize()

    def solve(self, rhs):
        return _Solve.apply(rhs, self)

    def solve_raw(self, rhs):
        out = torch.empty_like(rhs)
        for c0 in range(0, rhs.shape[1], self.width):
            k = min(self.width, rhs.shape[1] - c0)
            b = torch.zeros((self.width, rhs.shape[0]), dtype=dt, device=dev).T
            b[:, :k] = self.s[:, None] * rhs[:, c0:c0 + k]
            self.solver.reset_operands(b=b)
            out[:, c0:c0 + k] = self.s[:, None] * self.solver.solve()[:, :k]
        return out

    @staticmethod
    def panel(f):
        def g(self, x):
            if x.ndim == 1:
                return f(self, x[:, None])[..., 0]
            return f(self, x)
        return g

    def mm(self, name, x):
        """Sparse product whose backward uses the stored transpose (A, D symmetric; C^T kept explicitly), so the
        energy-gradient audit never has torch build a transposed copy of a large sparse block."""
        M, Mt = {'A': (self.A, self.A), 'D': (self.D, self.D), 'C': (self.C, self.Ct), 'Ct': (self.Ct, self.C)}[name]
        return _SpMM.apply(x, M, Mt)

    def extension(self, q):
        return DirectModel.panel(lambda m, x: -m.solve(m.mm('C', x)))(self, q)

    def extension_adjoint(self, w):
        return DirectModel.panel(lambda m, x: -m.mm('Ct', m.solve(x)))(self, w)

    def forward(self, q):
        return DirectModel.panel(lambda m, x: m.mm('D', x) + m.mm('Ct', m.extension(x)))(self, q)

    def energy(self, q):
        def e(m, x):
            z = m.extension(x)
            return (0.5 * ((x * m.mm('D', x)).sum(0) + 2 * (z * m.mm('C', x)).sum(0) + (z * m.mm('A', z)).sum(0)))[None]
        return DirectModel.panel(e)(self, q)[..., 0] if q.ndim == 1 else e(self, q)[0]


class _SpMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, M, Mt):
        ctx.Mt = Mt
        return torch.sparse.mm(M, x.detach())

    @staticmethod
    def backward(ctx, g):
        return torch.sparse.mm(ctx.Mt, g.contiguous()), None, None


class _Solve(torch.autograd.Function):
    """A^{-1} rhs through the factor; A is symmetric, so the backward pass is the same solve."""
    @staticmethod
    def forward(ctx, rhs, model):
        ctx.model = model
        return model.solve_raw(rhs.detach())

    @staticmethod
    def backward(ctx, g):
        from cuda.core import Device  # autograd runs backward on its own thread: make the device current there
        Device(dev.index or 0).set_current()
        return ctx.model.solve_raw(g.contiguous()), None


def main(a):
    global CPU_TRACE, COVER_INPUTS, GPU_EXPAND, TAU_EPS
    TAU_EPS = a.tau_eps
    CPU_TRACE = a.cpu_trace
    GPU_EXPAND = a.gpu_trace
    COVER_INPUTS = Path(a.cover_inputs) if a.cover_inputs else None
    T = Timer()
    out = Path(a.output); out.mkdir(parents=True, exist_ok=False)
    torch.cuda.synchronize(); t_all = time.perf_counter()
    if a.elements == 'polyref':
        A, C, D = blocks_from_polyref(a.case, a.s, a.rule_order, T, a.levels)
    else:
        with T('E1_teacher_blocks_load'):
            A, C, D = blocks_teacher(a.case, a.asset_root)
    comp = COVER_INPUTS if COVER_INPUTS is not None else (MN / 'assets' / a.case if a.asset_root is None else Path(a.asset_root) / a.case) / 'compiled'
    points = torch.tensor(np.load(comp / 'INTERIOR_POINTS.npy'), dtype=dt, device=dev)
    nn_ = len(points)
    if a.solver == 'cudss':
        import cross_case_chebyshev as X
        sys.path.insert(0, str(HS / 'source_chebyshev_full_01')); sys.path.insert(0, str(MN / 'source_v1'))
        runner = X.load('xc_run_mechanics', X.FROZEN['run_mechanics'])
        evaluator = X.load('xc_evaluate_mechanics', X.FROZEN['evaluate_mechanics'])
        evalcheb = X.load('xc_evaluate_chebyshev_strain', X.FROZEN['evaluate_chebyshev_strain'])
        tt = lambda x: torch.as_tensor(x, dtype=dt, device=dev)
        ub, ui = runner.orthonormalize_rigid_pair(tt(np.load(comp / 'Q_RIGID.npy')), tt(np.load(comp / 'INTERIOR_RIGID.npy')))
        model = DirectModel(A, C, D, ub, ui, T, width=a.direct_width, threads=a.direct_threads, reorder=a.direct_reorder)
        encode_seconds = time.perf_counter() - t_all
        if a.observables:
            observables(a, out, model, comp, T, encode_seconds)
            return
        if a.encode_only:  # deployment timing: the operator is query-ready here; time queries, no evaluation
            query = {}
            with torch.no_grad():
                for cols in (1, 7, 64):
                    q = torch.randn((D.shape[0], cols), dtype=dt, device=dev)
                    model(q); torch.cuda.synchronize(); t = time.perf_counter(); model(q); torch.cuda.synchronize()
                    query[cols] = time.perf_counter() - t
            (out / 'ENCODE.json').write_text(json.dumps(dict(case=a.case, encode_seconds=encode_seconds, stage_seconds=T.rows,
                query_seconds=query, torch_cuda_peak_gib=torch.cuda.max_memory_allocated() / 2**30,
                interior=A.shape[0], boundary=D.shape[0]), indent=2))
            print(json.dumps(dict(event='ENCODED', encode=round(encode_seconds, 2), query={k: round(v, 4) for k, v in query.items()})), flush=True)
            return
        finish(a, out, T, comp, model, A, D, encode_seconds, runner, evaluator, evalcheb,
               dict(solver='cudss', width=a.direct_width, reorder=a.direct_reorder, dimensions=[A.shape[0]]))
        if a.full_spectrum:
            full_spectrum(a, out, model, runner)
        return
    with T('E2a_patches_basis'):
        # node spacing of the Q2 background grid is 1/(2n) (computing it from the interior points fails when a
        # tiny retained region has no two distinct interior coordinates along some axis)
        h = 1.0 / (2 * int(json.loads((ROOT / 'packets' / a.case / 'FRESH_CONTEXT.json').read_text())['n']))
        def owner_of(span):
            b = torch.floor((points + 1e-10) / (span * h)).long(); b = b - b.min(0).values
            K = int(b.max()) + 1
            return torch.unique(b[:, 0] * K * K + b[:, 1] * K + b[:, 2], return_inverse=True)[1]
        o4, o8 = owner_of(4), owner_of(8)
        P1, bidx1, rank1, info1 = prolongation(points, o4, nn_, degree=a.degree1)
        P2, _, rank2, _ = prolongation(points, o8, nn_, degree=a.degree2)
    with T('E2b_galerkin'):
        P1t = P1.t().coalesce()
        P12 = spmm(P1t, P2).coalesce()
        A1 = galerkin(A, info1)
        A1 = ((A1 + A1.t()) * 0.5).coalesce()
        A2d = (P12.t() @ (A1 @ P12.to_dense()))
        A2d = (A2d + A2d.T) / 2
    with T('E2c_level1_blocks'):
        B1, valid1 = gather_blocks(A1, bidx1)
        F1 = block_factors(B1)
        F1 = torch.where((valid1[:, :, None] & valid1[:, None, :]), F1, torch.zeros_like(F1))
        bound1 = rowsum_bound(A1, bidx1, F1)
    with T('E2d_coarsest'):
        L2 = torch.linalg.cholesky(A2d)
        F2 = torch.linalg.solve_triangular(L2.T, torch.eye(len(A2d), dtype=dt, device=dev), upper=True)
    with T('E2e_fine_pairs'):
        fidx, fF, bound0, prec = (pair_fine_blocks_gpu if a.gpu_pairs else pair_fine_blocks)(A, a.pair_theta)
    # model: Codex's frozen core + my hierarchy correction + frozen Chebyshev recurrence
    import cross_case_chebyshev as X
    sys.path.insert(0, str(HS / 'source_chebyshev_full_01')); sys.path.insert(0, str(MN / 'source_v1'))
    runner = X.load('xc_run_mechanics', X.FROZEN['run_mechanics'])
    cheb = X.load('xc_chebyshev_strain_network', X.FROZEN['chebyshev_strain_network'])
    evaluator = X.load('xc_evaluate_mechanics', X.FROZEN['evaluate_mechanics'])
    evalcheb = X.load('xc_evaluate_chebyshev_strain', X.FROZEN['evaluate_chebyshev_strain'])
    hier = X.load('xc_hierarchy_strain_network_r1', HERE / 'hierarchy_strain_network_r1.py')
    with T('E2f_model_assembly'):
        tt = lambda x: torch.as_tensor(x, dtype=dt, device=dev)
        ub, ui = runner.orthonormalize_rigid_pair(tt(np.load(comp / 'Q_RIGID.npy')), tt(np.load(comp / 'INTERIOR_RIGID.npy')))
        core = hier.load_frozen_core(X.FROZEN['mechanics_network'], X.sha(X.FROZEN['mechanics_network']))
        corr = hier.HierarchyCorrection(A, P1, A1, P12, A2d, fidx, fF, bidx1, F1, F2, 1 / bound0, 1 / bound1, mode='vcycle')
        # the frozen core only needs a local 3x3 factor argument for its unused fine path
        local = torch.eye(3, dtype=dt, device=dev).expand(nn_, 3, 3).contiguous()
        base = core.MechanicsNetwork(A, C, D, ub, ui, local, (), layers=1, step_scale=1.0, learnable=False, share_layers=True)
        base = hier.install_hierarchy(base, corr, cycles=1)
    with T('E3_lanczos_design'):
        rv = X.lanczos_lower(lambda v: torch.sparse.mm(A, v), base.corrections[0], A.shape[0], dev, a.lanczos)
        a_design = float(a.safety * rv.min())
    slow = None
    if a.slow_m > 0:
        # route 1 oracle: the slowest modes of the exact V-cycle, restricted block by block, enrich the level-1 space
        with T('E4a_slow_modes'):
            theta, Vs = slow_modes(lambda v: torch.sparse.mm(A, v), base.corrections[0], A.shape[0], a.slow_m,
                                   rounds=a.slow_rounds, degree=a.slow_degree, a_cut=a.slow_cut)
        slow = dict(m=a.slow_m, k=a.slow_k, rounds=a.slow_rounds, degree=a.slow_degree, a_cut=a.slow_cut,
                    theta_first=theta[:8].tolist(), theta_last=float(theta[-1]), below_cut=int((theta < a.slow_cut).sum()),
                    ritz_min_before=float(rv.min()))
        with T('E4b_rebuild_level1'):
            P1, bidx1, rank1, info1 = prolongation(points, o4, nn_, degree=a.degree1, extra=Vs, extra_k=a.slow_k)
            P1t = P1.t().coalesce()
            P12 = spmm(P1t, P2).coalesce()
            A1 = galerkin(A, info1)
            A1 = ((A1 + A1.t()) * 0.5).coalesce()
            A2d = (P12.t() @ (A1 @ P12.to_dense()))
            A2d = (A2d + A2d.T) / 2
            B1, valid1 = gather_blocks(A1, bidx1)
            F1 = block_factors(B1)
            F1 = torch.where((valid1[:, :, None] & valid1[:, None, :]), F1, torch.zeros_like(F1))
            bound1 = rowsum_bound(A1, bidx1, F1)
            L2 = torch.linalg.cholesky(A2d)
            F2 = torch.linalg.solve_triangular(L2.T, torch.eye(len(A2d), dtype=dt, device=dev), upper=True)
            corr = hier.HierarchyCorrection(A, P1, A1, P12, A2d, fidx, fF, bidx1, F1, F2, 1 / bound0, 1 / bound1, mode='vcycle')
            base = core.MechanicsNetwork(A, C, D, ub, ui, local, (), layers=1, step_scale=1.0, learnable=False, share_layers=True)
            base = hier.install_hierarchy(base, corr, cycles=1)
        with T('E4c_lanczos_design'):
            rv = X.lanczos_lower(lambda v: torch.sparse.mm(A, v), base.corrections[0], A.shape[0], dev, a.lanczos)
            a_design = float(a.safety * rv.min())
        del Vs
    encode_seconds = time.perf_counter() - t_all
    extra = dict(solver='chebyshev', s=a.s, levels=a.levels, rule_order=a.rule_order, pair_theta=a.pair_theta,
                 degree1=a.degree1, degree2=a.degree2,
                 dimensions=[A.shape[0], P1.shape[1], P2.shape[1]], level1_rank=dict(min=int(rank1.min()), max=int(rank1.max())),
                 pair_record=prec, ritz_min=float(rv.min()), design_a=a_design, slow_mode_oracle=slow)
    return evaluate_depths(a, out, T, comp, base, cheb, X, a_design, A, D, encode_seconds, runner, evaluator, evalcheb, extra)


def evaluate_depths(a, out, T, comp, base, cheb, X, a_design, A, D, encode_seconds, runner, evaluator, evalcheb, extra):
    """Route 1 E0/E1a: the same hierarchy at several fixed Chebyshev depths; for each depth the worst-direction
    witness in the teacher-whitened frame (the model is an upper bound, mu >= 1, so the largest Rayleigh ratio is
    the error), the seven original probes, and query timing."""
    tt = lambda x: torch.as_tensor(x, dtype=dt, device=dev)
    R = tt(np.load(ROOT / 'targets' / (a.case + '_v1') / 'REFERENCE_RQ.npy'))
    quotient = runner.Quotient(np.load(ROOT / 'targets' / (a.case + '_v1') / 'RIGID_Q.npy'), dev)
    probes = np.load(comp / 'QUALIFICATION_PROBES.npz')
    probe_t = {k: tt(probes[key]) for k, key in [('q', 'q'), ('zref', 'z'), ('reference_force', 'reference_force')]}
    rows = []
    for depth in [int(x) for x in a.depths.split(',')]:
        model = cheb.ChebyshevStrainNetwork(base, cycles=depth)
        ca, cb = X.chebyshev_coefficients_interval(depth, a_design)
        model.chebyshev_alpha.copy_(torch.tensor(ca, dtype=dt)); model.chebyshev_beta.copy_(torch.tensor(cb, dtype=dt))
        query = {}
        with torch.no_grad():
            for cols in (1, 7):
                q = torch.randn((D.shape[0], cols), dtype=dt, device=dev)
                model(q); torch.cuda.synchronize(); t = time.perf_counter(); model(q); torch.cuda.synchronize()
                query[cols] = time.perf_counter() - t
        algebra = evalcheb.complete_algebra(model, probe_t, evaluator)
        witness, white = evaluator.spectral_witness(model, R, quotient.lift, quotient.reduce, seed=92219, maxiter=20, return_vector=True)
        replay, _ = evalcheb.replay_witness(model, R, quotient, white)
        row = dict(depth=depth, witness=replay['actual_rayleigh'], query_seconds=query,
                   original7_energy_ratio=(np.asarray(algebra['predicted_energy']) / np.asarray(algebra['reference_energy'])).tolist(),
                   numerical_checks=algebra['numerical_checks'])
        rows.append(row)
        np.save(out / f'WITNESS_WHITE_{depth:03d}.npy', white.cpu().numpy())
        print(json.dumps(dict(event='DEPTH', depth=depth, witness=row['witness'][0], query=query)), flush=True)
    res = dict(case=a.case, elements=a.elements, **extra, stage_seconds=T.rows, encode_seconds=encode_seconds, depths=rows,
               scope='finite witness only (largest Rayleigh ratio, 20 iterations); teacher R whitening; elements=' + a.elements)
    (out / 'RESULT.json').write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(dict(event='DONE', ritz_min=extra['ritz_min'], dims=extra['dimensions'],
                          witness={r['depth']: r['witness'][0] for r in rows})), flush=True)


def observables(a, out, model, comp, T, encode_seconds):
    """Design-sensitivity observables of the complete boundary operator S (dense, from the exact direct action),
    on the non-rigid complement N of the boundary rigid subspace (base Q_RIGID, identical across admitted eps):
    log det, trace of the inverse, softest and stiffest eigenvalue, and the compliance g_k^T (N^T S N)^-1 g_k of
    eight fixed loads (seeded, in the complement coordinates, identical across eps and across teacher / geometry)."""
    from scipy import linalg
    from threadpoolctl import threadpool_limits
    m = model.D.shape[0]
    S = np.empty((m, m), dtype=np.float64, order='F')
    t0 = time.perf_counter()
    with torch.no_grad():
        for lo in range(0, m, a.spectrum_batch):
            hi = min(m, lo + a.spectrum_batch)
            e = torch.zeros((m, hi - lo), dtype=dt, device=dev)
            e[torch.arange(lo, hi, device=dev), torch.arange(hi - lo, device=dev)] = 1
            S[:, lo:hi] = model(e).cpu().numpy()
    action_seconds = time.perf_counter() - t0
    asym = float(np.linalg.norm(S - S.T) / np.linalg.norm(S))
    S = (S + S.T) / 2
    U = np.linalg.qr(np.load(comp / 'Q_RIGID.npy'))[0]
    with threadpool_limits(a.spectrum_threads):
        Qf = linalg.qr(U, mode='full')[0]
        N = Qf[:, 6:]
        Sn = N.T @ S @ N
        w = linalg.eigvalsh(Sn, check_finite=False, driver='evd')
        L = linalg.cho_factor(Sn, lower=True, check_finite=False)
        g = np.random.default_rng(20260923).standard_normal((m - 6, 8))
        comp_k = np.sum(g * linalg.cho_solve(L, g, check_finite=False), axis=0)
    rigid_leak = float(np.linalg.norm(S @ U) / np.linalg.norm(S))
    res = dict(case=a.case, tau_eps=a.tau_eps, elements=a.elements, s=a.s, levels=a.levels, m=m,
               asymmetry=asym, rigid_leak=rigid_leak, log_det=float(np.sum(np.log(w))),
               trace_inverse=float(np.sum(1 / w)), softest=float(w[0]), stiffest=float(w[-1]),
               compliance=comp_k.tolist(), encode_seconds=encode_seconds, stage_seconds=T.rows,
               action_seconds=action_seconds)
    (out / 'OBSERVABLES.json').write_text(json.dumps(res, indent=2))
    print(json.dumps(dict(event='OBSERVABLES', **{k: res[k] for k in ('m', 'asymmetry', 'rigid_leak', 'log_det', 'trace_inverse', 'softest')})), flush=True)


def full_spectrum(a, out, model, runner):
    """Complete two-sided boundary spectrum, same definition as the frozen full-spectrum protocol:
    G[:, j] = R^-T B S B^T R^-1 e_j from the model's complete boundary action, all d physical directions kept,
    symmetrized (asymmetry must be < 1e-10), every eigenvalue computed; D/d = mean(mu - 1 - ln mu).
    Here S is applied exactly (cuDSS factor), so the spectrum measures the element (geometry) error alone."""
    import resource
    from scipy import linalg
    from threadpoolctl import threadpool_limits
    tt = lambda x: torch.as_tensor(x, dtype=dt, device=dev)
    R = tt(np.load(ROOT / 'targets' / (a.case + '_v1') / 'REFERENCE_RQ.npy'))
    quotient = runner.Quotient(np.load(ROOT / 'targets' / (a.case + '_v1') / 'RIGID_Q.npy'), dev)
    d = R.shape[0]
    G = np.empty((d, d), dtype=np.float64, order='F')
    t0 = time.perf_counter()
    with torch.no_grad():
        for lo in range(0, d, a.spectrum_batch):
            hi = min(d, lo + a.spectrum_batch)
            eye = R.new_zeros((d, hi - lo))
            eye[torch.arange(lo, hi, device=dev), torch.arange(hi - lo, device=dev)] = 1
            q = quotient.lift(torch.linalg.solve_triangular(R, eye, upper=True))
            col = torch.linalg.solve_triangular(R.T, quotient.reduce(model(q)), upper=False)
            if not torch.isfinite(col).all():
                raise ValueError('NONFINITE_ACTION')
            G[:, lo:hi] = col.cpu().numpy()
    action_seconds = time.perf_counter() - t0
    del R, quotient
    torch.cuda.empty_cache()
    num = den = 0.0
    for lo in range(0, d, 1024):  # symmetrize in place, tile by tile
        hi = min(d, lo + 1024)
        for lo2 in range(lo, d, 1024):
            hi2 = min(d, lo2 + 1024)
            x, y = G[lo:hi, lo2:hi2].copy(), G[lo2:hi2, lo:hi].T.copy()
            w = 1.0 if lo2 == lo else 2.0
            num += w * float(np.square(x - y).sum()); den += w * float(np.square(x).sum())
            m = (x + y) / 2
            G[lo:hi, lo2:hi2] = m; G[lo2:hi2, lo:hi] = m.T
    asym = float(np.sqrt(num / den))
    if asym > 1e-10:
        raise ValueError('ASYMMETRY_EXCEEDS_1e-10:%g' % asym)
    t = time.perf_counter()
    with threadpool_limits(a.spectrum_threads):
        mu = linalg.eigvalsh(G, overwrite_a=True, check_finite=False, driver='evd')
    eig_seconds = time.perf_counter() - t
    del G
    np.save(out / 'ALL_STIFFNESS_EIGENVALUES.npy', mu)
    res = dict(status='FULL_SPECTRUM_FINISHED', case=a.case, elements=a.elements, s=a.s, levels=a.levels, solver='cudss',
               physical_dimension=d, mu_min=float(mu.min()), mu_max=float(mu.max()),
               outside_work_gate=int(np.sum((mu < .9) | (mu > 1.1))), outside_target_gate=int(np.sum((mu < .97) | (mu > 1.03))),
               work_gate=bool(mu.min() >= .9 and mu.max() <= 1.1),
               target_gate_spectrum=bool(mu.min() >= .97 and mu.max() <= 1.03),
               D_over_d=float(np.mean(mu - 1 - np.log(mu))), D_gate=1e-4, asymmetry=asym,
               action_seconds=action_seconds, eig_seconds=eig_seconds,
               peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
               cuda_peak_bytes=torch.cuda.max_memory_allocated(), modes_removed=0, jitter=False, clipping=False,
               pseudoinverse=False,
               definition='G[:,j] = R^-T B S B^T R^-1 e_j via the complete boundary action; S applied exactly (cuDSS)')
    res['target_gate'] = bool(res['target_gate_spectrum'] and res['D_over_d'] < 1e-4)
    (out / 'FULL_SPECTRUM.json').write_text(json.dumps(res, indent=2))
    print(json.dumps(dict(event='FULL_SPECTRUM', **{k: res[k] for k in ('mu_min', 'mu_max', 'D_over_d', 'outside_target_gate',
                                                                        'target_gate', 'asymmetry', 'action_seconds', 'eig_seconds')})), flush=True)


def finish(a, out, T, comp, model, A, D, encode_seconds, runner, evaluator, evalcheb, extra):
    tt = lambda x: torch.as_tensor(x, dtype=dt, device=dev)
    # query timing: q panels of 1, 7 and 64 boundary displacement columns
    R = tt(np.load(ROOT / 'targets' / (a.case + '_v1') / 'REFERENCE_RQ.npy'))
    quotient = runner.Quotient(np.load(ROOT / 'targets' / (a.case + '_v1') / 'RIGID_Q.npy'), dev)
    query = {}
    with torch.no_grad():
        for cols in ((1, 7) if (A.shape[0] > 300000 or a.small_query) and a.solver != 'cudss' else (1, 7, 64)):
            q = torch.randn((D.shape[0], cols), dtype=dt, device=dev)
            model(q); torch.cuda.synchronize(); t = time.perf_counter(); model(q); torch.cuda.synchronize()
            query[cols] = time.perf_counter() - t
    # finite witness against the teacher
    probes = np.load(comp / 'QUALIFICATION_PROBES.npz')
    probe_t = {k: tt(probes[key]) for k, key in [('q', 'q'), ('zref', 'z'), ('reference_force', 'reference_force')]}
    algebra = evalcheb.complete_algebra(model, probe_t, evaluator)
    witness, white = evaluator.spectral_witness(model, R, quotient.lift, quotient.reduce, seed=92219, maxiter=20, return_vector=True)
    replay, _ = evalcheb.replay_witness(model, R, quotient, white)
    res = dict(case=a.case, elements=a.elements, **extra,
               stage_seconds=T.rows, encode_seconds=encode_seconds, query_seconds=query,
               original7_energy_ratio=(np.asarray(algebra['predicted_energy']) / np.asarray(algebra['reference_energy'])).tolist(),
               numerical_checks=algebra['numerical_checks'], witness=replay['actual_rayleigh'],
               scope='finite witness only; teacher R whitening; elements=' + a.elements)
    (out / 'RESULT.json').write_text(json.dumps(res, indent=2, default=float))
    np.save(out / 'WITNESS_WHITE.npy', white.cpu().numpy())
    print(json.dumps(dict(event='DONE', solver=extra['solver'], encode=round(encode_seconds, 2), stages={k: round(v, 3) for k, v in T.rows.items()},
                          query={k: round(v, 4) for k, v in query.items()},
                          witness=replay['actual_rayleigh'][0], original7=res['original7_energy_ratio'])), flush=True)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', required=True); ap.add_argument('--output', required=True)
    ap.add_argument('--asset-root', default=None); ap.add_argument('--reference', default=None)
    ap.add_argument('--elements', choices=['teacher', 'polyref'], default='teacher')
    ap.add_argument('--s', type=int, default=4); ap.add_argument('--rule-order', type=int, default=4)
    ap.add_argument('--levels', type=int, default=0)
    ap.add_argument('--cpu-trace', action='store_true')
    ap.add_argument('--gpu-trace', action='store_true')
    ap.add_argument('--cover-inputs', default=None)
    ap.add_argument('--gpu-pairs', action='store_true')
    ap.add_argument('--small-query', action='store_true')
    ap.add_argument('--pair-theta', type=float, default=0.9)
    ap.add_argument('--lanczos', type=int, default=80); ap.add_argument('--safety', type=float, default=0.5)
    ap.add_argument('--tolerance', type=float, default=1e-6)
    ap.add_argument('--solver', choices=['chebyshev', 'cudss'], default='chebyshev')
    ap.add_argument('--direct-width', type=int, default=8); ap.add_argument('--direct-threads', type=int, default=16)
    ap.add_argument('--direct-reorder', default='DEFAULT')
    ap.add_argument('--full-spectrum', action='store_true')
    ap.add_argument('--observables', action='store_true'); ap.add_argument('--tau-eps', default='0')
    ap.add_argument('--encode-only', action='store_true')
    ap.add_argument('--spectrum-batch', type=int, default=256); ap.add_argument('--spectrum-threads', type=int, default=16)
    ap.add_argument('--degree1', type=int, choices=[1, 2], default=1); ap.add_argument('--degree2', type=int, choices=[1, 2], default=1)
    ap.add_argument('--depths', default='8,16,32,64')
    ap.add_argument('--slow-m', type=int, default=0); ap.add_argument('--slow-k', type=int, default=12)
    ap.add_argument('--slow-rounds', type=int, default=4); ap.add_argument('--slow-degree', type=int, default=20)
    ap.add_argument('--slow-cut', type=float, default=0.05)
    main(ap.parse_args())
