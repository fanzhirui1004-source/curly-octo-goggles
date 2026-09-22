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


def elements_polyref(case, s, order, T):
    import element_moments as EM, polyref_torch as PT
    ctx = json.loads((ROOT / 'packets' / case / 'FRESH_CONTEXT.json').read_text())
    n = int(ctx['n']); E = float(ctx['material']['E']); nu = float(ctx['material']['nu'])
    lam = E * nu / ((1 + nu) * (1 - 2 * nu)); mu_ = E / (2 * (1 + nu))
    taus = [float(v) for v in ctx['case']['tau_corners']]
    normal = None if ctx['case'].get('normal') is None else [float(v) for v in ctx['case']['normal']]
    offset = None if normal is None else float(ctx['case']['offset'])
    arr = EM.members(case, ['NODES.npy', 'dofs.npy', 'CELL_INDICES.npy'])
    xi = EM.local_coordinates(case, n, arr)
    keys = np.unique(xi.reshape(len(xi), -1), axis=0)
    if len(keys) != 1:
        raise ValueError('ONE_NODE_ORDERING_EXPECTED')
    Tm = torch.tensor(EM.pattern_operators(keys[0].reshape(27, 3), lam, mu_, n)[1], dtype=dt, device=dev)  # 125x81x81
    with T('E1a_moments'):
        M = torch.tensor(PT.cell_moments(arr['CELL_INDICES.npy'], n, taus, normal, offset, s, rule_order=order), dtype=dt, device=dev)
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


def blocks_from_polyref(case, s, order, T):
    Kb, nb, ctx = elements_polyref(case, s, order, T)
    G = ghost(case, nb)
    gamma = float(json.loads((ROOT / 'packets' / case / 'SAMPLE.json').read_text())['gp']['gamma'])
    P = sparse.load_npz(ROOT / 'packets' / case / 'ORIGINAL_FROM_TRACE_FREE.npz').tocsr()
    m = json.loads((ROOT / 'packets' / case / 'SAMPLE.json').read_text())['full_trace_dimension']
    with T('E1d_trace_compile_PtKP'):
        K = (Kb + gamma * G).coalesce()
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


def blocks_teacher(case, asset_root):
    comp = (MN / 'assets' / case if asset_root is None else Path(asset_root) / case) / 'compiled'
    out = []
    for name in ('A', 'C', 'D'):
        M = sparse.load_npz(comp / (name + '.npz')).tocoo()
        out.append(coo(torch.as_tensor(M.row.astype(np.int64), device=dev), torch.as_tensor(M.col.astype(np.int64), device=dev),
                       torch.as_tensor(M.data, dtype=dt, device=dev), M.shape))
    return out


def affine_basis(points, owner, tol=1e-10):
    """Per patch: orthonormal basis of the 12 affine displacement fields (padded batched SVD)."""
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
    H = torch.zeros((npatch, width, 3, 12), dtype=dt, device=dev)
    for d in range(3):
        H[:, :, d, d] = valid.to(dt)                       # translations
    for d in range(3):
        for e in range(3):                                  # full displacement gradients e_d x_e (9)
            H[:, :, d, 3 + 3 * d + e] = u[:, :, e]
    H = H.reshape(npatch, 3 * width, 12)
    U, S, _ = torch.linalg.svd(H, full_matrices=False)
    keep = S > tol * S[:, :1]
    rank = keep.sum(1)
    return U, keep, node_of, rank


def prolongation(points, owner, n_nodes):
    U, keep, node_of, rank = affine_basis(points, owner)
    npatch, rows3, _ = U.shape
    width = node_of.shape[1]
    colstart = torch.cumsum(rank, 0) - rank
    dof = (3 * node_of[:, :, None] + torch.arange(3, device=dev)).reshape(npatch, 3 * width)
    rowvalid = (node_of[:, :, None] >= 0).expand(-1, -1, 3).reshape(npatch, 3 * width)
    # columns kept per patch are the leading singular vectors
    colidx = colstart[:, None] + torch.arange(12, device=dev)[None]
    m = rowvalid[:, :, None] & keep[:, None, :]
    r = dof[:, :, None].expand(-1, -1, 12)[m]; c = colidx[:, None, :].expand(-1, 3 * width, -1)[m]; v = U[m]
    ncol = int(rank.sum())
    P = coo(r, c, v, (3 * n_nodes, ncol))
    # block index lists of each patch's columns (for level-1 block Jacobi)
    bidx = torch.where(keep, colidx, torch.full_like(colidx, -1))
    # node-level rows of the patch basis (3 x 12, dropped columns zeroed) and the owning patch of each node
    Ukeep = U * keep[:, None, :]
    Unode = torch.zeros((n_nodes, 3, 12), dtype=dt, device=dev)
    vn = node_of >= 0
    Unode[node_of[vn]] = Ukeep.reshape(npatch, width, 3, 12)[vn]
    patch_node = torch.full((n_nodes,), -1, dtype=torch.long, device=dev)
    patch_node[node_of[vn]] = torch.arange(npatch, device=dev)[:, None].expand(-1, width)[vn]
    return P, bidx, rank, dict(Unode=Unode, patch=patch_node, colstart=colstart, rank=rank)


def galerkin(A, info, chunk=1 << 20):
    """P^T A P for a patch-block-diagonal P, accumulated as 12x12 blocks per coupled patch pair."""
    Unode, patch, colstart, rank = info['Unode'], info['patch'], info['colstart'], info['rank']
    npatch = len(rank)
    i, v = A.indices(), A.values()
    ni, ci, nj, cj = i[0] // 3, i[0] % 3, i[1] // 3, i[1] % 3
    key = patch[ni] * npatch + patch[nj]
    ukeys, inv = torch.unique(key, return_inverse=True)
    blocks = torch.zeros((len(ukeys), 12, 12), dtype=dt, device=dev)
    for lo in range(0, len(v), chunk):
        sl = slice(lo, lo + chunk)
        contrib = Unode[ni[sl], ci[sl]][:, :, None] * Unode[nj[sl], cj[sl]][:, None, :] * v[sl][:, None, None]
        blocks.index_add_(0, inv[sl], contrib)
    pp, qq = ukeys // npatch, ukeys % npatch
    a = torch.arange(12, device=dev)
    m = (a[None, :, None] < rank[pp][:, None, None]) & (a[None, None, :] < rank[qq][:, None, None])
    rows = (colstart[pp][:, None, None] + a[None, :, None]).expand(-1, 12, 12)[m]
    cols = (colstart[qq][:, None, None] + a[None, None, :]).expand(-1, 12, 12)[m]
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


def pair_fine_blocks(A, theta, max_nodes=4):
    import cross_case_chebyshev as X
    Ac = sparse.csr_matrix((A.values().cpu().numpy(), (A.indices()[0].cpu().numpy(), A.indices()[1].cpu().numpy())), shape=A.shape)
    idx, F, bound, rec = X.pair_blocks(Ac, theta, max_nodes)
    return torch.as_tensor(idx, device=dev), torch.tensor(F, dtype=dt, device=dev), bound, rec


def main(a):
    T = Timer()
    out = Path(a.output); out.mkdir(parents=True, exist_ok=False)
    torch.cuda.synchronize(); t_all = time.perf_counter()
    if a.elements == 'polyref':
        A, C, D = blocks_from_polyref(a.case, a.s, a.rule_order, T)
    else:
        with T('E1_teacher_blocks_load'):
            A, C, D = blocks_teacher(a.case, a.asset_root)
    comp = (MN / 'assets' / a.case if a.asset_root is None else Path(a.asset_root) / a.case) / 'compiled'
    points = torch.tensor(np.load(comp / 'INTERIOR_POINTS.npy'), dtype=dt, device=dev)
    nn_ = len(points)
    with T('E2a_patches_basis'):
        spacing = [torch.diff(torch.unique(points[:, i])) for i in range(3)]
        h = min(float(sd[sd > 1e-12].min()) for sd in spacing)
        o4 = torch.unique(torch.floor((points + 1e-10) / (4 * h)).long(), dim=0, return_inverse=True)[1]
        o8 = torch.unique(torch.floor((points + 1e-10) / (8 * h)).long(), dim=0, return_inverse=True)[1]
        P1, bidx1, rank1, info1 = prolongation(points, o4, nn_)
        P2, _, rank2, _ = prolongation(points, o8, nn_)
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
        fidx, fF, bound0, prec = pair_fine_blocks(A, a.pair_theta)
    # model: Codex's frozen core + my hierarchy correction + frozen Chebyshev recurrence
    import cross_case_chebyshev as X
    sys.path.insert(0, str(HS / 'source_chebyshev_full_01')); sys.path.insert(0, str(MN / 'source_v1'))
    runner = X.load('xc_run_mechanics', X.FROZEN['run_mechanics'])
    cheb = X.load('xc_chebyshev_strain_network', X.FROZEN['chebyshev_strain_network'])
    evaluator = X.load('xc_evaluate_mechanics', X.FROZEN['evaluate_mechanics'])
    evalcheb = X.load('xc_evaluate_chebyshev_strain', X.FROZEN['evaluate_chebyshev_strain'])
    hier = X.load('xc_hierarchy_strain_network', HERE / 'hierarchy_strain_network_xcase.py')
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
        depth = int(np.ceil(np.log(2 / a.tolerance) / (2 * np.sqrt(a_design))))
    encode_seconds = time.perf_counter() - t_all
    model = cheb.ChebyshevStrainNetwork(base, cycles=depth)
    ca, cb = X.chebyshev_coefficients_interval(depth, a_design)
    model.chebyshev_alpha.copy_(torch.tensor(ca, dtype=dt)); model.chebyshev_beta.copy_(torch.tensor(cb, dtype=dt))
    # query timing: q panels of 7 and 64 boundary displacement columns
    R = tt(np.load(ROOT / 'targets' / (a.case + '_v1') / 'REFERENCE_RQ.npy'))
    quotient = runner.Quotient(np.load(ROOT / 'targets' / (a.case + '_v1') / 'RIGID_Q.npy'), dev)
    query = {}
    with torch.no_grad():
        for cols in (1, 7, 64):
            q = torch.randn((D.shape[0], cols), dtype=dt, device=dev)
            model(q); torch.cuda.synchronize(); t = time.perf_counter(); model(q); torch.cuda.synchronize()
            query[cols] = time.perf_counter() - t
    # finite witness against the teacher
    probes = np.load(comp / 'QUALIFICATION_PROBES.npz')
    probe_t = {k: tt(probes[key]) for k, key in [('q', 'q'), ('zref', 'z'), ('reference_force', 'reference_force')]}
    refdir = Path(a.reference) if a.reference else None
    algebra = evalcheb.complete_algebra(model, probe_t, evaluator)
    witness, white = evaluator.spectral_witness(model, R, quotient.lift, quotient.reduce, seed=X.SEED, maxiter=X.ITERS, return_vector=True)
    replay, _ = evalcheb.replay_witness(model, R, quotient, white)
    res = dict(case=a.case, elements=a.elements, s=a.s, rule_order=a.rule_order, pair_theta=a.pair_theta,
               dimensions=[A.shape[0], P1.shape[1], P2.shape[1]], pair_record=prec,
               ritz_min=float(rv.min()), design_a=a_design, depth=depth, tolerance=a.tolerance,
               stage_seconds=T.rows, encode_seconds=encode_seconds, query_seconds=query,
               original7_energy_ratio=(np.asarray(algebra['predicted_energy']) / np.asarray(algebra['reference_energy'])).tolist(),
               numerical_checks=algebra['numerical_checks'], witness=replay['actual_rayleigh'],
               scope='finite witness only; teacher R whitening; elements=' + a.elements)
    (out / 'RESULT.json').write_text(json.dumps(res, indent=2, default=float))
    np.save(out / 'WITNESS_WHITE.npy', white.cpu().numpy())
    print(json.dumps(dict(event='DONE', encode=round(encode_seconds, 2), stages={k: round(v, 3) for k, v in T.rows.items()},
                          depth=depth, ritz_min=float(rv.min()), query={k: round(v, 3) for k, v in query.items()},
                          witness=replay['actual_rayleigh'][0], original7=res['original7_energy_ratio'])), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', required=True); ap.add_argument('--output', required=True)
    ap.add_argument('--asset-root', default=None); ap.add_argument('--reference', default=None)
    ap.add_argument('--elements', choices=['teacher', 'polyref'], default='teacher')
    ap.add_argument('--s', type=int, default=4); ap.add_argument('--rule-order', type=int, default=4)
    ap.add_argument('--pair-theta', type=float, default=0.9)
    ap.add_argument('--lanczos', type=int, default=80); ap.add_argument('--safety', type=float, default=0.5)
    ap.add_argument('--tolerance', type=float, default=1e-6)
    main(ap.parse_args())
