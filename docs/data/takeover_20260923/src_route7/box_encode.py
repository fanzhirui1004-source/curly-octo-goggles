"""Route 7: the exact per-cell boundary operator WITHOUT trace coordinates.

In the lattice a cut cell's cut surface is free (never glued, never loaded), so its trace coordinates are interior
unknowns there. The operator the lattice needs is the condensation of the background system onto the box nodal DOFs:
    T = K_bb - K_bi K_ii^-1 K_ib,   K = K_body + gamma * K_ghost on the background Q2 DOFs,
b = the DOFs of every node on the unit-box faces, i = all other DOFs. This equals the packet operator S condensed onto
its box coordinates, S_BB - S_BC S_CC^-1 S_CB: the box coordinates are the box nodal values and P only changes the
basis of the eliminated block, which a Schur complement does not see. So compile_trace (P) is not needed at all.

K_ii is factored once on the GPU (cuDSS, as in encode_r1.DirectModel); queries T q are exact.
--validate compares T with the packet's S condensed onto its box coordinates (use --elements body: teacher bulk
elements, so the only differences are rounding).
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
from scipy import sparse
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import encode_r1 as E

dev, dt = E.dev, E.dt
PACKETS = Path('/root/autodl-tmp/CUTFEM_FRESH_GP_20260921/packets')


def rigid_fields(points):
    R = np.zeros((3 * len(points), 6))
    for d in range(3):
        R[d::3, d] = 1
    c = points - points.mean(0)
    R[0::3, 3], R[1::3, 3] = -c[:, 1], c[:, 0]
    R[1::3, 4], R[2::3, 4] = -c[:, 2], c[:, 1]
    R[0::3, 5], R[2::3, 5] = c[:, 2], -c[:, 0]
    return R


def ghost_faces_gpu(body_dir, case, n, nb, group=8, chunk=1024):
    """Unit ghost-penalty matrix sum_f F_f^T F_f from the face list and the fixed face templates (GP_TEMPLATES_n<n>.npz,
    written once by fast_prep3 from the frozen stage_cutfem_gp.kernel), assembled on the GPU as upper-triangle
    135 x 135 blocks in chunks and mirrored. Same construction as fast_gp.ghost_matrix / sens_blocks v2."""
    d = Path(body_dir) / case
    faces = torch.as_tensor(np.load(d / 'GP_FACES.npy').astype(np.int64), device=dev)
    cells = torch.as_tensor(np.load(d / 'CELL_INDICES.npy').astype(np.int64), device=dev)
    nodes = torch.as_tensor(np.load(d / 'NODES.npy').astype(np.int64), device=dev)
    tpl = np.load(Path(body_dir) / f'GP_TEMPLATES_n{n}.npz')
    canon = torch.as_tensor(tpl['canonical'], dtype=dt, device=dev)
    offs = torch.as_tensor(tpl['offsets'].astype(np.int64), device=dev)
    tk = canon.transpose(1, 2) @ canon
    g = 2 * cells[faces[:, 0]][:, None, :] + offs[faces[:, 2]]
    M = 2 * n + 1
    local = torch.searchsorted(nodes, (g[..., 0] * M + g[..., 1]) * M + g[..., 2])
    dofs = (3 * local[:, :, None] + torch.arange(3, device=dev)).reshape(len(faces), -1)
    w = dofs.shape[1]
    iu = torch.triu_indices(w, w, device=dev)
    total, pending = None, []
    for lo in range(0, len(faces), chunk):
        dd = dofs[lo:lo + chunk]
        r, c = dd[:, iu[0]], dd[:, iu[1]]
        pending.append((torch.minimum(r, c).reshape(-1), torch.maximum(r, c).reshape(-1),
                        tk[faces[lo:lo + chunk, 2]][:, iu[0], iu[1]].reshape(-1)))
        if len(pending) == group or lo + chunk >= len(faces):
            part = E.coo(torch.cat([p[0] for p in pending]), torch.cat([p[1] for p in pending]),
                         torch.cat([p[2] for p in pending]), (nb, nb))
            total = part if total is None else (total + part).coalesce()
            pending = []
    i, v = total.indices(), total.values()
    off = i[0] != i[1]
    return E.coo(torch.cat([i[0], i[1][off]]), torch.cat([i[1], i[0][off]]), torch.cat([v, v[off]]), (nb, nb))


def split(K, b_mask):
    """A = K_ii, C = K_ib, D = K_bb as coalesced COO tensors, with new contiguous numbering."""
    nb = K.shape[0]
    new = torch.empty(nb, dtype=torch.long, device=dev)
    bi = torch.nonzero(b_mask).squeeze(1); ii = torch.nonzero(~b_mask).squeeze(1)
    new[bi] = torch.arange(len(bi), device=dev); new[ii] = torch.arange(len(ii), device=dev)
    i, v = K.indices(), K.values()
    rb, cb = b_mask[i[0]], b_mask[i[1]]
    A = E.coo(new[i[0][~rb & ~cb]], new[i[1][~rb & ~cb]], v[~rb & ~cb], (len(ii), len(ii)))
    C = E.coo(new[i[0][~rb & cb]], new[i[1][~rb & cb]], v[~rb & cb], (len(ii), len(bi)))
    D = E.coo(new[i[0][rb & cb]], new[i[1][rb & cb]], v[rb & cb], (len(bi), len(bi)))
    return A, C, D, bi, ii


def main(a):
    import os
    if a.body_dir:
        os.environ['G_BODY_DIR'] = str(Path(a.body_dir) / a.case)     # element_moments.members reads the fast CPU stage
    import element_moments as EM
    E.ELEMENTS = a.elements
    E.COVER_INPUTS = Path(a.cover_inputs) if a.cover_inputs else None
    T = E.Timer()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    if a.elements == 'body':
        Kb, nb, ctx = E.elements_body(a.case, T)
    else:
        Kb, nb, ctx = E.elements_polyref(a.case, a.s, a.rule_order, T, a.levels)
    gamma = float(json.loads((PACKETS / a.case / 'SAMPLE.json').read_text())['gp']['gamma'])
    with T('B1_ghost_and_assembly'):
        if a.ghost == 'faces':
            G = ghost_faces_gpu(a.body_dir, a.case, int(ctx['n']), nb)
        else:
            G = E.ghost(a.case, nb)
        K = (Kb + gamma * G).coalesce()
        del Kb
    if a.check_ghost and E.COVER_INPUTS is not None:
        Gref = sparse.load_npz(E.COVER_INPUTS / 'GHOST.npz').tocoo()
        Gr = E.coo(torch.as_tensor(Gref.row.astype(np.int64), device=dev), torch.as_tensor(Gref.col.astype(np.int64), device=dev),
                   torch.as_tensor(Gref.data, dtype=dt, device=dev), (nb, nb))
        diff = (G - Gr).coalesce().values().abs().max()
        ghost_check = float(diff / Gr.values().abs().max())
    else:
        ghost_check = None
    del G
    n = int(ctx['n'])
    nodes = EM.members(a.case, ['NODES.npy'])['NODES.npy']
    grid = np.stack(np.unravel_index(nodes, (2 * n + 1,) * 3), axis=1)
    bn_file = Path(a.body_dir) / a.case / 'BOX_NODES.npy' if a.body_dir else None
    if bn_file is not None and bn_file.exists():      # certified positive box-face patches (fast_prep3)
        onbox = np.isin(nodes, np.load(bn_file))
    else:
        onbox = np.any((grid == 0) | (grid == 2 * n), axis=1)
    with T('B2_partition'):
        b_mask = torch.as_tensor(np.repeat(onbox, 3), device=dev)
        A, C, D, bi, ii = split(K, b_mask)
    runner_src = E.MN / 'source_v1'
    sys.path.insert(0, str(runner_src))
    import cross_case_chebyshev as X
    runner = X.load('xc_run_mechanics', X.FROZEN['run_mechanics'])
    R = torch.as_tensor(rigid_fields(grid / (2 * n)), dtype=dt, device=dev)
    ub, ui = runner.orthonormalize_rigid_pair(R[bi], R[ii])
    model = E.DirectModel(A, C, D, ub, ui, T, width=a.direct_width, threads=a.direct_threads, reorder=a.direct_reorder)
    torch.cuda.synchronize(); encode_seconds = time.perf_counter() - t0
    query = {}
    with torch.no_grad():
        for cols in (1, 8, 64):
            q = torch.randn((D.shape[0], cols), dtype=dt, device=dev)
            model(q); torch.cuda.synchronize(); t = time.perf_counter(); model(q); torch.cuda.synchronize()
            query[cols] = time.perf_counter() - t
    rec = dict(case=a.case, elements=a.elements, ghost=a.ghost, ghost_check_relative=ghost_check, body_dir=a.body_dir,
               nb=int(nb), box_nodes=int(onbox.sum()), box_dofs=int(len(bi)),
               interior_dofs=int(len(ii)), encode_seconds=encode_seconds, stage_seconds=T.rows, query_seconds=query,
               torch_cuda_peak_gib=torch.cuda.max_memory_allocated() / 2 ** 30)
    if a.validate:
        rec['validation'] = validate(a, model, nodes[onbox], n)
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    (out / f'BOX_{a.case}_{a.elements}.json').write_text(json.dumps(rec, indent=2, default=float))
    print(json.dumps({k: v for k, v in rec.items() if k != 'stage_seconds'}, default=float), flush=True)


def validate(a, model, box_node_ids, n):
    """T q against the packet: S condensed onto its box coordinates, matched by global node id and component."""
    sys.path.insert(0, str(HERE))
    import lattice_v2 as LV
    from scipy import linalg
    z = np.load(PACKETS / a.case / 'TRACE.npz')
    bbo, bn = z['box_boundary_original'], z['background_nodes']
    m = int(json.loads((PACKETS / a.case / 'SAMPLE.json').read_text())['full_trace_dimension'])
    S = LV.packed_upper_dense(PACKETS / a.case / 'S_UPPER.npy', m)
    mb = len(bbo)
    if mb < m:
        Lc = linalg.cho_factor(S[mb:, mb:], check_finite=False)
        Tref = S[:mb, :mb] - S[:mb, mb:] @ linalg.cho_solve(Lc, S[mb:, :mb], check_finite=False)
    else:
        Tref = S
    key_ref = {(int(bn[d // 3]), int(d % 3)): k for k, d in enumerate(bbo)}
    order = np.array([key_ref[(int(g), c)] for g in box_node_ids for c in range(3)])   # our box DOF -> packet coordinate
    rng = np.random.default_rng(7)
    Q = rng.standard_normal((len(order), 16))
    with torch.no_grad():
        ours = model(torch.as_tensor(Q, dtype=dt, device=dev)).cpu().numpy()
    ref = np.zeros_like(Q)
    Qp = np.zeros((mb, 16)); Qp[order] = Q
    ref = (Tref @ Qp)[order]
    rel = np.linalg.norm(ours - ref, axis=0) / np.linalg.norm(ref, axis=0)
    return dict(box_coordinates_packet=int(mb), box_coordinates_ours=int(len(order)),
                same_box_set=bool(len(order) == mb and len(set(order.tolist())) == mb),
                relative_difference_max=float(rel.max()), relative_difference_median=float(np.median(rel)))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', required=True); ap.add_argument('--output', required=True)
    ap.add_argument('--elements', choices=['polyref', 'body'], default='polyref')
    ap.add_argument('--cover-inputs', default=None)
    ap.add_argument('--s', type=int, default=4); ap.add_argument('--rule-order', type=int, default=4); ap.add_argument('--levels', type=int, default=1)
    ap.add_argument('--direct-width', type=int, default=8); ap.add_argument('--direct-threads', type=int, default=16)
    ap.add_argument('--direct-reorder', default='DEFAULT')
    ap.add_argument('--validate', action='store_true')
    ap.add_argument('--body-dir', default=None); ap.add_argument('--ghost', choices=['file', 'faces'], default='file')
    ap.add_argument('--check-ghost', action='store_true')
    main(ap.parse_args())
