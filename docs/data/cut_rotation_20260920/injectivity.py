"""Can the model tell two cut coordinates apart at all?

A box coordinate is one background node, so its descriptor (a position) determines it.  A cut
coordinate is a signed functional over about 8.6 nodes, and everything the model learns about it
arrives through two channels:

  * explicit aggregates -- centroid, kind, signed sum, |c| sum, ||c||, log nnz, support radius,
    face membership, the tau/phi/margin field values at the centroid, the first and second moments;
  * the pullback of a learned volume field, `sum_a c_ia f(x_a)`, where `f` is trilinearly sampled
    from an n^3 = 32^3 grid of learned channels.

The second channel is linear in the grid values, so it is exactly a weight vector `w_i` over the
32^3 cells: `w_i[cell] = sum_a c_ia * trilinear(x_a, cell)`.  Two coordinates with the same `w`
and the same aggregates are **indistinguishable to the model whatever it learns**, and if their
rows of the trace operator differ, the training target is not a function of the training input.
That would be a defect no amount of data or capacity can repair, and it would explain the measured
28x presented-to-held-out gap on cut cells directly.

This probe is model-free and reads only frozen packets.  Box coordinates are the control.

    python injectivity.py <seat> [<seat> ...]
"""
import json, sys
import numpy as np

MANIFEST = '/root/autodl-tmp/CLAUDE_LABELS_20260917/V2_LABELS.json'
sys.path.insert(0, '/root/_inject_src')
from superelement.equi.context import compile_equi_inputs


def trilinear_weights(points, n):
    """The exact linear map of `sample_volume`: cell centres at (i+.5)/n, border padding.

    grid_sample with align_corners=False maps the unit cube to sample coordinates u = n*p - .5;
    the two neighbours are floor(u) and floor(u)+1, clamped to [0, n-1] (border padding), with
    weights (1 - frac) and frac.  Returned as (rows, 8) cell indices and weights.
    """
    u = np.asarray(points, dtype=np.float64) * n - .5
    lo = np.floor(u).astype(np.int64)
    frac = u - lo
    idx = np.empty((len(u), 8), dtype=np.int64)
    wgt = np.empty((len(u), 8), dtype=np.float64)
    for corner in range(8):
        bits = ((corner >> 2) & 1, (corner >> 1) & 1, corner & 1)
        cell = np.clip(lo + np.array(bits), 0, n - 1)
        w = np.ones(len(u))
        for axis in range(3):
            w *= frac[:, axis] if bits[axis] else (1. - frac[:, axis])
        idx[:, corner] = np.ravel_multi_index(cell.T, (n, n, n))
        wgt[:, corner] = w
    return idx, wgt


def check_against_grid_sample(n, rng):
    """The analytic weights must reproduce torch's grid_sample exactly."""
    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        return None
    pts = rng.random((256, 3))
    field = rng.standard_normal((1, n, n, n))
    grid = (2.0 * torch.as_tensor(pts).reshape(1, -1, 1, 1, 3) - 1.0)[..., [2, 1, 0]]
    want = F.grid_sample(torch.as_tensor(field)[None], grid, mode='bilinear',
                         padding_mode='border', align_corners=False)[0, :, :, 0, 0].T.numpy().ravel()
    idx, wgt = trilinear_weights(pts, n)
    got = (field.reshape(-1)[idx] * wgt).sum(axis=1)
    return float(np.abs(got - want).max())


def pullback_weights(ctx, n):
    """w_i over the n^3 cells, as a dense (count, n^3) array built row by row (sparse in practice)."""
    idx, wgt = trilinear_weights(ctx['support_pos'], n)
    rows = np.asarray(ctx['support_rows'])
    coeff = np.asarray(ctx['support_coefficients'])          # already normalised by |c| sum
    W = np.zeros((ctx['count'], n ** 3), dtype=np.float64)
    contrib = wgt * coeff[:, None]
    np.add.at(W, (np.repeat(rows, 8), idx.ravel()), contrib.ravel())
    return W


def aggregates(ctx):
    """Everything explicit the model is handed per coordinate, flattened."""
    return np.concatenate([np.asarray(ctx['node_scalar']).reshape(ctx['count'], -1),
                           np.asarray(ctx['node_vector']).reshape(ctx['count'], -1),
                           np.asarray(ctx['node_tensor']).reshape(ctx['count'], -1),
                           np.asarray(ctx['faces']).reshape(ctx['count'], -1),
                           np.asarray(ctx['pos']).reshape(ctx['count'], -1)], axis=1)


def unpack_rows(path, q):
    v = np.load(path, mmap_mode='r')
    m = int(round((np.sqrt(8 * v.size + 1) - 1) / 2))
    if m * (m + 1) // 2 != v.size or m != q:
        raise ValueError(f'PACKED_UPPER {path} {v.size} for q={q}')
    M = np.zeros((q, q))
    iu = np.triu_indices(q)
    M[iu] = v
    M.T[iu] = M[iu]
    return M


def run(seat, rng):
    row = [r for r in json.load(open(MANIFEST)) if int(r['seat']) == int(seat)][0]
    cache = dict(np.load(row['trace_cache'], allow_pickle=False))
    meta = json.loads((__import__('pathlib').Path(row['trace_cache']).parent / 'INPUT.json').read_text())['metadata']
    ctx = compile_equi_inputs(cache, meta)
    n = ctx['n']
    kind = np.asarray(ctx['kind'])
    W = pullback_weights(ctx, n)
    A = aggregates(ctx)
    S = unpack_rows(str(__import__('pathlib').Path(row['packet']) / 'S_UPPER.npy'), 3 * ctx['count'])
    # coordinate-level operator rows: the 3 x q slab of each coordinate
    rows3 = S.reshape(ctx['count'], 3, -1)
    rownorm = np.linalg.norm(rows3.reshape(ctx['count'], -1), axis=1)
    out = dict(seat=int(seat), count=int(ctx['count']), cut=int((kind > 0).sum()), box=int((kind == 0).sum()),
               n=int(n), grid_sample_agreement=check_against_grid_sample(n, rng))
    for name, mask in (('cut', kind > 0), ('box', kind == 0)):
        sel = np.flatnonzero(mask)
        if len(sel) < 2:
            continue
        # candidates: coordinates whose pullback weight vectors are closest
        Wm = W[sel]; Am = A[sel]
        wn = np.linalg.norm(Wm, axis=1)
        gram = Wm @ Wm.T
        d2 = np.maximum(wn[:, None] ** 2 + wn[None, :] ** 2 - 2 * gram, 0.)
        np.fill_diagonal(d2, np.inf)
        dw = np.sqrt(d2) / np.maximum(wn[:, None], 1e-300)
        near = np.dstack(np.unravel_index(np.argsort(dw.ravel())[:40], dw.shape))[0]
        pairs = []
        seen = set()
        for i, j in near:
            key = (min(i, j), max(i, j))
            if key in seen:
                continue
            seen.add(key)
            gi, gj = int(sel[i]), int(sel[j])
            da = float(np.linalg.norm(Am[i] - Am[j]) / max(np.linalg.norm(Am[i]), 1e-300))
            dr = float(np.linalg.norm(rows3[gi] - rows3[gj]) / max(rownorm[gi], 1e-300))
            pairs.append(dict(i=gi, j=gj, pullback_relative=float(dw[i, j]), aggregate_relative=da,
                              operator_row_relative=dr,
                              nnz=[int(np.diff(cache['indptr'])[gi]), int(np.diff(cache['indptr'])[gj])],
                              centroid_distance_over_h=float(np.linalg.norm(ctx['pos'][gi] - ctx['pos'][gj]) * n)))
            if len(pairs) == 10:
                break
        out[name] = dict(closest_pullback_relative=float(dw.min()),
                         median_pullback_relative=float(np.median(dw[np.isfinite(dw)])),
                         pairs=pairs)
    print(json.dumps(out), flush=True)
    return out


if __name__ == '__main__':
    rng = np.random.default_rng(0)
    rows = [run(int(s), rng) for s in sys.argv[1:]]
    json.dump(rows, open('/root/_injectivity.json', 'w'), indent=1)
