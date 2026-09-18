"""Train and evaluate the index-free model on the q-space label M_q = B^T A^{-1/2} B.

    python -m superelement.equi.train_equi --seats 253 --steps 20000 --augment --output OUT

Per step: one geometry, an optional cube symmetry g (random of 48 with --augment, the
canonical frame with --canonical), the geometry's inputs rotated by g on the device,
the label blocks of the sampled pairs rotated by the same g, the frozen three-bucket
loss with log pivots.  Buckets are geometric (diagonal / within the near radius /
beyond), not Morton patches, so they are the same in every frame.

Evaluation predicts every lower block, assembles the symmetric q x q M_q_hat, and scores
it in the frozen quotient: M_hat = B M_q_hat B^T, nu = eig((M_hat R^T)^T (M_hat R^T)),
mu = 1/nu, g = max(mu_max, 1/mu_min).  A_hat = M_hat^-2 is formed only for e_A and for
the two-cell physics test (its Cholesky factor is written as A_PRED_UPPER.npy for
sens_model.py).  Two memorisation probes follow: the prediction for the cell rotated
by g, mapped back, against the prediction for the cell (what augmentation is supposed
to buy), and the prediction for the cell with shuffled node indices (must be exact).
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch

from . import cubic_group as CG
from .context import compile_equi_inputs, permute_nodes, rigid_span_residual as rigid_span_residual_of
from .model import EquiModel, GroupTables, to_torch


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for b in iter(lambda: fh.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def write(path, value):
    path = Path(path); tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n'); tmp.replace(path)


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    return time.perf_counter()


def unpack(path, d, device):
    p = np.load(path, mmap_mode='r', allow_pickle=False)
    if p.shape != (d * (d + 1) // 2,) or p.dtype != np.float64:
        raise ValueError(f'FP64_UPPER_REQUIRED {path}')
    out = np.zeros((d, d), dtype=np.float64); off = 0
    for row in range(d):
        out[row, row:] = p[off:off + d - row]; off += d - row
    return torch.from_numpy(out).to(device)


def pack(dense, path):
    d = len(dense); host = dense.cpu().numpy()
    out = np.lib.format.open_memmap(path, mode='w+', dtype=np.float64, shape=(d * (d + 1) // 2,)); off = 0
    for row in range(d):
        out[off:off + d - row] = host[row, row:]; off += d - row
    out.flush(); del out


# ----------------------------------------------------------------------------- data
def selected_rows(manifest, seats, known_328):
    rows = json.loads(Path(manifest).read_text()); result = []
    for seat in seats:
        matches = [r for r in rows if int(r['seat']) == seat]
        if len(matches) != 1:
            raise ValueError('NONUNIQUE_OR_MISSING_SEAT')
        r = matches[0]
        if r['split'] != 'train' and not (seat == 328 and known_328 and seats == [328]):
            raise ValueError('VALIDATION_TEST_HOLDOUT_NUMERIC_ACCESS_FORBIDDEN')
        result.append(r)
    if len(set(seats)) != len(seats):
        raise ValueError('DUPLICATE_SEAT')
    return result


def prepare(row, args, device):
    from scipy.spatial import cKDTree
    reference = Path(row['reference']); cache_path = Path(row['trace_cache'])
    receipt = json.loads((reference / 'RESULT.json').read_text())
    d = int(receipt['dimension']); q = d + 6
    factor = reference / 'R_UPPER.npy'; label = reference / 'MQ_UPPER.npy'
    bindings = dict(factor_sha256=sha256(factor), trace_sha256=sha256(cache_path))
    if any(bindings[k] != receipt[k] for k in bindings):
        raise ValueError('FROZEN_REFERENCE_BINDING')
    mq = json.loads((reference / 'MQ_RESULT.json').read_text())
    if (mq['r_sha256'] != bindings['factor_sha256'] or mq['trace_sha256'] != bindings['trace_sha256']
            or mq['m_sha256'] != sha256(label) or int(mq['q']) != q or mq['read_blocks_roundtrip'] != 0.0):
        raise ValueError('FROZEN_MQ_LABEL_BINDING')
    bindings['mq_label_sha256'] = mq['m_sha256']
    cache = dict(np.load(cache_path, allow_pickle=False))
    meta = json.loads((cache_path.parent / 'INPUT.json').read_text())['metadata']
    ctx = compile_equi_inputs(cache, meta)
    if 3 * ctx['count'] != q or int(row['q']) != q:
        raise ValueError('FROZEN_DIMENSION_BINDING')
    radius = args.near_cells / ctx['n']
    pairs = cKDTree(ctx['pos']).query_pairs(radius, output_type='ndarray')
    if len(pairs) == 0:
        raise ValueError('EMPTY_NEAR_BUCKET')
    r = pairs.max(axis=1); c = pairs.min(axis=1)
    near_codes = np.sort(r.astype(np.int64) * ctx['count'] + c)
    if ctx['count'] * (ctx['count'] - 1) // 2 - len(near_codes) <= 0:
        raise ValueError('EMPTY_FAR_BUCKET')
    # The augmentation label identity M_q(g . cell) = G M_q G^T holds because span(rigid) is
    # the rigid-body 6-space of the cell -- the unique 6-space closed under G = blockdiag(Q_g).
    # For a general signed trace that space is the pullback of the background rigid modes
    # through the trace operator, and the identity rigid == (L (x) I3) rigid_body is EXACT
    # (0.0, measured on full and cut cells alike), so this also verifies that the CSR and the
    # frozen rigid array describe the same cell.  Nothing else in the pipeline checks it.
    rigid_span_residual = rigid_span_residual_of(cache, ctx['n'])
    if rigid_span_residual > 1e-12:
        raise ValueError(f'RIGID_IS_NOT_THE_TRACE_PULLBACK_OF_THE_RIGID_MODES {rigid_span_residual:.3e}')
    g_canon, _, ties = CG.canonical(ctx['corners'])
    record = dict(seat=int(row['seat']), split=row['split'], q=q, d=d, count=ctx['count'], n=ctx['n'],
                  near_radius=radius, near_pairs=int(len(near_codes)),
                  all_lower_pairs=int(ctx['count'] * (ctx['count'] - 1) // 2), canonical_frame=int(g_canon),
                  canonical_ties=int(ties), rigid_span_residual=rigid_span_residual,
                  bindings=bindings, label=str(label), whitener=str(factor),
                  trace_cache=str(cache_path), corners=ctx['corners'].tolist())
    return dict(seat=int(row['seat']), split=row['split'], q=q, d=d, count=ctx['count'], ctx=ctx,
                ctx_t=to_torch(ctx, device), near_codes=near_codes, radius=radius, canonical_g=int(g_canon),
                packed=np.load(label, mmap_mode='r', allow_pickle=False), label=label, whitener=factor,
                reference=reference, cache=cache, record=record, tilt=None, weight=None)


def node_scales(sample, read_blocks):
    """The label's own scale per coordinate: the mean of the three pivots of M_q's diagonal block.

    M_q = B^T A^{-1/2} B, so this is large exactly where the cell is soft.  That is also where
    the assembled solution's energy sits: the exact law for the assembled compliance is
    chat/c = sum_i w_i / mu_i with w the true solution's energy share per pencil mode, and the
    energy concentrates in the soft directions.  So tilting the sampler and the diagonal weight
    by this scale points the loss at the entries the physics reads.
    """
    count = int(sample['count'])
    diag = np.arange(count, dtype=np.int64)
    blocks = read_blocks(sample['packed'], diag, diag, sample['q'])
    pivots = np.diagonal(blocks, axis1=1, axis2=2)
    if not np.all(pivots > 0):
        raise ValueError('NONPOSITIVE_REFERENCE_PIVOT_IN_SCALES')
    return pivots.mean(axis=1)


def scale_tilt(scales, near_codes, count, alpha):
    """Categorical draws proportional to s^alpha per coordinate and to (s_r s_c)^alpha per pair."""
    if alpha == 0.0:
        return None
    s = np.asarray(scales, dtype=np.float64) ** float(alpha)
    node = s / s.sum()
    pair = s[near_codes // count] * s[near_codes % count]
    return dict(alpha=float(alpha), node_cdf=np.cumsum(node),
                near_cdf=np.cumsum(pair / pair.sum()),
                weight=(s / s.mean()))          # mean 1, so the loss keeps its scale


def sample_pairs_geometric(sample, per_bucket, rng, tilt=None):
    """Complete diagonal; near and far pairs drawn uniformly, or tilted by the label scale."""
    count = sample['count']; codes = sample['near_codes']
    diag = np.arange(count, dtype=np.int64)
    if tilt is None:
        near = codes[rng.integers(0, len(codes), size=per_bucket)]
        draw = lambda size: (rng.integers(0, count, size=size), rng.integers(0, count, size=size))
    else:
        near = codes[np.searchsorted(tilt['near_cdf'], rng.random(per_bucket))]
        cdf = tilt['node_cdf']
        draw = lambda size: (np.searchsorted(cdf, rng.random(size)).clip(0, count - 1),
                             np.searchsorted(cdf, rng.random(size)).clip(0, count - 1))
    far = []; needed = per_bucket; rounds = 0
    while needed:
        rounds += 1
        if rounds > 64:
            raise ValueError('FAR_BUCKET_REJECTION_DID_NOT_CONVERGE')
        a, b = draw(2 * needed + 32)
        r = np.maximum(a, b); c = np.minimum(a, b); code = r * count + c
        pos = np.searchsorted(codes, code)
        is_near = (pos < len(codes)) & (codes[np.minimum(pos, len(codes) - 1)] == code)
        keep = (r != c) & ~is_near
        code = code[keep][:needed]; far.append(code); needed -= len(code)
    far = np.concatenate(far)
    r = np.concatenate((diag, near // count, far // count)); c = np.concatenate((diag, near % count, far % count))
    bucket = np.repeat(np.arange(3), (count, per_bucket, per_bucket))
    return r, c, bucket


def smooth_face_loads(ctx, per_face, rng, degree=2):
    """Random polynomial tractions on each box face -- the load a lattice cell sees from a neighbour.

    The exact single-cell law is c(f) = ||M_q f||^2, so the compliance error under a load is the
    relative error of the label's ACTION on it.  A point load concentrates that action on a few
    modes and the network is 10-20 % off there; a smooth face traction spreads it over thousands
    and the +-random per-mode errors cancel to ~1 % (measured, 12 seats).  The loads that occur in
    a lattice are the smooth ones, so that is the family to train and to gate on.  Each load is
    (members, values): the face's coordinates and a (members x 3) field, a degree-<=2 polynomial in
    the two in-face coordinates with a random 3-vector per monomial.  Not rigid-projected: M_q has
    the rigid nullspace, so M_q f = M_q (Pi f) exactly, and a prediction's rigid leak is charged.
    """
    faces = ctx['faces']; pos = ctx['pos']
    loads = []
    for f_id in range(6):
        members = np.flatnonzero(faces[:, f_id])
        if len(members) < 8:
            continue
        axis = f_id // 2; u = [k for k in range(3) if k != axis]
        st = pos[members][:, u] - .5
        cols = [np.ones(len(members))]
        for dg in range(1, degree + 1):
            for i in range(dg + 1):
                cols.append(st[:, 0] ** (dg - i) * st[:, 1] ** i)
        basis = np.stack(cols, axis=1)
        for _ in range(per_face):
            values = basis @ rng.standard_normal((basis.shape[1], 3))
            values /= np.sqrt((values * values).sum())
            loads.append((members.astype(np.int64), values.astype(np.float64), int(f_id)))
    if not loads:
        raise ValueError('NO_FACE_WITH_ENOUGH_COORDINATES')
    return loads


def smooth_global_loads(ctx, n_loads, rng):
    """Low-frequency cosine fields over the whole trace: the easiest realistic family, gate only."""
    pos = ctx['pos']; count = ctx['count']; loads = []
    for _ in range(n_loads):
        k = rng.integers(1, 3, size=3); ph = rng.uniform(0, 2 * np.pi)
        field = np.cos(2 * np.pi * (pos @ k) + ph)[:, None] * rng.standard_normal(3)[None, :]
        field /= np.sqrt((field * field).sum())
        loads.append((np.arange(count, dtype=np.int64), field, -1))
    return loads


def dense_label(path, q):
    p = np.load(path, mmap_mode='r', allow_pickle=False)
    M = np.zeros((q, q)); off = 0
    for i in range(q):
        M[i, i:] = p[off:off + q - i]; off += q - i
    return M + np.triu(M, 1).T


def load_vectors(loads, q):
    F = np.zeros((len(loads), q))
    for i, (members, values, _) in enumerate(loads):
        F[i, (members[:, None] * 3 + np.arange(3)).ravel()] = values.ravel()
    return F


def response_bank(sample, per_face, seed, cache_dir, n_global=32):
    """The training family's exact responses y = M_q f from the label, cached by label sha."""
    cache_dir = Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
    sha = sample['record']['bindings']['mq_label_sha256'][:16]
    path = cache_dir / f"seat{sample['seat']}_{sha}_s{seed}_f{per_face}_g{n_global}.npz"
    q = sample['q']; ctx = sample['ctx']
    rng = np.random.default_rng((seed * 1000003 + int(sample['seat'])) % (2 ** 32))
    train_loads = smooth_face_loads(ctx, per_face, rng)
    held = smooth_face_loads(ctx, per_face, rng) + smooth_global_loads(ctx, n_global, rng)
    if path.exists() and 'colnorm2' in np.load(path):
        z = np.load(path)
        bank = dict(Y=z['Y'], ynorm2=z['ynorm2'], Yh=z['Yh'], yhnorm2=z['yhnorm2'], colnorm2=z['colnorm2'])
    else:
        M = dense_label(sample['label'], q)
        F = load_vectors(train_loads, q); Fh = load_vectors(held, q)
        Y = F @ M; Yh = Fh @ M
        # ||M_q e_k||^2 per coordinate dof, summed over the node's three dofs: the compliance of a
        # unit point load at that coordinate, and the normaliser of the column (point-load) term
        colnorm2 = (M * M).sum(axis=0).reshape(-1, 3).sum(axis=1); del M
        bank = dict(Y=Y, ynorm2=(Y * Y).sum(axis=1), Yh=Yh, yhnorm2=(Yh * Yh).sum(axis=1), colnorm2=colnorm2)
        np.savez(path, **bank)
    bank.update(train=train_loads, held=held, path=str(path))
    return bank


def response_term(model, enc, ctx, conditioning, bank, load_index, rows, g, Q, device):
    """Row-subsampled, unbiased estimate of ||(M_hat - M) f||^2 / ||M f||^2 in the view frame g.

    In the frame the model sees, blocks are Q B Q^T at the SAME index pair, so the response to the
    rotated load Q f_c is Q y_r: rotate the load and the bank response instead of the prediction.
    The prediction is queried on the lower pair (max, min) and transposed back where r < c; the
    diagonal block is the masked lower triangle plus its pivots, symmetrised like `evaluate` does.
    """
    members, values, _ = bank['train'][load_index]
    count = int(ctx['pos'].shape[0]); R = len(rows); m = len(members)
    r = np.repeat(rows, m); c = np.tile(members, R)
    lo = np.minimum(r, c); hi = np.maximum(r, c); swapped = r < c
    hi_t = torch.as_tensor(hi, device=device); lo_t = torch.as_tensor(lo, device=device)
    blocks = model.decode(enc, ctx, hi_t, lo_t, conditioning).float()      # (R*m, 3, 3) in frame g
    diag = hi_t == lo_t
    if bool(diag.any()):
        blocks = blocks.clone()
        blocks[diag] = blocks[diag] + torch.tril(blocks[diag], -1).transpose(1, 2)
    sw = torch.as_tensor(swapped, device=device)
    blocks = torch.where(sw[:, None, None], blocks.transpose(1, 2), blocks)   # now B_rc for every (r, c)
    f = torch.as_tensor(values, dtype=torch.float32, device=device) @ Q.T    # Q f_c, (m, 3)
    contrib = torch.einsum('pij,pj->pi', blocks, f.repeat(R, 1))             # (R*m, 3)
    resp = contrib.reshape(R, m, 3).sum(dim=1)                               # (R, 3) = (M_hat f)_r
    y = torch.as_tensor(bank['Y'][load_index].reshape(count, 3)[rows], dtype=torch.float32, device=device) @ Q.T
    num = (count / R) * (resp - y).square().sum()
    return num / float(bank['ynorm2'][load_index])


def column_term(model, enc, ctx, conditioning, sample, read_blocks, bank, columns, rows, g, tables, device):
    """Point loads: the mean over sampled coordinates c of ||(M_hat - M)[:, c]||^2 / ||M[:, c]||^2.

    The absolute bucket loss normalises every entry by one global rms, so a column whose norm is
    15 (an ordinary face node) is fit to ~15 % while a column of norm 2000 (a void pocket) is fit
    to 2-4 % -- measured on 12 seats.  Point loads read exactly one column, so this term asks for
    RELATIVE accuracy per column, equally for stiff and soft coordinates.  Rows are subsampled
    uniformly (unbiased, count/R), the label blocks are read on the fly, and the frame is handled
    like the bucket loss: the label blocks are rotated by g.
    """
    count = int(sample['count']); R = len(rows); K = len(columns)
    r = np.repeat(rows[None, :], K, axis=0).ravel(); c = np.repeat(columns, R)
    lo = np.minimum(r, c); hi = np.maximum(r, c); swapped = r < c
    target = torch.as_tensor(read_blocks(sample['packed'], hi, lo, sample['q']), dtype=torch.float32, device=device)
    diag_np = hi == lo
    bucket = torch.as_tensor(np.where(diag_np, 0, 2), device=device)
    target = tables.rotate_target_blocks(target, bucket, g)
    hi_t = torch.as_tensor(hi, device=device); lo_t = torch.as_tensor(lo, device=device)
    pred = model.decode(enc, ctx, hi_t, lo_t, conditioning).float()
    diag = hi_t == lo_t
    if bool(diag.any()):                                   # lower-masked diagonal blocks, both sides
        pred = pred.clone(); pred[diag] = torch.tril(pred[diag])
    err = (pred - target).square().sum(dim=(1, 2)).reshape(K, R)          # transposition does not change the norm
    norm2 = torch.as_tensor(bank['colnorm2'][columns], dtype=torch.float32, device=device)
    return ((count / R) * err.sum(dim=1) / norm2).mean()


def response_gate(M, bank, q):
    """Assembly-free compliance errors ||M_hat f||^2 / ||M f||^2 - 1 on the held-out family."""
    Fh = torch.as_tensor(load_vectors(bank['held'], q), dtype=M.dtype, device=M.device)
    Yh = Fh @ M
    err = ((Yh * Yh).sum(dim=1) / torch.as_tensor(bank['yhnorm2'], dtype=M.dtype, device=M.device) - 1.0).cpu().numpy()
    kinds = np.array([k for _, _, k in bank['held']])
    face = err[kinds >= 0]; glob = err[kinds < 0]
    summary = lambda v: dict(n=int(len(v)), median_abs=float(np.median(np.abs(v))), p90_abs=float(np.percentile(np.abs(v), 90)),
                             max_abs=float(np.abs(v).max()), signed_mean=float(v.mean()),
                             frac_within_3pct=float((np.abs(v) <= .03).mean()))
    return dict(face=summary(face), global_smooth=summary(glob), worst_abs=float(np.abs(err).max()))


def response_selftest(device, seed=20260919, n=6, q_nodes=40):
    """The estimator: exact label -> 0; scaled label -> the exact ratio; row subsampling unbiased."""
    rng = np.random.default_rng(seed)
    count = q_nodes; q = 3 * count
    A = rng.standard_normal((q, q)); M = A @ A.T / q + np.eye(q)
    pos = rng.random((count, 3)); faces = np.zeros((count, 6), dtype=bool)
    faces[:, 0] = pos[:, 0] < .3; faces[:, 1] = pos[:, 0] > .7
    ctx = dict(pos=pos, faces=faces, count=count)
    loads = smooth_face_loads(ctx, 2, rng)
    F = load_vectors(loads, q); Y = F @ M
    bank = dict(train=loads, Y=Y, ynorm2=(Y * Y).sum(axis=1))
    out = {}
    for scale in (1.0, 1.1):
        Mh = scale * M
        # emulate response_term with a "model" that returns the true (scaled) blocks
        worst = 0.; ests = []
        for li in range(len(loads)):
            members, values, _ = loads[li]; m = len(members)
            for trial in range(200):
                rows = rng.integers(0, count, size=4)
                resp = np.stack([sum(Mh[3 * r:3 * r + 3, 3 * c:3 * c + 3] @ values[k] for k, c in enumerate(members)) for r in rows])
                y = Y[li].reshape(count, 3)[rows]
                ests.append((count / 4) * ((resp - y) ** 2).sum() / bank['ynorm2'][li])
            exact = ((Mh @ F[li] - Y[li]) ** 2).sum() / bank['ynorm2'][li]
            worst = max(worst, abs(np.mean(ests[-200:]) - exact) / max(exact, 1e-12) if exact > 0 else abs(np.mean(ests[-200:])))
        out[f'scale_{scale}'] = dict(exact=float(exact), estimator_mean=float(np.mean(ests[-200:])), relative_bias=float(worst))
    if out['scale_1.0']['estimator_mean'] > 1e-24:          # summation-order noise only
        raise ValueError('EXACT_LABEL_MUST_GIVE_ZERO_RESPONSE_LOSS')
    if abs(out['scale_1.1']['exact'] - 0.1 ** 2) > 1e-9:
        raise ValueError('SCALED_LABEL_MUST_GIVE_EPSILON_SQUARED')
    if out['scale_1.1']['relative_bias'] > 0.2:      # 200 draws of 4 rows: a Monte-Carlo tolerance
        raise ValueError(f'ROW_SUBSAMPLING_IS_BIASED {out}')
    return out


def equi_loss(prediction, target, buckets, conditioning, weight=None, asymmetry=0.0, tail_beta=0.0,
              diagonal_loss='log'):
    """The frozen three-bucket loss plus three knobs; at neutral knobs it reproduces it exactly.

    * `weight` is the frozen `importance` hook: a per-pair multiplier, mean 1 within a bucket.
    * `asymmetry` puts extra weight on pivots predicted SOFTER than the truth.  The assembled
      compliance obeys chat/c = sum_i w_i / mu_i, so an over-soft direction (mu < 1) contributes
      w_i / mu_i and can blow up, while an over-stiff one contributes at most -w_i.  The label
      is M_q = B^T A^{-1/2} B, so a pivot predicted too LARGE is the over-soft direction.
    * `tail_beta` adds a smooth maximum over the per-coordinate diagonal error to its mean,
      because g is a max: on the outlier seats about 8 coordinates out of 6016 set it, which is
      0.13 % of a mean and invisible to it.

    Equivalence with `stage_cutfem_m4.factors.bucket_loss` at weight=None, asymmetry=0,
    tail_beta=0 is asserted bit for bit by `loss_selftest`.
    """
    if diagonal_loss not in ('absolute', 'log'):
        raise ValueError('UNKNOWN_DIAGONAL_LOSS')
    scales = (conditioning.diagonal_block_rms, conditioning.same_patch_rms, conditioning.cross_patch_rms)
    errors = [((prediction[buckets == j] - target[buckets == j]) / scales[j]).square() for j in range(3)]
    if weight is not None:
        errors = [v * weight[buckets == j, None, None] for j, v in enumerate(errors)]
    tail = None
    if diagonal_loss == 'log':
        pd = prediction[buckets == 0]; td = target[buckets == 0]
        if not bool((td.diagonal(dim1=-2, dim2=-1) > 0).all()):
            raise ValueError('NONPOSITIVE_REFERENCE_PIVOT')
        if not bool((pd.diagonal(dim1=-2, dim2=-1) > 0).all()):
            raise ValueError('NONPOSITIVE_PREDICTED_PIVOT')
        ratio = pd.diagonal(dim1=-2, dim2=-1).log() - td.diagonal(dim1=-2, dim2=-1).log()
        pivot = (ratio / conditioning.pivot_log_std).square()
        if asymmetry:
            pivot = pivot * (1.0 + asymmetry * (ratio > 0).to(pivot.dtype))
        lower = ((torch.tril(pd, -1) - torch.tril(td, -1)) / conditioning.diagonal_lower_rms).square()
        index = torch.tril_indices(3, 3, -1, device=lower.device)
        lower = lower[:, index[0], index[1]]
        if weight is not None:
            w = weight[buckets == 0, None]; pivot = pivot * w; lower = lower * w
        errors[0] = torch.cat((pivot, lower), dim=1)
        if tail_beta:
            per = errors[0].mean(dim=1)
            tail = torch.logsumexp(per * tail_beta, dim=0) / tail_beta
    pieces = [v.mean() for v in errors]
    loss = torch.stack(pieces).mean()
    if tail is not None:
        loss = loss + tail / 3.0                 # one bucket's worth of weight, not more
        pieces = pieces + [tail]
    return loss, torch.stack(pieces).detach()


def loss_selftest(bucket_loss, Conditioning, device, seed=20260919):
    """equi_loss at neutral knobs must equal the frozen bucket_loss bit for bit, in both modes."""
    torch.manual_seed(seed)
    conditioning = Conditioning(0.3, 1.7, 0.9, 1.3, 0.7, 0.5)
    n = (12, 9, 9)
    buckets = torch.as_tensor(np.repeat(np.arange(3), n), device=device)
    target = torch.randn(sum(n), 3, 3, device=device)
    prediction = target + 0.1 * torch.randn_like(target)
    for t in (target, prediction):
        for k in range(3):
            t[:n[0], k, k] = t[:n[0], k, k].abs() + 0.5
    out = {}
    for mode in ('absolute', 'log'):
        a, pa = bucket_loss(prediction, target, buckets, conditioning, None, mode)
        b, pb = equi_loss(prediction, target, buckets, conditioning, None, 0.0, 0.0, mode)
        out[mode] = dict(loss_difference=float((a - b).abs()), parts_difference=float((pa - pb).abs().max()))
        if out[mode]['loss_difference'] != 0.0 or out[mode]['parts_difference'] != 0.0:
            raise ValueError(f'EQUI_LOSS_IS_NOT_THE_FROZEN_LOSS {mode} {out[mode]}')
    # a weight of all ones is also the identity
    ones = torch.ones(sum(n), device=device)
    a, _ = bucket_loss(prediction, target, buckets, conditioning, None, 'log')
    b, _ = equi_loss(prediction, target, buckets, conditioning, ones, 0.0, 0.0, 'log')
    out['unit_weight_difference'] = float((a - b).abs())
    if out['unit_weight_difference'] != 0.0:
        raise ValueError(f'UNIT_WEIGHT_IS_NOT_THE_IDENTITY {out["unit_weight_difference"]}')
    # the knobs must actually change the objective, and only in the intended direction
    base, _ = equi_loss(prediction, target, buckets, conditioning, None, 0.0, 0.0, 'log')
    asym, _ = equi_loss(prediction, target, buckets, conditioning, None, 1.0, 0.0, 'log')
    tail, _ = equi_loss(prediction, target, buckets, conditioning, None, 0.0, 4.0, 'log')
    out['asymmetry_raises_loss'] = float(asym - base)
    out['tail_raises_loss'] = float(tail - base)
    if not (out['asymmetry_raises_loss'] > 0 and out['tail_raises_loss'] > 0):
        raise ValueError(f'KNOBS_ARE_INERT {out}')
    return out


def calibrate_geometric(samples, per_bucket, read_blocks, Conditioning, log_pivot_window=(-20., 20.), seed=20260917):
    rng = np.random.default_rng(seed); squares = np.zeros(4); counts = np.zeros(4); logs = []; bindings = []
    for s in samples:
        r, c, bucket = sample_pairs_geometric(s, per_bucket, rng)
        target = read_blocks(s['packed'], r, c, s['q'])
        for j in range(3):
            v = target[bucket == j]; squares[j] += np.sum(v * v); counts[j] += v.size
        diagonal = target[bucket == 0]; pivots = np.diagonal(diagonal, axis1=1, axis2=2)
        if np.any(pivots <= 0):
            raise ValueError('NONPOSITIVE_REFERENCE_PIVOT')
        logs.append(np.log(pivots).ravel())
        v = diagonal[:, np.tril_indices(3, -1)[0], np.tril_indices(3, -1)[1]]
        squares[3] += np.sum(v * v); counts[3] += v.size
        bindings.append(dict(seat=s['seat'], split=s['split'], label=str(s['label'])))
    log = np.concatenate(logs)
    if log.min() <= log_pivot_window[0] or log.max() >= log_pivot_window[1]:
        raise ValueError('REFERENCE_OUTSIDE_LOG_PIVOT_WINDOW')
    rms = np.sqrt(squares / counts)
    stats = Conditioning(float(log.mean()), float(log.std()), float(rms[3]), float(rms[0]), float(rms[1]), float(rms[2]))
    stats.validate()
    return stats, dict(source='training-label aggregate only; buckets are diagonal / near / far',
                       samples=bindings, per_sample_draws_per_bucket=per_bucket, seed=seed,
                       log_pivot_min=float(log.min()), log_pivot_max=float(log.max()), values=asdict(stats),
                       field_map='same_patch_rms = near bucket, cross_patch_rms = far bucket')


# ----------------------------------------------------------------------------- inference
@torch.no_grad()
def predict_full_q(model, sample, tables, conditioning, g_view, chunk, out_device, ctx_t=None):
    """Symmetric q x q M_q_hat in the ORIGINAL frame: the model sees the cell rotated by g_view."""
    model.eval()
    ctx = tables.rotate_context(ctx_t if ctx_t is not None else sample['ctx_t'], g_view)
    enc = model.encode(ctx)
    q = sample['q']; count = sample['count']; dev = ctx['pos'].device
    M = torch.zeros((q, q), dtype=torch.float64, device=out_device)
    total = count * (count + 1) // 2
    a = torch.arange(3, device=out_device)[None, :, None]; b = torch.arange(3, device=out_device)[None, None, :]
    g_back = int(CG.INVERSE[g_view])
    for lo in range(0, total, chunk):
        index = np.arange(lo, min(total, lo + chunk), dtype=np.int64)
        rows = np.floor((np.sqrt(8 * index + 1) - 1) / 2).astype(np.int64); cols = index - rows * (rows + 1) // 2
        r = torch.as_tensor(rows, device=dev); c = torch.as_tensor(cols, device=dev)
        blocks = model.decode(enc, ctx, r, c, conditioning).double()
        diag = r == c
        blocks[diag] = blocks[diag] + torch.tril(blocks[diag], -1).transpose(1, 2)   # full symmetric diagonal block
        blocks = tables.rotate_blocks(blocks, g_back).to(out_device)
        r = r.to(out_device); c = c.to(out_device)
        M[3 * r[:, None, None] + a, 3 * c[:, None, None] + b] = blocks
        M[3 * c[:, None, None] + a, 3 * r[:, None, None] + b] = blocks.transpose(1, 2)
    if float((M - M.T).abs().max()) != 0.0:
        raise ValueError('ASSEMBLY_NOT_SYMMETRIC')
    return M


def view_frame(sample, g_applied, canonical):
    """The frame the model sees for the cell rotated by g_applied."""
    if not canonical:
        return g_applied
    g_c, _, _ = CG.canonical(CG.permute_corners(sample['ctx']['corners'], g_applied))
    return int(CG.COMPOSE[g_c, g_applied])


@torch.no_grad()
def evaluate(model, sample, tables, conditioning, out, args, RigidQuotient):
    out.mkdir(exist_ok=False); dev = torch.device(args.eval_device); train_dev = sample['ctx_t']['pos'].device
    tick = time.perf_counter()
    g0 = view_frame(sample, 0, args.canonical)
    M = predict_full_q(model, sample, tables, conditioning, g0, args.inference_chunk, dev)
    inference = dict(seconds=time.perf_counter() - tick, view_frame=g0, blocks=sample['count'] * (sample['count'] + 1) // 2,
                     scalar_entries=sample['q'] * (sample['q'] + 1) // 2, complete=True)
    pack(M, out / 'MQ_PRED_UPPER.npy')
    q, d = sample['q'], sample['d']
    gate = response_gate(M, sample['bank'], q) if sample.get('bank') is not None else None
    label = unpack(sample['label'], q, dev)
    label = label + torch.triu(label, 1).T      # unpack fills the upper triangle only; M is full symmetric
    e_factor = float((M - label).norm() / label.norm())
    diag_rel = float(((M.diagonal() - label.diagonal()) / label.diagonal()).abs().max())
    del label; gc.collect()
    rigid = torch.from_numpy(sample['cache']['rigid']).to(dev)
    rigid_leak = float((M @ rigid).norm() / (M.norm() * rigid.norm()))
    Qt = RigidQuotient(rigid, torch.from_numpy(sample['cache']['order']).to(dev))
    Mhat = Qt(Qt(M).T.contiguous())                       # B M_q_hat B^T   (d x d)
    del M; gc.collect()
    R = unpack(sample['whitener'], d, dev)
    C = Mhat @ R.T; W = C.T @ C; del C
    W = (W + W.T) * .5
    tick_e = time.perf_counter(); nu, V = torch.linalg.eigh(W); eig_seconds = time.perf_counter() - tick_e
    ids = torch.cat((torch.arange(8, device=dev), torch.arange(d - 8, d, device=dev)))
    extreme = V[:, ids]; res = (W @ extreme - extreme * nu[ids]).norm(dim=0) / nu[ids].abs()
    del W, V, extreme; gc.collect()
    positive = bool((nu > 0).all())
    mu = (1.0 / nu).flip(0) if positive else None
    spectrum = dict(seat=sample['seat'], head='equi_mq', positive_numerical_spectrum=positive,
                    maximum_extreme_eigenpair_residual=float(res.max()), eigen_seconds=eig_seconds,
                    factor_relative=e_factor, diagonal_relative_max=diag_rel, rigid_nullspace_leak=rigid_leak)
    if positive:
        spectrum.update(mu_min=float(mu[0]), mu_max=float(mu[-1]),
                        eps_op=max(abs(float(mu[0]) - 1), abs(float(mu[-1]) - 1)),
                        under_stiff_factor=1. / float(mu[0]), g=max(float(mu[-1]), 1. / float(mu[0])),
                        D_per_mode=float((mu - mu.log() - 1).sum() / d),
                        below_0_9=int((mu < .9).sum()), above_1_1=int((mu > 1.1).sum()),
                        below_0_97=int((mu < .97).sum()), above_1_03=int((mu > 1.03).sum()),
                        g_definition='max_i max(mu_i, 1/mu_i) with mu = eig(R^-T A_hat R^-1), A_hat = (B M_q_hat B^T)^-2')
        np.save(out / 'EIGENVALUES.npy', mu.cpu().numpy())
    del nu; gc.collect()
    # A_hat, only for e_A and the assembled physics test
    Minv = torch.linalg.inv(Mhat); del Mhat
    Ahat = Minv @ Minv; del Minv
    Ahat = (Ahat + Ahat.T) * .5
    Astar = R.T @ R
    spectrum['e_A'] = float((Ahat - Astar).norm() / Astar.norm()); del Astar, R; gc.collect()
    chol, info = torch.linalg.cholesky_ex(Ahat, upper=True)
    spectrum['A_hat_spd'] = int(info) == 0
    if int(info) == 0:
        pack(chol, out / 'A_PRED_UPPER.npy')
    del chol, Ahat; gc.collect()
    if dev.type == 'cuda':
        torch.cuda.empty_cache()
    write(out / 'SPECTRUM.json', spectrum)
    print(json.dumps(dict(phase='evaluation', seat=sample['seat'], g=spectrum.get('g'), mu_min=spectrum.get('mu_min'),
                          mu_max=spectrum.get('mu_max'), e_A=spectrum['e_A'], factor_relative=e_factor)), flush=True)
    # --- probes ---------------------------------------------------------------
    probes = dict(view_frame=g0, canonical=args.canonical, augment=args.augment)
    M0 = unpack(out / 'MQ_PRED_UPPER.npy', q, dev); M0 = M0 + torch.triu(M0, 1).T
    norm0 = float(M0.norm())
    rng = np.random.default_rng(args.seed + 1)
    chosen = [int(g) for g in rng.choice(np.arange(1, CG.ORDER), size=min(args.probe_rotations, CG.ORDER - 1), replace=False)]
    rotation = []
    if args.canonical:
        probes['rotation_consistency'] = []
        probes['rotation_consistency_note'] = ('vacuous under --canonical: with a trivial corner stabiliser '
                                               'view_frame is constant in g, so the probe would re-run a '
                                               'bit-identical inference and report 0 whatever the model does')
        chosen = []
    for g in chosen:
        Mg = predict_full_q(model, sample, tables, conditioning, view_frame(sample, g, args.canonical), args.inference_chunk, dev)
        rotation.append(dict(g=g, proper=bool(CG.DET[g] > 0), view_frame=view_frame(sample, g, args.canonical),
                             relative_discrepancy=float((Mg - M0).norm() / norm0),
                             diagonal_relative_discrepancy=float(((Mg.diagonal() - M0.diagonal()) / M0.diagonal()).abs().max())))
        del Mg; gc.collect()
    probes['rotation_consistency'] = rotation
    probes['rotation_consistency_definition'] = ('predict the cell rotated by g, map the blocks back with g^-1, compare with the '
                                                 'unrotated prediction: ||M_g - M_0||_F / ||M_0||_F; 0 means exactly equivariant')
    pi = rng.permutation(sample['count'])
    ctx_p = to_torch(permute_nodes(sample['ctx'], pi), train_dev)
    Mp = predict_full_q(model, sample, tables, conditioning, g0, args.inference_chunk, dev, ctx_t=ctx_p)
    dof = (3 * torch.as_tensor(pi, device=dev)[:, None] + torch.arange(3, device=dev)[None]).ravel()
    back = torch.empty_like(Mp); back[dof[:, None], dof[None, :]] = Mp
    probes['index_shuffle_relative_discrepancy'] = float((back - M0).norm() / norm0)
    probes['index_shuffle_definition'] = 'shuffle node indices, predict, un-shuffle; must vanish for an index-free model'
    del Mp, back, M0; gc.collect()
    write(out / 'PROBES.json', probes)
    result = dict(seat=sample['seat'], split=sample['split'], inference=inference, spectrum=spectrum, probes=probes,
                  response_gate=gate,
                  seconds=time.perf_counter() - tick)
    write(out / 'RESULT.json', result)
    print(json.dumps(dict(phase='probes', seat=sample['seat'], rotation=[r['relative_discrepancy'] for r in rotation],
                          index_shuffle=probes['index_shuffle_relative_discrepancy'])), flush=True)
    return result


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', type=Path, default=Path('/root/autodl-tmp/CUTFEM_SUPERELEMENT_LABELS/V1_LABELS.json'))
    ap.add_argument('--source', type=Path, default=Path('/root/autodl-tmp/CLAUDE_SQRTHEAD_20260917/src_v5'),
                    help='frozen tree providing stage_cutfem_m4.factors and .quotient')
    ap.add_argument('--seats', type=int, nargs='+', required=True); ap.add_argument('--known-328', action='store_true')
    ap.add_argument('--train-seats', type=int, nargs='*', default=None); ap.add_argument('--eval-seats', type=int, nargs='*', default=None)
    ap.add_argument('--steps', type=int, default=256); ap.add_argument('--warmup', type=int, default=16)
    ap.add_argument('--pairs-per-bucket', type=int, default=8192); ap.add_argument('--calibrate-pairs', type=int, default=32768)
    ap.add_argument('--near-cells', type=float, default=2.0, help='near-bucket radius in background cells')
    ap.add_argument('--augment', action='store_true', help='random cube symmetry per step (48 elements)')
    ap.add_argument('--proper-only', action='store_true', help='augment with the 24 rotations only')
    ap.add_argument('--canonical', action='store_true', help='always present the cell in its canonical frame')
    ap.add_argument('--diagonal-loss', choices=['absolute', 'log'], default='log')
    ap.add_argument('--bf16', action='store_true', help='bf16 autocast for the training forward pass (losses in fp32; evaluation stays fp32)')
    ap.add_argument('--scale-importance', type=float, default=0.0,
                    help='tilt the near/far draws by (s_r s_c)^alpha and weight the diagonal by '
                         's^alpha, s = the mean pivot of the label diagonal block (0 = uniform)')
    ap.add_argument('--pivot-asymmetry', type=float, default=0.0,
                    help='extra weight on pivots predicted softer than the truth; over-softness is '
                         'the direction the assembled compliance sum_i w_i/mu_i can blow up on')
    ap.add_argument('--response-weight', type=float, default=0.0,
                    help='weight of the smooth-face-load response term ||(M_hat-M)f||^2/||Mf||^2 (0 = off)')
    ap.add_argument('--response-rows', type=int, default=8, help='rows sampled per step for the response term')
    ap.add_argument('--column-weight', type=float, default=0.0,
                    help='weight of the point-load (per-column relative) term (0 = off)')
    ap.add_argument('--column-count', type=int, default=32, help='columns sampled per step for the column term')
    ap.add_argument('--column-rows', type=int, default=256, help='rows sampled per column')
    ap.add_argument('--column-tilt', type=float, default=0.0,
                    help='sample columns with probability proportional to colnorm^-tilt (0 = uniform over coordinates)')
    ap.add_argument('--response-per-face', type=int, default=16, help='training loads per face in the bank')
    ap.add_argument('--response-seed', type=int, default=20260919)
    ap.add_argument('--response-bank-dir', type=Path, default=None, help='cache of exact bank responses (default OUTPUT/../RESPONSE_BANK)')
    ap.add_argument('--tail-beta', type=float, default=0.0,
                    help='smooth-max sharpness over the per-coordinate diagonal error, added to its '
                         'mean; g is a max and a handful of coordinates set it')
    ap.add_argument('--state-dim', type=int, default=192); ap.add_argument('--width', type=int, default=768)
    ap.add_argument('--depth', type=int, default=4); ap.add_argument('--volume-channels', type=int, default=32)
    ap.add_argument('--segment-samples', type=int, default=8)
    ap.add_argument('--lr', type=float, default=2e-4); ap.add_argument('--min-lr', type=float, default=1e-5)
    ap.add_argument('--weight-decay', type=float, default=1e-5); ap.add_argument('--clip', type=float, default=10.)
    ap.add_argument('--inference-chunk', type=int, default=8192); ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--device', default='cuda:0'); ap.add_argument('--eval-device', default='cuda:0')
    ap.add_argument('--probe-rotations', type=int, default=4)
    ap.add_argument('--checkpoint-every', type=int, default=64)
    ap.add_argument('--seed', type=int, default=20260917); ap.add_argument('--source-sha', required=True)
    ap.add_argument('--output', type=Path, required=True); args = ap.parse_args()
    if args.augment and args.canonical:
        raise ValueError('AUGMENT_AND_CANONICAL_ARE_ALTERNATIVES')
    if args.proper_only and not args.augment:
        raise ValueError('PROPER_ONLY_REQUIRES_AUGMENT')
    sys.path.insert(0, str(args.source))
    from stage_cutfem_m4.factors import read_blocks, bucket_loss, Conditioning
    from stage_cutfem_m4.quotient import RigidQuotient
    args.output.mkdir(parents=True, exist_ok=False); start = time.perf_counter()
    here = Path(__file__).parent
    protocol = dict(stage='EQUI_MQ_INDEX_FREE_PILOT', source_sha=args.source_sha,
                    source_files={p.name: sha256(p) for p in sorted(here.glob('*.py'))},
                    frozen_tree=str(args.source), label='MQ_UPPER.npy = packed upper of B^T A^{-1/2} B (q x q)',
                    seats=args.seats, known_328_diagnostic=args.known_328,
                    model=dict(state=args.state_dim, width=args.width, depth=args.depth, volume_channels=args.volume_channels,
                               segment_samples=args.segment_samples, volume_encoder='U-Net 32-16-8-4, avg-pool/nearest, zero pad',
                               index_free=True, transposition_symmetric=True),
                    augment=args.augment, proper_only=args.proper_only, canonical=args.canonical, group_order=CG.ORDER,
                    buckets='diagonal / near (<= near_cells background cells) / far', near_cells=args.near_cells,
                    field_scales='fixed nominal per-channel scales in equi.context; no per-case statistics',
                    diagonal_loss=args.diagonal_loss, log_pivot_window=[-20., 20.],
                    optimizer='AdamW', peak_lr=args.lr, min_lr=args.min_lr, weight_decay=args.weight_decay, clip_grad_norm=args.clip,
                    steps=args.steps, warmup=args.warmup, schedule='cosine', pairs_per_bucket=args.pairs_per_bucket,
                    calibrate_pairs=args.calibrate_pairs, seed=args.seed, checkpoint_every=args.checkpoint_every,
                    manifest_sha256=sha256(args.manifest), device=args.device, eval_device=args.eval_device,
                    presented_seats='all prepared seats' if args.train_seats is None else args.train_seats,
                    evaluated_seats='all prepared seats' if args.eval_seats is None else args.eval_seats,
                    acceptance='g = max(mu, 1/mu) on the frozen quotient; assembled physics via sens_model on A_PRED_UPPER')
    write(args.output / 'PROTOCOL.json', protocol)
    try:
        torch.set_num_threads(args.threads); torch.manual_seed(args.seed); np.random.seed(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        device = torch.device(args.device)
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        rows = selected_rows(args.manifest, args.seats, args.known_328)
        samples = [prepare(r, args, device) for r in rows]
        write(args.output / 'INPUTS.json', [s['record'] for s in samples])
        if args.train_seats is None:
            train = samples
        else:
            wanted = set(args.train_seats)
            if not wanted <= set(args.seats):
                raise ValueError('TRAIN_SEAT_NOT_PREPARED')
            train = [s for s in samples if s['seat'] in wanted]
        if not train:
            raise ValueError('EMPTY_TRAINING_SET')
        if args.canonical:
            tied = [s['seat'] for s in samples if s['record']['canonical_ties'] > 1]
            if tied:
                raise ValueError(f'CANONICAL_FRAME_NOT_UNIQUE_FOR_SEATS {tied}')
        presented = sorted(s['seat'] for s in train)
        write(args.output / 'SPLIT.json', dict(prepared=[s['seat'] for s in samples], presented=presented,
                                               never_presented=[s['seat'] for s in samples if s['seat'] not in presented]))
        conditioning, stats = calibrate_geometric(train, args.calibrate_pairs, read_blocks, Conditioning)
        write(args.output / 'CONDITIONING.json', stats)
        # The conditioning constants are calibrated on UNIFORM draws whatever the sampler does, so
        # the loss scale stays comparable across arms.
        selftest = loss_selftest(bucket_loss, Conditioning, device)
        scale_report = []
        for s_ in samples:
            scales = node_scales(s_, read_blocks)
            s_['tilt'] = scale_tilt(scales, s_['near_codes'], s_['count'], args.scale_importance)
            s_['weight'] = None if s_['tilt'] is None else \
                torch.as_tensor(s_['tilt']['weight'], dtype=torch.float32, device=device)
            scale_report.append(dict(seat=s_['seat'], count=int(s_['count']),
                                     pivot_scale_min=float(scales.min()), pivot_scale_max=float(scales.max()),
                                     pivot_scale_median=float(np.median(scales)),
                                     pivot_scale_p999=float(np.percentile(scales, 99.9)),
                                     tilted=bool(s_['tilt'] is not None)))
        bank_dir = args.response_bank_dir or (args.output.parent / 'RESPONSE_BANK')
        response_report = dict(selftest=response_selftest(device), weight=args.response_weight,
                               rows=args.response_rows, per_face=args.response_per_face, seed=args.response_seed,
                               column_weight=args.column_weight, column_count=args.column_count,
                               column_rows=args.column_rows, column_tilt=args.column_tilt)
        tick_b = time.perf_counter()
        for s_ in samples:
            s_['bank'] = response_bank(s_, args.response_per_face, args.response_seed, bank_dir)
        response_report.update(bank_seconds=time.perf_counter() - tick_b,
                               train_loads=[len(s_['bank']['train']) for s_ in samples],
                               held_loads=[len(s_['bank']['held']) for s_ in samples])
        write(args.output / 'LOSS_SETUP.json', dict(
            response=response_report,
            loss_selftest=selftest, scale_importance=args.scale_importance,
            pivot_asymmetry=args.pivot_asymmetry, tail_beta=args.tail_beta, scales=scale_report,
            note='equi_loss equals the frozen bucket_loss bit for bit at neutral knobs; the tilt '
                 'changes which pairs are drawn, not the conditioning constants'))
        n = samples[0]['ctx']['n']
        if any(s['ctx']['n'] != n for s in samples):
            raise ValueError('MIXED_GRID_SIZES')
        tables = GroupTables(n, device, len(samples[0]['ctx']['vol_scalar']))
        model = EquiModel(n=n, volume_in=int(samples[0]['ctx_t']['volume'].shape[0]),
                          volume_channels=args.volume_channels, state_dim=args.state_dim, width=args.width,
                          depth=args.depth, segment_samples=args.segment_samples,
                          near_radius=float(samples[0]['radius'])).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        parameter_count = sum(p.numel() for p in model.parameters())
        rng = np.random.default_rng(args.seed)
        # a separate stream for the group element, so the pair sequence is identical across
        # arms and the orientation variable is the only difference between them
        g_rng = np.random.default_rng(args.seed ^ 0x9E3779B9)
        losses = []; training_start = sync(device)
        elements = CG.PROPER if args.proper_only else np.arange(CG.ORDER)
        print(json.dumps(dict(phase='training_start', parameters=parameter_count, seats=args.seats,
                              augment=args.augment, canonical=args.canonical)), flush=True)
        for step in range(1, args.steps + 1):
            tick = sync(device); sample = train[(step - 1) % len(train)]
            lr = args.lr * step / args.warmup if step <= args.warmup else \
                args.min_lr + (args.lr - args.min_lr) * .5 * (1 + math.cos(math.pi * (step - args.warmup) / max(1, args.steps - args.warmup)))
            for group in opt.param_groups:
                group['lr'] = lr
            g = int(g_rng.choice(elements)) if args.augment else (sample['canonical_g'] if args.canonical else 0)
            model.train(); opt.zero_grad(set_to_none=True)
            r, c, b = sample_pairs_geometric(sample, args.pairs_per_bucket, rng, sample['tilt'])
            target = torch.as_tensor(read_blocks(sample['packed'], r, c, sample['q']), dtype=torch.float32, device=device)
            bucket = torch.as_tensor(b, device=device)
            target = tables.rotate_target_blocks(target, bucket, g)
            ctx = tables.rotate_context(sample['ctx_t'], g)
            autocast = torch.autocast('cuda', dtype=torch.bfloat16, enabled=bool(args.bf16 and device.type == 'cuda'))
            autocast.__enter__()
            enc = model.encode(ctx)
            pred = model.decode(enc, ctx, torch.as_tensor(r, device=device), torch.as_tensor(c, device=device), conditioning).float()
            weight = None
            if sample['weight'] is not None:
                weight = torch.cat((sample['weight'],
                                    torch.ones(2 * args.pairs_per_bucket, device=device)))
            loss, parts = equi_loss(pred, target, bucket, conditioning, weight, args.pivot_asymmetry,
                                    args.tail_beta, args.diagonal_loss)
            if args.response_weight > 0:
                bank = sample['bank']
                li = int(rng.integers(0, len(bank['train'])))
                rows_r = rng.integers(0, sample['count'], size=args.response_rows)
                Qg = torch.as_tensor(CG.Q_ALL[g], dtype=torch.float32, device=device)
                resp = response_term(model, enc, ctx, conditioning, bank, li, rows_r, g, Qg, device)
                loss = loss + args.response_weight * resp
                parts = torch.cat((parts, resp.detach()[None]))
            if args.column_weight > 0:
                bank = sample['bank']
                if args.column_tilt:
                    pc = bank['colnorm2'] ** (-.5 * args.column_tilt); pc = pc / pc.sum()
                    cols_c = rng.choice(sample['count'], size=args.column_count, replace=True, p=pc)
                else:
                    cols_c = rng.integers(0, sample['count'], size=args.column_count)
                rows_c = rng.integers(0, sample['count'], size=args.column_rows)
                colt = column_term(model, enc, ctx, conditioning, sample, read_blocks, bank, cols_c, rows_c, g, tables, device)
                loss = loss + args.column_weight * colt
                parts = torch.cat((parts, colt.detach()[None]))
            autocast.__exit__(None, None, None)
            if not torch.isfinite(loss):
                raise ValueError('NONFINITE_LOSS_NO_REPAIR')
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip, error_if_nonfinite=True); opt.step()
            record = dict(step=step, seat=sample['seat'], g=g, loss=float(loss.detach()), buckets=parts.cpu().tolist(), lr=lr,
                          gradient_norm_before_clip=float(grad), seconds=sync(device) - tick, wall_seconds=time.perf_counter() - start)
            losses.append(record)
            with (args.output / 'TRAIN.jsonl').open('a') as f:
                f.write(json.dumps(record) + '\n')
            if step <= 2 or step % 16 == 0:
                print(json.dumps(record), flush=True)
            if step % args.checkpoint_every == 0 or step == args.steps:
                tmp = args.output / 'CHECKPOINT.tmp'
                torch.save(dict(model=model.state_dict(), optimizer=opt.state_dict(), step=step, conditioning=asdict(conditioning),
                                protocol=protocol, near_radius=model.near_radius,
                                numpy_rng=rng.bit_generator.state, group_rng=g_rng.bit_generator.state,
                                torch_rng=torch.get_rng_state()), tmp)
                tmp.replace(args.output / f'CHECKPOINT_{step:06d}.pt')
        training_seconds = sync(device) - training_start
        del opt, pred, loss, target; gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        chosen = samples if args.eval_seats is None else [s for s in samples if s['seat'] in set(args.eval_seats)]
        if args.eval_seats is not None and len(chosen) != len(set(args.eval_seats)):
            raise ValueError('EVAL_SEAT_NOT_IN_RUN')
        results = [evaluate(model, s, tables, conditioning, args.output / f'EVAL_{s["seat"]:04d}', args, RigidQuotient) for s in chosen]
        result = dict(status='EQUI_PILOT_COMPLETE', parameters=parameter_count, steps=args.steps, training_seconds=training_seconds,
                      seconds=time.perf_counter() - start,
                      peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated() if device.type == 'cuda' else None,
                      first_loss=losses[0]['loss'], last_loss=losses[-1]['loss'],
                      median_step_seconds=float(np.median([r['seconds'] for r in losses[2:]])) if len(losses) > 2 else None,
                      evaluations={str(r['seat']): dict(g=r['spectrum'].get('g'), mu_min=r['spectrum'].get('mu_min'),
                                                       mu_max=r['spectrum'].get('mu_max'), e_A=r['spectrum']['e_A'],
                                                       factor_relative=r['spectrum']['factor_relative'],
                                                       response_gate=r.get('response_gate'),
                                                       rotation=[p['relative_discrepancy'] for p in r['probes']['rotation_consistency']],
                                                       index_shuffle=r['probes']['index_shuffle_relative_discrepancy']) for r in results},
                      interpretation='run completion does not imply accuracy, generalisation or physical validation')
        write(args.output / 'RESULT.json', result); print(json.dumps(result), flush=True)
    except BaseException as error:
        write(args.output / 'FAILURE.json', dict(error=str(error), traceback=traceback.format_exc(), seconds=time.perf_counter() - start))
        raise


if __name__ == '__main__':
    main()
