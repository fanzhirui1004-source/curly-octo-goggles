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
from .context import compile_equi_inputs, permute_nodes
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
    g_canon, _, ties = CG.canonical(ctx['corners'])
    record = dict(seat=int(row['seat']), split=row['split'], q=q, d=d, count=ctx['count'], n=ctx['n'],
                  near_radius=radius, near_pairs=int(len(near_codes)),
                  all_lower_pairs=int(ctx['count'] * (ctx['count'] - 1) // 2), canonical_frame=int(g_canon),
                  canonical_ties=int(ties), bindings=bindings, label=str(label), whitener=str(factor),
                  trace_cache=str(cache_path), corners=ctx['corners'].tolist())
    return dict(seat=int(row['seat']), split=row['split'], q=q, d=d, count=ctx['count'], ctx=ctx,
                ctx_t=to_torch(ctx, device), near_codes=near_codes, radius=radius, canonical_g=int(g_canon),
                packed=np.load(label, mmap_mode='r', allow_pickle=False), label=label, whitener=factor,
                reference=reference, cache=cache, record=record)


def sample_pairs_geometric(sample, per_bucket, rng):
    """Complete diagonal; uniform draws from the near pairs and from the far pairs."""
    count = sample['count']; codes = sample['near_codes']
    diag = np.arange(count, dtype=np.int64)
    near = codes[rng.integers(0, len(codes), size=per_bucket)]
    far = []; needed = per_bucket
    while needed:
        a = rng.integers(0, count, size=2 * needed + 32); b = rng.integers(0, count, size=2 * needed + 32)
        r = np.maximum(a, b); c = np.minimum(a, b); code = r * count + c
        pos = np.searchsorted(codes, code)
        is_near = (pos < len(codes)) & (codes[np.minimum(pos, len(codes) - 1)] == code)
        keep = (r != c) & ~is_near
        code = code[keep][:needed]; far.append(code); needed -= len(code)
    far = np.concatenate(far)
    r = np.concatenate((diag, near // count, far // count)); c = np.concatenate((diag, near % count, far % count))
    bucket = np.repeat(np.arange(3), (count, per_bucket, per_bucket))
    return r, c, bucket


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
        presented = sorted(s['seat'] for s in train)
        write(args.output / 'SPLIT.json', dict(prepared=[s['seat'] for s in samples], presented=presented,
                                               never_presented=[s['seat'] for s in samples if s['seat'] not in presented]))
        conditioning, stats = calibrate_geometric(train, args.calibrate_pairs, read_blocks, Conditioning)
        write(args.output / 'CONDITIONING.json', stats)
        n = samples[0]['ctx']['n']
        if any(s['ctx']['n'] != n for s in samples):
            raise ValueError('MIXED_GRID_SIZES')
        tables = GroupTables(n, device)
        model = EquiModel(n=n, volume_channels=args.volume_channels, state_dim=args.state_dim, width=args.width,
                          depth=args.depth, segment_samples=args.segment_samples).to(device)
        model.near_radius = float(samples[0]['radius'])
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        parameter_count = sum(p.numel() for p in model.parameters())
        rng = np.random.default_rng(args.seed); losses = []; training_start = sync(device)
        elements = CG.PROPER if args.proper_only else np.arange(CG.ORDER)
        print(json.dumps(dict(phase='training_start', parameters=parameter_count, seats=args.seats,
                              augment=args.augment, canonical=args.canonical)), flush=True)
        for step in range(1, args.steps + 1):
            tick = sync(device); sample = train[(step - 1) % len(train)]
            lr = args.lr * step / args.warmup if step <= args.warmup else \
                args.min_lr + (args.lr - args.min_lr) * .5 * (1 + math.cos(math.pi * (step - args.warmup) / max(1, args.steps - args.warmup)))
            for group in opt.param_groups:
                group['lr'] = lr
            g = int(rng.choice(elements)) if args.augment else (sample['canonical_g'] if args.canonical else 0)
            model.train(); opt.zero_grad(set_to_none=True)
            r, c, b = sample_pairs_geometric(sample, args.pairs_per_bucket, rng)
            target = torch.as_tensor(read_blocks(sample['packed'], r, c, sample['q']), dtype=torch.float32, device=device)
            bucket = torch.as_tensor(b, device=device)
            target = tables.rotate_target_blocks(target, bucket, g)
            ctx = tables.rotate_context(sample['ctx_t'], g)
            pred = model(ctx, torch.as_tensor(r, device=device), torch.as_tensor(c, device=device), conditioning)
            loss, parts = bucket_loss(pred, target, bucket, conditioning, None, args.diagonal_loss)
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
                                protocol=protocol, near_radius=model.near_radius, numpy_rng=rng.bit_generator.state,
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
                                                       rotation=[p['relative_discrepancy'] for p in r['probes']['rotation_consistency']],
                                                       index_shuffle=r['probes']['index_shuffle_relative_discrepancy']) for r in results},
                      interpretation='run completion does not imply accuracy, generalisation or physical validation')
        write(args.output / 'RESULT.json', result); print(json.dumps(result), flush=True)
    except BaseException as error:
        write(args.output / 'FAILURE.json', dict(error=str(error), traceback=traceback.format_exc(), seconds=time.perf_counter() - start))
        raise


if __name__ == '__main__':
    main()
