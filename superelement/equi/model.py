"""An index-free, geometry-only network for the q-space label M_q = B^T A^{-1/2} B.

What is gone from the legacy V7T2 model and why: the node GRU over Morton order, the
patch GRU, the elimination-position feature, the 18 Householder reflector entries and
the patch-pair tables all told the old network *where a coordinate sits in Codex's
elimination*, which is exactly the information a Cholesky emulator needs and exactly
what a physical operator does not depend on.  Here a node is its position, its faces
and the fields at it; a pair is its two nodes, the segment between them, and the
material sampled along that segment from a volumetric encoding of the cell.

Symmetry: every input transforms under the cube group as `equi.context.rotate_context`
specifies, and the block output transforms as Q B Q^T.  This model is made equivariant
by training (augmentation) or by canonicalisation, not by construction; the volume
encoder uses only 2x2x2 average pooling, nearest upsampling and zero padding, all of
which commute exactly with the 48 symmetries, so the only learned non-equivariance is
in the convolution kernels themselves (the tensor-basis decoder and G-CNN are the
next step).  Transposition is exact by construction: B_ij = (f(i,j) + f(j,i)^T)/2.

The diagonal block keeps the frozen head convention: strict lower entries scaled by the
diagonal-lower RMS, pivots as exp(log_mean + log_std * raw) bounded smoothly to a
window, strict upper zero.  That is what factors.read_blocks masks the label to.
"""
from __future__ import annotations

import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from . import cubic_group as CG
from .context import (VOLUME_SCALAR_NAMES, VOLUME_VECTOR_NAMES, NODE_SCALAR_NAMES,
                      NODE_VECTOR_NAMES, NODE_TENSOR_NAMES, rotate_plane)

TWO_PI = 2.0 * math.pi


def smooth_interval_bound(value, lower, upper):
    """Asymptotic to lower/upper, identity well inside; a bound, not a clamp of the label."""
    return lower + F.softplus(value - lower) - F.softplus(value - upper)


# ----------------------------------------------------------------------------- tensors
def to_torch(context, device):
    """The typed numpy context as float32 tensors; volume channels stacked (scalars, then vectors)."""
    vol = np.concatenate((context['vol_scalar'], context['vol_vector'].reshape(-1, *context['vol_scalar'].shape[1:])))
    f32 = lambda k: torch.as_tensor(context[k], dtype=torch.float32, device=device)
    return dict(n=int(context['n']), count=int(context['count']),
                pos=f32('pos'), faces=f32('faces'), node_scalar=f32('node_scalar'),
                node_vector=f32('node_vector'), node_tensor=f32('node_tensor'),
                volume=torch.as_tensor(vol, dtype=torch.float32, device=device),
                corners=f32('corners'), plane=f32('plane'), known=f32('known'),
                support_pos=f32('support_pos'), support_coefficients=f32('support_coefficients'),
                support_rows=torch.as_tensor(context['support_rows'], dtype=torch.long, device=device))


class GroupTables:
    """Per-device tensors of the cube group action for augmentation on the fly."""

    def __init__(self, n, device, n_scalar_channels=len(VOLUME_SCALAR_NAMES)):
        self.n = n
        self.n_scalar = int(n_scalar_channels)
        self.Q = torch.as_tensor(CG.Q_ALL, dtype=torch.float32, device=device)            # (48, 3, 3)
        self.cell_src = torch.as_tensor(np.stack([CG.cell_source_index(n, g) for g in range(CG.ORDER)]),
                                        dtype=torch.long, device=device)               # (48, n^3)
        self.corner_perm = torch.as_tensor(CG.CORNER_PERM, dtype=torch.long, device=device)
        self.face_perm = torch.as_tensor(CG.FACE_PERM, dtype=torch.long, device=device)
        self.inverse = CG.INVERSE

    def rotate_context(self, ctx, g):
        if g == 0:
            return ctx
        Q = self.Q[g]
        out = dict(ctx)
        out['pos'] = (ctx['pos'] - .5) @ Q.T + .5
        faces = torch.empty_like(ctx['faces']); faces[:, self.face_perm[g]] = ctx['faces']; out['faces'] = faces
        out['node_vector'] = ctx['node_vector'] @ Q.T
        out['node_tensor'] = torch.einsum('ab,pkbc,dc->pkad', Q, ctx['node_tensor'], Q)
        out['support_pos'] = (ctx['support_pos'] - .5) @ Q.T + .5
        a = Q @ ctx['plane'][:3]
        out['plane'] = torch.cat((a, (ctx['plane'][3:] + .5 * (a - ctx['plane'][:3]).sum())))
        vol = ctx['volume']
        flat = vol.reshape(vol.shape[0], -1)[:, self.cell_src[g]]
        scal = flat[:self.n_scalar]
        vec = flat[self.n_scalar:].reshape(-1, 3, flat.shape[1])
        vec = torch.einsum('ab,cbk->cak', Q, vec).reshape(-1, flat.shape[1])
        out['volume'] = torch.cat((scal, vec)).reshape(vol.shape)
        corners = torch.empty_like(ctx['corners']); corners[self.corner_perm[g]] = ctx['corners']; out['corners'] = corners
        return out

    def rotate_blocks(self, blocks, g):
        Q = self.Q[g].to(blocks.dtype)
        return torch.einsum('ab,pbc,dc->pad', Q, blocks, Q)

    def rotate_target_blocks(self, target, bucket, g):
        """Label blocks as read_blocks returns them: diagonal blocks are lower-masked, so
        rebuild the symmetric block, rotate, and re-mask; off-diagonal blocks are full."""
        if g == 0:
            return target
        diag = bucket == 0
        full = target.clone()
        full[diag] = full[diag] + torch.tril(full[diag], -1).transpose(1, 2)
        full = self.rotate_blocks(full, g)
        full[diag] = torch.tril(full[diag])
        return full


def permute_nodes_torch(ctx, pi):
    out = dict(ctx)
    for key in ('pos', 'faces', 'node_scalar', 'node_vector', 'node_tensor'):
        out[key] = ctx[key][pi]
    inverse = torch.empty_like(pi); inverse[pi] = torch.arange(len(pi), device=pi.device)
    out['support_rows'] = inverse[ctx['support_rows']]
    return out


# ----------------------------------------------------------------------------- volume
def sample_volume(volume, points):
    """Trilinear samples of a (C, n, n, n) field at unit-cube points (..., 3); cell centres at (i+1/2)/n.

    grid_sample's last coordinate indexes the last array dimension, so the point (x, y, z)
    is passed as (z, y, x).  With align_corners=False the cell centres map exactly onto the
    sample locations; the test below checks it.
    """
    shape = points.shape[:-1]
    grid = (2.0 * points.reshape(1, -1, 1, 1, 3) - 1.0)[..., [2, 1, 0]]
    out = F.grid_sample(volume[None], grid, mode='bilinear', padding_mode='border', align_corners=False)
    return out[0, :, :, 0, 0].T.reshape(*shape, volume.shape[0])


class ConvBlock(nn.Module):
    def __init__(self, cin, cout, groups=8):
        super().__init__()
        self.c1 = nn.Conv3d(cin, cout, 3, padding=1)
        self.n1 = nn.GroupNorm(groups, cout)
        self.c2 = nn.Conv3d(cout, cout, 3, padding=1)
        self.n2 = nn.GroupNorm(groups, cout)

    def forward(self, x):
        x = F.silu(self.n1(self.c1(x)))
        return F.silu(self.n2(self.c2(x)))


class VolumeUNet(nn.Module):
    """32 -> 16 -> 8 -> 4 -> back, with only symmetry-commuting resampling."""

    def __init__(self, cin, c):
        super().__init__()
        self.e1 = ConvBlock(cin, c)
        self.e2 = ConvBlock(c, 2 * c)
        self.e3 = ConvBlock(2 * c, 4 * c)
        self.e4 = ConvBlock(4 * c, 4 * c)
        self.d3 = ConvBlock(8 * c, 4 * c)
        self.d2 = ConvBlock(6 * c, 2 * c)
        self.d1 = ConvBlock(3 * c, c)
        self.out = nn.Conv3d(c, c, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(F.avg_pool3d(e1, 2))
        e3 = self.e3(F.avg_pool3d(e2, 2))
        e4 = self.e4(F.avg_pool3d(e3, 2))
        d3 = self.d3(torch.cat((F.interpolate(e4, scale_factor=2, mode='nearest'), e3), 1))
        d2 = self.d2(torch.cat((F.interpolate(d3, scale_factor=2, mode='nearest'), e2), 1))
        d1 = self.d1(torch.cat((F.interpolate(d2, scale_factor=2, mode='nearest'), e1), 1))
        return self.out(d1)


def mlp(cin, width, cout, depth):
    layers = [nn.Linear(cin, width), nn.SiLU()]
    for _ in range(depth - 1):
        layers += [nn.Linear(width, width), nn.SiLU()]
    layers.append(nn.Linear(width, cout))
    return nn.Sequential(*layers)


class ResidualTrunk(nn.Module):
    def __init__(self, cin, width, depth):
        super().__init__()
        self.inp = nn.Linear(cin, width)
        self.blocks = nn.ModuleList([nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.SiLU(),
                                                   nn.Linear(width, width)) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)

    def forward(self, x):
        h = self.inp(x)
        for block in self.blocks:
            h = h + block(h)
        return self.norm(h)


# ----------------------------------------------------------------------------- model
class EquiModel(nn.Module):
    def __init__(self, *, n, volume_in=len(VOLUME_SCALAR_NAMES) + 3 * len(VOLUME_VECTOR_NAMES),
                 n_node_scalar=len(NODE_SCALAR_NAMES), n_node_vector=len(NODE_VECTOR_NAMES),
                 n_node_tensor=len(NODE_TENSOR_NAMES), n_corners=8, n_known=4, n_plane=4,
                 volume_channels=32, state_dim=192, width=768, depth=4, segment_samples=8,
                 near_radius=.0625):
        super().__init__()
        self.n = n
        self.C = volume_channels
        self.K = segment_samples
        self.state_dim = state_dim
        self.volume_encoder = VolumeUNet(volume_in, volume_channels)
        node_in = 3 + 6 + 1 + n_node_scalar + 3 * n_node_vector + 6 * n_node_tensor + volume_channels
        self.node_encoder = mlp(node_in, width // 2, state_dim, 2)
        self.global_encoder = mlp(n_corners + n_known + n_plane + 2 * volume_channels, width // 2, state_dim, 2)
        pair_in = 3 * state_dim + 3 + 1 + 3 + 6 + 6 + 1 + segment_samples * volume_channels + state_dim
        self.trunk = ResidualTrunk(pair_in, width, depth)
        self.diagonal_head = nn.Linear(width, 9)
        self.near_head = nn.Linear(width, 9)
        self.far_head = nn.Linear(width, 9)
        self.log_pivot_lower = -20.0
        self.log_pivot_upper = 20.0
        # the segment sample parameters, symmetric under t -> 1 - t
        self.register_buffer('segment_t', (torch.arange(segment_samples, dtype=torch.float32) + .5) / segment_samples)
        # a buffer, not a bare attribute: it decides both which head runs and which
        # normaliser scales the output, so a checkpoint reloaded without it would
        # silently reroute pairs.  It travels in the state_dict.
        self.register_buffer('near_radius_t', torch.tensor(float(near_radius)))

    @property
    def near_radius(self):
        return float(self.near_radius_t)

    @near_radius.setter
    def near_radius(self, value):
        self.near_radius_t.fill_(float(value))

    def pull_volume(self, vol_feat, ctx):
        """The learned field sampled at each functional's support, weighted by its coefficients.

        For a box-only functional (one node, coefficient 1, abs_sum 1) this is exactly the field
        at that node; for a cut-surface functional it is the coefficient-weighted average over
        its up-to-20 background nodes.  Sampling at positions, not at owning cells, is what
        makes it commute with the cube group (a node on a cell boundary has no covariant owner).
        """
        values = sample_volume(vol_feat, ctx['support_pos']) * ctx['support_coefficients'][:, None]
        out = values.new_zeros((int(ctx['count']), values.shape[1]))
        return out.index_add_(0, ctx['support_rows'], values)

    def node_input(self, ctx, vol_feat):
        # context.py already divides every field by its fixed nominal scale.
        tri = torch.triu_indices(3, 3, device=ctx['node_tensor'].device)
        tensor = ctx['node_tensor'][:, :, tri[0], tri[1]].reshape(len(ctx['pos']), -1)   # 6 per tensor
        return torch.cat((ctx['pos'], ctx['faces'], ctx['faces'].sum(1, keepdim=True), ctx['node_scalar'],
                          ctx['node_vector'].reshape(len(ctx['pos']), -1), tensor,
                          self.pull_volume(vol_feat, ctx)), dim=1)

    def encode(self, ctx):
        vol_feat = self.volume_encoder(ctx['volume'][None])[0]
        node_state = self.node_encoder(self.node_input(ctx, vol_feat))
        pooled = torch.cat((ctx['corners'], ctx['known'], ctx['plane'], vol_feat.mean(dim=(1, 2, 3)),
                           vol_feat.amax(dim=(1, 2, 3))))
        global_state = self.global_encoder(pooled[None])[0]
        return dict(vol_feat=vol_feat, node_state=node_state, global_state=global_state)

    def _ordered_pairs(self, enc, ctx, rows, cols):
        hs = enc['node_state']
        pi, pj = ctx['pos'][rows], ctx['pos'][cols]
        delta = pi - pj
        dist = delta.norm(dim=1, keepdim=True)
        mid = .5 * (pi + pj)
        fi, fj = ctx['faces'][rows], ctx['faces'][cols]
        shared = (fi * fj).sum(1, keepdim=True)
        seg = pj[:, None, :] + self.segment_t[None, :, None] * delta[:, None, :]        # (P, K, 3), from j to i
        along = sample_volume(enc['vol_feat'], seg).reshape(len(rows), -1)
        g = enc['global_state'][None].expand(len(rows), -1)
        return torch.cat((hs[rows], hs[cols], hs[rows] * hs[cols], delta, dist, mid, fi, fj, shared, along, g), dim=1)

    def decode(self, enc, ctx, rows, cols, conditioning):
        """(P, 3, 3) blocks for lower pairs rows >= cols, transposition-symmetric off the diagonal."""
        diag = rows == cols
        off = ~diag
        r_off, c_off = rows[off], cols[off]
        # off-diagonal: both orderings through one trunk pass
        both = self._ordered_pairs(enc, ctx, torch.cat((r_off, c_off)), torch.cat((c_off, r_off)))
        state = self.trunk(both)
        dist = (ctx['pos'][r_off] - ctx['pos'][c_off]).norm(dim=1)
        near = dist <= self.near_radius
        m = len(r_off)
        raw = torch.empty((m, 9), dtype=state.dtype, device=state.device)
        raw[near] = .5 * (self.near_head(state[:m][near]) + self.near_head(state[m:][near]).reshape(-1, 3, 3).transpose(1, 2).reshape(-1, 9))
        raw[~near] = .5 * (self.far_head(state[:m][~near]) + self.far_head(state[m:][~near]).reshape(-1, 3, 3).transpose(1, 2).reshape(-1, 9))
        out = torch.zeros((len(rows), 3, 3), dtype=torch.float32, device=state.device)
        scale = torch.where(near, float(conditioning.same_patch_rms), float(conditioning.cross_patch_rms)).to(raw.dtype)
        out[off] = (raw * scale[:, None]).reshape(-1, 3, 3).float()
        # diagonal
        if bool(diag.any()):
            d_state = self.trunk(self._ordered_pairs(enc, ctx, rows[diag], cols[diag]))
            d_raw = self.diagonal_head(d_state).reshape(-1, 3, 3).float()
            lower = torch.tril(d_raw, -1) * float(conditioning.diagonal_lower_rms)
            log_pivot = float(conditioning.pivot_log_mean) + float(conditioning.pivot_log_std) * torch.diagonal(d_raw, dim1=-2, dim2=-1)
            log_pivot = smooth_interval_bound(log_pivot, self.log_pivot_lower, self.log_pivot_upper)
            out[diag] = lower + torch.diag_embed(torch.exp(log_pivot))
        return out

    def forward(self, ctx, rows, cols, conditioning):
        return self.decode(self.encode(ctx), ctx, rows, cols, conditioning)


# ----------------------------------------------------------------------------- tests
def selftest(device='cpu', seed=20260918, n=8):
    from .context import (compile_from_points, compile_from_trace, rotate_context, permute_nodes,
                          rotate_plane, _random_trace, VOLUME_SCALAR_NAMES)
    from .cubic_group import ORDER
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    report = {}
    # sampling axis order: cell centres reproduce cell values exactly
    vol = torch.rand(3, n, n, n)
    cells = np.array(list(np.ndindex(n, n, n)))
    centres = torch.as_tensor((cells + .5) / n, dtype=torch.float32)
    got = sample_volume(vol, centres)
    want = vol[:, cells[:, 0], cells[:, 1], cells[:, 2]].T
    report['sample_axis_order'] = float((got - want).abs().max())
    assert report['sample_axis_order'] < 1e-6
    # a context, its torch form, and the torch rotation against the numpy one
    top = 2 * n
    grid = np.array(list(np.ndindex(top + 1, top + 1, top + 1)))
    boundary = grid[((grid == 0) | (grid == top)).any(axis=1)]
    corners = rng.random(8) * .1 + .15
    # a cut cell: box-face nodes with nnz 1 plus cut-surface functionals with nnz 2..20
    indptr, support, coef, kind = _random_trace(rng, n, 120, 40)
    ctx = compile_from_trace(indptr, support, coef, kind, corners, (1., .52, 0., .27), n, [1., .3, 1e-4, 1. / n])
    ctx_t = to_torch(ctx, device)
    report['cut_cell_functionals'] = int(ctx['count']); report['cut_cell_nnz'] = int(len(ctx['support_rows']))
    tables = GroupTables(n, device)
    worst = 0.
    for g in range(ORDER):
        a = tables.rotate_context(ctx_t, g)
        b = to_torch(rotate_context(ctx, g), device)
        for key in ('pos', 'faces', 'node_vector', 'node_tensor', 'volume', 'corners', 'plane', 'support_pos'):
            worst = max(worst, float((a[key] - b[key]).abs().max()))
    report['torch_rotation_vs_numpy'] = worst
    assert worst < 1e-6
    # target rotation: g then g^-1 is the identity, and diagonal blocks stay lower-masked
    P = 40
    bucket = torch.as_tensor(np.r_[np.zeros(10, int), np.ones(15, int), 2 * np.ones(15, int)])
    sym = torch.rand(P, 3, 3); sym = sym + sym.transpose(1, 2)
    target = sym.clone(); target[bucket == 0] = torch.tril(target[bucket == 0])
    for g in range(ORDER):
        rt = tables.rotate_target_blocks(target, bucket, g)
        assert int(torch.count_nonzero(torch.triu(rt[bucket == 0], 1))) == 0
        back = tables.rotate_target_blocks(rt, bucket, int(CG.INVERSE[g]))
        assert float((back - target).abs().max()) < 1e-6
        # and the rotated diagonal block is the rotated symmetric block, masked
        full = tables.rotate_blocks(sym[bucket == 0], g)
        assert float((torch.tril(full) - rt[bucket == 0]).abs().max()) < 1e-6
    report['target_rotation'] = 'ok'
    # the model runs, is transposition-symmetric off the diagonal, and index-free
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class Cond:
        pivot_log_mean: float = 0.
        pivot_log_std: float = 1.
        diagonal_lower_rms: float = 1.
        diagonal_block_rms: float = 1.
        same_patch_rms: float = 1.
        cross_patch_rms: float = 1.
    model = EquiModel(n=n, volume_in=ctx_t['volume'].shape[0], volume_channels=8, state_dim=16, width=32,
                      depth=2, segment_samples=4, near_radius=3. / n).to(device)
    assert 'near_radius_t' in model.state_dict(), 'near_radius must travel in the state_dict'
    model.eval()
    count = ctx['count']
    r = torch.as_tensor(rng.integers(0, count, 60)); c = torch.as_tensor(rng.integers(0, count, 60))
    rows, cols = torch.maximum(r, c), torch.minimum(r, c)
    rows = torch.cat((torch.arange(8), rows)); cols = torch.cat((torch.arange(8), cols))   # force diagonal coverage
    assert int((rows == cols).sum()) >= 8
    with torch.no_grad():
        out = model(ctx_t, rows, cols, Cond())
        swapped = model(ctx_t, cols, rows, Cond())          # the model is told (j, i): must give B^T
    off = rows != cols
    report['transpose_symmetry'] = float((out[off] - swapped[off].transpose(1, 2)).abs().max())
    assert report['transpose_symmetry'] < 1e-5
    diag = out[rows == cols]
    assert int(torch.count_nonzero(torch.triu(diag, 1))) == 0 and bool((torch.diagonal(diag, dim1=-2, dim2=-1) > 0).all())
    pi = rng.permutation(count); inv = np.argsort(pi)
    ctx_p = to_torch(permute_nodes(ctx, pi), device)
    assert bool((ctx_p['support_rows'] == permute_nodes_torch(ctx_t, torch.as_tensor(pi))['support_rows']).all())
    with torch.no_grad():
        out_p = model(ctx_p, torch.as_tensor(inv[rows.numpy()]), torch.as_tensor(inv[cols.numpy()]), Cond())
    report['index_shuffle'] = float((out_p - out).abs().max())
    assert report['index_shuffle'] < 1e-5
    # the resampling claim: pooling and nearest upsampling must commute with all 48 elements
    x = torch.rand(2, n, n, n)
    worst_pool = worst_up = 0.
    for g in range(ORDER):
        src = torch.as_tensor(CG.cell_source_index(n, g), dtype=torch.long)
        rot = lambda t, s: t.reshape(t.shape[0], -1)[:, s].reshape(t.shape)
        p1 = F.avg_pool3d(rot(x, src)[None], 2)[0]
        p2 = rot(F.avg_pool3d(x[None], 2)[0], torch.as_tensor(CG.cell_source_index(n // 2, g), dtype=torch.long))
        worst_pool = max(worst_pool, float((p1 - p2).abs().max()))
        y = F.avg_pool3d(x[None], 2)[0]
        srch = torch.as_tensor(CG.cell_source_index(n // 2, g), dtype=torch.long)
        u1 = F.interpolate(rot(y, srch)[None], scale_factor=2, mode='nearest')[0]
        u2 = rot(F.interpolate(y[None], scale_factor=2, mode='nearest')[0], src)
        worst_up = max(worst_up, float((u1 - u2).abs().max()))
    report['avg_pool_commutes'] = worst_pool
    report['nearest_upsample_commutes'] = worst_up
    assert worst_pool < 1e-6 and worst_up == 0.0, (worst_pool, worst_up)
    report['parameters'] = sum(p.numel() for p in model.parameters())
    return report


if __name__ == '__main__':
    import json
    print(json.dumps(selftest(), indent=1))
