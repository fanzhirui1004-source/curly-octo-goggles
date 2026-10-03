"""Typed, index-free network inputs for one cell -- full or cut -- and the exact action of the
cube group on them.

A trace coordinate is a FUNCTIONAL over background nodes, not a node.  For a full cell every
functional is one node with unit coefficient (`kind` 0, nnz 1) and the distinction is
invisible; for a cut cell the interface is

    (the box-face nodes that survive the macro cut)   kind 0, nnz 1
  U (residual functionals on the macro cut surface)   kind 1, nnz 2..20

and the second family has no single position, only a weighted centroid.  Everything here is
therefore built from the signed CSR, with the box-only case as the special case nnz == 1 --
one code path, so the two cannot drift.

Every array is tagged by how it transforms, so `rotate_context` cannot be wrong in a way the
tests would miss:

    scalars   move with their functional or cell, unchanged
    vectors   move and rotate by Q            (field gradients, the CSR first moment)
    tensors   move and rotate by Q . Q^T      (the CSR second moment)
    faces     permute                          (a functional is "on face f" iff its whole
                                                support is on face f -- exact, not a tolerance)
    corners   permute                          (the 8 trilinear tau values)
    cells     permute                          (the n^3 volume grid, and the CSR support cells)
    plane     a -> Q a,  d -> d + (Q a - a) . 1 / 2, which is exactly what keeps the signed
              distance field invariant: margin'(g . p) = margin(p)

Nothing depends on functional index, elimination order, or the Householder quotient: those
were the memorisation handles of the legacy adapter and they are gone.

One correctness note on the frozen adapter it replaces: its `variance` feature was the three
diagonal entries of the CSR second-moment tensor.  Diagonal entries mix under rotation, so
that feature is not rotation-covariant; the covariant object is the whole symmetric tensor,
which is what this module carries.
"""
from __future__ import annotations

import numpy as np

from .cubic_group import (Q_ALL, map_int_positions, permute_corners, permute_faces, rotate_vectors,
                          rotate_blocks, rotate_scalar_volume, rotate_vector_volume, canonical, INVERSE)

# node channels, in the order the model consumes them
NODE_SCALAR_NAMES = ('tau', 'phi', 'margin', 'grad_tau_norm', 'grad_phi_norm', 'plane_margin',
                     'kind', 'signed_sum', 'abs_sum', 'coeff_norm', 'log1p_nnz', 'support_radius')
NODE_VECTOR_NAMES = ('grad_tau', 'grad_phi', 'first_moment')
NODE_TENSOR_NAMES = ('second_moment',)
VOLUME_SCALAR_NAMES = ('tau', 'phi', 'margin', 'plane_margin')
VOLUME_VECTOR_NAMES = ('grad_tau', 'grad_phi')

# Fixed nominal scales so no channel dominates the first convolution.  Unnormalised, grad_phi
# carried 97 % of the volume input variance -- and phi does not depend on tau at all -- while
# tau carried 0.0024 %.  These are constants, not per-case statistics: the network still sees
# absolute thickness, which is physical.  The three components of a vector channel, and the
# nine of a tensor channel, share one scale, so scaling commutes with the group exactly.
FIELD_SCALE = dict(tau=.2, phi=1.5, margin=1.5, grad_tau_norm=.2, grad_phi_norm=2 * np.pi,
                   plane_margin=1., kind=1., signed_sum=1., abs_sum=1., coeff_norm=1.,
                   log1p_nnz=1., support_radius=.05,
                   grad_tau=.2, grad_phi=2 * np.pi, first_moment=.05, second_moment=.0025)
NODE_SCALAR_SCALE = np.array([FIELD_SCALE[k] for k in NODE_SCALAR_NAMES])
NODE_VECTOR_SCALE = np.array([FIELD_SCALE[k] for k in NODE_VECTOR_NAMES])
NODE_TENSOR_SCALE = np.array([FIELD_SCALE[k] for k in NODE_TENSOR_NAMES])
VOLUME_SCALAR_SCALE = np.array([FIELD_SCALE[k] for k in VOLUME_SCALAR_NAMES])
VOLUME_VECTOR_SCALE = np.array([FIELD_SCALE[k] for k in VOLUME_VECTOR_NAMES])

ONES = np.ones(3)


# ----------------------------------------------------------------------------- the fields
def trilinear(points, corners):
    """tau and its gradient from the 8 corners (index 4bx+2by+bz); identical to adapter.geometry_values."""
    p = np.asarray(points, dtype=np.float64)
    corners = np.asarray(corners, dtype=np.float64)
    tau = np.zeros(p.shape[:-1]); gradient = np.zeros_like(p)
    for index, bit in enumerate(np.ndindex(2, 2, 2)):
        weight = [p[..., j] if bit[j] else 1 - p[..., j] for j in range(3)]
        tau += corners[index] * weight[0] * weight[1] * weight[2]
        for j in range(3):
            other = [k for k in range(3) if k != j]
            gradient[..., j] += corners[index] * (2 * bit[j] - 1) * weight[other[0]] * weight[other[1]]
    return tau, gradient


def schwarz_p(points):
    p = np.asarray(points, dtype=np.float64)
    return np.cos(2 * np.pi * p).sum(axis=-1), -2 * np.pi * np.sin(2 * np.pi * p)


def plane_margin(points, plane):
    """Signed distance to the macro cut plane, normalised by |a| so it is a length."""
    a = np.asarray(plane, dtype=np.float64)[:3]; d = float(plane[3])
    scale = np.linalg.norm(a)
    if scale <= 0:
        raise ValueError('DEGENERATE_CUT_PLANE')
    return (d - np.asarray(points, dtype=np.float64) @ a) / scale


def rotate_plane(plane, g):
    """a -> Q a and d -> d + (Q a - a) . 1 / 2, the unique choice with margin'(g.p) = margin(p)."""
    plane = np.asarray(plane, dtype=np.float64)
    a = Q_ALL[g].astype(np.float64) @ plane[:3]
    return np.concatenate((a, [plane[3] + .5 * (a - plane[:3]) @ ONES]))


def rotate_positions(points, g):
    """p -> Q(p - 1/2) + 1/2.  Exact for any float input: each component is p_j or 1 - p_j."""
    p = np.asarray(points, dtype=np.float64)
    Q = Q_ALL[g]
    axis = np.argmax(Q != 0, axis=1)
    sign = Q[np.arange(3), axis].astype(np.float64)
    moved = p[..., axis]
    return np.where(sign > 0, moved, 1.0 - moved)


def field_values(points, corners, plane):
    """The geometry at a set of points: scalars (6) and vectors (2, 3), unscaled."""
    tau, gtau = trilinear(points, corners)
    phi, gphi = schwarz_p(points)
    margin = tau - np.abs(phi)                      # solid where |phi| <= tau
    pm = plane_margin(points, plane)
    scalars = np.stack((tau, phi, margin, np.linalg.norm(gtau, axis=-1),
                        np.linalg.norm(gphi, axis=-1), pm), axis=-1)
    vectors = np.stack((gtau, gphi), axis=-2)
    return scalars, vectors


# ----------------------------------------------------------------------------- the trace
def csr_descriptors(indptr, support_pos, coefficients):
    """Per-functional position and covariant moments of a signed CSR functional.

    position       |c|-weighted centroid (for nnz == 1 this is the node itself, exactly)
    first_moment   sum_k c_k (x_k - p)                     a vector
    second_moment  sum_k |c_k| (x_k - p) (x_k - p)^T       a symmetric tensor
    """
    indptr = np.asarray(indptr, dtype=np.int64)
    x = np.asarray(support_pos, dtype=np.float64)
    c = np.asarray(coefficients, dtype=np.float64)
    count = len(indptr) - 1
    nnz = np.diff(indptr)
    if nnz.min() < 1:
        raise ValueError('EMPTY_TRACE_FUNCTIONAL')
    rowof = np.repeat(np.arange(count), nnz)
    absc = np.abs(c)
    abs_sum = np.bincount(rowof, weights=absc, minlength=count)
    if not np.all(abs_sum > 0):
        raise ValueError('ZERO_TRACE_FUNCTIONAL')
    pos = np.column_stack([np.bincount(rowof, weights=absc * x[:, j], minlength=count) / abs_sum
                           for j in range(3)])
    delta = x - pos[rowof]
    signed_sum = np.bincount(rowof, weights=c, minlength=count)
    coeff_norm = np.sqrt(np.bincount(rowof, weights=c * c, minlength=count))
    first = np.column_stack([np.bincount(rowof, weights=c * delta[:, j], minlength=count) for j in range(3)])
    second = np.zeros((count, 3, 3))
    for i in range(3):
        for j in range(3):
            second[:, i, j] = np.bincount(rowof, weights=absc * delta[:, i] * delta[:, j], minlength=count)
    radius = np.zeros(count)
    np.maximum.at(radius, rowof, np.linalg.norm(delta, axis=1))
    return dict(pos=pos, nnz=nnz, rowof=rowof, signed_sum=signed_sum, abs_sum=abs_sum,
                coeff_norm=coeff_norm, first_moment=first, second_moment=second, support_radius=radius)


def face_membership(indptr, support_int, top):
    """A functional is on face f iff its WHOLE support is on face f.  Exact, no tolerance."""
    u = np.asarray(support_int, dtype=np.int64)
    on = np.concatenate((u == 0, u == top), axis=1)          # (nnz, 6)
    count = len(indptr) - 1
    rowof = np.repeat(np.arange(count), np.diff(indptr))
    out = np.ones((count, 6), dtype=bool)
    np.logical_and.at(out, rowof, on)
    return out.astype(np.float64)


def compile_from_trace(indptr, support_int, coefficients, kind, corners, plane, n, known):
    """Build the context from the signed CSR in integer background-node coordinates."""
    indptr = np.asarray(indptr, dtype=np.int64)
    support_int = np.asarray(support_int, dtype=np.int64)
    coefficients = np.asarray(coefficients, dtype=np.float64)
    kind = np.asarray(kind, dtype=np.float64)
    top = 2 * n
    count = len(indptr) - 1
    if support_int.ndim != 2 or support_int.shape[1] != 3 or len(support_int) != indptr[-1]:
        raise ValueError('INVALID_TRACE_SUPPORT')
    if indptr[0] != 0 or np.any(np.diff(indptr) < 1) or len(coefficients) != len(support_int):
        raise ValueError('INVALID_SIGNED_TRACE_CSR')
    if support_int.min() < 0 or support_int.max() > top:
        raise ValueError('SUPPORT_OUTSIDE_BACKGROUND')
    if len(kind) != count:
        raise ValueError('KIND_LENGTH')
    d = csr_descriptors(indptr, support_int / top, coefficients)
    pos = d['pos']
    faces = face_membership(indptr, support_int, top)
    scal, vec = field_values(pos, corners, plane)
    node_scalar = np.column_stack((scal, kind, d['signed_sum'], d['abs_sum'], d['coeff_norm'],
                                   np.log1p(d['nnz']), d['support_radius'])) / NODE_SCALAR_SCALE
    node_vector = np.concatenate((vec, d['first_moment'][:, None, :]), axis=1) / NODE_VECTOR_SCALE[:, None]
    node_tensor = d['second_moment'][:, None, :, :] / NODE_TENSOR_SCALE[:, None, None]
    grid = np.stack(np.meshgrid(*[(np.arange(n) + .5) / n] * 3, indexing='ij'), axis=-1)
    vs, vv = field_values(grid, corners, plane)
    keep = [VOLUME_SCALAR_NAMES.index(k) for k in ('tau', 'phi', 'margin', 'plane_margin')]
    src = ('tau', 'phi', 'margin', 'grad_tau_norm', 'grad_phi_norm', 'plane_margin')
    vol_scalar = np.ascontiguousarray(
        (vs[..., [src.index(k) for k in VOLUME_SCALAR_NAMES]] / VOLUME_SCALAR_SCALE).transpose(3, 0, 1, 2))
    vol_vector = np.ascontiguousarray((vv / VOLUME_VECTOR_SCALE[:, None]).transpose(3, 4, 0, 1, 2))
    # The pullback of the learned feature field through the CSR samples the field AT the
    # support positions, trilinearly.  A nearest-cell lookup (what the frozen adapter used) is
    # not group-covariant: a node on a cell boundary has an ambiguous owner, and for an even
    # grid coordinate the two routes -- rotate-then-own and own-then-rotate -- differ by one
    # cell.  Interpolation at a rotating position commutes with the group exactly.
    support_pos = support_int / top
    context = dict(n=int(n), count=int(count), pos=pos, faces=faces, kind=kind,
                   node_scalar=node_scalar, node_vector=node_vector, node_tensor=node_tensor,
                   vol_scalar=vol_scalar, vol_vector=vol_vector,
                   support_rows=d['rowof'].astype(np.int64), support_pos=support_pos,
                   support_coefficients=coefficients / np.repeat(d['abs_sum'], d['nnz']),
                   corners=np.asarray(corners, dtype=np.float64).copy(),
                   plane=np.asarray(plane, dtype=np.float64).copy(),
                   known=np.asarray(known, dtype=np.float64).copy(),
                   box_only=bool(d['nnz'].max() == 1 and np.all(coefficients == 1) and not kind.any()))
    for key, value in context.items():
        if isinstance(value, np.ndarray) and not np.isfinite(value).all():
            raise ValueError('NONFINITE_FEATURE: ' + key)
    return context


def compile_from_points(u_int, corners, n, known, plane=(1., 0., 0., 2.)):
    """The box-only special case: one background node per functional, unit coefficient."""
    u_int = np.asarray(u_int, dtype=np.int64)
    count = len(u_int)
    return compile_from_trace(np.arange(count + 1), u_int, np.ones(count), np.zeros(count),
                              corners, plane, n, known)


def compile_equi_inputs(cache, metadata):
    """From a frozen TRACE_CACHE plus INPUT.json metadata.  Target-free.  Full or cut cell."""
    if metadata['geometry']['representation'] != 'eight_corner_trilinear_unit_box_v1':
        raise ValueError('UNSUPPORTED_GEOMETRY_REPRESENTATION')
    if metadata['geometry']['box_max'] != [1, 1, 1]:
        raise ValueError('UNIT_BOX_REQUIRED')
    n = int(metadata['n'])
    top = 2 * n
    indptr = np.asarray(cache['indptr'], dtype=np.int64)
    indices = np.asarray(cache['indices'], dtype=np.int64)
    coeff = np.asarray(cache['coefficients'], dtype=np.float64)
    nodes = np.asarray(cache['background_nodes'], dtype=np.int64)
    if len(indptr) != len(cache['kind']) + 1 or indptr[0] != 0 or indptr[-1] != len(coeff):
        raise ValueError('INVALID_SIGNED_TRACE_CSR')
    if len(indices) != len(coeff) or indices.min() < 0 or indices.max() >= len(nodes):
        raise ValueError('INVALID_TRACE_SUPPORT')
    support_int = np.column_stack(np.unravel_index(nodes[indices], (top + 1,) * 3)).astype(np.int64)
    order = np.asarray(cache['order'])
    if not np.array_equal(order, (3 * (order[::3] // 3)[:, None] + np.arange(3)).ravel()):
        raise ValueError('COMPILER_ORDER_IS_NOT_XYZ_GROUPED')
    gp = metadata['gp']; material = metadata['material']
    if gp['orders'] != [1, 2] or gp['scope'] != 'module local':
        raise ValueError('UNSUPPORTED_GP_CONTRACT')
    ctx = compile_from_trace(indptr, support_int, coeff, np.asarray(cache['kind'], dtype=np.float64),
                            np.asarray(cache['tau_corners'], dtype=np.float64),
                            np.asarray(cache['cut_plane'], dtype=np.float64), n,
                            [material['E'], material['nu'], gp['gamma'], 1. / n])
    # the compiler's own centroid is the |c|-weighted one, so this binds our CSR reading to it
    if not np.allclose(ctx['pos'], np.asarray(cache['support_centroid'], dtype=np.float64), atol=1e-12, rtol=0):
        raise ValueError('TRACE_CENTROID_BINDING')
    return ctx


def rigid_pullback(cache, n):
    """(L (x) I3) applied to the background rigid modes, on the (2n+1)^3 background grid.

    Exactly equals cache['rigid'] -- verified bit-for-bit on box-only and cut cells alike -- so
    this is the check that the CSR and the frozen rigid array describe the same cell, and the
    one algebraic assumption the augmentation label rests on (span(rigid) is the rigid-body
    space, the unique 6-space closed under blockdiag(Q_g)).

    n is passed, never inferred: a cut cell's surviving node ids need not reach the grid corner,
    so recovering the grid size from max(background_nodes) would be wrong exactly for the cells
    this generalisation is for.
    """
    from scipy import sparse
    nodes = np.asarray(cache['background_nodes'], dtype=np.int64)
    top = 2 * int(n)
    if nodes.min() < 0 or nodes.max() > (top + 1) ** 3 - 1:
        raise ValueError('BACKGROUND_NODE_OUTSIDE_GRID')
    xyz = np.column_stack(np.unravel_index(nodes, (top + 1,) * 3)) / top
    centred = xyz - xyz.mean(axis=0)
    rb = np.zeros((3 * len(xyz), 6))
    for d in range(3):
        rb[d::3, d] = 1.0
        rb[:, 3 + d] = np.cross(np.eye(3)[d], centred).ravel()
    L = sparse.csr_matrix((np.asarray(cache['coefficients'], dtype=np.float64),
                           np.asarray(cache['indices']), np.asarray(cache['indptr'])),
                          shape=(len(cache['kind']), len(nodes)))
    out = np.zeros((3 * L.shape[0], 6))
    for c in range(3):
        out[c::3] = L @ rb[c::3]
    return out


def rigid_span_residual(cache, n):
    got = rigid_pullback(cache, n)
    have = np.asarray(cache['rigid'], dtype=np.float64)
    if got.shape != have.shape:
        raise ValueError('RIGID_SHAPE_MISMATCH')
    return float(np.linalg.norm(got - have) / np.linalg.norm(have))


# ----------------------------------------------------------------------------- the group action
def rotate_context(context, g):
    """The context of the same cell rotated by g; functional indices unchanged."""
    if g == 0:
        return dict(context)
    n = context['n']
    out = dict(context)
    out['pos'] = rotate_positions(context['pos'], g)
    out['faces'] = permute_faces(context['faces'], g)
    out['node_vector'] = rotate_vectors(context['node_vector'], g)
    out['node_tensor'] = rotate_blocks(context['node_tensor'], g)
    out['vol_scalar'] = rotate_scalar_volume(context['vol_scalar'], g)
    out['vol_vector'] = rotate_vector_volume(context['vol_vector'], g)
    out['corners'] = permute_corners(context['corners'], g)
    out['plane'] = rotate_plane(context['plane'], g)
    out['support_pos'] = rotate_positions(context['support_pos'], g)
    return out


def permute_nodes(context, pi):
    """Re-index functionals: new index k is old index pi[k].  A model must not notice."""
    out = dict(context)
    for key in ('pos', 'faces', 'kind', 'node_scalar', 'node_vector', 'node_tensor'):
        out[key] = context[key][pi]
    inverse = np.argsort(pi)
    out['support_rows'] = inverse[context['support_rows']]
    return out


def canonical_frame(context):
    g, corners, ties = canonical(context['corners'])
    return g, ties


# ----------------------------------------------------------------------------- tests
def _close(a, b, tol):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    return a.shape == b.shape and np.abs(a - b).max() <= tol


def _random_trace(rng, n, count, cut):
    """A synthetic trace: box-face nodes with nnz 1, plus cut-surface functionals with nnz 2..20."""
    top = 2 * n
    grid = np.array(list(np.ndindex(top + 1, top + 1, top + 1)), dtype=np.int64)
    boundary = grid[((grid == 0) | (grid == top)).any(axis=1)]
    pick = boundary[rng.choice(len(boundary), size=count, replace=False)]
    rows, sup, coef, kind = [0], [], [], []
    for u in pick:
        sup.append(u[None]); coef.append(np.ones(1)); kind.append(0.); rows.append(rows[-1] + 1)
    for _ in range(cut):
        k = int(rng.integers(2, 21))
        base = rng.integers(1, top, size=3)
        off = rng.integers(-1, 2, size=(k, 3))
        u = np.clip(base + off, 0, top)
        sup.append(u); coef.append(rng.normal(size=k)); kind.append(1.); rows.append(rows[-1] + k)
    return np.array(rows), np.concatenate(sup), np.concatenate(coef), np.array(kind)


def selftest(seed=20260918, n=8):
    from .cubic_group import ORDER, COMPOSE
    rng = np.random.default_rng(seed)
    report = {}
    corners = rng.random(8) * .1 + .15
    known = [1., .3, 1e-4, 1. / n]
    for label, plane, cut in (('box_only', (1., 0., 0., 2.), 0), ('cut_cell', (1., .52, 0., .27), 40)):
        indptr, support, coef, kind = _random_trace(rng, n, 120, cut)
        ctx = compile_from_trace(indptr, support, coef, kind, corners, plane, n, known)
        assert ctx['box_only'] == (cut == 0 and label == 'box_only')
        worst = 0.
        for g in range(ORDER):
            rotated = rotate_context(ctx, g)
            recomputed = compile_from_trace(indptr, map_int_positions(support, g, 2 * n), coef, kind,
                                            permute_corners(corners, g), rotate_plane(plane, g), n, known)
            for key in ('pos', 'faces', 'kind', 'node_scalar', 'node_vector', 'node_tensor',
                        'vol_scalar', 'vol_vector', 'corners', 'plane', 'known',
                        'support_rows', 'support_pos', 'support_coefficients'):
                got, want = np.asarray(rotated[key], float), np.asarray(recomputed[key], float)
                assert got.shape == want.shape, (label, key)
                diff = np.abs(got - want).max() if got.size else 0.
                assert diff < 1e-12, (label, key, g, diff)
                worst = max(worst, diff)
            h = int(rng.integers(0, ORDER))
            twice = rotate_context(rotate_context(ctx, h), g); once = rotate_context(ctx, COMPOSE[g, h])
            for key in ('pos', 'faces', 'node_vector', 'node_tensor', 'vol_scalar', 'vol_vector',
                        'corners', 'plane', 'support_pos'):
                assert _close(twice[key], once[key], 1e-12), (label, 'compose', key)
            back = rotate_context(rotated, INVERSE[g])
            for key in ('pos', 'faces', 'node_vector', 'node_tensor', 'vol_scalar', 'corners', 'plane'):
                assert _close(back[key], ctx[key], 1e-12), (label, 'inverse', key)
        report[label + '_rotation_recompute_worst'] = float(worst)
        # positions rotate exactly, not just to 1e-12
        for g in range(ORDER):
            assert _close(rotate_positions(ctx['pos'], g), rotate_context(ctx, g)['pos'], 0.)
        # the plane's signed distance field is invariant, by construction of rotate_plane
        pts = rng.random((200, 3))
        for g in range(ORDER):
            assert _close(plane_margin(rotate_positions(pts, g), rotate_plane(plane, g)),
                          plane_margin(pts, plane), 1e-12), (label, 'plane margin')
        # channel balance
        allc = np.concatenate((ctx['vol_scalar'], ctx['vol_vector'].reshape(-1, *ctx['vol_scalar'].shape[1:])))
        var = allc.reshape(len(allc), -1).var(axis=1)
        report[label + '_max_channel_variance_share'] = float(var.max() / var.sum())
        assert report[label + '_max_channel_variance_share'] < .6
        # index shuffle is a pure relabelling, including the CSR row map
        pi = rng.permutation(ctx['count'])
        sh = permute_nodes(ctx, pi)
        assert _close(sh['pos'][np.argsort(pi)], ctx['pos'], 0.)
        assert np.array_equal(np.bincount(sh['support_rows'], minlength=ctx['count'])[np.argsort(pi)],
                              np.bincount(ctx['support_rows'], minlength=ctx['count']))
        report[label + '_count'] = int(ctx['count'])
        report[label + '_nnz'] = int(len(ctx['support_rows']))
    # the box-only path is the nnz == 1 special case of the general one, exactly
    top = 2 * n
    grid = np.array(list(np.ndindex(top + 1, top + 1, top + 1)), dtype=np.int64)
    boundary = grid[((grid == 0) | (grid == top)).any(axis=1)]
    u = boundary[rng.random(len(boundary)) < .6]
    a = compile_from_points(u, corners, n, known)
    b = compile_from_trace(np.arange(len(u) + 1), u, np.ones(len(u)), np.zeros(len(u)),
                           corners, (1., 0., 0., 2.), n, known)
    for key in ('pos', 'faces', 'node_scalar', 'node_vector', 'node_tensor', 'vol_scalar'):
        assert _close(a[key], b[key], 0.), key
    # for nnz == 1 the centroid IS the node and the moments vanish
    assert _close(a['pos'], u / top, 0.)
    assert np.abs(a['node_vector'][:, 2]).max() == 0. and np.abs(a['node_tensor']).max() == 0.
    report['box_only_is_the_nnz1_special_case'] = 'exact'
    g0, ties = canonical_frame(a)
    assert ties == 1
    report['canonical'] = 'ok'
    return report


if __name__ == '__main__':
    import json
    print(json.dumps(selftest(), indent=1))
