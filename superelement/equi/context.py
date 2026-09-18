"""Typed, index-free network inputs for one cell, and the exact action of the cube group on them.

Every array is tagged by how it transforms, so `rotate_context` cannot be wrong in a way
the tests would miss: scalars move with their node or cell, vectors move and rotate by Q,
face flags and corners permute.  Nothing in here depends on node index, elimination
order, or the Householder quotient: those were the memorisation handles of the legacy
adapter (elimination position, 18 reflector entries, Morton patches, GRU order) and they
are gone.

Only the box-only trace is supported (one background node per functional, unit
coefficient, no residual cut coordinates); every packet in the current datasets is one,
and anything else raises rather than being approximated.
"""
from __future__ import annotations

import numpy as np

from .cubic_group import (map_int_positions, permute_corners, permute_faces, rotate_vectors,
                          rotate_scalar_volume, rotate_vector_volume, canonical, INVERSE)

SCALAR_NAMES = ('tau', 'phi', 'margin', 'grad_tau_norm', 'grad_phi_norm')
VECTOR_NAMES = ('grad_tau', 'grad_phi')
VOLUME_SCALAR_NAMES = ('tau', 'phi', 'margin', 'solid')
VOLUME_VECTOR_NAMES = ('grad_tau', 'grad_phi')
INERT_PLANE = np.array([1., 0., 0., 2.])


def trilinear(points, corners):
    """tau and its gradient from the 8 corners (index 4bx+2by+bz), identical to adapter.geometry_values."""
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


def field_values(points, corners):
    tau, gtau = trilinear(points, corners)
    phi, gphi = schwarz_p(points)
    margin = tau - np.abs(phi)                      # solid where |phi| <= tau
    scalars = np.stack((tau, phi, margin, np.linalg.norm(gtau, axis=-1), np.linalg.norm(gphi, axis=-1)), axis=-1)
    vectors = np.stack((gtau, gphi), axis=-2)      # (..., 2, 3)
    return scalars, vectors


def compile_from_points(u_int, corners, n, known):
    """Build the context from integer node positions on the (2n+1)^3 grid and the corner taus."""
    u_int = np.asarray(u_int, dtype=np.int64)
    top = 2 * n
    if u_int.ndim != 2 or u_int.shape[1] != 3 or u_int.min() < 0 or u_int.max() > top:
        raise ValueError('NODE_POSITIONS_OUTSIDE_GRID')
    on_boundary = (u_int == 0) | (u_int == top)
    if not on_boundary.any(axis=1).all():
        raise ValueError('BOX_TRACE_NODE_NOT_ON_BOUNDARY')
    pos = u_int / top
    faces = np.concatenate((u_int == 0, u_int == top), axis=1).astype(np.float64)
    node_scalar, node_vector = field_values(pos, corners)
    grid = np.stack(np.meshgrid(*[(np.arange(n) + .5) / n] * 3, indexing='ij'), axis=-1)
    vs, vv = field_values(grid, corners)
    vol_scalar = np.concatenate((vs[..., :3], (vs[..., 2:3] >= 0).astype(np.float64)), axis=-1)
    vol_scalar = np.ascontiguousarray(vol_scalar.transpose(3, 0, 1, 2))          # (4, n, n, n)
    vol_vector = np.ascontiguousarray(vv.transpose(3, 4, 0, 1, 2))               # (2, 3, n, n, n)
    context = dict(n=int(n), count=int(len(u_int)), pos_int=u_int, pos=pos, faces=faces,
                   node_scalar=node_scalar, node_vector=node_vector,
                   vol_scalar=vol_scalar, vol_vector=vol_vector,
                   corners=np.asarray(corners, dtype=np.float64).copy(),
                   known=np.asarray(known, dtype=np.float64).copy())
    for key, value in context.items():
        if isinstance(value, np.ndarray) and not np.isfinite(value).all():
            raise ValueError('NONFINITE_FEATURE: ' + key)
    return context


def compile_equi_inputs(cache, metadata):
    """From a frozen TRACE_CACHE + INPUT.json metadata.  Target-free."""
    if metadata['geometry']['representation'] != 'eight_corner_trilinear_unit_box_v1':
        raise ValueError('UNSUPPORTED_GEOMETRY_REPRESENTATION')
    if metadata['geometry']['box_max'] != [1, 1, 1]:
        raise ValueError('UNIT_BOX_REQUIRED')
    n = int(metadata['n'])
    indptr = np.asarray(cache['indptr']); indices = np.asarray(cache['indices'])
    coeff = np.asarray(cache['coefficients'], dtype=np.float64)
    if not (np.all(np.diff(indptr) == 1) and np.all(coeff == 1.) and np.all(np.asarray(cache['kind']) == 0)):
        raise ValueError('BOX_ONLY_TRACE_REQUIRED')
    if not np.allclose(np.asarray(cache['cut_plane'], dtype=np.float64), INERT_PLANE):
        raise ValueError('ONLY_THE_INERT_CUT_PLANE_IS_SUPPORTED')
    nodes = np.asarray(cache['background_nodes'])[indices]
    u_int = np.column_stack(np.unravel_index(nodes, (2 * n + 1,) * 3)).astype(np.int64)
    if not np.allclose(u_int / (2 * n), np.asarray(cache['support_centroid']), atol=1e-12, rtol=0):
        raise ValueError('TRACE_CENTROID_BINDING')
    order = np.asarray(cache['order'])
    if not np.array_equal(order, (3 * (order[::3] // 3)[:, None] + np.arange(3)).ravel()):
        raise ValueError('COMPILER_ORDER_IS_NOT_XYZ_GROUPED')
    gp = metadata['gp']; material = metadata['material']
    if gp['orders'] != [1, 2] or gp['scope'] != 'module local':
        raise ValueError('UNSUPPORTED_GP_CONTRACT')
    known = [material['E'], material['nu'], gp['gamma'], 1. / n]
    return compile_from_points(u_int, np.asarray(cache['tau_corners'], dtype=np.float64), n, known)


def rotate_context(context, g):
    """The context of the same cell rotated by g, node indices unchanged."""
    n = context['n']
    out = dict(context)
    out['pos_int'] = map_int_positions(context['pos_int'], g, 2 * n)
    out['pos'] = out['pos_int'] / (2 * n)
    out['faces'] = permute_faces(context['faces'], g)
    out['node_vector'] = rotate_vectors(context['node_vector'], g)
    out['vol_scalar'] = rotate_scalar_volume(context['vol_scalar'], g)
    out['vol_vector'] = rotate_vector_volume(context['vol_vector'], g)
    out['corners'] = permute_corners(context['corners'], g)
    return out


def permute_nodes(context, pi):
    """Re-index nodes: new node k is old node pi[k].  A model must not notice."""
    out = dict(context)
    for key in ('pos_int', 'pos', 'faces', 'node_scalar', 'node_vector'):
        out[key] = context[key][pi]
    return out


def canonical_frame(context):
    g, corners, ties = canonical(context['corners'])
    return g, ties


# ----------------------------------------------------------------------------- tests
def _close(a, b, tol):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    return a.shape == b.shape and np.abs(a - b).max() <= tol


def selftest(seed=20260918, n=8):
    from .cubic_group import ORDER, COMPOSE, map_positions
    rng = np.random.default_rng(seed)
    top = 2 * n
    grid = np.array(list(np.ndindex(top + 1, top + 1, top + 1)), dtype=np.int64)
    boundary = grid[((grid == 0) | (grid == top)).any(axis=1)]
    keep = rng.random(len(boundary)) < .7                         # an asymmetric node subset
    u = boundary[keep]
    corners = rng.random(8) * .1 + .15
    known = [1., .3, 1e-4, 1. / n]
    ctx = compile_from_points(u, corners, n, known)
    report = dict(nodes=int(len(u)))
    worst = 0.
    for g in range(ORDER):
        rotated = rotate_context(ctx, g)
        recomputed = compile_from_points(map_int_positions(u, g, top), permute_corners(corners, g), n, known)
        for key in ('pos_int', 'pos', 'faces', 'node_scalar', 'node_vector', 'vol_scalar', 'vol_vector', 'corners', 'known'):
            diff = np.abs(np.asarray(rotated[key], dtype=np.float64) - np.asarray(recomputed[key], dtype=np.float64)).max()
            assert rotated[key].shape == recomputed[key].shape, key
            assert diff < 1e-12, (key, g, diff)
            worst = max(worst, diff)
        h = int(rng.integers(0, ORDER))
        twice = rotate_context(rotate_context(ctx, h), g); once = rotate_context(ctx, COMPOSE[g, h])
        for key in ('pos_int', 'faces', 'node_vector', 'vol_scalar', 'vol_vector', 'corners'):
            assert _close(twice[key], once[key], 1e-12), key
        back = rotate_context(rotated, INVERSE[g])
        for key in ('pos_int', 'faces', 'node_vector', 'vol_scalar', 'vol_vector', 'corners'):
            assert _close(back[key], ctx[key], 1e-12), key
    report['rotation_recompute_worst'] = float(worst)
    # the field values at the nodes are the volume field's values: same function
    vs, vv = field_values(ctx['pos'], corners)
    assert _close(vs, ctx['node_scalar'], 0) and _close(vv, ctx['node_vector'], 0)
    # solid indicator agrees with the margin channel
    assert np.array_equal(ctx['vol_scalar'][3], (ctx['vol_scalar'][2] >= 0).astype(float))
    # index shuffle is a pure relabelling
    pi = rng.permutation(len(u))
    shuffled = permute_nodes(ctx, pi)
    assert np.array_equal(shuffled['pos_int'][np.argsort(pi)], ctx['pos_int'])
    # canonical frame: the frame of a rotated cell maps it onto the same canonical inputs
    g0, ties = canonical_frame(ctx)
    assert ties == 1
    canon0 = rotate_context(ctx, g0)
    for h in range(ORDER):
        gh, _ = canonical_frame(rotate_context(ctx, h))
        canon_h = rotate_context(rotate_context(ctx, h), gh)
        for key in ('corners', 'vol_scalar', 'vol_vector'):
            assert _close(canon_h[key], canon0[key], 1e-12), key
        # node arrays agree up to the (unchanged) node indexing
        assert _close(canon_h['pos_int'], canon0['pos_int'], 0)
    report['canonical'] = 'ok'
    report['grid_top'] = top
    return report


if __name__ == '__main__':
    import json
    print(json.dumps(selftest(), indent=1))
