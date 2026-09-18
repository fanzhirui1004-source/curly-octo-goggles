"""The 48 symmetries of the unit cube about its centre, in every representation the pipeline needs.

One signed permutation matrix Q per element (entries in {-1, 0, 1}, Q Q^T = I) generates
everything else, so the representations cannot drift apart:

  vectors        v  -> Q v                          displacements, gradients, node deltas
  positions      p  -> Q (p - 1/2) + 1/2            exact on the integer grids used here
  corners        the 8 trilinear tau corners, index 4bx + 2by + bz, are a permutation
  faces          the 6 cube faces [x0, y0, z0, x1, y1, z1] (outward normal -e_a / +e_a)
  cell grid      the n^3 background cells, centres (i + 1/2)/n, are a permutation
  tensor blocks  B_ij -> Q B_ij Q^T for the 3x3 blocks of a nodal rank-2 field

The Schwarz-P field phi = sum cos(2 pi p) is invariant under all 48 elements about the
centre, so a geometry rotated by g is exactly the geometry with permuted corners, and the
operator of the rotated geometry is the permuted-and-rotated operator of the original.
That identity is what makes one label worth 48 (up to the teacher's own symmetry error,
which is measured, not assumed).

Element 0 is the identity.  Elements with det Q = +1 are the 24 proper rotations.
Pure numpy; no torch import, so the tests run anywhere.
"""
from __future__ import annotations

import itertools
import numpy as np

CORNER_BITS = np.array(list(itertools.product((0, 1), repeat=3)), dtype=np.int64)   # index 4bx+2by+bz
FACE_NORMALS = np.concatenate((-np.eye(3, dtype=np.int64), np.eye(3, dtype=np.int64)))  # x0 y0 z0 x1 y1 z1


def _elements():
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            Q = np.zeros((3, 3), dtype=np.int64)
            for a in range(3):
                Q[a, perm[a]] = signs[a]
            out.append(Q)
    Q = np.stack(out)
    # identity first, then a fixed deterministic order
    key = [tuple(q.ravel()) for q in Q]
    ident = key.index(tuple(np.eye(3, dtype=np.int64).ravel()))
    order = [ident] + [i for i in range(len(Q)) if i != ident]
    return Q[order]


Q_ALL = _elements()
ORDER = len(Q_ALL)
_LOOKUP = {tuple(q.ravel()): i for i, q in enumerate(Q_ALL)}
DET = np.array([int(round(np.linalg.det(q))) for q in Q_ALL])
PROPER = np.flatnonzero(DET > 0)


def index_of(Q):
    return _LOOKUP[tuple(np.asarray(Q, dtype=np.int64).ravel())]


def compose(g, h):
    """Index of the element acting as 'first h, then g': Q = Q_g Q_h."""
    return index_of(Q_ALL[g] @ Q_ALL[h])


def inverse(g):
    return index_of(Q_ALL[g].T)


COMPOSE = np.array([[compose(g, h) for h in range(ORDER)] for g in range(ORDER)])
INVERSE = np.array([inverse(g) for g in range(ORDER)])


def map_int_positions(u, g, top):
    """p -> Q (p - 1/2) + 1/2 on integer coordinates u = top * p in [0, top]; exact."""
    u = np.asarray(u, dtype=np.int64)
    if u.shape[-1] != 3 or u.min() < 0 or u.max() > top:
        raise ValueError('INTEGER_POSITIONS_OUTSIDE_GRID')
    twice = (2 * u - top) @ Q_ALL[g].T + top          # parity of every entry equals parity of top
    if np.any(twice % 2):
        raise AssertionError('CUBE_SYMMETRY_LEFT_THE_GRID')
    return twice // 2


def map_positions(p, g):
    """Real positions in the unit cube."""
    return (np.asarray(p, dtype=np.float64) - .5) @ Q_ALL[g].T.astype(np.float64) + .5


def rotate_vectors(v, g):
    """(..., 3) vectors: v -> Q v."""
    return np.asarray(v, dtype=np.float64) @ Q_ALL[g].T.astype(np.float64)


def _permutation_tables():
    corner = np.empty((ORDER, 8), dtype=np.int64)
    face = np.empty((ORDER, 6), dtype=np.int64)
    for g in range(ORDER):
        moved = map_int_positions(CORNER_BITS, g, 1)
        corner[g] = moved @ np.array([4, 2, 1])
        normals = FACE_NORMALS @ Q_ALL[g].T
        for f in range(6):
            match = np.flatnonzero((FACE_NORMALS == normals[f]).all(axis=1))
            if len(match) != 1:
                raise AssertionError('FACE_NORMAL_NOT_MAPPED')
            face[g, f] = match[0]
    return corner, face


# CORNER_PERM[g][c] = image index of corner c; FACE_PERM[g][f] = image index of face f
CORNER_PERM, FACE_PERM = _permutation_tables()


def permute_corners(corners, g):
    """tau'(g c) = tau(c): the corner vector of the rotated geometry."""
    corners = np.asarray(corners, dtype=np.float64)
    out = np.empty_like(corners)
    out[..., CORNER_PERM[g]] = corners
    return out


def permute_faces(flags, g):
    """(..., 6) face flags of nodes that keep their index but move with g."""
    flags = np.asarray(flags)
    out = np.empty_like(flags)
    out[..., FACE_PERM[g]] = flags
    return out


def cell_source_index(n, g):
    """src such that rotated_volume.reshape(C, -1)[:, k] = volume.reshape(C, -1)[:, src[k]]."""
    cells = np.array(list(np.ndindex(n, n, n)), dtype=np.int64)
    moved = map_int_positions(cells, g, n - 1)                      # centres (i+1/2)/n rotate about the centre
    flat_new = np.ravel_multi_index(moved.T, (n, n, n))
    src = np.empty(n ** 3, dtype=np.int64)
    src[flat_new] = np.arange(n ** 3)
    return src


def rotate_scalar_volume(volume, g):
    """(C, n, n, n) scalar channels."""
    volume = np.asarray(volume)
    n = volume.shape[-1]
    flat = volume.reshape(volume.shape[0], -1)[:, cell_source_index(n, g)]
    return flat.reshape(volume.shape)


def rotate_vector_volume(volume, g):
    """(C, 3, n, n, n) vector channels: moved with the grid and rotated by Q."""
    volume = np.asarray(volume, dtype=np.float64)
    n = volume.shape[-1]
    flat = volume.reshape(volume.shape[0], 3, -1)[:, :, cell_source_index(n, g)]
    flat = np.einsum('ab,cbk->cak', Q_ALL[g].astype(np.float64), flat)
    return flat.reshape(volume.shape)


def rotate_blocks(blocks, g):
    """(..., 3, 3) tensor blocks: B -> Q B Q^T."""
    Q = Q_ALL[g].astype(np.float64)
    return np.einsum('ab,...bc,dc->...ad', Q, np.asarray(blocks, dtype=np.float64), Q)


def node_permutation(u_int, g, top):
    """pi with u_int[pi[i]] == map(u_int[i]): which original node sits where node i lands.

    Needed only when comparing against a teacher run of the rotated geometry, whose
    node registry is sorted by background id; training never re-indexes nodes.
    """
    u_int = np.asarray(u_int, dtype=np.int64)
    key = u_int @ np.array([(top + 1) ** 2, top + 1, 1])
    order = np.argsort(key)
    moved = map_int_positions(u_int, g, top) @ np.array([(top + 1) ** 2, top + 1, 1])
    pos = np.searchsorted(key[order], moved)
    if pos.max() >= len(key) or not np.array_equal(key[order][pos], moved):
        raise ValueError('ROTATED_NODE_SET_IS_NOT_THE_NODE_SET')
    return order[pos]


def canonical(corners):
    """The element g* that makes the permuted corner vector lexicographically largest.

    Returns (g*, canonical corners, stabiliser size).  Every geometry in one orbit maps
    to the same canonical corners, so a network fed canonical inputs is exactly invariant;
    the price is a discontinuity in tau wherever the argmax switches, which is why the
    intrinsic route (tensor-basis decoder) exists as well.
    """
    corners = np.asarray(corners, dtype=np.float64)
    if corners.shape != (8,):
        raise ValueError('EIGHT_CORNERS')
    images = np.stack([permute_corners(corners, g) for g in range(ORDER)])
    best = max(range(ORDER), key=lambda g: tuple(images[g]))
    ties = int(sum(np.array_equal(images[g], images[best]) for g in range(ORDER)))
    return best, images[best], ties


# ----------------------------------------------------------------------------- tests
def _geometry_values(points, corners):
    """Local copy of adapter.geometry_values' tau and gradient (no torch, no plane)."""
    p = np.asarray(points, dtype=np.float64)
    tau = np.zeros(p.shape[:-1]); gradient = np.zeros_like(p)
    for index, bit in enumerate(itertools.product((0, 1), repeat=3)):
        weight = [p[..., j] if bit[j] else 1 - p[..., j] for j in range(3)]
        tau += corners[index] * weight[0] * weight[1] * weight[2]
        for j in range(3):
            other = [k for k in range(3) if k != j]
            gradient[..., j] += corners[index] * (2 * bit[j] - 1) * weight[other[0]] * weight[other[1]]
    phi = np.cos(2 * np.pi * p).sum(axis=-1)
    gphi = -2 * np.pi * np.sin(2 * np.pi * p)
    return tau, gradient, phi, gphi


def selftest(seed=20260918, n=8):
    rng = np.random.default_rng(seed)
    report = {}
    # group axioms
    assert ORDER == 48 and len(_LOOKUP) == 48
    assert np.array_equal(Q_ALL[0], np.eye(3, dtype=np.int64))
    for g in range(ORDER):
        assert np.array_equal(Q_ALL[g] @ Q_ALL[g].T, np.eye(3, dtype=np.int64))
        assert COMPOSE[g, INVERSE[g]] == 0 and COMPOSE[INVERSE[g], g] == 0
    for g in range(ORDER):
        for h in range(ORDER):
            assert index_of(Q_ALL[g] @ Q_ALL[h]) == COMPOSE[g, h]
    assert len(PROPER) == 24
    report['group'] = 'ok'
    # positions compose and are exact on the grid; corners/faces are permutations
    top = 64
    u = rng.integers(0, top + 1, size=(500, 3))
    for _ in range(64):
        g, h = rng.integers(0, ORDER, size=2)
        assert np.array_equal(map_int_positions(map_int_positions(u, h, top), g, top),
                              map_int_positions(u, COMPOSE[g, h], top))
        assert np.allclose(map_positions(u / top, g), map_int_positions(u, g, top) / top, atol=1e-15)
    for g in range(ORDER):
        assert sorted(CORNER_PERM[g]) == list(range(8)) and sorted(FACE_PERM[g]) == list(range(6))
        assert np.array_equal(CORNER_PERM[INVERSE[g]][CORNER_PERM[g]], np.arange(8))
    report['positions'] = 'ok'
    # corner permutation reproduces the rotated tau field, gradient rotates as a vector, phi invariant
    corners = rng.random(8)
    p = rng.random((300, 3))
    tau, grad, phi, gphi = _geometry_values(p, corners)
    worst = 0.
    for g in range(ORDER):
        tau2, grad2, phi2, gphi2 = _geometry_values(map_positions(p, g), permute_corners(corners, g))
        worst = max(worst, np.abs(tau2 - tau).max(), np.abs(grad2 - rotate_vectors(grad, g)).max(),
                    np.abs(phi2 - phi).max(), np.abs(gphi2 - rotate_vectors(gphi, g)).max())
    assert worst < 1e-12, worst
    report['tau_phi_field_consistency'] = float(worst)
    # face flags recomputed from moved positions equal the permuted flags
    ub = u.copy(); ub[:200, 0] = 0; ub[200:300, 1] = top; ub[300:, 2] = 0
    flags = np.concatenate((ub == 0, ub == top), axis=1)
    for g in range(ORDER):
        moved = map_int_positions(ub, g, top)
        assert np.array_equal(np.concatenate((moved == 0, moved == top), axis=1), permute_faces(flags, g))
    report['faces'] = 'ok'
    # volume grid: value at the moved cell equals the original value; vectors rotate
    vol = rng.random((2, n, n, n)); vec = rng.random((1, 3, n, n, n))
    cells = np.array(list(np.ndindex(n, n, n)))
    for g in range(ORDER):
        rs = rotate_scalar_volume(vol, g); rv = rotate_vector_volume(vec, g)
        moved = map_int_positions(cells, g, n - 1)
        assert np.array_equal(rs[:, moved[:, 0], moved[:, 1], moved[:, 2]], vol[:, cells[:, 0], cells[:, 1], cells[:, 2]])
        want = np.einsum('ab,cbk->cak', Q_ALL[g].astype(float), vec[:, :, cells[:, 0], cells[:, 1], cells[:, 2]])
        assert np.allclose(rv[:, :, moved[:, 0], moved[:, 1], moved[:, 2]], want, atol=0)
        # composition on the grid
        h = int(rng.integers(0, ORDER))
        assert np.array_equal(rotate_scalar_volume(rotate_scalar_volume(vol, h), g), rotate_scalar_volume(vol, COMPOSE[g, h]))
    report['volume'] = 'ok'
    # blocks: (Q B Q^T)(Q v) = Q (B v), and composition
    B = rng.random((50, 3, 3)); v = rng.random((50, 3))
    for g in range(ORDER):
        lhs = np.einsum('kab,kb->ka', rotate_blocks(B, g), rotate_vectors(v, g))
        rhs = rotate_vectors(np.einsum('kab,kb->ka', B, v), g)
        assert np.allclose(lhs, rhs, atol=1e-14)
        h = int(rng.integers(0, ORDER))
        assert np.allclose(rotate_blocks(rotate_blocks(B, h), g), rotate_blocks(B, COMPOSE[g, h]), atol=1e-14)
    report['blocks'] = 'ok'
    # node permutation on a symmetric node set (all boundary nodes of a 4-grid)
    grid = np.array(list(np.ndindex(5, 5, 5)))
    boundary = grid[(grid == 0).any(axis=1) | (grid == 4).any(axis=1)]
    for g in range(ORDER):
        pi = node_permutation(boundary, g, 4)
        assert np.array_equal(boundary[pi], map_int_positions(boundary, g, 4))
    report['node_permutation'] = 'ok'
    # canonical form is orbit-invariant, and the stabiliser of a generic vector is trivial
    for _ in range(20):
        x = rng.random(8)
        g0, c0, ties0 = canonical(x)
        assert ties0 == 1
        for h in range(ORDER):
            gh, ch, _ = canonical(permute_corners(x, h))
            assert np.array_equal(ch, c0)
            # and the recovered frame really maps the rotated vector onto the canonical one
            assert np.array_equal(permute_corners(permute_corners(x, h), gh), c0)
    assert canonical(np.ones(8))[2] == 48
    report['canonical'] = 'ok'
    return report


if __name__ == '__main__':
    import json
    print(json.dumps(selftest(), indent=1))
