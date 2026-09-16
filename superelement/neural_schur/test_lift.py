"""The custom lifting gradient must agree with the dense reference, in value and in gradient.

_LiftApply hand-writes its backward to avoid torch.sparse.mm's dense-materialising gradient.  A
hand-written backward is exactly the kind of thing that is silently wrong, so check it against
autograd through the explicit dense matrix, including the transpose path and the multi-layer
product actually used by QuotientOperator.
"""
import numpy as np
import torch

from backend import LiftingLayer, QuotientOperator, spatial_blocks


def _layer(d=40, n_pairs=60, seed=0, chunk=7):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(d)
    A, B = perm[: d // 2], perm[d // 2:]
    pairs = np.stack([rng.choice(A, n_pairs), rng.choice(B, n_pairs)], 1)
    pairs = np.unique(pairs, axis=0)                    # dense() assigns, so duplicates would differ
    return LiftingLayer(d, pairs, chunk=chunk), d


def test_forward_matches_dense():
    layer, d = _layer()
    k = torch.randn(layer.n_coeff, dtype=torch.float64)
    x = torch.randn(d, 9, dtype=torch.float64)
    assert torch.allclose(layer.apply(k, x), layer.dense(k) @ x, atol=1e-12)
    assert torch.allclose(layer.apply_transpose(k, x), layer.dense(k).T @ x, atol=1e-12)


def test_gradients_match_dense():
    layer, d = _layer(seed=1)
    x0 = torch.randn(d, 5, dtype=torch.float64)
    for transpose in (False, True):
        ka = torch.randn(layer.n_coeff, dtype=torch.float64, requires_grad=True)
        kb = ka.detach().clone().requires_grad_(True)
        xa = x0.clone().requires_grad_(True)
        xb = x0.clone().requires_grad_(True)
        w = torch.randn(d, 5, dtype=torch.float64)

        fast = layer.apply_transpose(ka, xa) if transpose else layer.apply(ka, xa)
        M = layer.dense(kb)
        ref = (M.T @ xb) if transpose else (M @ xb)
        (fast * w).sum().backward()
        (ref * w).sum().backward()
        assert torch.allclose(ka.grad, kb.grad, atol=1e-11), transpose
        assert torch.allclose(xa.grad, xb.grad, atol=1e-11), transpose


def test_gradcheck():
    layer, d = _layer(d=24, n_pairs=20, seed=2, chunk=5)
    k = torch.randn(layer.n_coeff, dtype=torch.float64, requires_grad=True)
    x = torch.randn(d, 4, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda kk, xx: layer.apply(kk, xx), (k, x), eps=1e-6, atol=1e-8)


def test_chunking_is_invisible():
    """Every chunk size must give the same answer, including one that does not divide the width."""
    rng = np.random.default_rng(3)
    d = 32
    perm = rng.permutation(d)
    pairs = np.unique(np.stack([rng.choice(perm[:16], 40), rng.choice(perm[16:], 40)], 1), axis=0)
    k0 = torch.randn(len(pairs), dtype=torch.float64)
    x = torch.randn(d, 13, dtype=torch.float64)
    outs, grads = [], []
    for chunk in (1, 5, 13, 64):
        layer = LiftingLayer(d, pairs, chunk=chunk)
        k = k0.clone().requires_grad_(True)
        y = layer.apply(k, x)
        y.pow(2).sum().backward()
        outs.append(y); grads.append(k.grad)
    for o, g in zip(outs[1:], grads[1:]):
        assert torch.allclose(o, outs[0], atol=1e-13)
        assert torch.allclose(g, grads[0], atol=1e-12)


def test_operator_product_and_det_one():
    """T is a product of these layers; det T = 1 is the property the fast path must not break."""
    rng = np.random.default_rng(4)
    d = 36
    pts = rng.random((d, 3))
    layers = []
    for s in range(3):
        perm = rng.permutation(d)
        pairs = np.unique(np.stack([rng.choice(perm[:18], 50), rng.choice(perm[18:], 50)], 1), axis=0)
        layers.append(LiftingLayer(d, pairs, chunk=7))
    op = QuotientOperator(d, layers, spatial_blocks(pts, 6))
    coeff = op.identity_coefficients()
    coeff[: op.n_layer_coeff] = torch.randn(op.n_layer_coeff, dtype=torch.float64) * 0.3
    T = op.apply_T(coeff, torch.eye(d, dtype=torch.float64))
    assert abs(float(torch.linalg.slogdet(T)[1])) < 1e-10
    x = torch.randn(d, 4, dtype=torch.float64)
    assert torch.allclose(op.apply_T(coeff, x), T @ x, atol=1e-11)
    assert torch.allclose(op.apply_T_transpose(coeff, x), T.T @ x, atol=1e-11)


def test_batched_floor_matches_naive():
    """The grouped floor must equal the per-block one, in value and gradient."""
    from fit import block_groups, floor_objective
    rng = np.random.default_rng(5)
    d = 30
    pts = rng.random((d, 3))
    layers = []
    for _ in range(2):
        perm = rng.permutation(d)
        pairs = np.unique(np.stack([rng.choice(perm[:15], 40), rng.choice(perm[15:], 40)], 1), axis=0)
        layers.append(LiftingLayer(d, pairs, chunk=4))
    op = QuotientOperator(d, layers, spatial_blocks(pts, 5))
    L = torch.tril(torch.randn(d, d, dtype=torch.float64)) + d * torch.eye(d, dtype=torch.float64)
    Rinv = torch.linalg.inv(L)

    def naive(c):
        G = op.apply_T(c, Rinv)
        tot = torch.linalg.slogdet(G)[1] * -2.0
        for idx in op.blocks:
            Gb = G[idx]
            tot = tot + torch.linalg.slogdet(Gb @ Gb.T)[1]
        return tot / op.d

    c0 = op.identity_coefficients()
    c0[: op.n_layer_coeff] = torch.randn(op.n_layer_coeff, dtype=torch.float64) * 0.2
    ka = c0[: op.n_layer_coeff].clone().requires_grad_(True)
    kb = c0[: op.n_layer_coeff].clone().requires_grad_(True)
    tail = c0[op.n_layer_coeff:]
    fa = floor_objective(op, torch.cat([ka, tail]), Rinv, groups=block_groups(op, Rinv.device))
    fb = naive(torch.cat([kb, tail]))
    assert abs(float(fa) - float(fb)) < 1e-11, (float(fa), float(fb))
    fa.backward(); fb.backward()
    assert torch.allclose(ka.grad, kb.grad, atol=1e-10)


def test_floor_constant_logdet_is_exact():
    """Passing logdet_G as the constant -logdet R* must change nothing."""
    from fit import block_groups, floor_objective
    rng = np.random.default_rng(6)
    d = 28
    pts = rng.random((d, 3))
    perm = rng.permutation(d)
    pairs = np.unique(np.stack([rng.choice(perm[:14], 30), rng.choice(perm[14:], 30)], 1), axis=0)
    op = QuotientOperator(d, [LiftingLayer(d, pairs, chunk=3)], spatial_blocks(pts, 5))
    L = torch.tril(torch.randn(d, d, dtype=torch.float64)) + d * torch.eye(d, dtype=torch.float64)
    Rinv = torch.linalg.inv(L)
    const = float(torch.linalg.slogdet(Rinv)[1])
    c = op.identity_coefficients()
    c[: op.n_layer_coeff] = torch.randn(op.n_layer_coeff, dtype=torch.float64) * 0.3
    g = block_groups(op, Rinv.device)
    a = float(floor_objective(op, c, Rinv, groups=g))
    b = float(floor_objective(op, c, Rinv, groups=g, logdet_G=torch.tensor(const, dtype=torch.float64)))
    assert abs(a - b) < 1e-12, (a, b)


def test_T_inverse_is_exact():
    """T^-1 must be exact, not iterative: it is what makes power iteration on H^-1 affordable."""
    rng = np.random.default_rng(7)
    d = 34
    pts = rng.random((d, 3))
    layers = []
    for _ in range(3):
        perm = rng.permutation(d)
        pairs = np.unique(np.stack([rng.choice(perm[:17], 45), rng.choice(perm[17:], 45)], 1), axis=0)
        layers.append(LiftingLayer(d, pairs, chunk=6))
    op = QuotientOperator(d, layers, spatial_blocks(pts, 6))
    c = op.identity_coefficients()
    c[: op.n_layer_coeff] = torch.randn(op.n_layer_coeff, dtype=torch.float64) * 0.4
    I = torch.eye(d, dtype=torch.float64)
    T = op.apply_T(c, I)
    Tinv = op.apply_T_inverse(c, I)
    assert torch.allclose(T @ Tinv, I, atol=1e-11), (T @ Tinv - I).abs().max()
    assert torch.allclose(Tinv @ T, I, atol=1e-11)
    assert torch.allclose(op.apply_T_inverse_transpose(c, I), Tinv.T, atol=1e-11)
    # and it must agree with a dense inverse, not merely be self-consistent
    assert torch.allclose(Tinv, torch.linalg.inv(T), atol=1e-10)


def test_smallest_eigenvalue_by_inverse_iteration():
    """H = M^T M with M = C T R^-1, so H^-1 = M^-1 M^-T and the top of H^-1 is the bottom of H."""
    rng = np.random.default_rng(8)
    d = 40
    pts = rng.random((d, 3))
    perm = rng.permutation(d)
    pairs = np.unique(np.stack([rng.choice(perm[:20], 60), rng.choice(perm[20:], 60)], 1), axis=0)
    op = QuotientOperator(d, [LiftingLayer(d, pairs, chunk=8)], spatial_blocks(pts, 7))
    c = op.identity_coefficients()
    c[: op.n_layer_coeff] = torch.randn(op.n_layer_coeff, dtype=torch.float64) * 0.3
    L = torch.tril(torch.randn(d, d, dtype=torch.float64)) + d * torch.eye(d, dtype=torch.float64)
    R = L.T.contiguous()
    Rinv = torch.linalg.inv(R)
    _, cs = op.split(c)
    chols = op._chol_blocks(cs)
    G = op.apply_T(c, Rinv)
    M = torch.empty_like(G)
    for idx, Cb in zip(op.blocks, chols):
        M[idx] = Cb @ G[idx]
    H = M.T @ M
    ref = torch.linalg.eigvalsh(0.5 * (H + H.T))

    def Minv(v):                              # M^-1 = R T^-1 C^-1
        w = torch.empty_like(v)
        for idx, Cb in zip(op.blocks, chols):
            w[idx] = torch.linalg.solve_triangular(Cb, v[idx], upper=True)
        return R @ op.apply_T_inverse(c, w)

    def MinvT(v):                             # M^-T = C^-T T^-T R^T
        w = op.apply_T_inverse_transpose(c, R.T @ v)
        out = torch.empty_like(w)
        for idx, Cb in zip(op.blocks, chols):
            out[idx] = torch.linalg.solve_triangular(Cb.T, w[idx], upper=False)
        return out

    v = torch.randn(d, 1, dtype=torch.float64)
    for _ in range(300):
        v = Minv(MinvT(v)); v = v / v.norm()
    mu_min = float((M @ v).pow(2).sum())
    assert abs(mu_min - float(ref[0])) < 1e-8 * float(ref[-1]), (mu_min, float(ref[0]))
