"""Fitting with D eliminated: minimise the oracle floor over T alone.

For a fixed T the best block-diagonal D is closed form, D_b = (C_bb)^-1 with C = T A^-1 T^T, and
the divergence it achieves is

    floor(T) = sum_b logdet C_bb - logdet C .

So D need not be optimised at all.  Descending floor(T) over T's coefficients is descending the
divergence with D always exactly optimal, which removes the co-adaptation between the two that
makes joint descent stall.

E0-B showed the stall concretely: joint Adam on (T, D) reached D/d 3.206e-5 against a floor of
3.198e-5 for the T it had arrived at.  D was optimal for that T to three digits; it was T that
had stopped moving.
"""
from __future__ import annotations

import torch

from oracle import oracle_floor


def block_groups(op, device):
    """Block index tensors grouped by size, so equal-sized blocks can be gathered in one go.

    Indexing G per block costs far more than the arithmetic: each `G[idx]` allocates a full
    zeros_like(G) for its gradient, so 200 blocks at d = 12792 push ~260 GB of allocation traffic
    per step.  Grouping turns 200 gathers into one per distinct block size.
    """
    groups = {}
    for idx in op.blocks:
        groups.setdefault(len(idx), []).append(idx.to(device))
    return [(n, torch.stack(v)) for n, v in sorted(groups.items())]


def floor_objective(op, coeff, Rinv, groups=None, logdet_G=None):
    """Differentiable floor(T).  Rinv = R*^-1 is precomputed and fixed.

    `logdet_G` may be passed as a constant: T is a product of unit-triangular lifting layers, so
    det T = 1 exactly and logdet(T R*^-1) = -logdet R* whatever the coefficients are.  Skipping
    that dense d x d slogdet every step is free, and the identity is asserted by
    test_lift.test_operator_product_and_det_one.
    """
    G = op.apply_T(coeff, Rinv)
    if logdet_G is None:
        logdet_G = torch.linalg.slogdet(G)[1]
    if groups is None:
        groups = block_groups(op, G.device)
    total = 0.0
    for n, idx in groups:                                  # idx: (m, n)
        Gb = G[idx.reshape(-1)].view(idx.shape[0], n, -1)  # (m, n, d), one gather per size
        total = total + torch.linalg.slogdet(Gb @ Gb.transpose(1, 2))[1].sum()
    return (total - 2.0 * logdet_G) / op.d


def fit_T(op, coeff, Rstar, steps=2000, lr=3e-3, log_every=500, verbose=True):
    """Descend floor(T) over the layer coefficients only; D stays implicit and optimal."""
    eye = torch.eye(op.d, dtype=Rstar.dtype, device=Rstar.device)
    Rinv = torch.linalg.solve_triangular(Rstar, eye, upper=True).detach()
    k = coeff[:op.n_layer_coeff].clone().requires_grad_(True)
    tail = coeff[op.n_layer_coeff:].detach()
    opt = torch.optim.Adam([k], lr=lr)
    hist = []
    for step in range(1, steps + 1):
        opt.zero_grad()
        f = floor_objective(op, torch.cat([k, tail]), Rinv)
        f.backward(); opt.step()
        if verbose and (step % log_every == 0 or step == 1):
            hist.append((step, float(f)))
            print('      floor step %-5d  D/d %.6e' % (step, float(f)), flush=True)
    return torch.cat([k.detach(), tail]), hist


def optimal_D_coefficients(op, coeff, Rstar):
    """Set the D blocks to their closed-form optimum for the current T."""
    _, blocks = oracle_floor(op, coeff, Rstar, return_blocks=True)
    out = coeff.clone()
    off = op.n_layer_coeff
    for n, Db in zip(op.block_sizes, blocks):
        tri = torch.linalg.cholesky(0.5 * (Db + Db.T), upper=True)
        # store so that softplus(stored diagonal) reproduces tri's diagonal
        diag = tri.diagonal()
        inv_softplus = diag + torch.log(-torch.expm1(-diag))
        tri = tri - torch.diag(diag) + torch.diag(inv_softplus)
        iu = torch.triu_indices(n, n, device=coeff.device)
        out[off:off + n * (n + 1) // 2] = tri[iu[0], iu[1]]
        off += n * (n + 1) // 2
    return out
