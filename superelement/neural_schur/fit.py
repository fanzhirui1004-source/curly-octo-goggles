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


def floor_objective(op, coeff, Rinv):
    """Differentiable floor(T).  Rinv = R*^-1 is precomputed and fixed."""
    G = op.apply_T(coeff, Rinv)
    sign, logdet_G = torch.linalg.slogdet(G)
    total = 0.0
    for idx in op.blocks:
        idx = idx.to(G.device)
        Gb = G[idx]
        total = total + torch.linalg.slogdet(Gb @ Gb.T)[1]
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
