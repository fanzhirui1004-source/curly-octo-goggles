"""The oracle floor: what the best block-diagonal D can reach for a GIVEN T, before any training.

Minimising the log-det divergence over block-diagonal positive definite D with a fixed partition
has a closed form.  With C = T A^-1 T^T,

    D_b^opt = (C_bb)^-1,        D_min = sum_b logdet C_bb - logdet C

and D_min >= 0 by Fischer's inequality, with equality exactly when C is block diagonal in that
partition.  So the floor measures one thing: **how far T takes A^-1 from being block diagonal**.

Cheap to compute.  C = (T R*^-1)(T R*^-1)^T, so with G = T R*^-1 each C_bb = G_b G_b^T is a small
determinant and logdet C = 2 logdet G.  The cost is one d x d triangular inverse, d^3/3, which is
sub-second at d = 12792 against 0.56-1.04 s per training step.  A structure can therefore be
screened before it is trained, and the divergence it can never beat is known in advance.

Two limits, both from the independent verification.  The closed form needs D to range over ALL
positive definite block-diagonal matrices with a fixed partition in a fixed basis; any further
constraint breaks separability and the formula is gone.  And it bounds the DIVERGENCE only, not
the worst-direction error eps_op, which is the acceptance criterion.
"""
from __future__ import annotations

import numpy as np
import torch


def oracle_floor(op, coeff, Rstar, chunk: int = 2048, return_blocks: bool = False):
    """D_min / d for this T and this block partition, plus the optimal blocks if asked."""
    d = op.d
    eye = torch.eye(d, dtype=Rstar.dtype, device=Rstar.device)
    Rinv = torch.linalg.solve_triangular(Rstar, eye, upper=True)        # R*^-1
    G = op.apply_T(coeff, Rinv)                                          # G = T R*^-1
    del Rinv, eye
    sign, logdet_G = torch.linalg.slogdet(G)
    if float(sign) == 0:
        raise ValueError('SINGULAR_G')
    logdet_C = 2.0 * float(logdet_G)
    total, blocks = 0.0, []
    for idx in op.blocks:
        idx = idx.to(G.device)
        Gb = G[idx]
        Cbb = Gb @ Gb.T
        s, ld = torch.linalg.slogdet(Cbb)
        if float(s) <= 0:
            raise ValueError('NONPOSITIVE_BLOCK')
        total += float(ld)
        if return_blocks:
            blocks.append(torch.linalg.inv(Cbb))
    floor = (total - logdet_C) / d
    return (floor, blocks) if return_blocks else floor


def block_offdiagonal_mass(op, coeff, Rstar):
    """A second, scale-free view of the same thing: the fraction of C's Frobenius mass off-block."""
    d = op.d
    eye = torch.eye(d, dtype=Rstar.dtype, device=Rstar.device)
    G = op.apply_T(coeff, torch.linalg.solve_triangular(Rstar, eye, upper=True))
    C = G @ G.T
    total = float(C.square().sum())
    inside = 0.0
    for idx in op.blocks:
        idx = idx.to(C.device)
        inside += float(C[idx.unsqueeze(1), idx.unsqueeze(0)].square().sum())
    return 1.0 - inside / total
