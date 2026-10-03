"""A two-sided, whitened training objective for a predicted Cholesky factor.

Motivation
----------
Codex's `relative` auxiliary loss is an unbiased estimate of

    L_rel = || M - I ||_F^2 / d,      M = R_hat R_star^{-1},

with H = M^T M and mu = eig(H) the whitened spectrum the acceptance gate reads.
Per mode its contribution is at best (sqrt(mu) - 1)^2, so as mu -> 0 it SATURATES
at 1 and its gradient decays to zero: at mu = 1e-6 the pressure is 2.0e-3, at
mu = 1e-12 it is 2.0e-6. A mode that has gone singular is therefore free to
sacrifice, which is exactly what the measured A/B shows (mu_min driven to 1e-15
and through zero while D/d improves).

The Bregman/Stein divergence

    D = mean(mu - log mu - 1)

has gradient pinned at -2 for every soft mode, at every scale. It is the same
quantity the project already uses as its necessary condition
(D/d <= 4.5921e-4 for +-3%, <= 5.3605e-3 for +-10%).

The point of this module: D needs NO new forward pass and NO model change.

    d * D = tr(H) - logdet(H) - d
    tr(H)      = ||M||_F^2          <- the probes Codex already draws estimate it;
                                       it is `result.square().mean()` in
                                       relative_action.relative_rows_loss, i.e. the
                                       same tensor, before the target is subtracted.
    logdet(H)  = 2 (sum log r_hat_ii - sum log r_star_ii)
                                    <- r_hat_ii = exp(log_pivot) is EXACTLY the
                                       diagonal of the predicted diagonal blocks
                                       (v7_t2_vectorized_m.py: `output[diagonal] =
                                       lower + diag_embed(exp(log_pivot))` with
                                       `lower = tril(..., diagonal=-1)`), and
                                       factors.sample_pairs already decodes the
                                       COMPLETE diagonal every step - it is not
                                       sampled. So this term is exact, noiseless,
                                       covers every one of the d modes, and is free.
    sum log r_star_ii               <- a per-seat constant from the teacher factor.

Verified against Codex's own LOSS_IDENTITY.json to 10 decimal places by
`verify_logdet_identity.py` in this directory.
"""
import torch


def predicted_logdet(diagonal_blocks):
    """sum_i log R_hat[i,i] from the decoded diagonal blocks. Exact, differentiable.

    `diagonal_blocks` is (n_nodes, 3, 3), the model's diagonal output blocks, whose
    diagonal is exp(log_pivot) by construction. Pass the COMPLETE set of diagonal
    blocks; the training step already decodes all of them.
    """
    if diagonal_blocks.ndim != 3 or diagonal_blocks.shape[-2:] != (3, 3):
        raise ValueError('DIAGONAL_BLOCKS_SHAPE')
    pivots = torch.diagonal(diagonal_blocks, dim1=-2, dim2=-1)
    if not bool(torch.all(pivots > 0)):
        raise ValueError('NONPOSITIVE_PREDICTED_PIVOT')
    return torch.log(pivots).sum()


def teacher_logdet(reference_upper):
    """sum_i log R_star[i,i]. A per-seat constant; compute once, cache it."""
    pivots = torch.diagonal(reference_upper)
    if not bool(torch.all(pivots > 0)):
        raise ValueError('NONPOSITIVE_REFERENCE_PIVOT')
    return torch.log(pivots).sum()


def divergence_from_parts(trace_per_mode, predicted_logdet_value, teacher_logdet_value, d):
    """D = mean(mu - log mu - 1), assembled from the three pieces above.

    `trace_per_mode` is tr(H)/d, i.e. `result.square().mean()` over the SAME probe
    tensor relative_rows_loss already forms (before subtracting the target).
    """
    return trace_per_mode - 2.0 * (predicted_logdet_value - teacher_logdet_value) / d - 1.0


def divergence_rows_loss(action, diagonal_blocks, teacher_logdet_value, d):
    """Drop-in companion to relative_action.relative_rows_loss.

    `action` is that function's `result` tensor: (rows, 3, probes) holding
    (R_hat R_star^{-1} Z) on the sampled output rows. Reuse it; do not recompute.
    """
    trace_per_mode = action.square().mean()
    value = divergence_from_parts(trace_per_mode, predicted_logdet(diagonal_blocks),
                                  teacher_logdet_value, d)
    if not bool(torch.isfinite(value)):
        raise ValueError('NONFINITE_DIVERGENCE_LOSS')
    return value, dict(trace_per_mode=float(trace_per_mode.detach()),
                       logdet_exact=True, logdet_sampled=False,
                       probes=int(action.shape[-1]), output_block_rows=int(action.shape[0]))


def whitened_maximum(apply_factor, solve_reference, d, iterations=20, device='cuda:0',
                     dtype=torch.float64, seed=0):
    """mu_max by power iteration on H = M^T M, using only forward applications.

    `apply_factor(v)`    returns R_hat v   (block decode, already available)
    `solve_reference(v)` returns R_star^{-1} v (triangular solve against the teacher)

    H v = R_star^{-T} R_hat^T R_hat R_star^{-1} v: four operations, no inverse of
    the PREDICTED factor, so nothing sequential in the unknown. Use this for the
    stiff end of the gate. The soft end is covered by the log-det term above, whose
    gradient does not vanish; do not attempt mu_min by power iteration on cI - H,
    which does not converge and returns plausible-looking garbage silently.
    """
    generator = torch.Generator(device=device).manual_seed(int(seed))
    v = torch.randn(d, 1, generator=generator, device=device, dtype=dtype)
    v = v / v.norm()
    value = None
    for _ in range(int(iterations)):
        w = solve_reference(v)
        w = apply_factor(w)
        w = apply_factor(w, transpose=True)
        w = solve_reference(w, transpose=True)
        value = float(v.T @ w)
        norm = w.norm()
        if not torch.isfinite(norm) or float(norm) == 0.0:
            raise ValueError('POWER_ITERATION_BREAKDOWN')
        v = w / norm
    return value
