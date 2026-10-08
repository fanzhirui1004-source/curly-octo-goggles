"""One registered energy-response term; teacher and deployed operator are unchanged.

`add_energy_gradient` accumulates into whatever gradients the neutral bucket loss already put on the
parameters, so the caller owns `zero_grad` and the optimizer step.  The weight comes from
`coefficient_from_preflight`, which refuses a preflight whose complete-factor replay is not exact to
1e-10 or whose frozen parameter state moved, and then takes the median ratio of the neutral to the
energy gradient norm so the new term enters at the scale of the one already there.

Cost, measured: 29-40 s per gradient (see `energy_response_vjp`).  This is a diagnostic or an
occasional term, not an inner-loop loss.

Provenance: Codex's, from `CLAUDE_HANDOFF_CUT_20260920`, landed unchanged.
"""
import statistics
import torch
from .energy_response_vjp import complete_factor,complete_factor_vjp,energy_response_loss


def coefficient_from_preflight(result):
    rows=list(result['seats'].values())
    if not rows or any(r['complete_factor_replay_relative']>1e-10 or
                       not r['frozen_parameter_state_unchanged'] for r in rows):
        raise ValueError('PREFLIGHT_NOT_QUALIFIED')
    weights=[r['gradient_neutral_norm']/r['gradient_energy_norm'] for r in rows]
    if any(not 0<w<float('inf') for w in weights):
        raise ValueError('INVALID_INITIAL_GRADIENT_SCALE')
    return statistics.median(weights)


def add_energy_gradient(model,ctx,conditioning,bank,chunk,weight):
    """Accumulate into existing neutral gradients. Same encoder graph for both passes."""
    if weight==0:
        return None
    if weight<0:
        raise ValueError('NEGATIVE_ENERGY_LOSS_WEIGHT')
    encoded=model.encode(ctx)
    matrix=complete_factor(model,ctx,conditioning,chunk,encoded=encoded).requires_grad_(True)
    loss,errors=energy_response_loss(matrix,bank['quotient'],bank['rstar'],bank['force'],bank['xstar'])
    value=float(loss.detach())
    worst=float(errors.detach().max().sqrt())
    (weight*loss).backward()
    grad=matrix.grad.detach()
    if not bool(torch.isfinite(grad).all()):
        raise ValueError('NONFINITE_COMPLETE_FACTOR_ADJOINT')
    del loss,errors,matrix
    complete_factor_vjp(model,ctx,conditioning,grad,chunk,encoded=encoded)
    return dict(loss=value,worst_teacher_energy_error=worst)


@torch.no_grad()
def energy_monitor(model,ctx,conditioning,bank,chunk):
    matrix=complete_factor(model,ctx,conditioning,chunk)
    loss,errors=energy_response_loss(matrix,bank['quotient'],bank['rstar'],bank['force'],bank['xstar'])
    return dict(loss=float(loss),worst_teacher_energy_error=float(errors.max().sqrt()))

