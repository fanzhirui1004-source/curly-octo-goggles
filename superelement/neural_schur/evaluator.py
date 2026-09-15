"""Independent, exact evaluation of a predicted quotient operator.

The headline number is the operator error

    eps_op = || H - I ||_2 = max_i |mu_i - 1|,      H = R*^-T A_hat R*^-1

because the +-3% target gate is exactly the Loewner condition (1-eps) A <= A_hat <= (1+eps) A
that the composability bound consumes, so eps_op is the certificate.  D/d is reported as a
training diagnostic only: a single direction at mu = 86 costs it just 6.3e-3, and the threshold
that would actually imply +-3% is min(f(0.97), f(1.03))/d = 3.449e-8, 2899 times tighter than the
1e-4 written in the gate.

This computes the full spectrum by dense eigendecomposition.  It never uses a one-sided iterative
estimate to declare a pass, and it reports a four-state verdict rather than a boolean.
"""
from __future__ import annotations

import math
import numpy as np
import torch

PASS = 'NUMERIC_PASS'
FAIL = 'FAIL_WITNESS'
UNVERIFIED = 'UNVERIFIED'
INVALID = 'INVALID_NUMERICS'


def whitened_spectrum(Ahat: torch.Tensor, Rstar: torch.Tensor) -> np.ndarray:
    """mu = eig(R*^-T A_hat R*^-1), exactly, by dense symmetric eigendecomposition."""
    X = torch.linalg.solve_triangular(Rstar.T, Ahat, upper=False)
    X = torch.linalg.solve_triangular(Rstar.T, X.T, upper=False).T
    X = 0.5 * (X + X.T)
    return torch.linalg.eigvalsh(X).detach().cpu().numpy()


def divergence(mu: np.ndarray) -> float:
    m = np.asarray(mu, dtype=float)
    if (m <= 0).any():
        return float('inf')
    return float(np.sum(m - np.log(m) - 1.0))


def sufficient_divergence_threshold(d: int, tau: float) -> float:
    """D/d below this PROVES every direction is inside +-tau, since D is a sum of non-negative terms."""
    f = lambda t: t - math.log(t) - 1.0
    return min(f(1 - tau), f(1 + tau)) / d


def assess(mu: np.ndarray, d: int, tau: float, label: str = '') -> dict:
    mu = np.asarray(mu, dtype=float)
    out = dict(label=label, tau=tau, d=int(d), n=int(mu.size))
    if mu.size != d:
        return dict(out, state=INVALID, reason='SPECTRUM_SIZE_MISMATCH')
    if not np.all(np.isfinite(mu)):
        return dict(out, state=INVALID, reason='NONFINITE_SPECTRUM')
    if (mu <= 0).any():
        return dict(out, state=FAIL, reason='NONPOSITIVE_EIGENVALUE',
                    mu_min=float(mu.min()), mu_max=float(mu.max()),
                    n_nonpositive=int((mu <= 0).sum()), eps_op=float('inf'))
    eps = float(max(mu.max() - 1.0, 1.0 - mu.min()))
    D = divergence(mu)
    out.update(mu_min=float(mu.min()), mu_max=float(mu.max()), eps_op=eps,
               D_per_mode=D / d, sufficient_D_per_mode=sufficient_divergence_threshold(d, tau),
               n_outside=int(((mu < 1 - tau) | (mu > 1 + tau)).sum()),
               n_too_soft=int((mu < 1 - tau).sum()), n_too_stiff=int((mu > 1 + tau).sum()))
    if eps <= tau:
        out['state'] = PASS
    else:
        out['state'] = FAIL
        worst = int(np.argmax(np.abs(mu - 1.0)))
        out['witness'] = dict(index=worst, mu=float(mu[worst]), relative_error=float(mu[worst] - 1.0))
    return out


def report(res: dict) -> str:
    if res['state'] == INVALID:
        return '%-18s %s  (%s)' % (res.get('label', ''), INVALID, res.get('reason'))
    s = ('%-18s %-14s eps_op %.4e   mu [%.4e, %.4e]   D/d %.3e   outside %d (soft %d, stiff %d)'
         % (res.get('label', ''), res['state'], res['eps_op'], res['mu_min'], res['mu_max'],
            res['D_per_mode'], res['n_outside'], res['n_too_soft'], res['n_too_stiff']))
    if 'witness' in res:
        w = res['witness']
        s += '\n%-18s witness: index %d at mu %.6f (%.2f%% off)' % ('', w['index'], w['mu'], 100 * w['relative_error'])
    return s


def action_error(Ahat: torch.Tensor, A: torch.Tensor, Rstar: torch.Tensor, probes: torch.Tensor) -> float:
    """Teacher-whitened action residual, the training-side quantity, reported for comparison only."""
    z = torch.linalg.solve_triangular(Rstar, probes, upper=True)
    r = torch.linalg.solve_triangular(Rstar.T, (Ahat - A) @ z, upper=False)
    return float(r.square().sum() / probes.shape[1] / Ahat.shape[0])
