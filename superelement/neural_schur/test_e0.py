"""E0: prove the implementation and the ruler before trusting either.

E0-A  structural identities on a system small enough to write out explicitly
E0-C  the evaluator must find a single planted bad direction, at the magnitudes we actually see
      in this project (mu up to 275), not just a mild one

Run: python test_e0.py
Every tolerance is fixed here, not adjusted until something passes.
"""
from __future__ import annotations

import sys
import numpy as np
import torch

from backend import QuotientOperator, multiscale_layers, spatial_blocks
import evaluator as EV

torch.manual_seed(20260916)
DT = torch.float64
FAILS = []


def check(name, ok, detail=''):
    print('  %-52s %s   %s' % (name, 'ok' if ok else 'FAIL', detail), flush=True)
    if not ok:
        FAILS.append(name)


def small_operator(d=120, levels=3, block=6, seed=0):
    rng = np.random.default_rng(seed)
    pts = rng.random((d, 3))
    layers, meta = multiscale_layers(pts, levels=levels, radius0=0.25, growth=1.6, seed=seed)
    blocks = spatial_blocks(pts, block)
    op = QuotientOperator(d, layers, blocks)
    return op, pts, meta


def e0a():
    print('\nE0-A  structural identities')
    op, pts, meta = small_operator()
    d = op.d
    print('  d %d, layers %s, pairs %s, blocks %d'
          % (d, len(op.layers), [m['pairs'] for m in meta], len(op.blocks)))
    c = op.identity_coefficients(dtype=DT)
    c = c + 0.05 * torch.randn(c.shape, dtype=DT)

    T = op.dense_T(c)
    Dm = op.dense_D(c)
    A = op.dense_A(c)

    # every lifting layer is unit triangular, so det T is exactly 1
    sign, ld = torch.linalg.slogdet(T)
    check('det T == 1', abs(float(ld)) < 1e-10 and float(sign) > 0, 'log|det T| = %.2e' % float(ld))

    # implicit action equals the explicit matrix
    X = torch.randn(d, 7, dtype=DT)
    check('apply_T matches dense T', torch.allclose(op.apply_T(c, X), T @ X, atol=1e-11),
          'max %.2e' % float((op.apply_T(c, X) - T @ X).abs().max()))
    check('apply_A matches dense A', torch.allclose(op.apply_A(c, X), A @ X, atol=1e-10),
          'max %.2e' % float((op.apply_A(c, X) - A @ X).abs().max()))

    # transpose is the transpose of the same map, not a second one
    Y = torch.randn(d, 5, dtype=DT)
    lhs = (op.apply_T(c, X).T @ Y)
    rhs = (X.T @ op.apply_T_transpose(c, Y))
    check('<T x, y> == <x, T^T y>', torch.allclose(lhs, rhs, atol=1e-11),
          'max %.2e' % float((lhs - rhs).abs().max()))

    # symmetry and positive definiteness of A_hat
    check('A_hat symmetric', float((A - A.T).abs().max()) < 1e-11, 'max %.2e' % float((A - A.T).abs().max()))
    ev = torch.linalg.eigvalsh(0.5 * (A + A.T))
    check('A_hat positive definite', float(ev.min()) > 0, 'lam_min %.3e' % float(ev.min()))

    # logdet by construction equals logdet of the explicit matrix
    ld_struct = float(op.logdet_A(c))
    ld_dense = float(torch.linalg.slogdet(A)[1])
    check('logdet_A == logdet(dense A)', abs(ld_struct - ld_dense) < 1e-8,
          'struct %.10f dense %.10f diff %.2e' % (ld_struct, ld_dense, abs(ld_struct - ld_dense)))
    ld_D = float(torch.linalg.slogdet(Dm)[1])
    check('logdet A == logdet D (det T = 1)', abs(ld_dense - ld_D) < 1e-8, 'diff %.2e' % abs(ld_dense - ld_D))

    # energy gradient is the reaction
    class Ident:
        def __call__(self, u): return u
        def lift(self, z): return z
    u = torch.randn(d, 1, dtype=DT, requires_grad=True)
    E = op.energy(c, u, Ident()).sum()
    (gu,) = torch.autograd.grad(E, u)
    check('grad_u energy == A_hat u', torch.allclose(gu, A @ u.detach(), atol=1e-10),
          'max %.2e' % float((gu - A @ u.detach()).abs().max()))

    # finite-difference gradient in the coefficients
    c2 = c.clone().requires_grad_(True)
    z = torch.randn(d, 1, dtype=DT)
    loss = (z * op.apply_A(c2, z)).sum() + op.logdet_A(c2)
    (gc,) = torch.autograd.grad(loss, c2)
    v = torch.randn(c.shape, dtype=DT); v /= v.norm()
    worst = 0.0
    for h in (1e-5, 1e-6):
        with torch.no_grad():
            f = lambda cc: float((z * op.apply_A(cc, z)).sum() + op.logdet_A(cc))
            fd = (f(c + h * v) - f(c - h * v)) / (2 * h)
        an = float(gc @ v)
        worst = max(worst, abs(fd - an) / max(abs(an), 1e-12))
    check('finite-difference gradient', worst < 1e-6, 'worst relative %.2e' % worst)

    # a lifting layer added with zero coefficients leaves the operator untouched
    from backend import LiftingLayer
    extra = LiftingLayer(d, np.array([[0, 1], [2, 3]], dtype=np.int64))
    op2 = QuotientOperator(d, op.layers + [extra], [b.numpy() for b in op.blocks])
    c_ext = torch.cat([c[:op.n_layer_coeff], torch.zeros(extra.n_coeff, dtype=DT), c[op.n_layer_coeff:]])
    check('identity-initialised new layer is a no-op',
          torch.allclose(op2.dense_A(c_ext), A, atol=1e-11),
          'max %.2e' % float((op2.dense_A(c_ext) - A).abs().max()))


def e0c():
    print('\nE0-C  the evaluator must find one planted bad direction')
    d = 400
    rng = np.random.default_rng(1)
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    Rstar = torch.eye(d, dtype=DT)                       # teacher whitening is the identity here
    A = torch.eye(d, dtype=DT)
    for name, alpha in (('mild stiff  (mu=1.5)', 0.5), ('severe stiff (mu=275)', 274.0),
                        ('severe soft  (mu=0.0036)', -0.9964)):
        v = torch.as_tensor(Q[:, 0], dtype=DT).reshape(-1, 1)
        Ahat = A + alpha * (v @ v.T)
        mu = EV.whitened_spectrum(Ahat, Rstar)
        res = EV.assess(mu, d, 0.03, label=name)
        print('   ', EV.report(res).replace('\n', '\n    '))
        expect_state = EV.FAIL
        check('%-26s state == FAIL_WITNESS' % name, res['state'] == expect_state)
        check('%-26s witness found' % name, res.get('n_outside', 0) == 1,
              'outside %d' % res.get('n_outside', -1))
        # the diagnostic that would have hidden it
        thr = EV.sufficient_divergence_threshold(d, 0.03)
        print('      D/d %.3e  vs the threshold that would prove a pass %.3e  -> %s'
              % (res['D_per_mode'], thr,
                 'D/d alone cannot certify' if res['D_per_mode'] > thr else 'D/d would certify'))


if __name__ == '__main__':
    e0a()
    e0c()
    print('\n%s' % ('E0 checks passed' if not FAILS else 'E0 FAILURES: %s' % FAILS))
    sys.exit(1 if FAILS else 0)
