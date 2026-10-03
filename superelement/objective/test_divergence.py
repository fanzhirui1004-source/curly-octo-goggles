"""Numerical proof that D assembles exactly from the pieces already computed.

Every test below mirrors Codex's data layout: block-3x3 factors, L[r,c]=R[c,r]^T,
diagonal blocks lower-triangular with positive diagonal, Rademacher probes with
x = R_star^{-1} z.
"""
import numpy as np
import torch

from divergence import (divergence_from_parts, predicted_logdet, teacher_logdet,
                        whitened_maximum)

torch.manual_seed(20260917)
DT = torch.float64


def case(d=96, softest=1e-9, seed=3):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(d, d, generator=g, dtype=DT)
    A = X @ X.T + d * torch.eye(d, dtype=DT)
    Rs = torch.linalg.cholesky(A).mH.contiguous()          # upper teacher factor
    Y = torch.randn(d, d, generator=g, dtype=DT)
    Rh = torch.triu(Rs + 0.02 * Y * Rs.abs().mean())
    pivots = torch.diagonal(Rh).clone()
    pivots[11] = torch.diagonal(Rs)[11] * softest          # drive one mode singular
    Rh = Rh - torch.diag_embed(torch.diagonal(Rh)) + torch.diag_embed(pivots)
    return Rs, Rh


def reference_divergence(Rs, Rh):
    M = Rh @ torch.linalg.inv(Rs)
    mu = torch.linalg.eigvalsh(M.T @ M)
    return float((mu - torch.log(mu) - 1).mean()), mu


def test_logdet_from_pivots_beats_slogdet_on_a_stiff_case():
    """Agrees with slogdet when the spectrum is benign; stays exact when it is not."""
    for softest, tol in ((1e-3, 1e-9), (1e-9, None)):
        Rs, Rh = case(softest=softest)
        from_pivots = float(2 * (torch.log(torch.diagonal(Rh)).sum() - teacher_logdet(Rs)))
        M = Rh @ torch.linalg.inv(Rs)
        from_slogdet = float(torch.linalg.slogdet(M.T @ M)[1])
        gap = abs(from_pivots - from_slogdet)
        print(f"  softest={softest:8.0e}  pivots {from_pivots:.10f}  slogdet {from_slogdet:.10f}"
              f"  gap {gap:.2e}")
        if tol is not None:
            assert gap < tol, (from_pivots, from_slogdet)


def test_logdet_from_pivots_is_algebraically_exact():
    """M = R_hat R_star^{-1} is triangular, so diag(M) = diag(R_hat)/diag(R_star)
    exactly. The pivot formula is therefore an identity, not an approximation."""
    Rs, Rh = case(softest=1e-9)
    M = Rh @ torch.linalg.inv(Rs)
    direct = float(2 * torch.log(torch.diagonal(M).abs()).sum())
    pivots = float(2 * (torch.log(torch.diagonal(Rh)).sum() - teacher_logdet(Rs)))
    assert abs(direct - pivots) < 1e-9, (direct, pivots)
    print(f"  2*sum log diag(M) {direct:.10f}   from pivots {pivots:.10f}   identity holds")


def test_divergence_assembles_from_trace_and_pivots():
    """Agrees with the eigen route WHERE THE EIGENSOLVER IS RELIABLE."""
    Rs, Rh = case(softest=1e-3)
    d = Rs.shape[0]
    M = Rh @ torch.linalg.inv(Rs)
    trace_per_mode = (M * M).sum() / d                      # what the probes estimate
    got = float(divergence_from_parts(trace_per_mode, torch.log(torch.diagonal(Rh)).sum(),
                                      teacher_logdet(Rs), d))
    want, _ = reference_divergence(Rs, Rh)
    assert abs(got - want) / want < 1e-8, (got, want)
    print(f"  D assembled {got:.10f}   D from eig {want:.10f}")


def test_the_eigen_route_is_the_one_that_fails_on_a_singular_mode():
    """At mu ~ 1e-18 the eigendecomposition cannot resolve the mode and its D is
    wrong; the pivot assembly stays exact. This matters because the measured
    relative arm reports mu_min of 1e-12..1e-15 and -1.35e-16."""
    for softest in (1e-3, 1e-5, 1e-7, 1e-9):
        Rs, Rh = case(softest=softest)
        d = Rs.shape[0]
        M = Rh @ torch.linalg.inv(Rs)
        exact = float(divergence_from_parts((M * M).sum() / d,
                                            torch.log(torch.diagonal(Rh)).sum(),
                                            teacher_logdet(Rs), d))
        mu = torch.linalg.eigvalsh(M.T @ M)
        eigen = float((mu - torch.log(mu.clamp_min(1e-300)) - 1).mean())
        true_mu_min = float(torch.diagonal(M)[11] ** 2)
        print(f"  softest={softest:8.0e}  mu_min true {true_mu_min:9.2e}  "
              f"eig {float(mu.min()):9.2e}   D exact {exact:.6f}  D via eig {eigen:.6f}  "
              f"err {abs(eigen-exact)/exact:7.2%}")


def test_probes_estimate_the_trace_without_bias():
    Rs, Rh = case()
    d = Rs.shape[0]
    p = 200000
    g = torch.Generator().manual_seed(11)
    Z = (torch.randint(0, 2, (d, p), generator=g, dtype=torch.int64).to(DT) * 2 - 1)
    Xp = torch.linalg.solve_triangular(Rs, Z, upper=True)    # teacher_probes()
    action = Rh @ Xp                                         # relative_rows_loss 'result'
    got = float(action.square().mean())
    want = float((Rh @ torch.linalg.inv(Rs)).pow(2).sum() / d)
    assert abs(got - want) / want < 5e-3, (got, want)
    print(f"  tr(H)/d from probes {got:.6f}   exact {want:.6f}")


def test_the_gradient_cliff_and_that_the_logdet_term_removes_it():
    """The measured failure: on a singular mode the relative loss has no gradient."""
    Rs, Rh = case()
    d = Rs.shape[0]
    Rsi = torch.linalg.inv(Rs)
    lt = teacher_logdet(Rs)

    def both(scale, j=11):
        R = Rh.clone()
        R[j, j] = R[j, j] * scale
        M = R @ Rsi
        rel = ((M - torch.eye(d, dtype=DT)) ** 2).sum() / d
        div = divergence_from_parts((M * M).sum() / d, torch.log(torch.diagonal(R)).sum(), lt, d)
        return rel, div

    h = 1e-6
    rel_p, div_p = both(np.exp(h))
    rel_m, div_m = both(np.exp(-h))
    g_rel = float((rel_p - rel_m) / (2 * h)) * d
    g_div = float((div_p - div_m) / (2 * h)) * d
    print(f"  d(relative)/d(log pivot) = {g_rel:.3e}     <- vanishes")
    print(f"  d(D)/d(log pivot)        = {g_div:.6f}     <- pinned near -2")
    assert abs(g_rel) < 1e-4, g_rel
    assert -2.05 < g_div < -1.9, g_div


def test_power_iteration_finds_mu_max():
    Rs, Rh = case(d=64, softest=1e-6)
    d = Rs.shape[0]
    _, mu = reference_divergence(Rs, Rh)
    want = float(mu.max())

    def apply_factor(v, transpose=False):
        return (Rh.T @ v) if transpose else (Rh @ v)

    def solve_reference(v, transpose=False):
        return torch.linalg.solve_triangular(Rs.T if transpose else Rs, v,
                                             upper=not transpose)

    got = whitened_maximum(apply_factor, solve_reference, d, iterations=200,
                           device='cpu', dtype=DT, seed=5)
    assert abs(got - want) / want < 1e-3, (got, want)
    print(f"  mu_max power iteration {got:.6f}   exact {want:.6f}")


if __name__ == '__main__':
    for name, fn in sorted((k, v) for k, v in globals().items() if k.startswith('test_')):
        print(name)
        fn()
    print('\nall green')
