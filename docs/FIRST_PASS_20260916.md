# The first achievable pass on the real teacher

Date 2026-09-16.  Seat 0328 of the production GP teacher, `d = 12792`, dense `S` = 8.182e7
independent entries.  Script `superelement/factor_fit/hmat4.py`; raw `docs/data/HMAT_ECONOMY4.json`.

Nothing in this project had previously passed the acceptance gate `eps_op = max_i |mu_i - 1|` on a
real teacher by any means.  A hierarchical low-rank approximation of `A` does, with no fitting of
any kind.

## Measured

Same construction as `HIERARCHICAL_ECONOMY`: bisection tree (511 nodes, leaf 64), admissibility
`min(diam) <= 2 dist`, 2965 admissible + 1344 near-field blocks, near field exact, admissible
blocks truncated by SVD at a blockwise relative-Frobenius tolerance.  Rank cap raised to 768; **0
of 2965 blocks reached it**.  The assembly self-test (every block filled straight from `A`) returns
`eps_op = 1.807e-13` with 0 entries uncovered and 0 written twice.

| tol | params | % of dense | `eps_op` | `D/d` | `n` out +-3% | `n` out +-10% | `mu` range | verdict |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1e-3 | 11,282,414 | 13.79% | 2.289e-01 | 1.274e-05 | 42 | 3 | 0.771 … 1.057 | fails |
| 3e-4 | 12,924,435 | 15.80% | 4.645e-02 | 1.041e-06 | 3 | 0 | 0.954 … 1.026 | **PASSES +-10%** |
| 1e-4 | 14,341,507 | 17.53% | 1.055e-02 | 1.267e-07 | 0 | 0 | 0.990 … 1.010 | **PASSES +-3%** |
| 3e-5 | 15,784,025 | 19.29% | 3.478e-03 | 7.593e-09 | 0 | 0 | 0.997 … 1.003 | PASSES +-3% |
| 1e-5 | 17,079,433 | 20.87% | 8.029e-04 | 4.956e-10 | 0 | 0 | 0.999 … 1.001 | PASSES +-3% |
| 3e-6 | 18,527,436 | 22.64% | 1.919e-04 | 4.276e-11 | 0 | 0 | 1.000 … 1.000 | PASSES +-3% |

**+-3% is reached at 14,341,507 parameters, 17.53% of the dense entry count.**

This is an upper bound on the question the line of work has been asking — how many
geometry-predictable numbers determine `S` to +-3% — which is the direction an upper bound should
err in: a better representation can only need fewer.

## Two things worth keeping

**The prediction held.**  `GATE_AXIS` predicted +-10% near 12.5M parameters from the two points
available then, and said it was worth falsifying rather than extrapolating.  It landed at 12.92M.

**The concentration is confirmed at the extreme.**  At `tol = 1e-3`, only **3 modes** of 12,792
lie outside +-10% and only 42 outside +-3%, yet `eps_op = 0.229`.  Three modes out of twelve
thousand decide the verdict.  No statistic that averages — the divergence included — can see them;
at that same row `D/d = 1.27e-05` clears the +-3% necessary line by 36x.

## Scope

One seat, one construction, achieved and audited.  It proves that operator passes.  It does not
bound any class, does not say 14.3M is the minimum, and says nothing about whether a network can
predict those numbers from geometry.  See `docs/CLAIM_SCOPE_20260916.md`.

## The immediate follow-up

14.3M is the cost of approximating `A` **entry-wise**, and `ERROR_METRIC_20260916` says that is
the expensive metric: with `Rhat = R* + E` the whitened operator is `X^T X` for `X = Rhat R*^-1`,
so `eps_op <~ 2||E|| / sigma_min(R*)` and approximating the **factor** buys a tolerance looser by
`cond(R*) = sqrt(cond(A)) = 269` for the same gate.  Whether that converts into fewer parameters
depends on how compressible `R*` is, which `hmat_factor.py` is measuring now on the same tree and
the same audit.  `Rhat` is not constrained to be triangular there — `H = X^T X` is positive
semidefinite whatever `Rhat` is, so the audit stays honest either way.
