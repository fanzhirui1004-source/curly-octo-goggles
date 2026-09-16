# Approximating the factor helps, but far less than the error-metric argument suggested

Date 2026-09-16.  Seat 0328, `d = 12792`.
Script `superelement/factor_fit/hmat_factor.py`; raw `docs/data/HMAT_FACTOR.json`.

## The prediction, and what happened

`ERROR_METRIC_20260916` established that the gate is multiplicative on the factor and additive on
`A`, with `cond(A) = 7.23e4` between them.  With `Rhat = R* + E` the whitened operator is `X^T X`
for `X = Rhat R*^-1`, so `eps_op <~ 2||E|| / sigma_min(R*)`, and approximating `R*` tolerates an
error looser by `cond(R*) = sqrt(cond(A)) = 269` than approximating `A` does.

I expected that to be a large parameter saving.  **It is not.**  Same tree, same audit, `Rhat`
unconstrained (`H = X^T X` is positive semidefinite whatever `Rhat` is), self-test `eps_op =
7.105e-14`:

| target | +-10% at | +-3% at |
|---|---:|---:|
| `A` (entry-wise) | 12,924,435 params (15.80%) | 14,341,507 params (17.53%) |
| `R*` (the factor) | **10,221,964 params (12.49%)** | **12,229,217 params (14.95%)** |
| saving | 1.26x | 1.17x |

Full factor sweep:

| tol | params | % of dense | `eps_op` | `D/d` | out +-3% | out +-10% | verdict |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1e-1 | 6,860,058 | 8.38% | 4.164e-01 | 8.171e-05 | 178 | 34 | fails |
| 3e-2 | 8,581,236 | 10.49% | 1.524e-01 | 1.826e-05 | 64 | 15 | fails |
| 1e-2 | 10,221,964 | 12.49% | 8.078e-02 | 4.121e-06 | 18 | 0 | **PASSES +-10%** |
| 3e-3 | 12,229,217 | 14.95% | 2.597e-02 | 6.322e-07 | 0 | 0 | **PASSES +-3%** |
| 1e-3 | 14,083,953 | 17.21% | 1.056e-02 | 1.043e-07 | 0 | 0 | PASSES +-3% |
| 3e-4 | 16,070,495 | 19.64% | 3.353e-03 | 1.123e-08 | 0 | 0 | PASSES +-3% |

## Why the 269x did not arrive

Two reasons, both visible in the run.

**A looser tolerance buys only a little rank.**  Block singular values decay, so relaxing the
tolerance by 269x removes a handful of ranks per block rather than most of them.  The error budget
and the parameter count are related through that decay, not proportionally.

**The target is no longer symmetric.**  `A` needs one SVD per unordered block pair; `R*` needs one
per *ordered* pair, so the partition went from 2965 admissible + 1344 near-field to 5930 + 2432.
Triangularity claws some of that back — 2086 of the 5930 admissible blocks are identically zero
and cost nothing — but not all of it.

The two effects nearly cancel, leaving a real but modest 1.17x-1.26x.

## What this changes

The best upper bound on the project's central question is now: **`S` is determined to +-3% by at
most 12.23e6 numbers on this seat, 14.95% of the dense entry count**, via a hierarchical low-rank
representation of the factor, with no fitting at all.

And a correction to how `ERROR_METRIC` was being read.  The 5.1e4 gap in *error budget* is real and
exact; I extrapolated it into an expected *parameter* saving, and that step was unjustified.  The
budget ratio and the parameter ratio are different quantities, connected by a spectral decay that
has to be measured.  Measured here, it is 1.2x, not 269x.

The metric argument still stands where it was actually established: a fit driven by `||Ahat - A||`
spends its accuracy in the metric that costs `cond(A)` more.  That is a statement about what to
optimise, and it is not a statement about how many parameters the result will need.
