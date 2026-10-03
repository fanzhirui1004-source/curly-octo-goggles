# Conditional optimal divergence under one fixed transform, against parameter count

**Read the title literally.**  Every number here is `F(T) = min_D divergence(T^T D T, A)/d` for **one
explicit, fixed `T`**.  Since `F_class = inf_T F(T) <= F(T)`, a large value here excludes that `T`
and nothing else.  This is **not** a minimum parameter requirement for the architecture, and the
extrapolation in section 3 is a property of this construction's trend only.  See
`docs/CLAIM_SCOPE_20260916.md` section 0 for why, including the two-by-two case where one trainable
lifting coefficient takes the same problem from `F(I) = log(2)/2` to an exact fit.

## 1. Setup

Seat 0328 of the production GP teacher.  `d = 12792`, so a dense symmetric `S` on the quotient has
`d(d+1)/2 = 8.18e7` independent entries.  `A = R*^T R*` with `R*` the teacher's shipped Cholesky
factor.

`T` is `R*` **truncated to a spatial band** of radius `r` in the unit cell, plus the exact
closed-form optimal block-diagonal `D` for that `T`.  This is a concrete, free, non-trivial
candidate, and `F(T_trunc)` is *achieved*, so it is an honest upper bound for the banded class at
that radius — the informative direction is downward.

Parameter count = band nonzeros + `sum_b n_b(n_b+1)/2` over the `D` blocks.  Gates, as necessary
conditions on the divergence: all `mu` within +-3% requires `D/d <= 4.5921e-4`; within +-10%
requires `D/d <= 5.3605e-3`.  Neither is sufficient — clearing them only means this particular
necessary condition did not exclude the point.

## 2. Measured (86.9 s for all 24 points, `PARAM_ECONOMY.json`)

Pareto frontier over the 6 radii x 4 block sizes:

| radius | block | params | % of dense | `F(T)` = D/d |
|---:|---:|---:|---:|---:|
| 0.00 | 1  | 25,584    | 0.03% | 1.5936 |
| 0.00 | 24 | 100,368   | 0.12% | 0.7378 |
| 0.03 | 24 | 248,862   | 0.30% | 0.3636 |
| 0.00 | 64 | 338,792   | 0.41% | 0.3355 |
| 0.03 | 64 | 487,286   | 0.60% | 0.1855 |
| 0.06 | 64 | 1,002,797 | 1.23% | 0.0782 |
| 0.10 | 64 | 1,879,433 | 2.30% | 0.0497 |
| 0.15 | 64 | 2,933,999 | 3.59% | 0.0374 |
| 0.25 | 64 | 4,931,495 | 6.03% | 0.0295 |

No point clears even the +-10% necessary condition.  The best, at 6.0% of the dense entry count,
sits at 0.0295 — 5.5x above `5.3605e-3`.  The curve is flattening: 1.0e6 -> 4.9e6 parameters (5x)
moved `F(T)` only 0.0782 -> 0.0295 (2.6x).

## 3. Trend of *this construction* (not a requirement)

A log-log fit over the frontier gives `F ~ params^-0.833` across all 18 points and `^-0.57` over
the last four.  Continued naively, the +-10% necessary line would land at 4e7-1e8 parameters
(roughly 0.4x-1.2x the dense entry count) and the +-3% line far beyond it.

**What this licenses:** band-truncating the teacher's own Cholesky factor is a bad `T`, and buying
more band is not turning it into a good one.
**What it does not license:** any statement about a trained `T`, about the banded class in
general, or about what the architecture needs.  The same limit was already on record for the
radius-0.2 truncation; this measures the whole curve rather than one point on it.

## 4. Why suspect the structural class rather than the budget

An interface Schur complement of an elliptic operator behaves like a Dirichlet-to-Neumann map: it
is **nonlocal**, and its far-field blocks are **low rank** rather than small.  A band keeps
near-field entries and discards exactly the far field that carries the coupling, which is the
shape of failure the table shows.  The matching structural prior is hierarchical low rank
(H-matrix / FMM), not a wider band.  `superelement/factor_fit/hmat.py` puts that class on these
same axes by explicit construction — near-field leaves dense, admissible blocks truncated by SVD —
so each of its points is an *achieved* operator carrying a real full-spectrum `eps_op` audit, the
fourth row of the table in `CLAIM_SCOPE_20260916.md` section 0.  That is evidence about a
construction, and still not a class bound.
