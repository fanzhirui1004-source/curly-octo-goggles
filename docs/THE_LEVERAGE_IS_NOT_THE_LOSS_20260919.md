# Three loss variants, three regressions: the leverage is not in the loss

2026-09-19. The box-only multi-geometry arm with the plain frozen three-bucket loss (absolute
entries, log pivots) remains the best model measured. Every deliberate improvement to the
objective made the physics worse, by a lot.

| arm | loss change | worst compliance | worst sensitivity | seats inside the 3 % sensitivity contract | median smooth-face error | point-load rel median |
|---|---|---|---|---|---|---|
| MULTI_AUGMENT | **plain** | **2.087 %** | **7.789 %** | **11/12** | **0.764 %** | **0.112** |
| S4_60K | `--scale-importance 0.5 --tail-beta 2.0` | 9.439 % | 9.599 % | 1/12 | 2.957 % | 0.279 |
| S4R_60K | `--response-weight 1.0` | 14.711 % | 15.883 % | 0/12 | 4.502 % | 0.338 |

Both were motivated, and both were wrong in the same direction: they take capacity away from the
bulk of the operator to spend it on a statistic that the measurement said mattered.

The response term is the more interesting failure, because it trains *exactly* the functional the
contract measures: `c(f) = ||M_q f||^2`, so the smooth-face compliance error is the relative
error of the label's action on that load family. The defect is variance, not bias. The estimator
subsamples R = 8 rows out of thousands and rescales by `count / R`, an amplification of 500 to
750, so at weight 1.0 it dominates a bucket loss of order 0.006 with almost pure noise — the
arm's final loss is 0.0664 against the baseline's 0.0056. Salvaging it needs variance reduction
(far more rows, or control variates), not a different weight; it is not obviously worth it.

`g` again behaved as an unreliable narrator in both directions: it exploded to 1e2..2e6 on both
arms, and on S4R the physics really was worse, while on the cut-cell single-seat arms it exploded
with the physics fine (see CUT_CELLS_ARE_LEARNABLE_20260919.md). `g` should be read as a
diagnostic of the fit's worst direction and never as accuracy.

## Where the leverage actually is, from the single-seat controls

One geometry, 20 000 steps, plain loss:

| seat | kind | `factor_rel` | median face error | inside 3 % | `g` | final loss |
|---|---|---|---|---|---|---|
| 100051 | full | 0.0130 | 0.393 % | 1.00 | 2.113 | 0.00067 |
| 100032 | 40 % cut | 0.0826 | 1.191 % | 0.85 | 5.17e5 | 0.0124 |
| 100079 | 73 % cut | 0.1802 | 12.726 % | 0.03 | 2.39e9 | 0.100 |

Seat 100051 lets the three regimes be compared on one geometry:

| training set | median face error on seat 100051 |
|---|---|
| that seat alone | 0.393 % |
| 48 presented box-only seats | 0.691 % |
| 56 presented seats, half of them cut | 6.193 % |

**Multi-geometry costs 1.8x. Adding cut cells to the mixture costs another 9x.** So the
interference is specific to mixing cut with full, not to breadth, and the two arms now running
test exactly that: `CUTONLY_32` (32 cut seats, no full cells) and `WIDE_BOX` (box-only at
width 1536, state 256, depth 5, about 2.5x the parameters), both with the plain loss.
