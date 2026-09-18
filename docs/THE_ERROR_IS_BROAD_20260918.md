# The loss fix worked on its target and the physics did not move

2026-09-18. This closes the log-pivot experiment, and its result rules out a whole
family of next steps — including the one I had lined up.

## 1. The A/B

One variable. chol head, seat 0253, 20 000 steps, 8192 pairs per bucket, seed
20260917, same model and sampler; only the diagonal-bucket loss differs. The control
is the stage-1 arm from the night before.

| | absolute (control) | **log pivots** | improvement |
|---|---|---|---|
| `mu_min` | 1.1709e−3 | **3.0442e−3** | **2.60x** |
| `mu_max` | 216.717 | 155.770 | 1.39x |
| `eps_op` | 215.717 | 154.770 | 1.39x |
| **`g` = `1/mu_min`** | **854.05** | **328.49** | **2.60x** |
| `D_per_mode` | 0.24935 | 0.16796 | 1.49x |
| `e_A` | 0.050747 | 0.049854 | 1.02x |
| `factor_relative` | 0.066112 | 0.059551 | 1.11x |
| eigenvalues below 0.9 | 3143 | **3203** | **0.98x (worse)** |
| eigenvalues above 1.1 | 3640 | 3348 | 1.09x |
| **worst assembled sensitivity** | **25.43 %** | **25.16 %** | **1.01x** |
| **worst assembled compliance** | **20.28 %** | **22.41 %** | **0.91x (worse)** |

The loss change did exactly what it was built to do. The measured defect was a 508x
weight deficit on the smallest pivots; removing it improved the soft extreme by 2.60x,
and `g` — the binding metric — by the same 2.60x. Nothing else about the run changed.

**And the physics did not move.** Worst sensitivity error 25.43 % → 25.16 %. Worst
compliance error got slightly worse.

I had recorded a prediction before the run: that improving the pivots could break the
diagonal/off-diagonal error cancellation and make `g` worse. That is not what happened.
`g` improved by 2.6x and it did not matter.

## 2. Why: the error is broad, not extreme

| | outside [0.9, 1.1] |
|---|---|
| chol + absolute | 6783 / 12 822 |
| chol + log pivots | 6551 / 12 822 |

**About half of all 12 822 modes are more than 10 % wrong, and the fix barely changed
that count** — the count below 0.9 actually went up. The single worst direction improved
2.6x; the bulk did not.

So the three network-regime points now available are:

| | `g` | worst sensitivity |
|---|---|---|
| chol + absolute | 854.05 | 25.43 % |
| chol + log pivots | **328.49** | 25.16 % |
| sqrt + absolute | 3057.80 | 542 % |

`g` still ranks correctly across heads (it puts chol above sqrt, which matches the
measurement where `eps_op` did not). But **within one head a 2.6x improvement in `g` buys
1 % of physics**, so `g` is a valid composable bound and not a calibrated predictor.

That is a property of the quantity, not a surprise: `g` is a maximum over 12 822
directions. When the error is spread over thousands of moderately wrong modes, no
single-direction functional can see it, and no fix aimed at the extreme can move it.

## 3. What this rules out, and what it points at

**Ruled out:** any further work targeted at the extreme of the spectrum. That includes
the obvious follow-ups I would otherwise have run — tightening the pivot loss further,
constraining the sqrt head's root to be positive definite, or chasing `mu_min` directly.
The extreme is no longer the binding thing; it was worth 2.6x and it is spent.

**Pointed at:** the acceptance criterion and the objective both have to be **aggregates
over the spectrum**, not extrema.

- For acceptance, the load-weighted form `sum_i (f^T v_i)^2 (1/mu_i - 1)` counts every
  mode and weights it by how much load energy actually reaches it. `g` stays as the
  rigorous composable bound (`g <= 1 + eps` implies `eps` compliance error on any load
  and any lattice size); it is just too loose to steer by.
- For the objective, the whitened aggregate `||R_*^-T A_hat R_*^-1 - I||_F^2 / d` counts
  every mode. **Codex's `relative_rows_loss` is already a Rademacher-probe estimate of
  exactly that**, and it carries a weight of **0.0258** as an auxiliary term. On this
  evidence that may be the most underweighted quantity in the whole objective, and
  raising it is the cheapest remaining experiment: no new labels, no new code, one
  hyperparameter.

**And it makes route 5 more interesting, not less.** Route 5 predicts the Cholesky factor
of `A^-1`, whose entry magnitudes are dominated by the directions that are soft in `A` —
the whole soft *bulk*, not its extreme. So the existing magnitude-weighted entrywise loss
lands on the broad soft population by construction, which is precisely the failure mode
this experiment just isolated. That arm is running.

## 4. Provenance

- `docs/data/loss_ab_20260918/S1_LOGPIVOT_SPECTRUM.json` — the log-pivot arm's spectrum
- `docs/data/loss_ab_20260918/SENS_MODEL_LOGPIVOT.json` — its assembled compliance and
  adjoint sensitivity, adjoint identity verified at 3e−15 to 8e−15
- `docs/data/loss_ab_20260918/INVERSE_RESULT.json` — the route-5 label's self-tests
- `docs/data/softend_20260918/` — the pivot-accuracy deciles and the diagonal swap that
  motivated the fix
