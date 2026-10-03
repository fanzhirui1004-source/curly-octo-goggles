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
- For the objective, see section 3b: the obvious candidate is already measured and it
  fails.

**And it makes route 5 more interesting, not less.** Route 5 predicts the Cholesky factor
of `A^-1`, whose entry magnitudes are dominated by the directions that are soft in `A` —
the whole soft *bulk*, not its extreme. So the existing magnitude-weighted entrywise loss
lands on the broad soft population by construction, which is precisely the failure mode
this experiment just isolated. That arm is running.

## 3b. Correction: the aggregate I proposed was already run, and it fails

Written an hour after section 3, which proposed raising the weight on Codex's
`relative_rows_loss` as "the cheapest remaining experiment". Both halves of that were
wrong, and the experiment was already on disk.

**The weight is not underweighting.** `train_relative.calibrate_weight` sets it to the
median ratio of legacy to auxiliary gradient norm at a frozen start on seats 0253, 0403
and 0206 — raw auxiliary norms 36.0, 20.2 and 1179.2 against legacy 0.50, 0.62 and 30.4 —
so 0.0258 puts the auxiliary at **gradient-norm parity**, not in the margins. The
gradient cosines are 0.0209, 0.0329 and 0.3316, so the term carries genuinely new
information; it is simply pointed the wrong way.

**Codex already ran the A/B**: `RUN_TRAIN32_RELATIVE_R2`, 32 seats, 30 512 steps, arms
`baseline` and `relative`, full spectra on eight seats. Scored in `g`:

| seat | baseline `eps_op` | baseline `g` | relative `eps_op` | relative `g` |
|---|---|---|---|---|
| 186 | 4520 | 7.23e4 | **1515** | **inf** (`mu_min` −7.3e−17) |
| 206 | 8600 | 8601 | **1982** | **inf** (−5.5e−15) |
| 210 | 1.124e4 | 1.921e8 | **6724** | **inf** (−7.1e−15) |
| 212 | 1897 | 2.317e7 | **926.4** | **inf** (−1.8e−14) |
| 253 | 291.1 | 9.278e4 | **266.9** | 1.529e5 |
| 353 | 9557 | 5.021e4 | **938.9** | **inf** (−7.2e−17) |
| 403 | 190.3 | 1511 | **164.6** | 1.592e4 |
| 814 | 1364 | 2.257e6 | **725.5** | **inf** (−4.7e−16) |

The aggregate improves `eps_op` on **all eight** seats, by 1.1x to 10.2x, and worsens `g`
on **all eight** — on six of them by driving `mu_min` to numerical zero or slightly
negative, which means the predicted operator is numerically singular and unusable.

The mechanism is the pathology this whole document is about. `C = R_hat R_*^-1` has
singular values `sqrt(mu)`, so `||C - I||_F^2` sums `(sqrt(mu_i) - 1)^2`, which tends to
**1** as `mu -> 0` and grows without bound as `mu -> infinity`. **The whitened Frobenius
aggregate is capped on the soft side and unbounded on the stiff side, exactly like
`eps_op`.** It therefore optimises `eps_op` beautifully and destroys the soft end. My
claim that it "counts every mode" was correct; the claim that this was sufficient was
not — it counts them with a weight that vanishes where the physics lives.

So the four functionals in play sort out as:

| | form | soft side | breadth | verdict |
|---|---|---|---|---|
| `eps_op = max abs(mu-1)` | max | **capped at 1** | blind | wrong, ranks arms backwards |
| `g = max(mu, 1/mu)` | max | unbounded | **blind** | correct ranking, cannot steer |
| `norm(R_hat R_*^-1 - I)_F^2` | sum | **capped at 1** | counts all | **catastrophic, measured above** |
| `D = sum(mu - log mu - 1)` | sum | unbounded | counts all but **/d dilutes** | GPT Pro's mean-masks-extremes |

Nothing available is both inversion-symmetric and undiluted. The one that is:
`sum_i (f^T v_i)^2 (1/mu_i - 1)`, weighted by the load energy that actually reaches each
mode rather than uniformly over 12 822 of them.

**This strengthens route 5 from a second, independent direction.** Predicting `L` with
`A_hat^-1 = L^T L` turns the inverse-direction residual into a forward quantity in `L`, so
the soft-side objective comes for free in the existing entrywise loss. Section 3 reached
route 5 from the target-magnitude argument; this reaches it from the loss argument.

**And a warning for the generalisation arm.** At 32 geometries the baseline's `g` spans
1511 to 1.921e8 across the eight evaluated seats — five orders of magnitude — against 854
for single-seat memorisation. Multi-geometry is very much harder than memorisation, and
our own 55-seat run should be read with that in mind.

## 4. Provenance

- `docs/data/loss_ab_20260918/S1_LOGPIVOT_SPECTRUM.json` — the log-pivot arm's spectrum
- `docs/data/loss_ab_20260918/SENS_MODEL_LOGPIVOT.json` — its assembled compliance and
  adjoint sensitivity, adjoint identity verified at 3e−15 to 8e−15
- `docs/data/loss_ab_20260918/INVERSE_RESULT.json` — the route-5 label's self-tests
- `docs/data/softend_20260918/` — the pivot-accuracy deciles and the diagonal swap that
  motivated the fix
- `CUTFEM_M4_MULTI_20260917/RELATIVE_CALIBRATION_V2/RESULT.json` — the gradient-norm
  parity calibration that sets the 0.0258 weight
- `CUTFEM_M4_MULTI_20260917/RUN_TRAIN32_RELATIVE_R2/{baseline,relative}/EVAL_030512_*` —
  the already-run aggregate-loss A/B scored in section 3b
