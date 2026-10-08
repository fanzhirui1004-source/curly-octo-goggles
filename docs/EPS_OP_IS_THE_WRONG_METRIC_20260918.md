# The acceptance metric ranked the two arms backwards

2026-09-18. This retires `eps_op = max|mu - 1|` as the headline acceptance quantity and
replaces it with the inversion-symmetric constant. It also withdraws an extrapolation I
made a few hours earlier, and corrects one of my own three hypotheses.

## 1. The measurement that broke it

`THE_REAL_GATE_20260917.md` measured a *constructed* module at `eps_op` 0.152 (0.37 %
sensitivity error) and `E1b` measured Codex's modules (41–48 %). Everything between was a
two-point extrapolation, and I had just used it to argue that `eps_op` ~ 4 would buy 3 %
sensitivity error. So I ran the same two-cell assembly and the same exact adjoint on the
factors the two stage-1 arms actually predicted.

Seat 0253, 23 508 assembled trace dofs, 716 shared nodes, four load cases, adjoint
identity `(sum dc/drho + c)/c` verified at 3e−15 to 8e−15 on the exact arm.

| arm | `eps_op` | worst compliance error | worst sensitivity error |
|---|---|---|---|
| chol | 215.72 | **20.3 %** | **25.4 %** |
| sqrt | **48.03** | **452 %** | **542 %** |

I predicted ~16 % for the sqrt arm. It is 542 %. And the arm with `eps_op` **4.5x larger**
is **21x better** on the physics. `D_per_mode` (0.249 vs 0.127) and `e_A` (0.0507 vs
0.0265) also both rank them the wrong way round. **All three metrics this project has been
steering by are inverted relative to the quantity in the contract.**

## 2. Why: the metric is capped on the side that matters

`eps_op = max_i |mu_i - 1|` is structurally asymmetric.

- Too stiff: `mu` is unbounded above, so it contributes without limit.
- Too soft: `mu >= 0`, so `|mu - 1| <= 1` — a direction that is a **thousand times too
  soft** contributes 0.999.

Compliance is `c = f^T K^-1 f`, and its sensitivity is `-u^T S u`. Both live on the soft
end. `eps_op` is very nearly blind there.

The numbers say exactly this. Both arms are soft-limited, and `eps_op` was reporting the
non-binding side in both cases:

| arm | `mu_max` (over-stiff factor) | `1/mu_min` (under-stiff factor) | binding side |
|---|---|---|---|
| chol | 216.72 | **854.05** | soft |
| sqrt | 49.03 | **3057.80** | soft |

## 3. The replacement

$$g \;=\; \max_i \max\!\left(\mu_i,\; \tfrac{1}{\mu_i}\right)
\qquad\Longleftrightarrow\qquad
\tfrac{1}{1+\varepsilon} A \preceq \hat A \preceq (1+\varepsilon) A$$

the Thompson metric `max(log mu_max, -log mu_min)` on the SPD cone. It is symmetric under
inversion, so it controls `A` and `A^-1` together; it composes exactly under assembly, as
the one-sided sandwich was only ever supposed to; and it costs nothing extra, being a
different function of the same eigenvalues.

| | `eps_op` | `g` | measured sensitivity |
|---|---|---|---|
| chol | 215.72 | **854.05** | 25.4 % |
| sqrt | 48.03 | **3057.80** | 542 % |
| ranking | `[sqrt, chol]` **wrong** | `[chol, sqrt]` **right** | `[chol, sqrt]` |

`run.py` now reports `eps_op`, `under_stiff_factor` and `g` in `SPECTRUM.json`. Every run
already saved its full `EIGENVALUES.npy`, so **every past run can be re-scored without
recomputing anything**, including the two generalisation arms now in flight.

Why the earlier extrapolation failed: at `eps_op` 0.152 the constructed module has
`g ≈ 1.18 ≈ 1 + eps_op`, so the two metrics coincide in the small-error regime and diverge
once the error reaches O(1). I extrapolated three decades out of the region where they
agree.

Re-doing it in the right variable: `(g = 1.18, 0.37 %)` to `(g = 854, 25.4 %)` gives a
log-log slope of 0.642, so 3 % sensitivity error wants **`g ≈ 31`**, against 854 today —
a factor of **28**, and it has to come from the soft end.

## 4. One of my three hypotheses was wrong, and it is provable on paper

I had suggested the sqrt arm's soft-end collapse might be caused by its predicted root
being indefinite (`det` sign −1). It cannot be. With `M_hat = V L V^T`,

$$\hat A = \hat M^2 = V L^2 V^T$$

which is invariant under the sign of any eigenvalue. Verified numerically rather than
asserted: flipping every negative eigenvalue changes `A_hat` by a relative **1.48e−14**.
Only 13 of 12 822 eigenvalues are negative, the most negative is −8.5e−4, and their energy
fraction rounds to 0. The indefiniteness is a statement about how far from the positive
definite target branch the fit landed — relevant to the gauge argument for route 3, and
irrelevant to the operator.

## 5. Where the soft end is controlled from — two answers that disagree

**In the linear regime.** Perturb one magnitude stratum of the *true* factor at a time by
1 % relative, random signs, and watch the spectrum:

| perturbed | entries | `eps_op` | `g` |
|---|---|---|---|
| **diagonal only** | **12 822** | **1.385** | **2.43** |
| off-diagonal deciles 0–8 | 65.8 M | ≤ 0.0013 | **1.00** |
| off-diagonal decile 9 (largest) | 8.2 M | 0.5865 | 1.60 |

So at 1 % relative error, 12 822 numbers — 0.016 % of the factor — do more damage than
65.8 million, and eight of the ten off-diagonal deciles do nothing measurable.

**At the network's actual error level, the split is different.** Swapping diagonals
between the prediction and the truth (chol arm; the swap is meaningless for a symmetric
root and is not reported for it):

| | `g` |
|---|---|
| as predicted | 854.05 |
| predicted diagonal + **true** off-diagonals | **38.03** |
| **true** diagonal + predicted off-diagonals | **1202.55** |

Giving it the true diagonal makes it **worse**. So the network's diagonal and
off-diagonal errors are correlated in a way that partially cancels, and removing half of
the error breaks the cancellation. It has learned a partially self-consistent factor, not
an accurate one — the coherence result from `WHERE_THE_ERROR_IS_20260917.md`, localised.

Both readings are correct in their own regime, and the 1 % oracle must not be quoted as
"65.8 million entries do not matter" at the network's error level. It says the *linear
response* is concentrated on the diagonal and the top decile.

## 6. A defect in the loss, measured twice

The network's relative pivot error, by decile of the **true** pivot, smallest first:

| arm | smallest decile | … | largest decile | median | worst |
|---|---|---|---|---|---|
| chol | **0.222** | 0.125, 0.0703, 0.0289, 0.0149, 0.00874, 0.00634, 0.00724, 0.0064 | **0.00524** | 0.0138 | 1.336 |
| sqrt | 0.206 | 0.362, 0.246, 0.0736, 0.0726, 0.0463, 0.0244, 0.0694, 0.0558 | 0.107 | 0.0842 | 3.579 |

The smallest pivots are **42x less accurate** than the largest, monotonically. Two
mechanisms cause it and they multiply:

1. the diagonal bucket residual is divided by **one global** `diagonal_block_rms` =
   0.0453, while the pivots span `e^-7.10` to `e^-1.95`. The smallest pivot therefore
   contributes `(8.2e-4 / 0.0453)^2 ≈ 3e-4` of a typical term.
2. the head emits pivots as `exp(mean + std * z)`, so `d(pivot)/dz` is proportional to the
   pivot, suppressing small-pivot gradients by that range all over again.

Measured directly in the loss, on a synthetic block set:

| +10 % relative error on | legacy loss | log-domain loss |
|---|---|---|
| a perfect prediction | 0 | 0 |
| the **smallest** pivot | 3.393e−9 | **4.4578e−6** |
| the **largest** pivot | 1.725e−6 | **4.4578e−6** |

The legacy objective weights an identical *relative* error **508x** less on the smallest
pivot. `--diagonal-loss log` scores pivots as `(log p_hat - log p_star)/pivot_log_std`,
uniform relative weight across all five decades, leaving every other entry alone.

**A prediction, stated before the result.** Section 5 says the current fit survives partly
by cancellation between diagonal and off-diagonal error. Improving the pivots could break
that cancellation and make `g` *worse* rather than better; the off-diagonals are trained
jointly and should re-adapt, but that is not guaranteed. The arm is running against the
existing stage-1 chol arm as its control (same seat, seed, steps, sampler, head — only the
diagonal loss differs), and `g` = 854.05 is the number to beat.

## 7. What this does not change

- The two generalisation arms in flight do **not** need restarting. The metric fix is post
  hoc, so they will yield correct `g` numbers from the eigenvalues they already save; and
  the chol arm is the first generalisation measurement this project has, which is worth
  having under the legacy loss precisely because it is the baseline.
- The chained `A^{1/2}` label build and sqrt generalisation arm **were** cancelled: 1.8 h
  of GPU and 214 GB to answer a head question posed by a metric that has just been
  retired, when the soft end is the binding constraint for both heads.
- The two-cell assembly test stays a *validation*, not an acceptance criterion: one
  assembly, one size, four loads, and far too expensive to run inside training. Its job is
  to check that the criterion predicts the physics — which is exactly the job it did here,
  by failing.
