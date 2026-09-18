# The floor is one decile, and 45% of route 5's residual is one number

2026-09-18. The projection probe asked which modes carry the assembled compliance error,
in a frame shared by all five arms: the eigenbasis of the true `A`, eigenvalues ascending.
The comparable statistic is `r_j = (e_j^T A_hat e_j) / lambda_j`, which is 1 when an arm
gets true mode `j` exactly right.

## 1. Where the load energy is

Fraction of load energy `a_j^2 lambda_j` by true-stiffness decile, softest first:

| load | d0 | d1 | d2 | d3 | d4–d9 |
|---|---|---|---|---|---|
| compression_x | **0.9610** | 0.0178 | 0.0155 | 0.0052 | ≤ 0.0003 |
| shear_y | **0.9447** | 0.0395 | 0.0140 | 0.0017 | ~0 |
| shear_z | **0.9448** | 0.0394 | 0.0140 | 0.0017 | ~0 |
| bending_xy | **0.9411** | 0.0226 | 0.0301 | 0.0054 | ≤ 0.0005 |

**94–96 % of the load energy sits on the softest 10 % of the spectrum**, and the compliance
error attributes to that decile and nowhere else — the per-decile attribution is the whole
error in `d0` and `−0.0000` in every other decile, for every arm.

**Nothing above the softest decile can affect the answer.** That is the floor.

## 2. What each arm does, by decile

`r_median` by true-stiffness decile, softest first:

| arm | d0 | d1 | d2 | d3 | d4 | d5 | d6 | d7 | d8 | d9 |
|---|---|---|---|---|---|---|---|---|---|---|
| ① chol + abs | **2.574** | 1.176 | 1.102 | 1.051 | 1.012 | 0.992 | 0.992 | 0.993 | 0.993 | 0.995 |
| ① chol + log | **1.855** | 1.145 | 1.070 | 1.029 | 1.007 | 0.995 | 0.995 | 0.994 | 0.992 | 0.995 |
| ③ sqrt + abs | **1.951** | 1.053 | 1.018 | 1.010 | 1.007 | 1.001 | 1.000 | 0.999 | 0.999 | 1.001 |
| ③ sqrt + log | **1.291** | 1.063 | 1.025 | 1.010 | 1.007 | 1.003 | 1.001 | 1.000 | 0.999 | 0.998 |
| **⑤ inverse + log** | **1.217** | 1.270 | 1.364 | 1.346 | 1.312 | 1.313 | 1.242 | 1.036 | 0.850 | 0.691 |

Fraction of modes more than 10 % off:

| arm | d0 | d1 | d2 | d3 | d4–d9 |
|---|---|---|---|---|---|
| ① chol + abs | 0.990 | 0.797 | 0.531 | 0.037 | ≤ 0.001 |
| ① chol + log | 0.971 | 0.732 | 0.248 | 0.004 | 0.000 |
| ③ sqrt + abs | 0.904 | 0.576 | 0.105 | 0.002 | 0.000 |
| ③ sqrt + log | 0.886 | 0.347 | 0.021 | 0.002 | 0.000 |
| ⑤ inverse + log | 0.879 | 0.856 | 0.980 | 0.998 | **0.989–1.000** |

**The forward-target arms are essentially exact on deciles 3–9 — which carry under 0.5 % of
the load energy.** Seventy per cent of their accuracy is spent on directions that cannot
change the answer. That is a precise statement of misallocated capacity, and it is the
mechanism behind the floor.

**Route 5 is wrong almost everywhere and it wins**, because its errors are bounded
(`r` in [0.443, 5.70] against up to 150.4 for chol) and because the one decile that matters
is the one it gets closest. Its target `A^-1` puts its mass on exactly those modes.

## 3. Different arms, same failure — except route 5

Correlation of `log r_j` between arms, 1 = identical failure pattern:

| pair | correlation |
|---|---|
| chol_abs vs chol_log | **0.9444** |
| chol_log vs sqrt_log | 0.8974 |
| chol_abs vs sqrt_abs | 0.8647 |
| chol_abs vs sqrt_log | 0.8578 |
| chol_log vs sqrt_abs | 0.7971 |
| sqrt_abs vs sqrt_log | 0.7728 |
| **chol_log vs inverse_log** | **0.2207** |
| **inverse_log vs sqrt_log** | **0.2201** |
| **chol_abs vs inverse_log** | **0.2158** |
| **inverse_log vs sqrt_abs** | **0.1644** |

The four forward arms fail on the **same** true modes (0.77–0.94) across both heads and both
losses. **Route 5 does not share that failure** (0.16–0.22). Changing the head or the loss
rearranged the magnitudes without moving the failure; changing the target moved it.

## 4. 45 % of route 5's residual is a single scalar

Route 5's `r_median` is 1.252 overall and 1.217 in the decisive decile — a systematic
over-stiffness, not a mode failure. The first-order optimal global scale is `alpha = m`
where `m − 1` is the first-order compliance error, and it comes out **independently
consistent across all four loads**: 0.8521, 0.8418, 0.8392, 0.8603.

A global scalar zeroes the first-order error identically, so the test has to be the full
re-solve, which is not linear in `alpha`. Scaling the factor by `sqrt(0.8521)` and
re-running the two-cell assembly:

| | worst compliance | worst sensitivity |
|---|---|---|
| ⑤ inverse + log | 17.92 % | 20.41 % |
| **⑤ inverse + log, x0.8521** | **10.73 %** | **11.26 %** |
| best forward arm (① chol + log) | 22.41 % | 25.16 % |

**11.26 % worst assembled sensitivity, 2.2x better than the best forward arm.**

The same test on ① chol + log, at its own first-order optimum `alpha = 0.3586`, gives
**241 % / 242 %** — ten times *worse*. The reason is that first order is only valid for a
small perturbation: route 5's `m = 0.852` is one, chol's `m = 0.359` is not. So this is
also a check on the forward arms' error being too large for linear analysis at all.

`alpha` is derived from the label, so it is a diagnostic and not a deployable number. What
it establishes is that **45 % of route 5's remaining error is bias rather than scatter**,
and bias is the tractable kind. A 25 % stiffness bias is a 12.5 % scale error in `L`, i.e.
a log-pivot offset of 0.118, i.e. the head's normalised output `z` being systematically off
by 0.118 / 1.333 = **0.088** — a small constant offset in one output, very plausibly just
an unconverged mean.

## 5. What to do next

1. **Remove route 5's calibration bias from inside the model.** The cheapest probes: train
   longer, and check whether the head's diagonal output has a systematic offset against
   `pivot_log_mean`. If the bias is a learnable constant, 11 % is reachable without any new
   idea.
2. **Stop supervising deciles 3–9.** Seventy per cent of the sampled blocks train modes
   that carry under 0.5 % of the load energy. A loss restricted to, or reweighted toward,
   the soft subspace is the structural version of what route 5 achieves by its choice of
   target.
3. **The acceptance criterion is now concrete**: the load-weighted sum over the softest
   decile. `g` stays as the composable bound.

## Provenance

`docs/data/floor_20260918/FLOOR_PROBE.json`, `SENS_MODEL_INVERSE_SCALED.json`,
`SENS_MODEL_CHOLLOG_SCALED.json`. Probe: `superelement/objective/floor_probe.py`.
