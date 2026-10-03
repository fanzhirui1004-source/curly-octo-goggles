# What the objective can and cannot fix: a correction, with measurements

Written 2026-09-17. **This document supersedes an earlier draft of itself that
claimed the divergence term `D` was "the missing gradient" for the gate. That claim
was wrong and is withdrawn in section 3.** Everything here is verified in
`superelement/objective/` (tests green on the box's torch) or measured on Codex's
saved factors (`/root/autodl-tmp/CLAUDE_DIAGSPLIT_20260917`).

## 1. What Codex's two losses actually are

**`baseline`** (`factors.bucket_loss`) is not naive Frobenius. It splits the factor
into diagonal / same-patch / cross-patch buckets, normalises each by one global RMS
from `factors.calibrate`, and averages the three equally; `GeometryMixture`
importance-samples near-cross pairs with an unbiased correction. This is better than
`docs/CODEX_M4_ASSESSMENT_20260917.md` credited it: the bucket split does give the
cross bucket 1/3 of the weight. What survives is narrower - *within* the cross
bucket a single global RMS leaves the nearest 0.444% of pairs carrying 97.09% of the
squared norm.

**`relative`** (`relative_action.relative_rows_loss`) is an auxiliary term added on
top, weight set by gradient-norm matching. With Rademacher `Z`, `x = R_star^-1 Z`,
`result = R_hat x`, its expectation is exactly

        L_rel = || M - I ||_F^2 / d,      M = R_hat R_star^-1,   H = M^T M,   mu = eig(H)

a **whitened** Frobenius error. Per mode its contribution is at best
`(sqrt(mu) - 1)^2`. Reproducing Codex's `LOSS_IDENTITY.json` from that closed form
matches to ten decimal places (`verify_logdet_identity.py`).

## 2. The asymmetry is real

| mu | `(sqrt(mu)-1)^2` | `D = mu - log mu - 1` | d/dt of each | ratio |
|---|---|---|---|---|
| 275 | 272.0 | 268.4 | 516.83 / 548.00 | **1.06** |
| 100 | 81.0 | 94.4 | 180.00 / 198.00 | 1.10 |
| 10 | 6.2 | 6.7 | 13.68 / 18.00 | 1.32 |
| 1e-2 | 0.810000 | 3.615 | -0.1800 / -1.9800 | 11 |
| 1e-6 | 0.998001 | 12.816 | -1.998e-3 / -2.0000 | 1.0e3 |
| 1e-12 | 0.999998 | 26.631 | -2.000e-6 / -2.0000 | **1.0e6** |

`L_rel` saturates at 1 as `mu -> 0` and its gradient vanishes, so a collapsed mode
is free to stay collapsed. Two measured consequences:

1. **Collapse.** The relative arm's `mu_min` is 3.9e-9, 2.2e-12, 4.2e-15, 1.2e-15
   and, on 0186 at 40000, **-1.35e-16** - so far under `mu_max = 2211` that the
   eigensolver cannot resolve it and Codex correctly reports `D_spectral: None`.
2. **A hedging bias with the right sign.** If a mode's correct log-scale is known
   only to `sigma`, the minimiser sits at `-1.5 sigma^2` for `L_rel`, `-1.0 sigma^2`
   for `D`, and `0` for Jeffreys `mu + 1/mu - 2` (verified to 4 decimals in
   `verify_hedging_bias.py`). Measured `closed_logdet_gap` is **positive on all 8
   baseline rows** (+2263..+4718, too stiff) and **negative on all 8 relative rows**
   (-10651..-25205, too soft). Adding the relative auxiliary flipped the global bias.

## 3. Why this does NOT fix the gate - the correction

`eps_op = max|mu - 1|`. For baseline/0253 the spectrum is `[0.001097, 274.9]`, so
`eps_op = 273.9` comes **entirely from `mu_max`**; the soft end can contribute at
most 1. And on the stiff side the two objectives agree to within 6% (table above,
first row). **Therefore adding `D` cannot be expected to move `eps_op`.**

Worse for the "wrong objective" story: one mode at `mu = 275` already carries
`(mu - log mu - 1)/d = 2.09e-2` of a total `D = 0.524`, i.e. **4.0%** of the
objective. The worst mode is not invisible to the current loss. It is being pushed
on, hard, and after 40000 steps it has not moved.

What `D` is still worth: it removes the soft-side cliff and the softness hedging, so
`mu_min` stops collapsing and `logdet(H)` stops drifting by +-1.4e4. That is a real
defect - a whitened condition number above 1e16 is not a usable operator - but it is
not the acceptance criterion, and it should be sold as hygiene, not as the fix.

It is also free, which is the one part of the earlier draft that stands:
`d*D = tr(H) - logdet(H) - d`, where `tr(H) = ||M||_F^2` is `result.square().mean()`
on the tensor `relative_rows_loss` already forms (measured unbiased: 0.992210 vs
0.992212 exact), and `logdet(H) = 2(sum log R_hat_ii - sum log R_star_ii)` is exact
from the predicted pivots, because `output[diagonal] = lower + diag_embed(exp(log_pivot))`
with `lower = tril(...,-1)` and `factors.sample_pairs` decodes the **complete**
diagonal every step. `M` is triangular so `diag(M) = diag(R_hat)/diag(R_star)`: the
identity is exact, and it is more accurate than the eigendecomposition where the
spectrum is extreme (12.11% D error via `eigvalsh` at `mu_min = 1e-18`; Codex's own
`divergence_spectrum_absolute_difference` shows the same, 27.46 on 0353).

## 4. Where the gate error actually lives - measured

New experiment (`CLAUDE_DIAGSPLIT_20260917`), on the saved 40000-step predicted
factors. Nothing trained, nothing of Codex's modified. Split the factor by **role**
rather than by distance: the diagonal (the pivots, which alone set `logdet H` and
which the network emits as an explicit bounded `log_pivot`) versus everything above.

| case | baseline/0253 | relative/0253 | baseline/0403 | relative/0403 |
|---|---|---|---|---|
| as predicted | **273.9** | 264.0 | 174.8 | 172.6 |
| teacher **diagonal**, predicted off-diagonal | **275.2** | 236.0 | 173.9 | 149.2 |
| teacher **off-diagonal**, predicted diagonal | **51.4** | 43.0 | 65.7 | 20.4 |
| teacher everything (self-test) | **0** | 0 | 0 | 0 |

`as_predicted` reproduces Codex's reported 273.9 and `D = 0.524` exactly, and the
self-test returns 0, so the harness is sound.

**Giving the network every pivot exactly changes `eps_op` by nothing** (273.9 ->
275.2 on baseline/0253; it gets slightly worse). Giving it the off-diagonals cuts
`eps_op` 5.3x. The gate error is in the off-diagonal blocks.

Two consequences:

- The log-pivot regression term proposed in the earlier draft is **demoted**. It
  would improve `D` and `logdet` and would not move the gate. Keep it as a cheap
  regulariser at most.
- Even with *every* off-diagonal exact, `eps_op` is still 20-66, i.e. 700-2200x the
  target gate, and the residual then lies in the pivots that were harmless a moment
  ago. Same non-additivity as Codex's distance-bucket substitution: the pieces must
  be **jointly** coherent, and no single piece's repair rescues the gate.

## 5. What this leaves

The objective is not the bottleneck on the side that decides acceptance. The
remaining candidates are the representation's accuracy on the off-diagonal blocks,
its capacity, and whether a mean-type loss of any shape can move a max. The
discriminating measurement is required-versus-achieved per-entry accuracy on these
same factors - running as `CLAUDE_ORACLE_20260917`. Until it reports, the honest
statement is: **we know where the error is (off-diagonal), we know the objective
shape is not the reason it is there, and we do not yet know whether the
representation could hold the answer at all.**
