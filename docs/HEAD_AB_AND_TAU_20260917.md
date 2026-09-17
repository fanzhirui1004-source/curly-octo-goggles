# Two measurements: the head matters, and the physics is smooth in tau

2026-09-17, night. Both are single-configuration results with self-tests; the caveats
are stated at the end of each section and are not decorative.

## 1. Stage 1: Cholesky head versus symmetric square-root head

One variable. Same encoder and decoder, same 5,077,241 parameters, same pair sampler,
same three-bucket normalised factor MSE, same optimizer, schedule, warmup, seed, seat
(0253) and step budget (20,000 at 8192 pairs per bucket). Both arms ran concurrently on
one GPU at the same 0.169 s/step and the same 11.91 GB peak.

The only difference is what the head emits and how the prediction is assembled:
`A_hat = R^T R` with `R` upper triangular, versus `A_hat = M M` with `M` symmetric. Both
are scored by the same yardstick, `eps_op = max|mu - 1|` with
`mu = eig(R_*^-T A_hat R_*^-1)` and `R_*` the Cholesky reference in both cases.

| metric | chol | sqrt | |
|---|---|---|---|
| **`eps_op`** | **215.72** | **48.03** | sqrt 4.5x better |
| `mu_max` | 216.72 | 49.03 | |
| `mu_min` | 1.1709e−3 | 3.2703e−4 | **chol 3.6x better** |
| `e_A` | 5.075e−2 | 2.655e−2 | sqrt 1.9x better |
| `D_per_mode` | 0.2493 | 0.1272 | sqrt 2.0x better |
| **`factor_relative`** | **6.61e−2** | **18.93e−2** | **chol 2.9x better** |
| eigenvalues below 0.9 | 3143 | 2440 | of 12 822 |
| above 1.1 | 3640 | 2817 | |
| below 0.97 | 5075 | 4255 | |
| above 1.03 | 5256 | 4558 | |
| numerical gate | pass | pass | |

Three things are worth separating.

**The square-root head is better at the operator level, by a lot.** 215.7 → 48.0 in the
acceptance metric, and fewer eigenvalues outside every band. Nothing else changed.

**It gets the factor entries more wrong while getting the operator more right.**
`factor_relative` is 0.189 against 0.066 — the predicted numbers are three times further
from their targets — yet `e_A` is half and `eps_op` is a quarter. That is the coherence
story from `WHERE_THE_ERROR_IS_20260917.md` seen from the other side: what matters is not
how big the entry errors are but how they line up. The square-root parameterisation
appears to make errors that cancel in the product.

**It is worse on the softest direction.** `mu_min` is 3.6x smaller, so the most
under-stiff mode got more under-stiff. `eps_op` is set by `mu_max` for both arms, so the
headline is unaffected, but the soft side did not improve — it regressed.

**The predicted root is indefinite.** `det(M_hat)` has sign −1, so `M_hat` is not a
matrix square root of anything positive definite; it is a symmetric matrix whose square
approximates `A`. `A_hat = M_hat M_hat` is positive semidefinite regardless, so `eps_op`
is well defined and the run is valid. But it qualifies the gauge argument that motivated
route 3: the uniqueness of `A^{1/2}` is a property of the *positive definite* root, and
the network is free to land on a different branch. How many eigenvalues of `M_hat` are
negative, and whether constraining them helps or hurts, is not measured.

**What this does and does not establish.** One seat, one seed, one budget: n = 1. It does
falsify the specific claim it was built to test — "the Cholesky gauge is not what limits
line ①" is now the losing hypothesis, because removing that gauge and changing nothing
else bought 4.5x. It does not establish that route 3 reaches the gate: 48.03 is still
320x above the 0.15 gate.

An honest accounting of the process: the sqrt arm trained all 20,000 steps and then
crashed in a guard I had written into `general_divergence`, which rejected a negative
`det(C)`. That guard was wrong — `W = C^T C`, so `logdet(W) = 2 log|det C|` and the sign
is irrelevant. The arm was re-evaluated from its final checkpoint with the weights and
normalisers the run ended with; nothing was retrained.

## 2. The tau gate: the acceptance metric jumps, the physics does not

Same five operators as `PIPELINE_UNBLOCKED_20260917.md` section 3, seat 0328, one
coordinate system. Bottom face clamped, top face a rigid platen given six unit motions;
`H` is the 6x6 platen stiffness, and the observable is `trace(H^-1)`, the compliance
summed over six unit loads.

| `eps` | `trace(H^-1)` | relative change | `eps_op` vs base |
|---|---|---|---|
| −1/1000 | 6108.1984063384 | **+2.282e−3** | 0.153 |
| −1/10000 | 6095.6790343656 | **+2.279e−4** | 0.016 |
| 0 | 6094.2903160099 | 0 | — |
| +1/10000 | 6092.9020434518 | **−2.278e−4** | **0.598** |
| +1/1000 | 6080.4284180583 | **−2.275e−3** | **0.675** |

Free-equilibrium residuals are 1.04–1.11e−13 for all five, so the solves are exact.

The compliance is linear and symmetric to four digits:

| | |
|---|---|
| one-sided slopes at `h` = 1e−4 | −13887.18 (left), −13882.73 (right); **ratio 0.99968** |
| one-sided slopes at `h` = 1e−3 | −13908.09, −13861.90; ratio 0.99668 |
| kink, `h` = 1e−4 | **3.2e−4** |
| kink, `h` = 1e−3 | 3.3e−3 |
| Richardson: central(1e−4) / central(1e−3) | **0.999997** |
| `dc/deps / c` | **−2.2784** |

The kink shrinks by a factor of ten when `h` shrinks by a factor of ten. That is the
expected `O(h)` second-order term of a smooth function, not a discontinuity. And the two
central differences agree to 3e−6, so −13885 is a converged derivative, not a difference
artifact. All six platen eigenvalues move monotonically across all five `eps` with no
jump at +1e−4.

So, stated sharply:

> The birth of one cut cell makes the acceptance metric asymmetric by a factor of **38**
> (`eps_op` 0.016 at `eps` = −1e−4 against 0.598 at +1e−4) at a point where the compliance
> is symmetric to **0.04 %** (2.279e−4 against 2.278e−4).

The `eps_op` jump lives entirely on directions that carry no compliance.

**What this settles.** GPT Pro's derivative-oscillation objection was, correctly, that
value accuracy does not imply derivative accuracy, and that `tau` is the design variable.
Measured here, the teacher's compliance is smooth and its `tau` derivative is well
defined and converged, in the exact place where the acceptance metric is not. That is
consistent with `THE_REAL_GATE_20260917.md`, where module `eps_op` 0.152 produced 0.31 %
compliance error and 0.37 % sensitivity error: `eps_op` is a conservative proxy that
over-weights physically irrelevant directions.

**What it does not settle.** This is the smoothness of the *teacher*, not of a network's
prediction. A network could be non-smooth in `tau` where the teacher is smooth, and
testing that needs `tau` labels on more seats plus a trained model — 180 s per cell per
`eps`, per `PIPELINE_UNBLOCKED_20260917.md` section 4. It is also one seat, one direction
in design space (uniform scaling of all eight `tau` corners), and one cell birth; a
different perturbation could cross a worse threshold. And the observable is a single
cell's six-platen response, not an assembled lattice — the assembled case is what
`THE_REAL_GATE` measured.

## 3. Consequence for the plan

The stage-1 result is not a tie, so by the decision rule set before it ran, the stage-2
sampling arms are not the best use of the remaining night: the sampling hypothesis was a
statement about the single-seat memorisation regime, and the head effect is larger and
now known. The label expansion to 266 training geometries is running; the night goes to
the first generalisation measurement this project has ever made, with the square-root
head.
