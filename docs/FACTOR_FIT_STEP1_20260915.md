# Step 1: the free complete factor fits; the difficulty was the coordinate system

Single fixed label (seat 0328, q = 12798, d = 12792, GP teacher). No geometry front-end.
Scripts in `superelement/factor_fit/`; raw results under `STEP1_FACTOR_FIT_R1/` on the GPU box.

## The objective is clean and convex in the operator

For a free upper-triangular student against teacher R*, the exact log-det divergence needs no
spectrum and no probes:

    D(R_hat) = ||R_hat R*^-1||_F^2 - 2 sum_i log R_hat_ii + logdet A - d
    dD/dR_hat = 2 R_hat A^-1 - 2 diag(1/R_hat_ii)          A = R*^T R*

D is convex in A_hat = R_hat^T R_hat (trace is linear, -logdet convex) with its only minimum at
A_hat = A. Evaluator check: D(R*) = 0.000000e+00, trace term = 12792.000000 = d, exactly.

## Every natural start is in the same place, so the start was never the problem

| start | D/d |
|---|---:|
| 0.01 I (what N1 used) | 7.44 |
| c* I, c* = sqrt(d / tr A^-1) = 4.13e-3, the best scalar | 4.35 |
| diag(R*) | 5.24 |
| diag(sqrt A_ii) | 7.34 |

The exact diagonal of the teacher is worse than a uniform scalar: the divergence lives entirely
in the off-diagonal structure. N1 ended at D/d = 6.84 after 100 steps, i.e. it had not moved.

## The curvature is set by dof compliance, column by column

    d2D / dR_hat_ij^2 = 2 (A^-1)_jj          (+ 2/R_ii^2 on the diagonal)

(A^-1)_jj is the compliance of trace coordinate j. Across the module it spans 2.7e3, so one
learning rate cannot serve stiff and floppy columns at once. Scaling column j by
1/sqrt((A^-1)_jj) makes the curvature exactly 2 everywhere; multiplying the gradient on the right
by A is the Newton step of the quadratic part. A log-diagonal reparameterization fixes only the
diagonal block (spread 4.9e3 -> 10.9) and the diagonal is not where the work is.

## Same start, same optimizer, same lr, only the coordinates change

| parameterization | 1000 Adam steps, lr 1e-3 | whitened mu |
|---|---:|---|
| raw (N1's) | NaN before step 100 | [2.5e-5, 1.7e4] |
| log-diagonal | 4.35 -> 4.85, non-monotone | [2.9e-9, 1.6e4] |
| column scaling | 4.35 -> 1.25, monotone; 0.081 at 4500 | [1.2e-3, 17] |
| Newton direction, no line search | NaN | - |
| Newton + Armijo backtracking | 4.35 -> 0.026 in 60 steps | [0.043, 22] |

The raw NaN reproduces "plain Adam went badly conditioned after two updates" from the N1 record.
The Newton NaN is the log barrier: the quadratic model's full step walks the diagonal through
zero; backtracking is the standard fix and it is what changes the step count from thousands to
hundreds.

## Run to the gate

Newton + Armijo, float32 steps, 228 steps, 25 min:

    step  100   D/d 4.54e-3
    step  150   D/d 5.87e-4
    step  200   D/d 6.72e-5      <- below the 1e-4 target gate
    step  228   D/d 2.11e-5      <- line search returns t = 0: float32 floor, not convergence

Final whitened spectrum: mu in [0.9558, 1.0468]; 0 modes below 0.9; 0 above 1.1;
12709 of 12792 (99.35%) inside +/-3%. Work gate passed outright; target gate on 99.35% of
directions. Against N1: D/d 6.84 -> 2.1e-5, mu range from six decades to +/-5%.

So the three questions separate: the representation reproduces the teacher (N0, and now this);
the optimization was the failure, and its mechanism is identified; geometry learning is untouched.

## What step 1 does not establish, and the problem it exposes

This fits one label with the teacher in hand. It is a reachability proof, not learning. And both
working methods use the teacher: the column scale needs diag(A^-1), the Newton step needs A. A
network has neither at prediction time, and in coordinates without that scale the measured result
is NaN. Step 2's first question is therefore where the scale comes from.

### Self-preconditioning does not supply it

Column scale refreshed every 100 steps from the student's own diag(A_hat^-1), with Adam's moments
carried into the new coordinates (the first run omitted that and was discarded):

| scale source | D/d at 3000 | log-corr of scale with teacher's |
|---|---:|---:|
| teacher | 0.289 | 1.000 |
| teacher start, then self-refresh | 1.248 | 0.963 |
| flat start, then self-refresh | NaN | 0.994 (last finite refresh) |

Even starting from the exact scale, refreshing from the student is 4.3x slower, and the refreshed
scale drifts to 0.92 correlation before recovering. A 0.96-0.99 log-correlation still leaves a
residual spread that one learning rate cannot absorb; correlation is the wrong statistic, spread
is the right one. Cold start diverges as the raw parameterization does.

### Geometry does supply it

Compliance per physical trace dof, c_k = ||column k of R*^-T B||^2, regressed on local features
only (support weight w, position, box-face id, near-field degree, signed cut-plane distance, 5^3
neighbourhood means of the 11 grid channels, component). Held out: 20% of nodes with all three
components together. The statistic that matters is the curvature spread left after dividing by the
prediction; ~10 is enough for one learning rate.

| model | R^2 (test) | residual spread | 5-95% core | as column scale |
|---|---:|---:|---:|---:|
| constant | 0 | 7.3e3 | 1.6e3 | 86 |
| log w alone | 0.867 | 692 | 21 | 26 |
| ridge, linear | 0.914 | 45 | 12 | 6.7 |
| **ridge, quadratic** | **0.981** | **14.4** | **3.1** | **3.8** |
| 4-NN spatial | 0.690 | 9.1e3 | 148 | 96 |

A quadratic function of local geometry takes the curvature spread from 7.5e3 to 14 on held-out
nodes (3.1 over the central 90%), and the column scale to 3.8. The spatial-neighbour baseline at
R^2 0.69 shows this is carried by the features, not by smoothness. The ruler can be a per-node
head.

## The complete factor is the right label and the wrong output

Right label: exact, PSD by construction, and the whitening in D is two triangular solves against
it. Freeze R*.

Wrong output, for three reasons, one of them measured here and one measured earlier:

1. Column j of an upper Cholesky factor is determined by S[:j, :j]: everything before j in the
   elimination order, a spatial region under Morton order, not a neighbourhood. A network cannot
   produce that from local features without performing the elimination itself, and any geometry
   change that shifts the ordering rewrites half the factor.
2. Its parameterization's curvature is the answer's compliance (this note).
3. Truncating R* to a spatial band of radius 0.2 drops only 2.6% of its Frobenius norm and still
   damages ~300 modes by more than 10% with mu_max 86: the small far entries matter collectively
   (`TRUNCATION_0328_*.json`, 2026-09-12; packet, RCM and raster orders agree). A rank-k softening
   correction recovers the over-stiff side to mu_max 1.016 / 1.0038 / 1.0011 at k = 512 / 1500 /
   3000; the over-soft side (mu_min 0.09) cannot be softened away and must come from a refit of
   the local factor.

So the network's output should be local and PSD by construction, with the compliance scale as a
per-node predicted quantity:

    S_hat = L^T (I + M M^T)^-1 L,   L banded on near-field pairs, columns scaled by the predicted
    1/sqrt(c_j), M low-rank

which is the existing `Operator` class with the ruler made explicit. The remaining capacity
question is whether a refit banded L plus rank ~1500 M reaches +/-3% on the teacher; the earlier
attempt at that was made without the column scaling and is presumed optimization-limited for the
same reason N1 was.
