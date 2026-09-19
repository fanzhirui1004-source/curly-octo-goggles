# The network carries the teacher's tau sensitivity to 1.2 %

2026-09-19. Seat 0328, **holdout** (never presented to any model), box-only cell, `d` = 12792,
`q` = 12798. Truth side: `superelement/objective/tau_gate.py`, run 2026-09-17. Prediction side:
`superelement/objective/tau_predict.py`, new. Data in `docs/data/tau_derivative_20260919/`.

This is the third of the project's three claims - the operator family is learnable, the modules
compose, and the result is differentiable in the design variable - and it is the one that had
never been tested.

## 1. What is on disk and why the comparison is exact

`CUTFEM_FULL_FACTOR_LOCAL_GEOMETRY_20260913T2330/ANALYSIS_R1/PATH_{1..4}/R_UPPER.npy` are four
teacher-built operators for seat 0328 with `tau_corners` scaled by `1+eps`, `eps` = -1e-3, -1e-4,
+1e-4, +1e-3, alongside the base at `CUTFEM_SUPERELEMENT_LABELS/REFERENCE_0328/R_UPPER.npy`. All
five are the packed upper triangle of `d` = 12792, every sha256 matches its own receipt, and all
four are recorded `ADMITTED_SAME_TRACE` at `q` = 12798 with the same ordering and quotient
reflectors agreeing to 2.9e-15.

The protocol takes the base trace cache, scales `tau_corners` by `1+eps` in memory, and compiles
the network's context from it. The precondition as first stated - that the trace CSR is
bit-identical across the five geometries - is **false**, and the teacher's own records say so:
`GEOMETRY_R1/PATH_k_H0/TRACE_COMPARISON.json` has `all_arrays_equal` false for `indices`,
`background_nodes` and `rigid` (up to 3967 of 4266 index entries differ; `background_nodes`
changes length by -64 to +120; `rigid` moves by up to 1.6e-4). The protocol is nevertheless exact,
for two reasons that were verified directly rather than assumed:

* `background_nodes[indices]` - the only form in which the featurizer ever uses either array
  (`context.py:250`) - is **bit-identical** across all five geometries;
* `rigid` moves only within its own span (exactly two of six columns, i.e. a shift of the rotation
  reference point), so `span(rigid)` and the Householder quotient `B` agree to 1e-15, which is
  what the recorded reflector differences say.

Everything else the featurizer reads is bit-identical except `tau_corners` itself. The context is
then **exactly affine in the tau scale**: second differences of every moving feature are at
machine epsilon, so the network's predicted compliance is analytic in `eps` by construction.

The numerical floor of the whole comparison is the teacher's own frozen-vs-replay difference,
`e_A` = 2.46e-8, four orders below the smallest signal.

## 2. The teacher's derivative exists and is clean

The observable is the platen: bottom face clamped, top face a rigid platen given six unit motions,
`H` the 6x6 platen stiffness, the scalar `trace(H^-1)`.

| `eps` | -1e-3 | -1e-4 | 0 | +1e-4 | +1e-3 |
|---|---|---|---|---|---|
| `trace(H^-1)` | 6108.198406 | 6095.679034 | 6094.290316 | 6092.902043 | 6080.428418 |

`d ln trace(H^-1) / d eps`: at `h` = 1e-4, left -2.27872, right -2.27799, central **-2.278355**;
at `h` = 1e-3, left -2.28215, right -2.27457, central -2.278360. One-sided kink 0.03 % at the
small step, Richardson ratio 0.999997, free-equilibrium residual exactly 0.0 in all five cases.

This settles a question open since 2026-09-13. Between `eps` = 0 and +1e-4 exactly one CutFEM cell
is born (7478 -> 7479 active cells) and that single cell takes the whitened spectrum's `mu_max`
from 1 to 1.598, four times the old acceptance gate. **It does not touch the physics.** The
clamped platen removes the near-mechanism the born cell stiffens, so the observable the project
must get right is smooth exactly where the acceptance metric jumps.

The free cell is a different story, and the same run shows it: for `c(f) = f^T A^-1 f` over 96
smooth face, 32 smooth global and 32 point loads, the truth's own one-sided log-slopes at
`h` = 1e-4 are -6.09 (left) and -21.36 (right), a factor 3.5. A free cell's compliance is
dominated by its softest direction, which is precisely what the born cell changes by 60 %, so the
free-cell compliance is **not** a differentiable function of tau at this point. In a lattice the
neighbours constrain that direction, which is the platen situation, not the free one. The network,
being analytic in `eps`, reproduces the constrained derivative and cannot reproduce the free one's
kink - and does not need to.

## 3. The network's derivative

| model | parameters | value error at `eps`=0 | derivative error `h`=1e-4 | `h`=1e-3 | own one-sided kink |
|---|---|---|---|---|---|
| MULTI_AUGMENT | 9.7 M | -0.769 % | **+1.92 %** | +1.88 % | 0.0036 |
| **ensemble of the two** | - | -0.832 % | **+1.16 %** | +1.11 % | 0.0014 |
| WIDE_BOX | 31.0 M | -2.876 % | -3.64 % | -3.70 % | 0.0007 |
| teacher | - | 0 | 0 | 0 | 0.0003 |

The baseline and the ensemble are inside the 3 % contract; WIDE_BOX is just outside, in proportion
to its worse value error, which is the same ordering the assembled compliance gives.

Writing `chat{c}(eps) = c(eps)(1 + e(eps))`, the relative error of the derivative is
`e + (de/deps)/(d ln c/d eps)` with `d ln c/d eps` = -2.278. Both terms are a few percent for all
three models - `e` is -0.8 to -2.9 % and the `de/deps` term contributes +2.7 %, +1.9 % and -0.8 %
respectively. **There is no catastrophic amplification: the tau derivative is as accurate as the
value.**

## 4. The measurement was wrong until cudnn TF32 was turned off

The test amplifies any relative error on the predicted compliance by `1/(2 h |dlnc/deps|)`, which
is 220 at `h` = 1e-3 and 2200 at `h` = 1e-4. `train_equi.main` sets
`torch.backends.cuda.matmul.allow_tf32 = False`, but a standalone script does not inherit it and
**`torch.backends.cudnn.allow_tf32` defaults to True** - and the volume encoder is `Conv3d`, so
that path is live. TF32's 10-bit mantissa put roughly 2e-5 relative noise on `trace(H^-1)`:

| | baseline derivative error | its own one-sided kink at `h`=1e-4 |
|---|---|---|
| cudnn TF32 **on** | +0.60 % (`h`=1e-4), +1.35 % (`h`=1e-3) - **step-dependent** | 0.164 |
| cudnn TF32 **off** | +1.92 %, +1.88 % - **step-independent** | 0.0036 |

With TF32 off the kink falls 45-fold, the two step sizes agree to 0.04 pp, and re-predicting
`eps` = 0 in the same process is **bit-identical** (`max_f |chat{c}_1/chat{c}_2 - 1|` = 0.0). The
TF32 numbers were wrong by about a percentage point and, worse, looked *better*. Any future run of
this test must set both flags and print the replication check.

## 5. Scope, and what is not established

* One seat, one direction. The perturbation is a **uniform** scaling of all eight corners, so it
  probes the `(1,1,...,1)` direction of the 8-dimensional tau space. The other seven directions
  need new teacher runs.
* A per-corner extension has a hazard the uniform direction does not:
  `cubic_group.canonical`'s docstring already names it - the lexicographic argmax over the 48
  permuted corner tuples is discontinuous in tau. A uniform positive scale cannot change it
  (monotonicity), but seat 0328's corners span only +-5 % about their mean, so a per-corner step
  of 1e-2 would cross an argmax switch. Under `--canonical` do not finite-difference a
  non-uniform direction at all; all arms here are `canonical=False`, and the view frame is pinned
  once from the base corners.
* The optimiser will use autodiff, not finite differences, so it gets the exact derivative of the
  network's own output; the noise discussed above is a property of our measurement, not of the
  gradient the optimiser sees. A forward-mode JVP through the closed-form context tangent
  (`d/ds` of `node_scalar[:,0]`, `(2/15) node_scalar[:,0]`, `node_scalar[:,3]`,
  `node_vector[:,0]`, `vol_scalar[0]`, `(2/15) vol_scalar[0]`, `vol_vector[0]`, `corners`; all
  else zero) would make the prediction side step-free, and is the natural next refinement.
