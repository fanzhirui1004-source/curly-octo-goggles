# Augmentation costs a factor of four on a cut cell; the cause is not yet established

**Correction, same day.** This note first claimed the measurement proved the augmentation identity
false. It does not, and the title has been changed. What is measured is the **cost**. The cause is
open, and the leading candidate is now capacity rather than an invalid label:

* the input side of the rotation is verified faithful -- `context.py`'s self-test has
  `cut_cell_rotation_recompute_worst` = 3.31e-14, i.e. rotating the compiled context equals
  recompiling from the rotated trace -- and the model is index-free to 3e-7;
* the trace operator is `L (x) I_3` (`rigid_trace` applies one scalar CSR to each of the three
  components), so a functional is a scalar functional tensored with the identity and the block form
  `G M_q G^T` is the right one;
* therefore, **on a single seat**, augmentation is a self-consistent equivariance regulariser:
  `f(g . x) = G f(x) G^T` together with `f(x) = M_q(x)` is satisfiable by any equivariant `f`. It
  cannot be contradictory for one geometry.

So the fourfold cost is most likely the price of forcing approximate equivariance through an
encoder that is not equivariant by construction: the volume U-Net is built from `Conv3d(3,3,3)`,
and a 3x3x3 convolution is not rotation equivariant (only the resampling is -- `avg_pool3d`
commutes to 1.19e-07 and nearest upsampling exactly). Equivariance has to be memorised over 48
orientations, and a cut cell needs the feature volume to carry the cut surface, which makes those
48 copies far more expensive than a full cell's smooth field.

The teacher's pivot-based choice of cut-surface functionals remains a real and separate concern,
but it bears on whether the model generalises to genuinely rotated geometries at test time, not on
whether single-seat augmentation is self-consistent. Whether it fails is being tested directly.

2026-09-19. Every arm in this project has trained with `--augment`: a random cube symmetry per
step, the label blocks rotated to match, resting on

    M_q(g . cell) = G M_q G^T,    G = blockdiag(Q_g).

For a **full** cell that identity is exact and was verified: the trace is the box-face background
nodes, they permute exactly under the group, `tau_phi_field_consistency` is 4.27e-15, and the
phase-2 arms found augmentation nearly free (non-equivariance 12.3 % -> 1.5 % while `g` moved
0.8 %).

For a **cut** cell, forcing it is expensive. One seat, 20 000 steps, plain loss, everything
identical but `--augment`:

| seat | cut fraction | augment | final loss | `factor_rel` | `g` | rotation probe | median face error | inside 3 % | point-load |
|---|---|---|---|---|---|---|---|---|---|
| 100051 | 0 (full) | on | 0.00067 | 0.0130 | 2.113 | 0.0125 | 0.393 % | 1.00 | 0.074 |
| 100032 | 0.398 | on | 0.01236 | 0.0826 | 5.17e5 | 0.0559 | 1.191 % | 0.85 | 0.141 |
| 100032 | 0.398 | **off** | **0.00359** | **0.0323** | **275.4** | 2.291 | **0.283 %** | **1.00** | 0.091 |
| 100079 | 0.731 | on | 0.10002 | 0.1802 | 2.385e9 | 0.0786 | 12.726 % | 0.03 | 0.455 |
| 100079 | 0.731 | **off** | **0.01761** | **0.1257** | **1.247e5** | 14.03 | **3.130 %** | 0.34 | 0.295 |

Switching augmentation off improves the median smooth-face compliance error **fourfold on both
cut seats**, cuts the final loss three to six fold, and drops `g` by three to four orders of
magnitude. The rotation probe moves the other way (0.056 -> 2.29 and 0.079 -> 14.03), which is the
proof that the augmented model really was learning the constraint — it was just learning a
constraint that is not true, and paying for it with accuracy everywhere else.

## The open question, and how it is being settled

A full cell's trace is one background node per coordinate, so the group maps coordinates to
coordinates. A cut cell's trace has, in addition, residual functionals on the macro cut surface,
selected by the compiler's coordinate convention `max_pivot_geometric_residuals_v2` — **by
pivoting**. Pivot-based selection need not be rotation covariant, which was flagged as "the
concrete suspect" in the 1.6a witness docstring in phase 1 and never tested.

The decisive form of the test is not whether the compiler picks the *same functionals*, because
those are only a basis for the cut surface's residual subspace and any basis gives a congruent
operator. It is whether it picks the *same subspace*. `witness_c.py` runs the teacher's
`compile_trace` on a cut geometry and on its rotation and measures the principal-angle residual
between the two row spaces, mapped into a common node basis:

* subspaces agree, bases differ -> the label is right up to a congruence, and re-expressing the
  cut-surface block in a covariant basis on our side fixes it without touching the frozen teacher;
* subspaces differ -> the teacher selects a genuinely different residual space and no basis change
  can repair it.

## What it explains

* cut cells fitting poorly even single-seat (they were being trained against contradictory targets);
* cut cells dragging the full cells down when mixed, since one network has to serve a true
  constraint and a false one at once (seat 100051: 0.393 % alone, 0.691 % in the box-only mixture,
  6.193 % in the half-cut mixture);
* the elevated rotation probe on cut seats, 0.10..0.26 against 0.032..0.047 on box-only ones;
* and part of the g explosion on cut cells, though not all of it -- `g` is still 275 with
  augmentation off on a seat whose physics is 0.283 %, so the near-mechanism argument in
  CUT_CELLS_ARE_LEARNABLE_20260919.md stands on its own.

## The fix, and what it does not fix

`--augment-full-only` rotates box-only cells and presents cut cells in the identity frame. Two
arms are queued with it: the 64-seat cut+full mixture, and the 32-seat cut-only set.

It does not recover the augmentation benefit for cut cells — they simply lose it, and a cut cell
seen in one frame will not generalise across orientations. The real repair is either a
rotation-covariant choice of cut-surface functionals in the teacher, or canonicalising each cut
cell's frame so that the one frame it is shown is a function of the geometry rather than of the
compiler's pivot order. Both are follow-on work; neither is needed to find out how well cut cells
can be fit.
