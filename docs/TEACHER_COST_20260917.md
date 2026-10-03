# The teacher's per-cell cost: the receipts, and the figure that was being misused

Recorded 2026-09-17 after the user pointed out - repeatedly - that "31.85 s" is not
the teacher's per-cell cost. They are right. This document is the correction and the
reference the rest of the corpus should cite.

## 1. What 31.85 s actually is

`docs/BRIEFING_20260915.md:598` and `docs/GP_TEACHER_TREE_COST_20260916.md:100`:

| item | time | scope |
|---|---|---|
| PARDISO numeric factorization | **31.85 s on 64 cores** | seat 0328 only; 231,192 dofs; q = 12,798 |
| full export, inclusive | 38.91 s | same seat |

So 31.85 s is **one step** (the numeric factorization) on **the smallest production
seat**. It is not the cost of producing one label.

## 2. The full-pipeline receipt

`docs/HANDOFF_20260913.md:11`, seat 0257 - 316,488 dofs, interior 297,450, interface
q = 19,038:

| stage | time |
|---|---|
| assembly | 0.6 s |
| geometry and factor preparation | ~29 s |
| factorization of the 316k system | 66 s |
| **total to export the Schur complement** | **~95 s** |

## 3. The honest range

Two receipts, two seats:

| seat | dofs | q | factorization | full pipeline |
|---|---|---|---|---|
| 0328 | 231,192 | 12,798 | 31.85 s | 38.91 s (export-inclusive line) |
| 0257 | 316,488 | 19,038 | 66 s | ~95 s |

**Per-cell teacher cost is 39-95 s across these two measured seats**, and the two
factorization points scale as roughly `dofs^2.3` (equivalently `q^1.8`). The training
set spans q = 12,828 to 25,932, so a two-point extrapolation puts the largest seats'
factorization near 120-165 s. That extrapolation is from two points and should be
treated as an order-of-magnitude statement, not a measurement.

The two receipts also do not decompose the pipeline the same way - 0328's
"export-inclusive 38.91 s" leaves ~7 s beyond factorization while 0257's receipt
carries a separate ~29 s of geometry and factor preparation. Until a single seat is
timed stage-by-stage twice, **quote the range, not a point.**

## 4. What this changes

With the cost target now set at **100 ms per cell** (user decision, 2026-09-17), the
required speedup is:

| baseline | speedup needed for 100 ms |
|---|---|
| 31.85 s (the figure previously quoted) | 320x |
| 39 s (smallest seat, full pipeline) | 390x |
| 95 s (measured larger seat, full pipeline) | **950x** |
| ~165 s (largest seats, extrapolated) | ~1650x |

And against Codex's measured predictor cost of 5.67e13 FLOP per cell - ~135 ms at
RTX 5090 TF32 tensor peak, 540 ms at fp32 peak, 2.07 s as actually run with TF32 off:

- the complete-factor route is **within ~1.35x of the 100 ms target at hardware
  peak**, and roughly **290-700x faster than the teacher**;
- the hierarchical low-rank route (1.22e7 numbers, the one measured to pass +-3%)
  sits at ~20 ms, i.e. **5x inside the target with headroom**.

**Cost is therefore no longer the binding constraint on either factor-shaped route.**
The binding constraint is accuracy (`docs/WHERE_THE_ERROR_IS_20260917.md`: median
per-entry relative error 3.2-5.7 against a required 2.9e-4).

## 5. Where the wrong figure was used, and what it now says

Corrected in this commit: `CODEX_M4_ASSESSMENT_20260917.md` (x2),
`REVIEW_20260916.md`, `ROADMAP_NEURAL_SCHUR_20260916.md`,
`FIRST_PRINCIPLES_PATH_20260917.md` (x2). Left unchanged where 31.85 s is correctly
labelled as the PARDISO factorization line for seat 0328
(`BRIEFING_20260915.md`, `GP_TEACHER_TREE_COST_20260916.md`, `REASSESS_20260916.md`,
`REVIEW_20260916.md:115`).
