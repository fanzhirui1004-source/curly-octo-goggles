# Gate 2: a cut cell's response does have a tau derivative, inside a bracket of 0.025 %

2026-09-20. Teacher side only, no network anywhere. First cut-cell result; the tau smoothness gate
had only ever been run on seat 0328, which is a **box** cell (`kind1 = 0`, cut-plane offset 2).

## Why it had to run first

The contract has a sensitivity leg, `tau` is the design variable, and a cut cell's cut surface is
free by construction - which is exactly the configuration where
`TAU_DERIVATIVE_PASSES_20260919.md` found the *truth itself* one-sided inconsistent: for the free
box cell the log-slopes of `c(f)` were **-6.09 and -21.36** at h = 1e-4, a factor of 3.5, because
one CutFEM cell being born moves the softest direction by 60 %. The clamped-platen observable was
smooth (Richardson 0.999997). If the truth has no derivative for a cut cell, no learned operator can
supply one and every route needs re-scoping, so this runs before any condensed label is built.

## How

`superelement/objective/cut_tau_geometry.py` calls the pinned teacher
(`source_independent_6624dc8_20260910`, commit `6624dc8670...`) at `tau * (1 + eps)` with the eight
corners scaled by an exact rational, exactly as the 0328 protocol did, and changes nothing inside
it. Two guardrails in the 0328 wrapper are box-only and were replaced, both recorded rather than
dropped:

* **admission.** The 0328 rule required the COMPLETE trace to be bit-identical. Here the rule is
  that the kind-0 (box-face) rows are identical and the box rigid trace is unchanged, with what the
  kind-1 rows did recorded beside it - because the cut surface is a fixed plane intersected with the
  material, so it moves when `tau` moves, while the condensed operator does not depend on the cut
  block's basis. **In the event the relaxation was never needed: all four admitted paths satisfy the
  original strict rule too.**
* **the export check.** The 0328 wrapper ended with `if np.any(expected['kind']): raise`, refusing
  cut seats outright, and indexed `registry['boundary']`. For a cut cell `boundary` is the full
  trace and the box part is `registry['box_boundary_original']`; indexed correctly, the export's box
  rows match the admitted trace **exactly and in order**. That is a correction, not a relaxation.

Two environment facts cost time and are worth recording: the teacher's `RUNTIME_ROOT` default points
at `CUTFEM_TRUE_IMPLICIT_GHOST_20260905_V1/runtime`, which no longer exists - the pinned native
runtimes (`r13_pardiso_v1`, `r17_algoim_v2`, `r22_gmpy2_v1`) live at
`CUTFEM_INGEST_R38/environment/runtime` and are selected with `CUTFEM_RUNTIME_ROOT`, as the
2026-09-13 launch records show. And the teacher's `preallocate_guard` compares against the cgroup's
`memory_current`, which was 92.6 GiB of 96.6 GiB while anonymous memory was only 2.2 GiB - the rest
was reclaimable page cache from the training arm's label reads. Releasing it with
`posix_fadvise(DONTNEED)` on the label trees brought `memory_current` to 41.1 GiB and the guard
passed. Nothing shared was modified and the guard was not bypassed.

Three untracked scripts (`gp_build.py`, `gp_launch.py`, `gp_tree_profile.py`, left by the
2026-09-16 GP-tree profiling) sit in the pinned teacher's working directory and break a literal
`git status --porcelain` cleanliness test. No tracked file is modified, so the teacher's code is
still the pinned commit. Moving them out means writing inside the frozen tree, which is not ours to
do, so the check here is HEAD at the pinned commit plus **no modified or staged tracked file**, with
every untracked path and its sha256 written into `SOURCE_BINDING.json`, and a stray file inside a
`stage_cutfem*` package still refused.

## The admitted bracket, and what bounds it

Seat 100000: q = 711, 95 box coordinates, 142 cut functionals, 19 active CutFEM cells.

| label | eps | active cells | q | box | cut | status |
| --- | --- | --- | --- | --- | --- | --- |
| BASE_REPLAY | 0 | 19 | 711 | 95 | 142 | replays the frozen cache exactly |
| PATH_1 | -1/1000 | **18** | 687 | 95 | 134 | rejected, box trace changed (rigid diff 4.09e-2) |
| PATH_1 | -1/2000 | **18** | 687 | 95 | 134 | rejected, same |
| PATH_1 | -1/4000 | 19 | 711 | 95 | 142 | admitted |
| PATH_2 | -1/10000 | 19 | 711 | 95 | 142 | admitted |
| PATH_3 | +1/10000 | 19 | 711 | 95 | 142 | admitted |
| PATH_4 | +1/1000 | 19 | 711 | 95 | 142 | admitted |

A CutFEM cell **dies** between eps = -2.5e-4 and -5e-4. So the bracket on which the operator is
comparable at all is about `|eps| <= 2.5e-4`, a **0.025 % step in tau**. Base conditioning:
`kappa(S) = 1.78e6`, `kappa(T) = 3.87e5` - the same numbers the condensation sweep produced
independently for this seat (1.783e6 -> 3.871e5), which cross-checks both scripts.

## The result: differentiable, and not marginally

One-sided log-slopes at h = 1e-4, both sides available. `T = S_BB - S_BC S_CC^-1 S_CB` is the
operator a lattice reads, because the assembly glues on kind-0 and leaves kind-1 free.

| observable | left | right | ratio | kink |
| --- | --- | --- | --- | --- |
| `trace(S^+)`, full trace | -15.9722 | -15.8847 | 1.0055 | 0.28 % |
| `trace(T^+)`, condensed | -17.3875 | -17.1009 | 1.0168 | 0.83 % |
| softest non-rigid eigenvalue of S | +13.8948 | +13.7606 | 1.0097 | 0.49 % |
| softest non-rigid eigenvalue of T | +12.4518 | +11.7068 | 1.0636 | 3.1 % |
| 8 box-supported loads through T | -14.13 … -19.95 | -15.49 … -19.24 | 1.006-1.148 | 0.3-6.9 % |
| 8 full-support loads through S | -14.54 … -18.68 | -14.34 … -18.73 | 1.0002-1.014 | 0.01-0.67 % |

**The worst one-sided ratio anywhere is 1.148, against 3.5 for the 0328 free box cell.** The wider
brackets agree: the left slope at h = 2.5e-4 and the right slope at h = 1e-3 reproduce the h = 1e-4
values to 0.1-1 % on every aggregate observable (`trace(S^+)` -15.955 / -15.883,
`trace(T^+)` -17.315 / -17.175, softest +13.851 / +13.839). So the derivative is stable across the
whole admitted bracket, not just at one step size.

## What this does NOT establish

* **One seat, and the cheapest one.** 19 active cells and only one intact box face
  (`face_0_0` alone), so it cannot be glued on opposite faces and has no assembled observable at
  all. A mid-size cut seat with a glueable face pair is running.
* **Nothing about continuity ACROSS a cell birth or death, which is the thing an optimiser
  crosses.** The bracket is 0.025 % in `tau`; a topology optimiser steps far further, so the
  operator's dimension changes repeatedly along a design path. Between those events the response is
  differentiable, as measured. Across one, the two sides are not on a common basis: when the cell
  died the cut coordinates went 142 -> 134 **and the box trace changed too** (box count stayed 95
  but the rigid trace differs by 4.09e-2, because the active node set shrank). So condensation alone
  does not make the comparison possible either - the box coordinate set moves as well.
  The measurement that would settle it is a **common geometric frame**: evaluate both sides'
  condensed operators against the same physical face traction, interpolated onto each side's own
  node set, so the comparison survives a change of coordinate set. That is the open question this
  document leaves, and it is the one the sensitivity leg of the contract actually rides on.

## Provenance

`docs/data/cut_tau_20260920/SLOPES_100000.json` and `GEO_100000_SELECTED.json`. Remote:
`/root/autodl-tmp/CLAUDE_CUT_TAU_20260920/` holds `GEO_100000/`, `NUM_100000_PATH_*/`,
`SLOPES_100000.json`, `ENV.sh` and `EXECUTION.json`. Scripts:
`superelement/objective/cut_tau_geometry.py`, `superelement/objective/cut_tau_slopes.py`.
Numeric cost on this seat: 6.0-6.7 s per perturbed operator, 8.7 s per geometry screen.
