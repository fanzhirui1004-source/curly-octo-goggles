# The storage floor, applied to the teacher we are actually fitting

Date 2026-09-16.  Script `superelement/factor_fit/floor_modes.py`; raw `docs/data/FLOOR_MODES.json`.

## Why

`docs/STORAGE_FLOOR_ACCEPTANCE_20260914.md` established, two days ago, that some directions of a
delivered factor are **not measurable**: their stored energy sits within a few times its own FP64
rounding, so a +-10% band cannot be evaluated there at all.  It defined the criterion
(`E/N`, signal to storage noise), fitted the deviation constant (`C = 3`), set the threshold
(`UNJUDGEABLE = 30`), shipped the tool (`superelement/storage_floor.py`), and said explicitly that
the whitened divergence weights every direction equally so such directions must be **masked or
down-weighted in the training loss**.

It was then never referenced again.  `grep` finds no mention of `storage_floor` or
`signal_to_storage_noise` in the roadmap, in the briefing's learning sections, or anywhere in
`superelement/neural_schur/`.  Every acceptance number produced since — including the whole of
2026-09-16 — was computed by a gate that does not know the criterion exists.

So: apply it to the operator the gate is actually being applied to.  The earlier measurement used
four screen directions on the **no-GP** packets; this uses **every stiffness eigenmode** of the
**GP** teacher's quotient operator.  For a unit eigenvector, `E = lam_i`, and `N` is one dense
`|R| |V|` away, so it is two matmuls.

## Measured

| seat | `d` | `lam_min` | `lam_max` | cond | modes with `E/N < 30` | min `E/N` | median `E/N` |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0328 | 12,792 | 4.8777e-07 | 3.5244e-02 | 7.23e4 | **0** | 1.52e27 | 1.32e31 |
| 0253 | 12,822 | 4.8818e-07 | 3.5265e-02 | 7.22e4 | **0** | 1.58e27 | 1.30e31 |

Modes where rounding alone could produce a deviation above 3%: **0 of 12,792**.  Above 10%: **0**.

The softest mode of seat 0253 (`lam = 4.88e-07`) stands `E/N = 7.73e29` above its own storage
floor, so `C/(E/N) = 3.9e-30` — twenty-eight orders of magnitude below the threshold.  The softest
end is as measurable as the stiffest.

## What changed between the two measurements

| | no-GP packets (2026-09-14) | GP teacher (today) |
|---|---:|---:|
| factor condition estimate | 4.11e16 (label 0448) | 269 (`sqrt` of `cond(A)`) |
| `cond(A)` | ~1.7e33 | 7.23e4 |
| diagonal ratio of `R` | 2.71e14 | — |
| softest direction `E/N` | 3.04 (seat 52), 9.73 (0448) | 1.5e27 |

Twelve orders of magnitude in the factor's conditioning.  That is what ghost penalty is for: it
stabilises exactly the near-singular modes that sliver cuts produce.  The same mechanism shows up
in the 59.76% sensitivity incident, where the perturbation "activated a background cell of material
volume fraction ~5.6e-15 and three new GP faces" and a controlled rewrite of that stabilisation
energy took the neighbouring-geometry max stiffness ratio from ~1.598 to ~1.016.

## Consequence

**On this teacher there is nothing to mask.**  The unmeasurable-direction problem was real, was
diagnosed correctly, and was engineered away by the stabilisation — before the current fitting work
started.  Relaxing the gate on the grounds that the soft directions are not physically meaningful
is not supported by the project's own criterion, applied to the current operator.

This does **not** rescue the gate.  `eps_op <= 0.03` on every mode is still **asserted rather than
derived**: the deliverable is an assembled lattice under realistic loads, and nobody has measured
how a module error propagates there.  `BRIEFING` section 3 records that assembly of **exact**
module Schur complements reproduces a direct FE solve to 1e-12, which establishes that condensation
composes and says nothing about error propagation.  `superelement/factor_fit/assemble.py` measures
that; it is what should set the tolerance, not this.

It also settles one thing about the failures seen today.  The banded and `T^T D T` arms produce
`mu_max` between 59 and 144 — the approximation is up to 144x **too stiff** somewhere.  With no
mode anywhere near its rounding floor, those are real approximation errors, not the teacher's own
noise being chased.
