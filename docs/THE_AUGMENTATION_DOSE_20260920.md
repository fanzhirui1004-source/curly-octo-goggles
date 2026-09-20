# Augmenting a cut cell helps, and the evidence that said otherwise was measured in-sample

2026-09-20. Pre-registered before the run in `docs/data/augcost_20260920/PREREGISTRATION.json`.
Four arms, matched in pool, capacity, schedule, steps, evaluation set and metric, run concurrently
on one box so that every arm saw the same machine.

## 1. What was open

No cut cell had ever been rotated in a multi-seat arm. Every arm that recorded `augmented_seats`
recorded zero of them for its cut seats - `cutonly_noaug` 0 of 32, `wide_cut` 0 of 73,
`ncut_02/04/08/16/59` all 0, and `cutmix_fullaug` 28 of 64 where those 28 are its *box* seats -
because they all ran `--augment --augment-full-only`. The flag that did that was built on one n = 1
pair, `solo_100079` against `noaug_100079`.

So the 48-fold cube symmetry, the only lever in this project measured to convert a fit into
generalisation, had never actually been applied to the family that fails.

## 2. The result

Held-out: the twelve seats pre-registered in `HELDOUT_H2.json`, presented by neither arm. Metric:
the assembly-free face-load response gate on the final checkpoint, per-seat median absolute error,
then the median over the twelve.

| arm | group elements on cut cells | held-out median | presented seat 100032 |
| --- | --- | --- | --- |
| DOSE_00 | 1 (identity) | 0.6400 | **0.0153** |
| DOSE_00B | 1, second seed | 0.7607 | 0.0243 |
| DOSE_24 | 24 proper rotations | **0.4092** | 0.2064 |
| DOSE_48 | all 48 | 0.5109 | 0.1894 |

The declared rule: `R = |DOSE_00 - DOSE_00B| = 0.1207`, and a dose counts only if it beats DOSE_00 by
more than R.

* **DOSE_24 is better by 0.2308 = 1.91 R.** Called.
* **DOSE_48 is better by 0.1291 = 1.07 R.** Called, but by a margin barely outside the resolution,
  so it is the weakest of the two statements and should not be quoted without its margin.

A sign test, **not pre-registered and reported only as support**: DOSE_24 beats DOSE_00 on 10 of the
12 held-out seats (one-sided p = 0.019); DOSE_48 on 8 of 12 (p = 0.19). So the DOSE_24 result has a
second, independent reason to be believed and the DOSE_48 one does not.

## 3. Why the old n = 1 evidence said the opposite, and why it was not wrong

`solo_100079` (augmented, 12.585 %) against `noaug_100079` (not augmented, 2.902 %) was measured on
**seat 100079 itself, the single seat each arm trained on**. It is an in-sample number. This
experiment reproduces exactly that effect and in the same direction:

| | presented seat 100032 | held-out median |
| --- | --- | --- |
| no augmentation | 0.0153 | 0.6400 |
| 24 elements | 0.2064 | 0.4092 |

Augmentation costs a factor of 13 in-sample and buys a factor of 1.6 out-of-sample. That is ordinary
regulariser behaviour, and the flag that switched augmentation off for every cut arm was built by
reading the first column.

The in-sample number is also the one that looks like success: **0.0153 is inside the 3 % contract**.
A cut arm without augmentation meets the contract on a cell it has seen and misses it by a factor of
21 on one it has not.

## 4. What this does NOT say

**It does not fix cut cells.** 0.4092 is 41 % where the contract is 3 %, and the fraction of loads
inside 3 % is 0.031 at best. Augmentation moved the failure from 64 % to 41 %; the gap to the
contract is still a factor of 14. Nothing here licenses "the cut problem is solved" or even "the cut
problem is now a tuning problem".

**The dose-response is not monotone.** 24 beats 48 on the median and on the sign test. Two readings
are available and this experiment cannot separate them: either the improper elements hurt, or at a
fixed 60000 steps a 48-fold wider input distribution gets too few visits per element. A third arm at
24 elements with a second seed, and an arm at 48 elements with twice the steps, would separate them.

**R is a crude resolution.** It is one absolute difference between two medians, not a standard error,
from a single pair of seeds. It is the number that was declared in advance and it is used as
declared, but a second seed per dose would make every statement here stronger.

## 5. What follows

The covariance result of the same day (`THE_CUT_OPERATOR_IS_COVARIANT_20260920.md`) said augmentation
generates *true* samples for cut cells. This says that using them helps. Together they retire the
`--augment-full-only` default for cut arms: the flag now refuses to run without `--augment`, and the
reason it was set is recorded above as an in-sample reading.

What remains is the same central fact as this morning: a cut cell that the network has not seen is
wrong by an order of magnitude more than the contract allows, and no measurement yet explains why.

## Provenance

`docs/data/augcost_20260920/` carries `PREREGISTRATION.json` (written before the arms ran),
`RESULT.json` (per-seat, per-arm, with the decision), `ARM_AUGMENTATION_STATE.json` (what every
historical arm actually augmented), `plan.py`, `run.sh` and `decode_equiv.py`.
Remote: `/root/autodl-tmp/CLAUDE_AUGCOST_20260920/`.
