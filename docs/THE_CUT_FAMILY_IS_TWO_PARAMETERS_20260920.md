# The cut family is two parameters, and one facet of a real part needs one to forty of them

2026-09-20. Teacher-side only, no training. This changes what the cut-cell requirement is, which
matters more than any of the methods under discussion.

## 1. Every cut plane in the dataset has normal (1, b, 0)

Read from `tau_corners` / `cut_plane` in each seat's frozen `TRACE_CACHE.npz`, all 184 cut seats:

* `cut_plane[0]` takes exactly one value, 1.0;
* `cut_plane[2]` takes exactly one value, 0.0 - **there is no z-tilt anywhere in the dataset**;
* `b = cut_plane[1]` spans 0.0028 to 0.9873;
* the offset `d = cut_plane[3]` spans 0.0342 to 1.6579.

So the cut family is **two** parameters on top of the eight `tau` corners, not the
high-dimensional thing the failure rate suggests. In the design domain (`tau_mean <= 0.45`) there
are 139 cut seats, and their nearest-neighbour distance in `(b, d)` has median 0.0329: we have been
sampling the whole two-dimensional plane family thinly rather than sampling `tau` densely at a few
cuts.

## 2. What one facet of a real part actually asks for

Put a flat facet `{y : n . y = t}` through a lattice of unit cells. The cell at integer offset `x`
sees, in its own coordinates, the plane `(n, t - n . x)`. **The normal is shared by every boundary
cell of that facet; only the offset differs.** Counting the distinct local offsets that actually
fall inside a cell:

| facet | distinct local cut geometries | N=4 | N=8 | N=16 | N=32 |
| --- | --- | --- | --- | --- | --- |
| lattice aligned, n = (1,0,0) | 1 | 1 | 1 | 1 | 1 |
| slope b = 1/2 | 2, independent of N | 2 | 2 | 2 | 2 |
| slope b = 1/4 | 4, independent of N | 4 | 4 | 4 | 4 |
| generic slope b = 0.234069 | ~N | 4 | 8 | 19 | 38 |

A rational slope quantises the offsets and the count stops growing; a generic slope gives about one
per cell along the facet. Two caveats. A facet with all three normal components non-zero would give
O(N^2) - but the dataset contains no such plane, so we have no data for that case either. And
`tau` still varies from cell to cell regardless, so the per-part family is
`{1 to ~N fixed cut planes} x {the 8-dimensional tau family}`.

## 3. Why that reframes the failure

Every cut arm so far has been trained to generalise over `(b, d, tau)` jointly. The application
asks it to generalise over `(d, tau)` at **fixed** `b`. Those are different requirements, and the
evidence that they are very different in difficulty is already on record: at n = 1 a cut cell is fit
to a 0.30 % median face-load response error with 100 % of loads inside 3 % (NCUT_02, seat 100032),
and at n >= 28 the same architecture and budget lands at 9-42 %.

So the central claim should be stated with its n attached - *the cut operator is not learned at
n >= 28 at this budget, and is learned at n = 1* - because then the next question is the right one:
how many distinct cut geometries does one part have? Section 2 answers it: 1 for a lattice-aligned
facet, 2-4 for a simple rational slope, about N for a generic one.

## 4. The experiment that tests the requirement rather than the method

Defined in `SLOPE_EXPERIMENT.json`, splits in `split_slope_S.txt` / `split_slope_W.txt`.

* **Arm S** - 15 training seats, all with `b < 0.10`: one facet slope, many offsets.
* **Arm W** - the same 15-seat budget spread over `b` 0.003-0.987 by quantiles of the pool, which
  still leaves 2 seats at `b < 0.10`, so W *covers* the held-out slope and only its concentration
  differs. (An earlier draft drew W from `b >= 0.11`, which made it an unseen-slope test rather
  than a concentration test.)
* **Held out, identical for both**: seats 100028, 100111, 100091, 100123 - `b` 0.044-0.099,
  `d` 0.092-0.604, `tau_mean` 0.303-0.427.
* Matched: n = 15, steps, capacity, `g = 0`, no sharding, same evaluation set and metric.

Reading: S >> W means the requirement the application has is much easier than the family we have
been training for, and the route is a per-part specialist at n <= 4 for aligned or simple-rational
facets and n ~ N for generic ones. S ~ W means concentration does not help and the difficulty is
intrinsic to the cut at any slope.

## 5. Inventory, with two corrections to earlier statements

Measured on disk per seat, not from the manifest's presence:

| | cut | box |
| --- | --- | --- |
| manifest entries | 184 | 89 |
| teacher factor `R_UPPER.npy` | 184 | 89 |
| teacher trace operator `S_UPPER.npy` | 184 | 89 |
| trainable label `MQ_UPPER.npy` | 140 | 57 |
| `INVERSE_RESULT.json` (carries `condition_of_A`) | **47** | **6** |

* The kappa(A\*) statistics quoted earlier - box median 3.27e4 over a 3.6x spread, cut median 7.75e5
  over a 100x spread - are over the **47 cut and 6 box seats that have `INVERSE_RESULT.json`**, not
  over 184 and 89. The values stand; the sample size was misstated.
* In the design domain there are 139 cut seats, 109 with a trainable label, and **all 139
  condensable**, because condensation needs only the teacher's `S_UPPER.npy`. The condensed route
  therefore unlocks 30 geometries that are not trainable today.

## 6. There is no reserved held-out cut set, and there cannot be a retrospective one

Of the 184 cut seats, **2** were never presented by any arm (207 and 100079), and both are outside
the design domain. Every in-domain cut seat has been shown to something. "Held out" in this project
has meant held out *by that arm*, which is sound for a within-arm reading and unsound for comparing
arms trained at different times.

Consequences, both adopted:

* the H2 comparison uses a set pre-registered in `HELDOUT_H2.json`: 12 in-domain seats presented by
  **neither** H2 arm, scored with the assembly-free 64-load response gate on both arms' final
  checkpoints;
* any future arm must reserve its evaluation seats *before* it runs. For the condensed-target arm
  that reservation has to be made when the condensed labels are built.

## Provenance

`docs/data/cut_diagnosis_20260920/` carries `h3_scc_sweep.py`, `slope_experiment.py`,
`heldout_h2.py`, `inventory.py`, the two split files, `SLOPE_EXPERIMENT.json` and `HELDOUT_H2.json`.
Remote outputs: `/root/autodl-tmp/CLAUDE_CONDENSE_20260920/H3_SCC.jsonl`,
`/root/autodl-tmp/CLAUDE_EQUI_20260918/{NCUT_08_60K,H2_GATE}`.
