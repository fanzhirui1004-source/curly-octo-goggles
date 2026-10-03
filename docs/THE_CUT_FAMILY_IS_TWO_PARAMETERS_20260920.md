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

## 2a. Retraction: this does NOT license a per-part specialist, and tau is why

An earlier version of this document read section 2 as "so the requirement is easier than we thought,
and a per-part specialist is the route". That reading is wrong, and three measurements kill it.

**The n = 1 result is memorisation, not generalisation.** Each seat carries exactly one `tau` field,
so n = 1 is one `(b, d, tau)` point. The 0.30 % was never a test of interpolation in `tau`, and
`tau` is the design variable: an optimiser moves it, and moves it toward the lower bound, which is
thin walls and near-mechanisms - the worst-conditioned end of the family. A specialist trained on 15
seats at one slope has seen 15 points of an eight-dimensional `tau` space.

**The `tau` sampling density is already the same for cut and box, and box generalised at it.** Full
eight-vectors from the frozen trace caches, over the seats that carry a trainable label:

| | box | cut, design domain |
| --- | --- | --- |
| trainable seats | 57 | 109 |
| `tau_mean` span | 0.185-0.490 | 0.179-0.447 |
| within-cell grading \|max-min\|, median | 0.0662 | 0.0635 |
| nearest neighbour in the 8-vector, relative, median | 0.081 | **0.067** |

The cut family is sampled *more* densely in `tau`, with twice as many seats, and box cells
generalised at that density to 0.3-3.3 % on twelve seats. So `tau` sparsity is not what is wrong
with cut cells, and concentrating the training set at one slope buys nothing on the axis that
actually binds.

**The specialist hypothesis is not even testable with the data we have.** Of the in-domain cut
seats, 27 pairs sit within `|d(b,d)| <= 0.03` of each other - but their relative `tau` distance has
median **0.511** and maximum **1.168**. The dataset was sampled independently in `(b, d, tau)`, so
there is no cluster anywhere with a nearly-fixed cut and densely-sampled `tau`. Testing the
specialist idea would need new teacher runs, and using it would need per-part teacher labels, which
is the cost the network exists to avoid.

What survives from section 2 is the geometry itself: a facet's boundary cells share `b`. What does
not survive is any suggestion that this makes the problem go away.

## 3. Why the failure is still the cut, not the family's size

Every cut arm so far has been trained to generalise over `(b, d, tau)` jointly. The application
asks it to generalise over `(d, tau)` at **fixed** `b`. Those are different requirements, and the
evidence that they are very different in difficulty is already on record: at n = 1 a cut cell is fit
to a 0.30 % median face-load response error with 100 % of loads inside 3 % (NCUT_02, seat 100032),
and at n >= 28 the same architecture and budget lands at 9-42 %.

So the central claim should be stated with its n attached - *the cut operator is not learned at
n >= 28 at this budget, and is learned at n = 1* - because then the next question is the right one:
how many distinct cut geometries does one part have? Section 2 answers it: 1 for a lattice-aligned
facet, 2-4 for a simple rational slope, about N for a generic one.

Box cells: 8 parameters, 57 trainable seats, contract met. Cut cells: 10 parameters, 109 trainable
seats, 9-42 %. More samples and only two more parameters, and it still fails - so the difficulty is
not the dimension of the family. What is left is the cut itself, and specifically the part of the
target the cut adds: a block carrying 60-93 % of the Frobenius mass, expressed in a basis that is
not rotation covariant, with its own condition number near 1e8, which the assembled answer never
reads. That is what section 4 of `CONDENSE_THE_FREE_CUT_SURFACE_20260920.md` proposes to remove.

## 4. The slope experiment, parked

Defined in `SLOPE_EXPERIMENT.json`, splits in `split_slope_S.txt` / `split_slope_W.txt`.
**Parked, and not a route-selection test** - see 2a. At most it measures how much of the difficulty
the `(b, d)` spread contributes, and the box/cut comparison above already answers that more cheaply.

* **Arm S** - 15 training seats, all with `b < 0.10`: one facet slope, many offsets.
* **Arm W** - the same 15-seat budget spread over `b` 0.003-0.987 by quantiles of the pool, which
  still leaves 2 seats at `b < 0.10`, so W *covers* the held-out slope and only its concentration
  differs. (An earlier draft drew W from `b >= 0.11`, which made it an unseen-slope test rather
  than a concentration test.)
* **Held out, identical for both**: seats 100028, 100111, 100091, 100123 - `b` 0.044-0.099,
  `d` 0.092-0.604, `tau_mean` 0.303-0.427.
* Matched: n = 15, steps, capacity, `g = 0`, no sharding, same evaluation set and metric.

## 4a. What any cut arm has to pass, decided before the arm runs

`tau` is the design variable, so the acceptance protocol is written around it, and the second gate
has **never been run on a cut cell**:

1. held-out seats, unseen in both `tau` and `(b, d)` - the 64-load assembly-free response gate and
   the box-node displacement, not the worst of four compliances;
2. **the `tau` derivative of a cut cell against the teacher.** Teacher side first: a free cut surface
   is exactly the configuration where `TAU_DERIVATIVE_PASSES_20260919.md` found the truth itself
   one-sided-inconsistent (log-slopes -6.09 against -21.36 at h = 1e-4). If the truth has no
   derivative there, no learned operator can supply one, and that kills every route - so this runs
   *before* condensed labels are built, not after;
3. the assembled cut-to-full stack.

## 4b. H2, on the pre-registered held-out set: there is no cross-geometry generalisation at all

Two arms, both 60 000 steps, both 9 692 283 parameters, both `g = 0` on every step, neither sharded,
scored by `sweep_checkpoints` on the 12 seats of `HELDOUT_H2.json` that **neither** arm presented,
plus seat 100032 which **both** presented. Metric: the assembly-free face-load response gate,
16 loads per seat, on each arm's final checkpoint.

| | n = 8 (7500 visits/geometry) | n = 28 (2143 visits/geometry) |
| --- | --- | --- |
| held-out face median, over 12 seats | 15.6 % … 554 %, **median 96.2 %** | 26.7 % … 1260 %, **median 87.9 %** |
| held-out `frac_within_3pct` | 0-6 % on every seat | 0-3 % on every seat |
| held-out factor error, median | 1.009 | 0.892 |
| presented seat 100032 | **3.14 %** median, 49 % of loads inside 3 % | **3.40 %** median, 46 % inside 3 % |

Read it twice, because the two readings point different ways.

**n makes almost no difference.** 96.2 % against 87.9 % median, with 3.5x the per-geometry exposure
on the n = 8 side. Whatever the cut family's problem is, it does not live between n = 8 and n = 28.
H2 is answered, negatively.

**The presented-to-held-out gap is a factor of 28** - 3.1-3.4 % on a geometry the arm has seen,
87-96 % on one it has not, with 0-6 % of loads inside 3 % there. Both arms are *memorising* the cut
geometries they are shown and carrying essentially nothing across to a new cut. That is the
concern stated plainly in 2a, now measured on a set that was declared before the arms ran and that
neither arm touched.

It also puts the earlier numbers in their place. The 9-42 % figures quoted for "internal unseen" cut
seats came from seats held out *by omission* from an arm that had presented 28-85 cut geometries, and
often sat near one it had. On a genuinely unseen cut the error is an order of magnitude worse.

This is what makes the condensed target the experiment worth running rather than one option among
several: the only lever anywhere in this project that has been measured to convert a fit into
generalisation is the 48-fold cube symmetry, and the cut family is the one family that cannot use it.

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
