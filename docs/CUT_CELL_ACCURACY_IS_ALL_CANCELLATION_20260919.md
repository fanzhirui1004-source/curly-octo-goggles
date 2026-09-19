# A cut cell's accuracy is all cancellation: fixing its worst directions makes it worse

2026-09-20, overnight. Seat 100032 (40 % cut) and the three-module stack
100051 - 100032 - 100051, plus the capacity and seat-count arms. Data in
`docs/data/cut_cancellation_20260919/`.

## 1. The experiment

The best cut-cell fit this project reaches is `NOAUG_100032`: one seat, no augmentation, 0.278 %
smooth-face-load median, `mu_max` = 275. In the realistic stack it gives 3.91 % compliance and
8.62 % sensitivity, outside the 3 % contract, and the dominant term is the energy split.

The design question was whether a low-rank correction could close that, since a cut cell's soft
direction is a near-mechanism and in `M_q = A^{-1/2}` that is one very large mode. To answer it
without training anything, take the pencil `(A_hat, A*)` of the existing prediction, clip the
eigenvalues that are worse than a factor `c`, rebuild `A_hat` from the clipped spectrum and re-run
the same stack. The number of clipped modes is the rank of the correction that would be needed;
the stack error after clipping says whether such a correction is sufficient.

    A_hat = R^T V diag(mu) V^T R,   A* = R^T R,   V orthonormal
    clipped: mu -> min(max(mu, 1/c), c)

## 2. The result: every partial correction is worse than no correction

| clip `c` | modes fixed | of `d` = 10 806 | stack compliance | stack sensitivity |
|---|---|---|---|---|
| none | 0 | 0 % | **3.91 %** | **8.62 %** |
| 4 | 311 | 2.9 % | 6.64 % | 12.55 % |
| 2 | 1 816 | 16.8 % | **8.05 %** | 13.88 % |
| 1.3 | 5 796 | 53.6 % | 4.95 % | 7.62 % |
| 1 (exact) | 10 806 | 100 % | 0 % | 0 % |

The curve is **not monotone**. Fixing the 311 worst directions doubles the error; fixing 1 816 of
them doubles it again; only after fixing more than half of all 10 806 directions does it start to
come back, and even there it has not caught up with doing nothing.

The mechanism is the harmonic law `chat/c = sum_i w_i / mu_i`. The prediction's spectrum runs from
`mu` = 0.027 to 275, with 821 modes above 2, 995 below 0.5, and **74 % of the total `log^2 mu`
energy in the modes outside a factor of two**. The over-stiff modes contribute almost nothing to
the predicted compliance and the over-soft ones contribute too much; the 3.91 % that comes out is
what is left after those two errors cancel. A symmetric clip removes the compensating over-soft
modes along with the over-stiff ones, so the cancellation goes before the error does.

**This rules out a family of fixes, not just one.** A rank-k mechanism head, a tail-weighted loss,
spectral clipping at inference, or any scheme that targets the worst directions will move the
delivered number the wrong way unless it fixes essentially the whole spectrum. The box cells are
the opposite case: their `mu` stays inside [0.28, 4.1], only 0.04-0.76 % of the load energy sits on
directions worse than a factor of two, and the average is a genuine law-of-large-numbers average
(`docs/TWO_ERROR_REGIMES_20260919.md`).

## 3. Capacity and seat count do not close it either

Three arms on the cut family, all with `--augment-full-only`, all in the tau <= 0.45 design
domain, evaluated on the same 14 seats (3 presented, 11 never presented):

| arm | presented | parameters | steps | final loss | trained median | held-out median |
|---|---|---|---|---|---|---|
| CUTONLY_NOAUG | 28 | 9.7 M | 60 k | 0.0146 | 2.31 % | (different eval set) |
| NCUT_59 | 59 | 9.7 M | 20 k | 0.1017 | 7.70 % | 19.54 % |
| WIDE_CUT | 59 | 31 M | 60 k | 0.0138 | 3.66 % | **20.45 %** |

3.2x the parameters and 7x the loss improvement move the held-out median from 19.54 % to 20.45 %,
i.e. not at all. Two-cell acceptance on the held-out seats is 8.8-28.6 % compliance. The
catastrophic individual cases do improve a lot - seat 100000 83 % -> 37 %, seat 100021 465 % ->
8.65 %, seat 115 20.6 % -> 8.43 % - so capacity is buying something, but the generalisation gap
stays at about 6x and nothing approaches the contract.

The earlier n-sweep (n = 1, 2, 4, 8, 16 at a fixed 20 k steps) cannot be used to separate data from
capacity, because holding the total step count fixed while raising n also divides the steps per
seat: n = 8 at 2 500 steps per seat gives 15.95 % while n = 28 at 2 143 steps per seat gives
2.31-5.07 %. That comparison was reported and withdrawn the same evening.

## 4. What this means for the deliverable

The cut-cell family is an open research problem: its operator has to be right almost everywhere in
its spectrum, which is a much stronger requirement than the box family's, and neither capacity nor
seat count at the scales tried gets close.

It is not, however, a blocker, because of a property of the application that the experiments above
ignore. **In a topology-optimisation run the cut planes are fixed by the part's outer shape and
never change; only tau moves.** So a given part's boundary cells are a handful of fixed cut
geometries, not a 607-member family. Three routes follow, in increasing ambition:

* an integer-cell design domain has no cut cells at all, and the box family already passes all
  three claims - learnable with no generalisation gap, composable to 1.32 % at eight modules, and
  differentiable in tau to 1.11 % (ensemble) on a holdout seat;
* a per-part specialist trained on that part's few boundary geometries sits at small `n`, which is
  the regime where single-seat quality (0.278 %) is reachable;
* recomputing a boundary cell exactly per iteration is not viable - it needs the teacher's full
  CutFEM solve, 39-95 s per cell per design iteration.
