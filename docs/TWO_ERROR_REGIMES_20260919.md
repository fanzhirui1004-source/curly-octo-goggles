# Two error regimes, and the cut cell's ceiling

2026-09-19. Box-only and cut cells, from the stored pencil spectra and load-energy weights
(`soft_diag`), plus one new assembly experiment. Data in `docs/data/two_regimes_20260919/`.

## 1. Almost every direction of the operator is wrong, in every arm

From the stored `SPECTRUM.json` of every evaluation: the number of pencil directions with
`|mu - 1| > 0.03` is essentially the whole spectrum, in every arm, box and cut alike. For the
plain-loss box baseline `MULTI_AUGMENT` at `d` between 11 500 and 16 900, between 11 535 and
16 893 directions are off by more than 3 % and between 9 514 and 14 432 by more than 10 %, with
`mu` spanning [0.28, 4.06]. The assembled compliance is nevertheless 0.17-2.09 %.

So the accuracy is not "the operator is 2 % right". It is an average.

## 2. But the load energy tells two completely different stories

`E>3%` is the fraction of the load's energy sitting on directions off by more than 3 %,
`E(mu>2)` the fraction on directions the prediction makes more than twice too stiff (whose
compliance contribution `w/mu` is then simply lost), `E(mu<0.5)` more than twice too soft. Worst
load per seat.

| arm | seat | kind | E>3% | E>10% | **E(mu>2)** | E(mu<0.5) | bias | halfvar | net |
|---|---|---|---|---|---|---|---|---|---|
| MULTI_AUGMENT | 100051 | box | 82.4 % | 54.8 % | **0.76 %** | 0.44 % | +0.038 | 0.026 | -1.23 % |
| MULTI_AUGMENT | 100054 | box | 78.7 % | 48.1 % | **0.35 %** | 0.19 % | +0.014 | 0.022 | +0.86 % |
| MULTI_AUGMENT | 347 | box | 85.5 % | 56.8 % | **0.36 %** | 0.08 % | +0.020 | 0.023 | +0.23 % |
| MULTI_AUGMENT | 403 | box | 89.4 % | 64.8 % | **0.60 %** | 0.18 % | +0.035 | 0.029 | -0.69 % |
| MULTI_AUGMENT | **100186** | box | 89.1 % | 68.7 % | **9.47 %** | 2.73 % | +0.107 | 0.162 | +2.42 % |
| WIDE_BOX | 100051 | box | 83.2 % | 49.4 % | **0.17 %** | 0.05 % | +0.028 | 0.020 | -0.92 % |
| WIDE_BOX | 347 | box | 80.7 % | 48.5 % | **0.04 %** | 0.02 % | -0.004 | 0.017 | +2.07 % |
| WIDE_BOX | 244 | box | 81.0 % | 45.0 % | **0.22 %** | 0.26 % | -0.019 | 0.019 | +3.83 % |
| CUTMIX_FULLAUG | 100032 | cut | 97.1 % | 90.8 % | **37.6 %** | 8.44 % | +1.02 | 2.55 | +4.93 % |
| CUTMIX_FULLAUG | 115 | cut | 98.7 % | 95.9 % | **68.5 %** | 5.82 % | +4.43 | 21.5 | **-39.1 %** |
| CUTMIX_FULLAUG | 100051 | box | 97.0 % | 91.0 % | **43.9 %** | 10.8 % | +0.62 | 0.77 | -8.15 % |
| CUTMIX_FULLAUG | 974 | box | 96.9 % | 90.2 % | **33.3 %** | 10.5 % | +0.30 | 0.36 | +2.82 % |

**Regime A, a genuine average** (the box arms). Four fifths of the energy sits on directions off
by more than 3 %, but only a few tenths of a percent sits on directions off by more than a factor
of two. Thousands of directions each wrong by 5-10 %, `mu` confined to [0.3, 4], `bias` and
`halfvar` both 0.01-0.04 and partially cancelling. The net is 0.2-3 % and its fluctuation is
rms/sqrt(n_eff). This is a sound basis for a contract, and `halfvar` is the honest bound.

**Regime B, a gross cancellation** (the cut arm, and the box seats inside it). A third to two
thirds of the load energy sits on directions made more than twice too stiff - that compliance is
simply lost - and 6-11 % on directions more than twice too soft, which puts it back. `bias` and
`halfvar` are 0.3-21, so the second-order expansion is meaningless and only the exact harmonic
sum `sum_i w_i/mu_i` means anything. On the trained seat 100032 the two halves cancel to +4.9 %;
on the held-out seat 115 the cancellation fails and the answer is -39 %.

**Consequence for how we judge cut cells.** A cut arm's smooth-face-load median is the residue of
that cancellation, not a measure of accuracy. Cut cells must be judged by the assembly or by the
`mu` spectrum, never by the face-load median alone.

**It also explains seat 100186**, whose sensitivity has been the one box-cell outlier at 7.8 %
in every model including the ensemble. It is the only box seat with a large energy share on
grossly wrong directions: 9.47 % against 0.04-0.76 % for every other box seat, `mu_max` 41. The
module energy split is wrong exactly where those directions carry energy. It is not model
variance - all three models give 7.3-7.8 % - it is that seat's spectrum.

## 3. The cut cell's ceiling: a well-fitted cut cell does assemble inside the contract

The open question was whether a cut cell's near-mechanism directions (`mu_max` 275 to 2.4e9, and
`lambda_min(A)` genuinely small by physics) make the assembled contract unreachable in principle.
They do not. Two-cell assembly of the same seat 100032 (40 % cut), three models:

| model | face-load median | worst assembled compliance | worst sensitivity | `e_A` |
|---|---|---|---|---|
| **NOAUG_100032** (one seat, no augmentation) | 0.278 % | **1.256 %** | **1.473 %** | 3.70 |
| SOLO_100032 (same seat, with augmentation) | 1.078 % | 5.91 % | 13.95 % | 1604 |
| CUTMIX_FULLAUG (28 cut + 32 box seats) | 4.15 % | - | - | - |

`e_A`, the operator error in the `A` metric, is 3.7 for the model that passes and 1604 for the one
that fails, on the same geometry and the same label. So the required fit quality is reachable and
the physics follows once it is reached. The cut-cell gap is therefore capacity and data - a
multi-seat arm not reaching single-seat quality - not an unlearnable target.

The label's condition number confirms the two families differ in kind but not catastrophically:
`cond(A^-1)` is 10^5.4 for box seats with `lambda_min` pinned at 28.0-28.5 across all 57, against
10^5.6 to 10^7.4 for the 116 cut seats with `lambda_min` 3.7-18.

Seat 100079 (73 % cut) cannot be self-glued at all - the cut removes every opposite face pair -
so it can only be accepted inside a cut-to-full stack.

## 4. A queue bug that would have failed the whole lattice programme

`/root/_after_arms.sh` invoked `superelement.equi.assemble_lattice_multi`, which does not exist:
the multi-seat rewrite lives in `assemble_lattice.py`, and the copy deployed on the box was still
the 291-line single-seat version. Every cut-to-full lattice run would have died on
`unrecognized arguments`. Fixed by shipping the 361-line multi-seat script and re-validating:
one seat, grid 2x1x1, reproduces `assemble_two` to `check_two` = 3.175e-14.

## 5. What this changes

* Judge cut cells by assembly or by `mu`, not by the face-load median.
* The target for the cut-cell arms is explicit: single-seat quality, `mu_max` of order 100, `e_A`
  of order 1. `n` = 2, 4, 8, 16 presented cut seats at fixed architecture and fixed steps is
  queued to separate capacity per seat from data, with seat 100032 in every arm so the in-sample
  curve is comparable to NOAUG_100032's 0.278 %, and seat 115 never presented for the
  generalisation curve.
* For box cells, `E(mu>2)` is a cheap per-seat predictor of a sensitivity outlier, computable from
  one `soft_diag` run without any assembly.
