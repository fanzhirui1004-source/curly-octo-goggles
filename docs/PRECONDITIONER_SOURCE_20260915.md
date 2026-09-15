# Where the column scale comes from at prediction time

Date: 2026-09-15. Seat 0328 for the structural measurement; leave-one-seat-out over twelve seats
for the transfer measurement.
Scripts: `superelement/factor_fit/step2_bridge.py`, `step2_quotient_scale.py`, `step2_cross.py`.

## The question

The only factor parameterization that trains scales column `j` of the raw upper-triangular
factor by `1/sqrt(s_j)`, with `s_j = (A^-1)_jj` the curvature of the divergence in that
coordinate. Without it the curvature spans 2.7e3 to 1.6e4 and no single learning rate works.
A network has no `A` at prediction time, so `s` has to come from geometry.

Step 2 regressed instead `c_k = (S^+)_kk`, the compliance of physical trace dof `k`, on the
stated ground that compliance "indexes nodes directly instead of quotient coordinates". That
premise was not checked.

## What the quotient map actually is

`B = E_{6:} H_6 ... H_1 P`: a stored permutation `order`, six dense Householder reflectors,
then drop the first six rows. So quotient coordinate `j` **is** physical dof `order[j+6]`. The
bijection is explicit and the same per-node features apply to both. Only the six anchor dofs
`order[0:6]` have no partner.

Measured on seat 0328:

| quantity | value |
|---|---|
| `\|\|BB^T - I\|\|_max` | 6.11e-15 |
| rows whose largest entry sits at `order[j+6]` | 100% |
| `\|B_{j,order[j+6]}\|` | 0.99937 to 0.99974 |
| off-peak row energy | 5.27e-4 to 1.25e-3 |

So rows of `B` are coordinate vectors to within about 0.1% of their energy.

## The values still differ, and the reason matters

| ratio | min | max | spread |
|---|---|---|---|
| `s_j` alone | 5.48e2 | 1.48e6 | 2.697e3 |
| `c_k` alone | 1.98e2 | 1.48e6 | 7.476e3 |
| `s_j / c_{order[j+6]}` | 0.885 | 3.628 | **4.098** |
| `s_j / (B.^2 c)_j` | 0.877 | 1.022 | **1.165** |
| best any relabelling could do | 0.759 | 2.746 | 3.616 |

Dividing by the aligned compliance leaves a factor of 4. That looks small next to 2.7e3, but it
is not noise: it beats the best possible relabelling of `c`, which means no index map fixes it.

The cause is the 0.1% off-peak energy. The rigid-mode correction puts it on directions whose
compliance is up to 7000 times larger, so a 1e-3 admixture contributes roughly 45% of the
value. The diagonal pullback `(B.^2) c` weights each `c_k` by `B_jk^2` and recovers `s`
to within 1.17.

`B` depends only on node positions and the stored ordering. It is pure geometry, so the
pullback is available at prediction time at no cost.

## Within one body, holding out 20% of nodes (seat 0328)

| model | preconditioned spread | 5-95% core | R2 |
|---|---|---|---|
| constant | 2646.6 | 607.7 | 0.000 |
| regress `log s`, ridge linear | 56.4 | 11.03 | 0.894 |
| regress `log s`, ridge quadratic | 13.3 | 2.77 | 0.979 |
| regress `log c` then pullback, linear | 44.8 | 8.79 | 0.924 |
| regress `log c` then pullback, quadratic | **10.9** | **2.39** | 0.986 |

The two-stage route wins. `c` is a local physical quantity and regresses better than `s`; the
pullback that turns one into the other is exact and free.

## Across bodies: leave one seat out

Each row fits on the other eleven seats and scores on the held-out one, so the test geometry is
never seen.

| held-out seat | d | raw spread | direct, quad | pullback, quad | 5-95% core (pullback) | oracle |
|---|---:|---:|---:|---:|---:|---:|
| 0120 | 16914 | 1.48e4 | 104.2 | 55.3 | 2.69 | 1.144 |
| 0196 | 18066 | 1.38e4 | 270.4 | 146.6 | 3.00 | 1.132 |
| 0244 | 17574 | 7.98e3 | 19.4 | 14.2 | 2.71 | 1.208 |
| 0253 | 12822 | 2.94e3 | 23.1 | 14.8 | 3.41 | 1.175 |
| 0328 | 12792 | 2.70e3 | 18.5 | 14.3 | 3.28 | 1.165 |
| 0347 | 15702 | 1.18e4 | 16.7 | 12.8 | 3.19 | 1.133 |
| 0403 | 14562 | 1.64e4 | 25.8 | 14.9 | 3.21 | 1.206 |
| 0575 | 16122 | 1.13e4 | 35.0 | 18.8 | 2.72 | 1.133 |
| 0882 | 16158 | 1.26e4 | 67.2 | 30.8 | 2.74 | 1.142 |
| 0920 | 17106 | 1.29e4 | 30.5 | 20.6 | 2.52 | 1.138 |
| 0941 | 16542 | 2.29e4 | 40.6 | 24.1 | 2.97 | 1.215 |
| 0974 | 16050 | 1.01e4 | 16.0 | 12.3 | 2.95 | 1.125 |

Pullback route, quadratic features: median 16.8, range 12.3 to 146.6. Its 5-95% core is
median 2.96, range 2.52 to 3.41. Direct regression of `log s`: median 28.2, range 16.0 to 270.4.

Two seats miss on the full spread, 0196 at 146.6 and 0120 at 55.3, yet both have 5-95% cores
tighter than the median. The misses are a handful of extreme coordinates, not the body. Seat
0575 read 65.5 when the fit used five training seats and 18.8 with eleven, so the outliers
shrink as training geometry is added rather than being intrinsic.

Linear features are not enough anywhere: 109 to 316 on the pullback route at six seats.

## Reading

**A network can carry its own preconditioner.** Predict `log c` per physical trace dof from
local geometry, then apply the exact `(B.^2)` map. On a geometry never seen this brings the
curvature spread from about 1e4 down to a median of 16.8, against an oracle floor of 1.13 to
1.21. The original target was "about 10 is enough for one learning rate".

**The remaining gap is regression error, not the coordinate mismatch.** With true `c` the
pullback lands at 1.17. Everything above that is the geometry model.

**The premise that sent step 2 to physical compliance was wrong, but the destination was
right.** Quotient coordinates do index nodes. Regressing `s` directly works and is simpler.
It is just slightly worse, because `c` is the more local quantity.

## Limits

- Twelve seats, `d` from 12792 to 18066, of the 32 that have reference factors. The largest
  (`d` above 25000) were not tested.
- Ridge regression on 28 hand-built features with a quadratic expansion, 435 parameters, about
  72000 training rows per fold. This is a floor on what a network should achieve, not a ceiling.
- One cut-plane family. Whether the features transfer across topologies is untested.
- The scores are spreads of `s / s-hat` over all coordinates. They say the learning rate can be
  shared; they do not say the factor fit succeeds.
