# Where the column scale comes from at prediction time

Date: 2026-09-15. Seat 0328 for the structural measurement; leave-one-seat-out over six seats
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

Each row fits on the other five seats and scores on the held-out one, so the test geometry is
never seen.

| held-out seat | raw spread | direct, quad | pullback, quad | 5-95% core (pullback) | oracle |
|---|---|---|---|---|---|
| 0328 | 2.70e3 | 25.1 | 19.9 | 3.10 | 1.165 |
| 0253 | 2.94e3 | 25.6 | 20.3 | 3.29 | 1.175 |
| 0403 | 1.64e4 | 23.6 | 16.7 | 3.32 | 1.206 |
| 0347 | 1.18e4 | 17.8 | 13.7 | 3.48 | 1.133 |
| 0974 | 1.01e4 | 19.9 | 16.8 | 2.81 | 1.125 |
| 0575 | 1.13e4 | 156.0 | 65.5 | 3.08 | 1.133 |

Linear features are not enough anywhere: 109 to 316 on the pullback route. Quadratic features
reach 14 to 20 on five of six seats.

Seat 0575 is the exception at 65.5, but its 5-95% core is 3.08, in line with everything else.
A handful of extreme coordinates miss, not the body as a whole.

## Reading

**A network can carry its own preconditioner.** Predict `log c` per physical trace dof from
local geometry, then apply the exact `(B.^2)` map. On a geometry never seen this brings the
curvature spread from about 1e4 down to 14 to 20, against an oracle floor of 1.13 to 1.21. The
original target was "about 10 is enough for one learning rate".

**The remaining gap is regression error, not the coordinate mismatch.** With true `c` the
pullback lands at 1.17. Everything above that is the geometry model.

**The premise that sent step 2 to physical compliance was wrong, but the destination was
right.** Quotient coordinates do index nodes. Regressing `s` directly works and is simpler.
It is just slightly worse, because `c` is the more local quantity.

## Limits

- Six seats, `d` from 12792 to 16122. The larger seats in the set were not tested.
- Ridge regression on 28 hand-built features with a quadratic expansion, 435 parameters, about
  72000 training rows per fold. This is a floor on what a network should achieve, not a ceiling.
- One cut-plane family. Whether the features transfer across topologies is untested.
- The scores are spreads of `s / s-hat` over all coordinates. They say the learning rate can be
  shared; they do not say the factor fit succeeds.
