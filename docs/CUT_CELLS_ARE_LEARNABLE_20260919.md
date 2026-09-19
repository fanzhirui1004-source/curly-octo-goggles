# Cut cells: what the single-seat test settles, and a claim I withdraw

2026-09-19. Commit `12790da` was titled "Cut cells are not learned by this architecture". The
architecture part of that is **wrong** and this note withdraws it. The single-seat memorisation
test — one geometry, 20 000 steps, the protocol the phase-2 arms used on seat 0253 — says the
representation is adequate at moderate cut fractions and that the mixed arm's failure was mostly
capacity and interference.

| seat | cut fraction | arm | `factor_rel` | face median | face worst | inside 3 % | point-load | `g` |
|---|---|---|---|---|---|---|---|---|
| 100032 | 0.398 | mixed, 64 seats | 0.357 | 18.079 % | 50.996 % | 0.04 | 0.478 | 1.42e5 |
| 100032 | 0.398 | **single seat** | **0.083** | **1.191 %** | 8.559 % | **0.85** | 0.141 | 5.17e5 |
| 100079 | 0.731 | mixed, 64 seats | 0.866 | 78.618 % | 88.985 % | 0.00 | 0.867 | 2.4e9 |
| 100079 | 0.731 | **single seat** | 0.180 | 12.726 % | 27.868 % | 0.03 | 0.455 | 2.385e9 |

## Three findings

**A cut cell at a moderate cut fraction is learnable to within the contract.** Seat 100032 alone
reaches a 1.191 % median smooth-face compliance error with 85 % of that load family inside 3 %.
That is the same order as the box-only multi-geometry arm's 0.764 %. So nothing about the kind-1
functional representation prevents learning a cut cell.

**The mixed arm was starved, not incapable.** The same two seats improve by 4x to 15x on every
measure when the model is given to one geometry. With 56 presented seats spanning 237 to 7668
coordinates, and the final loss 17x the box-only baseline's (0.0968 against 0.0056, worse than
the baseline at step 10 000), the 64-seat arm was nowhere near its fit. Capacity, curriculum or
per-size grouping is the lever, not a new geometry descriptor.

**Heavily cut cells are a separate, real wall.** Seat 100079 at 73 % cut reaches only 12.7 %
median with the entire model devoted to it, and 3 % of loads inside the contract. The error
tracks the cut fraction even in the single-seat limit, so that part is difficulty, not budget.

## `g` is structurally meaningless for a cut cell

Seat 100032 single-seat has `g = 5.17e5` while its physics is 1.191 % median with 85 % inside
3 %. Five orders of magnitude of `g` sitting on top of a good answer is the most extreme
decoupling measured in this project, and there is a reason it is structural rather than
accidental: a cut cell genuinely has near-mechanism directions — a sliver of material barely
attached to the rest — where `M_q`'s smallest eigenvalues are near zero *by physics*. `g` is
`1 / nu_min` of the pencil, so any imperfect prediction on a direction that is legitimately
almost free sends it to infinity while the load never goes there.

For cut cells `g` and `e_A` must not be reported as accuracy at all. The assembly-free response
gate is the measure.

## What this does not settle

The full cells in the mixed arm were also 9x worse than the box-only baseline (6.960 % against
0.764 %), which is still a controlled comparison and still says mixing cost them. The
single-seat control on a full cell (seat 100051) is running and will say how much of the
single-seat gain is simply that one geometry is easier than 56.
