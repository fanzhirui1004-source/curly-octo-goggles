# Dataset production for the CutFEM route: what to keep, what to fix, and what the symmetries buy

Handoff brief, 2026-09-08.  Audience: the agent working the CutFEM label route.  Everything numeric below is
measured on the produced population of the body-fitted route (2009 cells drawn, 1943 labels produced), from
`population/POPULATION.jsonl`, `population/CENSUS.jsonl` and `production/PRODUCTION_CENSUS.json`.  The generator is
`scripts/pred777h_full_cube_v1/make_sheet_population.py`.

The dataset design is **geometry sampling** and is therefore route independent: the same population can feed a
CutFEM route unchanged.  Three things are NOT route independent and section 5 says what you have to re-derive.

## 1. What is settled and should not change

**Graded thickness field.**  Material is the Schwarz-P sheet `|phi(x)| <= tau(x)`, `phi = sum_k cos(2 pi x_k)`, with
`tau` **trilinear** from the cell's eight corner values — a mean plus a linear gradient plus higher trilinear modes
up to 20 % of the gradient.  A uniform-thickness population would be a different and much easier problem; the graded
field is the point of the route, because it is what a density filter actually produces.

**tau in [0.1755, 0.8775].**  This is the full-unit-cell relative density rho in [0.10, 0.50] under `tau = 1.755 rho`.

That conversion is worth stating precisely, because the contract currently mislabels it.
`sheet_contract.py` carries

    density_law_reference = "Hao 2023 Table 2.2: C = 1.755 rho (tau = 0.4 -> rho = 0.228; measured 0.2286)"

which reads as a STIFFNESS law and, taken that way, exceeds the Voigt bound by 1.30x at rho = 0.228 — impossible for
any solid/void composite.  It is not a stiffness law.  **1.755 is the tau-to-density conversion**, and the generator
itself uses it correctly (`tau = 1.755 * rho` in the anchor loop).  Verified on the 707 uncut cells of the census:

    mean(tau) / measured material volume :  p5 = 1.7459   p50 = 1.7493   p95 = 1.7545
    predicting volume as tau / 1.755     :  relative error p50 = 0.32 %, p95 = 0.52 %

So the line should be renamed `tau_to_density` and a stiffness reference sought separately.  Do not carry the
mislabel into the CutFEM contract.

**One cut plane per cell, vertical family.**  The manifest carries a single `cut_plane` with normal
`n = (cos theta, sin theta, 0)` and retained side `n . x <= d`.  Two structural limits follow, both recorded as open
in STEP5 section 5 and neither addressed here: a cell at a concave corner of the cut region sees two or three planes
and is unsupported, and a fully oblique plane (all three normal components nonzero) is outside the family — the
population contains **0** such cells and no symmetry reaches one (section 4).

## 2. Thickness coverage: measured, and NOT a bug

I previously reported a sampling bias here.  That report measured the wrong variable and is withdrawn.  The design
parameter that matters is the **corner span** (max minus min over the eight corners), because that is the in-cell
thickness spread a density filter of radius 1.5 cells can produce, and the rejection test `span <= 0.47` is written
on it.  Over the 1939 usable cells:

| corner span >= | cells | share |     | \|g\|_2 >= | cells | share |
|---|---|---|---|---|---|---|
| 0.30 | 274 | 14.1 % |  | 0.30 | 10 | 0.5 % |
| 0.35 | 181 | 9.3 % |   | 0.35 | 3 | 0.2 % |
| 0.40 | 105 | 5.4 % |   | 0.40 | 2 | 0.1 % |

span: p50 = 0.085, p90 = 0.346, p95 = 0.404, p99 = 0.450, max = 0.702 (an anchor).

The two columns differ because the corner span over the unit cube is exactly `||g||_1`, not `||g||_2`, so a
body-diagonal gradient reaches at most `0.47 / sqrt(3) = 0.271` in `||g||_2` while an axis-aligned one reaches 0.470.
Sorting by `||g||_2` therefore selects axis-aligned cells and manufactures an apparent bias.  Sorting by span, which
is the physical quantity, there is none:

    axis-alignedness max|g_i| / |g| :  all cells p50 = 0.834
                                       top 5 % by SPAN   p50 = 0.813   (no bias, marginally the other way)
                                       top 5 % by |g|_2  p50 = 0.865   (the artefact of the wrong variable)

A 200 000-draw simulation confirms that re-parameterising the draw by span (draw the target span S, then set
`|g| = S / ||dir||_1`) changes the accepted distribution by nothing measurable: accepted span p95 0.3818 vs 0.3827,
share at span >= 0.40 3.78 % vs 3.84 %, axis-alignedness p50 0.834 vs 0.840.

**Conclusion: keep the thickness sampling as it is.**  One cosmetic defect remains and is worth removing when the
generator is next touched: the `force_top` branch (10 % of draws pushed to `|g|_2` in [0.40, 0.47]) is effectively
dead, because those draws have `||g||_1` above the cap and get rejected.  It costs acceptance rate (0.637 against a
possible 0.778 in simulation) and buys no coverage.  It does not affect the data already produced.

Two facts to know rather than fix: 359 cells (17.9 %) have span < 0.01 and 775 (38.6 %) have span < 0.05.  Nearly
40 % of the population is close to uniform thickness.  That is the deliberate consequence of `mag = 0.47 * u^2`
(small gradients dominate).  Whether that is the right prior for training is a sampling preference, not a defect.

## 3. Cut sampling: the azimuth is clean, the depth is not

**Azimuth: keep.**  `theta` uniform in [0, 45 deg], the other orientations reached by symmetry (section 4).  Nine
5-degree bins over the 1302 drawn cut cells: 155, 145, 149, 146, 141, 144, 147, 136, 139.  Nothing to fix.

**Depth: this is the real bug, and it is worth discussing before regenerating anything.**

The generator draws the **offset** uniformly: `c = row * (a + b)`, i.e. uniform over the range of `n . x` on the
cell.  The retained volume is not linear in `c` — for a plane cutting a square the area function is piecewise
quadratic with zero derivative at both ends — so a uniform offset piles draws up at both extremes.  Measured over
the 1234 cut cells of the training set, retained fraction of the cube:

    deciles: [190, 105, 102, 93, 91, 93, 97, 94, 125, 244]
    below 0.1: 15.4 %      in [0.4, 0.6]: 14.9 %      above 0.9: 19.8 %

A third of the cut cells are either almost the whole cube or almost nothing, and the genuinely-half-cut regime that
carries the most cut-face physics is the thinnest slice of the design.

**Proposed fix: sample the retained volume uniformly and invert.**  For `n = (a, b, 0)`, `a >= b > 0`, `s = a + b`,
the retained area in the unit square is exact and closed form, and so is its inverse:

```python
def V(a, b, c):                      # retained fraction of the cube for  a x + b y <= c
    a, b = max(a, b), min(a, b); s = a + b; c = np.clip(c, 0.0, s)
    return np.where(c <= b, c * c / (2 * a * b),
           np.where(c <= a, (c - b / 2) / a,
                    1.0 - (s - c) ** 2 / (2 * a * b)))

def Vinv(a, b, v):                   # the offset that retains fraction v
    a, b = max(a, b), min(a, b); s = a + b
    vb = b / (2 * a); va = (a - b / 2) / a
    return np.where(v <= vb, np.sqrt(np.maximum(v * 2 * a * b, 0.0)),
           np.where(v <= va, v * a + b / 2,
                    s - np.sqrt(np.maximum((1 - v) * 2 * a * b, 0.0))))
```

Draw `v` uniform (keep it on the Sobol dimension that currently carries the offset, so the design stays
space-filling) and set `c = Vinv(a, b, v)`.  Simulated over 200 000 draws:

| | deciles of retained volume | < 0.1 | [0.4, 0.6] | > 0.9 |
|---|---|---|---|---|
| offset-uniform (current) | 18.9, 8.9, 7.7, 7.3, 7.1, 7.3, 7.3, 7.7, 8.8, 19.1 | 18.9 % | 14.4 % | 19.1 % |
| volume-uniform (proposed) | 10.0, 10.1, 9.7, 9.9, 10.0, 10.1, 10.0, 10.0, 10.0, 10.1 | 10.0 % | 20.1 % | 10.1 % |

Flat, as intended, and the half-cut band grows by 40 %.

**Side effect on EMPTY cells, which is a decision, not a defect.**  All 59 EMPTY cells of the census were cut cells
whose retained cube volume was below 0.019 (p50 = 0.0018, p95 = 0.0098) — the retained sliver holds no band.  Under
volume-uniform sampling about 1.0 % of cut draws would land below that p95, against 5.8 % now.  The standing
decision on this route was that near-empty cells are INCLUDED and thin cut faces are RECORDED, not excluded; the
fix does not change that policy, it only stops spending a sixth of the cut budget on the two extremes.  If the
CutFEM route wants the small-cut regime *deliberately* oversampled (see section 5), draw `v` from a distribution
with a deliberate mass near zero rather than getting it by accident from the offset parameterisation.

## 4. Symmetry augmentation: what it is, and what it does not do

The population holds vertical cuts only and one canonical orientation per cell; the other orientations are produced
**at training time** by the 48 signed permutations of the cube (the full symmetry group including reflections).  No
rotated copies are stored.

**It is exact, and that was earned.**  Under the label's symmetric carrier the action `T S T^T` (T = carrier-node
permutation composed with R) is exact to **7.5e-15** on a fixed mesh, and the residual between a cell's label and its
image's independently produced label is **0.8 %**, i.e. the mesh-convergence level, unbiased.  Before the carrier
was made symmetric this residual was 2.8 % to 4.5 % and consistent in sign: the carrier triangulation (one diagonal
per square) is not cube-invariant, so a cell and its image received slightly different discretisations of the same
operator.  The fix was the symmetric square rule with the parity diagonal.  **This is the fact most worth carrying
across to CutFEM** (section 5).

**Orbit sizes, computed per label over the 1939 usable cells:**

    orbit size: p0 = 1, p50 = 48, p100 = 48, mean 47.80
    distribution: {1: 5, 6: 2, 8: 1, 24: 1, 48: 1930}
    effective training set: 92 689 distinct (geometry, operator) pairs from 1939 stored labels  (x47.8)

The small orbits are all anchors and all check out: the 5 uniform-rho anchors have the full group as stabiliser
(orbit 1); `anchor_grad_axis_x` has orbit 6 (six signed axes); `anchor_grad_diag` orbit 8 (eight sign choices);
`anchor_cut_theta00` orbit 6; and one Sobol draw, `pop_uncut_0305`, happened to be almost uniform and keeps a
residual order-2 stabiliser (orbit 24).

**What the augmentation buys, and what it does not.**  It multiplies ORIENTATIONS only:

* it **does** cover the whole family of planes parallel to a coordinate axis — a signed permutation sends
  `(cos t, sin t, 0)` to a vector with exactly one zero component, so that family is reached completely;
* it **does not** reach a fully oblique plane.  Zero cells in the population and zero reachable by symmetry.  If the
  CutFEM route needs arbitrary cut orientations, the population needs a polar-angle dimension and must be
  regenerated — the symmetries cannot substitute;
* it **does not** add thickness coverage.  A symmetry permutes the eight corner values, so the mean and the corner
  span of a cell are invariant along its whole orbit;
* it **does not** add depth coverage.  A symmetry sending `n -> -n` maps the retained set `{n.x <= d}` to
  `{n.x >= (a+b) - d}`, whose retained volume equals the original by the square's own centre symmetry.  **Retained
  volume fraction is an orbit invariant.**

So the U-shaped depth distribution of section 3 survives augmentation intact.  Augmenting a bad depth prior 48 times
gives 48 copies of the bad depth prior.  Fix the sampling; do not expect the symmetry to launder it.

## 5. What the CutFEM route has to re-derive rather than inherit

1. **Re-measure the symmetry residual on your own discretisation.**  The 0.8 % figure is a property of OUR carrier
   and mesh, not of the geometry.  The augmentation `T S T^T` is only as exact as the discretisation is
   cube-symmetric.  A CutFEM background grid that is itself the symmetric 32^3 Cartesian lattice with the same
   parity rule should inherit the result; anything else (a different cut-cell quadrature ordering, an
   agglomeration heuristic that is not symmetric, a level-set evaluated on a non-symmetric stencil) will not, and
   the failure mode is a consistent-sign bias of a few percent that no gate catches.  Run the equivalent of
   `sheet_rotation_check.py`: label a cell and its image independently, compare `T S T^T` against the image's label
   in the carrier low-frequency norm.  Do this BEFORE producing 2000 labels, not after.

2. **The small-cut regime is your stability question, not a meshing question.**  On the body-fitted route a
   vanishing cut fails by meshing (or produces a valid but nearly empty label); on CutFEM it fails by conditioning,
   and whatever you use — ghost penalty, cell agglomeration, or a stabilised extension — its parameters have to be
   validated exactly in the regime the depth sampling puts cells into.  That makes section 3 a coupled decision:
   the current design places 18.9 % of cut cells below 0.1 retained volume, the proposed fix 10 %.  Decide the
   stabilisation first, then decide whether you want that regime at 10 %, more, or less, and set the `v`
   distribution accordingly.  We measured on our route that the cut-face carrier node set SATURATES below one
   carrier spacing (98 nodes, port dimension 294, identically at every azimuth) and what degrades is only the
   triangle aspect ratio, about 4h / width — a CutFEM route should confirm the analogous statement for its own
   cut-cell quadrature before trusting thin-cut labels.

3. **Decide the port space before the population.**  Our labels are assemblable because the carrier is the exact
   intersection of a fixed 32^3 background grid with the cell's exact rational boundary planes, shared by the two
   cells of any internal interface and independent of the volume mesh.  If CutFEM adopts the same carrier, the
   augmentation permutation T and the whole assembly contract transfer unchanged and the two routes' labels are
   directly comparable — which would also give us the independent cross-check that the CGAL reference only
   partially provides.  If it adopts a different port space, the population is still reusable but nothing else is.

## 6. Concrete proposal to discuss

1. Keep: graded trilinear thickness, `tau` in [0.1755, 0.8775] (`rho` 0.10-0.50 at `tau = 1.755 rho`), theta
   uniform in [0, 45 deg], one vertical cut plane per cell, symmetry augmentation at training time.
2. Fix: cut depth sampled uniformly in retained volume via `Vinv` above.  This is the only sampling change I would
   make on the evidence.
3. Remove: the dead `force_top` branch (acceptance rate only).
4. Rename in the contract: `density_law_reference` -> `tau_to_density`, and drop the "C = " reading.
5. Decide, jointly with the CutFEM stabilisation choice: what share of cut cells should sit below 0.1 retained
   volume.  10 % is what plain volume-uniform gives; state the number you want rather than inheriting 18.9 %.
6. Open and NOT proposed here, both structural: fully oblique cut planes (needs a polar-angle dimension and a
   regenerated population) and more than one cut plane per cell (needs a manifest change).

## 7. Reproducing these numbers

`scripts/pred777h_full_cube_v1/diagnostics/` — `coverage.py` (design coverage), `cov2.py` (training-set coverage and
the orbit computation), `fixes.py` (the two simulations of section 2 and 3).  They read only
`population/POPULATION.jsonl`, `population/CENSUS.jsonl` and `production/PRODUCTION_CENSUS.json`, all of which are
in this repository, so they run without the production box.
