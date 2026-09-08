# STEP 9 — the full production run: 2009 cells

Written 2026-09-07, after the main dispatcher and the sweep both finished.  Everything below is measured over the
whole run, not a sample.  The census script is `scripts/pred777h_full_cube_v1/diagnostics/production_census.py`; its
output is `PRODUCTION_CENSUS.json`, the stiffness pass is `STIFFNESS_FULL.json` and `STIFFNESS_SUMMARY.json`.

## 1. What came out

| | cells |
|---|---|
| requested | 2009 |
| receipts written | 1999 (99.50 %) |
| PASS | 1943 (96.71 %) |
| EMPTY (retained material below the 1e-6 volume floor) | 55 |
| GEOMETRY_DEGENERATE | 1 (`pop_anchor_cut_theta45`) |
| no receipt at all | 10 |

Of the 1943 PASS labels: 1238 carry a world cut plane, of which **1186** have `cut_0` as a load-bearing port and 52
have a cut plane whose face touches no material; **705** are truly uncut.

Those two flags are not the same thing and an earlier draft of the census used one for both.  It put 52 cut cells
into the "uncut" density fit and, worse, reported the cut-face width statistics over the wrong subset, which hid the
thin faces STEP8 had measured.  Both are corrected here and in the script.

**Code identity.**  The receipts carry five different `git_head` values (`f7b3fda` 11, `7bdd3e1` 1896, `cd0417b` 5,
`8c61d04` 38, `8f263a5` 49) because reports were committed while the run was in flight.  `git diff f7b3fda 8f263a5 --
src scripts/pred777h_full_cube_v1/produce_sheet_label.py` is **empty**: every label in this run was produced by
byte-identical code.  The later commits touched only reports and the CGAL brief.

## 2. The operator gates, over all 1943 labels

Every gate holds at machine precision on every label; these are the full-set distributions, not spot checks.

| gate | p50 | p100 (worst) |
|---|---|---|
| rigid-body residual (relative) | 2.5e-15 | 5.8e-13 (`pop_cut_1142`) |
| min eigenvalue / max eigenvalue | -4.3e-16 | -2.3e-14 (`pop_uncut_0493`) |
| inactive-block max entry | 0 | **0** (exactly, all cells) |
| partition-of-unity error | 2.2e-16 | 2.2e-16 |
| affine-reproduction error | 2.6e-16 | 5.3e-15 |
| float32 storage rounding (relative) | 3.3e-8 | 4.9e-8 (p95) |
| support outside the active set | — | **0** (summed over all cells) |

The affine-reproduction error matters more than it looks: it is the guarantee that the six uniform-strain
stiffnesses of section 6 are exact quantities of the operator and not an artefact of the carrier.

## 3. The null space

`null_dim_lower_bound` in the receipt is `6 + 3 * n_unsupported`, which assumes ONE connected component.  Measured
against the pipeline's float64 spectrum at a 1e-14 relative threshold:

* **1927 / 1943 (99.18 %) equal the lower bound exactly.**
* 16 do not, and every one is attributable:
  * **+6 in 6 cells, all with 2 surface components** — the formula's one-component assumption.  A second free
    component contributes 6 more rigid modes.  Not an operator defect; the formula is what is incomplete.
  * **+3 in 7 cells** (6 with one component, 1 with two) — three extra dimensions is one carrier node whose three
    dofs are decoupled.  This is the residual tail of the STEP6 §8 family (a carrier column whose entire material
    trace collapses to a single mesh vertex).  The exact mechanism per cell is **not** pinned down; see section 10.
  * **+1 in 2 cells and -1 in 1 cell** — an eigenvalue straddling the threshold.  For `pop_cut_1142` the count at
    1e-10 is exactly the lower bound.  Extra null modes must come in multiples of 3 on a vector carrier, so a
    difference of one is by construction a threshold placement, not a mode.

Unsupported active carrier nodes are 14.0 % of the active set at the median (p95 23.5 %, p100 63.6 %).  These are
the deliberate consequence of the geometric active set being a superset of any mesh's support, and they are exactly
the rows the network must learn to output as zero — or, better, be told about (`support_mask` ships in the npz).

## 4. A precision finding the contract does not yet state

The receipt's `rank_at_1e-14` and `null_dim_at_1e-14` are computed inside the pipeline on the **float64** operator.
The file we ship is **float32** (`float32_rounding_relative` ~ 3.3e-8).  A consumer that loads
`FIXED_PORT_SCHUR_TET10.npz` and asks for the rank at 1e-14 gets **zero null modes** — the receipt's numbers do not
describe the shipped artefact.

Measured on the shipped file (supported block, six cells including 1- and 2-component cases), the null space is
cleanly resolvable, with a gap of **3 to 6 orders of magnitude**, and every relative threshold in **[1e-8, 1e-6]**
returns the identical count:

```
pop_cut_0473  gap after mode 6  6.1e-10 -> 1.7e-04  (ratio 2.8e5)   null dim: 0 @1e-14, 6 @1e-8 .. 1e-6
pop_cut_0355  gap after mode 12 1.1e-09 -> 5.1e-05  (ratio 4.8e4)   null dim: 0 @1e-14, 12 @1e-8 .. 1e-6
pop_uncut_0447 gap after mode 9 7.2e-10 -> 2.9e-04  (ratio 4.0e5)   null dim: 0 @1e-14, 9 @1e-8 .. 1e-6
```

**The consumer-side threshold of record is 1e-8 relative to lambda_max.**  This belongs in the contract and the
receipt should carry the rank at shipped precision alongside the float64 one.  It is a reporting defect, not an
operator defect: the gap is wide and unambiguous.

## 5. The sliver artefact after the STEP8 fix

STEP8's signature was a top eigenvector sitting almost entirely on the four corners of one carrier square, together
with a lambda_max well above its converged value.  Before the fix that was 26 % of labels; the 300-cell re-production
put ">= 0.99 on one square" at 0 and ">= 0.95" at 1.7 %.  Over the full set:

| top-mode mass on 4 nodes | cells | share |
|---|---|---|
| >= 0.99 | 10 | 0.51 % |
| >= 0.95 | 57 | 2.93 % |
| >= 0.90 | 308 | 15.8 % |
| median | 0.787 | |

A high top-mode share is **not** by itself a defect: lambda_max of a port Schur operator is naturally local, and the
median label sits at 0.79.  The discriminator is a localised top mode **together with** a lambda_max the density does
not explain.  Over the full set, `log lambda_max = -0.219 log rho - 3.474` with a residual sd of 0.383, and:

| flag | cells |
|---|---|
| top_mode >= 0.99 AND lambda_max residual > 3 sd | **2** (0.103 %) |
| top_mode >= 0.99 AND residual > 2 sd | 4 |
| top_mode >= 0.95 AND residual > 3 sd | 5 |
| residual > 3 sd alone | 39 |
| top_mode >= 0.99 alone | 10 |

The two cells that meet the strict flag are `pop_cut_0509` (residual 5.9 sd, mesh min dihedral 0.023 deg) and
`pop_cut_1046` (6.0 sd, 0.034 deg).  Both still carry unrepaired cap needles (5 and 3, cap min angle 0.017 and 0.024
deg).  **The STEP8 artefact went from 26 % of labels to 2 of 1943.**

Two supporting facts that matter for how to read lambda_max at all:

* `spearman(lambda_max, material volume) = -0.501`, and the 137 cells with lambda_max above 3x the median have a
  median volume of 0.0093 against the population's 0.208 — **22x smaller**.  A high lambda_max is now mostly the
  correct physics of a nearly empty cell, not an artefact.
* `spearman(lambda_max, mesh min dihedral) = -0.039` and `spearman(lambda_max, cap min angle) = +0.024`: bad
  tetrahedra exist in quantity (section 7) and are, in aggregate, **not** reaching the operator.  The Schur
  complement condenses them away.  Only the extreme tail gets through.

## 6. Physics: the six KUBC apparent stiffnesses

Computed for every PASS label (`stiffness_pass.py`, 1943 ok / 0 failed): `C[i] = u_i' S u_i` with `u_i` the affine
port displacement of the i-th unit strain.  Affine fields lie exactly in every carrier space, so this is the one
quantity comparable across meshes, across carrier resolutions, and against an independent mesher — it is what the
CGAL reference will be compared on.

* **All 1943 x 6 = 11 658 values are strictly positive** (minimum 7.8e-7).  Positive semidefiniteness holds on the
  physically meaningful subspace, everywhere.
* Medians: `C_xx = C_yy = C_zz = 0.125`, `C_yz = C_xz = C_xy = 0.054`; shear/normal ratio 0.418 at the median.

**Voigt bound.**  For a solid/void composite, `C_ii <= rho * C_1111(solid)` with `C_1111 = E(1-nu)/((1+nu)(1-2nu)) =
1.34615` at E=1, nu=0.3.  Checked on every cell and every normal direction: **0 violations out of 1943**, with the
ratio `C_max / (rho C_solid)` at p50 = 0.491, p95 = 0.830, p100 = 0.989.  This is a strong, cheap, physically
grounded guard that the pipeline did not previously have, and it should be added as a production gate.

**Density power law** (705 truly uncut cells, rho in [0.100, 0.503]):

```
C_normal = 0.9543 * rho^1.3487        R^2(log) = 0.99843
dense half (rho > 0.300):  exponent 1.4716, prefactor 1.080
```

An exponent of 1.35 sits between stretch dominated (1) and bending dominated (2), leaning stretch, which is what a
sheet TPMS should do.  The fit quality (R^2 = 0.998 over a 5x density range) is itself evidence that the labels are
internally consistent: 705 independently meshed and solved cells fall on one line.

**The reference line in the contract is wrong and must not be used as an acceptance criterion.**  The contract
carries `density_law_reference: "Hao 2023 Table 2.2: C = 1.755 rho (tau = 0.4 -> rho = 0.228)"`.  At rho = 0.228 that
gives 0.400, which sits at 1.304 times the Voigt bound — no solid/void composite can reach it, under either reading
of `C` (Young's modulus ratio or the `C_1111` component).  Either the transcription or the meaning of `rho` in that
source is wrong.  Our own values satisfy the bound with room to spare.  This line should be removed from the
contract or replaced once the source is checked.

## 7. Surface and mesh quality

Surface, over all PASS labels: **0** cells with a non-two-manifold edge, **0** with a self-intersecting triangle,
**0** with a dropped component, 10 with more than one component.  Minimum triangle quality p50 = 0.029, p0 = 8.1e-6.

Cap needles: `remaining > 0` in **1216 cells (62.6 %)**, distribution p50 = 1, p95 = 6, max = 17; cap minimum angle
p50 = 1.07 deg, p0 = 0.00042 deg.  The repair targets 2 deg and collapses what the 2D link condition allows, leaving
the rest.  This is worse than the 300-cell sample suggested and it is honest to say so — but
`spearman(lambda_max, cap min angle) = +0.024`, i.e. the remaining needles are, in aggregate, not reaching the
operator.  They matter only in the extreme tail (section 5).

Volume mesh: min dihedral angle **below 1 deg in 828 cells (42.6 %)**, tets below 5 deg p50 = 43; **0** cells with a
non-manifold volume facet.  Mesh volume against surface volume agrees to 1.1e-15 at the median — the volume mesher is
not losing geometry, it is producing badly shaped tetrahedra inside correct geometry.

## 8. What the dataset covers

| | p0 | p5 | p50 | p95 | p100 |
|---|---|---|---|---|---|
| material volume (all) | 2.5e-6 | 0.0069 | 0.208 | 0.427 | 0.503 |
| material volume (cut) | 2.5e-6 | 0.0021 | 0.139 | 0.378 | 0.489 |
| material volume (uncut) | 0.100 | 0.142 | 0.297 | 0.454 | 0.503 |
| tau corner value | 0.176 | 0.235 | 0.528 | 0.818 | 0.878 |
| per-cell tau spread (max-min over 8 corners) | 0 | — | 0.086 | 0.404 | 0.702 |
| active carrier nodes | 25 | 502 | 2019 | 3127 | 3692 |
| q_active (operator size) | 75 | 1506 | 6057 | 9381 | 11076 |
| lambda_max | 0.0247 | 0.0343 | 0.0390 | 0.144 | 1.453 |

345 cells (17.8 %) have an effectively uniform thickness field (spread < 0.01) and 747 have spread < 0.05.  Among
the 1196 cells with spread > 0.05, the normal-stiffness anisotropy `C_max/C_min` reaches p95 = 1.33 and p100 = 2.15,
so the thickness gradient does produce real anisotropy where it is present.

**Cut-face widths.**  Over the 1238 PASS cells that carry a cut plane: p0 = **0.087** carrier spacings, p1 = 1.74,
p5 = 6.30, p50 = p95 = p100 = 32.  **6 PASS cells sit below one carrier spacing** and 5 below half of one.  A further
6 sub-spacing cut faces came back EMPTY.  That 6 + 6 = 12 matches STEP8's population scan (12 of 1302 below one
spacing, minimum 0.087 h) exactly: the thin faces are recorded, not excluded, as decided.

## 9. The 10 cells with no receipt

| cell | failure |
|---|---|
| `pop_cut_0124`, `pop_cut_0628`, `pop_uncut_0074`, `pop_uncut_0215` | surface is not watertight, **exactly 3 non-two-manifold edges** in each |
| `pop_cut_0110`, `pop_cut_0164`, `pop_cut_1061` | Gmsh algorithm 10 produced 2-5 non-manifold volume facets |
| `pop_cut_0026` | HXT 3D mesh failed |
| `pop_cut_0599` | Gmsh worker segfaulted (rc -11) |
| `pop_cut_0539` | a fine port point fell outside its global cell-patch trace support (`cut_0`, uv = [1.00022, 0.4668]) |

Four independent cells failing with *exactly three* non-two-manifold edges is a pattern, not noise, and is the most
likely single fix to recover cells.  `pop_cut_0539`'s `u = 1.00022` is a parametric coordinate just past the patch
edge — a tolerance, not a geometry failure.  Neither is diagnosed here.

Also `pop_anchor_cut_theta45` returned GEOMETRY_DEGENERATE; it is an anchor case, not a population cell.

## 10. Cost

* Slot-time summed over all cells: **1 244 744 s = 345.8 slot-hours**; per cell p50 = 448 s, p95 = 1948 s, max 8290 s.
* **Waiting on the memory gate: mean 289 s of a mean 641 s total, i.e. 45 % of the per-cell wall time was idle**
  (p95 = 1380 s, max = 7605 s).  The gate is doing its job (0 OOM kills after the `anon` fix) but it is the single
  largest remaining inefficiency and a smarter admission policy — admitting by predicted footprint from
  `fine_dof` rather than a flat reservation — is the obvious next gain.
* Schur solve p50 = 143 s; fine-mesh dof p50 = 498 507, max 886 272.
* Operator storage: **145.0 GB** over 1943 files (p50 73 MB, max 245 MB); the labels directory is 148 GB.

## 11. What this run certifies, and what it does not

Certified, over the full set and not a sample: the operator gates hold at machine precision; the null space is the
combinatorial one in 99.18 % of labels with every exception attributed; all six KUBC stiffnesses are positive in
every label and satisfy the Voigt bound in every label; 705 independently produced uncut cells fall on a single
density power law to R^2 = 0.998; the surface is watertight and self-intersection free everywhere; the STEP8 sliver
artefact is down to 2 labels in 1943.

**Not certified.**  Every convergence number in this route still compares the route against itself at a finer
setting.  Nothing here rules out an error shared by every tier — which is precisely the class STEP8 turned out to
be.  That is what the CGAL reference (`cgal_convergence_reference_v1/BRIEF.md`) is for, and it is not yet run.  Note
also that the CGAL reference as specified is fed the label's own surface, so it isolates the volume mesher, the
element and the solve; the surface construction remains checked only against itself.

## 12. Follow-ups this run created

1. **Contract**: state the consumer-side rank threshold (1e-8 relative to lambda_max) and add the shipped-precision
   rank to the receipt alongside the float64 one (section 4).
2. **Contract**: remove or fix `density_law_reference` — as transcribed it violates the Voigt bound (section 6).
3. **Gate**: add the Voigt bound `C_ii <= rho * C_1111(solid)` as a production gate.  It is one line, it is physics,
   and it costs nothing.
4. **Receipt**: `null_dim_lower_bound` should use the component count, not assume one component (section 3).
5. Pin down the `+3` null-space family: is it a carrier column whose material trace is a single mesh vertex?
6. The four "exactly 3 non-two-manifold edges" surface failures (section 9).
7. Admission policy by predicted footprint instead of a flat memory reservation (section 10).
