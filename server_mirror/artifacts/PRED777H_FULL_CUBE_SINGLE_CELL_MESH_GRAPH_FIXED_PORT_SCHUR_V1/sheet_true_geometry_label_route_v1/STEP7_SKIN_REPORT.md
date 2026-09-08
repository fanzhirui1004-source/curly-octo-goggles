# Step 7: block with a bonded skin, end-to-end proof of the coupling contract

Sheet route, 2026-09-05.  `scripts/pred777h_full_cube_v1/sheet_skin_validation.py`.

The contract of record: a skin bonded to the cut face couples to the cells through the cut-face carrier, so the
block-to-skin joint traction lives in the carrier P1 space.  This is the one item that could still have invalidated
the route, because every earlier assembly figure was measured on INTERNAL interfaces, where both sides use the same
carrier by construction and agreement is not evidence.

## 1. The experiment

A 2 x 1 x 1 block of sheet cells cut by the world plane 3x + 10y <= 9, with a slab of thickness 0.05 extruded from the
cut plane on the discarded side and meshed as Tet10 (one prism layer per joint triangle, split into three tets by the
global-id rule, so the split is conforming).  The block is fixed at x = 0, the external box faces are tied to the
carrier, and the load is a nodal traction on the skin's outer face: the physical path, skin into core.

Three models of the same geometry, the same block mesh, the same skin mesh and the same loads.  They differ only in
the space the joint traction lives in:

| arm | the block's fine cut-cap nodes follow | the skin |
|---|---|---|
| REF | the skin's own P1 field at their positions (full-fidelity glue) | all its own degrees of freedom |
| CON | the CARRIER P1 field, whose nodes are the skin's own nodes at those positions | all its own degrees of freedom |
| ASM | nothing: the cells are their per-cell Schur operators on the carrier | all its own degrees of freedom |

Midpoint refinement keeps the carrier nodes as an exact subset of the skin's joint nodes, so a carrier node and its
skin node are the same unknown and the coupling is exact and conforming.  Neither body is coarsened in CON: only the
joint coupling is restricted.  `CON vs REF` is therefore the cost of the contract, and `ASM vs CON` is the assembly
error.

A fourth arm, `--skin-on-carrier`, meshes the SKIN itself on the carrier.  That measures the skin's own discretization,
not the coupling, and is reported separately below.

An earlier version of this script tied the skin's joint nodes to the carrier, which coarsens the skin to 1/32 as well;
it reported a 21 % to 42 % "contract cost" that was almost entirely the skin's own under-resolution.  The arms above
separate the two.

## 2. Results

Fast preset (block remesh 0.03, skin refine 1, joint spacing 0.0156) and production preset (block remesh 0.02, skin
refine 2, joint spacing 0.0078).  Wavelengths are in carrier spacings H = 1/32.

| load | CON vs REF, fast | CON vs REF, production | ASM vs CON, fast | ASM vs CON, production |
|---|---|---|---|---|
| uniform normal pull | +0.17 % | +0.36 % | −2.3e-5 | +3.2e-4 |
| uniform in-plane shear | +0.25 % | +0.51 % | −7.7e-5 | +1.9e-4 |
| linear bending | +0.24 % | +0.47 % | −3.7e-5 | +2.2e-4 |
| sine, 16 H | +1.13 % | +2.34 % | +1.0e-3 | +7.3e-4 |
| sine, 8 H | +2.35 % | +4.49 % | +2.2e-3 | +1.8e-3 |
| sine, 4 H | +1.95 % | +3.34 % | +4.7e-4 | +1.4e-3 |

Relative L2 difference of the displacement on the skin's outer face: CON vs REF 0.16 % to 1.5 % (fast), 0.34 % to
3.1 % (production); ASM vs CON 2.2e-4 to 3.0e-3.

Sizes: block 254 988 to 556 803 Tet10 dof, skin 381 321 to 1 518 345 Tet10 dof, carrier union 4825 nodes of which 2721
on the joint (1709 of those inactive: the skin spans the voids, which is the point).  The three systems factorize in
8 s to 34 s.

**Skin meshed on the carrier instead of refined** (fast preset, side arm): compliance −1.35 % (pull), −3.50 % (shear),
−2.40 % (bending), −1.55 % (16 H), −2.76 % (8 H), −3.83 % (4 H) against REF, i.e. the skin at 1/32 is 1.4 % to 3.8 %
too stiff.  Acceptable, but it is a real cost and the remedy is free: midpoint-refine the delivered joint mesh, which
keeps the carrier nodes as a subset and so keeps the coupling exact.

## 3. What this settles

* The coupling contract is sound.  Restricting the block-to-skin joint traction to the carrier P1 space costs about
  0.4 % on engineering-smooth loading and 2 % to 4.5 % on sinusoidal port loading down to four carrier spacings.  The
  sign is positive throughout, as a Rayleigh-Ritz restriction must be.
* The assembly of the per-cell labels reproduces the contract's monolithic solution to 2e-4 on smooth loads and 2e-3 on
  the oscillatory ones, with a skin attached and with 1709 of the joint carrier nodes carrying no cell stiffness at
  all.  This is the first assembly figure measured on an EXTERNAL loaded face.
* The delivered `CUT_FACE_TRACE.npz` mesh is usable as the skin's joint mesh directly, and refining it by midpoint
  keeps the coupling exact.

## 4. What it does not settle

The contract cost roughly doubles between the two presets (0.17 % to 0.36 % on pull), and the reference itself moves
1.2 % between them, so neither is converged: these are lower bounds, and a third resolution is needed to establish the
asymptotic value.  That belongs to the fine-mesh convergence study.

The skin here is an isotropic slab of the same modulus as the cells, one element through the thickness, bonded over
the whole cut plane.  A real skin (different modulus, a laminate, a bending-dominated thin plate) is not covered; the
`--skin-modulus` and `--skin-thickness` arguments exist for that sweep and it has not been run.
