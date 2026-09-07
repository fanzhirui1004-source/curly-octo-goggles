# CGAL convergence reference for the sheet fixed-port label route

Task brief, 2026-09-07.  Written for an agent working on the AutoDL box in
`/root/autodl-tmp/cut_control_tpms_v1_full_cube_single_cell_v1` (branch `codex/pred777h-full-cube-single-cell-v1`).

## 1. What this is for, and what it is not

Our production labels come from ONE discretisation path: an exact-geometry surface built to conform to the carrier,
remeshed, then Gmsh HXT for the volume, then a Tet10 fixed-port Schur complement.  Every convergence number we have
(mesh tiers, carrier tiers, reproducibility) compares that path against ITSELF at a finer setting.  That cannot catch
an error shared by every tier, and this route has already produced one such error (STEP8: a needle cap triangle from
a forced carrier support point made lambda_max up to 11x its converged value in a quarter of the labels, invisible to
every guard, and identical in kind at every resolution).

The job here is an INDEPENDENT reference: CGAL Mesh_3 on the same exact geometry, sharing no surface code, no
remesher, no volume mesher and no carrier, computing a quantity both routes can produce, so that agreement is
evidence about the PHYSICS rather than about the implementation.

This is not a replacement teacher, not a new label format, and not a port-space study.  Do not change anything under
`src/pred777h_full_cube_single_cell_mesh_graph_fixed_port_schur_v1/` or `scripts/pred777h_full_cube_v1/` other than
adding new files under a new directory.

## 2. The quantity of record

The two routes do not share a port space, so their operators cannot be compared entry by entry.  What they can both
produce exactly is the **six uniform-strain apparent stiffnesses** (kinematic uniform boundary conditions on the
outer faces):

    C[i] = u_i' S u_i,   u_i = the affine displacement of the i-th unit strain, i in (xx, yy, zz, yz, xz, xy),

with the affine field imposed on the OUTER PORT FACES ONLY (the six box planes, plus the cut plane when the cell has
one) and the TPMS free surface left free, E = 1, nu = 0.3, Tet10 elements.  Affine fields lie exactly in every P1
carrier space, so on our side this is `u' S u` with S the shipped operator; on the CGAL side it is the same energy
computed by constraining the port-face boundary nodes to `u = eps x` and solving for everything else.  No carrier is
involved on the CGAL side at all.

Also report the material volume, but read it for what it is on this route.  Because the CGAL side is fed the label's
OWN surface, the volume difference does NOT cross-validate the two surface constructions; it measures only how far
Mesh_3's remeshing of that surface has drifted from it, i.e. it is the `facet_distance` error.  That still makes it
the right gate -- a volume difference that does not fall under refinement means the meshing is not converging and
nothing downstream is meaningful -- but do not report it as agreement on the geometry.

## 3. The geometry contract, and which CGAL program to use

Material is the Schwarz-P sheet `|phi(x)| <= tau(x)`, `phi = sum_k cos(2 pi x_k)`, `tau` trilinear from the cell's
eight corner values, clipped by the six box planes and, when present, one world cut plane (retained side
`n . x <= d`).

There are two CGAL programs and only one of them is right here.

**Do not use `pred777h_cgal_mesh3`** (the implicit labelled domain).  Its band is

    collar_level = min over active faces of (distance to that face - collar)
    band         = max( min(phi - tau, collar_level), min(-phi - tau, collar_level) )

so with its default `--collar 0.03125` everything within one carrier spacing of every port face is material: a solid
shell lining the cell.  That was deliberate in the old route, where it guaranteed every carrier node had support, and
it is not the geometry our labels use, whose contract is `SHEET_SOLID_NO_COLLAR` (the route dropped the collars at
commits 739bbcd and 3c6a76c).  Working around it by passing a tiny collar is possible but pointless, because:

**Use `pred777h_cgal_polyhedral_mesh3`.**  It contains no collar logic at all: it takes a watertight surface with
`--surface-off` and meshes its interior.  Feed it the label's own `SHEET_SOLID_SURFACE.off`, which is already the
exact collar-free geometry.  Verified working on 2026-09-07 with no code change:

    build/pred777h_cgal_mesh3/pred777h_cgal_polyhedral_mesh3 --case-id pop_cut_0473 \
      --surface-off <label dir>/SHEET_SOLID_SURFACE.off --output-prefix <out> \
      --input-off-sha256 <sha> --source-surface-sha256 <sha> \
      --facet-size 0.05 --facet-distance 0.004 --cell-size 0.06 --edge-size 0.008 --random-seed 777

giving 10 507 vertices, 29 655 tetrahedra, volume 0.050599 against the label's 0.050475, a relative difference of
2.5e-3.  Two constraints the program enforces: `edge_size <= 2 * facet_distance` (a feature-protection stability
contract, it refuses otherwise), and the surface is REMESHED, not preserved (19 410 facets in the complex from 9 168
input triangles).  The remeshing is why this cannot produce labels, since the carrier nodes do not survive it, and is
exactly why it is a valid independent reference: only the analytic geometry is shared, not the discretisation.

`facet_distance` is therefore the geometry-error knob, and the volume difference against the label is the direct
measure of it: 2.5e-3 at 0.004 must fall as it is refined, and if it does not, stop and find out why before looking
at any stiffness.

Note on slivers: the wrapper reported `optimization_enabled: False` on that run and the raw mesh had a minimum
dihedral angle of 0.000 degrees with 216 tetrahedra below 5 degrees, against 2.736 degrees and 27 for our Gmsh mesh
of the same cell.  The program has `--sliver-bound-degrees` (default 10) and `--disable-sliver-exude`; run WITH the
exude for the reference, and report the before/after quality, because an unexuded CGAL mesh is a worse discretisation
than the one it is supposed to check.

## 4. What already exists

| thing | where |
|---|---|
| the program to use (built) | `build/pred777h_cgal_mesh3/pred777h_cgal_polyhedral_mesh3`; source `cpp/pred777h_cgal_mesh3/pred777h_cgal_polyhedral_mesh3.cpp` |
| its arguments | `--case-id --surface-off --output-prefix --input-off-sha256 --source-surface-sha256 --sharp-angle --facet-angle --facet-size --facet-distance --cell-radius-edge-ratio --cell-size --edge-size --sliver-bound-degrees --disable-sliver-exude --random-seed` |
| its output | `<output-prefix>.mesh` (MEDIT) and `<output-prefix>.wrapper.json` (criteria, input audit, facets in complex, seed) |
| the collared program, NOT to be used | `build/pred777h_cgal_mesh3/pred777h_cgal_mesh3` (implicit labelled domain, `--collar` defaults to 1/32) |
| the collar-free surface to feed it | `<label dir>/SHEET_SOLID_SURFACE.off`, written by every produced label |
| MEDIT to MeshArtifact | `src/.../cgal_mesh3_adapter.py` (`mesh_artifact_from_cgal_medit`, `audit_cgal_mesh_artifact`) |
| Tet10 assembly | `from cctpms.fem.tet10 import assemble_global_tet10_stiffness` (E, nu, workers) |
| Tet4 to Tet10 promotion | `scripts/pred777h_full_cube_v1/sheet_skin_validation.py`, function `tet4_to_tet10` |
| linear solver | `pypardiso` (`PyPardisoSolver`, `mtype=-2` and the upper triangle for a symmetric system) |
| geometry manifests | `/root/autodl-tmp/_claude_diag/population/cells/<case>/geometry_material_manifest.json` |
| our labels | `/root/autodl-tmp/_claude_diag/production/labels/<case>/` |
| the comparison targets | `CGAL_REFERENCE_TARGETS.json` next to this brief: per cell, the exact `--tau-corners` and `--cut` strings and our six stiffnesses |

## 5. What to run

Six cells, three uncut and three cut, spanning volume 0.010 to 0.423, are listed in the targets file.  For each, a
Mesh_3 resolution ladder of at least four levels, refining `facet-size`, `facet-distance` and `cell-size` together by
a constant factor (a factor of 2 in `facet-distance` per level is a reasonable start; the sheet is thin, so
`facet-distance` is what actually controls whether the geometry is resolved).  Record for every level: the criteria,
the tetrahedron count, the material volume, the minimum dihedral angle, the six stiffnesses, and the wall time.

Then, per cell: the observed convergence order of each stiffness against the finest level, and the relative
difference between the CGAL-extrapolated value and our label's value.

## 6. What counts as a result

* The material volume converges to the label's volume as `facet_distance` is refined (2.5e-3 at 0.004 is the measured
  starting point).  This is a meshing-convergence gate, not a geometry cross-check -- see section 2.  If it does not
  fall under refinement, stop: nothing downstream is meaningful.
* The six stiffnesses converge as the CGAL mesh refines, and the CGAL limit agrees with our label.  Our own mesh-tier
  study puts our production preset within 0.05 % to 0.14 % of our reference preset on these same six numbers, so the
  interesting threshold is about 1 %: agreement at that level is a genuine independent confirmation, and a systematic
  disagreement of several percent with a consistent sign is a finding worth chasing.
* A disagreement must be attributed before it is reported: check the volume first, then the boundary-node
  classification (which faces got the affine condition), then the element order, then the resolution.

## 7. Pitfalls this machine has already taught us

* **Memory.**  The cgroup is 128 GiB.  A production run of the label route may still be occupying it; check
  `/root/autodl-tmp/_claude_diag/production/DONE.txt` for `PRODUCTION_DONE` and `SWEEP_DONE` before taking the
  machine, or run at low concurrency.  Watch `anon` in `/sys/fs/cgroup/memory.stat`, NOT `memory.current`: the latter
  counts reclaimable page cache and guarding on it kills jobs needlessly (we did this 250 times).
* **Solver.**  `scipy` `splu` on a million-dof system took the machine down twice.  Use Pardiso, symmetric mode, and
  release each factorisation before building the next.
* **Determinism.**  Mesh_3 takes `--random-seed`; fix it and record it.  Report whether two runs at the same seed and
  the same criteria give the same mesh.
* **Do not reuse the old collared artefacts** under
  `artifacts/.../mesher_teacher_representation_route_v1/` as a reference: they are a different geometry.
* **An independent SURFACE is a second, harder question, and the obvious shortcut is a trap.**  Feeding the label's
  own OFF shares the surface construction, so this reference isolates the volume mesher, the element and the solve --
  which is exactly where the STEP8 error lived, so it is the right first target.  The tempting next step is to make
  `pred777h_cgal_mesh3` (the implicit labelled domain) collar-free and get a surface CGAL builds itself.  Do not do
  that, for two reasons found by reading it on 2026-09-07:

  1. Its port-face identification is load-bearing on the collar.  A void point inside the box is labelled by the
     NEAREST branch of the material union, so with a fat collar the void adjacent to a box face is material and the
     `(1, box_*)` surface patch occurs only where the box plane is actually crossed.  Set the collar to zero and that
     nearest-branch test degenerates into a bisector: genuine TPMS free surface that happens to lie near a box face
     acquires a `(1, box_*)` patch and is misclassified as an outer port.  That is a rewrite of the labelling, not a
     parameter change.
  2. It does not detect features.  It REQUIRES a complete externally supplied polyline network via
     `--feature-polylines` (V2 schema, with per-polyline patch incidences) and refuses partial ones by design.  With
     collars those curves are plane-plane intersections, i.e. straight segments.  Without collars they become
     `{|phi| = tau}` intersected with each box plane: curved, implicitly defined, with topology that changes with tau
     and with the cut.  The only thing on this machine that can produce them is our own surface builder -- so paying
     for the independent surface with that network would hand the coupling straight back.

  `pred777h_cgal_polyhedral_mesh3` has neither problem: it detects sharp features by angle
  (`exact_outer_feature_segments`, and it throws if no sharp outer-port feature is found).  So the right way to get an
  independent surface later is to KEEP this program and swap only its input OFF for an independently built surface (a
  different sampler on a different grid, or a collar-free mode for `implicit_polyhedral_surface.py`).  Holding
  everything downstream fixed also ATTRIBUTES a disagreement to the surface instead of merely detecting one.  Either
  way it is a follow-up, not part of this task.

## 8. Where to put the results

A new directory `artifacts/PRED777H_FULL_CUBE_SINGLE_CELL_MESH_GRAPH_FIXED_PORT_SCHUR_V1/cgal_convergence_reference_v1/`
with one JSON per (cell, level), a summary JSON, and a report that states the comparison, the attribution of any
disagreement, and what the reference does and does not certify.  Code under
`scripts/pred777h_full_cube_v1/cgal_reference/`.
