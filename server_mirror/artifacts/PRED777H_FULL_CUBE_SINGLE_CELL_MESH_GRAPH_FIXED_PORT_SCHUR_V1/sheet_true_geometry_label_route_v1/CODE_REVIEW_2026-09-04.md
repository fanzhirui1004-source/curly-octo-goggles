# Code review of the sheet label route, 2026-09-04

Scope: `sheet_solid_surface.py`, `tet10_label.py`, `produce_sheet_label.py`, `run_sheet_solid_prototype.py`,
`sheet_block_validation.py`, the tests.  Goal: errors, duplication, production hygiene.  Every change below leaves
the produced surfaces and operators byte-identical (checked on the five convergence cells and on a production G0 label).

## Findings and what was done
| # | finding | action |
|---|---|---|
| 1 | Schur solved for every layout carrier column although a third of them have no fine support (their coupling column is zero) | skip unsupported columns in the Pardiso solve; operator identical to the bit, 75 s -> 33 s on G0 fast (Step 4) |
| 2 | Eight functions and the whole marching-cubes-only path (no remesh) were dead after Step 4: `resnap_boundary_to_planes`, `restore_plane_corners`, `flip_flat_triangles`, `_even_odd_inside`, `collapse_short_boundary_edges`, `handle_flat_triangles`, `snap_near_plane`, `split_line_t_junctions` (subsumed by `split_all_t_junctions`), plus the `smooth=True` branch and the `project`/`snap_fraction`/`flat_mode` switches | removed; `remesh_size` is now a required, defaulted parameter; module 1634 -> 1428 lines |
| 3 | `orient_outward` flipped the whole surface by the total signed volume; with several closed shells (a cut can leave separate pieces, 5 of 342 stress cells) one inside-out shell could pass unnoticed | every shell is flipped by its own volume sign |
| 4 | The port-and-Schur block (adapter, fixed port, geometric active set, superset check, Schur, rigid/eigenvalue gates) was written twice (production command and prototype driver) with slightly different reporting | new `sheet_label_pipeline.py` (`compile_port`, `schur_label`, `carrier_norms`) used by both; the prototype's operator equals the panel's to the bit |
| 5 | The route had no frozen contract of its own; `contract.FullCubeContract` still describes the V1 collar route (collar on, 1/16 grid) and is used by that route's population/publication code, so it must not be edited | new `sheet_contract.SheetSolidContract` (no collar, carrier 1/32, presets, thickness scaling, gates, solver); every production receipt carries it |
| 6 | Gmsh ran in the production process: a Gmsh segmentation fault killed the worker (seen once in the stress population) | child process with one HXT retry (Step 4) |
| 7 | 2-D `np.cross` in the exact triangle test is deprecated in NumPy 2 | scalar helper |
| 8 | `produce_sheet_label.py` let a `RuntimeError` from Gmsh propagate as a traceback | reported as `MESH_FAIL` with the log tail |
| 9 | The block-validation script passed a removed keyword | updated; it builds through the same `_remesh_and_clip` path as production |

## Looked at and left alone
- `compile_fixed_port` point location lives in the vendored `cctpms` prolongation builder (about 10 % of a label);
  not touched.
- `dense_fixed_port_schur` still builds the full-layout dense Schur (18438^2 doubles, 2.7 GB on a tau = 0.4 cell)
  before slicing to the active block; the peak memory of a production label is dominated by the Pardiso factors
  (13-19 GB, see the production timing table), so this was not restructured.
- `smooth_on_polyhedron`, `carrier_shell` (conforming_port_mesh.py) stay for the V1 conforming route and the block
  validation.

## Checks after the review changes
- 15 unit tests pass (surface 9, tet10 6).
- Surfaces of G0/G1/G3/G4/G5 at three resolutions rebuilt with the reviewed module: md5 identical to the committed
  panel.
- Prototype driver after the refactor: operator identical to the panel's (max |dS| = 0).
- Production command after the refactor: operator identical to the pre-refactor production label (see below).
- Production command before vs after the refactor (the pre-review script from b462307 with only the removed keyword
  dropped, same module, run one after the other on an idle machine): every array of `FIXED_PORT_SCHUR_TET10.npz`
  identical to the bit (the 2e-15 seen between two contended runs is MKL thread scheduling, not code).
