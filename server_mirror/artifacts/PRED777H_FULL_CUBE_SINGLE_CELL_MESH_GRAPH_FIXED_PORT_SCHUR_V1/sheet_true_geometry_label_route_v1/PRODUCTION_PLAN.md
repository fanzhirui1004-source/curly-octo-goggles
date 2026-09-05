# Production plan: sheet labels on 32-core / 64 GB servers (2026-09-04)

Label definition of record: the sheet route at commit 3e8c450 (root-cause mesher fixes cc11fa1, symmetric carrier
534c56b, one shared layout builder ff9d147, degenerate-cut status 288b1c2, topology-aware near-plane rule 3e8c450);
the queue and memory budget (a5587de, 5819d12) change how labels are produced, not what they are.  Labels produced before those commits are not comparable and are not
part of the production set.

## Per-label cost (production preset 1/64 + 0.02, carrier 1/32, Tet10, Pardiso; measured on the 64-core box)
| cell class | share of the population | wall (8 threads, 4 jobs contending) | peak RSS | label size (fp32 upper triangle + COO norms) |
|---|---|---|---|---|
| rho 0.1 (thin, scaled 1/94 + 0.0194) | rho < 0.15: ~12 % | 2.1 min | 13 GB | ~30 MB |
| rho 0.23 (G0-like) | rho 0.15-0.35: ~50 % | 3.2 min | 19 GB | 64 MB |
| rho 0.5 (thick) | rho > 0.35: ~38 % | 7.7 min | 32 GB | 200 MB |
| cut, small retained piece | subset of the cut cells | 0.5 min | 1.4 GB | 2 MB |

Memory, not CPU, sets the concurrency on a 64 GB machine: two labels at a time is safe for every class (2 x 32 GB
at the thick end leaves nothing, so pair a thick cell with a thin one, or run thick cells alone); three at a time is
safe only when no thick cell is in flight.  Simplest robust policy: a memory-reservation queue (each job reserves
35 MB per 1000 fine Tet10 dof, known from the mesh before the Schur starts; the production script can print the
estimate after meshing and wait for headroom).

## Measured throughput with the memory-budgeted queue (64 cores, 128 GiB cgroup, 2026-09-04)
`run_sheet_label_queue.py --jobs 14 --threads 8 --budget-gb 110` on 30 random population cells (25 thin/medium,
5 thick, 20 cut): 30 labels in 16.0 min including the ramp and the drain tail, steady state 165 to 185 labels per
hour, cgroup memory at most 77 GB of 128, no failures.  Per class: thin/medium wall median 203 s (Schur 105 s),
thick median 249 s and up to 961 s (Schur up to 273 s); budget waits median 9 s, up to 555 s for a thick cell
behind other thick cells.  Jobs in the Schur stage use 3 to 4 cores each whatever the thread count (Pardiso at
this size), the other stages one core: the box runs 25 to 30 cores busy, bounded by the memory budget (about six
Schur stages in flight at 18 GB each), not by threads.  A second batch of 30 (8 thick, mean 109k tetrahedra) with `--jobs 16 --threads 12` took 19.1 min: the Schur time per
unit of work is the same at 8 and 12 threads (104 vs 105 s per 100k tetrahedra), the budget waits grew (median
142 s), so threads are not the lever; the memory reservation was 35 % above the measured peak (33 MB per 1000
Tet10 dof) and has been set to 36 MB (commit 5819d12), which admits about a quarter more Schur stages.
Recommended setting on this box: `--jobs 16 --threads 8 --budget-gb 115`.

## Throughput estimate per 32-core / 64 GB server
With the memory-budgeted queue the concurrency is set by the cores and the memory only bounds the Schur stages:
on a 32-core / 64 GB box, `--jobs 8 --threads 8 --budget-gb 52` keeps about three Schur stages in flight (two when a
thick cell is among them) and the surface/port stages of the other jobs on the remaining cores; scaled from the
64-core / 128 GiB measurement this is 70 to 90 labels per hour per server.

| population | labels | one server | four servers | storage |
|---|---|---|---|---|
| pilot 300 | 300 | ~4 h | ~1 h | 25 GB |
| main 2000 | 2009 | ~25 h | ~7 h | 160 GB |
| main 2000 on the 64-core / 128 GiB box alone | 2009 | ~12 h | | 160 GB |

## Operating rules
- one process per label (`produce_sheet_label.py`), create-only output directory, receipt with SHA-256, git head,
  the route contract and the execution settings; a scheduler script per server pulls the next case id from the
  population list and skips existing directories (idempotent restarts);
- memory sets the concurrency (a 128 GiB cgroup: five thin/medium labels or three thick ones at a time), the
  thread count per job then fills the cores (cores / concurrency, e.g. 12 or 20 on 64 cores): OMP_NUM_THREADS =
  MKL_NUM_THREADS = --threads = --workers.  The surface stage is single-threaded whatever the setting; Pardiso, the
  stiffness assembly and Gmsh use the threads.  Do not oversubscribe: the control plane of a box starves above
  ~1.2 x core count;
- statuses to expect and count: PASS, EMPTY (about 6 in 100 cut cells: offsets that leave no band), SURFACE_FAIL /
  SURFACE_BAD / MESH_FAIL (0 in 2009 at commit 3e8c450; the only refusal is a cut through a cell edge, GEOMETRY_DEGENERATE), each with a receipt;
  failed cases are re-queued once at the reference preset, then reported with their diagnosis (no case is dropped
  from the population by a rule);
- determinism: geometry and mesh byte-identical across machines; the operator agrees to 2e-15 across thread counts
  (MKL summation order); MKL_CBWR=COMPATIBLE is not needed for training but can be set for byte identity;
- rotation augmentation is applied at training time from the stored canonical labels (no rotated copies stored):
  with the symmetric carrier the action T S T^T is exact on a fixed mesh (7.5e-15) and the label-to-label residual
  between a cell and its image is 0.8 %, i.e. the mesh convergence level, so no rotated copies need labelling.
