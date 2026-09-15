# Layered elimination cost on a real production GP teacher

Date: 2026-09-16. Seat 0328 of `B1024_N32_INDEPENDENT_20260910`.
Scripts: `superelement/factor_fit/gp_build.py`, `gp_tree_profile.py`, `gp_launch.py`.

**Provenance.** The system was rebuilt on a host registered for profiling, not the host that
produced the dataset. It is a cost measurement, not certified dataset evidence, and both
authorization flags in the profiling configuration were left false. No checked-in file was
modified.

## Why a rebuild was needed

The shipped packets carry only the condensed Schur complement (`S_UPPER.npy`, 12798 x 12798),
`TRACE.npz` and `PROBES.npz`. The coupling graph a nested-dissection tree would actually run on
was never saved. It was rebuilt from the packet's own `SAMPLE.json` geometry through the
production chain at the packet's source commit `6624dc86`, which is checked out under the ingest
tree:

```
contract -> topology -> Q2 body (K_upper) -> unit ghost penalty -> K = K_body + 1e-4 * K_ghost
```

**The rebuild is verified, not trusted.** Against what the packet ships:

| quantity | packet | rebuilt | identical |
|---|---:|---:|---|
| `background_nodes` | 77064 | 77064 | yes |
| `boundary` | 12798 | 12798 | yes |
| `box_boundary_original` | 12798 | 12798 | yes |
| body upper nnz (from the `_O` receipt) | 18570906 | 18570906 | yes |
| combined upper nnz (from the `_O` receipt) | 33268819 | 33268819 | yes |

## The production cell

| quantity | value |
|---|---:|
| total dofs | 231,192 |
| interior dofs | 218,394 |
| trace dofs | 12,798 |
| **trace fraction** | **5.54%** |
| active Q2 cells | 7,478 |
| ghost-penalty faces | 18,092 |
| body upper nnz | 18,570,906 |
| combined upper nnz | 33,268,819 |
| **new entries from the ghost penalty** | **14,697,913, a 79% increase** |

The trace fraction is 5.5%, not the 13% assumed earlier and far from the 53-80% of the small
no-GP bodies profiled before. The ghost penalty is not a perturbation of the sparsity: each face
contributes a dense 135-dof block over the 45 nodes of its two cells, coupling pairs that lie in
no single element, and that adds 79% more nonzeros than the physical stiffness has.

## Layered cost

Recursive bisection of the cell set on the widest centroid axis, leaf 8 cells. A ghost face is
internal to a subtree only when both its cells are; a straddling face keeps its dofs on the
boundary. Dense frontal model `e^3/3 + e^2 b + e b^2`.

| depth | nodes | separator range | boundary range | flops | share |
|---:|---:|---|---|---:|---:|
| 0 | 1 | 10734 | 12798 | 3.645e12 | **26.5%** |
| 1 | 2 | 5424 - 5430 | 12918 - 13038 | 2.699e12 | 19.6% |
| 2 | 4 | 2655 - 2757 | 9774 - 9882 | 1.355e12 | 9.8% |
| 3 | 8 | 4101 - 5001 | 6540 - 6630 | 2.869e12 | **20.8%** |
| 4 | 16 | 1509 - 2229 | 5418 - 6714 | 1.559e12 | 11.3% |
| 5 | 32 | 987 - 1818 | 3654 - 4962 | 1.111e12 | 8.1% |
| 6 | 64 | 165 - 915 | 2478 - 3666 | 4.206e11 | 3.1% |
| 7 | 128 | 0 - 423 | 1536 - 2301 | 8.895e10 | 0.6% |
| 8 | 256 | 0 - 162 | 948 - 1395 | 1.503e10 | 0.1% |
| 9 | 512 | 0 - 72 | 543 - 807 | 2.354e9 | 0.0% |
| 10 | 1024 | 0 - 36 | 327 - 525 | 1.669e8 | 0.0% |

Tree total 1.377e13 flops. Peak front 4.126 GiB, the root's `(10734 + 12798)^2`.

## What this changes

**The root does not dominate.** 26.5%, against 65.4% on the largest small no-GP body. The
earlier caveat that trace-dominated bodies overstate the root share is confirmed, and the
magnitude is now measured rather than asserted.

**The leaves are worth nothing.** Depths 8 through 10 together carry 0.1%. A network that
replaced leaf construction entirely would save 0.1% of the elimination work. The earlier
suggestion to start by learning the leaves is not supported by this measurement.

**The cost is spread across the top seven levels**, and the largest single band is depths 3
through 5 at 40.2%, more than the root. Depth 3 alone (20.8%) exceeds depth 2 (9.8%), because
geometric bisection produces irregular separators on a cut body; a quality nested-dissection
ordering would smooth that and is not what was measured here.

**So "which level should a network replace" has no single answer from this profile.** Nothing
below depth 6 is worth replacing. Everything from the root down to depth 5 carries real cost,
in roughly comparable amounts.

## Against what production actually did

The `_O` stage receipt records the real condensation: Intel oneMKL PARDISO with the Schur
complement formed natively inside the factorization.

| quantity | value |
|---|---:|
| PARDISO factorization | 31.85 s on 64 cores |
| full export, inclusive | 38.91 s |
| factor nnz upper bound | 2.672e10 |
| body assembly | 0.44 s |

The tree model's 1.377e13 flops would be about 4 s at 3.2 TFLOP/s of float64, so **the idealized
tree is roughly an order of magnitude optimistic against what a real sparse direct solver
achieves on the same system.** The 299x ratio against a monolithic dense elimination
(4.118e15 flops) is a ceiling comparison and says nothing about a speedup over production.

## Limits

- One seat. Separator sizes depend on the cut geometry and this is a single cut plane.
- Geometric bisection, not a quality nested-dissection ordering. The depth-3 anomaly is its
  fault, and a better ordering would lower the total.
- Symbolic flops and dense front sizes only. Per-level construction, factorization, storage and
  application timings were not measured; the one real timing here is production's own 31.85 s
  for the whole condensation.
- The rebuild is verified against the packet's node and trace sets and against two nnz counts
  from the receipt, which is strong but not a check of the matrix values themselves.
