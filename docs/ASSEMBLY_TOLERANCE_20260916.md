# What module accuracy the assembled structure needs — measured, not asserted

Date 2026-09-16.  Script `superelement/factor_fit/assemble.py`; raw `docs/data/ASSEMBLY.json`.

Two copies of seat 0328 glued along x on their 696 matching face nodes (the trace is 100% on outer
faces; the -x (y,z) set is a strict subset of the +x set, so 20 +x nodes are left free — a genuine
two-body bonded problem, not a claim about any particular lattice).  Both arms use `S = B^T A B`
with the same `B`, so every difference is attributable to `Ahat`.  Self-equilibrated face
tractions; rigid modes removed by projection.  Modules are the hierarchical factor approximation
at four tolerances, each with its own full-spectrum `eps_op`.

| module tol | module `eps_op` | params | compression: displ. / compliance | shear_y | shear_z | bending_xy |
|---:|---:|---:|---|---|---|---|
| 3e-2 | **0.152** | 8.58M | 4.09e-2 / 1.29e-3 | 2.33e-2 / 2.61e-3 | 2.43e-2 / 3.09e-3 | 1.77e-2 / 1.50e-3 |
| 1e-2 | 0.081 | 10.22M | 2.27e-2 / 3.30e-3 | 1.48e-2 / 1.81e-4 | 1.59e-2 / 1.21e-3 | 1.14e-2 / 4.67e-4 |
| 3e-3 | 0.026 | 12.23M | 9.41e-3 / 2.54e-3 | 6.34e-3 / 9.45e-4 | 7.47e-3 / 1.37e-3 | 4.69e-3 / 3.29e-4 |
| 1e-3 | 0.011 | 14.08M | 3.64e-3 / 1.04e-3 | 2.15e-3 / 3.74e-4 | 2.15e-3 / 2.46e-4 | 2.08e-3 / 5.0e-6 |

(relative errors against the exact-module assembly; 23,508 assembled trace dofs)

## Reading

- **Displacement error ~ eps_op / 4**, monotone: 4.1% at `eps_op` 0.15, 0.94% at 0.026, 0.36% at 0.011.
  This is the trustworthy column.
- **Compliance error ≤ 0.33% at every tolerance**, including a module that fails ±10% by 1.5x.
  Compliance is quadratic in the displacement and second-order in the operator error.  But the
  column is **not monotone** (1.3e-3, 3.3e-3, 2.5e-3, 1.0e-3 for compression), which says the
  construction itself has a floor near 1e-3 — the rigid-subspace regularisation `sigma N N^T` or
  the 20 unmatched nodes — so compliance differences below ~3e-3 are not resolving the module.
- For a compliance-level deliverable the module gate can be **~0.1–0.15 instead of 0.03**, which
  moves the hierarchical cost from 12.2M to 8.6M numbers and brings the 23k-parameter collar
  model (`eps_op` 0.15, 2026-09-13) into range.

## Not established

Sensitivities.  Topology optimisation uses `d(compliance)/d(design)`, not compliance alone, and
the derivative is first-order in the operator error.  The adjoint on this same strip is the
missing experiment and it is what should set the gate; until it runs, "0.15 is enough" is a
statement about compliance values only.  Also: one seat, a two-cell strip, four load cases.
