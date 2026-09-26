# What our work claims (for prior-art / competitor screening)

Setting: gradient-thickness TPMS (Schwarz-P sheet) lattices for compliance topology optimisation. Each unit cell has a
trilinear thickness field (8 corner values); cells at the design boundary are CUT by planes (arbitrary retained volume
1%..100%). Discretisation: CutFEM on a common 65^3 Q2 background grid with ghost penalty, so neighbouring cells'
interface DOFs match by construction.

Claims (candidate contributions):
C1  Learned static condensation of NON-periodic, CUT TPMS cells: a geometry-conditioned neural operator that returns the
    boundary Schur complement (condensed stiffness / superelement operator) S of a cell for ANY boundary displacement,
    so cells can be assembled into lattices with arbitrary neighbours (no periodic BC, no homogenisation / scale
    separation assumption).
C2  Variational construction: the network outputs a LINEAR extension E_hat(q) of boundary displacements into the cell
    interior; the operator is S_hat = E_hat^T K E_hat with the exact CutFEM stiffness K. This makes S_hat symmetric,
    positive semi-definite, rigid-body exact, and one-sided (S_hat >= S, error = ||E_hat - E||_K^2, second order).
    Label-free energy-minimisation loss (minimum potential energy), normalised per direction.
C3  Design sensitivities: d(compliance)/d(corner thickness) through the learned field (envelope/adjoint formula),
    accuracy gate on lattice compliance AND 8-corner sensitivities (<= 3%) with continuous-thickness neighbours.
C4  Hybrid "two-grid" physics wrapper: network output -> k Chebyshev sweeps of exact interior equilibrium -> Galerkin
    coarse correction on a small Q1 coarse space (17^3) -> k sweeps; trained end-to-end. Network provides the global
    shape for complex geometry, the exact operator enforces local equilibrium / soft-mode amplitudes. Diagnostic
    finding: pure feed-forward networks are limited by a delta^2 * kappa law (thin walls, 100-3000x soft/stiff
    contrast) -- energy error of soft modes amplified by stiff error components.
C5  Engineering: fully learned multigrid-style network (hyperedge layers on active elements + U-Net over voxel levels),
    O_h equivariance by augmentation, lattice-level preconditioner (BNN with rigid coarse space) for assembling
    learned superelements.

Seed prior-art (found earlier, DOIs): 10.1007/s00158-026-04404-9; 10.1007/s10409-025-25942-x; 10.1016/j.cma.2026.118955;
10.1016/j.compstruct.2025.119330; 10.1016/j.compstruct.2026.120865; 10.1016/j.eml.2023.102041; 10.1016/j.eml.2024.102237;
10.1016/j.ijmecsci.2026.112007; 10.1016/j.jmps.2024.105893. Local PDFs: ../Xu2025-PIML-lattice-MMC.pdf,
../Zhang2024-isoparametric-PIML.pdf. Chinese patents to check: CN119249797B, CN119623208A, CN119918215A.

Grading rule (be calibrated, NOT alarmist):
- HIGH: the same object is learned for the same purpose (e.g. a published NN that outputs condensed/superelement
  stiffness of non-periodic or cut lattice cells for assembly, or a published NN+exact-smoother two-grid wrapper for
  condensed operators). Would force us to change a claim.
- MEDIUM: shares one key ingredient AND the application (e.g. learned superelements for periodic lattices; learned
  coarse spaces for DD in elasticity). Must be cited and differentiated explicitly.
- LOW: related background (homogenisation-based ML for TPMS, generic neural operators, generic learned solvers).
  Cite as context.
- NONE: superficial keyword overlap. Do not list unless commonly confused.
For every item give: full citation (authors, title, venue, year, DOI/URL), what exactly they learn and how it is used,
overlap with C1..C5 (which ones), grade + one-line reason, and the differentiator sentence we would write.
