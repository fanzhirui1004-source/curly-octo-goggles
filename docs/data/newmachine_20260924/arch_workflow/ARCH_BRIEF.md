# Architecture brief: a learned solution operator for cut TPMS cells (CutFEM + ghost penalty)

This brief is the complete context for designing the first version (v0) of the neural architecture.
The project owner wants a design derived from OUR requirements, OUR data and OUR experimental evidence,
grounded in mechanics, numerical analysis and operator-learning mechanisms. Borrowing modules from the
literature is fine; copying a paradigm wholesale is not the goal (other groups solve different problems).
Repository docs (Chinese) with full evidence: /home/user/curly-octo-goggles/docs/
 - OPERATOR_LEARNING_DESIGN_20260923_CN.md  (agreed decisions, sampling plan)
 - ARCHITECTURE_HISTORY_20260923_CN.md      (12 days, ~10 architectures, why each was dropped)
 - ROUTES_PROGRESS_20260923_CN.md           (exact pipeline numbers, route 1 skeleton depth study, BDD lattice counts)
Pipeline code (read-only reference): /tmp/claude-0/-home-user-curly-octo-goggles/76f53c89-e2a9-52bb-8817-9460fd780b76/scratchpad/xcase/r7/
 (design_encoder.py, schur_encoder.py, lattice_pcg.py, precision_study.py, fast_prep3.py, superset_encoder.py)

## 1. The goal

"Encode the geometry once, get a reusable structured operator; then for ANY boundary displacement of the cell,
quickly return the complete boundary reaction."  Cells are then assembled into lattices (~100 cells, number and
FULL/cut mix not fixed; designed by us) for thickness (tau) optimization: one global solve per design iteration,
compliance is self-adjoint, sensitivity dC/dtau = -sum_cells u^T (dK/dtau) u.

Main acceptance gate: lattice compliance AND sensitivity errors <= 3% (vs exact). Worst-direction mu (single-cell
whitened spectrum) is reported, not gated. Cut-surface-loaded cases: computed and reported, not gated (step 1).

## 2. Decisions already made with the owner (do not relitigate)

1. Learn the SOLUTION operator (harmonic/elastic extension), not the matrix. Network input: geometry g and port
   displacement q; output: the displacement field u on the cell interior (background grid). Reaction is read out
   variationally with the EXACT stiffness: S_hat = E_hat^T K E_hat.
2. Boundary at FULL resolution, no compression (no Bezier/corner-DOF reduction, no "one shape function per boundary
   DOF" paradigm of Guo Xu's PIML line). Cost should be ~independent of the number of port DOFs.
3. Everything on the FIXED background grid. This is why CutFEM was chosen: geometry changes only change masks and
   coefficients, never the coordinate system. (Verified: 15 rotated cases have identical node sets/ordering;
   pipeline is cube-group equivariant to 1e-7.)
4. Cut surface is a real load-carrying port: ports = box-face nodes UNION cut-band background nodes, all on the fixed
   grid, marked by masks. No trace coordinates that move with the plane. Load capability guaranteed structurally
   (forces on the cut band as nodal forces). Staged: step 1 may train mostly without cut loading, but architecture,
   data and harness support it from day one. A structural skin (future work) will attach via the cut-band port.
5. Strictly LINEAR in q. q passes only through linear operations; nonlinearity only in the geometry path, which
   generates the coefficients of those linear operations (a geometry-conditioned linear operator / hypernetwork).
6. Exact rigid modes. Cubic symmetry (48-element group) by augmentation or equivariant construction.
7. Main loss: energy (minimum potential energy) -> label-free possible; exact solutions (cheap) as auxiliary
   supervision and for validation. Loss normalized per sample by exact energy q^T S q (= mu-1 in that direction).
8. Main line (owner's choice): FEEDFORWARD neural operator combined with MULTISCALE GRAPH neural operator (GNO),
   "take the best of both", architecture custom-designed ("we will surely have to modify the architecture").
   Fallback A: learned iteration (each layer computes exact residual r = f - K u, geometry-conditioned linear
   net gives the correction). Fallback B: route-1 skeleton (paired smoother, coarse spaces, Chebyshev) with only a
   few geometry-generated coefficients. Only if the main line fails.
9. Sampling agreed (see design doc sec. 9): q mix = force-driven 30% (q = S^-1 f, smooth multiscale forces on box,
   ~10% with cut-band forces) / macro deformation 20% (rigid, 6 uniform strains, quadratic, cubic) / multiscale
   Gaussian random fields on ports 30% / adversarial worst directions via power iteration 20% after warm-up.
   Geometry: reuse Codex's generator (families.py), oversample thin walls, heavy cuts, planes grazing walls.
   Step 1: three fixed geometries (medium cut, heavy cut, their shared FULL parent). Step 2: geometry generalization
   (300/50/50 start, learning curve 100/300/1000). Step 3: lattice closed loop with BDD.

## 3. Discretization and data facts

- Background grid: n = 32 -> 32^3 hexahedral Q2 elements (27 nodes each), 65^3 = 274,625 Q2 nodes, 3 DOF/node.
  E = 1, nu = 0.3.
- Material: Schwarz-P TPMS shell {x : |phi(x)| <= tau(x)}, phi = cos(2 pi x)+cos(2 pi y)+cos(2 pi z) (cell = unit
  cube), tau(x) trilinear from 8 corner values in [0.1752, 0.6993] (span/gradient <= 0.47), intersected with a
  half-space (cut plane: normal, offset; parameterized by retained MACRO box volume and angle theta).
  Material set grows monotonically with each corner tau.
- CutFEM: active elements = background elements touching material; element stiffness integrated only over the
  material part via element moments: per element 125 monomial moments (x^a y^b z^c, a,b,c <= 4) of the cut
  volume; stiffness upper triangle = moments (125) x fixed template (125 x 3321) -> linear in moments.
  Ghost penalty on faces between active elements where not both are full: 135x135 face template, gamma = 1e-4.
  K = K_body + gamma K_ghost, SPD after removing rigid modes (floating cell).
- Per geometry the encoder computes moments in ~0.25-1 s on GPU; topology/masks on CPU ~1 s.
- Sizes (examples):
    case 0013 (small cut, retained 0.148): interior 62,961 DOF, box port 4,794 DOF
    case 0021 (large cut, retained 0.758): interior 197,658 DOF, box port 12,363 DOF
    FULL parent: interior 344,628 DOF, box port 23,616 DOF (~123k active nodes of 274,625)
  Box port = Q2 face nodes on box-face patches with positive-area material (3x3 face nodes per patch).
  Cut-band port = background nodes of elements intersected by the cut plane (definition to be fixed in step 0).
- Topology (active elements, GP faces, box-port nodes) changes under tiny tau changes (1e-3 relative changes
  active elements by +-4-6 in cut cells). Sensitivity is only defined within a topology; masks must be inputs.
- Exact pipeline (route 7): cuDSS sparse direct; factorization 0.08-13 s depending on precision/size; after that
  each solve is milliseconds. Schur mode gives dense box operator T (fp64 needed; FULL T = 4.5 GB).
  Exact solution fields u = E q for training are therefore cheap once a geometry is factorized.
- Hardware: RTX 5090 32 GB (fp64 1/64 of fp32), 16 CPU cores, 64 GiB RAM. One FULL cell's exact factor
  ~6.6 GB (fp32) -> the exact route can NOT hold 100 cells; the learned operator must be memory-light.

## 4. Mechanics of the problem (evidence)

- Spectrum: softest deformation modes of the box operator are 7e-7 (0013), 3.6e-6 (0021), 1.2e-5 (FULL) of the
  largest; FULL has 772 eigenvalues below 1e-4 lambda_max. fp32 rounding of a MATRIX already errs 0.8-2.2% in
  whitened spectrum -> matrices need fp64. For a FIELD with variational readout, errors enter second order.
- 94-96% of lattice load energy lies in the SOFTEST 10% of the cell spectrum; the compliance error is entirely
  attributable to that decile. Capacity spent on stiff directions is wasted.
- Soft/slow modes are LOCALIZED: in cut cells 17.6-21.9% of nodes are "weak" (3x3 diagonal block norm < 1% of
  median: thin slivers made by the cut, supported only by tiny body stiffness and gamma = 1e-4 GP terms); 94-99.4%
  of the slowest 64 modes' amplitude sits on those nodes; each slow mode covers ~11-30 nodes. Slow modes come in a
  dense band (128 slowest all within [0.0039, 0.0062] of the preconditioned spectrum).
- Global transmission: forces entering one face must reach the others (load paths through the shell network).
- Force-driven loads (q = S^-1 f) excite the soft modes: a skeleton with 0.3% error on smooth probes had 20-35%
  error on force-driven loads. Smooth-only training is insufficient.
- Variational readout property: S_hat - S = (E_hat - E)^T K (E_hat - E) >= 0 for any E_hat that matches q exactly
  on ports (Galerkin orthogonality): error is second order and one-sided (always too stiff). 1D example: middle
  displacement 32% off -> stiffness 1% off. Requires EXACT port values (hard constraint) and exact K in readout.
- Sensitivity -u^T (dK/dtau) u is FIRST order in the field error: the field must be accurate to a few percent in
  the actually loaded directions.

## 5. What was tried and what it taught (see ARCHITECTURE_HISTORY)

Structural dead ends (do not reintroduce):
- Entry-wise prediction of dense factors/operators with Frobenius or entry losses (0.47% Frobenius -> 39-70%
  response error; entry-loss descent direction provably fights the dangerous soft directions; symmetric-root
  sign branches gave 19 negative eigenvalues).
- One-sided low-rank corrections (can only soften). Fixed-geometry coarse bases for soft modes (3000 columns
  capture only 72%; soft modes follow thin material). Bandwidth truncation (soft stiffness = near/far-field
  cancellation 1e3-1e4). Compressed representations as regression targets (gauge non-unique; owner rejected
  compression). Generic operator networks that just concatenate q into the input without mechanical structure.
- Non-smooth output heads (global max/abs) broke optimization of learned local factors.
Evidence that works:
- EquiModel (index-free, cube-group augmentation) on FULL cells: unseen-cell assembled compliance 0.17-1.54%,
  64-cell lattice 1.12%. Failed on cut cells (41% median) due to training signal (entry loss vs physics),
  not generalization. Rotation augmentation helped held-out cut cells (0.64 -> 0.41).
- Route-1 skeleton (exact K in the loop; paired fine smoother + 2-level block polynomial coarse spaces +
  Chebyshev; energy readout): exact coefficients need 40-48 layers to pass the lattice gate; learning only 2
  scalars per layer (label-free energy minimization, 256 smooth samples, 300 Adam steps) -> 24 layers pass the
  lattice gate (0.72%/1.7% compliance, 0.93%/2.6% sensitivity). Query 32 layers ~174 ms per column.
  Smoother step from row-sum bound was 2-3x too conservative (Lanczos fixed it). Grouping weak nodes WITH their
  strong neighbors helps; separating them hurts.
- Collar + learned core material (B2): 23k numbers gave mu in [0.924, 1.150]; zero-learning mu>=1 theorem.
- Primal-dual certificate (route 2): equilibrated element forces give an upper bound; with GP outer ring, H=2,
  efficiency 1.03-1.15 -> teacher-free error estimate available later.
- BDD lattice solve (Mandel; coarse = partition-of-unity weighted rigid modes): 11-53 iterations independent of
  lattice size; needs per cell ~2-3 forward S*q and 1 approximate inverse S^+ r (Neumann solve) per iteration.
  Forward-only preconditioners need 300-1600 iterations. So the architecture should ALSO offer an approximate
  Neumann (force -> displacement) action for preconditioning (need not be very accurate).

## 6. Budgets and targets (to be confirmed with owner; use as design targets)

- Memory: 100 cells resident on a 32 GB card -> per-cell stored geometry coefficients << 300 MB; activations per
  query modest; fp32 network, fp64 only in energy reductions if needed.
- Throughput: a BDD design iteration with 100 cells needs ~70-160 forward + 35-55 inverse applications per cell
  -> ~2e4 network applications per design iteration; batched across cells. Target a few ms amortized per
  cell-application; current exact per-column query 0.5-40 ms, route-1 skeleton 174 ms.
- Encoding (geometry -> coefficients) per cell per design step: should beat exact fp32 factorization
  (0.08-0.85 s + moments 0.25-1 s), ideally <~0.5 s, and needs no factorization memory.
- Training: 5090, exact solves ms each after a 1-15 s factorization per geometry; geometry batches.

## 7. Open design questions the architecture must answer

a. How to consume the fixed-grid masked data: active-node graph (thin walls adjacent in space but disconnected in
   material must not talk) vs grid convolutions vs both; how to use element moments / element stiffness as edge
   features; how to handle GP faces and weak nodes.
b. How to hard-enforce port Dirichlet values (box + cut band) and exact rigid modes while staying linear in q.
c. How to get global transmission cheaply (coarse levels 65 -> 33 -> 17 -> 9; spectral vs coarse-grid vs
   geometry-adapted coarse spaces), and localized soft modes on slivers (fine level).
d. Where K may appear in the forward pass: pure feedforward (K only in readout) vs a few K-applications inside
   (hybrid toward fallback A). Trade-offs in accuracy, speed, generalization, training stability.
e. Reaction readout for lattice solves: S_hat q = E_hat^T K E_hat q needs the adjoint of the linear net
   (vjp); symmetric by construction. Alternative (K u_hat)_ports is not symmetric. Cost implications.
f. Approximate inverse (Neumann) action for BDD preconditioning.
g. Sensitivity: -u_hat^T (dK/dtau) u_hat with exact dK/dtau from moments; consistency, topology changes.
h. Equivariance (48-group) by augmentation vs construction; cut plane breaks per-cell symmetry but joint
   (g, q) equivariance holds.
i. Parameterization smoothness / conditioning; normalization so soft directions are not drowned
   (per-sample energy normalization; whitening; relative loss).
j. Capacity: per-geometry vs shared weights; how geometry encoder generates per-node/per-edge coefficients.
k. Precision: fp32 network vs fp64 readout.
l. How the architecture degrades gracefully into fallback A/B (shared components).
