# Mechanism-level literature scan for the v0 architecture

Scope: the seven topics requested, read against `ARCH_BRIEF.md` (the goal, the fixed decisions 1-9, the discretization facts and the evidence in sections 4-5).
Each entry gives the mechanism (what it does mathematically and why it works), its limits for our problem, and a verdict: **use / adapt / avoid**.

**Verification.** Every arXiv id below was checked on 2026-09-23 by fetching `arxiv.org/abs/<id>` and matching the title and first authors. Venues were checked with a web search, a publisher page or the arXiv journal-ref/comment field. Anything I could not confirm is marked **UNVERIFIED**. Where I describe a mechanism in more detail than the abstract gives, I either read the paper's HTML (marked "read") or I say it comes from my own knowledge of the paper. Numbers attributed to "us" come from the brief.

**One correction to the brief's wording.** Greenfeld et al. (2019) learn multigrid **prolongation**, not smoothers. Learned smoothers are Huang, Li & Xi (2021/2023) and, in a related form, Hsieh et al. (2019). Both are listed below.

---

## 0. Six mechanism facts that decide most verdicts

These follow from the brief combined with the literature below. The later sections refer back to them as F1-F6.

- **F1: learning the solution operator puts the singular part into exact arithmetic.** Our reaction operator S is a Dirichlet-to-Neumann / Steklov-Poincaré operator: a +1-order, non-smoothing map whose near field dominates. Ling, Ying & Zhou (arXiv:2606.25952) show that a network learns a DtN map only after its geometry-independent principal part is taken out analytically. Decision 1 does this implicitly. The network outputs E (the harmonic extension, an order ≤ 0 smoothing map). The singular near field enters only through the exact K in `S_hat = E_hat^T K E_hat`. The extension E = -K_II^{-1} K_IB is an interior Green's operator, and Green's operators of elliptic problems are provably compressible in multiscale form: Schäfer-Katzfuss-Owhadi's sparse Cholesky with screening (arXiv:2004.14455, SISC) and Schäfer-Owhadi (arXiv:2110.05351, SISC); H-matrix theory. The boundary-restricted Schur complement does not have this property, which the project docs already record. So the choice of the learned object is also the choice of the object that has a multiscale structure theorem.
- **F2: q-linearity plus the exact-K readout make the reaction symmetric PSD for any network.** For fixed g, `S_hat(g) = E_hat(g)^T K(g) E_hat(g)` is exactly the Hessian of the energy of the predicted field. The convex-neural-energy-element paper (arXiv:2608.02036) reports that a field-predicting operator trained by value regression gives an indefinite assembled Hessian (247% error). That failure needs a Hessian containing second derivatives of a q-nonlinear network. Decision 5 removes that term. Parish et al. (arXiv:2307.05434) and Melchers-Dolean-Abdelmalik (arXiv:2605.19867) both find symmetry/SPSD by construction necessary when a learned operator is used inside a coupled solve or a CG solve. Our readout gets both for free, as long as q never passes through a nonlinearity.
- **F3: with hard port values, the Ritz loss equals the error we report.** Let u* be the exact extension of q (K_II u*_I = -K_IB q), and let u_hat equal q on the ports. Galerkin orthogonality gives `Π(u_hat) - Π(u*) = ½ (u_hat - u*)^T K (u_hat - u*)`. With per-sample normalization by q^T S q, the energy loss is exactly μ(q) - 1. The label-free loss and the gate metric are therefore the same quantity per direction. Enforcing port values exactly is what makes this hold: a soft BC penalty would break the identity and the one-sided (too stiff) bound.
- **F4: the Ritz loss is badly conditioned in the output space.** Its Hessian with respect to the interior outputs is K_II, with condition numbers of 1e6-1e7 on cut cells. Gradient flow therefore corrects the stiff components of the field error first and the soft components last, and 94-96% of the lattice energy sits in the soft decile. The parameterization acts as the optimizer's preconditioner. Route 1 worked label-free because a multigrid-shaped q-path is such a preconditioner. The fixes in the literature are energy natural gradient (Müller & Zeinhofer, arXiv:2302.13163) or an auxiliary loss with a better-conditioned Hessian (supervised field L2 on exact solutions). The v0 design must take this into account; it is not only an optimizer detail.
- **F5: our slow modes break the frequency split that multigrid-style networks assume.** HINTS, MgNO, U-NO and spectral layers assume "smoother or pointwise layer = high frequency, coarse level or network = low frequency". In our cut cells the low-energy modes are spatially high-frequency and localized: 11-30 weak nodes on slivers supported by γ = 1e-4 ghost-penalty terms, with 94-99% of the slowest 64 modes' amplitude on 18-22% of the nodes. A smooth coarse channel cannot represent them. A pointwise smoother cannot fix them either, because they are slow for it by definition. The high-contrast DD literature (GenEO, CEM-GMsFEM, adaptive BDDC/FETI-DP) handles this case by adding **local low-energy eigenvectors to the coarse space**. The route-1 finding that "grouping weak nodes with their strong neighbours helps" is a small instance of the same principle.
- **F6: material connectivity is K's sparsity graph, and geometric coarsening breaks it.** On the fixed Q2 grid, the discrete problem couples node i and node j exactly when K_ij ≠ 0 (same active element, or two face-adjacent elements joined by a ghost-penalty face). A fine-level linear operator whose couplings are masked by K's sparsity cannot let two nodes talk unless the discretization already lets them. Proximity graphs (radius graphs in GINO/GNO, world-space edges in MeshGraphNets) and naive 2× geometric coarsening can merge two walls that are close in space but disconnected in material. BSMS-GNN (arXiv:2210.02573) names this failure explicitly ("wrong edges across geometry boundaries").

---

## 1. Linear operators with multigrid or hierarchical structure

### 1.1 MgNO: He, Liu & Xu, "MgNO: Efficient Parameterization of Linear Operators via Multigrid", ICLR 2024, arXiv:2310.19809 (verified; read)
- **Mechanism.** A layer is σ(W u + B u + b). Each linear operator W is a multi-channel **V-cycle** with no nonlinearity inside it. The smoothing step is `u ← u + B^{ℓ,i} * (f - A^ℓ * u)`, with learnable convolution kernels A^ℓ (a learned "stiffness") and B^{ℓ,i} (a learned smoother). Restriction and prolongation are **fixed** stride-2 FE transfer stencils. Parameters are shared across channels and differ per level. The default uses 5-6 levels, and boundary conditions are handled by the padding type. Nonlinearity (GELU) appears only between V-cycles. Why it works: a V-cycle is a near-optimal O(N) parameterization of an elliptic inverse, so a few learned kernels per level cover all scales. The authors report easier training than CNNs and less overfitting than spectral NOs.
- **Limits for us.** (i) The kernels are translation-invariant convolutions shared across the domain, but our operator is strongly heterogeneous: masks, cut coefficients, weak nodes. Coefficients must be generated per node and per edge by the geometry path, not shared. (ii) The fixed geometric R/P merges disconnected walls on coarse levels (F6). (iii) The GELU between cycles breaks q-linearity (decision 5), so in the q-path we must stack linear cycles only. (iv) Pointwise-kernel smoothers cannot handle localized soft modes (F5).
- **Verdict: adapt. This is the closest template for the q-path.** Keep the linear V-cycle skeleton (smoother as a learned correction of a learned or exact residual; restrict, prolong, repeat). Replace the conv kernels with geometry-generated, K-sparsity-masked node and edge blocks. Replace the fixed transfer with material-aware transfer (1.4, 2.4). Move all nonlinearity into the coefficient generator.

### 1.2 MgNet: He & Xu, "MgNet: A Unified Framework of Multigrid and Convolutional Neural Network", Sci. China Math. 62 (2019) 1331-1354, arXiv:1901.10415 (verified)
- **Mechanism.** Reads a CNN as a multigrid iteration: a data space f and a feature space u; smoothing `u ← u + σ∘B*(f - A*u)`; pooling as restriction of the residual. This is the conceptual root of MgNO.
- **Verdict: adapt (conceptually).** It justifies the two tensors (current field u, residual or "data" f) in the q-path, and it maps directly onto fallback A if the "A" in `f - A*u` is our exact K.

### 1.3 U-NO: Rahman, Ross & Azizzadenesheli, "U-NO: U-shaped Neural Operators", arXiv:2204.11127 (verified; published in TMLR per the search result, **year UNVERIFIED**)
- **Mechanism.** A U-Net of FNO integral layers with a contracting and expanding domain and channel dimension, plus skip connections. It gives depth and memory efficiency.
- **Limits.** Its blocks are Fourier layers, which fail on masked, high-contrast domains (7.x), and it is nonlinear throughout.
- **Verdict: avoid as a block. The U/V shape itself is already in 1.1.**

### 1.4 Learned AMG prolongation: Luz, Galun, Maron, Basri & Yavneh, "Learning Algebraic Multigrid Using Graph Neural Networks", ICML 2020 (PMLR v119), arXiv:2003.05744 (verified)
- **Mechanism (abstract plus my knowledge of the paper).** A single GNN maps a sparse SPD matrix A, used as a graph with the matrix entries as edge features, to the values of a sparse prolongation P on a fixed sparsity pattern taken from classical AMG. It is trained **unsupervised**: the loss is a Frobenius-norm surrogate of the two-grid error propagation matrix (smoother, then coarse correction with the Galerkin coarse operator P^T A P). It generalizes across a class of matrices, including unstructured FE problems with varying coefficients.
- **Why it matters.** It is the prototype of "geometry-generated coefficients of a linear operator, trained by a label-free operator-quality loss", with the matrix entries themselves as edge features. That answers part of question (a): use K's 3×3 blocks as edge features.
- **Limits.** It was demonstrated on scalar diffusion. Elasticity needs the near-nullspace (six rigid modes) preserved by P, which is the smoothed-aggregation requirement (Vaněk-Mandel-Brezina, 1.7). The loss needs P^T A P, so it assumes K-applications inside the method.
- **Verdict: adapt.** Coarse-level transfers should be generated from K's graph with edge features = normalized K blocks, not from geometry pixels. When we use K inside (fallback A/B), a two-grid-type energy loss is a valid label-free objective for the coarse space.

### 1.5 Greenfeld, Galun, Kimmel, Yavneh & Basri, "Learning to Optimize Multigrid PDE Solvers", ICML 2019, arXiv:1902.10248 (verified)
- **Mechanism.** A network maps a family of parameterized 2-D diffusion problems (structured grids, discontinuous coefficients) to **prolongation** stencils. The loss is unsupervised and approximates the two-grid convergence factor. It beats Black-Box multigrid.
- **Verdict: adapt (same lesson as 1.4).** Prolongation must be coefficient-dependent, which is our "fixed-geometry coarse bases capture 72%" dead end seen from the other side. Its structured-grid, periodic analysis setting does not transfer to masks.

### 1.6 Katrutsa, Daulbaev & Oseledets, "Deep Multigrid: learning prolongation and restriction matrices", arXiv:1711.03825 (verified; venue not checked)
- **Mechanism.** Treats P and R of a single problem class as trainable sparse matrices and minimizes a stochastic upper bound of the spectral radius of the multigrid iteration.
- **Verdict: avoid as a method (one matrix, no geometry conditioning). The stochastic spectral-radius bound is a usable diagnostic.**

### 1.7 Energy-minimizing coarse spaces: Xu & Zikatanov, "On an energy minimizing basis for algebraic multigrid methods", Comput. Visual. Sci. 7 (2004) 121-127 (verified); Vaněk, Mandel & Brezina, "Algebraic multigrid by smoothed aggregation for second and fourth order elliptic problems", Computing 56 (1996) 179-196 (verified)
- **Mechanism.** Xu-Zikatanov: the coarse basis that minimizes the total energy subject to reproducing constants is **locally harmonic** on each coarse "element" and has an explicit form. Smoothed aggregation: aggregates on the strength-of-connection graph, a tentative P that reproduces the **zero-energy modes** (rigid modes for elasticity), then one smoothing sweep of P.
- **Why it matters.** Our network is asked to produce an extension, which is a harmonic basis. The energy-minimizing coarse basis is the multilevel version of the same object. Aggregation on K's graph respects material connectivity (F6), and "reproduce rigid modes" is our exact-rigid-mode requirement moved to every level.
- **Verdict: use as design principles.** (a) Coarse nodes = aggregates of the K-graph, split by connected material component inside a coarse cell, not raw 65→33→17→9 grid points. (b) Each coarse level carries the 6 rigid modes exactly. (c) Transfer operators initialized as smoothed tentative prolongators, with learned corrections.

### 1.8 Learned smoothers: Huang, Li & Xi, "Learning optimal multigrid smoothers via neural networks", SIAM J. Sci. Comput. 45(3) (2023) S199-S225, arXiv:2102.12071 (verified); Hsieh, Zhao, Eismann, Mirabella & Ermon, "Learning Neural PDE Solvers with Convergence Guarantees", arXiv:1906.01200 (verified; published at ICLR 2019 per my knowledge, **venue UNVERIFIED**)
- **Mechanism.** Huang/Li/Xi: CNN smoothers are generated from operator stencils and trained with a loss derived from multigrid convergence theory. They generalize to larger sizes and other geometries. Hsieh et al.: a learned **linear** correction H is added to a standard iteration, u ← Ψ(u) + H(Ψ(u) - u). The exact solution stays a fixed point because the correction acts on an update that vanishes at the solution, and convergence is guaranteed when the spectral radius is < 1. Trained on one geometry, it generalizes to others.
- **Verdict: use for fallback A; adapt for the main line.** Hsieh's construction keeps consistency with the exact solution for any learned weights, which is the right invariant for any layer that sees an exact residual. The stencil-to-smoother generation (Huang/Li/Xi) is exactly the hypernetwork pattern of decision 5. Pointwise smoothers are still insufficient on slivers (F5), so the fine level needs **patch/block** smoothers over weak nodes plus their strong neighbours.

### 1.9 UGrid: Han, Hou & Qin, "UGrid: An Efficient-And-Rigorous Neural Multigrid Solver for Linear PDEs", ICML 2024 (PMLR 235), arXiv:2408.04846 (verified)
- **Mechanism.** A U-Net-shaped multigrid solver with a convergence and correctness proof, trained unsupervised with a residual loss. It generalizes to input geometries and values.
- **Verdict: adapt (evidence for fallback A).** The exact residual in the loop gives correctness guarantees. Nothing about elasticity, masks or soft modes carries over directly.

### 1.10 Hierarchical and multipole kernels: Li et al., "Multipole Graph Neural Operator for Parametric PDEs", NeurIPS 2020, arXiv:2006.09535 (verified); Liu, Xu, Cao & Zhang, "Mitigating spectral bias for the multiscale operator learning" (HANO), arXiv:2210.10890 (verified; venue UNVERIFIED); Gupta, Xiao & Bogdan, multiwavelet operator, arXiv:2109.13459 (verified); M2NO, arXiv:2406.04822 (verified; KDD 2026 per arXiv comment)
- **Mechanism.** The kernel integral operator is split level by level into a near-field part (full rank, on a fine graph) and far-field parts (low rank, on coarser graphs), which is an FMM/H-matrix analogue with O(N) cost. HANO uses hierarchical attention with a scale-adaptive interaction range and an H¹ loss to counter spectral bias on rough-coefficient elliptic problems. The multiwavelet and M2NO methods do the same on wavelet bases or multigrid structure.
- **Limits.** MGNO builds its graphs from spatial radius balls (F6 violation). All of these are nonlinear in the input. HANO's H¹ loss is a hint for F4.
- **Verdict: adapt the principle** (near field full rank at the fine level, far field on coarse levels, justified by F1). **Avoid** radius-ball graphs and spectral multiwavelet transforms on masked domains.

### 1.11 HINTS: Zhang, Kahana, Kopaničáková, Turkel, Ranade, Pathak & Karniadakis, "Blending neural operators and relaxation methods in PDE numerical solvers", Nat. Mach. Intell. 6 (2024) 1303-1313, arXiv:2208.13273 (verified)
- **Mechanism.** Every few Jacobi or Gauss-Seidel sweeps, one iteration replaces the relaxation by a DeepONet correction applied to the current residual, δu = DeepONet(r). The network's spectral bias (good at smooth components) complements relaxation (good at rough components), which gives a roughly uniform convergence rate across eigenmodes.
- **Limits for us.** The complementarity assumes the slow modes of relaxation are smooth. Ours are localized and rough (F5), so both halves miss the same modes. The DeepONet correction is nonlinear in r unless built linear, which breaks symmetry and linearity for PCG/BDD (see 5.10).
- **Verdict: adapt only the pattern "exact residual → linear learned correction" (= fallback A). Avoid the spectral-bias division of labour.** Evidence that this matters for us: the 0.3% → 20-35% jump on force-driven loads.

### 1.12 Kopaničáková & Karniadakis, "DeepONet Based Preconditioning Strategies For Solving Parametric Linear Systems of Equations", arXiv:2401.02016 (verified; venue UNVERIFIED)
- **Mechanism.** Uses HINTS-type DeepONet corrections as preconditioners inside Krylov methods, where nonlinearity forces flexible Krylov variants.
- **Verdict: avoid the nonlinear form for BDD/PCG. See 5.10 for the linear, symmetric alternative.**

---

## 2. Graph neural operators and geometry-informed operators on masked, unstructured or cut domains

### 2.1 GNO: Li et al., "Neural Operator: Graph Kernel Network for PDEs", arXiv:2003.03485 (verified); Kovachki et al., "Neural Operator: Learning Maps Between Function Spaces", JMLR 24 (2023) art. 89, arXiv:2108.08481 (verified)
- **Mechanism.** A layer is v ← σ(W v + ∫_{B(x,r)} κ_θ(x, y, a(x), a(y)) v(y) dy), a kernel integral over a radius ball, discretized by message passing, with κ generated by an MLP of positions and coefficients. If σ is dropped and κ depends only on (x, y, g), the layer is **linear in v with a geometry-generated kernel**. That is exactly our q-path pattern.
- **Limits.** The ball B(x, r) is spatial, so it crosses material gaps (F6). With O(r³) neighbours in 3-D, a large r is expensive and a small r gives a slow receptive-field growth of r per layer. Hence the need for multiscale.
- **Verdict: adapt.** Keep the "linear integral with generated kernel" form. Replace the ball with **K's sparsity pattern**: a 5×5×5 node stencil for Q2 elements, extended across ghost-penalty faces. Generate the kernel as 3×3 blocks, from invariants of the local geometry and the local K blocks (for equivariance, see 6.5).

### 2.2 GINO: Li et al., "Geometry-Informed Neural Operator for Large-Scale 3D PDEs", NeurIPS 2023, arXiv:2309.00583 (verified)
- **Mechanism.** SDF plus point cloud → GNO (radius graph) onto a regular latent grid → FNO on the latent grid → GNO back to the query points. It is discretization-convergent.
- **Limits.** Both halves conflict with our data: radius graphs cross thin gaps (F6), and FNO on a latent grid is a global translation-invariant spectral kernel on a masked, high-contrast body (7.x). Its target (surface pressure in external aerodynamics) has no thin load-carrying walls.
- **Verdict: avoid.** We already live on a fixed grid, so no GNO encoder/decoder is needed.

### 2.3 Geo-FNO: Li, Huang, Liu & Anandkumar, "Fourier Neural Operator with Learned Deformations for PDEs on General Geometries", JMLR 24 (2023) art. 388, arXiv:2207.05209 (verified)
- **Mechanism.** Learns a deformation from the physical domain to a latent uniform grid and runs FNO there.
- **Limits.** It needs a diffeomorphism to a box. A multiply connected TPMS shell whose topology changes with τ and the cut has none. It also contradicts decision 3 (fixed coordinates).
- **Verdict: avoid.**

### 2.4 MeshGraphNets: Pfaff et al., ICLR 2021, arXiv:2010.03409 (verified); MultiScale MeshGraphNets: Fortunato et al., ICML 2022 AI4Science workshop, arXiv:2210.00612 (verified); BSMS-GNN: Cao, Chai, Li & Jiang, arXiv:2210.02573 (verified; venue not checked)
- **Mechanism.** MGN encodes and processes with message passing on **mesh-space edges** (topology) and separate **world-space edges** (proximity, for contact). MS-MGN adds a coarse mesh with interpolation. BSMS builds coarse levels by bi-stride pooling on the graph (every other BFS frontier), explicitly to avoid the proximity-based coarsening that "can introduce wrong edges across geometry boundaries". It uses one message-passing step per level and non-parametric pooling/unpooling, U-Net style.
- **Why it matters.** The mesh-space/world-space distinction answers "walls close in space but disconnected in material must not talk": use topology (K-graph) edges only, since there is no contact in our physics. BSMS is a ready-made, cheap coarsening that respects connectivity.
- **Verdict: adapt.** Build the multilevel hierarchy by **graph-based** coarsening of the active K-graph (bi-stride or aggregation, see 1.7), or by grid coarsening with per-component splitting of coarse cells. Never by proximity. The nonlinear MLP messages must be replaced by linear, geometry-generated messages in the q-path.

### 2.5 Transolver: Wu, Luo, Wang, Wang & Long, "Transolver: A Fast Transformer Solver for PDEs on General Geometries", ICML 2024 (PMLR 235), arXiv:2402.02366 (verified)
- **Mechanism.** "Physics-attention": points are softly assigned to M learnable slices (weights = softmax of a projection of point features). Each slice becomes a token (a weighted average), attention runs among the M tokens, and the result is distributed back with the same weights. The cost is linear in N. This is a learned, data-adaptive **low-rank global channel**, u ← W^T A W v with W of size M × N.
- **Linear-in-q version.** If the slice weights W and the token mixing A are computed from geometry features only, the map v ↦ W^T A(g) W v is linear in v. It is a geometry-adaptive coarse space with M coarse functions (compare Galerkin attention in 3.4).
- **Limits.** A rank-M global channel cannot carry localized sliver modes (F5), and soft slices would blur across material gaps unless masked by connected components. The "one-sided low-rank corrections can only soften" dead end does not apply because this is a channel inside E, not an additive correction to S. The readout is still E^T K E.
- **Verdict: adapt, as an optional global-transmission channel at the coarsest level** (for example 9³ plus a few hundred geometry-generated slice functions), with slice weights masked by material components. Do not use it as the primary carrier.

### 2.6 DAFNO: Liu, Jafarzadeh & Yu, "Domain Agnostic Fourier Neural Operators", NeurIPS 2023, arXiv:2305.00478 (verified)
- **Mechanism.** Multiplies the FNO integrand by a smoothed characteristic function χ, roughly ∫ χ(x) χ(y) κ(x - y)(v(y) - v(x)) dy, so that the FFT still applies on the bounding box and the geometry enters explicitly. Demonstrated on material modeling, airfoils and fracture (topology change).
- **Limits.** χ masks the endpoints, but κ(x - y) stays **translation-invariant**, so two points in disconnected walls interact exactly as strongly as two points in the same wall at the same distance (F6). The spectral truncation plus the smoothed χ blurs features one or two elements thick, which is the scale of our slivers.
- **Verdict: avoid for the fine and middle levels.** Acceptable only as a coarse, smooth transmission channel, and 2.5 or a coarse graph does that better.

### 2.7 Other embedded- and unfitted-domain learners (all verified)
- Badia et al., "Unfitted finite element interpolated neural networks", arXiv:2501.17438: a network interpolated onto an unfitted FE space, trained on a discrete (dual-norm) weak residual. It reports that cut-cell **stabilization in the residual norm** strongly accelerates training and increases robustness. **Adapt:** our ghost-penalty-stabilized K is already the right norm (F3), and their result supports putting the stabilized operator in the loss.
- WINO, arXiv:2605.24651: φ-FEM on a fixed Cartesian mesh with a level set. Dirichlet data enter through a **φ-FEM lifting**, the network learns only the homogeneous part, and training is data-free on weak-form residuals. It is the closest discretization-wise neighbour. **Adapt** the lifting-plus-homogeneous-correction split (4.3). It is nonlinear and FNO-based, so avoid its backbone.
- Neural PDE Solvers for Irregular Domains, arXiv:2211.03241: a differentiable inside/outside identification on a grid. **Avoid.** Our masks are exact and come from CutFEM.
- Daviet et al., ACM TOG 2025, arXiv:2410.09417, and Saberi-Zhao-Vogel, arXiv:2403.11632: learn cut-cell quadrature or stabilization parameters, not operators. **Not applicable.** Our moments are exact.

---

## 3. Geometry-conditioned linear operators (exactly linear in one input, nonlinear in another)

### 3.1 VarMiON: Patel, Ray, Abdelmalik, Hughes & Oberai, "Variationally Mimetic Operator Networks", arXiv:2209.12871 (verified; CMAME per publisher link in search, **volume/year UNVERIFIED**)
- **Mechanism.** Mimics a Galerkin solution: u(x) = τ(x)^T D(θ) (sampled f, g). The PDE coefficient θ goes through a nonlinear branch that outputs a matrix, and the sources and boundary data enter **linearly** by matrix-vector product. The error analysis splits the error into data, training, quadrature and "covering" terms and involves the stability constants.
- **Verdict: use the principle, which is decision 5 exactly.** The limit shown in 3.3 applies: sampling the inputs at sensors rather than integrating them against a basis breaks symmetry when the operator is used as a preconditioner.

### 3.2 Neural Green's Operators: Melchers, Prins & Abdelmalik, "Neural Green's Operators for Parametric Partial Differential Equations", CMAME 455 (2026) 118893, arXiv:2406.01857 (verified; read)
- **Mechanism.** u_hat(x) = Σ φ_m(x) Â_mn(F[θ]) d_n[f, g], with d_n = ∫ψ_n f + Σ∫_Γ g B̃ψ_n. The inputs are **integrated against a basis** and the matrix Â is produced by a network from the discretized weak-form operator F[θ]. One variant sets Â ≈ F₀^{-1}[Σ_{k≤K}(-δF F₀^{-1})^k + NN(·)], a truncated Neumann series with the FE inverse of a reference operator as inductive bias plus a learned remainder. The paper reports better out-of-distribution generalization than DeepONet or FNO, and a **linear** preconditioner P_NGO = P Â R.
- **Verdict: adapt.** Two lessons. (i) Feed the network with the **FE operator itself** (our K blocks or moments), not with raw geometry. (ii) The Neumann-series form "reference inverse plus learned remainder" is a principled geometry-conditioned linear parameterization. For us the reference would be the route-1 skeleton or a smoother.

### 3.3 Melchers, Dolean & Abdelmalik, "When can a neural operator replace a coarse solve? Architectural principles for two-level preconditioning", arXiv:2605.19867 (verified; read)
- **Mechanism.** A 2×2 study with two axes: sampling vs integrating the input against the output basis, and linear vs nonlinear in the source. Only the linear, integrating architecture (NGO, preconditioner w = Z C(θ) Z^T W v) gives matching row and column spaces and therefore a **symmetric** preconditioned operator. Its symmetrized form (Z (C + C^T) Z^T / 2) is SPD whenever sym(C) is. It matches exact coarse solves in PCG. DeepONet and VarMiON (sampling) give non-symmetric spectra and PCG breakdown. The nonlinear variants fail on non-self-adjoint problems.
- **Verdict: use for the Neumann/BDD action (question f).** Build the approximate S^+ as M(g) = B_g^T C_g B_g, where the forces are integrated against the same basis that carries the output. Then M is symmetric PSD by construction and can be used inside BDD-PCG.

### 3.4 Galerkin/Fourier transformer: Cao, "Choose a Transformer: Fourier or Galerkin", NeurIPS 2021, arXiv:2105.14995 (verified)
- **Mechanism.** Softmax-free attention, (Q K^T) V / n, read as a Petrov-Galerkin projection. It is linear in V for fixed Q and K, at O(N d²) cost.
- **Verdict: adapt with 2.5.** If Q and K are generated from geometry only and q flows through V, this is a legal (linear-in-q) global channel with learned, geometry-dependent trial and test functions.

### 3.5 DeepONet: Lu, Jin & Karniadakis, arXiv:1910.03193 (verified; Nat. Mach. Intell. 2021 per my knowledge, **venue UNVERIFIED**); HyperDeepONet: Lee, Cho & Hwang, ICLR 2023, arXiv:2312.15949 (verified); NOMAD: Seidman et al., arXiv:2206.03551 (verified)
- **Mechanism.** u(x) = Σ_k b_k(input) t_k(x). With a linear branch (b = W q), the operator is rank ≤ p. HyperDeepONet lets a hypernetwork generate the target network's weights from the input function. NOMAD replaces the linear decoder with a nonlinear one (not allowed for q).
- **Limits.** A rank-p bottleneck on 5k-24k port DOFs is a compression (rejected, decision 2). Localized sliver modes need rank that grows with the number of slivers. The sampled-input form is non-symmetric (3.3).
- **Verdict: avoid as the q-path. Adapt only the hypernetwork idea (the geometry network emits the coefficients of the linear q-path), which is decision 5.**

### 3.6 Green's-function learning: Boullé & Townsend, "Learning elliptic PDEs with randomized linear algebra", arXiv:2102.00491 (verified; published in Found. Comput. Math. per my knowledge, **venue UNVERIFIED**); Boullé, Earls & Townsend, arXiv:2105.00266 (verified)
- **Mechanism.** Green's functions of elliptic operators have hierarchical low-rank off-diagonal structure. A randomized SVD on H-matrix blocks recovers them from O(polylog) input-output pairs, with provable rates. Random smooth forcings are the sampling scheme.
- **Verdict: use as theory.** Along with F1, it justifies a multiscale linear q-path and random-field boundary sampling (the 30% GRF slot of decision 9). It also warns that sample complexity depends on the spectrum being covered, which is why the force-driven and adversarial slots are needed.

### 3.7 HyperNetworks: Ha, Dai & Le, arXiv:1609.09106 (verified; ICLR 2017 per my knowledge, **venue UNVERIFIED**)
- **Verdict: use (vocabulary).** Memory cost is the practical constraint. Per-node 3×3 blocks × 123k active nodes × L layers in fp32 is about 4.4 MB per block-layer. So even 24 layers × 2 blocks per node is about 200 MB, which is over budget for 100 cells. Coefficients must be **shared across layers**, or generated as a few per-node scalars that modulate fixed or K-derived blocks (6.5).

---

## 4. Energy, Ritz and physics-informed losses; hard Dirichlet; rigid modes

### 4.1 Deep Ritz: E & Yu, arXiv:1710.00211 (verified; Commun. Math. Stat. 2018 per my knowledge, **venue UNVERIFIED**); Deep Energy Method: Nguyen-Thanh, Zhuang & Rabczuk, Eur. J. Mech. A/Solids 80 (2020) 103874 (verified); Samaniego et al., arXiv:1908.10407 (verified); VINO: Eshaghi et al., arXiv:2411.06587 (verified)
- **Mechanism.** Minimize the potential energy of a network field instead of the strong residual. It needs only first derivatives and is label-free. VINO carries this over to neural operators on element discretizations.
- **Limits.** Continuous Ritz with quadrature by sampling is noisy. Ours is **discrete and exact** (x^T K x with the true K), so the loss is exact (F3). The known weakness is F4 (conditioning).
- **Verdict: use.** The main loss is ½ u_hat^T K u_hat / (q^T S q), which is μ - 1 in that direction.

### 4.2 Energy natural gradient: Müller & Zeinhofer, "Achieving High Accuracy with PINNs via Energy Natural Gradients", arXiv:2302.13163 (verified; ICML 2023 per my knowledge, **venue UNVERIFIED**)
- **Mechanism.** Preconditions the parameter gradient with the Gram matrix of the energy inner product (a Gauss-Newton step in function space), which removes the ill-conditioning of the energy Hessian. It gives orders-of-magnitude accuracy gains.
- **Verdict: adapt (as a principle for F4).** Exact ENGD is too expensive for millions of parameters. Cheap substitutes: (i) a supervised auxiliary term in a better-conditioned norm (field L2 or D-weighted L2 against exact u*, which the brief already makes available); (ii) curriculum from rigid, macro and smooth loads toward force-driven and adversarial ones; (iii) most importantly, a q-path shaped like a preconditioner (multigrid), so that small parameter changes move soft-mode content efficiently. Oh, Lee, Darbon & Karniadakis (arXiv:2606.21828, verified) independently show that a short label-free energy fine-tune fixes indefiniteness left by mean-squared training, which supports "supervised warm start, then energy".

### 4.3 Hard boundary constraints: Sukumar & Srivastava, "Exact imposition of boundary conditions with distance functions in PINNs", arXiv:2104.08426 (verified; CMAME 2022 per the doi quoted in arXiv:2510.24557); Liu et al., "A Unified Hard-Constraint Framework for Solving Geometrically Complex PDEs", NeurIPS 2022, arXiv:2210.03526 (verified); BOON: Saad, Gupta, Alizadeh & Maddix, "Guiding continuous operator learning through physics-based boundary constraints", ICLR 2023, arXiv:2212.07477 (verified); "Imposing Boundary Conditions on Neural Operators via Learned Function Extensions", arXiv:2602.04923 (verified); arXiv:2510.24557 (verified)
- **Mechanism.** In continuous PINNs, u = g_lift + φ·N with φ an approximate distance function that vanishes on Γ_D. BOON modifies the operator kernel so that the output satisfies the BC exactly. The learned-extension paper maps boundary data to a **latent volume extension** that any operator network can consume, and reports large gains on elasticity with highly variable BCs. arXiv:2510.24557 shows that distance-function constructions become unstable on piecewise-C¹ boundaries (our box edges and corners, plus cut-band corners).
- **Our discrete case is simpler, and exact.** The ports are nodal DOF sets on the fixed grid, so the hard constraint is a masked overwrite: u_hat = P_port^T q + P_int^T N_g(q). No distance function is needed, and corner instability does not arise. What carries over is the **lifting idea**: how q enters the volume matters. A good first lift (discrete harmonic-ish extension of q into the active region: a few masked smoothing sweeps with exact K, or the extension through the route-1 skeleton's first levels) followed by a learned correction that vanishes on the ports is exactly WINO's "φ-FEM lifting + homogeneous part" and the learned-extension paper's thesis.
- **Verdict: use hard masking. Adapt "lift, then correct"** with the lift linear and K-based. **Avoid** distance-function multipliers. They are unnecessary here and they fail at corners.

### 4.4 Physics-informed graph Galerkin networks: Gao, Zahr & Wang, arXiv:2107.12146 (verified; CMAME 2022 per my knowledge, **venue UNVERIFIED**); PINO: Li et al., arXiv:2111.03794 (verified)
- **Mechanism.** A GNN outputs nodal FE coefficients. The loss is the discrete FE residual (weak form), and Dirichlet values are imposed strongly by setting the nodal values, which is exactly our discrete setting. PINO mixes data and PDE losses at the operator level.
- **Verdict: use (precedent).** "Nodal outputs + exact FE operator in the loss + strong Dirichlet" is established. Our energy loss is its symmetric, variational version.

### 4.5 Rigid modes
- **Mechanism set.** (a) The smoothed-aggregation near-nullspace requirement (1.7). (b) The regularization-nullspace principle in convex neural energy elements (arXiv:2608.02036, verified): if the regularizer's nullspace misses part of the physics nullspace, a one-signed irreducible bias appears. (c) The BDD coarse space built from partition-of-unity-weighted rigid modes (Mandel 1993, 5.7).
- **Construction for us.** Split q = R_p α + q_def with α = (R_p^T W R_p)^{-1} R_p^T W q (an O_h-invariant port weighting W), and set u_hat = R α + E_hat_g(q_def). Here R holds the 6 rigid fields on all active nodes, which Q2 represents exactly. Because K R = 0, the rigid part contributes nothing to S_hat and nothing to u^T dK u. It is extended exactly, and the network never sees it. Equivariance is preserved because span(R) is O_h-invariant and the projection commutes with the group action when W is invariant. On coarse levels, include R restricted to each level in the transfer (1.7), so the q-path cannot create spurious rigid-mode energy.
- **Verdict: use.**

---

## 5. Learned DtN / Steklov-Poincaré / substructuring and neural domain decomposition

### 5.1 SNI: Huang, Zhang, Wu & Cheng, "Operator Learning with Domain Decomposition for Geometry Generalization in PDE Solving", ICLR 2026, arXiv:2504.00510 (verified)
- **Mechanism.** Schwarz Neural Inference: the domain is split into small subdomains, a neural operator trained on randomly shaped local problems (with PDE symmetries used for data) solves each, and overlapping Schwarz iterations stitch the global solution. The paper gives a convergence-rate and error-bound analysis. Geometry generalization moves from the network to the algorithm.
- **For us.** Our "subdomain" is the cell, and the global algorithm is BDD at lattice level, which is already the SNI philosophy. SNI's local solvers are **nonlinear** in the boundary data and **overlapping**, and they use many local calls per global iteration. We need a non-overlapping, symmetric method with S and S^+ actions, which requires linearity (F2, 3.3).
- **Verdict: adapt the philosophy (train local, compose by a classical algorithm). Avoid its local-solver form.**

### 5.2 NEST: Secchi, Balint & Maurizi, "Neural-Schwarz Tiling for Geometry-Universal PDE Solving at Scale", arXiv:2605.12343 (verified)
- **Mechanism.** Learns a neural operator on minimal **3×3×3 voxel patches** with diverse local geometries and boundary/interface data, then tiles unseen voxel domains with overlapping patches, iterates Schwarz with partition-of-unity assembly, and generalizes to large neo-Hookean 3-D domains.
- **For us.** It shows that very local learned solvers generalize across geometry, which matches our finding that the soft physics lives in 11-30-node patches. But pure overlapping Schwarz with local solvers converges with a number of iterations that grows with the domain size in elements, unless there is a coarse space. Our cell is 32 elements across and needs global transmission inside the forward pass.
- **Verdict: adapt one idea: a fine-level operator made of learned local patch solves (a patch/block smoother) as the sliver handler. Avoid NEST's iteration as the whole method.**

### 5.3 Non-overlapping Schwarz FE-NO: Wang, Gupta, Ruan & Goswami, "A Non-Overlapping Schwarz Hybrid Finite Element-Neural Operator Framework for Solid Mechanics on Irregular Domains", arXiv:2606.08796 (verified)
- **Mechanism.** Neumann-Dirichlet alternating Schwarz between an FE region and a Point-DeepONet region. The NO passes traction to FE, and FE passes displacement back. Strains and stresses are derived analytically from the displacement operator rather than learned separately.
- **For us.** It supports "learn displacement, derive reactions from kinematics and constitutive law". Our variational readout is the discrete, symmetric version. Neumann-Dirichlet alternation is non-symmetric and sequential, so it is inferior to BDD for a lattice of about 100 interchangeable cells.
- **Verdict: adapt the "derive, don't learn, the reaction" principle (already decision 1). Avoid the coupling scheme.**

### 5.4 DD-DeepONet: Yang, Li, Zhao & Jiang, arXiv:2508.02717 (verified; a journal version exists per a ScienceDirect listing, **details UNVERIFIED**); Mosaic Flows: Wang, Planas, Chandramowlishwaran & Bostanabad, CMAME 389 (2022) 114424, arXiv:2104.10873 (verified)
- **Mechanism.** Both compose subdomain operators (DeepONet or genomic flow networks on simple shapes: rectangles, cuboids, a unit "genome") by Schwarz-type iteration, with stretching maps for shape variation.
- **Verdict: avoid.** They rely on simple subdomain shapes and nonlinear local operators. They only confirm that composition across subdomains is feasible.

### 5.5 Learned substructured energies and DtN maps (all verified): Parish et al., arXiv:2307.05434 (Comput. Mech., per the project docs); NOEM: Ouyang, Shin, Liu & Lu, arXiv:2506.18427; Beatson et al., "Learning Composable Energy Surrogates", arXiv:2005.06549 (NeurIPS 2020 per the project docs); Convex neural energy elements, arXiv:2608.02036; Du & Stechmann "element learning", arXiv:2308.02467; PPDNO, arXiv:2606.25952
- **Mechanisms and lessons.** Parish: the SPSD-constrained learned Schur complement is **required** for coupled solves, and unconstrained displacement→force regressors break them. NOEM: a neural operator exports the subdomain energy and the stiffness is its Hessian, with variational assembly. Beatson: composable energy surrogates, **data aggregation on assembled loads** (collect the interface modes that actually occur and add worst ones), which is our adversarial 20% slot. Convex neural energy elements: a hypernetwork-generated PSD quadratic form, the regularization-nullspace principle, and a failure report on field-regression elements. PPDNO: the DtN map has a singular principal part (F1).
- **Verdict: use as a set.** Our design (a linear E in q, S_hat = E^T K E) is a NOEM-type energy readout specialized to the linear case, where the Hessian is exact and PSD (F2). It is immune to the failure in arXiv:2608.02036 and meets Parish's SPSD requirement without learning a matrix. The main remaining risk is not structural but accuracy in the soft decile.

### 5.6 Learned Schwarz interface conditions and multilevel DD (all verified): Taghibakhshi et al., "Learning Interface Conditions in Domain Decomposition Solvers", NeurIPS 2022, arXiv:2205.09833; MG-GNN, ICML 2023 (PMLR v202), arXiv:2301.11378
- **Mechanism.** A GNN learns optimized Robin interface conditions and, in MG-GNN, the coarse-to-fine interpolation of two-level optimized Schwarz. The loss is an **unsupervised** bound on the error-propagation operator, evaluated on small problems and transferred to much larger unstructured grids. MG-GNN processes scales **in parallel** to avoid oversmoothing.
- **Verdict: adapt.** Evidence that operator-quality losses on small problems transfer to large ones. MG-GNN's parallel-scale processing is an alternative to a strictly sequential V-cycle for global transmission.

### 5.7 BDD, GenEO and learned adaptive coarse spaces
- Mandel, "Balancing domain decomposition", Commun. Numer. Methods Eng. 9(3) (1993) 233-241 (verified). Neumann-Neumann with a coarse balancing step built from subdomain nullspaces (rigid modes), which fixes the singular local Neumann problems. **Use** (lattice level).
- Spillane, Dolean, Hauret, Nataf, Pechstein & Scheichl, "Abstract robust coarse spaces for systems of PDEs via generalized eigenproblems in the overlaps" (GenEO), Numer. Math. 126(4) (2014) 741-770 (verified). Coarse vectors are **local generalized eigenvectors** with small eigenvalues, which makes the method robust to arbitrary coefficient contrast, elasticity included. **Use as the principle for F5** at two places: the fine sliver modes inside a cell, and, if needed later, lattice-level robustness when neighbouring cells have very different τ.
- Heinlein, Klawonn, Lanser & Weber: ML-predicted locations of critical edges for adaptive FETI-DP, then predicted coarse constraints (SIAM J. Sci. Comput., doi 10.1137/20m1344913, title verified by search; related arXiv:2312.14252 verified). Klawonn, Lanser & Weber-Hamacher, arXiv:2607.06261 (verified): networks **predict adaptive (AGDSW) coarse basis functions and their number**, with no online eigenproblem. They use **sign-invariant losses**, and transfer without retraining from diffusion to linear elasticity. Chung, Kim, Lam & Zhao, arXiv:2104.09162 (verified; venue UNVERIFIED): learned adaptive BDDC coarse spaces for high-contrast problems.
- **Verdict: adapt.** Two options for the sliver modes. (a) Compute them: small dense eigenproblems on patches around the weak nodes, using exact K, in the encoder. This is deterministic, cheap, allowed by decision 5 (nonlinear in g, linear in q), and needs no learning. (b) Predict them as in Klawonn et al., which needs sign- and rotation-invariant losses, or better, predict the **subspace projector** rather than individual vectors, because our slow modes form a dense band (128 within [0.0039, 0.0062]) and individual eigenvectors are not stable. Try (a) first.

### 5.8 Learned preconditioners for SPD systems (all verified)
Li, Chen, Du & Matusik, ICML 2023 (PMLR v202), arXiv:2305.16432; Häusner, Öktem & Sjölund, NeuralIF, TMLR, arXiv:2305.16368; Trifonov et al., arXiv:2405.15557 (a GNN starts from a classical preconditioner and learns corrections, for parametric problems with **high-contrast** coefficients); Chen, "Graph Neural Preconditioners", ICLR 2025, arXiv:2406.00809 (nonlinear, so it needs flexible GMRES); Trifonov, Muravleva & Oseledets, "Message-Passing GNNs Fail to Approximate Sparse Triangular Factorizations", TMLR 2026, arXiv:2502.01397 (negative result: non-local dependencies).
- **Verdict.** Adapt "start from a classical preconditioner and learn a correction", in the form of the route-1 skeleton plus learned coefficients. **Avoid** learning factorizations, which the negative result backs and which is already a dead end in our history. Avoid nonlinear preconditioners for BDD-PCG.

### 5.9 Robin-Neumann PINN/FEM coupling from a Steklov-Poincaré view, arXiv:2606.14181 (found in search; the title was on the arXiv HTML page, **not otherwise checked**)
- **Verdict: skip.** It concerns PINN-FEM coupling in FSI, not operator learning.

### 5.10 What this means for question (f), the Neumann action
BDD needs a symmetric positive (semi)definite approximation of S^+ on the balanced subspace (r ⊥ rigid modes). From 3.3 and F2, any learned inverse must be of the form M = B^T C B with the same basis on both sides, C SPD, and rigid modes projected out. Options, from cheapest to most accurate:
1. A **few steps of a symmetric iteration** on S_hat. Each step costs one forward pass plus one adjoint pass, so this is affordable only for 2-3 steps.
2. A **separate, light, linear Neumann network** of the NGO form (force → integrate against the port-trace basis → coarse geometry-generated SPD C → prolong).
3. **Reuse of the forward network's coarse levels.** The coarse Galerkin operator of the forward V-cycle defines a coarse Neumann solve by exact small factorization, and "M = coarse solve plus fine smoother" is then symmetric if the smoother is used symmetrically.

Option 3 shares components with the main line and with fallback B, which answers question (l) as well. The brief notes that the Neumann action "need not be very accurate", which fits.

---

## 6. Equivariance to the cubic group O_h (48 elements) on grids

### 6.1 G-CNN: Cohen & Welling, "Group Equivariant Convolutional Networks", ICML 2016, arXiv:1602.07576 (verified); 3D G-CNN: Winkels & Cohen, "3D G-CNNs for Pulmonary Nodule Detection", MIDL 2018, arXiv:1804.04656 (verified; the PDF text confirms the groups D4, D4h, O and **O_h**); CubeNet: Worrall & Brostow, arXiv:1804.04458 (verified; the 24 right-angle rotations)
- **Mechanism.** Lift features to functions on the group (48 copies for O_h). Filters are transformed by exact voxel permutations with sign flips for vector channels, and group convolutions keep exact equivariance. Winkels & Cohen report that O_h-CNNs reach baseline performance with about 10× less data.
- **Limits for us.** Memory and compute grow by up to 48× on the fine grid (65³ nodes × 3 components × channels × 48). The cut plane breaks per-cell symmetry, so equivariance is **joint** in (g, q), which G-CNNs handle but which removes no work.
- **Verdict: avoid full regular-representation lifting on the fine level.** It might be acceptable on the 9³ or 17³ levels.

### 6.2 Steerable and continuous: Weiler et al., "3D Steerable CNNs", arXiv:1807.02547 (verified; NeurIPS 2018 per my knowledge, **venue UNVERIFIED**); EqGINO, ICML 2026, arXiv:2606.03260 (verified); G-FNO: Helwig et al., ICML 2023, arXiv:2306.05697 (verified)
- **Verdict: avoid.** SO(3)-steerable filters are unnecessary (our symmetry is discrete, and the grid only supports O_h exactly). The equivariant spectral constructions (G-FNO, EqGINO) inherit the Fourier problems in 7.

### 6.3 Frame averaging and canonicalization: Puny et al., arXiv:2110.03336 (verified; ICLR 2022 per my knowledge, **venue UNVERIFIED**)
- **Mechanism.** Averaging f over a small frame F(x) ⊂ G gives exact equivariance. Full group averaging costs |G| = 48 passes. Canonicalization uses one pass but is discontinuous where the frame changes.
- **For us.** Canonicalizing by the cut-plane normal (map the normal into the fundamental domain of O_h, then average only over its stabilizer) is **constant within a design**, because τ is the design variable and the plane is fixed. Sensitivities dC/dτ therefore stay smooth. For FULL cells, canonicalizing by τ corners would be discontinuous in τ, so it is not allowed there.
- **Verdict: adapt for cut cells if augmentation proves insufficient. Keep it as a backup.**

### 6.4 Augmentation: Brandstetter, Welling & Worrall, "Lie Point Symmetry Data Augmentation for Neural PDE Solvers", ICML 2022, arXiv:2202.07643 (verified); Wang, Walters & Yu, arXiv:2002.03061 (verified; ICLR 2021 per my knowledge, **venue UNVERIFIED**)
- **Mechanism.** Augment by exact symmetries, applying the correct transformation to vector fields: u'(x) = ρ(h) u(h^{-1}x) for a displacement, with ρ the 3×3 signed permutation.
- **Our evidence.** EquiModel with 48-element augmentation succeeded on FULL cells, and rotation augmentation improved held-out cut cells (0.64 → 0.41). The CutFEM pipeline is equivariant to 1e-7.
- **Verdict: use (default for v0).** It costs nothing in architecture, and the exact pipeline provides consistent labels.

### 6.5 Equivariance by construction, specific to our operator
- **Construction** (derived from F6, 1.4 and 3.2, not from a single paper). K itself is O_h-equivariant: K_{h·i, h·j} = ρ(h) K_ij ρ(h)^T. So a linear stencil operator whose 3×3 blocks have the form W_ij = Σ_k a_k(inv_ij) B_ij^{(k)} is exactly equivariant, provided:
  - the scalar gates a_k are functions of **O_h-invariant** local features (for example norms or eigenvalues of normalized K blocks, moment invariants, mask counts, weak-node indicators);
  - the block basis B^{(k)} is built from equivariant objects (K_ij, the normalized D_i^{-1/2} K_ij D_j^{-1/2}, the identity, K_body vs K_ghost blocks, products such as K_ij K_jk along paths).
- This is "index-free" in EquiModel's sense and costs 1× (no group lifting). It also answers question (j) on memory: store a few scalars per node or edge, not full blocks. Q2 grids have several node classes (vertex, edge-mid, face-mid, centre), but O_h permutes them consistently, so the gates can depend on the node class.
- **Verdict: adapt; recommended for the fine level.** It combines exact equivariance with geometry-generated linear operators, and it reuses the exact K (moments to K blocks) as the equivariant carrier. Keep augmentation on top as a safety net for any non-equivariant part (for example Transolver slices).

---

## 7. High-contrast and near-singular elasticity; why spectral layers struggle

### 7.1 Spectral bias: Rahaman et al., "On the Spectral Bias of Neural Networks", ICML 2019, arXiv:1806.08734 (verified); HANO, arXiv:2210.10890 (verified)
- **Mechanism.** Gradient training fits low frequencies first. HANO shows this dominates multiscale operator learning and uses hierarchical attention plus an H¹ loss.
- **For us.** Combine with F5. Our hardest content (sliver modes) is both high-frequency and low-energy, the worst possible combination for a smooth network with an energy loss: spectral bias plus the conditioning of F4. **An architectural carrier for localized low-energy modes is therefore necessary.** It cannot be left to training.

### 7.2 Why FFT/spectral layers fail on masked, high-contrast bodies (all verified)
Nguyen & Schneider, "Universal Fourier Neural Operators for periodic homogenization problems in linear elasticity", J. Mech. Phys. Solids 206 (2026) 106418, arXiv:2507.12233; Kelly & Kalidindi, TherINO, arXiv:2411.06529 (CMAME 2025 per the publisher link in search); Walsh-Hadamard NO, arXiv:2511.07347; DAFNO, 2.6.
- **Mechanism, the clearest statement.** Nguyen & Schneider build an FNO that **exactly mimics the Moulinec-Suquet basic scheme** (the Lippmann-Schwinger fixed-point iteration with a homogeneous reference medium, whose Green's operator is diagonal in Fourier space). Its accuracy guarantee holds "only subject to a material-contrast constraint". That is the mechanism: a Fourier layer is the Green's operator of a **homogeneous** reference medium. The heterogeneity must be absorbed by iterating a polarization fixed point, whose contraction factor degrades with contrast and fails for voids (zero stiffness, which our empty space is). Their depth, equal to the iteration count, grows with contrast.
- **Other observations.** Discontinuities are broadband, so truncated modes give Gibbs artifacts (Walsh-Hadamard NO). Neural operators "struggle ... due to sharp transitions and high contrast", and TherINO's remedy is to **iterate in solution space with physics-derived (constitutive) encodings**, a fallback-A-like hybrid.
- **For us.** Void/solid contrast is infinite (inactive nodes) and, among active nodes, weak/strong stiffness ratios reach about 1e2 or more, since weak nodes have a diagonal < 1% of the median. Masks alone do not fix the translation-invariant kernel (2.6).
- **Verdict: avoid spectral layers at every level that resolves walls.** A coarse, smooth global channel does not need them either (2.5, coarse graph).

### 7.3 Contrast-robust multiscale bases: CEM-GMsFEM, Chung, Efendiev & Leung, arXiv:1704.03193 (verified; CMAME 2018 per my knowledge, **venue UNVERIFIED**); GenEO (5.7); learned adaptive BDDC/FETI-DP/GDSW (5.7)
- **Mechanism.** For high-contrast coefficients, a coarse space is contrast-robust only if it contains, per coarse region, the **eigenfunctions of local spectral problems with small eigenvalues** (high-conductivity channels, inclusions; in elasticity, thin stiff or soft features). CEM-GMsFEM then builds **energy-minimizing, exponentially decaying** multiscale basis functions on oversampled patches, and convergence becomes independent of contrast.
- **For us.** This is F5 in theory form. Our weak-node slivers play the role of "channels and inclusions", and their soft modes must be explicit coarse DOFs, computed or learned (5.7 a/b). The decay property also bounds how local the sliver handling can be. Oversampling by one or two element layers corresponds to "keep weak nodes paired with strong neighbours".
- **Verdict: use as the design rule for the fine level.** A local low-energy subspace per sliver patch, extended energy-minimally, with coefficients linear in q.

### 7.4 Conditioning of cut elements: de Prenter, Verhoosel, van Brummelen, Larson & Badia, "Stability and Conditioning of Immersed Finite Element Methods: Analysis and Remedies", Arch. Comput. Methods Eng. 30(6) (2023) 3617-3656, arXiv:2208.08538 (verified)
- **Mechanism.** Small cut elements cause loss of stability and severe ill-conditioning. The remedies are ghost penalty, aggregation, and **Schwarz-type (patch) preconditioning** that groups the DOFs supported on small cuts with neighbours.
- **Verdict: use.** It explains our weak nodes: γ = 1e-4 is a light ghost penalty, so the small-cut DOFs are held mostly by γ K_ghost. It confirms that the numerical-analysis remedy is patch-level (block) treatment, the same conclusion as F5 and 7.3. Aggregated unfitted FE (Badia et al., cited in the project docs) is the "extension" version, which relates to how coarse transfers should treat small cuts: slave them to interior neighbours.

---

## 8. Consequences for the open questions (a-l)

A condensed mapping of the above. These are research-informed constraints, not the final design.

| Q | Literature-informed answer |
|---|---|
| a | The fine level is a **linear, K-sparsity-masked stencil operator on the 65³ node grid**: 5×5×5 Q2 couplings plus ghost-penalty-face couplings, implemented as dense masked tensors. It is simultaneously the "graph" and the "grid" view (F6, 2.1, 2.4). Edge features are normalized K_body and K_ghost blocks; node features are weak-node indicators and moment invariants (1.4, 3.2). No radius or proximity edges, and no spectral layers (2.2, 2.6, 7.2). |
| b | Ports: exact masked overwrite (4.3). Rigid modes: exact projection and exact extension, with the network acting only on q_def (4.5). Rigid modes are carried on every coarse level (1.7). |
| c | Global transmission through **material-aware** coarse levels: graph or aggregation coarsening, split per connected component, energy-minimizing transfers (1.1, 1.7, 2.4). An optional geometry-only linear-attention channel at the coarsest level (2.5, 3.4). Localized soft modes through **patch/block fine operators plus local low-energy subspaces**, computed from exact K in the encoder (5.7, 7.3, 7.4). |
| d | The literature consistently favours a few exact-K applications for consistency and generalization (1.8 Hsieh, 1.9, 1.11, 7.2 TherINO, 3.2 Neumann-series NGO). K is available matrix-free from moments. My rough estimate: about 125 moments × active elements, so ≈10-20 MB per cell stored, and ≈10-30 GFLOP per application for FULL, i.e. sub-millisecond to about a millisecond on the 5090. This is an estimate, not measured. A pure feedforward net is allowed but has to learn the stiff near field, which F1 says to avoid. |
| e | S_hat q = E^T K E q: forward, then K, then the transpose of the linear net. It is symmetric PSD by construction (F2). With the net linear, the transpose can be an explicit transposed V-cycle (≈ forward cost). Avoid (K u)_ports, which is non-symmetric (3.3, 5.5 Parish). |
| f | Must be symmetric PSD with matched row and column spaces: M = B^T C B, rigid modes projected out (3.3, 5.10). Prefer reusing the forward net's coarse hierarchy. |
| g | Exact dK/dτ from moments is linear in the moment derivatives. Sensitivity is first order in the field error, so it is the field accuracy in the soft directions that matters (F4, F5). Topology is fixed within a design, which makes cut-plane canonicalization smooth (6.3). |
| h | Default: 48-element augmentation (6.4, our evidence). Upgrade: equivariance by construction through K-block bases with invariant gates, at 1× cost (6.5). Avoid regular-representation G-CNNs on the fine grid (6.1). |
| i | Energy loss normalized per sample = μ - 1 (F3). Counter the energy Hessian's ill-conditioning (F4) with a supervised field-L2 warm start, then an energy fine-tune (4.2, arXiv:2606.21828), and with a preconditioner-shaped parameterization. |
| j | Hypernetwork memory: share coefficients across layers, and emit per-node or per-edge scalars that gate K-derived blocks, not free blocks (3.7, 6.5). |
| k | The fp32 network is fine because readout errors are second order (F3). Energy and reaction reductions in fp64. |
| l | One component set serves three roles. Main line: a feedforward linear V-cycle with generated coefficients. Fallback A: the same cycle fed with exact residuals (Hsieh-style consistency, 1.8). Fallback B: the same cycle with only a few scalars learned (route 1). The coarse hierarchy doubles as the BDD Neumann preconditioner (5.10). |

---

## 9. Reference list with verification status

Verified means the arXiv abstract page was fetched and the title and authors matched, unless another method is stated.

| Ref | Id / venue | Status |
|---|---|---|
| MgNO (He, Liu, Xu) | arXiv:2310.19809; ICLR 2024 | verified (arXiv + ICLR proceedings/OpenReview via search); HTML read |
| MgNet (He, Xu) | arXiv:1901.10415; Sci. China Math. 62 (2019) | verified (journal-ref) |
| U-NO (Rahman, Ross, Azizzadenesheli) | arXiv:2204.11127; TMLR | arXiv verified; TMLR year UNVERIFIED |
| Luz et al. | arXiv:2003.05744; ICML 2020 (PMLR v119) | verified |
| Greenfeld et al. | arXiv:1902.10248; ICML 2019 | verified (arXiv comment) |
| Katrutsa et al. | arXiv:1711.03825 | verified (arXiv); venue not checked |
| Xu & Zikatanov | Comput. Visual. Sci. 7 (2004) 121-127 | verified (Springer page via search) |
| Vaněk, Mandel, Brezina | Computing 56 (1996) 179-196 | verified (Springer page via search) |
| Huang, Li, Xi | arXiv:2102.12071; SISC 45(3) (2023) | verified |
| Hsieh et al. | arXiv:1906.01200 | arXiv verified; ICLR 2019 UNVERIFIED |
| UGrid | arXiv:2408.04846; ICML 2024 | verified (arXiv comment) |
| Multipole GNO | arXiv:2006.09535; NeurIPS 2020 | arXiv verified; venue per project docs |
| HANO | arXiv:2210.10890 | arXiv verified; venue UNVERIFIED |
| Multiwavelet NO (Gupta et al.) | arXiv:2109.13459 | verified (arXiv) |
| M2NO | arXiv:2406.04822; KDD 2026 | verified (arXiv comment) |
| HINTS | arXiv:2208.13273; Nat. Mach. Intell. 6 (2024) 1303 | verified |
| Kopaničáková & Karniadakis | arXiv:2401.02016 | arXiv verified; venue UNVERIFIED |
| GNO (graph kernel network) | arXiv:2003.03485 | verified |
| Neural Operator (Kovachki et al.) | arXiv:2108.08481; JMLR 24 (2023) | verified |
| GINO | arXiv:2309.00583; NeurIPS 2023 | verified |
| Geo-FNO | arXiv:2207.05209; JMLR 24 (2023) | verified |
| MeshGraphNets | arXiv:2010.03409; ICLR 2021 | verified |
| MultiScale MeshGraphNets | arXiv:2210.00612; ICML 2022 workshop | verified |
| BSMS-GNN | arXiv:2210.02573 | arXiv verified; venue not checked |
| Transolver | arXiv:2402.02366; ICML 2024 | verified |
| DAFNO | arXiv:2305.00478; NeurIPS 2023 | verified |
| Unfitted FE interpolated NNs | arXiv:2501.17438 | verified |
| WINO | arXiv:2605.24651 | verified |
| Neural PDE solvers for irregular domains | arXiv:2211.03241 | verified |
| Daviet et al. | arXiv:2410.09417; ACM TOG 2025 | verified |
| Saberi, Zhao, Vogel | arXiv:2403.11632 | verified |
| VarMiON | arXiv:2209.12871; CMAME | arXiv verified; journal details UNVERIFIED |
| Neural Green's Operators | arXiv:2406.01857; CMAME 455 (2026) 118893 | verified; HTML read |
| Melchers, Dolean, Abdelmalik | arXiv:2605.19867 | verified; HTML read |
| Galerkin transformer (Cao) | arXiv:2105.14995; NeurIPS 2021 | verified |
| DeepONet | arXiv:1910.03193 | arXiv verified; Nat. Mach. Intell. UNVERIFIED |
| HyperDeepONet | arXiv:2312.15949; ICLR 2023 | verified |
| NOMAD | arXiv:2206.03551 | verified |
| Boullé & Townsend | arXiv:2102.00491 | arXiv verified; FoCM UNVERIFIED |
| Boullé, Earls, Townsend | arXiv:2105.00266 | verified |
| HyperNetworks | arXiv:1609.09106 | arXiv verified; ICLR 2017 UNVERIFIED |
| Deep Ritz | arXiv:1710.00211 | arXiv verified; journal UNVERIFIED |
| Deep Energy Method (Nguyen-Thanh et al.) | Eur. J. Mech. A/Solids 80 (2020) 103874 | verified (ScienceDirect/ADS via search) |
| Samaniego et al. | arXiv:1908.10407 | verified |
| VINO | arXiv:2411.06587 | verified |
| Müller & Zeinhofer (ENGD) | arXiv:2302.13163 | arXiv verified; ICML 2023 UNVERIFIED |
| Oh, Lee, Darbon, Karniadakis | arXiv:2606.21828 | verified |
| Sukumar & Srivastava | arXiv:2104.08426 | verified |
| Hard-constraint framework (Liu et al.) | arXiv:2210.03526; NeurIPS 2022 | verified |
| BOON (Saad et al.) | arXiv:2212.07477; ICLR 2023 | verified |
| BC via learned function extensions | arXiv:2602.04923 | verified |
| Enforcing BCs for PI neural operators | arXiv:2510.24557 | verified |
| PI graph neural Galerkin (Gao, Zahr, Wang) | arXiv:2107.12146 | arXiv verified; CMAME UNVERIFIED |
| PINO | arXiv:2111.03794 | verified |
| Convex neural energy elements | arXiv:2608.02036 | verified |
| SNI | arXiv:2504.00510; ICLR 2026 | verified |
| NEST | arXiv:2605.12343 | verified |
| Non-overlapping Schwarz FE-NO | arXiv:2606.08796 | verified |
| DD-DeepONet | arXiv:2508.02717 | arXiv verified; journal version UNVERIFIED |
| Mosaic Flows | arXiv:2104.10873; CMAME 389 (2022) 114424 | verified |
| Parish et al. (SPSD ML elements) | arXiv:2307.05434 | verified |
| NOEM | arXiv:2506.18427 | verified |
| Beatson et al. | arXiv:2005.06549 | verified |
| Du & Stechmann | arXiv:2308.02467 | verified |
| PPDNO (Ling, Ying, Zhou) | arXiv:2606.25952 | verified |
| Taghibakhshi et al. | arXiv:2205.09833; NeurIPS 2022 | verified |
| MG-GNN | arXiv:2301.11378; ICML 2023 | verified |
| Mandel BDD | Commun. Numer. Methods Eng. 9(3) (1993) 233-241 | verified (Wiley page via search) |
| GenEO (Spillane et al.) | Numer. Math. 126(4) (2014) 741-770 | verified (Springer page via search) |
| Heinlein, Klawonn, Lanser, Weber (ML + adaptive FETI-DP) | SISC doi 10.1137/20m1344913 | title verified via search; details UNVERIFIED |
| Learning adaptive constraints in nonlinear FETI-DP | arXiv:2312.14252 | listed in search results; not fetched (UNVERIFIED) |
| Klawonn, Lanser, Weber-Hamacher | arXiv:2607.06261 | verified |
| Chung, Kim, Lam, Zhao (learned BDDC) | arXiv:2104.09162 | arXiv verified; venue UNVERIFIED |
| Li, Chen, Du, Matusik | arXiv:2305.16432; ICML 2023 | verified |
| NeuralIF | arXiv:2305.16368; TMLR | verified |
| Trifonov et al. (GNN preconditioner) | arXiv:2405.15557 | verified |
| Graph Neural Preconditioners (Chen) | arXiv:2406.00809; ICLR 2025 | verified |
| MP-GNNs fail on factorizations | arXiv:2502.01397; TMLR 2026 | verified |
| G-CNN | arXiv:1602.07576; ICML 2016 | verified |
| 3D G-CNN (Winkels, Cohen) | arXiv:1804.04656; MIDL 2018 | verified; O_h confirmed in PDF text |
| CubeNet | arXiv:1804.04458 | verified |
| 3D Steerable CNNs | arXiv:1807.02547 | arXiv verified; NeurIPS 2018 UNVERIFIED |
| G-FNO | arXiv:2306.05697; ICML 2023 | verified |
| EqGINO | arXiv:2606.03260; ICML 2026 | verified |
| Frame averaging | arXiv:2110.03336 | arXiv verified; ICLR 2022 UNVERIFIED |
| Lie point symmetry augmentation | arXiv:2202.07643; ICML 2022 | verified |
| Wang, Walters, Yu | arXiv:2002.03061 | arXiv verified; ICLR 2021 UNVERIFIED |
| Spectral bias (Rahaman et al.) | arXiv:1806.08734; ICML 2019 | verified |
| Universal FNO for homogenization (Nguyen, Schneider) | arXiv:2507.12233; JMPS 206 (2026) 106418 | verified |
| TherINO (Kelly, Kalidindi) | arXiv:2411.06529 | arXiv verified; CMAME 2025 per publisher link |
| Walsh-Hadamard NO | arXiv:2511.07347 | verified |
| CEM-GMsFEM | arXiv:1704.03193 | arXiv verified; CMAME UNVERIFIED |
| de Prenter et al. | arXiv:2208.08538; Arch. Comput. Methods Eng. 30 (2023) 3617-3656 | verified |
| Schäfer, Owhadi | arXiv:2110.05351 | verified |
| Schäfer, Katzfuss, Owhadi | arXiv:2004.14455 | verified |
| Liu-Schiaffini et al. (local kernels) | arXiv:2402.16845; ICML 2024 | verified (not used in the verdicts) |
| CNO (Raonić et al.) | arXiv:2302.01178 | verified (not used in the verdicts) |
| GNOT | arXiv:2302.14376 | verified (not used in the verdicts) |
