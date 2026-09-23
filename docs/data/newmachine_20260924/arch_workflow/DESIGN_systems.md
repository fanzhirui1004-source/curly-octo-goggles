# DESIGN_systems: v0 architecture designed from the deployment side

Lens: systems and lattice integration. The requirement is 100 resident cells on one 32 GB RTX 5090. A BDD lattice solve
needs, per cell, a forward action S q, a symmetric readout Eᵀ K E q, an approximate Neumann action and the
sensitivities −uᵀ(∂K/∂τ)u. Topology changes under τ steps.
This is nevertheless a complete v0 architecture, not only the deployment part.

Evidence labels:
- **[brief]**: ARCH_BRIEF.md.
- **[DI]**: DATA_INTERFACE.md, recomputed counts.
- **[RP]**: ROUTES_PROGRESS_20260923_CN.md.
- **[AH]**: ARCHITECTURE_HISTORY_20260923_CN.md.
- **[RS]**: RESEARCH.md.
- **[est]**: my estimate. Every FLOP, latency and memory number below is an estimate from operation counts, not a
  measurement. The first engineering task (§7, risk 6) is a microbenchmark that replaces them.

---

## 0. Position in one page

**Name:** GLMO, the Geometry-gated Linear Multilevel Operator.

**What it is.** A fixed-depth, feedforward network, strictly linear in q.
- Its q-path is a symmetric multilevel operator on the exact finite-element hierarchy that the fixed grid already
  contains: Q2 on 32³ → 16³ → 8³ → 4³, nested, with 65³ → 33³ → 17³ → 9³ node grids.
- Messages on every level are exact element-stiffness actions, applied matrix-free from element moments.
- A nonlinear geometry hypernetwork does not predict operators. It emits positive scalar **gates** and **spectral-filter
  weights**, which modulate exact, equivariant carriers:
  - Jacobi blocks;
  - local soft-mode (sliver) subspaces computed from K;
  - Galerkin coarse operators;
  - polynomial smoothers.
- Global transmission comes from an exact dense solve at the 9³ level inside every cycle.
- Localized soft modes are carried by exact patch eigenspaces on two symmetric, overlapping block families.

**How many exact K applications belong inside the forward pass: about 10.** For one forward extension this is 9 full
applications plus 1 restricted to the elements touching a port. A symmetric S q therefore needs about 21 (two passes
plus the readout), and a Neumann action needs 4. Three reasons:
1. **Cost.** On the fine level a matrix-free exact K application costs about as much as one learned vector-channel
   hyperedge layer. Both move the same 27- and 45-node gathers and scatters; see §2.11. The budget therefore does not
   favour learned fine layers over exact K.
2. **Accuracy.** A pure feed-forward field has to cancel near-field energy to a relative 1e-4 or better, because
   q^T K_ΓΓ q / q^T S q is about 1e3–1e5 in soft directions [brief §4, AH]. Entry-level learned precision of 1e-4 is the
   wall that ended phases 0–7. With exact-residual smoothing, the stiff and boundary-layer error is contracted
   **independently of the learned coefficients**. The learned parts then need to be right to about 10–30%, not to 1e-4.
3. **Why not more.** Going above about 24 applications (the fallback A/B regime of route 1) costs roughly twice the
   lattice budget. The same code reaches it by raising the cycle count m, so no second architecture is needed.

**Headline numbers for a FULL cell, fp32 [est].**

| quantity | value |
|---|---|
| resident per cell | ≈ 75 MB, so 100 cells ≈ 7 GB |
| encode per design step | ≈ 0.1–0.3 s on top of the exact moments (≈ 1 s), no factorization |
| symmetric S q | ≈ 32 K-equivalents ≈ 1.6–3.2 ms per column, batched over cells |
| Neumann action | ≈ 8 K-equivalents ≈ 0.4–0.8 ms |
| geometry network | ≈ 0.6 M shared parameters |
| per-cell generated numbers | ≈ 0.3 M |
| one design iteration, 100 cells, balanced BDD | ≈ 15–30 s per load case for the network part |

- A K-equivalent (K_eq) is one exact fine-level K application on one column: about 1 GFLOP.
- The design iteration is also charged about 80–100 s for the exact moments and their τ-derivatives. That cost is
  inherited from decision 1 and dominates.

**Guarantees, by construction:**
- exact linearity in q;
- exact port values on box ∪ band;
- exact rigid modes;
- S_hat symmetric positive semidefinite;
- μ(q) ≥ 1 (the upper bound);
- exact joint O_h equivariance, up to the ~1e-4 asymmetry of polyref;
- a symmetric positive definite Neumann action whose BDD condition number is bounded by the V-cycle's spectral bounds;
- a **free, two-sided error indicator** on every S q application, and at lattice level an estimate of the compliance
  error with no extra work (§4.7).

---

## 1. Requirements (derived; each tied to evidence)

| # | Requirement | Evidence |
|---|---|---|
| R1 | Output field linear in q exactly. Nonlinearity only in the geometry path, which generates coefficients. | Decision 5; F2 in [RS]; nonlinear preconditioners break PCG [RS 3.3, 5.8] |
| R2 | Hard, exact port values on box ∪ Γ-band, with the port set given as a **mask input**. The same weights must also run with Γ = box only and Γ = ∅ (Neumann). | Decisions 3, 4; F3 (the Ritz identity needs exact ports); the owner's design doc §3 (q = (q_box, q_cut)) |
| R3 | Exact rigid modes: E R_Γ β = R β and S_hat R_Γ = 0, in fp64 where it matters. | Decision 6; R7_15: an unprojected rigid residual gave a 12× energy error in a 2-cell lattice [RP 7.6] |
| R4 | S_hat symmetric positive semidefinite **by construction**, with an explicit adjoint that is cheap in memory. | BDD-PCG [RP 7.7]; Parish, Melchers [RS 3.3, 5.5]; brief question e |
| R5 | One-sided, second-order error: S_hat ≥ S, μ ≥ 1, and an upper bound on lattice stiffness. | Decision 1; brief §4 (1-D example: 32% field error → 1% stiffness error) |
| R6 | Lattice compliance and sensitivity ≤ 3%. Operationally: μ − 1 ≲ 1e-2 in the softest decile of lattice-loaded directions. Field energy-norm error of a few % in loaded directions; the sensitivity error was measured at ≈ 1.3–1.5× the compliance error. | Brief §1; route-1 D2 lattice table [RP] (0.72/1.7% compliance vs 0.93/2.6% sensitivity) |
| R7 | Stiff, near-field and boundary-layer errors must not depend on learned precision: an exact K inside the q-path, and rigid-exact arithmetic in the readout. | Soft stiffness comes from 1e3–1e4 near/far-field cancellation [AH §1]; an fp32 matrix is off by 0.8–2.2% [brief §4]; entry precision of 3e-4 is unreachable [AH phase 4] |
| R8 | A fine-level carrier for **localized** soft modes. It must group weak nodes with their strong neighbours, with **per-group** damping, and have capacity for the whole dense band. | 17.6–21.9% weak nodes carry 94–99.4% of the slowest modes, each 11–30 nodes, 128 modes in [0.0039, 0.0062] [brief §4, RP]; grouping helps, splitting hurts; a global step crushes exact blocks [RP] |
| R9 | Global transmission in every application, with coarse spaces rich enough for shell bending (quadratic). | Forces must cross the cell [brief §4]; route-1 affine vs quadratic coarse spaces: 7.75 vs 1.26 at 64 layers [RP E0] |
| R10 | Coverage of force-driven directions in training and in validation. | Skeleton: 0.3% on probes, 20–35% on force-driven loads [brief §4, RP D2] |
| R11 | Approximate Neumann action: SPD on the balanced space, spectrally equivalent to S⁺, cost ≪ one S q. | BDD needs 11–53 iterations; forward-only preconditioning needs 300–1600 [RP 7.7] |
| R12 | ≤ ~100 MB resident per FULL cell. No per-pair storage, no dense element matrices, no factors. | [DI §3]: a K_e store is 171–338 MB and an fp32 factor 6.6 GB per cell |
| R13 | S q ≲ 3 ms per column per FULL cell, amortized over a batch across cells. Neumann action ≲ 1 ms. | Brief §6 (~2e4 applications per design iteration) |
| R14 | Encode per design step ≲ 0.5 s beyond the moments, batched, with no factorization memory. | Brief §6; [RP 7.5] (exact design step 0.33–1.58 s, 3–14.5 GB) |
| R15 | Topology changes are handled by masks over a superset per design band. The operator is continuous within a topology. Sensitivities are defined within a topology, and ∂K_ghost/∂τ = 0. | Brief §3; [RP 7.5] (±1e-3 changes ±4–6 elements); [DI 1.5, 1.6] |
| R16 | Joint O_h equivariance of (g, q). | Pipeline equivariant to 1e-7; augmentation helped held-out cut cells (0.64 → 0.41) [brief §5] |
| R17 | fp32 network with fp64 reductions, **no TF32**, and rigid-exact K application. | [DI §4 precision]; R7_15 |
| R18 | Training: per-sample normalized energy loss (μ − 1), no entry losses, smooth coefficient heads, adversarial directions, a warm start at a working solver. | Decision 7; [AH phases 0, 4, 7] (non-smooth heads, entry-loss conflict); route-1 E1b learned 2 scalars/layer label-free |
| R19 | Degrades gracefully into fallback A or B while sharing components and code. | Decision 8 |
| R20 | The cut band is a port from day one. Cut loads are supported, computed and reported. | Decision 4 |
| R21 | Cost is independent of the number of port DOFs. | Decision 2 |
| R22 | The operator is a pure function of (g, mode): deterministic, cacheable, reproducible. No history-dependent state such as warm-started eigensolvers. | Needed for sensitivity consistency and lattice caching [inference] |

---

## 2. Architecture

### 2.0 Notation, modes, sizes

**Grids and levels.**
- Fine level ℓ = 0: Q2 on 32³ elements, node grid 65³, N active nodes, 3 DOF per node.
- Coarse levels ℓ = 1, 2, 3: Q2 on 16³, 8³ and 4³ elements, node grids 33³, 17³ and 9³.
- The coarse spaces are **nested**: a Q2 polynomial on a coarse element is Q2 on each of its 8 children.

**Masks.**
- 𝔇 is the set of Dirichlet nodes for the current **mode**. I is the set of the remaining active nodes.
- Modes:
  - **X** (extension): 𝔇 = Γ = box ∪ Γ-band (default), or box only.
  - **N** (Neumann): 𝔇 = ∅ for a floating cell, or the clamped face nodes for a cell touching a lattice support.
- The Γ-band is all 27 nodes of the active elements whose plane section carries positive-area material. Box
  membership takes precedence [DI 2.3].

**Other symbols.**
- R: the 6 rigid fields on all active nodes; R_Γ: their restriction to Γ.
- K: the exact CutFEM matrix K_body + γ K_ghost, **never assembled**.

**Sizes** [DI, q2_hierarchy_counts.json]:

| | FULL (τ = 0.5 stand-in; real parent 122,748 nodes) | cut, retained 0.758 (≈ 0021) | cut, retained 0.148 (≈ 0013) |
|---|---|---|---|
| fine elements E / ghost faces F | 12,880 / 21,936 | ~9.6k / ~16.5k | ~2.3k / ~3.5k |
| fine nodes N / DOF | 121k / 363k | 91.5k / 274k | 23k / 69k |
| box port nodes / Γ-band nodes | 7,872 / 0 | 5,434 / 6,328 | 1,844 / 6,883 |
| weak nodes (‖D_i‖ < 1% of the median) | ~20k (16.7%) | ~16k | ~4.2k |
| level 1 (Q2 16³): elements / nodes | 2,080 / 21,196 | 1,596 / 16,560 | 402 / 4,488 |
| level 2 (Q2 8³): elements / nodes | 352 / 3,886 | 284 / 3,243 | 79 / 1,005 |
| level 3 (Q2 4³): elements / nodes | 64 / 729 | 56 / 667 | 18 / 269 |
| fine elements touching Γ | 1,776 | 2,232 | 1,290 |

### 2.1 Data flow, end to end

```
 GEOMETRY PATH (once per cell per design step; nonlinear in g; no q)
 tau corners, plane, superset-band masks
   |  CPU fast_prep3 topology (existing, ~1 s, overlaps GPU)
   v
 moments M (E x 125, fp64)  --polyref (existing)-->  dM/dtau (E x 125 x 8, transient, only for sensitivities)
   |
   +--> EXACT products (deterministic, no learning)
   |      w = V^-1 M (signed 5^3 Gauss weights, E x 125 fp32)            -> matrix-free K apply
   |      D_i (3x3 diagonal blocks, N x 6) -> Jacobi carrier, weak indicator
   |      coarse moments M_1, M_2, M_3 by exact binomial shift (Prop. 2.4)  -> Galerkin A_1, A_2, A_3 (body)
   |      coarse ghost templates by sub-face pattern                       -> Galerkin ghost part
   |      Gamma-band masked-prolongation corrections (cut cells, mode X)
   |      patch spectral spaces V_p, Lambda_p (two O_h-symmetric block families; mode X / N variants)
   |      Lanczos lambda_max bounds per level (for smoother scaling)
   |      coarsest Cholesky: A_3^X (<= 1,029 DOF), A_3^N (+ rigid regularization, <= 2,187 DOF)
   |
   +--> LEARNED geometry hypernetwork G_theta (invariant features -> element MP -> coarse U-graph -> heads)
          outputs only positive invariant scalars:
          omega_i (fine node gates), (sigma_p, theta_p) patch filter, g^(c)_{1,i}, g^(c)_{2,i} level gates,
          global head: smoother polynomial and cycle-polynomial coefficients
   |
   v
 CELL RECORD (~75 MB FULL): topology, w, D^-1, gates, patches, level-1 moments, A_2 BSR, A_3 factors, bounds

 Q-PATH (per column; linear in q; batched over all cells as a disjoint union)
 q on Gamma (fp64 from lattice) -> rigid split (fp64): alpha, q_d = Pi q
   -> lift residual  b = -(K [q_d; 0])_I      (K restricted to Gamma-touching elements)
   -> x = N_m b,  N_m = cycle polynomial in the symmetric V-cycle V:
        y = V b ;  z = V (K y) ;  x = a1 y + a2 z          (m = 2: 9 full K in total)
      V = symmetric V(1,1) over levels 0..3:
        fine pre-smoother (degree-2 polynomial in S0 K; S0 = gated Jacobi + patch spectral)
        -> exact residual -> P1 [level-1 V-cycle: gated multichannel polynomial smoothers on exact A_1, A_2;
           exact A_3^-1] P1^T -> exact residual -> fine post-smoother (= transpose of pre)
   -> u = R alpha + [q on Gamma; x + (R alpha)_I]      (ports overwritten with q: exact)
 READOUT
   v = K u (rigid-exact fp32 with fp64 local rigid removal and fp64 accumulation)
   S_hat q = Pi [ v_Gamma - K_Gamma,I N_m v_I ]          (the adjoint is the same symmetric N_m: no activation storage)
   free error indicator  eta^2 = v_I^T N_m v_I           (Sec. 4.7)
 NEUMANN ACTION (BDD)
   z_Gamma = Pi [ V^N (J_Gamma Pi r) ]_Gamma             (one V-cycle in mode N; SPD; Sec. 2.7)
 SENSITIVITY
   g_c = - sum_e sum_q (V^-1 dM_e/dtau_c)_q eps_q(u)^T C eps_q(u)   (strain energy densities at Gauss points)
```

### 2.2 The cell record: what is stored per cell (FULL, fp32 unless noted) [est]

| object | shape | MB | notes |
|---|---|---|---|
| element→node map, ghost faces, coarse maps, patch index lists | E×27 int32, F×3, … | ≈ 3 | the topology *is* the graph; no edge lists [DI 2.2] |
| signed Gauss weights w | E×125 | 6.4 | K_e = Σ_q w_{e,q} B_qᵀ C B_q exactly [DI 1.4]; used by every fine K application |
| D_i⁻¹ (symmetric 3×3) | N×6 | 2.9 | exact, from moments and ghost diagonal |
| fine gates ω_i | N | 0.5 | learned, invariant scalar |
| patch spectral spaces | ≈ 3,000 patches × ≤ 192 DOF × 16 | ≈ 23 (+ ≈ 4 for port-touching patches in both modes) | exact, from K; §2.5.3 |
| patch filter parameters (σ_p, θ_p) | P×2 | 0.02 | learned |
| level-1 coarse moments, D₁⁻¹, gates (C₁ = 2), ghost pattern ids | 2,080×125; 21k×6; 21k×2 | ≈ 2 (+ ≤ 5 band corrections, cut cells only) | A₁ is applied matrix-free |
| level-2 Galerkin A₂ (block sparse, 3×3 blocks), D₂⁻¹, gates (C₂ = 4) | ≈ 3.9k nodes × ≤ 125 nbrs × 9 | ≈ 18 | assembled at encode time |
| level-3 Cholesky factors | A₃^X ≤ 1,029², A₃^N ≤ 2,187² (packed) | ≈ 12 | fp32; fp64 is an optional knob (+12 MB) |
| Lanczos bounds, global head outputs | O(100) | ~0 | |
| **total** | | **≈ 70–75** | cut 0.758: ≈ 55; cut 0.148: ≈ 15 |

- 100 cells (80 FULL + 20 cut) come to about **6.5 GB**. For comparison, one exact fp32 factor is 6.6 GB.
- Transient during encode: dM/dτ (E×125×8 = 51 MB), processed one cell at a time, and the geometry-net activations
  (< 50 MB).

### 2.3 Geometry path

#### 2.3.1 Exact, deterministic part (no learning)
1. **Moments** M (E×125, fp64) come from the existing polyref GPU encoder: 0.24 s (0013), 0.79 s (0021), 0.98 s
   (FULL) [RP 7.5].
2. **Gauss weights** w = V⁻¹M, with V the 125×125 Vandermonde matrix of the 5³ Gauss–Legendre points and
   cond(V) = 1.2e4. Computed in fp64, then stored in fp32. The 4 null moments (degrees 11 and 12) drop out
   automatically [DI 1.4].
3. **Diagonal blocks** D_i = Σ_e (K_e)_ii + γ Σ_f (G)_ii come from a fixed 125×27×9 slice of the template plus a
   constant per face axis [DI 2.4]. The weak indicator is log₁₀(‖D_i‖ / median).
4. **Coarse Galerkin operators** by the coarse-moment identity (Prop. 2.4) and ghost-pattern templates (§2.4).
5. **Patch spectral spaces** (§2.5.3): a batched `eigh` of Jacobi-scaled patch matrices.
6. **Lanczos λ_max estimates** of S₀K, S₁A₁ and S₂A₂: 12 steps each, about 15 fine K applications per cell in total.
   They are used to scale the smoothers (stop-gradient).
7. **Coarsest factorizations**: Cholesky of A₃^X and of A₃^N + α R₃R₃ᵀ, plus clamped variants when a cell touches a
   lattice support.

#### 2.3.2 Learned hypernetwork G_θ: nonlinear, O_h-invariant, about 0.6 M parameters

All inputs are invariant scalars attached to elements, nodes or faces, so the network commutes with the cube group by
construction (§4.6).

**Element features** (≈ 32 per element):
- volume fraction and its log;
- eigenvalues of the normalized second-moment tensor about the centroid (3), and the centroid offset norm;
- the 6 eigenvalues of the element's uniform-strain stiffness C_e = [ε_iᵀ K_e ε_j]. This matrix is 6×6, linear in the
  moments, and its eigenvalues are O_h invariants;
- full, cut and band flags;
- τ at the centre and |∇τ|;
- signed plane distance divided by h;
- number of active face neighbours and of ghost faces;
- count, minimum and mean of the weakness of its 27 nodes.

**Node features** (≈ 12 per node):
- log‖D_i‖ and the eigenvalues of D_i;
- parity class (vertex, edge-mid, face-mid, centre);
- box, band and in-material flags;
- number of incident active elements.

**Trunk:**
- **Element encoder:** MLP 32 → 64 → 64 (≈ 6k parameters).
- **Element-graph message passing:** 6 layers, width 64.
  - Graph: active elements of the design-band superset, face adjacency (≤ 6), and ghost faces as edge features.
  - **Isotropic sum aggregation**, with invariant edge features only: ghost flag, full/full flag, the two volume
    fractions, tr(Σ_iΣ_j)/(‖Σ_i‖‖Σ_j‖) of the neighbours' second-moment tensors.
  - Message MLP 129 → 64 → 64 and update MLP 128 → 64 → 64: ≈ 25k parameters per layer, ≈ 150k in total.
  - Cost: about 2.4 GFLOP per layer (96k directed edges), ≈ 14 GFLOP forward, ≈ 1–3 ms.
- **Coarse U-graph:**
  - Pool to level-1 coarse elements (mean ‖ max → 128). Run 3 message-passing layers on the coarse face graph.
  - Pool to level 2 and run 3 layers. Pool to level 3 (≤ 64 tokens) and run 2 layers of full self-attention with
    invariant positional features |x − c|₂ and |x − c|_∞.
  - Unpool with skip connections back to the fine elements.
  - ≈ 350k parameters; ≈ 1 GFLOP.
- **Heads** (smooth, bounded; no max or abs; [AH phase 7]):
  - **Fine node gate:** ω_i = exp(0.7 tanh(h_i)) ∈ [0.50, 2.0], a multiplier on the theory value. h_i is an MLP of
    [mean of the 27-incident element embeddings, node features]: 80 → 64 → 1.
  - **Patch filter:**
    - σ_p = exp(0.7 tanh(·)) is a multiplier on the patch correction;
    - θ_p ∈ [0.03, 0.3] is the spectral cutoff (sigmoid map);
    - input: the mean embedding of the patch nodes.
  - **Level gates:** g^{(c)}_{ℓ,i} = exp(0.7 tanh(·)) for C₁ = 2 channels at level 1 and C₂ = 4 at level 2. Input:
    the mean of the incident coarse-element embeddings.
  - **Global head:** the pooled level-3 tokens (mean) feed an MLP. It outputs:
    - the smoother polynomial coefficients per level and channel, as deviations around their Chebyshev values;
    - the cycle-polynomial coefficients (a₁, a₂), as deviations around (2, −1).
    Bounded parameterizations keep every polynomial non-negative on [0, 1] (§2.5.2).

**Initialization.** The last layer of every head is zero, so that at initialization every gate is 1 and every
coefficient takes its theory value. **The untrained network is a working multigrid.**
- It has μ ≥ 1 and a finite error: the zero-learning baseline.
- Training starts from route-1-like quality, not from noise (R18).

**Output volume per FULL cell:** ≈ 121k + 6k + 42k + 16k + 100 ≈ 0.19–0.3 M numbers.

### 2.4 The exact multilevel hierarchy: Galerkin operators without assembling K

**Prolongations.** P_{ℓ+1→ℓ} is the fixed Q2 → Q2 interpolation from a coarse element to its 8 children. It is
tensor-product sum-factorized with a 1-D 3→5 stencil.
- Coarse nodes are active if their prolongation touches an active fine node.
- In mode X, coarse nodes on box faces are excluded. Coarse functions that vanish at the coarse face nodes vanish on
  the whole face, so they are consistent with fine Dirichlet box ports.
- Nodes on box planes that are not port nodes are merely over-constrained at coarse levels. The subspace becomes
  slightly stiffer, and the result is still exactly Galerkin.

**Proposition 2.4 (coarse body operators are "template × coarse moments").** Let ξ_E ∈ [−1, 1]³ be the local
coordinates of a coarse element E, and M_{E,abc} = ∫_{E ∩ material} ξ_E^a η_E^b ζ_E^c dx. Then

  Σ_{children e ⊂ E} P_eᵀ K_e^{body} P_e = Σ_m M_{E,m} T_m^{(H)},

where T^{(H)} is the fine template rescaled by (h/H)². The coarse moments follow exactly from the children's moments:
ξ_E = (ξ_e + s)/2 with s ∈ {±1}³, and binomial expansion gives 8 fixed 125×125 matrices.

*Proof.* The fine operator is K_e = Σ_m M_{e,m} T_m because ∂N_i ∂N_j is a polynomial of degree ≤ 4 in each variable
[DI 1.4]. A coarse Q2 basis function restricted to a child is a Q2 polynomial on that child, so the entry
P_eᵀK_eP_e is the same moment functional applied to the product of derivatives of coarse basis functions. That product
is a degree-≤ 4-per-variable polynomial in ξ_E, and after the affine change of variables it is still one in ξ_e.
Summing over the children gives the integral over E ∩ material, which is Σ_m M_{E,m} T_m^{(H)}. The coarse moments of
degree ≤ 4 per variable involve only child moments of degree ≤ 4 per variable, so no information is lost. ∎

Consequences:
- A₁ is applied **matrix-free**: the same strain-form kernel as the fine level, on 2,080 elements.
- The coarse operators are exactly the Galerkin operators of the discrete fine K, polyref error included, so the
  V-cycle is a genuine Galerkin multigrid.
- Rigid modes lie in every coarse space and are annihilated at every level (R3 at every level).

**Ghost part of the coarse operators.**
- A fine ghost face lying **inside** a coarse element contributes nothing. The coarse function is one polynomial across
  that face, so all derivative jumps vanish. This rests on the inference in [DI 1.5] that F measures normal-derivative
  jumps; the stored F_axis template has to be checked for it (§7, risk 4).
- Fine ghost faces lying **on** a coarse face contribute P_sᵀ FᵀF P_s. Here P_s is a fixed 135×135 map from the 45
  coarse nodes of the two coarse elements to the 45-node fine stencil, for each axis and each of the 4 sub-face
  positions.
- Level 1: the 16 on/off sub-face patterns × 3 axes give **48 precomputed 135×135 matrices**. Each coarse face
  selects one by its pattern id, and the apply costs one 135×135 GEMV per coarse face (≈ 3.5k faces ≈ 0.13 GFLOP).
- Levels 2 and 3 are assembled at encode time by accumulating (F P_s)ᵀ(F P_s) over the fine ghost faces on
  level-2/3 coarse planes, with 48 templates per level (≈ 5k faces, ≈ 10 GFLOP, ≈ 1 ms).
- A₂ is stored as a block-sparse matrix with 3×3 blocks (≈ 18 MB). A₃ is stored densely and factorized.

**Γ-band correction** (cut cells, mode X). In mode X the band nodes are Dirichlet, but coarse functions do not vanish
on an oblique band. The masked prolongation P̄ = M_I P then gives

  A_ℓ^X = A_ℓ^{template} − Σ_{e touching Γ} P_eᵀ(K_e − M_I K_e M_I) P_e.

- The correction needs explicit K_e for the ≤ 2.2k fine elements that touch Γ, about 1.7 GFLOP at encode time.
- It is stored per affected level-1 element as a dense 81×81 correction, ≤ 5 MB.
- Levels 2 and 3 are formed by Galerkin products of the corrected level-1 element matrices.

### 2.5 The q-path

Every primitive below is linear, has an explicit transpose, and **saves no input** at deployment (a linear
autograd.Function whose backward is the transpose operator). Memory for the adjoint is one working set.

#### 2.5.1 Primitives and costs (FULL, per column) [est]

| primitive | definition | GFLOP | K_eq | transpose |
|---|---|---|---|---|
| K (fine, strain form) | gather 27 → sum-factorized gradients at 5³ Gauss points → σ = C ε weighted by w_{e,q} → Bᵀσ → scatter; plus the ghost term via the 54×135 factor per face | 0.37 + 0.64 | 1 | self |
| K_{·,Γ}q_d (lift) | K applied to [q_d; 0], restricted to the 1,776 Γ-touching elements and their faces | 0.15 | 0.15 | K_{Γ,·} |
| S₀ | ω_i D_i⁻¹ ω_i (3×3) + Σ_p R_pᵀ V_p Φ_p V_pᵀ R_p | 0.004 + ≈ 0.05 | 0.06 | self |
| P₁ / P₁ᵀ | Q2 16³ ↔ 32³ sum-factorized interpolation, masked in mode X | ≈ 0.02 | 0.02 | each other |
| A₁ | strain form on 2,080 elements + 48-template ghost | ≈ 0.06 + 0.13 | 0.2 | self |
| A₂ | block-sparse matrix-vector product | ≈ 0.009 | 0.01 | self |
| A₃⁻¹ | two fp32 triangular solves with the ≤ 1,029² packed factor | ≈ 0.004 | ~0 | self |

K_eq is taken as 0.05–0.1 ms per column in fp32 when batched: about 1 GFLOP per column, ≥ 64 columns in flight, custom
kernels. **This is the number most in need of measurement** (§7, risk 6).

#### 2.5.2 Smoothers: gated, multi-channel, polynomial (the "multiscale GNO")

**Fine level.**
- Carrier: S₀ = Ω D⁻¹ Ω + Π_patch, with Ω = diag(ω_i I₃).
- The smoother applied is the degree-2 symmetric polynomial

  Ŝ₀ = s₀ (c₁ S₀ + c₂ S₀ K S₀), with s₀ = 1/(1.1 λ̂_max(S₀K)).

  Its coefficients (c₁, c₂) are initialized at the degree-2 Chebyshev smoother values on [λ̂_max/10, λ̂_max].
- Each application costs 1 K plus 2 S₀.

**Levels ℓ = 1, 2.** For c = 1..C_ℓ, with W_c = G_c D_ℓ^{−1/2} and G_c = diag(g^{(c)}_{ℓ,i}):

  S_ℓ = Σ_c W_c p_{ℓ,c}(W_c A_ℓ W_c) W_c.

- Here p_{ℓ,c}(t) = Σ_{j=0}^{2} a_{ℓ,c,j} t^j (degree 3 in the smoother sense, i.e. 2 A-applications) on the
  Lanczos-normalized interval.
- Parameterization: coefficients are deviations from a Chebyshev reference. A smooth barrier keeps
  min_{t∈[0,1]} p(t) ≥ 0 and the scaled λ_max(S_ℓA_ℓ) ≤ 1.9, which guarantees the smoother condition used in §4.
- Choices: C₁ = 2, C₂ = 4.
- This is linear message passing on the level-ℓ K-hypergraph. The kernel is the exact Galerkin stiffness, gated per
  node by the geometry, and each channel is a differently weighted local inverse that the hypernetwork can steer to
  where it is needed (slivers, thin walls, near ports).
- It is the only place where learned **channel width** lives, because it is cheap there: level-1 channels cost
  ≈ 0.2 K_eq per A₁ application.

#### 2.5.3 Sliver carrier: patch spectral spaces (the fine-level answer to R8)

**Patch families.** Two O_h-symmetric partitions of the 32³ element grid into 2×2×2 element blocks:
- 𝔅₀: blocks aligned at 0;
- 𝔅₁: blocks shifted by one element, with half blocks at the boundary.

Both partitions map to themselves under the 48 cube symmetries about the centre. Together they overlap, so any
11–30-node mode lies inside at least one block unless it spans more than about 5 nodes in some direction.

**Patch membership.** Patch p (one per block that contains weak nodes) is
- the weak nodes of the block (‖D_i‖ < 1% of the median; the threshold is invariant),
- plus every node of the block that shares an active element with one of them.

This keeps weak nodes paired with their strong neighbours, the route-1 lesson. Size ≤ 125 nodes, typically about 40
nodes (120 DOF). About 3,000 patches per FULL cell. Membership is **frozen per design band** (§2.9).

**Patch numerics** (per design step):
- Form Ǩ_p = D_p^{−1/2} K_pp D_p^{−1/2}, with the rows and columns of the mode's Dirichlet nodes removed.
- Take its k = 16 smallest eigenpairs with batched fp32 `eigh`: ≈ 0.05–0.2 s per FULL cell (≈ 5e10 FLOP).
- Filter:

  Π_patch = Σ_p σ_p R_pᵀ D_p^{−1/2} V_p diag(φ_{θ_p}(λ_{p,j})) V_pᵀ D_p^{−1/2} R_p,
  φ_θ(λ) = (1/λ − 1)·τ(λ/θ),

  where τ is a C¹ taper from 1 at λ ≤ θ/2 to 0 at λ ≥ θ.
- The taper makes φ a spectral function of Ǩ_p. It is therefore continuous in the geometry and exactly equivariant,
  even when eigenvalues are degenerate, as long as fewer than 16 eigenvalues lie below θ. Rank saturation is
  monitored (§6).
- The (1/λ − 1) form adds to Jacobi exactly the missing part of the local inverse on the soft local modes. Jacobi acts
  as 1 in scaled coordinates.
- **Per-patch learned σ_p fixes the route-1 failure mode**: a single global step crushed the exact blocks [RP].
- Patches touching Γ get a second variant for mode X, with the Dirichlet nodes removed (≈ 15% of patches).

**Capacity.** Up to 16 local soft modes per patch × ≈ 3,000 patches is ≈ 48k local modes. The dense band of 128 slow
modes is 1/400 of that. Compare E1 in route 1: 12 global-mode columns per block helped only modestly, because the
band continues [RP E1].

#### 2.5.4 The symmetric V-cycle 𝒱 and the extension

**𝒱**, one symmetric V(1,1) cycle from zero initial guess, level 0 (mode-dependent masks throughout):

```
z  = Ŝ0 b                         (1 K inside Ŝ0)
r  = b - K z                      (1 K)
z += P1 V1(P1^T r)                (level-1 V-cycle: S1, A1 residual, P2 V2 P2^T ..., A3^-1 exact, symmetric)
r  = b - K z                      (1 K)
z += Ŝ0 r                         (1 K inside Ŝ0)      -> 4 fine K per V
```

**Extension** (mode X, cycle polynomial with m = 2 applications of 𝒱):

```
alpha = (R_G^T R_G)^-1 R_G^T q    (fp64, 6x6);   q_d = q - R_G alpha
b  = -(K [q_d; 0])_I              (restricted K, 0.15 K_eq)
y  = V b ;  z = V (K y) ;  x = a1 y + a2 z          (4 + 1 + 4 = 9 fine K)
u_G = q (exact overwrite);  u_I = x + (R alpha)_I
```

- With (a₁, a₂) = (2, −1) this is exactly two V-cycles: N₂ = 𝒱 + 𝒱(I − K𝒱) = 2𝒱 − 𝒱K𝒱.
- Learned (a₁, a₂) give a Chebyshev-like acceleration across cycles.
- N_m = p(𝒱K)𝒱 is **symmetric** because 𝒱 is.

**Forward cost** (FULL, per column): 9 K + 0.15 + 8 S₀ + 2 × (level-1 visit ≈ 2.3 K_eq) + transfers ≈ **15.5 K_eq**,
i.e. ≈ 15 GFLOP.

The level-1 visit (≈ 2.3 K_eq) breaks down as:
- level-1 pre/post smoothers: 2 × 2 channels × 2 applications of A₁ = 8 applications;
- 2 level-1 residuals;
- level 2: 4 channels, ≈ 18 applications of A₂ ≈ 0.18 K_eq;
- level 3: negligible.

#### 2.5.5 Why the fine level is exact-K plus cheap gates, and learned channels live on coarse levels

This is a systems argument, made quantitative:
- A learned fine layer with C vector channels moves the same 348k element incidences and 987k face incidences per
  channel as K [DI 2.2]. It is memory-bound at the same speed, and it needs ≥ 20 MB of per-cell coefficients per
  layer-channel if its element operators are free.
- So a learned fine channel costs ≈ 1 K_eq and ≈ 2 GB for 100 cells per layer, and it carries learned error.
- An exact K application carries zero error.
- At level 1 a learned channel costs 0.2 K_eq and ≈ 0.2 MB per cell.

**Therefore:** capacity goes where it is cheap (levels 1 and 2, patches, gates), and exactness goes where error would
be expensive (the fine level).

**Upgrade path v0.1**, only if step 1 shows the need: one fine "gated K-message" channel, H = G₁D^{−1/2}KD^{−1/2}G₂,
used as a symmetric additive HᵀΘH term. It costs 2 K_eq per use and 1 MB per cell.

### 2.6 Readout: symmetric S_hat q, energy, and the free indicator

- **Energy** (training loss, lattice diagnostics): ûᵀKû = Σ_e Σ_q w_{e,q} ε_q(û_e)ᵀ C ε_q(û_e), with the
  **rigid-exact arithmetic** of §2.10. Cost: 1 K_eq (energy only, ≈ 0.2 GFLOP).
- **Reaction.** From E = [I; −N_m K_{IΓ}]Π + (rigid part) and K R = 0:

  S_hat q = Π [ v_Γ − K_{ΓI} N_m v_I ], with v = K û.

  - This is Eᵀ K E q (§4.4), computed with one more pass of the **same** symmetric N_m. No transposed network code
    and no stored activations are needed.
  - Total per S_hat q: 2 × 15.5 + 1 (v) + 0.15 + ≈ 0.3 (rigid-exact extras) ≈ **32 K_eq ≈ 1.6–3.2 ms per column**.
  - Cut 0.758: ≈ 0.8×. Cut 0.148: ≈ 0.2×.
- **Free error indicator.** η² = v_Iᵀ(N_m v_I) is a dot product with a vector computed anyway. §4.7 gives
  η² ≤ qᵀ(S_hat − S)q ≤ η²/(1 − ρ^{m}).
- **Rejected alternative:** the non-symmetric reaction (K û)_Γ. It is half the cost, but it breaks symmetry for PCG
  and the one-sided bound [RS 3.3, 5.5]. It is kept only as a diagnostic: (S_hat q − (Kû)_Γ) measures the adjoint
  correction.

### 2.7 Neumann (force → displacement) action for BDD

- **Construction:** M_c r = Π_Γ J_Γᵀ 𝒱^N J_Γ Π_Γ r.
  - 𝒱^N is one symmetric V-cycle (as in §2.5.4) in mode N: 𝔇 = ∅ for a floating cell, or the clamped nodes.
  - Coarsest step: Cholesky of A₃^N + α R₃R₃ᵀ followed by rigid projection, which gives the pseudo-inverse on
    range(A₃^N).
  - Π_Γ = I − R_Γ(R_ΓᵀR_Γ)⁻¹R_Γᵀ.
- **Cost:** 4 K + 4 S₀ + one level-1 visit ≈ **8 K_eq ≈ 0.4–0.8 ms** per column.
- **Guarantee** (§4.8): M_c is symmetric positive semidefinite, with kernel exactly span(R_Γ). On the balanced space it
  is spectrally equivalent to S⁺ with the V-cycle's constants: [1 − ρ_N, 1]·S⁺.
- **Knob:** m_N ∈ {1, 2, 3} applications of the same 𝒱^N (polynomial-accelerated). This shrinks the constants toward
  1 at +8 K_eq each.
- **Default:** m_N = 1. Measure the BDD iteration counts, then raise m_N only if the counts exceed exact BDD by more
  than about 1.5×. M is about 4× cheaper than S_hat q.

**Balanced BDD bookkeeping** (a systems saving, independent of the network):
- Start from the coarse solution, so every residual is balanced (Φᵀr = 0).
- Store A Φ from the coarse-space build: ≤ 42 S_hat columns per cell [RP 7.7], ≈ 8 MB fp64 per cell.
- Then each BDD iteration needs **1 S_hat application + 1 M**, not 3 + 1 as in lattice_pcg.py:
  - A z_k is obtained from A w_k minus the stored AΦ term;
  - A p_k follows by recurrence, with a true-residual refresh every 20 iterations.

**Design-iteration budget** (per load case) [est]:

| per FULL-equivalent cell | count | K_eq |
|---|---|---|
| coarse-space build (A Φ) | ≤ 42 S_hat | 1,340 |
| BDD iterations (34–53 exact; assume 50) | 50 S_hat + 50 M | 1,600 + 400 |
| sensitivity field | 1 forward | 16 |
| **total** | | **≈ 3,360 K_eq ≈ 0.17–0.34 s** |

- For 100 cells (≈ 89 FULL-equivalents) this is **≈ 15–30 s** per load case.
- Encode: moments and their τ-derivatives take ≈ 80–100 s (existing pipeline, decision 1). The network-specific encode
  takes ≈ 10–30 s, of which patch eigensolves are ≈ 5–20 s. CPU topology work overlaps.

### 2.8 Sensitivities

- Formula: g_c = −Σ_e Σ_q (∂w_{e,q}/∂τ_c) ε_q(û)ᵀCε_q(û), with ∂w = V⁻¹ ∂M/∂τ_c.
  - ∂M/∂τ_c comes from forward-mode differentiation through polyref: 8 tangents, ≈ 1 extra moment pass [DI §4],
    computed per cell and discarded.
  - The ghost term has zero τ-derivative within a topology [DI 1.5].
- Cost per cell: 1 forward (û from the converged lattice port values), the Gauss-point energy densities (≈ 0.2 GFLOP)
  and a 26 MFLOP contraction.
- **This is the "physical" sensitivity** evaluated with the network field. It is not d(C_hat)/dτ. The difference is
  2(∂E·U)ᵀ K E U = 2(∂E·U)_Iᵀ r_I, which is first order in the interior residual r_I that the indicator already
  measures. The gate compares against the exact dC/dτ, and the physical form avoids differentiating the network
  through the geometry.
- Across a topology change both the operator and the sensitivity jump, exactly as the exact pipeline does [RP 7.5].

### 2.9 Topology changes: design bands, supersets, masks

- **Design band:** each cell gets a superset topology valid for τ_c(1 ± δ), with δ = 0.02 [DI 1.6, RP 7.5 superset].
  Built once per band:
  - index arrays;
  - element-graph edges for G_θ;
  - coarse maps and ghost pattern ids;
  - **patch membership**;
  - the band-correction element list.
- **Per design step:** masks select the active elements, faces and nodes. Nodes outside every active element get an
  identity row (decoupled, as in superset_encoder). Recomputed each step:
  - moments;
  - w, D, the level operators;
  - patch eigenpairs;
  - Lanczos bounds;
  - the coarsest factors (≈ 2 ms);
  - one forward pass of G_θ.
- **Continuity within a topology:** every generated coefficient is a smooth function of the moments. The patch filter
  is a continuous spectral function. So S_hat(τ) is continuous and piecewise smooth, and it jumps only where the exact
  K jumps, at element or ghost-face activation.
- **Leaving the band:** rebuild the superset, which costs a few seconds per cell on the CPU and overlaps with GPU work.

### 2.10 Precision plan (R17)

| operation | precision | reason |
|---|---|---|
| lattice vectors, PCG dots, rigid split α, Π | fp64 | cheap; soft directions need it [RP 7.6] |
| q into the network | fp32 after the fp64 rigid split | δq ~ 1e-7‖q‖ causes a relative energy error ≲ 1e-7 |
| K inside the q-path | fp32 strain form | errors are field errors: second order in energy, and contracted by the next smoothing |
| K in the readout (energy, reaction) | **rigid-exact**: gather in fp64, subtract the element's least-squares rigid fit with a fixed 6×81 pseudo-inverse (the same for all elements), cast to fp32 for gradients and stresses, accumulate element energies and scatter reactions in **fp64** | rounding then scales with the element *deformation*, not with λ_max‖u‖. This is the fp32-T problem (0.8–2.2% whitened error) avoided without fp64 compute. Extra cost ≈ 0.3 K_eq |
| ghost 54×135 GEMMs | fp32 with **TF32 disabled** (`allow_tf32 = False`) | TF32 injects ~1e-3 relative error into K |
| patch eigh, Lanczos | fp32 on Jacobi-scaled matrices | preconditioner components |
| coarsest Cholesky | fp32 (fp64 optional, +12 MB per cell) | coarse error is corrected by the fine cycle |
| training loss μ − 1 | fp64 ratio of fp64-accumulated energies | |

### 2.11 Cost, memory and throughput summary [est]

| item | FULL | cut 0.758 | cut 0.148 |
|---|---|---|---|
| forward extension (K_eq / GFLOP) | 15.5 / 15 | 12 / 9 | 3 / 2.7 |
| S_hat q (K_eq / ms at 0.05–0.1 ms per K_eq) | 32 / 1.6–3.2 | 26 / 1.0–2.0 | 7 / 0.3–0.6 |
| Neumann action (K_eq / ms) | 8 / 0.4–0.8 | 6.5 / 0.3–0.6 | 2 / 0.1–0.2 |
| resident record | ≈ 75 MB | ≈ 55 MB | ≈ 15 MB |
| query working set per column | ≈ 10 MB | ≈ 8 MB | ≈ 2 MB |
| network-specific encode (on top of moments) | 0.1–0.3 s | 0.1–0.25 s | 0.03–0.08 s |

- **Batching across cells:** all cells are concatenated as a disjoint union, with element, face, patch and coarse lists
  offset per cell. Each primitive is **one kernel launch for all 100 cells and all load columns**.
  - The coarsest solves are batched triangular solves on factors padded to 1,029 (mode X) or 2,187 (mode N).
  - A 100-cell × 4-column batch needs ≈ 3.5 GB of working set, and the lattice vectors (fp64) ≈ 0.6 GB.
- **Training throughput (step 1):**
  - Forward plus energy ≈ 16.5 K_eq per column; backward ≈ 2×. That is ≈ 50 GFLOP per column, ≈ 3.2 TFLOP per batch
    of 64, **≈ 0.2–0.4 s per step**.
  - Saved activations: ≈ 35 fine vectors per column (≈ 50 MB per column), ≈ 3.2 GB per batch of 64. Checkpoint the
    two 𝒱 applications to reach batch 256.
  - Exact labels come from cuDSS factors, which must be resident while their geometry is being trained: FULL fp32
    6.6 GB + 0021 3.2 GB + 0013 0.9 GB, and the Neumann factors the same again. So **rotate geometries** (the design
    doc's "geometry batches"), keeping at most two cells' factors resident.

### 2.12 The position on exact K inside the forward pass

| option | fine K per forward | S_hat q (K_eq) | what carries stiff and boundary-layer error | verdict |
|---|---|---|---|---|
| 0: pure feed-forward (K only in the readout) | 0 | ≈ 2 × (learned fine layers ≈ K cost) + 1 | the learned coefficients, to ~1e-4 relative, because qᵀK_ΓΓq / qᵀSq ≈ 1e3–1e5 | **rejected**: this is the precision wall of phases 0–7, and it is not cheaper |
| 1: "sandwich" (one symmetric pre/post smoothing around a K-free learned core) | 2 | ≈ 11 | 2 exact smoothings | fallback of the fallback, if latency ever dominates; unlikely to reach 1e-2 on soft loads |
| **2: GLMO v0, m = 2** | **9 (+ 1 restricted)** | **≈ 32** | 8 exact polynomial smoothing steps across 2 cycles | **chosen**: meets the ~3 ms budget; stiff error is contracted independently of learning |
| 3: m = 4 | 19 | ≈ 64 | 16 steps | first escalation (≈ 6 ms); same code |
| 4: fallback A (per-step learned corrections, 24+ K) | ≥ 24 | ≥ 70 | exact residual at every layer | budget × 2; same components |

Why m = 2 is the right default:
- Let 𝒱's error propagator have K-norm ρ on the soft or smooth part and ρ_hf ≪ ρ on the boundary layer.
- The energy error after N_m is ρ^{2m}(μ_lift − 1)·qᵀSq, restricted to each part.
- The boundary layer, which is the huge part, needs ρ_hf^{2m} ≲ 1e-6. With 4 Chebyshev-type smoothing applications
  across two cycles at ρ_hf ≈ 0.2–0.3 per step, the reduction is 4e-6 to 3e-7.
- The smooth or soft part needs ρ^{2m} ≲ 1e-2, i.e. ρ ≲ 0.3.
- **Step 1 measures ρ and ρ_hf before any training** (§7, risk 1), and m is then set from data.

---

## 3. Answers to the open questions a–l

**a. Consuming fixed-grid masked data.**
- **Structure:**
  - The fine operator is K's own **hypergraph**: 27-node element hyperedges and 45-node ghost-face hyperedges, applied
    matrix-free. There are no edge lists and no radius or grid convolutions in the q-path.
  - Two walls that are close in space interact **only** where K couples them, i.e. through a shared active element or
    a ghost face, and then exactly as strongly as K says.
  - Element moments are used through the signed 5³ Gauss weights. The "edge feature" is the exact element stiffness
    action itself, never a learned 3×3 block.
- **GP faces:** at the fine level, 54×135 factor GEMMs. At coarse levels, sub-face pattern templates (§2.4).
- **Weak nodes:**
  - a feature (log‖D‖) in G_θ;
  - a selector for patch membership;
  - the target of per-patch filter weights.
- **The geometry network** uses the element graph (isotropic, invariant). Grid convolutions are not needed anywhere.
- **Masks are first-class inputs:** element-on, full, cut, band; face-on; node-on, box, band, in-material; the mode
  mask 𝔇; the superset band.

**b. Hard ports and exact rigid modes while staying linear.**
- Ports: the output port values are overwritten with q, so they are exact bit for bit.
- Rigid modes: a fp64 rigid split q = R_Γα + q_d, then u = Rα + extension(q_d). Every coarse space contains the rigid
  fields exactly (Q2), and K annihilates them at every level.
- Dirichlet masks enter the q-path as masked restrictions and prolongations. For the oblique band this uses the
  masked-prolongation Galerkin correction (§2.4). All operations are linear in q.

**c. Global transmission and localized soft modes.**
- Global transmission is **exact** in every cycle: a dense Cholesky solve on the Q2 4³ Galerkin space (≤ 2,187 DOF),
  reached through the exact Galerkin hierarchy 65³ → 33³ → 17³ → 9³.
- Quadratic coarse spaces represent shell bending (route-1 E0).
- There are no spectral layers, which fail on masked, high-contrast bodies [RS 7.2].
- Localized soft modes are carried by the patch spectral spaces with learned per-patch weights at the fine level
  (§2.5.3).
- Learned channel width sits on levels 1 and 2.

**d. Where K appears.** In the q-path: 9 full fine applications plus 1 restricted one per forward pass; in the
readout: 1. Coarse levels use exact Galerkin operators, which are cheap. The justification is §2.12: pure
feed-forward is rejected on precision grounds, and more than 24 applications is rejected on budget grounds, with the
count set by the cycle number m.

**e. Readout.**
- S_hat q = Π[v_Γ − K_{ΓI} N_m v_I] with v = K û. This equals Eᵀ K E q.
- It is symmetric positive semidefinite by construction, and its adjoint is the same symmetric operator N_m, with no
  stored activations.
- Cost: 2 forward passes + 1 K ≈ 32 K_eq.
- The non-symmetric reaction (K û)_Γ is rejected for the lattice and kept as a diagnostic.
- Bonus: the free indicator η².

**f. Approximate inverse.**
- One symmetric V-cycle of the **same** machinery in mode N, restricted to Γ and rigid-projected.
- It is SPD on the balanced space, with spectral constants [1 − ρ_N, 1] (§4.8), and costs ≈ 8 K_eq.
- It is used in balanced BDD with 1 S_hat + 1 M per iteration.

**g. Sensitivity.**
- −ûᵀ(∂K/∂τ)û is evaluated in strain form with ∂w = V⁻¹∂M/∂τ from forward-mode differentiation through polyref.
- Topology is fixed per design step, and masks come from the design-band superset. ∂K_ghost/∂τ = 0.
- It is the physical sensitivity, not d(C_hat)/dτ. The difference is O(‖r_I‖), and it is monitored.

**h. Equivariance.** By construction; see §4.6:
- the carriers are equivariant (K, D, Galerkin operators, patch spectral functions, symmetric grids and patch
  families);
- the gates are invariant (an isotropic GNN on invariant features);
- the rigid-split weighting is invariant.

Augmentation is used only as a **test** and to absorb the ~1e-4 Kuhn-split asymmetry of polyref. The cut plane needs
no special treatment, because joint (g, q) equivariance is automatic.

**i. Smoothness, conditioning, normalization.**
- Parameterization: bounded multiplicative gates around theory values, smooth heads, and zero-initialized last layers.
  So at initialization the network is a convergent multigrid, and every parameter moves a preconditioner-shaped
  operator (F4 in [RS]).
- Loss: per-sample log μ with the fp64 ratio (μ − 1 = energy error / qᵀSq), a CVaR tail term, and a D-weighted field
  term during warm-up (§6).
- Smoother scaling uses stop-gradient Lanczos bounds, which keeps the convergence conditions true during training.

**j. Capacity.**
- Shared weights: G_θ has ≈ 0.6 M parameters, plus ≈ 100 global coefficients from the global head.
- Per-cell generated numbers: ≈ 0.3 M, all invariant scalars modulating exact carriers.
- There is no per-pair or per-element matrix storage.
- Where capacity grows if step 1 needs it, in order:
  1. more level-1/2 channels and higher polynomial degree;
  2. patch rank 16 → 24, or a third patch family;
  3. m = 3;
  4. v0.1 fine gated K-message channel.

**k. Precision.** fp32 network, rigid-exact fp32 K with fp64 accumulation in the readout, fp64 lattice algebra and
rigid split, TF32 off (§2.10).

**l. Degradation.**
- Main line: G_θ-generated gates, m = 2.
- **Fallback A** = the same 𝒱 with m ≥ 4 and per-cycle learned coefficients. At that point each "layer" is a learned
  correction of an exact residual, which is exactly fallback A, with the same code, primitives and cell record.
- **Fallback B** = G_θ frozen at initialization (all gates 1, theory polynomials), learning only the ≈ 100 global
  scalars (route-1 style, label-free). It is route 1 on a better hierarchy: exact Galerkin Q2 with 4 levels instead of
  2-level block polynomials, and patch spectral spaces instead of pair blocks.
- The Neumann action is the same 𝒱 in all three.

---

## 4. Guarantees (with short proofs)

Throughout: g is fixed and the mode is fixed, so every generated coefficient is a constant.

### 4.1 Exact linearity in q
Every q-path primitive (§2.5.1) is a matrix whose entries depend only on (g, mode):
- the rigid split;
- the restricted K;
- K;
- S₀ and Π_patch;
- P and Pᵀ;
- A_ℓ and A₃⁻¹;
- gates;
- the fixed polynomials.

The forward pass is a finite composition and sum of such matrices, hence linear. No nonlinearity or data-dependent
step (no Krylov coefficient, no max, no normalization by q) touches q. ∎

### 4.2 Exact port values (box and cut band)
u_Γ := q by assignment. In mode X every interior update is multiplied by the interior mask, including the band rows of
P̄ = M_I P. So no later operation writes to Γ. Hence (E q)_Γ = q exactly, for any coefficients. ∎

### 4.3 Exact rigid modes
Let q = R_Γβ. R_Γ has full column rank, since the port contains ≥ 3 non-collinear nodes. Then α = β and q_d = 0, so
b = 0 and x = N_m 0 = 0, which gives u = Rβ exactly. The rigid split is done in fp64, so α = β up to fp64 rounding.
S_hat R_Γβ = EᵀK Rβ = 0 because K R = 0. In the rigid-exact readout, the element rigid fit removes R's contribution
before the fp32 stage, so the computed reaction is zero to fp64 rounding. ∎

### 4.4 Symmetric positive semidefinite readout and the adjoint formula
**Extension operator.** Write E = R(R_ΓᵀR_Γ)⁻¹R_Γᵀ + E_d Π with E_d = [I; −N_m K_{IΓ}]. Since K R = 0,

  S_hat = EᵀKE = Π E_dᵀ K E_d Π,

which is symmetric and satisfies xᵀS_hat x = ‖E x‖²_K ≥ 0. Its kernel contains span(R_Γ).

**Adjoint formula.** E_dᵀ v = v_Γ − K_{ΓI}N_mᵀ v_I, and N_mᵀ = N_m:
- 𝒱 is symmetric because its post-smoother is the transpose of its pre-smoother and its coarse correction is
  recursively symmetric;
- N_m = p(𝒱K)𝒱 is a polynomial in 𝒱K times 𝒱, which is symmetric.

This gives the formula of §2.6. ∎

### 4.5 One-sided (upper-bound) error
Let E* be the exact extension: (K E* q)_I = 0 and (E* q)_Γ = q. For any E with (E q)_Γ = q, write e = (E − E*)q. Then
e vanishes on Γ, and

  qᵀS_hat q = (E*q + e)ᵀ K (E*q + e) = qᵀSq + 2 eᵀ(K E* q) + eᵀKe.

The middle term is zero because K E* q vanishes on I and e vanishes on Γ. Hence S_hat − S = (E − E*)ᵀK(E − E*) ≽ 0
and μ(q) ≥ 1.

**Lattice.** Â = Σ_c R_cᵀ S_hat,c R_c ≽ A, so Ĉ = fᵀÂ⁻¹f ≤ fᵀA⁻¹f = C: the surrogate lattice is always too stiff.
Sensitivities have no sign guarantee. ∎

### 4.6 Equivariance
Let h ∈ O_h act on the cell about its centre. Its action:
- node and element indices are permuted by h;
- vectors are acted on by ρ(h) (a signed permutation matrix);
- moments are acted on by the signed permutation of monomials [DI 2.4].

**Equivariant objects** (K_{h·g} = ρ̄ K_g ρ̄ᵀ, with ρ̄ the permutation composed with ρ blockwise):
- K;
- D (conjugated blockwise);
- the prolongations, because the nested 32/16/8/4 grids are symmetric about the centre;
- the coarse operators, by Galerkin;
- the patch families 𝔅₀ and 𝔅₁, which are invariant as partitions;
- the weak-node threshold, which is an invariant;
- patch membership;
- Ǩ_p, and therefore every spectral function of it (the filter in §2.5.3);
- the Lanczos bounds, which are invariant because the Krylov spectra are invariant under similarity. With a fixed
  random start vector this holds only to the bound tolerance, so use the start vector ρ̄·1, which is invariant;
- R, Π and the rigid split with W = I, since span(R_Γ) is invariant;
- the port and mode masks, which are permuted.

**Invariant outputs.** G_θ uses only invariant features and isotropic sum aggregation over an equivariant graph, so
its outputs are permuted scalars. Every q-path operator is then O(g·h) = ρ̄ O(g) ρ̄ᵀ, and hence

  E(h·g)(ρ̄ q) = ρ̄ E(g) q and S_hat(h·g) = ρ̄ S_hat(g) ρ̄ᵀ,

exactly in exact arithmetic. ∎

**Caveats:**
- polyref moments of a rotated geometry differ from the permuted moments by ~1e-4 [DI 2.4];
- floating-point summation order;
- global-head attention uses only invariant positions.

### 4.7 Two-sided, free error indicator
Assume every level smoother satisfies the scaled condition λ(SA) ∈ (0, 1.9] (enforced in §2.5.2) and the coarsest
solve is exact.
- Then the V-cycle error propagator E_V = I − 𝒱K = (I − Ŝ₀ᵀK) C (I − Ŝ₀K), where C is the coarse-correction
  propagator.
- C is K-self-adjoint with spectrum in [0, 1] (recursively), so E_V is K-self-adjoint and positive semidefinite, with
  spectrum in [0, ρ] where ρ = ‖E_V‖_K < 1.
- With the plain two cycles (a₁, a₂) = (2, −1), N₂K = I − E_V² has spectrum in [1 − ρ², 1].
- The residual is r_I = v_I = (Kû)_I = K_II e, so the true energy error is eᵀK_IIe = r_Iᵀ K_II⁻¹ r_I.
- Since N₂ K_II has spectrum in [1 − ρ², 1], we get (1 − ρ²) r_Iᵀ K_II⁻¹ r_I ≤ r_Iᵀ N₂ r_I = η² ≤ r_Iᵀ K_II⁻¹ r_I.

**Result:** η² ≤ qᵀ(S_hat − S)q ≤ η²/(1 − ρ²). η² is available after the adjoint pass at no extra cost, and ρ is
estimated at encode time by 10 steps of power iteration on E_V.
- With learned (a₁, a₂), the same bound holds with [min, max] of 1 − p(λ) over [1 − ρ, 1]. This is monitored, and a
  barrier keeps it ⊂ (0, 1].

**Lattice.** From Â = A + Dm with Dm ≽ 0 and Û the surrogate solution:

  C − Ĉ = Ûᵀ Dm Û + Ûᵀ Dm A⁻¹ Dm Û ≥ Σ_c q̂_cᵀ(S_hat,c − S_c) q̂_c ≥ Σ_c η_c².

So Σ_c η_c² / Ĉ is a free, per-solve **lower estimate of the relative compliance error**. It is sharp to first order,
and the neglected term is O(Dm²). It can drive adaptivity: raise m only on cells with large η_c². The primal-dual
certificate of route 2 remains the rigorous two-sided bound for later steps.

### 4.8 The Neumann action is SPD and spectrally equivalent to S⁺
Let K_f be the floating-cell matrix restricted to the complement of span(R), and 𝒱^N the mode-N V-cycle with the
pseudo-inverse coarsest solve. As in §4.7, (1 − ρ_N) K_f⁻¹ ≼ 𝒱^N ≼ K_f⁻¹ on range(K).

Congruence with the injection J_Γ preserves Loewner order:

  (1 − ρ_N)(K_f⁻¹)_ΓΓ ≼ (𝒱^N)_ΓΓ ≼ (K_f⁻¹)_ΓΓ.

Block inversion gives (K_f⁻¹)_ΓΓ = S⁻¹ on the balanced space.

Hence M_c = Π(𝒱^N)_ΓΓΠ is symmetric positive semidefinite, and the BDD condition number grows by at most
1/(1 − ρ_N) relative to exact local Neumann solves. For ρ_N = 0.5 that is ≤ 2×, i.e. ≤ √2× iterations. ∎

The approximate M is used with S_hat (not S) in the lattice. Since S ≼ S_hat ≼ μ_max S on the balanced space:
- the lower constant only improves: M ≽ (1 − ρ_N)S⁻¹ ≽ (1 − ρ_N)S_hat⁻¹;
- the upper constant degrades by at most μ_max: M ≼ S⁻¹ ≼ μ_max S_hat⁻¹.

μ_max is ≈ 1 for an operator that passes the gate.

### 4.9 Continuity within a topology
Within a design band and a fixed topology:
- w, D and the level operators are linear in the moments;
- the gates are smooth MLP outputs of smooth features;
- the patch filter is a continuous spectral function (C¹ taper), provided it does not saturate the rank of 16;
- the Lanczos scalings are continuous under small perturbations (they are stop-gradient scalars).

Hence S_hat(τ) is continuous, and piecewise smooth, as the moments are.

---

## 5. Why the soft, force-driven directions come out right

**The target.** 94–96% of lattice load energy sits in the softest 10% of the cell spectrum. Force-driven q = S⁻¹f
amplify exactly those directions. Physically they are of three kinds:
1. global bending and twisting of the shell network, smooth along the walls;
2. sliver near-mechanisms on weak nodes: 11–30 nodes each, a dense band;
3. port-localized slivers where the cut or fringe meets a box face.

**Each kind has a dedicated exact carrier, not a learned approximation:**
1. **Bending.** Q2 coarse spaces on 16³ and 8³ contain quadratic fields on each coarse element. Route 1 showed that
   quadratic coarse content is decisive (64-layer witness 7.75 → 1.26 [RP E0]). The coarse operators are exact
   Galerkin (Prop. 2.4), so the coarse correction is the best energy approximation in that space, not a learned
   imitation. Global transmission is exact through the 9³ Cholesky solve in every cycle.
2. **Slivers.** The patch spectral spaces solve the soft local modes exactly within each patch (up to the per-patch
   learned weight σ_p).
   - They contain the weak nodes **and** their strong neighbours, as route 1 required.
   - There are two overlapping, symmetric families, so modes of 11–30 nodes fit inside a patch.
   - Capacity is ≈ 48k local modes against a 128-mode band.
   - The per-patch weight replaces the single global damping that suppressed route-1's exact blocks (λ_min improved
     only 14% [RP]).
   - Because 94–99.4% of slow-mode amplitude is on weak nodes, **patch coverage of the weak set is coverage of the slow
     band**. This is measured before training (§6 Phase 0).
3. **Port slivers.** Port values are exact (hard constraint). In mode X the patches touching Γ are re-solved with Γ
   removed, so the sliver interior adjacent to a moving port node is extended by an exact local solve.

**The stiff and boundary-layer part is handled by exact residuals, not by learning.** Soft q have
qᵀK_ΓΓq ≫ qᵀSq, i.e. near/far-field cancellation of 1e3–1e4. The zero-interior lift carries that huge energy in a
boundary layer. Eight exact-residual polynomial smoothing steps contract it by ρ_hf^{2m}, with no dependence on
learned precision (§2.12). This is precisely the component that entry-wise or feed-forward approaches had to get right
to 1e-4.

**The training signal points at the soft directions:**
- the per-sample loss is log μ, i.e. normalized by qᵀSq, so soft directions weigh as much as stiff ones;
- 30% of samples are force-driven (10% with band forces), and 20% are adversarial LOBPCG directions of the pencil
  (S_hat − S, S) after warm-up;
- a CVaR tail term;
- a sensitivity-consistent auxiliary term on element energy densities (§6).

**Precision does not wash out the soft directions.** The rigid-exact readout makes rounding proportional to element
deformation. The fp32-T failure (0.8–2.2% [RP 7.6]) came from rounding proportional to λ_max‖q‖.

**The gate is watched directly.** The free indicator η² is computed on the actual lattice solutions (§4.7), so soft
failures in production are visible cell by cell.

---

## 6. Training for step 1 (three fixed geometries)

**Geometries:** the medium-cut and heavy-cut second-generation samples selected in step 0, and their shared FULL
parent.

**Modes:**
- **X-band** (box ∪ Γ-band Dirichlet; primary, per decision 4);
- **X-box** (cut band free; trained jointly at no extra architectural cost, same weights, only masks differ);
- **N** (evaluated for BDD; not trained in step 1).

**Exact side:** per geometry, cuDSS factors of K_II for each X mode and of K + αRRᵀ for N. Keep ≤ 2 cells' factors
resident and rotate through the geometries.

**Phase 0: zero learning, ≈ 30 min. Run this before anything else.**
1. Build the cell record with all gates set to 1 and theory polynomials.
2. Measure:
   - ρ and ρ_hf by power iteration on E_V; ρ restricted to the slowest 64 exact modes;
   - with and without patches;
   - the fraction of slow-mode amplitude inside patches;
   - the μ distribution per sampling class for m = 1, 2, 3, 4, 6;
   - the lattice_v2 gate, i.e. compliance and sensitivity for 6 cases × 2 configurations;
   - the BDD iteration count on a 2×2×2 FULL lattice with M = 𝒱^N (lattice_pcg with the exact T for S);
   - μ ≥ 1 on every sample (a sanity check), symmetry, and rigid exactness.
3. This is the zero-learning baseline, and it decides m.

**Phase A: representation upper bound, ≈ 1 h, one geometry at a time.** Free per-node, per-patch and per-level gate
parameters, with no G_θ. This isolates q-path capacity from hypernetwork generalization.

**Phase B: the real thing, ≈ 2–3 h.** G_θ generates everything, trained on the 3 geometries × 2 X modes.

**Loss** (per batch of 64 q, one geometry and mode per step):

  L = mean_q log μ(q) + 0.5·CVaR₀.₉[log μ] + λ_D(t)·ℓ_D + 0.1·ℓ_sens

- **μ(q)** = ûᵀKû / qᵀSq, an fp64 ratio. log μ = log1p((ûᵀKû − qᵀSq)/qᵀSq). Samples are pre-scaled to qᵀSq = 1.
  This is label-free apart from the exact scalar qᵀSq, which is also available as the energy of the exact field.
- **ℓ_D** = ‖û − u*‖²_D / ‖u* − Rα‖²_D, with D = the blockwise diagonal of K. It is well conditioned: F4 is countered
  in the early phase. λ_D: 1 → 0 linearly over the first 5k steps.
- **ℓ_sens** = Σ_e |a_e(û) − a_e(u*)| / Σ_e a_e(u*), with a_e the element strain energy, on force-driven samples only.
  It targets the first-order sensitivity quantity directly.
- The exact fields u* come from cuDSS, a few ms per batch.

**Sampling (decision 9):**
- Force-driven: 30%. q = S⁻¹f with f a smooth multiscale box force, computed through the N-mode factor; 10% of these
  also get band forces. f_band = 0 produces the physical free-cut band displacements.
- Macro: 20%. Rigid, 6 uniform strains, quadratic and cubic fields.
- GRF on ports: 30%. Correlation lengths from the whole face down to 2 elements.
- Adversarial: 20%, from step 2k on.
  - Block LOBPCG (block 32, 10 iterations, warm-started) on (S_hat − S)v = θSv every 250 steps.
  - Push the top 8 into a replay bank of 256.
  - Sample from the bank and from fresh LOBPCG vectors.
  - Cost ≈ 20 TFLOP per refresh, ≈ 1–2 s.
- Lattice-induced port vectors from lattice_v2 are **validation only**, never trained on, so the gate cannot leak.
- O_h augmentation: 10% of batches (moments are permuted, not re-integrated), as an equivariance test. By §4.6 it
  should change nothing beyond 1e-4.

**Optimizer:**
- Adam with β = (0.9, 0.99): lr 3e-4 for G_θ and 1e-3 for the global coefficients, warm-up 500 steps, cosine decay to
  1e-5 over 20k steps.
- Gradient clipping at 1.0. fp32 parameters. TF32 off.
- Lanczos smoother scalings are recomputed every 200 steps and held stop-gradient in between. The barrier keeps
  polynomial positivity and the cycle-polynomial range.

**What to monitor** (every 250 steps unless stated):
1. μ − 1 quantiles (median, p90, p99, max) per sampling class and per mode; softest-decile energy error.
2. Worst-direction μ (LOBPCG witness), reported, not gated.
3. **Lattice gate:** compliance and sensitivity error on lattice_v2, 6 cases, every 1k steps. Also Σ η² vs the true
   compliance error, to validate §4.7.
4. ρ and ρ_hf of E_V; ρ restricted to the slow band; N₂K spectrum bounds.
5. Where the error energy is: weak / port-adjacent / bulk nodes; inside vs outside patches; per level.
6. Patch rank saturation: the fraction of patches with ≥ 14 eigenvalues below θ_p. Gate histograms (saturation at the
   bounds means capacity is missing).
7. Symmetry defect |xᵀS_hat y − yᵀS_hat x| / (‖x‖_{S_hat}‖y‖_{S_hat}) and rigid residual; both should be ≤ 1e-6.
8. N-mode: κ(M S) by Lanczos, and BDD iteration counts on 2×2×2 FULL.
9. Timing per primitive against the cost model of §2.11.

**Step-1 ablations** (cheap, same code):
- m ∈ {1, 2, 3, 4};
- patches off;
- σ_p fixed at 1;
- C₁/C₂ ∈ {1/2, 2/4, 4/8};
- Phase A vs Phase B;
- X-band vs X-box on the lattice gate.

---

## 7. Risks, ranked, with the cheapest exposing experiment and the degradation path

| rank | risk | cheapest experiment that exposes it | if it materializes |
|---|---|---|---|
| 1 | **Soft-band contraction too weak**: ρ on the slow band ≳ 0.5, so m = 2 misses μ − 1 ≤ 1e-2 on force-driven loads | Phase 0 (30 min, no training): ρ on the slowest exact modes with and without patches; lattice_v2 error vs m for the untrained skeleton | raise m to 3–4 (+32 K_eq each; still ≈ 6 ms); patch rank 24, third family, patches at level 1. If m > 6 is needed, this *is* fallback A, with the same code |
| 2 | **The training signal does not move the soft directions** (F4); the learned gates improve smooth probes only, as in route-1 E1b | Phase A on the heavy cut with force-driven and adversarial q only, 2k steps: does softest-decile μ − 1 drop ≥ 3× from Phase 0? | increase the ℓ_D and ℓ_sens weights; natural-gradient-like preconditioning of the gate updates (divide by the Jacobi diagonal of the gate Fisher); if still flat, freeze G_θ and learn global scalars only (fallback B, known to pass at 24 layers) |
| 3 | **The approximate Neumann M inflates BDD iterations** beyond ~1.5× exact | lattice_pcg.py on 2×2×2 FULL: replace the local Cholesky solve by 𝒱^N (untrained) and count iterations (minutes) | m_N = 2–3 (0.4–0.8 ms each, still ≪ S_hat q); Chebyshev-accelerate M with fixed coefficients |
| 4 | **The geometric Q2 hierarchy fails on cut cells**: coarse functions merge pieces, the Galerkin coarse space is too stiff near slivers, or the ghost template is not a pure jump operator (Prop. 2.4 ghost claim) | (a) verify F_axis·P_s = 0 for interior sub-faces, a 5-minute template check; (b) a two-level test with an exact level-1 solve (cuDSS on 63k DOF): two-grid contraction vs "no coarse" | component-split coarse elements (aggregation on the K-graph within each coarse element, as in [RS 1.7]) with the same Galerkin machinery; or a smoothed prolongation at level 1 only (+1 K per transfer) |
| 5 | **Precision:** fp32 K or strain rounding pollutes soft directions or symmetry | on 0013: S_hat q in rigid-exact fp32 vs full fp64 for the 64 softest exact S eigenvectors; symmetry defect | fp64 for the final post-smoothing residual and the readout (≈ +2 K_eq at 1/64 rate, still ms); fp64 coarsest factor |
| 6 | **The cost model is off by > 3×** (atomics, ragged union batching, patch eigh, level-1 ghost templates) | microbenchmark before training: K apply, 𝒱, S_hat q on a union of 100 synthetic cells (FULL and cut) × 1/8/64 columns | colour elements to avoid atomics; dense 32³ layout [DI 2.2]; bucket patch sizes; fp16 storage of V_p; m_N and channel cuts |
| 7 | **The X-band mode inflates lattice unknowns and iterations** (≈ +30k local DOF per heavy cut cell with sliver conditioning) | lattice_pcg with exact operators: cut cells in X-band vs X-box; count iterations | use X-box for free-cut cells in production (already trained); X-band for loaded cuts and skins |
| 8 | **Topology changes:** patch membership or superset structure produces jumps larger than the exact ones; frozen membership goes stale within a band | τ ±1e-4 … ±1e-2 sequences on 0021 (as in topo_change.py): ‖ΔS_hat‖ vs ‖ΔS‖, μ along the path | shrink δ; recompute membership each step (loses continuity only at membership flips) |
| 9 | **Equivariance leaks** (Lanczos start vector, attention, patch threshold ties) | the 15 rotated cases: ‖E(h·g)ρ̄q − ρ̄E(g)q‖ | canonical start vector; augmentation during training |
| 10 | **Step 2 (outside step 1): G_θ generalization** | 100/300 learning curve | more invariant features; bigger G_θ (cheap: encode time is dominated by moments) |

**How the design degrades.** Every component is shared:
- Raising m walks continuously into fallback A.
- Freezing G_θ at its zero-initialized state gives fallback B on a stronger skeleton.
- Both fallbacks keep the cell record, the rigid-exact readout, the Neumann action, the indicator and the lattice
  integration unchanged.
- Nothing needs to be rebuilt to switch.

---

## 8. Borrowed vs new

**Borrowed, adapted to our constraints:**
- the linear multigrid V-cycle as an operator parameterization, with nonlinearity outside the q-path (MgNO, MgNet
  [RS 1.1–1.2]);
- Galerkin coarse operators and nested FE transfers (classical);
- polynomial / Chebyshev smoothers with Lanczos-bounded scaling (route 1 [RP E3]);
- learned smoothers generated from stencils (Huang–Li–Xi [RS 1.8]);
- local low-energy eigenspaces as the contrast-robust carrier (GenEO, CEM-GMsFEM [RS 5.7, 7.3]);
- patch/Schwarz treatment of small cut elements (de Prenter et al. [RS 7.4]);
- a symmetric preconditioner "same basis on both sides" (Melchers et al. [RS 3.3]);
- BDD with a balancing coarse space (Mandel [RS 5.7]);
- Deep-Ritz / discrete energy loss and hard nodal Dirichlet values [RS 4.1, 4.3, 4.4];
- the hypernetwork with invariant heads (Ha et al.; EquiModel's index-free augmentation lesson);
- isotropic message passing on the mesh (topology) graph (MGN mesh-space edges [RS 2.4]);
- superset topology bands (route 7 [RP 7.5]).

**New in this design, to my knowledge:**
1. **Galerkin coarse operators on nested Q2 CutFEM spaces from "coarse moments"** (Prop. 2.4): the exact algebraic
   multigrid coarse operator of a cut-cell discretization at the cost of a binomial moment transform, plus the
   sub-face-pattern ghost templates. It needs no assembly, and it keeps rigid modes at every level.
2. **Exact K as the fine-level message kernel**, with learned capacity placed only on coarse levels. The placement is
   derived from a memory-traffic argument (§2.5.5), not taken as a convention.
3. **Mode masks:** one cell record and one set of weights serve the Dirichlet extension on box ∪ band, the extension
   on box only, and the floating or clamped Neumann action. The Neumann action inherits SPD-ness and explicit
   spectral constants through the principal-submatrix identity (K⁻¹)_ΓΓ = S⁻¹ (§4.8).
4. **A symmetric extension operator N_m = p(𝒱K)𝒱**, which makes the adjoint the forward operator itself (no activation
   storage) and yields the **free two-sided error indicator** η² per S q, and a lattice compliance-error estimate
   Σ_c η_c² (§4.7).
5. **Rigid-exact mixed precision:** fp64 element-local rigid removal before fp32 strain evaluation, and fp64
   accumulation. It avoids the fp32-operator failure without fp64 compute.
6. **Spectral-filter patch spaces on two O_h-symmetric overlapping block families**, with weak-node-plus-neighbour
   membership and per-patch learned weights. They are continuous in the geometry and exactly equivariant.
7. **A sensitivity-consistent auxiliary loss** on element energy densities, and a "physical" sensitivity whose
   deviation from the surrogate derivative is bounded by the same residual that drives the indicator.
8. **Balanced-BDD bookkeeping** with stored AΦ: 1 S_hat + 1 M per iteration. This halves the lattice query count
   relative to the current lattice_pcg.

**Deliberately not used:**
- spectral / FNO layers at any level [RS 7.2];
- radius graphs [RS F6];
- learned per-pair or per-element matrices (memory [DI §3]);
- non-smooth heads [AH phase 7];
- entry or Frobenius losses [AH phases 0, 4, 7];
- one-sided low-rank corrections [AH];
- fixed-geometry coarse bases [AH].
