# v0 architecture from the mechanics: MMX-Net (Mechanism-enriched Multilevel eXtension network)

Lens: mechanics first. Each component of the network corresponds to a mechanical mechanism of the thin TPMS shell in its
CutFEM discretization. The proposal is nevertheless a complete architecture: data flow, shapes, costs, guarantees,
training, risks.
Sources: ARCH_BRIEF.md (the "brief"), DATA_INTERFACE.md ("DI"), RESEARCH.md ("RS", facts F1-F6), the repository docs
(ROUTES_PROGRESS, ARCHITECTURE_HISTORY, OPERATOR_LEARNING_DESIGN), and the route-1 code
(`docs/data/takeover_20260923/src_route1/encode_r1.py`, `hierarchy_strain_network_r1.py`), read only.
Numbers marked [est] are my estimates. Numbers without a mark come from the brief, the DI or the logs.

---

## 0. The design in one paragraph

The q-path is a **linear, U-shaped, multiscale graph operator applied in two sweeps**. Its input is the port
displacement q. It turns q into the exact out-of-balance nodal forces that the ports exert on the first interior
layer, and returns the interior displacement. Every coupling in it is one the mechanics has:
- **Fine level (K-structured GNO).** Linear message passing over the element hyperedges (27 nodes) and ghost-face
  hyperedges (45 nodes), whose kernels are the exact element stiffnesses scaled by learned per-element and per-face
  gates. Nodes are grouped into clusters: strongly coupled nodes together, and each weak (sliver or fictitious) node
  together with the strong node it hangs on. Each cluster is solved by an exact local inverse.
- **Coarse levels (material-aware aggregates).**
  - Level 1: 2³-element blocks carrying rigid, membrane (affine) and bending (quadratic) fields, **plus the local
    soft mechanisms of the block**, found by a small exact generalized eigenproblem (a GenEO / spectral-AMGe local
    problem).
  - Level 2: 4³-element blocks with quadratic fields.
  - Level 3: 8³-element blocks with affine fields, solved exactly by a dense Galerkin inverse. This is the global
    kernel.
- **Rigid motion and ports.** Rigid motion is split off and extended exactly. Port values are imposed by masking.
- **Readout.** The reaction is read variationally with the exact K: Ŝ = Êᵀ K Ê.
- **The only learned nonlinear part** is an O_h-invariant geometry GNN. It emits O(1) gates around this mechanical
  skeleton. With all gates equal to 1, the network is exactly a (mechanism-enriched) route-1 V-cycle. That is its
  initialization, and also fallback B.

---

## 1. Requirements (derived; each tied to evidence)

| # | Requirement | Evidence (brief section / doc) |
|---|---|---|
| R1 | Output is a field u(g, q) on the fixed grid; **exactly linear in q**; nonlinearity only in g → coefficients | decisions 1, 3, 5 |
| R2 | Ports at full resolution (box ∪ cut band); cost independent of the number of port DOF (port data may enter only through masks and a port-local force map) | decisions 2, 4 |
| R3 | **Port values imposed exactly** (hard mask), K in the readout exact. Only then is S_hat − S = (Ê−E)ᵀK(Ê−E) ≥ 0 (second order, one-sided) | §4 "variational readout property" |
| R4 | **Rigid modes exact and removed analytically in the readout**. An fp32 rigid residual times ~1e-7 gave a 12× cell-energy error in a 2-cell lattice | decision 6; ROUTES_PROGRESS 7.6 |
| R5 | Accuracy where the load is: the softest decile carries 94–96% of lattice load energy. A skeleton that errs 0.3% on smooth probes errs 20–35% on force-driven loads | §4; ROUTES D2 |
| R6 | A carrier for **localized soft mechanisms** whose capacity grows with the number of mechanisms in the geometry: 17.6–21.9% weak nodes, 94–99.4% of the slowest-64 amplitude on them, 11–30 nodes per mode, dense band of ≥128 modes in [0.0039, 0.0062]. Global low rank fails (3000 fixed columns → 72%). Adding the 128 slowest global modes only moved the witness 1.26 → 1.09 | §4; history stage 1; ROUTES E1 |
| R7 | **Weak nodes are handled together with their strong anchors**, never separated (separating: witness 1.26 → 3.80; merging: 1.26 → 1.13) | ROUTES "slow modes" section |
| R8 | **Global transmission** (forces entering one face must reach the others) via an exact coarse Galerkin solve. No bandwidth truncation: soft stiffness is a near/far-field cancellation of 1e3–1e4 | §4; history stage 1 |
| R9 | Nodes may couple **only through material, i.e. through K's sparsity** (shared active element or ghost face). No spatial-radius edges | RS F6; decision 3 |
| R10 | **Bending-capable coarse spaces**. Quadratic patch fields beat affine ones by 6× in the 64-layer witness (7.75 → 1.26). Q1 on 1–7-element-thick walls locks in bending | ROUTES E0; §2.3 below |
| R11 | **Fictitious and weak nodes (35–55% of active nodes) must be output correctly.** In soft directions the physical energy per unit displacement is so small that the ghost-penalty energy (γ = 1e-4) of fictitious nodes is no longer negligible | DI §2.1, §2.4; §4 |
| R12 | Symmetric PSD reaction with an explicit adjoint (BDD); an approximate SPD Neumann action (BDD: 11–53 iterations vs 300–1600 without it) | §5 BDD; question e/f |
| R13 | Sensitivity −uᵀ(∂K/∂τ)u is first-order in the field error and lives on the shell-surface elements. It uses the exact ∂K/∂τ from moments and is defined only within one topology (±1e-3 in τ already changes 4–6 elements) | §3, §4 |
| R14 | Memory: 100 cells resident on 32 GB → ≲100–150 MB stored per FULL cell. No per-element dense matrices and no pair-edge lists | §6; DI §3 |
| R15 | Throughput: ~2e4 applications per design iteration, a few ms per column. Encode ≲0.5 s beyond moments, without factorization memory | §6 |
| R16 | Joint (g, q) O_h equivariance. Rotation augmentation helped held-out cut cells (0.64 → 0.41); the pipeline is equivariant to 1e-7 | decision 6; §5 |
| R17 | Smooth parameterization, no max/abs heads (a non-smooth head made the first learned-factor attempt diverge) | §5; history stage 7 |
| R18 | The cut band supports **both** displacement input and nodal-force input on the same mask; a free cut is zero band force | decision 4 (staged) |
| R19 | The training signal is per direction and physics-aligned: an energy loss normalized by qᵀSq (= μ−1), with force-driven and adversarial directions. Entry losses provably fight the soft directions | decisions 7, 9; history S1 |
| R20 | Components are shared with fallback A (learned iteration) and fallback B (route-1 skeleton), so degradation is continuous | decision 8 |

---

## 2. Architecture

### 2.1 What the exact extension looks like (the target, mechanically)

- **FULL cell.** The Schwarz-P shell (thickness 2τ/|∇φ| ≈ 0.03–0.22 of the cell, i.e. 1–7 elements; typically 2–3)
  forms 6 necks that cross the box faces; the port patches are the neck cross-sections. The extension of a port
  displacement has four parts.
  1. A rigid part.
  2. A macro part: membrane stretching of the necks plus bending at the saddle junctions. This is the load path, and
     it spans the whole cell.
  3. Boundary layers at the ports. Self-equilibrated port fluctuations decay over roughly one wall thickness
     (Saint-Venant). Shell edge bending decays over 1/β ≈ 0.78√(Rt) ≈ 2–3 elements (R ≈ 0.2, t ≈ 0.06).
  4. **The fictitious fringe.** 35% of the active nodes lie outside the material. Their values are the
     ghost-penalty continuation of the material field: for a Q2 element held mainly by γ-weighted jumps of normal
     derivatives up to second order, the minimizer is close to the polynomial continuation of the well-supported
     neighbour (the aggregated-FEM picture).

  The weak-node proxy is 16.7% in FULL, so the fringe exists there too.
- **Heavily cut cell (retained 0.148).** A few shell fragments remain: flaps attached to box patches, wedge slivers
  where the plane grazes a wall, and islands joined by thin ligaments or only through ghost faces. For soft loads the
  extension consists of:
  - near-rigid motion of each sliver, which follows its anchor;
  - cantilever bending of flaps;
  - strain concentrated in the ligaments.

  These are **local mechanisms**: low-energy motions of a few-element region whose stiffness comes from a ligament,
  a tiny volume fraction and the γ = 1e-4 ghost terms. Their eigenvalues are 7e-7 of λ_max (0013). This is the 11–30
  node dense band of R6.

Consequently the exact field has three distinct kinds of content: the load path (global, smooth along the shell),
boundary layers (local to the ports), and mechanisms (local, rough, low energy). **MMX-Net gives each kind its own
carrier.**

### 2.2 Data flow

Each pass through the q-path below is one sweep 𝒮.

```
GEOMETRY PATH (once per geometry; nonlinear in g; no q)
 moments M (E×125) ──► exact mechanics encoder (deterministic)
    ├─ strain-form Gauss weights w_e (E×125), node blocks D_i (N×3×3), weak flags, port masks (per port mode)
    ├─ fine clusters: strong pairs (≤4 nodes) ∪ anchored weak clusters (≤24 nodes); exact inverse factors F_c
    ├─ L1 aggregates (2³-el blocks, PoU): 30 polynomial fields + ≤8 mechanism modes (local generalized eigen)
    │     Galerkin diagonal blocks A1_aa → inverse factors
    ├─ L2 aggregates (4³-el, PoU): 30 polynomial fields; explicit Galerkin A2 (block-sparse)
    └─ L3 aggregates (8³-el, PoU): 12 affine fields; dense A3⁻¹ (fp64)
 invariant features ──► geometry GNN (learned) ──► gates {a_e, b_f, ω_c, ω1_a, ω2_A, g_AB, κ3} (all ≈1 at init)

Q PATH (linear in q; coefficients from the geometry path)
 q (port DOF) ─► rigid split: α = A_R q (6),  q_d = q − R_P α
             ─► port force: f0 = −K_{I,P} q_d            (+ f_band in force mode)
             ─► sweep 𝒮1:  w1 = 𝒮(f0)
             ─► exact equilibrium residual r1 = f0 − K_II w1 ─► sweep 𝒮2: w = w1 + 𝒮(r1)
             ─► u = R α + [q_d on ports ; w on unknowns]
READOUTS
 energy qᵀŜq = u_dᵀ K u_d (fp64 accumulation, rigid dropped exactly)
 reaction Ŝq = Π_Rᵀ E_dᵀ K E_d Π_R q (explicit transposed sweeps)
 sensitivity s_c = −Σ_e Σ_gp ∂_τc w_e,gp · ε_gp(u)ᵀ C ε_gp(u)
 Neumann (BDD) M_P r ≈ S⁺ r: symmetric V-cycle on the free cell, restricted to ports, rigid-projected
```

One sweep 𝒮 maps a force r on the unknown DOFs to a displacement correction w, and is exactly a U-Net:

```
 fine  pre (2 layers, C0=2) ──► restrict exact residual ──► L1 block solve ──► L2 (2 layers, C2=4) ──► L3 exact
                                                                                                         │
 fine post(2 layers, C0=2) ◄── prolong ◄── L1 block post-solve ◄── prolong ◄── L2 (2 layers) ◄──────────┘
```

### 2.3 Components, mechanism by mechanism

Reference sizes are those of the FULL parent: active nodes N ≈ 122.7k; box-port nodes 7,872 (23,616 DOF); unknown
nodes n_I ≈ 114.9k (344,628 DOF); elements E ≈ 13k; ghost faces F ≈ 22k. The q-path acts on a batch of k columns.
The layout is the dense parity packing of DI §2.2: 33³ slots × 8 parity sub-lattices, so hyperedge gather and scatter
are fixed-offset slices.

**C0. Port modes and masks** (geometry only). Every call has a port mode:
- `DBOX`: box ports Dirichlet, band free. This is the lattice default for a free cut.
- `DALL`: box and band Dirichlet. Decision 4's full port; it covers skins and a displacement-controlled cut.
- `NEU`: all nodes free, for the Neumann action.

Per mode we store the unknown mask 𝕀 and the port mask ℙ. In `DBOX`, a band force f_band enters as a nodal force on
the band nodes, which are unknowns (decision 4: "forces on the cut band as nodal forces"; R18). The band is the
Γ-band of DI §2.3; on overlap the box port takes precedence.

**C1. Rigid split** (linear in q; exact).
- R ∈ R^{3N×6} holds the rigid fields on all active nodes, R_P its port rows, and W the port-node lumped-area weights
  (O_h-invariant).
- A_R = (R_Pᵀ W R_P)⁻¹ R_Pᵀ W, of size 6 × 3|ℙ|, is computed in fp64. Then α = A_R q and q_d = Π_R q with
  Π_R = I − R_P A_R.
- Cost: 6·3|ℙ|·2 ≈ 0.3 MFLOP per column.
- Mechanism: rigid motion needs no mechanics, so it never enters a learned or rounded operator.

**C2. Port force** (linear in q; exact, local). f0 = −K_{𝕀,ℙ} q_d.
- These are the out-of-balance forces that the prescribed port displacement exerts on the first interior node layer.
  This is how a displacement boundary condition acts on the body.
- It is computed only on the elements and ghost faces that touch ports: 1,776 elements for the FULL box; 1.3–2.2k
  elements for cut cells with the band.
- Cost ≈ 0.02–0.05 GFLOP per column.
- No lift is used. The multilevel sweep with its exact coarsest Galerkin transmits the macro content. A global
  polynomial lift would be an ill-conditioned least-squares fit in heavy cuts, where the ports cover only a few
  patches.

**C3. Fine level: K-structured gated hyperedge GNO** (linear in the state; gates geometry-only).
- For layer j, the kernel is
  Ψ_j(X) = Σ_e a_{e,j} P_eᵀ K_e P_e X + γ Σ_f b_{f,j} P_fᵀ G_{ax(f)} P_f X, masked to 𝕀.
  - K_e is applied in **strain form**: sum-factorized Q2 gradients at the 5³ signed Gauss points with weights
    w_e = V⁻¹M_e (DI §1.4, exact to 1.4e-15). No 81×81 matrix is stored. Each element's rigid part is subtracted
    before the contraction, for fp32 robustness.
  - The ghost term uses the shared 54×135 factor per axis (a tensor-core-shaped GEMM).
- Why K-structured is mechanically necessary:
  - It is the only local kernel that has exactly K's sparsity (R9) and exactly the rigid kernel.
  - It is exactly O_h-equivariant.
  - It costs nothing to store. Free learned element kernels would need 81×81×C² per element (211 MB per cell at
    C = 64; DI §3).
  - The gates a_e and b_f act as **spatially variable step sizes**: they amplify or damp the smoothing on fictitious
    and cut elements. This is the "per-block damping" that route-1 identified as the next learnable parameter, after
    one global step size proved 2–3× too conservative.
- **Anchored clusters** (R7): B_j = Σ_c ω_{c,j} P_cᵀ (K_cc)⁻¹ P_c, stored as upper inverse-Cholesky factors F_c.
  - Strong pairs: node-block strength s_ij = ‖K_ij‖/√(‖K_ii‖‖K_jj‖) ≥ θ = 0.1, components of at most 4 nodes (as in
    route-1).
  - Anchored weak clusters: every weak node (‖D_i‖ < 1% of the median) is merged into the pair cluster of the strong
    node it couples to most strongly; the merged components are chunked to at most 24 nodes along L1 block
    boundaries (O_h-symmetric; §4.6).
  - Mechanism: a weak node's value is defined by its anchor (ghost-penalty continuation). Solving the anchor and its
    dependents together is exact local static condensation of the fringe onto its root.
- **Layer** (C0 = 2 vector channels; X ∈ R^{n_I×3×C0}; r is the sweep input force):

  X ← X Γ_j + B_j ( r β_jᵀ − Ψ_j(X) Λ_j ),   with Γ_j, Λ_j ∈ R^{C0×C0} and β_j ∈ R^{C0}.

  The channel mixers are shared scalars, which keeps the layer equivariant. At initialization (gates = 1, and
  Γ, Λ, β the coefficients of a 2-term Chebyshev recurrence on [λ_min, λ_max] from Lanczos) the layer is the
  route-1 smoother with two-history Chebyshev, B_j is the cluster block-Jacobi and Ψ_j = K_II.
- Receptive field: 2 pre-layers plus 2 post-layers reach about 4–8 elements from a node, covering the Saint-Venant
  and shell edge-bending decay lengths (§2.1). This is where the boundary layers live.
- Stored per cell: F_c for about 45k pair clusters and 2k weak clusters ≈ 6.5M + 10M floats ≈ 35 MB fp32 (upper
  packed); gates for 4 fine layers ≈ 4×(13k + 22k + 47k) ≈ 0.33M floats.

**C4. Level 1: 2³-element PoU aggregates, bending-capable, mechanism-enriched** (basis geometry-only; action linear).
- **Aggregates.** 16³ grid of element blocks (O_h-symmetric: block b ↔ 15−b); about 2,080 active blocks for FULL,
  402 for the heavy cut and 1,596 for 0021 (q2_hierarchy_counts).
  - PoU: node i gets weight χ_a(i) = 1/#{active blocks containing i}. The weights sum to 1, so global rigid fields
    lie exactly in the span.
  - The partition is nested: a coarser block's χ is exactly the sum of the χ of the blocks it contains.
- **Polynomial fields per block**: χ_a · {rigid 6, membrane / affine strain 6, bending / quadratic 18} = 30 columns,
  centred and scaled per block, orthonormalized by a batched Gram eigendecomposition (route-1 `affine_basis`
  degree 2), with rank-deficient columns dropped.
  - Mechanically, affine fields carry the membrane action of a wall segment and quadratic fields its curvature,
    i.e. bending.
  - We deliberately skip the Q1-vertex level (33³). A trilinear hexahedron through a 1–3-element-thick shell locks
    in bending, so its Galerkin operator would hide exactly the soft bending response we need.
  - Polynomial columns are generated on the fly from coordinates; nothing is stored.
- **Mechanism modes (GenEO / spectral-AMGe local problem).**
  - Patch p(a) = closed block plus a one-node-layer halo, at most 7³ node positions; active ≈ 100–200 nodes,
    ≤ 600 DOF (capped at 450 by keeping the weak nodes plus their anchors).
  - Pencil: Ã_p z = λ (χ_a K_pp χ_a) z. Ã_p is the **Neumann** patch stiffness (elements and ghost faces inside the
    patch only, so the patch floats). K_pp is the principal submatrix of the port-masked global K.
  - Eigenproblem: solved on the complement of the 30 polynomial columns (projected pencil); keep λ < λ_c = 0.05,
    at most k_max = 8 per block. The columns are χ_a z, K-orthonormalized per block in fp64.
  - Mechanically, small λ means a motion that is cheap when the patch floats but carries real global energy weight:
    the relative rigid motion of a sliver or island, a flap's cantilever mode, or the decoupling of a fringe pocket.
    This is the constructive form of R6: **the coarse space grows exactly where there is thin, cut or weakly
    supported material**, and is geometry-generated, not fixed-geometry (a history dead end).
  - It also replaces explicit material-component splitting: two pieces that share a block but are not connected
    within it have a zero-energy relative motion, which the eigenproblem finds.
  - Per port mode the eigenproblem is solved with the mode's mask, because clamped band nodes remove some mechanisms.
- **Level-1 operator**:
  - A1 = P1ᵀ K_II P1 is applied **matrix-free** as P1ᵀ(K(P1 y)). Storing 38-column block rows with 27 neighbours
    would be about 320 MB, which violates R14.
  - We store only the diagonal blocks A1_aa (≤ 38×38) as inverse-Cholesky factors.
  - B1 = Σ_a ω1_a P_aᵀ(A1_aa)⁻¹P_a with a learned gate ω1_a.
  - The mechanism amplitude is therefore set, in one step, by (generalized force)/(exact generalized stiffness) (§5).
- Stored per cell: mechanism columns for about 2k blocks × 3 on average × 450 ≈ 2.7M floats ≈ 11 MB; A1_aa factors
  2,080 × 38² ≈ 3.0M ≈ 12 MB.

**C5. Level 2: 4³-element PoU aggregates, quadratic** (the load paths at the scale of the necks).
- 8³ grid; about 352 active blocks (FULL); 30 columns each → n2 ≈ 10.6k.
- Nesting: quadratic × χ_A = Σ_{a⊂A} χ_a·(quadratic restricted) ∈ span L1, so P12 is exact and closed-form.
- A2 = P2ᵀ K P2 (explicit, block-sparse over the 27 neighbours, computed from element contributions at encode time):
  352 × 27 × 30² / 2 ≈ 4.3M floats ≈ 17 MB. Block inverses B2.
- Layers: C2 = 4 channels, 2 pre and 2 post, of the same form as C3, with gated A2 blocks g_AB and B2 gates ω2_A.
  Cost is negligible (≈ 17 MFLOP per A2 apply).

**C6. Level 3: 8³-element PoU aggregates, affine, exact** (global transmission, the NO's global kernel).
- 4³ grid; at most 64 blocks × 12 columns ≈ 770 DOF. A3 = P23ᵀ A2 P23, and its dense Cholesky inverse is stored in
  fp64 (4.7 MB).
- z3 = κ3 A3⁻¹ ρ3 with a scalar gate κ3 (init 1).
- Why affine here: at quarter-cell scale the coarsest level must carry the resultants (forces and moments, i.e.
  rigid motions, plus stretching) between quarter-cells. Bending at that scale is represented by L2 quadratics.
  In `NEU` mode A3 has the 6 global rigid modes in its kernel; we use the rigid-deflated pseudo-inverse.

**C7. The sweep 𝒮** (all linear in r). Given r ∈ R^{n_I×3}:
1. Fine pre-layers j = 1, 2 from X = 0. The first layer yields B_1 r β_1ᵀ, the local cluster Green's response.
2. ρ1 = P1ᵀ(r − K_II x), with x = X[:, 0]. This is the **exact** residual (gates are not used for restriction), so the
   coarse levels receive the true out-of-balance force.
3. y = B1 ρ1; ρ2 = P12ᵀ(ρ1 − A1 y) (one matrix-free K apply).
4. L2 pre-layers (C2 = 4); ρ3 = P23ᵀ(ρ2 − A2 z).
5. z ← z + P23 κ3 A3⁻¹ ρ3; then L2 post-layers.
6. y ← y + P12 z[:, 0]; y ← y + B1(ρ1 − A1 y) (one K apply).
7. x ← x + P1 y; X[:, 0] = x; fine post-layers j = 3, 4.
8. Output w = X[:, 0], masked to 𝕀.

Fine K-structured applications per sweep: 4 layers × C0 + 1 + 2 = 11 (on one column).

**C8. Two sweeps with one exact equilibrium re-entry** (answers question d):
w1 = 𝒮1(f0); r1 = f0 − K_II w1; w = w1 + 𝒮2(r1).
The two sweeps share the geometry gates and have their own channel mixers. Mechanically, the second sweep measures
what is still out of balance and corrects it. This is the only place where an "exact residual" enters beyond
restriction, and it is the continuous path to fallback A (§7).

**C9. Geometry GNN (learned; nonlinear; O_h-invariant scalars only).**
- Inputs (all invariant, dimensionless, smooth within a topology):
  - Node (~16): log(tr D_i / median); eigenvalue ratios of D_i; weak and inside-material flags; node class
    (vertex / edge / face / centre); port type; number of active incident elements /8 and incident ghost faces /12;
    maximum and mean strength s_ij.
  - Element (~20): volume fraction and its log; the 3 sorted inertia-tensor eigenvalues; centroid offset norm;
    invariant Rayleigh quotients of K_e on uniform strains (sum over the axial ones, sum over the shear ones, both
    ÷ vf); flags full / cut / band / port-touching; tau, |φ|−τ and signed plane distance at the centroid (h-scaled).
  - Ghost face (~6): sum and |difference| of the two volume fractions; full-flag count; weak count.
  - L1 block (~16): number of active and weak nodes; mechanism count and log λ (8, padded); log λ_min of the
    D-normalized A1_aa; port fraction.
  - Global (4): retained fraction, mean τ, cut flag, port mode.
- Network: node↔element bipartite message passing (3 rounds, width 48, SiLU, mean aggregation), plus one
  element↔ghost-face round; pool (mean) to L1 blocks for 3 rounds on the 26-neighbour graph (width 64); pool to L2
  for 2 rounds; unpool back (U-shaped).
  - Edge features are invariant only (no relative position vectors).
  - No max, abs or hard gates, only mean/sum pooling and smooth activations (R17).
- Heads: g = softplus(z + c0) with c0 chosen so that g = 1 at z = 0; the last layer is initialized to zero (exact
  skeleton at start).
- Parameters ≈ 0.2–0.25M. Generated coefficients per FULL cell ≈ 0.4M numbers (1.6 MB). Encoder FLOPs ≈ 3 GFLOP.

**C10. Readouts** (details in §3 e, f, g).

### 2.4 Linear in q vs geometry-only

| object | depends on q? | depends on g? | learned? |
|---|---|---|---|
| A_R, Π_R, R, masks, K (w_e, ghost), D_i, clusters F_c, P1/P2/P3 bases, mechanism modes, A1_aa, A2, A3⁻¹ | no | yes (deterministic) | no |
| gates a_e, b_f, ω_c, ω1_a, g_AB, ω2_A, κ3 | no | yes | yes (GNN) |
| channel mixers Γ, Λ, β (per layer, per sweep) | no | no (shared) | yes |
| α, q_d, f0, X, ρ_ℓ, y, z, w, u | **linear** | through coefficients only | — |

### 2.5 Cost and memory (FULL parent, fp32 unless noted)

| item | per column | notes |
|---|---|---|
| one K-structured fine apply (body in strain form + ghost in factor form) | ≈1.0 GFLOP | DI §4: 0.37 + 0.64 |
| sweep: 11 fine applies + clusters (~0.4) + P-transfers (~0.06) + L2/L3 (~0.3) | ≈12 GFLOP | |
| forward (port force + 2 sweeps + 1 re-entry residual) | **≈25 GFLOP** | ≈1.5–3 ms per column at 8–16 TFLOP/s effective [est]; batching k ≥ 8 columns amortizes the moment and weight streaming |
| energy readout | +1 GFLOP | fp64 accumulation |
| reaction Ŝq (forward + K + explicit transposed sweeps) | **≈51 GFLOP** | ≈3–6 ms per column [est] |
| Neumann action (1 symmetric V-cycle, NEU mode) | ≈7 GFLOP | ≈0.5–1 ms |
| activations (inference) | ≈123k × 3 × 2 × 4 B ≈ 3 MB per column per live tensor | ~10 live tensors |
| activations (training, k = 32, checkpointing inside hyperedge ops) | ≈1.3–2 GB | |

- **Stored per FULL cell:** topology and masks ≈ 3 MB; w_e ≈ 8 MB; D_i 3 MB; clusters ≈ 35 MB; L1 ≈ 23 MB;
  L2 ≈ 18 MB; L3 ≈ 5 MB; gates ≈ 2 MB.
  - Total ≈ 97 MB for `DBOX`. `NEU` adds about 20 MB (port clusters and the Galerkin blocks that touch ports).
  - Total ≈ 120 MB, i.e. ≈ 12 GB for 100 resident FULL cells, leaving about 15 GB for activations.
- **Encode per cell** (after moments, which cost 0.25–1 s):
  - strength graph + clusters + batched Cholesky ≈ 0.05 s
  - bases + PoU ≈ 0.02 s
  - mechanism eigenproblems (2,080 × ≤450², batched, fp32 with an fp64 Rayleigh–Ritz polish) ≈ 0.1–0.2 s [est]
  - Galerkin A1_aa / A2 / A3 ≈ 0.1 s
  - GNN ≈ 0.02 s
  - Total **≈ 0.3–0.4 s**, without any sparse factorization.
- **One BDD design iteration** (80 FULL + 20 cut cells, about 160 reactions and 55 Neumann actions per cell):
  ≈ 100 × (160 × 4 ms + 55 × 0.8 ms) ≈ **70 s** [est], against about 20 min for the exact pipeline, which moreover
  cannot hold 100 cells.

---

## 3. Answers to the open questions a–l

**a. Consuming the masked fixed-grid data.**
- Use hyperedges, not a pairwise graph and not plain grid convolutions: element hyperedges (27 nodes) and ghost-face
  hyperedges (45 nodes) on the active set, executed as fixed-offset slices of the dense parity-packed 33³×8 layout.
  - Couplings exist only where K has them (R9). Two walls that are close in space but not in material never
    exchange information unless CutFEM itself couples them through a ghost face.
  - There are no edge lists (0 MB, against 76 MB for K's pair graph).
- Element information enters in two ways:
  1. As **the kernel itself**: K_e in strain form from moments, scaled by a gate.
  2. As **invariant features** for the GNN (volume fraction, inertia invariants, Rayleigh quotients of K_e).
- Ghost faces are a separate hyperedge type with their own gates b_f. The ghost terms are what hold the fictitious
  nodes, so their effective step size is learned separately.
- Weak nodes are handled by (i) anchored clusters with exact local inverses at the fine level and (ii) mechanism
  modes at L1.
- Diagonal 3×3 blocks D_i serve four purposes: weak flags, the strength graph, cluster construction, and invariant
  features.

**b. Hard port values and exact rigid modes while staying linear.**
- Rigid part: split by a fixed fp64 least-squares projector and extended exactly as R α.
- Ports: the network outputs only the unknown-mask rows and ports carry q_d, so u|_ℙ = q holds by construction.
- Port data enter the network only through the local force map f0 = −K_{𝕀ℙ} q_d, which is linear, exact and
  independent of the number of port DOF (R2).
- On the coarse levels rigid fields are in every span (PoU); in the Dirichlet modes the ports are simply not
  unknowns.

**c. Global transmission and localized soft modes.**
- Transmission: L1 (2³ el) → L2 (4³ el) → L3 (8³ el, exact dense Galerkin inverse). The levels are PoU aggregates,
  not geometric point grids.
  - The bases are rigid + membrane + bending, i.e. they follow the load-carrying kinematics of a shell.
  - They are nested exactly, and the coarse operators are exact Galerkin products.
  - Spectral / FFT layers are excluded: a Fourier layer is the Green's operator of a homogeneous medium (RS 7.2),
    and our contrast between void and solid is infinite.
- Localized soft modes: fine-level anchored clusters plus **L1 mechanism enrichment** from local generalized
  eigenproblems. The enrichment is the new element here; see §5.

**d. Where K appears in the forward pass.** Three places, all with exact K:
1. The local port-force map.
2. The fine GNO kernels. The kernel is K_e and ghost templates with learned gates; this is the only kernel that is
   memory-feasible and rigid-exact.
3. Three exact residual evaluations: for the L1 restriction, inside L1, and one re-entry between the two sweeps.

In total about 25 K-structured applies per forward, ≈ 25 GFLOP. This is a hybrid. The reasons:
- **Cancellation.** Soft-direction stiffness is a near/far-field cancellation of 1e3–1e4 (history stage 1). A net
  that never evaluates equilibrium must reproduce that cancellation through its coefficients: a 1e-3 coefficient
  error becomes an O(1) error in the soft amplitude. An exact residual measures the out-of-balance force directly,
  which turns the cancellation into a well-conditioned correction.
- **Evidence** (RS 1.8, 3.2): consistency with an exact residual is what generalizes.
- **Cost.** K in strain form is cheap: ≈ 0.1 ms per column [est], with no storage.

The pure-feedforward limit (P = 1 sweep, no re-entry) is ablation A0 on day 1 (§6.5).

**e. Reaction readout.**
- Ŝq = Π_Rᵀ E_dᵀ K E_d Π_R q, where E_d is the deformation part of the extension; K R = 0 is used analytically, so
  no rigid rounding enters.
- E_dᵀ is the network's adjoint. Every block is linear and symmetric (Ψ_j, B_j, B1, A2, A3⁻¹ are symmetric), so the
  adjoint is the same kernels applied in reverse order with transposed channel mixers. It is an explicit program and
  needs no autograd tape at deployment. Cost ≈ 1× forward, so the reaction ≈ 2× forward + 1 K apply.
- The alternative (K û)_ℙ is rejected: it is not symmetric (RS 3.3), which breaks CG / BDD.
- When only the energy is needed (training, gate checks), use qᵀŜq = u_dᵀ K u_d, with no adjoint.

**f. Approximate Neumann action for BDD.**
- For nonsingular K, S⁻¹ = (K⁻¹)_ℙℙ. So S⁺r ≈ Π_Rᵀ R_ℙ M R_ℙᵀ Π_R r, where M ≈ K⁺ is a **symmetric V-cycle of
  the same hierarchy in `NEU` mode**:
  - symmetric pre/post cluster smoothing with Lanczos steps;
  - L1 and L2 block solves;
  - L3 rigid-deflated pseudo-inverse.
- M is SPD on the complement of the rigid modes by construction (pre and post smoothers are adjoint, coarse solves
  exact), so BDD-PCG applies unchanged (RS 3.3).
- Learned: one scalar per level (ω0, ω1, ω2), trained on the S-norm error of the inverse action against exact S⁺r.
- Cost ≈ 7 GFLOP. It need not be accurate. The step-3 check is BDD iterations ≲ 1.5× those of the exact S⁺.

**g. Sensitivity.**
- s_c = −u_dᵀ(∂K/∂τ_c)u_d = −Σ_e Σ_gp (∂w_{e,gp}/∂τ_c) ε_gp(u)ᵀ C ε_gp(u). The rigid part has zero strain. Within a
  topology the ghost term has zero τ-derivative.
- ∂w/∂τ = V⁻¹ ∂M/∂τ comes from forward-mode differentiation through polyref: one extra moment pass, or 51 MB stored.
  It is nonzero only on elements crossed by the shell surface |φ| = τ(x) and the plane.
- The formula is applied to the network field. It is the lattice sensitivity to first order in the field error.
- Partial cancellation of the first-order error:
  - The exact extension is K-harmonic and the error e = û − u* vanishes on the ports, so u*ᵀ K e = 0 exactly.
  - Hence the first-order sensitivity error 2u*ᵀ(∂K/∂τ)e vanishes for the part of ∂K/∂τ proportional to K. It is
    driven only by the surface-localized remainder.
  - This is why the design adds a sensitivity auxiliary loss (§6) rather than relying on the energy loss alone.
- Topology: all hierarchy structure (clusters, aggregates, **mechanism count k per block**) is frozen per design band
  on the superset masks of DI §1.6. Values (factors, eigenvectors, Galerkin blocks, gates) are recomputed smoothly
  within the band. The operator is then smooth in τ within the band. Sensitivities are defined within the current
  masks, as the brief requires.

**h. Equivariance.**
- Joint (g, q) O_h equivariance holds **by construction** for all continuous parts (§4.6).
- Augmentation (48 elements, on the fly) is still applied in training. It costs little: rotated exact fields come
  free by signed permutation. It covers two residual sources: tie-breaks in weak-cluster chunking, and the Kuhn-split
  asymmetry of polyref at the 1e-4 level.
- The cut plane breaks each cell's own symmetry but not the joint equivariance.

**i. Smoothness, conditioning and normalization.**
- Normalization:
  - The fine state is Jacobi-scaled (clusters give dimensionless B_jΨ_j).
  - Gates are softplus around 1.
  - Encoder inputs are logs and ratios.
- Loss per sample: μ − 1 = eᵀKe / qᵀSq, with exact normalization.
- Against the energy loss's K-conditioning (RS F4):
  1. A **D-norm field loss** ‖e‖²_D / ‖u*_d‖²_D, whose Hessian is block-diagonal D rather than K, so soft components
     get no weaker gradient than stiff ones.
  2. Initialization at a working solver (fallback B), so the energy landscape starts inside the "correct physics"
     basin.
  3. Force-driven and adversarial directions.
- Smoothness in τ comes from smooth features, frozen combinatorics per band, and softplus heads.

**j. Capacity; shared vs per-geometry.**
- The encoder weights (~0.25M) are shared. Per geometry the network generates about 0.4M gate values.
- The deterministic mechanical encoder provides everything exact: clusters, bases, mechanisms, Galerkin operators.
  **The network learns only what cannot be computed cheaply**: how to turn a slow iteration into a fast operator that
  is accurate in the loaded directions.
- A per-geometry "free-gate" mode, with gates as free parameters and no GNN, is a step-1 diagnostic only: it tests
  representability separately from generalization.
- Capacity knobs, in escalation order:
  1. More fine layers per sweep.
  2. Per-node equivariant 3×3 modulation of cluster blocks, ω_i I + β_i D̂_i.
  3. More mechanism modes (larger k_max or λ_c).
  4. More sweeps (toward fallback A).

**k. Precision.**
- fp32:
  - network, gathers, and strain-form element contractions (rigid-subtracted per element);
  - cluster and block factors, stored after local normalization.
- fp64:
  - rigid projector;
  - L3 dense inverse (≤ 770 DOF);
  - mechanism-eigenvector polish;
  - **all energy, reaction and sensitivity accumulations**.
- Risk check in step 1: μ from fp32 vs fp64 readout on the adversarial set (§7, R6).

**l. Graceful degradation.** See §7.3. In short: the same modules with more sweeps become fallback A, and with gates
frozen at 1 (only per-layer scalars learned) they become fallback B. MMX-Net is initialized at B.

---

## 4. Guarantees (short proofs)

Notation: 𝕀 and ℙ are the unknown and port node sets; E_d is the linear map from q_d to u_d = [q_d; w]; Ê q =
R A_R q + E_d Π_R q.

**4.1 Exact linearity in q.**
- The encoder takes no q input.
- Every q-path operation is a composition of maps whose coefficients are functions of g only: A_R, Π_R, K_{𝕀ℙ},
  Ψ_j, B_j, P_ℓ, A1, A2, A3⁻¹, gates, and the constant mixers Γ, Λ, β.
- The re-entry residual r1 = f0 − K_II w1 is also linear.
- Hence Ê(g) is a matrix, and E_dᵀ is the reversed composition of the transposes.

**4.2 Exact port values.**
w is supported on 𝕀 by the output mask, so u|_ℙ = R_P α + q_d = R_P A_R q + (I − R_P A_R)q = q. The subtraction is
done in fp64 on port DOF; its error is rounding only.

**4.3 Exact rigid modes.**
- For q = R_P c: α = c and q_d = 0 (since A_R R_P = I), so u = R c exactly.
- In the readout Ŝ = Π_Rᵀ E_dᵀ K E_d Π_R, we have Π_R R_P = 0, so Ŝ R_P = 0 to rounding of the fp64 projector, never
  to fp32 K rounding.
- The global rigid fields are in the span of every coarse level (PoU), so in `NEU` mode the coarse operators carry
  the rigid kernel exactly.

**4.4 Symmetric PSD readout.**
Ŝ = (E_d Π_R)ᵀ K (E_d Π_R) is a Gram matrix of the PSD K. It is symmetric PSD for **any** parameters, including
untrained, adversarially bad or diverged ones.

**4.5 One-sided error, and what it means for the lattice gate.**
1. Let E be the exact extension (K_II (Eq)_𝕀 = −K_𝕀ℙ q), and let e = (Ê − E)q, which is zero on ℙ. Then
   (Eq)ᵀKe = (K E q)_𝕀ᵀ e_𝕀 + (KEq)_ℙᵀ e_ℙ = 0 + 0. So qᵀŜq = qᵀSq + eᵀKe, i.e. **Ŝ − S = (Ê−E)ᵀK(Ê−E) ≥ 0**.
   The energy error is second order in e, and always of the "too stiff" sign.
2. Lattice: K̂_L = K_L + Δ with Δ = ⊕_i (Ŝ_i − S_i) ≥ 0. Matrix inversion is operator-convex, so
   (K+Δ)⁻¹ ⪰ K⁻¹ − K⁻¹ΔK⁻¹. Hence

   0 ≤ C − Ĉ ≤ u_Lᵀ Δ u_L = Σ_i q_iᵀ(Ŝ_i − S_i)q_i,

   where q_i are the cells' port displacements in the **exact** lattice solution. So:

   (C − Ĉ)/C ≤ Σ_i (μ_i(q_i) − 1) · w_i,   with w_i = q_iᵀS_iq_i / C.

   **The gated lattice compliance error is at most the energy-weighted mean of the per-direction training loss
   μ − 1, evaluated at the directions the lattice actually loads.** The force-driven samples (q = S⁻¹f) are those
   directions for single-cell loads. This is why the loss, the sampling and the gate are one quantity.
3. With band forces (force mode) the same argument holds for the total potential Π(u) = ½uᵀKu − fᵀu:
   Π(û) − Π(u*) = ½eᵀKe ≥ 0.

**4.6 Equivariance.**
- Let h ∈ O_h act by a signed node permutation Q_h (node permutation ⊗ ρ(h)).
- Moments transform by a signed index permutation (DI §2.4). With moments transformed rather than re-integrated,
  K(h·g) = Q_h K(g) Q_hᵀ exactly.
- Invariant features are permuted, so the GNN (mean aggregation, invariant edges) outputs gates with
  a_{h·e}(h·g) = a_e(g). Hence Ψ_j(h·g) = Q_h Ψ_j(g) Q_hᵀ.
- The 16³ / 8³ / 4³ block grids and their PoU weights are O_h-symmetric. The polynomial and eigen spans are
  O_h-covariant subspaces. Block inverses, Galerkin products and scalar channel mixing are basis-independent.
- Strength-graph components are equivariant.
- Therefore Ê(h·g) = Q_h Ê(g) Q_{h,ℙ}ᵀ and Ŝ(h·g) = Q_{h,ℙ} Ŝ(g) Q_{h,ℙ}ᵀ, except for two sources:
  (i) chunking ties in weak clusters (rules in C3: split along the symmetric L1 blocks, and assign a shared node to
  the adjacent block with larger incident material volume; residual ties are rare); (ii) the polyref Kuhn asymmetry
  (~1e-4) when moments are re-integrated for rotated geometry.
- Both are monitored (§6) and covered by augmentation.

**4.7 A safe sub-family with guaranteed contraction** (used for the Neumann action and available for the forward
pass).
- Constrain post-layers to be the adjoints of the pre-layers, use a single channel, and choose ω with
  ω·λ_max(B Ψ) ≤ 1 (Lanczos, not the row-sum bound, which is 2–3× loose).
- Then each sweep is a symmetric V-cycle with 0 < B_V K ≤ I, so the error propagation I − B_V K has K-norm < 1.
  Adding sweeps never increases the energy error.
- The full main line relaxes this (free Γ, Λ) for capacity. The readout guarantees 4.1–4.5 hold regardless.

**4.8 One-step mechanism amplitude (why mechanisms need no learned cancellation).**
- Let z be a K-normalized L1 mechanism column of block a with zᵀ K z = κ_z ≪ 1 (relative), and let y be the
  remaining columns. If the strengthened Cauchy–Schwarz constant between span{z} and the other columns is
  θ = |zᵀKy| / √(κ_z yᵀKy), then the block solve gives the Galerkin amplitude zᵀρ/κ_z with relative error O(θ²).
- Both numerator and denominator are exact: ρ is an exact residual, and κ_z comes from exact moments.
- θ is small precisely because a mechanism is weakly coupled; that is what makes it a mechanism.
- The soft amplitude is therefore a quotient of exact quantities, not a difference of large learned ones.

---

## 5. Why it gets the soft, force-driven directions right

The softest decile of the whitened box spectrum carries 94–96% of the lattice load energy. It contains three kinds of
mechanical content, and each has a dedicated, exact carrier.

**(S1) Global compliance: load paths and neck bending.**
- A force entering one face is carried through the necks, as membrane force, and around the saddles, in bending.
- Carriers:
  - L3 exact Galerkin solve (global rigid and stretching of quarter-cells);
  - L2 quadratic fields (neck bending at 4-element scale);
  - L1 quadratic fields (wall curvature at 2-element scale).
- Because all coarse corrections are exact energy minimizations over their subspaces given an exact residual, the
  error left after them is K-orthogonal to those subspaces (Galerkin).
- Route-1 evidence: quadratic spaces in this exact arrangement already drove force-driven lattice errors to 0.72% /
  1.7% (24 learned layers) and 0.04% (64 layers).
- MMX-Net keeps this carrier and adds L2 quadratics. Route-1 had only two coarse levels, each with 12 or 30 columns
  per patch.

**(S2) Local mechanisms: slivers, flaps, islands, fringe pockets.** These are the dense band.
- A pointwise or pair smoother cannot resolve them: they are slow for it by definition (RS F5). A polynomial coarse
  space cannot represent them: the sliver moves and its surroundings do not. Global low rank cannot hold them:
  there are thousands of them.
- MMX-Net computes them **all, locally**, from the exact K. A block's pencil (Neumann patch stiffness vs PoU-weighted
  global stiffness) has a small eigenvalue exactly when part of the block can move cheaply relative to the rest
  while carrying global energy weight.
- The number of columns therefore scales with the thin and cut material, not with a fixed budget: about 0 per block
  in the solid core, up to 8 per block on slivers. This contrasts with E1, which added only the 128 global slowest
  modes, when there are thousands.
- By §4.8 their amplitudes follow from exact generalized force over exact generalized stiffness in one block solve.
- GenEO-type theory: a two-level method whose coarse space contains all local eigenvectors below λ_c has a condition
  number bounded by a function of λ_c and the overlap multiplicity only, independent of contrast. So the minimum
  eigenvalue can no longer sit on the dense band at ~0.004–0.006: it is lifted to O(λ_c).
- This is the step-1 experiment X0 (§6.5). If it fails, the design has to change (risk 1, §7).

**(S3) Boundary layers and weakly supported port nodes.**
- A force-driven q moves soft port nodes a lot: box-patch nodes on thin necks or fringes, and in `DBOX` the band is
  interior. The extension must carry that motion into the adjacent fringe with the right ghost-penalty continuation.
- Carriers: the anchored clusters (exact local condensation of the weak nodes onto their anchors, R7) and the gated
  fine layers, whose receptive field covers the Saint-Venant and edge-bending decay lengths (§2.1).
- This is also R11: in soft directions the fictitious nodes' γ-energy is comparable to the physical energy, and exact
  local condensation gets it right without learning.

**Training closes the remaining gap.**
- After (S1)–(S3), the learned gates and mixers mainly tune step sizes and polynomial acceleration.
- They are trained on exactly the gate quantity:
  - μ − 1 on force-driven directions (30%) and adversarial worst directions (20%), whose energy weight is the lattice
    compliance error bound of §4.5;
  - a D-norm term that keeps soft components visible to the gradient (RS F4);
  - a sensitivity term on the surface elements.
- Smooth probes alone are insufficient (0.3% vs 20–35%), so they are only 50% of the mix.

**Where the error ends up.**
- The residual error is concentrated in stiff, local, high-frequency components that the second sweep's fine layers
  damp. These carry little of the lattice energy.
- The readout counts every error only quadratically.

---

## 6. Training for step 1 (three fixed geometries)

**6.1 Data** (the teacher pool, which avoids holding three factorizations during training).
- Geometries: heavy cut (0013), medium cut (0021) and their FULL parent, second-version data. Cut cells run in
  `DBOX` (70%) and `DALL` (30%).
- Per geometry, one exact fp64 cuDSS factorization, run offline and one geometry at a time. It produces a pool of
  exact pairs (q, u*) on host RAM: 4k per geometry ≈ 6 GB total for FULL-size fields.
- Force-driven samples come from one **free-cell Neumann solve** K u = [f; 0] (rigid-projected), which gives
  **q = u|_ℙ and u* = u at once**.
- Class mix (decision 9):
  - force-driven 30%: smooth multiscale f on the box, ~10% with band forces;
  - macro 20%: rigid, 6 uniform strains, quadratic, cubic;
  - multiscale GRF 30%: correlation lengths from a whole face down to 2 elements;
  - adversarial 20%, after 2k steps.
- O_h augmentation is applied on the fly by signed permutation of (g, q, u*): 48× data at zero solve cost.
- Adversarial pool: every 500 steps, per geometry, load that factor and run block LOBPCG (k = 32, 20 iterations) on
  (Ŝ − S)v = λ Sv. S v comes from the factor, Ŝ v from the network reaction. The top 32 directions and their exact
  u* go into a FIFO of 512 per geometry. A second generator draws worst **force** directions from the same pencil
  mapped through S⁻¹.

**6.2 Loss** (per sample i, all fp64 reductions):

L = Σ_i w_i (μ_i − 1) + λ_D(t) Σ_i ‖e_i‖²_D / ‖u*_{d,i}‖²_D + λ_s(t) Σ_i Σ_c (ŝ_{i,c} − s_{i,c})² / Σ_c s²_{i,c}

- μ_i − 1 = e_iᵀKe_i / q_iᵀSq_i, i.e. qᵀŜq/qᵀSq − 1, with exact normalization (decision 7).
- Weights w_i: 2 for force-driven and adversarial samples, 1 otherwise.
- λ_D decays from 1 to 0.05 over the first 40% of the steps (F4 warm start). λ_s rises from 0.1 to 1 after warm-up.
- The label-free variant (for step 2 and later) replaces qᵀSq by a detached deep-sweep reference, as in route-1 E1b.
  With the upper bound of §4.5 the minimizer does not change.

**6.3 Optimizer and schedule.**
- AdamW with separate learning rates: GNN 3e-4, mixers 1e-3, head bias 1e-3. Cosine schedule, 20k steps,
  gradient-norm clip 1.0.
- Batches: one geometry × one random group element × 32 columns, geometries in round-robin.
- fp32 forward with fp64 reductions; checkpointing inside the hyperedge ops.
- No gradient flows through the exact solver.
- Estimated cost ≈ 0.15 s per step on FULL, about 1–2 h per run [est].

**6.4 What to monitor** (every 500 steps; lattice every 2k).
1. μ quantiles (50/90/99/max) on fixed held-out sets per geometry and class: force-driven 256, GRF 256, macro 64,
   band-loaded 64.
2. Witness μ_max via LOBPCG (reported, not gated).
3. **Lattice gate**: lattice_v2 x/y configurations (cut cell plus FULL neighbour). Two variants: only the cut cell
   replaced by the network, and both cells replaced. Also a 2×2×2 FULL-only lattice. Compliance and per-cell
   sensitivity errors vs exact.
4. Error localization: the energy error eᵀKe split by node class (weak / strong / port-adjacent / band) and by
   level-1 block. This is the mechanics diagnostic that shows **which carrier is failing**.
5. The implied preconditioner's spectrum: Lanczos λ_min of the sweep as an operator; cluster and mechanism counts.
6. Gate statistics: range, saturation, and correlation with volume fraction and weakness.
7. Equivariance error ‖Ê(h·g)h·q − h·Ê(g)q‖ / ‖Ê q‖ for random h.
8. fp32 vs fp64 readout gap on the adversarial pool.

**6.5 Step-1 experiment ladder** (cheapest first; each has a kill or continue criterion).

| id | what | cost | pass criterion → next |
|---|---|---|---|
| X0 | **No learning.** Route-1 skeleton on 0013/0021/FULL with (a) PoU aggregates, (b) + L1 mechanism modes, (c) + L2 quadratic. Measure Lanczos λ_min(B_V K), the fraction of the slowest-64 modes' energy captured by span(L1), and the Chebyshev witness and lattice error at 2, 4, 8, 16 cycles | ~1–2 h GPU | λ_min raised ≥ 5× over 0.0070 (0013), and lattice error < 3% at ≤ 8 cycles |
| X1 | **Free gates** per geometry (no GNN) at P = 1 and 2 sweeps, V(2,2) | ~2–4 h | force-driven μ−1 p90 ≤ 1% and lattice ≤ 3% at P = 2 → representability shown |
| X2 | **Shared GNN** on the 3 geometries with augmentation | ~4–8 h | matches X1 within 1.5× |
| X3 | Ablations: no mechanism modes; affine-only L1; P = 1 (pure feedforward A0); no D-norm term; `DALL` vs `DBOX` | ~4 h | ranks the components; decides whether the re-entry stays |
| X4 | Transfer probe: train on two geometries, test on the third (FULL ↔ cut) | ~2 h | an early read on step-2 generalization |

---

## 7. Risks, cheapest exposing experiment, graceful degradation

### 7.1 Ranked risks

1. **Local enrichment does not remove the soft band.** The modes may straddle blocks, λ_c may be mis-set, or the slow
   modes may be less local than 11–30 nodes suggests. The weak-merged clusters raised λ_min by only 14%, which
   hints that the slow modes are not confined to single ≤32-node clusters.
   - *Exposed by:* X0 (b) vs (a). Check λ_min and the captured slowest-64 energy fraction (target ≥ 90%).
   - *Response:* enlarge the halo to one element (GenEO overlap), use a second shifted block partition (overlap 2),
     raise k_max to 16 and λ_c to 0.1.
   - If κ still stays above 50, the main line needs more sweeps: move toward fallback A (§7.3).
2. **Learned gates add too little over fixed ones**, so depth stays near route-1's 24.
   - *Exposed by:* X1 vs X0 at equal depth. Target: the same error with ≤ half the sweeps.
   - *Response:* capacity knobs 2–3 (§3 j); otherwise accept P = 3–4 (fallback A).
3. **The training signal stalls on the soft directions** (RS F4): the mean μ improves while force-driven and
   adversarial μ do not.
   - *Exposed by:* monitor 1 vs the training loss, per class, in X1.
   - *Response:* a larger λ_D, a higher adversarial share, a per-class μ cap loss (mean of top-10%).
4. **Throughput and memory above budget.** The ~0.1 ms/column K-structured apply is an estimate.
   - *Exposed by:* a microbenchmark of one strain-form + ghost apply on FULL (k = 1, 8, 32 columns, fp32) before
     any training.
   - *Response:* C0 = 1 in post-layers, V(1,1) with P = 3, a fused kernel, fp16 storage of the normalized cluster
     factors.
5. **`DBOX` (band interior) is much harder than `DALL`**, because the slivers become interior unknowns.
   - *Exposed by:* X3 on 0013.
   - *Response:* keep `DALL` for training and band loading; handle the free cut in the lattice by condensing the
     band inside the cell with a few PCG steps on Ŝ_band,band (Neumann action as preconditioner). This costs more
     but needs no new component.
6. **The fp32 readout loses soft energy.**
   - *Exposed by:* monitor 8.
   - *Response:* an fp64 strain contraction on the ≤ 10% of elements that carry ≥ 90% of eᵀKe (adaptive).
7. **Sensitivity error exceeds compliance error** (first order).
   - *Exposed by:* the lattice sensitivity monitor, plus a within-topology finite-difference check of s_c.
   - *Response:* raise λ_s; weight the D-norm loss toward surface elements.
8. **Encode time grows because of the eigenproblems.**
   - *Exposed by:* timing X0's encoder.
   - *Response:* run the eigenproblems only on blocks that contain weak or cut elements; use batched LOBPCG for
     ≤ 8 vectors.
9. **Combinatorics are discontinuous in τ** (cluster or mode count flips).
   - *Exposed by:* a τ ±1e-4…1e-2 sweep on 0021 with frozen band structure, checking that Ŝ and s vary smoothly.
   - *Response:* freeze the structure per superset band (§3 g).

### 7.2 What cannot go wrong structurally

Linearity, exact port values, exact rigid modes, a symmetric PSD Ŝ, the one-sided bound and the lattice bound of
§4.5 hold for every parameter value. Failures show up as **accuracy**, never as indefiniteness, sign flips or rigid
leakage. The history lists those as the failure modes of stages 0, 4 and 7.

### 7.3 Degradation into the fallbacks (shared components)

- **Fallback A (learned iteration)**: increase P. The sweep 𝒮 on the exact residual *is* the learned iteration
  body. Per-sweep mixers become per-iteration coefficients; gates stay shared. With the symmetric sub-family (§4.7)
  every added sweep provably reduces the energy error. No code changes.
- **Fallback B (route-1 skeleton)**: freeze all gates at 1 and C0 = 1, and learn only per-sweep (α, β) or Chebyshev
  scalars. This is exactly route-1 E1b plus the mechanism enrichment, PoU and an L2 quadratic. MMX-Net is
  initialized here, so the main line starts from B's accuracy at equal depth.
- **The Neumann action** is B-type in every variant, so the BDD path (step 3) is unaffected by which line wins.
- **The route-2 primal-dual certificate** attaches to any variant (the field û is exact on ports). It gives a
  teacher-free error estimate once implemented on GPU.

---

## 8. Borrowed vs new

**Borrowed (adapted):**
- The linear V-cycle as a neural operator: MgNO (He–Liu–Xu, ICLR 2024); the MgNet data/feature spaces.
- Smoothed-aggregation near-nullspace coarse spaces (Vaněk–Mandel–Brezina 1996) and energy-minimizing bases
  (Xu–Zikatanov 2004), becoming PoU aggregates with rigid, affine and quadratic fields.
- Local generalized-eigen coarse enrichment: GenEO (Spillane et al. 2014), spectral AMGe, CEM-GMsFEM, which becomes
  the L1 mechanism modes.
- Patch/block treatment of small cuts (de Prenter et al. 2023) and aggregated FEM (Badia et al.), which become the
  anchored clusters.
- Learned smoothers and a consistent exact residual (Hsieh et al. 2019; Huang–Li–Xi 2023), which become the gated
  fine layers and the re-entry sweep.
- Matrix entries as network inputs (Luz et al. 2020) and hypernetwork coefficients (Ha et al.; VarMiON / NGO),
  which become the invariant K-derived features and the gates.
- A symmetric linear coarse inverse for Krylov methods (Melchers–Dolean–Abdelmalik 2026), which becomes the Neumann
  action.
- BDD (Mandel 1993); energy loss (Deep Ritz / VINO); the natural-gradient insight (Müller–Zeinhofer), which becomes
  the D-norm term; symmetry augmentation (Brandstetter et al.).
- Our own route-1 skeleton, Lanczos steps, pair clusters and weak-merge clusters.

**New in this design:**
1. **Mechanism enrichment inside a learned operator.** Mechanism modes are computed from the exact CutFEM stiffness
   per aggregate and placed where route-1 had only polynomials, so the operator's capacity follows the thin and cut
   material automatically.
2. **A coarse hierarchy chosen by shell mechanics.** Rigid, membrane and bending fields on PoU aggregates; the Q1
   level is deliberately skipped (locking); quadratics are used where necks bend; affine fields at the coarsest
   level carry resultants.
3. **A K-structured, gated hyperedge GNO.** The fine kernel is the exact element stiffness in strain form with
   learned invariant gates. It is exactly equivariant, rigid-exact and storage-free, and it equals a known-good
   smoother at initialization.
4. **Port handling as mechanics.** A local exact port-force map instead of a lift or learned encoder; runtime port
   modes (`DBOX` / `DALL` / `NEU`) on one mask set; band loading as nodal forces with reciprocity-consistent
   readouts.
5. **Rigid-exact readout algebra.** Ŝ = Π_Rᵀ E_dᵀ K E_d Π_R never applies K to a rigid field. This fixes the 12×
   lattice error mode of R4 by construction.
6. **A theorem linking training loss and gate.** The lattice compliance error is at most the energy-weighted mean of
   the per-direction μ − 1 at the actually loaded directions (§4.5). It justifies the loss, the sampling mix and the
   adversarial slot as one objective.
7. **The K-orthogonality argument for the sensitivity error** (§3 g), which motivates a dedicated
   surface-sensitivity loss.
