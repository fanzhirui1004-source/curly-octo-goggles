# v0 architecture from the operator-learning lens: HGM-NO (Hyper-Galerkin Multilevel Neural Operator)

Author lens: operator-learning mechanisms (kernel integrals, multiscale GNO, hypernetwork-generated kernels, global
mixing, expressivity, equivariance). This is a complete architecture, not only that lens's part.
Inputs read: ARCH_BRIEF.md, DATA_INTERFACE.md, RESEARCH.md, the three repository docs (design, history, routes),
route-1 `lattice_v2.py`, `r7/lattice_pcg.py`. New counts computed for this proposal: `arch/_ol/hier_counts.py` and
`arch/_ol/patch_counts.py`, with outputs in `hier_counts.json` and `patch_counts.json`. Both use uniform-tau
estimates: body-only weak proxy, sampled moments, and the same rules as `count_topology.py`.

---

## 0. The design in one page

**Central claim.** The main line is "a feedforward neural operator combined with a multiscale GNO". In a
fixed-grid FE setting, the operator-learning analysis gives this line a precise form:

1. **The fine-level kernel of the network should be the exact stiffness K itself, applied matrix-free from the
   moments.**
   - K is the kernel that a perfect hypernetwork would output on the active-node hypergraph.
   - It is also the cheapest fine-level kernel that can couple nodes: about 1 GFLOP per column for FULL, one
     channel.
   - A learned fine kernel costs at least as much. It also has to reproduce near/far-field cancellations at 1e-3
     to 1e-4 relative accuracy: the same accuracy on soft modes that the dense-factor route failed at (entry
     accuracy 3e-4 needed, 3 to 6 achieved).
2. **Learned capacity belongs where exact numerics is either conservative or expensive.** That means three
   things:
   - the per-block relaxation (route 1: the row-sum step was 2-3x too conservative);
   - the effective (homogenized, locking-relieved) coarse operator (B2: learned coarse materials, 23k numbers, μ in
     [0.924, 1.150]);
   - the fixed polynomial and recurrence that combines the sub-steps (route 1: 2 learned scalars per layer halved
     the depth).
   All three are local, smooth functions of geometry, so a hypernetwork can generate them and they can generalize.
3. **q passes through a fixed-depth linear DAG** with no q-dependent step sizes (Krylov and CG are forbidden
   because they are nonlinear in q). All coefficients are generated once per geometry.

**The architecture** has two paths:

- **Geometry path (nonlinear, once per geometry).**
  - Moments give K in strain form (exact).
  - A pair-block smoother (exact block inverses).
  - Local low-energy modes on two staggered macro-element patch partitions (exact, GenEO-type).
  - A nested Q2 three-level hierarchy Q2(32³) ⊃ Q2(16³) ⊃ Q2(8³) ⊃ Q2(4³), with Galerkin coarse operators, learned
     per-macro-element effective Lamé moduli, and an exact dense level-3 Cholesky.
  - An O_h-equivariant hypernetwork (isotropic-kernel U-Net on the vertex grid) that emits about 86k invariant
     scalars per FULL cell: dampings, mode gates, coarse moduli and step coefficients.
  - Lanczos certification of every sub-step.
- **q-path (strictly linear).**
  - An exact rigid/affine lift of the port data.
  - n_K = 8 sub-steps that alternate a fine smoother S and a global coarse correction C. Each sub-step is followed
     by one exact matrix-free K residual, and the sub-steps are combined by a learned two-history recurrence.
  - Deep supervision at sub-steps 4, 6 and 8.

**Readout.**
- Ŝ = Êᵀ K Ê, computed by the explicit transpose sweep.
- Energy and sensitivity use fp32 strains with fp64 accumulation.
- The last residual gives a free a-posteriori error indicator.

**Neumann (BDD) action.** A palindromic S-C-S sequence on the floating cell, with the same components. It is
symmetric positive definite by construction.

**Port model.**
- Dirichlet on the box nodes.
- On the cut band, a per-node port-type mask:
  - **v0 default: nodal-force input (Neumann)**. A free cut is f_C = 0, so the lattice needs no extra interface
    unknowns, and a loaded cut is f_C ≠ 0.
  - **Dirichlet on the cut band**, for the future skin: the same code, mask switched.
- Both modes keep exact port values, linearity and the one-sided bound.

**Budget, FULL cell (estimates):**

| quantity | value |
|---|---|
| stored geometry state | ≈ 110 MB (vs 6.6 GB for the exact fp32 factor) |
| forward pass | ≈ 10 fine K-equivalents ≈ 10.5 GFLOP ≈ 0.35-0.7 ms per column (batched) |
| symmetric reaction | ≈ 2.1× the forward pass |
| Neumann action | ≈ 0.15 ms |
| encoding, excluding moments | ≈ 0.1-0.3 s |
| hypernetwork parameters | ≈ 0.7M |

**Knob toward fallback A:** n_K. Extra "certified" steps only improve accuracy. **Knob toward fallback B:** freeze
the hypernetwork heads to constants.

---

## 1. Requirements (R1..R20), each tied to evidence

| # | Requirement | Evidence / source |
|---|---|---|
| R1 | q enters only linear operations. The network output is Ê_g q, with no q-dependent normalisation, gating or step size (so no CG or Krylov inside). | Decision 5. RESEARCH F2: an SPSD readout needs linearity. Convex-element paper: q-nonlinear fields give indefinite Hessians. |
| R2 | Dirichlet port values are exact (hard) and the readout uses the exact K. | Decision 1. Galerkin identity Ŝ − S = (Ê−E)ᵀK(Ê−E) ⪰ 0. 1D example: 32% field error → 1% stiffness error. RESEARCH F3. |
| R3 | Rigid modes are exact: Ŝ R = 0 to fp64 rounding, and the rigid part never enters K numerically. | Decision 6. R7_15: without exact rigid projection, the cut-cell energy in a 2-cell lattice is 12× wrong. |
| R4 | Box ports at full resolution. Cost scales with active nodes, not with port DOF. | Decision 2. |
| R5 | Fixed background grid. Masks, and the superset masks for design bands, are inputs. Sensitivity is taken within one topology. | Decision 3. topo_change: τ ±1e-3 changes 4-6 active elements. superset_encoder at 1e-14. |
| R6 | The cut band is a load-carrying port through nodal forces. A free cut in the lattice must not add interface unknowns. | Decision 4. DATA_INTERFACE §2.3: a Dirichlet Γ-band is 3-4× the box DOF (≈31k unknowns per heavy-cut cell). |
| R7 | Accuracy where the lattice loads: μ − 1 ≲ 1e-2 in lattice-loaded directions, and sensitivity (first order) within a few percent. | 94-96% of lattice energy is in the softest decile. Force-driven error 20-35% at 0.3% probe error (D2). Route-1 lattice: 24 learned layers give 0.72/1.7% compliance and 0.93/2.6% sensitivity. |
| R8 | An exact, undamped local carrier for slow modes on the weak sheet, with weak nodes grouped together with strong neighbours. | Slow modes: 94-99.4% on weak nodes, 11-30 nodes each. Merged groups damped by a global step helped little (λ_min 0.0070 → 0.0080). Separating weak from strong hurt (1.26 → 3.80). **New (this doc):** weak nodes percolate into 2-5 giant sheets (the fringe of the shell). Connected components are therefore unusable as clusters and **patches must be spatial** (§2.7). |
| R9 | Global transmission in every application. Coarse spaces must be material-aware and bending-capable (no Q1 locking on 1-2-element walls). | Global load paths (brief §4). Route 1: quadratic coarse spaces far better than affine (64-layer witness 1.26 vs 7.75). Walls are 1-2 fine elements thick. |
| R10 | Symmetric PSD reaction. A symmetric PSD approximate Neumann action S⁺ for BDD. | Brief §5 BDD: 11-53 iterations with a Neumann action, 300-1600 without. RESEARCH 3.3 and 5.10 (Melchers et al.). |
| R11 | Per-cell resident state ≪ 300 MB, 100 cells on one 32 GB card, no factorization memory. | R7_13: FULL factor 6.6 GB, dense T 4.5 GB. |
| R12 | A few ms per cell-application amortized. Encoding (excluding moments) well under the exact factorization time. | Brief §6: about 2e4 applications per design iteration. Factorization takes 0.08-0.85 s (fp32). |
| R13 | O_h (48-element) equivariance, joint in (g, q). | Decision 6. Pipeline equivariant to 1e-7. EquiModel. Augmentation helped held-out cut cells (0.64 → 0.41). |
| R14 | The training signal is physics: energy normalised per sample, worst-direction pressure, no entry losses. | History phases 0/4/7: entry losses fight soft directions, and averages hide the worst direction. Decision 7. |
| R15 | Smooth parameterisation, with no global max/abs heads. | Phase 7: a non-smooth head broke the learned local factors. |
| R16 | Sensitivity −uᵀ(∂K/∂τ)u uses the exact ∂K/∂τ from moments. The ghost-penalty term has zero τ-derivative within a topology. | DATA_INTERFACE §1.5 and §4. |
| R17 | fp32 network; fp64 for reductions, rigid projection and energy accumulation. | R7_15: fp32 matrices err 0.8-2.2%. For fields the error is second order (RESEARCH F3). DATA_INTERFACE §4 (strain form). |
| R18 | Degrades gracefully into fallback A and fallback B with shared code. | Decision 8. |
| R19 | Generalizes across geometries (step 2): local coefficient maps, self-diagnosis when out of distribution. | Step-2 plan. EquiModel had no train/test gap on FULL. |
| R20 | None of the dead ends: no boundary compression, fixed-geometry coarse bases, bandwidth truncation, one-sided low-rank corrections, entry regression, or proximity edges. | History §3.1. RESEARCH F6. |

---

## 2. Architecture

### 2.1 Where learning goes, and why exact K sits inside a "feedforward" operator

Let A = K_II denote the stiffness on the free nodes, with ports Dirichlet. The target is E_g q = L q + Z A⁻¹(f − K L q)
for any lift L that matches q on the ports.

**(i) The accuracy argument.** A K-free linear network N_θ has to represent A⁻¹ acting on the lift residual.
- In soft directions the relevant stiffness comes from near/far-field cancellation of about 1e3-1e4 (the
  bandwidth-truncation dead end), and cond(A) is 1e6-1e7.
- A relative coefficient error δ in the learned internal stiffness gives a field error of about
  A⁻¹ δA A⁻¹ r, amplified by 1/λ_soft.
- So the learned coefficients would need about 1e-4 relative accuracy on soft modes. That is exactly the
  entry-accuracy wall of phases 0 and 4 (3e-4 needed, 3-6 achieved), moved inside the network.

With exact residuals r_j = f − K u_j:
- The cancellation happens in exact arithmetic (fp32 is enough, see §2.8).
- The learned operators only need to be **spectrally equivalent preconditioners** (O(1) accuracy). Errors of 10-50%
  in a damping or a coarse modulus cost convergence rate, not correctness.

**(ii) The cost argument.**
- In strain form, K costs ≈ 0.37 (body) + 0.64 (ghost) ≈ 1.0 GFLOP per column for FULL, with 125 weights per
  element and 3 ghost templates (DATA_INTERFACE §4).
- Any learned fine hyperedge layer with C ≥ 1 channels costs at least as much and needs per-element coefficient
  storage.
- Using K as the fine kernel is therefore weakly dominant in cost and strictly dominant in accuracy.

**(iii) Consequence.** The network is feedforward in the operator-learning sense:
- fixed depth, a fixed linear DAG, trained end to end;
- no convergence loop, no Krylov, no q-dependent scalars.

Its fine layers are exact-K residual evaluations. Its learned multiscale GNO is:
- the coarse hierarchy with hypernetwork-generated moduli and transfers (the "feedforward NO" half: global, dense
  grids, one shot per visit);
- the fine patch kernels (the "GNO" half: local, on K's hypergraph).

The number of fine K evaluations, n_K, is the depth.
- Pure K-free feedforward is kept as an **explicit step-1 ablation** (§6.6) that tests this claim.
- More K evaluations is literally fallback A (§7).

### 2.2 Data flow (FULL numbers; heavy-cut 0013-like ≈ 1/5)

```
GEOMETRY PATH (once per geometry / design step; nonlinear allowed)
 moments M (E×125, fp64→Gauss weights w_e, E×125 fp32)  masks, element map (E×27), GP faces (F×3)
   │
   ├─► K apply (strain form, matrix-free)  ──────────────────────────────┐ exact kernel, reused by q-path + readout
   ├─► node diag blocks D_i, pairing (canonical, tie-merged) → pair-block Cholesky {A_bb^-1}
   ├─► 2 staggered macro-patch partitions → local gen. eigenproblems (A_pp, M0_pp) → modes V_p, λ_p (k≤8)
   ├─► Galerkin λ/μ/γ parts of Q2(16^3) macro elements, Q2(8^3), GP-face templates
   ├─► HYPERNETWORK H_θ (O_h-isotropic U-Net 33^3→17^3→9^3→5^3 + heads)
   │     outputs (all invariant scalars): ω_b (pair damping), g_pk (mode gates), θ^λ,θ^μ (coarse moduli, lvl 1,2),
   │                                      ω_v (coarse smoother dampings), (α_j, β_j) j=1..16, Neumann α's
   ├─► Ã1 = Σ_E θ^λ_E A^λ_E + θ^μ_E A^μ_E + A^γ ;  Ã2 (same, level 2) ;  Ã3 = P3ᵀ Ã2 P3 → dense Cholesky (fp64)
   ├─► floating (Neumann) level-3 factor with rigid deflation
   └─► Lanczos certification: caps c_S = 1.9/λmax(S A), c_C = 1.9/λmax(C A) (≈ 30 fine K + coarse applies)

Q-PATH (strictly linear in (q_B, f_C))
 q_B (N_b×3), f_C (Γ-band nodal forces; 0 = free cut)
   │  rigid/affine fit (fp64, 12×12)          u0 = A_aff α + Π_B(q_B − A_aff,B α)      (u0 = q on ports)
   ▼
 r0 = f − (K u0)_free   [K never applied to the rigid part]
   for j = 1..n_K (T_j alternates S, C, S, C, ...):
        z_j = T_j r_{j-1}                     (S: fine pair+patch smoother;  C: P1 · Vcycle(Ã1,Ã2,Ã3) · P1ᵀ)
        d_j = α_j c_{T_j} z_j + β_j d_{j-1}   (learned two-history recurrence; certified tail: β=0)
        u_j = u_{j-1} + Z d_j ;  y_j = K Z d_j (1 fine K) ;  r_j = r_{j-1} − (y_j)_free
   û = u_{n_K}      (deep supervision at j = 4, 6, 8)
   ▼
READOUT: energy ûᵀKû (fp64 acc.), reaction Ŝq = Êᵀ(Kû) by explicit transpose sweep, sensitivity −ûᵀK'_τ û,
         indicator η² = r_nᵀ C_η r_n
```

### 2.3 Port model and lift (questions b, and decision 4)

**Port-type mask per node.**
- D (Dirichlet): box-port nodes, always; cut-band nodes only in "cut-Dirichlet" mode.
- N (force input): cut-band nodes in the default mode.
- F (free): everything else.
- Γ-band = the 27 nodes of each active element whose plane section contains positive-area material. A node on
  both the box and the band belongs to the box (DATA_INTERFACE §2.3).

**Mixed variational problem.** Minimize Π(u) = ½uᵀKu − f_Cᵀu_C subject to u_D = q_D.
- The free set I = F ∪ N. The correction space is Z = the injection of I.
- f_C = 0 is exactly the free-cut lattice case, with no extra interface unknowns.
- f_C ≠ 0 gives the cut-loaded cases, computed and reported.
- The union-Dirichlet operator of decision 4 and this mixed operator are partial Legendre transforms of each other.
  For exact operators they carry the same information.
- We train the mixed one because it is what the step-1 lattice evaluates directly.
- **This interpretation of decision 4 should be confirmed with the owner in step 0.** Both modes use the same arrays,
  and the harness supports both.

**Rigid and affine lift (linear, fp64).**
- W = diag(w_i ⊗ I₃), with w_i = ∫_{box} N_i dA the geometric Q2 face weight. W is invariant under O_h.
- A_aff has 12 columns: 3 translations, 3 rotations and 6 symmetric uniform strains, evaluated on all active
  nodes. Q2 represents them exactly.
- Fit: α = (A_aff,Bᵀ W A_aff,B)⁻¹ A_aff,Bᵀ W q_B.
- Lift: u0 = A_aff α + Π_B(q_B − A_aff,B α). So u0 = q exactly on D, and in the interior u0 is the affine (Taylor)
  guess.

**Rigid exactness.**
- The residual r0 = f − K(A_strain α_ε + Π_B q_rem) never applies K to the rigid columns: they are a symbolic zero,
  because B R ≡ 0 in strain form.
- For a rigid q_B: α_ε and q_rem are at fp64 rounding level, r0 ≈ 1e-16|q|, and û = Rβ to fp64 rounding.

### 2.4 Geometry path, part 1: exact products (no learning)

| product | shape (FULL, τ≈0.5) | cost | notes |
|---|---|---|---|
| Gauss weights w_e = V⁻¹M_e | 12.9k × 125 fp32 (6.4 MB) | trivial | fp64 solve then cast; K_e = Σ_q w_eq B_qᵀCB_q exact (1.4e-15) |
| K apply, strain form | – | 1.0 GFLOP / column | body sum-factorised at 5³ Gauss points + ghost factor form (54×135 per face, 3 templates); fp32, **TF32 off** |
| node diag blocks D_i | 121k × 3×3 | from a 125×27×3×3 slice + ghost constants | features and pairing |
| pair blocks (1-4 nodes) | ≈55k blocks, Cholesky 5 MB | ms | mutual-strongest coupling ‖D_i^{-1/2}K_ijD_j^{-1/2}‖; **ties merged, never broken by index** (keeps equivariance) |
| patch modes | ≈3.0k patches × k≤8 × ≤375 DOF (36 MB) | batched LOBPCG ≈ 0.05-0.2 s | §2.7 |
| λ/μ/γ Galerkin parts of level-1 macro elements | 2,080 × 3 × 3321 (transient; only Ã1 is stored, 28 MB) | ≈ 40 GFLOP ≈ 5 ms | nested P1 is exact, so A^λ_E = Σ_children P_cᵀK^λ_cP_c; K = λK^λ + μK^μ splits the template linearly |
| level-1 GP across macro faces | 48 templates (3 axes × 2⁴ patterns of the 4 fine faces) | 0 storage | P1ᵀ(Σγ Fᵀ F)P1 depends only on the pattern |

**New count (this doc, `_ol/hier_counts.json`): nested Q2 hierarchy with component splitting (26-connectivity of
active elements within each node's support).**

| level | space | node grid | FULL τ=0.5 (2/3) | cut 0.758 | cut 0.148 |
|---|---|---|---|---|---|
| 0 | Q2(32³) | 65³ | 121.0k (146.4k) nodes | 91.5k | 23.0k |
| 1 | Q2(16³) | 33³ | 2,080 macro el., 21.2k nodes (23.6k) | 16.6k | 4.5k |
| 2 | Q2(8³) | 17³ | 352 macro el., 4.0k nodes | 3.3k | 1.0k |
| 3 | Q2(4³) | 9³ | 64 macro el., 766 nodes (2.3k DOF) | 711 | 269 |

Component splitting barely matters at level 1 (0%) and is small at levels 2-3 (+3-6%). Ghost-penalty faces tie
the fringe to the body, which is consistent with route-1 E2.

### 2.5 Geometry path, part 2: the hypernetwork H_θ (how coefficients are generated)

**Inputs: O_h-invariant scalars only.**
- Per element, 20 features:
  - log and linear volume fraction;
  - three symmetric invariants of the normalised central second-moment tensor (trace, trace of the square,
    determinant);
  - |centroid offset|/h;
  - full, cut and Γ-band flags;
  - fraction of weak nodes;
  - invariants of K_e: the sum and sum of squares of the axial and of the shear uniform-strain Rayleigh energies
    (linear in the moments), and ‖K_e‖_F normalised;
  - number of ghost-penalty faces;
  - signed plane distance/h, τ at the element centre, and (|φ|−τ)/h.
- These are scattered onto the 33³ vertex grid by symmetric pooling over the 8 incident elements (mean, max,
  sum).
- Vertex-node features are added:
  - log tr D_i;
  - two invariant eigenvalue ratios of D_i;
  - the weak indicator log₁₀(‖D_i‖/median);
  - lumped mass;
  - incidence count;
  - box and Γ-band flags;
  - component count.

**Backbone: IsoU-Net.**
- It runs on dense masked grids: 33³ (32 channels), 17³ (64), 9³ (128) and 5³ (128), with two convolutions per
  stage on each side.
- Each 3×3×3 convolution is **O_h-isotropic**: its weights are shared over the 4 offset orbits (centre, 6 faces,
  12 edges, 8 corners), so there are 4·Cin·Cout parameters per convolution. It is exactly equivariant on scalar
  fields; there is no regular-representation lifting and no 48× cost.
- GELU and GroupNorm are used. Normalisation is allowed here because this is the geometry path.
- Stride-2 average pooling and trilinear upsampling are symmetric.
- A global token (masked mean and max at 5³ → MLP, 128 wide) is broadcast back.

The dense grid sees across material gaps. That is harmless and useful as context: the network only emits
coefficients for couplings that K's hypergraph already contains (RESEARCH F6).

**Heads.** Each head is a small MLP, 2×64, with bounded smooth output maps (R15).

| head | entity | inputs | output map | count (FULL) |
|---|---|---|---|---|
| H_ω | pair block b | trilinear sample of the 33³ embedding at the block centroid; block invariants (log λ_min and λ_max of D_b⁻¹A_bb over its nodes) | ω_b = σ(·)∈(0,1), relative to the certified cap | ≈55k |
| H_g | patch mode (p,k) | patch-pooled embedding; log λ_pk; rank k/k_max | g_pk = 1.5σ(·)∈(0,1.5) | ≈24k |
| H_θ1 | level-1 macro element | pooled 33³ embedding over the element | (θ^λ, θ^μ) = 4^{tanh(·)} ∈ [¼,4] | 2×2,080 |
| H_θ2 | level-2 macro element | 17³ embedding | same | 2×352 |
| H_v | coarse node (levels 1, 2) | grid embedding | Chebyshev-2 damping σ(·) | ≈25k |
| H_step | global token | – | α_j = σ(·)∈(0,1), β_j = 0.9σ(·), j = 1..16; 3 Neumann α's | 35 |

**Parameters.** About 16k (33³) + 66k (17³) + 262k (9³) + 131k (5³) + 60k (resampling and 1×1) + 2k (input MLP) +
≈150k (heads and global) ≈ **0.7M**, shared across all geometries.

**Cost per geometry.**
- IsoU-Net ≈ 15 GFLOP, about 1-2 ms.
- Heads ≈ 0.2 GFLOP.
- Output: ≈86k generated scalars (0.35 MB) per FULL cell.

**Per-geometry vs shared weights (question j).**
- Weights are shared; generated coefficients are per geometry.
- Step 1 also runs a "free-coefficient oracle": the same 86k scalars as free parameters per geometry. It separates
  "the structure is insufficient" from "the hypernetwork is insufficient".

### 2.6 The q-path components

**Sub-step S (fine smoother; K's hypergraph; exact local kernels).**

    S r = Σ_b Z_b ω_b c_S A_bb⁻¹ Z_bᵀ r  +  Σ_p Σ_k s(λ_pk) g_pk (1/λ_pk − 1) v_pk v_pkᵀ r

- The v_pk are M0-orthonormal generalized eigenvectors of (A_pp, M0_pp), with M0 = blockdiag(A_bb) (the
  undamped pair operator). They are geometry-only and computed once. There is no backpropagation through
  eigensolvers.
- s(λ) is a smoothstep from 1 at λ ≤ 0.1 to 0 at λ ≥ 0.2. This keeps û continuous in τ when an eigenvalue crosses
  the threshold (§3g).
- **Exactness:** if r = A v_pk for one mode, then with ω = g = 1 the operator returns S r = v_pk exactly: an
  undamped local solve (§4.6). This fixes route 1's issue that "the global step also damps the exact blocks".
- The operator is symmetric, and PSD for gates in range.
- Cost: pairs ≈ 55k × (≤12)² × 2 ≈ 10 MFLOP; modes 3k × 8 × 375 × 4 ≈ 36 MFLOP.

**Sub-step C (global coarse correction; the multiscale "feedforward NO" half).**

    C r = P1 · V₂(Ã1; Ã2; Ã3) · P1ᵀ r

- P1 is the nested Q2(16³) → Q2(32³) prolongation. It is tensor-product quadratic refinement (1-D weights
  [3/8, 3/4, −1/8] and its mirror), masked by component and zero on Dirichlet ports. The Dirichlet coarse space
  drops the macro nodes on the box plane: by the Lagrange property, the remaining macro functions vanish on the box
  faces.
- V₂ is two symmetric V-cycles:
  - level 1: Chebyshev-2 of 3×3 node-block Jacobi with learned dampings;
  - level 2: the same;
  - level 3: exact dense Cholesky solve (fp64, 2.3k DOF, 21 MB).
- Transfers between coarse levels are nested Q2 (P2, P3).
- Coarse operators are Ã1, Ã2, and Ã3 = P3ᵀÃ2P3.
- Cost per visit:
  - Ã1 apply 2,080 × 81² × 2 = 27 MFLOP;
  - ≈ 10 applies per V₂;
  - levels 2-3 < 10 MFLOP;
  - P1 and P1ᵀ ≈ 5 MFLOP;
  - total ≈ 0.3 GFLOP, well below one fine K.

**Learned coarse moduli (the B2 mechanism, made safe).** Ã1 = Σ_E (θ^λ_E A^λ_E + θ^μ_E A^μ_E) + A^γ.
- The A^λ_E and A^μ_E are the exact Galerkin λ- and μ-parts over the 8 children of macro element E.
- θ = 1 reproduces Galerkin.
- In a one-shot multilevel method, the best coarse operator is the one whose solution best matches the exact field
  at the coarse nodes. For thin walls that is softer than Galerkin: a homogenized, locking-relieved operator.
  θ^λ < 1 relieves volumetric stiffening and θ^μ < 1 relieves shear stiffening.
- B2 showed that learned coarse materials capture a cell with 23k numbers. We keep Galerkin as the initialisation
  and a certified trust region (§4.7) as the safety net.

**Recurrence (the route-1 mechanism).**
- d_j = α_j c_{T_j} z_j + β_j d_{j−1}.
- c_S and c_C are the certified caps. α_j ∈ (0,1) and β_j ∈ [0, 0.9) come from H_step, geometry-conditioned through
  the global token.
- Steps beyond the trained n_K (inference-time dial) use β = 0 and α = 1/1.9 ("certified tail").

**Exact residual evaluations.**
- One fine K per sub-step, y_j = K Z d_j, applied to all nodes. This maintains r_j = f − (K u_j)_I exactly up to
  fp32, and also accumulates K û for the readout.
- The readout recomputes K û once, cleanly, in the fp64-accumulated strain form.

### 2.7 Why patches, not clusters (new evidence) and how patches are built

`_ol/hier_counts.json` computes connected components of weak nodes (body-only proxy: ‖D_i‖ < 1% of median) in the
element graph:
- FULL τ=0.5: 20.2k weak nodes (16.7%) in **2 components of about 10.1k nodes each**. These are the fringe sheets on
  the two sides of the shell.
- Cut 0.758: 2 components.
- Cut 0.148: 5 components.

So the "11-30-node slow modes" are localized **inside** giant sheets. Weak-node components cannot serve as local
clusters: this breaks any "cluster = component" design, and it is probably part of why the first route-1 weak-group
attempt behaved oddly. Localization is by weakness gradient, not by topology.

**Patch definition.**
- Two staggered partitions of the 32³ elements into 2×2×2 macro elements, one at offset 0 and one at offset 1.
  Both are symmetric under O_h.
- A patch is all active nodes of one macro element: ≤125 nodes, ≤375 DOF, minus Dirichlet nodes.

`_ol/patch_counts.json`:
- Median weak nodes per patch: 15-17, maximum 33-54.
- The weak nodes plus their strong ring cover 45-117 of the 125 nodes, so a patch is effectively the whole macro
  element.
- Patches containing weak nodes: FULL 1,552 + 1,476; cut 0.758: 1,226 + 1,174; cut 0.148: 328 + 343.

**Consequences.**
- Any slow mode of 11-30 nodes (≈2-3 elements across on a sheet) lies inside a patch of at least one stagger, or
  straddles the one-element overlap. That case is covered by additive combination and learned gates.
- The mode count scales with the number of patches, about 24k local modes for FULL, and needs **no global budget**.
  Route-1 E1 ("ideal enrichment" with the 128 slowest global modes) was limited because the band continues past 128.
- Building the patches needs no weak threshold, which is discontinuous. The generalized eigenproblem selects the
  slow directions itself.
- Encoding: patch matrices are assembled from the 8 child K_e plus internal ghost-penalty faces (transient
  ≈140k floats per patch, chunked). Batched LOBPCG uses block 8 and 25 iterations: 3k × 375² × 16 × 2 × 25 ≈
  3e11 FLOP ≈ 20-60 ms, plus Rayleigh-Ritz with batched syevj on 24×24.

### 2.8 Readout (question e), sensitivity (g), precision (k), error indicator

- **Energy.** ûᵀKû = Σ_e Σ_q w_eq ε_qᵀCε_q + γ Σ_f ‖F j_f‖².
  - Strains and jumps are formed in fp32, from û minus its rigid part (a symbolic zero).
  - Products and sums are accumulated in fp64.
  - The rounding is relative to the strain, not the displacement (DATA_INTERFACE §4).
- **Reaction.** Ŝ q = Êᵀ(K û).
  - Ê is linear, so its transpose is the reversed sweep: the Jacobian is constant and no forward activations need
    storing.
  - The operator order is reversed: Kᵀ = K, Sᵀ = S, Cᵀ = C (the symmetric V-cycle), the recurrence reversed, and
    the lift transposed.
  - Cost ≈ forward + 1 K ≈ 2.1× forward. It is symmetric PSD by construction (§4).
  - Identity: Ŝq = (Kû)_D − Ê_Iᵀ r_n. The non-symmetric consistent reaction (Kû)_D differs from it by the
    transposed extension of the final residual. **We never use (Kû)_D in the lattice** (RESEARCH 3.3).
  - With f_C ≠ 0 each cell also gives a load vector b = Ê_Bᵀ(K û(0,f_C) − J_Cᵀf_C), one extra forward and transpose
    per load case.
- **Sensitivity.** dC/dτ_c = −Σ_cells Σ_e Σ_q (∂w_eq/∂τ_c) ε_qᵀCε_q.
  - The ∂w/∂τ come from forward-mode through polyref (≈ one moment pass), or are stored (51 MB per FULL cell) while
    the sensitivity is being evaluated.
  - The ghost term has zero derivative within a topology.
  - This is the consistent estimator of the exact sensitivity. We do not differentiate the network with respect to
    τ (§3g).
- **Precision.**
  - q-path in fp32, including K with TF32 off (the residual needs 1e-7, not 1e-3).
  - Level-3 Cholesky in fp64: 2.3k³/3 ≈ 4 GFLOP ≈ 3 ms at 1/64 rate, and cheap per solve.
  - Lift fit, rigid projection, energies and lattice vectors in fp64.
  - Hypernet in fp32. Patch eigen in fp32 after an fp64 assembly of A_pp.
- **Free error indicator.**
  - η² = r_nᵀ C_η r_n, with C_η one Neumann-style symmetric pass ≈ A⁻¹. This estimates ‖û − u*‖²_K
    (e = A⁻¹r_n). It costs 1-2 ms per column.
  - It flags out-of-distribution cells in step 2 and step 3. It is not a guaranteed bound; route 2's equilibrated
    certificate plugs into the same û later.

### 2.9 Neumann action for BDD (question f)

**Structure.** B_N = Π_R⊥ J_B 𝒱 J_Bᵀ Π_R⊥.
- 𝒱 is the **palindromic** sequence S, C^N, S on the floating cell: no Dirichlet nodes, full level-1 space including
  the box-plane macro nodes (the same stored Ã1, no principal restriction).
- Level 3 is the floating factor of Ã3^N + ρ R3R3ᵀ, fp32, 21 MB.
- Recurrence: β = 0, α values symmetric.
- There are 2 fine K evaluations between the three steps.
- Π_R⊥ is the rigid projection in fp64.

**SPD.**
- I − 𝒱K = (I − SK)(I − C^N K)(I − SK) with symmetric S and C^N.
- Each factor is certified non-expansive, and S is strictly contracting off the rigid kernel.
- So 𝒱K has eigenvalues in (0, 2) on the complement of R, which makes 𝒱 SPD on the balanced subspace. This is the
  Melchers-Dolean-Abdelmalik condition: same basis on both sides.

**Cost and accuracy.**
- ≈ 3 K + 1 coarse ≈ 3.4 GFLOP ≈ 0.1-0.2 ms per column.
- Target spectral-equivalence ratio κ(B_N S) ≲ 5 against the exact S⁺. That should cost BDD at most about 2× its
  iteration count, and is measured in step 1 (§6.7).
- The operator shares every stored object except the floating level-3 factor.

### 2.10 What is linear in q and what is geometry-only

| object | depends on q? | depends on g? |
|---|---|---|
| lift fit α, u0, r0 | linear | W, A_aff, K (geometry) |
| S, C, P1-3, Ã1-3, V-cycles, pair and patch kernels | applied linearly | coefficients geometry-only |
| K apply at every sub-step | linear | exact from moments |
| α_j, β_j, caps c_S and c_C | no | geometry-only (global token and Lanczos) |
| û, Ŝq | linear | – |
| energy, sensitivity | quadratic (readout only) | exact K, ∂K/∂τ |

**Hard rules.**
- No activation, normalisation, attention or batch statistic touches a q-path tensor.
- No step size computed from q-path inner products (this rules out CG).
- No hypernetwork input derived from q.
- Unit test: ‖Ê(a q1 + b q2) − aÊq1 − bÊq2‖ at fp32 rounding, on random q.

### 2.11 Sizes, parameters, FLOPs, memory

**Per-cell resident state (fp32 unless noted).**

| object | FULL τ=0.5 | FULL τ=2/3 | cut 0.758 | cut 0.148 |
|---|---|---|---|---|
| topology, masks, element map, faces | 3 MB | 3.5 MB | 2.3 MB | 0.6 MB |
| Gauss weights (E×125) | 6.4 MB | 8.0 MB | 4.8 MB | 1.2 MB |
| pair blocks (Cholesky) | 5 MB | 6 MB | 4 MB | 1 MB |
| patch modes (k≤8, ≤375 DOF) | 36 MB | ≈40 MB | 29 MB | 8 MB |
| Ã1 (81×81 upper per level-1 macro element) | 28 MB | 31 MB | 21 MB | 5 MB |
| Ã2 | 4.7 MB | 4.7 MB | 3.8 MB | 1.0 MB |
| level-3 Cholesky, Dirichlet fp64 + Neumann fp32 | 21 + 11 MB | same | ≈ 19 + 10 MB | ≈ 3 + 2 MB |
| generated scalars, caps | 0.4 MB | 0.5 MB | 0.3 MB | 0.1 MB |
| **total** | **≈ 115 MB** | **≈ 125 MB** | **≈ 95 MB** | **≈ 22 MB** |

- 100 all-FULL cells come to ≈ 12 GB, which leaves ≈ 18 GB on the 32 GB card for batched columns.
- Each live fine vector per column is N0 × 3 × 4 B ≈ 1.45 MB (FULL).
- The forward pass keeps ≈ 6 live vectors (u, d_{j−1}, r, z, y, lift), ≈ 9 MB per column.

**FLOPs per column (FULL, n_K = 8).**

| operation | count | FLOP |
|---|---|---|
| fine K (lift residual + 8 steps + clean readout) | 10 | ≈ 10 GFLOP |
| S steps | 4 | ≈ 0.2 GFLOP |
| C steps | 4 | ≈ 1.2 GFLOP |
| **forward + energy** | | **≈ 11.4 GFLOP** |
| reaction Ŝq (forward + transpose) | | ≈ 23 GFLOP |
| Neumann action | | ≈ 3.4 GFLOP |

**Time (estimate, to be benchmarked in step 0).**
- The strain-form K apply is batched GEMM-like work (ghost part) plus sum-factorisation (body).
- At an effective 25-40 TFLOPS in fp32 that is ≈ 25-40 µs per K per column, so forward ≈ 0.3-0.5 ms,
  reaction ≈ 0.6-1.0 ms and Neumann ≈ 0.1-0.15 ms.
- For comparison, the route-1 skeleton measured ≈ 5 ms per layer (174 ms for 32 layers, per column). That gap is
  implementation (trace coordinates, per-column execution) and must be verified by a microbenchmark before anyone
  relies on it.

**Per design iteration (100 cells, BDD).**
- Per cell: (70-160) Ŝ + (35-55) B_N ≈ 0.05-0.17 s (FULL). For 100 cells that is ≈ 5-17 s.
- Encoding, excluding moments, is ≈ 0.1-0.3 s per cell.
- Moments are 0.25-1 s per cell. Every K-based route needs them, so they are the dominant encoder cost.

**Training memory (B = 32 columns, n_K = 8, FULL).**
- Autograd stores ≈ 3 fine vectors per sub-step: 8 × 3 × 1.45 MB × 32 ≈ 1.1 GB.
- Hypernet activations ≈ 0.2 GB.
- The coarse operators and the level-3 Cholesky backward are small.
- Gradient checkpointing per sub-step brings the total below 2 GB.

### 2.12 Operator-learning analysis

**Kernel view.**
- Each q-path operator is a kernel integral u(x) ↦ Σ_y κ_g(x,y) u(y), whose kernel is geometry-only:
  - K: element and ghost-face hyperedge integrals, exact;
  - S: block-diagonal plus patch low-rank kernels;
  - C: a multiscale kernel P1 (Ã1⁻¹)_approx P1ᵀ;
  - Ã3⁻¹: a dense, full-rank global kernel on 2.3k DOF.
- This is a multipole/H-matrix-type factorisation (RESEARCH 1.10): near field exact (K, patches), far field
  through nested Galerkin levels.
- F1 justifies it: E = −A⁻¹K_IP is an interior Green's operator with multiscale compressibility.

**Receptive field.**
- **Global in every C step.** The exact dense level-3 solve couples every residual to every node, so information
  crosses the cell in one sub-step, not one hop per layer as in message-passing GNNs.
- Local operators (K: 5 node layers; ghost penalty: 5 layers along the axis; patches: 2 elements) provide the
  near field.

**Attention-like global mixing with geometry-only weights.** Two places:
- Ã3⁻¹ is an exact, geometry-only, full-rank global kernel. Galerkin optimality makes it the best global mixing
  within its space.
- The learned θ fields reweight it (a geometry-conditioned "attention temperature" per macro element).

Transolver/Galerkin-attention slices were considered and **rejected for v0**:
- A rank-M softmax channel on the same coarse information is dominated in the energy norm by the exact Galerkin
  solve.
- It cannot carry localized sheet modes (F5).
- Soft slices blur across material gaps unless masked.

**Spectral layers: rejected at every q-path level.**
- A Fourier layer is the Green's operator of a homogeneous reference medium (the Moulinec-Suquet mechanism,
  RESEARCH 7.2). With voids (infinite contrast) and weak/strong ratios above 1e2 it needs fixed-point depth that
  grows with contrast.
- Masks do not fix a translation-invariant kernel (DAFNO).
- The exact coarse Galerkin operator gives a geometry-exact global kernel for free.
- The hypernetwork does not need an FFT either: its U-Net reaches 5³ plus a global token.

**Expressivity.**
- (i) For any geometry and any admissible hypernetwork output, n_K → ∞ with the certified tail converges to E_g
  exactly (§4.7). The class is dense in the solution operators, with no representational ceiling.
- (ii) At finite n_K the error is ‖Π_j (I − α_j c_j T_j A) (+ recurrence) e0‖_A. The learned coefficients select
  the polynomial and the local and coarse spectra.
- (iii) With patches off, θ = 1, uniform ω and route-1 α/β, the class is structurally close to the route-1
  skeleton at an equal number of K evaluations. It differs in the coarse space (nested Q2 in place of block
  polynomials) and in the port coordinates. Route 1 passed the lattice gate with 24 learned layers, so a
  gate-passing member at n_K ≈ 24 is expected but **not proven**. Step 1a checks this directly.

**Linear in q, strongly geometry-conditioned.**
- Every nonlinearity sits in the map g ↦ (w, D, pairs, V_p, λ_p, Ã, α, β, caps).
- The q-path is a composition of linear maps whose matrices are those outputs. Linearity holds by construction
  (§4.1).

**Resolution behaviour.**
- The grid is fixed at n = 32 (decision 3), so discretisation-convergence is not a v0 goal.
- Every component is FE-consistent: exact K, nested Q2 transfers, Galerkin products. Multigrid contraction factors
  are h-robust, and the hypernetwork's features are dimensionless (h-scaled).
- At n = 64 the same code adds a level and reuses H_θ unchanged. Its receptive field is in macro-element units.

**Generalization across geometries (step 2).**
- (a) Every head is a function of a bounded neighbourhood plus one global token. Energy-minimizing multiscale bases
  decay exponentially (LOD/gamblets), so optimal local coefficients are local functionals of geometry.
- (b) The geometry-specific structure (K, pairs, modes, Galerkin products) is **computed, not learned**, so it
  transfers perfectly.
- (c) Out of distribution, the certified steps turn a bad coefficient into slower convergence, not a wrong answer.
  The indicator η detects it, and extra steps repair it.

**Cubic group O_h.** Joint (g, q) equivariance holds by construction (§4.8). Augmentation is kept only as a monitor
and as a safety net for knobs that break symmetry.

---

## 3. Answers to the open questions a-l

**a. Consuming the fixed-grid masked data.**
- The q-path never uses learned message passing at the fine level. The fine coupling is K itself, as element (27
  nodes) and ghost-face (45 nodes) hyperedge integrals in strain form. That is exactly the hypergraph DATA_INTERFACE
  recommends (§2.2).
- Walls that are adjacent in space but disconnected in material interact only if K couples them, which is exactly
  CutFEM's definition.
- Coarse levels are nested Q2 on dense 33³/17³/9³ grids with component splitting.
- The hypernetwork runs on dense masked grids. It may see across gaps, harmlessly, because it only emits
  coefficients for K-graph couplings.
- Element moments serve three roles:
  1. exact K (Gauss weights);
  2. invariant features;
  3. Galerkin λ/μ/γ coarse parts.
- The ghost penalty enters exactly through K_ghost and through level-1 face templates, and as feature counts.
- Weak nodes are handled by patch modes on staggered macro patches (§2.7), and by pairing weak nodes with strong
  ones, as route 1 found.
- Element stiffness is never an edge feature of a learned fine layer, because there is none. Its invariants feed
  the hypernetwork.

**b. Hard ports and rigid modes, staying linear.**
- The correction space Z excludes Dirichlet nodes, so every update leaves the ports untouched.
- The lift puts q on the ports exactly, and puts the rigid/affine fit in the interior.
- Rigid modes are split in fp64 and never pass through K. All coarse spaces contain the rigid modes, because nested
  Q2 reproduces linear fields.

**c. Global transmission and localized soft modes.**
- Global: every C step is global (exact level-3 dense solve) and bending-capable (Q2 coarse spaces, not Q1).
  Material-awareness comes from component splitting and learned moduli.
- Local: localized soft modes are handled by exact local modes on patches, inside S.
- No spectral layers. A graph-coarsened hierarchy is unnecessary, because splitting is almost never triggered.

**d. Where K appears.**
- At n_K + 2 fine evaluations per forward (lift residual, one after each sub-step, one clean readout). Default n_K =
  8, dial 4-16.
- Justification in §2.1: this turns a 1e-4 coefficient-accuracy requirement into an O(1) one, at no cost premium.
- Coarse Galerkin products use K only in the encoder.
- The pure K-free variant is a step-1 ablation.

**e. Reaction readout.**
- Ŝq = Êᵀ(K û), using the explicit transposed sweep (≈ 2.1× forward, no stored activations). Symmetric PSD by
  construction.
- (Kû)_D is rejected (non-symmetric). Its relation to Ŝq is in §2.8.

**f. Approximate Neumann action.** The palindromic S-C^N-S sequence on the floating cell, with rigid projection
(§2.9). It is SPD by construction, shares all stored objects except one small factor, and costs ≈ 0.15 ms per
column.

**g. Sensitivity.**
- −ûᵀ(∂K/∂τ)û is computed with the exact ∂w/∂τ; the ghost-penalty term is zero within a topology.
- Continuity in τ within a topology:
  - the hypernetwork is smooth;
  - pairs, patches and the hierarchy are frozen per design band (superset idea, DATA_INTERFACE §1.6);
  - patch modes vary continuously, with the soft threshold s(λ) removing jumps;
  - Galerkin products and Cholesky are smooth.
- Across a topology change the encoder rebuilds masks. That step is discrete, exactly as in the exact pipeline.
- Error: −2u*ᵀK′e − eᵀK′e.
  - If K′_e were proportional to K_e element by element, the first term would vanish, because u*ᵀKe = 0 for
    interior-supported e (Galerkin orthogonality).
  - The observed sensitivity/compliance error ratio of 1.3-1.5 in route 1 is consistent with this.
  - The residual first-order part comes from the non-proportional (surface) part of K′. It is largest on fringe
    elements, where K′/K is large. That is risk 6, with a dedicated loss term in §6.2.

**h. Equivariance.**
- By construction (§4.8): invariant-scalar hypernetwork with isotropic kernels, symmetric partitions and grids,
  tie-merged pairing, and eigen-subspaces rather than individual eigenvectors.
- Augmentation is used as a monitor. The cut plane breaks per-cell symmetry, but the joint (g, q) map is
  equivariant, and that is what the construction delivers.
- Training rotations use signed-permutation moment transforms, not re-integration, which avoids the 1e-4 Kuhn-split
  asymmetry.

**i. Conditioning and normalisation.**
- Per-sample normalisation by ‖u*‖²_K = qᵀSq (the rigid-projected q), so each loss term is exactly μ − 1 in that
  direction.
- log(1 + ℓ) during warm-up.
- A CVaR (top 25%) term, because averages hide the worst direction.
- The parameterisation itself preconditions the problem: coefficients act as dampings and moduli inside a multigrid
  structure, so the parameter gradient reaches soft components directly. This addresses F4.
- There is no whitening of q-path tensors (that would break linearity).
- All head outputs are bounded smooth maps.

**j. Capacity.**
- One shared 0.7M-parameter hypernetwork, generating ≈ 86k invariant scalars per FULL cell.
- There are no per-node 3×3 free blocks: 4.4 MB per block-layer is over budget, and not equivariant.
- Step 1 adds a free-coefficient oracle.
- The capacity ladder is in §7.

**k. Precision.** fp32 network with TF32 off in K. fp64 for the level-3 Dirichlet factor, lift fit, rigid
projection, energy and reaction accumulation, and lattice vectors (§2.8).

**l. Degradation.** One codebase with three dials, n_K, head freezing, and the "fallback-A head" (§7.2).

---

## 4. Guarantees (short proofs)

Notation: D = Dirichlet nodes, I = free nodes (including force nodes), A = K_II, e = û − u*.

**4.1 Exact linearity in q.**
- Every q-path map is one of the following:
  - a fixed matrix (K, S, P, the V-cycle with fixed coefficients, the lift fit);
  - a sum of such maps;
  - a composition of such maps.
- All coefficients are functions of g only (§2.10 rules). Compositions and sums of linear maps are linear.
- Floating point gives linearity to rounding (the unit test in §2.10).

**4.2 Exact port values.**
- u0|_D = q_D by construction of the lift.
- Every update is u_j = u_{j−1} + Z d_j, and Z has zero rows on D. By induction û|_D = q_D bit-exactly, because
  nothing is ever added to port entries.
- This holds in both cut modes (D = box, or D = box ∪ Γ).

**4.3 Exact rigid modes.**
- For q_B = R_B β, the fit gives α = (β, 0) exactly in exact arithmetic, because the weighted least-squares fit
  reproduces vectors in its range.
- Then q_rem = 0 and r0 = f − K(A_strain·0 + 0) = 0. A linear map of 0 is 0, so û = Rβ.
- In fp64 the error is |q_rem| ≈ 1e-16|q|.
- Readout: K is applied only to û − Rβ (symbolic B R = 0), so Ŝ R_B = O(1e-16).
- The BDD coarse space (partition-of-unity-weighted rigid modes) needs exactly this.

**4.4 Symmetric PSD readout.**
- Ŝ = Êᵀ K Ê with Ê a fixed matrix for fixed g. Symmetric; xᵀŜx = ‖Êx‖²_K ≥ 0.
- With the rigid split: Ŝ = P⊥ᵀ Ê_netᵀ K Ê_net P⊥ (congruence, fp64 P⊥), still symmetric PSD.

**4.5 One-sided (upper-bound) error.**
- (K u*)_I = f_I and e|_D = 0 give e_Iᵀ(K u*)_I = e_Iᵀ f_I. Expanding: Π(û) − Π(u*) = ½ eᵀKe ≥ 0.
- For f = 0 (the Dirichlet-only extension): qᵀŜq − qᵀSq = ‖e‖²_K ≥ 0, so μ(q) ≥ 1 for every q.
- Lattice: the model's global space {û_c = Ê_c q_c + Ĝ_c f_C, glued q} is an affine subset of the exact global FE
  space. So Π̂* ≥ Π*, and the compliance C = −2Π* satisfies **Ĉ ≤ C**: the model is always too stiff, for force or
  cut loading.
- Holds for any network weights; needs only 4.2 and the exact K in the readout.

**4.6 Exact local correction of slow patch modes.**
- Take v_pk with A_pp v = λ M0_pp v and vᵀM0v = 1, and suppose r = A v is supported in the patch (no leakage).
- Let s = 1, g = 1, ω_b = 1 (with c_S = 1 for this argument; in general the gate absorbs the cap).
- The pair part gives M0⁻¹Av = λv. The mode part gives (1/λ − 1)·v·(vᵀAv) = (1/λ − 1)λv = (1 − λ)v.
- The sum is v: the local error component is removed exactly, whatever λ is (λ → 0 is the soft case).
- Leakage across the patch boundary is handled by alternation with C and exact residuals (§5).

**4.7 Certified non-expansiveness, and convergence as n_K → ∞.**
- Let T ∈ {S, C} be symmetric positive semidefinite (C is SPD on range(P1)). The encoder estimates
  λ̂ = λ_max(TA) by 20-step Lanczos with a safety factor of 1.05, and sets c_T = 1.9/λ̂.
- For α ∈ (0,1], the step I − α c_T T A has A-norm spectrum in [1 − 1.9·1.05·α, 1] ⊂ [−1, 1], so it is
  non-expansive in ‖·‖_A.
- S is SPD on all of I (block inverses plus PSD mode terms), so I − αc_S S A is strictly contracting.
- Hence the certified tail (β = 0) is monotone: error(n) ≤ error(n_K) for n > n_K, and error → 0 as n → ∞.
- The trained head (β > 0) is not certified step by step; its overall polynomial is monitored (§6.5).
- "Certified" means up to the Lanczos estimate (a lower bound on λ_max, corrected by the safety factor). It is not a
  proof. A 2-3 extra-iteration check is cheap and advisable.
- The θ moduli are unrestricted in sign structure (all θ > 0 keep Ã SPD). Safety comes from c_C, not from
  restricting θ, which leaves the network free to learn aggressive homogenization.

**4.8 Equivariance.**
- For h ∈ O_h, write ρ(h) for the signed 3×3 permutation acting on node vectors and index maps. The data satisfy:
  - K_{h·g} = ρ K_g ρᵀ (pipeline to 1e-7; exact for signed-permutation-transformed moments);
  - the masks permute;
  - W is invariant.
- Every constructed object is equivariant:
  - pairing is canonical with ties merged;
  - both stagger partitions, the Q2 grids and the transfers are O_h-symmetric;
  - patch modes enter only through the projector Σ_k s(λ_k)(·)v_kv_kᵀ, which is basis-independent within
    eigenspaces;
  - the hypernetwork maps invariant fields to invariant fields with O_h-isotropic kernels, and pair and entity heads
    take invariant inputs;
  - α and β are functions of invariant global pooling;
  - the Lanczos caps are invariant scalars, up to Krylov randomness (use a start vector transformed with g, or the
    canonical orientation).
- Each q-path operator therefore satisfies T_{h·g} = ρ T_g ρᵀ, the lift commutes with ρ, and so Ê_{h·g}(ρq) =
  ρ Ê_g(q) and Ŝ_{h·g} = ρ Ŝ_g ρᵀ.
- For cut cells this is the joint (g, q) equivariance; per-cell invariance is not claimed.
- Residual caveats:
  - exact eigenvalue ties exactly at a mode-count cap (k_max = 8) could break the projector argument; this is rare
    and at the tie level;
  - fp non-associativity at 1e-7.

---

## 5. Why the soft, force-driven directions come out right

**What the soft directions physically are.** Evidence plus the new counts:
- Slow modes live 94-99.4% on weak nodes, with 11-30 nodes per mode.
- Weak nodes form **continuous fringe sheets on both sides of the shell** (FULL cells too: 16.7% weak, 2 giant
  components).
- They are fictitious-domain and cut-sliver nodes, held by tiny body stiffness and γ = 1e-4 ghost-penalty terms.
- Global soft deformation also exists: bending of 1-2-element-thick walls.
- The lattice gate's loads are **unit nodal forces on every box-port node of a plane** (`lattice_v2.py` l.95-99,
  `lattice_pcg.py` l.61), including fringe port nodes with tiny material support. A traction-consistent load
  f_i = ∫_{Γ∩mat} N_i t dA would give those nodes almost nothing. The gate's load model therefore pushes energy
  into the fringe sheet, which plausibly explains part of the 94-96% soft-decile share. This is an **observation for
  the owner**, not a change of the gate (risk 4). The architecture must, and does, handle it either way.

**Mechanism 1: exact, undamped local modes on spatial patches (§2.7, §4.6).**
- Every mode that is slow for the pair smoother and supported in one macro patch is removed exactly by one S step.
- Two staggers cover modes of up to about 3 elements.
- The number of modes grows with the number of patches, so the dense band is covered without a global budget
  (unlike E1's 128).
- This fixes route 1's two failures:
  - the global step damped the exact blocks;
  - weak groups were built from a threshold or components, and those percolate.

**Mechanism 2: boundary coupling is controlled by the body.**
- After an S step, the patch error equals the discrete harmonic extension of the error on the patch boundary.
- Its energy is at most that of any extension of the same boundary error (energy minimisation), so the soft sheet
  error is bounded by the error of the stiff body nearby.
- C and the exact residuals reduce the body error. Alternating S and C is the multiplicative two-level Schwarz of
  GenEO, whose contraction is contrast-independent once the local slow spaces are included (RESEARCH 5.7 and 7.3,
  Spillane et al.).

**Mechanism 3: soft global bending is carried by Q2 coarse spaces, not Q1.**
- Quadratic reproduction removes the locking that affine or Q1 coarse spaces suffer on thin walls (route 1:
  quadratic ≫ affine).
- The learned θ moduli add homogenization-type softening where Galerkin is still too stiff for a one-shot
  correction (the B2 evidence).
- The exact dense level-3 solve transmits face-to-face in one step.

**Mechanism 4: Dirichlet data on fringe port nodes.**
- For a soft q that moves a weak box-port node, the adjacent interior fringe nodes must follow nearly rigidly.
- That is a local Dirichlet problem inside the port-adjacent patch. Its patch modes are computed with the port
  nodes excluded, so the continuation is exact to the patch-boundary argument above.
- For a force-type cut band, f_C hits Γ-band weak nodes (22-28% of band nodes are weak, and in heavy cuts 35% of all
  weak nodes are in the band). They are free nodes inside patches: same mechanism.

**Mechanism 5: the training signal targets exactly these directions (§6).**
- Force-driven samples (30%) include (i) unit nodal face forces like the gate and (ii) traction-consistent forces.
- The adversarial pool comes from LOBPCG on (Ŝ − S, S), seeded with force-driven q.
- A CVaR term, and per-sample normalisation, so each soft direction counts in full.
- A sensitivity term.

**Mechanism 6: one-sided safety and diagnosis.**
- Any residual soft error makes the cell too stiff, so compliance is underestimated and never erratic.
- η flags it, and the certified tail reduces it if more accuracy is needed at inference.

**Expected numbers (a hypothesis to test, not a claim).**
- Suppose the S-C pair gives an energy contraction of ρ ≈ 0.3-0.4 on force-driven residuals once patch modes remove
  the slow band. Then the preconditioned λ_min moves from ≈ 0.005 (route 1) to the patch threshold scale 0.1-0.2,
  a factor of 20-40.
- With the learned recurrence, 4 pairs (n_K = 8) then give ‖e‖/‖u*‖ ≲ 0.1, i.e. μ − 1 ≲ 1e-2 in loaded directions.
- Step 1a measures exactly this, with no training.

---

## 6. Training for step 1 (three fixed geometries: heavy cut 0013-like, medium cut 0021-like, shared FULL parent)

### 6.1 Data per geometry
- Exact factorisations from the route-7 encoder:
  - the Dirichlet interior K_II;
  - the floating K, regularised by pinning or R Rᵀ, then projected.
  Two cuDSS factorisations: fp32 plus 2-step refinement, or fp64 for FULL where memory allows.
- Every sample gives an exact field u* in milliseconds.
- **Rotations:** the 48 group elements are applied on the fly to the geometry features (signed-permutation moments
  and permuted masks), and to q and u*. That gives 144 geometry instances with no new solves.
- The patch modes and pairs of rotated instances are the rotated originals (they are equivariant), so eigenproblems
  are precomputed only for the 3 base geometries.

### 6.2 Loss
Per sample i at checkpoint j: ℓ_ij = ‖û_ij − u*_i‖²_K / ‖u*_i‖²_K = μ_ij − 1.
- The label u* is available and cheap, and computing ℓ directly avoids the cancellation in Π(û) − Π(u*).
- Accumulated in fp64. Pure rigid samples are excluded from the loss (exact by construction, 0/0).

    J = Σ_{j∈{4,6,8}} λ_j [ 0.7·mean_i φ(ℓ_ij) + 0.3·CVaR_25%(φ(ℓ_·j)) ] + 0.1·J_sens + 0.25·J_Neu,
    λ = (0.25, 0.5, 1),  φ(ℓ) = log(1+ℓ) for the first 1k steps, then φ(ℓ) = ℓ.

- **J_sens.** Mean over samples and the 8 corners c of |ûᵀK′_c û − u*ᵀK′_c u*| / Σ_c |u*ᵀK′_c u*|.
  - Cheap: per-element energy densities contracted with the stored ∂w/∂τ (51 MB per FULL cell, only the 3 training
    geometries).
  - Turned on after 2k steps. It trains the quantity the gate measures and counters risk 6.
- **J_Neu.** Neumann-action energy error ‖u_N − u*_N‖²_K/‖u*_N‖²_K on balanced box-force samples: 50% smooth, 30%
  GRF, 20% "BDD-like" rough residuals taken from lattice_pcg runs.
- No entry losses and no Frobenius terms (R14).
- Optional warm start: a D-norm field term ‖û − u*‖²_D/‖u*‖²_D with weight 0.1 for 1k steps, annealed to 0.
  It helps the stiff-first ordering of F4. Keep it only if the ablation in §6.6 says it helps.

### 6.3 Sampling (decision 9, with two additions)
- **Force-driven, 30%.** The field is u* = the floating Neumann solution for box forces f. Its box trace is q and
  its interior is exactly E q.
  - f is smooth multiscale on the box.
  - One third is **unit nodal face forces as in the gate**, and one third is **traction-consistent**
    f_i = ∫ N_i t dA (face moments of the patches).
  - About 10% carry cut-band forces f_C ≠ 0 (mixed mode).
- **Macro, 20%.** Rigid (loss-free exactness check), 6 uniform strains, quadratics, cubics.
- **Gaussian random fields on ports, 30%.** Correlation lengths from the whole face down to 2h.
- **Adversarial, 20% after 1.5k steps.**
  - Every 250 steps, run block LOBPCG (block 16, 8 iterations) on the pencil (Ŝ − S, S): Ŝ from forward and
    transpose, S⁻¹ from the exact floating solve.
  - Seeded with force-driven q. This targets the soft subspace, which is what matters for the lattice.
  - Keep a pool of the 64 worst directions per geometry, with exact u*.
  - Also run one unseeded LOBPCG for the reported μ_max.
- The mix is re-weighted only after the first lattice-gate readout.

### 6.4 Optimisation
- AdamW, lr 3e-4 on the hypernetwork (1e-2 on the global step biases), wd 1e-5.
- 500 warm-up steps, cosine decay to 20k steps. Gradient clipping 1.0.
- fp32 with TF32 disabled in K.
- One geometry instance per step, B = 32 columns, cycling FULL / 0021 / 0013 with 48 random rotations.
- Gradient checkpointing per sub-step.
- **Stop-gradient** through the Lanczos caps and the patch eigenpairs (geometry-only constants).
- Gradients flow through the Galerkin products, the level-3 Cholesky (torch backward) and all heads.
- Estimated ≈ 0.3 s per step (FULL) → ≈ 2 h for 20k steps.

### 6.5 What to monitor (every 250 steps unless noted)
1. μ − 1 per sample class (median, p90, max) at j = 2, 4, 6, 8 and at the certified tail j = 12 and 16. This gives
   the n_K curve.
2. μ_max from LOBPCG, reported and not gated. Also the soft-subspace worst direction.
3. Lattice gate every 1k steps:
   - `lattice_v2` (x and y configurations, 6 loads), with the model cut cell and the exact parent;
   - an all-model variant;
   - compliance and sensitivity errors;
   - also under traction-consistent loads.
4. Share of the error energy eᵀKe on weak nodes, on patch interiors and on patch boundaries. This localizes what is
   not captured.
5. Per-step contraction: the ratio of ‖e‖_A before and after each step, on the adversarial pool. Also the power-
   iteration spectral radius of the full trained error propagator (the stability of the β recurrence).
6. How often the caps bind: fraction of geometries where λ̂(TA) > 1.9/α. Also the θ histogram (saturation at ¼ or
   4?).
7. Equivariance error ‖Ê_{h·g}(ρq) − ρÊ_g q‖/‖Ê_g q‖ on random h. Expected 1e-6.
8. Calibration of the indicator η/‖e‖_K.
9. Neumann action: κ(B_N S) estimate by Lanczos, and BDD iteration counts on `lattice_pcg` at 2×2×2 (every 2k
   steps).
10. Gradient norms per head, and the linearity unit test.

### 6.6 Step-1 run list (in order; the cheapest first)
- **1a. Zero-learning structure. No training, about 1-2 h.**
  - Settings: θ = 1, ω = 1, g = 1, α = 1/1.9-capped, β = 0.
  - Measure the n_K curve per class, the lattice gate versus n_K, and the 64 slowest modes of the S-C propagator and
    their overlap with the patch-mode span.
  - Ablations: no patch modes; Q1(32³) level 1 instead of Q2(16³); one stagger instead of two.
- **1b. Free-coefficient oracle** (the same 86k scalars as free parameters, per geometry; about 1 h each). This is
  the upper bound of the structure.
- **1c. Hypernetwork with rotations** (the main run, about 2 h).
- **1d. K-free ablation.**
  - Replace the fine K in the sub-step residuals by a learned strain-form operator A_θ = Σ_e Σ_{q∈2³} B_qᵀ D_eq B_q
    at equal cost, with D_eq generated SPD 6×6.
  - The exact K is kept in the readout.
  - This tests the central claim (§2.1). If it wins, the owner's pure-feedforward preference is vindicated and the
    fine K moves out of the q-path.
- **1e. Neumann action in BDD.** Iteration counts with the exact forward operator plus the learned B_N, on 2×2×2
  and 3×3×3 FULL and on a mixed cut lattice.

**Step-1 pass criterion.**
- Lattice compliance and sensitivity errors ≤ 3% on all configurations at n_K ≤ 8 (target) or ≤ 16 (acceptable).
- Cut-loaded cases and μ_max are reported.

---

## 7. Risks (ranked), cheapest exposing experiment, graceful degradation

| # | Risk | Cheapest experiment that exposes it | Mitigation / degradation |
|---|---|---|---|
| 1 | The slow band is not captured by 125-node patch modes: modes straddle patches, or the "slow for pairs" criterion misses slow-for-S-C modes. The result is a plateau on force-driven μ. | 1a on 0013 and 0021: Lanczos for the 64 slowest modes of the zero-learning S-C propagator; fraction of their energy outside span(patch modes) + range(P1). About 1 h, no training. | 3 staggers or 3×3×3-element patches (≤ 343 nodes), k_max 16, lower threshold. Promote patch modes into the level-1 Galerkin space (enriched coarse). |
| 2 | Coarse transmission and locking on thin walls: the C step contracts poorly for macro and smooth-force loads on FULL. | 1a ablation: Q1(32³) vs Q2(16³) level 1, and free-fit θ (1b) on the FULL parent; two-level contraction on macro loads. A few hours. | Capacity knob K2 (§7.1): learned enrichment channels (a hypernetwork-generated multiscale basis) at levels 1-2. Rotation/strain partition-of-unity enrichment at level 3. |
| 3 | The n_K needed exceeds 16. | The 1a and 1c curves. | Accept up to 24 (≈ 1-1.5 ms per column is still within budget). Stronger recurrence (full Chebyshev tail). Fallback-A heads (§7.2). |
| 4 | The gate's uniform nodal forces load fictitious fringe nodes, so accuracy is spent on physically meaningless modes, or the gate over-states difficulty. | `lattice_v2` with the exact operators: energy split between weak and strong nodes, uniform nodal vs traction-consistent loads. Minutes, no network. | Report both. The owner decides the gate's load model. The architecture is unchanged. |
| 5 | Training instability: through the Galerkin products and Cholesky backward, the caps binding, or β recurrence blow-up. | 1b oracle on 0013, 1 h. Watch the monitors in 6.5 items 5 and 6. | Stop-gradient on θ (freeze at 1). β = 0. Reduce to route-1-style scalars (fallback B). |
| 6 | Sensitivity amplified on fringe elements, where K′/K is large (surface term). Sensitivity error ≫ compliance error. | 1a: lattice sensitivity error decomposed by element class, vs n_K. | J_sens term; patch modes cover fringe elements; more steps. |
| 7 | The Neumann action is too weak: BDD iterations grow more than 2×. | 1e with the zero-learning B_N (hours). | More symmetric steps (S C S C S). Exact level-2 sparse Cholesky (~20-40 MB). Train J_Neu harder. |
| 8 | The K-apply throughput estimate (25-40 µs per column) is wrong by 10×. | Step-0 microbenchmark of batched strain-form K (body + ghost) on FULL, fp32, B = 32. About 1 h. | Fuse kernels. Reduce n_K via risks 1-2. Batch across cells. |
| 9 | Encoding time: patch LOBPCG plus Galerkin exceeds 0.5 s per FULL cell. | Benchmark the encoder on the FULL parent. | Only weak-touched patches (≈ 70% of patches). Warm-start LOBPCG from the previous design step (superset band). k = 6. |
| 10 | Equivariance slip in the implementation. | Monitor 6.5 item 7 on day 1. | Canonical orientation for cut cells (frame by the plane normal, RESEARCH 6.3) plus augmentation. |

### 7.1 Capacity ladder (the upgrades, in this order)
- **K1.** Full Chebyshev-type learned recurrence over longer horizons.
- **K2.** Learned multiscale basis enrichment. The hypernetwork generates k_e ∈ {1, 2} extra vector channels per
  level-1 or level-2 node, with prolongation kernel ψ_θ(i,v) = a_θ I + b_θ d̂d̂ᵀ (equivariant). Galerkin keeps the
  guarantees. Memory +10-30 MB. This is the genuine "GNO-generated kernel" upgrade.
- **K3.** Bigger patches or more modes.
- **K4.** A learned multi-channel linear correction head per sub-step on levels 1-2 (fallback A proper).
- **K5.** More steps.

### 7.2 Graceful degradation
- **Into fallback A (learned iteration).** The same code with n_K raised: the certified tail makes this monotone.
  Optionally add K4, a geometry-conditioned linear network acting on the exact residual. Everything is shared: K,
  lift, readout, hierarchy, patches, hypernetwork, harness.
- **Into fallback B (route-1 skeleton).** Freeze the heads to constants and learn only (α_j, β_j) plus a few global
  dampings. That is route 1 with Q2 coarse spaces, patch modes and a better implementation.
- **The other direction.** If 1d shows that K-free wins, move K out of the sub-steps (keep it in the readout). The
  hypernetwork then also generates D_eq, and the design becomes the owner's pure-feedforward main line on the same
  hierarchy.

---

## 8. Borrowed vs new

**Borrowed.**

| item | source |
|---|---|
| Variational readout, energy loss, exact ports | decisions 1 and 7; Galerkin theory |
| Nested Q2 multigrid, Galerkin coarse operators, symmetric V-cycle as a preconditioner | classical |
| Multigrid as a linear-operator parameterisation | MgNet/MgNO (RESEARCH 1.1-1.2) |
| Learned multigrid parameters from operator features | Luz et al., Greenfeld et al., Huang-Li-Xi |
| Local low-energy (spectral) coarse components | GenEO, CEM-GMsFEM |
| Pair smoother, Lanczos step caps, learned recurrence scalars | route 1 |
| Learned coarse materials | B2 (project history) |
| Superset masks for design bands | route 7 |
| Strain-form, Gauss-weight K | DATA_INTERFACE |
| BDD with partition-of-unity rigid coarse space | Mandel |
| Symmetric Neumann action with the same basis on both sides | Melchers-Dolean-Abdelmalik |
| Orbit-shared isotropic kernels | G-CNN idea restricted to scalar fields |
| Hypernetwork paradigm | Ha et al., VarMiON, NGO |
| Consistency through exact residuals | Hsieh et al. |

**New in this proposal.**
1. The K-dominance argument (§2.1): exact K is the optimal fine kernel of a linear feedforward operator on a
   fixed-grid FE discretisation. The learned parts only need spectral equivalence, not 1e-4 accuracy.
2. The mixed Dirichlet-box / force-cut port operator, with the one-sided lattice bound for free and loaded cuts.
3. **Patches instead of weak clusters**, with the computed percolation evidence (§2.7), and exact undamped patch
   modes (§4.6).
4. Learned per-macro-element effective Lamé moduli on nested Q2 coarse spaces (a homogenization-type,
   locking-relieving coarse operator), with safety by Lanczos certification instead of trust-region restriction.
5. The certified tail: an inference-time accuracy dial that is monotone by construction. The main line and
   fallback A become one model.
6. One hierarchy serving the forward extension, the symmetric Neumann action and the error indicator η.
7. Equivariance-by-construction recipe for this pipeline: invariant scalar heads, tie-merged pairing, symmetric
   staggered partitions, eigen-projectors.
8. The Galerkin-orthogonality reading of the sensitivity error (first-order term only from the non-proportional
   part of K′). It motivates J_sens and the fringe-element risk.
9. The observation that the gate's uniform nodal forces load fringe and fictitious port nodes at O(1) (risk 4).

---

## Appendix: new counts (estimates; body-only weak proxy, uniform τ, sampled moments)

`_ol/hier_counts.json`: weak-node components (element-sharing graph, 1% threshold).

| cell | nodes | weak | components | largest components |
|---|---|---|---|---|
| FULL τ=0.5 | 121,036 | 20,196 (16.7%) | 2 | 10,212 / ≈10.0k |
| FULL τ=2/3 | 146,364 | 21,332 (14.6%) | 2 | 10,912 / ≈10.4k |
| cut 0.148 | 22,966 | 4,232 (18.4%) | 5 | 3,884 max; median 117 |
| cut 0.758 | 91,466 | 16,007 (17.5%) | 2 | 15,978 / 29 |

`_ol/patch_counts.json`: 2×2×2-element macro patches, two staggers.

| cell | patches with weak nodes (stagger 0 / 1) | weak per patch (median / max) | weak + strong ring (median / max) |
|---|---|---|---|
| FULL τ=0.5 | 1,552 / 1,476 | 16-17 / 33 | 45-75 / 109 |
| cut 0.148 | 328 / 343 | 15 / 37-42 | 63 / 109 |
| cut 0.758 | 1,226 / 1,174 | 16-17 / 46-54 | 45-66 / 109-117 |

The nested Q2 hierarchy counts are in §2.4.
