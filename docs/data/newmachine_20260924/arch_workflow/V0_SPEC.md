# V0_SPEC: SGNO, the Symmetric Galerkin-sandwich multiscale Neural Operator (v0)

Status: v0 architecture specification. It synthesizes the four proposals in `arch/DESIGN_*` and their critiques in `arch/CRITIQUE_*`.
It is binding on every decision in ARCH_BRIEF §2. Appendix A lists each critique point I accept, modify or reject, with the reason.

Tags used below:
- **[ev]**: repository evidence (ROUTES_PROGRESS, ARCHITECTURE_HISTORY, DATA_INTERFACE counts).
- **[est]**: an estimate from operation counts. Not yet measured.
- **[pre-reg]**: a pre-registered decision rule.
- **K_eq**: one exact fine-level K application on one column of the FULL parent, ≈ 1.0 GFLOP (0.37 body in strain form + 0.64 ghost factor form).

Planning throughput is 0.05–0.15 ms per K_eq per column, batched. The E0 microbenchmark in §6 replaces this figure.

---

## 0. Summary

**What SGNO computes.** SGNO is a fixed-depth, feedforward linear operator q ↦ û(g, q) on the fixed Q2 grid.
- Its only nonlinear part is a geometry path. The geometry path produces two kinds of object:
  - **exact geometry-generated kernels:**
    - the stiffness K, matrix-free from the 125 signed Gauss weights per element;
    - Galerkin coarse operators on the nested Q2 hierarchy 65³ → 33³ → 17³ → 9³;
    - a **local soft space** G(g), made of the low-energy modes of oversampled block problems (principal submatrices of K), solved exactly by a small sparse Galerkin factorization;
  - **learned coefficients:** a hyperedge GNN on K's hypergraph emits invariant scalars that shape the multiscale GNO smoothers:
    - Loewner-capped fine gates;
    - two- and four-channel polynomial smoothers on the 33³ and 17³ levels;
    - smoother polynomials.
- The q-path is a **palindrome of five blocks**, F → G → M → G → F. The blocks are:
  - F: fine gated Jacobi-Chebyshev;
  - G: exact projection onto the soft space;
  - M: learned multiscale V-cycle with an exact 9³ bottleneck.
- The exact residual is re-evaluated between blocks, 4 times. That puts about 8 K_eq in one forward pass. These are the "few K applications" of question d.
- Because the palindrome is symmetric, **the extension operator N is its own adjoint**. The reaction Ŝq = Πᵀ[v_Γ − K_ΓI N v_I], with v = Kû, is exactly ÊᵀKÊ. It is symmetric PSD, one-sided (μ ≥ 1) and rigid-null for any parameter values, and it needs no stored activations. It costs ≈ 18 K_eq ≈ 1–3 ms per column.
- Ports are exact by masking. Box ∪ cut band is Dirichlet in mode XP; the box is Dirichlet and the band takes nodal forces in mode XB. The rigid split is done in fp64.
- The same record serves a Neumann action (mode N). Wrapped in a Chebyshev polynomial, it is SPD, which BDD needs.
- Resident memory is ≈ 140–160 MB per FULL cell (modes XB and N), ≈ 13 GB for 100 cells [est].

**Design principle.**
- Every stiffness cancellation is computed exactly: residuals with exact K, and Galerkin solves on geometry-generated spaces.
- Learning goes where an exact solve is unaffordable (the 33³/17³ scales) or too conservative (step sizes).
- The dense soft band is removed by an exact solve on a subspace that contains it, never by relaxation. Relaxation is what forced route 1 to 24–48 layers [ev].

**What SGNO does not claim.** It does not claim that one pass passes the gate.
- The depth dial m (N_m = p_m(NA)N, which is still symmetric) runs continuously into fallback A.
- Freezing the heads gives fallback B.
- Step 1 decides m and the fate of the main line by pre-registered rules. Its baselines are:
  - the zero-parameter prior X0;
  - the per-geometry capacity ceiling X1;
  - a pure-feedforward arm with K only in the readout.

---

## 1. Requirement traceability

| # | Requirement | Evidence | Satisfied by |
|---|---|---|---|
| R1 | Output strictly linear in q (and in band forces f_c). Nonlinearity only on the geometry path. | Decision 5. RESEARCH F2. A q-nonlinear field regressor gave an indefinite Hessian (arXiv:2608.02036). | q-path rules (§2.5): every q-path op is a g-only matrix; no q-dependent scalar (no CG, no norms). Proof §3.1. Unit test §6-E0. |
| R2 | Exact port values on box ∪ cut band (hard), with exact K in the readout. | Decisions 1 and 4. Galerkin identity Ŝ − S = (Ê−E)ᵀK(Ê−E) ⪰ 0 needs exact ports (brief §4). | Corrections live only on free DOFs (Z), plus a final overwrite û_Γ := q. Modes XB/XP/N by masks (§2.2). Proof §3.2. |
| R3 | Ports at full resolution; cost independent of the number of port DOFs. | Decision 2. | q enters only through the port-force map b = −(K[q_d;0])_I, evaluated on port-touching elements (FULL: 1,776 of 12.9k) (§2.5 Q1). No port encoder, no compression. |
| R4 | Fixed background grid. Topology enters as masks. Sensitivity is taken within a topology. | Decision 3. τ ± 1e-3 changes ±4–6 active elements; the superset encoder matches to 1e-14 [ev]. | Superset design bands; node, element and face masks; discrete structures frozen per band (§4.5). |
| R5 | Exact rigid modes, in fp64, never passed through fp32 K. | Decision 6. An unprojected 1e-7 rigid residual gave a 12× cell-energy error in a 2-cell lattice (R7_15) [ev]. | Q0 fp64 rigid split. Rigid part added analytically. Rigid-exact readout (§2.6). Proof §3.3. |
| R6 | Joint (g, q) O_h equivariance. | Decision 6. Pipeline equivariant to 1e-7. Rotation augmentation helped held-out cut cells 0.64 → 0.41 [ev]. | O_h-symmetric blocks (even-start ℓ1 partition), hierarchy and windows; spectral (span) objects; invariant GNN; invariant W (§3.6). Augmentation used only as a test. |
| R7 | Symmetric PSD reaction with a cheap adjoint. One-sided error. | Decision 1; question e; Parish, Melchers (RESEARCH 3.3, 5.5); BDD-PCG. | Palindromic N = Nᵀ gives the self-adjoint readout formula (§2.6). Proofs §3.4–3.5. |
| R8 | Main loss is energy, normalized per sample (= μ − 1 in that direction). | Decision 7. Entry losses fight the soft directions (history phase 7 S1) [ev]. | §5 loss: ℓ_E = ‖û − u*‖²_K / ‖u*‖²_K in fp64. The label-free form is equal by Galerkin orthogonality. |
| R9 | Accuracy in the soft decile, i.e. in force-driven directions. | 94–96% of lattice load energy is in the softest 10%. The skeleton had 0.3% error on probes and 20–35% on force-driven loads (D2) [ev]. | G-stage (exact soft subspace); decision-9 sampling mix; CVaR tail; adversarial LOBPCG; lattice-bound theorem (§3.5) linking the loss to the gate. |
| R10 | A carrier for localized soft modes that is not throttled by damping and covers the dense band. | 17.6–21.9% weak nodes carry 94–99.4% of the slowest modes; 11–30 nodes per mode; 128 modes in [0.0039, 0.0062]. Merged groups under global damping moved λ_min only +14%. 128 ideal global modes helped only modestly (E1) [ev]. | G-stage: oversampled local spectral modes, centre-assigned, projected **exactly** by one sparse Galerkin solve (§2.4 E2, §2.5 Q3). No step size, no overlap damping. Capacity grows with thin and cut material. |
| R11 | Global transmission in every application, with bending-capable coarse spaces. | Brief §4. Quadratic ≫ affine coarse spaces (64-layer witness 1.26 vs 7.75, E0) [ev]. Q1 locks on 1–6-element walls. | M-stage: nested Q2 hierarchy with an exact dense fp64 ℓ3 (9³) solve in every pass (§2.5 Q4). No spectral layers (RESEARCH 7.2). |
| R12 | Coupling only through material, i.e. K's sparsity. No proximity edges. | RESEARCH F6; DATA_INTERFACE §2.2. | K's element and ghost-face hypergraph is used in the q-path, in the G local problems (principal submatrices of K) and in the GNN (K-connectivity). |
| R13 | Stiff and near-field cancellation must not depend on learned precision. | Soft stiffness is a 1e3–1e4 near/far-field cancellation. Entry accuracy 3e-4 was needed, 3–6 achieved (phase 4) [ev]. | Exact residuals between blocks. Exact Galerkin in G and ℓ3. The learned parts are spectral-equivalence-level shapers only. |
| R14 | Approximate Neumann (force → displacement) action, SPD on the balanced space. | BDD takes 11–53 iterations with S⁺ and 300–1600 without it [ev]. | Mode N palindrome, wrapped in a positive Chebyshev polynomial, rigid-projected (§4.2, proof §3.7). |
| R15 | Sensitivity −uᵀ(∂K/∂τ)u with exact ∂K, first order in field error. | Brief §4. Route-1 sensitivity/compliance error ratio 1.3–1.6 [ev]. | Strain-form contraction with ∂w = V⁻¹∂M/∂τ; ∂K_ghost = 0 within a topology (§4.4). ℓ_sens on the 8-vector of corner sensitivities (§5). |
| R16 | Memory ≪ 300 MB per cell; 100 cells resident on 32 GB. | Brief §6. The exact FULL fp32 factor is 6.6 GB [ev]. | ≈ 140–160 MB per FULL cell for XB + N; ≈ 13 GB for 100 cells (§4.6). No per-pair or per-element dense storage. |
| R17 | A few ms per cell-application, amortized. | Brief §6: ~2e4 applications per design iteration. | Ŝq ≈ 18 K_eq; Neumann action ≈ 28 K_eq at d_N = 3 (§2.7). |
| R18 | Encode ≲ 0.5 s beyond moments, with no large factorization. | Brief §6. | 0.3–0.6 s cold and 0.1–0.25 s warm-started for two modes; one sparse factor ≤ 64 MB per mode (§2.4). |
| R19 | Smooth parameterization; no max or abs heads. | A non-smooth head broke local-factor learning (phase 7) [ev]. | Bounded sigmoid/tanh heads with zero-initialized last layers; mean and logsumexp pooling; eigen-selection frozen per band. |
| R20 | Graceful degradation into fallbacks A and B with shared components. | Decision 8. | Depth dial m (fallback A); freeze heads (fallback B); the same record and readout (§2.9). |
| R21 | The cut band is a load-carrying port from day one: displacement or nodal-force type. | Decision 4 (staged). | Mode XP: band Dirichlet (Ŝ on box ∪ band). Mode XB: band free with nodal forces f_c, and load vector ĝ = Êᵀ(J_c f_c − KF f_c) (§2.6). |
| R22 | fp32 network, fp64 where soft directions need it. | fp32 matrices err 0.8–2.2% in the whitened spectrum (R7_15) [ev]. | Precision plan §2.8: fp64 rigid split, readout accumulation, G and ℓ3 factors, lattice vectors. |
| R23 | None of the dead ends come back. | History §3.1. | No entry or Frobenius loss, no dense factor output, no one-sided low-rank correction, no fixed-geometry coarse basis for soft modes (G is computed per geometry), no bandwidth truncation, no compression, no plain q-concatenation. |

---

## 2. Architecture

### 2.1 From mechanics to structure (why these blocks)

Let Γ be the Dirichlet port set of the mode, I the free nodes, A = K_II, and Z the injection I → all. The exact
extension is u* = Rα + [q_d ; A⁻¹ b] with b = −(K[q_d;0])_I + J_c f_c. Any linear N gives û = Rα + [q_d ; N b]. The
energy error of one sample is ‖(I − NA)x*‖²_A with x* = A⁻¹b (Galerkin orthogonality). This makes the design problem
concrete: **build a geometry-generated linear N whose error propagator I − NA is small on the x* that force-driven
loads produce.**

Four facts fix the structure of that N:

1. **x* has three kinds of content, and each gets its own exact or learned carrier.**
   - (a) The boundary ramp of the zero lift. It carries 20–64× the solution energy, is concentrated one element from
     the ports and is high-frequency. It is killed by the fine smoother F and by port-truncated coarse functions.
   - (b) Load paths: membrane action and bending of walls and necks, smooth along the shell, cell-wide. Carrier: M, the
     nested Q2 V-cycle with an exact 9³ solve.
   - (c) Localized soft mechanisms: sliver rigid motion, fringe pockets, flap bending. The dense band. Carrier: G.
2. **Relaxation cannot remove (c).** A stationary cycle contracts the band by 1 − λ_min ≈ 0.993 per application (route 1,
   λ_min ≈ 0.007) [ev]. The only one-shot remedy is an **exact Galerkin solve on a space that contains the band**
   (GenEO, spectral AMGe, CEM-GMsFEM). That is what G is.
   - G's space consists of the low eigenmodes of (A_Ω, D_Ω) on oversampled windows: modes that are slow *for the
     Jacobi smoother F*.
   - The union of these modes is then solved jointly and exactly. This glues modes that straddle block boundaries and
     removes overlap double-counting without any step size.
3. **Cancellations must happen in exact arithmetic.** A 1e3–1e4 near/far-field cancellation [ev] means a learned
   operator would need about 1e-4 relative accuracy in the soft directions.
   - With exact residuals between blocks and exact Galerkin matrices, each learned part only has to be a
     spectral-equivalence-level shaper, where 10–30% errors cost accuracy and not correctness.
   - This is why the forward pass has 4 exact inter-block residuals. §2.9 gives the pure-feedforward alternative that
     step 1 tests.
4. **The readout must be ÊᵀKÊ, symmetric and cheap.** A palindrome of symmetric blocks makes N symmetric, so the
   adjoint of the network is the network (§3.4). A lattice reaction then costs two forward passes plus one K, with no
   activation storage.

**Operator-learning reading.** Every block is a kernel integral with a geometry-generated kernel on K's hypergraph at its
own scale:
- F and M: local kernels at h, 2h and 4h (the multiscale GNO);
- G and the ℓ3 bottleneck: global, geometry-exact Galerkin kernels (the one-shot feedforward-NO part);
- F ∘ G ∘ M ∘ G ∘ F: a multipole-like near/far split (RESEARCH F1, 1.10), with the soft near field solved in a
  geometry-adaptive basis.

### 2.2 Modes (one record, masks only)

| mode | Dirichlet set Γ | band | use |
|---|---|---|---|
| **XB** | box-port nodes | free unknowns; nodal forces f_c on band nodes (f_c = 0 is a free cut) | lattice production for free or loaded cuts; the step-1 gate |
| **XP** | box ∪ Γ-band (decision 4's full port) | Dirichlet | Ŝ on box ∪ band; future skin; displacement-controlled cut |
| **N** | ∅ (floating) or the clamped face nodes | free | Neumann action for BDD |

- Γ-band = all 27 nodes of active elements whose plane section has positive-area material. Box membership takes
  precedence (DATA_INTERFACE §2.3).
- FULL cells have XB = XP.
- Per mode, the following are recomputed: masks, D, the G blocks touching Γ (the rest are shared), coarse masks and
  factors.

### 2.3 Sizes (FULL = the real parent; cut cells: real DOF where logged, else the uniform-τ stand-in)

| | FULL parent | medium cut (0021-like, retained 0.758) | heavy cut (0013-like, retained 0.148) |
|---|---|---|---|
| fine nodes / DOF (interior + box) | 122.7k / 344,628 + 23,616 | ≈ 70k / 197,658 + 12,363 | ≈ 22.6k / 62,961 + 4,794 |
| elements E / GP faces F | ≈ 12.9k / ≈ 21.9k | ≈ 9.6k / ≈ 16.5k | ≈ 2.3k / ≈ 3.5k |
| Γ-band nodes (XP) | 0 | ≈ 6.3k | ≈ 6.9k (≈ 3–4× the box port) |
| ℓ1 = Q2(16³): elements / nodes | 2,080 / 21.2k | 1,596 / 16.6k | 402 / 4.5k |
| ℓ2 = Q2(8³): elements / nodes | 352 / 3.9k | 284 / 3.2k | 79 / 1.0k |
| ℓ3 = Q2(4³): DOF (X / N) | ≤ 1,029 / 2,187 | ≤ 1,029 / 2,001 | ≤ 807 |
| ℓ1 blocks containing weak nodes (G windows) | 1,552 | 1,226 | 328 |
| G dimension n_G (planning, k̄ = 4; cap k ≤ 8 per block) | ≈ 6.2k (cap 12.4k) | ≈ 4.9k | ≈ 1.3k |

The weak-block counts come from `_ol/patch_counts.json` (body-only proxy). n_G is a planning value; E0 measures it.

### 2.4 Geometry path (once per cell per design step; nonlinear in g; no q)

```
tau corners, plane, superset band ──► CPU topology (fast_prep3, ~1 s, overlapped)
moments M (E×125, fp64, polyref) ─┬─► E1 exact products: w = V⁻¹M (E×125 fp32); D_i (N×3×3); masks per mode;
                                  │     ℓ1 coarse moments (8 binomial 125×125 maps); A₂ (BSR); A₃ + fp64 Cholesky
                                  ├─► E2 soft space G: per ℓ1 block b, oversampled window Ω_b (block + 1-element halo),
                                  │     A_Ω = principal submatrix of A (all element + GP terms), pencil (A_Ω, D_Ω),
                                  │     modes λ < θ_s, core-mass ≥ 0.4 → vectors φ_{b,j}; A_G = P_Gᵀ A P_G (fp64), Cholesky
                                  ├─► E3 certified bounds at gates = 1: λ̂_max(D⁻¹A), λ̂_max(D_ℓ⁻¹A_ℓ); Lanczos λ̂_min(NA)
                                  └─► E4 hypernetwork H_θ (invariant features → K-hypergraph GNN → coarse U-graph → heads)
                                        → ω_i (fine gates), g^{(c)}_{ℓ,v} (level gates), polynomial coefficients
CELL RECORD per mode: topology/masks, w, D⁻¹, gates, ℓ1 moments, A₂, ℓ3 factor, G vectors + factor, bounds
```

#### E1. Exact products (deterministic)

| object | shape (FULL) | cost | memory | notes |
|---|---|---|---|---|
| signed Gauss weights w_e = V⁻¹M_e | E×125 fp32 | trivial (fp64 solve, then cast) | 6.5 MB | K_e = Σ_q w_eq B_qᵀCB_q, exact to 1.4e-15 [ev]. All K applies are matrix-free in strain form. |
| node diagonal blocks D_i (mode-masked), and D_i⁻¹ | N×6 (symmetric 3×3) | 125×27×9 slice + ghost constants | 2.9 MB per mode | smoother carrier and GNN feature |
| ℓ1 operator A₁ | 2,080 coarse elements | matrix-free: coarse moments × template (h/H)², plus 48 ghost sub-face templates | 1.1 MB | Galerkin of the nested Q2 P₁ (systems Prop. 2.4). **Build order** (critique S8): first direct fine-element Galerkin (2,080 × 81² upper = 28 MB, fine for 3 geometries), then switch to coarse moments once they agree to 1e-12 and F·P_s = 0 is verified for interior sub-faces. |
| A₂ = P₁₂ᵀA₁P₁₂ | BSR 3×3 blocks, ≤ 3.9k × 125 | ≈ 27 GFLOP (fine-element Galerkin), a few ms | 18 MB fp32 (fp64 accumulation); shared across modes via row masks | |
| A₃ = P₂₃ᵀA₂P₂₃, Cholesky | X: ≤ 1,029²; N: 2,187² (pinned + projected) | ≤ 4 GFLOP fp64 ≈ 2–6 ms | 4.2 / 19 MB fp64 packed | the global bottleneck |

Masking of coarse functions:
- **XB and N.** Coarse nodes on box planes are excluded in XB. The coarse functions then vanish on the whole face,
  consistent with Dirichlet box ports. Non-port box-plane nodes are over-constrained, which is still exact Galerkin
  (systems §2.4).
- **XP.** The prolongation is masked on band nodes, P̄ = ZZᵀP. A_ℓ is formed by direct fine-element Galerkin with P̄
  rows zeroed, which includes the ghost faces (systems-critique S5). Storage ≤ 14 MB per cut cell.

#### E2. The soft space G (the carrier of the dense band)

For each ℓ1 block b (2×2×2 fine elements) that contains weak or cut elements:

1. **Window.** Ω_b = block b plus a one-element halo: 4³ elements. The unknowns are the free nodes strictly inside
   the window's outer node boundary: ≤ 7³ = 343 nodes, ≤ 1,029 DOF, ≈ 500 active on average [est]. The window family
   is O_h-symmetric: blocks start at even indices, so s ↦ 30 − s maps the family onto itself.
2. **Local operator.** A_Ω is the principal submatrix of the mode's global A on those nodes. It includes every element
   term and every GP-face term among them, so the Dirichlet (zero) outer boundary is implied.
   - This is route 2's "GP outer ring" lesson in its exact form: slivers stay supported by all their GP faces.
   - It is also op-learning-critique F1a: no floating Neumann patch, no spurious kernel.
3. **Pencil.** A_Ω z = λ D_Ω z. Here D is the 3×3 point-block diagonal of the global A, the same D that the fine
   smoother F inverts.
   - Small λ means slow for F: a local near-mechanism.
   - Smooth solid modes of a 4-element window sit at λ ≈ O(0.1–1) and are left to M.
   - Keep λ < θ_s with θ_s = 0.05·λ̂_max(D⁻¹A). Extend θ_s to the largest relative spectral gap in [θ_s/2, 2θ_s].
   - Cap k ≤ 8 per window. Batched LOBPCG (block 16, 20 iterations), fp32 on D-scaled matrices, with an fp64
     Rayleigh–Ritz polish.
4. **Centre assignment.** Keep a mode only if ≥ 40% of its D-mass lies on block b's own nodes. The vector keeps its
   full window support; no PU truncation is needed.
   - A localized mode is represented, largely whole, in the window of the block that holds its centre.
   - Coverage, recomputed with node parity: the window interior is 7 nodes per axis, and the block centre's period is
     4 nodes.
     - Exact containment is guaranteed per axis only for node extent ≤ 4 (every position). Extent 5 is contained at
       3 of the 4 centre positions, extent 6 at 2 of 4, extent 7 at 1 of 4 (checked by enumeration).
     - A 30-node sheet mode (≈ 5–6 × 5–6 × 1–2 nodes) is therefore often **not** contained whole. Its tail is carried
       by the neighbouring windows' vectors and glued by the joint exact solve, which is GenEO-type gluing through
       overlap.
     - Whether this suffices is exactly what E0d measures (≥ 90% of the slow-mode energy in span(G)). Route-1 modes are
       11–30 nodes [ev].
   - The 0.4 threshold admits exact ties (0.5/0.5) from both sides. The duplicates are harmless and keep the rule
     O_h-invariant.
5. **Galerkin.** P_G = [φ_{b,j}] (n_G columns). A_G = P_GᵀAP_G is assembled in fp64: each column needs one local K on
   its support, ≈ 5 GFLOP in total. It is regularized by 1e-10·diag against near-duplicates, which is span-invariant
   and order-free, and factorized by sparse Cholesky (cuDSS, ≤ 64 MB [pre-reg cap]).
   - Coupling stencil: 5³ blocks, because GP faces reach 4 node layers.
   - Planning: nnz(A_G) ≈ 1.2M; factor ≈ 6–8M entries; fp32 factor + fp64 A_G for one refinement step ≈ 40 MB.
   - Vectors: ≈ 12 MB fp32.
6. **Cost** [est]:
   - local eigensolves 1,552 × ≈ 7e8 FLOP ≈ 1.1 TFLOP, i.e. 0.1–0.2 s cold and ≈ 3× less warm-started from the
     previous design step (subspace iteration);
   - A_G assembly and factorization ≈ 10–40 ms.
7. **Frozen per design band:** the selected count k_b per block. Within a band the spans vary continuously as long as
   the selection sits at a gap (Davis–Kahan); the gap ratio is monitored and must be ≥ 1.5.

**Why this carrier and not patches in the smoother.** All four proposals put local slow spaces inside a smoother, and
all four critiques found the same failures:
- global damping crushing the exact blocks (route 1: λ_min +14% only [ev]);
- coverage holes for sheet-like modes;
- straddling modes that block-Jacobi cannot glue.

Putting the modes into one exact Galerkin solve removes all three at once, at a memory cost of ≈ 50 MB. It is
spectral-AMGe / GenEO with the soft space defined relative to the actual smoother.

#### E3. Certified bounds (per geometry and mode, computed once at gates = 1)

- λ̂_max(D⁻¹A) and λ̂_max(D_ℓ⁻¹A_ℓ): 20-step Lanczos from 3 random starts, with the residual bound θ_k + ‖r_k‖. About
  20 fine K.
- **Loewner caps** (systems-critique S3): every learned gate is ≤ 1, so S(gates) ⪯ S(1). The bound computed at gates = 1
  therefore holds for all gate values. It is gate-independent, needs no stop-gradient refresh and has no stale-bound
  instability.
- λ̂_min(NA) and λ̂_max(NA): 30-step Lanczos (for the depth dial, the Neumann Chebyshev interval and the indicator).

#### E4. Hypernetwork H_θ (learned, nonlinear, O_h-invariant)

**Inputs** (invariant, dimensionless, smoothly clamped: log(x + 1e-4)):
- **Element (≈ 30):**
  - vf and log vf;
  - 3 sorted eigenvalues of the normalized second-moment tensor; centroid offset/h;
  - the 18 eigenvalues of the element's linear-strain (quadratic-displacement) energy matrix ÷ vf. It is linear in
    the moments. The uniform-strain 6×6 matrix is **not** used: it carries only vf (systems-critique m1);
  - trace and ‖·‖_F of D^{-1/2}K_eD^{-1/2};
  - full, cut, band and port-touching flags;
  - τ at the centre; (|φ| − τ)/h; signed plane distance/h;
  - |cos| of the angle between ∇φ and the plane normal (an oriented invariant; op-learning-critique m9).
- **Node (≈ 12):**
  - log(‖D_i‖/median) and the eigenvalue ratios of D_i;
  - parity class (4);
  - Dirichlet-in-mode flag and face count (face-agnostic);
  - band flag, in-material flag;
  - number of incident elements /8, number of incident GP faces /12.
- **GP face (≈ 5):**
  - the sum and |difference| of the two volume fractions;
  - number of full sides;
  - weak count.
  - No axis identity.
- **ℓ1 block (≈ 12):**
  - k_b and the sorted log λ of its G modes (padded with smooth defaults);
  - weak, active and port fractions.
- **Global (≈ 8):**
  - retained fraction; mean τ; cut flag; mode one-hot;
  - log λ̂_max bounds and log λ̂_min(NA) of the untrained operator. These are explicit spectral normalizers
    (mechanics-critique m6).

**Trunk.**
- Node ↔ element ↔ GP-face bipartite message passing on K's hypergraph: 4 rounds, width 64, SiLU, mean + sum
  aggregation. Node → element weights depend only on the node's orbit class within the element (corner, edge, face,
  centre), which makes the layer exactly invariant.
- Pool to ℓ1 blocks (mean ‖ logsumexp). Then 3 rounds on the ℓ1 K-connectivity graph (blocks sharing an active element
  face or a GP face; not proximity), width 96.
- Pool to ℓ2, 2 rounds, width 128. Global token (mean ‖ logsumexp → MLP 128), broadcast back.
- U-shaped unpool with skips back to ℓ1 and to fine nodes.

**Heads.** Zero-initialized last layers. Bounded smooth maps; no max or abs.

| head | entity | map | count (FULL) |
|---|---|---|---|
| fine gate ω_i | fine node | σ(3 + z) ∈ (0, 1), ≈ 0.95 at init; Loewner-capped | 122.7k |
| level gates g^{(c)}_{1,v}, g^{(c)}_{2,v} | ℓ1 nodes (C₁ = 2), ℓ2 nodes (C₂ = 4) | σ(3 + z) | 42.4k + 15.5k |
| smoother polynomials | fine (degree 2); ℓ1 and ℓ2 per channel (degree 2) | Chebyshev reference + 0.3·tanh(z), with a smooth barrier keeping p(t) ≥ 0 and t·p(t) ≤ 1.9 on [0, 1] | ≈ 30 |
| cycle polynomial (only when m > 1) | global | Chebyshev on [λ̂_min(NA), 1] + 0.3·tanh(z) | m |

- Parameters ≈ 0.45M: trunk 0.10M, coarse U-graph 0.26M, global 0.05M, heads 0.04M.
- Generated per FULL cell: ≈ 0.18M scalars (0.7 MB).
- Cost ≈ 15 GFLOP, 3–5 ms.
- **At initialization every gate is ≈ 1 and every polynomial is Chebyshev: the untrained SGNO is a working solver**
  (X0, the zero-parameter baseline).

### 2.5 The q-path (strictly linear in q and f_c; B columns per call)

**Hard rules (unit-tested).**
- No activation, normalization, attention, max or batch statistic touches a q-path tensor.
- No step size is computed from q-path inner products, so no CG or LOBPCG runs inside the cell operator.
- No q-derived input reaches H_θ.

| id | operation | shapes (FULL) | q-linear / geometry-only | FLOP per column [est] |
|---|---|---|---|---|
| Q0 | Rigid split in fp64. α = (R_ΓᵀWR_Γ)⁻¹R_ΓᵀWq; q_d = q − R_Γα. W is the lumped port-area weight, O_h-invariant. | q (B, 3N_Γ) fp64 → α (B, 6), q_d | linear; W, R geometry | 0.3 MFLOP |
| Q1 | Port force. b = −(K[q_d; 0])_I + J_c f_c, evaluated only on elements and GP faces touching Γ (FULL: 1,776 elements). | b (B, 3\|I\|) fp32 | linear; K exact | 0.15 GFLOP |
| Q2 | **F block** (fine smoother, K's hypergraph). B_F = s₀·p_F(S₀A)S₀ with S₀ = ΩD⁻¹Ω, Ω = diag(ω_i) ≤ 1, s₀ = 1/(1.05 λ̂_max), p_F of degree 2 (Chebyshev at init). Symmetric; λ(B_F A) ∈ [0, 1.9] by the barrier. | x, r (B, 3\|I\|) | linear; ω, p_F learned | ≈ 1.0 (1 K inside) |
| Q3 | **G block** (exact soft projection). B_G = P_G A_G⁻¹ P_Gᵀ: restrict with n_G local dot products, fp64 sparse triangular solves, prolong. | coefficients (B, n_G) fp64 | linear; exact, no learning | ≈ 0.05 (plus fp64 solves) |
| Q4 | **M block** (learned multiscale GNO V-cycle). B_M = P₁V₁P₁ᵀ. V₁ is symmetric: pre T₁ → ℓ1 residual (A₁) → P₁₂ᵀ → pre T₂ → ℓ2 residual (A₂) → P₂₃ᵀ → **A₃⁻¹ exact (fp64)** → P₂₃ → post T₂ → P₁₂ → post T₁. T_ℓ = (1/C_ℓ) Σ_c W_c p_{ℓ,c}(W_cÂ_ℓW_c)W_c, W_c = diag(g^{(c)}_ℓ)(s_ℓD_ℓ⁻¹)^{1/2}. The 1/C_ℓ factor keeps λ(T_ℓA_ℓ) ≤ 1.9, because λ_max is subadditive over PSD terms. C₁ = 2 channels at 33³, C₂ = 4 at 17³. | ℓ1: (B, C₁, 3N₁); ℓ2: (B, C₂, 3N₂); ℓ3: (B, ≤ 2,187) | linear; gates and polynomials learned; A_ℓ exact Galerkin | ≈ 2.1 (8 A₁ at 0.19 each + residuals + ℓ2) |
| Q5 | **Palindrome with exact residuals**: x = N b, where I − NA = (I − B_F A)(I − B_G A)(I − B_M A)(I − B_G A)(I − B_F A). Evaluated as 5 corrections, each preceded by the exact residual r ← b − Ax (4 fine K between blocks). Output û = Rα + [q_d on Γ ; x on I], then the port overwrite û_Γ := q. | û (B, 3N) fp32; α fp64 | linear | total forward ≈ **8.3 GFLOP ≈ 8.3 K_eq** |

**Rationale per block.**

**F** (mechanics: stiff and boundary-layer content; numerical analysis: smoothing property).
- Point 3×3 blocks, not mutual-strongest pairs. The bulk is tie-dominated because every interval-full element has an
  identical K_e (multilevel-critique S7).
- Gates are Loewner-capped. The spatially variable step size (ω) is what route 1 identified as the next learnable
  parameter [ev].
- K itself is never gated (mechanics-critique S6), so every stage stays consistent with the exact residual.

**G.** Covered in §2.4 E2. It is the exact local Green's response of every mechanism that is slow for F. The amplitude
of each mechanism is a quotient of exact quantities (the mechanics design's §4.8 argument, now valid because the solve is
joint).

**M** (global load paths, bending, transmission).
- Q2 h-coarsening keeps through-thickness bending on walls 1.5–6 elements thick; Q1 would lock.
- The ℓ3 solve couples every node to every node once per pass.
- The learned C-channel polynomial smoothers are the multiscale-GNO kernels. Each channel is a differently gated local
  inverse on the exact Galerkin hypergraph of its level, steered by the geometry network to thin walls, necks and ports.
  This is where learning is cheaper than exactness: an exact ℓ1 solve (63.6k DOF) would need a factor of ≈ 200 MB.
- No learned transfer smoothing and no learned coarse moduli; Appendix A (A7, A11) gives the reasons.

**The palindrome order F G M G F.**
- The soft band is projected out both before and after the global correction.
- The global correction sits in the centre, the bottleneck of the "U".
- Every stage is consistent with the exact residual (Hsieh), so x* is a fixed point of every stage and of N_m for every m.

**Depth dial (fallback-A direction).** N_m = p_m(NA)N, with p_m of degree m − 1 on [λ̂_min(NA), 1]: Chebyshev at init,
learned deviations bounded. It is symmetric for every m. Cost ≈ 9.3·m − 1 K_eq per forward. m = 1 is the v0 default.

### 2.6 Readout (question e), load vector, indicator

- **Energy** (training, gate checks): ûᵀKû = Σ_e Σ_q w_eq ε_q(û_e)ᵀCε_q(û_e) + γ Σ_f ‖F_f û_f‖². Evaluated
  rigid-exactly:
  - gather in fp64;
  - subtract each element's least-squares rigid fit (a fixed 6×81 pseudo-inverse);
  - form strains in fp32;
  - accumulate in fp64.

  Rounding then scales with the element deformation, not with λ_max‖u‖. Cost 1 K_eq.
- **Reaction (lattice).** Write E_d = [I ; −N K_IΓ]. Since KR = 0,

  Ŝ = Πᵀ E_dᵀ K E_d Π,  so  **Ŝq = Πᵀ[ v_Γ − K_ΓI N v_I ]**, where v = K[q_d ; x] (rigid-exact) and Π = I − R_Γ(R_ΓᵀWR_Γ)⁻¹R_ΓᵀW.

  The adjoint of the network is N itself, applied once more to v_I. No stored activations and no transposed code are
  needed. Cost: forward 8.3 + K 1.0 + N 8.3 + K_ΓI 0.15 ≈ **18 K_eq ≈ 0.9–2.7 ms per column**. Symmetric PSD for any
  parameters (§3.4).
- **Loaded cut (mode XB, f_c ≠ 0).** û(q, f_c) = E q + F f_c with F = [0 ; N J_c]. The lattice load vector is
  ĝ = Πᵀ E_dᵀ (J_c f_c − K F f_c). It costs one extra forward and one adjoint per load case, and the potential
  Π(q) = ½qᵀŜq − ĝᵀq is exactly variational.
- **Rejected lattice operator: (Kû)_Γ.** It is non-symmetric (RESEARCH 3.3). It is kept as a free diagnostic:
  Ŝq − (Kû)_Γ = −Πᵀ K_ΓI N r_I measures the interior imbalance.
- **Indicator.** η² = v_Iᵀ N v_I, with v_I = (Kû)_I = A e. Since NA has spectrum in [λ_min, 1],
  η² ≤ ‖e‖²_A ≤ η²/λ̂_min(NA).
  - It is an **estimate**, because λ̂_min comes from Lanczos.
  - It is free inside Ŝq.
  - Route 2's equilibrated certificate stays the rigorous two-sided bound for later steps.

### 2.7 Cost and memory summary (FULL parent; cut cells scale with E: medium ≈ 0.75×, heavy ≈ 0.2×) [est]

| item | K_eq per column | time per column (0.05–0.15 ms per K_eq) |
|---|---|---|
| forward extension (m = 1) | 8.3 | 0.4–1.2 ms |
| energy (forward + readout K) | 9.3 | 0.5–1.4 ms |
| Ŝq, symmetric reaction | 18 | 0.9–2.7 ms |
| Neumann action M_c (d_N = 3, §4.2) | ≈ 28 | 1.4–4.2 ms |
| forward at m = 2 / 4 / 8 | 17.6 / 36 / 73 | Ŝq 36 / 73 / 147 K_eq |

| stored per FULL cell | size | notes |
|---|---|---|
| topology, masks, maps (E×27 int32, faces, coarse maps) | 3 MB | shared by all modes |
| w (E×125 fp32) | 6.5 MB | shared |
| ℓ1 coarse moments + ghost pattern ids; A₂ BSR | 1.1 + 18 MB | shared (row masks per mode) |
| D⁻¹, fine and level gates | 3.6 MB per mode | |
| ℓ3 fp64 factor | 4.2 (X) / 19 (N) MB | |
| G: vectors (fp32) + A_G (fp64) + factor (fp32 with fp64 refinement) | ≈ 12 + 9 + 30 ≈ 50 MB per mode (cap 64 MB) | blocks away from Γ share vectors across modes |
| **total, modes XB + N** | **≈ 140–160 MB** | heavy cut ≈ 30 MB; medium cut ≈ 110 MB |
| working set per column (inference) | ≈ 9 MB | 6 fine vectors + coarse |
| training activations (B = 32, m = 1) | ≈ 0.6 GB | only learned-coefficient ops save inputs; the exact ops have fixed matrices |

**Encode per design step (network-specific, on top of moments), per mode:**
- G eigensolves 0.1–0.2 s;
- A_G assembly and factorization 0.01–0.04 s;
- A₂, ℓ3 and Lanczos bounds ≈ 0.02 s;
- H_θ ≈ 0.005 s.

That is ≈ 0.15–0.3 s per mode, **0.3–0.6 s cold for XB + N, and 0.1–0.25 s warm-started** in the design loop. There is
no factorization larger than ≈ 64 MB.

### 2.8 Precision plan (question k)

| operation | precision | reason |
|---|---|---|
| rigid split, Π, lattice vectors, PCG/BDD dot products | fp64 | soft directions need it; the 12× rigid failure [ev] |
| K inside the q-path (residuals), smoothers, M block | fp32, **TF32 disabled** | these are field errors, second order in energy, and later stages contract them |
| G solve (A_G) and ℓ3 solve | fp64 accumulation; fp32 factor + 1 fp64 refinement for A_G; fp64 factor for ℓ3 | G and ℓ3 carry the soft and global directions uncorrected. Their field error enters the sensitivity at first order (critiques: multilevel S3, systems m5). |
| readout: energy, Ŝq reductions, sensitivity | rigid-exact fp32 strains with fp64 accumulation | rounding relative to deformation (§2.6) |
| local eigensolves (E2) | fp32 on D-scaled matrices, fp64 Rayleigh–Ritz polish | spans only; they are then used inside an exact Galerkin solve |
| training loss ℓ_E | fp64 difference form ‖û − u*‖²_K | avoids the cancellation in Π(û) − Π(u*) |

### 2.9 The main line: feedforward NO + multiscale GNO, and where K appears (questions d and l)

**Mapping to the owner's decision 8.**
- SGNO is a fixed-depth DAG. There is no convergence loop, no Krylov method, no q-dependent scalar. It is trained end to
  end as an operator.
- Its "feedforward NO" half: the global geometry-generated kernels, the ℓ3 bottleneck and the G projection. Both are
  one-shot.
- Its "multiscale GNO" half: the learned local kernels on K's hypergraph at 65³ (F gates), 33³ and 17³ (M channels).
- **Four exact inter-block residuals (≈ 4 K_eq) plus one K inside F on each side are the "few K applications inside"
  of question d.**

This is a hybrid by the brief's own vocabulary. I recommend it because of R13, the cancellation argument: without exact
residuals, the learned kernels must carry the 20–64×-energy lift ramp to 1–2% relative accuracy.

**The owner decides; open item O1.** Step 1 runs the literal alternative on the same stored objects:
- **Arm A-FF (pure feedforward, K only in the readout):** N_FF = η_F S₀ + η_G B_G + η_M B_M. The blocks are additive,
  there are no inner residuals, the η are learned global scalars, and M gets C₁ = C₂ = 4 with degree 3, so the arm is
  compared at equal wall-clock time.
- If A-FF passes the gate, the inner K are removed ([pre-reg], §6).

**Degradation (question l). One codebase, three dials.**

| configuration | what changes | when used |
|---|---|---|
| **main line** | m = 1, learned heads | default |
| **main line at depth** | m = 2–4 (symmetric cycle polynomial) | step-1 rule "GO-with-depth" |
| **fallback A** (learned iteration) | m ≥ 4 with per-cycle learned heads: each cycle is a geometry-conditioned linear correction of the exact residual | step-1 rule A |
| **fallback B** (route-1-like skeleton) | heads frozen at init. Only ≈ 30 global polynomial scalars are learned, label-free (route-1 E1b style). This is route 1 on a stronger skeleton: 4-level exact Galerkin Q2 + G instead of 2-level block polynomials + pairs. | step-1 rule B |

All configurations keep §3 (linearity, ports, rigid modes, symmetric PSD readout, one-sided error), the readout, the
Neumann action and the lattice integration unchanged.

### 2.10 Borrowed vs new

**Borrowed, adapted.**
- The linear V-cycle as an operator parameterization with nonlinearity only on the coefficients (MgNO, MgNet).
- Galerkin nested Q2 coarse operators from coarse moments (systems Prop. 2.4).
- Spectral-AMGe / GenEO / CEM-GMsFEM local low-energy spaces, solved exactly, for contrast-robustness (Spillane et al.;
  Chartier et al.).
- The principal-submatrix and GP outer-ring lesson (route 2 D1).
- Hsieh-consistent corrections with exact residuals.
- Loewner-capped gates (systems critique).
- The self-adjoint symmetric extension and free indicator (systems design).
- Port-force entry (mechanics).
- W-weighted rigid split (op-learning / mechanics).
- Rigid-exact mixed precision (systems / DATA_INTERFACE).
- Hyperedge GNN on K's hypergraph with orbit-class aggregation (multilevel).
- Invariant features and augmentation-as-test (EquiModel lesson).
- The Melchers-Dolean-Abdelmalik symmetric preconditioner for BDD.
- Balanced BDD with stored AΦ (systems).
- Lattice compliance bound (mechanics §4.5).
- Energy loss (Deep Ritz, VINO).
- ENGD insight → D-norm warm-up (Müller-Zeinhofer).
- Adversarial data aggregation (Beatson et al.).

**New in SGNO.**
1. **The soft band in an exact Galerkin solve on "slow-for-the-smoother" local modes.** The modes are oversampled
   principal-submatrix eigenmodes of the pencil (A_Ω, D_Ω), centre-assigned, and the solve is joint. It replaces every
   "patch smoother" variant of the four proposals and their three shared failure modes (damping, coverage,
   straddling).
2. **A symmetric five-block palindrome as a feedforward operator.** Exact where cancellation lives, learned where
   exactness is unaffordable, with the adjoint equal to the operator.
3. **One O_h-symmetric window family** (even-start ℓ1 blocks + one-element halo) serving the soft space. I checked the
   parity argument: stride-2 windows of odd width are not reflection-symmetric on a 32-element grid.
4. **Loewner-capped learned smoothers.** Certified once per geometry and independent of the learned values, so training
   never sees a stale or stop-gradient bound.

### 2.11 Answers to the open questions a–l (index)

| Q | Answer (section) |
|---|---|
| a | K's hypergraph at every level: the q-path, the G local problems and the GNN. Moments enter as exact K (w) and as invariants. GP faces enter as hyperedges. Weak nodes are handled by the G modes; they are not thresholded in the q-path. (§2.4, §2.5) |
| b | Z-masked corrections plus an exact overwrite; fp64 rigid split; mode masks for box and band. (§2.5 Q0–Q5, §3.2–3.3) |
| c | Global: nested Q2 M + exact ℓ3 in every pass. Localized: G, an exact joint solve. (§2.4 E2, §2.5) |
| d | 4 inter-block residuals + 2 inside F ≈ 8 K_eq per forward; the depth dial m; the pure-FF arm A-FF. (§2.9) |
| e | Ŝq = Πᵀ[v_Γ − K_ΓI N v_I], with the network as its own adjoint. (§2.6) |
| f | Mode-N palindrome in a positive Chebyshev polynomial, rigid-projected. (§4.2) |
| g | Strain-form −ûᵀ(∂K/∂τ)û with ∂w = V⁻¹∂M/∂τ; frozen band structures; ℓ_sens. (§4.4, §5) |
| h | By construction. Augmentation only as a test. (§3.6) |
| i | Per-sample μ − 1, CVaR, D-norm warm-up, bounded heads around theory values, Loewner caps. (§5) |
| j | 0.45M shared weights, ≈ 0.18M generated scalars per cell. The exact spaces (G, A_ℓ) are computed, not learned. The capacity ladder is in §6. |
| k | §2.8. |
| l | §2.9. |

---

## 3. Guarantees (short proofs)

The geometry g, the mode and the learned weights θ are fixed. Every generated coefficient is then a constant.
Notation: A = K_II; Z injects I; e = û − u*; Q_h is the signed node permutation of h ∈ O_h.

### 3.1 Exact linearity in (q, f_c)
Every q-path operation (§2.5) is multiplication by a matrix whose entries depend on (g, mode, θ) only:
- Π and α (Q0);
- K restricted (Q1);
- S₀ and p_F (F);
- P_G, A_G⁻¹ (G);
- P_ℓ, A_ℓ, T_ℓ, A₃⁻¹ (M);
- the residual updates b − Ax;
- the port overwrite, which is an affine map in (q, x) with a q-linear right-hand side.

Sums and compositions of linear maps are linear, so û = Ê(g)q + F̂(g)f_c exactly. The nonlinear maps (MLPs, sigmoid,
Lanczos, eigensolves, Cholesky) take (g, mode) only; the rules in §2.5 are enforced by a unit test. In floating point,
‖Ê(aq₁ + bq₂) − aÊq₁ − bÊq₂‖ is at the level of rounding. ∎

### 3.2 Exact port values
- u₀|_Γ = R_Γα + q_d = q.
- Every correction is Z(·), which has zero rows on Γ. In XP, P̄ = ZZᵀP carries the mask into the coarse functions too.
- The final assignment û_Γ := q makes the result bit-exact for box and band.
- This holds for every θ and every mode. ∎

### 3.3 Exact rigid modes
Take q = R_Γβ. R_Γ has full column rank (≥ 3 non-collinear port nodes; for the heavy cut this is checked at encode).
- Then α = β and q_d = 0 up to fp64 rounding (≈ 1e-16‖q‖).
- So b = 0 (f_c = 0), all corrections vanish, and û = Rβ. Hence Ê R_Γ = R.
- Readout: Ŝ = ΠᵀE_dᵀKE_dΠ with ΠR_Γ = 0 in fp64, so ŜR_Γ = 0 to fp64 rounding.
- K is never applied to Rα: Q1 applies K to [q_d; 0], and the readout subtracts the element rigid fit in fp64. So the
  fp32 leakage mechanism behind the 12× failure [ev] cannot occur. ∎

### 3.4 N is symmetric, so Ŝ is symmetric PSD and the adjoint formula holds

**N is symmetric.** Each block B_k ∈ {B_F, B_G, B_M} is symmetric:
- S₀ is diagonal-block SPD, and p_F(S₀A)S₀ is a polynomial in S₀A times S₀;
- B_G = P_G A_G⁻¹ P_Gᵀ;
- B_M = P₁V₁P₁ᵀ, where V₁ is a symmetric V-cycle: its post-smoother is the transpose of the symmetric pre-smoother, and
  its coarse correction is recursively symmetric with an exact A₃⁻¹.

In the A-inner product, I − B_kA is self-adjoint. The palindrome E_N = I − NA = X*(I − B_MA)X, with
X = (I − B_GA)(I − B_FA), is therefore A-self-adjoint: AE_N = E_NᵀA. Hence
N = (I − E_N)A⁻¹ = Nᵀ. The same holds for N_m = p_m(NA)N, since (NA)^jN = N(AN)^j.

**Ŝ is symmetric PSD.** With E_d = [I; −NK_IΓ], Ŝ = ΠᵀE_dᵀKE_dΠ is a Gram matrix of the PSD K: xᵀŜx = ‖E_dΠx‖²_K ≥ 0.
- Adjoint formula: E_dᵀv = v_Γ − K_ΓI Nᵀ v_I = v_Γ − K_ΓI N v_I. So Ŝq = Πᵀ[v_Γ − K_ΓI N v_I] with v = KE_dΠq = Kû
  (KR = 0).
- This holds for every θ, including untrained or diverged weights.

**N is PSD and NA has spectrum in (0, 1].** The barrier gives λ(B_FA) ∈ (0, 1.9], and every level smoother of V₁
satisfies λ(T_ℓA_ℓ) ∈ (0, 1.9] (Loewner cap plus barrier).
- So I − B_MA has A-spectrum in [0, 1]: a symmetric V-cycle with convergent symmetric smoothers and an exact coarsest
  solve.
- I − B_GA is an A-orthogonal projector.
- ‖(I − B_FA)v‖_A < ‖v‖_A for v ≠ 0.

Hence 0 ⪯_A E_N ≺_A I. ∎

### 3.5 One-sided error and the lattice bound

**Cell.** u* satisfies (Ku*)_I = f_I, and e|_Γ = 0. Then

  Π(û) − Π(u*) = ½ eᵀKe ≥ 0.

With f = 0 this reads qᵀŜq − qᵀSq = ‖e‖²_K ≥ 0: Ŝ − S = (Ê − E)ᵀK(Ê − E) ⪰ 0 and μ(q) ≥ 1 for every q, and the
loss ℓ_E equals μ(q) − 1. The error is second order in e and always too stiff.

**Lattice.** Write K̂_L = K_L + Δ with Δ = Σ_i R_iᵀ(Ŝ_i − S_i)R_i ⪰ 0. Then:
- Ĉ ≤ C.
- (K + Δ)⁻¹ = K⁻¹ − K⁻¹ΔK⁻¹ + K⁻¹Δ(K+Δ)⁻¹ΔK⁻¹ ⪰ K⁻¹ − K⁻¹ΔK⁻¹. Therefore

  **0 ≤ (C − Ĉ)/C ≤ Σ_i w_i (μ_i(q_i*) − 1),  with w_i = q_i*ᵀS_iq_i*/C and Σ_i w_i = 1,**

  where q_i* are the **exact** lattice port displacements.
- The gate compliance error is at most the energy-weighted training loss at the directions the lattice actually loads.
  In lattice_v2 the cut cell carries 44–79% of the energy [ev], so μ − 1 ≲ 3–4% in those directions suffices for
  compliance.
- **Loaded cuts.** The same argument applies to the total potential with ĝ. The model lattice is still too stiff.
- **Sensitivity has no sign guarantee.** δs_c = −(2u*ᵀK′_c e + eᵀK′_c e), plus a lattice term from q̂ − q*. It is first
  order in e and weighted by ∂w on surface-crossed elements. Hence ℓ_sens (§5), the internal target of compliance
  ≤ 2% (the sensitivity/compliance error ratio was 1.3–1.6 in route 1 [ev]), and the polish option (§4.4). ∎

### 3.6 Joint O_h equivariance

Transform moments by the signed monomial permutation (DATA_INTERFACE §2.4); do not re-integrate them. Then
K(h·g) = Q_hK(g)Q_hᵀ exactly, and the following objects are equivariant:
- D (blockwise conjugation);
- the masks (permuted);
- W and span(R);
- the nested Q2 grids and the even-start ℓ1 partition. It is symmetric: [2i, 2i+2) ↦ [30−2i, 32−2i);
- P_ℓ, and A_ℓ by Galerkin;
- the window family, and A_Ω ↦ Q_hA_ΩQ_hᵀ;
- the selected G span. The selection rule uses only invariants: λ, θ_s, gaps, the core-mass fraction. B_G depends only
  on the span, not on the eigenvector basis or its sign gauge.

The GNN sees only invariant inputs, aggregates isotropically and uses orbit-class weights, so every gate is permuted
with its entity. Hence B_k(h·g) = Q_hB_k(g)Q_hᵀ for every block, and

  Ê(h·g)(Q_{h,Γ}q) = Q_hÊ(g)q  and  Ŝ(h·g) = Q_{h,Γ}Ŝ(g)Q_{h,Γ}ᵀ.

For cut cells this is joint (g, q) equivariance: the plane moves with h. Residual leaks:
- the Lanczos scalars from random starts, at tolerance level, scaling s₀ only;
- a degenerate eigen-cluster straddling the k ≤ 8 cap. The gap rule prevents this, and it is monitored;
- the polyref Kuhn-split asymmetry (≈ 1e-4) when moments are re-integrated instead of permuted. ∎

### 3.7 The Neumann action is SPD and has an explicit BDD constant

Mode N is the same palindrome with Γ = ∅: floating ℓ3 pseudo-inverse (pinned + projected), and window problems still
Dirichlet at the window boundary, so A_Ω and A_G stay nonsingular. By §3.4 it is symmetric and N^NK has spectrum in
(0, 1] on range(K).

Let M_c = Π_ΓᵀJ_Γᵀ p_d(N^NK)N^N J_ΓΠ_Γ, where p_d is the Chebyshev polynomial that makes t·p_d(t) ∈ [1 − ε_d, 1 + ε_d]
on [λ̂_min(N^NK), 1]. It is positive there, so M_c is SPD on the balanced space.
- If c₁K⁺ ⪯ p_d(N^NK)N^N ⪯ c₂K⁺ on range(K), the congruence with J_Γ and (K⁺)_ΓΓ = S⁺ on the balanced space (checked
  in the systems critique) give c₁S⁺ ⪯ M_c ⪯ c₂S⁺.
- So the BDD condition number grows by at most c₂/c₁ = (1 + ε_d)/(1 − ε_d). d_N is chosen for a ratio ≤ 2, i.e. ≤ √2×
  iterations. Because M_c is used with Ŝ, not S, the upper constant also picks up μ_max ≈ 1 for an operator that passes
  the gate. ∎

### 3.8 Consistency and a monotone depth dial

Every block corrects the exact residual, so x* = A⁻¹b is a fixed point of every block, of N and of N_m. The error after
N_m is (I − p_m(NA)NA)x*, and its A-norm is ≤ max over [λ̂_min, 1] of |1 − t·p_m(t)|. For Chebyshev p_m this decreases
monotonically in m, with the rate set by λ̂_min(NA). This gives fallback A a convergence argument.

The learned cycle polynomial is kept within 0.3 of Chebyshev, and its realized spectral radius is monitored (§5). ∎

---

## 4. Lattice integration (step 3 design; interfaces fixed now)

### 4.1 Forward action Ŝq and its adjoint
- **Per cell, per column:** Ŝq = Πᵀ[v_Γ − K_ΓI N v_I] with v = K[q_d; x] and x = N b(q). Cost ≈ 18 K_eq.
- **The adjoint is the same operator.** Ŝ is symmetric (§3.4), so ŝᵀ-products, the lattice operator and its
  transpose are one code path. Nothing is stored between the two N applications.
- **Batching.** All cells and all load columns form one disjoint union: element, face, window and coarse lists are
  concatenated with per-cell offsets, and each primitive is one kernel launch. Details:
  - The fine level uses compacted, Morton-ordered active element and face lists. The dense 33³×8 slot layout wastes
    2.5–4.3× on FULL cells and 3–19× on cut cells (mechanics-critique m3).
  - Coarse levels use dense masked grids.
  - The per-cell sparse G and ℓ3 solves are batched triangular solves. Planning: ≈ 400 small solves per BDD iteration
    for 100 cells, ≈ 10–20 ms.
- **Box-port gluing** uses global grid keys, as in `lattice_pcg`. Shared corner τ makes neighbouring port sets
  coincide (DATA_INTERFACE §1.3).
- **Cut cells in production:** mode XB. The band is internal, so there are no extra interface unknowns. Loaded cuts
  add ĝ to the lattice right-hand side (§2.6).
- **Mode XP (the full port):** used when a skin attaches, or when the owner makes box ∪ band canonical (O2).

### 4.2 Approximate Neumann action for BDD (question f)
- **Definition.** M_c r = Π_Γᵀ J_Γᵀ p_d(N^NK) N^N J_Γ Π_Γ r (§3.7), with Π_Γ the Euclidean rigid projector on box DOFs
  (the balanced space of Mandel's BDD).
  - Clamped cells, i.e. cells touching a lattice support, use the clamped-face mode instead of the floating one.
  - d_N comes from the encode-time Lanczos interval, for κ(M_cS) ≤ 2. Planning d_N = 3, cost ≈ 28 K_eq.
- **Not trained in step 1** (the systems choice). Step 1 measures BDD iteration counts against exact local Neumann
  solves (X4, §6).
- **If counts exceed 1.5× exact:**
  - raise d_N (+9.3 K_eq per degree);
  - train an ℓ_N head in step 3 on the S-norm error of M_c against exact S⁺ on balanced BDD-like residuals.
- **Rejected:** fixed-count PCG or LOBPCG inside the action (nonlinear in r), and one plain V-cycle
  (κ ≈ 1/λ_min(NA), which inflates BDD iterations by ≈ √κ; all four critiques).

### 4.3 Balanced BDD bookkeeping
- Start every global solve from the coarse solution, so every residual is balanced (Φᵀr = 0).
- Store AΦ from the coarse-space build: ≤ 42 Ŝ columns per cell [ev], ≈ 8 MB fp64 per cell. It is built once per design
  iteration, not per load case (systems-critique m8).
- **Each BDD iteration then needs 1 Ŝq + 1 M_c per cell.** `lattice_pcg`'s hybrid formula needs 3 + 1.
- The true residual is refreshed every 20 iterations.
- Lattice vectors and dot products are fp64.

### 4.4 Sensitivities (question g)

  s_{i,c} = −Σ_e Σ_q (∂w_{e,q}/∂τ_c) ε_q(û_i)ᵀ C ε_q(û_i),  ∂w = V⁻¹ ∂M/∂τ_c,  ∂K_ghost/∂τ = 0 within a topology.

- **Inputs.** ∂M/∂τ_c comes from forward-mode differentiation through polyref: 8 tangents, ≈ one extra moment pass.
  It is nonzero only on elements crossed by the shell surface or the plane, and is held transiently (51 MB per FULL
  cell).
- **Evaluation.** û_i is the model extension of the converged lattice port values q̂_i: 1 forward (8.3 K_eq). Strains
  are rigid-exact, with fp64 accumulation.
- **Lattice gradient:** dC/dτ_v = Σ over cells i incident to vertex v of s_{i,c(i,v)}.
- **Physical vs surrogate.** This is the physical sensitivity evaluated on the model field, not dĈ/dτ. The difference
  is 2(∂x/∂τ)ᵀ(Kû)_I, first order in the interior residual that η² measures. Step 1 reports both against the exact
  dC/dτ.
- **Polish option (owner decision, O3).** Re-extend q̂_i with N_{m_deep} (m_deep = 4, ≈ 36 K_eq, one column per cell)
  before the contraction. This removes most of the field error from the first-order term. The step-1 gate is
  evaluated **without** polish; the polish is reported separately.

### 4.5 Topology changes and design bands
- **Once per band** (δ = 0.02 around the corner τ; `fast_superset` idea), built on the CPU and overlapped with GPU work:
  - superset index arrays;
  - the GNN graph;
  - coarse maps and ghost pattern ids;
  - the G window set (ℓ1 blocks containing weak or cut elements at *either* band end; systems-critique S6a);
  - the frozen counts k_b;
  - the band-correction lists.
- **Per design step:**
  - element-on, face-on and node-on masks select the design's own topology. Nodes outside every active element are
    **excluded from every q-path vector**; there are no superset identity rows in the q-path (systems-critique S6b);
  - recomputed: moments, w, D, the A_ℓ, A₃, the G eigenpairs (warm-started, 3–5 subspace iterations, k_b fixed), A_G
    and its factor, the bounds, and H_θ.
- **Continuity.** Within a band and a topology:
  - w, D and A_ℓ are linear in the moments;
  - the gates are smooth in smooth, clamped features;
  - the G spans are continuous while the kept cluster is separated by a gap (ratio ≥ 1.5, monitored; otherwise the
    whole cluster is kept).

  So Ŝ(τ) is continuous and piecewise smooth. It jumps only where exact K jumps (element or face activation), exactly
  as in the exact pipeline.
- **Box-port set changes** inside a band (0021: ±28–34 nodes at ±1% [ev]) rebuild that cell's gluing and its AΦ
  (42 Ŝq).
- **Band switching uses hysteresis:** switch when a corner τ leaves [τ_c(1 − 0.9δ), τ_c(1 + 0.9δ)].

### 4.6 Memory for 100 resident cells (80 FULL + 20 cut) [est]

| item | per FULL cell | 100 cells |
|---|---|---|
| cell record, modes XB + N (§2.7) | 140–160 MB | ≈ 12–13.5 GB (cut cells 30–110 MB) |
| AΦ (fp64, ≤ 42 columns) | 8 MB | 0.8 GB |
| lattice vectors (fp64, ≈ 10 vectors) | — | ≈ 0.2 GB |
| query working set (4 columns per cell in flight) | 36 MB | 3.6 GB |
| **total** | | **≈ 17–18 GB of 32 GB** |

Comparison: one exact FULL fp32 factor is 6.6 GB [ev].

If memory binds, apply in this order:
1. fp16 storage of the G vectors (−6 MB per mode);
2. a light N mode with no G and a higher d_N (−50 MB per cell);
3. a shared fine-level cell record for cells with identical τ (uniform regions of a design).

### 4.7 Latency per design iteration (100 cells ≈ 89 FULL-equivalents; network part per load case) [est]

| stage | K_eq per FULL cell | time for 100 cells (0.05–0.15 ms per K_eq) |
|---|---|---|
| coarse-space build AΦ (≤ 42 Ŝq, once per design iteration) | 756 | 3.4–10 s |
| BDD iterations (assume 50): 50 × (Ŝq 18 + M_c 28) | 2,300 | 10–31 s |
| sensitivity field: 1 forward (+ polish 36) | 8 (44) | 0.04–0.6 s |
| **network total** | **≈ 3.1k** | **≈ 14–41 s** |
| network encode, warm-started, 2 modes | — | 10–25 s |
| moments + ∂M/∂τ (inherited from decision 1, not network) | — | ≈ 175 s; the dominant term (§7 R12) |

Per design iteration this is ≈ 3.5–4.5 min, dominated by the inherited moment pipeline. For comparison, the exact route
cannot hold 100 cells, and a dense fp64 T would take ≈ 20 min plus 220–450 GB [ev].

---

## 5. Step-1 training protocol (three fixed geometries)

**Geometries and modes.** Step 0 fixes a heavy cut H (0013-like), a medium cut M (0021-like) and their shared FULL
parent F, all from the second-version data. This gives 5 training streams: F-X, M-XB, M-XP, H-XB, H-XP. Mode N is
evaluated (BDD, X4) but not trained.

### 5.1 Exact data (offline; one geometry resident at a time)

**Factors.** Per geometry-mode, one cuDSS factorization of K_II: fp64, or fp32 + 2 refinement steps. Per geometry, one
pinned floating factor: 6 DOF pinned, then rigid projection. The dense K + αRRᵀ is not used (systems-critique m4).

**Pools.** 2,048 (q, f_c, u*) samples per stream, in host RAM. Fields are stored in fp32 and energies in fp64: F 3.0 GB,
M 2 × 1.7 GB, H 2 × 0.55 GB, ≈ 7.6 GB in total. Labels take milliseconds per column once a factor exists. Factors are
not resident during training, except at adversarial refreshes.

**One floating solve feeds both cut modes.**
- **Force-driven samples** come from K u = [f; 0] (pinned, rigid-projected). That single solve gives:
  - **XB:** input (q = u|_box, f_c), target u;
  - **XP:** input q = u|_{box ∪ band}, target u.
- **Other classes** use the X-mode factor.

**Rotations.** 10% of batches apply a random h ∈ O_h by signed permutation of (moments, masks, q, u*). This is an
equivariance *test*: the expected change in loss is ≤ 1e-4. It is not a data multiplier, because the model is
equivariant by construction (mechanics-critique m5).

### 5.2 Sampling (decision 9, per stream)

| class | share | generation |
|---|---|---|
| force-driven | 30% | Smooth multiscale box forces f: ⅓ unit nodal face forces (the gate's load model), ⅓ traction-consistent f_i = ∫N_i t dA, ⅓ random multiscale. ≈ 10% also carry band forces f_c ≠ 0 (loaded cut). |
| macro | 20% | Rigid (exactness check only, excluded from the loss); 6 uniform strains; 18 quadratic; cubic port fields. |
| GRF | 30% | Multiscale Gaussian random fields on port nodes, per component. Correlation lengths run from a whole face down to 2h. |
| adversarial | 20%, from step 2k | Block LOBPCG (block 32, 10 iterations, warm-started, rigid-deflated) on (Ŝ − S)v = λSv, seeded with force-driven q. Refreshed every 1,000 steps per stream, with the factor loaded temporarily (≈ 15 s). The top 16 go into a 256-deep replay bank with exact u*, sampled with random mixing and 10% noise. One unseeded run per refresh gives the reported μ_max. |

**Validation (never trained on):**
- 256 per class per stream;
- the port vectors that lattice_v2's exact solutions induce on the cut cell;
- loaded-cut cases;
- traction-consistent gate loads.

### 5.3 Losses and normalization (fp64 reductions)

Every sample is pre-scaled to ‖u*‖²_K = 1, which is qᵀSq = 1 for f_c = 0. The loss is

  J = mean_i w_i φ(ℓ_E,i) + 0.5·CVaR₀.₉(ℓ_E) + λ_D(t)·mean ℓ_D + λ_s(t)·mean ℓ_sens

| term | definition | role |
|---|---|---|
| ℓ_E | ‖û − u*‖²_K / ‖u*‖²_K, in the fp64 difference form with rigid-exact strains. It equals μ(q) − 1 for f_c = 0, and 2(Π(û) − Π(u*)) for loaded cuts. | decision 7; the gate quantity (§3.5) |
| w_i | 2 for force-driven and adversarial samples, 1 otherwise; rigid samples excluded (0/0) | soft-decile emphasis |
| φ | log(1 + ℓ) for the first 1k steps, then ℓ | tames large errors at X0 |
| CVaR₀.₉ | mean of the top 10% of ℓ_E per class in the batch | averages hide the worst direction (history, phase 1) |
| ℓ_D | ‖û − u*‖²_D / ‖u*_d‖²_D, D = point-block diagonal of K. λ_D: 1 → 0.05 linearly over the first 30% of steps. | F4: the energy Hessian is K_II; D-weighting gives soft components gradient early |
| ℓ_sens | ‖g(û) − g(u*)‖² / ‖g(u*)‖², with g ∈ ℝ⁸ the ∂w-weighted corner sensitivities. Force-driven and adversarial samples only. λ_s: 0 until step 2k, then a ramp to 0.3 by 5k. Needs ∂w for the 3 geometries (51 MB for F). | sensitivity is the binding gate [ev]; ∂w weighting, not volume weighting (systems-critique S4) |

There are no entry or Frobenius terms. For step 2, the label-free gradient ∇ log(ûᵀKû) needs only the scalar qᵀSq
(systems-critique m3).

### 5.4 Optimizer and schedule

**Optimizer.**
- AdamW, β = (0.9, 0.99).
- Learning rate: 3e-4 for the GNN trunk and heads; 1e-3 for the global polynomial heads.
- Weight decay 1e-5, none on head biases.
- 500 warm-up steps, then cosine decay to 1e-5 over 20k steps.
- Gradient-norm clip 1.0.
- fp32 parameters, TF32 off.

**Batches.**
- One stream per step, B = 32 columns.
- Streams are **interleaved every step** in round-robin order (multilevel-critique m9).

**Gradient paths.**
- Gradients reach θ only through the gates and polynomial coefficients. These multiply fixed exact matrices.
- There is **no** gradient through eigensolves, Lanczos bounds, Galerkin products or factorizations. They are
  geometry-only and θ-independent: the Loewner caps make the bounds gate-independent.
- This removes the riskiest gradient path of the proposals (learned moduli through a Cholesky) and the stop-gradient
  mismatch (op-learning-critique S9).

**Cost** [est].
- One F step: (9.3 + 2 × 9.3) K_eq × 32 ≈ 0.9 TFLOP ≈ 0.05–0.15 s. Cut-cell steps are cheaper.
- 20k steps ≈ 0.5–1 GPU-hour, plus ≈ 25 min of adversarial refreshes per run.

### 5.5 Monitoring (every 250 steps; the lattice gate every 1k)

1. μ − 1 quantiles (50/90/99/max) per class × stream on the validation sets, plus the softest-decile energy error.
2. **The lattice gate.**
   - lattice_v2 x and y configurations: cut cell with its FULL parent as neighbour, 6 load cases. The model cut cell is
     in XB.
   - Measured: compliance error and cut-cell sensitivity error against exact.
   - Also: a both-cells-model variant; a 2×2×2 all-model FULL lattice; loaded-cut and XP reports; traction-consistent
     loads.
3. μ_max from unseeded LOBPCG. Reported, not gated.
4. **Stage-wise error energies** ‖x* − x_k‖²_A after each of the 5 blocks, per class. This shows which block leaves
   which error.
5. Error localization: eᵀKe by node class (weak, strong, port-adjacent, band, outside material) and inside vs outside the
   G windows.
6. Spectra: λ̂_min(NA), n_G, G gap ratios, the fraction of blocks at the k_b cap.
7. Gate saturation histograms; activity of the polynomial barrier.
8. **Invariant tests. Any failure aborts the run.**
   - linearity;
   - bitwise ports;
   - rigid residual ≤ 1e-12;
   - symmetry defect |xᵀŜy − yᵀŜx|/√(xᵀŜx·yᵀŜy) ≤ 1e-6 on LOBPCG-soft pairs;
   - equivariance ≤ 1e-6 with permuted moments;
   - an fp32 vs fp64 shadow readout on 8 columns.
9. Sensitivity error by element class (full, cut, sliver); physical vs dĈ/dτ; finite differences in τ within a band
   (H).
10. Calibration of η² against ‖e‖²_A.
11. κ(M_cS) by Lanczos, and BDD iteration counts on a 2×2×2 FULL lattice (every 2k steps).
12. Timing per primitive against §2.7; peak memory.

### 5.6 Stopping

- **Success:** the gate passes with the internal margin (compliance ≤ 2%, sensitivity ≤ 3%, all configurations) at 3
  consecutive evaluations.
- **Plateau:** force-driven p90 μ − 1 and the gate errors improve by < 5% over 4k steps. The run's result then enters
  the §6 rules as is.
- **Budget:** 20k steps.
- **Abort:**
  - an invariant test fails (a bug);
  - a non-finite loss;
  - more than 50% of gates saturated for 2k steps. This is logged as "capacity missing" and goes to the §6 ladder.

---

## 6. Minimal step-1 experiment plan and pre-registered decision rules

**The gate, as used below.**
- Lattice compliance error ≤ 3% **and** sensitivity error ≤ 3% against exact.
- On every lattice_v2 configuration: x and y, each with 6 load cases.
- The model cut cell runs in mode XB (free cut), without polish.
- The FULL cell is checked in addition through the 2×2×2 all-model lattice.
- μ_max, loaded-cut cases and mode XP are reported and never gated.

### 6.1 E0: build checks and zero-learning measurements (≈ 1 day, no training)

| id | what | pass criterion [pre-reg] |
|---|---|---|
| E0a | Unit tests on all 5 streams: linearity; bitwise ports; rigid exactness; symmetry defect; equivariance with permuted moments; the adjoint formula of §2.6 against an autograd VJP of the forward | all to rounding (§5.5 item 8). A failure is a bug. |
| E0b | Microbenchmark: strain-form K + ghost (FULL/M/H; B = 1, 8, 32; fp32, TF32 off); A₁ matrix-free; batched G and ℓ3 solves; encode timing; G factor size | K_eq ≤ 0.15 ms per column at B = 32 is the target; ≤ 0.3 ms is tolerated. Encode ≤ 0.6 s (XB + N, FULL, cold). G factor ≤ 64 MB per mode. |
| E0c | Exactness of the coarse operators: coarse-moment A₁ vs direct fine-element Galerkin; F·P_s = 0 on interior ghost sub-faces | ≤ 1e-12 relative. Otherwise keep direct Galerkin. |
| E0d | **Soft-band coverage.** Take the 128 slowest eigenvectors of the zero-learning propagator *without* G (palindrome F-M-F). Measure the fraction of their A-energy inside span(P_G). Streams H-XB and M-XB. | ≥ 90% |
| E0e | **X0 curves.** The zero-parameter SGNO (all heads at init) with and without G, m ∈ {1, 2, 4, 8}. Record: μ − 1 quantiles per class; the gate; stage-wise error energies; λ̂_min(NA); preliminary BDD counts with M_c. | Diagnostic. Feeds rules R1–R7. |

### 6.2 Baselines, main arm, ablations

**Baselines.**
- **B0a, trivial lift.** The 12-column W-weighted rigid + affine fit, extended to the interior as the affine field,
  with no correction. It gives the μ reference and checks the harness.
- **B0b = X0, zero-parameter SGNO.** The numerical prior: every learned coefficient at its theory value. This is the
  zero-parameter lifting baseline. Under R6 it is also fallback B's starting point.
- **B0c, external calibration.** The logged route-1 results: learned-24 gives 0.72/1.7% compliance and 0.93/2.6%
  sensitivity; Chebyshev-64 gives 0.04/0.07% [ev]. Optionally re-run Chebyshev-64 in the new harness as a positive
  control.
- **B1 = X1, capacity upper bound.** Every gate and polynomial coefficient is a free parameter per stream, with no GNN,
  the same loss and m ∈ {1, 2}. It is the representational ceiling of SGNO at that depth. The structural ceiling is X0
  at m = 16.

**Main arm X2.** GNN-generated coefficients, shared across the 5 streams.
- It is trained with **random depth** m ∈ {1, 2, 4} (60/25/15%), with shared gates and a cycle polynomial per m. The
  depth dial is therefore trained, not extrapolated (op-learning-critique S1).
- It is evaluated at m = 1, 2 and 4.

**Ablations.** One component each. Run against X2 on H and F with the same budget.

| id | change | isolates |
|---|---|---|
| A-G | G block removed (palindrome F-M-F) | the soft-band carrier |
| A-FF | pure feedforward, K only in the readout: N_FF = η_F S₀ + η_G B_G + η_M B_M, with M at C₁ = C₂ = 4 and degree 3 (equal wall-clock) | the exact inner residuals. This arm is the owner's literal main line. |
| A-Mch | M with 1 channel and spatial gates frozen at 1 (polynomials still learned) | the learned multiscale-GNO capacity |
| A-smp | macro + GRF samples only (same count); no force-driven or adversarial samples | the sampling claim (probes 0.3% vs force-driven 20–35% [ev]) |
| A-sens | λ_s = 0 | ℓ_sens |

**Reports, not ablations:**
- XB vs XP in the gate;
- X4: BDD iterations with M_c on a 2×2×2 FULL lattice and on 2×1×1 cut + FULL, against exact local Neumann solves;
- optional X5: train on F + M, test on H (an early generalization read).

**Compute.** ≈ 3–4 GPU-days in total, including factorizations and adversarial refreshes.

### 6.3 Decision rules [pre-reg]

Apply the rules in order. Every rule is judged on the gate defined above.

| rule | condition | decision |
|---|---|---|
| R0 | E0a fails, **or** E0d < 90% | Fix bugs first. For coverage: θ_s ×2, k ≤ 12, 2-element halo; re-test once. If still < 90%, the soft band is not local at block scale: stop and report the measured mode supports to the owner before training. (E0b failures do not block the 3-geometry step 1; they open R5-type systems work.) |
| R1 **GO** | X2 passes at m = 1 | Main line confirmed. Proceed to step 2 with SGNO at m = 1. |
| R2 **GO-with-depth** | X2 fails at m = 1 and passes at m = 2 | Main line at m = 2 (Ŝq ≈ 36 K_eq ≈ 2–5 ms). Report under O1 and proceed. |
| R3 **FIX, generator** | X1 passes at m ≤ 2 and X2 does not | Generator gap: widen or deepen H_θ, and add the features that the error localization points to. At most 2 fix iterations; then R4. |
| R4 **FIX, capacity** | X1 fails at m ≤ 2, and X0 or X1 passes at some m ≤ 4 | One capacity-ladder step (K1 and/or K2 below), then re-run X1 and X2. If the gate then needs m ≤ 2, go to R1/R2; otherwise R5. |
| R5 **FALLBACK A** | the gate needs m ≥ 4 even with learned heads | Adopt fallback A: m = 4–8 with per-cycle learned heads, same code. Budget from §2.7: Ŝq 73–147 K_eq ≈ 4–22 ms. |
| R6 **FALLBACK B** | X1 and X2 improve force-driven p90 μ − 1 over X0 by < 1.5× at equal m, **and** X0 passes the gate at some m ≤ 8 | Learning does not pay. Freeze the heads and learn ≈ 30 global scalars label-free, at the smallest m where X0 passes. |
| R7 **structural failure** | X0 at m = 8 fails the gate **and** force-driven p90 μ − 1 > 10% at m = 8 | The carriers are wrong; this is not a learning problem. Use the stage-wise energies and the error localization to name the failing block, and redesign it. Route-1 learned-24 remains the known-good reference. |
| R8 **pure FF** | A-FF passes the gate at ≤ X2's wall-clock time | Adopt the pure-feedforward form (inner K removed). This overrides R1–R2. |
| R9 components | at the operating m | **G:** drop it if A-G is within 20% of X2 on force-driven p90 μ − 1 and on the gate. **M channels:** go to 1 channel if A-Mch is within 10%. **ℓ_sens:** keep it if A-sens has ≥ 1.2× X2's sensitivity error. If A-smp passes the gate, report that the sampling claim is refuted; the decision-9 mix stays (owner decision). |
| R10 Neumann | X4 | BDD iterations ≤ 1.5× the exact-S⁺ count: accept. Otherwise raise d_N. If d_N > 6 is needed, train ℓ_N in step 3. |

**Capacity ladder** (used only by R4, in this order):
- **K1.** A fine learned hyperedge channel. The PSD element templates are all built from the element's own 125 Gauss
  weights: cut mass, cut Laplacian, K̂_e. They are combined with generated non-negative coefficients. Symmetric,
  equivariant, ≈ 0.4 K_eq per use.
- **K2.** Learned enrichment of G. For each window, 1–2 vectors from a local dictionary: spare eigenmodes plus
  Jacobi-smoothed ℓ1 functions, with coefficients from the GNN. Implicit gradients through the A_G factor; the factor
  is recomputed each step (≈ 20 ms).
- **K3.** Learned rigid-preserving transfer smoothing at ℓ1 → ℓ2, with a Galerkin recompute. Last in the ladder,
  because of the route-1 E3 evidence [ev].
- **K4.** More depth (m).

---

## 7. Ranked risks

| rank | risk | how it is detected (cheapest first) | mitigation / degradation |
|---|---|---|---|
| R1 | **One pass is not accurate enough on force-driven directions**; the gate needs m > 1. Evidence: every depth claim in the four proposals was refuted by the route-1 contraction data [ev]. | E0e X0 curves vs m; stage-wise error energies; λ̂_min(NA) with and without G. Hours, no training. | The depth dial is priced (§2.7). Rules R2/R5. If G does not lift λ_min: θ_s ↑, k ≤ 12, 2-element halo. |
| R2 | **The soft band is not local at block scale:** modes wider than the oversampled window, or sheet-scale fringe motion that G cannot represent | E0d coverage (≥ 90% of the 128 slowest modes' energy in span(G)); support diameter of the slow Ritz vectors | Larger halo; add a second window family (odd-start blocks, also O_h-symmetric); fold ℓ1 fringe coarse functions into G. Otherwise R7. |
| R3 | **G is too big or too slow:** n_G, fill, eigensolve time | E0b: n_G, factor MB, encode s | Gap-based θ_s, a higher core-mass threshold, fp16 vectors, warm-started subspace iteration, blocks restricted to weak/cut regions |
| R4 | **Learning adds little beyond the numerical prior**, because gates and polynomials cannot add directions (all four critiques) | X1 vs X0 at equal m | Capacity ladder K1–K2 (R4); otherwise fallback B (R6). Honest outcome: a strong numerical prior with learned step shaping. |
| R5 | **Sensitivity is the binding gate:** first-order error on surface and sliver elements, where ∂w/w is large (route-1 ratio 1.3–1.6 [ev]) | Gate monitor; sensitivity error by element class; A-sens | ℓ_sens; internal compliance target ≤ 2%; polish option (O3); K2 enrichment near surface elements |
| R6 | **Throughput:** the 0.05–0.15 ms/K_eq assumption fails (atomics, ragged batches). Route 1 measured ≈ 2.5 ms per layer per column [ev]. | E0b microbenchmark, before training | Colour elements to avoid atomics; column-innermost layout; compacted lists; fewer channels. Memory remains the decisive advantage (6.6 GB → 0.15 GB). |
| R7 | **The Neumann action is too weak** (BDD iterations inflate) | X4 counts vs exact S⁺ | Raise d_N; ℓ_N training in step 3; add G to the N mode if it was dropped |
| R8 | **Mode XP / band conditioning:** the oblique masked coarse space (energy grows ≈ H/h near the band) and the sliver-heavy band | XB vs XP gate and μ reports; two-grid contraction per mode | XB is the production mode for free cuts. For XP, use a band-local smoothed prolongation (systems-critique S5). |
| R9 | **Topology and band continuity:** mode counts or window sets change inside a band; box-port changes | τ ± 1e-4…1e-2 sweeps on M (topo_change style): ‖ΔŜ‖ vs ‖ΔS‖, and finite-difference sensitivities | k_b frozen per band; gap rule; hysteresis; band-robust window sets |
| R10 | **Precision:** fp32 residuals in weak regions; coarse factors; symmetry defect of the adjoint term K_ΓI N v_I | fp64 shadow forward and readout; symmetry defect on soft pairs | fp64 for the last inter-block residual; fp64 A_G factor (+30 MB) |
| R11 | **Implementation slips** in equivariance or masks | E0a tests on the 15 rotated cases [ev] | Tests gate every run |
| R12 | **The design-iteration time is dominated by inherited moments and ∂M/∂τ** (≈ 175 s per 100 cells), not by the network | Timing in the step-3 harness | Outside the architecture: compute ∂M only on surface-crossed elements, reuse fully-in-material closed forms, overlap with GPU work. Flagged to the owner (O6). |
| R13 | **The gate's load model:** unit nodal forces on every box-port node, including fictitious fringe port nodes, put energy into physically meaningless modes (op-learning observation) | lattice_v2 with exact operators: energy on weak vs strong nodes, unit-nodal vs traction-consistent loads (minutes) | Report both load models. The gate is never softened (O4). |

---

## 8. Open items that need the owner's decision

| id | question | recommendation |
|---|---|---|
| **O1** | **Is a fixed-depth operator with 4 exact inter-block residuals (≈ 8 K_eq per forward) acceptable as the "feedforward + multiscale GNO" main line (question d)?** And is "main line at depth m = 2" (R2) still the main line? | Yes. The cancellation argument (R13) says learned kernels alone would need 1–2% accuracy on a 20–64× lift energy. The pure-FF arm A-FF is run anyway, and R8 adopts it if it passes. |
| O2 | Cut-band port contract. XP (Ŝ on box ∪ band) is decision 4's canonical form. XB (band free, nodal-force input) is what the lattice uses for free cuts, and adds no interface unknowns. Both run on the same record and both are trained. Which one is canonical for the step-1 report? | Gate on XB; report XP. Make XP canonical once skins attach. |
| O3 | May the deployed method compute sensitivities from a polished field (N_{m_deep}, one column per cell per design iteration, ≈ 36 K_eq)? | Allow it as a deployment knob. The step-1 gate is judged without it and the polish is reported separately. |
| O4 | The gate's load model is unit nodal forces on all box-port nodes, including fringe or fictitious port nodes. Keep it, or add or replace it with traction-consistent loads? | Keep it as the gate. Report traction-consistent loads next to it. |
| O5 | Confirm the budgets: ≈ 140–160 MB per FULL cell (XB + N); ≈ 18 GB for 100 cells including working sets; encode 0.3–0.6 s cold and 0.1–0.25 s warm for 2 modes; Ŝq ≈ 1–3 ms. | These are the planning targets used by the rules above. |
| O6 | Moments and ∂M/∂τ (≈ 175 s per 100 cells per design iteration) dominate the design-iteration time. Should a separate pipeline task (surface-only ∂M, closed forms) be started now? | Yes, in parallel with step 1. It is outside the network. |
| O7 | Step-1 geometry pair: which heavy and medium cut share one FULL parent in the second-version data? 0013/0021 are used here as stand-ins. | Fix it in step 0. |
| O8 | Deployment: per-cell adaptive depth m driven by η² (80% of the cells in a design are FULL and probably need m = 1)? | Defer to step 3. |

---

## Appendix A. Critique disposition: what was accepted, modified or rejected

### A.1 What each proposal contributes to SGNO

| proposal | taken into SGNO |
|---|---|
| multilevel (LCMO) | Error-product design with exact residuals; ports by truncation or masking; Q2 h-hierarchy; exact coarsest solve; hyperedge GNN with orbit-class aggregation; numerical-prior initialization and the X0/X1/X2 ladder; stage-wise error energies; (Kû)_Γ − Ŝq as a diagnostic |
| operator learning (HGM-NO) | Hard q-path rules with a linearity unit test; strain-form K, TF32 off, fp64 accumulation; W-weighted rigid split; mixed box-Dirichlet / cut-force port (mode XB); free-coefficient oracle; random-depth training; sensitivity-error reading via Galerkin orthogonality |
| mechanics (MMX-Net) | Port-force entry b = −K_IΓq_d; the lattice compliance theorem (§3.5); "K is never gated"; mechanism-type soft space, rebuilt; error localization by node class; force-driven data from one floating solve |
| systems (GLMO) | Self-adjoint symmetric N and its readout formula; mode masks on one record; coarse-moment Galerkin; rigid-exact mixed precision; Neumann action via (K⁺)_ΓΓ = S⁺; balanced BDD with stored AΦ; the η² indicator; zero-initialized heads around theory values |

### A.2 Accepted (fixed in v0)

Every fatal and serious point below is accepted. The fix is given with its location in the spec.

- **Multilevel critique.**
  - F1: the ℓ2 split is not nested. G is now a separate exact stage with fine residuals around it.
  - F2: the T = 1 claim is invalid. No one-pass claim is made; the soft band gets an exact subspace solve; m is
    measured.
  - S1: the Neumann action is too weak. Chebyshev wrapper (§4.2).
  - S2: this is a hybrid. Disclosed (O1), and the A-FF arm is run.
  - S3: fp32 factors. fp64 ℓ3, and A_G with fp64 refinement.
  - S5: P̃ (learned transfer smoothing). Dropped.
  - S7: pair ties. Point blocks.
  - S8: build order. Direct fine-element Galerkin first.
  - Minors 1, 3–5, 7–11 accepted.
- **Operator-learning critique.**
  - F1a: principal submatrix. Accepted (A_Ω).
  - F1c: projection form. Accepted, as an exact Galerkin solve.
  - S1: honest depth, random-depth training. Accepted.
  - S2: the tail is inert. Chebyshev tail on [λ̂_min, 1]; η² labelled an estimate; route 2 is the rigorous bound.
  - S3: decision 8. O1 and A-FF.
  - S4: learned coarse moduli θ. Dropped.
  - S5: coarse SPD not certified. Loewner caps and a barrier per level.
  - S6: throughput. E0b comes first; compacted lists.
  - S7: Neumann action. Chebyshev.
  - S8: band semantics. Both modes trained; O2.
  - S9: stop-gradient caps. Gate-independent Loewner bounds.
  - Minors 2, 3, 5, 7, 9 accepted.
- **Mechanics critique.**
  - F1: depth. Priced at all m (§2.7).
  - F2: GenEO without an exact coarse solve. The G solve is exact and joint.
  - S1: Neumann action. Accepted.
  - S2: eigensolve cost. LOBPCG with k ≤ 8, chunked, warm-started.
  - S3: Neumann patch = route 2's failure. Principal oversampled windows.
  - S4: fixed k_max. Gap selection, frozen per band.
  - S5: sensitivity is binding. ℓ_sens, a 2% internal target, polish (O3).
  - S6: do not gate Ψ. K is never gated.
  - S8: affine L3. The ℓ3 level is Q2.
  - S9: no Krylov iteration inside the cell operator.
  - S10: budget items. AΦ and per-mode records included.
  - Minors 1, 3–7, 9–12 accepted.
- **Systems critique.**
  - F1: m = 2 claim. Replaced by measurement; priced.
  - F2: decision 8. O1 and A-FF.
  - S1: additive patches. Replaced by G.
  - S3: Loewner monotonicity. Adopted.
  - S4: ∂w-weighted ℓ_sens. Adopted.
  - S5: band masking including ghost faces. Adopted.
  - S6: topology hygiene. Adopted.
  - S7: storage. Recounted.
  - Minors 1–10 accepted.

### A.3 Rejected or modified, with reasons

| # | critique point | disposition | reason |
|---|---|---|---|
| A1 | Op-learning F1b: "3×3×3-element windows at stride 2" | **rejected**; replaced by even-start ℓ1 blocks + a 1-element halo, with centre assignment | On a 32-element grid, a width-3 window [s, s+3) reflects to [29 − s, 32 − s): the start parity flips, so no stride-2 family of odd-width windows is O_h-symmetric. This would break R6. The replacement is symmetric (s ↦ 30 − s) and avoids duplicates by centre assignment. Its exact containment is also limited (node extent ≤ 4 always, 5 in 3/4 of positions, 6 in 1/2; §2.4 E2). Instead of relying on containment, it relies on the joint exact solve to glue neighbouring windows, and E0d tests that. |
| A2 | Multilevel S4 and systems S1: "multiplicative 8-colour patch sweeps" | **rejected** as the carrier | Exact for modes inside a window, but still unable to glue sheet-scale modes (weak nodes percolate into 2–5 giant sheets [ev]). Cost ≈ 3.4 partial K per sweep (≈ 9 K_eq for the 4³ windows needed for coverage). One exact Galerkin solve over the local soft modes covers all three failure modes at ≈ 0.05 K_eq per application and ≈ 50 MB. |
| A3 | Multilevel F2 fix (ii-b): energy-minimizing MsFEM ℓ1 basis | **deferred** (not in v0) | ≈ 55 MB and 36 GFLOP per cell, plus new machinery. G captures the local soft content that locked polynomial spaces miss (bending of thin wall pieces is a low-λ mode of (A_Ω, D_Ω)). Revisit only if the E0e stage energies show residual ℓ1-scale bending error. |
| A4 | Mechanics S7 and systems F1 fix 2: learned prolongation smoothing as *the* learned channel | **rejected for v0** (ladder K3) | Route-1 E3: one smoothing step doubled the per-layer cost for a 32-layer witness gain of only 7.07 → 5.59 [ev]. It also breaks exact Galerkin unless A_ℓ is recomputed and differentiated through a factorization. |
| A5 | Mechanics F1: "realistic P ≈ 8–16; price the design there" | **modified** | The arithmetic is accepted, and every m is priced. The extrapolation is rejected: route 1's λ_min comes from a carrier without an exact soft-space solve, and G changes exactly that quantity. E0e measures it before anything else. |
| A6 | Systems F2: "present GLMO as the chassis and put a K-free learned core in it as *the* main-line candidate" | **modified** | The diagnosis is accepted: the q-path is a hybrid. But the main arm keeps the exact inter-block residuals, on the precision grounds of R13. The K-free core is run as arm A-FF, and R8 promotes it if it passes. |
| A7 | Op-learning S3: "a feedforward inverse needs only ≈ 10% accuracy on soft amplitudes" | **partially accepted** | True for the soft directions. But the zero-lift residual carries 20–64× the solution energy in stiff and boundary-layer content, which needs 1–2% relative accuracy without exact smoothing. So the arm is tested (A-FF) rather than assumed. |
| A8 | Mechanics S5 fix 1: "compute sensitivity only from a deep polish" | **modified** | Kept as a deployment option (O3). The step-1 gate is judged on the main-arm field, so the architecture is not credited for depth it does not have by default. |
| A9 | Multilevel minor 3: "weak nodes forming sheets is a tautology" | **accepted** | Not used as mechanical evidence. G selects modes by eigenvalue, not by a weak-node threshold, so the q-path uses no weak mask at all. |
| A10 | Op-learning S4 (θ moduli) and multilevel S6 (G_θ templates) | **accepted and dropped** | K1 in the capacity ladder reintroduces a fine learned kernel with moment-weighted templates if R4 fires. |

