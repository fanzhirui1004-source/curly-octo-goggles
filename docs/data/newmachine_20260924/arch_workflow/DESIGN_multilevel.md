# v0 architecture from the numerical-analysis / multilevel lens: the Lift-and-Correct Multilevel Operator (LCMO)

Author lens: numerical analysis and multilevel methods. I treat u = E q as the discrete elastic (harmonic) extension and
derive the structure that a cheap, fixed-depth, q-linear approximation of E must have. The result is a complete v0
architecture, not only its multilevel part.

Inputs read: ARCH_BRIEF.md (binding), RESEARCH.md, DATA_INTERFACE.md, docs/ROUTES_PROGRESS, ARCHITECTURE_HISTORY,
OPERATOR_LEARNING_DESIGN, and the route-1 code (`xcase/r1/encode_r1.py`, `hierarchy_strain_network_r1.py`).
Two small new computations are in `arch/` and are cited as [new]:
- `q2_hierarchy_counts.py` produces `q2_hierarchy_counts.json`: sizes of the Q2 h-coarsening hierarchy and of the port-touching element sets.
- `weak_clusters.py` produces `weak_clusters.json`: the connectivity of weak nodes. It uses the same body-only proxy as `weak_proxy.py`.

Labels: **[est]** marks an estimate that has not been run on the real pipeline. **[new]** marks a number from the two scripts
above, computed for uniform tau = 0.5 with the rules calibrated in DATA_INTERFACE §0.

---

## 0. Summary in ten lines

1. Write the extension as `E = (I - Z K_II^{-1} Z^T K) L`, a lift of the port data followed by an interior correction. An approximation `Ê = (I - Z B Z^T K) L` has
   energy error `μ(q) - 1 = ||(I - B K_II) w*||²_K / ||u*||²_K`. The whole design problem is therefore: **build a
   geometry-generated linear approximate inverse B whose one-pass error operator (I - B K_II) is small on the w* that
   force-driven loads produce.**
2. Ports are exact by construction, because B only writes interior DOFs. The rigid modes are exact through an fp64 rigid split. Ŝ = Êᵀ K Ê is
   symmetric PSD and one-sided (Ŝ ≥ S) for any value of the learned parameters.
3. B is a fixed five-stage DAG: rigid split → lift residual → FMG coarse-to-fine pass → fine sliver+bulk smoothing → one
   V-cycle → fine post-smoothing. Between stages the residual is recomputed with the **exact K** (3 full + ~1.1 partial fine K
   applications). A stage error operator multiplies the errors left by the previous stages; it does not add to them.
4. The hierarchy is **Q2 h-coarsening on the fixed grid, 65³ → 33³ → 17³ → 9³ nodes**. It is not Q1 p-coarsening, because Q2 keeps the
   through-thickness bending of thin walls. The coarse operators are **exact truncated Galerkin operators** formed in the
   encoder from aggregated element moments and masks, and the coarsest (≤ 2,187 DOF) is solved by a dense fp64 Cholesky factor. Box and cut-band ports
   are handled in the same way, by truncating the coarse functions on the port mask.
5. The soft-mode carrier is the **weak fringe**. [new] The weak nodes are not isolated slivers: they form a connected sheet on each side of
   the shell (2 components in FULL, 5 in the heavy cut). They are handled by (i) overlapping grid-aligned fringe patches
   with exact local solves and learned, undamped gates at the fine level, and (ii) a **strong/fringe split of the 17³
   coarse functions**. The split gives the fringe its own coarse DOFs (a multi-continuum coarse space in the sense of GenEO/CEM).
6. The learned part (~0.16 M weights) is a hyperedge GNN on K's own hypergraph (elements and ghost-penalty faces) with O_h-invariant
   inputs. It emits ~0.2 M coefficients per FULL cell: per-block and per-patch damping, element-kernel scales, coarse
   smoother and transfer gates, stage step lengths. **At initialization these coefficients reproduce a sound numerical method**, and training
   only improves it.
7. The same network, with a different port mask, gives mode F (box ports, free or loaded cut: the lattice gate), mode D
   (box ∪ cut-band Dirichlet: decision 4) and mode N (floating cell). Mode N is an additive multilevel operator that is SPD by construction, for the BDD S⁺
   action.
8. Cost for FULL per column [est]: ~5 GFLOP forward, ~11.5 GFLOP for the symmetric reaction Ŝq, ~0.6 GFLOP for the Neumann
   action. Resident memory is ~75–110 MB per FULL cell. Encoding is ~0.1–0.2 s on top of the existing moment computation.
9. Main risk: one V-cycle (T = 1) may not be enough on the dense soft band. The knob T (the number of correction cycles)
   turns the design continuously into fallback A. Freezing the gates at their numerical priors turns it into fallback B.
10. Step 1 order: X0 (numerical prior, no learning), X1 (free per-geometry coefficient fields: the capacity ceiling), X2 (GNN-generated coefficients), plus
    a small set of ablations, each tied to one risk.

---

## 1. Requirements (derived; each tied to evidence in the brief or the docs)

| # | Requirement | Evidence |
|---|---|---|
| R1 | Output exactly linear in (q, f); every nonlinearity is geometry-only. | Decision 5. RESEARCH F2: q-linearity plus the exact-K readout makes Ŝ SPSD; a q-nonlinear field regressor gave an indefinite Hessian (arXiv:2608.02036). |
| R2 | Port values exact on box ∪ cut band, bit for bit. | Brief §4: one-sided, second-order readout error requires exact port values (Galerkin orthogonality). |
| R3 | Rigid modes exact **in floating point**: Ê R_p = R, Ŝ R_p = 0. | ROUTES 7.6: an unprojected 1e-7 rigid residual gave a 12× energy error in a 2-cell lattice. |
| R4 | Accuracy concentrated on the soft decile: μ-1 ≲ 1% and field error of a few % on force-driven q. | Brief §4: 94–96% of lattice load energy lies in the softest 10%. Sensitivity is first order in field error. The skeleton had 0.3% error on smooth probes and 20–35% on force-driven loads. |
| R5 | An explicit carrier for localized low-energy error on weak nodes, effective **in one pass** (no damping that throttles it). | ROUTES "慢模态在哪里": 17.6–21.9% of nodes are weak and carry 94–99.4% of the slow-mode amplitude; each mode covers 11–30 nodes; the 128 slowest modes lie in [0.0039, 0.0062]. E3: smoother strength was the bottleneck. Merged weak groups helped only under a global damping. [new] Weak nodes form connected fringe sheets. |
| R6 | Global transmission inside one forward pass, with coarse fields that can bend thin walls. | Brief §4: global transmission. E0: quadratic block coarse spaces beat affine ones (64-layer witness 1.26 vs 7.75). |
| R7 | Fine-level coupling no wider than K's hypergraph. Coarse levels must not merge material that is disconnected. | RESEARCH F6. DATA_INTERFACE §2.2. |
| R8 | Cost ≈ a few ms per cell-application amortized; ≲ 100 MB resident per FULL cell; encode ≲ 0.5 s beyond moments; no fine factorization. | Brief §6. ROUTES 7.6: exact factors do not fit 100 cells. |
| R9 | An SPD approximate Neumann action (S⁺) for BDD, built from the same components. | Brief §5: BDD takes 11–53 iterations with an S⁺ action, 300–1600 with forward actions only. RESEARCH 3.3 / 5.10: the preconditioner needs a matched basis. |
| R10 | Symmetric reaction for lattice solves. | Question e. Parish et al.: SPSD is required in coupled solves. |
| R11 | fp32 network; fp64 for rigid projection, energy reductions and the coarsest factor. | ROUTES 7.6: fp32 matrices err 0.8–2.2%. For fields with a variational readout, errors enter at second order. |
| R12 | Fixed grid; topology enters as masks; discrete structures frozen within a design band so that sensitivities are smooth. | Decision 3. A 1e-3 change in tau changes ±4–6 active elements. DATA_INTERFACE §1.6 (superset). |
| R13 | The cut band is a port of two types on the same mask: displacement (mode D) or nodal force (mode F; free cut = f_c = 0). | Decision 4 ("forces on the cut band as nodal forces"). DATA_INTERFACE §2.3: a Dirichlet band adds ~31k private unknowns per cut cell to BDD. The step-1 lattice harness treats the cut as free. |
| R14 | Trainable: smooth parameterization; start from an operator that already works; loss = normalized energy error. | HISTORY: a non-smooth head broke local-factor learning; entry losses fight the soft directions. F4: the energy Hessian with respect to the output is K_II. Route 1: 2 learned scalars per layer passed the gate. |
| R15 | Coefficients generated from local, O_h-invariant mechanical features (index-free), so that step 2 generalizes. | EquiModel: index-free, augmented, generalized on FULL cells. Rotation augmentation helped held-out cut cells. |
| R16 | Graceful degradation into fallback A and B with the same components. | Decision 8. |

---

## 2. Architecture

### 2.1 Derivation: what the structure must be

Notation:
- Nodes are split into ports p (mode-dependent mask) and interior I. `Z` injects interior DOFs into the full vector.
- `L0` is the zero extension: q on the ports, 0 elsewhere.
- `R` (3N×6) holds the rigid fields on all active nodes; `R_p` is its restriction to the ports.
- Exact extension: `u* = R α + L0 q_d + Z w*` with `w* = K_II^{-1} Z^T (f - K L0 q_d)`, where `q_d = q - R_p α`.

**Error identity.** Replace K_II^{-1} by any linear B. Then `ŵ - w* = -(I - B K_II) w*`. Because `Z^T K u* = Z^T f`, Galerkin orthogonality gives

  `û^T K û - u*^T K u* = ||(I - B K_II) w*||²_{K_II}` (for f = 0), so `μ(q) - 1 = ||(I - B K_II) w*||²_K / ||u*||²_K`.

Three design consequences follow.

(i) **Use exact residuals between stages.** Suppose B is a sequence of corrections `w ← w + C_k (r - K_II w)` with the exact K_II. Then
`I - B K_II = Π_k (I - C_k K_II)`, a product of stage error operators. Three stages that each leave 30% of their
target component give ≈ 3% of the field and ≈ 0.1% of the energy. If the stages are summed without intermediate residuals, the operator is `I - (Σ C_k) K_II` and every
stage must be accurate on its own. This product property is the numerical-analysis reason to put a few exact K applications
into the forward pass. It is the same fixed-point consistency as Hsieh et al. (RESEARCH 1.8): w* is a fixed point of every stage.
Using K_e exactly is not a concession either. A geometry-generated fine kernel on K's hypergraph would, at best, learn
something close to K_e, and K_e = M_e · T is already the cheapest "generated coefficient" there is: it is linear in the moments and has zero
parameters.

(ii) **The lift must not create a boundary layer that coarse spaces cannot see.** With the zero lift, w* ramps from 0 at
the ports to u*(interior) over one fine element. Its energy is ~(1/h)·|q|² per unit port area, which is ~64× the energy of a smooth u*. That ramp
must be captured in the **first** stage, or μ-1 is multiplied by 64. Coarse functions that are **truncated** at the port nodes
(`P_I = Z Z^T P`) contain exactly this one-element ramp. So the truncated-Galerkin coarse solve represents the lift
residual correctly, and the construction is the same for an arbitrarily shaped port mask (box patches, the oblique cut band). This choice is what
lets box and cut-band ports be handled by masks alone.

(iii) **Each level must be matched to a component of w\*.** With the soft-decile evidence and the new fringe finding, the components are:

| component of w* (force-driven q) | where it lives | carrier |
|---|---|---|
| global transmission and macro fields | whole cell, wavelength ≥ 1/4 | ℓ3 exact coarse solve |
| wall-scale membrane and bending, load paths | along the shell, 1/16 – 1/4 | ℓ1, ℓ2 (quadratic, conforming) |
| relative motion of the weak fringe against the strong shell | fringe sheet, ≥ 1/8 | ℓ2 fringe-split coarse functions |
| localized weak / sliver / fictitious-node modes (11–30 nodes) | fringe patches ≲ 1/16 | fine fringe patches (exact local solves) |
| element-scale residual and port detail (wavelength ≤ 2h) | everywhere, mostly near ports | fine bulk smoother + learned hyperedge kernel |

**Why four levels.** The coarsest level must be small enough for a dense exact solve that 100 resident cells can afford (≤ ~2.2k DOF,
a 9.6 MB packed fp32 factor). Q2 coarsening by a factor of 2 keeps the approximation property between neighbouring levels. From 32³
elements this gives exactly 32 → 16 → 8 → 4. One fewer level would force a 11.7k-DOF coarsest (0.5 GB dense) or an inexact
coarsest. One more level (Q2 on 2³) adds nothing a 2.2k dense solve does not already give.

**Why Q2 h-coarsening and not Q1 p-coarsening (65³ → 33³ as Q1 vertices).**
- Walls are ~2 fine elements thick. Q1 fields on one or two elements through the thickness lock in bending, so a Q1 coarse
  operator is too stiff exactly for the soft bending directions.
- Q2 on 16³ has element size ≈ wall thickness and represents through-thickness bending.
- Route-1 E0 is the empirical version of this argument: discontinuous quadratic block spaces beat affine ones by 6× in witness at
  equal depth. Our ℓ1 is the conforming, nested version of those quadratic block spaces, with the same 2×2×2-element block size.
- The node grids 65³/33³/17³/9³ also coincide with the grids suggested in the brief, so the dense slot layouts of DATA_INTERFACE §2.2 apply.

### 2.2 Data flow (one geometry, one mode, a batch of B query columns)

```
 GEOMETRY PATH (once per geometry and mode; nonlinear allowed)                 q-PATH (linear in q, f; fp32 unless noted)
 ──────────────────────────────────────────────────────────                 ───────────────────────────────────────────
 moments M_e (E0×125), masks, GP faces, port mask (mode)                     q (B, 3Np), f (B, 3N0) optional
   │                                                                          │
   ├─ E1 exact objects (deterministic):                                       Q0  rigid split (fp64): α = G⁻¹R_pᵀWq, q_d = q − R_p α
   │    K_e on the fly from M_e; node blocks D_i; strength s_ij               │
   │    mutual-strongest pairs → bulk blocks + inverse factors                Q1  lift residual  r = Zᵀ(f − K L0 q_d)      [K, port-local]
   │    weak mask; fringe patches (per ℓ1 element) + factors                  │
   │    Q2 hierarchy: coarse moments ℓ1; truncation corrections;              Q2  FMG pass (coarse → fine):
   │      ℓ2 split Galerkin (strong/fringe) element matrices;                 │     c3 = A3⁻¹ R3 r
   │      ℓ3 dense A3 + fp64 Cholesky                                         │     c2 = P23 c3 + S2 (R2 r − A2 P23 c3)
   │                                                                          │     c1 = P12 c2 + S1 (R1 r − A1 P12 c2)
   ├─ E2 invariant features (node / element / GP face)                        │     w1 = P01 c1
   │                                                                          Q3  ρ1 = r − K_II w1                          [K #1]
   └─ E3 hyperedge GNN (6 rounds, C=32) → heads:                              │     fringe patches (overlapping, exact, gated) → local ρ update
        ω_b (bulk blocks), ω_P (fringe patches), s_e (element kernel),        │     bulk: gated block-Jacobi + G_θ  → w2
        d_v^ℓ (coarse smoother), Ω_v^ℓ (transfer smoothing),                  Q4  ρ2 = r − K_II w2                          [K #2]
        γ_ℓ,j (coarse polynomials), η_k (stage steps)                          │     one V-cycle on ℓ1..ℓ3 → w3 = w2 + η P01 V(R1 ρ2)
                                                                              Q5  ρ3 = r − K_II w3                          [K #3]
                                                                              │     fringe + bulk post-smoothing (own gates) → w4
                                                                              OUT û = Rα + L0 q_d + Z w4 ; û|ports := q (exact)
 READOUT (fp64 accumulation, strain form):  Π(û) = ½ ûᵀKû − fᵀû ;  Ŝq − ĝ = ∇_q Π(û(q,f))  (one reverse pass = exact transpose)
```

Default depth: T = 1 (one V-cycle correction, Q4–Q5). Each extra cycle repeats Q4–Q5.

### 2.3 Components, shapes, linear vs geometry-only

Sizes: FULL parent (tau ≈ 0.5): N0 ≈ 121–123k active nodes, E0 = 12,880 elements, F = 21,936 GP faces, 7,872 box-port
nodes. Hierarchy sizes [new] (`q2_hierarchy_counts.json`):

| level | space | node grid | FULL: elements / nodes / DOF | heavy cut 0.148: el / nodes / DOF | moderate-large cut 0.758 |
|---|---|---|---|---|---|
| ℓ0 | Q2(32³) | 65³ | 12,880 / 121,036 / 363k | 2,307 / 22,966 / 69k | 9,627 / 91,466 / 274k |
| ℓ1 | Q2(16³) | 33³ | 2,080 / 21,196 / 63,588 | 402 / 4,488 / 13,464 | 1,596 / 16,560 / 49,680 |
| ℓ2 | Q2(8³) (+ fringe split) | 17³ | 352 / 3,886 / 11,658 (≤ 23.3k split) | 79 / 1,005 / 3,015 | 284 / 3,243 / 9,729 |
| ℓ3 | Q2(4³) | 9³ | 64 / 729 / 2,187 | 18 / 269 / 807 | 56 / 667 / 2,001 |

- Fine elements touching ports [new]: FULL 1,776 of 12,880 (14%). Heavy cut, mode F: 413 of 2,307; mode D (box ∪ band): 1,290
  of 2,307 (56%).
- No coarse element at any level has material split into separate face-connected components [new, uniform tau 0.5]. The
  component-split guard (C3 below) is kept only as insurance for thin tau and grazing cuts.

#### Q-path components (all linear in (q, f); their coefficients are geometry-only)

**Q0 Rigid split (fp64).** `G = R_pᵀ W R_p` (6×6), with W = lumped port area weights (O_h-invariant). `α = G⁻¹ R_pᵀ W q`,
`q_d = q − R_p α`. Cost O(6 Np). The rigid field R α is added analytically at the output and never passes through K.

**Q1 Lift residual.** `r = Zᵀ (f − K L0 q_d)`, shape (B, 3|I|). L0 q_d is supported on the port nodes, so only the elements and GP faces
touching ports are evaluated: 14% of elements for FULL, 0.15 GFLOP/col. f carries the cut-band nodal forces in mode F (the "loaded
cut" case), or interior loads in general.

**Transfers.** `P_{ℓ-1←ℓ}` is the fixed nested Q2 interpolation stencil. In 1-D the Q2 child values are (1, 0, 0), (3/8, 3/4, −1/8), … and
they are applied per axis. It is **truncated at the fine port mask** (`P_I = Z Zᵀ P` on the ℓ0 side). The composite restriction is `R_ℓ = P_{0←ℓ,I}ᵀ`.
- ℓ1 and ℓ2 fields live in dense masked grids (33³ and 17³, slot layout). The transfer is a fixed separable stencil: 20 MFLOP/col at ℓ0↔ℓ1.
- **Learned, rigid-preserving transfer smoothing** at ℓ1←ℓ2 and ℓ2←ℓ3: `P̃ = (I − Ω D_ℓ⁻¹ A_ℓ) P`, with Ω = diag(Ω_v) (gate
  per coarse node, init 0). Because `A_ℓ R = 0`, P̃ R_c = R_f for **any** Ω (smoothed-aggregation argument; RESEARCH 1.7).
  This is where the energy-minimizing transfer (Xu–Zikatanov) is learned. It is not applied at ℓ0←ℓ1, because there it would cost two extra fine
  K applications. The fringe machinery covers that level instead.

**Coarse operators (exact Galerkin, geometry-only, no learning).**
- **Moment aggregation (ℓ1).** For coarse Q2 functions, `Pᵀ K_body P` equals CutFEM on the coarse element with
  aggregated moments. The coarse integrand ε:C:ε has degree ≤ 4 per variable, and coarse moments are exact linear
  combinations of the child moments: 8 fixed 125×125 binomial shift/scale matrices, one per child position. The template scales by
  (n_c/n)² = 1/4. `Pᵀ K_ghost P` receives contributions only from fine GP faces that lie on coarse faces, because coarse functions are single polynomials inside a
  coarse element. Per axis, 4 fixed templates (one per sub-face position) are combined by the GP mask. Rigid modes are in the kernel exactly.
  - Port truncation: `A_I = PᵀKP − Pᵀ(K − (I−Π)K(I−Π))P`, evaluated only on port-touching fine elements and faces, with K_e formed
    from moments (FULL: 1,776 elements; ~10 GFLOP once at encode).
  - A_1 is applied on the fly: coarse K^c_E = M^c_E · T at 1.7 GFLOP per batch, amortized, then 27 MFLOP/col, plus the stored truncation
    corrections (element-form, ~12 MB FULL).
- **Strong/fringe split (ℓ2).** Every ℓ2 coarse function φ_v is duplicated into φ_v·1[strong] and φ_v·1[weak], using the
  fine nodal weak mask. Every fine node is either weak or strong, so Σ duplicates = φ_v. Partition of unity and exact rigid/linear reproduction therefore hold.
  - The masked functions are not polynomial per coarse element, so A_2 is formed by fine-element Galerkin: `Σ_e P_eᵀ K_e P_e` with P_e of size 81×162 (port rows
    zeroed = truncation). That is 6.4 MFLOP per element, 82 GFLOP per FULL cell, ~3–8 ms fp32, stored element-form: 352 × 162² upper ≈ 18.6 MB.
  - Duplicates whose weak part is empty are dropped, so the split DOF count is ≤ 23.3k.
- **ℓ3.** `A_3 = P_{23}ᵀ A_2 P_{23}`, where P_23 feeds both duplicates with the same coefficient, so the recursion stays exactly Galerkin. It is assembled dense
  (≤ 2,187²), factorized by Cholesky in fp64 (~2–5 ms), and the packed lower factor is stored in fp32 (9.6 MB).
- In mode N (no ports), A_3 has the 6-dimensional kernel R_3. It is solved on the orthogonal complement: `(A_3 + R_3 R_3ᵀ κ)⁻¹`, then project.

**Q2 FMG pass (coarse → fine, "backslash cycle").** As in the diagram. The coarse smoothers are learned polynomials
`S_ℓ = Σ_{j<m_ℓ} γ_{ℓ,j} (D̃_ℓ A_ℓ)^j D̃_ℓ` with `D̃_ℓ = diag(d_v) · blockdiag(A_ℓ)⁻¹` (3×3 per node, or 6×6 for split pairs). m_1 = 3,
m_2 = 4. The γ are initialized to Chebyshev coefficients for [λ_max/30, λ_max], using a Gershgorin estimate. Output: w1 (B, 3|I|).

**Fine stage F (used in Q3 and Q5, with separate gates).**
- **F-a Fringe patches.** Weak mask: `||D_i||_F < 1% median` (the definition used in ROUTES). Each patch is formed from one ℓ1 element (a closed 5×5×5
  node box). It contains the weak interior nodes in that box plus their mutual-strongest partners. Patches overlap on the shared node layers.
  - Exact inverse factors of `K[patch, patch]` (fp32, computed in fp64). For FULL: ~2k patches, 20–60 weak nodes typical, cap 64 nodes (192 DOF); larger patches are split
    by octant.
  - Applied as a **PU-weighted symmetric additive Schwarz**: `Σ_P R_Pᵀ χ_P ω_P K_PP⁻¹ χ_P R_P`, with χ = diag(1/√multiplicity) and ω_P = gate per patch, **init 1**. No global damping.
  - The residual is then updated locally, only on elements and GP faces touching weak nodes (~35–50% of elements, **[est]**).
  - Why: a slow mode supported inside one patch is removed exactly in one application. That is the property route-1's merged weak groups lost under the global
    ω ≈ 0.22.
- **F-b Bulk.**
  - Mutual-strongest pairs (node i is paired with j if each is the other's strongest neighbour and s_ij ≥ 0.1; otherwise it stays a singleton). These form 3×3/6×6 exact block inverses,
    each gated by ω_b (init = 1/(1.05·λ̂_max), with λ̂ from a local Gershgorin bound; E3 showed that the bound matters).
  - Plus the learned element-hyperedge kernel
    `G_θ = D^{-1/2} Σ_e P_eᵀ [ (s_e1 I + s_e2 M̂ + s_e3 L̂) ⊗ I_3 + s_e4 K̂_e ] P_e D^{-1/2}`. Here M̂ and L̂ are the normalized Q2 mass and scalar
    Laplacian templates (27×27, invariant under the element's O_h action), and K̂_e is the normalized exact element stiffness. The s_e are initialized to 0.
    This is the "GNO layer" of the fine level: a linear, geometry-generated kernel with K's sparsity.

**Q4 V-cycle (T = 1).** On `R_1 ρ2`: pre S_1 → restrict → pre S_2 → ℓ3 exact → post S_2 → prolong → post S_1, with coarse residuals
from A_1 and A_2 (cheap). Result `w3 = w2 + η_4 P_01 c`.

**Q5.** Recompute ρ3 with the exact K, then apply fine stage F with its own gates.

**Stage step lengths.** η_k ∈ (0, 2) per stage, init 1. They are generated from an invariant global pooling of the GNN embeddings.
This is the learned-recurrence idea of route 1 E1b, reduced to 5 numbers.

#### Geometry path

**E1 exact objects**: listed in the diagram. They are all deterministic functions of the moments, masks and mode. The strength s_ij = ||K_ij||_F/√(||K_ii|| ||K_jj||) is
computed over node pairs sharing an element or a GP face. This needs one pass over K's pattern (9.5M pairs, transient ~0.35 GB).

**E2 invariant features.** All are invariant under the joint O_h action, and none uses absolute coordinates:
- node: log(||D_i||/median), eigenvalues of D_i/median, the weak indicator, inside-material flag, number of incident active elements, graph
  hop distance to the nearest port (clipped at 8), port/band flags, |phi|−tau and plane distance scaled by h;
- element: volume fraction, eigenvalues of the second-moment tensor, eigenvalues of the 6×6 uniform-strain energy matrix `ε_kᵀ K_e ε_l`
  (linear in moments, invariant as a spectrum), full/cut/band flags;
- GP face: volume fractions of both sides, the full flags.

**E3 hyperedge GNN.** Messages move only along K's hypergraph (element hyperedges with 27 incidences, GP-face hyperedges with 45), which satisfies R7.
- Node→element aggregation weights depend only on the node's O_h orbit class within the element (corner, edge, face, centre), which makes it exactly
  invariant.
- 6 rounds, C = 32. Each round has element, face and node MLPs (64→64→32, SiLU). Weights ≈ 25k per round, ≈ 0.15 M total. Heads ≈ 10k.
- Heads (smooth bounded maps; no max/abs):
  - block / patch gates: `ω = ω_prior · exp(log 4 · tanh z)`, pooled over the block's or patch's node embeddings;
  - `s_e = 0.5 tanh z`; in mode N, softplus with PSD templates;
  - `d_v, Ω_v`: pooled over each coarse node's support by `P_{0←ℓ}ᵀ`;
  - `γ`, `η`: global pooled.
- Separate heads per mode, shared trunk.

#### Parameter counts, generated coefficients, FLOPs, memory (FULL; [est] except where noted)

| item | size |
|---|---|
| learned weights | ≈ 0.16 M (trunk 0.15 M + heads) |
| generated coefficients per cell and mode | ω_b ~100k, ω_P ~2k, s_e 52k, d_v ~25k, Ω_v ~25k, γ 7, η 5 → ≈ 0.2 M floats (0.8 MB) |
| forward FLOP/col | lift 0.15 + 3 full K 2.7 + 2 local residual updates 0.8 + fine smoothers 0.5 + ℓ1 ops 0.45 + ℓ2/ℓ3 0.25 + transfers 0.1 ≈ **5 GFLOP** |
| energy readout | +0.9 GFLOP (strain form, fp64 accumulation) |
| Ŝq (forward + reverse) | ≈ **11.5 GFLOP**; mode N action ≈ **0.6 GFLOP** (no fine K) |
| time per column at batch 32–64 | forward 0.2–0.35 ms, Ŝq 0.4–0.8 ms, mode N 0.03–0.05 ms (15–30 TFLOPS effective fp32; must be measured, see R-6) |
| design iteration, 100 FULL cells (160 Ŝq + 55 S⁺ per cell) | ≈ 10 s |
| resident per FULL cell, modes F + N | moments 6.4 + topology 1.7 + bulk block factors ~8 + fringe factors ~24 + ℓ1 truncation corrections ~12 + ℓ2 split 2×18.6 + ℓ3 2×9.6 + gates 1.6 ≈ **110 MB**; ≈ 75 MB if ℓ2 is regenerated per use (+80 GFLOP each time) |
| 100 mixed cells (80 FULL, 20 cut) | ≈ 9–10 GB resident; ≥ 15 GB left for batched activations |
| per-column activations (inference) | ~10 fine fields × 1.45 MB ≈ 15 MB/col. The reverse pass of a linear net needs no stored activations if written as the transposed DAG |
| training activations (B = 64) | ≈ 1.5 GB fields + 0.34 GB K_e + 0.25 GB GNN |
| encode (beyond moments) | K_e and strengths ~50 ms, pairs/patches + batched Cholesky ~30 ms, hierarchy (ℓ1 corrections, ℓ2 Galerkin, ℓ3 Cholesky) ~20 ms, GNN ~15 ms, second mode +~30 ms → **≈ 0.1–0.2 s** |

Cut cells cost 1.3–6× less at every item (sizes table above).

---

## 3. Answers to open questions a–l

**a. Consuming the masked fixed-grid data.**
- The q-path never uses a pairwise graph or a proximity graph. At the fine level it uses only K's hypergraph (element and GP-face hyperedges), through
  (1) the exact K, (2) the block and patch inverses, which are principal submatrices of K, and (3) G_θ, whose kernels live on elements.
- The coarse levels are dense masked grids (33³, 17³, 9³) with fixed Q2 stencils and element-form Galerkin operators.
- Adjacent walls that are disconnected in material interact only if K couples them. On coarse levels a guard duplicates coarse functions per
  connected component of active children. It never triggered at uniform tau 0.5 [new], but it costs nothing.
- Element moments are used in three places: (i) exactly, via K_e = M·T and the coarse aggregated moments; (ii) as invariant features (volume fraction, moment tensor spectrum,
  6×6 strain-energy spectrum); (iii) summed into D_i, which defines the weak nodes.
- GP faces enter as hyperedges of K and of the GNN.
- Weak nodes are handled by the fringe patches (fine) and the fringe split (ℓ2).

**b. Hard constraints while staying linear.**
- Ports: B writes only interior DOFs (Z), and the output port entries are overwritten with q.
- Rigid modes: the fp64 split. R α never touches K, and Ŝ is evaluated on the projected q_d.
- Coarse spaces reproduce rigid fields exactly. This does not matter for Dirichlet modes, but it is required in mode N (§4).

**c. Global transmission and localized soft modes.**
- Global: the ℓ3 exact solve, together with the FMG pass that visits it first, and the quadratic conforming ℓ1 and ℓ2.
- No spectral or FFT layer anywhere (RESEARCH 7.2).
- Localized soft modes: fine fringe patches (exact, undamped, overlapping) plus the ℓ2 fringe-split coarse DOFs for fringe relative motion
  at longer wavelength. Individual eigenvectors are not captured one by one (E1's failure mode with the dense band). The subspace that contains the whole family is built
  structurally instead.

**d. Where K appears.** In the forward pass: 3 full fine applications plus about 1.1 port-local or fringe-local ones. In the readout: 1.
- The case for them: the product structure of the error (§2.1 (i)); exactness near ports; rigid consistency; the
  generalization argument (the residual is exact for any unseen geometry, so the learned parts only have to distribute correct forces); and fixed-point
  consistency.
- The cost is about 55% of the forward FLOPs.
- The pure-feedforward ablation (X3-d) replaces K #2 and K #3 by G_θ-type learned kernels. I expect it to lose, and it is cheap to test.

**e. Reaction readout.** `Ŝq = ∇_q Π(û(q))`, one reverse pass through the linear DAG. This is exactly Êᵀ K Ê q, symmetric to rounding.
- It can be written as the transposed DAG: transfers swap roles, smoothers are self-adjoint or reverse their order, and no activations are stored.
- Cost ≈ 2.3× forward.
- `(K û)_ports` is rejected as the lattice operator because it is not symmetric. Its difference from Ŝq is a free a-posteriori indicator of interior imbalance, and it is used as a monitor.
- The lattice load from cut forces is `ĝ = −∇_q Π|_{q=0} = Êᵀ(f − K u_f)`, from the same reverse pass.

**f. Neumann action for BDD (mode N).** Same geometry path, with an empty port mask (the cell floats) and box forces as input.
- `B_N = S_0ᴺ + Σ_ℓ η_ℓ P_{0←ℓ} S_ℓᴺ P_{0←ℓ}ᵀ + P_{0←3} A_3⁺ P_{0←3}ᵀ`, an additive (BPX-type) form. Here S_0ᴺ = gated bulk blocks + PU fringe
  patches + G_θ with nonnegative scales and PSD templates, and S_ℓᴺ = gated coarse block-Jacobi (positive node gates times the inverse 3×3 or 6×6
  diagonal blocks of A_ℓ). The coarse polynomial smoothers are used only in modes F and D.
- `S⁺ r ≈ Π_R T_b B_N T_bᵀ r`. It is SPD on the balanced subspace for **any** gate values (proof §4.6), matches the NGO/Melchers principle of the same basis on both sides, needs no fine K, and costs 0.6 GFLOP.
- Upgrade path if BDD iteration counts exceed about 2× those with the exact Neumann solve: a symmetric V(1,1)-cycle with 2 fine K applications.

**g. Sensitivity.** `dĈ/dτ ≈ −Σ_e û_eᵀ (∂K_e/∂τ) û_e = −Σ_e Σ_m a_e,m(û) ∂M_e,m/∂τ`, with element energy densities a in strain form. ∂K_ghost/∂τ = 0 within a topology.
- Consistency: this equals the derivative of the approximate compliance up to a term `(Zᵀ K û)ᵀ ∂ŵ/∂τ`. That term is first order in
  the interior residual, which is small after Q5.
- The exact sensitivity differs from ours at first order in ||û − u*||_K. This is why R4 asks for field accuracy, not only energy accuracy, and why λ_sens appears in the loss.
- Topology: all discrete structures (weak mask, pairs, patches, splits, truncation sets) are frozen per superset design band (§1.6
  of DATA_INTERFACE). Only continuous coefficients vary within a band, and they are smooth in the moments.

**h. Equivariance.**
- Exact by construction: exact K, grid hierarchy, truncation masks, fringe patches (grid-aligned, symmetric about the cube centre), PU
  weights, invariant GNN, rigid projection with invariant W.
- The one non-exact piece is the tie-breaking in mutual-strongest pairs. It only matters for geometries that have exact symmetric ties.
- 48-element augmentation on 25% of steps covers it. The label transforms by permutation, so no refactorization is needed.
- The joint (g, q) equivariance is what holds: the cut plane breaks per-cell symmetry. The polyref Kuhn split limits moment equivariance to ~1e-4.

**i. Smoothness and conditioning.**
- Every learned coefficient multiplies an operator that is already sound. Gates are bounded exp–tanh maps around numerical priors, s_e is initialized to 0, and
  there is no max or abs anywhere.
- The loss is normalized per sample by the exact energy (= μ − 1).
- F4 is countered in three ways: (1) the preconditioner-shaped parameterization (the gradient with respect to a stage gate is a residual-weighted quantity, which is well scaled); (2) a D-norm
  auxiliary field loss that weights soft, large-amplitude error; (3) adversarial directions.

**j. Capacity.** Shared weights (≈ 0.16 M); per-geometry capacity sits in the ≈ 0.2 M generated coefficients.
- Step 1 isolates the generator gap with X1 (free per-geometry coefficient fields).
- If X1 passes and X2 does not, grow the GNN (C = 64, 8 rounds) before touching the q-path.
- Optional q-path width: two vector channels in the coarse smoothers (MgNO-style), only if X1 shows the coarse polynomials saturate.

**k. Precision.**
- fp32: the q-path, K applications (strain form, element rigid part removed before contraction), GNN.
- fp64: the rigid split, energy and Ŝq reductions, the ℓ3 factorization (stored fp32), the labels, and the normalization qᵀSq.
- Galerkin orthogonality puts interior fp32 rounding in û into the energy at second order (§4.4).

**l. Graceful degradation.** Three knobs on one codebase (§7.2): T (number of V-cycle+smoothing corrections), gate freezing (learned ↔
numerical prior), and K-inside (exact ↔ learned surrogate).

---

## 4. Guarantees (short proofs)

Let Ê denote the full map q ↦ û (for f = 0).

**4.1 Exact linearity.** Every q-path operation is one of: a matrix-vector product with an operator whose entries are functions of (g, mode)
only (K, P, A_ℓ, block and patch inverses, G_θ, diagonal gates), a sum, or an injection or overwrite by q. The gates come from the GNN, whose inputs are geometry and mode only.
A composition of linear maps is linear, so û = Ê(g) q + F̂(g) f exactly. Nonlinearity (MLPs, tanh, exp, and the Cholesky factorizations of geometry-only matrices) acts only on geometry. ∎

**4.2 Exact port values.** Every correction is Z w, which is zero on the ports. R α + L0 q_d equals R_p α + q_d = q on the ports. The final overwrite
û|_p := q makes this bitwise. It holds for box, for band (mode D), and for any mask. ∎

**4.3 Exact rigid modes.** If q = R_p β, then α = G⁻¹R_pᵀW R_p β = β (in fp64, to 1e-16) and q_d = 0, so r = Zᵀf = 0 for f = 0, all w_k = 0 and û = Rβ. So Ê R_p = R.
- Ŝ is evaluated as `Ŝ = P_Wᵀ Ê_dᵀ K Ê_d P_W` with `P_W = I − R_p G⁻¹ R_pᵀ W`, so Ŝ R_p = 0 **exactly** (P_W R_p = 0 in fp64).
- The rigid field never enters a K application, so the fp32 leakage of ROUTES 7.6 cannot occur.
- Mode N: every P_{ℓ-1←ℓ} reproduces rigid fields: the nested Q2 space contains linear fields; the split duplicates sum to the parent; the smoothed P̃ preserves R because A_ℓR = 0. So A_ℓ R_ℓ = 0 at every level. The coarsest pseudo-inverse deflates R_3, and the output projection removes the rest. ∎

**4.4 Symmetric PSD readout and one-sided error.** Ŝ = P_Wᵀ Ê_dᵀ K Ê_d P_W is symmetric and PSD for every parameter value, since K is SPSD. Let u* = Eq with
Zᵀ K u* = 0. Then û − u* = Z v for some v (the ports agree), and

`ûᵀKû = u*ᵀKu* + 2 vᵀ(ZᵀKu*) + vᵀK_II v = qᵀSq + ||v||²_K`, so Ŝ − S = (Ê − E)ᵀK(Ê − E) ⪰ 0.

- The model is **never softer** than the truth. Compliance is underestimated, with an error equal to the energy of the field error.
- Interior rounding perturbations δ in û enter only as ||δ||²_K (second order). That is why fp32 is enough for the field.
- With cut forces (mode F, f ≠ 0): Π(û) − Π(u*) = ½||û − u*||²_K ≥ 0, so the total potential energy is an upper bound. ∎

**4.5 Stage consistency (the error-product structure).** Each stage has the form w ← w + C_k(r − K_II w) with exact K_II. So w* is a fixed point of every stage, and
`w* − w_out = Π_k (I − C_k K_II)(w* − 0)` for the multiplicative part. The FMG pass is one linear stage C_FMG applied to the lift residual. Increasing T
converges to w* whenever each cycle is an energy contraction, which gives fallback A its convergence argument (Hsieh-type). ∎

**4.6 Mode N is SPD on the balanced subspace.** B_N is a sum of terms `X_ℓ Y_ℓ X_ℓᵀ` with Y_ℓ SPD for every gate value:
- block and patch inverses of SPD principal submatrices, times positive gates and PU weights;
- G_θ with nonnegative scales on PSD templates;
- coarse smoothers: positive gates times the inverse SPD diagonal blocks of A_ℓ;
- A_3⁺ is PSD with kernel R_3.

The fine block term alone is SPD on all DOFs, so B_N is SPD. After projection onto the balanced subspace, BDD-PCG applies unchanged. ∎

**4.7 Equivariance statement.** For h ∈ O_h acting on the grid, with displacement representation ρ(h): Ê(h·g)(ρ(h) q∘h⁻¹) = ρ(h)(Ê(g)q)∘h⁻¹ holds exactly for every
component except the pair tie-breaking. That is:
- K_{h i, h j} = ρ K_ij ρᵀ;
- the hierarchy and patches are grid-symmetric;
- the GNN inputs are invariant and its aggregations are orbit-class-symmetric;
- G_θ templates are invariant, and K̂_e is equivariant;
- W is invariant.

What holds is joint (g, q) equivariance, not per-cell invariance, because the cut plane moves with h. ∎

---

## 5. Why it gets the soft, force-driven directions right

1. **The loaded directions are soft directions, and their w\* lives in known places.**
   - A force-driven q = S⁻¹f puts most of its energy where the cell is compliant: thin-wall bending (global and wall scale), weakly
     supported cut slivers, and the fictitious fringe nodes next to them.
   - The one-pass error is `Π_k (I − C_k K_II) w*`, so each such component needs a stage that removes it almost completely. §2.1 (iii) assigns one to each:
     ℓ3 and ℓ2 (global and wall scale, quadratic and conforming), the ℓ2 fringe split (fringe versus shell), fine fringe patches (the 11–30-node modes), and the bulk
     smoother (element scale).
2. **The fringe is treated as what it is.** [new] Weak nodes percolate into one connected sheet per shell side:
   - FULL: 20,244 weak nodes in 2 components of about 10k each;
   - heavy cut: 4,239 weak nodes in 5 components;
   - 0.758 cut: 15,981 weak nodes in 2 components (`weak_clusters.json`, body-only proxy).

   So the slow modes of ROUTES are **localized modes of a connected, weakly supported sheet**. That explains the dense band: a continuum of similar
   local modes, not a few isolated pieces. Two consequences:
   - A "cluster by connected components" design cannot work. Route 1 had to cut components into BFS chunks, and that is order-dependent.
     Grid-aligned overlapping patches of about 1/16 size (≈ the 11–30-node mode size) contain each localized mode entirely in at least
     one patch. The overlap covers modes that straddle patch boundaries.
   - Wavelengths longer than a patch are relative motions of the sheet against the strong shell. Standard coarse spaces cannot represent them, because they move
     sheet and shell together. The strong/fringe duplicate coarse functions (the multi-continuum idea of GenEO and CEM-GMsFEM) can. The weak attachment acts as
     an elastic foundation for the sheet, so modes longer than the foundation length are nearly diagonal and the ℓ2 smoother handles them.
3. **No damping throttles the soft carrier.**
   - Route-1 E3 and the merged-group result showed that exact weak blocks under a global ω ≈ 0.22 lose most of their effect.
   - Here the fringe patches are PU-weighted with gates initialized at 1, and they are applied multiplicatively before the bulk
     smoother, with a local residual update in between.
   - The gates can learn where overlap requires less.
4. **The forces are exact where the soft physics is.** The lift residual and the three stage residuals use the exact K, including the γ-weighted GP terms that hold the
   weak nodes. A learned kernel would have to reproduce γ = 1e-4 couplings to about 1e-3 relative accuracy to get the same local forces. Here they are free.
5. **The training signal targets exactly these directions.**
   - The loss per sample is μ − 1, so a soft direction counts as much as a stiff one.
   - 30% of samples are force-driven and 20% adversarial (LOBPCG on the pencil (Ŝ − S, S)).
   - The D-norm auxiliary loss rewards amplitude accuracy on low-energy, large-displacement error, the part that F4 says pure energy descent reaches last.
   - The error is one-sided, so improvement in a soft direction is monotone in the loss.
6. **Evidence mapping.**
   - Route 1 needed 24 learned layers of a *two-level* cycle, with discontinuous block quadratic coarse spaces, pair smoothers under global
     damping, and no weak-specific coarse DOFs.
   - Every one of these is strengthened here: 4-level conforming Q2 with an exact coarsest solve, FMG ordering, undamped fringe patches, a fringe coarse split, and
     per-block gates.
   - The claim that T = 1 suffices is the main thing step 1 must test (R-1). The design degrades to T = 2–3 without any change.

---

## 6. Training for step 1 (three fixed geometries: medium cut, heavy cut, shared FULL parent)

**Modes and batch.**
- Each step uses one geometry and 64 columns.
- Cut cells: 40 in mode F (the lattice gate operator), 16 in mode D, 8 in mode N. FULL: 56 in mode F (= D) and 8 in mode N.
- Geometries rotate every 250 steps. One cuDSS factor stays resident per active geometry and mode; refactoring takes 1–13 s.
- Dense fp64 Schur complements T (box for F; box ∪ band for D) are kept in pinned CPU memory. They serve normalization, force-driven sampling and adversarial pencils.

**q sampling (the agreed mix, applied per mode).**
- Force-driven, 30%: q = T⁻¹f for smooth multiscale box forces. In mode F, 10% of these carry cut-band forces f_c ≠ 0, which gives the loaded-cut
  report.
- Macro, 20%: rigid (a check only: the loss is exactly 0 by construction), 6 uniform strains, quadratic, cubic.
- Multiscale GRF on the ports, 30%.
- Adversarial, 20%, after 2k steps.
- Mode N: balanced multiscale box forces and GRFs.

**Loss** (per sample, with the fp64 reduction):
- `L_E = ||û − u*||²_K / ||u*||²_K`, computed label-free as `2(Π(û) − Π(u*))/||u*||²_K`. Π(u*) and ||u*||²_K come from the exact solve (cheap). For f = 0 this is μ(q) − 1.
- `L_D = ||û − u*||²_D / ||u*||²_D` (D = block diagonal of K).
- `L_s = Σ_e |a_e(û) − a_e(u*)| / Σ_e a_e(u*)` (element energies = the sensitivity integrand).
- Mode N: `L_N = ||B_N f − K⁺f||²_K / ||K⁺f||²_K`, restricted to the balanced subspace, with weight 0.1.
- Total: `L = mean(L_E) + λ_D mean(L_D) + 0.1 mean(L_s) + 0.1 L_N`, where λ_D goes from 0.3 to 0.03 over the first 5k steps.

**Adversarial directions.**
- Every 500 steps, per geometry and mode: block LOBPCG with k = 32 vectors and 8 iterations on `(Ŝ − S) v = λ S v`. Ŝv takes one forward and one reverse pass; S v uses the dense T.
- The top directions go into a replay buffer (256 per geometry and mode). They are sampled with random mixing and 10% noise.
- The largest λ is logged as the worst-direction μ (reported, not gated).

**Optimizer.**
- Adam, lr 3e-4 for heads and 1e-4 for the trunk, cosine decay over 20k steps, gradient-norm clip 1.0, no weight decay on gates.
- fp32 parameters, fp64 loss reductions.
- Budget: ~1 TFLOP per FULL step (forward + backward, 64 columns) ≈ 60–120 ms, so ≈ 1 GPU-hour per 20k steps **[est]**.

**Step-1 experiment sequence.**
- **X0.** Numerical prior with gates at init, T ∈ {0, 1, 2, 3}, no learning, half a day. Measure μ by category, LOBPCG worst, the lattice gate, and the
  stage-wise error energies (the energy of w* − w_k after each stage, which the exact u* makes available). This identifies which stage leaves which error.
- **X1.** Free per-geometry coefficient fields (one parameter per block, patch, element and coarse node; no GNN): the capacity ceiling.
- **X2.** The architecture with GNN-generated coefficients: the generator gap relative to X1.
- **X3 ablations**, each run on 0013 only:
  - (a) fringe patches off;
  - (b) ℓ2 split off;
  - (c) FMG pass replaced by zero initial guess;
  - (d) K #2 and K #3 replaced by learned kernels;
  - (e) ℓ1 replaced by route-1 block-quadratic spaces;
  - (f) T sweep.
- **X4.** Mode N inside BDD on 2-cell and 8-cell lattices (exact operators elsewhere): iteration counts.

**Monitoring.**
- μ − 1 median, p90 and max per category and mode.
- Lattice compliance and sensitivity every 2k steps (the `lattice_v2` harness; the cut cell uses mode F Ŝ via the reverse pass).
- Stage-wise error energies; share of the remaining error energy on weak nodes; distributions of the gates (weak vs strong).
- fp32 vs fp64 shadow forward on 8 columns; equivariance gap on 4 random group elements.
- Timing and peak memory; BDD iteration counts (X4).

---

## 7. Risks, cheapest exposing experiments, degradation

### 7.1 Ranked risks

| rank | risk | cheapest experiment that exposes it | response |
|---|---|---|---|
| R-1 | One pass (T = 1) leaves too much error in the dense soft band | X0 and X1 with T = 0..3 on 0013 force-driven q plus the lattice gate (hours; X0 needs no training) | T = 2–3 (+2.4 GFLOP each); if T > 4 is needed, go to fallback A explicitly |
| R-2 | Fringe hypothesis wrong: slow error sits at scales patches do not cover, or ℓ2 split useless | X3a/b, plus Ritz vectors of the one-pass error operator on 0013: measure the support diameter and weak share of the 32 worst | move the split to ℓ1 (costly: fine Galerkin at ℓ1, ~110 MB, so step 1 only) or enlarge patches to ℓ1 vertex stars |
| R-3 | Q2 geometric coarse functions too stiff for thin walls (approximation property) | projection test: `||u* − P_ℓ Π_ℓ u*||_K / ||u*||_K` per level for macro and force-driven u* (coarse solves only) | learned transfer smoothing also at ℓ0←ℓ1 (+2 K per cycle), or aggregation-based ℓ1 |
| R-4 | Generator gap: the GNN cannot produce the X1 fields | X2 vs X1 on the same budget | widen/deepen the GNN; add raw K_e Rayleigh features; per-geometry fine-tune head |
| R-5 | F4 conditioning: gates learn only stiff content | loss curves split by category; L_E-only vs +L_D runs | larger λ_D, longer warm-up, curriculum macro → GRF → force |
| R-6 | Cost above budget (gather-heavy K, reverse pass) | day-1 microbenchmark on FULL: strain-form K vs K_e GEMV vs fused ghost; full forward and Ŝq at B = 1, 32, 64 | fused kernels; slot-packed dense layout; drop ℓ2 storage |
| R-7 | Mode N too weak, BDD iterations explode | X4 on 2×2×2 | symmetric V(1,1) (+2 K) |
| R-8 | fp32 cancellation in late residuals ρ_k | fp64 shadow forward (monitor) | fp64 residual only for ρ3 |
| R-9 | Discrete-structure jumps spoil sensitivities | finite differences in tau within one superset band on 0013 | freeze masks per band (already planned); soft weak indicator in the gates |
| R-10 | Equivariance gap from pair ties | gap monitor on the symmetric FULL parent | drop pairs (point blocks only) if the gap is > 1e-3 |

### 7.2 Degradation into the fallbacks (same components)

- **Main line**: T = 1, learned gates, exact K at 3 points.
- **Fallback A (learned iteration)**: T = 4–12, with per-cycle gate sets that are generated or shared. Each cycle is an exact-residual correction (4.5), so
  convergence is monotone for contractive cycles, and the hierarchy, GNN and training code are unchanged.
- **Fallback B (route-1 skeleton)**: gates frozen at numerical priors, cycles accelerated by a Chebyshev or learned 2-scalar recurrence (route 1
  E1b). The Q2 hierarchy replaces route 1's block-polynomial spaces; X3e checks this swap in the other direction. Only the ~5 η per cycle are learned.
- **Pure feedforward variant** (if K inside the network is ruled out): X3d, where K #2 and K #3 become G_θ kernels. This is the least favoured option.
- All three keep §4.1–4.4, because those rest on the port overwrite, the rigid split and the readout, not on the learned parts.

---

## 8. Borrowed vs new

**Borrowed.**
- MgNO: linear V-cycle as an operator parameterization, no nonlinearity inside cycles.
- FMG nested iteration: the coarse-to-fine pass.
- BPX and additive multilevel: mode N.
- Smoothed aggregation (Vaněk–Mandel–Brezina) and energy-minimizing transfers (Xu–Zikatanov): rigid-preserving learned transfer smoothing.
- GenEO / CEM-GMsFEM / multi-continuum coarse spaces: local low-energy content must be in the coarse space (the fringe split).
- Patch / Schwarz smoothing for small cut elements (de Prenter et al.).
- Hsieh et al.: consistency of residual corrections.
- Luz / Greenfeld / Huang–Li–Xi: transfer and smoother coefficients generated from operator features.
- NGO / Melchers–Dolean–Abdelmalik: symmetric preconditioner with a matched basis.
- Deep Ritz / energy loss; Beatson-style data aggregation on assembled loads (adversarial slot); EquiModel (index-free, invariant, augmentation).
- Mandel BDD.

**New in this design.**
1. **Truncated Galerkin Q2 hierarchy generated in closed form from aggregated CutFEM moments and masks.** The coarse operators are
   exact, cost milliseconds and near-zero storage, and handle arbitrary port masks (box patches, oblique cut band) by truncation.
2. **Lift-residual formulation with exact-K stage residuals** as the q-path of a neural operator, with the error-product identity
   `μ − 1 = ||Π_k(I − C_kK)w*||²/||u*||²` as the design principle and the loss.
3. **Evidence that the weak nodes form connected fringe sheets** [new], and the matching two-part carrier: grid-aligned PU overlapping
   exact patches that are O_h-symmetric and order-free, plus strong/fringe duplicated coarse functions.
4. **One network, three port types by mask** (F, D, N). This includes the cut band as a dual-type port and the loaded-cut load vector obtained from the same reverse
   pass.
5. **Numerical-prior initialization of every generated coefficient**: training starts from a working multilevel operator and can only
   refine it. The step-1 X0/X1/X2 ladder separates the numerical method, the capacity of the coefficient fields, and the geometry generator.
