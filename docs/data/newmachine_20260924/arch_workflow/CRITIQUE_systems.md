# CRITIQUE_systems: adversarial review of GLMO (DESIGN_systems.md)

Reviewer lens: what would make GLMO fail the lattice gate (compliance and sensitivity ≤ 3%), or break a decision the
owner has already made. Sources: ARCH_BRIEF, DESIGN_systems, DATA_INTERFACE [DI], RESEARCH [RS], ROUTES_PROGRESS [RP],
and the counts already in `arch/` (`_ol/patch_counts.json`, `weak_clusters.json`, `q2_hierarchy_counts.json`).
New check script: `arch/_verify/patch_cover.py`.

**Bottom line.** The *chassis* is excellent and should be kept almost whole:
- the symmetric extension operator whose adjoint is itself;
- mode masks (X-band / X-box / N) on one record;
- the coarse-moment Galerkin hierarchy;
- rigid-exact mixed precision;
- the Neumann action with a provable BDD constant;
- balanced-BDD bookkeeping.

What does not survive review:
- the **operating point** (m = 2, about 9 fine K, 32 K_eq per S q);
- the **soft-band carrier** (additive block patches);
- the **claim to be the main line**.

As specified it will very likely miss the gate at m = 2. At the m it will actually need, it is fallback A by its own
definition (§3.l).

---

## 1. Fatal issues (as specified)

### F1. The m = 2 operating point, and every headline cost, rests on a contraction factor the evidence contradicts

**Claim (§2.12).** m = 2 works if the V-cycle has ρ ≲ 0.3 on the soft part: ρ⁴ ≤ 1e-2. That means
κ(𝒱K) = 1/(1 − ρ) ≈ 1.46.

**Evidence** [RP route 1]:
- **The only measured analogue.** Route 1 used a paired fine smoother, a quadratic 2-level coarse space, exact K and
  Lanczos-scaled Chebyshev.
  - Its preconditioned λ_min is about 0.007–0.008, with λ_max normalized to 1, so κ ≈ 125–140.
  - Its 128 slowest modes sit in [0.0039, 0.0062].
- **What it takes to pass.** Exact coefficients need 40–48 layers; learned ones need 24 (0.72/1.7% compliance,
  0.93/2.6% sensitivity).
- **What fails.** 16 learned layers fail (6.8/13.2%), and 32 Chebyshev layers fail the y configuration (5.7/8.9%).
- **What GLMO spends.** m = 2 is about 9–10 fine K and 2 coarse corrections per forward. Route 1 used 24 of each.

**Why GLMO's components do not plausibly close a ~90× gap in κ.**
- **Multiplicative vs additive, and degree-2 smoothing.** A symmetric V(1,1) versus route 1's additive-style layer is
  worth maybe 2–4×. Degree-2 Chebyshev smoothing on each side is worth maybe 2×. That leaves at least 10–20× unexplained.
- **Extra coarse levels.** They help transmission and the smooth part. But the soft band is **fine-scale and
  localized**: F5 in [RS]; 94–99.4% of slow amplitude sits on weak nodes. Coarse levels cannot carry it.
- **Patches.** They are the only fine-level carrier. Their route-1 analogue, grouped exact block solves of weak nodes
  plus strong neighbours, raised λ_min by only 14%. GLMO's version is additive too, and covers modes worse than
  claimed (S1).
- **Learning cannot add directions.** Every learned quantity is a bounded multiplicative gate on a fixed carrier:
  - ω ∈ [0.5, 2];
  - σ_p ∈ [0.5, 2];
  - channel gates;
  - polynomial coefficients.

  The representational ceiling is the span of the zero-learning multigrid. If its patch or coarse spaces miss a slow
  direction, no training reaches it. Route 1's E1b gained 2–3× from learned scalars. That is the plausible order of the
  learning gain here, not 90×.

**Consequence (my arithmetic, using the design's own unit costs).** Per V-cycle ≈ 7.75 K_eq. S q ≈ 15.5m + 1.5.
Per FULL cell and load case = 92 S q (42 for AΦ + 50 BDD) + 50 M + 1 forward:

| m | S q (K_eq) | ms per S q at 0.05–0.1 ms per K_eq | K_eq per cell per load case | 100 cells (89 FULL-eq), s per load case |
|---|---|---|---|---|
| 2 (claimed) | 32.5 | 1.6–3.2 | 3,406 | 15–30 |
| 4 | 63.5 | 3.2–6.4 | 6,273 | 28–56 |
| 8 (my central guess) | 125.5 | 6.3–12.6 | 12,008 | 53–107 |
| 12 | 187.5 | 9.4–18.8 | 17,743 | 79–158 |

- At m ≈ 8, one S q (6–13 ms) is **no faster than the exact fp32-factor query** (FULL: 57 ms per 8 columns, about
  7 ms per column [RP 7.5]).
- GLMO's real advantage is then **memory**: about 90 MB instead of 6.6 GB. That is decisive, but it must be stated as
  the advantage. Speed is not.

**Fix.**
1. Demote m = 2 from design point to hypothesis. Publish the table above as the budget and let Phase 0 choose m.
2. Put learned capacity where it can **add directions**, not just re-weight them:
   - a learned, rigid-preserving level-1 prolongation (rows constrained to reproduce constant and linear fields,
     solved per row by a 4×4 least-squares projection, which keeps it smooth);
   - and/or the v0.1 fine "gated K-message" channel (2 K_eq), from Phase A onward, not as a late upgrade.
3. Fix the soft-band carrier (S1).
4. Use per-cell adaptive m driven by η², since 80% of lattice cells are FULL.

### F2. By the owner's own taxonomy, GLMO is fallback A/B; adopting it as the main line breaks decision 8

**What decision 8 says.**
- Fallback A: "each layer computes exact residual r = f − Ku; a geometry-conditioned linear net gives the correction".
- Fallback B: "route-1 skeleton (paired smoother, coarse spaces, Chebyshev) with a few geometry-generated coefficients".
- The fallbacks are to be used "only if the main line fails".

**What GLMO is.**
- It computes 8 exact fine residuals per forward (21 fine K per symmetric S q).
- It learns about 0.2 M *bounded gates* on a smoother, coarse-space and Chebyshev skeleton.
- §3.l concedes that m ≥ 4 "is exactly fallback A", and F1 says m ≥ 4 is where it will land.
- Question d allows "a few K applications inside". 21 per S q, with all learning confined to damping gates, is not a
  feedforward or GNO operator in the owner's sense.
- This is a governance failure, not a technical one. The owner rejected paradigm-copying and wants operator learning.

**Fix.**
- Present GLMO honestly as the **shared chassis plus fallback B**. The untrained, zero-initialized network is literally
  fallback B on a better hierarchy, which is valuable.
- Put a genuinely learned **linear multiscale GNO core** into the same chassis as the main-line candidate. One option:
  - L_θ(g) = Σ_ℓ P̃_ℓ(g) C_ℓ(g) P̃_ℓ(g)ᵀ, with learned rigid-preserving transfers and learned channel mixing on
    levels 1–2, plus the exact level-3 inverse;
  - wrapped in a **2–4 K exact symmetric sandwich** N = Ŝ + (I − ŜK) L (I − KŜ). This is symmetric whenever L is,
    and L = BᵀCB is symmetric by construction [RS 3.3].
- Then Phase 0, A and B compare the main line against fallback A/B on identical infrastructure, and the owner decides
  with data.

---

## 2. Serious issues

### S1. The sliver carrier covers far less than claimed, and is combined additively (route 1's failure mechanism)

**(a) Coverage arithmetic is wrong.**
- §2.5.3 claims any mode of 11–30 nodes "lies inside at least one block unless it spans more than about 5 nodes".
- With blocks of 5 nodes per axis and two families offset by 2 nodes *simultaneously in all axes*, containment is
  guaranteed only for bounding boxes whose extent is ≤ 2 nodes in at least two axes.
- `_verify/patch_cover.py`, fraction of positions contained in one block:

  | bounding box (nodes) | 2×2×3 | 2×3×3 | 3×3×3 | 3×3×4 | 2×4×4 | 3×4×4 | 5×5×2 (oblique sheet) |
  |---|---|---|---|---|---|---|---|
  | contained | 100% | 88% | 72% | 56% | 50% | 38% | 12.5% |

- The situation is worse than isolated slivers. Weak nodes **percolate into giant sheets** (`weak_clusters.json`):
  - FULL: 2 clusters of about 10k nodes;
  - cut 0.758: one of about 16k;
  - cut 0.148: 3 of more than 64 nodes.

  So block boundaries cut the slow-mode support everywhere.
- The §5 sentence "patch coverage of the weak set is coverage of the slow band" is false for an additive one-level
  method. What matters is containment of a mode's support, not coverage of its nodes.

**(b) Additive composition inside a globally scaled smoother.**
- S₀ = ΩD⁻¹Ω + Σ_p Π_p is scaled as a whole by s₀ = 1/(1.1 λ̂_max(S₀K)).
- The two overlapping families plus Jacobi give λ_max(S₀K) ≈ 2–3. So every exact local inverse is damped by 2–3×.
- This is the "global step crushes exact blocks" mechanism of route 1 [RP, +14% λ_min]. σ_p ≤ 2 cannot undo it,
  because λ_max grows with σ_p.

**(c) Sizes are underestimated** (`_ol/patch_counts.json`).
- Weak nodes plus their ring have a median of 45–75 nodes and a p90 of 97–109 nodes, against "typically 40".
- Storage is Σ ≈ 192k nodes × 3 × 16 × 4 B ≈ **37 MB**, not 23. The eigh cost is ≈ 2e11 FLOP, not 5e10.
- Batched `eigh` at n = 135–330 is often not batch-efficient on CUDA.

**Fix, in order of preference.**
1. **Multiplicative symmetric patch sweeps.**
   - Pre-smoother: gated Jacobi-Chebyshev → exact residual → family 𝔅₀ local solves at full weight (within a family
     the blocks are disjoint, so there is no damping) → residual → 𝔅₁.
   - Post-smoother: the reverse order, which keeps 𝒱 symmetric.
   - Cost: ≈ +2 restricted residuals per sweep, about 0.75 K_eq each (patches touch 75% of elements), so ≈ +3 K_eq per
     V-cycle.
2. **Better coverage.**
   - Use vertex-star patches, i.e. 2×2×2-element blocks centred at every weak Q1 vertex, which gives all 8 per-axis
     offsets: every 3×3×3 box is then contained. Colour them into 8 disjoint families for the multiplicative sweep.
   - Or put PU-weighted patch eigenvectors into the level-1 coarse space (GenEO-style [RS 5.7, 7.3]), so straddling
     modes are glued by the Galerkin solve.
3. Replace full `eigh` by batched Cholesky of K_pp plus 6–8 steps of block inverse subspace iteration (k = 16–24),
   using only well-supported batched kernels. Store V_p in fp16.

### S2. Rank saturation of the patch filter is likely, and it breaks continuity and equivariance (§4.6, §4.9)

- θ_p reaches 0.3 in Jacobi-scaled units, on patches of 135–330 DOF.
- A clamped 5³-node elastic patch has on the order of 10–30 scaled eigenvalues below 0.3, before counting sliver modes.
- Once more than 16 eigenvalues lie below θ, the filter is no longer a spectral function. It then depends on the
  eigensolver's choice inside a degenerate or near-degenerate cluster.

**Fix.**
- Use θ_eff,p = smoothmin(θ_p, λ_{p,k}), which is continuous because λ_{p,k} is.
- The taper then vanishes at the k-th eigenvalue by construction.
- Monitor how often θ_eff < θ_p. That rate is the true capacity signal.

### S3. The smoother bound is not a bound: invariant Lanczos underestimates λ_max, and training uses stale bounds

- **The start vector is not invariant.** §4.6 proposes the start vector ρ̄·1. It is not O_h-invariant, because
  reflections flip vector components.
- **An invariant start vector underestimates λ_max.** Any start vector that *is* invariant confines the Krylov space
  to the symmetric sector. For symmetric geometries (FULL parents, uniform-τ lattices) λ_max(S₀K) is then
  **underestimated**. With the 1.1 margin, λ(Ŝ₀K) > 2 becomes possible. Then:
  - 𝒱 is no longer convergent;
  - E_V is no longer PSD;
  - the §4.7 indicator bound and the §4.8 BDD constant are both false.
- **Training uses stale bounds.** Bounds are refreshed every 200 steps. Meanwhile ω ∈ [0.5, 2] can move ΩD⁻¹Ω by up
  to 4×, so the stale bound can be exceeded between refreshes. This is a source of training instability.

**Fix (it also removes the equivariance problem).** Use a **Loewner-monotone parameterization**:
- Cap every gate at ≤ 1: ω ∈ (0, 1], σ_p ∈ (0, 1], and level gates ≤ 1. Move the upward freedom into one per-level
  global scalar that is bounded explicitly.
- Then S₀(gates) ≼ S₀(1), so λ_max(S₀(gates)K) ≤ λ_max(S₀(1)K).
- Compute that bound **once per geometry**, independent of the gates:
  - Lanczos from 2–3 random start vectors;
  - plus the residual-based upper bound θ_k + ‖r_k‖, which makes it rigorous.
- The equivariance leak is then confined to a few scalars at the 1e-3 level, which is harmless.

### S4. Sensitivity, the gate route 1 barely passed, is under-weighted by the training signal

- Route 1 at 24 layers: sensitivity 2.6% against compliance 1.7%. **Sensitivity is the binding gate.**
- ∂K/∂τ = Σ ∂w·(BᵀCB) lives on elements crossed by |φ| = τ: 57% of FULL elements are cut [DI 2.1].
- On sliver elements, (∂v_e/∂τ)/v_e ≫ 1. The energy loss weights a local field error by v_e; the sensitivity weights
  it by ∂v_e/∂τ.
- So an energy-accurate field can be sensitivity-inaccurate exactly on the fringe, including on outside-material
  nodes, which are 35–55% of active nodes.
- ℓ_sens as proposed uses element strain energies a_e, which are volume-weighted. That is the wrong weighting.

**Fix.**
- ℓ_sens = ‖g(û) − g(u*)‖² / ‖g(u*)‖² on the 8-vector g_c = −Σ ∂w_{e,q} ε_qᵀCε_q. It is cheap, because ∂w is
  computed per step anyway.
- Add an element-level smooth proxy Σ_e |∂w_e|·(δ energy density)².
- Evaluate on force-driven and lattice-like samples.
- Report sensitivity error by element class (full, cut, sliver).

### S5. The masked Γ-band coarse space degrades the primary mode, and the correction formula is incomplete

- **Energy penalty of masked coarse functions.** P̄ = M_I P zeroes coarse functions on an oblique band, which creates
  kinks one fine element wide. The energy of a truncated coarse function near the band grows roughly as H/h: about 2,
  4 and 8 at levels 1, 2 and 3. This lands exactly where the cut slivers live.
- **Incomplete correction.** The A^X formula corrects only K_e. Ghost faces whose 45-node stencil touches Γ need the
  same M_I masking.
- **Storage underestimate.** For 0021 about 550 level-1 elements are affected, at 81² × 4 B each, so ≈ **14 MB**,
  not ≤ 5 MB.

**Fix.**
- Use a band-local smoothed prolongation: one Jacobi step of K restricted to the elements touching Γ, applied to
  P̄'s columns at encode time. It costs about 0.15 K_eq per transfer if done online, or it can be folded into the
  stored corrections.
- Include ghost faces in the correction.
- Measure the two-grid contraction in X-band vs X-box in Phase 0.
- Default the lattice to X-box for free cuts (risk 7 already points there).

### S6. Topology fragility inside a design band

**(a) Frozen patch membership goes stale.**
- Membership is frozen per ±2% band. Yet ±1% changes 35–73 active elements [RP 7.5], so new slivers appear that are in
  no patch.
- Fix: define membership on the superset with a band-robust weakness test (weak at either band end), or use smooth
  per-node weakness weights inside fixed block membership.

**(b) Identity rows must not enter the q-path.**
- The superset's "identity rows" for decoupled nodes have unit diagonals. These are orders of magnitude off the K
  scale. They are also inconsistent with the moment-built Galerkin coarse operators, which contain no identity rows,
  so the V-cycle stops being Galerkin.
- Fix: exclude those nodes everywhere with the node-on mask (in P̄, D, S₀ and all vectors).

**(c) G_θ features diverge at activation.**
- log(vf) and log‖D_i‖ go to −∞ as an element or node activates.
- Fix: clamp them smoothly, e.g. log(x + ε) with ε at the element-error scale of about 1e-4.

**(d) Box-port sets change inside a band** (0021: ±28–34 box nodes at ±1% [RP 7.5]).
- The mode masks handle this. But lattice gluing and BDD Φ have to be rebuilt when it happens, and that cost is not
  in the budget.

### S7. Encode and storage estimates are low (not fatal, but R12 and R14 become tight)

| item | design | corrected |
|---|---|---|
| patch spaces | 23 MB | ≈ 37 MB (+ X variants) |
| band corrections (cut) | ≤ 5 MB | ≈ 7–14 MB (half-stored or full) |
| FULL record | ≈ 75 MB | ≈ 90 MB |
| patch eigensolve | 5e10 FLOP, 0.05–0.2 s | ≈ 2e11 FLOP; wall time unknown (batched eigh at n ≈ 200–330) |
| moments + ∂M/∂τ per design iteration | 80–100 s | ≈ 170–180 s: ∂M/∂τ is another full moment pass [DI §4], 88 s + 88 s |

- Fix: microbenchmark (risk 6) and switch to subspace iteration (S1).
- A₂ can share one BSR across modes X-box and N by masking rows. Say so, or budget a second copy.

---

## 3. Minor issues (each with fix)

1. **C_e feature is redundant.** The element feature eig(C_e) with C_e = [ε_iᵀK_eε_j] is vol_e·eig(C) for uniform
   strains, so it carries only volume.
   - Fix: use Rayleigh quotients on linear-strain (quadratic-displacement) modes, or second-moment invariants.
2. **The lattice-level indicator is not free.** BDD applies S_hat to search directions, not to q̂.
   - Σ_c η_c² at the solution needs one more forward and adjoint pass per cell (≈ 32 K_eq; about 1% of the
     iteration).
   - Power iteration gives a *lower* estimate of ρ, so η²/(1 − ρ²) is an estimate, not a bound. Label it so.
3. **The main loss needs no label.** ∇ log μ = ∇ log(ûᵀKû), because qᵀSq does not depend on θ.
   - The exact scalar is needed only for CVaR ranking and monitoring. Exploit this for label-light training in step 2.
4. **The Neumann training factor is wrong.** "cuDSS factor of K + αRRᵀ" is not sparse (RRᵀ is dense).
   - Fix: pin 6 DOF (3 non-collinear nodes), then project out rigid modes, or use Woodbury on the pinned factor.
   - Also deflate rigid modes in the LOBPCG pencil (S_hat − S, S).
5. **Coarsest Cholesky precision.** fp32 Cholesky of A₃ in cut cells (weak-node contrast, γ = 1e-4) risks losing
   definiteness.
   - Fix: make fp64 with diagonal scaling the default. It costs about 3.5 GFLOP per cell per design step, a few ms.
6. **The patch term ignores the gates.** It assumes "Jacobi acts as 1", but with gates Jacobi acts as ω², so the
   patch no longer gives the exact local inverse.
   - Fix: form Ǩ_p in the gated scaling (ΩD^{-1/2}), so the property holds for any gates.
7. **Symmetry defect.**
   - The rigid-exact readout protects v_Γ. It does not protect K_ΓI N v_I, whose partner b is lift-sized (1e3–1e5
     times the energy).
   - Fix: measure the symmetry defect and the S_hat error on the 64 softest exact directions, whitened, not raw
     (extend risk 5).
8. **AΦ is charged per load case** in the budget. It is per design iteration, so the budget is conservative.
9. **§2.1 typo.** u = Rα + [q; x + (Rα)_I] double-counts Rα. It should read u_Γ = q, u_I = x + (Rα)_I.
10. **G_θ uses face adjacency of active elements.** That can link walls that are disconnected in material.
    - Fix: use K-connectivity (shared active element or ghost face). This matters only on the geometry path, so the
      impact is low.

---

## 4. Arithmetic audit

| quantity | design | check |
|---|---|---|
| message and update MLPs: 12.5k + 12.4k per layer, 6 layers ≈ 150k; G_θ ≈ 0.6 M | ✓ | ✓ |
| GNN 2.4 GFLOP per layer (96k edges × 25k MAC) | ✓ | ✓ |
| w: E×125×4 B = 6.4 MB; D⁻¹ 2.9 MB; A₂ BSR ≤ 17.5 MB; A₃ packed 2.1 + 9.6 MB | ✓ | ✓ |
| K apply: 0.37 body + 0.64 ghost (4·54·135·F) GFLOP; ≈ 30 MB of traffic per column → 0.05–0.1 ms batched is plausible | ✓ | ✓ (atomics and ragged batching are the risk) |
| forward 15.5 K_eq (my recount 14.3); S q 32; BDD 3,356 K_eq per cell; 15–30 s per 100 cells | ✓ at m = 2 | ✗ as a forecast (F1 table) |
| symmetric-readout identity, N₂ = 2𝒱 − 𝒱K𝒱, (1 − ρ²) bounds, C − Ĉ ≥ Σ η² | ✓ | ✓ (the proofs hold, given S3's fix) |
| (K_f⁺)_ΓΓ = S⁺ on the balanced space with Π on both sides | ✓ | ✓ (checked, including rigid gauge) |
| Prop. 2.4 coarse moments (8 binomial 125×125 maps, (h/H)² scale) | ✓ | ✓ |
| patch storage / eigh FLOP / coverage | 23 MB / 5e10 / "≤ 5 nodes" | ✗ 37 MB / ~2e11 / fully contained only for extent ≤ 2 nodes in at least two axes (S1) |
| band correction ≤ 5 MB; moments + ∂M 80–100 s | | ✗ ≈ 14 MB; ≈ 175 s |
| training: 50 GFLOP per column, 3.2 GB of activations per 64 columns | ✓ | ✓ |

---

## 5. Component-by-component verdict

| component | verdict | note |
|---|---|---|
| strain-form matrix-free K from signed 5³ Gauss weights | **keep** | exact; 125 numbers per element |
| fp64 rigid split + exact port overwrite (box ∪ band) | **keep** | exact ports and rigid modes, linear |
| symmetric N_m; S q = Π[v_Γ − K_ΓI N v_I]; no stored activations | **keep** | the best idea in the proposal |
| rigid-exact mixed-precision readout | **keep** | extend the tests to the adjoint term (minor issue 7) |
| nested Q2 Galerkin hierarchy via coarse moments and ghost sub-face templates | **keep** | verify F·P_s = 0 (risk 4) |
| masked-P̄ Γ-band corrections | **modify** | include ghost faces; band-smoothed transfer (S5) |
| exact dense coarsest solve (9³) | **keep** | fp64 by default |
| gated Jacobi ω ∈ [0.5, 2] | **modify** | cap at ≤ 1 for a Loewner bound (S3) |
| additive patch spectral carrier on 2 families | **modify heavily** | multiplicative sweeps, 8-offset coverage or GenEO coarse enrichment, θ_eff, band-robust membership (S1, S2, S6) |
| multichannel polynomial smoothers on levels 1–2 | **keep** | but not enough as the only learned capacity (F1) |
| learned transfers / fine gated-K channel | **add** | the capacity that can add directions (F1, F2) |
| G_θ trunk (invariant element MP + coarse U-graph) | **keep** | fix features (minor issue 1, S6c) |
| m = 2 as the design point | **drop** | replace with m chosen by Phase 0 and adaptive m per cell |
| Lanczos with "invariant" start vector, refreshed every 200 steps | **drop** | per-geometry bound at gates = 1 (S3) |
| mode-N V-cycle Neumann action | **keep** | its constant is valid only after the S3 fix |
| balanced BDD with stored AΦ (1 S q + 1 M per iteration) | **keep** | |
| η² indicator | **keep** | call it an estimate; +1 pass at lattice level |
| physical sensitivity with ∂w = V⁻¹∂M, ∂K_ghost = 0 | **keep** | |
| ℓ_sens on element energies | **modify** | use the ∂w-weighted sensitivity vector (S4) |
| superset masks / design bands | **keep** | no identity rows in the q-path; band-robust patches (S6) |
| Phase 0 (zero learning) → Phase A (free gates) → Phase B | **keep** | Phase 0 is the right first experiment; add the main-line core as a Phase-B arm (F2) |
| X-band and X-box trained jointly, N evaluated | **keep** | |
| K + αRRᵀ exact Neumann factor | **drop** | pinned factor + projection |

---

## 6. Ideas to carry into a synthesis

1. **Self-adjoint extension operator.** N = p(𝒱K)𝒱 with 𝒱 symmetric makes S_hat = EᵀKE computable with the *same*
   operator twice.
   - No stored activations.
   - Symmetric PSD by construction.
   - A free per-query error indicator η² = v_Iᵀ N v_I with a proof.

   Any synthesis whose q-path is a symmetric linear operator inherits all of this.
2. **One record, three modes.** Masks switch between X-band, X-box and N. The Neumann action inherits SPD-ness and an
   explicit BDD constant through (K⁺)_ΓΓ = S⁺. This answers questions f and l in one construction.
3. **Coarse moments.** Exact Galerkin coarse operators on nested Q2 CutFEM spaces come from a binomial moment
   transform: matrix-free, rigid-exact at every level, no assembly.
4. **Rigid-exact mixed precision.** Subtract the element rigid fit in fp64 before fp32 strains. This avoids the fp32-T
   failure without fp64 compute.
5. **Balanced BDD with stored AΦ.** One S q and one M per iteration; this halves lattice queries.
6. **Zero-initialized heads around theory values.** The untrained network is a working solver (fallback B). Phase 0
   measures it before any training, and the fallbacks are "freeze" and "raise m", not rebuilds.
7. **Placement by memory traffic.** An exact fine K costs the same traffic as one learned fine channel and carries no
   error. So put exactness at the fine level and learned width on coarse levels, with the caveat from F1: the fine
   level still needs a carrier that can add directions.
8. **Physical sensitivity from ∂w = V⁻¹∂M** with ∂K_ghost = 0, and its deviation from the surrogate derivative
   bounded by the interior residual.
9. **(From this review.) Loewner-monotone gates** (≤ 1 times a per-geometry bound). They make smoother bounds
   gate-independent, rigorous and equivariant, and they remove stale-bound instability during training.
