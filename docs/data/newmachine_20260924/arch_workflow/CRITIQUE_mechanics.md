# Adversarial review of DESIGN_mechanics.md (MMX-Net)

Reviewed against: ARCH_BRIEF.md (goal, decisions 1–9, §3–§7), DATA_INTERFACE.md (DI), RESEARCH.md (RS), and the
repository evidence it cites (ROUTES_PROGRESS §route 1/2, OPERATOR_LEARNING_DESIGN, route-1 source
`hierarchy_strain_network_r1.py`, arch/`q2_hierarchy_counts.json`, `weak_clusters.json`, `weak_proxy.json`).
Tags: **[checked]** = I recomputed or read the source; **[est]** = my estimate.

Verdict in one line: the *readout algebra and the port/rigid handling are correct and should be kept verbatim*;
the *q-path as specified (P = 2 sweeps) will not pass the gate*, because the only argument that it can be that
shallow (GenEO-type mechanism enrichment lifting the dense soft band) is applied without the ingredient that makes
GenEO work: an exact joint solve over the enriched coarse space. The cost and throughput headlines (25 GFLOP,
≈70 s per design iteration, encode 0.3–0.4 s) are computed for that unsupported depth, and the encode estimate
is additionally ~10× optimistic.

---

## 1. Fatal issues (as specified, the design fails the gate or its headline budget)

### F1. The depth premise (P = 2 sweeps) has no quantitative support and is contradicted by our own data
- **What the design claims.** Forward = 2 V(2,2)-type sweeps with one exact re-entry, about 22 fine K-applies
  (§2.5, C8). All cost figures (25 GFLOP forward, 51 GFLOP reaction, 70 s per design iteration) assume P = 2.
- **Arithmetic [checked].** For a symmetric preconditioner with condition number κ, a degree-d polynomial
  (Chebyshev-optimal) reduces the K-norm error by about 2((√κ−1)/(√κ+1))^d.
  - Calibration on route 1: κ ≈ 1/λ_min = 1/0.0070 ≈ 143 (0013, E3 table). Reducing the error by the factor
    ≈ 0.022 needed to go from witness ~1e3 to ~1.5 takes **d ≈ 27** by this formula. Route 1 needed 40–48
    Chebyshev layers or 24 learned-coefficient layers to pass the lattice gate. The formula is right to within 2×,
    so it can be trusted for extrapolation.
  - For **d = 2** the same reduction needs κ ≲ 1.5 per sweep (T₂((κ+1)/(κ−1)) ≥ 45 ⇒ κ ≤ 1.53). That is textbook
    Poisson-multigrid quality, on a CutFEM elasticity problem with γ = 1e-4 fringe, a dense band of ≥128 slow
    modes, and 17–22% weak nodes. It is a **~100× improvement in κ** over route 1.
  - The design's own X0 pass criterion ("λ_min raised ≥ 5× over 0.0070", i.e. κ ≈ 29) implies **d ≈ 12**
    sweeps, not 2. The experiment ladder is inconsistent with the main line.
  - Lattice calibration: route-1 learned 16 layers (witness 6.8) failed at 6.8%/13.2%, 24 layers (witness 1.44)
    passed at 0.72%/1.7%. So the gate needs roughly witness μ ≲ 1.5–2, i.e. the full reduction above.
- **Why the learned gates cannot close a 100× gap.** In two-level theory, κ is set by the coarse-space
  approximation property (stable decomposition). Scalar gates on smoothers and 2×2 channel mixers change step
  sizes and polynomial coefficients. Route 1 shows what that buys: learned per-layer scalars ≈ 2× fewer layers
  than Chebyshev. Per-element step sizes might buy another ~2×. Neither changes what the coarse spaces can
  represent.
- **Consequence.** A realistic depth is P ≈ 8–16 (it must be measured). At P = 12: forward ≈ 145 GFLOP, reaction
  ≈ 290 GFLOP ≈ 20–30 ms per column [est]. That gives ≈ 100 × 160 × 25 ms ≈ **7 min per design iteration**, before
  the BDD inflation of S1. The owner already called 30 min "too slow". 7 min may be acceptable, but it is not
  the 70 s the design advertises.
- **Fix.**
  1. Make depth a first-class, measured quantity. X0 measures κ(B_V K) per geometry (Lanczos, both port modes)
     and the design sets P from the Chebyshev formula with a 2× margin. Budgets are quoted at that P.
  2. Put the depth-reducing effort where theory says it matters: the coarse-space approximation property. See F2
     (exact enriched coarse solve), S8 (L3 quadratic) and S7 (learned prolongation smoothing).
  3. Keep the exact-residual form of every sweep (Hsieh consistency). Then "more sweeps" is a smooth dial, not
     a redesign.

### F2. The mechanism enrichment is GenEO without GenEO's exact coarse solve, so the dense band is not lifted
- **What the design claims** (§4.8, §5 S2). The L1 mechanism columns get their amplitude in one block solve
  ("exact generalized force / exact generalized stiffness"). "GenEO-type theory" then bounds κ independently of
  contrast.
- **What the design actually does** (C4, C5, C7).
  - Mechanism columns exist **only at L1**, and L1 is treated **only by block Jacobi** over 2,080 PoU blocks,
    twice per sweep.
  - L2 is polynomial only: the nesting proof in C5 covers only the polynomial columns, so mechanisms are not in
    span(L2). L3 is affine.
  - So a mechanism is never part of any exactly solved coarse problem.
- **Why that breaks the argument.**
  1. GenEO's contrast-independent bound needs (a) an **exact solve over the whole enriched coarse space** and
     (b) local solves on the **same overlapping patches** where the eigenproblems were posed. MMX-Net has neither.
     Its fine smoother is ≤4-node pair / ≤24-node clusters, not 2³-element patch solves. Its L1 coarse problem is
     relaxed by block Jacobi. The cited bound does not apply.
  2. Even where it does apply, the GenEO constant is roughly (1+k₀)(2 + k₀(2k₀+1)/λ_c). With overlap
     multiplicity k₀ = 8 for 2³ blocks and λ_c = 0.05, that is ~2·10⁴: useless as a design guarantee.
  3. §4.8's "θ is small because a mechanism is weakly coupled" confuses two couplings: the coupling of the
     mechanism to its surrounding material (small), and the coupling between the **PoU pieces of the same
     mechanism in neighbouring blocks** (O(1)).
     - Slivers lie along the cut plane. That plane crosses the 16³ block grid obliquely, so a sliver mode of 11–30
       nodes routinely straddles block faces. Those are 2-element blocks of ~125 nodes, each with 35% of its nodes
       in the fringe.
     - χ_a z and χ_b z of a straddling mode have θ ≈ 1. Block Jacobi converges on such a pair at rate
       (1+θ)/(1−θ) → the band reappears as the slow modes of the L1 relaxation.
     - Route 1's weak-merge result already hints at this: λ_min rose only 14% (0.0070 → 0.0080) even with
       ≤32-node clusters (risk 1 in the design says the same).
- **Fix** (cheap, and keeps the design's best idea).
  1. Solve the enriched coarse space **jointly and exactly**. Form a "mechanism + global" coarse level
     M = span{all mechanism columns} ∪ span{L3}, Galerkin A_M = P_Mᵀ K P_M.
     - Size ≈ 2k blocks × ~3 + 768 ≈ 7k DOF (FULL); block-sparse with a 27-block stencil.
     - Sparse Cholesky by nested dissection on the 16³ block grid: the top separator is ~16² × 3 ≈ 770 dense DOF,
       total fill a few M entries, ≲ 20–40 MB fp32/fp64 [est]. It is formed at encode time and is not a large
       factorization.
     - Apply it once per sweep in place of κ3·A3⁻¹, as an additive or multiplicative exact coarse correction.
  2. Alternatively, or in addition, carry aggregated mechanism modes into L2: per L2 block, a K-orthonormal SVD
     of the restricted L1 mechanism columns. Then L2 relaxation and the L3 solve both see them.
  3. Use overlapping mechanism patches: 3³-element patches, or a second partition shifted by one element. Then a
     sliver mode lies wholly inside at least one patch (the design lists this only as a fallback).
  4. X0(b) must measure the right thing: λ_min of the **full sweep operator**, and the captured energy fraction of
     the slowest-128 (not 64) modes by span(M). Pass at ≥ 90% captured **and** κ small enough for the chosen P
     (F1).

---

## 2. Serious issues

### S1. One V-cycle as the BDD Neumann action inflates the BDD iteration count by ~√κ_V
- **Mechanism.** If c₁ S⁺ ⪯ M ⪯ c₂ S⁺ on the balanced subspace, BDD's κ is multiplied by c₂/c₁ = κ_V (restriction to
  ports is a congruence, so R_P B_V R_Pᵀ inherits κ(B_V K)). PCG iterations then grow by ≈ √κ_V.
  - With κ_V ≈ 30 (X0 target): 11–53 iterations → ~60–290.
  - With κ_V ≈ 143 (route-1 V-cycle): ×12.
- **Consequences.** The step-3 check "≲1.5× exact S⁺" cannot hold. The 55 Neumann + 160 forward applications
  per cell used in the 70 s estimate become 300–800 forward applications.
- **Fix.** Make the Neumann action a **fixed-degree Chebyshev polynomial of the symmetric V-cycle** (NEU mode):
  - interval from encode-time Lanczos with a fixed start vector;
  - degree chosen for an effective κ ≈ 1.5–2 (d ≈ 3–5 at κ_V ≈ 30);
  - it is linear, symmetric, and positive when the polynomial is positive on the interval.
  - Include the exact mechanism coarse solve of F2 in the NEU cycle: the floating cell's soft mechanisms are
    exactly what S⁺ amplifies.
  - Never use PCG or LOBPCG inside the action (they are nonlinear in the right-hand side).

### S2. The encode-time estimate for the mechanism eigenproblems is ~10–100× optimistic
- **What is claimed.** "2,080 × ≤450², batched, fp32 + fp64 polish ≈ 0.1–0.2 s".
- **Arithmetic** [checked FLOPs, est time].
  - A dense generalized eigen-decomposition costs ~9n³ ≈ 0.8 GFLOP at n = 450, so ≈ 1.7 TFLOP for the batch.
  - The tridiagonal reduction is BLAS-2/latency-bound. cuSOLVER's batched Jacobi eigensolver is limited to
    n ≤ 32. Looped syevd on GPU, or LAPACK on 16 CPU cores (~10–25 ms each), gives **≈ 1.5–10 s**.
  - Add the projection onto the complement of the 30 polynomial columns, Cholesky of the singular χKχ (χ = 0 on
    the halo), and the per-port-mode repetition (DBOX/DALL/NEU).
- **Consequence.** The brief's encode budget (≲ 0.5 s, "should beat fp32 factorization 0.08–0.85 s") is violated.
  Risk 8's mitigation ("only blocks with weak or cut elements") does little: weak nodes are 17–22% of nodes in
  FULL cells too, spread along the whole fringe (weak_proxy.json), so most shell blocks qualify.
- **Fix.**
  1. Pose the eigenproblem on the weak nodes + anchors + one element ring (≲ 150 DOF): n³ scaling gives 27×.
  2. Compute only the k ≤ 8–16 smallest eigenpairs by batched LOBPCG or subspace iteration with the D-scaled
     block as preconditioner.
  3. **Warm-start from the previous design iteration's columns.** τ moves little per design step, so 2–3 subspace
     iterations suffice. This also fixes the continuity problem of S4.
  4. Recompute only port-touching blocks per port mode.
  5. Microbenchmark before X0.

### S3. The Neumann patch matrix is route 2's failed "H = 1 without GP outer ring" construction
- **The construction.** Ã_p uses only the elements and ghost faces *inside* the patch. The halo is one node layer,
  so it contains no whole element. A ghost-face stencil spans 5 node layers, so every cross-block GP face is
  excluded.
- **Consequence.**
  - A sliver element whose only support comes from a cross-boundary GP face floats in Ã_p. It produces spurious
    near-zero eigenvalues: 6 rigid modes per floating fragment.
  - These modes crowd the k_max = 8 slots, are ill-conditioned in fp32, and are exactly the failure route 2
    recorded: "upper bound flew to 1e12" at H = 1 without the GP ring, fixed by H = 2 plus the GP outer ring.
- **Fix.**
  - Build Ã_p on the block plus **one full element layer**, including every GP face that touches a patch element
    and the element on its far side (route-2 recipe). Keep χ_a = 0 weights on the outer layer.
  - Either keep all kernel modes of genuinely floating pieces (GenEO requires them; then do not cap), or remove
    those already in span(polynomial columns).

### S4. Fixed k_max / frozen mode counts inside a dense near-degenerate band are discontinuous and non-equivariant
- **The problem.**
  - The slow band is dense: 128 modes in [0.0039, 0.0062]. Local pencils will likewise have clusters near λ_c.
  - "At most 8 per block", and later "count frozen per design band", both pick an **arbitrary subspace of a
    near-degenerate eigenspace**. When λ_k and λ_{k+1} cross inside a band, the span jumps: Ŝ and the network
    field jump in τ.
  - Under O_h stabilizers of a symmetric block the degenerate space is an invariant subspace. Truncating it
    breaks equivariance, and augmentation will not repair this.
- **Fix.**
  - Select by **spectral gap**: take all modes below the largest relative gap in [λ_c/2, 2λ_c], with no hard cap
    (cap only memory, by raising λ_c's tolerance per cell).
  - Track the selected invariant subspace continuously across design iterations (warm-started subspace
    iteration, S2).
  - Close the selection under the block's stabilizer.
  - Note that the block-inverse formulation is already gauge-invariant within a span, so only the span must be
    continuous.

### S5. Sensitivity, not compliance, is the binding gate, and the design's fix for it is thin
- **Evidence** [checked, ROUTES lattice table]. Sensitivity error / compliance error = 1.3–1.6 in every route-1
  row (0.93/0.72, 2.6/1.7, 8.9/5.7).
- **Why.** The §4.5 theorem bounds only compliance. Sensitivity is first order in e, and it is concentrated on
  elements crossed by the shell surface and the plane. Those are the slivers, the weak fringe and the mechanisms:
  where the network is weakest and where ∂vf/∂τ / vf is largest.
- **The design's §3g "K-orthogonality" remark does not help.** ∂K/∂τ is surface-localized and has no meaningful
  "part proportional to K". The usable bound is |δs_c| ≤ 2‖u*‖_{|∂K|}‖e‖_{|∂K|} + ‖e‖²_{|∂K|}.
- **Fix** (cheap, and exploits the problem structure).
  1. **One deep polish per cell per design iteration.** Sensitivity needs one field per cell per design iteration,
     not 2e4 applications. After the lattice solve, re-extend q̂_i with P_deep = 16–32 exact-residual sweeps
     (fallback-A iteration, same modules): 100 cells × ~0.1–0.2 s ≈ 10–20 s. Compute s_c from that field.
  2. **A-posteriori compliance correction.** With the same deep field, C ≈ Ĉ + Σ_i q̂_iᵀ(Ŝ_i − Ŝ_deep,i)q̂_i. This
     moves the compliance error from first order in Δ to roughly second order.
  3. **Make the targets consistent.** Target lattice compliance ≤ 1.5–2% so that sensitivity ≤ 3% at the observed
     ratio. Keep the surface-weighted sensitivity loss, and add a D-norm field term restricted to
     surface-crossed elements.

### S6. Gating Ψ (the operator), not B (the step), breaks the stated safety properties and adds a near-gauge
- **The problem.**
  - With Ψ_j = Σ a_e K_e + γΣ b_f G ≠ K, a smoother step x ← x + B(r − Ψx) no longer reduces the K-energy error,
    and a sweep mixing Ψ (smoothing) with K (restriction, re-entry) is **not symmetric**.
  - §4.7's safe sub-family ("0 < B_V K ≤ I; adding sweeps never increases the energy error") therefore holds only
    with a_e = b_f = 1. §7.3's claim "every added sweep provably reduces the energy error" is false for the main
    line.
  - The per-element Ψ gates and per-cluster ω_c gates are nearly redundant: a uniform scale on the elements
    around a cluster trades against 1/ω_c. This gives flat directions and poor conditioning for the GNN.
- **Evidence.** Route 1's identified next lever was "per-block smoothing damping", i.e. B, not a modified
  operator.
- **Fix.** Keep Ψ = K exactly: strain form, no gates. Put all spatial step-size freedom on B:
  - ω_c per cluster, or ω_i I + β_i D̂_i per node (capacity knob 2);
  - per-element-group damping, implemented as a scaling of the cluster inverse rather than of K_e.

  Then every sweep is a consistent smoother on the exact residual, the symmetric sub-family is honest, and the
  fallback-A dial is monotone.

### S7. Decision 8 tension: the q-path has no learned kernel, so its capacity ceiling is fallback B
- **The problem.**
  - The owner chose a feedforward NO + multiscale GNO as the main line, with fallbacks A and B.
  - MMX-Net's q-path uses only exact K_e, exact cluster/block inverses and exact Galerkin blocks, scaled by
    scalar gates, with 2×2 or 4×4 shared mixers. With gates = 1 it *is* fallback B, and adding sweeps turns it
    into fallback A.
  - The only learned objects are ~0.4M positive scalars. If they cannot compress 24 cycles into P, there is no
    kernel capacity to add: every escalation in §3j is "more of B/A".
- **Fix.** Add one learned, K-structured, depth-reducing channel: **learned prolongation smoothing**
  (Luz/Greenfeld, energy-minimizing basis).
  - Set P̃_ℓ = (I − Ω_ℓ(g) D⁻¹K) P_ℓ, with a geometry-generated per-node (or per-cluster) Ω. Optionally take two
    steps.
  - It is linear in q, rigid-preserving (K R = 0 ⇒ P̃ keeps the global rigid fields), exactly equivariant with
    invariant Ω, and costs one K apply per basis at encode time. It is stored as column values: L1 ≈ 2080 × 38 ×
    ~350 nodes × 3 ≈ 83M floats, too much, so apply it matrix-free per sweep (two extra K applies), or only
    at L2/L3.
  - This is the lever that improves the approximation property, i.e. κ, i.e. P (F1), and it gives the
    "neural operator" part substance.
  - Also allow learned (C₂ = 4–8) channel mixing at L2/L3, where it is cheap.

### S8. The coarsest level is affine, which contradicts the design's own locking argument
- **The problem.** The design skips Q1 because trilinear fields lock in bending, then uses **affine** fields at L3.
  So the only exactly solved level cannot represent quarter-cell bending, which is the neck/saddle bending load
  path of Schwarz-P.
  - Bending is left to L2 (quadratic), which is only relaxed (2 + 2 layers moving about 1 block per layer on an
    8³ grid).
  - Global bending-dominated transmission therefore depends on sweep count.
- **Fix.** Make L3 quadratic: 64 × 30 = 1,920 DOF, dense fp64 inverse 29 MB, or upper Cholesky factor ≈ 15 MB
  [checked]. Nesting still holds, since quadratic × χ_{A3} = Σ χ_{A2}·quadratic ∈ span L2. Or merge it with the F2
  mechanism coarse solve. Handle rank deficiency (few-node blocks in heavy cuts) with a regularized
  pseudo-inverse, not a plain Cholesky that assumes full rank.

### S9. Risk 5's fallback ("condense the band with a few PCG steps") violates decision 5
- **The problem.** Krylov iterates with a fixed step count are **nonlinear in the right-hand side**: the
  polynomial depends on b. So the cell operator would no longer be linear in q, and would not be symmetric for
  BDD.
- **Fix.** Use a fixed Chebyshev polynomial (linear, symmetric), or simply use DALL as the canonical cut-cell mode
  (owner's formulation: Ŝ on box ∪ band) and let the lattice carry the band as cell-local interface unknowns with
  f_band = 0.

### S10. The throughput headline also omits the BDD coarse-space setup and the port-mode duplication
- ROUTES 7.7 lists **up to 42 forward columns per cell** at the start of every global solve (coarse-space setup).
- Cut cells need DBOX *and* NEU (and DALL if loaded) hierarchies. Each has its own clusters near ports, its own
  eigenproblems and its own Galerkin blocks.
- Neither affects feasibility, but both belong in the budget table. With S1 and F1 they move the realistic design
  iteration from 70 s to **~5–15 min** [est].

---

## 3. Minor issues

1. **Cluster storage arithmetic is internally inconsistent.**
   - "6.5M + 10M floats ≈ 35 MB": 16.5M floats is 66 MB.
   - Upper bounds: 45k clusters × 78 (≤12 DOF, packed) = 3.5M; 2k × 2,628 (≤72 DOF) = 5.3M. The total ≈ 8.8M ≈
     35 MB, so the stated MB is right and the components are wrong.
   - Also uncounted: the per-block 30×30 orthonormalization transforms (≈ 7.5 MB).
2. **Misquote.** "Free learned element kernels would need 81×81×C² per element (211 MB at C = 64)". DI's 211 MB is
   a C×C mixing per element (12.9k × 64² × 4 B). 81×81×C² per element would be ~1.4 TB.
3. **Dense parity packing vs active FLOPs.**
   - The design adopts the dense 33³×8 layout but quotes active-element FLOPs. Dense costs 2.54× for the body
     and 4.34× for the ghost term (95,232 vs 21,936 faces) [checked], so ≈ 3.7 GFLOP per apply, not 1.0.
   - Cut cells would be 3–19× wasteful.
   - Fix: use compacted, Morton-ordered active element and face lists at the fine level; dense layout only at the
     coarse levels.
4. **The training-pool size is underestimated.** 4k fields per geometry is ≈ 5.9 GB (FULL alone, fp32) and
   ≈ 10.3 GB for all three, not 6 GB. fp64 is ≈ 21 GB. It fits 64 GiB only in fp32.
5. **"48× data at zero solve cost" is illusory for a model that is equivariant by construction.** The rotated
   loss equals the original loss. Augmentation only exercises the tie-break and Kuhn-asymmetry residue. Use it
   as an equivariance test, not as a data multiplier.
6. **Channel mixers are "shared, g-independent" (§2.4) but initialized from a per-geometry Lanczos Chebyshev
   interval.** Route 1 shows the step must be per-geometry: the row-sum bound was 2–3× off. Feed the
   deterministic per-geometry λ_max(BK) and λ_min estimates (fixed start vector, ~20 K-applies at encode) as
   explicit normalizers. Do not ask a 3-round local GNN to infer a global spectral quantity.
7. **Hard thresholds in the encoder:** weak flag at 1%, strength θ = 0.1, chunk ties, and **rank-dropping of
   near-dependent polynomial columns** (not in the list of items frozen per band). Freeze them all per superset
   band. Replace column dropping by a smooth regularized pseudo-inverse of the block Gram and Galerkin matrices.
8. **Weak-cluster size.** Chunking at ≤24 nodes is smaller than route 1's best-tested ≤32 (witness 1.13). Weak
   nodes form giant connected sheets (weak_clusters.json: two ~10k-node components in FULL), so chunk boundaries
   are everywhere. Start from route 1's ≤32, anchored.
9. **Cluster factors.** fp32 inverse-Cholesky of anchored weak clusters: the D-scaled κ reaches ~1/λ_mech. Factor
   in fp64 and store fp32 only after a residual check. This is harmless for the energy readout, which is second
   order, but it matters for sensitivity (S5).
10. **Band loads in DBOX.** Define the load vector variationally (∂Π̂/∂q) and the compliance as −2Π̂*. That keeps
    the q–q block PSD and Ĉ ≤ C. Any (Kû)_box-style reading of band loads is non-symmetric. Make DALL canonical
    (owner: "Ŝ on box ∪ band"). Treat DBOX + f_band = 0 as a cost optimization, not a different operator
    contract.
11. **GNN "port type" feature.** It must be face-agnostic (a count of box faces, band yes/no). A face identity
    breaks O_h invariance.
12. **Adversarial LOBPCG in step 2.** It needs a factorization per geometry per round: at 300 geometries, subsample
    ~10–20 geometries per round.
13. **"≈ 20 min for the exact pipeline"** has no source. ROUTES 7.4 quotes 30 min (100 cells). Cite or drop.
14. **§4.8's O(θ²) amplitude claim.** It holds only for coupling to *other-block* columns and the fine remainder.
    Same-block polynomial coupling is already exact in A1_aa. State this precisely (it matters for F2).

---

## 4. Component-by-component verdict

| Component | Verdict | Note |
|---|---|---|
| C0 port modes DBOX / DALL / NEU on one mask set | **keep, modify** | DALL canonical (owner contract); DBOX = optimization for f_band = 0; band loads variational (minor 10) |
| C1 rigid split (fp64 projector, exact R α) | **keep** | correct; proof 4.2/4.3 checked |
| C2 port force f0 = −K_{𝕀ℙ} q_d | **keep** | exact, local, cost independent of port DOF (1,776 elements FULL, checked) |
| C3 fine K-structured hyperedge layers | **modify** | Ψ = K exact (no a_e, b_f); gates on B only (S6); compacted active lists (minor 3) |
| C3 anchored clusters | **keep, modify** | route-1 size ≤32; fp64 factorization (minor 8, 9) |
| C4 L1 PoU polynomial fields (rigid + membrane + bending) | **keep** | skipping the Q1 level is well argued |
| C4 L1 mechanism enrichment (local generalized eigen) | **keep the idea, rebuild it** | GP outer ring and element overlap (S3); gap-based selection with continuous tracking (S4); cheaper eigensolves with warm start (S2); **exact joint coarse solve** (F2) |
| C4 L1 block-Jacobi-only treatment | **drop as sole treatment** | replace by, or supplement with, the exact mechanism coarse solve |
| C5 L2 quadratic, explicit block-sparse A2 | **keep, modify** | add aggregated mechanism modes, or rely on F2's joint level; learned C₂ mixing |
| C6 L3 affine dense inverse | **modify** | quadratic (1,920 DOF, ≈ 15–29 MB), merged with the mechanism coarse level (F2, S8) |
| C7 sweep with exact restriction residual | **keep** | good: coarse levels see the true out-of-balance force |
| C8 P = 2 sweeps + one re-entry | **modify** | P from measured κ (F1); expect 8–16; exact residual at every sweep |
| C9 invariant geometry GNN, softplus heads, zero-init | **keep, modify** | add per-geometry spectral normalizers (minor 6); face-agnostic features; outputs step-size gates + learned prolongation weights (S7) |
| C10 energy readout (strain form, fp64 accumulation, rigid-free) | **keep** | the strongest part of the design |
| C10 reaction Π_Rᵀ E_dᵀ K E_d Π_R via explicit adjoint | **keep** | symmetric PSD for any parameters (checked) |
| C10 sensitivity from network field | **modify** | compute from a one-column deep polish per cell (S5) |
| Neumann action = 1 symmetric V-cycle | **modify** | Chebyshev polynomial of V-cycles + mechanism coarse solve (S1) |
| §4.5 lattice compliance bound | **keep** | correct (operator convexity of X⁻¹; weights sum to 1); add that it does not cover sensitivity |
| §4.7 safe sub-family / §7.3 "monotone in P" | **modify** | true only with Ψ = K (S6) |
| Training: μ−1 + D-norm + sensitivity loss, force-driven via one Neumann solve | **keep** | fix pool size (minor 4); per-class top-10% μ cap from the start |
| Risk-5 PCG band condensation | **drop** | nonlinear in q (S9) |
| X0–X4 ladder | **keep, modify** | pass criteria consistent with the chosen P; measure captured slowest-128 energy in span(M) |

---

## 5. Best ideas to carry into a synthesis

1. **Rigid-exact variational readout algebra.** Ŝ = Π_Rᵀ E_dᵀ K E_d Π_R never applies K to a rigid field, and the
   strain form (w_e = V⁻¹M_e, exact to 1e-15, rigid-free strains) gives an fp32 gather with fp64 accumulation.
   The result is symmetric PSD, exactly rigid-null and one-sided for any parameters. It closes the 12×
   rigid-residual failure mode.
2. **Port-force entry f0 = −K_{𝕀ℙ} q_d.** It is the mechanically exact way a displacement BC loads the body:
   local, linear, cost independent of the number of port DOF, no lift and no port encoder. It satisfies
   decision 2 structurally.
3. **K-structured hyperedges on K's own sparsity.** Element (27-node) and ghost-face (45-node) gather/scatter give
   no edge lists and respect material connectivity in exactly CutFEM's sense. Exact O_h equivariance comes with
   invariant scalar coefficients.
4. **Geometry-generated local mechanism enrichment** (GenEO / spectral AMGe from exact moments). Coarse capacity
   grows where thin, cut or weakly supported material is. It is the constructive answer to "fixed-geometry coarse
   bases capture only 72%", **provided** the enriched space is solved jointly (F2) and built on GP-ring patches
   (S3).
5. **Span-based (gauge-free) coarse treatment.** Block inverses over spans make eigenvector sign and rotation
   gauges irrelevant. Only span continuity matters, and that can be engineered (S4).
6. **The lattice compliance theorem.** 0 ≤ (C − Ĉ)/C ≤ Σ_i w_i(μ_i − 1) at the lattice's loaded directions, with
   Σ w_i = 1. It unifies loss, sampling mix and gate for compliance. It needs a companion statement for
   sensitivity (S5).
7. **Exact-residual sweeps.** Exact residual at restriction and at re-entry, initialization at fallback B, and
   P as a continuous dial to fallback A, with the symmetric sub-family giving monotonicity once Ψ = K.
8. **Force-driven data from one free-cell Neumann solve.** K u = [f; 0] gives (q, u*) at once, and O_h labels come
   free by signed permutation.
9. **PoU aggregate hierarchy with shell kinematics.** Rigid, membrane and bending fields, exact nesting, and a
   deliberately skipped Q1 level (locking). Carry it with a quadratic coarsest level.
10. **Mechanics diagnostics.** Energy-error localization by node class (weak / strong / port-adjacent / band) and
    by block, Lanczos λ_min of the sweep, and gate saturation statistics. They tell *which carrier* fails.
11. **New ideas from this review worth adopting in any synthesis.**
    - A one-column deep polish per cell per design iteration, for sensitivity and a first-order compliance
      correction.
    - Warm-started mechanism subspaces across design iterations, for cost and τ-continuity.
    - Learned prolongation smoothing as the depth-reducing learned channel.
