# Adversarial review of DESIGN_operator-learning.md (HGM-NO, "Hyper-Galerkin Multilevel Neural Operator")

Reviewer lens: find what makes HGM-NO fail the lattice gate (compliance and sensitivity within 3%) or break an owner
decision. Inputs: ARCH_BRIEF.md (binding), DESIGN_operator-learning.md, RESEARCH.md, DATA_INTERFACE.md,
`docs/ROUTES_PROGRESS_20260923_CN.md` (route-1 E0-E3, the slow-mode statistics, the merged weak-group table, E1b, D1/D2 and
the lattice_v2 results), and `arch/_ol/*.json`. Numbers marked **[rc]** I recomputed here. Everything else is quoted from
the sources.

**Overall verdict.** The invariants are right. q-linearity, exact Dirichlet port values, rigid modes exact to fp64, a
symmetric PSD readout that is always too stiff, and a symmetric Neumann action all follow from the construction; §4.1-4.5
check out. None of the section-5 dead ends comes back. The design is also the most complete of the proposals on
precision (strain form, TF32 off, fp64 accumulation), on budgets and on step-1 diagnostics.

What does not hold is the accuracy story. The design says 8 K evaluations (n_K = 8) should pass the gate, and that claim
rests on one mechanism: exact, undamped local patch modes that lift the preconditioned λ_min by 20-40×. As specified,
that mechanism:
- has an ill-posed local eigenproblem;
- covers only a fraction of the slow modes geometrically;
- is damped by the same single global step cap that route 1 blamed for its weak-group result.

The repository evidence says every "exact local slow-mode" intervention so far moved λ_min by ≤ 1.2×. Expect a route-1-like
n_K ≈ 24-32 unless the patch mechanism is repaired as described below. Separately, the q-path is fallback A/B in substance.
Presenting it as the owner's feedforward main line needs an explicit owner decision.

---

## 1. Fatal issues (as written)

### F1. The sliver-mode carrier (patch modes inside S) is ill-posed, under-covering and damped, and it carries the whole accuracy claim

Each of the three defects below is enough to reproduce route 1's plateau (λ_min 0.0070 → 0.0080). Together they make n_K = 8
very unlikely.

**(a) The local pencil is singular as written.** §2.7 assembles the patch matrix "from the 8 child K_e plus internal
ghost-penalty faces". That is a Neumann-type local matrix, and it has two problems:
- A floating patch (almost every patch) has 6 exact zero eigenvalues, the rigid modes of the patch.
- Sliver pieces that are tied to the rest only through GP faces crossing the patch boundary are near-singular. This is exactly
  the failure route 2 hit in its first version: "thin-sliver elements inside a small block are supported only by their own
  tiny stiffness, while part of the GP faces that tie them to their neighbours falls outside the block, so the local problem
  is near-singular" (D1, first version, upper bound > 1e12).

With the (1/λ − 1)·v vᵀ scaling of §2.6, λ → 0 gives an unbounded S. Lanczos then sets c_S ≈ 0, and the S step stops acting on
everything else. The exactness proof in §4.6 silently assumes the principal submatrix: it needs (A v)_p = A_pp v. The Neumann
matrix does not satisfy that.

- **Fix.** Use the principal submatrix A_pp = R_p A R_pᵀ of the global operator: Dirichlet on the patch boundary, with all
  element and GP contributions, including faces whose other side lies outside the patch. Equivalently, add route 2's
  "GP outer ring".

**(b) Two diagonal staggers of 2×2×2-element patches do not cover the slow modes.**
- The claim "any slow mode of 11-30 nodes (≈2-3 elements across) lies inside a patch of at least one stagger" is false.
  Stagger 0 has windows [2k, 2k+2) and stagger 1 has [2k+1, 2k+3), per axis, **with the same parity in all three axes**
  (offset (0,0,0) or (1,1,1)).
- Consider a mode whose bounding box is 2 elements in d axes. It fits a patch only if its start parities agree across those d
  axes. That happens for 2 of 2^d parity classes: 50% for sheet-like modes (d = 2) and 25% for d = 3 **[rc]**. A mode 3
  elements across fits no 2-element window at all.
- 30 nodes on a one-node-thick sheet is about 5.5 × 5.5 Q2 nodes, which is about 2.7 elements across **[rc]**. So the larger
  route-1 modes (0013: about 30 nodes each) are mostly uncovered.
- The dense band sets λ_min through its worst member. The uncovered modes remain, so λ_min barely moves: the same result as
  E1 (128 ideal global modes: 32-layer witness only 56.7 → 34.6) and as the merged groups (+14%).
- Mechanism 2 (§5) does not rescue the uncovered modes. "Patch error = harmonic extension of the patch-boundary error, bounded
  by the stiff body" fails wherever the patch boundary cuts through the weak sheet. The design's own count shows the sheets
  are giant (2-5 components of up to 10k nodes), so the boundary cuts them everywhere.
- **Fix.** One family of 3×3×3-element windows at stride 2 (one-element overlap). In every axis these contain every
  interval of ≤ 2 elements, whatever its parity. Restrict it to weak-touched windows (≈ 70%) with k ≤ 8.
  - Cost **[rc]**: ≈ 2080 × 0.7 × 8 × 1029 × 4 B ≈ 48 MB (vs 36 MB now).
  - Batched LOBPCG: ≈ 1.2 TFLOP ≈ 0.1-0.2 s. That fits the ≤ 0.5 s encode target, barely.
  - Alternative: all 8 parity staggers of 2×2×2 patches with k ≤ 4, about 72 MB **[rc]**.
  - Either way, make coverage a step-0 go/no-go. The route-1 slowest-64 modes of 0013 and 0020 already exist. For each
    candidate patch family, measure the fraction of their energy inside span(patch modes). This takes minutes and needs no
    network.

**(c) One global cap for the whole S re-creates route 1's "global step damps the exact blocks".**
- On a slow patch mode, α c_S S A v ≈ α c_S (ω λ + g(1 − λ)) v ≈ α c_S g v, with c_S = 1.9 / λ_max(SA).
- The pair part alone has λ_max(M0⁻¹A) ≈ 4.59 (route-1 E3, fine level). With ω ≈ 1 and the head bound g ≤ 1.5, the best
  per-step factor is 1.9·1.5/(4.59·1.05) ≈ **0.59** **[rc]**. Removal is not exact, and the contraction is at best 0.41.
- Exact removal is reachable only in a corner: ω ≲ 0.5 (pair smoothing at half strength everywhere) and no double counting.
- Double counting is built in:
  - the two staggers are summed additively;
  - within one stagger, neighbouring 125-node patches share face, edge and corner nodes.
  A mode seen by m patches has an SA eigenvalue of about m·g, which forces c_S down again.
- **Fix.**
  - Apply the local slow space as an energy projection, not as a 1/λ-scaled smoother term:
    z = Σ_p V_p (V_pᵀ A V_p)⁻¹ V_pᵀ r. V_pᵀ A V_p is k×k per patch, computed once with the global A. Its spectrum is {0, 1} per
    patch, so no step size is needed.
  - Apply it multiplicatively by colour (8 colours for stride-2 windows), with a patch-local residual update. That costs about
    (3/2)³ ≈ 3.4 patch-restricted K-equivalents per S step.
  - Or at least give S_pair and S_mode separate sub-steps with separate caps.
  - Keep ω_b per block (route-1 E3 evidence) for the pair part.

**Consequence if unfixed.** 1a would show a plateau on force-driven μ at n_K = 8-16. The design would then slide to n_K ≈
24-32 (route-1 territory), and the "new" part would contribute nothing measurable.

---

## 2. Serious issues

### S1. The n_K = 8 target (and "≤ 16 acceptable") is arithmetically inconsistent with the evidence unless F1 is fixed and works
- Take a Chebyshev-optimal polynomial of degree m in a preconditioned operator with spectrum [λ_min, 1], and require the
  energy-error ratio ≤ 0.1 (μ − 1 ≈ 1e-2, R7). This needs 2ρ^m ≤ 0.1 **[rc]**:

  | m (≈ n_K) | required κ | required λ_min | vs route 1 (0.0039-0.0070) |
  |---|---|---|---|
  | 8 | 29 | 0.034 | 5-9× higher |
  | 16 | 115 | 0.0087 | 1.2-2.2× higher |
  | 24 | 257 | 0.0039 | ≈ route 1 (and route-1 learned-24 did pass the gate) |

- One HGM-NO sub-step is one of S or C with one K. One route-1 layer applied the whole preconditioner with one K. So n_K is at
  best equivalent to the route-1 layer count, not better.
- Route-1 learned-16 failed the lattice gate (6.8% / 13.2% compliance, 8.9% / 20.4% sensitivity).
- §5's "expected λ_min 0.1-0.2 (20-40×)" has no support. Every exact-local intervention so far gave ≤ 1.2×.
- **Fix.**
  - State the planning target honestly: n_K = 16-24, with 8 as a stretch goal.
  - Train with random depth n_K ∈ {8, …, 32} (per-depth α/β from H_step), so that the dial is trained rather than
    extrapolated.
  - Budget the cost at n_K = 24 (see S6).

### S2. The "certified tail" and "extra steps repair it" are nearly inert on exactly the directions that matter
- The tail uses β = 0 and α = 1/1.9, i.e. Richardson with step 1/λ̂. Per step it contracts slow modes by 1 − λ_min/λ_max
  ≈ 0.993-0.996. Eight extra tail steps remove about 3-5% of the soft error **[rc]**.
- Monotone, yes. Useful, no. The claims that "fallback A = raise n_K" and that "η flags, extra steps repair" (§2.12(c), §7.2)
  are therefore overstated.
- η uses C_η built from the same weak preconditioner, so it also under-weights the soft error by about λ_min. It will miss the
  error it is meant to flag.
- **Fix.**
  - Make the tail a Chebyshev recurrence on [λ̂_min, λ̂_max]. Both come from the same encode-time Lanczos. It stays linear in
    q and symmetric.
  - Use route 2's equilibrated certificate as the out-of-distribution flag in step 2 and step 3. Keep η only as a cheap
    trend monitor.

### S3. Owner decision 8: the q-path is fallback A/B, relabelled as the feedforward main line
- Decision 8: the main line is a FEEDFORWARD neural operator plus a multiscale GNO. Fallback A is "each layer computes the
  exact residual r = f − K u; a geometry-conditioned linear net gives the correction". Fallback B is "route-1 skeleton with
  only a few geometry-generated coefficients".
- HGM-NO computes an exact residual after every sub-step (n_K + 2 fine K), and its learned content is only scalars: dampings,
  gates, moduli and step coefficients generated by a hypernetwork. That is fallback B with a stronger skeleton, run the
  fallback-A way.
- The "multiscale GNO" half is nominal. The q-path contains no learned kernel. The only real generated-kernel component,
  K2 enrichment, is postponed to the capacity ladder.
- Brief question (d) allows "a few K-applications inside". Ten applications with a residual after every step is not "a few";
  it is the fallback's defining structure.
- The justification (§2.1(i), "a K-free linear net needs 1e-4 coefficient accuracy") is not a theorem. It holds for a net
  that learns a *stiffness* and iterates on it. The dead-end evidence (3e-4 needed) was about learning S or its factors,
  where soft eigenvalues come from cancellation.
  - A feedforward net that represents the *inverse* directly only needs about 10% relative accuracy on the soft amplitudes,
    which are the large eigenvalues of the inverse.
  - That is attainable when the net has a carrier for local soft modes with exact 1/λ from exact local eigen-solves, and
    HGM-NO already computes those.
- Ablation 1d replaces K by a *learned stiffness* A_θ inside the same iteration. That tests the variant §2.1 already predicts
  will fail. It is a straw man for the owner's main line.
- **Fix.**
  - (i) Tell the owner plainly that the proposal makes a fallback-A/B hybrid the main line, and get a decision.
  - (ii) Replace 1d with a fair feedforward variant built on the same stored objects and linear in q. Suggested form:
    lift → [patch-mode inverse terms Σ V_p (V_pᵀAV_p)⁻¹ V_pᵀ applied to the lift residual] + [P1 · learned multi-channel
    linear coarse map with hypernetwork-generated kernels · P1ᵀ], with only 1-3 exact residual corrections.
    Compare it at equal wall-clock time.
  - (iii) If the hybrid wins, that is evidence the owner can accept. Until then the design should not declare K-in-the-loop
    "weakly dominant".

### S4. Learned coarse moduli θ are theoretically counter-productive in an exact-residual method
- With exact residuals, the Galerkin correction P(PᵀAP)⁻¹Pᵀr is the A-orthogonal projection. It minimises the energy error
  over range(P) in one step. Any θ ≠ 1 is strictly worse per step on the coarse component.
- The B2 evidence (learned coarse materials, 23k numbers) was for a **one-shot, coarse-only** model with no fine correction
  after it. That is not this setting.
- θ < 1 also makes C overshoot (C A eigenvalues up to 1/θ_min = 4). That drives c_C down and damps every other coarse
  component.
- θ also adds the riskiest gradient path: backpropagation through the Galerkin products and the fp64 level-3 Cholesky, which is
  risk 5.
- **Fix.** Freeze θ = 1 in v0 (drop H_θ1 and H_θ2, about 4.9k outputs). Spend that capacity on K2 (hypernetwork-generated
  enrichment vectors at levels 1-2, Galerkin-coupled). That enlarges the space, which is what locking on 1-2-element walls
  actually needs. Keep θ only as a 1b oracle experiment.

### S5. Coarse-level SPD is not certified, so the non-expansiveness argument can fail
- §4.7 certifies only λ_max(TA). The argument assumes T ⪰ 0.
- C = P1 V₂ P1ᵀ is PSD only if V₂ is SPD. A symmetric V-cycle is SPD iff each level's symmetrised smoother
  R + Rᵀ − RᵀÃR is SPD. So every coarse Chebyshev-Jacobi smoother must be convergent for its own Ã_ℓ.
- The dampings ω_v = σ(·) are not certified against λ_max(D_ℓ⁻¹Ã_ℓ), which moves with θ. If a coarse smoother diverges,
  C becomes indefinite. λ(αcCA) < 0 then gives I − αcCA > 1: an expansive step that no cap catches.
- **Fix.** At encode, run 10-step Lanczos on D_ℓ⁻¹Ã_ℓ per coarse level (negligible cost) and parameterise ω_v relative to
  that cap, as ω_b already is. In the 1a/1c monitors, check λ_min(CA) ≥ 0 by Lanczos. The fp32 floating level-3 factor
  Ã3^N + ρRRᵀ should be fp64 (+11 MB): an fp32 Cholesky of a κ ~ 1e5-1e7 matrix can produce negative pivots.

### S6. Throughput estimate is optimistic, and the budget at a realistic n_K is tight
- Per column, one K apply moves ≈ 32 MB of gathers and scatter-adds (body 81 values per element, ghost 135 per face, both
  directions) **[rc]**, plus weights.
  - 25-40 µs is a compute-bound floor. With DRAM-level atomics, 50-150 µs is realistic unless node vectors stay L2-resident
    and the batch is laid out column-innermost.
  - Route 1 measured about 2.5 ms per layer per column even when batched (565 ms / 7 columns / 32 layers).
- At n_K = 24 and 100 µs per K, one reaction Ŝq costs about 2.1 × 26 × 0.1 ≈ 5.5 ms. Per design iteration:
  100 cells × (160 × 5.5 ms + 55 × 0.5 ms) ≈ 90 s, plus encoding (moments dominate, 25-100 s).
- That is "a few ms" per application, but only just. It stays within budget only if F1 lowers n_K.
- **Fix.** The step-0 microbenchmark (risk 8) must come before any other work. Also benchmark a dense-slot layout (the 33³ × 8
  parity packing of DATA_INTERFACE §2.2), which turns gathers into fixed-offset slicing and avoids atomics.

### S7. The Neumann (BDD) action is too weak to deliver "κ(B_N S) ≲ 5"
- Suppose c₁K⁺ ⪯ 𝒱 ⪯ c₂K⁺ on the balanced space. Then κ(J𝒱Jᵀ S) ≤ κ(𝒱K).
- One palindromic S-C-S sweep has λ_min(𝒱K) of the order of the two-level λ_min (≈ 0.005-0.03), so κ ≈ 30-200.
- BDD iterations scale roughly as √κ, so they would grow 5-14×, from 11-53 up to about 100-700. That erodes the advantage
  over forward-only preconditioners (300-1600).
- **Fix.** Use a symmetric Chebyshev polynomial of degree 6-10 in 𝒱K, with 𝒱 as the base, on the Lanczos interval. The
  polynomial is positive on (0, λ̂_max], so the action stays SPD and linear. Cost is about 6-10 K, roughly 0.5-1 ms. Target
  κ ≤ 5 is realistic only after F1.

### S8. Cut-band port semantics: the default reinterprets decision 4 (flagged by the design, but it must not ship silently)
- Decision 4 says "ports = box-face nodes UNION cut-band nodes". The design defaults to Dirichlet on the box and a
  **force input** on the cut band. The exact operators are partial Legendre transforms of each other (correct), and the
  free-cut lattice then needs no extra unknowns (a real advantage).
- The approximations are not interchangeable:
  - A net trained only in force mode says nothing about accuracy in cut-Dirichlet mode, which the future skin needs.
  - In force mode, 22-28% of band nodes (35% of all weak nodes in heavy cuts) stay inside the interior problem, which is the
    harder variant for F1.
- The cut-Dirichlet mode also changes the coarse spaces. P1 must be masked by Z on band nodes, and Ã1 must be the Galerkin
  product with the masked P1. Otherwise C is non-Galerkin and overshoots. The design does not budget this.
- **Fix.**
  - Get the owner's decision in step 0.
  - Carry both modes in the harness, and include ≈ 10% cut-Dirichlet samples in step-1 training.
  - Build masked-P1 Galerkin templates for band macro elements. This is a pattern-dependent extra of about 5-10 MB.

### S9. Training/inference mismatch through stop-gradient caps
- The caps c_S and c_C depend on the generated coefficients (ω, g, θ) but are stop-gradient. The optimizer therefore never
  sees that raising g or lowering θ is later undone by a smaller cap. That produces oscillation or a drift into saturation
  (monitor 6.5 item 6 will show it).
- **Fix.** Either make λ̂_max differentiable (power iteration with a warm start, differentiating the Rayleigh quotient, which
  is smooth away from multiplicity), or drop runtime caps during training and use bounded parameterisations. Then certify
  post hoc at encode and fall back to caps only when a bound is violated.

---

## 3. Minor issues

1. **k_max = 8 truncation and the "rank k/k_max" gate feature** make û discontinuous in τ when λ₈ and λ₉ cross, and break the
   eigenspace-invariance argument of §4.8 inside the dense band.
   - Fix: drop the rank feature; gate as a smooth function of λ only. Pick k_max so that λ_{k_max+1} > 0.2 on ≥ 99% of
     patches (measure it), or taper s(λ) against λ_{k_max+1}.
2. **Global max pooling** in the vertex scatter and the global token is the same kind of non-smooth global head as the
   phase-7 dead end. Fix: use logsumexp or softmax pooling.
3. **Mutual-strongest pairing** is discrete. Near-ties broken by 1e-7 pipeline noise (or by the 1e-4 Kuhn-split asymmetry of
   re-integrated rotated moments) change the pairing at O(1). Equivariance then holds only for moment-permuted rotations, not
   for re-encoded ones. Fix: tie tolerance 1e-5 relative, and freeze pairs per design band (already planned).
4. **Galerkin FLOPs are underestimated.** The naive P_cᵀK_cP_c route costs ≈ 71 GFLOP for the λ and μ parts **[rc]**, not
   40 GFLOP. Precomputing Galerkin moment templates per child position brings it to ≈ 28 GFLOP. Still milliseconds either way.
5. **Level-3 Dirichlet size.** With box-plane macro nodes dropped, the Dirichlet level 3 has ≤ 343 interior nodes
   (≈ 1k DOF), not 2.3k. The 2.3k figure is the floating space. Memory and time drop accordingly.
6. **§2.7 contradicts itself.** "Needs no weak threshold" conflicts with counting and keeping only weak-touched patches. Decide
   which one applies: all patches cost ≈ 4.2k × 8 × 375 × 4 B ≈ 50 MB.
7. **J_sens** uses |·|, which is non-smooth at 0. Use a squared relative error.
8. **Heterogeneous momentum.** d_j mixes S-step and C-step directions through β. That is not a polynomial in one preconditioned
   operator, so Chebyshev intuition does not apply to the trained head. Monitor item 5 (spectral radius of the full
   propagator) is essential. Keep it.
9. **Isotropic scalar convolutions** can express only O_h-invariant functions of neighbourhoods of invariant scalars. Cross-
   directional invariants (for example a wall normal against a plane normal) have to come from the input features. Consider
   regular-representation lifting at 9³/5³ only (cheap there), or add products of oriented moment invariants to the inputs.
10. **Gate load model (risk 4).** It is a valid observation. It must be reported, never used to soften the gate. Training
    already includes gate-like nodal loads; keep that.

---

## 4. Component-by-component verdict

| Component | Verdict | Note |
|---|---|---|
| Exact K, matrix-free strain form, TF32 off, fp64 accumulation | **keep** | Correct precision analysis. Rounding is relative to strain. |
| K inside the q-path after every sub-step | **modify** | Owner decision needed (S3). Build a fair feedforward comparison. Target 1-3 residual corrections if the feedforward net holds. |
| Rigid/affine W-weighted lift, rigid part symbolic in K | **keep** | Exact, O_h-invariant, and gives BDD Ŝ R = 0. |
| Mixed port model (Dirichlet box, force on the cut band), mask switch | **modify** | Owner confirmation, both modes trained, masked-P1 Galerkin (S8). |
| Pair-block smoother with per-block ω_b | **keep** | Route-1 E3 evidence. Give it its own cap. |
| Patch modes on 2 staggered 2×2×2 partitions, Neumann-assembled, 1/λ-scaled, additive | **modify (heavily)** | Principal submatrix, 3-wide stride-2 windows (or 8 parities), projection form, multiplicative colouring (F1). Step-0 coverage go/no-go. |
| Nested Q2 hierarchy 16³/8³/4³ with component splitting | **keep** | Nesting holds with splitting. Q2 is needed (route-1 quadratic ≫ affine). |
| Learned coarse moduli θ (H_θ1, H_θ2) | **drop in v0** | Galerkin is optimal with exact residuals (S4). Replace with K2 enrichment. |
| Coarse Chebyshev-Jacobi with learned ω_v | **modify** | Certify λ_max per level so that V₂ is SPD (S5). |
| Level-3 exact dense Cholesky (fp64) | **keep** | The global transmission carrier. Also fp64 for the floating factor. |
| Lanczos step caps (stop-gradient) | **modify** | Per-component caps. Fix the training mismatch (S9). |
| Learned two-history recurrence α_j, β_j from the global token | **keep** | Route-1 E1b mechanism. Train with random depth. |
| Certified tail (β = 0, α = 1/1.9) | **modify** | Chebyshev tail with λ̂_min (S2). |
| Deep supervision at j = 4, 6, 8 | **keep** | |
| IsoU-Net hypernetwork on 33³ → 5³ plus global token | **keep** | Replace max pooling (m2). Optionally regular-representation at the coarse stages (m9). |
| Bounded smooth heads | **keep** | Drop the rank feature (m1). |
| Ŝ = ÊᵀKÊ via explicit transpose sweep | **keep** | Symmetric PSD, no stored activations, one-sided bound. |
| Sensitivity with exact ∂w/∂τ; ghost term zero within a topology; J_sens | **keep** | Squared form (m7). |
| Palindromic S-C^N-S Neumann action | **modify** | Symmetric Chebyshev polynomial of degree 6-10 (S7). |
| η indicator from the final residual | **keep as monitor only** | Route-2 certificate for out-of-distribution flags (S2). |
| Loss (μ − 1 per sample, CVaR, log warm-up), sampling, seeded adversarial LOBPCG | **keep** | |
| Free-coefficient oracle 1b | **keep** | The right way to separate "structure" from "hypernetwork". |
| Ablation 1d (learned stiffness A_θ) | **replace** | Use a fair feedforward inverse-carrier variant (S3). |
| Step 1a zero-learning structure test | **keep and extend** | Add the patch-coverage test on the route-1 slowest-64 modes first. |

---

## 5. Best ideas to carry into a synthesis

1. **Hard q-path rules with a unit test.** A fixed linear DAG, geometry-only coefficients, no q-dependent scalars (so no CG),
   and a linearity check to fp32 rounding. Linearity, symmetry, the one-sided bound and exact rigid modes then hold for any
   weights.
2. **Strain-form exact K from 125 signed Gauss weights per element.** fp32 with TF32 off, fp64 reductions, and the rigid part
   of the lift never passed through K. Precision is solved structurally.
3. **W-weighted rigid/affine lift** (12 columns, O_h-invariant weights). The interior gets a Taylor guess, and the ports are
   bit-exact because Z has zero rows on D.
4. **The percolation finding.** Weak nodes form 2-5 giant sheets, so "cluster = connected component" is unusable and local
   slow spaces must be spatial windows. Also the principle that local slow spaces are *computed* by local generalized
   eigenproblems, not learned, and are applied without a weak threshold.
5. **Nested Q2 coarse hierarchy.** Galerkin products are formed from moments through λ/μ template splitting (linear in the
   moments), with an exact dense coarsest solve. That gives global transmission in every coarse visit, with geometry-exact
   far-field mixing and no spectral layers.
6. **Mixed box-Dirichlet / cut-force port with a mask switch.** The one-sided lattice bound also holds for loaded cuts, and a
   free cut costs zero extra unknowns (pending owner confirmation).
7. **One stored hierarchy serving three roles:** the forward extension, a symmetric Neumann action (same basis on both sides,
   the Melchers-Dolean-Abdelmalik condition), and an error indicator.
8. **Galerkin-orthogonality reading of the sensitivity error.** The first-order term comes only from the non-proportional
   (surface) part of K′. This motivates J_sens and fringe-element monitoring, and it matches route 1's sensitivity/compliance
   error ratio of 1.3-1.5.
9. **Per-component Lanczos certification.** It keeps every certified step non-expansive, provided C is also certified PSD.
10. **Free-coefficient oracle, zero-learning structure test, and a K-free (fair) ablation before any hypernetwork training.**
    That run order makes every failure attributable.
11. **Rotations by signed-permutation moment transforms,** with pairs and patches of rotated instances obtained by rotating
    the originals: 144 training instances from 3 factorisations and 3 eigen-precomputes.
