# Adversarial review of DESIGN_multilevel.md (LCMO, "Lift-and-Correct Multilevel Operator")

Reviewer lens: find what makes LCMO fail the lattice gate (compliance and sensitivity within 3%) or break an owner decision.
Inputs: ARCH_BRIEF.md (binding), DESIGN_multilevel.md, RESEARCH.md, DATA_INTERFACE.md, ROUTES_PROGRESS (route-1 E0/E1/E2/E3/E1b/D2
tables), `xcase/r1/encode_r1.py` (the pair and strength rules), `arch/q2_hierarchy_counts.*`, `arch/weak_clusters.*`,
`arch/_ol/patch_counts.json` (weak-node counts per ℓ1 box, computed by another design; body-only proxy).
Numbers marked **[rc]** are ones I recomputed here. Everything else is quoted from the sources.

**Overall verdict.** The invariants hold: q-linearity, exact ports, rigid modes exact to fp64, and a symmetric PSD, one-sided
readout (§4.1–4.4 check out). What does not hold is the accuracy claim. It rests on an error-product argument that is
mathematically wrong for a stationary cycle, and on a split coarse level that the stated FMG/V-cycle recursion cannot use. At
the advertised operating point (T = 1), the design will very likely miss the gate on force-driven loads. The skeleton is still
the best available chassis for a synthesis, after the fixes below.

---

## 1. Fatal issues (as written)

### F1. The ℓ2 strong/fringe split is not nested in ℓ1, so the stated FMG pass and V-cycle are ill-defined
- The split functions `φ_v·1[weak]` and `φ_v·1[strong]` use the **fine** (ℓ0) nodal weak mask. The fringe sheet is 1–2 fine node
  layers thick, and those functions are not in the ℓ1 space Q2(16³). So `ℓ2split ⊄ ℓ1`, although `ℓ3 ⊂ ℓ2split ⊂ ℓ0` does hold.
- In the data flow, `c1 = P12 c2 + S1(R1 r − A1 P12 c2)` needs `P12` on a split vector. No such operator exists.
  - If the duplicates are summed on the way up (`P12(c_s + c_w)`), the fringe relative motion that the split was introduced for
    is erased before it reaches the fine field. The split then only affects the ℓ2 solve itself, which gives a non-Galerkin
    stage whose energy contraction is not guaranteed.
  - The correct ℓ1 residual would be `R1 r − A_{1,2s} c2` with the cross operator `A_{1,2s} = P_{0←1,I}ᵀ K P_{0←2s,I}`. The design
    neither builds it nor budgets it.
- The V-cycle has the same problem: after pre-smoothing at ℓ1, the ℓ2split residual cannot be formed from ℓ1 quantities.
- **Fix, in order of preference.**
  - (a) Take the fringe coarse space out of the nested chain. Make it its own multiplicative stage, placed between Q3 and Q4, with
    an exact fine residual: `w ← w + P_{0←2s}(A_2s)⁻¹P_{0←2s}ᵀ(r − K_II w)`. A_2s is the 23k split operator the design already
    builds. It can be solved by a few PCG-free Chebyshev steps, or factorized, since it is 23k sparse with 162-DOF element blocks.
    The prolongation `P_{0←2s}` is applied element-wise with the 81×162 maps that are already needed to form A_2. Cost: one extra
    residual. With a fringe-local residual update this is ≈0.4 GFLOP, not a full K.
  - (b) Precompute A_{1,2s} in element form at encode. Per ℓ2 element that is (ℓ1 DOF in it) × 162. This is cheaper at run time
    but memory-heavier and more complicated.
  - (c) Drop the split in v0 and let X3b decide whether it comes back.
  - ℓ3 = P23ᵀ A_2s P23 remains valid in all three options.

### F2. The one-pass (T = 1) accuracy argument is invalid; the soft band is contracted by ≈ (1 − λ_min), not by a "30% per stage" product
- §2.1(i) claims that "three stages that each leave 30% of their target component give ≈3% of the field". The identity
  `I − B K = Π_k (I − C_k K)` is correct. The inference is not.
  - Each factor has K-norm ≈ 1 on the components it does not target: a smoother leaves smooth error, and a coarse correction
    leaves rough error. So the product of per-component residual fractions is not a bound.
  - The correct tool is the Xu–Zikatanov identity: one multiplicative sweep has `‖E‖²_K = 1 − 1/K₀`, where K₀ is the
    stable-decomposition constant of the whole subspace family. It is set by the worst component, the dense soft band.
- **The evidence says K₀ is huge.** Route 1 put the slowest 128 modes of its preconditioned operator in [0.0039, 0.0062]. Merged
  weak groups raised λ_min only from 0.0070 to 0.0080 (+14%).
  - A stationary cycle contracts those modes by 1 − λ ≈ 0.992–0.996 per cycle **[rc]**.
  - Chebyshev over 24 steps gives 2σ²⁴ ≈ 0.03–0.10 **[rc]**, which is exactly why route 1 needed 24 learned or 40–48 Chebyshev
    layers.
  - For T = 1 to reach even a 10× reduction on the soft band, λ_min on that band must be ≳ 0.5–0.9: a 60–100× improvement over
    the best route-1 variant. Nothing measured supports that.
- **The accuracy target is also larger than the design's framing suggests.** The initial error is the zero-lift field Zw*, whose
  energy the design itself puts at ≈64 ‖u*‖² for smooth data (§2.1(ii)).
  - To reach μ − 1 ≈ 3e-3 at a ratio of 20–64, the whole pass must cut ‖Zw*‖_K by roughly 80–150× in norm.
  - Route-1 D2 calibration: the lattice gate tolerated per-cell force-driven energy errors of roughly 3–10% in the 2-cell harness
    (at Chebyshev 32 layers, 21–35% per-cell error gave 1.9–5.7% lattice error). Even so, the reduction still has to be about 25–45×.
- **A second mechanism points the same way: locked geometric coarse spaces.** Walls are 1.5–6 fine elements thick
  (2τ/|∇φ| **[rc]**), while ℓ2 has H = 4h and ℓ3 has H = 8h. A single Q2 polynomial per coarse element cannot bend a thin wall,
  and it ties the two legs of a U-bend together.
  - The FMG start is therefore too stiff in exactly the global soft bending modes that carry 94–96% of the lattice energy.
  - Only ℓ1 (H = 2h) and a few polynomial smoothing steps can relax them, and in one pass information crosses only a few ℓ1
    elements.
  - This is the brief's "fixed-geometry coarse bases" dead end in disguise: the operators are Galerkin, but the functions are
    geometry-agnostic polynomials. Route-1 E0 is consistent with this: quadratic block spaces still needed 32–64 layers.
- **Learned gates cannot close this gap.** ω, d_v, η and s_e rescale existing operators. They add no new directions, except the
  weak Ω smoothing (see S5) and G_θ's 4 scalars per element. So X1 (the capacity ceiling) may itself fail the gate at T = 1.
- **Fix.**
  - (i) Make T a budgeted variable from day one and accelerate across cycles: learned two-term (Chebyshev-type) recurrence
    coefficients per cycle. Route-1 E1b halved the depth this way, 2 scalars per layer. The budget allows it: forward cost is
    ≈ 5 + 2.4(T − 1) GFLOP, and a 100-FULL-cell design iteration takes ≈ 9 / 14 / 23 / 31 / 40 s at T = 1 / 2 / 4 / 6 / 8 at
    20 TFLOPS **[rc]**, which is within "a few ms per cell-application" up to T ≈ 8.
  - (ii) Remove the soft band by **direct solves on a subspace**, not by relaxation.
    - Fine fringe: multiplicative exact patch solves (S4) or a GenEO-type local spectral coarse space: per patch, k ≈ 8–16
      low-energy eigenvectors of the local pencil, assembled into one sparse coarse level.
    - Global bending: an energy-minimizing ℓ1 basis. For example, an MsFEM-type K-harmonic extension of the Q2 coarse-node values
      inside each 2×2×2 fine block, which is 81 interior DOF × 81 coarse DOF per ℓ1 element, ≈ 55 MB per FULL cell and
      ≈ 36 GFLOP at encode **[rc]**. ℓ2 and ℓ3 are then built by Galerkin on top of it, so the hierarchy stays nested.
  - (iii) Before building anything learned, the X0 gate must measure the Ritz values of `B K_II` on the Krylov space of
    force-driven w*, and the stage-wise error energies. The design already lists the second.

---

## 2. Serious issues

### S1. Mode N (the approximate Neumann action for BDD) is an additive BPX with block-Jacobi smoothers: probably too weak
- S⁺ is dominated by the softest modes of S, which are the soft decile. An additive BPX with point or pair blocks has
  κ(B_N K) ~ 1/λ_min on that band, which is O(10²) by the route-1 evidence.
- Inexact BDD Neumann solves multiply the BDD condition number by that spectral-equivalence constant. So 11–53 iterations can
  become several hundred, which is the forward-only regime (300–1600) that BDD was chosen to avoid. The loss L_N (field error
  of B_N f) does not measure spectral equivalence either.
- **Fix.**
  - Build mode N from the same multiplicative pieces, used symmetrically: symmetric V-cycle, fringe patches forward then
    backward, exact ℓ3 pseudo-inverse. Wrap it in a fixed-degree Chebyshev polynomial in B_N K with Lanczos bounds. That map
    is linear and SPD, whereas fixed-count PCG is nonlinear in r and not allowed.
  - Cost: k fine K applications, ≈ 1 GFLOP each, still far below the 11.5 GFLOP of Ŝq.
  - Train it with a spectral surrogate: the energy error of k preconditioned iterations on random balanced loads, or the ratio
    of the extreme Ritz values.
  - Measure BDD counts in X4 against the exact Neumann counts.

### S2. The owner's decision 8 is bent: the main line is structurally fallback A/B
- The q-path is a classical multigrid with ~0.2 M geometry-generated gates and 3 exact K residuals. That is "learned iteration
  with exact residuals" (fallback A) combined with "skeleton with generated coefficients" (fallback B). The feedforward
  multiscale GNO survives only as G_θ, 4 scalars per element.
- Question d allows "a few K applications inside", but the owner must be told plainly that this *is* the hybrid and that the
  learned capacity is small.
- **Fix.**
  - Present LCMO as "hybrid, K inside".
  - Add one genuinely learned, q-linear GNO component that can create directions, not only rescale:
    - geometry-generated local enrichment functions at ℓ1/ℓ2 (a learned GenEO: k per coarse element, linear combinations of
      K-derived local vectors emitted by the GNN);
    - or 2–4 vector channels in the coarse smoothers.
  - Keep the X3d pure-feedforward ablation as the owner's main-line reference.

### S3. fp32 storage of the ℓ3 factor and the fringe-patch factors corrupts precisely the soft directions
- A factor stored in fp32 has a forward error of about ε₃₂ κ, with ε₃₂ ≈ 6e-8.
  - A_3 contains the softest global coarse modes. With the S spectrum reaching 7e-7 λ_max, κ(A_3) of 1e5–1e6 is plausible, which
    means 1–6% error on exactly the loaded directions.
  - Fringe blocks mix weak diagonals (< 1% of the median, down to γ-only support; volume fractions go down to 1e-9) with strong
    partners, so κ can reach 1e4–1e7. In the last stage (Q5) nothing corrects that error.
- **Fix.**
  - Store the ℓ3 factor in fp64 (+9.6 MB per mode). The fp64 triangular solves cost 2 × 2187² × 2 ≈ 19 MFLOP per column, which
    is negligible. Alternatively, keep fp32 and add one fp64 refinement step.
  - Factor the fringe blocks after symmetric diagonal equilibration (D^{-1/2} K_PP D^{-1/2}). Keep blocks whose scaled κ is above
    1e5 in fp64.
  - Also accumulate A_2 and A_3 in fp64 at encode, which is cheap.

### S4. The fringe-patch claims do not match the counts, and the additive χ-weighted Schwarz is not "exact" and may not be stable
- **The claim does not match the counts.** The design says "20–60 weak nodes typical" and "each localized mode is entirely in
  at least one patch". The ℓ1-box counts (`_ol/patch_counts.json`) give weak nodes per box: median 16, p90 29, max 33–54. The
  0013 slow modes cover ≈ 30 nodes. So a large share of slow modes straddle patches.
- **Straddling modes are not solved exactly.** The additive, χ = 1/√m weighted Schwarz solves a straddling mode only partially.
  Even a mode inside one patch is removed exactly only if its K-image avoids the overlap layers.
- **Stability at initialization is not guaranteed.** With gates initialized to 1, λ_max(C K) of an overlapping weighted
  additive Schwarz is not bounded by 2: node multiplicity is up to 8 at box corners. The stage can then amplify error at
  initialization, which contradicts "sound numerical method at init".
- **Patch composition may repeat route 1's failure.** The design uses weak nodes plus mutual-strongest partners. Route 1's first
  weak-group version, which separated weak nodes from the strong nodes they hang on, made the witness *worse* (1.26 → 3.80).
  Whether mutual-strongest partners actually bring in the supporting strong nodes has not been checked. A one-ring of strong
  nodes brings a patch to a median of 45–75 nodes and p90 ≈ 100.
- **Fix.**
  - Apply patches as an **8-colour multiplicative block Gauss–Seidel** over the 2×2×2 colouring of ℓ1 boxes, with fringe-local
    residual updates after each colour. That is exact on in-patch modes, energy-contractive for any SPD blocks, and needs no
    χ weights and no damping.
  - Add a second pass on a partition staggered by one fine element, to catch straddling modes (already counted in `_ol`).
  - Include the strong one-ring, capped.
  - If memory binds, replace full factors by k local low-energy eigenvectors per patch (GenEO).
  - Initialize any additive gate from a Lanczos estimate of λ_max (the E3 lesson).

### S5. Learned transfer smoothing is inconsistent with "exact Galerkin", and route 1 already measured it as poor value
- The design uses `P̃ = (I − Ω D⁻¹A) P`, but A_2 and A_3 are formed from the unsmoothed P. With P̃ in the transfers and A_ℓ from
  P, the coarse correction is non-Galerkin. It is no longer an A-orthogonal projection, and the stage can increase energy.
- Recomputing `A = P̃ᵀ A P̃` makes the coarse operators depend on the gates, so ℓ3 must be refactorized, with a backward pass
  through the Cholesky, every step. It also interacts with the fringe split (F1).
- Route-1 E3: one damped-Jacobi smoothed aggregation doubled the per-layer cost and moved the 32-layer witness only from 7.07 to
  5.59, and it was rejected.
- **Fix.** Drop P̃ in v0. Put the effort into the energy-minimizing ℓ1 basis (F2 fix ii), which is computed once, is geometry-only,
  and keeps the operators Galerkin.

### S6. The G_θ parameterization is badly scaled on weak nodes and uses geometry-agnostic templates
- `D^{-1/2}(…)D^{-1/2}` with the M̂ and L̂ templates maps a residual at a strong node j to a correction at a weak node i of size
  ~ s·e_j·√(d_j/d_i). That is 10–1000× amplification for d_i/d_j of 1e-2 to 1e-6, so tiny s_e changes swing the fringe values:
  poor conditioning for training, and fringe values are what the sensitivity integrand uses.
- M̂ and L̂ are full-element templates regardless of the material fraction. They couple nodes through near-empty cut elements
  with O(1) normalized strength, where K has ~0.
- **Fix.**
  - Use moment-weighted templates: the cut mass `∫_{mat∩e} N_i N_j` and the cut Laplacian are also linear in the moments.
  - Use left Jacobi scaling D⁻¹, or keep only s_e4 K̂_e.
  - Bound s_e so that the stage stays contractive: tanh times a Lanczos-based cap.

### S7. The mutual-strongest pairs are tie-driven everywhere, not "rarely"
- About 43–45% of elements are interval-full, and all full elements have the *identical* K_e. Nodes in the full bulk therefore
  have exactly tied normalized strengths to their symmetric neighbours.
- So the pairing is decided by index order throughout the bulk. That makes it index-dependent (against the EquiModel
  "index-free" lesson) and non-equivariant at O(1) level, not 1e-3. Also, the pairing is argmax-based and switches
  discontinuously in τ at band changes.
- **Fix.** Use point 3×3 blocks in the bulk. Or use symmetric, tie-free blocks: element-centred overlapping blocks in a
  multiplicative colour sweep. All full elements can then share one factor template, so only the cut elements need their own.
  Keep pairs only as an ablation.

### S8. Implementation risk and schedule
- The v0 has: 4 levels, truncation corrections through moment aggregation plus an unverified assumption that ghost faces inside
  a coarse element contribute nothing (the GP template is not available locally, and "54 = jumps" is DATA_INTERFACE's guess),
  split fine Galerkin, 2k patch factors, a transposed DAG, and 3 modes. "X0 in half a day" is the run time, not the build time.
- **Fix: stage the build.**
  - X0a: ℓ0 fringe/bulk plus a 2-level exact ℓ3. Form every coarse operator by **direct fine-element Galerkin** with composite
    81×81 maps: each fine element lies in exactly one coarse element per level, ≈ 27 GFLOP per level, i.e. ms. Truncation is
    then "zero the port rows of P_e", and no GP-template assumption is needed.
  - X0b: add ℓ1 and ℓ2.
  - Moment aggregation is an optimization, to be verified against X0a.

---

## 3. Minor issues

1. **The target is internally inconsistent (R4).** "μ − 1 ≲ 1%" means a K-norm field error of √0.01 = 10%, not "a few %". Take
   the target from the lattice harness instead: route-1 D2 relates per-cell force-driven energy error to lattice error by a
   factor of ≈ 5–15 in the 2-cell harness. For sensitivity, report the field error in the dK/dτ-weighted norm.
2. **Fringe-factor memory is underestimated.** "~24 MB" becomes ≈ 29–95 MB for weak plus partners (32–58 nodes per patch ×
   1,552 patches, packed fp32), and ≈ 57–330 MB with a strong ring **[rc]**. fp64 doubles these numbers. The 100-cell total
   rises to roughly 12–16 GB: still inside the budget, but the S3/S4 fixes must be costed.
3. **"Weak nodes form connected sheets" is close to a tautology.** It is element-incidence connectivity of the fictitious layer on
   each side of a connected TPMS surface, which gives 2 components in FULL. It says nothing about coupling strength, and it uses
   the body-only proxy with no ghost term. Do not use it as mechanical evidence for the split. Measure instead: the support
   diameter and weak share of the Ritz vectors of the one-pass error operator (the R-2 test).
4. **"No coarse element has split material" is weaker than it sounds.** It checks face-connectivity of *active fine elements*
   (not material), only at τ = 0.5, with one normal. Rerun it at τ = 0.175 with grazing cuts, before deciding the guard "costs
   nothing".
5. **Coarse Chebyshev intervals from Gershgorin.** E3 showed row-sum bounds are 2–3× loose. Use 10-step Lanczos per level at
   encode, as the design already does for ω_b.
6. **Rigid exactness is fp64-exact, not bitwise.** q_d is about 1e-16·‖q‖ rather than 0. This is harmless, since the 7.6 failure
   was at 1e-7. Say "exact to fp64" so that nobody later relies on bitwise zeros.
7. **Readout symmetry in fp32.** The reverse pass is the fp32 transpose, so asymmetry is ~ε₃₂ × (near/far-field cancellation of
   1e3–1e4), i.e. 1e-4–1e-3 in soft directions. That is probably fine for BDD-PCG. Test `|xᵀŜy − yᵀŜx|/√(xᵀŜx·yᵀŜy)` on
   LOBPCG-soft pairs, and do the last K application and the port reduction in fp64.
8. **Throughput assumption.** The 15–30 TFLOPS figure assumes B = 32–64 columns per geometry. In BDD each cell needs only 2–3
   columns per iteration, so the batching must run across heterogeneous cells.
   - A dense-slot layout wastes 2.5× on FULL cells and ≈ 14× on the 0.148 cut (2.3k of 32,768 elements).
   - Ragged gathers lose tensor-core efficiency.
   - Budget 3–8 TFLOPS until the day-1 microbenchmark (R-6) is done.
9. **Training rotates geometries every 250 steps.** With 3 geometries this invites forgetting between blocks. Interleave per
   step, using precomputed exact-solution pools (u* in fp32, energies in fp64) so that factors need not stay resident.
   Recompute on the fly only for adversarial columns.
10. **L_s uses an absolute value.** That is acceptable in a loss (the history's problem was non-smooth *output heads*), but it
    gives sign-noisy gradients. Prefer a squared, element-energy-weighted form.
11. **Band switches make Ê piecewise.** The discrete structures (weak mask, pairs, patches, split) jump at band switches, so Ê,
    and hence Ĉ, is piecewise in τ. Add hysteresis to band switching in the optimizer loop, and let the gates depend smoothly on
    log‖D_i‖ rather than on the binary mask.
12. **The ℓ1 coarse levels also have "weak" coarse nodes.** Their support is mostly fictitious. The coarse polynomial smoother on
    [λ_max/30, λ_max] ignores them. The fringe split handles this only at ℓ2.

---

## 4. Arithmetic check (FULL, τ = 0.5)

| claim | check | verdict |
|---|---|---|
| ℓ2 split Galerkin 6.4 MFLOP/el, 82 GFLOP, 18.6 MB | 2·81·81·162 + 2·162·81·162 = 6.4M; ×12,880 = 82 G; 352 × 13,203 × 4 B = 18.6 MB | correct |
| ℓ3 packed fp32 9.6 MB; 11.7k coarsest 0.5 GB | 2187·2188/2·4 = 9.57 MB; 11,658²·4 = 0.54 GB | correct |
| 3 fine K ≈ 2.7 GFLOP | 0.37 (strain) or 0.17 (K_e) + 0.64 (ghost) per apply → 2.4–3.0 | correct |
| Ŝq ≈ 11.5 GFLOP, 0.4–0.8 ms | 5 + 0.9 + ~5.6 reverse; 11.5 G / 15–30 TFLOPS | arithmetic correct; throughput unverified (minor 8) |
| 100 FULL cells ≈ 10 s per design iteration | 100·(160·11.5 + 55·0.6) G / 20 TFLOPS = 9.4 s | correct at T = 1; 23 s at T = 4, 40 s at T = 8 |
| fringe factors ~24 MB | 1,552 patches with weak nodes; 32–58 nodes → 29–95 MB | underestimated 1.2–4× |
| "20–60 weak nodes per patch" | median 16, p90 29, max 33–54 | overstated |
| GNN ≈ 25k weights per round | 3 × (64·64 + 64·32 + biases) ≈ 18.7k, plus wider inputs | plausible |
| encode 0.1–0.2 s | K_e 10.7 GFLOP; strengths 9.5M pairs; split Galerkin 82 GFLOP; ≈2k small Cholesky (fp64 rate 1/64) | plausible; the fp64 fringe factors could add 10–50 ms |
| Cholesky of ℓ3 fp64 in 2–5 ms | 2187³/3 = 3.5 GFLOP at ≤ 1.6 TFLOPS fp64 | ≥ 2.2 ms at peak; realistically 5–10 ms |

---

## 5. Component-by-component verdict

| component | verdict | note |
|---|---|---|
| Q0 fp64 rigid split, invariant W, P_W on both sides | **keep** | clean; fixes the ROUTES 7.6 leakage |
| Zero lift L0 + port-truncated coarse functions (ports by mask) | **keep** | the correct way to handle box and oblique band by masks alone |
| Exact-K stage residuals (error-product structure) | **keep** | but budget with the XZ identity, not per-component products (F2) |
| Q2 h-hierarchy 65³→33³→17³→9³ | **keep** | nesting, dense slot layout |
| Coarse basis = plain Q2 polynomials | **modify** | energy-minimizing ℓ1 basis (MsFEM-type local K-harmonic) to cure locking (F2) |
| Moment-aggregation Galerkin at ℓ1 | **modify** | build by direct fine-element Galerkin first; aggregation only as a verified optimization (S8) |
| ℓ3 dense Cholesky | **keep** | store fp64, or fp32 + refinement (S3) |
| ℓ2 strong/fringe split | **modify** | a separate multiplicative stage with its own residual (F1a), or defer to X3b |
| Learned transfer smoothing P̃ | **drop** (v0) | non-Galerkin; poor value in route-1 E3 (S5) |
| Coarse polynomial smoothers | **keep** | Lanczos bounds (minor 5) |
| Fringe patches | **modify** | multiplicative 8-colour + staggered pass, strong ring, equilibrated factors; GenEO variant if memory binds (S4) |
| Bulk mutual-strongest pairs | **modify** | tie-free symmetric blocks or point blocks (S7) |
| G_θ element kernel | **modify** | moment-weighted templates, D⁻¹ or K̂_e scaling, contractive bound (S6); optionally grow into a multichannel linear GNO (S2) |
| FMG backslash pass | **keep** | |
| T = 1 default | **modify** | T budgeted, plus a learned cross-cycle recurrence (F2) |
| Stage step lengths η | **keep** | extend to per-cycle recurrence coefficients |
| Hyperedge GNN generator (K's hypergraph, orbit-class aggregation, invariant features) | **keep** | good answer to question a and to equivariance |
| Readout Ŝq = reverse pass; (Kû)_ports − Ŝq as an indicator | **keep** | fp64 final K and port reduction; symmetry test (minor 7) |
| Mode N additive BPX | **modify** | symmetric multiplicative cycle plus Chebyshev degree k; spectral training loss (S1) |
| Sensitivity −ûᵀ ∂K/∂τ û with frozen per-band structures | **keep** | add the dK-weighted field error metric |
| Equivariance by construction + 25% augmentation | **keep** | after the S7 fix |
| Loss L_E + L_D + L_s + L_N | **modify** | squared L_s; spectral L_N; target from the lattice harness (minor 1) |
| X0 / X1 / X2 ladder with stage-wise error energies | **keep** | add X0a (2-level) first and the Ritz test of `B K_II` on the force-driven Krylov space |

---

## 6. Best ideas to carry into a synthesis

1. **Error-product design principle with exact-K residuals between stages.** w* is a fixed point of every stage, the
   generalization comes for free (the residual is exact for unseen geometry), and it degrades continuously into fallback A
   through T.
2. **Ports by truncation.** Every coarse function is truncated on the port mask. That one mechanism handles box patches, the
   oblique cut band, and mode N (an empty mask), and it represents the one-element lift ramp in the first stage.
3. **Exact Galerkin coarse operators generated in milliseconds from moments and masks**, with a dense exact coarsest level
   (≤ 2.2k DOF) for global transmission. No learned far field and no spectral layers.
4. **One network, three port types by mask (F, D, N)**, including the loaded-cut load vector ĝ from the same reverse pass.
5. **Numerical-prior initialization of every generated coefficient**, and the X0 → X1 → X2 ladder that separates method
   quality, coefficient-field capacity and generator gap. This is the most useful process idea in the document.
6. **Stage-wise error energies from exact u*** as the diagnostic that says which stage leaves which error. Also
   **(Kû)_ports − Ŝq** as a free a-posteriori interior-imbalance indicator.
7. **Hyperedge GNN on K's own hypergraph** (element and GP-face hyperedges, O_h orbit-class aggregation, invariant mechanical
   features such as the 6×6 uniform-strain energy spectrum) as the coefficient generator.
8. **An explicit, structural carrier for the weak fringe** (patches plus a fringe coarse space). The concept is right and backed
   by the ROUTES slow-mode data. It needs the S4/F1 repairs and a direct-solve (GenEO) character rather than relaxation.
   - The fringe also matters for sensitivity: ∂M/∂τ lives on the boundary elements whose nodes are the fringe.
