# Consolidated issue register: internal review round 1

**Manuscript:** "Learned static condensation for cut thin-walled TPMS cells with equilibrium correction" (method: NICE; CMAME submission)
**Inputs:** R1_mechanics, R2_ml, R3_solvers, R4_optimisation, R5_editor, R6_factcheck, R7_critical (135 numbered findings, plus the Section 6.11 requirement lists of R1, R2, R4 and R7 and R5's title recommendation).
**Clerk's note:** Nothing in the manuscript or the review reports was changed. Finding IDs follow the reports. "(part)" means only one part of a multi-part finding belongs to the issue. R6 severities are mapped as follows: Error → Major; Inconsistency/Unsupported → Minor, or Major when a headline claim (Abstract/Conclusions) is affected; Style → Minor. Word and cost estimates marked "clerk's estimate" are not the reviewers' own.

---

## 0. Statistics

### 0.1 Reviewer recommendations

| Reviewer | Focus | Recommendation | Findings (Major / Minor) |
|---|---|---|---|
| R1 | Mechanics, condensation theory, CutFEM | Major revision | 17 (6 / 11) |
| R2 | ML for computational mechanics | Major revision | 17 (8 / 9) |
| R3 | Numerical linear algebra, solvers, benchmarking | Major revision | 15 (9 / 6) |
| R4 | Structural/topology optimisation | Major revision | 17 (9 / 8) |
| R5 | Handling editor (pre-review) | Desk-reject risk **medium**. It is **high** if submitted with the placeholders, and falls to **low–medium** once §6.11 is complete and the length plan is applied. No accept/reject recommendation. | 20 (8 / 12) + title recommendation |
| R6 | Fact-check | No recommendation. About 430 numbers verified. | 32 (2 Error, 12 Inconsistency, 5 Unsupported, 13 Style) |
| R7 | PIML / learned substructures (critical) | Major revision | 17 (10 / 7) |

All five scientific reviewers recommend **major revision**. None considers the construction itself flawed, and all praise the candour of the disclosures.

### 0.2 Consolidated issues

- **Total: 59 issues** (from 135 findings).
- Severity: 36 Major, 23 Minor.
- Consensus:
  - 6 reviewers: 5 issues;
  - 5 reviewers: 5 issues;
  - 4 reviewers: 11 issues;
  - 3 reviewers: 6 issues;
  - 2 reviewers: 18 issues;
  - 1 reviewer: 14 issues.
- Status:
  - 4 issues **DECIDED**: I-01 (baseline choice), I-10, I-29, I-33 (pilot running);
  - 28 issues flagged **NEEDS AUTHOR DECISION** (list in §0.5).

### 0.3 Issues per theme

| Theme | # | Issues |
|---|---:|---|
| T1 Cost & baselines | 13 | I-01, I-02, I-06, I-07, I-11, I-12, I-13, I-22, I-27, I-36, I-41, I-51, I-52 |
| T2 Sensitivity / design derivative | 5 | I-03, I-08, I-25, I-31, I-50 |
| T3 Accuracy evidence & evaluation independence | 8 | I-16, I-17, I-18, I-24, I-30, I-39, I-42, I-48 |
| T4 Numerics / precision / solver | 4 | I-04, I-20, I-23, I-38 |
| T5 Novelty, attribution & PIML fairness | 4 | I-09, I-14, I-15, I-35 |
| T6 ML ablations & training | 5 | I-19, I-26, I-33, I-34, I-43 |
| T7 Scope, limitations & reproducibility | 5 | I-05, I-32, I-40, I-44, I-49 |
| T8 Structure, length & presentation | 7 | I-21, I-37, I-45, I-53, I-54, I-55, I-56 |
| T9 Fact-check corrections | 7 | I-28, I-29, I-46, I-47, I-57, I-58, I-59 |
| T10 Design-optimisation example requirements | 1 | I-10 (see also §2) |
| **Total** | **59** | |

### 0.4 Issues per fix type

Each issue is counted once, under the most demanding action that the recommended fix needs.

| Primary fix type | # | Issues |
|---|---:|---|
| Text only | 27 | I-01*, I-05, I-09, I-14, I-15, I-16, I-21, I-25, I-27, I-28, I-29, I-39, I-40, I-41, I-42, I-44, I-45, I-46, I-47, I-49, I-50, I-52, I-55, I-56, I-57, I-58, I-59 |
| Re-analysis of existing data / checkpoints | 13 | I-02, I-03, I-08, I-18, I-19, I-23, I-26, I-32, I-38, I-43, I-48, I-51, I-53 |
| New experiment | 15 | I-04, I-06, I-07, I-10, I-11, I-12, I-13, I-17, I-20, I-24, I-30, I-31, I-33, I-34, I-36 |
| Cut or move | 4 | I-22, I-35, I-37, I-54 |

\* I-01 is text only under the DECIDED host baseline. It would become a new experiment if that decision were revisited.

A new experiment is at least one proposed option (including optional ones) in 24 issues: I-01\*, I-04, I-05, I-06, I-07, I-10, I-11, I-12, I-13, I-17, I-19, I-20, I-24, I-27, I-30, I-31, I-33, I-34, I-35, I-36, I-42, I-48, I-51, I-52.

### 0.5 Issues flagged NEEDS AUTHOR DECISION

| ID | One-line title | Why a decision is needed |
|---|---|---|
| I-01 | GPU-vs-host cost comparison and the headline cost claims | Claim wording in the Abstract and §8 (the baseline itself is DECIDED) |
| I-02 | Offline cost and break-even not reported | Claim "cheaper for any number of queries" |
| I-03 | Complete surrogate derivative never evaluated | Choice of the gradient used in §6.11 |
| I-04 | Finite-precision residual floor in assembled solves | fp64 re-runs; Abstract numbers may change |
| I-05 | Scope stated too generally; no Limitations section; no OOD test | Title/Abstract scope; OOD experiment |
| I-06 | Asymmetric timing conditions (front end, shared host, stopping) | Benchmark re-runs |
| I-07 | Weak graph-harmonic baseline; scope of "7 to 290 times" | Abstract claim; new runs |
| I-11 | No whole-lattice iterative / DD baseline | New experiment; relation to the DECIDED baseline |
| I-12 | No matched-accuracy (Pareto) comparison against non-learned corrections | New experiment |
| I-13 | Scalability beyond eight cells not shown; assembled solver under-specified | New experiment |
| I-14 | "Accurate compliance ≠ accurate sensitivity" framed as the principal finding | Headline claim in Abstract/§8 |
| I-15 | Classical/inherited results presented as contributions | Contributions list and Abstract |
| I-16 | "80 unseen geometries" contradicts the selection overlap | Abstract wording (low controversy) |
| I-17 | Assembled accuracy rests on development cells | New experiment; Abstract claim |
| I-18 | Worst-direction / operator-norm error not reported | Abstract wording "at most 0.65% for any geometry" |
| I-19 | Benefit of training through the correction is confounded | Seeds (new); contribution (ii) and Abstract claim |
| I-20 | No runtime error control; one-sided bias invites exploitation | New implementation and experiment |
| I-22 | Supplementary Note S4 / Table ST14 still contains same-GPU exact-factor timings | Keep, remove or re-frame (explicit author decision) |
| I-23 | Two-grid bound proves only non-expansion; "controllable" claim | Abstract/contribution (i) wording |
| I-24 | CutFEM reference verification too thin | New (small) experiment |
| I-28 | Memory units (GiB vs GB) and memory content | Mandatory fix; the Abstract number changes |
| I-30 | Lattice examples small, favourable and not shown | New experiment |
| I-31 | Non-smoothness of Ĉ(τ) across discrete switches | New (inexpensive) experiment |
| I-34 | Sensitivity term and architecture not ablated | New experiment |
| I-35 | §6.8 is not a fair PIML proxy (reviewers disagree on its role) | Disagreement R5 vs R7; Intro claim |
| I-36 | Host PARDISO baseline configured unfavourably | Benchmark re-runs |
| I-45 | Title wording | Title (the NICE name is DECIDED) |
| I-55 | Abstract too long and dense | Which claims and numbers stay in the Abstract |

---

## 1. Issue register

Order: first the issues with consensus ≥3 (sorted by consensus, then severity), then the remaining Major issues, then the remaining Minor issues.

### Summary index

| ID | Title | Theme | Cons. | Sev. | Primary fix | Status |
|---|---|---|---:|---|---|---|
| I-01 | GPU-vs-host cost comparison and headline cost claims | T1 | 6 | Major | Text | DECIDED + NEEDS DECISION |
| I-02 | Offline cost (data, training) and break-even not reported | T1 | 6 | Major | Re-analysis | NEEDS DECISION |
| I-03 | Complete surrogate derivative never evaluated | T2 | 6 | Major | Re-analysis | NEEDS DECISION |
| I-04 | Finite-precision residual floor in assembled solves | T4 | 6 | Major | New exp. | NEEDS DECISION |
| I-05 | Scope too general; no Limitations; no OOD test | T7 | 6 | Major | Text | NEEDS DECISION |
| I-06 | Asymmetric timing conditions | T1 | 5 | Major | New exp. | NEEDS DECISION |
| I-07 | Weak harmonic start; scope of "7–290×" | T1 | 5 | Major | New exp. | NEEDS DECISION |
| I-08 | Sensitivity metric not the one an optimiser sees | T2 | 5 | Major | Re-analysis | |
| I-09 | Closest prior work missing | T5 | 5 | Major | Text | (context: repositioning DECIDED) |
| I-10 | §6.11 design-optimisation example (placeholder) | T10 | 5 | Major | New exp. | DECIDED |
| I-11 | No whole-lattice iterative / DD baseline | T1 | 4 | Major | New exp. | NEEDS DECISION |
| I-12 | No matched-accuracy Pareto against non-learned corrections | T1 | 4 | Major | New exp. | NEEDS DECISION |
| I-13 | Scalability beyond 8 cells; assembled solver under-specified | T1 | 4 | Major | New exp. | NEEDS DECISION |
| I-14 | Contribution (iv) as the principal finding | T5 | 4 | Major | Text | NEEDS DECISION |
| I-15 | Classical/inherited results presented as contributions | T5 | 4 | Major | Text | NEEDS DECISION |
| I-16 | "80 unseen geometries" | T3 | 4 | Major | Text | NEEDS DECISION |
| I-17 | Assembled accuracy rests on development cells | T3 | 4 | Major | New exp. | NEEDS DECISION |
| I-18 | Worst-direction / operator-norm error missing | T3 | 4 | Major | Re-analysis | NEEDS DECISION |
| I-19 | Training through the correction confounded | T6 | 4 | Major | Re-analysis | NEEDS DECISION |
| I-20 | No runtime error control; surrogate exploitation | T4 | 4 | Major | New exp. | NEEDS DECISION |
| I-21 | Labels, notation, colour coding | T8 | 4 | Major | Text | (context: NICE name DECIDED) |
| I-22 | Note S4 / Table ST14 same-GPU exact-factor timings | T1 | 3 | Major | Cut/move | NEEDS DECISION |
| I-23 | Two-grid bound vacuous; "controllable" claim | T4 | 3 | Major | Re-analysis | NEEDS DECISION |
| I-24 | CutFEM reference verification too thin | T3 | 3 | Major | New exp. | NEEDS DECISION |
| I-25 | FD-step statement contradictory; PSD of numerical K,c | T2 | 3 | Major | Text | |
| I-26 | View dependence and checkpoint-selection description | T6 | 3 | Minor | Re-analysis | |
| I-27 | Benchmark reporting and table layout | T1 | 3 | Minor | Text | |
| I-28 | Memory units (GiB vs GB) and memory content | T9 | 2 | Major | Text | NEEDS DECISION (must fix) |
| I-29 | Ratios computed from rounded Table 5 values | T9 | 2 | Major | Text | DECIDED |
| I-30 | Lattice examples small, favourable, not shown | T3 | 2 | Major | New exp. | NEEDS DECISION |
| I-31 | Non-smoothness across discrete switches | T2 | 2 | Major | New exp. | NEEDS DECISION |
| I-32 | τ → wall thickness/density map; admissible design domain | T7 | 2 | Major | Re-analysis | |
| I-33 | Dependence on exact data; data-free variant | T6 | 2 | Major | New exp. | DECIDED (pilot running) |
| I-34 | Sensitivity term and architecture not ablated | T6 | 2 | Major | New exp. | NEEDS DECISION |
| I-35 | §6.8 not a fair PIML proxy | T5 | 2 | Major | Cut/move | NEEDS DECISION |
| I-36 | Host PARDISO configured unfavourably | T1 | 1 | Major | New exp. | NEEDS DECISION |
| I-37 | Length far above CMAME norm; Discussion repeats | T8 | 1 | Major | Cut/move | |
| I-38 | Chebyshev interval and coarse-solve verification | T4 | 2 | Minor | Re-analysis | |
| I-39 | δ²κ definitions inconsistent and over-interpreted | T3 | 2 | Minor | Text | |
| I-40 | Connectivity / rank-six rigid-mode assumption | T7 | 2 | Minor | Text | |
| I-41 | Reuse and warm starts across design iterations | T1 | 2 | Minor | Text | |
| I-42 | Surrogate error vs reference discretisation error | T3 | 2 | Minor | Text | |
| I-43 | Training-population counts (591 / 305) | T6 | 2 | Minor | Re-analysis | |
| I-44 | Architecture reproducibility; code/data availability | T7 | 2 | Minor | Text | |
| I-45 | Title | T8 | 2 | Minor | Text | NEEDS DECISION (NICE name DECIDED) |
| I-46 | "Nine configurations common to all predictors" | T9 | 2 | Minor | Text | |
| I-47 | Lattice numbers: §6.9 vs Table 6 vs §8 | T9 | 2 | Minor | Text | |
| I-48 | Cut-surface loads not reported for A3 | T3 | 1 | Minor | Re-analysis | |
| I-49 | Interface matching with cut neighbours | T7 | 1 | Minor | Text | |
| I-50 | Design-dependent loads frozen | T2 | 1 | Minor | Text | |
| I-51 | Learned application barely benefits from batching | T1 | 1 | Minor | Re-analysis | |
| I-52 | Retained cut band when cut surfaces are traction-free | T1 | 1 | Minor | Text | |
| I-53 | Fig. 10 omits NICE; overlaps Fig. 9 / Table 4 | T8 | 1 | Minor | Re-analysis | |
| I-54 | Main text relies on appendix equations; uncited appendices | T8 | 1 | Minor | Cut/move | |
| I-55 | Abstract too long and dense | T8 | 1 | Minor | Text | NEEDS DECISION |
| I-56 | Training described before the correction | T8 | 1 | Minor | Text | |
| I-57 | Operator-consistency numbers in §6.2 | T9 | 1 | Minor | Text | |
| I-58 | Descriptive wording precision (§6.3, §6.4, §6.6) | T9 | 1 | Minor | Text | |
| I-59 | Numbers without an archived source | T9 | 1 | Minor | Text | |

---

### Part A: Consensus ≥3

#### I-01: GPU-vs-host cost comparison and the headline cost claims
- **Theme:** T1 · **Severity:** Major · **Consensus:** 6 (R1, R2, R3, R4, R5, R7)
- **Sources:** R1-06 (part), R2-04 (part), R3-01 (part), R3-15, R4-07 (part), R5-05, R7-05 (part)
- **Status:**
  - **DECIDED.** The GPU interior factorisation (cuDSS) comparison was deliberately removed from Table 5 and the discussion. The authors consider host MKL PARDISO the conventional practice and the relevant baseline, as in the PIML literature, which compares against full-scale FEM.
  - **NEEDS AUTHOR DECISION** on how the Abstract, §6.10, §7.4 and §8 cost claims are worded under that decision.
- **Problem:** The main text compares A3 on one RTX 5090 with MKL PARDISO on 16 host cores. The resulting claims are stated without their hardware condition:
  - Abstract: "condenses a cell 9 to 29 times faster, stores 4 to 14 times less, and applies it 2 to 23 times faster";
  - §6.10 and §8: "cheaper for any number of queries".

  All scientific reviewers and the editor object that algorithm and hardware are confounded (R2-04, R5-05). GPU sparse direct solvers (cuDSS, CHOLMOD-GPU) and GPU batched triangular solves exist, and a CMAME reviewer is expected to ask for one (R5-05, R3-01, R2-04, R1-06). R7-05 notes that the single-vector speed-up of 7.4–23× is largely a host-vs-GPU factor: G1 takes 826 ms with host PARDISO against about 29 ms for a GPU factor. R3-15 lists the missing qualifiers: host LU with 16 threads against a GPU, a host front end, and a shared host. R4-07 adds that the claim is only true relative to the chosen CPU direct baseline. R7-05 and R3-10 suggest that memory, not speed, may be the most defensible advantage.
- **Proposed actions (under the decision):**
  1. Add one sentence to §6.10 justifying the baseline: host sparse direct condensation is conventional practice, and PIML studies compare against full-scale FEM.
  2. Qualify every cost claim with the hardware, e.g. "on one GPU against a 16-core host direct solver", in the Abstract, §6.10, §7.4 and §8.
  3. Decide whether "cheaper for any number of queries" survives. Reviewers ask for removal or qualification; offline cost (I-02) and the same-GPU data still present in the supplement (I-22) both bear on it.
  4. Consider leading with the memory advantage (after I-28 and I-36).
  5. Prepare a response-letter argument, because R2, R3, R5 and R7 will very likely repeat the request.
- **Fix type:** Text only (under the decision). It becomes a new experiment (same-GPU timings) if the decision is revisited.
- **Related:** I-02, I-06, I-11, I-12, I-22, I-28, I-29, I-36.

#### I-02: Offline cost (data generation and training) and break-even not reported
- **Theme:** T1 · **Severity:** Major · **Consensus:** 6 (R1, R2, R3, R4, R6, R7)
- **Sources:** R1-06 (part 3), R2-05, R3-08, R4-08 (part), R6-15, R7-06 (part)
- **Status:** NEEDS AUTHOR DECISION (the "cheaper for any number of queries" wording in the Abstract and §8, shared with I-01).
- **Problem:** Only "A3's continuation took 2.7 h on one RTX 5090" is reported. R6-15 finds that the log starts at step 4,050 after a restart and ends at 9,806.6 s, so it covers about 11,000 of the 15,000 steps; C took 5,799 s for all 15,000.

  The following are not reported:
  - the time of B's 40,000 steps;
  - label and bank generation for 148 + 305 + 591 geometries: interior factorisations, Neumann/pinned solves for the mechanically defined direction classes, qᵀSq normalisation, eight-corner finite-difference sensitivity labels, and S⁻¹ applications in the adversarial search;
  - checkpoint selection;
  - total GPU-h and CPU-h.

  By the nominal class weights, the classes derived from exact solves make up 77.5% of sampled directions (R7-06). R2-05 gives a rough break-even: learned preparation saves about 7–53 s per cell, so the 2.7 h continuation alone needs about 180–1,450 cell preparations to amortise. Because the operator is tied to one family, n, ν and γ, the offline cost recurs for every new setting (R3-08, R4-08). Offline cost is also the standard criticism of the PIML-type approaches the paper contrasts itself with (R2-05, R7-06).
- **Proposed actions:**
  - Report, per training geometry, the directions per class, the number of exact solves and the generation time.
  - Report the total offline cost (data plus all training stages) in GPU-h and CPU-h, and the storage.
  - Correct the 2.7 h statement: either the full wall time, or "2.7 h for the final 11,000 steps after a restart".
  - Give a break-even count in cell preparations and in design iterations (e.g. for the §6.11 lattice).
  - State the reuse domain next to the cost claims.
  - Qualify "any number of queries".
- **Fix type:** Re-analysis of existing logs, plus text.
- **Related:** I-01, I-05, I-33, I-43.

#### I-03: The complete derivative of the surrogate objective is never evaluated
- **Theme:** T2 · **Severity:** Major · **Consensus:** 6 (R1, R2, R4, R5, R6, R7)
- **Sources:** R1-02 (part), R1-17(d), R2-14, R4-01, R4-13, R4-17, R5-01 (part), R6-14, R7-17
- **Status:** NEEDS AUTHOR DECISION on which gradient drives §6.11 (field-based s̃_c or complete Ĉ,c).
- **Problem:** Every reported sensitivity is the field estimate s̃_c = −ûᵀK,cû, which is only the first term of Eq. (14). The gradient of the objective the surrogate actually evaluates is Ĉ,c = s̃_c − 2(F_{I,c}q̂)ᵀr_I, and it is never computed. The (H.5) bound involves ‖F_{I,c}q̂‖_A, which is never measured.

  The §7.2 argument does not settle the question:
  - The "about 2.7% of ‖û‖_K" is relative to ‖û‖_K, not to |s_c|. For weakly participating cells |s_c| ≪ ‖û‖²_K (R1-02).
  - R4-01 estimates a relative bound of at least about 5% per component, and about 16% at the worst-geometry excess of 0.65%, i.e. an order above the reported 0.7%. The bound may be pessimistic: the J.5 cancellation could make Ĉ,c *more* accurate than s̃_c.
  - J.9 example 5 shows that the complete derivative can even have the wrong sign.
  - Numerical details of the argument are off: "26% for B" should be 25.4% for the ‖û‖_K normalisation (R6-14), and a population mean ε is combined as though √mean = mean√ (R1-17d).

  §7.2 defers the measurement to "subsequent work", which conflicts with the planned §6.11 (R5-01). The deployed "reverse-mode sensitivities" (S7, Table 6) are not specified: which quantity, how K,c is formed, and the cost per cell (R4-13). No recommendation is given on which sensitivity to use with OC or MMA (R4-17). An optimiser that evaluates Ĉ but is driven by s̃ uses an inconsistent gradient, which affects line search, GCMMA inner loops and KKT checks (R2-14, R4-01, R7-17).
- **Proposed actions:**
  1. Compute Ĉ,c by reverse-mode differentiation through the network, the rigid split and the correction (training already backpropagates through the correction). Alternatively, use central differences of Ĉ with the full pipeline re-evaluated.
  2. Verify against finite differences at several step sizes.
  3. On all 14 pair configurations and both lattices, report exact C,c, s̃_c and Ĉ,c side by side. Give the residual term relative to |s_c| and to ‖∇C‖.
  4. Rewrite §7.2 (including the R6-14 and R1-17d corrections).
  5. Add a short "Recommendation for design use" paragraph (§4.3 or §7.2).
  6. Specify the deployed sensitivity (quantity, K,c formation, cost per cell) and confirm that it matches the teacher.
- **Fix type:** Re-analysis (new computations on existing configurations and checkpoints).
- **Related:** I-08, I-10, I-31, I-50.

#### I-04: Finite-precision residual floor in the assembled solves
- **Theme:** T4 · **Severity:** Major · **Consensus:** 6 (R1, R2, R3, R4, R5, R7)
- **Sources:** R1-03, R2-15, R3-05, R4-12, R5-12, R7-07 (part), R7-10 (part 4)
- **Status:** NEEDS AUTHOR DECISION (fp64 verification re-runs; the Abstract's lattice numbers may change).
- **Problem:** In §6.9 the recursive PCG residual reaches about 1e-10, but the recomputed residual stagnates at 3.6×10⁻⁴ (2×2×2) and 3.2×10⁻³ (3×3×1), i.e. 3–7 orders higher. The floor grows tenfold with layout. Meanwhile:
  - lattice compliance errors are about 1×10⁻⁴ relative, and pair errors go down to 10⁻⁶ relative, yet no residual was saved for the pairs (S1.2);
  - by Eq. (18), such residuals can contaminate compliance through Ūᵀρ, and through ω if the fp32 action differs from the energy operator. Neither term is stored (J.6).

  The "0.3–0.4%" agreement with β and the smallest compliance errors therefore cannot yet be read as properties of the operator (R1-03). Further inconsistencies and gaps:
  - Table 1 says the correction runs in fp64, but App. F.3 rounds fields to fp32 between stages and on output. The ST08a fused/sparse action difference is 2.5e-5 (R3-05).
  - Attributing the floor to the network's fp32 arithmetic is asserted, not shown (R3-05, R5-12).
  - The floor sits uneasily with the 10⁻⁸ consistency figures of §6.2 (R5-12).
  - Iterating past the attainable accuracy inflates the 183/186 application counts used in the cost argument (R3-05).
  - The floor may grow with lattice size (R7-07). It limits how far the correction budget can reduce the error (R7-10) and may set the attainable gradient accuracy in optimisation (R4-12).
  - The historical 27-cell D run had a recomputed residual of 8.99e-3 and a 1.86% compliance error (ST14d).
- **Proposed actions:**
  - Re-run one pair and one lattice with the network, rigid split and retained restoration in fp64, all casts removed. Show whether the stall disappears and whether the errors change.
  - For every assembled result, report the recomputed residual, Ūᵀρ/C and ω/C, or the residual-corrected J(Ū) (J.6).
  - Save true residuals for the pair runs.
  - Check GPU determinism (bitwise-repeatable applications).
  - Report the floor against the number of cells.
  - Use a stopping criterion tied to attainable accuracy (residual replacement or mixed-precision iterative refinement; Carson & Higham 2018).
  - Make Table 1 consistent with F.3.
  - State that the pair runs use the exact reference factor as preconditioner, so their iteration counts (6–16) say nothing about deployable cost.
  - State which compliance errors are resolved above the algebraic floor.
- **Fix type:** New experiment (small re-runs), plus re-analysis and text.
- **Related:** I-13, I-23, I-47, I-57.

#### I-05: Scope stated too generally; no Limitations section; no out-of-distribution test
- **Theme:** T7 · **Severity:** Major · **Consensus:** 6 (R1, R2, R3, R4, R5, R7)
- **Sources:** R1-14 (part), R2-07, R3-08 (part: reuse domain), R4-08 (part), R5-07, R7-14
- **Status:** NEEDS AUTHOR DECISION (Title/Abstract scope wording; whether to run an OOD test).
- **Problem:** Training and all tests cover:
  - one TPMS family (Schwarz-P, Eq. 1);
  - one planar cut with normal (cos ϑ, sin ϑ, 0), 0 < ϑ < π/4. Under the 48 cube symmetries this reaches only normals with at least one zero Cartesian component; general oblique cuts are unseen (R2-07);
  - no edge or corner cells (two or three cuts);
  - n = 32, fixed by the 65/33/17/9 network hierarchy;
  - E = 1, ν = 0.3, γ = 1e-4;
  - τ ∈ [0.1752, 0.6993] with corner span and gradient norm ≤ 0.47.

  Every test is in distribution (lattice corners 0.25–0.56). Title and abstract say "cut thin-walled TPMS cells" in general. There is no Limitations section; limitations are scattered through §6.1, §6.2 and §7 (R4-08, R5-07). Gyroid and diamond cells, more common in metal AM, are not addressed (R4-08). R7-14 notes that a family of about 10 parameters is served by a 603k-parameter network, and that parametric ROM or operator interpolation (SCRBE-style) is a natural competitor. The network is not equivariant, and A3 is not evaluated in another view (R1-14; see I-26). An optimiser may drive designs out of the box (R1-14, R2-07).
- **Proposed actions:**
  - Write "Schwarz-P-type" in the Abstract (title: see I-45).
  - Add a "Limitations and extensions" paragraph of about 150–200 words at the end of §7. Cover family and resolution dependence and retraining cost; the single planar cut; evidence confined to the discrete reference (about 1% discretisation error for heavy cuts, I-42); and the hardware-specific cost comparison.
  - State the reuse domain beside the cost claims.
  - Optionally add a small OOD stress test: oblique normals with three non-zero components; corners 10–20% outside the range; another ν; a doubly cut corner cell; one gyroid cell. Run each with and without correction and against harmonic start + correction. Demonstrate the Eq. (10) residual indicator as an OOD flag (I-20).
  - Discuss parametric ROM alternatives.
- **Fix type:** Text only (minimum); new experiment (OOD test, requested by R2 and R7, optional for R4).
- **Related:** I-20, I-26, I-32, I-45.

#### I-06: Asymmetric timing conditions (front end, shared host, stopping criteria)
- **Theme:** T1 · **Severity:** Major · **Consensus:** 5 (R3, R4, R5, R6, R7)
- **Sources:** R3-03, R4-07 (parts 3–4), R5-05 (part), R6-02, R7-05 (part), R7-13
- **Status:** NEEDS AUTHOR DECISION (benchmark re-runs). This issue is compatible with the DECIDED host baseline: it asks for a fair host baseline, not a different one.
- **Problem:** The front end (setup, moment integration, assembly) is method-independent, yet it runs on the host for the conventional route and on the GPU for the learned route. The "ready for queries" factor of 3.1–8.1 is therefore largely hardware.
  - In Table 6 the host front end is 140–152 s of the 242–256 s direct time. The solver phases alone are 91 s (Cholesky) against 30–31 s (learned setup and PCG) (R3-03, R4-07, R7-13, R5-05).
  - Table 6's direct runs were made on a shared host (load average 11–37), while the Table 5 caption claims an "otherwise idle" job.
  - R6-02 (Error): the caption's "a first run with other jobs … agreed within 10%" is false for the G1 64-vector application (4.38 vs 5.81 s, −25%) and for G3's explicit S (1,873 vs 1,609 s, +16%). The quiet run's load average (10–18) is no lower than the first run's (9–12.5).
  - The routes are stopped differently: direct solves six loads to 1e-10, while the learned route solves three loads to a recursive 1e-6, with its true residual unreported (R3-03).
  - The eight-cell Cholesky factorisations were skipped under a 60 GB threshold, although 66.6 GB plus 5–6 GB overhead fits in 90 GB. The eight-cell entries are lower bounds.
  - The direct times exclude sensitivities (R7-13).
- **Proposed actions:**
  - Run the front end on the same device for both routes, or make the solver-phase comparison primary. Remove the front-end ratio from the method claims (R5-05).
  - Repeat the Table 6 direct runs on an idle host, with repetitions and spread.
  - Match load counts and accuracy (e.g. compliance to 1e-4).
  - Run the eight-cell Cholesky factorisation.
  - Report sensitivities for both routes.
  - Correct the Table 5 caption (R6-02).
- **Fix type:** New experiment (benchmark re-runs); text only as the minimum.
- **Related:** I-01, I-27, I-36.

#### I-07: Weak graph-harmonic starting-field baseline and the scope of "7 to 290 times"
- **Theme:** T1 · **Severity:** Major · **Consensus:** 5 (R1, R2, R3, R6, R7)
- **Sources:** R1-06 (part 2), R1-15 (part), R2-16, R3-07 (part), R3-15 (part), R6-27, R7-12
- **Status:** NEEDS AUTHOR DECISION (Abstract claim; cheap new runs).
- **Problem:** The Abstract's "the learned starting field is 7 to 290 times more accurate than a harmonic extension" rests on:
  - five development cells;
  - one loading class (consistent tractions);
  - 32 directions;
  - the fixed-weight B rather than A3 (R1-15, R2-16, R7-12, R3-15).

  The comparator is a scalar, material-volume-weighted graph Laplacian with no elastic coupling. It is weak: in ST18 it ends *worse* than the zero start after 32/Q1/32 for U2 (1.25 vs 1.19) and H2 (0.0246 vs 0.0056) under nodal forces (R1-06). R3-07 adds two points. The single geometric Q1 coarse space is not topology-aware (one coarse function spans disconnected wall segments), and it is truncated at the retained nodes, exactly where B's error concentrates. The 32/Q1/32 rows only add smoothing, the component known to stall on low modes. The Conclusions' "zero interior 2,400 to 12,000 times larger" is correct, but it is not stated in §6.5 or Table 3 (R6-27).
- **Proposed actions:**
  - Add cheap non-learned starts under the same correction:
    - coarse Galerkin solve first;
    - vector elastic harmonic extension (a few AMG V-cycles on A);
    - lower-resolution exact extension (n = 16) interpolated;
    - m two-grid cycles;
    - an SA-AMG V-cycle with rigid near-nullspace;
    - a three-level V-cycle on the 33/17/9 hierarchy.
  - Run zero, harmonic and elastic starts on all 80 geometries (no training needed).
  - Express the network's cost in stiffness-action equivalents.
  - Scope the Abstract claim ("on five cells, with the fixed-weight network, under consistent tractions").
  - State the 2,400–12,000 ratio in §6.5.
- **Fix type:** New experiment (cheap, per cell, no training), plus text.
- **Related:** I-12, I-19.

#### I-08: The sensitivity metric is not the one an optimiser sees
- **Theme:** T2 · **Severity:** Major · **Consensus:** 5 (R1, R2, R4, R5, R7)
- **Sources:** R1-08, R2-13, R4-03 (part), R5-16 (part: 3% threshold), R7-11 (part). See also R7-03 (item 4) under I-35.
- **Status:** —
- **Problem:** e_s = ‖s̃ − s‖/‖s‖ is normalised by each cell's own eight-vector.
  - The headline example (U1/x, N-z; target energy share 0.13%) has local errors of 3.89–5.8%, which may be negligible in the design update. M1/x (31% share, C up to 11.4%) does support a real effect (R1-08).
  - The Fig. 10(d) "separation" (local sensitivity error independent of participation) is largely what dividing by a local, participation-dependent norm produces (R4-03).
  - The 2-norm over eight components hides the per-variable errors that matter for OC updates near the multiplier threshold (R4-03).
  - The shared-variable gradient ∇_{τg}C = Σ(∂τ_m/∂τ_g)ᵀs_m of §4.3 is defined but never evaluated, even in the lattices where each corner is shared by up to eight cells.
  - The 3% "accuracy reference" is not motivated (R2-13, R5-16).
- **Proposed actions:**
  - For every pair and lattice case, report ‖∇̃C − ∇C‖/‖∇C‖, cosine or angle, the distribution of per-variable relative errors (median, 95th percentile, max) and sign agreement.
  - Report each cell's eight-vector norm as a fraction of the global gradient norm, next to the local error.
  - Justify 3% (e.g. by optimiser tolerance), or call it a visual reference only.
- **Fix type:** Re-analysis of existing data.
- **Related:** I-03, I-14, I-35.

#### I-09: The closest prior work is missing, so the positioning is incomplete
- **Theme:** T5 · **Severity:** Major · **Consensus:** 5 (R2, R3, R4, R5, R7)
- **Sources:** R2-01 (part), R3-04 (part: DD literature), R4-03 (part: TO literature), R5-17, R7-02
- **Status:** Context: the authors have DECIDED to reposition the paper as an own framework with PIML as related work. The requested related-work coverage fits that decision; nothing here conflicts with it.
- **Problem:** Approximating the interior extension on the full port space is the defining idea of the static-condensation reduced basis element method: full ports, reduced bubbles, a posteriori bounds (Huynh, Knezevic & Patera 2013; Eftang & Patera 2013; Ballani et al. 2018). Citing Smetana & Patera (2016) only for port reduction places SCRBE on the wrong "route" (R7-02).

  Also missing:
  - inexact subdomain solvers in BDDC/FETI-DP (Dohrmann 2003; Farhat et al. 2001; Li & Widlund 2007; Klawonn & Rheinbach 2007; Toselli & Widlund 2005);
  - learned FETI-DP coarse spaces (Heinlein, Klawonn et al.);
  - learned multigrid (Greenfeld et al. 2019; Luz et al. 2020);
  - solver-in-the-loop training (Um et al. 2020);
  - geometry-aware neural operators and graph networks (Geo-FNO, GINO, MeshGraphNets, MgNO);
  - learned Green's functions (Boullé & Townsend);
  - learned CG preconditioners, hypernetworks and Deep-Ritz-type losses (R2-01);
  - multiscale TO with coarse-basis correction (Alexandersen & Lazarov 2015) and GMsFEM oversampling (R7-02);
  - approximate reanalysis and inexact solves in TO (Amir, Bendsøe & Sigmund 2009; Amir, Stolpe & Sigmund 2010; Gogu 2015) (R4-03, R7-02);
  - smoothed aggregation (Vaněk, Mandel & Brezina 1996) (R3-04).

  R5-17: the distinction from "learned initial guess plus multigrid" (fixed-budget correction turned into a variational operator, complete transpose, training through the correction, sensitivity analysis) is spread over several paragraphs. Two key comparators are 2026 arXiv preprints whose status should be checked.
- **Proposed actions:**
  - Add a related-work paragraph on component/substructure ROMs with full ports (SCRBE), inexact substructuring, learned solvers and multigrid, and neural operators. State what differs: SCRBE needs offline RB bubble spaces per parameter set and gives certified bounds; NICE conditions on geometry and has no certified bound (links I-20).
  - Add one explicit "difference" sentence after the hybrid-solver citations.
  - Update the preprint references.
  - Authors to verify all bibliographic data suggested by reviewers (R7-02 flags medium confidence on individual citations).
- **Fix type:** Text only.
- **Related:** I-11, I-14, I-15, I-35.

#### I-10: Section 6.11 design-optimisation example (placeholder)
- **Theme:** T10 · **Severity:** Major · **Consensus:** 5 (R1, R2, R4, R5, R7)
- **Sources:** R5-01, R1-§4, R2-§4, R4-§4, R7-§4 (and R7-11 on deferring design-impact claims to §6.11)
- **Status:** **DECIDED.** §6.11 is a placeholder and will be added, together with the Abstract and §8 sentences. The scope choices inside §6.11 are listed as must-have / nice-to-have in §2 of this register.
- **Problem:** Placeholders remain in the Abstract, §6.11 and §8. R5 rates the desk-reject risk as high if the paper is submitted like this. §7.2 defers measuring the complete-derivative term "to subsequent work", which conflicts with an optimisation driven by the surrogate (R5-01; I-03).

  Current plan (placeholder text):
  - compliance minimisation under a volume constraint;
  - the eight corner thicknesses as shared variables;
  - OC or MMA;
  - 8 or 27 cells;
  - final design verified with the exact reference;
  - history compared with an exact-condensation run.

  Reviewers want the example to show that the learned operator reaches the same design quality as exact condensation, with verified gradients, at stated cost, inside the training domain.
- **Proposed actions:** See §2, "Requirements for Section 6.11". Keep it to about 600–900 words, one figure (history plus final design) and one table (verification plus cost) (R5-01). Move the shared-design-variable paragraph from §4.3 into §6.11. Rewrite the §7.2 sentence.
- **Fix type:** New experiment (planned).
- **Related:** I-03, I-08, I-13, I-20, I-30, I-31, I-32, I-41, I-50.

#### I-11: No whole-lattice iterative or domain-decomposition baseline
- **Theme:** T1 · **Severity:** Major · **Consensus:** 4 (R1, R3, R4, R7)
- **Sources:** R1-06 (part 1), R3-04, R4-07 (part 1), R7-07 (part), R7-13 (part)
- **Status:** NEEDS AUTHOR DECISION. The DECIDED baseline covers GPU direct factorisation (cuDSS) versus host PARDISO. This request, a fine-scale iterative solver of the whole lattice, is not covered by that decision.
- **Problem:** The only whole-lattice competitor in Table 6 is a sparse direct solve of 0.9–2.1 M DOFs.
  - R1-06: the correction is itself a two-grid method, so the natural competitor is PCG on the full CutFEM lattice with GMG or AMG built from the same components. The learned route needs about 130–165 PCG iterations × about 33 fine stiffness actions per cell, i.e. several thousand full stiffness actions per design iteration. The retained system (139k of 1.8M DOFs) is only about a 13× reduction.
  - R3-04: AMG-PCG with rigid-body near-nullspace (PETSc GAMG, ML/MueLu, BoomerAMG, AmgX) usually solves this size in tens of seconds with modest memory. If AMG struggles with thin walls, Q2 and the ghost penalty, that is itself a useful result. The setting is also classical iterative substructuring (cells = subdomains, retained set = interface), for which BDDC and FETI-DP are standard.
  - R4-07: matrix-free GPU GMG-CG is the standard tool in large-scale TO (O(10–100) iterations of a few matvecs each).
  - R7-07 and R7-13: NICE is in effect an inexact matrix-free substructuring solver of the fine model, so its competitors are fine-scale iterative solvers and DD, which would not need 60–160 GB. §7.4 itself calls the "total work to attain the prescribed response accuracy" the relevant comparison.
- **Proposed actions:**
  - Add an AMG-PCG baseline (GPU if possible, otherwise CPU on the full node) to Table 6 for the four- and eight-cell lattices, at matched compliance accuracy: setup, solve, iterations, memory.
  - Discuss, and ideally run, BDDC with cells as subdomains.
  - If not run, state why in §7 and position the method relative to inexact-subdomain DD (links I-09).
- **Fix type:** New experiment, plus text.
- **Related:** I-01, I-09, I-12, I-13.

#### I-12: No matched-accuracy (Pareto) comparison against non-learned corrections
- **Theme:** T1 · **Severity:** Major · **Consensus:** 4 (R2, R3, R5, R7)
- **Sources:** R2-04 (part), R3-07 (part: cost-matched), R5-06, R7-10 (parts 1, 2, 5)
- **Status:** NEEDS AUTHOR DECISION (new experiment). R7 also asks for exact GPU factorisation as a Pareto point, which touches the DECIDED removal of cuDSS; decide whether to include it as a reference point.
- **Problem:** Table 3 varies only the smoothing per stage around a single coarse solve, at a fixed budget. It does not show how many cycles, or how much time, a zero or harmonic start needs to match A3 (R2-04, R5-06). §7.4 says application time is "dominated by the 32 stiffness actions of the correction, not by the network", so network cost versus extra cycles is the central trade-off (R2-04).

  R7-10:
  - on M1, going from 8 to 32 steps per stage gives only a 2.6× reduction (0.186% → 0.0727%) at about 4× cost;
  - on H2 at 32/Q1/32 the harmonic start already reaches 0.0006% with no network;
  - multiple two-grid cycles are never shown;
  - the network is trained for one budget.

  Without a Pareto front, a reviewer can argue that the network only replaces a stronger preconditioner (R5-06).
- **Proposed actions:** Produce an accuracy-vs-cost plot (energy excess and sensitivity error against wall time or stiffness-action equivalents, same GPU) on the five detailed cells (minimum) or the 20 heavy-cut geometries. Include:
  - NICE with k = 2, 4, 8, 16, 32 and 1, 2, 4 … cycles;
  - zero, harmonic and elastic starts with k = 8…64 plus Q1(17) and repeated cycles;
  - interior PCG with SA-AMG to matched energy excess.

  Show whether any NICE configuration is non-dominated. Use one panel (e.g. in Fig. 8) plus two sentences, with details in the supplement (R5-06).
- **Fix type:** New experiment.
- **Related:** I-07, I-20, I-23.

#### I-13: Scalability beyond eight cells not shown; the assembled solver is under-specified
- **Theme:** T1 · **Severity:** Major · **Consensus:** 4 (R2, R3, R4, R7)
- **Sources:** R2-§4(e), R3-09, R4-06 (part: ≥27 cells), R4-07 (part 2), R7-07
- **Status:** NEEDS AUTHOR DECISION (new experiment; positioning).
- **Problem:** The main text never names the preconditioner of the A3 lattice solves. S1.1 suggests a K_PP⁻¹ fine action plus a q1r coarse space, which needs a factorisation of the global retained block that grows superlinearly. K_PP ⪰ K̂ is not guaranteed (S1.1) (R3-09). Trends:
  - iterations of 114–165 to 1e-6, growing from 4 to 8 cells;
  - a residual floor that grows tenfold;
  - GPU memory that already forces host streaming at 8 cells ("operator state of four cells on the GPU and stream the remainder") (R4-07);
  - the only 27-cell evidence, predictor D in ST14d/ST17b, has the learned route at 222 s against 94.6 s exact on the same GPU (R3-09).

  R7-07 extrapolates to 1,000 cells: about 17–26 M retained DOFs, 0.25–1.4 TB of operator state, 1–2 h of front end and several hours of PCG per design iteration. NICE is therefore at a different point on the accuracy/cost curve from PIML-type ROMs (fine-scale accurate analysis of tens of cells), and the positioning should say so.
- **Proposed actions:**
  - Describe the lattice preconditioner in the main text.
  - Run a scaling study with A3 for 4/8/27/64 (R7: ≥125) cells: iterations, CG-Lanczos condition estimate, setup, K_PP factorisation scaling, time and memory per design iteration, residual floor.
  - State the number of cells beyond which one GPU is impractical.
  - Discuss the 27-cell D result or remove it (I-22).
  - Reframe the positioning in §1 and §7. If a coarse-level model is the eventual aim, say so.
- **Fix type:** New experiment, plus text.
- **Related:** I-04, I-11, I-22, I-30, I-52.

#### I-14: "Accurate compliance does not imply accurate sensitivity" framed as the principal finding
- **Theme:** T5 · **Severity:** Major · **Consensus:** 4 (R1, R4, R5, R7)
- **Sources:** R1-01 (part: contribution iv), R4-03 (part: literature), R5-03, R7-11 (part)
- **Status:** NEEDS AUTHOR DECISION (headline claim in the Abstract, contribution iv and §8).
- **Problem:** The distinction between energy-norm and local-functional accuracy is the basic premise of goal-oriented error estimation (Becker & Rannacher 2001; Oden & Prudhomme 2001; R1-01, R7-11). It is also documented for approximate reanalysis in TO (R4-03). The contribution is therefore a quantification for learned substructures, not a new principle.

  R5-03: the separation is demonstrated only on predictors that are *not* the method (B, C, S8; Table 4, Fig. 10), while for NICE both errors are small, so it reads better as the design criterion NICE satisfies. R7-11: the flagship case is a cell carrying 0.13% of the energy. It has never been shown that the separation changes an optimised design.

  R5-03 also finds:
  - no single storyline: the Abstract spends about half its length on analysis and about 20 numbers;
  - contributions (i) and (ii) overlap;
  - §8 follows yet another order.
- **Proposed actions:**
  - Reword contribution (iv), the Abstract and §8 as a quantification for learned substructures, with citations.
  - Present the separation as the design criterion that NICE meets.
  - Use one storyline (problem → idea → analysis → evidence) in the Abstract, contributions and §8, with three contributions (R5-03), coordinated with I-15.
  - Optionally demonstrate the design consequence in §6.11 (§2, N1).
- **Fix type:** Text only.
- **Related:** I-08, I-09, I-15, I-55.

#### I-15: Classical and inherited results presented as contributions
- **Theme:** T5 · **Severity:** Major · **Consensus:** 4 (R1, R2, R6, R7)
- **Sources:** R1-01, R2-01 (part), R6-32, R7-01
- **Status:** NEEDS AUTHOR DECISION (contributions list and Abstract wording). Context: the DECIDED repositioning as an own framework is compatible with separating "inherited" from "new".
- **Problem:** For any admissible linear extension N (N_P = I, N R_P = R), NᵀKN is SPSD, has exactly the rigid kernel and satisfies NᵀKN − S = HᵀAH ⪰ 0. This holds for MsFEM, SCRBE and PIML alike. The "complete transpose" is NᵀKN evaluated matrix-free (R7-01). The following are classical (R1-01):
  - Ritz identity (Eq. 9, B.1);
  - A-orthogonal Galerkin projection (Eq. 16);
  - Chebyshev semi-iteration (D.1–D.4);
  - two-grid error operator (Eq. 17);
  - compliance ordering (C.2–C.3);
  - residual-corrected functional (J.6);
  - self-adjoint compliance sensitivity (H.1).

  Only Xu (1992), Adams et al. (2003) and Giles & Pierce (2000) are cited, and the last is an aerodynamic-adjoint tutorial. Guyan (1965) is in the .bib but not cited (R6-32). Yet the Abstract, contribution (i) and §8 list these properties as features of NICE, and contribution (iii) reads as a new error analysis.

  Genuinely new, per R1-01, R2-01 and R7-01:
  - the displacement-linear, matrix-free learned extension on the complete retained trace of cut CutFEM cells;
  - the fixed correction inside the energy form, with its transpose;
  - training through that correction;
  - the participation bound β/(1+ρ) ≤ e_C ≤ β/(1+β) (J.3), the global-to-local bound (J.4) and the O(t) cancellation (J.5);
  - the cut-band retained space;
  - the empirical study.
- **Proposed actions:**
  - Add a "Classical background" paragraph at the start of §4 that labels each relation as classical (cited) or new. Citations: Guyan 1965; Irons 1965; Craig & Bampton 1968; Toselli & Widlund 2005 / Smith, Bjørstad & Gropp 1996; Golub & Varga 1961; Saad 2003; Hackbusch 1985; Trottenberg et al. 2001; Xu & Zikatanov 2002; Bendsøe & Sigmund 2003; Haftka & Gürdal 1992; Becker & Rannacher 2001; Oden & Prudhomme 2001; Fraeijs de Veubeke 1965.
  - Split the contributions into "inherited and retained" and "new" (R7-01), or three contributions (R5-03).
  - Rephrase contribution (iii) as an "application of classical Ritz and goal-oriented arguments to …".
  - Attribute the properties in the Abstract and §8.
  - Cite Guyan (resolves R6-32).
- **Fix type:** Text only.
- **Related:** I-09, I-14.

#### I-16: "80 unseen geometries" contradicts the disclosed selection overlap
- **Theme:** T3 · **Severity:** Major · **Consensus:** 4 (R1, R2, R5, R6)
- **Sources:** R1-15 (part), R2-06 (part), R5-04 (part), R6-07
- **Status:** NEEDS AUTHOR DECISION (Abstract wording; low controversy).
- **Problem:** Twenty of the 80 validation geometries (6 uncut, 14 cut; valmeta `sel=True`) entered checkpoint selection, and all detailed cells U1–L1 belong to that subset (R6-07). "Unseen" appears in the Abstract, the Fig. 5 title and §8 ("every one of 80 unseen geometries"). The effect is negligible (60 held-out mean 0.077% vs 0.074%), but reviewers will read the wording as overclaiming (R5-04).
- **Proposed actions:** Write "80 validation geometries not used in training (20 of them entered checkpoint selection)" and quote the 60-geometry figure next to it. Note that the detailed cells belong to the selection subset.
- **Fix type:** Text only.
- **Related:** I-17, I-26.

#### I-17: Assembled accuracy rests on development cells; worst geometries never assembled
- **Theme:** T3 · **Severity:** Major · **Consensus:** 4 (R1, R2, R4, R5)
- **Sources:** R1-15 (part: lattice provenance), R2-06 (part), R4-04, R5-04 (part)
- **Status:** NEEDS AUTHOR DECISION (new experiment; the Abstract claim "below 0.7% in all fourteen configurations").
- **Problem:** The 14 pair configurations are 7 development target cells in two orientations, each with an exact neighbour. The same cells underlie the fixed-weight study, the spectral analysis and the choice of 8/Q1(17)/8 and a = b/30 (R2-06). A3's largest geometry-mean excess (0.65%) is about 5× M1's (0.13%), and sensitivity error can exceed energy error by 2× or more (H2 with B: 35% energy, 75% sensitivity). Assembled sensitivity errors on the worst validation geometries are therefore probably above 0.7%, and they were never measured (R4-04). The weight-selection score also includes a sensitivity term on the overlapping list. The provenance of the lattice cells is not stated in the main text; they may already be fresh cells (R5-04, R1-15).
- **Proposed actions:** Run the pair protocol (six face loads plus cut loads) on held-out cells. Options range from small to large:
  - R5: 4–6 held-out cells, one per stratum and x/y;
  - R4: all 60 held-out cells, or the 10–15 worst per stratum;
  - R2: a sealed new test set of 40 geometries and 20 random pairs, evaluated once for A3, B+W, C+W and the baselines.

  Report the distribution, with local and global-gradient versions (I-08). Rewrite the Abstract claim as a distributional statement. State the provenance of the lattice cells.
- **Fix type:** New experiment (the pipeline exists), plus text.
- **Related:** I-08, I-16, I-30.

#### I-18: Worst-direction / operator-norm error not reported
- **Theme:** T3 · **Severity:** Major · **Consensus:** 4 (R1, R2, R3, R6)
- **Sources:** R1-05, R2-08, R3-06 (part: all-direction bound), R6-20
- **Status:** NEEDS AUTHOR DECISION (Abstract wording "at most 0.65% for any geometry").
- **Problem:** All population statistics average over sampled directions within a geometry ("the maximum is not a worst individual direction", ST01). An assembled solve, and an optimiser, select their own directions, and J.7 concedes that sampled means give no bound on unsampled ones. The paper already defines ε_* and has the machinery: the G.3 bank/adversarial search, Lanczos/LOBPCG on the pencil (ZᵀŜZ, S_*), and the I.1 lower bound. Yet no worst-found directional error, λ_max(S⁻¹Ŝ) − 1, or lower bound on ε_* is reported for A3 (R1-05, R2-08). R3-06 adds that a measured ‖T‖_A would give an all-direction bound ε_*(F) ≤ ‖T‖²_A ε_*(Ê).

  Other gaps:
  - direction counts per geometry are not retained (Supp. R2);
  - there are no confidence intervals or seeds;
  - "at most 0.65% for any geometry" reads as a worst case, and the value is 0.651% (R6-20).
- **Proposed actions:**
  - For A3 (and B+W, C+W) on all 80 geometries (at least the 20 heavy-cut ones) and on the lattice cells, report the 99th percentile and maximum directional errors per class. Report Lanczos estimates of λ_max(S⁻¹Ŝ) − 1 on the non-rigid complement (50–100 iterations, preconditioned by the available exact Schur action).
  - Record the direction counts.
  - Give bootstrap CIs.
  - Reword the Abstract to "largest geometry-mean directional error 0.65%".
- **Fix type:** Re-analysis of existing checkpoints and teachers.
- **Related:** I-19, I-23.

#### I-19: The benefit of training through the correction is confounded with extra data and steps
- **Theme:** T6 · **Severity:** Major · **Consensus:** 4 (R1, R2, R5, R7)
- **Sources:** R1-07, R2-01 (part: framing), R2-02, R5-18, R7-08
- **Status:** NEEDS AUTHOR DECISION (seed runs are new experiments; contribution (ii) and the Abstract framing may change).
- **Problem:** A3 is B continued for 15,000 steps on 591 geometries (vs 305) with the correction in the loop. B+W is B's weights plus the correction. The A3/B+W factors (1.3 in the population mean; 1.35–2.2 in worst sensitivities) therefore mix three effects. The clean control is C+W: C's existing checkpoint (591 geometries, same continuation, no correction) evaluated with A3's correction. It needs no training, yet it is not reported (R1-07, R2-02, R7-08).

  Only one seed exists, and the factors are small enough to lie within seed variability (R2-02). R5-18 and R2-01: B+W ("NICE-post") already meets every assembly criterion (0.0965% population mean), and the correction delivers most of the accuracy (6.89% → 0.0965%). Yet the Abstract attributes all headline numbers to the trained-through variant. R6-23 (under I-58) shows that B+W beats A3 by 1.6–2.1× on M2. The training-population count behind the confound is itself uncertain (I-43).
- **Proposed actions:**
  - Evaluate C+W on the 80 geometries and the 14 configurations. Report paired per-geometry differences (A3 − C+W) with bootstrap CIs.
  - Train A3 and C with at least three seeds (new).
  - If C+W ≈ A3, rephrase contribution (ii), §6.3, §6.6, §7.3 and §8.
  - State in the Abstract that the correction alone brings an existing network within tolerance. Present training through the correction as a refinement for the hardest cells.
  - Consider reframing contribution (ii) as "a learned linear warm start for a fixed variational correction" (R2-01).
- **Fix type:** Re-analysis (C+W, evaluation only); new experiment (seeds).
- **Related:** I-07, I-34, I-43, I-58.

#### I-20: No runtime error control; the one-sided bias invites exploitation by the optimiser
- **Theme:** T4 · **Severity:** Major · **Consensus:** 4 (R2, R3, R4, R7)
- **Sources:** R2-07 (part: indicator), R3-11 (part: runtime check of b), R4-09, R7-10 (part 3)
- **Status:** NEEDS AUTHOR DECISION (new implementation and experiment).
- **Problem:** Ĉ ≤ C always (Eq. 12), and the error is largest for heavily cut, thin or out-of-distribution cells. A compliance minimiser is therefore rewarded for moving towards designs where the surrogate is most over-stiff, which is classic surrogate exploitation (R4-09). Optimisation creates thousands of unvalidated geometries, while the Chebyshev containment is verified only offline (80 validation plus 5 deployed cells) (R4-09, R3-11). Appendix I gives only lower bounds, so there is no computable upper bound. The correction budget therefore cannot be chosen to meet a tolerance without reference data, which weakens the "controllable" claim; SCRBE offers such estimators (R7-10). The Eq. (10) residual/energy identity is a label-free indicator that could flag OOD inputs, but it is not used (R2-07).
- **Proposed actions:**
  - Add an a-posteriori per-cell indicator to the deployed route (e.g. L_z/(ûᵀKû) from App. I with z from one extra coarse or Krylov step, or a few interior CG iterations estimating ‖r_I‖_{A⁻¹}) with a verified effectivity. Otherwise, state that the budget cannot be chosen without reference data.
  - Define a policy for flagged cells: raise the budget or fall back to exact condensation.
  - Add a runtime Chebyshev-endpoint check.
  - In §6.11, report surrogate error at the start and at the optimum.
- **Fix type:** New experiment (implementation), plus text.
- **Related:** I-05, I-10, I-23, I-38.

#### I-21: Too many labels; overloaded notation; inconsistent naming and colour coding
- **Theme:** T8 · **Severity:** Major · **Consensus:** 4 (R1, R2, R5, R6)
- **Sources:** R1-16, R2-09, R5-08, R6-11
- **Status:** Context: the method name NICE is DECIDED. The choice between the reviewers' naming schemes is the authors' (not a disagreement).
- **Problem:**
  - **Predictor labels.** There are eight (P0, B, C, S8, A2b, B+W, A3, D), plus internal run IDs in the appendices (v2L1, A0_ctrl, B2grid, …). "W" is never tied to 𝒲. S8 has a partial, unspecified training protocol and is "not interpreted", yet it carries Table 4 and Fig. 10 results. P0 appears only in ST01/S01, and D only in the supplement.
  - **Method name.** NICE is not used in the Results, which say A3. Table 3's "B (learned)" is actually B+W.
  - **Cell labels.** Twelve labels (U1 … L1, G1–G4) are defined only in Supp. R1.
  - **Colour and markers.** An orange square is A3 in Figs. 5 and 9 but S8 in Fig. 10 (R5-08).
  - **Notation overload (R1-16).** The same letter serves several meanings:
    - C: compliance, predictor and coefficient map;
    - A: K_II, predictors A2b/A3, 𝔸;
    - D: diag(A), K,c, predictor D;
    - also H, P, ρ, T, M_I, I, γ, b and n.
  - **Support-class terms (R6-11).** `support_k` is called both "spring-supported faces" and "stiffness-scaled supports", and "spring-support" also names `support`.
- **Proposed actions:** Adopt one scheme, using NICE throughout the Results. Proposals:
  - R5: NICE / NICE-post / Smoothing-trained / Uncorrected / Base network;
  - R2: factorial labels (N305, N591, N591⊕𝒲 …) on a 2×2(×2) grid.

  In addition:
  - drop S8, P0 and D from the main text (or specify S8's protocol);
  - add a cell key next to Fig. 1 or Table 2;
  - use one colour map;
  - add a nomenclature table and reserve A, C, D, H, P for mechanics;
  - remove internal IDs;
  - use one term for `support_k`, distinct from `support`.
- **Fix type:** Text only (plus re-plot).
- **Related:** I-37, I-53.

#### I-22: Supplementary Note S4 / Table ST14 still contains same-GPU exact-factor timings
- **Theme:** T1 · **Severity:** Major · **Consensus:** 3 (R3, R5, R7)
- **Sources:** R3-01, R3-09 (part: 27-cell D result), R5-19, R7-05 (part)
- **Status:** **NEEDS AUTHOR DECISION (keep / remove / re-frame).** This is the open issue that follows from the DECIDED removal of the cuDSS comparison from the main text: the supplement still holds same-GPU exact-factor data, and two reviewers found them.
- **Problem:** Table ST14 (historical deployment study with the earlier uncorrected predictor D) gives GPU fp64 exact interior-factor timings on the same RTX 5090 and the same cells G1–G4:
  - factorisation 0.64–8.6 s;
  - application 7.9–73 ms (1 vector), 13–100 ms (16), 35–316 ms (64).

  Table 5 gives A3 as 25–99, 205–1,052 and 388–1,828 ms. On the same device, the exact factor therefore applies about 1.4–3× faster for one vector and 6–12× faster for 16/64 vectors. The learned route wins only in preparation, by about 4–8× (0.16–1.09 s) (R3-01, R7-05).

  Consequences:
  - Break-even is about 90–320 applications per cell, fewer than the 183–234 per design iteration of §6.9 (R7-05).
  - R3-01's G1 estimate is about 9 s exact vs about 40 s learned per design iteration.
  - S4 itself says "the exact factor is faster for batches of 64 directions".
  - ST14d/ST17b: the 27-cell D array takes 222 s learned vs 94.6 s exact on the same GPU.
  - Exact-factor timings do not depend on the predictor (R3-01).
  - §6.10 points to S4 only as "the historical deployment study of an earlier uncorrected predictor".

  Reviewers call this a contradiction of "cheaper for any number of queries". The ST14 figures come from an earlier software stack and benchmark campaign, so the cross-campaign comparison is approximate (R3-01 medium confidence on size; R7-05 "unlikely to reverse"). R5-19 independently says the D study no longer supports any main-text claim and adds a label and about 1,300 supplementary words.
- **Options for the authors:**
  - (a) Remove S4, ST08–ST10, ST14, ST17 and Figs. S04–S05 from the supplement to a data/code repository, and delete the §6.1/§6.10 mentions of D (R5-19). Risk: reviewers who saw this version may regard the removal as concealment.
  - (b) Keep them and re-frame. Acknowledge in §6.10 that on the same GPU an exact factor applies faster for batched queries, and that the learned route's advantages are preparation time and memory (fp64 factor 4.7–5.7 GiB for G1/G3/G4 vs 0.7–1.4 GiB learned state; ST08a). Reword the cost claims accordingly (I-01).
  - (c) Keep them under a separate "Archive: earlier uncorrected predictor" heading, with an explicit note that the timings come from an earlier stack and are not comparable with Table 5, and with the exact-factor columns removed or caveated.
- **Fix type:** Cut or move (a, c) or text only (b).
- **Related:** I-01, I-13, I-51.

#### I-23: The two-grid bound proves only non-expansion; the "controllable" claim is empirical
- **Theme:** T4 · **Severity:** Major · **Consensus:** 3 (R1, R3, R7)
- **Sources:** R1-10, R3-06 (part), R7-10 (parts 1–2)
- **Status:** NEEDS AUTHOR DECISION (Abstract and contribution (i) wording: "its error … can be reduced through both the network and the correction budget").
- **Problem:** ρ_k is the maximum of |p_k| over the whole spectrum of D⁻¹A. Since λ₁ ≈ 4–6×10⁻⁴ ≪ a ≈ b/30 ≈ 0.17 (ST02b), ρ_k ≈ 1, and the "ρ_k⁴ energy estimate" of App. D gives only non-expansion. No approximation property of the Q1 space for walls about one element thick is established; M1 keeps 0.186% after 8/Q1/8. The zero start leaves 28–1,097% after 8/Q1/8, which suggests weak worst-case contraction (R3-06). The ordering S ⪯ Ŝ_tg ⪯ Ŝ₀ is correct. The empirical evidence is thin: M1 at k = 2/4/8 (1.02/0.42/0.186%) and five cells at k = 32; multiple cycles are never shown (R7-10).
- **Proposed actions:**
  - State that only monotone non-increase is guaranteed, and reword App. D.
  - Compute ‖T‖_A and ‖Tᵐ‖_A per geometry (Lanczos on the A-symmetric error propagator, a few dozen cycle applications) on the detailed cells, ideally on all 80. Compare them with the directional reduction.
  - Show multi-cycle convergence.
  - Cite Xu & Zikatanov (2002) and Falgout, Vassilevski & Zikatanov (2005).
  - Qualify "controllable" and "can be reduced" in the Abstract and contribution (i) as empirical.
- **Fix type:** Re-analysis (cheap new computation on existing operators), plus text.
- **Related:** I-12, I-18, I-20.

#### I-24: Verification of the CutFEM reference is too thin for thin, heavily cut walls
- **Theme:** T3 · **Severity:** Major · **Consensus:** 3 (R1, R4, R6)
- **Sources:** R1-04, R1-17(c), R4-05 (part: convergence at τ bounds), R6-21, R6-22
- **Status:** NEEDS AUTHOR DECISION (small new experiment).
- **Problem:** The mesh check has several limitations (R1-04):
  - Only four cells (U1, M1, M2, H1) are checked. The finest level is n = 48, and convergence is non-monotone: for M1, n = 40 is further from n = 48 than n = 32 is (Fig. S06a). So "difference from finest" is not an error estimate, and no observed rate is given.
  - The τ_c of the detailed cells are not reported. Near τ ≈ 0.176 the walls are about 1–1.5 background elements thick (R1: about 0.05, i.e. 1.5 h; R4-05: about 0.03 box lengths, i.e. about 1 h). This is exactly where Q2 CutFEM is least reliable in bending, and where optimisation pushes lightly loaded regions (R4-05).
  - There is no independent (body-fitted) reference.
  - γ = 10⁻⁴ is orders below usual values, and conditioning robustness over cut positions is not shown.
  - Ghost faces never cross cell boundaries, so the reference is a *modular* CutFEM, a fact stated only in A.1/J.1.
  - The Fig. S06 axis reads "nodes per cell edge" although n counts elements per axis, and Fig. S06(a) stops at n = 40 while the text cites n = 48 (R1-17c).

  Text corrections:
  - penalty energy "about 4×10⁻⁴" → up to about 6×10⁻⁴ (1.2e-4 to 5.6e-4) (R6-21);
  - H1's "in-plane load" is the y-directed load, normal to the loaded face (R6-22).
- **Proposed actions:**
  - Add the thinnest-wall and most heavily cut validation cells, and cells at both τ bounds (R4-05).
  - Add n = 64, or Richardson extrapolation over ≥3 levels with an observed order.
  - Add one body-fitted P2 tetrahedral comparison.
  - Report cond(D⁻¹A) or λ_min over cut positions for γ ∈ [10⁻⁵, 10⁻¹].
  - Move the modular-reference statement into §2 and justify it.
  - Give τ_c for the detailed cells.
  - Fix Fig. S06 and apply the R6-21/22 corrections.
- **Fix type:** New experiment (small), plus text.
- **Related:** I-32, I-42.

#### I-25: Contradictory finite-difference step statement; PSD of the numerical K,c unchecked
- **Theme:** T2 · **Severity:** Major (by the R6 "Error" mapping; substantive impact small) · **Consensus:** 3 (R1, R4, R6)
- **Sources:** R1-17(e), R4-11, R6-01, R6-31
- **Status:** —
- **Problem:**
  - R6-01 (Error): §6.2 says the FD stiffness derivative "changes by less than 10⁻⁷ when the step is varied between 10⁻⁶τ_c and 10⁻³τ_c". The records give up to 2.56e-7 (M2) and 2.19e-7 (H1).
  - R4-11: App. J.5 says "No derivative eigenvalues or step-refinement study are saved", which contradicts §6.2 and Fig. S06(c). Whether the numerical K,c with adaptive subcell refinement is PSD is unchecked, yet J.5–J.7 and the sign guarantee of s̃_c used by OC rely on it.
  - R1-17(e): "exact integration gives K,c ⪰ 0" (§4.3) should refer to the continuous integral, since fixed quadrature branches are not guaranteed.
  - R6-31: App. H cites "Eq. (J.6) in Appendix J.5", which is confusing because §J.6 is a different topic.
- **Proposed actions:**
  - Write "less than 3×10⁻⁷" (or give per-cell values).
  - Update J.5.
  - Report the smallest Lanczos eigenvalue of the numerical K,c for a few cells, or explain why nesting is preserved.
  - Soften §4.3.
  - Write "Eq. (J.6) (Section J.5)".
- **Fix type:** Text only (plus small re-analysis).
- **Related:** I-03.

#### I-26: View dependence of A3 not evaluated; checkpoint selection described inconsistently
- **Theme:** T6 · **Severity:** Minor · **Consensus:** 3 (R1, R2, R6)
- **Sources:** R1-14 (part: view 17), R2-12, R6-10
- **Status:** —
- **Problem:** Selection is described three ways:
  - App. G.3: "selected at step 15,000";
  - ST12: evaluated every 7,500 updates, lowest finite score; A2b with EMA weights at 15,000;
  - §6.1: "two checkpoints per predictor", when in fact C, S8 and A3 had two and A2b had one (R6-10).

  Selection used views 0 and 17, but A3 is reported only in view 0. The network is not equivariant: B's mean rises from 6.62% to 7.49% in view 17. ST01c covers only P0 and B, on an earlier 15–20-geometry evaluation (R1-14, R2-12).
- **Proposed actions:** Harmonise the text. Report A3 and B+W on all 80 geometries in 4–8 random views of the 48 with the view-to-view spread (R2-12), or at least in view 17 (R1-14).
- **Fix type:** Re-analysis, plus text.
- **Related:** I-05, I-16.

#### I-27: Benchmark reporting incomplete; table layout
- **Theme:** T1 · **Severity:** Minor · **Consensus:** 3 (R3, R5, R6)
- **Sources:** R3-14, R5-20 (part: table width), R6-29, R6-30
- **Status:** —
- **Problem:** Not reported:
  - CPU model, sockets, memory bandwidth, NUMA;
  - MKL, CUDA and PyTorch versions;
  - GPU clocks and power;
  - repetitions and spread (host timings are single runs apart from a "within 10%" remark, itself partly wrong; I-06);
  - whether the three PCG loads are solved as a block;
  - warm-up/JIT handling;
  - any energy or cost normalisation for a 575 W GPU against 16 cores (R3-14).

  Layout: Table 5's column order switches (host/GPU vs learned/host) (R6-29). ST20c's phase columns do not add up to the totals (38.36 vs 38.65 s; 109.70 vs 110.12 s) (R6-30). Tables 5 and 6 (8 and 7 columns) will not fit a single-column elsarticle page (R5-20).
- **Proposed actions:**
  - Add a benchmark-environment table (supplement).
  - Report the median and range over ≥3 runs (piggyback on I-06/I-36 re-runs).
  - Consider energy-to-solution.
  - Use one column order.
  - Note the unlisted overhead in the ST20c caption.
  - Simplify Table 5 (ratios or stacked units). Table 6 moves to the supplement under the length plan.
- **Fix type:** Text only (re-runs optional, shared with I-06/I-36).
- **Related:** I-06, I-36.

---

### Part B: Remaining Major issues (consensus ≤2)

#### I-28: Memory units (GiB vs GB) and what the memory figures contain
- **Theme:** T9 · **Severity:** Major (affects the Abstract's "stores 4 to 14 times less") · **Consensus:** 2 (R3, R6)
- **Sources:** R6-04, R3-10
- **Status:** **NOT DECIDED. MUST BE FIXED.** Flagged NEEDS AUTHOR DECISION only because the Abstract number changes, and because the convention for what counts as operator memory is the authors' choice.
- **Problem:** R6-04 finds three different units in Tables 5–6:
  - the learned state is GiB (0.736328125 = 754 MiB/1024);
  - the host factor is factor_kB/10⁶, which is GB only if PARDISO's kB means 1000 B;
  - Supplementary Note S7 treats PARDISO kB as 1024 B and reports GiB (Table 6: 32.1), and the explicit-S column is GiB (4.52 = 24,636²·8/2³⁰).

  In consistent GiB the host factor is 5.30 / 1.05 / 11.76 / 17.67, and the ratio becomes 4.2–12.9 ("4 to 13"). R3-10 adds that it is unclear whether the learned 0.25–1.37 GB includes K, which applying FᵀKF needs. ST08a lists K (1.39–2.71 GiB) separately, and it is larger than the learned state. It is also unclear whether the host figure includes K_IP and K_PP. A Cholesky factor would roughly halve the host memory (I-36).
- **Proposed actions:**
  - Use GiB (as S7 does), labelled explicitly, in Tables 5 and 6.
  - Recompute the ratios in §6.10, the Abstract and §8.
  - State exactly what each memory figure contains, and report totals including all matrices needed for one application.
  - Recompute again if the Cholesky rerun (I-36) is done.
- **Fix type:** Text only (re-tabulation).
- **Related:** I-01, I-29, I-36.

#### I-29: Ratios in §6.10 computed from rounded Table 5 values
- **Theme:** T9 · **Severity:** Major (headline numbers), DECIDED · **Consensus:** 2 (R1, R6)
- **Sources:** R6-03, R1-17(b)
- **Status:** **DECIDED.** Ratios are computed from the displayed (rounded) Table 5 values by author decision. The reviewers' arguments are recorded for the response letter.
- **Problem:** Recomputed from the unrounded evidence:

  | Quantity | Unrounded | Current text |
  |---|---|---|
  | Condensation speed-up | 8.53–28.7 | 8.8 to 29 |
  | Memory ratio | 4.37–13.57 | 4.4 to 13.5 |
  | Single-vector application | 7.26–22.97 | 7.4 to 23 |
  | Learned ready time | 3.12–7.63 s | |
  | Host ready time | 9.81–61.4 s | |
  | Ready-time ratio | 3.15–8.05 | |

  The Abstract and §8 say "9 to 29", while §6.10 says "8.8 to 29" (R1-17b). 8.53 rounding to 9 is borderline (R6-03).
- **Proposed actions (compatible with the decision):**
  - State in the Table 5 caption or §6.10 that ratios are formed from the displayed values.
  - Keep the Abstract and §8 consistent with §6.10 (e.g. "about 9 to 29").
  - The memory ratio changes anyway under I-28.
- **Fix type:** Text only.
- **Related:** I-28.

#### I-30: The lattice examples are small, favourable and not shown
- **Theme:** T3 · **Severity:** Major · **Consensus:** 2 (R4, R6) (+ R2-§4(e) via §6.11)
- **Sources:** R4-06, R6-19
- **Status:** NEEDS AUTHOR DECISION (new experiment; may be merged with §6.11 and I-13).
- **Problem:** Only two eight-cell lattices are shown. Both are clamped on one face and loaded by uniform consistent traction on the opposite face, which gives near-uniform participation.
  - τ spans 0.25–0.56, well inside the range, and no sliver like H2 (2.7% retained) appears.
  - There is no figure of the lattices, their τ fields or their cuts.
  - Loads act on open wall sections, whereas practice uses skins or plates.
  - No bending-dominated or localised-load case is tested with every cell learned, although the paper's own theory predicts the largest local errors there (R4-06).
  - R6-19: "0.25 to 0.56" cannot be checked. The 2×2×2 range is 0.2516–0.5350, and the 3×3×1 layout is not archived.
- **Proposed actions:**
  - Add a figure of both lattices (geometry, τ, cut cells, supports and loads).
  - Add at least one bending-dominated lattice of ≥27 cells with a localised load and support, cut cells from all strata including a sliver, and τ near both bounds with steep gradients. This can be the §6.11 example.
  - Report the per-cell participation distribution and the global-gradient error (I-08).
  - State whether solid non-design regions are supported.
  - Archive the 3×3×1 layout or give per-lattice ranges.
- **Fix type:** New experiment (plus figure).
- **Related:** I-10, I-13, I-17, I-59.

#### I-31: Non-smoothness of Ĉ(τ) across the discrete switches an optimiser crosses
- **Theme:** T2 · **Severity:** Major · **Consensus:** 2 (R1, R4)
- **Sources:** R1-02 (part), R4-02
- **Status:** NEEDS AUTHOR DECISION (inexpensive new experiment).
- **Problem:** Section 4.3 assumes fixed active, retained and ghost sets. Along an optimisation, the following switch discretely:
  - element activation;
  - ghost-face membership (γg_h does not scale with material fraction, so stiffness jumps);
  - the retained cut-band nodes (and hence P);
  - the binary network features (weak support at <0.01× median; retained and cut-band membership);
  - the adaptive subcell branches;
  - the Cholesky shift picked from {0, 10⁻¹², …, 10⁻⁴};
  - the finite power-iteration estimate of b.

  Ĉ is therefore only piecewise smooth, and it may have jumps that C does not. The ghost share is below 0.05% under consistent tractions but 49–81% under nodal forces. The jump size relative to the design change per iteration is unknown. MMA/OC tolerate small non-smoothness but may stall near convergence (R4-02, R1-02).
- **Proposed actions:**
  - Plot 1-D sweeps of C, Ĉ, s̃_c and Ĉ,c over τ_c from 0.18 to 0.70 for 2–3 cells, including a cut cell, with steps fine enough to cross the switches.
  - Report jump magnitudes against per-iteration objective changes.
  - List all switching quantities.
  - Discuss remedies: a smooth weak-support feature; a volume-fraction-scaled or fixed-set ghost penalty; a fixed seed and iteration count for the spectral estimate.
- **Fix type:** New experiment (inexpensive).
- **Related:** I-03, I-10.

#### I-32: No map from τ to wall thickness and relative density; admissible design domain unstated
- **Theme:** T7 · **Severity:** Major · **Consensus:** 2 (R1, R4)
- **Sources:** R4-05, R1-04 (part: wall-thickness estimate)
- **Status:** —
- **Problem:** τ are band parameters, "not pointwise physical wall thicknesses" (§2.1), yet the paper speaks of "thickness sensitivity" and "corner thicknesses". No map to physical wall thickness, relative density or homogenised stiffness is given.
  - Reviewer estimates put the thinnest admissible walls at about 1–1.5 background elements (R4: |∇φ| = 2π√3 ≈ 10.9, band about 0.03 box lengths at τ ≈ 0.18; R1: about 0.05).
  - The generator limits within-cell span and gradient to 0.47, whereas an optimiser with bounds [0.18, 0.70] can create spans up to 0.52. Neighbouring cells also set in-cell gradients, so designs leave the training domain unless constrained.
  - The design resolution (one trilinear C⁰ field per cell) should be stated as a modelling choice.
- **Proposed actions:**
  - Add a table or figure relating τ to minimum and mean wall thickness (in h and in cell size) and to relative density, for uniform τ with and without cuts.
  - Check reference convergence at the τ bounds (I-24).
  - State the bounds and span/gradient constraints an optimiser must impose, and how §6.11 enforces them.
  - Use "band parameter" consistently, or define "corner thickness" once.
- **Fix type:** Re-analysis (geometry post-processing), plus text.
- **Related:** I-05, I-10, I-24.

#### I-33: Dependence on exact data; data-free variant not discussed
- **Theme:** T6 · **Severity:** Major · **Consensus:** 2 (R2, R7)
- **Sources:** R2-17, R7-06 (part: data-free)
- **Status:** **DECIDED (in progress).** A data-free training pilot (prescribed directions only, no sensitivity labels) is running, and its outcome may address this criticism. Whether and how to report it is decided after the outcome.
- **Problem:** Because Ŝ ⪰ S, log(qᵀŜq) − log(qᵀSq) has the same θ-gradient as log(qᵀŜq). The energy term is therefore label-free apart from direction weighting, i.e. a Ritz principle (R2-17, R7-06). Exact solves are really needed for:
  - the mechanically defined directions: force, force_c, face, face_c, support, support_k and glued need Neumann/pinned solves, and adv needs S⁻¹. Only macro and grf are solve-free, and the solve-derived classes make up 77.5% of sampled directions;
  - the eight-corner sensitivity labels.

  The paper does not spell this out, nor contrast it with data-free PIML (Huang et al. 2024). Reliance on exact data limits extension to larger cells, finer n, other families or nonlinear materials (R7-06).
- **Proposed actions:**
  - Now: state in §3.3 which labels need exact solves, and add a paragraph contrasting NICE with data-free PIML.
  - After the pilot: report it as an ablation (population energy excess, 14-configuration sensitivity maxima, lattice sensitivities), with proxy normalisation (e.g. qᵀK_PPq or the corrected energy) and only prescribed/random directions. With w_s = 0, the pilot also informs I-34.
- **Fix type:** New experiment (running), plus text.
- **Related:** I-02, I-34.

#### I-34: Sensitivity term of the loss and the architectural components not ablated
- **Theme:** T6 · **Severity:** Major · **Consensus:** 2 (R2, R7)
- **Sources:** R2-03, R7-09
- **Status:** NEEDS AUTHOR DECISION (new training runs).
- **Problem:** Every arm uses w_s = 1, so there is no evidence that the sensitivity term, which the Abstract and contribution (ii) list, has any effect. The reviewers' arguments differ, but both conclude that an ablation is needed:
  - R2: §6.4 finds the quadratic term dominant, so energy training alone might already control sensitivity.
  - R7: after correction (energy ≈ 0.1%), the linear term dominates the sensitivity error (J.3), and the energy objective does not control it. That is exactly where the term could matter.

  The architecture is not ablated either: 603k parameters, 24 local interaction layers with 4 heads, 12 coarse convolutions and a separate weak-region block. Since the correction does most of the work, a much simpler linear-in-q or geometry-agnostic network might suffice (R2-03).
- **Proposed actions:**
  - Minimum: a w_s = 0 continuation with A3's protocol (about 2.7 h), compared on population energy, 14-configuration sensitivity maxima and lattice sensitivities (R7-09). Drop or downgrade the claim if the effect is negligible.
  - R2's small set: no latent hierarchy; no weak-region layers; non-conditioned fixed coefficients; reduced width. Report parameter counts and application cost.
  - The data-free pilot (I-33) covers part of this.
- **Fix type:** New experiment.
- **Related:** I-19, I-33.

#### I-35: §6.8 is not a fair proxy for PIML-type substructures
- **Theme:** T5 · **Severity:** Major · **Consensus:** 2 (R4, R7)
- **Sources:** R7-03, R7-04, R4-15
- **Status:** **NEEDS AUTHOR DECISION (reviewer disagreement + Introduction claim).**
  - R5 (length plan, item 38; R5-03 storyline) treats §6.8 and Fig. 11 as the positioning evidence and keeps a trimmed version (about 270 words).
  - R7 wants it moved to the supplement and relabelled, or made fair with new experiments.
  - Context: the DECIDED repositioning (own framework, PIML as related work) reduces the need for §6.8 to characterise PIML at all.
- **Problem:** The Introduction says: "restricting every box face to corner-linear displacements gives compliance errors of 78–85% even with exact cell operators". It presents this as characterising PIML-type substructures. R7-03 lists five biases:
  1. Substructure scale is frozen at one TPMS cell; PIML controls boundary error through partition refinement and super-node count.
  2. There is no oversampling or overlap (PIML-OFEM, Guo et al. 2026b; MsFEM oversampling).
  3. The loads are fine-scale face tractions projected onto a degree-r space (Eq. I.3), which is adversarial to any reduced boundary.
  4. The sensitivity metric inflates errors (200–750% in ST07a) on targets carrying about 0.1–1% of the energy.
  5. Cost is unmatched: r = 1 on U1 has 24 coordinates vs 28,206; at r = 8 (about 14k coordinates on H1), errors are already 0.49–0.74%.

  R7 asks the authors to reconcile the 78–85% with PIML's reported percent-level 3D accuracy. R4-15: reduced-boundary methods trade accuracy for coarse models of tens to hundreds of DOFs per cell, while NICE keeps 17k–26k retained DOFs (about 13× reduction, ST20a), so accuracy must be shown against cost.

  R7-04 on describing the PIML papers:
  - verify the "corner-based linear interpolation" of the 3D examples (substructure size, super-nodes);
  - App. I calls Huang et al. (2023) "physics-informed", but it is data-driven; the data-free variant is 2024;
  - "independent of how accurately the interior is learned" should mention PIML's levers (partition refinement, Bézier enrichment in Guo 2026a, oversampling in Guo 2026b);
  - PIML's boundary description is a deliberate trade for 10⁴–10⁶ substructures.
- **Options:**
  - (a) Relabel as "effect of box-face polynomial restriction on single-cell substructures", move it to the supplement, and remove the PIML attribution and the 78–85% figure from the Introduction (R7 minimum).
  - (b) Keep a trimmed version in the main text (R5), but drop the PIML attribution and add a cost column (retained DOFs, solve time) (R4-15).
  - (c) Make it fair (R7 preferred): substructure refinement (1, 2³, 4³ per cell; several super-nodes per face); an oversampled or overlapping basis; coarse-representable loads; the global-gradient metric; accuracy vs coordinates and time on ≥27 cells.

  In all options, fix the "physics-informed" label and add the sentence on PIML's error levers.
- **Fix type:** Cut or move (a) / text (b) / new experiment (c).
- **Related:** I-08, I-09, I-14, I-37.

#### I-36: The host sparse-direct baseline is configured unfavourably
- **Theme:** T1 · **Severity:** Major · **Consensus:** 1 (R3)
- **Sources:** R3-02
- **Status:** NEEDS AUTHOR DECISION (benchmark re-runs). Because host PARDISO is the DECIDED baseline, reviewers will expect it to be configured well; this issue strengthens the decision rather than contradicting it.
- **Problem:** R3-02 raises five points:
  1. **LU on an SPD matrix.** Table 5 uses PARDISO LU on SPD A = K_II. Table 6 shows that LU costs about 2× the memory (57–64 vs 29–32 GB) and about 2× the factorisation time of Cholesky. The "4 to 14×" memory and the condensation ratios are therefore inflated by about 2, and single-vector solves read both factors.
  2. **Low implied bandwidth.** It is implausibly low: G1 reads 5.56 GB in 826 ms, about 7 GB/s, and ST20b solves 6 RHS with a 32 GB factor in 20.6 s. This is consistent with a sequential solve phase (iparm(25) default), so "limited by reading the factor from memory" may misattribute a configuration effect.
  3. **Explicit S column by column.** S is formed by one solve per retained coordinate (4–59 min), whereas PARDISO's Schur-complement feature (iparm(36)), MUMPS, or blocked multi-RHS solves are standard.
  4. **Cores.** Only 16 of 128 logical cores are used.
  5. **Settings.** No iparm settings or MKL version are reported.
- **Proposed actions:**
  - Use Cholesky/LDLᵀ (mtype 2/−2) in Table 5.
  - Enable the parallel solve and report the iparm settings.
  - Form S with the Schur feature or blocked RHS.
  - Report the CPU, bandwidth and threads; run on the full node or justify 16 cores.
  - Recompute all ratios (I-01, I-28, I-29).
- **Fix type:** New experiment (benchmark re-runs), plus text.
- **Related:** I-01, I-06, I-27, I-28.

#### I-37: Length far above the CMAME norm; the Discussion repeats the Results
- **Theme:** T8 · **Severity:** Major · **Consensus:** 1 (R5)
- **Sources:** R5-02, R5-15
- **Status:** — (approval of the plan in §3 is an author choice)
- **Problem:** The main text is about 14,650 words (typical CMAME: 8–12k), the appendices about 9,090, and the preprint 86 pages. The Discussion (1,738 words) adds perhaps 700 new words:
  - §7.1 ¶3 repeats the Introduction ¶4 and §6.8;
  - §7.2 ¶1–3 repeat §4.2–4.3 and §6.7;
  - §7.3 ¶2 repeats §6.5;
  - §7.4 repeats §6.10, and its last paragraph repeats §5.3/Eq. (18).
- **Proposed actions:**
  - Apply the length plan (§3).
  - Condense §7 to about 850 words in three subsections: (a) boundary restriction against interior approximation; (b) what learning and correction each contribute, including NICE-post; (c) limitations and extensions.
  - Move the cost interpretation into §6.10.
- **Fix type:** Cut or move.
- **Related:** §3, I-54, I-55.

---

### Part C: Remaining Minor issues (consensus ≤2)

#### I-38: Chebyshev interval and coarse-solve verification
- **Theme:** T4 · **Severity:** Minor · **Consensus:** 2 (R1, R3)
- **Sources:** R1-11, R3-11 (part), R3-12
- **Status:** —
- **Problem:**
  - §6.2 says the 80-geometry check used its own power-iteration start, but App. D calls that endpoint "operational". The deployed b is verified on five cells only, and the minimum margin is 2.1% (R1-11).
  - Forty power steps × 1.05 is weaker and costlier than a 10–20-step Lanczos estimate with a Ritz-residual margin. a = b/30 is not justified (R3-11).
  - The colour-probed A_c is assumed exact, and the selected Cholesky shifts are not reported, although (F.1) needs probed = Galerkin (R1-11, R3-12).
  - In Table 3, the B columns come from the fixed-weight study (explicit VᵀAV with PARDISO) and the A3 column from deployment (probed, shifted Cholesky); the implementations differ (R3-12).
  - The paper should note that SPD and the lower bound do not depend on contraction; only the ordering and the accuracy do.
- **Proposed actions:**
  - Make §5.1 and App. D consistent.
  - Use a certified upper bound (Lanczos Ritz value plus residual).
  - Report ‖A_c^probe − VᵀAV‖/‖VᵀAV‖ and the selected shifts on the 80 geometries.
  - Report sensitivity to b/a (10, 30, 100) on the detailed cells.
  - Evaluate B+W and A3 through the same implementation in Table 3, or quantify the difference.
- **Fix type:** Re-analysis.
- **Related:** I-20, I-23.

#### I-39: δ²κ decomposition: inconsistent definitions and over-interpretation
- **Theme:** T3 · **Severity:** Minor · **Consensus:** 2 (R1, R6)
- **Sources:** R1-09, R6-12
- **Status:** — (note: R5 lists δ²κ as a strength and the length plan keeps it; R1 asks only that it be presented as a diagnostic)
- **Problem:**
  - §6.4 uses an interior Jacobi-weighted norm with a mixed denominator, while App. B.1 uses the full-field Euclidean norm (W = I). It is not stated which one produced ST19.
  - ε = δ²κ holds by definition for any norm, so it is a reparametrisation, not an explanation.
  - The "bending" interpretation (contribution iii) contradicts the J.7 caveat.
  - §7.3 equates large κ with high Jacobi-spectrum modes, which J.7 distinguishes.
- **Proposed actions:**
  - Use one definition throughout (align B.1 with §6.4, or state W = J_IᵀDJ_I), and state the norm used for ST19.
  - Present the identity as a diagnostic decomposition.
  - Support the bending claim (membrane/bending energy split, or Jacobi-spectrum location of the error) or drop it.
- **Fix type:** Text only (possibly re-tabulation).

#### I-40: Connectivity and the rank-six rigid-mode assumption
- **Theme:** T7 · **Severity:** Minor · **Consensus:** 2 (R1, R4)
- **Sources:** R1-17(f), R4-14
- **Status:** —
- **Problem:** §2.2 assumes exactly six rigid modes with a rank-six restriction to P. Heavily cut cells may contain disconnected islands, or material touching no shared face (H2, 2.67% macro volume, looks like an isolated ring in Fig. 1d), and thinning during optimisation can disconnect walls. Either case produces floating bodies and breaks the rigid split C_R.
- **Proposed actions:**
  - State the connectivity check used in generation, and report the per-geometry rank check.
  - Describe the treatment of disconnected components (removal, or a per-component rigid split).
  - Add an optimisation-time safeguard (a τ lower bound tied to connectivity).
- **Fix type:** Text only.
- **Related:** I-10.

#### I-41: Reuse and warm starts across design iterations dismissed
- **Theme:** T1 · **Severity:** Minor · **Consensus:** 2 (R1, R4)
- **Sources:** R4-16, R1-06 (part: warm start)
- **Status:** —
- **Problem:** §7.4 says "neither preparation can be reused across iterations". In late iterations τ changes little:
  - previous factorisations can precondition new systems (approximate reanalysis);
  - the previous global solution is an effective CG warm start;
  - the learned route could reuse b and the coarse factor while the active set is unchanged (R4-16);
  - the previous iteration's corrected field is a strong starting field (R1-06).

  The current statement biases the cost comparison.
- **Proposed actions:** Discuss reuse for both routes, state which the §6.11 timings use, and optionally include a warm-start variant.
- **Fix type:** Text only.
- **Related:** I-10.

#### I-42: Surrogate error lies below the reference discretisation error
- **Theme:** T3 · **Severity:** Minor · **Consensus:** 2 (R5, R7)
- **Sources:** R5-16 (part), R7-15
- **Status:** —
- **Problem:** For heavily cut cells, the n = 32 reference has about 1% discretisation error (§6.2), an order above NICE's 0.01–0.1%. R5-16: make explicit that the surrogate is not the limiting error. R7-15: this means (1) the correction budget could be reduced, and (2) the conventional route could use a coarser n (e.g. 24). Cost comparisons should therefore be made at matched continuum-level accuracy.
- **Proposed actions:** Add one sentence in §6.6 or §7, and discuss the implication for the cost comparison. Optionally add one accuracy-matched cost point (NICE with a reduced budget vs the conventional route at n = 24).
- **Fix type:** Text only (optional re-analysis).
- **Related:** I-05, I-24.

#### I-43: Training-population counts (591 / 305) uncertain
- **Theme:** T6 · **Severity:** Minor (but it affects the reading of I-19) · **Consensus:** 2 (R2, R6)
- **Sources:** R2-11, R6-16
- **Status:** —
- **Problem:**
  - R2-11: with a pool of three geometries and one replacement every 100 steps, a 15,000-step continuation sees at most about 150 new geometries, about a quarter of 591. B (40,000 steps) could have seen all 305. The data-scaling reading P0 148 → B 305 → C 591 is therefore uncertain.
  - R6-16: no evidence file contains 305. gen_supp_tables.py copies it from Table 2 (circular), and meta_p1.json records B's split as SPLIT_V3.json with lists.train = 591, identical to SPLIT_ARMS.json.
- **Proposed actions:** Report the distinct geometries (and geometry–view pairs) visited per arm. Archive B's 305-geometry list or correct the count. Optionally add a learning curve.
- **Fix type:** Re-analysis of existing logs.
- **Related:** I-02, I-19.

#### I-44: Architecture reproducibility; code and data availability
- **Theme:** T7 · **Severity:** Minor · **Consensus:** 2 (R2, R5)
- **Sources:** R2-10, R5-20 (part)
- **Status:** — (the scope of any code/data release is the authors' choice)
- **Problem:** Not specified (R2-10):
  - the values of the coefficient bounds a_max;
  - the initialisation of λ_mix, σ_ℓ and the weights;
  - the "eight-dimensional slot embedding";
  - whether k_b = 0.5 and the disabled feature extension apply to A2b/A3;
  - the element/face split of the 24 layers;
  - the per-block parameter breakdown;
  - pool sampling and the adversarial refresh schedule.

  There is no code/data availability statement, although evidence/*.json and figures_src/*.py exist (R5-20).
- **Proposed actions:**
  - Add a full hyperparameter table (supplement).
  - Add pseudo-code of one forward pass with tensor shapes.
  - Add a code and data availability statement with an archive DOI.
  - Release the model, generator and teacher as far as the authors decide.
- **Fix type:** Text only.
- **Related:** I-59.

#### I-45: Title
- **Theme:** T8 · **Severity:** Minor · **Consensus:** 2 (R2, R5)
- **Sources:** R5-§4, R5-07 (part), R2-07 (part)
- **Status:** The method name NICE is **DECIDED**. The title wording NEEDS AUTHOR DECISION.
- **Problem:** The current title:
  - does not carry the method idea (the network initialises, the correction completes);
  - says "TPMS cells" generally, although only P-type cells are studied;
  - omits thickness sensitivity and design;
  - lets "with equilibrium correction" dangle after "cells".
- **Proposed actions:**
  - R5's recommendation (once §6.11 exists): "Neural-initialised static condensation with equilibrium correction for the analysis and thickness design of cut thin-walled TPMS lattices".
  - R5's alternatives: foreground sensitivity, or drop "design" if §6.11 is small.
  - Put NICE in the Abstract and keywords, not at the start of the title.
  - Write "Schwarz-P-type" in the first Abstract sentence.
  - Add the keywords "neural-initialised condensation" and "thickness optimisation".
- **Fix type:** Text only.
- **Related:** I-05.

#### I-46: "Nine configurations common to all predictors"
- **Theme:** T9 · **Severity:** Minor · **Consensus:** 2 (R1, R6)
- **Sources:** R6-06, R1-17(a)
- **Status:** —
- **Problem:** §6.6 and the ST13 note say "the nine configurations common to all predictors", but B has only seven (no U2/y, no M1/y). The nine are common to C, S8, A2b, B+W and A3.
- **Proposed actions:** Write "the nine configurations common to C, S8, A2b, B+W and A3 (B was evaluated on seven of them)".
- **Fix type:** Text only.

#### I-47: Lattice numbers: §6.9 vs Table 6 vs §8
- **Theme:** T9 · **Severity:** Minor · **Consensus:** 2 (R5, R6)
- **Sources:** R5-11, R6-08, R6-09, R6-26
- **Status:** —
- **Problem:**
  - Learned compliance errors for the same lattices and loads differ between Table 6 and §6.9:

    | Lattice | Table 6 | §6.9 |
    |---|---|---|
    | 2×2×2 | 0.014, 0.011, 0.0097 | 0.014, 0.011, 0.0094 |
    | 3×3×1 | 0.015, 0.014, 0.015 | 0.015, 0.013, 0.010 |

    The 3×3×1 z-load differs by about 50% relative. S7 calls this "slightly"; the likely cause is the 10⁻⁶ CG tolerance vs the tighter §6.9 solve together with the 3.2e-3 floor. The Table 6 caption claims comparison with "the exact condensation of Section 6.9".
  - §6.9 gives the reference residual as "9×10⁻¹¹"; it is 9.08e-11 (2×2×2) and 1.29e-10 (3×3×1).
  - The random loads are "4–5%" above the bound: 5.1/4.4/5.0% and 5.0/4.2/4.5%.
  - §8's "to within 0.4%" applies to consistent loads only.
  - Confirm that the 5.83% appearing both as S8 M1/x T-y and as B U1/x N-z is correct.

  The Abstract's "at most 0.015%" holds for both sets of values.
- **Proposed actions:**
  - Explain the difference and its cause in the Table 6 caption and S7, or harmonise the numbers.
  - Write "at most 1.3×10⁻¹⁰".
  - Write "about 4–5%".
  - Qualify §8 to consistent loads.
  - Confirm the 5.83% values.
- **Fix type:** Text only.
- **Related:** I-04.

#### I-48: Cut-surface loads not reported for the principal predictor
- **Theme:** T3 · **Severity:** Minor · **Consensus:** 1 (R1)
- **Sources:** R1-12
- **Status:** —
- **Problem:** Cut targets receive three macro-cut tractions, but ST06 and Fig. 10 report only B, C and S8. These loads exercise the cut band, where B's error concentrates (39–43% of the error energy near the cut); B/M1/x reaches 9.9% sensitivity error under them. The lattices load only box faces.
- **Proposed actions:** Add A3 and B+W rows to ST06 and Fig. 10. Include a cut-surface load in one lattice or in §6.11.
- **Fix type:** Re-analysis (plus small new runs for the lattice).
- **Related:** I-52, I-53.

#### I-49: How cell interfaces are matched with cut neighbours
- **Theme:** T7 · **Severity:** Minor · **Consensus:** 1 (R1)
- **Sources:** R1-13
- **Status:** —
- **Problem:** Face patches are certified per cell. When a cut target neighbours an uncut or differently cut cell, the active face-node sets on the shared face can differ. The treatment of unmatched face nodes is not stated, nor whether the neighbour in the two-cell tests is cut by the same plane (physical consistency).
- **Proposed actions:** State the matching rule and how the certification tolerance is handled. Confirm that the reference and the surrogate use identical maps B_m, and that the pair geometries are physically consistent.
- **Fix type:** Text only.

#### I-50: Design-dependent loads frozen in the sensitivities
- **Theme:** T2 · **Severity:** Minor · **Consensus:** 1 (R4)
- **Sources:** R4-10
- **Status:** —
- **Problem:** Consistent tractions are integrated over τ-dependent material patches and normalised to a unit resultant, so the load depends on the design. The sensitivities hold the load at its base-design value and omit 2f_{g,c}ᵀÛ (App. H). That is fine as a definition, but not for an optimisation with loads on design cells.
- **Proposed actions:** State the §6.11 load model. Either include the term, or apply loads and supports through non-design regions.
- **Fix type:** Text only.
- **Related:** I-10.

#### I-51: The learned application barely benefits from batching
- **Theme:** T1 · **Severity:** Minor · **Consensus:** 1 (R3)
- **Sources:** R3-13
- **Status:** —
- **Problem:** G1 takes 54 ms for 1 vector, 531 ms for 16 and 923 ms for 64, almost linear in batch size. A memory-bound SpMM with K ≈ 1.4 GiB should amortise across columns: 34 K-actions at about 1.8 TB/s is roughly 25–30 ms up to moderate widths. This points to an implementation limit (per-vector loops or the network transpose). The learned route may be faster than reported, and the stated cause (§7.4) may be wrong. It matters for the batch gap in I-22.
- **Proposed actions:** Break an application down by component (network forward/transpose, smoothing SpMVs, coarse solves, K action) for batches of 1, 6 and 64, and compare with a bandwidth estimate.
- **Fix type:** Re-analysis (profiling).
- **Related:** I-22.

#### I-52: Retaining the cut band when cut surfaces are traction-free
- **Theme:** T1 · **Severity:** Minor · **Consensus:** 1 (R7)
- **Sources:** R7-16
- **Status:** — (tension with I-48, which asks for more cut-surface loading, and with the cut band counted as new in I-15)
- **Problem:** Cut-band coordinates are private to each cell and carry only cut-surface loads. In lattices and in design, specimen-boundary cuts are typically traction-free, so these coordinates could be condensed into the interior exactly. On M1 that is about 15k of 40k retained DOFs (ST07: 15,423 even at r = 1). Keeping them increases global DOFs and PCG work, and it is also part of the novelty claim.
- **Proposed actions:** Justify retaining them (cut-surface loads and supports), or offer a traction-free option and report its cost and accuracy effect.
- **Fix type:** Text only (optional experiment).
- **Related:** I-13, I-48.

#### I-53: Fig. 10 omits NICE; Fig. 9(c,d), Fig. 10(b) and Table 4 overlap
- **Theme:** T8 · **Severity:** Minor · **Consensus:** 1 (R5)
- **Sources:** R5-09
- **Status:** —
- **Problem:** The key mechanism figure has "192 observations from the 25 B, C and S8 model–configuration combinations" and no NICE or NICE-post points. Fig. 9(c,d), Fig. 10(b) and Table 4 show the same load-level relation three times.
- **Proposed actions:** Add NICE and NICE-post to Fig. 10 (the data are in ST13). Reduce Fig. 9 to panels (a,b). Fold Table 4 into two sentences and move it to ST13/ST15.
- **Fix type:** Re-analysis (re-plot), plus cut or move.
- **Related:** I-21, I-48, §3.

#### I-54: The main text relies on appendix equations; some appendices are never cited
- **Theme:** T8 · **Severity:** Minor · **Consensus:** 1 (R5)
- **Sources:** R5-10
- **Status:** —
- **Problem:** §5.1 relies on Eq. (B.8); §6.7 on (J.4) and (H.2); §4.3 and §7.2 on (J.5) and (H.5). More than 20 ST tables are cited from the main text. Appendices C, I, J.8 and J.9 are never cited.
- **Proposed actions:**
  - State the one-line results inline (the J.4 bound; the J.5 condition "uniformly C¹-small H").
  - Remove the reliance on B.8.
  - Move uncited appendices to the supplement.
  - Cap ST citations at about one per subsection.
- **Fix type:** Cut or move.
- **Related:** I-37, §3.

#### I-55: Abstract too long and too dense
- **Theme:** T8 · **Severity:** Minor · **Consensus:** 1 (R5)
- **Sources:** R5-13
- **Status:** NEEDS AUTHOR DECISION (which claims and numbers stay; this interacts with I-01, I-05, I-07, I-14, I-16, I-18, I-19, I-23 and I-28).
- **Problem:** The Abstract is about 370 words before the placeholder, with about 20 numbers; CMAME abstracts are typically 150–250 words. The method sentences pack in Chebyshev, Q1, Galerkin, matrix-free, SPSD and rigid-motion details, and the analysis sentences repeat the Introduction.
- **Proposed actions:**
  - Rewrite to about 220 words: problem (1 sentence), idea and guarantees (2), analysis insight (1), evidence with four numbers (2), design example (1).
  - Prepare Highlights.
  - Rewrite once, after the claim decisions above. Note that hardware and scope qualifiers add about 25 words.
- **Fix type:** Text only.
- **Related:** I-14, I-37.

#### I-56: Training described before the correction it is trained through
- **Theme:** T8 · **Severity:** Minor · **Consensus:** 1 (R5)
- **Sources:** R5-14
- **Status:** —
- **Problem:** §3.3 ("The correction of Section 5 can also be placed inside the training loop") comes before the correction is defined, with the whole of §4 in between.
- **Proposed actions:** Move §3.3 to a new §5.4 "Training through the correction" (the smaller change), or reorder to §2 Setting → §3 NICE operator → §4 Training → §5 Error analysis.
- **Fix type:** Text only (reorder).

#### I-57: Operator-consistency numbers in §6.2
- **Theme:** T9 · **Severity:** Minor · **Consensus:** 1 (R6)
- **Sources:** R6-05
- **Status:** —
- **Problem:** §6.2 says "symmetric to a relative 3×10⁻⁸ … agrees … to 2×10⁻⁸". The cited ST19 and p1_checks_cpu/_u2 give ≤ 8.7e-9 and ≤ 4.5e-9; p1_checks.json gives 1.15e-8 and 1.0e-8. The field-difference (1.1e-7) and rigid-energy (≤ 2e-11) values match. R5-12 contrasts these figures with the residual floor (I-04).
- **Proposed actions:** Write "9×10⁻⁹" and "5×10⁻⁹" per ST19, or name the other source.
- **Fix type:** Text only.
- **Related:** I-04.

#### I-58: Descriptive wording precision (§6.3, §6.4, §6.6)
- **Theme:** T9 · **Severity:** Minor · **Consensus:** 1 (R6)
- **Sources:** R6-13, R6-23, R6-24, R6-25, R6-28
- **Status:** —
- **Problem:** Five statements overstate or understate the records:
  - "reduces the element error energies by two to three orders of magnitude throughout the cell": the total ratio is 113–138 (about 2.1 orders), and the elementwise log10 ratios at the 10th/50th/90th percentiles are 1.4/1.9/2.5 (R6-13).
  - "marginally more accurate" / "equivalent" understates the gaps: on M2, B+W 0.083/0.075% vs A3 0.136/0.154% (1.6–2.1×); on U2, A3 is about 2× better (R6-23).
  - "from 0.35%": ST13 has 0.339–0.350% (R6-24).
  - C "fails the sensitivity requirement in four of them": C also fails compliance on M1/x (4.119%) and M1/y (3.458%), and B on M1/x (4.384%) (R6-25).
  - "ghost-penalty … below 0.05% in all cells examined in §6.4": ST19 covers only five cells, with no U1 record (R6-28).
- **Proposed actions:**
  - "about two orders of magnitude (roughly 25–300× elementwise)";
  - quantify the B+W/A3 differences;
  - "0.34–0.35%";
  - add "and, on M1, also the compliance requirement";
  - "in the five cells of Table ST19".
- **Fix type:** Text only.
- **Related:** I-19.

#### I-59: Numbers without an archived source
- **Theme:** T9 · **Severity:** Minor · **Consensus:** 1 (R6)
- **Sources:** R6-17, R6-18
- **Status:** —
- **Problem:**
  - The Fig. 7 layer shares (8%; 13% of elements; 39–43%; 19–22%) cannot be reconstructed: p1_field_M1_small.npz lacks the cut-plane definition, and plausible reconstructions did not reproduce them (R6-17).
  - §6.7 attributes the replacement values 15.6% and 19.3% to the T-y load, but ST05 gives maxima over six loads, and no per-load record exists. The 12.2% is confirmed (R6-18).
  - R6 also lists sources not archived in evidence/: ST02, ST03, ST04, per-load ST05, ST08–ST10, ST14, ST17, and the R2 element-group shares.
- **Proposed actions:** Archive the plane or layer mask with the figure data. Give per-load T-y values or write "maxima over the six loads". Archive the sources of the listed tables.
- **Fix type:** Text only (archiving).
- **Related:** I-30, I-44.

---

## 2. Requirements for Section 6.11 (design-optimisation example)

This section merges R1-§4, R2-§4, R4-§4, R5-01 and R7-§4/R7-11, with the relevant parts of R3-05, R3-09 and R4-10. The status is DECIDED (the section will be added); the choice of nice-to-haves is the authors'.

**Baseline plan (current placeholder):**
- compliance minimisation under a volume constraint;
- eight shared corner parameters per cell;
- OC or MMA;
- 8 or 27 cells;
- final design verified with the exact reference;
- history compared with an exact-condensation run.

### 2.1 Must-have (in priority order)

| # | Requirement | Requested by | Linked issues |
|---|---|---|---|
| M1 | **State which gradient drives the optimiser** (s̃_c or complete Ĉ,c), and why. **Verify it** at the initial, an intermediate and the final design against the exact C,c and central finite differences of Ĉ (all variables, or a random subset of ≥50). Report the global relative error, cosine, per-variable error distribution (median, 95th percentile, max) and sign agreement. Rewrite the §7.2 "left to subsequent work" sentence. | R1, R2, R4, R5, R7 | I-03, I-08 |
| M2 | **Twin optimisation with exact condensation** from the same start with identical settings. Compare the objective, volume and design-change histories, the iterations to convergence, and the final τ fields (max/RMS difference, visual). Report the exact compliance of both final designs (target: difference well below 1%). If the twin is too costly at the larger size, run it at 27 cells and use exact checks at selected iterates beyond. | R1, R2, R4 (placeholder already plans it) | I-10 |
| M3 | **Final-design verification with the exact reference:** compliance error, exact gradient, and the KKT/stationarity residual computed with *exact* gradients (is it an optimum of the true problem?). | R1, R2, R4, R7 | I-10 |
| M4 | **Training-domain control:** enforce τ ∈ [0.18, 0.70] plus the generator's span and gradient limits (0.47, ST11) as explicit constraints, or report how often and how far iterates leave the domain. | R1, R2, R4 | I-05, I-32 |
| M5 | **Cost:** wall time per iteration split into front end, correction setup, global solve (with iterations), sensitivities and design update; total time; peak GPU and host memory. Give the same for the exact twin, with hardware stated consistently with the I-01 decision. State any reuse or warm starts. Include the amortised offline cost (break-even). | R1, R2, R4 | I-01, I-02, I-41 |
| M6 | **Algebraic accuracy along the path:** report the recomputed residual (and Ūᵀρ) at every iteration, with a stopping criterion tied to attainable accuracy. | R1, R3, R4 | I-04 |
| M7 | **Lattice with cut cells, at least 27 cells (3×3×3).** Tests the fp32 floor, error accumulation over many learned cells and scalability. The placeholder allows 8; all requesting reviewers want ≥27. | R2, R4 (size); R1, R2, R4 (cut cells) | I-13, I-30 |
| M8 | **Complete problem statement:** objective, volume constraint, number of design variables, optimiser and all parameters (move limit, damping, asymptotes), convergence criterion, maximum iterations. State the load model: include 2f_{g,c}ᵀÛ, or introduce loads and supports through non-design regions. | R4 | I-50 |
| M9 | **Exploitation / surrogate-error check:** per-cell energy excess and assembled compliance error of the surrogate at the start and at the optimum; whether any cell leaves the validated domain; whether error grew during optimisation. Use the Eq. (10) residual indicator as a monitor if available. | R2, R4 | I-20 |
| M10 | **Compact presentation:** about 600–900 words, one figure (history + final design as TPMS geometry) and one table (verification + cost). Move the shared-design-variable paragraph from §4.3 into §6.11. The Abstract and §8 sentences must state quantitative outcomes (M2, M3, M5), not merely that the optimisation ran. | R4, R5 | I-37, I-55 |

### 2.2 Nice-to-have (in priority order)

| # | Requirement | Requested by | Note |
|---|---|---|---|
| N1 | **Design consequence of the compliance/sensitivity separation:** repeat the optimisation with an uncorrected predictor (C or S8) and with B+W; compare final designs and their exact compliance with A3 and the exact twin. | R4, R7 | Becomes a **must-have if contribution (iv) stays a headline claim** (I-14). If the designs do not differ, say so. |
| N2 | **Bending-dominated case** (cantilever, MBB or three-point bending) with localised load and support, on a non-box domain with cut cells from all three strata, including a sliver (<1/3 retained volume). | R4 (also R4-06 for lattices) | Can double as the I-30 lattice. |
| N3 | **Smoothness check:** Ĉ and C along one line through the optimum, crossing at least one active-set or ghost-set change. | R1, R4 | I-31 |
| N4 | **Practitioner baselines:** a uniform-density lattice of equal volume (improvement ratio), and a homogenisation-based graded design evaluated with the exact full model. | R4 (R4-07 item 5) | Quantifies what full-resolution condensation of cut cells buys. |
| N5 | **Cost against a full-model GPU multigrid-CG route** and a warm-start variant of the learned route. | R1, R4 | I-11, I-41 |
| N6 | **Second, larger lattice** (≥64 cells, e.g. 8×4×2) showing the scaling of CG iterations, memory (host streaming) and time. | R4 (R7 wants ≥125 in a scaling study) | I-13 |
| N7 | **Manufacturability:** min/max wall thickness in h and in cell size, and the final relative-density field. | R4 | I-32 |
| N8 | **Cut-surface load or support** in the example. | R1 | I-48 |
| N9 | **Second load case** (multi-load compliance), to show that the per-load participation argument holds in aggregate. | R4 | |
| N10 | **Fair PIML-type baseline optimisation**, verified exactly. | R7 | Depends on the I-35 decision. |

**Tension to resolve:** meeting M1–M9 in 600–900 words requires putting the gradient-check, twin-history and cost-breakdown tables in the supplement, with only headline numbers in the main text.

---

## 3. Length-reduction plan

### 3.1 R5's plan (summary)

- **Current size:** main text about 14,650 words (the file itself has about 16,800 words including tables and captions), appendices about 9,090, 86 pages, 11 figures, 6 tables.
- **Target:** about 9,500 words before §6.11 and about 10,300 with it (window 9,500–11,000).
- **Main text:** a net saving of about 5,125 words (including +180 for a new Limitations paragraph), by moving rather than deleting throughout:
  - §6 goes from 6,643 to about 3,823 words;
  - §7 from 1,738 to about 890;
  - the Introduction from 1,465 to about 915.
- **Appendices:** about 4,300 words move to the supplement (F, G, I.1–I.2, J.7–J.9), with J.1–J.6 deduplicated into B, C and H. About 4,790 words remain.
- **Other savings:** about 380 caption words; the Abstract from 370 to 220 words.
- **Tables and figures:** Tables 4 and 6 move to the supplement; Fig. 6 moves; Fig. 9 is reduced to panels (a,b).
- **Supplement:** reordered to follow the main text; the legacy D material (S1, S4 and associated items) goes to an archive or repository.
- **Page count:** about 55–62 pages.

**R5 main-text table (kept verbatim):**

| # | Item (location) | Action | Destination | Est. words saved |
|---|---|---|---|---:|
| **Introduction (1,465 → ~915)** | | | | |
| 1 | ¶3 L15 "Multilevel iteration supplies…": the operator formulas F=𝒲Ê, Ŝ=FᵀKF, lower bound, complete transpose | Delete the formulas and keep one sentence of idea plus the hybrid-solver citations; the content is in §3.2 and §5.2 | — (already in §3.2 and §5.2) | 110 |
| 2 | ¶4 L17 PIML paragraph | Condense to about 150 words: PIML line of work → reduced-boundary error → one-line pointer to §6.8 | — | 110 |
| 3 | ¶6 L21 "We construct the initial extension…" | Merge into ¶3 and the contributions; it repeats §3.1 and contribution (ii) | §3.1 | 120 |
| 4 | ¶7 L23 "We analyse this construction…" | Condense to 2–3 sentences; it repeats §4 and contributions (iii) and (iv) | — | 80 |
| 5 | Contributions L25 | Reduce to 3 items (R5-03) | — | 60 |
| 6 | ¶1, ¶2, ¶5 (L11, L13, L19) | Light trimming | — | 70 |
| **Section 2 (732 → ~640)** | | | | |
| 7 | Paragraph after Table 1 (L105) | Reduce to one sentence | Table 1 footnote | 55 |
| 8 | §2.3 last ¶ (L85), aggregation sentences | Keep the metric definitions and move the aggregation rules | App. A.2 (already there) | 40 |
| **Section 3 (1,776 → ~1,325)** | | | | |
| 9 | §3.1 ¶1 L117, PIML contrast | Delete (repeats the Introduction) | — | 50 |
| 10 | §3.1 architecture ¶¶ L121, L123, L133, L135 | Condense from about 550 to about 300 words; Fig. 3 and App. G carry the details | App./Supp. G | 250 |
| 11 | §3.1 last ¶ L145, last two sentences | Delete | — | 30 |
| 12 | §3.3 ¶3 L190: difficult-direction search, 48-symmetry augmentation | One sentence plus pointer | Supp. G.3 | 90 |
| 13 | §3.3 | Relocate as §5.4 (R5-14); light trim | — | 30 |
| **Section 4 (993 → ~853)** | | | | |
| 14 | §4.3 ¶3 L252, Δq decomposition | Move together with the replacement study | App. H | 60 |
| 15 | §4.3 ¶5 L267, shared design variables | Relocate to §6.11, where it is used | §6.11 | 0 (relocated) |
| 16 | §4.1 last ¶, §4.2 last ¶, §4.3 ¶2: cross-reference sentences | Trim | — | 80 |
| **Section 5 (820 → ~700)** | | | | |
| 17 | §5.1 ¶2 L285 | Trim | — | 40 |
| 18 | §5.1 ¶3 L287: interval verification numbers "2.1–5.0%" (duplicated in §6.2) | Keep in one place | §6.2 / App. D | 40 |
| 19 | §5.3 L324–331 (Eq. 18 text) | Keep the equation; trim the explanation | App. J.6 | 40 |
| **Section 6 (6,643 → ~3,823 before §6.11)** | | | | |
| 20 | §6 intro L335 | Reduce to 2 sentences | — | 45 |
| 21 | §6.1 ¶1 L339: parameter ranges, cut-severity definitions | Keep strata; move ranges | Supp. ST11 | 40 |
| 22 | §6.1 ¶2 L341 and Table 2: predictor descriptions | Rewrite with the 5 descriptive names; drop S8, P0 and D; Table 2 goes from 7 rows to 5 | Supp. R1 | 100 |
| 23 | §6.1 ¶4 L359, aggregation rules | Move | App. A.2 | 90 |
| 24 | §6.1 ¶5 L361, assembly set-up | Trim; the Fig. 4 caption carries the geometry | — | 50 |
| 25 | §6.2 ¶¶ L369–373, verification | Keep a summary of about 180 words (reference error ≤0.06% for U and M and about 1% for H at n = 32; penalty and integration insensitivity; symmetry, energy–action consistency and rigid energy near 10⁻⁸); move the details | Supp. Note S5 + Fig. S06 + ST19 | 400 |
| 26 | §6.3 ¶1 L377, nodal-force stress-test rationale | Reduce to one sentence | Supp. R2 | 60 |
| 27 | §6.3 ¶3 L381, intermediate predictors | Condense | — | 80 |
| 28 | §6.3 ¶4 L383, orientation dependence (B only) | Move | Supp. ST01c / R2 | 70 |
| 29 | §6.4 ¶¶1–2 L391–393 and **Figure 6**, spectral distribution | Keep a summary of about 80 words (the M1 low modes below *a* motivate the coarse space); move Fig. 6 and the U1 nodal-force reversal | Supp. (new Fig. S07) + ST02b | 200 (+ 90 caption) |
| 30 | §6.4 L405 δ²κ | Keep; trim | — | 30 |
| 31 | §6.4 L407, linear against quadratic sensitivity term | Reduce to 2 sentences | Supp. ST02 / Fig. S03 | 70 |
| 32 | §6.5 ¶¶ L411–417: smoothing only, coarse, budget | Merge into one paragraph; the numbers stay in Fig. 8 | Supp. ST03, ST04 | 180 |
| 33 | §6.5 Table 3: the five 32/Q₁/32 rows | Keep the 8/Q₁/8 rows; state the 32-step result in one sentence | Supp. ST18 | 0 (table only) |
| 34 | §6.5 ¶ after Table 3 (L436), B+W against A3 locally | Move the B+W argument to §6.6; trim | — | 60 |
| 35 | §6.6 ¶1 L444: per-predictor listing, L1 host-memory detail | Condense; Fig. 9 and ST13 carry the numbers | Supp. ST13 | 230 |
| 36 | §6.6 **Table 4** | Remove; quote the 2–3 key numbers in the text | Supp. ST13 / ST15 | 40 (+ table) |
| 37 | §6.7, merged into §6.6: replacement study L466 | Keep the participation example (about 100 words); move the replacement decomposition | Supp. ST05 + App. H | 190 |
| 38 | §6.8, Bernstein restriction | Trim to about 270 words; keep Fig. 11, which is the positioning evidence | Supp. ST07 | 100 |
| 39 | §6.9 ¶1 set-up and ¶2 (3×3×1 repetition) | Condense; report both lattices in parallel form | — | 120 |
| 40 | §6.9 ¶3, bound comparison | Condense to 2 sentences | — | 80 |
| 41 | §6.10 ¶¶ L492, L494, L505, L507 | Trim and merge; remove the front-end ratio from the method claims (R5-05) | — | 140 |
| 42 | §6.10 L509–520 and **Table 6**, whole-lattice direct solution | Keep an 80-word summary with the key numbers (4-cell: 38 s against 242–333 s; 8-cell factor memory above 90 GB); move the table and details | Supp. Note S7 / ST20 | 420 (+ table) |
| 43 | §6.10 L522, legacy study | Delete the sentence (R5-19) | Supp. S4 or repository | 25 |
| **Section 7 (1,738 → ~890 including a new Limitations paragraph)** | | | | |
| 44 | §7.1 (427 → ~150): ¶3 repeats the Introduction and §6.8 | Condense into the "Boundary restriction against interior approximation" subsection | — | 277 |
| 45 | §7.2 (494 → ~200): ¶1–3 repeat §4.2–4.3 and §6.7 | Keep ¶4 (complete derivative), updated for §6.11 | — | 294 |
| 46 | §7.3 (460 → ~250) | Remove the numbers restated from §6.5; keep the complementarity argument and the NICE-post message | — | 210 |
| 47 | §7.4 (349 → ~100) | Merge reuse and amortisation into the end of §6.10; delete the Eq. (18) ¶ (it repeats §5.3) | §6.10 | 249 |
| 48 | New "Limitations and extensions" (R5-07) | Add | — | −180 |
| **Section 8 (482 → ~382)** | | | | |
| 49 | §8 ¶3 L570, which repeats abstract numbers | Condense; keep 3–4 numbers; leave room for the §6.11 sentence | — | 100 |
| | **Total main-text saving (net, including +180 for Limitations)** | | | **≈ 5,125** |

**R5 resulting size:**

| Stage | Main text (words) |
|---|---:|
| Current | 14,650 |
| After the cuts | ≈ 9,525 |
| After adding §6.11 (≈750) and its §8 sentence (≈50) | ≈ 10,325 |

**R5 appendix plan (summary):**

| # | Action | Words moved or saved |
|---|---|---:|
| A1 | A.2 metrics: keep only what §2.3 and §6.1 no longer contain; deduplicate | 130 |
| A2 | Move F.1–F.4 to the supplement (implementation note) | 809 |
| A3 | Move G.1–G.3 to the supplement | 1,587 |
| A4 | Move I.1–I.2 (uncited) to the supplement | 145 |
| A5 | I.3 → 2 sentences in §6.8 or App. C | 80 |
| A6–A8 | Merge J.1 and J.2 into B, and J.3 into C (deduplicate) | 400 |
| A9, A10 | J.4–J.5 merged with H; J.6 kept beside C | 0 |
| A11–A13 | Move J.7, J.8 and J.9 to the supplement (J.9 next to S2) | 1,149 |
| | **Total** | **≈ 4,300** |

Remaining appendices: A (condensed), B (with J.1–J.2), C (with J.3, J.6, I.3), D, E, H (with J.4–J.5).

### 3.2 Where other reviewers' requests would add text

The estimates below are the clerk's. "Main" is the recommended main-text increment, assuming each new result is reported in one to three sentences with tables in the supplement. "Full" is the increment if written out in the main text.

| Issue | Addition | Requested by | Full (words) | Main (words) | Recommended placement |
|---|---|---|---:|---:|---|
| I-15 | "Classical background" paragraph with citations (§4) | R1, R7 | 200 | 120 | Main |
| I-09 | Related-work paragraph (SCRBE, inexact DD, learned MG, neural operators, TO reanalysis) | R2, R3, R4, R5, R7 | 350 | 200 | Main (partly absorbed by plan item 2, the condensed PIML ¶) |
| I-05 | Limitations beyond R5's +180 (OOD result, reuse domain) | R1, R2, R3, R4, R7 | 100 | 60 | Main (§7) |
| I-01/02/36 | Hardware qualifiers, baseline justification, offline cost and break-even, PARDISO settings | R1–R5, R7 | 250 | 120 | Main (§6.10); settings to S7 |
| I-03 | Complete-derivative results; "recommendation for design use" | R1, R2, R4, R5, R7 | 200 | 100 | Main (replaces the kept §7.2 ¶4) |
| I-08 | Gradient-level metric | R1, R2, R4, R7 | 120 | 80 | Main (§6.6/§6.9) + supplement table |
| I-04 | Residual/precision reporting, fp64 check | R1–R5, R7 | 120 | 80 | Main (§6.9) |
| I-18 | Worst-direction statistics | R1, R2, R3 | 100 | 60 | Main (§6.3) + ST01 |
| I-19 | C+W result, seeds | R1, R2, R5, R7 | 80 | 60 | Main (§6.3/§6.6) |
| I-23 | Measured contraction factor; reworded bound | R1, R3, R7 | 60 | 50 | Main (§5.2/§6.5) |
| I-12 | Pareto panel | R2, R3, R5, R7 | 100 | 60 | Panel in Fig. 8 + supplement |
| I-20 | Runtime indicator and policy | R2, R3, R4, R7 | 150 | 100 | Main (§5 or §6.11) — only if implemented |
| I-17 | Held-out assembly tests | R1, R2, R4, R5 | 80 | 60 | Main (§6.6) + ST13 |
| I-11 | Whole-lattice iterative baseline | R1, R3, R4, R7 | 150 | 80 | Row in ST20/Table 6 (supplement); only if run |
| I-13 | Scaling study; preconditioner description | R2, R3, R4, R7 | 200 | 120 | Main (§6.9/§6.10) — only if run |
| I-34 | Ablations (w_s, architecture) | R2, R7 | 150 | 40 | Supplement table + 1 sentence |
| I-24 | Extended reference verification | R1, R4 | 150 | 30 | Supplement (S5); §6.2 summary stays ≈180 |
| I-30/I-32 | Lattice figure; τ → thickness/density map | R4, R6 | 140 | 80 | +1 figure (or panel in Fig. 1) |
| I-31 | Smoothness sweeps | R1, R4 | 100 | 30 | Supplement + 1 sentence |
| I-39–I-42, I-49, I-50, I-52 | Short clarifications (connectivity, interfaces, loads, reuse, discretisation, cut band) | R1, R4, R5, R7 | 250 | 150 | Main (scattered) |
| I-10 | §6.11 must-haves beyond R5's 750 words | R1, R2, R4, R5, R7 | 500 | 300 | Main + supplement tables |
| I-35 | §6.8: option (a) moves it | R7 | −270 | −270 | Supplement (saves beyond R5 item 38) |
| I-35 | §6.8: option (c) makes it fair | R7 | +200 | +100 | Main (alternative to (a)) |

**Net projection (clerk's estimate):**

| Scenario | Main-text words | Within the 9.5–11k window? |
|---|---:|---|
| R5 plan including §6.11 | ≈ 10,325 | Yes |
| + text-only and re-analysis additions (I-01/02, I-03, I-04, I-05, I-08, I-09, I-15, I-18, I-19, I-23, clarifications) and the §6.11 must-haves | ≈ 11,700 | No (+700) |
| … with §6.8 moved to the supplement (I-35 option a) and the robustness results (I-18/I-19/I-34) merged into one ≈150-word paragraph | ≈ 11,250 | Marginal |
| + all new experiments reported in the main text (I-11, I-12, I-13, I-17, I-20, I-30, I-31) | ≈ 11,900–12,300 | No |

**Implications for a realistic plan:**
1. **Adopt R5's cuts in full.** They are the only large source of savings, and none of the other reviewers objects to them. The exception is §6.8, where R5 and R7 disagree (I-35).
2. **Report new results in a fixed format:** one or two sentences in the main text plus a supplementary table. This applies to I-11, I-12, I-13, I-17, I-18, I-19, I-24, I-31 and I-34.
3. **For experiments that are not run,** add a single Limitations sentence (about 20 words each) instead of new sections.
4. **Merge the §6.11 verification and cost details into one table**, and keep §6.11 near the upper end of R5's 600–900 words.
5. **Figures and tables.** R5's plan ends at 11 figures and 5 tables with §6.11. Additions from I-30/I-32 (lattice and τ map) and a nomenclature/cell key (I-21) would bring this to about 12 figures and 6 tables unless they become panels (Fig. 1, Fig. 8) or go to the supplement.
6. **Abstract.** R5's 220-word target must absorb the hardware and scope qualifiers (I-01, I-05, I-16, I-18): about +25 words.
7. **Supplement.** It grows (hyperparameter table I-44, benchmark environment I-27, ablations, scaling, reference verification). This is acceptable to R5's plan, but the supplement's cross-reference table (R5 §3.3) should be updated accordingly.
