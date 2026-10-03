# Review R3: numerical linear algebra, correction and cost

Manuscript: "Learned static condensation for cut thin-walled TPMS cells with equilibrium correction" (CMAME submission)
Reviewer expertise: multigrid and domain decomposition, polynomial smoothers, preconditioned Krylov methods, sparse direct solvers, CPU/GPU benchmarking.
Scope: Section 5, Sections 6.2, 6.5, 6.9 and 6.10, Tables 3, 5 and 6, Appendices D–F and J.6–J.8, Supplementary Notes S1, S4, S5 and S7, and Tables ST04, ST08, ST09, ST14 and ST17–ST20. Section 6.11 and the placeholders were not reviewed, as instructed.

---

## 1. Summary

The manuscript approximates the Schur complement of cut thin-walled TPMS cells on the full retained space. A geometry-conditioned network, linear in the retained displacement, supplies an initial interior field. A fixed linear two-grid correction then improves it: k Jacobi–Chebyshev steps, a Galerkin solve on a trilinear Q1(17) interior coarse space, and k more steps. The condensed operator is applied matrix-free as F^T K F, so it is symmetric positive semidefinite, bounded below by the exact S, and reproduces rigid modes. The algebra behind these statements (Ritz identity, complete transpose, A-orthogonal coarse projection, Chebyshev non-expansion) is correct and clearly written. The Chebyshev upper endpoint is checked carefully against Lanczos on all 80 geometries, and the historical and negative results (Supplementary Note S4) are reported with unusual candour. The accuracy evidence for the corrected operator is convincing at the scale tested (two-cell pairs and two 8-cell lattices). The cost evidence is not. The main-text comparison sets a GPU implementation against a 16-thread host PARDISO **LU** factorisation of an SPD matrix, apparently with a sequential solve phase. The host route runs its front end on the CPU although a GPU front end exists. The supplement's own same-GPU exact-factor timings (Table ST14) show exact condensation applying faster than the corrected learned operator. The expected iterative and domain-decomposition baselines are also missing.

## 2. Recommendation

**Major revision.**

The method and its variational analysis are sound, and the accuracy study of the correction is thorough. The cost conclusions are not supported as stated, and they appear in the abstract and the conclusions. Examples: "applies it 2 to 23 times faster" and "cheaper for any number of queries". These claims rest on a baseline configuration that a solver specialist would not accept, and the manuscript's own supplementary data contradict them. Several other points need attention:

- the finite-precision behaviour of the assembled solve: the true residual stagnates at 3.6e-4 to 3.2e-3;
- the lack of any measured contraction factor for the two-grid correction;
- the strength of the correction and starting-field baselines used to establish the network's contribution.

Most of these can be fixed with re-analysis of existing data, a few new benchmark runs and rewording. None of them undermines the construction itself.

---

## 3. Findings

### R3-01. The same-GPU exact baseline in the supplement contradicts the main-text cost claims

- **Severity:** Major
- **Location:** Section 6.10 and Table 5 ("applies it 2 to 23 times faster"; "the learned route is cheaper for any number of queries per design iteration"); Abstract; Section 8; compared with Table ST14b/c and Supplementary Note S4 ("Exact double precision ... the exact factor is faster for batches of 64 directions").
- **Issue:** Table ST14c gives GPU fp64 exact interior-factor applications on the same RTX 5090 and the same four cells (G1–G4):
  - one vector: 7.9–73 ms;
  - 16 vectors: 13–100 ms;
  - 64 vectors: 35–316 ms.

  The corrected A3 operator in Table 5 takes 25–99, 205–1,052 and 388–1,828 ms for the same batches. On the same device, the exact route therefore applies about 1.4–3x faster for a single vector and 6–12x faster for 16 or 64 vectors.

  GPU fp64 factorisation (ST14b) takes 0.64–8.6 s against 0.16–1.09 s for learned preparation, so the learned route wins on setup by only about 4–8x.

  A rough per-design-iteration estimate for G1 uses the 183 batched applications of six loads reported in Section 6.10:
  - exact on GPU: about 2.7 s + 183 x 34 ms, or about 9 s;
  - learned: about 0.7 s + 183 x 210 ms, or about 40 s.

  On equal hardware, "cheaper for any number of queries" is therefore reversed, not merely weakened. The main text never mentions this comparison. It points to S4 only as a "historical deployment study of an earlier uncorrected predictor", but the exact-factor timings in S4 do not depend on the predictor.
- **Why it matters:** It is the central practical claim of the paper, and it appears in the abstract and the conclusions.
- **Suggested fix:**
  - Re-run the exact route on the GPU in the current software stack alongside A3, for example with cuDSS or CHOLMOD-GPU Cholesky, including factorisation, applications for batches of 1, 6, 16 and 64, and memory. Report it in Table 5 next to the host route.
  - Restate the cost conclusions per hardware configuration.
  - If the honest advantage is memory or capacity (for example, all factors of a lattice not fitting on one GPU, since each G1 factor is about 4.7 GiB) or setup time, say so explicitly. Do not claim uniform speed.
  - Remove "cheaper for any number of queries" unless it survives the same-hardware comparison.
- **Fix type:** new experiment (timings only), plus text.
- **Confidence:** High for the discrepancy, which is visible in the manuscript's own tables. Medium for the size of the per-iteration estimate, because ST14 comes from an earlier benchmark run.

### R3-02. The host sparse-direct baseline is configured unfavourably

- **Severity:** Major
- **Location:** Section 6.10, Table 5 and Table 6 ("LU, the factorisation used in Table 5"); Supplementary Note S7; the sentence "the host solves are limited by reading the factor from memory".
- **Issue:** Five aspects of the host configuration disadvantage the conventional route.
  1. The interior matrix A = K_II is SPD, yet Table 5 uses PARDISO's unsymmetric LU. Table 6 shows what this costs: LU needs about 2x the memory (57–64 GB against 29–32 GB) and about 2x the factorisation time of Cholesky. The memory ratios "4 to 14 times less" and the condensation-time ratios are inflated by roughly that factor. Single-vector solve times also double, because the solve reads both factors.
  2. The implied solve bandwidth is implausibly low. For G1, one application reads the 5.56 GB factor in 826 ms, about 7 GB/s. In Table ST20b, the Cholesky solve of six right-hand sides with a 32 GB factor takes 20.6 s. Both figures match a single-threaded forward and backward substitution. In MKL PARDISO the parallel solve phase is controlled by iparm(25), and the default is sequential. If that default was used, "limited by reading the factor from memory" misattributes a configuration effect.
  3. The dense S is formed by "one host solve per retained coordinate" (4–59 min). MKL PARDISO can return the Schur complement directly through a partial factorisation (iparm(36)), and so can MUMPS. Even without that feature, blocked multi-right-hand-side solves are standard. The column-by-column baseline is a straw man.
  4. The comparison uses only 16 of the host's 128 logical cores against a whole flagship GPU.
  5. No PARDISO settings are reported: ordering (iparm(2)), scaling and matching, parallel solve (iparm(25)), MKL version.
- **Why it matters:** Every host-side ratio in Tables 5 and 6, the abstract and Section 8 depends on these settings.
- **Suggested fix:**
  - Use Cholesky or LDL^T (mtype = 2 or -2) for Table 5.
  - Enable the parallel solve phase and report the iparm settings.
  - Form the explicit S with the Schur-complement feature, or with blocked right-hand sides.
  - Report the CPU model, memory bandwidth and thread count, and add a run on the full node or a justification for using 16 cores.
  - Recompute all quoted ratios.
- **Fix type:** new experiment (benchmark reruns), plus text.
- **Confidence:** High for items 1, 3, 4 and 5. Medium for item 2, which is inferred from the implied bandwidth.

### R3-03. What each timing includes is asymmetric, and the Table 6 timing conditions are not comparable

- **Severity:** Major
- **Location:** Table 5, front-end column ("on the host cores for the conventional route and on the GPU for the learned route"; "ready for queries after 3.2 to 7.6 s ... 9.9 to 61 s"); Table 6 and ST20b ("Cells: setup + assembly 139.6–152.5 s"); Supplementary Note S7 ("The host was shared with other jobs ... load average between 11 and 37"); the Table 5 caption ("otherwise idle job").
- **Issue:** Four asymmetries affect the comparison.
  1. The front end (setup, moment integration, stiffness assembly) is a shared, method-independent stage. A GPU implementation exists, and the paper runs it for the learned route only. Assembling K on the GPU and copying it to the host costs a fraction of a second. The "factor 3.1 to 8.1" readiness advantage is therefore largely a hardware difference. In Table 6 the host front end accounts for 140–152 s of the 242–256 s direct time. The solver phases alone are 91 s (Cholesky) against about 31 s (learned setup and PCG), before correcting for R3-02.
  2. Table 6's direct runs used a host shared with other jobs (load average 11–37), whereas Table 5 claims an idle host.
  3. The direct route solves six loads to a residual of 1e-10. The learned route solves three loads to a recursive residual of 1e-6, and its true residual is not reported for these runs (see R3-05).
  4. The eight-cell Cholesky factorisations were skipped under a 60 GB threshold, although 66.6 GB plus the observed 5–6 GB overhead fits in the 90 GB available. The eight-cell direct entries are therefore only lower bounds up to the symbolic analysis.
- **Why it matters:** The headline ratios mix method, hardware and run conditions.
- **Suggested fix:**
  - Run the front end on the same device for both routes, or report the solver-phase comparison as the primary one.
  - Repeat the Table 6 direct runs on an idle host, with repetitions and a reported spread.
  - Stop both routes at matched response accuracy (for example, compliance to 1e-4) with matched load counts.
  - Run the eight-cell Cholesky factorisation.
- **Fix type:** new experiment, plus text.
- **Confidence:** High.

### R3-04. Iterative and domain-decomposition baselines for the whole lattice are missing

- **Severity:** Major
- **Location:** Section 6.10 ("A further alternative avoids condensation altogether ... factorised once by a sparse direct solver"); Section 7.4; the reference list.
- **Issue:** The only whole-lattice baseline is a sparse direct solve of 0.9–2.1 M degrees of freedom. The obvious alternative is PCG with algebraic multigrid on the full cut-FE model. Candidates are smoothed aggregation with the six rigid-body modes as near-nullspace (PETSc GAMG, ML/MueLu, hypre BoomerAMG in its elasticity/nodal mode) and GPU AMG such as AmgX. At this problem size such solvers usually run in tens of seconds with modest memory. Thin walls, Q2 elements and ghost penalty may make AMG struggle. That would itself be a useful result and would strengthen the paper.

  The setting is also classical iterative substructuring: cells are subdomains, the retained set is the interface, and PCG on the assembled Schur complement is primal substructuring. BDDC and FETI-DP (for example PETSc PCBDDC) are the standard solvers for it, and the balanced two-level preconditioner of S1.1 is a variant of that family. The literature on inexact subdomain solvers in BDDC and FETI-DP addresses the same question the paper asks: replacing the interior factorisation with a cheaper approximate extension. None of it is cited or compared. Relevant references include:
  - Toselli & Widlund, *Domain Decomposition Methods* (2005);
  - Dohrmann, SIAM J. Sci. Comput. (2003);
  - Farhat et al., IJNME (2001);
  - Li & Widlund, CMAME (2007);
  - Klawonn & Rheinbach, IJNME (2007);
  - Vaněk, Mandel & Brezina (1996) for smoothed aggregation.
- **Why it matters:** A solver expert will ask whether the learned operator beats a well-tuned iterative solver of the full problem, or a standard DD method, at equal accuracy and on the same hardware. Without that answer, the claim that the learned route is the practical choice for lattices is unsupported.
- **Suggested fix:**
  - Add an AMG-PCG whole-lattice baseline (GPU if possible, otherwise CPU on the full node) to Table 6 for at least the four- and eight-cell lattices. Report setup and solve time, iterations and memory at matched compliance accuracy.
  - Discuss, and ideally run, BDDC with cells as subdomains.
  - Position the method relative to inexact-subdomain-solver DD in the introduction and in Section 7.
- **Fix type:** new experiment, plus text.
- **Confidence:** High.

### R3-05. Finite precision: the true residual stagnates, precision statements conflict, and residual terms are not reported

- **Severity:** Major
- **Location:**
  - Section 6.9 ("its recomputed residual stagnates at 3.6 x 10^-4, the level set by the network's single-precision arithmetic"; 3.2 x 10^-3 for the 3x3x1 layer);
  - Table 1 ("stiffness actions, smoothing, coarse solve and energies in double precision"), compared with Appendix F.3 ("each smoothing routine returns its input dtype ... a float32 pre-smoothed field can be rounded before entering the float64 coarse solve; the final corrected field returns to the original dtype");
  - ST08a ("Fused/sparse action difference 2.5e-05");
  - S1.2 ("The saved result summaries ... omit the residual returned by the solver");
  - Appendix J.6 (signed residual work and omega "are not stored").
- **Issue:** There are five related problems.
  1. In the lattices the recursive PCG residual reaches 1e-10, but the true residual of the same operator stalls three to seven orders of magnitude higher. The stall grows by 10x from the 2x2x2 block to the 3x3x1 layer. Iterating well past the attainable accuracy wastes applications and inflates the 183/186 application counts used in the cost argument. The divergence between the two residuals shows that PCG is working with an operator that is not exactly symmetric and linear in floating point.
  2. Table 1 says the correction runs in fp64, but F.3 shows its fields are rounded to fp32 between stages and on output. The deployed operator is therefore fp32-limited beyond the network itself. The ST08a fused/sparse action difference of 2.5e-5, if the fused path is deployed, is a further source.
  3. The attribution to "the network's single-precision arithmetic" is asserted, not demonstrated.
  4. The reported compliance errors are 1e-2 to 7e-5 percent (for example U1/x N-z in Table 4). By Eq. (18), a relative residual of 1e-3 can contaminate compliance through the residual work U^T rho at a level that the Euclidean bound in J.6 does not rule out. For the two-cell runs the final residual was not even saved.
  5. The same issue will govern gradient accuracy in the optimisation example.
- **Why it matters:** The accuracy numbers at the 1e-4-percent level, and the iteration counts in the cost argument, depend on it. Both will likely degrade as lattices and condition numbers grow.
- **Suggested fix:**
  - Re-run one pair and one lattice with the network, rigid split and retained restoration in fp64 and all casts removed, to show whether the stall disappears and whether the reported errors change.
  - Report U^T rho / C and the action–energy term omega (J.6) for the reported solves.
  - Save and report true residuals for the pair runs.
  - Use a stopping criterion tied to attainable accuracy, or residual replacement, or mixed-precision iterative refinement (Carson & Higham, SISC 2018).
  - Make Table 1 consistent with F.3.
  - State explicitly that the pair runs use the exact reference factor as preconditioner, so their iteration counts (6–16) say nothing about deployable cost.
- **Fix type:** new experiment (small), re-analysis, text.
- **Confidence:** High for the existence of the issue. Medium for its effect on the reported errors.

### R3-06. The two-grid analysis proves only non-expansion, and no contraction factor is measured

- **Severity:** Major
- **Location:** Section 5.2, Eq. (17) and following ("S ⪯ Ŝ_tg ⪯ Ŝ_0"); Appendix D ("which proves the rho_k^4 energy estimate"); Abstract ("its error ... can be reduced through ... the correction budget").
- **Issue:** rho_k in Eq. (D.2) is the maximum of |p_k| over the whole spectrum of D^-1 A. The spectrum extends to lambda_1 ≈ 4e-4 to 6e-4 (ST02b), far below a ≈ 0.17. So p_k(lambda_1) ≈ 1, rho_k ≈ 1, and the "rho_k^4 estimate" is only non-expansion. The analysis never uses an approximation property of the coarse space, so it gives no rate. The data suggest the worst-case two-grid contraction may be weak: a zero interior start is left with 28–1,097% excess after 8/Q1/8 (Table 3). Every "controllability" statement is therefore empirical and directional, and only the monotone ordering is proved.
- **Why it matters:** Two consequences follow.
  - The claim that the correction budget controls the error needs either a rate or a measured factor.
  - A measured factor would give something the paper currently lacks: an all-direction bound. With T = P_k C_V P_k, which is A-self-adjoint, H_tg = T H_0 and hence epsilon_*(F) ≤ ||T||_A^2 epsilon_*(Ê), and m cycles give ||T||_A^{2m}.
- **Suggested fix:**
  - Compute ||T||_A (and ||T^m||_A for a few m) per geometry by Lanczos on the A-symmetric error-propagation operator, at least on the detailed cells and ideally across the 80. This costs a few dozen applications of the cycle.
  - Report the directional reduction factor against this worst-case factor. This separates what the correction guarantees from what the network's error distribution enables.
  - Cite two-grid theory for context, for example Falgout, Vassilevski & Zikatanov, NLAA (2005), or the XZ identity.
  - Reword the Appendix D sentence so it does not suggest a rate.
- **Fix type:** re-analysis of existing operators (new computation, but cheap), plus text.
- **Confidence:** High.

### R3-07. The correction and starting-field baselines are too weak to isolate the network's contribution

- **Severity:** Major
- **Location:** Section 6.5, Table 3, Table ST18 ("harmonic extension solves a graph Laplacian on the element connectivity weighted by material volume"; "lower than the harmonic start by factors of 7 to 290"; "The learned field therefore supplies the part of the interior equilibrium that the smoothing and the coarse space do not reach at a practical budget"); Section 5.2 and Appendix F.1 (geometric Q1 functions "restricted to the internal degrees of freedom").
- **Issue:** The claim that the network is needed rests on two comparisons, and each has a weak partner.
  1. **The starting field.** A scalar graph Laplacian is a weak starting field for 3D elasticity with thin walls.
  2. **The correction.** One geometric two-grid cycle with a trilinear coarse space has two known weaknesses for this geometry:
     - it is not topology-aware: one coarse function spans disconnected wall segments within its support, a known failure mode for perforated and thin structures;
     - it is truncated at the retained nodes, which produces a poor coarse approximation exactly in the cut-adjacent layer where Figure 7 shows B's error concentrates.

     The Q2(17) and Q1(33) variants in ST04 are finer geometric spaces of the same kind, not algebraic ones.

  Several obvious baselines are missing:
  - m repeated 8/Q1/8 cycles (still a fixed linear map);
  - one or more smoothed-aggregation AMG V-cycles with rigid-body near-nullspace as W;
  - an elasticity-based cheap start, for example the coarse Galerkin elasticity solve interpolated with the retained values prescribed, or an MsFEM-type extension;
  - a cost-matched comparison that expresses the network's forward plus transpose cost in stiffness-action equivalents (Section 7.4 says the network is cheaper than the 32 correction actions) and gives the harmonic start the same total budget.

  The 32/Q1/32 rows only increase smoothing, which is exactly the component known to stall on the low modes.
- **Why it matters:** The complementarity statements in the abstract ("7 to 290 times more accurate than a harmonic extension") and Section 7.3 depend on these baselines. If a standard AMG-based correction from a cheap start reached 0.1% at similar cost, the network's role would shrink substantially. The evidence is also limited to five cells and the untrained B.
- **Suggested fix:**
  - Add, on the five detailed cells: m-cycle two-grid, an SA-AMG V-cycle correction, and an elasticity-based start, each at matched cost measured in stiffness-action equivalents or wall time.
  - Report the network's cost in those units.
  - Qualify the abstract's "7 to 290" as five cells with the fixed-weight B.
- **Fix type:** new experiment (local, cheap), plus text.
- **Confidence:** Medium-high. The outcome is uncertain, but the comparison is necessary.

### R3-08. Offline cost, amortisation and scope of reuse are left out of the cost conclusions

- **Severity:** Major
- **Location:** Section 6.10, Section 7.4 and Section 8 ("cheaper for any number of queries"); Section 6.1 ("A3's continuation took 2.7 h on one NVIDIA GeForce RTX 5090").
- **Issue:** The comparison excludes several offline costs:
  - generating the reference data: exact condensed responses and eight-corner sensitivity labels for many directions on 305 or 591 training geometries, plus the adversarial searches that need S^-1 solves;
  - training B (40,000 steps, time not reported) and the continuation;
  - checkpoint selection.

  The trained operator is also tied to one discretisation: n = 32, E = 1, nu = 0.3, gamma = 1e-4, P-type level set and a single planar cut. Any change requires retraining, whereas the conventional route has no such restriction.
- **Why it matters:** A per-query advantage only matters beyond a break-even number of cell evaluations. CMAME readers will expect that number.
- **Suggested fix:**
  - Report the data-generation and training wall time and hardware for all stages.
  - Compute the break-even number of cell condensations or design iterations against the corrected same-hardware baseline (R3-01, R3-02).
  - State the reuse domain explicitly beside the cost claims.
- **Fix type:** re-analysis of existing records, plus text.
- **Confidence:** High.

### R3-09. The assembled solver is under-specified in the main text, and its scalability is untested

- **Severity:** Major
- **Location:** Section 6.9 and Section 6.10 ("the preconditioned iteration needs 183 applications per cell ..."); Table 6 and ST20c (iterations 114, 119, 129, 165; "preconditioner setup"); Supplementary Note S1.1; Tables ST09 and ST17b (27-cell D runs: learned 222 s against exact 94.6 s on the same GPU).
- **Issue:** The main text never names the preconditioner used for the A3 lattice solves. S1.1 suggests K_PP^-1 as fine action plus a q1r coarse space. That requires factorising the global retained block, which is itself a sparse direct solve that grows superlinearly with lattice size.

  Several trends are worrying:
  - iteration counts are high (114–165 to 1e-6) and grow from 4 to 8 cells;
  - the true-residual floor grows by 10x;
  - the only 27-cell evidence in the manuscript (predictor D) has the learned route 2.3x slower than exact condensation on the same GPU.

  Using K_PP (the retained block with the interior clamped) as fine preconditioner for S can be poorly spectrally equivalent for thin-walled cells. S1.1 itself notes that K_PP ⪰ K̂ is not guaranteed.
- **Why it matters:** The design-optimisation use case (8 or 27 cells, Section 6.11) and the claims in Section 7.4 depend on solver behaviour beyond 8 cells.
- **Suggested fix:**
  - Describe the A3 lattice preconditioner in the main text.
  - Report iteration counts, the condition estimate from the CG Lanczos coefficients, and setup cost for 4, 8 and 27 (or more) cells with A3.
  - Report how the K_PP factorisation scales.
  - Discuss the 27-cell D result explicitly rather than only in S4.
- **Fix type:** new experiment, plus text.
- **Confidence:** Medium.

### R3-10. Memory accounting may omit the stiffness matrix needed by the learned operator

- **Severity:** Minor
- **Location:** Table 5 memory column ("Memory: interior factor, or the learned operator's stored state"); Section 6.10 ("stores 4 to 14 times less"); ST08a (K storage 1.39–2.71 GiB, listed separately from the learned state of 0.27–1.37 GiB).
- **Issue:** Applying F^T K F needs K, either stored or regenerated by a fused matrix-free kernel. Applying the exact S needs K_IP and K_PP besides the factor. It is unclear whether the 0.25–1.37 GB includes K. ST08a lists K separately, and it is larger than the learned state. It is also unclear whether the host figure includes the off-diagonal blocks.
- **Why it matters:** It affects the memory ratio, which may be the method's most defensible advantage.
- **Suggested fix:** State exactly what each memory figure contains. Report totals including all matrices needed for an application, and recompute them with the Cholesky factor (R3-02).
- **Fix type:** text only, or re-analysis.
- **Confidence:** Medium.

### R3-11. Chebyshev interval: the estimator, the safeguard for unseen designs, and the choice a = b/30

- **Severity:** Minor
- **Location:** Section 5.1 and Appendix D ("40 power iterations ... factor 1.05"; "on all 80 validation geometries this endpoint exceeds the converged largest eigenvalue by 2.1–5.0%"); Section 6.2 ("verified per cell rather than guaranteed a priori").
- **Issue:** The verification is good practice. In optimisation, however, every iteration creates new geometries, and a power estimate falling below lambda_max would amplify the top modes. The ordering Ŝ_tg ⪯ Ŝ_0 would then fail, although Ŝ ⪰ S still holds.
  - A 10–20-step Lanczos estimate (as in hypre and ML) is cheaper and sharper than 40 power steps, and gives a Ritz residual to set a principled safety margin.
  - The ratio a = b/30 is not justified for this problem. Since much of M1's error lies below a (Section 6.4), a short sensitivity study of b/a (for example 10, 30 and 100) would be informative.
- **Suggested fix:**
  - Use Lanczos for the estimate, or add a runtime check (for example the Rayleigh quotient of the last iterate against b).
  - Report the sensitivity to b/a on the detailed cells.
  - Note explicitly that the SPD and lower-bound properties do not depend on contraction; only the ordering and accuracy do.
- **Fix type:** re-analysis or small experiment, plus text.
- **Confidence:** High.

### R3-12. The deployed coarse solve is not verified against the analysed one

- **Severity:** Minor
- **Location:** Appendix F.2 ("Recovery assumes the resulting colour separation resolves every interacting coarse pair"; "trying diagonal shifts in the order 0, 1e-12, ..., 1e-4. The selected value is stored"); Appendix F.1 compared with F.2 (explicit V^T A V with PARDISO in the fixed-weight study, probed and shifted Cholesky in training and deployment); Table 3 (B columns from the fixed-weight study, A3 column from deployment).
- **Issue:** Three points are unreported:
  - the error of the probed A_c against V^T A V;
  - which shifts were selected, and on how many geometries;
  - whether Table 3 compares B and A3 through the same implementation.

  Equation (F.1) guarantees energy decrease only when the probed matrix equals the Galerkin matrix. The fp32 casts of F.3 add further differences.
- **Suggested fix:**
  - Report ||A_c^probe - V^T A V|| / ||V^T A V|| and the selected shifts for the 80 geometries.
  - Evaluate B+W and A3 in Table 3 through the same implementation, or quantify the difference between the two implementations.
- **Fix type:** re-analysis.
- **Confidence:** Medium.

### R3-13. The learned application barely benefits from batching

- **Severity:** Minor
- **Location:** Table 5 (54 ms for 1 vector, 531 ms for 16, 923 ms for 64 on G1); Section 7.4 ("The learned applications are dominated by the 32 stiffness actions of the correction").
- **Issue:** Cost grows almost linearly with batch size. A memory-bound SpMM with K of about 1.4 GiB should amortise across columns. As a rough roofline, 34 K-actions at about 1.8 TB/s take about 25–30 ms regardless of batch width up to moderate sizes. This suggests an implementation limit (per-vector loops, or the network transpose), not an intrinsic cost. It matters in both directions: the learned route may be faster than reported, and the stated cause of the cost may be wrong.
- **Suggested fix:** Break an application down by component (network forward and transpose, smoothing SpMVs, coarse solves, K action) for batches of 1, 6 and 64, and compare it with a bandwidth estimate.
- **Fix type:** re-analysis or small experiment.
- **Confidence:** Medium.

### R3-14. Benchmark reporting is incomplete

- **Severity:** Minor
- **Location:** Section 6.10, Tables 5 and 6, Table ST20.
- **Issue:** The following are not reported:
  - CPU model, sockets, memory bandwidth and NUMA placement;
  - MKL, CUDA and PyTorch versions;
  - GPU clocks and power;
  - the number of repetitions and the spread for the host timings, which are single runs apart from a "within 10%" remark;
  - whether the three PCG loads are solved as a block;
  - whether warm-up or JIT is excluded consistently on both sides.

  No hardware-cost or energy normalisation is given for a 575 W flagship GPU against 16 cores.
- **Suggested fix:** Add a benchmark-environment table and report the median and range over at least three runs. Consider an energy-to-solution or cost-normalised comparison.
- **Fix type:** text, plus re-runs.
- **Confidence:** High.

### R3-15. Cost and complementarity claims in the abstract and conclusions need scoping

- **Severity:** Minor
- **Location:**
  - Abstract: "condenses a cell 9 to 29 times faster, stores 4 to 14 times less, and applies it 2 to 23 times faster";
  - Abstract: "the learned starting field is 7 to 290 times more accurate than a harmonic extension";
  - Section 7.4: "A direct solution of the whole lattice ... already takes more time and memory than the learned route at four cells";
  - Section 8: "so it is cheaper for any number of queries".
- **Issue:** These are stated without their conditions: host LU with 16 threads against a GPU; five cells with the fixed-weight B; a host front end; a shared host.
- **Suggested fix:** After R3-01 to R3-03, restate each claim with its hardware and baseline qualifiers, or remove it.
- **Fix type:** text only.
- **Confidence:** High.

---

## 4. Top three concerns

1. **The cost conclusions are not supported on equal hardware (R3-01, R3-02, R3-03).** The main-text baseline is a 16-thread host PARDISO LU of an SPD matrix, apparently with a sequential solve, a column-wise explicit S, a host-only front end, and (for Table 6) a shared host. The supplement's same-GPU exact-factor timings show exact condensation applying faster than the corrected learned operator. Under those timings, "cheaper for any number of queries" is reversed.
2. **The expected solver baselines are missing (R3-04, R3-07).** There is no AMG-PCG or BDDC/FETI-DP solution of the whole lattice, and no connection to the inexact-subdomain-solver DD literature. Locally, the network's contribution is measured against a scalar graph-harmonic start and a single geometric two-grid cycle, not against multi-cycle or AMG corrections at matched cost.
3. **Algebraic accuracy and correction guarantees are not established (R3-05, R3-06).** The true residual of the learned assembled solve stagnates at 3.6e-4 to 3.2e-3 while compliance errors are reported at the 1e-2 to 1e-5 percent level. The residual work and action–energy terms are not reported, the two-cell residuals were not saved, and Table 1 contradicts Appendix F.3 on precision. The two-grid analysis proves only non-expansion (rho_k ≈ 1), and no worst-case contraction factor is measured, although one would give an all-direction bound at little cost.
