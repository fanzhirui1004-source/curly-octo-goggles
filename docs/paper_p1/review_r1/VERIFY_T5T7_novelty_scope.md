# Stage 2b verification: T5 (novelty, attribution, PIML fairness) and T7 (scope, limitations, reproducibility)

**Verifier role:** adversarial. For each issue I first tried to refute the reviewers' concern from the manuscript, appendices, supplement, evidence and archived code.
**Issues:** I-09, I-14, I-15, I-35 (T5); I-05, I-32, I-40, I-44, I-49 (T7).
**Author decisions taken as given:**
- NICE is framed as the authors' own framework, not as an extension of PIML.
- Guo et al. (2026a) is cited.
- §6.11 is a placeholder.

**Scope of this check:**
- Nothing in the manuscript, evidence or code was edited.
- Line numbers refer to `docs/paper_p1/MANUSCRIPT_EN.md` (610 lines) at commit 693de8e.
- References were checked against Crossref records, and against publisher or proceedings pages through web search, on 2026-09-28. Anything I could not confirm is marked **UNVERIFIED**.

---

## Summary table

| Issue | Verdict | Action class | Priority |
|---|---|---|---|
| I-09 Closest prior work missing | CONFIRMED | (a) text (+ about 20 references) | High |
| I-14 "Compliance ≠ sensitivity" as the principal finding | PARTLY | (a) text; (b) optional re-analysis via I-08 | High |
| I-15 Classical results presented as contributions | CONFIRMED | (a) text | High |
| I-35 §6.8 is not a fair PIML proxy | CONFIRMED as a PIML proxy; REFUTED as "useless" | (a) reframe as an own-pipeline ablation + (d) rebuttal of the "make it fair" option | High (decision) |
| I-05 Scope; no Limitations; no OOD test | CONFIRMED (the in-house gyroid test is not in the manuscript) | (a) text; (c) optional (data already exist in-house) | High |
| I-32 τ → thickness/density map; admissible domain | CONFIRMED | (b) cheap re-analysis + (a) text | Medium |
| I-40 Connectivity / rank-six rigid modes | PARTLY (a check exists in code but is not stated) | (a) text | Low–Medium |
| I-44 Architecture reproducibility; availability | PARTLY (pool/refresh schedule is given; the rest is missing) | (a) text + supplement table | Medium |
| I-49 Interface matching with cut neighbours | CONFIRMED (the answer is in the code, not the paper) | (a) text | Low |

---

## T5. Novelty, attribution and PIML fairness

### I-09: The closest prior work is missing

**Verdict: CONFIRMED.**

Attempted refutation: the manuscript cites Smetana & Patera (2016) for port reduction and Hou & Wu (1997) for multiscale bases. It also cites HINTS, DeepONet preconditioning and GMT for hybrid solvers, and PIML, CNEE and Parish et al. for learned substructures. This does not answer the concern:
- **The second route is presented without prior work.** L13 says "Approximating the interior extension on the complete retained space … This work takes the second route". No prior work is attached to that route. SCRBE (full ports, reduced interior bubbles, a posteriori bounds) and inexact or approximate substructuring (approximate harmonic extensions in BDDC/FETI-DP) are exactly that route.
- **SCRBE is cited in the wrong place.** Smetana & Patera (2016) is itself an SCRBE-lineage paper, but it is cited only for the first route.
- **The reference list is short.** The manuscript has 17 references (18 bib entries, including the uncited Guyan). That is thin for a CMAME methods paper, and CMAME referees will notice.

**Missing works named by the reviewers, with verification status:**

| # | Reference (as verified) | Named by | Status |
|---|---|---|---|
| 1 | Huynh, D.B.P., Knezevic, D.J., Patera, A.T. A static condensation reduced basis element method: approximation and a posteriori error estimation. *ESAIM: M2AN* 47(1) (2013) 213–251. doi:10.1051/m2an/2012022 | R2, R7 | Verified (Crossref; online 2012) |
| 1b | Huynh, Knezevic, Patera. A static condensation reduced basis element method: complex problems. *CMAME* 259 (2013) 197–216. doi:10.1016/j.cma.2013.02.013 | (companion) | Verified |
| 2 | Eftang, J.L., Patera, A.T. Port reduction in parametrized component static condensation: approximation and a posteriori error estimation. *IJNME* 96(5) (2013) 269–302. doi:10.1002/nme.4543 | R7 | Verified |
| 3 | Ballani, J., Huynh, D.B.P., Knezevic, D.J., Nguyen, L., Patera, A.T. A component-based hybrid reduced basis/finite element method for solid mechanics with local nonlinearities. *CMAME* 329 (2018) 498–531. doi:10.1016/j.cma.2017.09.014 | R7 | Verified |
| 4 | Dohrmann, C.R. A preconditioner for substructuring based on constrained energy minimization. *SIAM J. Sci. Comput.* 25(1) (2003) 246–258. doi:10.1137/S1064827502412887 | R3 | Verified |
| 5 | Farhat, C., Lesoinne, M., LeTallec, P., Pierson, K., Rixen, D. FETI-DP: a dual–primal unified FETI method—part I. *IJNME* 50(7) (2001) 1523–1544. doi:10.1002/nme.76 | R3 | Verified |
| 6 | Li, J., Widlund, O.B. On the use of inexact subdomain solvers for BDDC algorithms. *CMAME* 196(8) (2007) 1415–1428. doi:10.1016/j.cma.2006.03.011 | R3, R7 | Verified |
| 7 | Klawonn, A., Rheinbach, O. Inexact FETI-DP methods. *IJNME* 69(2) (2007) 284–307. doi:10.1002/nme.1758 | R3, R7 | Verified (online 2006) |
| 8 | Toselli, A., Widlund, O. *Domain Decomposition Methods — Algorithms and Theory.* Springer (2005). doi:10.1007/b137868 | R1, R3 | Verified |
| 9 | Heinlein, A., Klawonn, A., Lanser, M., Weber, J. Combining machine learning and adaptive coarse spaces—a hybrid approach for robust FETI-DP methods in three dimensions. *SIAM J. Sci. Comput.* 43(5) (2021) S816–S838. doi:10.1137/20M1344913 | R2, R7 | Verified |
| 10 | Greenfeld, D., Galun, M., Basri, R., Yavneh, I., Kimmel, R. Learning to optimize multigrid PDE solvers. *Proc. ICML 2019*, PMLR 97, 2415–2423. | R2, R7 | Verified (PMLR page; no DOI) |
| 11 | Luz, I., Galun, M., Maron, H., Basri, R., Yavneh, I. Learning algebraic multigrid using graph neural networks. *Proc. ICML 2020*, PMLR 119, 6489–6499. | R2, R7 | Verified (PMLR page; no DOI) |
| 12 | Um, K., Brand, R., Fei, Y., Holl, P., Thuerey, N. Solver-in-the-loop: learning from differentiable physics to interact with iterative PDE-solvers. *NeurIPS 2020* (33). | R2, R7 | Verified (NeurIPS proceedings page; no DOI) |
| 13 | Li, Z., Huang, D.Z., Liu, B., Anandkumar, A. Fourier neural operator with learned deformations for PDEs on general geometries (Geo-FNO). *JMLR* 24 (2023) paper 388, 1–26. | R2 | Verified via web search; not Crossref-indexed |
| 14 | Li, Z., Kovachki, N., Choy, C., et al. Geometry-informed neural operator for large-scale 3D PDEs (GINO). *NeurIPS 2023* (36). | R2 | Verified (NeurIPS proceedings page) |
| 15 | Pfaff, T., Fortunato, M., Sanchez-Gonzalez, A., Battaglia, P. Learning mesh-based simulation with graph networks (MeshGraphNets). *ICLR 2021*. | R2 | Verified (ICLR/dblp) |
| 16 | He, J., Liu, X., Xu, J. MgNO: efficient parameterization of linear operators via multigrid. *ICLR 2024*. | R2 | Verified (ICLR proceedings, OpenReview) |
| 17 | Boullé, N., Townsend, A. Learning elliptic partial differential equations with randomized linear algebra. *Found. Comput. Math.* 23 (2023) 709–739. doi:10.1007/s10208-022-09556-w | R2 | Verified |
| 17b | Boullé, N., Earls, C.J., Townsend, A. Data-driven discovery of Green's functions with human-understandable deep learning. *Sci. Rep.* 12 (2022) 4824. doi:10.1038/s41598-022-08745-5 | R2 (alt.) | Verified (article number from memory; DOI verified) |
| 18 | E, W., Yu, B. The Deep Ritz method. *Commun. Math. Stat.* 6 (2018) 1–12. doi:10.1007/s40304-018-0127-z | R2 | Verified |
| 19 | Alexandersen, J., Lazarov, B.S. Topology optimisation of manufacturable microstructural details without length scale separation using a spectral coarse basis preconditioner. *CMAME* 290 (2015) 156–182. doi:10.1016/j.cma.2015.02.028 | R7 | Verified |
| 20 | Efendiev, Y., Galvis, J., Hou, T.Y. Generalized multiscale finite element methods (GMsFEM). *J. Comput. Phys.* 251 (2013) 116–135. doi:10.1016/j.jcp.2013.04.045 | R7 | Verified |
| 20b | Efendiev, Y., Hou, T.Y., Wu, X.-H. Convergence of a nonconforming multiscale finite element method (oversampling). *SIAM J. Numer. Anal.* 37(3) (2000) 888–910. doi:10.1137/S0036142997330329 | (classical oversampling) | Verified |
| 21 | Amir, O., Bendsøe, M.P., Sigmund, O. Approximate reanalysis in topology optimization. *IJNME* 78(12) (2009) 1474–1491. doi:10.1002/nme.2536 | R4, R7 | Verified (online 2008) |
| 22 | Amir, O., Stolpe, M., Sigmund, O. Efficient use of iterative solvers in nested topology optimization. *Struct. Multidiscip. Optim.* 42 (2010) 55–72. doi:10.1007/s00158-009-0463-4 | R4 | Verified (online 2009) |
| 23 | Gogu, C. Improving the efficiency of large scale topology optimization through on-the-fly reduced order model construction. *IJNME* 101(4) (2015) 281–304. doi:10.1002/nme.4797 | R4, R7 | Verified |
| 24 | Vaněk, P., Mandel, J., Brezina, M. Algebraic multigrid by smoothed aggregation for second and fourth order elliptic problems. *Computing* 56 (1996) 179–196. doi:10.1007/BF02238511 | R3 | Verified |
| 25 | Becker, R., Rannacher, R. An optimal control approach to a posteriori error estimation in finite element methods. *Acta Numerica* 10 (2001) 1–102. doi:10.1017/S0962492901000010 | R1, R7 | Verified |
| 26 | Oden, J.T., Prudhomme, S. Goal-oriented error estimation and adaptivity for the finite element method. *Comput. Math. Appl.* 41 (2001) 735–756. doi:10.1016/S0898-1221(00)00317-5 | R1, R7 | Verified |
| — | Learned CG preconditioners (R2, no specific work named) | R2 | Not a reference; Kopaničáková & Karniadakis (2025) is already cited |
| — | Hypernetworks (R2, no specific work named) | R2 | Not named; cite only if a specific work is chosen (e.g. Ha, Dai & Le, ICLR 2017, **UNVERIFIED** here) |

**Additional works the authors should consider** (not named by the reviewers; all verified):
- Dohrmann, C.R. An approximate BDDC preconditioner. *Numer. Linear Algebra Appl.* 14 (2007) 149–168, doi:10.1002/nla.514. This is the closest classical analogue of "approximate harmonic extension + multilevel solve inside substructuring".
- Babuška & Lipton, *Multiscale Model. Simul.* 9 (2011) 373–406, doi:10.1137/100791051.
- Ma, Scheichl & Dodwell, *SIAM J. Numer. Anal.* 60 (2022) 244–273, doi:10.1137/21M1406179 (MS-GFEM). These sit next to Smetana & Patera (2016) as "optimal local spaces".
- Bonilla Moreno, Guarino & Antolin, *CMAME* 463 (2027) 119304, doi:10.1016/j.cma.2026.119304 (ROM-BDDC for unfitted 2D lattices). This is the closest non-ML cut-lattice work; it is listed in `docs/PRIOR_ART_SCREEN_20260926_CN.md` but is not in the manuscript.
- O'Leary-Roseberry et al., Derivative-informed neural operator, *J. Comput. Phys.* 496 (2024) 112555, doi:10.1016/j.jcp.2023.112555. This is prior art for training with derivative labels (contribution ii).

**Preprint status (R5-17):**
- Guo et al. (2026b), PIML-OFEM, arXiv:2607.22019, was submitted on 24 Jul 2026 and is still a preprint.
- Jiang et al. (2026), arXiv:2608.02036, is still a preprint.
- Re-check both at submission.

**Action (a), text:**

1. **L13.** Replace the sentences from "Two approximations reduce this cost." through "This work takes the second route." with:

> Two approximations reduce this cost. Reducing the retained coordinates, as in port reduction for component-based static condensation [Eftang & Patera (2013)], [Smetana & Patera (2016)], changes the displacement patterns that substructures can exchange. Approximating the interior extension on the complete retained space leaves those patterns unchanged and alters only the interior response to each of them. This is the route of the static-condensation reduced basis element method, which replaces interior solves by parametric reduced bubble spaces with a posteriori error bounds [Huynh et al. (2013)], [Ballani et al. (2018)], and of inexact substructuring, which replaces exact discrete harmonic extensions by approximate ones [Dohrmann (2007)], [Li & Widlund (2007)], [Klawonn & Rheinbach (2007)]. This work follows the second route with a learned, geometry-conditioned interior extension.

2. **L15.** Add the "difference" sentence (R5-17) after "…[Xing et al. (2026)].":

> Unlike these solvers, which accelerate the solution of one global system, the correction here is a fixed, geometry-specific linear map inside the condensed energy, so that the corrected operator remains symmetric, bounded below by the exact Schur complement and consistent with its complete transpose.

3. **New paragraph** between L19 and L21 (about 220 words):

> Three further lines of work bear on this construction. First, the component reduced-basis methods named above build their interior approximation offline from snapshots of a prescribed parameter family and certify it a posteriori; NICE instead conditions one network on the discrete geometry of each cell, including its cut, and provides no certified bound, although Eq. (10) supplies a computable residual in the norm that governs the stiffness error. Second, learned components of iterative solvers, such as multigrid prolongations [Greenfeld et al. (2019)], [Luz et al. (2020)], coarse spaces for FETI-DP [Heinlein et al. (2021)] and networks trained through the solver they assist [Um et al. (2020)], accelerate the solution of a global system; here a network and a fixed correction define, per geometry, a variational approximation of the condensed operator that is itself assembled, and training through the correction transfers the solver-in-the-loop idea to that operator. Third, geometry-aware neural operators and graph networks [Li et al. (2023)], [Li et al. (2023b)], [Pfaff et al. (2021)], multigrid-structured linear operators [He et al. (2024)] and learned Green's functions [Boullé & Townsend (2023)] learn solution maps across geometries; the present extension is restricted to be exactly linear in the retained displacement so that its energy defines a stiffness. In topology optimisation, approximate reanalysis and inexact solves have shown that an accurate objective need not give accurate sensitivities [Amir et al. (2009)], [Amir et al. (2010)], [Gogu (2015)], and coarse multiscale bases with fine-scale correction have been used without scale separation [Alexandersen & Lazarov (2015)].

4. **§5.1 (L285).** Add after the Adams et al. citation: "…and the combination of a trial extension with smoothing is the principle of smoothed-aggregation prolongation [Vaněk et al. (1996)]."

---

### I-14: "Accurate compliance does not imply accurate sensitivity" framed as the principal finding

**Verdict: PARTLY.**

Attempted refutation:
- **Where the reviewers are too strong:**
  - The manuscript scopes the claim to *learned substructures* (L25), and it already calls Ritz orthogonality "classical" (L23, L564). The quantification (participation masking, J.3/J.4 bounds, δ²κ) is not in the goal-oriented literature.
  - R7-11's "limited practical weight" is partly wrong. The M1/x T-y case (target energy share 0.311, L458) is not a weak-participation case, and there C gives 11.4% sensitivity error.
  - A per-variable relative error matters for OC updates, as R4-03 itself notes, so the local metric is not meaningless.
- **Where the reviewers are right:**
  - *The principle is classical.* That an energy-norm-accurate approximation need not approximate a local functional is the basic premise of goal-oriented estimation (Becker & Rannacher 2001; Oden & Prudhomme 2001). It is also documented for reanalysis in topology optimisation (Amir et al. 2009, 2010; Gogu 2015). None of these is cited.
  - *"Principal mechanical finding" overclaims.* The phrase at L566 and contribution (iv) at L25 claim more than the evidence shows.
  - *The separation is shown only on predictors that are not NICE.* It appears for B, C and S8 (Table 4, Fig. 10). For A3 both errors are small.
  - *No gradient-level metric is reported.* The shared-variable gradient of L267 is never evaluated (see I-08).

**Action (a), text:**
- **Abstract, L5.** Replace "Accurate compliance therefore need not imply accurate local sensitivity: participation weighting masks the error of weakly participating cells." with:
  > As in goal-oriented error estimation, accurate compliance therefore need not imply accurate local sensitivity; for learned substructures, participation weighting masks the error of weakly participating cells, and we require both errors to be small.
- **Intro, L23.** After "…while its sensitivity remains inaccurate." insert "This is the distinction on which goal-oriented error estimation rests [Becker & Rannacher (2001)], [Oden & Prudhomme (2001)], and it has been observed for approximate reanalysis in topology optimisation [Amir et al. (2009)]; the relations here quantify it for learned substructures."
- **Contribution (iv), L25.** Fold it into the three-contribution list proposed under I-15, where it becomes the design criterion.
- **§8, L566.** Replace the first sentence with:
  > For learned substructures, these relations quantify a distinction familiar from goal-oriented error estimation and approximate reanalysis: accurate compliance does not imply accurate local thickness sensitivity. For an uncorrected learned extension they also explain why.

**Action (b), optional re-analysis:** report the assembled shared-variable gradient error and cosine for the pairs and lattices (R4-03, I-08; this is another theme's issue). If it stays small for C while the local error is large, say so. Contribution (ii) below then survives only as a design criterion.

---

### I-15: Classical and inherited results presented as contributions

**Verdict: CONFIRMED.**

Attempted refutation: L17 does concede that "the energy construction and rigid-motion constraints … are shared by the operator used here", and "Classical Ritz orthogonality" appears at L23 and L564. However:
- the Abstract (L5), contribution (i) (L25) and §8 (L564) still list SPSD, the rigid kernel and the Schur lower bound as properties of NICE;
- contribution (iii) reads as a new error analysis;
- Giles & Pierce (2000) is the only sensitivity reference;
- Guyan (1965) is in the .bib but not cited (R6-32).

**Claim-by-claim labelling:**

| # | Claim (location) | Label | Cite |
|---|---|---|---|
| 1 | Approximate S on the complete retained space, with no boundary reduction (Abstract; L13; L25 (i)) | **Classical route**; new in its learned, geometry-conditioned form | Huynh et al. 2013; Dohrmann 2007; Li & Widlund 2007 |
| 2 | Admissible extension: J_P Ê = I, Ê R_P = R (Eq. 5) | Classical/inherited | Guyan 1965; Hou & Wu 1997; Huang et al. 2023 (rigid constraints) |
| 3 | Ŝ = FᵀKF is SPSD with exactly the rigid kernel (Eq. 6; App. B) | Classical/inherited | Toselli & Widlund 2005; Fraeijs de Veubeke 1965 |
| 4 | Ritz identity Ŝ − S = HᵀAH ⪰ 0 (Eq. 9) | Classical | Toselli & Widlund 2005; Fraeijs de Veubeke 1965 (Guyan/Irons for condensation) |
| 5 | ε = residual in the A⁻¹-norm (Eq. 10) | Classical (error–residual equivalence) | Becker & Rannacher 2001 |
| 6 | "Complete transpose" (Eq. 7; App. E) | Classical construct (NᵀKN, adjoint of a fixed linear iteration); **new use**: needed so that a *corrected* learned extension stays variational | — |
| 7 | Compliance = total reconstructed error energy (Eq. 11; C.2–C.3) | Classical (energy orthogonality; compliance ordering) | Toselli & Widlund 2005; Bendsøe & Sigmund 2003 |
| 8 | Participation bound β/(1+ρ) ≤ e_C ≤ β/(1+β) (Eq. 12; J.3) | **New derivation** (elementary; spectral calculus + maximum principle) | Say "derived here"; no priority claim |
| 9 | Self-adjoint sensitivity s_c = −uᵀK,cu; S,c = EᵀK,cE (L240; H.1) | Classical | Haftka & Gürdal 1992; Bendsøe & Sigmund 2003 (replace or supplement Giles & Pierce) |
| 10 | Sensitivity error expansion (Eq. 13) and global-to-local bound (J.4) | Eq. 13 elementary; **J.4 new** | — |
| 11 | Complete surrogate derivative and O(t) cancellation (Eq. 14; J.5) | **New** | — |
| 12 | "Accurate compliance ≠ accurate sensitivity" (L23; (iv); L566) | Classical principle; **new quantification** for learned substructures | Becker & Rannacher 2001; Oden & Prudhomme 2001; Amir et al. 2009 |
| 13 | ε = δ²κ amplification (Eq. B.8; §6.4) | Elementary identity; **new diagnostic** and measurement in thin walls | — |
| 14 | Chebyshev semi-iteration and its energy bound (§5.1; D.1–D.4) | Classical | Golub & Varga 1961; Saad 2003; Adams et al. 2003 |
| 15 | Galerkin coarse energy decrease (Eq. 16) | Classical | Xu 1992 |
| 16 | Two-grid error operator and ordering S ⪯ Ŝ_tg ⪯ Ŝ_0 (Eq. 17) | Classical | Hackbusch 1985; Trottenberg et al. 2001; Xu & Zikatanov 2002 |
| 17 | Residual-corrected compliance functional (Eq. 18; J.6) | Classical | Becker & Rannacher 2001 |
| 18 | Learned initial field + relaxation/coarse correction (L15; §5) | **New combination** | Zhang et al. 2024; Xing et al. 2026; Dohrmann 2007 |
| 19 | Fixed geometry-specific correction composed *inside* the energy form with its transpose (§3.2; Alg. 1) | **Genuinely new** (as far as the verified literature shows) | — |
| 20 | Training through the correction (§3.3) | **New combination** (solver-in-the-loop applied to a condensation operator) | Um et al. 2020; Xing et al. 2026 (end-to-end V-cycle) |
| 21 | Sensitivity term in the objective (Eq. 8) | **New combination**; not ablated (I-34) | O'Leary-Roseberry et al. 2024 (derivative-informed training) |
| 22 | Geometry-conditioned, matrix-free, exactly linear-in-q extension on 1.7–2.6×10⁴ retained coordinates of cut CutFEM cells (§3.1) | **Genuinely new** | Contrast Huang et al. 2023/2024, Guo et al. 2026a, Jiang et al. 2026 |
| 23 | Cut-band retained space (§2.2) | **New modelling choice**; its necessity is questioned (R7-16/I-52) | — |
| 24 | Empirical study (population, pairs, lattices, ablations, cost) | **New** | — |

**Action (a), text:**

1. **Replace the contributions paragraph (L25)** with three contributions, following R5-03 and consistent with the NICE framing:

> The construction inherits from any admissible, rigid-motion-reproducing extension evaluated in the energy form, as in static condensation [Guyan (1965)], multiscale finite elements [Hou & Wu (1997)], component reduced-basis methods [Huynh et al. (2013)] and learned shape-function substructures [Huang et al. (2023)], a symmetric positive semidefinite condensed stiffness with the rigid kernel, bounded below by the Schur complement with a quadratic excess; the correction inherits the energy properties of Chebyshev smoothing and Galerkin coarse projection [Golub & Varga (1961)], [Xu (1992)], [Xu & Zikatanov (2002)]. The contributions of this work are threefold. (i) NICE: a geometry-conditioned, matrix-free extension, exactly linear in the retained displacement, on the complete retained space of cut cells in a stabilised cut finite element model, including a band of cut-element coefficients that carries cut-surface virtual work, combined with a fixed, geometry-specific multilevel correction composed inside the energy form with its complete transpose, and trained through that correction with an objective that includes thickness sensitivity. (ii) An application of classical Ritz and goal-oriented arguments to learned substructures, which yields a load-specific two-sided participation bound on the compliance error, a bound relating it to the local eight-corner sensitivity error and the amplification of displacement error into energy error in thin walls, and which makes the joint accuracy of compliance and local sensitivity a design criterion for learned substructures. (iii) Numerical evidence that NICE meets this criterion on 80 validation geometries, fourteen two-cell configurations and two eight-cell lattices, together with an ablation of the retained representation and a cost comparison with conventional condensation [placeholder: and a thickness design, Section 6.11].

2. **New opening paragraph of §4** ("Classical background", about 120 words):

> Several relations below are classical and are restated in the present notation: the energy minimality of the discrete harmonic extension and the Ritz identity (9) [Fraeijs de Veubeke (1965)], [Toselli & Widlund (2005)]; the residual form (10) and the residual-corrected functional (18) [Becker & Rannacher (2001)]; the energy orthogonality behind Eq. (11); the self-adjoint compliance sensitivity [Haftka & Gürdal (1992)], [Bendsøe & Sigmund (2003)]; and, in Section 5, Chebyshev semi-iteration [Golub & Varga (1961)] and the two-grid error operator [Hackbusch (1985)], [Xu & Zikatanov (2002)]. Derived here for learned substructures are the participation bounds of Eq. (12) and Appendix J.3, the sensitivity bound of Appendix J.4, the order of the complete surrogate derivative in Appendix J.5 and the identity ε = δ²κ of Eq. (B.8).

3. **Abstract, L5.** Replace "its energy in a stabilised three-dimensional cut finite element model defines a symmetric, positive semidefinite condensed stiffness, bounded below by the exact one and never formed" with:
   > its energy in a stabilised three-dimensional cut finite element model defines, as for any admissible extension, a symmetric, positive semidefinite condensed stiffness bounded below by the exact one; here it is applied without being formed

4. **§8, L564.** Replace "its complete transpose defines a variational condensed stiffness, bounded below by the exact Schur complement," with:
   > its complete transpose defines a condensed stiffness that inherits the variational properties of admissible extensions, including the Schur-complement lower bound,

5. **L240.** Cite Haftka & Gürdal (1992) or Bendsøe & Sigmund (2003) in place of, or in addition to, Giles & Pierce (2000). Cite Guyan (1965) and Irons (1965) at L13 ("Static condensation provides such a model…"). This resolves R6-32.

**Classical references to add (all verified):**
- Guyan, *AIAA J.* 3 (1965) 380, doi:10.2514/3.2874
- Irons, *AIAA J.* 3 (1965) 961–962, doi:10.2514/3.3027
- Craig & Bampton, *AIAA J.* 6 (1968) 1313–1319, doi:10.2514/3.4741
- Golub & Varga, *Numer. Math.* 3 (1961) 147–156 and 157–168, doi:10.1007/BF01386013 / BF01386014
- Saad, *Iterative Methods for Sparse Linear Systems*, 2nd ed., SIAM (2003), doi:10.1137/1.9780898718003
- Hackbusch, *Multi-Grid Methods and Applications*, Springer (1985), doi:10.1007/978-3-662-02427-0
- Xu & Zikatanov, *J. Amer. Math. Soc.* 15 (2002) 573–597, doi:10.1090/S0894-0347-02-00398-3
- Haftka & Gürdal, *Elements of Structural Optimization*, Kluwer (1992), doi:10.1007/978-94-011-2550-5
- Fraeijs de Veubeke, "Displacement and equilibrium models in the finite element method", in Zienkiewicz & Holister (eds), *Stress Analysis*, Wiley (1965). The 1980 memorial-volume reprint, doi:10.1007/978-94-009-9147-7_3, is verified; the original chapter pages are **UNVERIFIED**.
- Trottenberg, Oosterlee & Schüller, *Multigrid*, Academic Press (2001): **UNVERIFIED** here (not in Crossref), but a standard book.
- Bendsøe & Sigmund, *Topology Optimization*, Springer (2003): **UNVERIFIED** here (search returned wrong hits); the usual DOI is 10.1007/978-3-662-05086-6, which the authors should confirm.
- Smith, Bjørstad & Gropp (1996), Cambridge UP: **UNVERIFIED** here.

---

### I-35: §6.8 is not a fair proxy for PIML-type substructures (R5 vs R7)

**Verdict: CONFIRMED as a characterisation of PIML; REFUTED as a claim that the experiment is worthless.**

Attempted refutation of R7-03, bias by bias:
- **Bias 3 (adversarial loads) is overstated.** Target-face loads alone already give 77–82% compliance error at r = 1 (ST07a, "Target-face compliance" column). The headline 78–85% does not come from neighbour-face loads.
- **Bias 4 (sensitivity metric) does not touch the Intro claim.** The Intro number is a compliance error.
- **Bias 1 (scale frozen at one cell), bias 2 (no oversampling) and bias 5 (unmatched cost) are valid.** At r = 1 on U1 there are 24 coordinates against 28,206. At r = 8 the errors are 0.49–0.74% with about 14k coordinates on H1 (ST07a). PIML controls this error through partition refinement, Bézier enrichment (Guo 2026a) and oversampling (Guo 2026b). The PIML-OFEM abstract itself states that PIML "can be limited by prescribed boundary displacement interpolation", which supports the *concept* the ablation isolates, but not the 78–85% as a statement about PIML.
- **The restricted model is not PIML-like.** In ST07a, M1 at r = 1 still controls 15,423 DOFs, because the cut band is unrestricted.
- **The attribution cannot be confirmed.** "Degree one … corresponds to the corner-linear boundary interpolation of the three-dimensional PIML examples" (L474, L17, L534) cites Huang et al. (2023). I could not access that paper (closed; no abstract in Crossref or Semantic Scholar), so this is **UNVERIFIED**. The in-house prior-art notes attribute the 24-corner-DOF extension to the 2024 JMPS paper (`docs/data/newmachine_20260924/prior_art_0926/NOTES_cluster_TPMS_lattice_ML.md`), not to the 2023 EML paper.
- **"Physics-informed" is wrong.** Appendix I (APPENDICES_EN.md:553) applies it to Huang et al. (2023), which is supervised; the data-free variant is Huang et al. (2024).

**Recommendation: reframe §6.8 as an ablation of the authors' own pipeline, and keep a trimmed version (about 200 words) in the main text.** Put the degree sweeps in ST07. Figure 11 can stay (R5) or shrink to panels (b,c). Reasons:
1. The DECIDED own-framework positioning removes the need to characterise PIML at all. What NICE must justify is its own design choice (the complete retained space), and this experiment is the only direct evidence for it. Moving it entirely to the supplement (R7 option a) leaves that choice unsupported, which R5's storyline needs.
2. As an ablation ("same exact interiors, restricted retained space"), all five R7 biases stop being biases. They become the stated conditions of the ablation: one cell, fine-scale tractions, no oversampling, accuracy not cost.
3. The interface-only variant is informative and fair in both directions. At r = 3 compliance is already 0.17–0.44%, while neighbour-load sensitivity stays at 8–21% (ST07b). This should be reported, because it shows a restricted interface is adequate for compliance.
4. R7 option (c), a fair PIML re-implementation, is not recommended. See the rebuttal paragraph below.

**Replacement wording:**

- **Intro, L17.** Replace the text from "A reduced boundary description also introduces a model error…" to the end of the paragraph with:
  > A reduced boundary description also introduces a model error that does not depend on the accuracy of the learned interior. This line of work controls it by refining the partition, enriching the boundary interpolation [Guo et al. (2026a)] or oversampling overlapping local bases [Guo et al. (2026b)], in exchange for coarse models small enough for very large structures. The present framework makes the opposite trade: it keeps the complete retained space, so that its approximation error lies in the interior extension, where the network and the correction can reduce it, at the price of a coarse model with tens of thousands of coordinates per cell. Section 6.8 isolates, within our own pipeline, what restricting the box-face displacements of single cut cells would cost in compliance and local sensitivity.

  Also delete, or verify with page and figure references, the sentence "the three-dimensional examples use a corner-based linear interpolation [Huang et al. (2023)]" (R7-04).

- **§6.8 heading:** "6.8. Ablation: restricting the retained box-face representation".

- **§6.8 first paragraph (L474).** Replace with:
  > To isolate the role of the complete retained space, pairs in configuration x are solved with exact cell operators while the box-face displacements of each cell are restricted to tensor Bernstein polynomials of degree r; the non-box cut-band coordinates remain unrestricted. This is an ablation of the retained representation at a fixed substructure size of one cell, under the fine-scale face tractions used throughout and without oversampling. It is not a model of reduced-boundary substructure methods, which control this error by partition refinement, boundary enrichment or oversampling [Guo et al. (2026a)], [Guo et al. (2026b)], [Efendiev et al. (2013)], and it compares accuracy rather than accuracy at equal cost: at r = 1 the restricted U1 pair controls 24 coordinates against 28,206, and at r = 8 the H1 pair controls 14,001 of 32,991 (Table ST07). Two variants are examined: every box face restricted, and only the shared interface restricted.

- **§6.8, L476 last two sentences.** Replace "The separation between global and local accuracy thus persists, and is stronger, when the approximation enters through the retained space; the full retained representation used here avoids it by construction." with:
  > Restricting only the shared interface is thus adequate for compliance at low degree, but not for the local sensitivity under neighbour loads. Keeping the complete retained space removes this component of the error, at the cost of the larger coarse model quantified above.

- **§7.1, L534.** Replace "This comparison also clarifies … independent of how accurately the interior is learned." with:
  > Substructure methods that describe the boundary by a few coordinates [Huang et al. (2023)], [Guo et al. (2026a)] accept this boundary error, which they control by partition refinement, boundary enrichment or oversampling [Guo et al. (2026b)], in exchange for coarse models small enough for 10⁴–10⁶ substructures.

  Keep the next two sentences. Change "(Sections 6.6 and 6.8)" to "(Sections 6.6 and 6.8, for single cut cells at fixed size)".

- **Appendix I, APPENDICES_EN.md:553.** Replace "physics-informed local shape-function methods [Huang et al. (2023)]" with "learned shape-function substructures [Huang et al. (2023)], [Huang et al. (2024)]".

- **Minor.** The supplement cites data files `evidence/piml4_*.json`. The file names can stay, but the ST07 caption should not call the model "PIML".

**(d) Rebuttal paragraph** for R7's option (c):

> We agree that Section 6.8 cannot characterise PIML-type substructures, and we have removed that attribution and the Introduction's quantitative statement. We now present the comparison as an ablation of our own retained representation, with exact interiors, and state its conditions: one cell per substructure, fine-scale tractions, no oversampling, accuracy not cost. We have not re-implemented PIML with partition refinement, Bézier enrichment and overlapping bases on cut TPMS cells. That would be our implementation of another group's method, outside the method's intended voxel-density setting, and it could not be made fair to either side without the original code and training data. The paper makes no claim about the accuracy of PIML. Its claim is that, for the cut cells and responses studied, removing boundary reduction places the entire error in the interior, where the correction budget controls it.

---

## T7. Scope, limitations and reproducibility

### I-05: Scope too general; no Limitations section; no OOD test

**Verdict: CONFIRMED** for the scope wording and the missing Limitations section.

Attempted refutation: the facts are in the manuscript, but they are scattered:
- P-type level set: Eq. (1), L33.
- Single planar cut, with multiple cuts and curved boundaries excluded: L357.
- n, E, ν, γ: Table 1.
- About 1% reference error: L371.
- Development cells: L357.

However, the title (L1), Abstract (L5) and keywords say "TPMS" without qualification, and §7 has no limitations subsection.

**New finding the reviewers did not have.** An out-of-distribution test already exists in-house but is not in the manuscript (`docs/GCELL_ZERO_SHOT_20260928_CN.md`, data in `docs/data/newmachine_20260924/gcell/`):
- **Zero-shot.** A3 applied to 16 gyroid twins of P validation cells gives class-mean energy excess of 3.0–7.4% for the loaded classes, against 0.03–0.07% for the P twins. The glat222 lattice gives compliance 0.98–2.14% and eight-corner sensitivity 5.6–13.5%, which fails the 3% reference.
- **After fine-tuning.** Fine-tuning on 18 gyroid cells for 8000 steps (A3G) gives 0.23% / 1.84% under consistent loads, and 3.57% sensitivity under random loads.
- **Relevance.** This answers R2-07, R4-08 and R7-14 directly: graceful degradation, rescue by retraining, and retraining cost.
- **Caveats.** The gyroid twins keep the canonical cut family. The result is a night-run with a patched generator, so its evidence would need archiving before it goes into the paper. Whether to include it is an author decision.

**Action (a):**
- **Title and Abstract.** Write "Schwarz-P-type" in the first Abstract sentence ("For cut thin-walled Schwarz-P-type triply periodic minimal surface (TPMS) cells…"). Add it to the title as well if the title is revised (I-45).
- **New §7.5 "Limitations and extensions"** (239 words; draws only on content already in the manuscript):

> The trained operator applies to the setting in which it was trained and tested: Schwarz-P-type cells defined by Eq. (1), eight trilinear corner parameters in [0.175, 0.699] with corner span and gradient norm at most 0.47 (Table ST11), at most one planar cut whose normal is a cube-symmetry image of (cos ϑ, sin ϑ, 0), and one discretisation (n = 32, E_Y = 1, ν = 0.3, γ = 10⁻⁴), to which the 65/33/17/9 latent hierarchy is tied. Other TPMS families, cells with two or three cuts, curved boundaries, other resolutions or other materials require new training data and training. All reported tests lie inside this domain, and the network is not equivariant under the cube symmetries. The variational properties of Section 3.2 hold for any input, but the accuracy does not: no certified error bound is provided, although the residual of Eq. (10) can be computed at run time. Accuracy is measured against the discrete reference, whose discretisation error for heavily cut cells at n = 32 is about 1% (Section 6.2). The assembled evidence comprises fourteen two-cell configurations built from development cells and two eight-cell lattices. The reported sensitivities are field-based and omit the design dependence of the extension in Eq. (14). The cost comparison sets one GPU against host cores and excludes data generation and training. Finally, Section 2.2 assumes a connected cell with six rigid modes, and an optimiser must keep designs inside the parameter domain above.

- **Optional sentence** if the gyroid result is added, as a short supplement note plus one sentence here:
  > Applied without retraining to gyroid-type twins of the validation cells, the P-trained operator's energy error rises by one to two orders of magnitude and a gyroid lattice fails the 3% sensitivity reference; fine-tuning on 18 gyroid cells restores it under consistent loads (Supplementary Note S8).
- **Placement.** State the reuse domain next to the cost claims (L505) as well; this links to I-01/I-02.

**Action (c), optional:** use the existing gyroid data, and optionally add one oblique-normal and one doubly cut cell. Evaluate A3 in view 17 (R1-14). ST01c has view 17 only for B.

---

### I-32: No map from τ to wall thickness and relative density; admissible design domain unstated

**Verdict: CONFIRMED.**

Attempted refutation: §2.1 (L44) states that τ is "not pointwise physical wall thickness", which is honest. But no map is given, and "thickness sensitivity" is used throughout.

**Indicative scratch estimate** (uncut cell, uniform τ, 160³ sampling of |φ| ≤ τ; not the authors' integrator):

| τ | Relative density | Mean wall ≈ ρ/A (in h = 1/32) | Thinnest wall ≈ 2τ/max‖∇φ‖ on φ = 0 (in h) |
|---|---|---|---|
| 0.175 (lower bound) | 0.100 | ≈ 1.4 h | ≈ 1.1 h |
| 0.30 | 0.171 | ≈ 2.5 h | ≈ 1.9 h |
| 0.56 (lattice max) | 0.320 | ≈ 4.6 h | ≈ 3.5 h |
| 0.699 (upper bound) | 0.400 | ≈ 5.8 h | ≈ 4.3 h |

- The densities agree with ST11's "centre volume fraction stratified between 0.1 and 0.4", a useful cross-check.
- The thinnest walls at the lower bound are about one background element thick, which confirms R4's estimate.
- A is the estimated surface area, about 2.2–2.3; the mean-wall column is accurate to about ±5%.

**Action (b):** tabulate τ against relative density and against minimum and mean wall thickness (in h and in cell size), uncut and for the M/H strata, with the authors' moment integrator. Run the reference convergence at τ = 0.175 (links I-24).

**Action (a):**
- **After L44** add:
  > For uniform τ in an uncut cell, the relative density of the band grows from about 0.10 at the lower bound τ = 0.175 to 0.40 at the upper bound τ = 0.70, and the wall thickness near the thinnest point is about 2τ/|∇φ|, i.e. from about one to four background elements (Table S…). We call τ_c corner thickness parameters and derivatives with respect to them thickness sensitivities.
- **§6.11 placeholder (L524).** State the bounds [0.175, 0.699] and the span/gradient constraints (≤ 0.47) the optimiser imposes, or report how far the iterates leave the domain (M4 in CONSOLIDATED). Replace "corner thicknesses" with "corner thickness parameters".

---

### I-40: Connectivity and the rank-six rigid-mode assumption

**Verdict: PARTLY.**

The generator does enforce a check, but the paper never states it.

**Evidence** (archived source snapshot, not in the paper):
- **Face-connectivity check.** `docs/data/newmachine_20260924/src_v2_wip/fast_prep4.py:59–69` (the "Step 0 (operator learning)" pipeline) computes the face-connected components of the active background elements and raises `ACTIVE_CELLS_NOT_FACE_CONNECTED` unless there is exactly one.
- **Failed draws are redrawn.** `gen_new.py:250–259` (`fix_neighbour`, "topology not certified (fast_prep4)") replaces draws that fail it.
- **Port check.** `comp_check.py` checks that every component carries box nodes, since a component without ports would make K_II singular.
- **Numerical confirmation.** The rigid-body energy of the deployed operator is at most 2×10⁻¹¹ (L373; ST19).

**Caveats:**
- The docstring says "no global connectedness proof, by the user's decision; the face adjacency of the active cells is checked instead". The check therefore guarantees *algebraic* connectivity of the stabilised model (continuous Q2 plus ghost penalty), not physical connectivity of the material. Material that touches the rest only through partially filled elements is coupled through shared basis functions and the ghost penalty.
- The authors must confirm that this pipeline generated all 591/305/148/80 geometries and the lattice cells, and that H2 passed it. Fig. 1d looks like an isolated ring.

**Action (a).** In §2.2 after L68 add:

> Geometry generation enforces this assumption: a cell is accepted only if its active background elements form a single face-connected set that carries retained box-face coordinates; draws that fail are replaced. With continuous Q2 functions and the ghost penalty, K then has exactly the six rigid-body modes, and the deployed operator reproduces them to a relative energy of 2×10⁻¹¹ (Section 6.2). The check establishes connectivity of the stabilised discrete model; material connected only through partially filled elements is coupled through their shared basis functions and the ghost penalty.

For §6.11, add one sentence. For uncut cells, every positive τ keeps the band around the connected P surface connected. For cut cells, the check is repeated at every design iteration.

---

### I-44: Architecture reproducibility; code and data availability

**Verdict: PARTLY.**

- **Already specified** (refuting part of R2-10):
  - pool sampling: three geometries, one replaced every 100 steps, a new geometry every step (APPENDICES_EN.md:456);
  - adversarial refresh: eight candidates, four block iterations on loading a geometry (:462);
  - Adam, one-cycle schedule, EMA 0.9997 (:458);
  - class weights (:456).
- **Missing, confirmed:**
  - the numerical values of a_max per coefficient group ("fixed model-state arrays", :414);
  - initialisation of λ_mix, σ_ℓ and the weights;
  - whether k_b = 0.5 and the disabled feature extension also apply to A2b/A3 (:343 and :414 name only B, C, S8);
  - the source and dimension of the "eight-dimensional slot embedding" (learned table?);
  - the per-block parameter breakdown of the 603,464 parameters;
  - there is no availability statement at all.

**Action (a), supplement table:** a hyperparameter table with:
- all a_max values by coefficient type, layer and role;
- k_b per arm;
- initialisations;
- slot-embedding definition;
- per-block parameter counts (encoders, heads, local layers, U-hierarchy, weak-region block, in/out maps);
- per-arm feature flags.

Also add a pseudo-code listing of one forward pass with tensor shapes; G.1 already gives most shapes (APPENDICES_EN.md:339).

**What is realistic to release.** The repository holds partial source snapshots only (`docs/data/newmachine_20260924/src_s0`, `src_v2_wip`):
- they contain hard-coded server paths (e.g. `/root/autodl-tmp/...` in `fast_prep4.py`);
- they depend on cuDSS/nvmath and MKL PARDISO;
- the full pipeline, checkpoints (`/root/autodl-tmp/OPL/S1/V2/A3_2grid/best.pt`) and label banks live on the training server.

A frozen, cleaned snapshot is realistic. Releasing the full label banks is probably not, because of their size. Proposed statement (place before the References):

> **Code and data availability.** The implementation of the neural extension, the equilibrium correction and its transpose, the cut-cell preprocessing and reference condensation, and the assembly, benchmark and figure scripts, together with the trained weights of predictors B, C, S8, A2b and A3, the parameter files of all training, validation and lattice geometries, and the result records from which every table and figure is generated, will be archived at [Zenodo DOI] upon acceptance. Training labels can be regenerated from the geometry parameters with the archived reference pipeline; the complete label banks are available from the corresponding author on reasonable request.

If the authors decide not to release code, the minimum is weights + geometry parameters + evidence records + figure scripts. CMAME requires a data-availability statement in any case.

---

### I-49: How cell interfaces are matched with cut neighbours

**Verdict: CONFIRMED.**

The paper does not state the matching rule. The code answers the question:
- **Matching rule** (`docs/data/newmachine_20260924/src_v2_wip/lattice3.py:102–114`):
  - box-face coordinates are identified by global background-grid position and component;
  - cut-band coordinates are private to each cell;
  - a face node present on one side only simply remains a coordinate of that cell, uncoupled to the neighbour;
  - the exact and learned assemblies use the same index maps, so B_m is identical in both.
- **Why patches usually match.** The trilinear τ on a shared face depends only on the four shared corner values, and φ is 1-periodic. Material patches therefore coincide on any shared face not intersected by a cut.
- **The two-cell neighbours are uncut.** The two-cell neighbours are uncut ("FULL") continuous-thickness cells: `gen_new.py:262–290`, `kind='FULL'`, and e.g. `fresh_val_2005_d1_v0_nbmx` in `evidence/gate_A3_2grid_fresh_val_2005_d1_v0_x.json:188`.
- **Consequence.** The neighbour is not cut by the target's plane. For config x the neighbour lies wholly on the retained side only if b ≥ sin ϑ. Otherwise the pair is a diagnostic assembly, not a physically cut specimen, and the neighbour may carry unmatched face nodes where the target was cut away.
- **Face-match check.** `face_match.py` exists to count unmatched shared-face nodes. For the gyroid lattice generator, the note reports ports matching exactly except on edges.

**Action (a):**
- **§2.3, after L73:**
  > Box-face coordinates of neighbouring cells are identified by their background-grid position. Because the thickness field on a shared face depends only on its four corner values and φ is periodic, the material patches of two neighbours coincide on every face not intersected by a cut; where a cut removes material from one side only, the unmatched face coordinates remain coordinates of the cell that carries them. The reference and learned assemblies use the same maps B_m.
- **§6.1, L361:**
  > The neighbour is an uncut cell whose thickness is continuous across the interface; where the target's cut plane meets the shared face, the pair is therefore a diagnostic assembly rather than a physically cut specimen.
- **§6.9.** Add one sentence confirming that the lattice cells share one cut plane, with the `face_match.py` counts. The authors should verify this; I did not check the lattice layouts.

---

## Top three points the authors must decide

1. **§6.8 and the PIML sentence in the Introduction (I-35).**
   - Adopt the "own-pipeline ablation" reframing.
   - Drop the 78–85% figure and the PIML attribution from the Introduction; verify or delete the "corner-linear 3D PIML examples" claim, which is UNVERIFIED against Huang et al. (2023).
   - Keep a trimmed §6.8 in the main text rather than moving it wholly to the supplement.
   - Decline R7's "fair PIML re-implementation" with the rebuttal above.
2. **Contribution structure and headline (I-14 + I-15).**
   - Rewrite the contributions as "inherited (cited)" plus three new contributions.
   - Demote "accurate compliance ≠ accurate sensitivity" from principal finding to a design criterion, with goal-oriented and TO-reanalysis citations.
   - Add about 20 verified references (I-09), including SCRBE and inexact BDDC/FETI-DP as prior work on the "complete trace + approximate interior" route.
   - This changes the Abstract, L25 and §8.
3. **Scope and OOD evidence (I-05, with I-44).**
   - Put "Schwarz-P-type" in the Abstract and title, and add the §7.5 Limitations text.
   - Decide whether to include the existing gyroid zero-shot/fine-tune result. It directly answers three reviewers, but shows a zero-shot sensitivity failure and needs its evidence archived.
   - Decide the release scope for the code and data statement.

---
**Coordinator note:** the gyroid (G-cell) zero-shot and fine-tune results are, by standing author decision, reserved for the next paper and are NOT to enter P1. For I-05 the scope is therefore handled by "Schwarz-P-type" wording and the Limitations subsection (other TPMS families not tested), without citing the in-house G-cell data.
