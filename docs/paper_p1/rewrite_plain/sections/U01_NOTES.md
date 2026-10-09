# U01 notes: Section 1.1 Related work

Source: MANUSCRIPT_EN.md / MANUSCRIPT_CN.md lines 40-51 (old 1.1), plus the items moved out of the old introduction (old lines 23, 25, 27 and 29; see the remark in (d)).

New paragraphs (EN and CN aligned): P1 PIML and the trade-off of few boundary DOFs; P2 component models with a mechanical structure, the shared minimum-energy property and the three respects of comparison; P3 what a component exchanges (component reduced-basis methods); P4 how the interior is represented; P5 where the approximation is improved; P6 learning in solvers and neural operators; P7 static condensation as a two-level method.

Length (EN words, citations excluded): old 1.1 = 967; moved from the old introduction = 269 (line 23: 92, line 25: 30, line 27: 12, line 29: 135); total source = 1236. New 1.1 = 1292. The new text is about 5% longer than its sources because long sentences were split and two items are stated in plain words where they are used (the minimum-energy property in P2, the meaning of \(E\) in P7). No cited work was dropped.

## (a) Sentences and numbers removed or moved

### Old 1.1, paragraph 1 (old line 42)

| Old sentence | Where it is now |
|---|---|
| "Learned substructures have been developed furthest as PIML, which predicts ... \(N^TKN\)" [Huang 2022, 2023] | P1 S1-S2 (split) |
| "Its variants train without labelled shape functions ... [Huang 2024], enrich the boundary with Bézier functions ... [Guo 2026a], join oversampled bases ... [Guo 2026b], impose symmetry equivariance [Jiang C], treat isoparametric substructures [Zhang L], transfer them ... [Zhang 2026] and optimise lattices [Xu 2025]" | split: P1 S3 (Huang 2024, Jiang C, Zhang L, Zhang 2026, Xu 2025); P1 S4 (Guo 2026b); P1 S10-S11 (Guo 2026a; the citation stands at the end of S11, after the finding it supports, as in the source). The Guo 2026a comparison was moved after the trade-off sentences so that it leads directly into "NICE addresses both difficulties". It stays verbal: no numbers, no numerical comparison with PIML. |
| "A few boundary degrees of freedom per substructure give coarse models small enough ..., with the boundary model error controlled by refining the partition, enriching the interpolation or oversampling." | P1 S5-S6 |
| "NICE sits at the other end of this trade-off: it keeps every retained DOF ..., and it learns the interior extension ... rather than shape functions for a fixed set of boundary DOFs." | P1 S7-S9 |
| "NICE addresses both obstacles of retaining every boundary node." | P1 S12 |
| "The interior of a thin-walled cell is several times larger than its retained set (the cell of Figure 1b retains 9,674 of its 72,631 nodes) ...; and the size of the network does not grow ..., while the equilibrium correction reduces the error that the network leaves (Section 5.5)." | P1 S13-S14; numbers 9,674 and 72,631 kept; "equilibrium correction" written as "two-grid correction" |

### Old 1.1, paragraph 2 (old line 44)

| Old sentence | Where it is now |
|---|---|
| "Closest to the present construction are component models that keep a mechanical structure." | P2 S1 |
| "They differ from NICE in the retained space, in how the component is represented and in where its approximation is improved." | merged with old line 29 S1 into P2 S6 ("differ in three respects") |
| Component reduced-basis sentence [Huynh 2013], [McBane & Choi 2021], [Chasapi 2023] | P3 S1-S4 |
| "NICE keeps the complete retained space although that space itself changes with the cut, and therefore conditions its network on the discrete geometry of each cell ..." | P3 S5-S6 |

### Old 1.1, paragraph 3 (old line 46)

All sentences kept in P4, split into shorter sentences; all seven citations kept. Last sentence ("In NICE the learned object is the interior extension itself: one corrected extension ... defines the assembled operator, the recovered field and the thickness sensitivities") is P4 S9-S10, with "interior extension" written as "displacement recovery" and "assembled operator" as "assembled condensed stiffness". "Schur-complement targets" is written as "from targets computed with the exact condensed stiffness" (the words "Schur complement" are left to Section 2.2, where U02 says it once).

### Old 1.1, paragraph 4 (old line 48)

All sentences kept in P5. "NICE provides a computable residual (Eq. (5)) instead of a certified bound" is P5 S5, with Eq. (5) renumbered to Eq. (7). The last sentence is P5 S6.

### Old 1.1, paragraph 5 (old line 50)

| Old sentence | Where it is now |
|---|---|
| "Learning has also been combined with multilevel and domain-decomposition solvers, through ... [Greenfeld], [Luz], [Heinlein], [Zhang E], [Kopaničáková], [Xing] and [Um]." | P6 S1-S3 (split; hybrid neural solvers in their own sentence) |
| "Geometry-aware neural operators ... and Ritz-type training [E & Yu 2018] instead learn solution maps of the whole problem." | P6 S5; "Ritz-type training" written as "training by energy minimisation" |
| "Spectral multiscale coarse bases [Efendiev] have likewise served ... [Alexandersen & Lazarov]." | P6 S4 (placed before the neural operators, next to the solver items) |
| "These methods accelerate a global solve ...; here the learned map is a per-cell extension whose energy defines a condensed operator that can be assembled, differentiated and improved." | P6 S8-S9 |
| "The corrected extension \(F\) is in effect a geometry-conditioned prolongation from the retained to the active DOFs, with \(\widehat S=F^TKF\) as its Galerkin operator; learned multigrid prolongations [Greenfeld], [Luz] are likewise nonlinear in the operator and linear in the vector they act on, but serve as part of a solver, whereas \(F\) is a component model: ..." | P7 S4-S5 and S7-S9; "in effect" deleted (banned); "Galerkin operator" written as "coarse-grid matrix" (brief 3.1); the dependence on the geometry is stated as "nonlinear in the cell geometry" |

### Items moved in from the old introduction

| Old text | Where it is now |
|---|---|
| Line 23: "Static condensation is itself a two-level method: the exact extension \(E\) is the ideal interpolation of algebraic multigrid ..., and \(S=E^TKE\) is its Galerkin coarse operator [Falgout & Vassilevski (2004)]. NICE replaces this interpolation by a learned and smoothed one, and condenses with the Galerkin operator of that interpolation." | P7 S1-S5 ("Galerkin coarse operator" written as "coarse-grid matrix") |
| Line 23: "The same Ritz property underlies static condensation itself [Toselli & Widlund (2005)], multiscale finite elements [Hou & Wu (1997)], component reduced-basis methods [Huynh et al. (2013)] and learned shape functions evaluated as \(N^TKN\) [Huang et al. (2023)]; here it carries a learned extension on a retained space that changes with the cut." | P2 S4-S5. The property is stated in plain words in P2 S2-S3, with the conditions of Proposition 1 / assumption (A2): an approximate displacement field "that is linear in the retained displacements and reproduces them and rigid-body motions"; "by the principle of minimum potential energy, this condensed stiffness is never softer than the exact one, and its error is quadratic in the interior displacement error (Section 3.1)". The word "Ritz" is not used. |
| Line 23, rest (division of tasks, Ritz approximation, properties of the energy form) | not in this unit: the banned triad is deleted; the properties are in the approved introduction (paragraph 5) and new Section 3.1 |
| Line 25: "And a solution map that is nonlinear in its input, as most neural operators learn, gives up superposition and with it the quadratic condensed energy on which the structure rests." | P6 S6-S7 |
| Line 25, rest (cost of an extension matrix, one output per retained DOF, network acting on stencils, channels and grid levels, about \(6\times10^5\) parameters) | not in this unit: approved introduction (paragraph 4) and new Section 3.2 |
| Line 27: "much as smoothing improves a tentative prolongation in smoothed-aggregation multigrid [Vaněk et al. (1996)]" | P7 S6. Vaněk et al. (1996) also stays cited in new Section 3.3 (old 4.3) with the same analogy. |
| Line 27, rest (Xu 1992, Proposition 4, Golub & Varga, Xu & Zikatanov, 5.4 to 265) | not in this unit: approved introduction (paragraph 5) and new Sections 3.3-3.4 and 5.5 |
| Line 29 S1: "Learned and reduced component models differ in three choices: what a component exchanges ..., how its interior response is represented, and where its approximation is improved." | P2 S6 |
| Line 29 S2, clause 1: "Methods with few boundary DOFs exchange restricted displacement patterns, in return for small assembled systems" | duplicate of P1 S5-S6 and of the approved introduction (paragraph 3); not repeated |
| Line 29 S2, clause 2: "component reduced-basis methods keep the interface but represent the interior on a discretisation shared by the parameter family" | P3 S1 |
| Line 29 S2, clause 3: "inexact substructuring improves approximate local solves inside a preconditioner" | duplicate of P5 S2; not repeated |
| Line 29 S3: "The learned substructures of [Huang 2023] and [Guo 2026a] reduce what a component exchanges ...; NICE keeps what a component exchanges and learns only how its interior responds." | P2 S7-S8 |
| Line 29 S4: "It keeps the complete retained space, represents the interior by a geometry-conditioned network, and improves the approximation inside the condensed operator that is assembled." | P2 S9 (complete set of retained DOFs: P2 S8 and P3 S5) |
| Line 29 S5: "Section 1.1 relates it to each family." | removed (self-reference; this text is now Section 1.1) |

### Numbers

Old EN source numbers (outside citations and inline mathematics): 1.1, 1 (Figure 1b), 9,674, 72,631, 5.5, 6.2, 5 (Eq. (5)). All are present in the new EN and CN text; Eq. (5) appears as Eq. (7). New numbers not in the source: 3.1 (added cross-reference, new numbering) and 7 (renumbered equation). The years 2004, 2005, 1997 and 1996 enter only through the citations moved from the old introduction. EN and CN contain the same numbers, the same inline mathematics and the same citation links, paragraph by paragraph.

### Terms replaced (brief 3.1)

interior extension / learned extension / corrected extension → interior displacements, displacement recovery, recovery operator \(F\), exact recovery \(E\); retained space / retained set → (complete) set of retained degrees of freedom; equilibrium correction → two-grid correction; condensed operator / condensed energy → condensed stiffness / strain energy; Galerkin (coarse) operator → coarse-grid matrix; geometry-conditioned network → network that takes the cell geometry as input; Ritz property → property that follows from the principle of minimum potential energy; Ritz-type training → training by energy minimisation; Schur-complement targets → targets computed with the exact condensed stiffness. CN: 主自由度, 位移恢复, 位移恢复算子, 精确位移恢复, 凝聚刚度矩阵, 精确凝聚刚度矩阵, 两重网格修正, 粗网格矩阵, 以胞元几何为输入的神经网络.

## (b) Cross-references renumbered

| Old | New | Place |
|---|---|---|
| Eq. (5) | Eq. (7) | P5 S5 |
| Section 5.5 | Section 5.5 (unchanged) | P1 last sentence |
| Section 6.2 | Section 6.2 (unchanged) | P5 S2 |
| Figure 1b | Figure 1b (unchanged) | P1 |
| (none) | Section 3.1 (added, new numbering) | P2 S3, pointing to the condensed stiffness from the strain energy and Proposition 1 |

## SUPPLEMENT ADDITIONS

None. The source of this unit does not mention the Uncorrected continuation or the Smoothing-trained variant.

## (d) Open questions

1. Resolved by the verifier: U02 (Section 2.2) already says "the exact condensed stiffness \(S\), that is, the Schur complement" at the definition. The duplicate in P4 was removed, so the brief's "say once" holds.
2. Vaněk et al. (1996) with the smoothed-aggregation analogy now appears in P7 here and in new Section 3.3 (old 4.3). Both were kept because no citation may be dropped; the assembler may shorten the 3.3 sentence if the repetition is unwanted.
3. Old line numbers: the task named old lines 25, 27 and 29 as the source of the moved items. In MANUSCRIPT_EN.md the Falgout & Vassilevski sentence and the Ritz-property citation list are on old line 23, the neural-operator sentence on line 25, the Vaněk clause on line 27 and the three-choices paragraph on line 29. All four lines were used.
4. "degrees of freedom" is spelled out throughout 1.1, as in the approved introduction draft; the abbreviation "DOFs" is left to Section 2.1, where the old text defines it.
5. "Ideal interpolation" (the algebraic multigrid term of Falgout & Vassilevski) is kept and explained by the preceding sentence on \(E\); in CN it is 理想插值, and prolongations of the cited works are written as 插值算子, as in the old CN text.

## VERIFIER

Checks run: sentence-by-sentence comparison of old 1.1 (old lines 40-50) and the moved items (old lines 23, 25, 27, 29) with the new EN; Python diff of numbers, citation links and inline mathematics (old EN vs new EN, new EN vs new CN, per paragraph and in citation order); sentence lengths; grep for banned patterns and old terms (brief 1, 3.1); cross-check of the destinations claimed for items not in this unit (U00_EN.md paragraphs 3-5, U02_EN.md Section 2.2, U03_EN.md \tag{7}, U05_EN.md Section 3.3).

Results after the changes below: every number of the source is present (9,674; 72,631; Figure 1b; Section 5.5; Section 6.2; old Eq. (5) as Eq. (7)); every citation link of old 1.1 and of old lines 23-29 is present, except Xu (1992), Golub & Varga (1961) and Xu & Zikatanov (2002), which belong to the content kept in U00 (introduction paragraphs 4-5) and are cited there. No citation link was added that is not in the source. EN and CN have 8 blocks each, with identical numbers, inline mathematics and citation links in the same order in every paragraph. No EN sentence exceeds 34 words; no CN sentence exceeds 80 characters. No PIML number, no priority claim, nothing on measurements, machine load or training cost. New EN length 1,326 words without citations.

Changes made:

1. P1 (Guo et al. 2026a): the citation was attached to the sentence describing the Bézier variant, and the finding ("cost more than it saved ... less accurate on the larger substructures") stood without a citation. The citation now closes the finding sentence, as in the source, and the first sentence says the comparison was made "in the same study" / "在同一研究中". The finding stays verbal.
2. P1: "boundary error" / "边界误差" restored to "boundary model error" / "边界模型误差" (source wording; "boundary error" could be read as the error at the boundary).
3. P2 S2: the condition for the minimum-energy property was weaker than in the source (only "reproduces the retained displacements"). Old Proposition 1 holds under (A1) and (A2), and the approved introduction (U00 paragraph 5) states "linear in the retained displacements and reproduces them and rigid-body motions". The sentence now carries this condition in EN and CN, so the claim is not strengthened.
4. P2 CN: "其主自由度集合随切割变化" had an unclear referent; now "而胞元的主自由度集合随切割变化".
5. P3: the paragraph now opens with its point, "As to what a component exchanges, ..." / "就组件交换的内容而言，……", matching P4 and P5, which open with the other two respects.
6. P3 CN: passive "定义在不同激活集上的解在降阶之前先被扩展到" rewritten as "已有研究在降阶之前将不同激活集上的解扩展到".
7. P4: "the exact condensed stiffness, that is, the Schur complement" removed. U02 (Section 2.2) says "that is, the Schur complement" at the definition of \(S\); brief 3.1 allows it once. The Jiang, H. et al. (2026) sentence now reads "learn boundary energies from targets computed with the exact condensed stiffness" / "依据精确凝聚刚度矩阵给出的目标学习边界能量".
8. P4: "Its corrected recovery operator" became "A single corrected recovery operator" / "同一个修正后的位移恢复算子", which restores the source's "one corrected extension ... defines" all three quantities.
9. P4 CN: "已有研究针对小切割改善了数值形函数的条件性" became "也已有研究改善了数值形函数在小切割情形下的条件数" (条件性 is not a standard term; the cited title is "Conditioned numerical shape functions"; "also" restored). "有一种点阵方法" became "一种点阵方法".
10. P6: "lose the quadratic strain energy" was inaccurate (the strain energy is always quadratic in the full displacement field; what is lost is the quadratic dependence on the retained displacements, which the CN already said). EN now: "Such maps do not obey superposition. The strain energy of the predicted field is then not a quadratic form in the retained displacements, and the properties of the condensed stiffness rest on this quadratic form." CN split into the same three sentences.
11. P6: "the network predicts the interior displacements of a single cell, and their strain energy defines a condensed stiffness" attributed the strain energy to the interior displacements only. Now "the learned map recovers the displacements of a single cell from its retained displacements. The strain energy of these displacements defines ..." (source: "a per-cell extension whose energy defines a condensed operator"). CN aligned.
12. P6 CN: the modifier chain "细尺度全分辨问题两层预条件子的粗空间" rewritten as "两层预条件子的粗空间，用于求解完全分辨的细尺度问题".
13. P7: "In algebraic multigrid, \(E\) is the ideal interpolation" became "In algebraic multigrid terms, ..." / "按代数多重网格的术语，……" (\(E\) is not an object of algebraic multigrid itself; the source says it "is the ideal interpolation of algebraic multigrid").
14. Notes updated: the rows for Guo 2026a, the Schur-complement wording and the P2 condition, the term list, and open question 1 (resolved).

Not resolved (for the assembler or the author):

- "active degrees of freedom" / "激活自由度" (P7) is used before its definition in Section 2.2, as in the source. A short gloss could be added if the advisor stumbles on it.
- Open questions 2 (Vaněk et al. (1996) cited both here and in Section 3.3) and 5 ("ideal interpolation" kept as the algebraic multigrid term) remain author decisions.
