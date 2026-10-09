# U00 notes: title, abstract, keywords, Introduction (Section 1 without 1.1) and Figure 1

Source: MANUSCRIPT_EN.md / MANUSCRIPT_CN.md lines 1–39 (old numbering).
Output: `rewrite_plain/sections/U00_EN.md` and `rewrite_plain/sections/U00_CN.md`, 35 lines each. They replace source lines 1–39, so the unit is 4 lines shorter than the source.

## 0. Format and alignment

- The title line (line 1) is byte-identical to the source in both languages.
- EN and CN are line-aligned with each other: the same 35 lines hold the same heading, paragraph, figure line, caption or bullet. Citation links and inline mathematics match line by line (checked by script).
- Order: title, `## Abstract`, abstract, keywords, `## 1. Introduction`, paragraph 1, Figure 1 (image line, blank line, caption line), paragraphs 2–6, "The main contributions are as follows.", four bullets, roadmap paragraph. Figure 1 sits between the first and second introduction paragraphs, as in the source.
- The EN abstract has 250 words (source 250; approved draft 250).
- Unit word count (EN): old 2,488, new 1,982.
- The approved drafts `R5_术语对照与摘要草稿.md` and `R5_引言草稿.md` are the base text. Section 1.3 below lists every place where the base text was changed and why.

## 1. (a) Material removed or moved

### 1.1 Abstract and keywords (old line 5, line 7)

1. "whose discrete space changes with the cut": removed from the abstract. It is now stated in the introduction ("The number of retained degrees of freedom is therefore large and changes with the geometry") and in U01 (Section 1.1).
2. "NICE divides the tasks of condensation: a geometry-conditioned network supplies the interior field, the energy form the structure, and a fixed two-grid correction the improvability": deleted (banned triad). The content is restated plainly in abstract sentences 5–8.
3. "The network must be accurate, not structure-preserving.": deleted (banned slogan).
4. "its network ... forms no matrix": removed from the abstract to keep it at 250 words. It is still stated in introduction paragraph 4 and contribution 2.
5. "The condensed stiffness is a Ritz approximation, bounded below by the exact Schur complement": restated as "never softer than the exact one" (term map).
6. "a correction nonexpansive in this energy cannot increase that error": restated as "Under stated conditions the correction cannot increase this error."
7. "for any approximate extension" (compliance does not imply sensitivity accuracy): removed from the abstract. It is kept in introduction paragraph 6 and in Section 4 (old 3).
8. The expansion "Neural-initialised static condensation with equilibrium correction (NICE)" is removed from the abstract (word limit). It remains in the title and in introduction paragraph 4. See open question 2.
9. Old keywords "learned substructures; Schur complement; Ritz approximation; cut finite elements; TPMS lattices" were replaced, as the unit instructions specify, by "superelement; static condensation; neural network; multigrid; cut finite element method; TPMS lattice; thickness optimisation".

### 1.2 Introduction and Figure 1 (old lines 11–38)

Old paragraph 1 (line 11)

10. "components of the same kind but of different geometry: graded lattices whose cells vary in wall thickness ..., architected materials trimmed to the shape of a part, cellular cores supported and loaded along cut boundaries": condensed to "Graded lattices ... and lattices trimmed to the shape of a part consist of many cells of different geometry". The phrases "whose cells vary in wall thickness" and "cellular cores supported and loaded along cut boundaries" were removed. The plate example and Figure 1 cover supports on a cut boundary.
11. "which favours replacing each component by something cheaper" / "which favours keeping each component's detail": removed (rhetorical). The two requirements themselves are kept.
12. "Homogenisation resolves this tension in favour of economy": deleted (flourish). Restated as "Homogenisation replaces the cells by an equivalent material; it is inexpensive ...".
13. "the homogenised model of that section": shortened to "the homogenised model". Section 5.10 is still named in the same paragraph.
14. The NICE expansion in old paragraph 1 moved to new paragraph 4 ("This paper proposes NICE (...)").
15. "Resolving every wall of every cell settles the tension in favour of fidelity" and "at a cost that each design iteration ... pays again in full": banned flourish, restated plainly. "(Section 5.9)" is kept.
16. "Substructuring takes the middle course.": banned phrase. Restated as "Substructuring lies between these two approaches."
17. "and returns the forces conjugate to their displacement, so that the structure can be assembled": removed. This is a duplicate of old Section 4.1 (new Section 3.1, old Eq. (12) = new Eq. (5)).
18. "Static condensation does this exactly ..., but at the price of an interior factorisation for every geometry": restated plainly (paragraph 2). The Guyan and Irons citations moved to the preceding sentence.
19. "Learned component models seek to remove this price": restated as "Learned component models replace the interior solution by a neural network."

Old paragraph 2 (line 17)

20. "Static condensation serves analysis and design well because its condensed stiffness is symmetric and positive semidefinite with the rigid-body kernel and therefore assembles like a finite element.": removed. The same properties are stated for NICE in paragraph 5, and "assemble ... like finite elements" appears in paragraph 3.
21. "A learned component model that takes its place is most useful if it [four properties]" and "This paper shows that a learned component model can have these four properties while keeping every degree of freedom ...": the four properties are no longer listed as such. Each one is stated in paragraphs 2–6 and in the contributions: structure (paragraph 5), no interior factorisation for a new geometry (paragraphs 2 and 4), error traceable to compliance and sensitivity (paragraph 6), improvement without retraining (contribution 3). The "keeping every degree of freedom ..." part is in paragraph 4. See open question 1.
22. "It obtains the structure of static condensation from where its approximation is placed ..., and its improvability from a fixed correction of its interior field.": deleted (banned "improvability"). The content is in paragraph 5.
23. "The network then has to be accurate, but it does not have to preserve structure.": deleted (banned slogan).

Old paragraph 3 (line 19)

24. "the box-face degrees of freedom carry exterior loads and intercell coupling, and those of the cut band carry virtual work on the cut surface": shortened. It is now "through which it connects to its neighbours, supports and loads" (paragraph 2) and "supports and loads on the cut surface act on all degrees of freedom of the cut-plane elements" (paragraph 3). The full statement remains in Section 2.2 (U02).
25. "Forming the Schur complement explicitly needs one interior solve per retained degree of freedom": removed. This is a duplicate of old Section 4.2 (new Section 3.2: "an extension matrix formed for every geometry costs one interior solve per retained DOF"). The second half, "applying it needs an interior factorisation for every geometry", is kept in paragraph 2.
26. "An approximation can reduce the retained set or approximate the interior extension.": removed as a sentence. The two options are contrasted in paragraph 3 (reduction) and paragraph 4 (NICE keeps all retained DOFs and approximates the interior).
27. "Approximating the interior extension on the complete retained space, as in [Huynh et al. (2013)](https://doi.org/10.1051/m2an/2012022), alters only the interior response to each pattern.": moved to U01 (Section 1.1, component reduced-basis methods keep the complete interface; Huynh et al. (2013) is cited there).
28. "We place the approximation in the interior extension, learned and conditioned on the geometry of each cell, and keep every box-face degree of freedom and the whole cut band.": now in paragraph 4, sentences 2–3.
29. "Keeping the cut band also keeps loads and supports on the cut surface available: the plate of Section 5.10 is clamped through its cut band without retraining.": now in contribution 1 (Sections 2, 5.7 and 5.10) and paragraph 1 (plate of Section 5.10, Figure 1).

Old paragraph 4 (line 21): kept as paragraph 6 (see item 51 and Section 2).

Old paragraph 5 (line 23)

30. "The central idea is a division of tasks within static condensation: learning supplies the trial field, the variational form supplies the structure, and a fixed correction supplies the improvability.": deleted (banned triad).
31. "The approximation is placed in the interior extension; ... and the error that remains is followed through the energy share and the stiffness derivative to the compliance and the design sensitivity.": summary sentence. Its content is restated in paragraphs 4–6.
32. "Whatever admissible extension it produces, as long as it reproduces rigid-body motion, the energy form yields ...; exact static condensation is the case in which the extension is exact.": now in paragraph 5, sentences 1–5.
33. "Static condensation is itself a two-level method: the exact extension \(E\) is the ideal interpolation of algebraic multigrid ..., and \(S=E^TKE\) is its Galerkin coarse operator [Falgout & Vassilevski (2004)](https://doi.org/10.1137/S0036142903429742). NICE replaces this interpolation by a learned and smoothed one, and condenses with the Galerkin operator of that interpolation.": moved to U01 (Section 1.1, last paragraph; present in `U01_EN.md`).
34. "The same Ritz property underlies static condensation itself [Toselli & Widlund (2005)](https://doi.org/10.1007/b137868), multiscale finite elements [Hou & Wu (1997)](https://doi.org/10.1006/jcph.1997.5682), component reduced-basis methods [Huynh et al. (2013)](https://doi.org/10.1051/m2an/2012022) and learned shape functions evaluated as \(N^TKN\) [Huang et al. (2023)](https://doi.org/10.1016/j.eml.2023.102041); here it carries a learned extension on a retained space that changes with the cut.": moved to U01 (present in `U01_EN.md`, paragraph 2).

Old paragraph 6 (line 25)

35. "Learning supplies the trial field, and three obstacles shape how.": deleted (banned).
36. "An extension matrix formed for every geometry would cost one interior solve per retained degree of freedom.": removed. Duplicate of old Section 4.2 (new Section 3.2).
37. "A network with one output per retained degree of freedom, as for learned shape functions, computes such a matrix column by column, with an output dimension fixed by the retained set it was built for.": shortened to the last two sentences of paragraph 3. The detail "column by column" is removed; new Section 3.2 states that such a network "would have to change its output dimension with the cut".
38. "And a solution map that is nonlinear in its input, as most neural operators learn, gives up superposition and with it the quadratic condensed energy on which the structure rests.": moved to U01 (present in `U01_EN.md`, paragraph on learned solvers and neural operators).
39. "... computes the interior field of each cell by local interactions and a multilevel latent hierarchy whose weights act on stencils, channels and grid levels rather than on individual degrees of freedom": kept in paragraph 4 as "Its parameters act on local node positions, feature channels and grid levels rather than on individual degrees of freedom". The architecture details (local interactions, grid hierarchy) remain in new Section 3.2 (old 4.2).
40. "Its size, about \(6\times10^5\) parameters, does not grow with the number of retained degrees of freedom, one network serves cells whose retained sets differ, and neither the extension nor the condensed matrix is ever formed.": now in paragraph 4, sentences 6 and 11, and contribution 2.

Old paragraph 7 (line 27)

41. "A fixed correction supplies the improvability:": deleted (banned).
42. "one symmetric two-grid cycle per geometry, with Chebyshev smoothing and a coarse-grid Galerkin correction [Xu (1992)] ...": now "one two-grid cycle of Chebyshev smoothing and coarse-grid correction [Xu (1992)]" (paragraph 4). The details "symmetric" and "coefficients fixed per geometry" remain in new Sections 3.1 and 3.4 (old 4.1 and 4.4).
43. "much as smoothing improves a tentative prolongation in smoothed-aggregation multigrid [Vaněk et al. (1996)](https://doi.org/10.1007/BF02238511)": removed from the introduction. Duplicate of old Section 4.3 (new Section 3.3), and also present in U01.
44. "and the transpose of the complete corrected extension returns the force conjugate to that energy": removed. Duplicate of old Section 4.1 (new Section 3.1, Eq. (5)).
45. "a deterministic harmonic start": renamed "a graph-harmonic initial field, which has no trainable parameters" (the wording of old Section 5.5 and Table ST07). "the error" became "the mean energy error" (Table 4 reports mean energy errors).
46. "Together, the learned extension and this correction make up NICE.": restated in paragraph 4 ("The network and the correction together define a linear map ...").

Old paragraph 8 (line 29)

47. The whole paragraph "Learned and reduced component models differ in three choices: ... Section 1.1 relates it to each family." moved to U01 (Section 1.1, item 2 of the draft's list). Huang et al. (2023) and Guo et al. (2026a) are still cited in paragraph 3. "Section 1.1 relates it to each family" became "Section 1.1 reviews related work." in the roadmap.

Old paragraph 9 and bullets (lines 31–36)

48. "The contributions of this work follow this division.": became "The main contributions are as follows." (banned "division").
49. The bold bullet titles "(i)"–"(iv)" were removed, following the approved draft. The content is kept.
50. Old (ii): "with the retained values and rigid-body motion imposed by construction, so that the relations of Section 3 apply to it and its transpose returns the work-conjugate force". Rigid-body reproduction is stated in paragraph 5. "Its transpose returns the corresponding nodal forces" is a duplicate of new Sections 3.1 and 3.2. "One network of about \(6\times10^5\) parameters serves retained sets of 2,679 to 45,900 degrees of freedom": these numbers are stated in paragraphs 2 and 4.
51. Old (iv): "which the energy error bounds only at the order of its square root" moved to paragraph 6 ("The energy error controls this term only at the order of its square root"). "(Propositions 2 and 3; ...)" moved to paragraph 6 as "(Propositions 3 and 4)". "its designs are verified against exact condensation" is kept in contribution 4.

Old paragraph 10, roadmap (line 38)

52. The roadmap was rewritten for the new section order. "the properties sought for an approximate condensed stiffness" (Section 2) became "the assumptions of the analysis" (new Section 2.4). The text on required properties stays in Section 2.3 (U02). "as Propositions 1–3: the Ritz identity, the share-weighted bound on the compliance and the error of the thickness sensitivity" became the Section 4 description. Proposition 1 now lives in Section 3.1, so the condensed stiffness is no longer listed under Section 4. "(Proposition 4)" was dropped from the roadmap. The Section 5 description keeps the source details: cut thin-walled Schwarz-P cells, stabilised cut finite element model, 80 validation geometries, two-cell assemblies, ablation, eight-cell lattices of cells not used in training or checkpoint selection, cost against whole-lattice direct solution and conventional exact condensation, and designs checked against exact condensation and compared with a homogenised model.

Figure 1 caption (old line 15)

53. Terms changed: "box faces" became "cell faces"; "cut band" became "cut-plane elements"; "clamped through its cut" became "clamped at its cut"; "in-plane traction" became "in-plane traction load"; "eliminated by condensation" became "eliminated by static condensation". The title "trimmed to a part" became "trimmed to the shape of a part". Long sentences were split. All numbers are kept (\(8\times4\), \(\tau=0.40\), 16, eight, 72,631, 9,674, Eq. (1) twice). The image already uses the new terms ("clamped cut-plane elements", "retained cell-face nodes", "retained nodes of cut-plane elements"); this was checked against `figures/F00_problem.png`.

### 1.3 Changes to the approved drafts and the reasons

- Abstract EN: "have cut cells that carry supports and loads" became "... that can carry ..." (source: "can carry"; the draft overstated the claim). "than a direct solve" became "than direct solution" (source wording), which keeps the abstract at 250 words. The long sentence was split at its semicolon ("Under stated conditions ...").
- Abstract CN: "往往承担" became "可能承担" (same reason as EN). "快约 10 倍" became "快约十倍", to match EN "ten" and the source CN. "以胞元表面节点和切割平面单元节点的自由度为主自由度" became "将胞元表面节点以及与切割平面相交的单元节点的自由度取为主自由度", because 切割平面单元 is not defined in the abstract. The long sentence was split at "；".
- Paragraph 1: "where the cells are not small compared with the structure" became "where the cells are not small compared with the scale on which the field varies". The source says "small compared with the variation of the field", and supports and loads are places of rapid field variation, not of large cells. "(Section 5.9)" was restored from the source. Long sentences were split.
- Paragraph 2: "cut-plane elements" is now defined at first use ("the elements intersected by the trimming plane, called cut-plane elements below").
- Paragraph 3: "To limit the size of the network output, these methods reduce the boundary degrees of freedom by linear or Bézier interpolation" became "These methods describe the boundary displacement of each substructure by a few degrees of freedom, for example through Bézier interpolation". The source does not mention linear interpolation or that motive. Old Section 1.1 states only "a few boundary degrees of freedom" and Bézier enrichment. "the retained set" became "the number of retained degrees of freedom" (term map).
- Paragraph 4: "serves all cells, whatever their number of retained degrees of freedom" became "serves cells with different numbers of retained degrees of freedom". The source says "serves cells whose retained sets differ", and applicability is limited to the trained domain (Section 6.3). "no matrix is formed" became "neither this map nor the condensed stiffness is formed as a matrix", because the coarse matrix \(V^TAV\) is formed and factorised (old Sections 4.4 and 4.6). Two sentences were added. The first explains what smoothing and coarse-grid correction each reduce; it is the one-sentence definition of the two-grid cycle and is taken from old Section 6.1. The second states that the network and correction define a linear map from retained to whole-cell displacements; this introduces the recovery operator used in U01 and Section 3.
- Paragraph 5: the draft sentence "By the principle of minimum potential energy, a condensed stiffness ... is symmetric positive semidefinite, ..." was split. Symmetry and semidefiniteness now follow from the energy form; the bound and the quadratic error follow from the minimum principle (old Proposition 1 and Section 4.1). "linear in the retained displacements" was added to the condition, because assumption (A2) requires a linear interior approximation. A reference to Proposition 2 (old Proposition 4) was restored. "that of the uncorrected one" became "the error before the correction", to avoid confusion with the removed "Uncorrected continuation" variant.
- Paragraph 6: two sentences were added from the source. "The energy error controls this term only at the order of its square root" comes from old contribution (iv). "For any approximate static condensation used in design, an accurate compliance thus does not guarantee an accurate sensitivity" comes from old paragraph 4. "for any approximate interior displacement" became "for any interior approximation that meets the conditions above", because old Section 3 requires an admissible linear extension. "(Propositions 3 and 4)" and "(Section 5.6)" were restored.
- Contribution 2: "(Section 3)" became "(Sections 3.1, 3.2 and 3.6)" (old: Sections 4.2 and 4.6; Proposition 1 is now in 3.1).
- Contribution 3: the source condition "for any discrete model with a symmetric positive semidefinite stiffness matrix and a positive definite interior block" was restored. The draft's "in all cases tested a larger number of smoothing steps reduced the error further" became "In every example of Table ST07, a larger number of smoothing steps reduced the energy error further". The source claim is in old Section 7: "in every example of Table ST07 a larger correction budget reduced it further". The sensitivity error is not monotone in the number of steps (old Section 5.5, cell U1), so the claim is limited to the energy error. References: "(Sections 3 and 5.5)" became "(Sections 3.3 and 3.4)" plus "(Section 5.5)".
- Contribution 4: "(Sections 4 and 5)" became "(Sections 4, 5.6, 5.8 and 5.10)" (old: Sections 5.6, 5.8 and 5.10). "compared with exact static condensation" became "verified against exact static condensation" (source wording).
- Roadmap: see item 52. "Section 1.1 reviews related work." was added.

### 1.4 Number check (script)

- Every number of the old EN source appears in the new EN text, except:
  - 1996: Vaněk et al. (1996), moved to U01 and kept in new Section 3.3.
  - 1997: Hou & Wu (1997), moved to U01.
  - 2004: Falgout & Vassilevski (2004), moved to U01.
  - 2005: Toselli & Widlund (2005), moved to U01 and kept in old Section 3 introduction (new Section 4).
  - 4.2, 4.3, 4.4, 4.6: old section references, renumbered to 3.2, 3.3, 3.4 and 3.6.
- Numbers in the new EN text that are absent from the old source: 3.1, 3.2, 3.3, 3.4, 3.6 (renumbered section references) and 07 (Table ST07 cross-reference, see Section 2).
- EN and CN contain the same set of numbers. The only count difference is "8": CN writes "8 个切割胞元" where EN writes "eight cut cells", as the source does. Word numbers kept from the source: "eight-cell", "two-cell", "about ten times" (CN 八胞元, 两胞元, 约十倍).
- Sentence lengths: no EN sentence is longer than 35 words. One CN sentence is longer than 80 characters: the NICE definition, whose length comes from the embedded English expansion. It was kept.
- Banned words and old terms (brief Section 3.1): none remain. The remaining script hits are false positives: "nodal forces" (Capuano & Rimoli map displacements to nodal forces, not the stress-test load class) and 正是 inside 修正是.

## 2. (b) Cross-references renumbered (each occurrence mapped once)

| New location | Old | New |
|---|---|---|
| Paragraph 1 | Section 5.10; Section 5.9 | Section 5.10; Section 5.9 (unchanged) |
| Figure 1 caption | Eq. (1), twice | Eq. (1), twice (unchanged) |
| Paragraph 5 | Proposition 4 (old paragraph 7) | Proposition 2 |
| Paragraph 5 | Section 5.5 | Section 5.5 (unchanged) |
| Paragraph 6 | Section 3 ("derives it for any admissible extension") | Section 4 |
| Paragraph 6 | Propositions 2 and 3 (old contribution iv) | Propositions 3 and 4 |
| Paragraph 6 | Section 5.6 | Section 5.6 (unchanged) |
| Contribution 1 | Sections 2, 5.7 and 5.10 | unchanged |
| Contribution 2 | Sections 4.2 and 4.6 | Sections 3.1, 3.2 and 3.6 (3.1 added for the Proposition 1 properties stated in the bullet) |
| Contribution 3 | Proposition 4; Sections 4.3, 4.4 and 5.5 | Proposition 2; Sections 3.3 and 3.4; Section 5.5 in a separate sentence |
| Contribution 3 | none | Table ST07 (added with the claim taken from old Section 7) |
| Contribution 4 | Sections 5.6, 5.8 and 5.10 | Sections 4, 5.6, 5.8 and 5.10 |
| Roadmap | Sections 2, 3 (Propositions 1–3), 4 (Proposition 4), 5, 6, 7 | Sections 1.1, 2, 3, 4, 5, 6, 7 in the new order; proposition numbers dropped |
| Old paragraph 8 | Section 1.1 | Section 1.1 in the roadmap |

## 3. (c) SUPPLEMENT ADDITIONS

None. The source unit contains no statement about the Uncorrected continuation or the Smoothing-trained variant.

## 4. (d) Open questions

1. Old Section 2.3 (last paragraph) begins "Of the four properties named in Section 1, two can now be made precise." The new introduction no longer lists four properties. The writer of Section 2.3 (U02/U03) should rephrase this sentence, for example "Two properties are required of an approximate condensed stiffness ...".
2. The abstract does not expand NICE; the title does. Adding "(neural-initialised static condensation with equilibrium correction)" would bring the abstract to 256 words. This is an author decision.
3. The abstract keeps the approved wording "homogenisation is inaccurate where cells are not small relative to it" (that is, the structure). The introduction now uses the source meaning: "not small compared with the scale on which the field varies". The author may want the abstract aligned, for example "where the scales do not separate", which saves two words.
4. Falgout & Vassilevski (2004) and Hou & Wu (1997) were cited only in the old introduction. They are now cited only in U01 (`U01_EN.md` lines 5 and 15). If U01 drops them, they become uncited in the reference list.
5. The one-sentence explanation of the two-grid cycle in paragraph 4 is taken from old Section 6.1: smoothing reduces the high-frequency part of the interior error, and coarse-grid correction reduces its smooth part. Please confirm that it is wanted in the introduction.
