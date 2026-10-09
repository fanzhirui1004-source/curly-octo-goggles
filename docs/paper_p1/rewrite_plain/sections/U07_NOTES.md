# U07 notes: Section 5 introduction, 5.1 Geometries, variants and loads (Tables 1-3, Figure 4), 5.2 Verification

Source (old numbering): MANUSCRIPT_EN.md / MANUSCRIPT_CN.md lines 359-429 (old Section 5 introduction, 5.1, 5.2, Tables 1-3, Figure 4).

New blocks (EN and CN line-aligned, 69 lines each): heading `## 5.`; P0 overview; heading `### 5.1.`; P1 validation geometries and cut-severity groups; Figure 4 and caption; P2 load classes; Table 1; P3 scope of Table 1; P4 the variants of the main text; Table 2 (two rows); P5 Base network + correction; P6 architecture and training geometries, pointer to Note S8; P7 checkpoint selection and population statistics; Table 3 with caption; heading `### 5.2.`; P8 verification of the reference model; P9 verification of the deployed operator; P10 spectral condition of the smoothing.

Length (EN words, tables and display equations excluded, table captions included, inline mathematics counted as one word): old source 1127, new 1360. The increase comes from splitting long sentences, from the plain gloss of the work–energy identity, from naming NICE where the source said "the corrected learned extension" or "the principal variant", and from the two pointer sentences to Supplementary Note S8 that replace the removed variant descriptions.

Self-check (Python, on the final files):
- (a) Every number of the old EN source is in the new EN text, except the old section numbers 4.3, 4.4 and 4.5 and the old "Section 4", which are renumbered (list (b)). Occurrence counts fall only where values were moved to the supplement additions (15,000 twice, 591 twice and the "8 smoothing steps" cells of the removed Table 2 rows) or where a repeated value was stated once ("20 uncut, 20 lightly, 20 moderately and 20 heavily cut cells" → "four cut-severity groups of 20 cells each"). One "Supplementary Note S1" was dropped from "(Table ST10; Supplementary Note S1)" under brief rule 4; Note S1 is still cited twice in the same paragraph.
- (b) The new EN text contains no number absent from the old source except the renumbered references 3.2, 3.3, 3.4 and 3.5 (Section 3.2 for old "Section 4", Sections 3.3-3.5 for old 4.3-4.5) and "Proposition 3" (old Proposition 2; the digit 3 also occurs in the source).
- (c) EN and CN contain the same set of numbers and identical inline mathematics. The only count difference is "20": EN writes "Twenty of the 80 validation geometries" as a word at the start of a sentence, as the source does; CN writes "20 个".
- (d) No EN sentence above 35 words (82 sentences, longest 32, mean 16.6). No CN sentence above 80 characters (79 sentences, longest 71, mean 35.7).
- (e) No banned word or old term in EN or CN. Remaining grep hits are not old terms: "viewing direction" / 观察方向 in the Figure 4 caption; "the returned nodal forces" / 返回的节点力 (the corresponding nodal forces of Section 3.1, not the nodal-point-load class); the ordinals "first average" and "first design"; "smoothing interval" and "Chebyshev smoothing" (not the variant name).

## (a) Sentences and numbers removed or moved

All content stays in this unit except the description of the Smoothing-trained variant and the Uncorrected continuation, which moves to Supplementary Note S8 (SUPPLEMENT ADDITIONS below).

### Old Section 5 introduction (old line 361) → new Section 5 introduction, P0

| Old sentence | Where it is now |
|---|---|
| "The examples show three things." | P0 S1, which names the three areas (single cells, assemblies, thickness design). |
| "On single cells, the corrected learned extension reaches a mean energy error of 0.074% on 80 validation geometries, and the network and the correction each remove error that the other leaves (Sections 5.3–5.5)." | P0 S2-S3. "the corrected learned extension" → "NICE" (its recovery operator with the correction). |
| "After assembly, compliance follows the share-weighted relation of Proposition 2, whereas the local sensitivity has to be checked on its own (Sections 5.6–5.8)." | P0 S4-S5. "share-weighted" written out as "the energy error of each cell is weighted by its energy fraction" (brief 3.1: energy share → energy fraction). Proposition 2 → 3. |
| "In design, an eight-cell analysis with sensitivities is about ten times faster than direct solution, and the optimised designs agree with exact condensation in compliance and gradient (Sections 5.9 and 5.10)." | P0 S6-S7, one reference each. "direct solution" → "a direct solution of the whole lattice" (old 5.9: "the whole-lattice direct solution takes 868 and 1,042 s, about ten times as long"). "the optimised designs agree with exact condensation in compliance and gradient" → "For the optimised designs, the compliance and the gradient computed with NICE agree with those of exact static condensation" (this is what Table 3 reports for Case A and the plates). |
| "Table 3 collects the accuracy against exact condensation in every assembled example." | P0 S8 ("of NICE", as in the Table 3 title) |
| "Sections 5.1 and 5.2 first state the settings and verify the discrete reference." | P0 S9 ("the reference model", matching the new 5.2 title) |

### Old 5.1 "Geometries, variants and loading conditions" (old lines 363-420) → new 5.1 "Geometries, variants and loads"

| Old sentence | Where it is now |
|---|---|
| Heading | `### 5.1. Geometries, variants and loads` (brief Section 4) |
| "The 80 validation geometries comprise 20 uncut, 20 lightly, 20 moderately and 20 heavily cut cells, with uniform, affine or mixed trilinear thickness fields and corner parameters from 0.1762 to 0.6983 (Table ST02)." | P1 S1-S2. S1 introduces the term "cut-severity groups" (brief 3.1: cut strata → cut-severity groups). |
| "Each carries at most one planar cut with normal \((\cos\vartheta,\sin\vartheta,0)\), \(0<\vartheta<\pi/4\); heavy cuts retain less than one third of the volume of the cell box, moderate cuts between one and two thirds, and light cuts more than two thirds." | P1 S3-S4 |
| "U, L, M and H identify the uncut, lightly, moderately and heavily cut cells used for detailed comparisons (U1, U2, L1, M1, M2 and H1–H3; Supplementary R1)." | P1 S5-S6 ("Supplementary Section R1") |
| "Table 1 lists the discretisation and correction settings, and Figure 4 shows four of the cells used for detailed comparisons." | P1 S7-S8 |
| Figure 4 caption, S1 "The same unit-box scale and viewing direction are used for (a) uncut U1, (b) moderately cut M1, and (c,d) heavily cut H1 and H2." | Caption S1 |
| Caption S2 "Light blue denotes the material surface, dark blue the wall section on the box faces and sand the section on the cut plane;" | Caption S2 ("cell faces", brief 3.1) |
| Caption S2 (cont.) "the part removed by the cut is drawn translucent grey, and the cells are viewed from the side of the cut." | Caption S3 |
| Caption S3 "Percentages indicate the box volume remaining after the cut, relative to the unit box, before intersection with the thin-wall material." | Caption S4 |
| Caption S4 "Surfaces are reconstructed from Eq. (1) on a grid of 129 positions per axis, for visualisation only." | Caption S5 ("129 points per axis") |
| "Each validation geometry is probed with retained displacement directions from the nine validation classes of Table ST03 (defined in Appendix G.2): imposed polynomial and multiscale displacements, equilibrated nodal forces and consistent tractions on all faces or on a single face, spring-supported responses and traces induced by a neighbouring cell." | P2 S1-S3. "directions" → "test displacements", "validation classes" → "load classes" (as in the rewritten Table ST03 title), "nodal forces" → "nodal point loads", "consistent tractions" → "traction loads", "traces induced by a neighbouring cell" → "displacements imposed by a neighbouring cell" (brief 3.1). |
| "Consistent tractions (64 directions per geometry), which represent surface loads and neighbour tractions, are the principal loading class;" | P2 S4-S5 |
| "equal nodal forces also load weakly supported nodes and serve as a stress test of the stabilised problem:" | P2 S6 ("Equal nodal point loads also act on weakly connected nodes"). Weakly connected nodes are defined in Section 3.2 (U04), so no definition is repeated here. |
| "in the five cells of Table ST04 the ghost penalty carries on average less than 0.05% of the exact field energy under consistent tractions, but 49–81% under equal nodal forces." | P2 S7 |
| Table 1 title | unchanged |
| Table 1 rows | Content unchanged; terms updated: "Retained space" → "Retained DOFs"; "box-face nodes ... on the box faces" → "cell-face nodes ... on the cell faces", with "(cut-plane elements)" naming the second part; "partial subcells" → "partially filled subcells"; "Principal interior coarse space" → "Principal coarse space" (the cell still says "restricted to the interior DOFs"); "Thickness-difference step" → "Finite-difference step of the thickness derivative", "with fixed active DOFs and ghost contribution" → "with the active DOFs and the ghost-penalty contribution held fixed"; "Correction of NICE" → "Two-grid correction of NICE", "\(Q_1(17)\) coarse-grid Galerkin correction" → "coarse-grid correction with \(Q_1(17)\)" (brief 3.1: "Galerkin" only in the appendices); "Arithmetic" → "Floating-point precision", "stiffness actions" → "stiffness products", "timed route" → "timed analyses"; Section 4.3 → 3.3. |
| "The mesh, material and stabilisation entries apply to all 80 validation geometries; lengths are relative to the unit box and the modulus is normalised." | P3 S1-S2 |
| "Coarse dimensions and smoothing counts of other correction sequences are reported with their comparisons." | P3 S3 |
| "Four variants are compared (Table 2)." | P4 S1 ("Three variants are used in the main text, and two of them are trained networks (Table 2)"). The count changes because two variants move to Note S8 (brief 3.3). |
| "The base network was trained without correction." | P4 S2 |
| "Three networks continue it for 15,000 training steps: with the complete correction \(\mathcal W\) of Sections 4.3 and 4.4 inside the training loop (NICE, Section 4.5), ..." | P4 S3 (NICE part; Sections 3.3, 3.4, 3.5) |
| "... with eight smoothing steps and no coarse-grid correction in every training step (Smoothing-trained), or without correction (Uncorrected, referred to in the text as the Uncorrected continuation)." | Moved to SUPPLEMENT ADDITIONS (Note S8, P1 S1-S3). Main text keeps only the pointer in P6 S4. |
| Table 2 title | unchanged |
| Table 2 header "Training pool (geometries)" | "Training set (geometries)", as in Table ST01 |
| Table 2 row NICE | unchanged |
| Table 2 row Base network, "40,000 training steps, selected at 30,000" | "40,000 training steps; parameters of step 30,000 selected" (Table ST01: "the parameters of step 30,000 were selected") |
| Table 2 rows Smoothing-trained and Uncorrected (every cell: "Base network continued for 15,000 training steps", 591, "8 smoothing steps", "8 smoothing steps"; "Base network continued for 15,000 training steps", 591, None, None) | Moved to SUPPLEMENT ADDITIONS (Note S8 table), every cell value kept. |
| "The base network with the correction \(\mathcal W\) applied at deployment, without retraining, is also evaluated and denoted 'Base network, corrected' (Sections 5.3, 5.5 and 5.6)." | P5 S1-S2, kept directly after Table 2 (unit instruction). Label 'Base network, corrected' → 'Base network + correction' (brief 3.3). |
| "All variants share the architecture of Section 4 (about \(6\times10^5\) trainable parameters; Table ST15)." | P6 S1 (Section 3.2; one parenthetical reference). Repeated in Note S8 for the two further continuations. |
| "The three continuations draw from the same set of 591 geometries, which contains 304 of the base network's 305 training geometries, and see the same geometries in the same order, at most 153 of the 591 in their 15,000 steps (Table ST01)." | P6 S2-S5: the facts about NICE (S2-S3) and a pointer to Note S8 for the other two continuations, which use the same set and the same order (S4-S5). The full sentence is also in Note S8 (P2 S2-S4). |
| "Twenty of the 80 validation geometries (6 uncut, 14 cut) were used for checkpoint selection (Table ST01);" | P7 S1 |
| "statistics are therefore also given for the 60 geometries outside checkpoint selection." | P7 S2 |
| "The target cells of the two-cell assemblies of Section 5.6 belong to these 20 selection geometries;" | P7 S3 ("test cells", brief 3.1) |
| "nine further cells outside checkpoint selection are assembled there as an additional check." | P7 S4 |
| "The principal variant was designated after all variants had been compared on the 80 validation geometries and the two-cell configurations (Table ST01);" | P7 S5. Names NICE as the principal variant and adds "including the two of Supplementary Note S8", so that "all variants" keeps the scope it had in the source (Table ST01: "NICE was designated the principal variant after all variants had been compared"). |
| "the sixteen cells of the lattices of Section 5.8 entered no selection step." | P7 S6 |
| "Population statistics average the directional energy error within each geometry and loading class and weight geometries equally (Appendix A.1), so the population maximum is the largest geometry mean." | P7 S7-S8 ("energy error", "test displacements", "load class", brief 3.1) |
| Table 3 title and caption | Title: "exact condensation" → "exact static condensation". Caption split into five sentences; "a cell's eight-component thickness-sensitivity vector" → "the eight-component thickness-sensitivity vector of a cell"; "over the shared vertex parameters" → "with respect to the shared vertex parameters"; "target cell" → "test cell". |
| Table 3 rows | All values unchanged. Terms: "cut-face tractions" → "cut-face traction loads" (row 2, twice); "Plate clamped through its cut band" → "Plate clamped at its cut-plane elements" (brief 3.1). CN "配置" → "构型", as in the rewritten supplement (S1). |

### Old 5.2 "Verification of the discrete reference and of the corrected operator" (old lines 422-428) → new 5.2 "Verification of the reference model and of the deployed operator"

| Old sentence | Where it is now |
|---|---|
| Heading | `### 5.2. Verification of the reference model and of the deployed operator` (brief Section 4) |
| "The \(n=32\) reference was verified by refinement on seven cells, to \(n=40\) for U1 and to \(n=48\) for M1 and M2 (Figure S01), and to \(n=64\) for H1, H2 and two further validation cells (Table ST10; Supplementary Note S1):" | P8 S1-S3. "(Table ST10; Supplementary Note S1)" → "(Table ST10)"; Table ST10 lies in Note S1 (brief rule 4). |
| "apart from the heavily cut H2, the compliance changes by at most 0.28% and the thickness sensitivities by at most 0.44%, whereas in H2 the changes reach about 1% in compliance and 3.6% in sensitivity." | P8 S4-S5 ("refinement changes ...") |
| "On uncut cells and assemblies of them, the compliance of the discrete model agrees with an independent body-fitted quadratic-tetrahedral discretisation to 0.05–0.25% (Supplementary Note S1)." | P8 S6 |
| "Varying the ghost-penalty coefficient between \(10^{-5}\) and \(10^{-3}\) and refining the volume integration change compliance and sensitivities by at most 0.12%," | P8 S7 ("each change"; Note S1: ghost-penalty variation at most 0.053% / 0.12%, integration at most 0.010%) |
| "and the finite-difference stiffness derivative and its step are verified in Supplementary Note S1 (Figure S01(c))." | P8 S8 |
| "All comparisons that follow are made against this discrete reference (Section 2.1)." | P8 S9 |
| "On five cells (U2, M1, M2, H1 and H2), the deployed operator is symmetric and satisfies the work–energy identity to relative deviations of at most \(3\times10^{-8}\)," | P9 S1-S2. S2 states the identity in one sentence (Table ST04: "maximum relative difference between returned work and field energy"). |
| "gives the rigid-body modes zero energy to round-off" | P9 S3 |
| "and reproduces the training-time field to \(1.1\times10^{-7}\) (Table ST04)." | P9 S4 ("agrees with the field computed by the training implementation"; Table ST04 column "Deployed vs training field") |
| "The nonexpansiveness of Section 4.3 is guaranteed by \(b\ge\lambda_{\max}(D^{-1}A)\)." | P10 S1 ("The Chebyshev smoothing of Section 3.3 is guaranteed not to increase the energy error when ..."; brief 3.1, nonexpansive) |
| "On all 80 validation geometries the power-iteration endpoint exceeds the converged largest eigenvalue by 2.1–5.0% (Appendix D)," | P10 S2 ("the endpoint \(b\) obtained from the power iteration") |
| "whereas the guaranteed Gershgorin bound is 4.7 to 19.4 times larger (median 18.4) and would place the smoothing interval far above the actual spectrum." | P10 S3-S4 ("4.7 to 19.4 times larger than \(b\)"; Appendix D: "4.7–19.4 times the operational endpoint") |
| "The spectral condition is thus supported numerically on these geometries; it was not checked for the cells of the lattices and optimisations of Sections 5.8–5.10, which use the same power-iteration estimate." | P10 S5-S6 |
| "Symmetry, positive semidefiniteness and \(\widehat S\succeq S\) do not depend on it." | P10 S7 |

## (b) Cross-references renumbered (each occurrence mapped once)

| Old | New | Place |
|---|---|---|
| Proposition 2 | Proposition 3 | Section 5 intro, P0 S4 |
| Section 4.3 | Section 3.3 | Table 1, row "Smoothing interval" |
| Sections 4.3 and 4.4 | Sections 3.3 and 3.4 | 5.1 P4 S3 |
| Section 4.5 | Section 3.5 | 5.1 P4 S3 |
| Section 4 ("the architecture of Section 4") | Section 3.2 | 5.1 P6 S1. The map gives Section 3; the network architecture is in new 3.2 (old 4.2), the more specific reference (brief rule 4). |
| Section 4.3 ("The nonexpansiveness of Section 4.3") | Section 3.3 | 5.2 P10 S1 |
| Sections 5.9 and 5.10 (one parenthesis) | Section 5.9; Section 5.10 | Section 5 intro, P0 S6 and S7 (one reference per sentence) |
| Supplementary R1 | Supplementary Section R1 | 5.1 P1 S6, as in the rewritten supplement |
| (Table ST10; Supplementary Note S1) | (Table ST10) | 5.2 P8 S3 |
| (none) | Supplementary Note S8 | 5.1 P6 S4 and P7 S5 (new pointers, brief 3.3 and unit instruction) |
| Sections 5.1-5.10, 2.1; Eq. (1); Tables 1, 2, 3, ST01, ST02, ST03, ST04, ST10, ST15; Figures 4, S01, S01(c); Appendices A.1, D, F.2, G.2; Supplementary Note S1 | unchanged | throughout |

## SUPPLEMENT ADDITIONS

Destination: the new **Supplementary Note S8. Further continuations of the base network** (brief Section 6). This block comes from old Section 5.1 (old lines 392, 399-400 and 405). It introduces the two continuations, so it should open the note; the units for old Sections 5.3-5.6 add their results after it. The small table carries no label; see open question 2.

### EN

## Supplementary Note S8. Further continuations of the base network

Besides NICE, two further networks continue the training of the base network for 15,000 training steps. The Smoothing-trained variant applies eight smoothing steps without coarse-grid correction in every training step. The Uncorrected continuation is trained without correction. The table below gives their training and evaluation settings in the format of Table 2 of the main text.

| Variant | Network parameters | Training set (geometries) | Correction in training | Correction at evaluation |
| --- | --- | ---: | --- | --- |
| Smoothing-trained | Base network continued for 15,000 training steps | 591 | 8 smoothing steps | 8 smoothing steps |
| Uncorrected continuation | Base network continued for 15,000 training steps | 591 | None | None |

Both continuations use the network architecture of Section 3.2, with about \(6\times10^5\) trainable parameters (Table ST15). The three continuations, NICE, the Smoothing-trained variant and the Uncorrected continuation, draw from the same set of 591 geometries. This set contains 304 of the 305 training geometries of the base network. The three continuations see the same geometries in the same order, at most 153 of the 591 in their 15,000 training steps (Table ST01).

### CN

## 补充说明 S8. 基础网络的其他继续训练变体

除 NICE 外，另有两个网络以基础网络为初值继续训练 15,000 步。平滑训练变体在每个训练步中施加八步光滑，不进行粗网格校正。未修正延续在训练中不施加修正。下表按正文表 2 的格式列出二者的训练与评估设置。

| 变体 | 网络参数 | 训练集（几何数） | 训练中的修正 | 评估时的修正 |
| --- | --- | ---: | --- | --- |
| 平滑训练 | 基础网络继续训练 15,000 步 | 591 | 8 步光滑 | 8 步光滑 |
| 未修正延续 | 基础网络继续训练 15,000 步 | 591 | 无 | 无 |

两个变体均采用第 3.2 节的网络结构，可训练参数约 \(6\times10^5\) 个（表 ST15）。NICE、平滑训练变体和未修正延续这三个继续训练变体取自同一个含 591 个几何的集合。该集合包含基础网络 305 个训练几何中的 304 个。三者按相同顺序使用相同的几何，在 15,000 个训练步中至多用到这 591 个几何中的 153 个（表 ST01）。

## (d) Open questions

1. **Table 2 title.** The title "Variants compared in the numerical examples" is kept. Table 2 now lists the two trained networks (NICE, Base network), and the third main-text variant, Base network + correction, is defined in the sentence directly after the table, as in the source. The rewritten supplement (S1) says "Variant labels follow the main text (Table 2). The two variants not listed there, the Uncorrected continuation and the Smoothing-trained variant, are evaluated in this supplement." Base network + correction is also not a row of Table 2, but it was not a row in the old table either. If the assembler wants an exact statement, S1 could read "Variant labels follow the main text (Section 5.1, Table 2)".
2. **Label of the Note S8 table.** The two-row table in the supplement additions has no label, because S1 and S2 report a collision for "ST08b" (old main-text Table 5 and the existing ST08b sub-table). The assembler can give it a label once the S8 numbering is fixed, or leave it unnumbered inside the note.
3. **Old "Section 4" for the architecture** is mapped to Section 3.2, not Section 3 (see (b)). If the assembler applies the map literally, "Section 3" is also correct.
4. **Mention of the two further continuations in P7 S5.** The designation sentence keeps "including the two of Supplementary Note S8". Without it, "all variants" in the main text would read as the three main-text variants, which narrows the source statement. This is the second of two pointer sentences to Note S8 in this unit (P6 S4 is the first).
5. **Plate name in Table 3.** "Plate clamped through its cut band" became "Plate clamped at its cut-plane elements" (CN 在切割平面单元处固支的板). The Section 5.10 writer should use the same name for this plate.
6. **CN for "configuration".** This unit uses 构型 (Table 3 and P7), as the rewritten supplement S1 does; the rewritten S2 uses 配置 in Table ST08. The assembler should unify the term.
7. **0.074%.** The Section 5 introduction gives the mean energy error of NICE over the 80 validation geometries as 0.074% without naming a load class, as the source does. Table ST03 shows that this is the traction-load (force_c) mean, 0.0737%. If the author wants the class named, the clause "under traction loads" can be added without any new number.
8. **"Deployed operator"** is used in the 5.2 title (brief Section 4) and in P9 without a separate definition; P9 S4 contrasts it with the training implementation, as Table ST04 does.
