# U09 notes: Section 5.6 Two-cell assemblies (Figures 9, 10, 11; old Table 5) and Section 5.7 Ablation (Figure 12)

Source (old numbering): MANUSCRIPT_EN.md / MANUSCRIPT_CN.md lines 486-533 (old 5.6 "After assembly: compliance follows the energy share, the sensitivity does not", old Table 5, old 5.7 "Ablation: restricting the box faces costs sensitivity before compliance").

New blocks (EN and CN line-aligned, 38 lines and 19 blocks each): heading `### 5.6.`; P1 set-up of the two-cell assemblies; Figure 9 and caption; P2 NICE on the seven test cells; P3 nine held-out cells; P4 effect of the correction; P5 cut-face traction loads; P6 compliance and sensitivity must be checked separately; Figure 10 and caption; P7 energy fraction and the U1/x example; Figure 11 and caption; heading `### 5.7.`; P8 ablation set-up and compliance; P9 ablation sensitivity; Figure 12 and caption. Old Table 5 and its note are no longer in the main text (moved to Supplementary Note S8, see SUPPLEMENT ADDITIONS).

Length (EN words; tables excluded, captions included, inline mathematics counted as one word): old source 1533, new 1705. The increase comes from splitting long sentences, from writing out "target-face"/"neighbour-face" loads, and from the restatement of the correction argument with the base network and a pointer to Note S8 (P4).

Self-check (Python, on the final files):
- (a) Every number of the old EN source is in the new EN text, except the numbers listed in table (a) below as moved to Note S8: all cells of old Table 5 (1.06, 2.47, 0.144, 0.636, 0.0076, 0.0133, 0.643, 0.0023, 4.89, 0.00109, 3.15, 7.3×10⁻⁵, 0.598, 0.0013, 3.44, 11.4, 1.14, 4.43, 0.0481, 0.145, 0.311) and the numbers of the removed sentences on the Uncorrected continuation and the Smoothing-trained variant (11.4, 4.1, 3.15–4.43, 14.1, 0.00109, 3.15, 0.13, 0.0481, 0.145, 3.44, 11.4, Section 5.4). 0.598 is kept once in the main text (P7). The reference numbers that dropped are "Table 5" (four occurrences, see (b)) and "Table 2" (old Figure 10 caption, see (b)).
- (b) The new EN text contains no number absent from the old source except "Section 5.1" (new reference in P1, see (b)) and the renumbered "Proposition 4(b)" (the digit 4 also occurs in the source).
- (c) EN and CN contain the same numbers with the same counts in every block, the same inline mathematics and the same citation links.
- (d) No EN sentence above 35 words (96 sentences, mean 17.6). The checker reports 44 words for the Figure 11 caption title only because it joins the bold title with the next sentence; the two sentences have 16 and 28 words. No CN sentence above 80 characters (97 sentences, mean 35.6), except the sentence in P8 that carries the three citation links (38 characters without the links).
- (e) No banned word or old term in EN or CN (checked: target, energy share, box, cut band, consistent traction, nodal force, directional, strat-, field-based, nonexpansive, admissible, Ritz, Galerkin, extension, retained space, Schur, em dash, question mark, Uncorrected, Smoothing-trained, Table 5, "Proposition 3(b)"; CN 目标胞元, 能量份额, 边界面, 切割带, 一致面力, 分层, 场基, 未校正, 平滑训练, 正是, 值得注意的是, 换言之, 把, 当作, 总是, 其实, 表 5 and others). Remaining EN hits of "direction" are "three Cartesian directions", "x, y and z directions" and "the directions of the thickness-sensitivity vectors", which are geometric directions and not the old term for test displacements.

## (a) Sentences and numbers removed or moved

### Old 5.6 (old lines 486-522) → new 5.6 "Two-cell assemblies: compliance and sensitivity"

| Old sentence | Where it is now |
|---|---|
| Heading "After assembly: compliance follows the energy share, the sensitivity does not" | `### 5.6. Two-cell assemblies: compliance and sensitivity` (brief Section 4) |
| "Do the small single-cell energy errors of Sections 5.3–5.5 yield accurate compliance and local sensitivity once cells are assembled?" | P1 S1, as a statement ("This section checks whether ...") (brief 1: no rhetorical questions) |
| "The two-cell assemblies join a learned target to an exact, uncut neighbouring cell whose thickness is continuous across the interface (Figure 9)." | P1 S2 ("test cell", brief 3.1) |
| "The seven targets from the selection geometries are U1, U2, L1, M1, M2, H1 and H3, each in configurations x and y; 'M1/x' denotes target M1 in configuration x." | P1 S3-S4. "selection geometries" written out as "the geometries used for checkpoint selection (Section 5.1)". |
| "Each configuration has six face loads, the three Cartesian traction directions applied separately to the target and neighbour faces." | P1 S5-S6 ("traction loads", "on the loaded face of the test cell and on that of the neighbouring cell", as in rewritten Note S4.2) |
| "For each load, compliance and each cell's thickness sensitivity, an eight-component vector, are compared with their exact counterparts, with the nodal load fixed at its base-design value (load normalisation and pair solver: Supplementary Note S4.2)." | P1 S7-S8 |
| "We use 3% as a common reference tolerance for compliance and for each cell's sensitivity vector, drawn as a 3% line in the figures, and report maxima over the six loads and, for sensitivity, over both cells." | P1 S9-S10 (impersonal voice) |
| "Three tractions on a target cut face are considered separately." | P1 S11 |
| Figure 9 title "Supports and loading of the two-cell assemblies." | "Supports and loads of the two-cell assemblies." |
| Figure 9 caption (a), (b) | unchanged in content; "face tractions act at \(y=0\)" → "the traction loads act on the faces at \(y=0\)" (the figure labels them "loaded faces, y = 0"); same for (b) |
| "Each target (T) and neighbour (N) face carries separate x-, y- and z-directed consistent-traction loads." | Caption S5 ("The loaded faces of the test cell (T) and of the neighbouring cell (N) each carry separate traction loads in the x, y and z directions.") |
| "Coincident box-node DOFs are shared across the interface; non-box cut-band DOFs remain local." | Caption S6 ("coincident DOFs of the cell-face nodes"; "the DOFs of cut-plane elements that do not lie on the cell faces remain local to their cell", wording of rewritten Note S2) |
| "Cut targets also receive three cut-surface tractions, analysed separately." | Caption S7 |
| "Figure 10 compares the variants in these configurations." | P2 S1, names the two variants now drawn ("Figure 10 compares Base network + correction and NICE ..."). |
| "NICE stays below both 3% lines in all fourteen configurations of the seven selection cells: its largest compliance error over the six face loads is 0.056% (M1/x), and its largest sensitivity error, taken over both cells, is 0.67% (U1/y)." | P2 S2-S4 |
| "The directions of the thickness sensitivity vectors are reproduced more closely than their magnitudes: in these fourteen configurations, NICE's vectors have cosines of at least 0.999998 with the exact ones (Supplementary Note S4.3)." | P2 S5-S6 |
| "Nine further validation cells outside checkpoint selection were assembled in the same way: the five geometries with NICE's largest single-cell errors and one randomly chosen cell from each of the uncut, lightly, moderately and heavily cut strata." | P3 S1-S2 ("one from each of the four cut-severity groups"; the four groups are named in Section 5.1) |
| "All eighteen configurations stay below both 3% lines, with maxima of 0.27% in compliance and 1.49% in sensitivity, both on the geometry with the largest single-cell error (Supplementary Table ST08b)." | P3 S3-S4. "Supplementary Table ST08b" here is the existing held-out table (sub-table of ST08), unchanged. |
| "The five largest-error geometries account for all compliance errors above 0.02%; the four stratum cells stay below 0.011% and 0.22%." | P3 S5-S6 ("the four randomly chosen cells"; "compliance errors ... below 0.011% and the sensitivity errors below 0.22%", checked against Table ST08b: RU-RH compliance ≤ 0.0101, sensitivity ≤ 0.218) |
| "The Uncorrected continuation exceeds the 3% sensitivity line in four of the eleven configurations on which it was evaluated (up to 11.4%; Table ST08) and, on M1, also the compliance line (up to 4.1%)." | Moved to SUPPLEMENT ADDITIONS (Note S8, P2 S1-S3). Replaced in the main text by P4 S2-S3 (see below). |
| "Smoothing-trained reduces these errors but still exceeds the line in the same four configurations (3.15–4.43%): smoothing without the coarse-grid correction leaves the slowly damped low-mode error of Section 5.4, consistent with the sensitivity errors that remain in U1 and M1." | Moved to Note S8 (P2 S4-S6). The main text keeps one pointer sentence (P4 S3: "Continuing the training of the base network without the correction, or with smoothing alone, also leaves sensitivity errors above the 3% line on U1 and M1 (Supplementary Note S8)."). Checked: Uncorrected and Smoothing-trained exceed 3% sensitivity exactly on U1/x, U1/y, M1/x, M1/y (Table ST08). |
| "'Base network, corrected' also stays below both lines (largest sensitivity error 0.95%, U1/x), so the correction is what brings the assembled responses below them." | P4 S1 and S4 ("Base network + correction", brief 3.3). The comparison that the conclusion needs is restated with the base network in P4 S2: "Without the correction, the base network exceeds the 3% sensitivity line on U1 and M1 (Table ST08)." No number is quoted; Table ST08 gives Base network sensitivity maxima of 5.830 (U1/x), 5.390 (U1/y) and 12.153 (M1/x), and below 3% on U2/x, H1/x, H1/y and M2/x (brief 3.3: restate with the remaining variants where the supplement contains the numbers). The sentence is also repeated in Note S8 (P2 S7-S8) for context. |
| "Under tractions on the target cut face, NICE reaches at most 0.10% in compliance and 0.16% in sensitivity," | P5 (with "(Table ST09)", which carried the whole old sentence) |
| "whereas the Uncorrected continuation exceeds the 3% line in four of its seven configurations, up to 14.1% in sensitivity (Table ST09)." | Moved to Note S8 (P3). No pointer in the main text, because the NICE result in P5 does not depend on it. |
| "Table 5 shows why both quantities are needed." | P6 S1 ("An accurate compliance does not guarantee an accurate local sensitivity, so the two quantities have to be checked separately."). The reference to Table 5 is removed (unit instruction: the argument rests on NICE and Figure 11). |
| "Without the complete correction, an accurate compliance does not guarantee an accurate local sensitivity: under a neighbour-face load on U1/x, the Smoothing-trained compliance error is 0.00109% while the target-cell sensitivity error is 3.15%, with the target carrying 0.13% of the exact assembled energy." | General claim kept in P6 S1. The Smoothing-trained example (0.00109%, 3.15%, 0.13%) moved to Note S8 (P4 S1-S3). The main text keeps one pointer sentence, P6 S5: "For the variants without the complete correction, the sensitivity error under this load exceeds the 3% line, while the compliance error stays far below it (Supplementary Note S8)." Checked for this load (U1/x, N-z): Uncorrected 0.0023 / 4.89, Smoothing-trained 0.00109 / 3.15 (old Table 5), base network 0.002827394% / 5.82992% (note to Table ST08). |
| "NICE reduces both errors: the target-cell sensitivity error of that load falls to 0.598%," | 0.598% stays in P7 S5 (the NICE example). The comparison "reduces both errors ... falls to" moved to Note S8 (P4 S4). |
| "and on M1/x under the target-face y load NICE gives 0.0481% and 0.145% instead of the Uncorrected continuation's 3.44% and 11.4%." | Moved to Note S8 (P4 S5-S6). The values are also in Table ST08b of Note S8. |
| "The separation remains visible in NICE, whose sensitivity error exceeds its compliance error by a factor of about two on H1 and by four orders of magnitude for the U1 target, whose energy share is small, but both lie far below the 3% line." | P6 S2-S4. The loads are named ("the x traction load on the test-cell face of H1/x", "the z traction load on the neighbour face of U1/x"), because without old Table 5 "on H1" could be read as the H1 maxima of Table ST08, whose ratio (0.098 / 0.0076) is not about two. |
| Figure 10 title | unchanged |
| "The target cell is represented by a learned substructure and the neighbour by exact condensation." | Caption S1 ("exact static condensation") |
| (new) | Caption S2: "The figure shows NICE and Base network + correction, which is the base network with the correction applied at deployment, without retraining." This replaces the old last sentence (below) and states the content of the regenerated figure (unit instruction). |
| "(a,b) Maximum errors over the six face loads in each configuration; the sensitivity error is also maximised over both cells." | Caption S3 |
| "(c,d) Compliance and target-cell sensitivity errors of the individual face loads on M1/x and U1/x; filled markers denote target-face loads and open markers neighbour-face loads." | Caption S4-S5 |
| "Dashed lines mark the 3% lines." | Caption S6 ("Dashed lines mark 3%.") |
| "The base network is not shown, and not every variant was evaluated on every configuration; all results are in Table ST08." | Caption S7: "Table ST08 gives all results, including those of the base network and of the two variants of Supplementary Note S8." "Not every variant was evaluated on every configuration" is removed from the caption: the two variants shown were both evaluated on all fourteen configurations (Table ST08), and Table ST08 already states "Combinations without a row were not evaluated." |
| "Variants as in Table 2; 'Base network, corrected' is the base network with the correction applied at deployment, without retraining." | Caption S2 (definition kept); "Variants as in Table 2" removed, because Base network + correction is not a row of Table 2 and is defined in the caption itself. |
| Old Table 5 title, all three rows, and its note ("T and N identify the loaded face of the target or neighbour; x, y and z give the traction direction. The last column gives the target cell's share of the exact assembled energy.") | Moved in full to Note S8 as Table ST08b (SUPPLEMENT ADDITIONS), every cell value unchanged; column headers renamed ("Test cell / configuration", "Uncorrected continuation", "Energy fraction of the test cell"). |
| "The target's energy share (Table 5) explains why the compliance error can be orders of magnitude smaller than the sensitivity error." | P7 S1, without "(Table 5)"; the values needed are given in P7 itself. |
| "Under the neighbour-face z load in U1/x, NICE's target cell carries 0.127% of the exact assembled energy and has a local energy error of 0.0575%;" | P7 S2-S3 |
| "their product \(\beta=w\varepsilon\) gives a relative compliance bound of \(7.33\times10^{-5}\)%, close to the observed \(7.31\times10^{-5}\)%," | P7 S4 |
| "whereas the target-cell sensitivity error is 0.598%, about 8,200 times the compliance error." | P7 S5 |
| "Two factors make up this ratio." | P7 S6 |
| "The compliance error is relative to the compliance of the whole assembly, so the local error enters it weighted by \(w\), a factor \(1/w\approx780\)." | P7 S7 |
| "The sensitivity error is relative to the target's own sensitivity, which scales with the target's energy as its error does, so that \(w\) cancels;" | P7 S8-S9 |
| "at fixed retained displacement the energy error controls it only at the order \(\sqrt\varepsilon\) (Proposition 3(b)), and here it is about ten times \(\varepsilon\)." | P7 S10-S11 (Proposition 4(b)) |
| "Figure 11 shows this weighting for NICE over all loads; the other four variants follow the same relations (Figure S04)." | P7 S12-S13. "the other four variants" written out as "The base network, Base network + correction and the two variants of Supplementary Note S8", so that the main text names only its three variants. |
| Figure 11 title "The target's energy share scales down the compliance error but not the sensitivity error." | "The energy fraction of the test cell scales down the compliance error but not the sensitivity error." |
| "NICE in the fourteen configurations of the seven selection cells, one point per load: 84 face loads (filled markers) and 30 cut-surface loads (open markers); the maxima over the loads are given in Tables ST08 and ST09." | Caption S1-S2 ("seven test cells") |
| "(a) Compliance error against \(\beta=\sum_mw_m\varepsilon_m\), with the energy shares \(w_m=q_m^TS_mq_m/C\) and energy errors \(\varepsilon_m=\dots\), evaluated at the exact assembled retained displacement; only the learned target contributes to \(\beta\)." | Caption S3-S4 ("energy fractions"; mathematics unchanged) |
| "The line denotes equality." | Caption S5 |
| "For every load the compliance error lies between 0.963 and 1.004 times \(\beta\); the five ratios above 1 correspond to at most \(7.8\times10^{-10}\) of the compliance, whereas the residual work of the pair solves reaches \(1.8\times10^{-8}\) of the compliance (Supplementary Note S4.3)." | Caption S6-S7. "correspond to an excess of at most ..." makes explicit what the old CN text already said (超出量至多为柔度的 ...). |
| "(b) Compliance error and (c) target-cell sensitivity error against the target's exact energy share \(w\); solid lines are least-squares fits in logarithmic coordinates." | Caption S8 |
| "The circled load is U1/x under the neighbour-face z traction (Table 5)." | Caption S9, without "(Table 5)"; the values of this load are given in P7. |
| "The other four variants are shown in Figure S04." | Caption S10 ("Figure S04 shows the corresponding results of the other variants.") |

### Old 5.7 (old lines 524-532) → new 5.7 "Ablation: restricting the cell-face displacements"

| Old sentence | Where it is now |
|---|---|
| Heading "Ablation: restricting the box faces costs sensitivity before compliance" | `### 5.7. Ablation: restricting the cell-face displacements` (brief Section 4) |
| "Keeping the complete retained space matters more for the local sensitivity than for the compliance." | P8 S1 ("Keeping all retained DOFs") |
| "To isolate its role, pairs in configuration x are solved with exact cell operators while the box-face displacements of each cell are restricted to tensor Bernstein polynomials of degree \(r\); the non-box cut-band DOFs remain unrestricted." | P8 S2-S3 ("the exact condensed stiffness of each cell"; "cell-face displacements"; "tensor-product Bernstein polynomials"; "The DOFs of the cut-plane elements that do not lie on the cell faces remain unrestricted") |
| "The ablation restricts only the retained representation of the present cells at fixed cell size;" | P8 S4 ("the representation of the retained displacements") |
| "reduced-boundary learned substructures control the resulting error by partition refinement, boundary enrichment or oversampling [Huang et al. (2024)], [Guo et al. (2026a)], [Guo et al. (2026b)] and are not reproduced here (Supplementary Note S2)." | P8 S5-S6, citations unchanged. Words only, no numerical comparison (brief 2). |
| "With every box face restricted, degree one gives compliance errors of 78–85% on U1, M1, M2 and H1, decreasing to about 4% at \(r=5\) and 0.49–0.74% at \(r=8\), where the H1 pair has 14,001 retained DOFs instead of 32,991 (Figure 12)." | P8 S7-S9 |
| "The local sensitivity converges more slowly: at \(r=8\) the target-cell sensitivity error is 1.3–4.5% under target-face loads and 24–64% when neighbour-face loads are included." | P9 S1-S2 |
| "Restricting only the shared interface removes most of the compliance error (0.17–0.44% at \(r=3\) on U1, M1 and M2), but the sensitivity error under neighbour loads remains 8–21% at \(r=3\) and 2.2–3.4% at \(r=5\)." | P9 S3-S4 |
| "A restricted interface is thus adequate for compliance at low degree but not for local sensitivity under neighbour loads;" | P9 S5 |
| "keeping the complete retained space removes this component of the error, at the cost of the larger coarse model (Table ST11)." | P9 S6. "the larger coarse model" → "a larger assembled model" (the assembled system on the retained DOFs; same wording as Section 1.1, "enlarges the assembled model"). |
| Figure 12 title "Response errors caused by restricting box-face displacements." | "Response errors caused by restricting the cell-face displacements." |
| "Both cells of H1/x use exact operators and Bernstein degree \(r\) on every box face, with unrestricted non-box cut-band DOFs." | Caption S1-S2 |
| "(a) Number of retained DOFs; the dashed line denotes the 32,991 DOFs of the full representation." | Caption S3 |
| "(b,c) Maximum compliance and target-cell sensitivity errors over the three target-face loads or all six target- and neighbour-face loads; cut-surface tractions are excluded." | Caption S4-S5 |
| "Errors are relative to the full retained-space solution; horizontal lines mark 3%." | Caption S6 ("relative to the solution with all retained DOFs", as in rewritten Table ST11) |

## (b) Cross-references renumbered or changed (each occurrence mapped once)

| Old | New | Place |
|---|---|---|
| Proposition 3(b) | Proposition 4(b) | 5.6 P7 S10 (brief 5: old Proposition 3 → 4; part (b) is the \(\sqrt\varepsilon\) bound in the rewritten Section 4.2) |
| Table 5 ("Table 5 shows why ...") | removed | 5.6 P6 S1 (unit instruction: argument rests on NICE and Figure 11) |
| Table 5 (table itself) | Table ST08b in Supplementary Note S8 | SUPPLEMENT ADDITIONS (brief 5) |
| (Table 5) after "The target's energy share" | removed | 5.6 P7 S1 (values given in the same paragraph) |
| (Table 5) in Figure 11 caption | removed | Figure 11 caption S9 |
| Table 2 ("Variants as in Table 2") | removed | Figure 10 caption |
| (none) | Section 5.1 | 5.6 P1 S3, where "selection geometries" is written out as "the geometries used for checkpoint selection (Section 5.1)" |
| (none) | Table ST08 | 5.6 P4 S2 (base-network restatement) |
| (none) | Supplementary Note S8 | 5.6 P4 S3, P6 S5 (pointer sentences), Figure 10 caption S7, P7 S13 (in-text naming of the two further continuations) |
| (up to 11.4%; Table ST08) | moved with its sentence | Note S8 P2 |
| Section 5.4 (Smoothing-trained explanation) | moved with its sentence | Note S8 P2 |
| (Table ST09) in the Uncorrected clause | kept in main text P5 for NICE; repeated in Note S8 P3 | |
| Sections 5.3–5.5; Figures 9, 10, 11, 12, S04; Tables ST08, ST08b (held-out), ST09, ST11; Supplementary Notes S2, S4.2, S4.3 | unchanged | throughout |

## SUPPLEMENT ADDITIONS

Destination: **Supplementary Note S8. Further continuations of the base network** (brief Section 6), after the block contributed by U07 (introduction of the two continuations) and after the blocks of the units for old Sections 5.3-5.5. The sub-heading carries no number; the assembler can number it with the other S8 sub-sections. The table label follows the brief and the unit instruction (old Table 5 → ST08b); see open question 1 for the label collision.

### EN

### Two-cell assemblies

The two further continuations were also evaluated in the two-cell assemblies of Section 5.6. The configurations, loads and error measures are those of Section 5.6. Tables ST08 and ST09 list the maxima for every variant and configuration evaluated.

The Uncorrected continuation was evaluated in eleven configurations. It exceeds the 3% sensitivity line in four of them, with sensitivity errors up to 11.4% (Table ST08). On M1 it also exceeds the 3% compliance line, with compliance errors up to 4.1%. The Smoothing-trained variant reduces these errors, but it still exceeds the 3% sensitivity line in the same four configurations, with errors of 3.15–4.43%. Smoothing without the coarse-grid correction leaves the slowly damped error in the low modes described in Section 5.4. This is consistent with the sensitivity errors that remain in U1 and M1. Base network + correction stays below both 3% lines, with a largest sensitivity error of 0.95% on U1/x. The correction therefore brings the assembled responses below the 3% lines.

Under the traction loads on the cut face of the test cell, the Uncorrected continuation exceeds the 3% line in four of its seven configurations, with sensitivity errors up to 14.1% (Table ST09). Under the same loads, the errors of NICE are at most 0.10% in the compliance and 0.16% in the sensitivity.

Without the complete correction, an accurate compliance does not guarantee an accurate local sensitivity (Table ST08b). Under the z traction load on the neighbour face of U1/x, the compliance error of the Smoothing-trained variant is 0.00109%, whereas its test-cell sensitivity error is 3.15%. The test cell carries 0.13% of the exact assembled energy under this load. NICE reduces both errors, and its test-cell sensitivity error under this load is 0.598%. On M1/x under the y traction load on the test-cell face, NICE gives errors of 0.0481% in the compliance and 0.145% in the sensitivity. The Uncorrected continuation gives 3.44% and 11.4% under the same load.

### Table ST08b. Compliance and test-cell sensitivity under individual face loads

T and N identify the loaded face of the test cell or of the neighbouring cell; x, y and z give the traction direction. The last column gives the fraction of the exact assembled energy carried by the test cell.

| Test cell / configuration | Load | Uncorrected continuation: compliance / sensitivity (%) | Smoothing-trained: compliance / sensitivity (%) | NICE: compliance / sensitivity (%) | Energy fraction of the test cell |
| --- | --- | --- | --- | --- | ---: |
| H1/x | T-x | 1.06 / 2.47 | 0.144 / 0.636 | 0.0076 / 0.0133 | 0.643 |
| U1/x | N-z | 0.0023 / 4.89 | 0.00109 / 3.15 | 7.3×10⁻⁵ / 0.598 | 0.0013 |
| M1/x | T-y | 3.44 / 11.4 | 1.14 / 4.43 | 0.0481 / 0.145 | 0.311 |

### CN

### 两胞元装配

第 5.6 节的两胞元装配同样评估了这两种继续训练变体。构型、载荷和误差度量均与第 5.6 节相同。各变体在所评估构型上的最大误差见表 ST08 和表 ST09。

未修正延续在十一种构型中经过评估，其中四种构型的灵敏度误差超过 3% 线，最高为 11.4%（表 ST08）。在 M1 上，未修正延续的柔度误差也超过 3% 线，最高为 4.1%。平滑训练变体降低了这些误差，但在同样的四种构型中，灵敏度误差仍超过 3% 线，为 3.15–4.43%。不含粗网格校正的光滑会留下第 5.4 节所述衰减缓慢的低阶模态误差。这与 U1 和 M1 中残留的灵敏度误差相一致。基础网络加修正低于两条 3% 线，最大灵敏度误差为 0.95%，出现在 U1/x。因此，修正使装配响应降至 3% 线以下。

在被测胞元切割面上的面力载荷作用下，未修正延续在其七种构型中有四种超过 3% 线，灵敏度误差最高为 14.1%（表 ST09）。在相同载荷下，NICE 的柔度误差至多为 0.10%，灵敏度误差至多为 0.16%。

若不施加完整的修正，柔度准确并不保证局部灵敏度准确（表 ST08b）。在 U1/x 相邻胞元表面的 z 向面力载荷下，平滑训练变体的柔度误差为 0.00109%，而被测胞元的灵敏度误差为 3.15%。在该载荷下，被测胞元承担装配结构精确应变能的 0.13%。NICE 使两项误差均有所降低，该载荷下被测胞元的灵敏度误差为 0.598%。在 M1/x 被测胞元表面的 y 向面力载荷下，NICE 的柔度误差和灵敏度误差分别为 0.0481% 和 0.145%。在同一载荷下，未修正延续的相应误差为 3.44% 和 11.4%。

### 表 ST08b. 单个面载荷下的柔度与被测胞元灵敏度

T 和 N 分别表示被测胞元和相邻胞元的加载面；x、y、z 表示面力方向。最后一列为被测胞元的应变能在装配结构精确应变能中的占比。

| 被测胞元 / 构型 | 载荷 | 未修正延续：柔度 / 灵敏度（%） | 平滑训练：柔度 / 灵敏度（%） | NICE：柔度 / 灵敏度（%） | 被测胞元应变能占比 |
| --- | --- | --- | --- | --- | ---: |
| H1/x | T-x | 1.06 / 2.47 | 0.144 / 0.636 | 0.0076 / 0.0133 | 0.643 |
| U1/x | N-z | 0.0023 / 4.89 | 0.00109 / 3.15 | 7.3×10⁻⁵ / 0.598 | 0.0013 |
| M1/x | T-y | 3.44 / 11.4 | 1.14 / 4.43 | 0.0481 / 0.145 | 0.311 |

## (d) Open questions

1. **ST08b label collision (needs a decision; also reported by S1 and S2).** The supplement already has `#### ST08b. Held-out two-cell configurations (NICE)`. The main text of this unit cites "Supplementary Table ST08b" only for that held-out table (P3, as in the source; CN 补充表 ST08b). Old Table 5 is never cited by label in the main text: P4, P6, the Figure 10 caption and P7 point to "Supplementary Note S8". In the supplement additions, old Table 5 is labelled "Table ST08b" as the brief and the unit instruction require; the label occurs twice there (EN and CN: the table heading and "(Table ST08b)" in the last paragraph). Suggested resolution: relabel the moved table "Table ST08c" in those two places per language. Nothing in the main text then has to change.
2. **Was the base network evaluated in the two-cell assemblies?** The unit instruction says that "the base network alone was not evaluated in the two-cell assemblies". The supplement shows otherwise: Table ST08 has seven Base network rows (U1/x, U1/y, U2/x, M1/x, H1/x, H1/y, M2/x), Table ST09 has four, Figure S04(a–c) plots 54 base-network loads, the note to Table ST08 gives its values for U1/x under the z traction load on the neighbour face, and `evidence/` holds seven `gate_v2L1_*` files. I therefore did not write that the base network was not evaluated. The Figure 10 caption says only which two variants the regenerated figure shows (it matches `figures/F05_assembly.png`, which already has the two variants) and that Table ST08 includes the base-network results. P4 S2 uses the base network, without numbers, to restate the correction argument (brief 3.3). If the author prefers not to mention the base network here, P4 S2 and the clause "including those of the base network and" in the Figure 10 caption can be deleted. The argument then rests on Base network + correction and the pointer to Note S8 (P4 S3).
3. **"selection geometries"** (P1) is written out as "the geometries used for checkpoint selection (Section 5.1)". The Section 5.1 reference is new; Section 5.1 (U07, P7) states that the test cells of Section 5.6 belong to these geometries.
4. **CN term for "configuration".** This unit uses 构型, as U07 does in the main text and Table 3. The rewritten supplement (S2: Tables ST08, ST08b, ST09, ST11) uses 配置. The supplement additions above also use 构型. The assembler should unify the term (U07 open question 6).
5. **Figure 11 image.** Panels (b) and (c) of `figures/F10_energy_share.png` print the fitted slopes ("fit, slope 1.10", "fit, slope −0.02"). The caption does not quote them, as in the source.
6. **"coarse model"** in the old 5.7 ("at the cost of the larger coarse model") is rendered as "a larger assembled model" (CN 装配模型的规模更大), to avoid a clash with the coarse grid of the two-grid correction.
7. **Sub-heading in Note S8.** The block above is headed "Two-cell assemblies" / "两胞元装配" without a number; the assembler numbers it with the other S8 blocks (from the units for old 5.1 and 5.3-5.5).

## VERIFIER

Checks run: sentence-by-sentence comparison of old EN/CN lines 486-533 with the new EN/CN; Python number diff old EN vs new EN (all differences are the old Table 5 cells, the removed Uncorrected/Smoothing-trained numbers, "Table 5"/"Table 2", Proposition 3 → 4, the new "Section 5.1" reference and "Note S8" pointers; every moved number is present in the SUPPLEMENT ADDITIONS in EN and CN with equal counts); number, inline-mathematics, citation-link and blank-line parity between new EN and new CN (38 lines each, identical); sentence lengths; banned-pattern and old-term grep. Sources opened to confirm moved or restated facts: SUPPLEMENTARY_EN.md Tables ST08, ST08b, ST09 and Figure S04; rewritten S2_EN.md (Tables ST08, ST08b, ST09, ST11a/b) and S4_EN.md (Figure S04); U06_EN.md (Proposition 4(b) is the \(\sqrt\varepsilon\) bound); U07_EN.md (Section 5.1 states that the test cells belong to the 20 selection geometries; definition of the Smoothing-trained variant); U01/U02/U11 (wording "cut-plane elements that do not lie on the cell faces", "enriching the interpolation", "assembled model"); figures F05_assembly.png (now shows only Base network + correction and NICE, all fourteen configurations), F09_assembly_loads.png, F10_energy_share.png.

Verified numbers against the supplement: Base network sensitivity maxima above 3% exactly on U1/x (5.830), U1/y (5.390), M1/x (12.153); Uncorrected and Smoothing-trained above 3% exactly on U1/x, U1/y, M1/x, M1/y (eleven and twelve configurations evaluated); Uncorrected cut-surface: four of seven above 3%, up to 14.053; Base network + correction max sensitivity 0.945 (U1/x); ST11a/b ranges 78–85, 0.49–0.74, 1.3–4.5, 24–64, 0.17–0.44, 8–21, 2.2–3.4 all match.

Changes made:
1. 5.6 P3 (EN, CN): "one from each of the four cut-severity groups" → "one from each of the uncut, lightly cut, moderately cut and heavily cut groups" (CN 在未切割、轻度切割、中度切割和重度切割四个分组中各随机选取一个), restoring the named groups of the source.
2. 5.6 P6 S2 (EN, CN): "NICE also shows this separation, although all its errors lie far below the 3% line." → "For NICE, the sensitivity error exceeds the compliance error, but both lie far below the 3% line." The old "also"/"this separation" referred back to the Smoothing-trained example, which now comes last in the paragraph; the new sentence states the source content ("whose sensitivity error exceeds its compliance error ... but both lie far below the 3% line") without a dangling reference. CN: NICE 的灵敏度误差大于柔度误差，但二者均远低于 3% 线。
3. Figure 10 caption (EN, CN): restored the qualifier of the old caption that not every variant was evaluated in every configuration, now attached to the variants that are only in Table ST08 ("It also includes the base network and the two variants of Supplementary Note S8, which were not evaluated in every configuration."). Confirmed in Table ST08: base network 7, Uncorrected 11, Smoothing-trained 12 of the 14 configurations.
4. 5.7 P9 (EN, CN): "removes most of the compliance error, which is 0.17–0.44% at r=3" split into two sentences so that 0.17–0.44% reads as the remaining compliance error (checked against ST11b: 0.169, 0.440, 0.261).
5. CN only: "NICE 均低于两条 3% 线", "全部十八种构型均低于", "基础网络加修正同样低于" → "……的误差均/同样低于" (subject was the variant or configuration, not the error); "与精确值进行比较" → "与精确值比较"; "七个被测胞元十四种构型" → "七个被测胞元的十四种构型"; P1 S1 rewritten to remove a pre-nominal modifier chain ("第 5.3–5.5 节中单个胞元的能量误差较小，本节检验这些误差在装配后能否……").
6. SUPPLEMENT ADDITIONS CN: two "进行了评估" constructions rewritten (第 5.6 节的两胞元装配同样评估了这两种继续训练变体；在十一种构型中经过评估).

Checked and left unchanged:
- P6 S1 states the general claim without "Without the complete correction"; the qualifier is kept in P6 S5 ("For the variants without the complete correction ..."), and the unit instruction asks that the main-text argument rest on NICE and Figure 11. Not a strengthening: NICE's own errors are stated to lie far below 3% in the next sentence.
- P4 S2 ("Without the correction, the base network exceeds the 3% sensitivity line on U1 and M1 (Table ST08)") is a restatement with a main-text variant whose numbers are in Table ST08 (brief 3.3); no number added.
- "enriching the boundary interpolation" (old "boundary enrichment") matches U01 ("enriching the interpolation") and the Bézier boundary enrichment described there; kept.

Unresolved (for the assembler):
- The unit instruction says the base network alone was not evaluated in the two-cell assemblies. Table ST08 (seven base-network rows), Table ST09 (four rows), the note to Table ST08 and Figure S04(a–c) (54 loads) show that it was evaluated in seven of the fourteen configurations. The text therefore does not claim it was not evaluated; Figure 10 shows only Base network + correction and NICE, and the caption says so. Writer's open question 2 stands.
- ST08b label collision (writer's open question 1): the main text cites "Supplementary Table ST08b" only for the existing held-out table; old Table 5 is labelled ST08b inside the Note S8 additions as instructed. One of the two needs a new label (for example ST08c for old Table 5, two occurrences per language in the additions).
- CN 构型 vs 配置 between main text and rewritten supplement (writer's open question 4).
