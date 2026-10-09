# U08 notes: Sections 5.3 Accuracy on single cells (Figure 5), 5.4 Where the network error lies (Figures 6, 7), 5.5 Correction of a fixed network (Table 4, Figure 8)

Source (old numbering): MANUSCRIPT_EN.md / MANUSCRIPT_CN.md lines 430-485 (old 5.3-5.5, Figures 5-8, Table 4).

New blocks (EN and CN paragraph-aligned, 25 blocks each): heading `### 5.3.`; P1 traction loads by cut-severity group; P2 nodal point loads, other classes, 60 geometries outside selection; P3 source of the improvement; Figure 5 + caption; heading `### 5.4.`; P4 overview; P5 spectrum; Figure 6 + caption; P6 spatial distribution; Figure 7 + caption; P7 \(\delta\), \(\kappa\); P8 linear and quadratic sensitivity terms; heading `### 5.5.`; P9 smoothing only; P10 coarse-grid correction; P11 initial fields; Table 4 title, table, note; Figure 8 + caption.

Length (EN words, table rows excluded, inline mathematics counted as one word): old source 1644, new 1942. The increase comes from splitting long sentences and from restating the comparisons against the base network.

Self-check (Python, on the final files):
- (a) Numbers of the old EN source missing from the new EN text: 1.03, 11.2, 6.33, 42, 102, 7.8 (all moved to Supplementary Note S8, see (a) and SUPPLEMENT ADDITIONS) and 9 (old Eq. (9), renumbered to Eq. (17), see (b)). Every other number of the source is in the new text.
- (b) Numbers in the new EN text absent from the old source: 1.142 and 12.702 (base-network group means under traction loads, copied from Supplementary Table ST03b, column "Base network force/force_c", rows Uncut and Heavy cut) and 71, 117 (computed ratio range, see (a) below). "17" occurs as the new tag of old Eq. (9) and, as before, inside \(Q_1(17)\).
- (c) EN and CN contain the same numbers, block by block, with identical counts; inline mathematics is identical; no citations occur in this unit.
- (d) No EN sentence above 35 words (longest 35, mean 16.8 words over 110 sentences). No CN sentence above 80 characters (longest 75, mean 35.4).
- (e) No banned word or old term in EN or CN. The only grep hit is the ordinal "first" ("The first two are", "The first two classes", "The first column", "first order"), which is not a priority claim.

## (a) Sentences and numbers removed or moved

### Old 5.3 "Single cells: about 0.1% mean energy error or less in every cut stratum" (old lines 430-440) → new 5.3 "Accuracy on single cells"

| Old sentence | Where it is now |
|---|---|
| Heading "5.3. Single cells: about 0.1% mean energy error or less in every cut stratum" | Heading "5.3. Accuracy on single cells" (brief Section 4). The claim of the old heading is kept as P1 S1: "In every cut-severity group, the mean energy error of NICE is about 0.1% or less (Figure 5)." (Group maxima of NICE: 0.109% under traction loads, 0.090% under nodal point loads.) |
| "NICE's stratum-mean errors are 64 to 102 times lower than those of the Uncorrected continuation," | Restated against the base network in P1 S2: "lower than those of the base network by factors of 71 to 117". Computation below. The original ratio 64 to 102 moves to Supplementary Note S8. |
| "although its error still grows about sevenfold from uncut to heavily cut cells (Figure 5)." | P1 S3. The reference (Figure 5) moved to P1 S1. |
| "Under consistent tractions, NICE's stratum means rise from 0.016% in uncut to 0.109% in heavily cut cells," | P1 S4 |
| "against 1.03% to 11.2% for the Uncorrected continuation (Supplementary Table ST03b)." | Moved to Supplementary Note S8. P1 S4 instead gives the base network's group means from Table ST03b, 1.142% to 12.702%, at the printed precision of that table. |
| "Over all 80 geometries the mean is 0.074% for NICE, 6.33% for the Uncorrected continuation and 6.89% for the base network," | P1 S5 keeps 0.074% (NICE) and 6.89% (base network). 6.33% moves to Supplementary Note S8. |
| "and the largest geometry means are 0.65%, 42% and 48.6%." | P1 S6 keeps 0.65% (NICE) and 48.6% (base network). 42% (Uncorrected continuation) moves to Supplementary Note S8. |
| "Taken direction by direction, 99% of NICE's errors over the 5,120 sampled consistent-traction directions lie below 0.69%, and the largest is 1.24% (Supplementary Table ST03)." | P1 S7 ("test displacements under traction loads") |
| "Under nodal forces, the base network's mean rises from 0.908% in uncut to 11.3% in heavily cut cells (largest 70.1%)," | P2 S1 ("nodal point loads"; "largest single-geometry mean is 70.1%") |
| "whereas NICE's stratum means stay between 0.013% and 0.090%;" | P2 S2 |
| "under the other loading classes NICE's means are of the same order (Supplementary Table ST03; per-geometry distributions in Figure S02)." | P2 S3 (Table ST03) and P2 S4 ("Figure S02 shows the distributions over the geometries."), split to keep one cross-reference per sentence. |
| "On the 60 geometries outside checkpoint selection, NICE's class means differ from those over all 80 by at most 0.005 percentage points;" | P2 S5 |
| "under consistent tractions its heavily cut stratum mean is higher (0.126% against 0.109%)," | P2 S6 |
| "and the five largest geometry means all belong to these 60 geometries (Supplementary Table ST03)." | P2 S7. "of NICE under traction loads" added as clarification; this is what Table ST03 states ("Under consistent tractions, NICE's stratum means on the 60 geometries ... and the five largest geometry means of the 80 ... all belong to them"). |
| "The intermediate variants locate this improvement." | Replaced by P3 S1 "Most of the improvement comes from the correction itself." (conclusion kept, as instructed). |
| "Continuing the base network without correction changes its mean only from 6.89% to 6.33%, and Smoothing-trained reaches 1.28% (largest 7.8%)." | Moved to Supplementary Note S8 in full (6.89%, 6.33%, 1.28%, 7.8%). Main text keeps P3 S4: "Continuing the training of the base network without the correction leaves its mean nearly unchanged (Supplementary Note S8)." 6.89% stays in the main text (P1 S5, P3 S2). 1.28 still occurs in the main text, but only as the H1 graph-harmonic entry of Table 4. |
| "Applying the correction to the base network without retraining already gives 0.0965% (largest 0.91%)," | P3 S2 ("lowers the mean under traction loads from 6.89% to 0.0965%") and P3 S3 (largest 0.91%). 0.0965% is the traction-load (force_c) mean of the base network + correction in Table ST03. |
| "so most of the improvement comes from the correction itself;" | P3 S1 |
| "the NICE continuation, which also adds training and widens the training pool, lowers the mean by a further factor of 1.31 (95% bootstrap interval over geometries 1.20–1.40; Supplementary Table ST03)." | P3 S5-S7 (factor 1.31 and interval 1.20–1.40 kept, as instructed). |
| "Under the same correction, a deterministic harmonic start leaves 5.4 to 265 times the error of NICE (Section 5.5)." | P3 S8 ("deterministic graph-harmonic initial field") |
| Figure 5 caption, title "Directional energy error of the learned substructures on the 80 validation geometries." | New title "Energy error of the base network, the base network + correction and NICE on the 80 validation geometries.", adapted to the regenerated figure with three variants (figures_src/fig05_population.py plots B, B+W and A3 only; legend labels 'Base network', 'Base network + correction', 'NICE'). |
| Figure 5 caption body | Kept in full: observation definition, (a) 20 geometries per group, mean marker and 10th-percentile-to-maximum bar, (b) open uncut / filled cut markers and dotted equality line, (c) four classes with 75 / 80 geometries and the asterisk, twenty selection geometries (6 uncut, 14 cut), "Variants as in Table 2". "Base network, corrected" renamed 'Base network + correction' and "without retraining" added, as in the new note below Table 2 (U07). The panel titles of the regenerated figure ("Traction loads, by cut severity", "Per geometry: base network vs NICE", "Other load classes, all cells"; classes "imposed by a neighbour", "stiffness-scaled springs", "single-face traction", "nodal point loads*") match the caption. |

Computation of the ratio range (Supplementary Table ST03b, column force_c, NICE versus base network): uncut 1.142/0.016 = 71.4; lightly cut 5.801/0.079 = 73.4; moderately cut 7.899/0.091 = 86.8; heavily cut 12.702/0.109 = 116.5. Range 71 to 117. The same computation with the Uncorrected column (1.029, 5.325, 7.823, 11.160) gives 64.3, 67.4, 86.0 and 102.4, which reproduces the source's "64 to 102" and confirms the method. The sevenfold growth of NICE is 0.109/0.016 = 6.8 (unchanged claim).

### Old 5.4 "What the network leaves: slow modes, the cut band and a stiff error" (old lines 442-460) → new 5.4 "Where the network error lies"

| Old sentence | Where it is now |
|---|---|
| Heading | "5.4. Where the network error lies" (brief Section 4). |
| "The error that the network leaves lies partly in modes that smoothing damps slowly and partly next to the cut, and it is stiffer than the solution." | P4 S1-S2 ("stiffer than the exact solution") |
| "Before the correction is applied, it is examined in four ways: where it lies in the interior spectrum, where it lies in space, why an interior displacement error of about one percent becomes an energy error of several percent, and how its effect on the sensitivity divides between the linear and quadratic terms of Eq. (9)." | P4 S3-S6, split into four sentences; Eq. (9) → Eq. (17). |
| "The base network's error occupies different parts of the Jacobi-scaled interior spectrum in different cells (Figure 6)." | P5 S1 |
| "Under consistent tractions on M1, the lowest 200 modes contain 24.5% of the error energy but only 4.87% of the exact-field energy," | P5 S2. The reference to Table ST05b, which in the source was attached to the last sentence together with Section 5.5, moved here. |
| "and all lie below the lower endpoint \(a=0.173\) of the smoothing interval (\(a=b/30\), Table 1);" | P5 S3 |
| "in H2 only 66 of the 200 modes lie below \(a=0.138\)." | P5 S4 |
| "Smoothing damps these modes slowly, which leaves them to the coarse-grid correction (Section 5.5, Table ST05b)." | P5 S5 (Section 5.5); Table ST05b moved to P5 S2. |
| Figure 6 caption | Kept in full. "extension error" → "interior displacement error" / "the error"; "solid circles ... dashed triangles" written as "Solid lines with circles ... dashed lines with triangles" (checked in figures_src/codex/build_publication.py, f03: force_c uses ls "-" with marker "o", force uses ls "--" with marker "^"); "directional means / percentiles" → "means / percentiles over the test displacements". |
| "Spatially (Figure 7), for the plotted consistent-traction direction the exact field of M1 places 8% of its energy in the elements within two element widths of the cut plane, which are 13% of the elements;" | P6 S1-S3 (S1 opens with the point, taken from the source's "partly next to the cut"; S2 13%; S3 8%). |
| "over the four directions evaluated, 39–43% of the base network's error energy lies there, next to the retained cut-band DOFs whose values the extension must propagate." | P6 S4-S5 ("retained DOFs of the cut-plane elements, whose values the network must propagate into the interior") |
| "Relative to the base network, NICE reduces the total error energy by about two orders of magnitude (113–138 times) and the element error energies by roughly 25 to 300 times (10th–90th percentiles), and halves the share of this layer to 19–22%." | P6 S6-S8 ("by factors of 113 to 138"; "fraction of the error energy in this layer") |
| Figure 7 caption | Kept in full. "box-face DOFs" → "DOFs on the cell faces"; "cut-band DOFs / elements" → "DOFs of the cut-plane elements / cut-plane elements"; "extension error" → "interior displacement error"; "consistent-traction direction" → "traction-load test displacement". "Bulk element energies" kept (the figure script computes the element energies \(x_e^TK_ex_e\)). Legend labels of the regenerated figure ("retained: cell face", "retained: cut-plane elements", "interior (I)") match. |
| "The size of these energy errors reflects a measurable amplification." | P7 S1 ("A small displacement error produces a large energy error when the error is stiff, and this amplification can be measured.") |
| "For interior error \(d_I\) and exact interior field \(u_I\), let \(\delta^2=\dots\) be the Jacobi-weighted relative displacement error and \(\kappa=\dots\) the ratio of the Rayleigh quotients of the interior error and of the exact field." | P7 S2-S4 (mathematics unchanged) |
| "Then \(\varepsilon=\delta^2\kappa\) holds identically (Appendix B.3; values in Supplementary Table ST04)." | P7 S5 (Appendix B.3); Table ST04 moved to P7 S6, the first sentence that gives the values. |
| "Under consistent tractions, the base network's fields in M1, M2, H1 and H2 have mean \(\delta\) of 0.7–1.3% but mean \(\kappa\) of 124 to 7089: the displacement error is small but lies in directions far stiffer, relative to their amplitude, than the exact field, so that energy errors of 1.8–35% result." | P7 S6-S8 ("directions" here means displacement patterns; rewritten as "relative to its amplitude it is far stiffer than the exact field"). |
| "In the uncut U2 a similar \(\delta\) of 0.74% meets \(\kappa=85\) and gives 0.40% (Supplementary Table ST04)." | P7 S9 ("combines with"; "an energy error of 0.40%"). The repeated Table ST04 reference is carried by P7 S6. |
| "NICE's fields have \(\delta\) of 0.05–0.19% and \(\kappa\) of 33 to 582 in the cut cells (0.11% and 47 in U2)." | P7 S10 (parenthesis written out) |
| "Where the base network's errors are large, the quadratic term of Eq. (9) dominates:" | P8 S1 (Eq. (17); "dominates the sensitivity error") |
| "in M1 and H2, with sensitivity errors of 14.1% and 75.1% at fixed retained displacement, the linear term contributes only 7.2% and 0.79% of the sum of the linear- and quadratic-term norms." | P8 S2-S3. "under traction loads" added to S2: Table ST05a gives these values (14.128, 75.072; first-order shares 7.164, 0.786) in the force_c rows of M1 and H2. |
| "Under consistent tractions it contributes 28–43% in the three cells whose energy errors are below 2% (U1, U2 and H1), and under nodal forces 24–72% in all six cells." | P8 S4-S5 |
| "Being first order in the field error, it dominates as that error tends to zero (Table ST05, Figure S03)." | P8 S6 ("(Supplementary Table ST05 and Figure S03)", one parenthesis; Figure S03 is cited only here, so both references are kept). |

### Old 5.5 "Correction at fixed network parameters: learned field and correction are complementary" (old lines 462-484) → new 5.5 "Correction of a fixed network"

| Old sentence | Where it is now |
|---|---|
| Heading | "5.5. Correction of a fixed network" (brief Section 4). |
| "Holding the base network fixed isolates the effect of the correction (Figure 8)." | P9 S1 |
| "Eight Chebyshev steps at fixed retained displacement reduce the mean energy error everywhere, but unevenly (Table ST05a):" | P9 S2 ("in every cell") |
| "under consistent tractions H2 falls from 35.0% to 0.205%, whereas M1 falls only from 13.5% to 4.69% (2.95% after 32 steps), because much of its error lies in modes below the smoothing interval (Section 5.4)." | P9 S3-S4 |
| "M1's sensitivity error falls from 14.1% to 2.76%, whereas U1's sensitivity error varies non-monotonically with the number of steps although its energy error decreases at every step (Figure 8a,b)." | P9 S5-S6 |
| "A trilinear coarse-grid correction before the same eight steps brings M1 to 0.230%, and eight further pre-smoothing steps to 0.186%, with 5,601 coarse DOFs for 165,927 interior DOFs." | P10 S1-S3 |
| "Two, four and eight steps per stage give 1.02%, 0.422% and 0.186% on M1, and the eight-step cycle about 0.027% on M2 and U1 (Table ST07); other coarse spaces are compared in Table ST06." | P10 S4-S6 |
| "Table 4 applies the same correction to four starting fields at the same retained displacements: a zero interior, a graph-harmonic extension (a volume-weighted graph Laplacian on the element connectivity, with the same exact rigid-body split), the base network's field and NICE." | P11 S1-S3. "starting fields" → "initial fields"; "graph-harmonic extension" → "graph-harmonic initial field" (term "extension" replaced, as in the approved introduction draft, "graph-harmonic initial field"); the parenthesis became S3 ("after the same exact separation of the rigid-body part as for the network"; Table ST07: "Harmonic and zero starting fields receive the exact rigid-body split"). |
| "A zero interior leaves 28–1100% energy error and the harmonic start 0.08–17%, whereas the base network's field ends at 0.004–0.19%, lower by factors of 7 to 290 than the harmonic start and 2,400 to 12,000 than the zero interior." | P11 S4-S6 |
| "Even with 64 steps per stage, eight times the smoothing work, the harmonic start remains 2.6 to 50 times above the base network's eight-step result under consistent tractions in five of the six cells of Table ST07 (those of Table 4 and U1); only in H2 does it reach that result." | P11 S7-S9. The parenthesis "(those of Table 4 and U1)" describes the six cells of Table ST07, and S8 states it that way. Checked against Table ST07: harmonic 64 steps / base network 8 steps = 2.63 (H1), 17.8 (M2), 7.7 (M1), 49.8 (U1), 45.8 (U2); H2 1.22e-05 < 0.0117. |
| "The learned field thus supplies the part of the interior equilibrium that smoothing and the coarse space do not reach." | P11 S10 ("The network prediction therefore accounts for ...") |
| Table 4 title "Mean energy error (%) after the same 8 / \(Q_1(17)\) / 8 correction applied to different starting fields (consistent tractions, 32 directions)" | "**Table 4. Mean energy error (%) after the same 8 / \(Q_1(17)\) / 8 correction applied to different initial fields.** Traction loads, 32 test displacements." (form of brief Section 2; parsed by latex/build_tex.py `tables()`). |
| Table 4 columns | "Zero interior" → "Zero interior displacements"; "Harmonic" → "Graph-harmonic"; "Base network, corrected" → "Base network + correction". All entries unchanged. |
| "The first column gives the base network's error without correction. Results with 16 to 64 smoothing steps per stage are given in Table ST07." | Note S1-S2 |
| Figure 8 caption | Kept in full. Title "Accuracy gained by correcting the base network at fixed network parameters." → "Correction of the base network with fixed network parameters."; "directional energy error" → "energy error"; "field-based sensitivity error" → "sensitivity error"; "consistent tractions" → "traction loads". |

No numbers or sentences of old 5.4 and 5.5 were removed or moved.

## (b) Cross-references renumbered (each occurrence mapped once)

| Old | New | Where |
|---|---|---|
| Eq. (9) | Eq. (17) | P4 S6 (old 5.4 paragraph 1) |
| Eq. (9) | Eq. (17) | P8 S1 (old 5.4 last paragraph) |

Unchanged references (same number in the new structure): Sections 5.4 and 5.5; Table 1; Table 2; Table 4; Figures 5, 6, 7, 8, 8a,b; Figures S02, S03; Appendix B.3; Supplementary Tables ST03, ST03b, ST04, ST05, ST05a, ST05b, ST06, ST07. New reference: Supplementary Note S8 (P3 S4). The source's mixed "Table STxx" / "Supplementary Table STxx" is written uniformly as "Supplementary Table STxx" (CN 补充表).

## SUPPLEMENT ADDITIONS

For Supplementary Note S8 (Further continuations of the base network). These sentences come from old Section 5.3. The numbers are those of Tables ST03 and ST03b, under traction loads (class force_c).

### EN

**Accuracy on single cells (from Section 5.3).** Under traction loads, the means of NICE by cut-severity group are 64 to 102 times lower than those of the Uncorrected continuation. The group means of the Uncorrected continuation rise from 1.03% for uncut to 11.2% for heavily cut cells (Table ST03b). Over all 80 geometries, the mean energy error of the Uncorrected continuation under traction loads is 6.33%, and its largest single-geometry mean is 42%. Continuing the base network without correction therefore changes its mean only from 6.89% to 6.33%. The Smoothing-trained variant reaches a mean of 1.28%, and its largest single-geometry mean is 7.8% (Table ST03).

### CN

**单个胞元的精度（来自第 5.3 节）。** 在面力载荷下，未修正延续各切割程度分组的均值为 NICE 的 64 至 102 倍。未修正延续的分组均值由未切割胞元的 1.03% 增至重度切割胞元的 11.2%（表 ST03b）。在全部 80 个几何上，未修正延续在面力载荷下的平均能量误差为 6.33%，单个几何平均误差的最大值为 42%。因此，不加修正地继续训练基础网络，仅使其平均误差由 6.89% 变为 6.33%。平滑训练变体的平均误差为 1.28%，单个几何平均误差的最大值为 7.8%（表 ST03）。

## (d) Open questions

1. **Ratio 71 to 117 (computed).** The unit instruction allowed restating "64 to 102 times lower" against the base network because Table ST03b prints both sets of group means. The computation is recorded in (a). The new numbers 71 and 117 do not appear verbatim in any source file.
2. **Precision of the base-network group means.** P1 S4 quotes 1.142% and 12.702% exactly as printed in Table ST03b. Elsewhere the main text rounds to three significant digits (0.908% → 11.3% from 11.278). If the assembler prefers that convention, the values would be 1.14% and 12.7%. I did not round them, because rounding would put new strings into the text.
3. **"Nearly unchanged."** P3 S4 uses the wording suggested by the brief for the change from 6.89% to 6.33% (the Uncorrected continuation). The exact numbers are in Note S8.
4. **"under traction loads" added in P8 S2.** The source sentence on M1 and H2 (14.1%, 75.1%, 7.2%, 0.79%) did not name the load class. Table ST05a shows that these are the force_c values, so the condition was added for clarity. Please confirm.
5. **Figure 5 regeneration.** The caption assumes the regenerated F02_validation.png from figures_src/fig05_population.py (three variants). If an older five-variant PNG is still in figures/, it must be regenerated before the build.
6. **Supplementary Figure S03 caption.** It still mentions "five of Uncorrected" (supplement, unit S4). This is consistent with S03 staying in the supplement, but the main text now cites S03 only for the base network's linear-term shares.
7. **Term "largest single-geometry mean".** Used for the population maximum (U07 defines it as "the largest mean of a single geometry"). The CN text writes 单个几何平均误差的最大值 rather than 几何平均值, because 几何平均值 reads as "geometric mean" in Chinese. The CN supplement (S1) uses 几何均值 in the ST03 note. The assembler may want to align these.

## VERIFIER

Checked against MANUSCRIPT_EN.md / MANUSCRIPT_CN.md lines 430-485, SUPPLEMENTARY_EN.md Tables ST03, ST03b, ST04, ST05a, ST05b, ST06, ST07, the regenerated Figure 5 script (figures_src/fig05_population.py: three variants; panel (c) classes glued, support_k, face_c, force), U07_EN/CN (Section 5.1 variant definitions) and the approved introduction draft.

Python checks (final files):
- Old EN vs new EN: numbers absent from the new text are 1.03, 11.2, 6.33, 42, 102, 7.8 (and one occurrence each of 64 and 1.28), all in SUPPLEMENT ADDITIONS (EN and CN), and 9 (old Eq. (9) → Eq. (17)). Numbers new to the text are 1.142 and 12.702 (Table ST03b, base network force_c, Uncut and Heavy cut rows) and 71, 117 (ratio range; recomputed: 1.142/0.016 = 71.4, 5.801/0.079 = 73.4, 7.899/0.091 = 86.8, 12.702/0.109 = 116.5; the Uncorrected column reproduces the source's 64 to 102 only for force_c, which confirms the restriction "under traction loads").
- New EN vs new CN: 25 blocks each; numbers and inline mathematics identical block by block.
- Every remaining number recomputed or traced: 5.4 to 265 (Table 4 harmonic/NICE: 5.45 to 264.5), 7 to 290 and 2,400 to 12,000 (Table 4), 2.6 to 50 (Table ST07, 64-step harmonic / 8-step base network), 0.005 percentage points (ST03 glued 0.060 vs 0.055), δ, κ ranges (ST04), 28–43% and 24–72% (ST05a first-order shares; nodal 23.516 rounds to 24), 1.8–35% (ST05a k = 0), 1.31 and 1.20–1.40 (ST03 note).
- Eq. (9) → Eq. (17) confirmed: old Eq. (9) is the linear-plus-quadratic sensitivity-error identity of old Proposition 3. Both occurrences mapped once. No displayed equations, no citations in this unit.
- Banned words, old terms and variant names: none in EN or CN ("first" only as ordinal; CN "校正" only in 粗网格校正). EN sentences above 30 words occur only in figure captions (at most 35 words).

Changes made:
1. EN P3 (and CN): "NICE adds further training through the correction and draws on a larger set of training geometries" → "NICE also continues the training, with the correction inside the training loop, and draws its training geometries from a larger set." The CN "且采用更多的训练几何" said that NICE uses more training geometries, which is wrong: NICE sees at most 153 of the 591 geometries, against 305 for the base network (U07, Table ST01). The source says "widens the training pool". CN now reads "其训练几何取自更大的几何集合". "This lowers" → "This continuation lowers" (CN 这一继续训练).
2. CN P3: "按几何进行 bootstrap 重抽样" → "以几何为单位作 bootstrap 重抽样" (removes 进行).
3. EN P7 S1 split into two sentences (one claim per sentence). CN P7 S1 rewritten: the old CN "之所以……是因为误差场刚硬" stated a general cause, whereas EN states a condition ("when the error is stiff"); CN now reads "当误差较为刚硬时，较小的位移误差会引起较大的能量误差。这种放大作用可以定量衡量。"
4. CN P1: "NICE 有 99% 的能量误差低于 0.69%" → "NICE 的能量误差有 99% 低于 0.69%" (word order).
5. CN P5: "光滑对这些模态衰减缓慢，因而将其留给粗网格校正" → "光滑对这些模态的衰减较慢，这些模态因而留给粗网格校正" (grammar; drops 其).

Confirmed, no change needed:
- Open question 4 (added "under traction loads" in P8 S2): ST05a gives 14.128 / 75.072 and first-order shares 7.164 / 0.786 in the force_c rows of M1 and H2. The condition is correct and does not change the claim.
- Open question 2 (1.142% and 12.702%): left at the printed precision of Table ST03b. If the assembler rounds to the three-digit convention of the text, the values are 1.14% and 12.7%, and the ratio range 71 to 117 is unchanged.
- "Nearly unchanged" for 6.89% → 6.33% follows the wording the brief proposes; the numbers are in SUPPLEMENT ADDITIONS.
- "as for the network" in the graph-harmonic description: the network's interior field also uses the exact rigid-body split (old Section 4.2, MANUSCRIPT_EN line 281), and Table ST07 states that the harmonic and zero fields receive it.

Unresolved: none in this unit. The CN supplement (S1) writes 几何均值 where this unit writes 单个几何平均误差的最大值 (open question 7); the assembler should choose one form.
