# U04 notes: Section 3.2 Network for the interior displacements, Figure 3

Source (old numbering): MANUSCRIPT_EN.md / MANUSCRIPT_CN.md lines 247-288 (old Section 4.2, Eqs. (13), (14), Figure 3).

New blocks (EN and CN aligned, 16 blocks each, same order as the source): heading `### 3.2.`; P1 what the network approximates and why the two direct approaches are excluded; P2 the three exact properties; P3 the remaining design choices (stencil, grid hierarchy, weakly connected nodes); P4 the two branches; P5 separation of the rigid-body motion; P6 geometry branch; P7 displacement branch and local interaction; Eq. (8); P8 meaning of Eq. (8), grid hierarchy, parameter count, cost; P9 superposition; Eq. (9); P10 the two exact relations and the transpose; P11 resemblance to an iterative solver, limited to the data flow; Figure 3 image; Figure 3 caption.

Length (EN words, display equations excluded): old source 1132, new 1523. The increase comes from splitting long sentences and from plain definitions at first use: stencil, ghost-penalty face, element moments, retained node, U-shaped grid hierarchy. The Figure 3 caption also gains one sentence on the position embedding (see (a)).

Self-check (Python, on the final files): (a) every number of the old EN source is in the new EN text, except the renumbered references (old Eqs. (12), (13), (14), old Sections 4.1-4.5); (b) the new EN text has no number that is absent from the old source, except renumbered references and tags (Sections 3.1, 3.3, 3.4, 3.5, 4, heading 3.2, Eq. (8)) and the added reference "Appendix A.1"; (c) EN and CN have the same numbers in every block, the same inline mathematics in every block and identical display equations; both display equations are byte-identical to the old ones apart from their tags; (d) no EN sentence above 35 words (longest 33, mean 14.5), no CN sentence above 80 characters (longest 66, mean 29); (e) no banned word or old term in EN or CN after two fixes ("supplies" → "provides"; "the first relation" → "the former"). In CN, "其" occurs only in 其一/其二/其三, 其余 and 其中.

## (a) Sentences and numbers removed or moved

All source sentences remain in this unit. Nothing moved to another unit or to the supplement. No number was removed.

| Old sentence (EN) | Where it is now |
|---|---|
| Heading "4.2. Geometry-conditioned multilevel displacement extension" | heading "3.2. Network for the interior displacements" (brief Section 4) |
| "The learning target is the interior equilibrium map \(q\mapsto E_Iq=-A^{-1}K_{IP}q\)." | P1 S1: "The network approximates the interior part of the exact recovery, that is, the map ..." |
| "Its approximation provides both a displacement field and, through its energy, the condensed operator on the complete retained space (Section 4.1)." | P1 S2 ("the condensed stiffness on all retained DOFs", Section 3.1) |
| "The retained set has 2,679 to 45,900 DOFs and changes with the cut, which rules out the obvious alternatives: an extension matrix formed for every geometry costs one interior solve per retained DOF, and a network with one output per retained DOF would have to change its output dimension with the cut." | P1 S3-S6. The range 2,679 to 45,900 also appears in the introduction (U00); kept here because the argument of this paragraph uses it. |
| "Three properties are needed exactly, because the relations of Sections 3 and 4.1 rest on them, and the architecture provides them by construction." | P2 S1-S2 (Sections 3.1 and 4, see (b)) |
| "The extension is linear in \(q\) at fixed geometry: nonlinearity is confined to a geometry branch, and every operation on displacement features is linear." | P2 S3-S4 |
| "It reproduces the retained values and rigid-body motion: rigid motion is separated before learning and reconstructed, and the retained values are overwritten at the output (Eq. (14))." | P2 S5-S6 (Eq. (9)) |
| "Its transpose, needed for the work-conjugate force of Eq. (12), follows by transposing the linear displacement path operation by operation (Appendix G.1, Eq. (E.1))." | P2 S7-S8 ("the corresponding nodal forces of Eq. (5)"). Only "(Appendix G.1)" is kept here (brief rule 4); "Eq. (E.1)" is kept in P10 S4, where the same transpose is stated in detail. |
| "Section 5.2 verifies all three numerically." | P2 last sentence |
| "The remaining choices govern how well the map is approximated and were not ablated individually." | P3 S1 ("not tested individually by ablation") |
| "Weights act on stencil slots, channels and grid levels, with coefficients generated per stencil from the geometry, so that one network serves the retained sets of the trained domain." | P3 S3-S5. P3 S2 is added to define "stencil" at its first use (definition from Appendix G.1: element stencils and ghost-face stencils). "Stencil slots" → "local node positions of the stencils"; "the trained domain" → "all geometries within the range covered by training". |
| "Element moments enter as geometry input, because wall surfaces and the cut plane intersect the background elements." | P3 S7. P3 S6 is added to define element moments (Appendix A.1: "The 125 moments \(M_{e\alpha}\) integrate local monomials ... over the material part of an element"); the number 125 is not quoted. |
| "A U-shaped latent hierarchy carries the long-range coupling of the nonlocal map," | P3 S8-S9 ("The map couples distant parts of the cell. A U-shaped grid hierarchy, which passes the features to successively coarser grids and back, carries this long-range coupling.") The relative clause explains "U-shaped" from Appendix G.1 (restrict 65→33→17→9, then prolong back). |
| "and additional local interactions act on stencils containing weakly supported nodes, which few, small stiffness entries couple to the rest of the cell." | P3 S10-S11 ("weakly connected nodes") |
| "Accordingly, the network has two branches: a nonlinear geometry branch computes, once per geometry, the coefficients of a linear operator, and the displacement branch is that operator (Figure 3)." | P4 S1-S3 |
| "Rigid-body motion is separated before learning." | P5 S1 ("before the network is applied") |
| "We take for the columns of \(R\) three translations and three rotations about a common centre." | P5 S2 (impersonal) |
| "The coefficient extractor \(C_R=\ldots\) and projector \(\Pi_P=\ldots\) decompose the input into rigid coefficients \(C_Rq\) and retained deformation \(q_d=\Pi_Pq\)." | P5 S3-S4 |
| "The network extends only \(q_d\), so rigid reconstruction is independent of training accuracy." | P5 S5-S6 |
| "In the geometry branch, element moments resolve the material distribution within each background element, and node features identify retained and cut-band membership, weak support (a diagonal stiffness block below 1% of the median over active nodes; Appendix G), local stiffness magnitude and position." | P6 S1-S4. The parenthesis became its own sentence (S4); "diagonal stiffness block" → "the norm of its diagonal stiffness block" (Appendix G.1: Frobenius norm of the node's diagonal 3×3 block). 1% kept. |
| "Encoders form element and node feature vectors, which exchange information in two rounds;" | P6 S5 |
| "coefficient heads (small output networks) then assign weights to prescribed element and ghost-face incidences, to transfers between latent grids and to coarse-grid convolutions." | P6 S6-S7 ("fixed element–node and ghost-face–node connections", "transfers between the grids of the hierarchy") |
| "These coefficients are fixed for all displacement directions on the same geometry." | P6 S8 ("For a given geometry, these coefficients do not depend on the retained displacements.") |
| "The displacement branch lifts the three components of \(q_d\) to 32 channels on retained nodes, with the interior features set to zero, defining \(X^0(q)\)." | P7 S1-S2; "retained nodes" defined as "the nodes that carry retained DOFs" |
| "Local interactions propagate these values through the element and ghost-face neighbourhoods:" | P7 S3 |
| "each gathers the features on a 27-slot stencil into four weighted heads, mixes the channels within each head and scatters the result back to the incident nodes." | P7 S4-S5 ("27 local node positions") |
| "With interior and retained node masks \(\mathsf M_I,\mathsf M_P\) acting on nodal rows, an interaction updates the latent field by" | P7 S6-S7 (inline math split into \(\mathsf M_I\) and \(\mathsf M_P\)) |
| Eq. (13) | Eq. (8), unchanged except the tag |
| "Here \(\mathcal G_{\ell h}\) and \(\mathcal S_{\ell h}\) are geometry-weighted gather and scatter maps, \(W_{\ell h}\) mixes channels, and the final term restores the retained features." | P8 S1 |
| "A U-shaped latent hierarchy through grids with 65, 33, 17 and 9 positions per axis, the first being the \(Q_2\) node grid of the cell, supplies the long-range coupling;" | P8 S2-S3 ("the finest of which is the \(Q_2\) node grid") |
| "pairs of local interactions before and after it, and further pairs on stencils containing weakly supported nodes, complete the branch, and a linear map reconstructs nodal displacement (Figure 3b,c)." | P8 S4-S5 |
| "Because the weights do not act on individual DOFs, the same network, of about \(6\times10^5\) parameters (Table ST15), serves every retained set." | P8 S6 |
| "For a given channel width and stencil size, one application costs work proportional to the number of element and ghost-face stencils and to the size of the fixed latent grids, independently of how many of the nodes are retained, and treats a batch of displacement directions at once." | P8 S7-S9. "displacement directions" → "retained displacement vectors" (not "test displacements"), because the statement concerns any input \(q\), including deployment, and "test displacement" is defined in Section 3.1 as a retained displacement used to measure accuracy. |
| "Appendix G gives the complete order of operations and coefficient definitions." | P8 last sentence |
| "Every displacement operation is linear, and the nonlinear encoders act only on geometry, so the raw map \(\mathcal N_\theta(\eta)\), with network parameters \(\theta\), obeys superposition at fixed \(\eta\)." | P9 S1-S2 |
| "Adding back the rigid field and restoring the original retained values gives" | P9 S3 ("... gives the network recovery operator") |
| Eq. (14) | Eq. (9), unchanged except the tag |
| "Consequently, \(J_P\widehat E=I_p\) and \(\widehat ER_P=R\): admissibility, from the final overwrite of the retained values, and rigid reproduction, from the rigid-body split, hold by construction." | P10 S1-S3 ("admissibility" → the relation itself; brief 3.1) |
| "The transpose \(\mathcal N_\theta(\eta)^T\) required by Eq. (12) is the composition of the transposed operations in reverse order (Appendix G.1, Eq. (E.1)), so that, in exact arithmetic, the returned force is the derivative of the energy of the learned extension." | P10 S4-S5. Only "(Eq. (E.1))" is kept here (rule 4); Appendix G.1 is cited for the same statement in P2. "The energy of the learned extension" → "the strain energy \(\widehat{\mathcal U}\)", the symbol of Eq. (4) in Section 3.1 (U03 uses the same wording for Eq. (5)). |
| "The data flow of the displacement branch resembles that of an iterative solver for the interior problem with the retained displacement as Dirichlet data:" | P11 S1 ("prescribed (Dirichlet) boundary values") |
| "each local interaction, Eq. (13), updates the interior features from their stencil neighbours with geometry-dependent coefficients while the retained features are reset to their lifted prescribed values," | P11 S2-S3 (Eq. (8)) |
| "and the latent hierarchy carries information between distant parts of the cell, as coarse levels do in multigrid." | P11 S4 |
| "Unlike a solver, the branch never applies the stiffness matrix or forms a residual, and the equilibrium field is not a fixed point of its layers; it is a fixed sequence of learned linear maps." | P11 S5-S8. S5 ("The resemblance is limited to the data flow.") states the unit instruction in one plain sentence; the content is that of "Unlike a solver". |
| "The correction of Sections 4.3 and 4.4, in contrast, acts on the residual of the same interior problem with prescribed coefficients;" | P11 S9 ("The two-grid correction of Sections 3.3 and 3.4") |
| "trained through it (Section 4.5), the network supplies the part of the interior field that a fixed number of these steps does not recover." | P11 S10-S11 ("The network is trained with this correction applied (Section 3.5)", wording as in U03) |

### Figure 3 caption

| Old | New |
|---|---|
| Title "Geometry-conditioned neural displacement architecture." | "Architecture of the network for the interior displacements." (follows the new section title; "geometry-conditioned" removed, brief 3.1) |
| (a) "Element and node encoders produce 64-channel geometry feature vectors, followed by two rounds of residual exchange; coefficient heads condition the local interactions, grid transfers and convolutions." | (a) S1-S2, with the panel name "Geometry branch" as in the image |
| (none) | (a) S3 added: "The heads of the local interactions also receive a position embedding, which identifies the local node position within a stencil (Appendix G.1)." It explains the relabelled image text "position embedding" (formerly "slot embedding"). Source: Appendix G.1 ("an eight-dimensional slot embedding, a learned \(27\times8\) table indexed by the local slot"); the number 8 is not quoted. |
| (b) "The displacement branch propagates the nonrigid retained input through four local interaction pairs, the multilevel block, four further local pairs and four pairs on weakly supported stencils;" | (b) S1 ("four pairs on stencils that contain weakly connected nodes", matching the relabelled image text "Weakly connected") |
| "linear maps connect the three displacement components to 32 latent channels," | (b) S2 ("to 32 feature channels at the input and back at the output", matching the image labels 3 → 32 and 32 → 3) |
| "and deterministic bypasses reconstruct rigid-body motion and restore the retained values." | (b) S3 ("Bypass paths without learned parameters") |
| (c) "Restriction and prolongation connect grids with 65, 33, 17 and 9 background positions per axis; each coarse level has two residual convolutions on each pass." | (c) S1-S2; "on each pass" written out as "on the downward pass and two on the upward pass" (Appendix G.1 table) |
| "\(\mathcal R\) and \(\mathcal I\) in (c) denote restriction and prolongation." (last sentence) | merged into (c) S1 ("Restriction \(\mathcal R\) and prolongation \(\mathcal I\) ...") |
| (d) "A local interaction uses geometry-weighted gathering, four channel-mixing heads and scattering, followed by residual addition and retained-value restoration." | (d) S1-S2, with "the 27 local nodes of a stencil" (relabelled image text "27 local nodes") |
| "E and G denote element and ghost-face interactions." | kept |
| "Blue dashed arrows carry geometry-dependent coefficients; solid arrows carry features or displacement states." | kept ("features or displacements") |
| "For fixed geometry, the complete displacement path is linear." | kept |

### Added explanatory sentences (no new numbers or results)

- P3 S2: definition of "stencil" and of "ghost-penalty face" (Appendix G.1; U02 uses "ghost-penalty faces", CN "ghost 罚项作用面").
- P3 S6: definition of element moments (Appendix A.1).
- P7 S1: "retained nodes, the nodes that carry retained DOFs".
- Figure 3 caption (a) S3: position embedding (Appendix G.1).

## (b) Cross-references renumbered (each occurrence mapped once)

| Old | New | Place |
|---|---|---|
| Section 4.2 (heading) | Section 3.2 | heading |
| Section 4.1 | Section 3.1 | P1 S2 |
| Sections 3 and 4.1 | Sections 3.1 and 4 | P2 S1. Old Section 3 is now split into 3.1 (old 3.1) and 4, 4.1, 4.2 (old 3 intro, 3.2, 3.3); old 4.1 is now 3.1. |
| Eq. (14) | Eq. (9) | P2 S6; tag of the second display |
| Eq. (12) | Eq. (5) | P2 S7; P10 S4 |
| Eq. (13) | Eq. (8) | tag of the first display; P11 S2 |
| Sections 4.3 and 4.4 | Sections 3.3 and 3.4 | P11 S9 |
| Section 4.5 | Section 3.5 | P11 S10 |
| Appendix G.1, Appendix G, Eq. (E.1), Section 5.2, Table ST15, Figure 3, Figure 3b,c | unchanged | P2, P6, P8, P10, P4, caption |
| (none) | Appendix A.1 | P3 S6 (added definition of element moments) |
| (none) | Appendix G.1 | caption (a) S3 (added position embedding) |

## SUPPLEMENT ADDITIONS

None. The source of this unit does not mention the Uncorrected continuation or the Smoothing-trained variant.

## (d) Open questions

1. FIGURES.md carries a copy of the old Figure 3 caption (line 27). The assembler should replace it with the new EN caption.
2. The image F11_network_architecture.png already shows "position embedding" and "27 local nodes", but panel (b) still shows "Weak". The caption assumes the pending relabel to "Weakly connected".
3. Terms shared with Appendix G (A3 unit): this text uses "stencil", "local interaction", "position embedding", "grid hierarchy", "weakly connected" and "Eq. (8)", which match A3_EN. In Chinese it uses 模板, 局部交互, 收集/散布, 头, 多层网格 and 主节点, which match A3_CN. The old main-text CN used 局部相互作用; this unit uses 局部交互 to agree with A3_CN.
4. "Displacement directions" in old P8 became "retained displacement vectors" (EN) and 主自由度位移向量 (CN), not "test displacements". A3_EN translates the corresponding sentence of Appendix G.1 as "several test displacements". The assembler may unify these if a single term is preferred. In training the inputs are test displacements. In deployment they are arbitrary retained displacements.
5. The introduction (U00) and Section 1.1 (U01) already state that the network is nonlinear in the geometry and linear in the retained displacements, that one network of about \(6\times10^5\) parameters serves all cells, and the range 2,679 to 45,900. This unit repeats these points because the method section must state them in full. The assembler may shorten the introduction instead.
6. P2 cites "(Appendix G.1)" and P10 cites "(Eq. (E.1))" for the transpose. The source cited both at both places. If the assembler prefers both references at P10, the sentence can read "... in reverse order (Appendix G.1); Eq. (E.1) gives the resulting transpose of \(\widehat E\)."

## VERIFIER

Checked against MANUSCRIPT_EN.md / MANUSCRIPT_CN.md lines 247-288, Appendix A.1 and G.1 of APPENDICES_EN.md (for the added definitions of element moments, stencils, the weak-node norm, the position embedding and the two-plus-two coarse convolutions), the current image F11_network_architecture.png, and the neighbouring units U02, U03 and A3 for term consistency.

Result of the checks:

- A. Fidelity. Every sentence, number, condition and citation of the old text is in the new text; the table in (a) is correct. Python number diff old EN vs new EN: the only differences are the renumbered references and tags (old 4.1, 4.2-4.5, Sections 3 and 4.1, Eqs. (12), (13), (14) → 3.1, 3.2-3.5, Sections 3.1 and 4, Eqs. (5), (8), (9)), plus "Appendix A.1" and the second "27" (caption (d), "27 local nodes", from the relabelled image and Appendix G.1). Both display equations are byte-identical to the old ones apart from the tags. The added sentences (stencil, element moments, retained nodes, position embedding, "on the downward pass and two on the upward pass", "without learned parameters") are supported by Appendices A.1 and G.1 and the image; no claim is strengthened or weakened. The solver analogy keeps "only in its data flow", "never applies the stiffness matrix" and "does not compute residuals".
- B. Cross-references follow brief Section 5 exactly once each; tags are (8) and (9).
- C/E/F. No banned pattern, old term, em dash, PIML comparison, priority claim or cost statement. Figure block format is correct (image line, blank line, one-line bold caption). EN sentences: three of 31-33 words, none above 35.
- D. EN and CN have 16 aligned blocks with identical numbers, inline mathematics and display equations in every block (Python check after the edits).

Changes made:

1. EN P6: "compute the coefficients of the network" → "compute the geometry-dependent coefficients" (the network also has learned weights \(W_{\ell h}\) that are not computed by the heads). CN: 计算网络的系数 → 计算依赖几何的系数.
2. EN P11: "... is not a fixed point of its layers either." → "Nor is the equilibrium solution of the interior problem a fixed point of its layers." (register).
3. Figure 3 caption (c): the panel name "Grid hierarchy" → "Multiscale displacement propagation", the title printed in panel (c) of the image, so that all four panel names in the caption match the image; the sentence now reads "connect the grids of the hierarchy, which have 65, 33, 17 and 9 background positions per axis". CN: (c) 多尺度位移传播。……连接多层网格中的各层网格，各层每轴分别有 65、33、17 和 9 个背景位置。
4. CN wording (no content change):
   - P1: 每个胞元的主自由度为……个，且随切割而变化 → 各胞元的主自由度在 2,679 至 45,900 个之间，且该集合随切割而变化 (the set, not the count per cell, changes with the cut).
   - P2: 位移恢复对 \(q\) 线性 → 位移恢复关于 \(q\) 是线性的.
   - P3: 未逐项进行消融试验 → 但未逐项经过消融试验检验; 单元矩作为几何输入 → 网络以单元矩作为几何输入; 在含弱连接节点的模板上另加局部交互 (no subject) → 网络还在含弱连接节点的模板上附加若干局部交互.
   - P6: weak-node definition rewritten as 若某节点对角刚度块的范数低于该范数在全部激活节点上中位数的 1%，则称该节点为弱连接节点 (the old wording 低于激活节点上中位数 did not say whose median).
   - P8: 约 \(6\times10^5\) 个参数的同一网络 → 同一个约含 \(6\times10^5\) 个参数的网络.
   - P9: 所有位移特征上的运算 → 作用于位移特征的运算 (matches P2 and EN).
   - P11: 二者的相似仅限于数据流 (ambiguous: could be read as the grid hierarchy and multigrid) → 位移分支与迭代求解器的相似仅限于数据流; 从不施加刚度矩阵 → 从不与刚度矩阵相乘; 网络各层 → 位移分支各层 (EN "its layers" refers to the branch); 网络在训练时即包含这一修正 → 网络训练时计入这一修正 (same wording as U03_CN).
   - Caption (a): 随后进行两轮残差信息交换 (subject unclear) → 这些向量随后经过两轮残差信息交换.

Not resolved (for the assembler):

- The image still prints "Weak" above the third group of interactions in panel (b) (figures_src/codex/build_network.py line 167). The caption already reads "weakly connected nodes"; the relabel to "Weakly connected" must still be done in the image.
- If the separate relabel also renames the panel (c) title, the caption name "(c) Multiscale displacement propagation" must follow it (build_network.py line 183).
- Open questions 1, 4, 5 and 6 above remain for the assembler; they do not affect the correctness of this unit.
