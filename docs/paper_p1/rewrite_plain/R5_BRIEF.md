# R5 rewrite brief (binding for every rewrite agent)

## 0. Why this rewrite

The author's advisor, a leading expert in structural and topology optimisation, judged the current manuscript unreadable: it is full of coined terms and mathematical jargon he does not use, its sentences are long and over-qualified, and it reads as machine-written. The author has approved the following: rewrite the main text in plain, formal academic language with field-standard terminology, reorder it so that the method comes before the error analysis, keep every technical result, number and condition, and reduce the network variants in the main text to three. The Chinese version is what the advisor reads; it must read as native Chinese academic prose, not as a translation.

Reference drafts already approved in style by the author: `rewrite_plain/R5_术语对照与摘要草稿.md` (term map, abstract) and `rewrite_plain/R5_引言草稿.md` (introduction). Read both before writing; match their register.

## 1. Register and style

Target register: a well-edited paper in CMAME or Structural and Multidisciplinary Optimization; in Chinese, a well-edited paper in 力学学报 or 计算力学学报. Formal, objective, concrete. Neither chatty nor ornate.

Rules (EN and CN):

1. One claim per sentence. EN sentences mostly 15–30 words, rarely above 35. CN sentences mostly 30–60 characters, rarely above 80.
2. Field-standard terms (Section 3 below). A term not in common use in computational mechanics or structural optimisation is defined in one plain sentence at first use, or avoided.
3. Keep every technical condition, but state it once, in a subordinate clause or a following sentence, not in stacked parentheses.
4. At most one parenthetical cross-reference per sentence; prefer "(Section 3.2)" at the end of the sentence. Do not chain "(Proposition 2; Section 5.5; Table ST07)" when one reference suffices; keep the most specific one.
5. Paragraphs open with their point. No paragraph-level throat clearing.
6. Numbers: report the numbers that carry the argument; keep all numbers that the source reports, but a long run of numbers may move into a table already present (only if that table already contains them).
7. CN: write natural Chinese. Avoid translationese: long pre-nominal modifier chains ("一个对几何非线性、对保留位移精确线性的几何条件化网络"), "对……进行……" stacks, "其" overuse, English word order. Use formal written forms: 将……视为, 采用, 所得, 由于, 因而, 其中, 在……条件下. Avoid colloquial forms: 把……当作, 直接给出, 这样得到的, 总是, 一下, 其实.

Banned patterns (delete wherever they occur):

- Triads and slogans: "learning supplies the trial field, the variational form the structure, and a fixed correction the improvability"; "division of tasks/labour"; "the network must be accurate, not structure-preserving"; any "X supplies A, Y supplies B, Z supplies C" cadence; CN 分工, 可改进性, "网络只需准确，不必保持结构".
- Aphorisms and flourishes: "in effect", "precisely", "it is exactly", "notably", "crucially", "importantly", "this is the price", "settles the tension", "takes the middle course"; CN 这正是, 正是, 值得注意的是, 换言之, 由此可见 (when decorative).
- Em dashes as punctuation in prose; rhetorical questions; "not X but Y" contrasts used for effect.
- "first" or priority claims of any kind.

## 2. Hard constraints (never violate)

- Never add a number, result or claim that is not in the source text, the appendices or the supplement. Copy every number exactly (digits, units, ranges, signs).
- Never weaken or strengthen a claim. Keep qualifiers such as "under stated conditions", "in all cases tested", "on these geometries", "was not checked".
- No numerical comparison with PIML (Huang et al., Guo et al., Jiang et al., Zhang et al. learned substructures) in the paper. Describing their approach in words is fine.
- No mention of repeated measurements, machine load, throttling or shared hosts. Do not add statements about GPU hours, dataset generation cost or offline training cost.
- No model identifiers, no tokens, no hostnames.
- Do not touch the AI-use declaration, author information, CRediT, competing interests or acknowledgements.
- Keep all LaTeX mathematics and symbols unchanged (\(E\), \(F\), \(\widehat E\), \(\mathcal W\), \(\widehat S\), \(H\), \(A\), \(q\), \(P\), \(I\), etc.). Keep equations displayed exactly as they are, except for their \tag numbers, which follow Section 5 below.
- Keep the markdown structures that the build scripts parse, byte for byte in form:
  - figure: `![Figure N](figures/FILE.png)`, blank line, `**Figure N. Title.** caption` on one line;
  - table: `**Table N. Title.** optional text` on one line, then a blank line, then the pipe table;
  - citations as markdown links exactly as in the source, e.g. `[Guyan (1965)](https://doi.org/10.2514/3.2874)`.
- EN and CN files must be paragraph-aligned: the same number of paragraphs, lists, equations, figures and tables, in the same order, with the same numbers and the same citations. Each EN paragraph and its CN paragraph say the same thing.

## 3. Terminology

### 3.1 Replace (EN and CN)

| Old EN | Old CN | New EN | New CN |
|---|---|---|---|
| retained DOFs / retained displacement / retained space / retained set | 保留自由度 / 保留位移 / 保留空间 / 保留集 | retained DOFs (define once as "retained (master) DOFs"); retained displacements | 主自由度；主自由度位移 |
| interior DOFs | 内部自由度 | interior DOFs | 内部自由度 |
| extension, learned extension, complete extension, equilibrium extension | 延拓、学习延拓、完整延拓、平衡延拓 | displacement recovery; recovery operator \(F\); the network's interior displacements; exact recovery \(E\) | 位移恢复；位移恢复算子 \(F\)；网络预测的内部位移；精确位移恢复 \(E\) |
| condensed stiffness / condensed operator | 凝聚刚度 / 凝聚算子 | condensed stiffness (matrix) | 凝聚刚度矩阵 |
| exact Schur complement | 精确 Schur 补 | exact condensed stiffness (say once "the Schur complement") | 精确凝聚刚度矩阵（首次注明“即 Schur 补”） |
| box face(s), box-face DOFs | 胞元边界面 | cell face(s), cell-face DOFs | 胞元表面、胞元表面自由度 |
| cut band | 切割带 | cut-plane elements (the active elements intersected by the trimming plane) | 切割平面单元 |
| energy share | 能量份额 | energy fraction | 应变能占比 |
| equilibrium correction (as a mechanism) | 平衡校正 | two-grid correction (the name NICE keeps "equilibrium correction"; say once that it denotes this two-grid cycle) | 两重网格修正（方法名保留“平衡校正”，说明一次即指该两重网格循环） |
| consistent tractions | 一致面力 | traction loads | 面力载荷 |
| nodal forces (stress test class) | 节点力 | nodal point loads | 集中节点载荷 |
| direction(s), retained direction(s) | 方向、保留方向 | test displacement(s) | 测试位移 |
| directional energy error | 方向能量误差 | energy error | 能量误差 |
| weakly supported nodes | 弱支撑节点 | weakly connected nodes | 弱连接节点 |
| latent features / latent hierarchy / latent grids | 潜空间、潜空间层级 | feature channels / grid hierarchy / coarse grids | 特征通道、多层网格、粗网格 |
| slot(s), slot embedding | 槽位 | local node position(s), position embedding | 单元局部节点位置、位置嵌入 |
| cut strata, stratum | 切割分层 | cut-severity groups, group | 切割程度分组 |
| target cell | 目标胞元 | test cell | 被测胞元 |
| neighbour-induced traces | 邻胞元诱导迹 | displacements imposed by a neighbouring cell | 相邻胞元施加的边界位移 |
| geometry-conditioned network | 几何条件化网络 | the network (at first use: a network that takes the cell geometry as input) | 网络（首次：以胞元几何为输入的神经网络） |
| field-based sensitivity (estimate) | 场基灵敏度 | sensitivity (define once that it is \(-\widehat u^TK_{,c}\widehat u\) and ignores the design dependence of the network) | 灵敏度（定义处说明一次） |
| work-conjugate force | 功共轭力 | the corresponding nodal forces | 对应的节点力 |
| nonexpansive | 非扩张 | does not increase the energy error | 不增大能量误差 |
| admissible | 容许 | reproduces the retained displacements | 在主自由度上等于给定位移 |
| trial field | 试探场 | (delete; at most once when stating the minimum potential energy argument) | （删除） |
| Ritz approximation / Ritz identity | Ritz 近似 / Ritz 恒等式 | "by the principle of minimum potential energy"; Proposition 1 may keep the label "(Ritz identity)" once | 由最小势能原理；命题 1 可保留“（Ritz 恒等式）”一次 |
| Galerkin operator | Galerkin 算子 | appendices only; in the main text "coarse-grid matrix \(V^TAV\)" | 仅附录；正文写“粗网格矩阵 \(V^TAV\)” |
| uncut / cut stratum names light, moderate, heavy | 轻度、中度、重度切割 | lightly, moderately, heavily cut | 轻度、中度、重度切割（不变） |

### 3.2 Keep (field-standard), define once in one sentence where helpful

static condensation, superelement, substructure, cut finite element method (CutFEM), ghost penalty, two-grid cycle, Chebyshev smoothing, coarse-grid correction, compliance, sensitivity, homogenisation, TPMS, Schwarz-P, cut cell, Q2 element, rigid-body modes, conjugate gradient, preconditioner, MMA.

### 3.3 Variant names (main text keeps three)

| Key | New EN (main text and figures) | New CN |
|---|---|---|
| B (v2L1) | Base network | 基础网络 |
| B+W (B2grid) | Base network + correction | 基础网络加修正 |
| A3 (A3_2grid) | NICE | NICE |
| C (A0_ctrl) | Uncorrected continuation (supplement only) | 未修正延续（仅补充材料） |
| A2b (A2b_tail8) | Smoothing-trained (supplement only) | 平滑训练（仅补充材料） |

Main text: remove every statement about the Uncorrected continuation and the Smoothing-trained variant, and remove them from Table 2. Where a main-text argument relied on them, restate it with the three remaining variants if the source contains the needed numbers (for the base network, the base network with correction, or NICE), or with one plain sentence pointing to the new Supplementary Note S8, for example: "Continuing the training without the correction leaves the mean energy error nearly unchanged (Supplementary Note S8)." Every removed sentence, number and table goes, verbatim in content, into the supplement additions (Section 6 below).

## 4. New structure

| New | Title (EN) | Title (CN) | Source (old numbering) |
|---|---|---|---|
| Abstract | Abstract | 摘要 | approved draft |
| 1 | Introduction | 引言 | approved draft; Figure 1 and its caption stay here |
| 1.1 | Related work | 相关工作 | old 1.1, plus items moved out of the old introduction |
| 2 | Cut cells, static condensation and the design problem | 切割胞元、静力凝聚与设计问题 | old 2 intro |
| 2.1 | Geometry and finite element model | 几何与有限元模型 | old 2.1 |
| 2.2 | Retained degrees of freedom and static condensation | 主自由度与静力凝聚 | old 2.2 |
| 2.3 | Assembly, compliance and the design problem | 装配、柔度与设计问题 | old 2.3 |
| 2.4 | Assumptions | 基本假设 | assumptions (A1)–(A3), (D) from the old Section 3 introduction |
| 3 | The NICE method | NICE 方法 | old 4 intro and Figure 2 |
| 3.1 | Condensed stiffness from the strain energy | 由应变能计算凝聚刚度矩阵 | old 4.1, plus old 3.1 (Proposition 1, Eqs. (4), (5)) |
| 3.2 | Network for the interior displacements | 预测内部位移的神经网络 | old 4.2 and Figure 3 |
| 3.3 | Chebyshev smoothing | Chebyshev 光滑 | old 4.3 |
| 3.4 | Coarse-grid correction and the two-grid cycle | 粗网格校正与两重网格循环 | old 4.4 (Proposition 4 becomes Proposition 2; Remark 1 stays) |
| 3.5 | Training | 训练 | old 4.5 |
| 3.6 | Applying the condensed stiffness | 凝聚刚度矩阵的施加 | old 4.6 and Algorithm 1 |
| 4 | Effect of interior displacement errors on compliance and sensitivities | 内部位移误差对柔度与灵敏度的影响 | old 3 intro (without the assumptions) |
| 4.1 | Compliance | 柔度 | old 3.2 (Proposition 2 becomes Proposition 3) |
| 4.2 | Sensitivities | 灵敏度 | old 3.3 (Proposition 3 becomes Proposition 4) |
| 5 | Numerical examples | 数值算例 | old 5 intro |
| 5.1 | Geometries, variants and loads | 几何、网络变体与载荷 | old 5.1 |
| 5.2 | Verification of the reference model and of the deployed operator | 参考模型与部署算子的验证 | old 5.2 |
| 5.3 | Accuracy on single cells | 单个胞元的精度 | old 5.3 |
| 5.4 | Where the network error lies | 网络误差的分布 | old 5.4 |
| 5.5 | Correction of a fixed network | 固定网络参数下的修正 | old 5.5 |
| 5.6 | Two-cell assemblies: compliance and sensitivity | 两胞元装配中的柔度与灵敏度 | old 5.6 |
| 5.7 | Ablation: restricting the cell-face displacements | 消融：限制胞元表面位移 | old 5.7 |
| 5.8 | Lattices of learned cells | 全部由学习胞元组成的点阵 | old 5.8 |
| 5.9 | Computational cost | 计算成本 | old 5.9 |
| 5.10 | Thickness design | 厚度设计 | old 5.10 |
| 6 | Discussion | 讨论 | old 6 |
| 6.1 | Retained degrees of freedom, network and correction | 主自由度、网络与修正 | old 6.1 |
| 6.2 | Computational value | 计算价值 | old 6.2 |
| 6.3 | Limitations | 局限性 | old 6.3 |
| 7 | Conclusions | 结论 | old 7 |

## 5. Renumbering maps (apply to every cross-reference; all numbers in the source files are OLD numbers, map each occurrence exactly once, never chain replacements)

Sections: old 1→1, 1.1→1.1, 2.k→2.k, old 3 (intro)→4, old 3.1→3.1, old 3.2→4.1, old 3.3→4.2, old 4→3, old 4.k→3.k (k = 1…6), 5.k→5.k, 6.k→6.k, 7→7. Assumptions (A1)–(A3), (D) now live in Section 2.4.

Equations (main text): (1)→(1), (2)→(2), (3)→(3), (11)→(4), (12)→(5), (4)→(6), (5)→(7), (13)→(8), (14)→(9), (15)→(10), (16)→(11), (17)→(12), (18)→(13), (6)→(14), (7)→(15), (8)→(16), (9)→(17), (10)→(18). Appendix equations (A.1), (B.3), (D.6), (E.1), (G.1), (H.4) … keep their labels.

Propositions: old 1→1, old 4→2, old 2→3, old 3→4.

Tables (main text): 1→1, 2→2, 3→3, 4→4, old 5→Supplementary Table ST08b, old 6→5, old 7→6. Supplementary tables ST01…ST21 keep their labels.

Figures: unchanged (Figures 1–15, S01–S04).

## 6. Supplement additions

Material removed from the main text because it concerns the Uncorrected continuation or the Smoothing-trained variant goes to a new **Supplementary Note S8. Further continuations of the base network** (EN and CN), and the old Table 5 becomes **Table ST08b** inside it. Write these additions in your notes file, under the heading `## SUPPLEMENT ADDITIONS`, in EN and CN, in the plain style of this brief, with every number preserved.

## 7. Output contract for section agents

For your unit `Uxx`, write:

- `rewrite_plain/sections/Uxx_EN.md` — the new English text of the unit, starting with its heading line(s) in the form used by the source (`## 3. The NICE method`, `### 3.1. Condensed stiffness from the strain energy`).
- `rewrite_plain/sections/Uxx_CN.md` — the Chinese text, paragraph-aligned with the EN file (`## 3. NICE 方法`, `### 3.1. 由应变能计算凝聚刚度矩阵`).
- `rewrite_plain/sections/Uxx_NOTES.md` — (a) every sentence or number of the source that you removed or moved, each with where it now lives (another unit, an appendix, the supplement, or "duplicate of …"); (b) every cross-reference you renumbered; (c) `## SUPPLEMENT ADDITIONS` if any; (d) open questions.

Do not edit any other file.
