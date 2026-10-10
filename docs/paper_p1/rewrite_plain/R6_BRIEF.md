# R6 brief: case-by-case formal revision of the main text (binding for every R6 editor)

## 0. Why

After the R5 rewrite the author judged the text too colloquial and lecture-like: short choppy sentences chained with
这些/这一/因此, explanatory asides (即……, 称为……, parenthetical explanations of ordinary words), meta-commentary, and
colloquial words (做法、有的、却、多得多、反过来、那部分 …). A first R6 guide distilled from the PIML group's papers
(`R6_STYLE_FROM_PIML.md`) was then judged **over-templated**. The author's instruction is: **handle each paragraph on its
merits** ("按情况处理"). Use `R6_STYLE_FROM_PIML.md` only as background (its §3 term map and §4.8 word list are useful);
do **not** apply its sentence-length targets (§4.2) or its paragraph templates (§2).

The reader is a senior professor of computational mechanics who reads the Chinese version first. The target register is
formal academic prose that is easy to follow, not bureaucratic and not lecture notes.

## 1. Default: leave the paragraph alone

For every paragraph decide one of three actions and record it:

- **不动** (unchanged): the paragraph already reads as formal academic prose. Derivations, definitions, propositions,
  table/figure blocks and most of Sections 2–4 should normally stay as they are.
- **词句修改** (word/sentence level): replace colloquial words, drop lecture-style asides, and re-join sentences that
  were split from one point. Expected for most edited paragraphs.
- **重组** (restructure): only where the logic of the paragraph is genuinely unclear or written as a lecture
  (for example "第一项是……第二项是……"). Rare. Keep the paragraph boundaries.

## 2. What to change (only these problems)

a. **Colloquial words (CN)**, replace in context: 做法→方法/方案；有的→一类/部分；却→而/但/则（视语境）；多得多→远多于；
得不偿失→平实陈述其结论；反过来→反之/另一方面；那部分→……的部分/分量；很→较/相当/少量（与 EN 程度一致）；
合在一起→合并；"有多/为何/如何/什么/哪些/何处" in indirect questions → noun phrases（……的精度/原因/方式/内容）；
因为→由于（视语境）；下面→以下/本节；可以看出→由……可知/……表明；发展得最为充分→研究最为系统；大部分时间→主要部分；
数字写成文字的百分比（如"百分之一"）→阿拉伯数字（数值本身不变）；从不→不。
Do **not** mass-replace 都→均, 只→仅, 可以→可, 需要→需, or 被-sentences; change them only where the particular
sentence reads colloquially.

b. **Lecture-style asides and meta-commentary**: "，即" that explains an ordinary word (keep "即" for a mathematical
equivalence or a symbol, e.g. "精确凝聚刚度矩阵，即 Schur 补"); "称为" except at the single place a term is named;
parenthetical explanations such as "（小型输出网络）", "（即施加 ghost 罚项的单元面）" → a short definition clause or
delete if the term is already defined; "由……可以看出二者各自的作用", "下面说明", "本段说明" → state the point.
EN analogues: "that is," explaining ordinary words; "called" used repeatedly; "This shows that …" chains.

c. **Fragmented sentences**: merge a sentence into the previous one only when it merely adds the cause, condition,
number or consequence of the previous claim (typically it starts with 这些/这一/该/其/因此 or This/These/It). Keep
sentences that make a separate claim. There is **no** sentence-length target; do not make a sentence longer than it
needs to be (CN rarely above about 90 characters, EN rarely above about 40 words). Pair numbers with
"分别为……和……" / "respectively" where that reads naturally.

d. **Unclear logic** → restructure the paragraph (rare; see the unit notes in your task).

## 3. What not to do (the over-templating the author rejected)

- No sentence-length quotas, no rule like "at most two short sentences per paragraph".
- No imposed paragraph templates ("然而……为此……", "目的→设置→结果→基准→原因"), no new opening sentences such as
  "图 N 给出了……" or "本节……" when the paragraph already opens with its point. Conclusion-first paragraphs stay.
- Keep the bullet list of contributions in Section 1 as a list; do not rewrite the first sentence of Section 7 into a
  formula; do not convert figure titles.
- No "需要指出的是", "值得注意的是", "It is worth noting"; keep 载荷 (not 荷载).
- No hype words (显著、大幅、优异、新颖、novel、significant(ly) as praise, remarkable); no rhetorical questions, dashes,
  slogans, aphorisms or parallel triads.

## 4. Terminology

The R5 term table (`R5_BRIEF.md` §3) stays in force. Two adoptions from the PIML papers, each made **once**:

1. First mention of the superelement in the Introduction (Section 1, paragraph 2): EN "a substructure, or
   superelement," / CN "一个子结构（超单元）". Elsewhere keep the current wording.
2. Section 3.1, in the paragraph after Eq. (4), add one sentence:
   EN: "The form \(\widehat S=F^TKF\) coincides with that of substructure methods in which the condensed stiffness
   matrix is computed from a matrix of numerical shape functions \(N\) as \(N^TKN\)
   [Huang et al. (2023)](https://doi.org/10.1016/j.eml.2023.102041); NICE, however, does not form \(F\)."
   CN: "\(\widehat S=F^TKF\) 与子结构法中由数值形函数矩阵 \(N\) 按 \(N^TKN\) 计算凝聚刚度矩阵的形式相同
   [Huang et al. (2023)](https://doi.org/10.1016/j.eml.2023.102041)，但 NICE 不显式形成 \(F\)。"

Never call the retained DOFs "boundary DOFs", the network output "shape functions", exact static condensation the
"standard substructure method", or the substructure level a "coarse grid"; do not describe NICE as "problem-independent".
When describing the PIML papers, use their own terms (边界位移线性插值、三次 Bézier 插值、数值形函数、过采样).

## 5. Hard constraints (unchanged)

- No change to any number, unit, range, condition, scope word ("under traction loads", "at most", "in every component"),
  citation, equation, symbol, or cross-reference (section, equation, proposition, table, figure, appendix, supplementary
  label). The only intended additions are the sentence of §4.2 (U03) and the words of §4.1 (U00).
- Abstract and keywords: unchanged (EN and CN).
- Paragraph count of each unit unchanged; EN and CN keep the same paragraphs and the same content.
- Markdown structure unchanged: headings, `![Figure N](...)` lines, caption lines `**Figure N. …**` / `**图 N. …**`,
  table caption lines, pipe tables (cells: word-level only), display equations with `\tag{n}`, the Algorithm block.
- No numerical comparison with PIML, no "first" claims, no GPU/dataset/offline training cost, nothing about repeated
  measurements, machine load or shared hosts.
- EN register: plain formal English; CN is primary for readability, EN is edited where it has the same problem.

## 6. Approved samples (use verbatim for these paragraphs)

**Section 3.2, paragraphs 1–2 (unit U04)**

CN:
> 网络逼近精确位移恢复的内部部分，即映射 \(q\mapsto E_Iq=-A^{-1}K_{IP}q\)；由该近似可得位移场，并经其应变能得到全部主自由度上的凝聚刚度矩阵（第 3.1 节）。各胞元的主自由度为 2,679 至 45,900 个，且主自由度集合随切割变化，因而两种直接方案均不适用：对每个几何形成位移恢复矩阵，需对每个主自由度进行一次内部求解；而对每个主自由度设置一个网络输出，则输出维数须随切割改变。

> 第 3.1 节和第 4 节的关系以下列三项性质为前提，网络结构由构造保证其精确成立：（1）几何固定时，位移恢复关于 \(q\) 线性，非线性仅出现于几何分支，作用于位移特征的运算均为线性运算；（2）位移恢复精确再现主自由度位移和刚体运动，刚体运动在输入端分离、在输出端重构，主自由度位移在输出端重新赋值（式 (9)）；（3）式 (5) 中节点力的计算需要位移恢复的转置，该转置由线性位移路径逐个运算转置得到（附录 G.1）。第 5.2 节对上述三项性质进行了数值验证。

EN:
> The network approximates the interior part of the exact recovery, the map \(q\mapsto E_Iq=-A^{-1}K_{IP}q\); this approximation yields a displacement field and, through its strain energy, the condensed stiffness on all retained DOFs (Section 3.1). Since a cell has 2,679 to 45,900 retained DOFs and this set changes with the cut, two direct approaches are unsuitable: forming the recovery matrix for every geometry requires one interior solve per retained DOF, and a network with one output per retained DOF would need an output dimension that changes with the cut.

> The relations of Sections 3.1 and 4 rest on three properties, which the architecture enforces exactly by construction: (i) at fixed geometry the recovery is linear in \(q\), since all nonlinearity is confined to a geometry branch and every operation on the displacement features is linear; (ii) the recovery reproduces the retained displacements and the rigid-body motions, since the rigid-body motion is separated before the network and reconstructed after it, and the retained displacements are overwritten at the output (Eq. (9)); (iii) the transpose of the recovery, which the nodal forces of Eq. (5) require, is obtained by transposing the linear displacement path operation by operation (Appendix G.1). Section 5.2 verifies the three properties numerically.

**Section 5.3, paragraph 1 (unit U08)**

CN:
> 在各切割程度分组中，NICE 的平均能量误差均约为 0.1% 或更低（图 5）；在面力载荷下，基础网络各分组的均值为 NICE 的 71 至 117 倍。从未切割胞元到重度切割胞元，NICE 的误差仍增大约七倍，其分组均值由 0.016% 增至 0.109%，基础网络则由 1.14% 增至 12.7%（补充表 ST03b）。在全部 80 个几何上，NICE 与基础网络的平均能量误差分别为 0.074% 和 6.89%，单个几何平均误差的最大值分别为 0.65% 和 48.6%。在面力载荷下的 5,120 个采样测试位移中，NICE 的能量误差有 99% 低于 0.69%，最大值为 1.24%（补充表 ST03）。

EN:
> In every cut-severity group, the mean energy error of NICE is about 0.1% or less (Figure 5), and under traction loads its group means are lower than those of the base network by factors of 71 to 117. The error of NICE still grows about sevenfold from uncut to heavily cut cells: its group means rise from 0.016% to 0.109%, whereas those of the base network rise from 1.14% to 12.7% (Supplementary Table ST03b). Over all 80 geometries, the mean energy errors of NICE and the base network are 0.074% and 6.89%, and their largest single-geometry means are 0.65% and 48.6%, respectively. Over the 5,120 sampled test displacements under traction loads, 99% of the energy errors of NICE lie below 0.69%, and the largest is 1.24% (Supplementary Table ST03).

**Section 6.1, paragraph 1 (unit U11)**

CN:
> 采用完整的主自由度集合而不缩减界面，可将约化子结构模型中合并处理的两项选择分离开来：一是相邻胞元之间可传递的位移模式，二是胞元内部对每种位移模式的响应精度。网络和修正仅影响后一项选择，第 5.7 节的消融算例则表明了前一项选择的作用。即使内部位移精确，限制胞元表面位移也会改变装配后的响应，而能够给出准确柔度的多项式阶次仍可能留下明显的灵敏度误差。在所考察切割胞元的两胞元装配中，限制胞元表面位移引起的局部灵敏度误差大于 NICE 内部近似引起的误差（第 5.6 和 5.7 节）；这一比较在胞元尺寸固定的条件下进行，未采用缩减边界自由度的方法所用的分区加密或富集。此外，切割平面单元的主自由度直接承担第 5.10 节板的支撑，而均匀化模型须在切割平面上另加约束方能表示该支撑。另一方面，在切割面不承受载荷或支撑之处，这些自由度增大了装配系统的规模（第 6.3 节），而装配系统的求解占 NICE 分析时间的主要部分（补充表 ST12d）。

EN:
> Using the complete set of retained DOFs, without reducing the interface, separates two choices that a reduced substructure model combines: the displacement patterns that neighbouring cells can exchange, and the accuracy with which the interior responds to each of them. The network and the correction affect only the second choice, and the ablation of Section 5.7 shows the role of the first. Even with exact interior displacements, restricting the cell-face displacements changes the assembled response, and a polynomial degree that gives an accurate compliance can still leave an appreciable error in the sensitivity. In the two-cell assemblies of the cut cells examined, restricting the cell-face displacements produced larger errors in the local sensitivity than the interior approximation of NICE (Sections 5.6 and 5.7). This comparison was made at a fixed cell size and without the refinement of the partition or the enrichment that methods with reduced boundary DOFs use. In addition, the retained DOFs of the cut-plane elements carry the support of the plate of Section 5.10 directly, whereas the homogenised model needs a separate constraint on the plane of the cut for this support. Where the cut surfaces carry no loads or supports, however, these DOFs enlarge the assembled system (Section 6.3), whose solution takes most of the time of a NICE analysis (Supplementary Table ST12d).

**Reference (may be used or adapted): Section 1, paragraph 2 (unit U00)** — see `R6_STYLE_FROM_PIML.md` §6.1, "改写（EN）"
and "改写（CN）".

## 7. Output contract

- Edit `sections/Uxx_EN.md` and `sections/Uxx_CN.md` in place. Do not edit any other file except your notes file; do not
  commit.
- Write `sections/Uxx_R6.md`: a table with one row per paragraph (EN/CN pairs share a row):
  `| # | 段首（前 12 字） | 决定（不动/词句修改/重组） | 改动说明（一句） |`, followed by the output of the check below and a
  justification for every reported difference.
- Run `python3 /home/user/curly-octo-goggles/docs/paper_p1/rewrite_plain/check_r6.py Uxx` after editing. It must report
  unchanged paragraph counts, EN = CN paragraph counts, no number/link/maths differences other than the intended ones of
  §4, and no structure differences. Colloquial word counts should fall; sentence statistics are information only.
