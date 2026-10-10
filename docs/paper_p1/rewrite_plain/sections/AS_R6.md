# R6 word-level pass: appendices (A1–A3) and supplement (S1–S4, S8)

Scope: colloquial words (R6 brief §2a), hype words, lecture-style asides (§2b). No sentence merged or split, no paragraph
restructured; derivations, equations, numbers, conditions, citations, labels, cross-references and table cells untouched.
All changes are in the CN files. The EN files needed no change: they contain no hype word used as praise and no aside
explaining an ordinary word (see "Flagged words left unchanged").

## Changes per file

| File | Changes | Paragraphs changed |
|---|---:|---:|
| A1_CN.md | 7 | 6 |
| A2_CN.md | 3 | 2 |
| A3_CN.md | 1 | 1 |
| S1_CN.md | 2 | 2 |
| S4_CN.md | 3 | 2 |
| A1_EN, A2_EN, A3_EN, S1_EN, S2_EN, S2_CN, S3_EN, S3_CN, S4_EN, S8_EN, S8_CN | 0 | 0 |
| **Total** | **16** | **13** |

## Change list (old phrase → new phrase)

| # | File, location | Old | New | Reason |
|---|---|---|---|---|
| 1 | A1_CN, A.1 para. 1 (last sentence) | ……也取为主自由度，因为其基函数决定…… | ……也取为主自由度，因其基函数决定…… | 因为 (§2a); post-positioned cause, formal 因其 |
| 2 | A1_CN, B.2 para. "进一步假设……" (before Eq. (B.2)) | ……精确场为 \(Ra\)，因为它在主自由度上取这些位移…… | ……精确场为 \(Ra\)，因其在主自由度上取这些位移…… | 因为 (§2a) |
| 3 | A1_CN, B.3 para. after Eq. (B.8) | 此时即使很小的相对位移误差 | 此时即使较小的相对位移误差 | 很 (§2a); EN "small" |
| 4 | A1_CN, same sentence | ……在力学上也是显著的。 | ……在力学上也不可忽略。 | 显著 (non-praise use, replaced by neutral wording; EN "mechanically significant" = same content) |
| 5 | A1_CN, B.3 last para. | 在未缩放时有一个非常软的模态 | 在未缩放时有一个极软的模态 | 非常 (colloquial intensifier); EN "very soft", \(0<\epsilon_0\ll1\) |
| 6 | A1_CN, C.1 last para. | 该结论针对特定载荷，因为 \(w_m\) 与…… | 该结论针对特定载荷，原因在于 \(w_m\) 与…… | 因为 (§2a) |
| 7 | A1_CN, C.2 last para. | 其中第二个不等式成立是因为 \(\widehat{\mathbb K}\succeq\mathbb K\) | 其中第二个不等式成立是由于 \(\widehat{\mathbb K}\succeq\mathbb K\) | 因为→由于 (§2a) |
| 8 | A2_CN, D.1 para. after Eq. (D.6) | ……满足该条件，因为 \(2G-GA_cG=G(A_c+2\Lambda)G\)；但它并非精确投影 | ……满足该条件，这是由于 \(2G-GA_cG=G(A_c+2\Lambda)G\)；但它并非精确投影 | 因为 (§2a) |
| 9 | A2_CN, same para. | 仍可得到 \(A_c+\Lambda\succ0\)，因为 \(V\) 中保留的每一列都具有正能量 | 仍可得到 \(A_c+\Lambda\succ0\)，原因在于 \(V\) 中保留的每一列都具有正能量 | 因为 (§2a) |
| 10 | A2_CN, F.1 para. 2 | 这一做法使前向与转置粗求解保持一致 | 这一规则使前向与转置粗求解保持一致 | 做法 (§2a); "规则" matches "粗因子主元规则" in Table ST16 (EN "pivot rule") |
| 11 | A3_CN, H.2 para. after Eq. (H.6) | 两者中哪一个给出更准确的梯度，取决于…… | 两者所给梯度的精度高低取决于…… | indirect question (哪) → noun phrase (§2a) |
| 12 | S1_CN, opening para. | 下方索引给出各变体…… | 以下索引给出各变体…… | 下面/下方 → 以下 (§2a) |
| 13 | S1_CN, R1 para. after the variant/cell tables | d0 和 d1 表示切割角 \(\vartheta\) 位于 \((0,\pi/4)\) 的哪一半区间 | d0 和 d1 表示 \((0,\pi/4)\) 中包含切割角 \(\vartheta\) 的半区间 | indirect question (哪) → noun phrase (§2a); now mirrors EN "the half of \((0,\pi/4)\) that contains the cut angle" |
| 14 | S4_CN, S6.1 para. | 该方法从不沿搜索方向检验 | 该方法不沿搜索方向检验 | 从不→不 (§2a) |
| 15 | S4_CN, same para. | ……KKT 残差，因为该估计值并非 \(\widehat C\) 的梯度 | ……KKT 残差，原因在于该估计值并非 \(\widehat C\) 的梯度 | 因为 (§2a); 因该 avoided (reads as 应该) |
| 16 | S4_CN, S7 "单个胞元" para. (last sentence) | ……不能迁移，因为它与网络训练所用的胞元族相关。 | ……不能迁移，因其与网络训练所用的胞元族相关。 | 因为 (§2a) |

## Flagged words left unchanged (check-script hits that are not problems)

- A1_EN '`significant`' (B.3): "mechanically significant" is a technical statement, not praise. The CN counterpart now
  reads 不可忽略 (change 4); content identical.
- A3_CN '`大幅`' ×2 (G.1, coefficient bounds): substring of 最大幅值 ("largest magnitude"), not the hype word.
- S3_EN '`significant`' (S4.3): "three significant digits".
- S3_CN '`首次`' ×2 (S3 route (c) and iterative solvers): temporal "first" (首次作用, 残差首次低于), not a priority claim;
  EN '`first `' hits in all files are likewise ordinal.
- S3_EN '`that is,`' and S3_CN（即……）(S4.3): defines the vertex aggregation as a sum, a mathematical definition (§2b keeps 即 for these).
- S3_EN '`called`' (S3 route (a)): "The PARDISO phases are called", programming sense.
- S1_EN, S2_EN, S4_EN '` — `': the dash placeholder in empty table cells and in "— marks entries not tabulated", not prose dashes.
- '`，即`' kept everywhere (A1 ×5, A2 ×1, A3 ×2, S1 ×1, S4 ×5): each is a mathematical consequence or identity
  (即得/即证得, \(v_h\) 刚体, \(D=\operatorname{diag}(K_{II})\) 即 Jacobi 加权范数, numeric restatements such as
  "即每次迭代 0.0255", "即 591 个训练几何和 100 个验证几何", the cube-symmetry image, \(u=0\) on the section, the
  definition of failing cells in the ST19a note); none explains an ordinary word.
- A2_CN '`称为`' (F.1, 粗因子平移): the single place where the term is named (A1 B.1 and Table ST19b refer to it).

## Considered but left unchanged (outside the §2a/§2b lists; for the author's decision)

- S4_CN S7 first para.: 训练所得的网络只见过 Schwarz-P 胞元 (EN "has seen Schwarz-P cells only") reads slightly informal;
  a formal variant would be 仅以 Schwarz-P 胞元训练 (EN "was trained on Schwarz-P cells only"). Not on the brief's list.
- A3_CN G.4 这样的半正定扰动, S2_CN S2.1 两个这样的内部空间: 此类 would be more formal; not on the list.
- S1_CN R1/ST01 数据划分共包含 691 个几何，即 591 个训练几何和 100 个验证几何: numeric breakdown, kept (EN uses a colon).

## Check output

`python3 docs/paper_p1/rewrite_plain/check_r6.py A1 A2 A3 S1 S2 S3 S4 S8`

Paragraph counts are unchanged in every file, EN = CN in every unit, and there are no numbers, links, maths or structure
differences (no DIFF lines). Colloquial counts fell: A1_CN 很 1→0, 非常 1→0, 显著 1→0; A2_CN 做法 1→0. The changes
to 因为, 从不, 下方 and 哪 are not counted by the script; they are verified by the character diff above.

```
== A1
  A1_EN.md: paragraphs 68 -> 68, changed 0
    words (old->new): 'significant' 1->1, 'first ' 3->3
    sentences old: 148 sentences, mean 14.4 words, <=15 words 58%
    sentences new: 148 sentences, mean 14.4 words, <=15 words 58%
  A1_CN.md: paragraphs 68 -> 68, changed 6
    words (old->new): '很' 1->0, '非常' 1->0, '显著' 1->0, '，即' 5->5
    sentences old: 145 sentences, mean 26.6 chars, <=30 chars 69%
    sentences new: 145 sentences, mean 26.6 chars, <=30 chars 69%
== A2
  A2_EN.md: paragraphs 40 -> 40, changed 0
    words (old->new): 'first ' 2->2
    sentences old: 95 sentences, mean 15.4 words, <=15 words 54%
    sentences new: 95 sentences, mean 15.4 words, <=15 words 54%
  A2_CN.md: paragraphs 40 -> 40, changed 2
    words (old->new): '做法' 1->0, '，即' 1->1, '称为' 1->1
    sentences old: 88 sentences, mean 29.9 chars, <=30 chars 58%
    sentences new: 88 sentences, mean 30.0 chars, <=30 chars 58%
== A3
  A3_EN.md: paragraphs 76 -> 76, changed 0
    words (old->new): 'first ' 1->1
    sentences old: 152 sentences, mean 17.4 words, <=15 words 47%
    sentences new: 152 sentences, mean 17.4 words, <=15 words 47%
  A3_CN.md: paragraphs 76 -> 76, changed 1
    words (old->new): '大幅' 2->2, '，即' 2->2
    sentences old: 157 sentences, mean 29.9 chars, <=30 chars 59%
    sentences new: 157 sentences, mean 29.8 chars, <=30 chars 59%
== S1
  S1_EN.md: paragraphs 52 -> 52, changed 0
    words (old->new): 'first ' 5->5, ' — ' 36->36
    sentences old: 84 sentences, mean 22.7 words, <=15 words 27%
    sentences new: 84 sentences, mean 22.7 words, <=15 words 27%
  S1_CN.md: paragraphs 52 -> 52, changed 2
    words (old->new): '，即' 1->1
    sentences old: 109 sentences, mean 35.6 chars, <=30 chars 50%
    sentences new: 109 sentences, mean 35.6 chars, <=30 chars 50%
== S2
  S2_EN.md: paragraphs 36 -> 36, changed 0
    words (old->new): ' — ' 1->1
    sentences old: 70 sentences, mean 19.0 words, <=15 words 43%
    sentences new: 70 sentences, mean 19.0 words, <=15 words 43%
  S2_CN.md: paragraphs 36 -> 36, changed 0
    sentences old: 81 sentences, mean 31.4 chars, <=30 chars 49%
    sentences new: 81 sentences, mean 31.4 chars, <=30 chars 49%
== S3
  S3_EN.md: paragraphs 50 -> 50, changed 0
    words (old->new): 'that is,' 1->1, 'significant' 1->1, 'first ' 4->4, 'called' 1->1
    sentences old: 133 sentences, mean 18.7 words, <=15 words 38%
    sentences new: 133 sentences, mean 18.7 words, <=15 words 38%
  S3_CN.md: paragraphs 50 -> 50, changed 0
    words (old->new): '首次' 2->2
    sentences old: 135 sentences, mean 36.0 chars, <=30 chars 41%
    sentences new: 135 sentences, mean 36.0 chars, <=30 chars 41%
== S4
  S4_EN.md: paragraphs 73 -> 73, changed 0
    words (old->new): 'first ' 7->7, ' — ' 17->17
    sentences old: 130 sentences, mean 23.0 words, <=15 words 26%
    sentences new: 130 sentences, mean 23.0 words, <=15 words 26%
  S4_CN.md: paragraphs 73 -> 73, changed 2
    words (old->new): '，即' 5->5
    sentences old: 177 sentences, mean 35.4 chars, <=30 chars 42%
    sentences new: 177 sentences, mean 35.4 chars, <=30 chars 42%
== S8
  S8_EN.md: paragraphs 14 -> 14, changed 0
    sentences old: 32 sentences, mean 17.0 words, <=15 words 44%
    sentences new: 32 sentences, mean 17.0 words, <=15 words 44%
  S8_CN.md: paragraphs 14 -> 14, changed 0
    sentences old: 31 sentences, mean 35.0 chars, <=30 chars 39%
    sentences new: 31 sentences, mean 35.0 chars, <=30 chars 39%
```
