# P1 再审交接说明（2026-10-08）

给接手审稿与改稿的新会话。先读完本文件，再动稿子。

## 一、必须读的文件（`docs/paper_p1/`）

改稿的源文件是 Markdown，LaTeX/PDF/Word 都由它们生成：

| 文件 | 内容 |
| --- | --- |
| `MANUSCRIPT_EN.md` / `MANUSCRIPT_CN.md` | 正文（中英逐行对齐，均 802 行） |
| `APPENDICES_EN.md` / `APPENDICES_CN.md` | 附录 A–I（命题证明在 B.2、C.1、D.1、H.1） |
| `SUPPLEMENTARY_EN.md` / `SUPPLEMENTARY_CN.md` | 补充材料（补充表 ST01–ST23、补充说明 S1–S7）；**正文数字以这里的表为准** |

当前版本的 PDF 在 `latex/main.pdf`、`latex/main_cn.pdf`、`latex/supp.pdf`、`latex/supp_cn.pdf`，可用于看版面。

## 二、按需读的文件

| 文件 | 什么时候读 |
| --- | --- |
| `study/P1_STUDY_GUIDE_CN.md` | 想快速理解全文的方法、理论与结果时（作者的通读讲义） |
| `review_r3_20261005/CMAME_BENCHMARK_CN.md` | 想知道上一轮对照 17 篇 CMAME 论文得出的诊断 |
| `review_r3_20261005/ELEVATION_PLAN_CN.md`、`ELEVATION2_DRAFTS_CN.md` | 想知道上两轮改了什么、为什么改 |
| `review_r1/GUO_SERIES_COMPARISON_CN.md` | 只在改动 1.1 节 PIML 相关句子时读；其中有逐篇核查过的事实与页码 |
| `evidence/` | 需要核对某个数字的原始记录时 |

## 三、不要读或不要用的

- `review_r1/` 下的旧草稿（如 `DRAFT_6_11_EN.md`、`PROPOSED_EDITS_*`）：数字和节号已过时，会误导。
- `advisor_review/`、`../followup/`：给导师的材料和后续工作，与改稿无关。
- 会话 scratchpad 中的任何文件，特别是含服务器凭据的脚本：**永远不要读取或复制**。

## 四、工作规则（作者长期要求）

1. 用中文回复。改稿时**先给 EN + CN 对照稿**，作者确认后再同时写入中英文源文件。
2. 提交并推送到分支 `claude/wizardly-euler-3m9cwx`；不要在提交、论文中写入模型名称，不要写入任何 token 或主机名。
3. 整轮改完之前不重建 PDF/Word/PPT，除非作者要求。重建命令：在 `latex/` 下 `python3 build_tex.py && python3 build_supp_tex.py && python3 build_tex_cn.py`，然后 `pdflatex` main/supp 各两遍、`lualatex` main_cn/supp_cn 各两遍；导师审阅版再跑 `python3 build_advisor.py`。
4. 摘要不超过 250 词（**当前正好 250 词**，加词必须同时删词）。
5. 不写、也不强调 GPU、数据集或离线训练的成本。
6. 正文不与 PIML 做数值对比，只允许定性定位。
7. 不做"首次""首创"之类的优先权声明，包括隐含的（如"have so far"）。
8. 可读性优先于压缩；第 5 节只做轻度精简。
9. 论文中不提重复测量、机器负载、降频、共享主机之类的内容。
10. AI 使用声明由作者本人撰写，不要代写。

## 五、已定的决定（不要重新提）

- 标题固定："NICE: neural-initialised static condensation with equilibrium correction, with application to the thickness design of cut TPMS lattices"。
- 第 3 节保留"结论式"小节标题；保留术语 "field-based sensitivity"；不加误差链示意图；第 4 节原表 1 已删除；不引入中间网络变体。
- 摘要不提"32% lower compliance"；正文不加板尺寸方面的说明，补充表 ST20 下"88 和 110 胞元板未计算精确参考解"的说明保留。
- gyroid 迁移测试留在补充说明 S7；多胞族联合训练写作后续工作。
- 引言第 5 段的多重网格句（"Static condensation is itself a two-level method…"）、命题 3(b)、命题 4 后的注 1、表 3（装配精度一览）都是上一轮刚确认加入的。
- 编号现状：图 1–15，表 1–7，附录新增式 (H.6)，原 (H.6) 已改为 (H.7)。

## 六、这一轮建议的审稿重点

1. **防御型写作检查**，用下面的判据（作者已认可的版本）：

   > 逐句检查带有限定、免责或预先回应质疑性质的句子。对每一句问：删掉它，认真的读者会不会得出错误结论，或缺少判断结论所需的信息（适用条件、假设、误差的参考基准、比较是否公平、已知局限）？会：保留，但改写成具体事实（条件、数字、出处），只在最相关的位置说一次，删掉其他位置的重复。不会：删除——包括回应没人会提的质疑、重复已说过的限定、"值得注意的是"一类只表姿态的话、用"可能""在一定程度上"代替具体条件的模糊对冲。假设、适用范围、误差基准和局限不能删，但要集中写在对应位置。先列出每一处拟删或拟改的句子、理由和改后文字，由作者确认。

2. **数字一致性**：摘要、引言、表 3、第 5 节、结论、补充表之间的同一数字是否一致。
3. **中英文对齐**：两版内容是否逐句对应（行数应保持一致）。
4. **可读性**：第 5 节证据句中数字是否过密（已移入表 3 的数字不必在正文重复），但不要为压缩牺牲可读性。

审完先给出问题清单与 EN + CN 改稿，不要直接改动源文件。
