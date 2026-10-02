# P1 论文交接（逐节修改阶段结束，2026-10-02）

本文档交给接手的 Codex，用来继续 P1 论文（CMAME 投稿稿）的收尾工作。本文档中没有任何服务器凭据；本阶段只改了论文文本和一张图，没有运行实验，也不需要访问 GPU 服务器。

## 1. 仓库与构建

- 仓库：`fanzhirui1004-source/curly-octo-goggles`，分支 `claude/wizardly-euler-3m9cwx`。只往这个分支推。
- 论文源文件（Markdown 是唯一的源，LaTeX/PDF 由脚本生成，不要手改 .tex）：
  - `docs/paper_p1/MANUSCRIPT_EN.md`：正文
  - `docs/paper_p1/APPENDICES_EN.md`：附录 A–I
  - `docs/paper_p1/SUPPLEMENTARY_EN.md`：补充材料（Note S1–S10、Table ST01–ST27）
- 构建（在 `docs/paper_p1` 下执行）：
  ```
  python3 latex/gen_figures_md.py
  cd latex && python3 build_tex.py && python3 build_supp_tex.py
  pdflatex main.tex && pdflatex main.tex     # 正文 PDF，当前 91 页
  pdflatex supp.tex && pdflatex supp.tex     # 补充材料 PDF（supp.tex/supp.pdf 在 .gitignore 中）
  ```
  每次构建后检查 `grep -c undefined main.log` 和 `grep -c "^! " main.log` 都为 0。
- 文献链接写法：正文中引用为 `[Author et al. (Year)](https://doi.org/...)`，`build_tex.py` 据此生成 `\href`。新增引用时沿用同一格式，并在文末参考文献表中补条目。
- 图：`docs/paper_p1/figures_src/` 下的脚本生成 `docs/paper_p1/figures/*.{png,pdf,svg}`；配色统一在 `figures_src/figstyle.py`（方案 A）。`figures_src/codex/` 是早先由 Codex 生成的构建管线，图 5/8 等的数据绑定在其中。
- 提交信息末尾的署名行按仓库已有提交的惯例写；提交信息、正文和代码注释中都不要出现任何模型标识。

## 2. 本阶段做了什么

作者和 Claude 逐节讨论、逐节修改了摘要、引言、第 2–7 节，共 18 个提交（`4cbbdcc` 到 `657f6be`）。每节的做法是：先给作者英文原文和中文对照，再列建议的改动（同样中英对照），作者同意后写入、重编 PDF、推送。

### 2.1 已定的主线与措辞（后续改动必须保持一致）

- **核心主张**：学习组件模型可以作用在"离散空间随几何变化"的组件的**完整迹**上（每胞 1.7–3.4×10⁴ 个保留自由度），同时保持静力凝聚的结构（以精确 Schur 补为下界），精度可在部署时不重训地改进，误差可追溯到柔度和局部灵敏度。切割 Schwarz-P 胞是这一主张的实现实例，不是全部范围。
- **口号句**（摘要、引言第 4 段骨架、结论都用）："Learning supplies the trial field, the variational form the structure, and the correction the improvability."
- **对比对象的表述**：均匀化（尺度分离失效，单层切割低估柔度 27–37%）；学习子结构/PIML（能组装、对称正定，但只保留少数边界自由度，输出维数增大时学习变差，训练后精度固定）；组件降基法（需要参数族共享离散空间，切割破坏了这一点）。对 PIML 要公允：承认其能组装，限制按 Guo et al. (2026a) 原文表述。
- **谱条件**：单调性保证只要求光滑区间上端点不低于最大特征值，b ≥ λ_max(D⁻¹A)。全文统一写作 "when the upper end of the smoothing interval bounds the spectrum"（或"谱位于 (0,b] 内"）。**不要**再写 "the smoothing interval contains the spectrum"，因为区间是 [b/30, b]，低于 a 的模态确实存在。
- **3% 线**：两胞和格架算例中的 3% 只是共同参照线。正文一律写 "3% line"，不要写 "reference"，因为 "reference" 专指精确离散参考解。
- **变体名称**：NICE、Base network、Uncorrected、Smoothing-trained、'Base network, corrected'（部署时施加校正、不重训的基础网络，第 5.1 节已命名）。
- **术语**：retained DOFs、interior、cut band、active element（背景单元与 Ω 交集测度为正，用区间包络认证；第 2.1 节和附录 A.1 已定义）、weak support（对角刚度块低于活跃节点中位数的 1%，第 4.4 节已括注）、energy share、two-grid correction。"background cell" 一律改为 "background element"，cell 专指 TPMS 胞。

### 2.2 本阶段改正的事实性问题（供核对时参考）

1. 能量误差范围原写 "2–35%"，实际 H1 为 1.8%（表 ST05a 1.814%，δ²κ 复核 1.88%），全文改为 1.8–35%（贡献段 (ii)、5.4、6.1、结论）。
2. "校正把未带校正训练的网络误差降两个数量级"不实（6.89% → 0.0965%，约 71 倍），摘要中已不用该说法；结论里写的是实测数字。
3. "NICE 消除了大部分几何依赖"改为如实陈述：各分层比 Uncorrected 低 64–103 倍，但从未切割到重度切割仍增长约 7 倍。
4. 图 10 原先无理由排除 L1，已补回 L1 重绘（`figures_src/fig10_participation.py`，465 个点、58 个组合）。
5. 5.10 删除了"NICE 最终设计的精确柔度更低"（差 1.25×10⁻⁵，属噪声）；补了均匀化对比的结论句：均匀化对响应误判 27–37%，但其设计在切割几何上与 NICE 设计相差约 1% 以内，并可被 NICE 继续改进。
6. 5.9 补充：32 线程时直接求解 341–956 s（表 ST17b）；明确各路线硬件不同，比较的是部署方式下的成本。
7. 表 1 从第 2 节移到第 5.1 节；第 5.1 节新增验证方向九类的定义段（附录 G.2），ghost-penalty 能量份额数据随之移入。
8. 引言第 5 段（PIML）引用 Guo 2026a 引言原话说明输出维数问题，并给出规模对比：其最大全节点算例 602 个节点、1,806 个自由度，我们每胞保留自由度约为其 10–19 倍。
9. 引言第 4 段补了光滑聚集多重网格 [Vaněk et al. (1996)] 作为"初始延拓加光滑"的先例，贡献段不再声称校正本身是新的。

### 2.3 投稿格式（已核实 CMAME 官方 Guide for authors）

- 摘要 ≤250 词，当前 245 词；关键词 1–7 个，当前 7 个。见 `review_r1/CMAME_FORMAT_NOTES_20261002_CN.md`。
- Highlights 3–5 条、每条 ≤85 字符，草稿已写好；生成式 AI 使用声明必须有，见 `review_r1/SUBMISSION_ITEMS_DRAFT_20261002_CN.md`。

## 3. 还没做的

按优先级排列：

1. **生成式 AI 使用声明（需作者填写，不能代写）。** 模板在 `SUBMISSION_ITEMS_DRAFT_20261002_CN.md`。作者给出工具和用途后，在参考文献前加一节，标题固定为 "Declaration of generative AI and AI-assisted technologies in the manuscript preparation process"。
2. **附录与补充材料的逐段审查。** 本阶段只核对了被正文引用的表和小节。需要检查：与 2.1 节措辞是否一致（谱条件、3% line、变体名称、"background cell"）；数字是否与正文一致；交叉引用是否有效（如 S9.1/S9.4、G.2）。
3. **篇幅（方向 8）。** 正文 16,385 词，其中第 5 节 7,817 词、引言 2,204 词。压缩前先列出候选段落和理由，交作者决定，不要直接大段删除。
4. **英文润色（方向 9）。** 通读全文统一语言；不改动数字和结论。
5. **归档与文件命名（方向 10）。** `docs/paper_p1/` 下草稿、日志等文件很多；删除或重命名前先列清单给作者确认。

## 4. 工作约定（作者的长期要求）

- 用中文和作者交流。作者希望对方是讨论伙伴，不是一味顺从；有不同意见要直说并给理由。
- 改稿时先给英文原文和中文对照，再给建议改动（同样中英对照），作者同意后再写入。作者已授权剩余部分可以自行审改，但涉及定位、结论口径、删除内容的改动，仍应先问。
- 删除文件前先列出来请作者确认。
- 不编造数字。每个数字都要能追到补充表、`docs/paper_p1/evidence/` 下的记录或 `RESULTS_A3_CN.md`；找不到出处就标出来，不要猜。
- 论文正文不提重复测量、机器负载、限流、共享主机之类的运行细节。硬件只写一块 CPU（AMD EPYC 9654）和一块 GPU（RTX 5090）。
- 论文中不得出现早期内部代号（H2/y、P0、S8、NICE-post 等）。
- 不要在仓库文件里写入任何凭据。
- `dataset_independent_20260910` 永远不删。
