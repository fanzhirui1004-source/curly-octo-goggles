# 主图 11–13 与补图 S01–S06 独立审查

审查日期：2026-10-02。仓库：`/Users/fffffreeze/Desktop/curly-octo-goggles`。基线经 `git rev-parse HEAD` 核对为 `700af856f2b9f152ab2c263406c54132a11255bf`；开始时工作区干净。本报告仅记录问题候选，编号 `TF01`–`TF04` 供主报告重新分配，不占用主报告的 S 编号。未改论文、图形、代码或生成文件，未重编、未运行有限元或训练实验。

先阅读了正文 5.7–5.10、相关补充材料和九张当前图注，再阅读 `review_r1/HANDOVER_CODEX_20261002_CN.md`。历史交接只作为语境。当前图号按正文引用识别，不按文件的 F/S 数字识别；未把 `FIGURES.md` 的派生索引当成投稿图的事实依据。

## 1. 结论与检查边界

九张当前 PNG 全部实际查看；现有构建 PDF 中对应九页全部渲染查看。未见丢图、面板裁切、当前图号错配，抽核的主要数值与当前表格/证据记录一致。主要建议是改善最终尺寸下的文字可读性、图 12 下条带的比较对象标注、补图 S01 的跨面板图例，以及补图 S05 的数值结果资格说明。这些问题均不要求改变论文定位或主要科学结论。

PDF 只抽检本报告列出的九页，不能据此宣称正文 91 页、补充 78 页逐页合格。没有重推数学命题、没有重新执行图形脚本、没有复现数值实验，没有阅读全文引用文献。数字核验覆盖范围见下一表；未抽核部分仍由其他审查角色负责。

现有 PDF：

- [main.pdf](/Users/fffffreeze/.cache/p1-paper/verification-20261002/docs/paper_p1/latex/main.pdf)，91 页；查看印刷页 38、48、49。
- [supp.pdf](/Users/fffffreeze/.cache/p1-paper/verification-20261002/docs/paper_p1/latex/supp.pdf)，78 页；查看印刷页 73–78。
- 辅助渲染：`/Users/fffffreeze/.cache/p1-paper/review-20261002/structure_figures/tail_qa/`，120 dpi，文件为 `main_p38.png`、`main_p48.png`、`main_p49.png` 与 `supp_p73.png`–`supp_p78.png`。

缓存构建目录中九张图 PDF 与基线图 PDF 的文件哈希不同，但逐图比较的解码绘图内容流、提取文字及页面边界全部相同。例如 F06 的创建日期元数据不同；因此没有把文件哈希差异判成图形版本冲突。该检查不是整份论文 PDF 与 Markdown 的逐字一致性证明。

## 2. 逐图覆盖表

下表路径均相对于上述已核对仓库；全部九图均检查了实际原图的面板标签、轴名、图例、图注、论证用途及现有 PDF 排版。

| 当前图号 / 源行 | 当前图文件 | 实际 PDF 页 | 用途与可保留内容 | 数字/绑定核验范围 | 结果 |
|---|---|---:|---|---|---|
| 11；正文 450–452，论述 448 | `figures/F06_bernstein.png` / `.pdf` | main 38 | 将保留自由度数、柔度和局部灵敏度并列，明确只画 H1/x、两胞均用精确算子、非盒面切带不受限。可保留。 | `evidence/piml4_all_fresh_val_2005_d1_v0.json:/results/0` 全部五阶；字段 `free`、各阶 `ctrl_dofs`、`test_face_compliance_max`、`gate_compliance_max`、`test_face_sens_max`、`gate_sens_max`，均与 ST16a 的 H1 行及图形趋势一致；检查了小数转百分数。未重解受限系统。 | 未见数值或图注错误；“All faces: 6 loads”在图注中已限定为目标/邻胞六个面载荷且排除切面牵引，可保留。 |
| 12；正文 519–521，论述 491–499 | `figures/F13_optimisation.png` / `.pdf` | main 48 | 同时展示精确孪生优化、同设计精确核验、均匀化起点继续优化，保留体积下降导致早期柔度上升的阴影说明。 | 阅读 `fig_opt.py:218–289` 的绑定；抽核 case A/B1/B2 历史的初末柔度、迭代数及前十次 `V_rel`，阴影覆盖的 0–3 次迭代确实体积超限；case A 0/12/23 精确检查的绑定已读。未逐点重算全部曲线。 | TF01、TF02；无证据要求删图。 |
| 13；正文 523–525，论述 501 | `figures/F14_designs_scale.png` / `.pdf` | main 49 | B2 与 X-z 使用同一厚度色标；图注明确显示 z=0 层、圆/方点意义，以及被移除区域内的点仍是切胞角点。规模结果与最终设计在同图但用途可分辨。 | `review_r1/results/X6_opt/scale/scale_summary.json` 的四个成功规模：全部 `iter_s`、`host_peak_gb`、`sampler_gpu_used_max_gib` 与 ST26 抽核；24/51/88/110 胞分别 4/2/4/4 次分析。未逐顶点核对两个厚度场；未验证硬件容量口径。 | TF01；另有待核验容量线，见第 5 节。 |
| S01；补充 1183–1185，Note S2 569–599 | `figures/S06_reference_verification.png` / `.pdf` | supp 73 | 区分背景加密、ghost penalty（幽灵惩罚）系数、矩导数差分步长。图注明确 H1 的非单调性以及 n=48 差值不能当 n=32 真误差，必须保留。 | 阅读 `fig_refconv.py:24–35`：`ref_valid.json` / `ref_valid_h1.json` 的 `vs_finest`、`gamma`、`fd.rel_to_h1e_5`、`fd.direct_vs_sens` 绑定；对照 Note S2/ST14 对参考层级的限定。未逐条重算三个面板的全部数值。 | TF04；不要因当前补表含 n=64 就声称本图本身画了 n=64。 |
| S02；补充 1187–1189，ST03 112–167 | `figures/S01_distributions.png` / `.pdf` | supp 74 | 每几何散点与中位数可揭示尾部及离群值，不能用均值柱状图替代。九类使用同一对数轴，五变体颜色/符号一致。 | 对 `newval2_v2L1.json`、`newval2_A0_ctrl.json`、`newval2_A2b_tail8.json`、`newval2_B2grid.json`、`newval2_A3_2grid.json` 的 `per_geo/*/0/<class>` 做全部 5×9 组计数：七类均 80，`support_k` 与 `glued` 均 75，吻合图注。阅读百分数和中位数绘制代码，未重算每个点的上游能量。 | TF01；S02(h) 与 (i) 标题间距很紧，但未观察到实际字形覆盖，不能报告为已发生重叠。 |
| S03；补充 1191–1193，ST05a 183–210 | `figures/S03_sensitivity_diagnostics.png` / `.pdf` | supp 75 | 区分误差与一阶项份额，材料体积分数组别互斥，误差贡献和单元数量并列有效。六胞/五胞数量及 H2 缺少 Uncorrected 已说明。 | 读 `figures_src/codex/_build/data/S03_sensitivity_diagnostics_bindings.json` 的来源映射，抽核 M1 consistent traction（一致牵引）下基础网络能量 13.5119%、一阶项份额 7.1644%，与 ST05a 和图相符。未逐元素重算(c,d)归一化。 | 可保留。 |
| S04；补充 1195–1197，ST06 236–268 | `figures/S02A_smoothing.png` / `.pdf` | supp 76 | 网络/零内场保持同一保留位移；实/虚线、实/空心区分起点；零起点灵敏度只记录 k=32，图注明确，未画出虚构轨迹。 | 阅读 `S02A_smoothing_bindings.json` 和作图代码，抽核 U1 consistent traction 下 k=32：网络/零起点能量 0.27998%/2680.874%、灵敏度 0.45509%/1671.32%，吻合 ST06；未逐胞逐步抽核。 | 可保留；独立使用时可补一个简短线型图例，属作者选择。 |
| S05；补充 1199–1201，Note S3 602–615 | `figures/S02B_coarse_spaces.png` / `.pdf` | supp 77 | 六类粗空间与四个校正顺序展示必要消融；均值与 p90（第 90 百分位）区分明确，保留界面不变。 | 阅读 `S02B_coarse_spaces_bindings.json`；抽核 U1/Q1(17)/consistent traction 四序列的 7,950 个粗系数及均值：0.588904%、0.0379093%、0.0267101%、133.129%；未重新计算粗空间秩/解残差或全部 p90。 | TF01、TF03。 |
| S06；补充 1203–1205，ST23b 1069–1086 | `figures/S06_homogenised_law.png` / `.pdf` | supp 78 | 三个独立弹性分量与体积分数独立面板、离散采样点与样条曲线区分明确，图注明确单位胞和工程剪应变。 | `review_r1/results/X6_opt/homog/homog_cells.json:/cells` 的全部 12 个 τ、ρ、CH11、CH12、CH44 与 ST23b 在给定舍入精度内一致；检查 `fig_opt.py:424–445` 的读取和平均规则。未复算周期均匀化或插值导数。 | 可保留。 |

## 3. 问题候选

### TF01 — 最终 PDF 中若干普通标签小至约 4 pt

- **严重性：建议修，优先处理。**
- **当前源位置：**[正文 519–525](https://github.com/fanzhirui1004-source/curly-octo-goggles/blob/700af856f2b9f152ab2c263406c54132a11255bf/docs/paper_p1/MANUSCRIPT_EN.md#L519)；[补充 1187–1201](https://github.com/fanzhirui1004-source/curly-octo-goggles/blob/700af856f2b9f152ab2c263406c54132a11255bf/docs/paper_p1/SUPPLEMENTARY_EN.md#L1187)。图尺寸配置对应 `figures_src/fig_opt.py:369–388`、`fig_s01_distributions.py:28–29`、`figures_src/codex/build_publication.py:355–382`。
- **短英文 / 中文：**图 12 的 “exact condensation (twin)” / “精确凝聚的孪生优化”；图 S05 的 “Coarse family / coarse DOFs” / “粗空间族 / 粗自由度数”。这些是读懂比较所必需的普通标签。
- **证据：**main 48、49 和 supp 74、77 的整页渲染。用 pdfplumber（PDF 字符提取工具）读取实际放置后的字符尺寸：main 48 的大量图例字母约 4.38 pt，main 49 注释约 4.4–4.8 pt；supp 77 横轴家族/数字等普通标签约 3.85 pt，图例约 4.01 pt。这里未把公式上、下标的更小字号计为主要问题；也不声称某个官方最小字号被违反。
- **建议方向：**以最终放置尺寸重新设定字号。优先放大补图 S02、S05 在独立页上的占用面积；图 12 可以增大整体尺寸、缩短局部图例。图 13(c) 可考虑移到设计图下方或单列以增加可读面积，但是否拆图由作者选择。避免只提高 PNG 分辨率：矢量文字已经清晰，问题是物理尺寸。
- **影响：**定位否；科学结论否；删除否。可能改变排版与图页占用。
- **确信度：高。核验范围：**上述四页整页视觉检查和实际字符字号；其余五页也查看，但未对每个标签逐字测量。

### TF02 — 图 12(a) 下条带的轴名未就地区分两个比较对象

- **严重性：建议修。**
- **当前源位置：**[正文图注 521](https://github.com/fanzhirui1004-source/curly-octo-goggles/blob/700af856f2b9f152ab2c263406c54132a11255bf/docs/paper_p1/MANUSCRIPT_EN.md#L521)；`figures_src/fig_opt.py:238–250`。
- **短英文 / 中文：**“NICE vs exact (%)” / “NICE 相对精确值的百分比”；图注已有 “the designs of the two runs coincide at iterations 0–4 and differ afterwards” / “两条运行的设计在第 0–4 次迭代相同，此后不同”。
- **证据：**`rel_twin = 100 * (CA[:n] / CX[:n] - 1)` 将两条各自变化的设计路径比较；`rel_chk` 则为同一 NICE 设计上的精确检查。图上连线/叉号和三个空心圆确实画在同一纵轴，原图和 main 48 可见。当前图注已正确解释，因此不是图中数值错误。
- **建议方向：**把下条带轴名改为较中性的 “Relative compliance difference (%)”（相对柔度差），并在下条带就地标注 “vs twin design” / “vs exact, same design”（相对孪生设计 / 同一设计的精确解）；同设计核验点保持空心圆。不应把整条线叙述为 surrogate error（代理模型误差）。
- **影响：**定位否；结论否；删除否。避免读者把第 16/19 次迭代差值解读为同设计代理误差突然增大。
- **确信度：高。核验范围：**作图公式、当前图注和实际上下条带；没有重新运行精确检查。

### TF03 — 图 S05 需要就地带出 PU 数值记录的资格边界

- **严重性：建议修。**
- **当前源位置：**[补充图注 1201](https://github.com/fanzhirui1004-source/curly-octo-goggles/blob/700af856f2b9f152ab2c263406c54132a11255bf/docs/paper_p1/SUPPLEMENTARY_EN.md#L1201)，对应 [Note S3 615](https://github.com/fanzhirui1004-source/curly-octo-goggles/blob/700af856f2b9f152ab2c263406c54132a11255bf/docs/paper_p1/SUPPLEMENTARY_EN.md#L615)。
- **短英文 / 中文：**“Appendix F.1 and Supplementary Note S3 explain the rank and solve conditions” / “附录 F.1 和补充 Note S3 解释秩与求解条件”；Note S3 明确写 “numerical observations rather than verified Galerkin projections” / “数值观察，而非经过核验的 Galerkin 投影”。
- **证据：**图中 PU(9)、PU(17) 与 Q1/Q2 共用同类标记序列；横轴称“coarse DOFs”，读图时很容易视为已验证独立维数。Note S3 已指出生成函数的系数冗余，且粗解精度尚未验证，Q1(17) 是主要粗校正结果。此处仅核对已有资格声明，不独立判定 PU 解错误。
- **建议方向：**图注用一句直接限定代替泛指跨引，如 “PU entries are numerical observations; the accuracy of their potentially singular coarse solves was not verified (Note S3).”（PU 条目是数值观察；其可能奇异的粗系统求解精度尚未核验，见 Note S3。）必要时将 PU 下方数值称 surviving coefficients（筛选后保留的系数数），避免将数量等同秩。保留 Q1(17) 的主要论证地位。
- **影响：**定位否；主要结论否；删除否。影响 PU 诊断结果被读者理解的证据等级，不建议删掉诊断结果或添加伪逆/抖动来改变它。
- **确信度：高。核验范围：**Note S3、图注、当前图、绑定的粗系数数；未重算秩、Galerkin 投影或线性解精度。

### TF04 — 图 S01 的共同图例在(c)面板发生语义切换

- **严重性：建议修。**
- **当前源位置：**[补充图注 1185](https://github.com/fanzhirui1004-source/curly-octo-goggles/blob/700af856f2b9f152ab2c263406c54132a11255bf/docs/paper_p1/SUPPLEMENTARY_EN.md#L1185)；`figures_src/fig_refconv.py:27–35,51–54`。
- **短英文 / 中文：**共同图例 “compliance” / “柔度”对应实线、实心，“sensitivity” / “灵敏度”对应虚线、空心；(c)图注却正确写 “relative change of the sensitivity … (filled)” / “灵敏度的相对变化……实心”。
- **证据：**(a,b)的实心/空心确实区分柔度/灵敏度；(c)的两组数据都是灵敏度相关核验：实心是换步长后的灵敏度变化，空心是直接柔度差分与灵敏度之差。实际图中该共同图例横跨全图，(c)也有局部解释“open: direct compliance difference vs sensitivity”。这不是数值错误，图注可解开，但读图需要处理一次相反的视觉语义。
- **建议方向：**将共同“compliance / sensitivity”图例显式限定为 panels (a,b)（面板 a、b），并给(c)两类点独立短标记；也可仅保留共同胞体标记，其他意义写在各面板。保留 n=48/n=64 的参考层级警示。
- **影响：**定位否；结论否；删除否；只统一视觉编码解释。
- **确信度：高。核验范围：**图例、三个面板、图注和直接作图字段，未重算导数。

## 4. 可压缩/调整菜单与科学内容保留理由

以下均为作者选择，未实施；减字区间是编辑目标估计，不是已经完成的节省量。计词方法：将行内数学替为一个 `MATH` 占位，再用英文词/数字正则计数；因此不等同投稿系统的字数。

| 候选 | 当前近似词数 | 可考虑的调整 | 估计与必须保留内容 |
|---|---:|---|---|
| 图 12 图注，正文 521 | 221 | 用 TF02 的局部图例承接两个比较对象；把扰动的两个确切索引与三点误差数值交给 ST24a/ST22，以一句带出处的说明保留事件。 | 可尝试约 30–50 词的压缩。必须保留：两条路径设计不同、同设计精确核验、归一化分母 C0、继续优化各用自己的迭代轴、体积阴影意义。不能为缩短而删除这些可比性条件。 |
| 图 S01 图注，补充 1185 | 202 | 将具体步长误差数值和“百倍下降”转由 Note S2 承接；图注保留面板量的定义与用于正式比较的 n=32。 | 可尝试约 25–40 词的压缩。必须保留 H1 不单调及 n=48 差值不能估计 n=32 误差的限定。 |
| 图 13(c)图注/图内注记 | 图注全段 122 | 作者可选择补 “first 4 analyses; 2 at 51 cells”（前四次分析；51 胞为两次），而把 CPU/GPU 的详细计量口径留在 Note S9.6。 | 这会略增字但让图独立传播时仍限定为短程计时；不要把图改成已完成 110 胞优化或已验证精度的证据。无需写运行负载、共享主机等细节。 |
| 图 12、13 的 PDF 浮动位置 | — | 让 5.10 的两张图在 Discussion（讨论）前集中出现。main 48 已开始 6.1，main 49 的图 13 打断其第一段。 | 属阅读顺序优化，正常浮动不等于科学内容错误。根报告负责全局篇幅与布局判断。 |
| 图 S02、S05 | — | 保留全部数据类别，以放大占页面积解决阅读问题；不优先删面板。 | S02 保留尾部信息；S05 保留初始化和校正顺序的区分。应先给图更多空间，再考虑拆成两个关联图。 |

建议顺序：先解决最终尺寸可读性，再修图例/比较对象的局部解释，再压缩图注的重复数值，最后协调全局浮动位置。图 11 的“精确算子下仅改变边界表示”、图 S01 的离散参考局限、图 S04 的同一保留位移与未记录点、图 S05 的 PU 资格条件、图 S06 的工程剪应变约定都应保留。

## 5. 另列待核验项，不据此断言数据错误

图 13(c)的 `GPU capacity`（GPU 容量）横线在 `fig_opt.py:345` 写成 `32607/1024 = 31.8428 GiB`；补充 1154 写“31.4 of 31.4 GiB”。`scale_summary.json:/plateS110/sampler_gpu_used_max_gib` 为 31.35，旧 `scale/runs/plateS110.log:25` 的内存分配失败记录则报告总可用 31.36 GiB。物理容量和计算框架可用容量可能存在不同口径，本次没有找到直接记录 `memory.total` 的原始字段，不能判定某一个容量数字错。

若保留容量横线，应给出其数据来源并说明图中究竟使用哪一种容量；若不影响叙事，也可只保留实测显存占用和 135 胞分解阶段失败的事件。此项确信度为“内部口径差异：高；错误归因：低”，不影响已记录的成功规模时间、内存峰值或主要结论。任何删线决定均由作者讨论。

## 6. 可保留的关键内容

1. 图 11 没有把本流水线的边界表示消融冒充为对已有学习子结构方法的复现；正文和补充都限定了比较范围。
2. 图 12 通过体积阴影解释早期目标上升，并真实保留扰动事件；不应润色成单调优化轨迹。
3. 图 13 的颜色是角点厚度参数而非已生成材料的三维实体图，当前图注对此明确；不应将其误报为几何渲染造假或缺少三维结果。
4. 图 S01 的 H1 非单调警示、S02 的样本数差异、S03 的五/六胞差异、S04 的零起点灵敏度缺失说明均已存在，属于应保留的证据边界。
5. 图 S06 的 12 个点及 ST23b 舍入一致，曲线/采样点区别清楚；无需增加无依据的新数值。

审查完成后应由主报告统一合并上述候选；本报告没有对其他图、整篇文献、全部数学推导或所有原始数值作覆盖承诺。
