# P1 补充材料英文与作者声音独立审读

日期：2026-10-02。仓库：`/Users/fffffreeze/Desktop/curly-octo-goggles`。固定提交：`700af856f2b9f152ab2c263406c54132a11255bf`。审读时 `git status --short` 无输出，未修改仓库。本文仅是候选清单，不是已应用的修改。L01–L09 为本辅助报告内编号，主报告整合时可以映射到全篇编号。

## 范围与证据边界

完整通读 `docs/paper_p1/SUPPLEMENTARY_EN.md` 第1–1205行，包括R1、ST01–ST27及全部子表、Note S1–S9和六张补充图的英文图注。Note S10 不存在于当前源；交接文件第11行的“S1–S10”与当前源不符。ST26、ST27用加粗段落而非标题，已纳入覆盖。正文对应读到3.3、5.1、5.6、5.9、5.10、6.2、6.4相关段落；完成独立通读之后才读交接文件。

本审读核对的是语言中的比较对象、因果连接、统计层级和证据范围，未运行有限元、训练、构建或服务器实验。数字只与当前表和少量已存记录对照，**不是逐数全量复算**。数学仅对L02的标量比值解释给出直接代数反例；未重新证明全部公式。文献只保留作者已有引文口径，未在本子任务中查阅原论文全文。六图仅读图注，未作像素级核验；正文13图的图形核验由主审合并覆盖。本报告不使用检测器，也不根据词汇风格判断写作来源。

优先结论：补充材料整体有明确的试验对象、定义和证据限定，最需要修改的不是词汇，而是两处具体的源内解释问题（L01、L02），以及少数把证据、数字和结论压进一个句子的段落。L06–L08涉及解释范围，建议交作者确认后再写入，不自动改变定位或结论。

## 候选修改

### L01｜必须修｜ST10 对3% line的说明与正文冲突，并残留旧术语

- 位置：`docs/paper_p1/SUPPLEMENTARY_EN.md:474`；关联第94、390、392–451、522行。
- 英文原文：

> Maximum relative errors (%) over the three cut-surface traction directions, for every variant and configuration of Table ST09 with a cut target. Sensitivity maxima include both cells. The target is learned and the neighbour exact. The 3% reference is not applied to these loads in the main text; values above it are marked in bold.

- 中文：对ST09中所有切割目标的变体与配置，报告三个切面牵引方向下的最大相对误差；灵敏度最大值包括两胞。目标胞使用学习模型、邻胞精确。正文不对这些载荷使用3%参照；超过该值的数值加粗。
- 问题：正文 `MANUSCRIPT_EN.md:422` 已直接写出切面载荷下Uncorrected超过3% line（3%参照线）的四个配置，并引用ST10；“is not applied ... in the main text”已不成立。用户固定术语是3% line，补充材料仍反复使用reference（参考）。ST09 Outcome（比较结果）列的“Within reference / Above reference”也会混淆精确参考解和参照线。
- 建议英文：

> Maximum relative errors (%) over the three cut-surface traction directions, for every variant and configuration of Table ST09 with a cut target. Sensitivity maxima include both cells. The target is learned and the neighbour exact. Values above the common 3% line are marked in bold, consistent with the comparison in Section 5.6.

- 建议中文：对ST09中所有切割目标的变体与配置，报告三个切面牵引方向下的最大相对误差（%）。灵敏度最大值包括两胞。目标胞使用学习模型、邻胞精确。超过共同3%参照线的数值加粗，与5.6节的比较一致。
- 为何更清楚：不改变任何误差数值或判据，只修复补充说明与正文之间的冲突，并将精确参考解和比较线分开。关联位置统一为“3% line”，ST09列名可用“Relative to 3% line”，值用“Below line / Above line”，避免被读成接受阈值。
- 证据：当前正文第422行；ST10第478–516行；交接第34行仅用于确认作者术语偏好。无需依赖历史review。
- 影响：定位否；结论否；删除否；局部事实说明需修正。
- 确信度与核验范围：高；已逐字对照正文422与补充474，并看过ST09/ST10全部行；未重算误差。

### L02｜必须修｜ST15把谱比值−1解释为负半定，超出了该统计量的信息

- 位置：`docs/paper_p1/SUPPLEMENTARY_EN.md:588`。
- 英文原文：

> A ratio of −1 means that the element derivative is negative semidefinite.

- 中文：比值为−1意味着该单元导数矩阵是负半定的。
- 问题：表中比值是 `lambda_min / max(abs(lambda))`。例如对角矩阵 `diag(-2,1)` 的比值为−1，但矩阵不定。因此该比值只说明负特征值达到了全谱的最大绝对值，不能排除较小的正特征值。这是具体数学解释问题，不是表达习惯。
- 建议英文：

> A ratio of −1 means that the most negative eigenvalue has the largest absolute magnitude; the ratio alone does not establish negative semidefiniteness.

- 建议中文：比值为−1表示最负特征值的绝对值达到全谱最大值；仅凭这个比值，不能判定矩阵负半定。
- 为何更清楚：让统计量的解释严格对应其定义，避免让读者把“存在主导负特征值”误读为“所有特征值非正”。如果作者确实希望报告负半定，应补充这些实际矩阵的最大特征值及容差证据，由数学主审核验；不能凭上述反例断言实际矩阵含正特征值。
- 证据：ST15的列定义及第592–599行；`docs/paper_p1/review_r1/results/X3/psd.json` 中抽查 `cells.fresh_val_2003_d1_v1.ad[0].q0`、`ad_uniform.q0`（均为−1）以及相应 `worst.ratio`。记录抽查确认其报告的是比值，没有在本审读中复算原矩阵全谱。代数反例由定义直接得到。
- 影响：定位否；总体实验结论未判定；删除否；局部数学解释需修正，实际矩阵符号仍待数学主审核验。
- 确信度与核验范围：对逻辑错误为高；对实际矩阵是否负半定未作判断。

### L03｜建议修｜ST03用同一段混装方向尾部统计和配对几何比较

- 位置：`docs/paper_p1/SUPPLEMENTARY_EN.md:114`，从“For NICE...”起的两句。
- 英文原文：

> For NICE under consistent tractions, the 5,120 individual sampled directions of the 80 geometries have a 95th percentile of 0.331%, a 99th percentile of 0.693% and a maximum of 1.24% (3,840 directions of the 60 geometries outside model selection: 0.355%, 0.734%, 1.24%). Under consistent tractions the mean of the corrected base network is 0.0965% and that of NICE 0.0737%, a ratio of 1.31; resampling the 80 geometries with replacement (paired, ratio of geometry-equal means, percentile interval) gives a 95% interval of 1.20–1.40.

- 中文：在一致牵引下，80个几何的5,120个NICE采样方向，其第95百分位、第99百分位和最大值分别为0.331%、0.693%、1.24%；模型选择之外60个几何的3,840个方向对应值为0.355%、0.734%、1.24%。校正基础网络与NICE的均值为0.0965%和0.0737%，比值1.31；对80个几何有放回地配对重采样，几何等权均值比值的95%百分位区间为1.20–1.40。
- 建议英文：

> For NICE under consistent tractions, the 5,120 sampled directions across all 80 geometries have 95th- and 99th-percentile errors of 0.331% and 0.693%, and a maximum of 1.24%. For the 3,840 directions from the 60 geometries outside model selection, the corresponding values are 0.355%, 0.734% and 1.24%.
>
> The geometry-equal means under consistent tractions are 0.0965% for 'Base network, corrected' and 0.0737% for NICE. Their ratio is 1.31. Paired resampling of the 80 geometries with replacement gives a 95% percentile interval of 1.20–1.40 for this ratio of means.

- 建议中文：在一致牵引下，全部80个几何共5,120个NICE采样方向的第95与第99百分位误差为0.331%和0.693%，最大值为1.24%。模型选择之外60个几何共3,840个方向的对应值为0.355%、0.734%和1.24%。另起一段：一致牵引下，'Base network, corrected'与NICE的几何等权均值为0.0965%和0.0737%，比值为1.31。对80个几何有放回地配对重采样，得到这个均值比值的95%百分位区间1.20–1.40。
- 为何更清楚：区分方向级尾部统计与几何级配对比较，避免读者把重采样单位理解成5,120个方向。保留全部数字、样本量、模型选择限定和作者比较。
- 证据：ST03第114行、第116–126行、ST03d第155–167行；ST01第72行说明20/80几何用于模型选择。
- 影响：定位否；结论否；删除否。仅重排成两段。
- 确信度与核验范围：高；完整读表并对照统计定义，未重跑重采样计算或逐一追溯5,120个方向。

### L04｜建议修｜ST04列说明没有阅读层次，八类诊断在一个句子中连续堆叠

- 位置：`docs/paper_p1/SUPPLEMENTARY_EN.md:171`第一句。后续delta、kappa公式和误差上界不动。
- 英文原文：

> Columns: \(\lambda_{\max}\) of \(D^{-1}K_{II}\) by Lanczos; a power-iteration estimate \(b\) of the upper smoothing endpoint; the Gershgorin bound; the maximum relative asymmetry \(\max_{i,j}|G_{ij}-G_{ji}|/\sqrt{|G_{ii}G_{jj}|}\) of \(G=Q^T\widehat SQ\), where the columns of \(Q\) are the first eight retained-displacement directions of the cell's consistent-traction (force_c) validation set; the maximum relative difference between returned work and field energy, the deployed-versus-training field difference and the maximum rigid-body energy ratio, all over the consistent-traction validation directions; the ghost-penalty share of the exact field energy, mean over directions (consistent tractions / nodal forces); and the mean \(\delta\) and \(\kappa\) of the base network and of NICE under consistent tractions.

- 中文：列依次为：Lanczos最大特征值、幂迭代端点、Gershgorin界、由前八个一致牵引方向构成的G矩阵相对非对称性、返回功与场能量之差、部署与训练场之差、刚体能量比、精确场中鬼罚能量占比，以及基础网络与NICE的delta和kappa均值。
- 建议英文：

> The columns report three groups of diagnostics. Spectral checks compare the Lanczos estimate of \(\lambda_{\max}(D^{-1}K_{II})\), the power-iteration smoothing endpoint \(b\), and the Gershgorin bound. Operator checks report the relative asymmetry of \(G=Q^T\widehat SQ\), the returned-work versus field-energy difference, the deployed-versus-training field difference, and the rigid-body energy ratio. For the asymmetry check, \(Q\) contains the first eight retained-displacement directions of the cell's consistent-traction (force_c) validation set, and the metric is \(\max_{i,j}|G_{ij}-G_{ji}|/\sqrt{|G_{ii}G_{jj}|}\); the other operator checks take maxima over the consistent-traction directions. Energy diagnostics report the mean ghost-penalty share under consistent tractions and nodal forces, and the mean \(\delta\) and \(\kappa\) of the base network and NICE under consistent tractions.

- 建议中文：这些列分为三组诊断。谱检查比较Lanczos得到的最大特征值、幂迭代给出的光滑端点b以及Gershgorin界。算子检查报告G矩阵的相对非对称性、返回功与场能量之差、部署场与训练场之差及刚体能量比。非对称性只使用该胞一致牵引验证集的前八个保留位移方向和原公式；其他算子检查在一致牵引方向上取最大值。能量诊断报告两类加载下的平均鬼罚能量占比，以及基础网络与NICE在一致牵引下的delta、kappa均值。
- 为何更清楚：表格仍保存列顺序，文字按读者想回答的问题组织，使谱端点、算子实现检查、误差机理诊断不再互相挤压；特别保留八方向抽检的范围，不能把它写成完整矩阵对称性证明。
- 证据：ST04第171–179行；ST02第91行；正文5.2。这里按原文“the maximum ... all over ... directions”理解后面三个量均取方向最大值，整合时宜与该表的生成字段再核对一次。
- 影响：定位否；结论否；删除否。
- 确信度与核验范围：对结构建议高；对字段顺序与最后三个量的聚合方式中高，已核正文/表头，未重读生成脚本或重算算子。

### L05｜建议修｜ST09b的“before evaluation”没有区分单胞选择与两胞评价

- 位置：`docs/paper_p1/SUPPLEMENTARY_EN.md:457`第一句；正文第420行有同类表达。
- 英文原文：

> Cells outside model selection, fixed before evaluation: the five validation geometries with NICE's largest single-cell errors and one random cell per stratum.

- 中文：未参与模型选择、在评价前固定的胞：NICE单胞误差最大的五个验证几何，加上每个分层随机一个胞。
- 建议英文：

> These cells are outside model selection. Before the two-cell evaluations, we selected the five validation geometries with NICE's largest single-cell errors and one random cell from each stratum.

- 建议中文：这些胞均未参与模型选择。在两胞评价之前，我们选择了NICE单胞误差最大的五个验证几何，并从每个分层随机选取一个胞。
- 为何更清楚：选择已使用单胞评价结果，而“在评价前固定”容易被读成从未看过任何误差结果。明确是哪一级评价，既保留压力测试的作者意图，也不把它包装成完全独立随机样本。
- 证据：ST09b标题、第457–469行的Selection列；ST03d第155行的五个最大几何均值；正文第420行。这里只澄清文本内部给出的顺序，未追溯选择清单的时间戳；若不能证明两胞计算前固定，应改成更朴素的“ The two-cell tests use ... ”，不加时间声明。
- 影响：定位否；结论否；删除否；前置选择时间仍应由实验记录确认。
- 确信度与核验范围：对歧义高；对实际选择时间未核验。

### L06｜建议修｜S5把很小的精度差写成“不会改变”，应直接用实测范围

- 位置：`docs/paper_p1/SUPPLEMENTARY_EN.md:706`末句；连同第806行核验。
- 英文原文：

> The single-precision correction itself does not change the operator accuracy: repeating the \(2\times2\times2\) solve of Section 5.8 with the correction in single precision (to \(10^{-10}\)) changes its compliance errors by less than \(3\times10^{-9}\).

- 中文：单精度校正本身不改变算子精度：在2×2×2复算中把校正换为单精度，柔度误差变化小于3×10^-9。
- 建议英文：

> The evaluated lattice re-solves show a much smaller effect of correction precision. Repeating the \(2\times2\times2\) solve of Section 5.8 with the correction in single precision, to a recursive residual of \(10^{-10}\), changes its compliance errors by less than \(3\times10^{-9}\). Supplementary Note S6.3 reports the corresponding checks for both eight-cell lattices.

- 建议中文：已评估的格架复算中，校正精度的影响要小得多。在5.8节2×2×2算例中，把校正换为单精度，并仍求解到10^-10递归残差，柔度误差变化小于3×10^-9。S6.3给出了两个八胞格架的对应检查。
- 为何更清楚：句子直接服务于本段论证——计时误差的增量主要来自求解停止精度。第806行已经补充两种八胞格架的柔度变化小于4×10^-8、梯度变化小于10^-6；不能误称只有一个复算。建议保留两处不同范围的界，用明确交叉引用连接，避免一般性的“does not change”。
- 证据：S5第704–706行；S6.3第806行；`docs/paper_p1/evidence/lat_hetero222_stream.json` 的 `args.tol=1e-10`、`args.deploy=true`、`lattices.hlat222.A3.compliance_rel_err` 已抽查。未独立复算相差3×10^-9的成对记录，数字沿用当前源。
- 影响：定位否；数值结论否；删除否；局部保证措辞限定到实测范围，交作者确认。
- 确信度与核验范围：对语言范围问题高；精度差数字仅做当前文内交叉核验及单记录抽查。

### L07｜作者选择｜S6.3结尾把三个扫描的相近跳变写成一般机制等同

- 位置：`docs/paper_p1/SUPPLEMENTARY_EN.md:823`末句。
- 英文原文：

> The non-smoothness of the surrogate objective is therefore that of the discrete model, which an optimiser driven by exact condensation meets in the same way.

- 中文：因此，代理目标的非光滑性就是离散模型的非光滑性，采用精确凝聚的优化器也会以同样方式遇到它。
- 建议英文：

> In these sweeps, the surrogate and exact discrete compliances depart from the sensitivity-predicted changes by similar amounts. This points to switching in the discrete model as the main source of the observed non-smoothness; the corresponding exact-condensation evaluations show the same behaviour.

- 建议中文：在这些扫描中，代理柔度与精确离散柔度相对于灵敏度所预测变化的偏离幅度相近。这表明离散模型的切换是所观测非光滑性的主要来源；对应的精确凝聚评价也表现出同样行为。
- 为何更清楚：把作者判断建立在直接对照上，保留“离散模型切换是主要来源”的解释；不把有限扫描写成排除了所有其他非光滑性的普遍结论。前文第808、823行也承认网络二值指示器和粗因子移位具有自身切换。建议不删这些限定。
- 证据：第823行全部上下文；`docs/paper_p1/review_r1/results/X3/E13/E13_SUMMARY.json` 的 `m1x/u1y/u1ym.n` 分别为4/27/9，`n_switch`为3/24/8，`ref.jumps[*].ref_nonsmooth_rel` 与 `hat_nonsmooth_rel`；`u1y.max_nonsmooth_rel_noswitch=6.941650664471272e-08`。已直接读这三个扫描的摘要字段，未重跑扫描，也未假定这是全部可能路径。
- 影响：定位否；解释结论的普遍性会收窄，须作者讨论；删除否。此处不应以纯润色名义静默改写。
- 确信度与核验范围：对局部证据范围高；对“主要来源”的归因是数据支持的解释，不是本子任务新增的数学证明。

### L08｜建议修（须作者确认推理范围）｜S9.1从“无线搜索”转到梯度不一致问题，连接容易过强

- 位置：`docs/paper_p1/SUPPLEMENTARY_EN.md:934`前两句。
- 英文原文：

> The update is the method of moving asymptotes [Svanberg (1987)](https://doi.org/10.1002/nme.1620240207): one convex separable approximation per analysis, solved by the primal–dual interior-point method of that note, with no line search and without the conservativeness test of its globally convergent variant GCMMA (Table ST21). The optimiser thus uses the gradient estimate only to build its approximation and never tests the decrease of \(\widehat C\) along a search direction, where the inconsistency discussed in Section 6.2 would matter.

- 中文：采用移动渐近线法，每次分析构建一个凸可分近似，由所述原始–对偶内点法求解，不用线搜索或全局收敛变体的保守性检验。因此优化器只在构造近似时使用梯度估计，从不检查沿搜索方向的代理柔度下降，而6.2节所述不一致会在那里产生影响。
- 建议英文：

> We use the method of moving asymptotes [Svanberg (1987)](https://doi.org/10.1002/nme.1620240207), with one convex separable approximation per analysis and the primal–dual interior-point solve specified in Table ST21. The update uses the field-based gradient estimate to construct this approximation. It includes neither a line search on \(\widehat C\) nor the conservativeness test of GCMMA. Its observed performance is assessed by the exact checks and design comparisons below.

- 建议中文：我们采用移动渐近线法，每次分析构造一个凸可分近似，并按ST21规定用原始–对偶内点法求解。更新利用基于场的梯度估计构造该近似，不使用代理柔度的线搜索，也不使用GCMMA的保守性检验。其实际表现由下文的精确检查与设计对比评估。
- 为何更清楚：原句的“thus”“only”“where ... would matter”容易被连读为“因为不做线搜索，所以梯度与目标不一致无关紧要”。建议直写算法事实，然后用真实检查支撑实用性；不新增收敛保证，也不猜测会失败。第934行后面的停止规则与未使用KKT残差的限定保留。
- 证据：ST21第946–947、952、955行；ST22b、ST27；正文第170行明确field-based estimate（基于场的估计）不等于完整代理导数，正文第537行讨论线搜索。本审读未查阅Svanberg原文或验证求解器实现；引文按原口径保留。
- 影响：定位否；实验结论否；删除否；改变的是局部论证连接，须作者确认。若仅要求最小改动，可把“thus ... where ... would matter”改成直接算法叙述，保留原文献句。
- 确信度与核验范围：对阅读歧义中高；未判定优化算法无效或需要改算法。

### L09｜必须修（量纲说明）；作者选择（是否强调方向）｜S9.2微小柔度差缺“相对”一词

- 位置：`docs/paper_p1/SUPPLEMENTARY_EN.md:968`末两句；对应ST22c第1023、1031行。
- 英文原文：

> Their corner parameters differ by at most 0.0051 (root mean square 0.0011), and the exact compliance of the NICE design, 24.110504, is \(1.25\times10^{-5}\) below that of the twin's design, 24.110806. The final compliance is 2.0% below that of the initial design, which holds 25% more material.

- 中文：两最终设计的角点参数最大差0.0051、均方根差0.0011；NICE设计的精确柔度24.110504比孪生设计的24.110806低1.25×10^-5。最终柔度比初始设计低2.0%，初始设计多含25%材料。
- 问题：“is 1.25×10^-5 below”没有说明该数为相对差，易被读成绝对柔度差；两个明确柔度值的绝对差并不是这个数。第1031行实际写的是relative to（相对于）。正文第491行已正确写“of their value”，并避免宣称NICE取得更优最优解。
- 建议英文：

> Their corner parameters differ by at most 0.0051 (root mean square 0.0011). The exact compliances of the final designs are 24.110504 for NICE and 24.110806 for the twin, a relative difference of \(1.25\times10^{-5}\). The final compliance is 2.0% below that of the initial design, which holds 25% more material.

- 建议中文：两最终设计的角点参数最大差为0.0051，均方根差为0.0011。NICE与孪生设计的精确柔度分别为24.110504和24.110806，相对差为1.25×10^-5。最终柔度比初始设计低2.0%，而初始设计多含25%的材料。
- 为何更清楚：给出数值层级，读者无需回表才能辨认相对差；保持“两条路径得到相近设计”的作者判断。若作者希望保留方向，可以写“the former is lower by 1.25×10^-5 relative to the latter”，但不应将该微小差异包装成优化优越性。
- 证据：ST22c第1023、1031行；正文第491行。仅做当前源内量纲与算术检查，没有对最终设计做新的精确求解。
- 影响：定位否；“相对”补全不改变结论；删除否；是否避免强调低于方向属于作者选择。
- 确信度与核验范围：高；数值与表内定义对应。

## 全部补充表覆盖

“通读”指读过英文表头、注释、所有行及邻近解释，不代表全部数字已追溯至原始计算。

| 表 | 当前行范围 | 覆盖与结论 |
|---|---:|---|
| ST01 | 44–72 | 完整通读；模型选择名单重叠、未记录的Smoothing-trained分数和主变体后选均明确，保留 |
| ST02 | 74–110 | 完整通读；误差定义、采样坐标与材料体积分数分开，保留 |
| ST03及b–d | 112–167 | 完整通读；L03；每个几何均值与最坏单方向分开，保留 |
| ST04 | 169–179 | 完整通读；L04；八方向对称性抽检范围必须保留 |
| ST05a–b | 181–234 | 完整通读；两类谱份额使用各自分母，保留 |
| ST06a–b | 236–268 | 完整通读；零起点仅清零interior（内部），保留 |
| ST07a–b | 270–326 | 完整通读；生成列数不等于秩的限定，保留 |
| ST08及b | 328–386 | 完整通读；相同校正下不同起场和固定保留位移，保留 |
| ST09及b | 388–469 | 完整通读；L01术语、L05选择时点；缺行未评价的说明，保留 |
| ST10 | 472–516 | 完整通读；L01必须修；不改数值 |
| ST11 | 518–530 | 完整通读；替换范数不可相加、各列最大值可能来自不同载荷，保留 |
| ST12 | 532–555 | 完整通读；同一载荷下误差与energy share（能量份额）对应清楚 |
| ST13 | 557–563 | 完整通读；乘积界与局部灵敏度并列有明确目的，保留 |
| ST14 | 575–584 | 完整通读；不单调背景收敛及H2 n=56缺项，保留 |
| ST15 | 586–599 | 完整通读；L02必须修；实际矩阵符号待数学主审 |
| ST16a–b | 645–694 | 完整通读；载荷集合、目标胞灵敏度和受控自由度定义保留 |
| ST17a–e | 710–765 | 完整通读；与S5共同评价L06；计时误差不等同算子误差的解释保留 |
| ST18 | 767–776 | 完整通读；持有K的内存不计入两种状态内存，范围明确 |
| ST19 | 810–821 | 完整通读；差分商未解析完整导数的判断应保留 |
| ST20 | 829–846 | 完整通读；参数数与固定界值分开，保留；未逐数求和复算 |
| ST21 | 940–962 | 完整通读；与S9.1共同评价L08；优化器、停止规则、回退均明示 |
| ST22a–c | 974–1031 | 完整通读；L09；同迭代号不一定同设计的脚注明确，保留 |
| ST23a–c | 1043–1095 | 完整通读；宏观预测误差与细尺度设计效果分开，保留 |
| ST24a–b | 1103–1130 | 完整通读；按计数发现的切换仅为下界，保留 |
| ST25 | 1136–1148 | 完整通读；比较路线同时改实现与存储，不能当单一精度成本基准，限定保留 |
| ST26 | 1156–1166 | 完整通读；135胞失败在首个求解前、只评估早期2–4次分析，限定保留 |
| ST27 | 1170–1179 | 完整通读；只对两个最终设计核梯度，未查中间/部分横向设计的限定保留 |

## 全部Notes与图注覆盖

| 单元 | 当前行范围 | 覆盖与判断 |
|---|---:|---|
| R1 | 5–42 | 全读；固定变体名称和几何键保持 |
| S1 | 565–567 | 全读；视觉采样与机械离散区分明确 |
| S2 | 569–600 | 全读；L02；参考误差与学习误差不混同 |
| S3 | 602–615 | 全读；未验证PU求解的明示限定保留，不把范围压缩当作已验证投影 |
| S4 | 617–694 | 全读；这是本流程保留表示消融，不是复现其他文献，公平限定保留 |
| S5 | 696–776 | 全读；L06；长实现段可随后拆段，不删计时边界 |
| S6.1–S6.3 | 778–823 | 全读；L07；局部interior粗空间与全局预条件粗空间区分保留 |
| S7 | 825–846 | 全读；构建设置与参数表，保留 |
| S8及S8.1 | 848–922 | 全读六例；数学未全部复算，例子目的与物理假设限定保留 |
| S9.1–S9.6 | 924–1179 | 全读；L08、L09；实际最多110胞短程分析不能写成完整优化验证 |
| Figure S01 | 1183–1185 | 仅图注；H1不单调及n=48不是n=32误差估计，保留 |
| Figure S02 | 1187–1189 | 仅图注；几何均值而非逐方向点、75/80差别，保留 |
| Figure S03 | 1191–1193 | 仅图注；绝对单元贡献先加再归一化的定义，保留 |
| Figure S04 | 1195–1197 | 仅图注；零起点灵敏度只在32步记录，保留 |
| Figure S05 | 1199–1201 | 仅图注；需要继续保持生成列数与粗空间维数的差别，图中较低数字称coarse DOFs可能宜随L04/秩核查统一，但本任务未看图像，不单列定论 |
| Figure S06 | 1203–1205 | 仅图注；计算点与插值线分开，保留 |

## 建议统一原则与顺序

1. 先修当前源可直接证伪的解释：L01、L02、L09。它们不应埋在一般润色清单中。
2. 再对齐评价对象：L05的单胞选择与两胞评价、L03的方向统计与几何统计。精确参考解用reference，3% line（3%参照线）只用于横向比较，不写成接受标准。
3. 接着拆解定义密集段：L04。每个句子只回答一个问题：谱是否在界内、算子实现是否一致、误差何以放大。拆段不删必要限定。
4. 最后讨论解释口径：L06–L08。作者可以明确判断，但让判断紧跟对应数据与条件。不要把“未观察到显著改变”润色成“不会改变”，也不要把“无某项算法检查”解释成“相关不一致无影响”。
5. 保留固定词汇：retained DOFs（保留自由度）、interior（内部）、cut band（切割带）、active element（活跃单元）、weak support（弱支撑）、energy share（能量份额）、two-grid correction（双网格校正）、background element（背景单元）、3% line（3%参照线）；保留NICE、Base network、Uncorrected、Smoothing-trained、'Base network, corrected'。不为求词汇变化引入新名称。
6. 正文口号是作者偏好；补充材料没有需要强行补入口号的位置。这里的作者声音来自“为什么做这个对照、这个数据支持哪一步判断”，而不是在每张表后重新声明贡献。

未编辑论文、代码、图或生成文件；未运行科学计算；只写本辅助报告。上述问题不授权变更论文定位、总体结论或删节。
