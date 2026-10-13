# astra 第二轮修订稿：与作者逐批讨论的决定（2026-10-13）

总原则（作者重申）：不能为了叙述严谨而把句子写得啰嗦难读。只采纳 astra 指出的真实问题，按我们的写法改；
其余保留原文。所有改动中英文同步，英文摘要不超过 250 词。

## 第一批：摘要

- 摘要-1/2/5/6/10（删背景三句、写全称、改灵敏度句、0.074% 加限定、速度句加硬件限定）：不采纳。
- 摘要-3：采纳。中文"利用以胞元几何为输入的神经网络，由主自由度位移预测内部位移"；英文
  "a network with the cell geometry as input maps retained to interior displacements"。
- 摘要-4：中文"对任意网络参数，所得凝聚刚度矩阵给出的应变能都不低于精确值，超出部分为内部位移误差的
  二阶小量。"英文"For any network parameters, the condensed strain energy is never below the exact value,
  exceeding it to second order in the interior displacement error."（"上界"一词难懂，摘要改用应变能表述）
- 摘要-7：采纳但不写细节。中文"两个八胞元点阵在面载荷下的柔度误差低于 0.015%，点阵梯度误差在 0.09%
  以内"；英文"Two eight-cell lattices have compliance errors below 0.015% and gradient errors within
  0.09% under face loads."
- 摘要-8：保留"复现"，补"各角点参数的差异不超过 0.0051"。
- 摘要-9：保留原文（0.03%/0.3%）。
- 英文摘要压缩（不删论断）：静力凝聚句、"NICE, proposed here,"、网络句、"training and checks cover
  both"、板的句子。定稿约 247 词，草稿见 scratchpad r7/abs_en_v6（再去掉 "of sixteen new cells"，开头恢复
  "in every iteration"）。

## 第二批：引言和第 1.1 节

- 引言 ¶1–¶4、贡献列表、提纲、1.1 首段：不采纳 astra 改写。
- 引言-5（¶5）：补"由主自由度位移"："利用以胞元几何为输入的神经网络，由主自由度位移给出内部位移的初值"。
- 引言-6（¶6）：作者定："由最小势能原理，该矩阵给出的应变能不低于精确值，超出部分恰为内部位移误差的
  应变能，构成精确凝聚刚度矩阵的上界；"（删去"在任一主自由度位移下""该矩阵因而"等冗词）。
- 引言-7（¶7）：(a)"柔度的相对误差不超过各胞元能量误差按其应变能占比的加权和，为内部位移误差的二阶
  小量"；(b) 删去"（命题 3 和命题 4）"。
- 1.1-2（末段）：(a)"以网络预测、再经两重网格循环改进的位移恢复算子 F 近似 E"；(b)"F^TKF 作为凝聚刚度
  矩阵装配到点阵中，其恢复位移场还用于计算厚度灵敏度（第 4.2 节）"。其余保留。
