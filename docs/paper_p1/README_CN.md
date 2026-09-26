# P1 论文阅读入口

当前稿件围绕一个问题展开：保持胞间耦合坐标不变时，如何学习和修正内部位移场，以及这种近似怎样影响组装柔度与局部厚度灵敏度。

- [英文主稿](MANUSCRIPT_EN.md)：方法构造、力学推导、数值结果与讨论。
- [证明与实现附录](APPENDICES_EN.md)、[完整补充结果](SUPPLEMENTARY_EN.md)。
- [全部图件](FIGURES.md)：11幅主图、5幅附图，提供PDF、SVG和PNG。
- [独立神经网络架构图](../snapshot/workstreams/04_figures/architecture3/new_figures/F11_network_architecture.pdf)、[方法总览](../snapshot/workstreams/04_figures/architecture3/new_figures/F01_method_overview.pdf)。
- [本轮独立复审与重要修正](../snapshot/SELF_REVIEW4_CN.md)、[前轮全稿修订说明](../snapshot/ARCHITECTURE_REVISION_CN.md)、[中文科学主线](PAPER_OVERVIEW_CN.md)、[42项计算力学术语库](TERMINOLOGY_CN_EN.md)。

当前版本为 publication_cleanup5：完成独立整稿复审、18项术语条目同步，以及用户指定的发表内容删除。正文、表3、图8与补充材料已统一为当前比较口径；图8显示B的7个、C和S8各9个构型。方法总览、独立网络架构图、必要推导及有效局部结果保持完整。历史审计另行留档。

## 复现与工作材料

```sh
python3 papers/p1_full_trace_20260926/build_manuscript.py
python3 papers/p1_full_trace_20260926/validate_manuscript.py
```

脚本重建并检查文档。来源及编号对应见[整合清单](../snapshot/INTEGRATION_MANIFEST.json)、[编号映射](../snapshot/PUBLICATION_NUMBERING.json)和[文件检查](../snapshot/MANUSCRIPT_QA_CN.md)。

保留的工作材料：[原始证据](../snapshot/evidence)、[本轮修改前稿件](../snapshot/revisions/self_review4_20260927/baseline)、[前轮科学审计](../snapshot/AUDIT2_REPORT_CN.md)、[待补研究资料](../snapshot/MISSING_ITEMS_CN.md)、[CMAME/IJMS全文写法对照](../snapshot/workstreams/05_references/revision2/JOURNAL_BENCHMARK_CN.md)、[六个独立任务索引](../snapshot/TASKS/INDEX.md)。
