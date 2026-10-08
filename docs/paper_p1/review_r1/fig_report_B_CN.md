# 图件重绘报告 B（图 2、图 12 拆分、图 1）— 2026-10-01

范围：`docs/paper_p1`。未改动 `MANUSCRIPT_EN.md`、`APPENDICES_EN.md`、`SUPPLEMENTARY_EN.md`；未 git commit；未删除任何原有文件。所有脚本在 `figures_src/` 下用 `python3` 运行，沿用 `figstyle.py`（rcParams、配色、MM、panel 样式），输出 300 dpi PNG + PDF + SVG。

## 1. 图 2 — `figures/F01_method_overview.{png,pdf,svg}`（重绘；新脚本 `figures_src/fig02_overview.py`）

旧图是两行朴素方框链，未体现"学习给出试探内部延拓、变分形式给出力学结构、固定多层修正给出可改进性"这一原则及算子性质。新图 178 mm × 92 mm，matplotlib 方框/箭头 + mathtext，颜色沿用 figstyle（文字 `#243447`，灰框，学习块蓝边 `#0072B2`，修正块橙边 `#D55E00`，变分块深灰边）。

- **(a) Condensed operator of one cell**：链 `q → [rigid split: C_R q | Π_P q] → [learned extension N_θ(η)Π_P q：trial interior field, geometry-conditioned, linear in q] → [rigid field + R C_R q; restore q on P ⇒ Ê q] → Ê q → [equilibrium correction W：Chebyshev → Q1 coarse → Chebyshev；q held fixed；coefficients set once per geometry] → u = F q → [K] → [Fᵀ: complete chain transposed, reverse order] → Ŝ q`。链上方细线标出刚体系数 `C_R q` 与给定 `q` 绕过网络的旁路。三个块下方的角色标签：`learning: trial field`（蓝）、`correction: improvability`（橙）、`variational form: structure`（深灰，跨 K 与 Fᵀ）。
- 两个性质框：右侧（由 Ŝq 引线连接）`Ŝ = FᵀKF: symmetric, ⪰ 0, rigid kernel`；`Ŝ − S = HᵀAH ⪰ 0 (error quadratic in the field error H)`；`S ⪯ Ŝ_W ⪯ Ŝ_0 (correction cannot increase the error)`。左侧 "by construction (Eq. 5)"：`J_P Ê = I_p`（admissible）、`Ê R_P = R`（rigid motion exact）、`Ê` 在固定几何下对 `q` 线性、权重 θ 全胞共享（正文 3.1 原话）。
- **(b) Assembly and design**：`cells m (Ŝ_m = F_mᵀ K_m F_m) → assembly K̂ = Σ_m B_mᵀ Ŝ_m B_m → global solve K̂ Û = f → field recovery u_m = F_m B_m Û → compliance C = fᵀÛ, thickness sensitivity s̃_c = −u_mᵀ K_{,c} u_m`，下方设计迭代回路（更新 τ_c：新几何 η、同一权重 θ、修正系数按几何重算）。
- 图注建议（作者自定）：(a) 可补一句"K 与 Fᵀ 构成变分读出，框中为 3.2/4.1/5.2 节性质"；(b) 可提及设计回路。

## 2. 图 12 拆分 — `figures_src/fig_opt.py`（就地修改；数据与数值完全不变）

- **`figures/F13_optimisation.{png,pdf,svg}`（图 12，新版）**：仅保留 (a) Case A（含下方相对差条带）与 (b) 平板 B1/B2（含下方放大条带），178 mm × 84 mm。顶部三列大图例取消，改为面板内紧凑图例：(a) 的三项放在 (a) 右上（`NICE` / `exact condensation (twin)` / `exact C, NICE design`），(b) 的六项放在 B1 面板（`NICE, uniform start` / `Hom-y, Hom-z with NICE`（星） / `X-y, X-z from Hom`（六边形线） / `final, uniform start`（点线） / `V>V*, uniform start`（阴影） / `range enlarged below`（括号））。为给图例留空，(a) 纵轴上限 39.5→42.5，B1 纵轴上限 1.38→1.5（仅坐标范围，数据不变）。
- **`figures/F14_designs_scale.{png,pdf,svg}`（新图，建议编号图 13）**：原 (c)(d)(e) → 新 (a)(b)(c)，178 mm × 64 mm；色条保留在 (a)(b) 下方；scale 面板的标签重新放置（`GPU capacity` 在容量线上方靠右、`host`/`GPU` 在末点右侧、`out of GPU memory at 135 cells` 在右下空白区、虚线左侧），互不碰撞。

**面板字母映射（供改图注/正文引用）**

| 旧图 12 | 新位置 |
| --- | --- |
| (a) Case A | 图 12 (a)（不变） |
| (b) Plates B1/B2 | 图 12 (b)（不变） |
| (c) Final design B2 | **图 13 (a)** |
| (d) Final design X-z | **图 13 (b)** |
| (e) Scale demonstration | **图 13 (c)** |

需作者改动的文字：图 12 图注的 (c,d) 与 (e) 两段移入新图 13 的图注（(a,b)、(c)）；正文 "Figure 12b–d"（6.11 节 Hom/X 段）→ "Figure 12b, Figure 13a,b"，"Figure 12e"（规模演示段）→ "Figure 13c"；`FIGURES.md`/LaTeX 中增加图 13 条目（`figures/F14_designs_scale.png`）。目前主文图 12 为最后一张主图，故其他图号无需顺延。

**验证**：数据读取与一致性断言代码段（`import json` 至绘图函数之前）及 `fig_law()`（S06）逐行与原脚本相同（diff 为空）；运行后 `S06_homogenised_law.png` 字节相同，PDF/SVG 仅嵌入日期与 matplotlib 随机 clip-path id 不同。旧的五面板单图代码保留在 `fig_main_single()`，`python3 fig_opt.py --single` 可输出 `figures/F13_optimisation_single.*`（已验证其 PNG 与原 `F13_optimisation.png` md5 相同：`7d02fc49…`）；验证后我已删掉这组由我生成的副产品文件，避免混淆。

## 3. 图 1 — `figures/F08_geometry.png`（**未重绘，原文件原样保留**）

- 四个胞的 id：U1 = `fresh_val_2000_full`、M1 = `fresh_val_2003_d1_v1`、H1 = `fresh_val_2005_d1_v0`、H2 = `fresh_val_2010_d0_v0`（SUPPLEMENTARY R1 表）。
- **可得**：8 个角厚度参数 τ_c（角序 4x+2y+z）四胞齐全——`docs/data/newmachine_20260924/gcell/gval.json` 中 G 胞记录的 `tau_corners_P`（P 孪生胞参数）；U1/M1/H1 与 `review_r1/results/X3/pairs/pair_<case>.json` 的 `taus0` 完全一致。保留宏观体积分数 `evidence/valmeta.json` `vol` = 0.6478 / 0.08604 / 0.02666，与图中 64.8% / 8.6% / 2.67% 一致。
- **不可得**：M1、H1、H2 的切割面法向 `n` 与偏移 `b_cut` 不在本仓库（在生产机 packets 的 `FRESH_CONTEXT.json` `case.normal` / `case.offset`；生成器 `fresh_gp.families` 亦不在库中，无法由种子复现）。已搜遍 `evidence/`、`review_r1/results/`、`docs/data/`（含 ref 日志、gate 文件、X3 对、审计 json）与 git 历史，均无。
- 旁证（仅供核对，不能用于作图）：H2 面 x=0 的材料面积分数由 τ_c 算得 0.15525，与存档 face weight 0.1552463 一致，说明 H2 的切割面几乎平行于 x=0 面（θ ≤ 3.05°，由体积 2.666% 推出）但 θ 不唯一；M1 可由 `evidence/p1_field_M1_small.npz` 的切割带节点拟合 θ≈35.56°、b≈0.8186（保留体积 0.6488 vs 0.6478，近似）。
- 已备好脚本 **`figures_src/fig01_geometry.py`**（已 `pip install scikit-image` 0.26；129³ marching cubes，`g = max(|φ|−τ, n·x−b)`；光照着色；切割截面橙色；统一正交视图；曲面栅格化、文字矢量；约 80 s）。脚本内 τ_c 已填；只需把三组 `(n, b_cut)` 填入 `CUT` 后运行 `python3 fig01_geometry.py`，即输出 `F08_geometry.{png,pdf,svg}`；缺参数时脚本拒绝写 `figures/`。`--test-planes` 用占位平面生成带水印预览 `figures_src/_preview_F08_geometry.png`（已生成，供判断渲染风格；平面为占位，不可用于论文）。脚本还打印各胞保留体积与 H2 面分数核对值。
- 结论：按要求"参数不可得则保留原文件"，`F08_geometry.png` 未改（md5 与 HEAD 一致）。

## 4. 文件清单

- 修改：`figures_src/fig_opt.py`；`figures/F01_method_overview.{png,pdf,svg}`；`figures/F13_optimisation.{png,pdf,svg}`；`figures/S06_homogenised_law.{pdf,svg}`（重新生成，内容相同，仅日期/id）。
- 新增：`figures_src/fig02_overview.py`；`figures_src/fig01_geometry.py`；`figures_src/_preview_F08_geometry.png`；`figures/F14_designs_scale.{png,pdf,svg}`；本报告。
- 环境：新装 scikit-image（及 imageio、lazy-loader、tifffile）。
