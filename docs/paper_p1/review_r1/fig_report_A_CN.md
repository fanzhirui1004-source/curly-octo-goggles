# 图件修订报告 A（Figure 7 / Figure 4 / Figure 10）

日期：2026-10-01。范围：仅改表现形式，所有数据、归一化、坐标范围和统计量与修订前完全一致；未修改 MANUSCRIPT_EN.md、APPENDICES_EN.md、SUPPLEMENTARY_EN.md，未提交 git，未删除任何文件。三个脚本均在 `figures_src/` 下用 `python3 <脚本>` 运行，输出沿用原基名（PNG 300 dpi + PDF + SVG），样式全部取自 `figstyle.py`（rcParams、`C`/`MODEL` 颜色、`MM`、`panel()`、`save()`）。

## 1. Figure 7 — `figures/F12_field_error_M1.{png,pdf,svg}`

脚本：`figures_src/fig_field_error.py`（改写）。数据：`evidence/p1_field_M1_small.npz`，仍取第 4 个方向（`k=3`），单元能量除以精确场总能量。

修改内容：

- 面板标题缩短为 "(c) Base network error energy" 和 "(d) NICE error energy"，(a)(b) 标题不变；(c)(d) 标题不再互相碰撞。
- (a) 的图例移出坐标区，放在该面板下方一行（三项横排，自定义等大标记，便于辨认浅灰的 internal 点），不再压在三维散点上。
- (b)(c)(d) 统一使用同一个顺序色图（`magma_r`，去掉最浅的 8%，使最小值在白底上仍可见；浅 = 小，深 = 大）。
- 布局改为 2×3 网格：两列三维面板 + 一列专用色条，四个三维面板尺寸和视角完全相同（`elev=24, azim=-58`）。(c)(d) 共用同一归一化（1e-9 … 1e-3，"element error energy / total"），因此第二行只画一个共享色条；(b) 自己一个色条（1e-7 … 1e-2）。色条不再挤占面板宽度。
- 四个面板的散点全部 `rasterized=True`，并把 `savefig.dpi` 设为 300，使 PDF/SVG 内的点云以 300 ppi 位图嵌入，文字和坐标轴仍为矢量（PDF 内嵌 DejaVu TrueType）。
- 保留 (c)(d) 内的 "total 17.7% / 0.156% of exact energy" 注记，数值不变。

文件大小（修订前 → 修订后）：

| 文件 | 修订前 | 修订后 |
|---|---|---|
| F12_field_error_M1.pdf | 8,471,698 B (8.5 MB) | 1,059,529 B (1.06 MB，< 1.5 MB 要求) |
| F12_field_error_M1.png | 1,177,818 B | 1,580,372 B |
| F12_field_error_M1.svg | 6,952,108 B | 1,712,085 B |

PDF 页面 501 × 443 pt（约 177 × 156 mm），用 pdftoppm 渲染核对，与 PNG 一致。

## 2. Figure 4 — `figures/F09_assembly_loads.{png,pdf,svg}`

脚本：`figures_src/fig04_loads.py`（新建；原图无脚本）。纯示意图，无数据。

修改内容（整幅重画，178 mm × 66 mm，两面板，二维斜投影，深度 +y 朝左上）：

- (a) Configuration x：邻胞 N（exact，灰）在 x∈[-1,0]，目标胞 T（learned，浅蓝）在 x∈[0,1]；夹持面 x = -1 画阴影线；两胞的 y = 0 面（正面）为受载面，浅黄填充，每个面各画一组 x、y、z 三支小箭头（表示每个受载面分别施加的三个牵引方向）；共享盒面 x = 0 用绿色虚线轮廓加淡色填充，标注 "shared box face / coincident box-node coordinates shared"，引线只接到盒子轮廓顶点，不穿过盒体；右下角画整体坐标三元组。
- (b) Configuration y：N 在 y∈[-1,0]（在前），T 在 y∈[0,1]（在后）；夹持面 y = -1（N 的正面）阴影；受载面为两胞的 x = 0 面（左侧斜面），同样浅黄填充加三支箭头；共享面 y = 0 为隐藏面，用虚线矩形按隐藏线惯例画出，标注在盒子右侧。
- 两面板一眼可辨差异（夹持面/受载面/共享面各不相同），比例一致；图下方一行小字说明图例（阴影 = 夹持面，浅填充 = 受载面及三向牵引，虚线 = 共享盒面）。字号按 figstyle（标题 8.5 粗体，标注 7，小字 6–6.5）。

文件大小（修订前 → 修订后）：

| 文件 | 修订前 | 修订后 |
|---|---|---|
| F09_assembly_loads.pdf | 20,812 B | 29,881 B |
| F09_assembly_loads.png | 111,209 B | 172,645 B |
| F09_assembly_loads.svg | （原无） | 21,841 B |

正文引用 `figures/F09_assembly_loads.png` 基名不变。PDF 页面 509 × 186 pt，渲染核对阴影线和虚线均正常。

## 3. Figure 10 — `figures/F10_energy_participation_r1.{png,pdf,svg}`

脚本：`figures_src/fig10_participation.py`（仅改图例部分）。

修改内容：

- 顶部图例只保留五个预测器（Base network、Uncorrected、Smoothing-trained、NICE-post、NICE），一行五列。
- 实心/空心约定改为图下方一行小字（与 Figure 9 的 `fig08_assembly.py` 做法一致）："Filled markers: face loads; open markers: cut-surface loads. Dashed lines: equality."，同时说明了 (a)(b) 中的虚线（与图注 "Dashed lines denote equality" 一致）。
- 其余（四个面板、数据、(b) 面板的注记引线、坐标范围）全部不变；脚本打印的计数仍为 411 observations / 52 combinations，与图注一致。

文件大小（修订前 → 修订后）：

| 文件 | 修订前 | 修订后 |
|---|---|---|
| F10_energy_participation_r1.pdf | 73,619 B | 73,701 B |
| F10_energy_participation_r1.png | 534,239 B | 539,943 B |
| F10_energy_participation_r1.svg | 356,964 B | 355,602 B |

## 4. 未做 / 需要注意的事项

- Figure 4 的可选项"T 内的虚线切割面（cut targets）"没有画：在 (b) 中 T 位于 N 后方，切割面几乎全部被遮挡，只能以隐藏虚线叠在夹持阴影面上，会与共享面虚线混在一起；为保持两面板一致和清爽，两图都省略，宏切割牵引仅由图注说明。如需加入，可在 `config_x`/`config_y` 中按 `shared_face()` 的方式再加一个用 `C['cut']` 颜色的虚线多边形。
- 受载面上的 y（(a)）和 x（(b)）箭头按面的外法线方向画，与整体坐标三元组的正向相反；牵引方向本身不分正负，这样画是为了让箭头留在面内且读作"法向牵引"。
- 三张图都沿用 `figstyle.save()` 的 `bbox_inches='tight'`，最终页面宽度为 177–180 mm（与仓库其他图一致）。
- Figure 7 的 PNG 比修订前略大（1.18 → 1.58 MB），因为面板放大（`zoom=1.12`）、色彩更丰富；PDF 要求（< 1.5 MB）已满足。
- 工作树中另有 `figures/F01_method_overview.*`、`figures/F13_optimisation.{pdf,svg}`、`figures/S06_homogenised_law.{pdf,svg}` 和新文件 `figures_src/fig02_overview.py` 在本次工作期间被其他进程修改/新增（时间戳 09:41–09:55），不属于本报告范围，本次未触碰。
- 修订前的原文件副本保存在会话临时目录（`.../scratchpad/before/`），仓库内未另存副本。
