"""Build the advisor-meeting deck (P1, NICE) on the group's PowerPoint template.

Usage: python3 build_talk.py <template.pptx> <out.pptx>
The template's first slide (title page) is kept and edited; its other slides are removed; every new slide uses the
template's single layout (red title at the top, group footer). Figures come from docs/paper_p1/figures; the cost and
scale charts are native PowerPoint charts. Numbers follow docs/followup/ADVISOR_TALK_OUTLINE_20261003_CN.md, the
manuscript and docs/paper_p1/review_r1/GUO_SERIES_COMPARISON_CN.md. Speaker notes are written on every slide."""
import sys, copy
from pathlib import Path
from PIL import Image
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_LABEL_POSITION
from pptx.oxml.ns import qn
from lxml import etree

FIG = Path(__file__).resolve().parents[2] / 'paper_p1' / 'figures'
RED, DARK, GRAY, BLUE = 'C00000', '262626', '595959', '0070C0'
LIGHT, LRED, LBLUE = 'F2F2F2', 'FBEDED', 'EAF2FB'
LATIN, EA, MATH = 'Arial', 'Microsoft YaHei', 'Cambria Math'
X0, X1, Y0, Y1 = 0.5, 12.83, 1.05, 6.72        # content box (inches)


def rgb(h):
    return RGBColor.from_string(h)


def font(run, size=16, color=DARK, bold=False, italic=False, math=False):
    f = run.font
    f.size = Pt(size); f.bold = bold; f.italic = italic; f.color.rgb = rgb(color)
    f.name = MATH if math else LATIN
    rPr = run._r.get_or_add_rPr()
    for tag in ('a:ea',):
        el = rPr.find(qn(tag))
        if el is None:
            el = etree.SubElement(rPr, qn(tag))
        el.set('typeface', MATH if math else EA)


def para_runs(p, spec, size, color, bold):
    """spec: str, or list of str / (text, dict) runs; dict keys: color, bold, size, math, italic."""
    if isinstance(spec, str):
        spec = [spec]
    for item in spec:
        text, o = (item, {}) if isinstance(item, str) else item
        r = p.add_run(); r.text = text
        font(r, o.get('size', size), o.get('color', color), o.get('bold', bold), o.get('italic', False), o.get('math', False))


def bullet(p, level=0, char='•'):
    pPr = p._p.get_or_add_pPr()
    pPr.set('marL', str(int(Emu(Inches(0.25 + 0.25 * level)))))
    pPr.set('indent', str(int(-Emu(Inches(0.22)))))
    for t in ('a:buNone', 'a:buChar', 'a:buAutoNum'):
        el = pPr.find(qn(t))
        if el is not None:
            pPr.remove(el)
    bf = etree.SubElement(pPr, qn('a:buFont')); bf.set('typeface', 'Arial')
    bc = etree.SubElement(pPr, qn('a:buChar')); bc.set('char', char)


def text(slide, x, y, w, h, paras, size=16, color=DARK, bold=False, align='l', anchor='t', fill=None,
         bullets=False, space=6, line=None, name=None, margin=0.08):
    """paras: list of paragraph specs; a paragraph spec may be ('-', spec) for a bullet, ('--', spec) for level 2."""
    shp = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        shp.name = name
    tf = shp.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(margin); tf.margin_top = tf.margin_bottom = Inches(0.04)
    tf.vertical_anchor = {'t': MSO_ANCHOR.TOP, 'm': MSO_ANCHOR.MIDDLE, 'b': MSO_ANCHOR.BOTTOM}[anchor]
    if fill:
        shp.fill.solid(); shp.fill.fore_color.rgb = rgb(fill)
    for i, spec in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = {'l': PP_ALIGN.LEFT, 'c': PP_ALIGN.CENTER, 'r': PP_ALIGN.RIGHT}[align]
        p.space_after = Pt(space)
        if line:
            p.line_spacing = line
        lvl = None
        if isinstance(spec, tuple) and spec and spec[0] in ('-', '--'):
            lvl = 0 if spec[0] == '-' else 1
            spec = spec[1]
        elif bullets:
            lvl = 0
        if lvl is not None:
            bullet(p, lvl, '•' if lvl == 0 else '–')
        para_runs(p, spec, size, color, bold)
    return shp


def card(slide, x, y, w, h, head, body, fill=LIGHT, head_color=RED, size=15, head_size=17, name=None):
    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    box.adjustments[0] = 0.06
    box.fill.solid(); box.fill.fore_color.rgb = rgb(fill); box.line.fill.background(); box.shadow.inherit = False
    if name:
        box.name = name
    text(slide, x + 0.12, y + 0.08, w - 0.24, 0.5, [head], size=head_size, color=head_color, bold=True)
    text(slide, x + 0.12, y + 0.58, w - 0.24, h - 0.66, body, size=size, color=DARK, space=4)


def stat(slide, x, y, w, number, label, color=RED, nsize=36, lsize=14, fill=None, h=1.45):
    if fill:
        box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
        box.adjustments[0] = 0.08
        box.fill.solid(); box.fill.fore_color.rgb = rgb(fill); box.line.fill.background(); box.shadow.inherit = False
    text(slide, x, y + 0.08, w, 0.75, [number], size=nsize, color=color, bold=True, align='c', anchor='m')
    text(slide, x + 0.1, y + 0.82, w - 0.2, h - 0.85, [label], size=lsize, color=GRAY, align='c')


def image(slide, path, x, y, w, h, crop=None, align='c'):
    """Fit the picture inside (x, y, w, h), keeping its aspect ratio; crop = (l, t, r, b) fractions."""
    im = Image.open(path); iw, ih = im.size
    l, t, r, b = crop or (0, 0, 0, 0)
    ar = (iw * (1 - l - r)) / (ih * (1 - t - b))
    if w / h > ar:
        pw, ph = h * ar, h
    else:
        pw, ph = w, w / ar
    px = x + (w - pw) / 2 if align == 'c' else x
    py = y + (h - ph) / 2
    pic = slide.shapes.add_picture(str(path), Inches(px), Inches(py), Inches(pw), Inches(ph))
    if crop:
        pic.crop_left, pic.crop_top, pic.crop_right, pic.crop_bottom = l, t, r, b
    return pic


def table(slide, x, y, w, rows, widths, size=13, head_fill=RED, row_h=0.36, hl_rows=(), hl_fill=LBLUE, name=None):
    nr, nc = len(rows), len(rows[0])
    gt = slide.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(w), Inches(row_h * nr))
    if name:
        gt.name = name
    tbl = gt.table
    tblPr = tbl._tbl.tblPr
    for k in ('bandRow', 'firstRow'):
        tblPr.set(k, '0')
    style = tblPr.find(qn('a:tableStyleId'))
    if style is not None:
        style.text = '{5940675A-B579-460E-94D1-54222C63F5DA}'     # "No Style, Table Grid"
    for j, fw in enumerate(widths):
        tbl.columns[j].width = Inches(w * fw / sum(widths))
    for i, row in enumerate(rows):
        tbl.rows[i].height = Inches(row_h)
        for j, val in enumerate(row):
            c = tbl.cell(i, j)
            c.margin_left = c.margin_right = Inches(0.06); c.margin_top = c.margin_bottom = Inches(0.03)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = c.text_frame; tf.word_wrap = True
            p = tf.paragraphs[0]
            head = i == 0
            para_runs(p, val, size, 'FFFFFF' if head else DARK, head)
            c.fill.solid()
            c.fill.fore_color.rgb = rgb(head_fill if head else (hl_fill if i in hl_rows else ('FFFFFF' if i % 2 else 'F7F7F7')))
    return gt


def cite(slide, s):
    text(slide, 0.08, 6.80, 12.0, 0.3, [s], size=11, color=GRAY)


def arrow(slide, x, y, w, h, color=GRAY):
    a = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x), Inches(y), Inches(w), Inches(h))
    a.fill.solid(); a.fill.fore_color.rgb = rgb(color); a.line.fill.background(); a.shadow.inherit = False
    return a


class Deck:
    def __init__(self, template):
        self.prs = Presentation(template)
        sld = self.prs.slides._sldIdLst
        for sid in list(sld)[1:]:                                   # keep only the title page
            self.prs.part.drop_rel(sid.rId); sld.remove(sid)
        self.layout = self.prs.slide_layouts[0]

    def slide(self, title, notes):
        s = self.prs.slides.add_slide(self.layout)
        tp = s.shapes.title
        tp.text_frame.text = title
        for r in tp.text_frame.paragraphs[0].runs:
            r.font.color.rgb = rgb(RED)
        s.notes_slide.notes_text_frame.text = notes
        return s

    def total(self, n):
        for sh in self.layout.shapes:                               # '/61' next to the slide number
            if sh.has_text_frame and sh.text_frame.text.strip().startswith('/'):
                r = sh.text_frame.paragraphs[0].runs
                r[0].text = '/%d' % n
                for extra in r[1:]:
                    extra.text = ''


def build(template, out):
    D = Deck(template)

    # ---------------------------------------------------------------- 1 title page (template slide, edited)
    s1 = D.prs.slides[0]
    for sh in s1.shapes:
        if not sh.has_text_frame:
            continue
        t = sh.text_frame.text
        if t.startswith('组会汇报'):
            r = sh.text_frame.paragraphs[0].runs
            r[0].text = 'P1 进展汇报：带平衡校正的神经初始化静力凝聚（NICE）'
            for extra in r[1:]:
                extra.text = ''
            for rr in r:
                rr.font.size = Pt(34)
    text(s1, 0.5, 2.45, 12.33, 0.6, ['切割薄壁 TPMS 格栅的分析与厚度设计 · 投稿 CMAME 前与导师单独汇报 · 2026 年 10 月'],
         size=18, color=GRAY, align='c')
    s1.notes_slide.notes_text_frame.text = (
        '老师好。今天汇报 P1 的定稿情况：方法、结果、和 PIML 系列的对比，以及投稿计划和后续方向。'
        '最后有几件事需要您拍板。')

    # ---------------------------------------------------------------- 2 one-page conclusion
    s = D.slide('一页结论：保留完整边界，只近似内部，误差能追到灵敏度',
                '先说结论。我们的学习模型保留每个胞全部盒面和切割带自由度，约两万四千个，但保住了静力凝聚的结构：'
                '对称半正定、刚体核、被精确 Schur 补从下方界住。四个数字：单胞平均能量误差 0.074%；两胞装配柔度 0.28%、'
                '灵敏度 1.5% 以内；八胞分析约为 16 线程直接解的十分之一时间；切割固支板上均匀化低估柔度 27%。'
                '和 PIML 最根本的区别：他们删边界，我们保留边界、只近似内部。目标期刊 CMAME。')
    text(s, X0, 1.15, 5.7, 3.3, [
        [('学习模型作用在每个胞的', {}), ('全部盒面与切割带自由度', {'bold': True, 'color': RED}), ('上（中位数 2.4×10⁴ 个），同时保住静力凝聚的结构：', {})],
        ('-', '对称半正定，含刚体核'),
        ('-', [('被精确 Schur 补从下方界住：', {}), ('S ⪯ Ŝ', {'math': True})]),
        ('-', '误差可以一路追到柔度和厚度灵敏度'),
        ('-', '部署后加大校正预算还能继续改进'),
    ], size=17, space=8)
    stat(s, 6.55, 1.15, 3.0, '0.074%', '单胞平均能量误差（80 个验证几何）', fill=LIGHT)
    stat(s, 9.75, 1.15, 3.0, '≤0.28% / 1.5%', '两胞装配：柔度 / 灵敏度误差', fill=LIGHT, nsize=30)
    stat(s, 6.55, 2.85, 3.0, '≈10×', '八胞分析（含灵敏度）快于 16 线程直接解', fill=LIGHT)
    stat(s, 9.75, 2.85, 3.0, '27%', '切割固支板上均匀化低估柔度', fill=LIGHT)
    text(s, X0, 4.75, 12.33, 1.15, [
        [('和 PIML 的根本区别：', {'bold': True, 'color': RED}),
         ('他们删边界（每个子结构 24 个角点自由度），误差 2–20%；我们保留边界、只近似内部，误差 0.01% 量级，并且说清误差怎么进入灵敏度。', {})]],
        size=17, fill=LRED, anchor='m', margin=0.2)
    text(s, X0, 6.05, 12.33, 0.5, [[('状态：', {'bold': True}), ('正文 63 页 + 补充 37 页，可投；目标 ', {}), ('CMAME', {'bold': True, 'color': BLUE}),
                                    ('，同时挂 arXiv', {})]], size=16, color=GRAY)

    # ---------------------------------------------------------------- 3 problem
    s = D.slide('问题：组件同类但几何各异，整体太大、驱动设计的量却是局部的',
                '我们关心的结构：梯度格栅、按零件外形切割的格栅、在切割面上支撑或加载的芯材。矛盾在于：整体太大，'
                '每次设计迭代都重算一遍不现实；但驱动设计的量是局部的，比如支撑旁切割胞的刚度、某处壁厚的灵敏度。'
                '左图是四个代表性胞：未切割、中度切割、两个重度切割，保留体积从 100% 到 2.6%。')
    image(s, FIG / 'F08_geometry.png', X0, 1.05, 5.6, 5.6)
    text(s, 6.4, 1.2, 6.4, 5.3, [
        [('对象', {'bold': True, 'color': RED})],
        ('-', '梯度格栅：胞的壁厚逐个变化'),
        ('-', '按零件外形切割的格栅：边界胞被平面切开'),
        ('-', '在切割面上支撑、加载的芯材'),
        [('矛盾', {'bold': True, 'color': RED})],
        ('-', '整体太大：每次设计迭代都改变每个胞，全分辨率代价每轮全额重付'),
        ('-', '驱动设计的量是局部的：支撑旁切割胞的刚度、对某处壁厚的灵敏度'),
        [('需要', {'bold': True, 'color': RED})],
        ('-', '一个可组装、能用于设计、误差可追溯的组件模型，且对新几何不需要内部分解'),
    ], size=17, space=7)
    cite(s, '图：论文 Fig. 1（代表性验证几何 U1、M1、H1、H2）')

    # ---------------------------------------------------------------- 4 routes
    s = D.slide('现有三条路线各卡在一处，学习子结构是第四条',
                '均匀化依赖尺度分离，在支撑、载荷和切割胞处失效，我们切割板上低估 27%。全分辨率每次迭代都全额重付，'
                '八胞已经两百万自由度、Cholesky 因子 66 到 81 GiB。精确凝聚每个几何要一次内部分解。'
                '学习子结构，也就是 PIML 系列，是第四条路，下面三页专门讲它的问题。')
    W4 = (12.33 - 3 * 0.3) / 4
    cards = [
        ('均匀化', ['依赖尺度分离', '在支撑、载荷、切割胞处失效', [('切割固支板：柔度低估 ', {}), ('27%', {'bold': True, 'color': RED})]]),
        ('全分辨率', ['每次设计迭代改变每个胞，代价每轮全额重付', [('八胞 183–211 万自由度，Cholesky 因子 ', {}), ('66–81 GiB', {'bold': True, 'color': RED})]]),
        ('精确凝聚', ['每个几何一次内部分解', '显式 Schur 补：每个保留自由度一次内部求解', [('一胞中位 2.4×10⁴ 个保留自由度', {'bold': True, 'color': RED})]]),
        ('学习子结构（PIML）', ['每个子结构只保留少量边界自由度', '换来规模，但带来与网络无关的边界误差', [('→ 下面三页专门体检', {'bold': True, 'color': RED})]]),
    ]
    for k, (h, body) in enumerate(cards):
        card(s, X0 + k * (W4 + 0.3), 1.3, W4, 3.4, h, [('-', b) for b in body], fill=LRED if k == 3 else LIGHT, size=15)
    text(s, X0, 5.0, 12.33, 0.9, [[('我们的位置：', {'bold': True, 'color': BLUE}),
                                    ('走子结构这条中间路线，但把近似放在内部延拓上，保留完整的保留空间；结构由构造保证，不靠训练。', {})]],
         size=17, fill=LBLUE, anchor='m', margin=0.2)

    # ---------------------------------------------------------------- 5 PIML overview
    s = D.slide('PIML 系列速览：先说清他们做了什么、强在哪',
                '先公平地讲清楚他们做了什么，表明我们读透了。六篇，共同框架是每个子结构保留八个角点、24 个自由度，'
                '网络预测形函数 N，刚度取 N 转置 K N。他们的强项是应用广、规模大：串行做到十亿单元，并行版本一百亿自由度；'
                '还有无数据训练、等变网络。这些导师可能会问，主动说。')
    table(s, X0, 1.15, 7.4, [
        ['论文', '期刊', '主要做法'],
        ['Huang et al. 2023', 'EML', '3D PIML；24 个角点自由度；监督训练'],
        ['Huang et al. 2024', 'JMPS', '无数据训练（最小势能）；DeepONet'],
        ['Guo et al. 2026a', 'CMAME', 'Bézier 边界（168 自由度）；含"全节点"对照'],
        ['Guo et al. 2026b', 'arXiv', 'OFEM：过采样基 + 重叠单位分解'],
        ['Jiang C. et al. 2026', 'Compos. Struct.', '立方对称等变；投影校正'],
        ['Zhang et al. 2026', 'IJMS', '等参子结构迁移学习；应力问题'],
    ], (2.3, 1.4, 4.0), size=14, row_h=0.5)
    card(s, 8.2, 1.15, 4.63, 2.35, '共同框架', [
        ('-', '每个子结构保留少量边界自由度（多为 8 个角点、24 个）'),
        ('-', [('网络预测形函数 N，刚度取 ', {}), ('NᵀKN', {'math': True})]),
    ], size=15)
    card(s, 8.2, 3.7, 4.63, 2.75, '他们的强项（导师可能问）', [
        ('-', '应用广：柔顺机构、应力、扭转盒、复杂域'),
        ('-', '规模大：串行 10⁹ 单元；并行版本 10¹⁰ 自由度'),
        ('-', '无数据训练、等变网络、Bézier 边界'),
    ], fill=LBLUE, head_color=BLUE, size=15)
    cite(s, '来源：review_r1/GUO_SERIES_COMPARISON_CN.md 第 1 节（六篇全文逐页核对）')

    # ---------------------------------------------------------------- 6 audit 1: accuracy
    s = D.slide('PIML 体检一：边界一删，误差就和网络无关了',
                '第一，精度。只保留 24 个角点，就带来与网络无关的边界模型误差：JMPS 2024 表 2 用精确内部，误差仍有 1% 到 12%。'
                '集中载荷下加载点位移，也就是柔度，误差线性边界 63%、Bézier 15%，作者自己写了"必然过刚"。'
                '局部量更糟：IJMS 2026 的应力 p 范数低估约 5.3 倍，只好降格成 warm start 再在细网格上优化 31 步。'
                '经细网格复核的柔度全部偏刚，他们只是观察到，没有解释也没有界；我们的 Ritz 恒等式正好解释这个现象。')
    W3 = (12.33 - 3 * 0.3) / 4
    items = [('0.98–11.9%', '精确内部、只限制边界时的位移误差', 'JMPS 2024, Table 2, p.5'),
             ('63% / 15%', '集中载荷下的柔度误差（线性 / Bézier 边界）', 'CMAME 2026, p.15–17'),
             ('≈5.3×', '应力 p 范数被低估（0.00913 对细网格 0.04820）', 'IJMS 2026, p.13'),
             ('−20.7%', '中等规模算例柔度相对全尺度 FEA', 'IJMS 2026, App. C, p.20')]
    for k, (n, lab, src) in enumerate(items):
        x = X0 + k * (W3 + 0.3)
        stat(s, x, 1.2, W3, n, lab, fill=LRED, nsize=32, lsize=14, h=1.9)
        text(s, x, 3.12, W3, 0.35, [src], size=11, color=GRAY, align='c')
    text(s, X0, 3.65, 12.33, 1.05, [[('作者原话：', {'bold': True, 'color': RED}),
                                     ('"any algorithm that assumes the substructure boundary displacements will inevitably lead to an overly stiff numerical response"', {'italic': True}),
                                     ('（CMAME 2026, p.17）', {'color': GRAY, 'size': 13})]], size=15, fill=LIGHT, anchor='m', margin=0.2)
    text(s, X0, 4.9, 12.33, 1.6, [
        [('经细网格复核的柔度全部偏刚：', {'bold': True}), ('−4.7%（JMPS）；−9.17%、−2.79%、−3.14%（CS 2026）；−20.7%（IJMS）', {})],
        [('他们只"观察到"，没有解释、没有界。', {}), ('我们的 Ritz 恒等式 ', {'bold': True, 'color': BLUE}),
         ('Ŝ − S = HᵀAH ⪰ 0', {'math': True, 'bold': True, 'color': BLUE}), (' 正好解释这一点。', {'bold': True, 'color': BLUE})],
    ], size=16, space=8)
    cite(s, '来源：GUO_SERIES_COMPARISON_CN.md 第 1、3(b) 节；页码为期刊版面页')

    # ---------------------------------------------------------------- 7 audit 2: baselines
    s = D.slide('PIML 体检二：参考解和加速比的口径',
                '第二，口径。参考解常常不是细网格：CS 2026 表 3、IJMS 主体都以同一线性边界下的精确子结构法为参考，'
                '只量了网络误差，边界误差没算进去；CMAME 2026 的优化设计全部没有细网格复核。基准偏弱或说不清：'
                '472 倍只比经典 SIMP 前十步平均；CMAME 2026 的直接 FEM 没写求解器，串行单 CPU，时间与规模还不单调。'
                '数字前后不一：摘要一万到十万倍、正文 328 到 472 倍，等等。')
    card(s, X0, 1.2, 3.95, 5.25, '参考解经常不是细网格', [
        ('-', 'CS 2026 Table 3、IJMS 2026 主体：以"同一线性边界下的精确子结构法"为参考 → 只量网络误差，边界误差不计'),
        ('-', 'CMAME 2026 的优化设计全部没有细网格复核'),
    ], fill=LIGHT, size=15)
    card(s, X0 + 4.19, 1.2, 3.95, 5.25, '基准偏弱或说不清', [
        ('-', 'EML 2023 的 472×：只比经典 SIMP 前 10 步的平均时间'),
        ('-', 'CMAME 2026：直接 FEM 未写求解器，串行单 CPU；2.69 M 自由度 1473 s，6.15 M 自由度只要 1262 s'),
        ('-', 'JMPS 2024：Matlab 基准，0.8 M 自由度 1320 s'),
    ], fill=LIGHT, size=15)
    card(s, X0 + 8.38, 1.2, 3.95, 5.25, '数字前后不一', [
        ('-', 'EML 2023：摘要"10⁴–10⁵ times"，正文 328–472×'),
        ('-', 'JMPS 2024：Table 3 与正文两个时间写反'),
        ('-', 'CMAME 2026：引言"2–3 个数量级"，表中设计迭代 7.4–63×，结论改称"one to two orders"；最大 722× 对应 13.53% 误差'),
        ('-', '10⁹ 单元算例串行 9–10 天，"laptop"实为 Xeon Gold 6256 + 512 GB'),
    ], fill=LRED, size=14)
    cite(s, '来源：GUO_SERIES_COMPARISON_CN.md 第 3(c) 节与"其他可引用事实"')

    # ---------------------------------------------------------------- 8 audit 3: full-node
    s = D.slide('PIML 体检三：他们撞墙的地方，正是我们出发的地方',
                '第三，最关键的一页。CMAME 2026 表 1 自己试过保留全部边界节点：加速只有 7.3 到 7.5 倍，位移误差 0.84% 到 3.46%，'
                '子结构越大反而越差，原因是网络输出变多、预测能力下降；桥形算例超过一千秒，作者说"不再算有效的加速方法"。'
                '我们每胞保留的自由度多一到两个数量级，网络对保留位移精确线性、参数量不随保留自由度增长，'
                '他们解决不了的"输出变多网络变差"由固定校正收掉。另外整个系列没有一篇报告灵敏度误差。'
                '最后口径提醒：他们的输入是随机模量体素场，比我们难，这一点要主动承认。')
    table(s, X0, 1.15, 12.33, [
        ['', 'CMAME 2026"保留全部边界节点"方案', 'NICE（本文）'],
        ['每个子结构 / 胞的保留自由度', '456–1,806', [('2,679–45,900（中位 23,604）', {'bold': True, 'color': BLUE})]],
        ['加速', '7.3–7.5×', [('≈10×（16 线程）；57–77×（单线程）', {'bold': True, 'color': BLUE})]],
        ['精度', '位移误差 0.84%（5³）/ 3.46%（10³），子结构越大越差', [('八胞柔度 ≤0.015%，灵敏度 ≤0.14%', {'bold': True, 'color': BLUE})]],
        ['"输出变多，网络变差"', '无对策', '网络对保留位移精确线性；固定校正 𝒲 收掉内部误差'],
        ['灵敏度误差', '整个系列未报告；JMPS"for ease of implementation"忽略 ∂N/∂ρ', '逐胞报告；推导完整导数 Eq. (10)'],
        ['理论', '刚体不变性；收敛分析"forthcoming"', 'Ritz 下界 + 份额加权界 + 灵敏度一次项'],
        ['作者结论', '"no longer counted as an effective acceleration method"', '—'],
    ], (2.6, 4.9, 4.8), size=14, row_h=0.56)
    text(s, X0, 5.75, 12.33, 0.85, [[('口径提醒（导师问到要主动承认）：', {'bold': True, 'color': RED}),
                                     ('他们的输入是随机模量体素场，比我们的低维参数族难；但体检一、二是方法和口径问题，不是任务难度问题。', {})]],
         size=15, fill=LIGHT, anchor='m', margin=0.2)
    cite(s, '来源：CMAME 2026 Table 1–2（p.14–15）；GUO_SERIES_COMPARISON_CN.md 第 3(a)、3(d) 节')

    # ---------------------------------------------------------------- 9 core idea
    s = D.slide('核心想法：近似放在哪、刚度怎么形成，决定了结构',
                '我们的核心想法就三处放置：近似只放在内部延拓上；保留完整的迹；刚度用能量形式 F 转置 K F。'
                '由构造就得到对称半正定、刚体核、被精确 Schur 补从下方界住，误差是内部误差的二次型，这些对任何输入都成立，'
                '不依赖网络训练得好不好。再加一个固定的两层网格校正，在给定条件下不会增大误差。')
    image(s, FIG / 'F01_method_overview.png', X0, 1.05, 8.1, 4.3)
    text(s, 8.85, 1.15, 4.0, 4.4, [
        [('三处放置', {'bold': True, 'color': RED})],
        ('-', '近似只放在内部延拓'),
        ('-', '保留完整的迹（盒面 + 切割带）'),
        ('-', [('刚度用能量形式 ', {}), ('Ŝ = FᵀKF', {'math': True})]),
        [('由构造得到（对任何输入成立）', {'bold': True, 'color': RED})],
        ('-', '对称半正定、刚体核'),
        ('-', [('S ⪯ Ŝ', {'math': True}), ('，误差为内部误差的二次型', {})]),
    ], size=16, space=6)
    text(s, X0, 5.6, 12.33, 0.95, [[('固定的两层网格校正 𝒲', {'bold': True, 'color': BLUE}),
                                    ('（Chebyshev 8 / 粗网格 Q₁(17) / Chebyshev 8）在能量内修正内部场；精确或非负平移的粗求解下，', {}),
                                    ('Ŝ ⪯ Ŝ_net', {'math': True}), ('，校正不会增大误差。', {})]],
         size=15, fill=LBLUE, anchor='m', margin=0.2)
    cite(s, '图：论文 Fig. 2；理论：第 3、4 节，附录 B–D')

    # ---------------------------------------------------------------- 10 error chain
    s = D.slide('误差链：柔度准不等于灵敏度准',
                '误差链有四环。第一，Ritz 恒等式：近似刚度减精确刚度等于 H 转置 A H，半正定，误差二次。第二，装配柔度误差按'
                '各胞能量份额加权，所以份额小的胞可以让柔度看起来很准。第三，灵敏度里有一个一次项，能量误差管不住它。'
                '第四，能量误差分解成 δ 平方乘 κ：约 1% 的位移误差落在刚性方向上，放大成 1.8% 到 35% 的能量误差。'
                '所以判据必须是柔度和灵敏度同时准。第六页那串"全部偏刚"，就是这条链的第一环。')
    steps = [('① Ritz 恒等式', [('Ŝ − S = HᵀAH ⪰ 0', {'math': True})], '单侧、二次'),
             ('② 份额加权', ['柔度误差按各胞能量份额加权，Eq. (7)'], '份额小 → 柔度"看起来"准'),
             ('③ 灵敏度一次项', ['场基灵敏度含线性交叉项，Eq. (9)'], '能量误差管不住它'),
             ('④ ε = δ²κ', ['1% 位移误差落在刚性方向'], '→ 1.8–35% 能量误差')]
    bw = (12.33 - 3 * 0.45) / 4
    for k, (h, body, foot) in enumerate(steps):
        x = X0 + k * (bw + 0.45)
        card(s, x, 1.3, bw, 2.6, h, [body, [(foot, {'color': GRAY})]], fill=LIGHT if k != 2 else LRED, size=15)
        if k < 3:
            arrow(s, x + bw + 0.07, 2.4, 0.31, 0.38)
    text(s, X0, 4.25, 12.33, 1.0, [[('结论：', {'bold': True, 'color': RED}),
                                    ('柔度准不保证灵敏度准；设计用途的判据是两者同时准。', {'bold': True})]],
         size=19, fill=LRED, anchor='m', margin=0.2)
    text(s, X0, 5.45, 12.33, 1.1, [
        ('-', '体检一里 PIML 经细网格复核的柔度全部偏刚，正是第①环的直接后果（边界受限时下界为 LᵀSL）'),
        ('-', '第③环在 PIML 系列里从未量化：没有一篇报告灵敏度误差'),
    ], size=15, space=4)
    cite(s, '论文第 3 节；附录 B、C、H')

    # ---------------------------------------------------------------- 11 network
    s = D.slide('网络：对保留位移精确线性，一个网络服务所有保留集',
                '网络对几何是非线性的，对保留位移是精确线性的。权重作用在模板、通道和网格层级上，不对应单个自由度，'
                '所以一个约六十万参数的网络服务保留自由度从两千多到四万多的所有胞。我们不形成显式矩阵：一个中位尺寸的胞，'
                '显式延拓双精度就要 4.3 GiB。')
    image(s, FIG / 'F11_network_architecture.png', X0, 1.05, 4.6, 5.65)
    stat(s, 5.6, 1.2, 3.45, '6×10⁵', '网络参数（不随保留自由度增长）', fill=LIGHT, h=1.5)
    stat(s, 9.35, 1.2, 3.45, '2,679–45,900', '同一网络服务的保留自由度范围', fill=LIGHT, nsize=28, h=1.5)
    text(s, 5.6, 2.95, 7.2, 3.6, [
        ('-', '几何条件化：输入是背景单元上的材料矩，不是设计参数'),
        ('-', '对保留位移精确线性 → 凝聚算子线性、可组装'),
        ('-', '权重作用在模板、通道和网格层级（65/33/17/9）上，不对应单个自由度'),
        ('-', '不形成显式矩阵：中位尺寸胞的显式延拓需 4.3 GiB（双精度）'),
        ('-', '训练：691 个几何（591 训练 + 100 验证），一次性约 60 GPU-h'),
    ], size=16, space=8)
    cite(s, '图：论文 Fig. 3；第 4.4–4.6 节')

    # ---------------------------------------------------------------- 12 correction
    s = D.slide('校正：网络固定不变，部署后仍可继续改进',
                '固定网络、只加校正。H2 从 35% 降到 0.205%。M1 只做 Chebyshev 光滑降到 4.69%，因为它的误差有一大块落在'
                '光滑区间以下的低频模态，要靠粗网格校正，加上后到 0.186%。同一校正作用在零初值或调和延拓上远远不够，'
                '网络仍承担绝大部分精度。')
    image(s, FIG / 'F04_correction.png', X0, 1.05, 7.0, 5.5)
    text(s, 7.8, 1.15, 5.0, 0.45, [[('同一网络、只加校正（能量误差）', {'bold': True, 'color': RED})]], size=17)
    table(s, 7.8, 1.7, 5.03, [
        ['几何', '基础网络', '仅光滑 8 步', '完整校正'],
        ['H2', '35.0%', '0.205%', '—'],
        ['M1', '13.5%', '4.69%', [('0.186%', {'bold': True, 'color': BLUE})]],
    ], (1, 1.3, 1.4, 1.3), size=15, row_h=0.5)
    text(s, 7.8, 3.45, 5.0, 3.1, [
        ('-', 'M1：最低 200 个模态（全部低于光滑区间下端）含 24.5% 的误差能量 → 需要粗网格校正'),
        ('-', '同一校正作用在零初值或调和延拓上远不够（Table 3）：网络承担了大部分精度'),
        ('-', '校正预算可以在部署时加大，不用重训'),
    ], size=15, space=8)
    cite(s, '图：论文 Fig. 7；第 5.4–5.5 节，Table 3')

    # ---------------------------------------------------------------- 13 single cell
    s = D.slide('单胞精度：80 个验证几何平均 0.074%',
                '80 个验证几何上，NICE 的平均能量误差 0.074%，未校正延续是 6.33%，各分层平均低 64 到 102 倍。'
                '逐方向看，5120 个采样方向的 95 分位 0.33%，最大 1.24%。')
    image(s, FIG / 'F02_validation.png', X0, 1.05, 8.2, 5.6)
    stat(s, 9.0, 1.15, 3.83, '0.074%', 'NICE 平均能量误差（一致面力）', fill=LIGHT, h=1.4)
    stat(s, 9.0, 2.75, 3.83, '64–102×', '各分层平均低于未校正延续', fill=LIGHT, h=1.4)
    text(s, 9.0, 4.35, 3.83, 2.2, [
        ('-', '5,120 个方向：95 分位 0.33%，最大 1.24%'),
        ('-', '60 个非选择几何上基本不变'),
        ('-', '未切割 → 重度切割仍增约 7 倍'),
    ], size=15, space=6)
    cite(s, '图：论文 Fig. 4；第 5.3 节，Table ST03')

    # ---------------------------------------------------------------- 14 assembly
    s = D.slide('装配后：柔度准，灵敏度不一定准',
                '装配之后，不完整的校正柔度可以在 3% 线以内，灵敏度却可以到 11.4%。NICE 在 14 个选择配置加 18 个留出配置'
                '全部在两条 3% 线以内，最大 0.28% 和 1.49%。这就是误差链第三环的实证。')
    image(s, FIG / 'F05_assembly.png', X0, 1.05, 5.9, 5.6)
    card(s, 6.75, 1.2, 6.08, 2.25, '不完整的校正', [
        ('-', '未校正延续：灵敏度最高 11.4%，4 个配置越过 3% 线'),
        ('-', '仅光滑训练：同样 4 个配置 3.15–4.43%'),
    ], fill=LRED, size=15)
    card(s, 6.75, 3.65, 6.08, 2.85, 'NICE', [
        ('-', '14 个选择配置 + 18 个留出配置全部低于两条 3% 线'),
        ('-', [('最大：柔度 ', {}), ('0.28%', {'bold': True, 'color': BLUE}), ('，灵敏度 ', {}), ('1.49%', {'bold': True, 'color': BLUE})]),
        ('-', '基础网络部署时加校正（不重训）也全部低于 3% 线'),
    ], fill=LBLUE, head_color=BLUE, size=15)
    cite(s, '图：论文 Fig. 9；第 5.6 节，Table 4，Table ST08')

    # ---------------------------------------------------------------- 15 ablation
    s = D.slide('消融：压缩边界对柔度尚可，对局部灵敏度不行',
                '这一页最直接打 PIML 的路线。用精确胞算子，只把盒面位移限制成 r 次 Bernstein 多项式：r 等于 1 时柔度误差'
                '78% 到 85%；r 到 8 时柔度 0.5% 到 0.7%，但邻载下灵敏度仍有 24% 到 64%。只限制共享界面、r 等于 3 时'
                '柔度只有零点几，灵敏度仍有 8% 到 21%。压缩边界对柔度尚可，对局部灵敏度不行。')
    image(s, FIG / 'F06_bernstein.png', X0, 1.05, 12.33, 3.3)
    W3 = (12.33 - 2 * 0.3) / 3
    stat(s, X0, 4.5, W3, '78–85%', '全部盒面 r = 1：柔度误差', fill=LRED, h=1.6)
    stat(s, X0 + W3 + 0.3, 4.5, W3, '24–64%', '全部盒面 r = 8：柔度已 0.49–0.74%，邻载灵敏度误差仍', fill=LRED, h=1.6)
    stat(s, X0 + 2 * (W3 + 0.3), 4.5, W3, '8–21%', '只限制界面 r = 3：柔度 0.17–0.44%，邻载灵敏度误差', fill=LRED, h=1.6)
    cite(s, '图：论文 Fig. 11；第 5.7 节，Table ST11（精确胞算子，只改保留表示）')

    # ---------------------------------------------------------------- 16 lattices
    s = D.slide('八胞格栅：全部是学习胞，而且都是训练中没见过的胞',
                '真实设计里每个胞都是学习胞，邻胞误差进入同一个装配解。两个八胞格栅，十六个胞都没进过训练和检查点选择。'
                '面载下柔度误差不超过 0.015%，灵敏度不超过 0.14%；随机载荷下 0.069% 和 0.45%。份额加权的界 β 和实际误差'
                '在面载下只差不到 β 的 0.4%。')
    table(s, X0, 1.2, 7.6, [
        ['', '2×2×2 块体（4 胞切割）', '3×3×1 层（3 胞切割）'],
        ['柔度误差（三个面载）', '0.014 / 0.011 / 0.0094%', '0.015 / 0.013 / 0.010%'],
        ['柔度误差（随机载荷，最大）', '0.069%', '0.056%'],
        ['灵敏度最大误差（面载 / 随机）', '0.14% / 0.45%', '0.14% / 0.43%'],
        ['装配保留解差异', '0.13%', '0.12%'],
        ['各胞能量误差（面载）', '0.006–0.026%', '0.005–0.052%'],
    ], (2.5, 2.55, 2.55), size=13, row_h=0.6)
    stat(s, 8.5, 1.2, 4.33, '≤0.015%', '面载下格栅柔度误差', fill=LBLUE, color=BLUE, h=1.5)
    stat(s, 8.5, 2.9, 4.33, '≤0.14%', '面载下灵敏度最大误差', fill=LBLUE, color=BLUE, h=1.5)
    text(s, 8.5, 4.6, 4.33, 1.9, [
        ('-', '份额加权界 β 与实际柔度误差之差 < β 的 0.4%（面载）'),
        ('-', '格栅误差低于两胞最大误差：Eq. (7) 按能量份额加权'),
    ], size=14, space=6)
    cite(s, '论文第 5.8 节；参考解：装配精确凝聚矩阵，相对残差 ≤1.3×10⁻¹⁰')

    # ---------------------------------------------------------------- 17 cost
    s = D.slide('成本：约为 16 线程直接解的 1/10，单线程口径 57–77×',
                '成本。八胞两个格栅，NICE 79 秒和 105 秒；16 线程 PARDISO 直接解 868 和 1042 秒，常规精确凝聚 950 和 1218 秒；'
                '单线程直接解 5864 和 8131 秒，这个口径和 PIML 的串行基准一致，是 57 到 77 倍。内存上，直接法的 Cholesky 因子'
                '从四胞的 29 到 32 GiB 涨到八胞的 66 到 81 GiB，NICE 显存峰值 9.2 GiB。')
    cd = CategoryChartData()
    cd.categories = ['2×2×2（4 胞切割）', '3×3×1（3 胞切割）']
    cd.add_series('NICE（一块 GPU）', (79.2, 105.5))
    cd.add_series('直接解，16 线程', (867.8, 1041.6))
    cd.add_series('常规精确凝聚，16 线程', (950, 1218))
    cd.add_series('直接解，单线程', (5863.9, 8131.0))
    gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(X0), Inches(1.1), Inches(7.9), Inches(5.5), cd)
    ch = gf.chart
    ch.has_title = True; ch.chart_title.text_frame.text = '一次含灵敏度的八胞格栅分析耗时（s）'
    for r in ch.chart_title.text_frame.paragraphs[0].runs:
        font(r, 15, DARK, True)
    ch.has_legend = True; ch.legend.position = XL_LEGEND_POSITION.BOTTOM; ch.legend.include_in_layout = False
    ch.legend.font.size = Pt(12)
    for ser, col in zip(ch.plots[0].series, (BLUE, '7F7F7F', 'BFBFBF', RED)):
        ser.format.fill.solid(); ser.format.fill.fore_color.rgb = rgb(col)
    pl = ch.plots[0]; pl.has_data_labels = True; pl.data_labels.font.size = Pt(11)
    pl.data_labels.number_format = '#,##0'; pl.data_labels.number_format_is_linked = False
    pl.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END; pl.gap_width = 60
    va = ch.value_axis; va.has_major_gridlines = True; va.major_gridlines.format.line.color.rgb = rgb('E0E0E0')
    va.tick_labels.font.size = Pt(11); va.format.line.fill.background()
    ch.category_axis.tick_labels.font.size = Pt(12)
    stat(s, 8.75, 1.2, 4.08, '≈10×', '相对 16 线程直接解（868 / 1,042 s 对 79 / 105 s）', fill=LIGHT, h=1.5)
    stat(s, 8.75, 2.9, 4.08, '57–77×', '单线程口径（与 PIML 的串行基准一致）', fill=LIGHT, h=1.5)
    stat(s, 8.75, 4.6, 4.08, '66–81 GiB', '八胞直接法 Cholesky 因子；NICE 显存峰值 9.2 GiB', fill=LIGHT, nsize=30, h=1.5)
    cite(s, '论文第 5.9 节，Table 5，Table ST12b；不同处理器：一块 GPU 对 16 个 CPU 线程，比较的是各自的部署方式')

    # ---------------------------------------------------------------- 18 speedup table
    s = D.slide('怎么读加速比：他们的 10²× 是拿两个数量级的误差换的',
                '把加速比和精度放在一张表里看。他们的一百倍以上，都来自每个子结构 24 个角点自由度，代价是 2% 到 20% 的误差；'
                '同一精度档，也就是全节点方案，他们只有 7 倍；而且基准多是串行或者没说明的求解器。我们 10 倍是对 16 线程 PARDISO，'
                '单线程口径 57 到 77 倍，精度是 0.01% 量级。答复信里只陈述事实，不下"更优"的结论。')
    table(s, X0, 1.15, 12.33, [
        ['工作', '加速比', '相对细网格的精度', '基准'],
        ['EML 2023', '328–472×', '无全尺度误差表', 'SIMP 前 10 步的平均'],
        ['JMPS 2024', '>230×', '柔度 −4.7%，位移 6.3%', 'Matlab，串行'],
        ['CMAME 2026', '设计迭代 7.4–63×', '位移 1.3–13.6%；集中载荷柔度 15–63%', '求解器未说明，串行'],
        ['CMAME 2026 全节点方案', '7.3–7.5×', '位移 0.84–3.46%', '同上'],
        ['CS 2026', '46×', '柔度 2.8–9.2%', 'PCG + 不完全 Cholesky'],
        ['IJMS 2026', '70–129×', '柔度 −20.7%，应力约低估 5.3 倍', 'OptiStruct 32 核'],
        [[('NICE', {'bold': True, 'color': BLUE})], [('≈10×（16 线程）；57–77×（单线程）', {'bold': True})],
         [('柔度 ≤0.015%，灵敏度 ≤0.14%', {'bold': True})], [('MKL PARDISO Cholesky + 常规精确凝聚', {'bold': True})]],
    ], (2.6, 2.9, 4.0, 2.8), size=14, row_h=0.56, hl_rows=(7,))
    text(s, X0, 5.85, 12.33, 0.7, [[('要点：', {'bold': True, 'color': RED}),
                                    ('同一精度档（全节点方案）他们只有 7×；正文不比 PIML 数值，被审稿人问到时按此表答（RESPONSE_PREP_SPEEDUP_CN.md）。', {})]],
         size=15, fill=LIGHT, anchor='m', margin=0.2)
    cite(s, '来源：GUO_SERIES_COMPARISON_CN.md 第 2 节；本文 Table 5、ST12b')

    # ---------------------------------------------------------------- 19 case A
    s = D.slide('优化算例 A：与精确凝聚的孪生运行几乎重合',
                '算例 A 是八胞块体，用 NICE 和精确凝聚各优化一遍。分别 24 和 23 次迭代停，柔度都降 2.0%、用料少 20%；'
                '最终角点参数最多差 0.0051。迭代 0、12、23 处 NICE 柔度比精确值低 0.011% 到 0.028%，符号正是式 (7) 给的；'
                '梯度误差不超过 0.33%，余弦不低于 0.999996，每个分量符号都对。')
    image(s, FIG / 'F13_optimisation.png', X0, 1.05, 7.3, 5.6, crop=(0, 0, 0.5, 0))
    stat(s, 8.1, 1.15, 4.73, '0.0051', '与孪生设计的最大角点参数差', fill=LBLUE, color=BLUE, h=1.45)
    text(s, 8.1, 2.8, 4.73, 3.8, [
        ('-', '24 / 23 次迭代停止；柔度都降 2.0%，用料少 20%'),
        ('-', '柔度比精确值低 0.011–0.028%（符号由 Eq. (7) 给出）'),
        ('-', '梯度误差 ≤0.33%，余弦 ≥0.999996，每个分量符号正确'),
        ('-', '每对相邻迭代之间离散模型都发生切换，仍收敛到同一设计'),
    ], size=15, space=8)
    cite(s, '图：论文 Fig. 12a；第 5.10 节，Table ST16–ST17')

    # ---------------------------------------------------------------- 20 plate
    s = D.slide('切割固支板：均匀化低估柔度 27%，设计在真实几何上更软',
                '切割固支板，8 乘 4 层切掉 8 个胞，剩 24 个，固支在切割面上，这由保留的切割带直接表示，不需要重训。'
                '均匀化在均匀初始设计上低估柔度 27%；它的设计放回切割几何上，比 NICE 设计软 2.44%，精确柔度 80.198 对 78.287，'
                '角点参数最多差 0.26。NICE 的 24 次迭代共 1.3 小时。')
    image(s, FIG / 'F13_optimisation.png', X0, 1.05, 6.0, 3.2, crop=(0.5, 0, 0, 0))
    image(s, FIG / 'F14_designs_scale.png', X0, 4.3, 6.0, 2.4, crop=(0, 0, 0.34, 0))
    stat(s, 6.85, 1.15, 2.85, '27.0%', '均匀化在初始设计上低估柔度', fill=LRED, h=1.55)
    stat(s, 9.98, 1.15, 2.85, '2.44%', '均匀化设计在切割几何上更软', fill=LRED, h=1.55)
    text(s, 6.85, 2.95, 5.98, 3.6, [
        ('-', '24 胞（16 未切割 + 8 切割），切割带全部自由度固支：保留切割带直接表示，不重训'),
        ('-', '精确柔度：均匀化设计 80.198，NICE 设计 78.287；用 NICE 或精确凝聚评估结论相同'),
        ('-', '两设计角点参数最多相差 0.26（相关系数 0.95）'),
        ('-', 'NICE 柔度误差 ≤0.030%，梯度误差 ≤0.30%；24 次迭代共 1.3 h'),
    ], size=15, space=7)
    cite(s, '图：论文 Fig. 12b、Fig. 13a,b；第 5.10 节，Table ST18c、ST21')

    # ---------------------------------------------------------------- 21 scale
    s = D.slide('规模：每次设计迭代耗时与胞数成正比，每胞约 10.7 s',
                '规模演示：24、51、88、110 胞的板，650 万到 3270 万自由度，每次设计迭代 4 到 19.4 分钟，和胞数成正比，'
                '每胞约 10.7 秒。CPU 内存每胞约 0.7 GiB，线性增长。现在的上限在预条件子里 K_PP 的直接分解，不在学习胞本身。')
    cd = CategoryChartData()
    cd.categories = ['24', '51', '88', '110']
    cd.add_series('每次设计迭代（min）', (4.0, 9.1, 16.0, 19.4))
    gf = s.shapes.add_chart(XL_CHART_TYPE.LINE_MARKERS, Inches(X0), Inches(1.1), Inches(7.4), Inches(5.5), cd)
    ch = gf.chart
    ch.has_title = True; ch.chart_title.text_frame.text = '每次设计迭代耗时（min）随胞数'
    for r in ch.chart_title.text_frame.paragraphs[0].runs:
        font(r, 15, DARK, True)
    ch.has_legend = False
    ser = ch.plots[0].series[0]; ser.format.line.color.rgb = rgb(BLUE); ser.format.line.width = Pt(2.5); ser.smooth = False
    ser.marker.format.fill.solid(); ser.marker.format.fill.fore_color.rgb = rgb(BLUE); ser.marker.size = 9
    pl = ch.plots[0]; pl.has_data_labels = True; pl.data_labels.font.size = Pt(12)
    pl.data_labels.position = XL_LABEL_POSITION.ABOVE
    va = ch.value_axis; va.has_major_gridlines = True; va.major_gridlines.format.line.color.rgb = rgb('E0E0E0')
    va.tick_labels.font.size = Pt(11); va.minimum_scale = 0; va.format.line.fill.background()
    ch.category_axis.tick_labels.font.size = Pt(12); ch.category_axis.has_title = True
    ch.category_axis.axis_title.text_frame.text = '胞数'
    for r in ch.category_axis.axis_title.text_frame.paragraphs[0].runs:
        font(r, 12, GRAY)
    stat(s, 8.2, 1.15, 4.63, '10.7 s / 胞', '每次设计迭代（10.1–10.9 s）', fill=LIGHT, h=1.45)
    stat(s, 8.2, 2.8, 4.63, '0.7 GiB / 胞', 'CPU 内存线性增长', fill=LIGHT, h=1.45)
    text(s, 8.2, 4.45, 4.63, 2.1, [
        ('-', '650 万–3270 万自由度，凝聚为 39 万–162 万自由保留自由度'),
        ('-', '上限在预条件子里 K_PP 的直接分解，不在学习胞'),
        ('-', '24 胞以上没有做精确核对'),
    ], size=14, space=6)
    cite(s, '论文第 5.10 节，Fig. 13c，Table ST20（每个规模跑前 4 次设计迭代）')

    # ---------------------------------------------------------------- 22 limitations
    s = D.slide('局限与审稿人可能的追问',
                '局限我们在 6.4 节都写了：只有一种胞族、一种离散，网络不等变；没有认证误差界，只有可计算的残差；'
                '灵敏度用场基估计；24 胞以上没有精确核对；没和 AMG、BDDC 比。审稿人最可能问加速比，答复材料已经准备好。')
    card(s, X0, 1.2, 5.95, 3.9, '局限（6.4 节已如实写出）', [
        ('-', '一种胞族（Schwarz-P）、一种离散（n = 32）；网络不等变'),
        ('-', '没有认证误差界，只有可计算残差'),
        ('-', '灵敏度为场基估计，省略延拓的设计导数'),
        ('-', '24 胞以上无精确核对；未与 AMG、BDDC 比较'),
        ('-', '训练对比只用一个随机种子'),
    ], fill=LIGHT, size=15)
    card(s, X0 + 6.38, 1.2, 5.95, 3.9, '可能的追问 → 已备答复', [
        ('-', [('"加速只有 10×"', {'bold': True}), ('→ 精度层级不同；同精度档他们 7×；基准强度不同（单线程口径 57–77×）', {})]),
        ('-', [('"只有一种胞族"', {'bold': True}), ('→ 构造与误差关系通用，网络适用范围 6.4 节写明', {})]),
        ('-', [('"一个种子"', {'bold': True}), ('→ 如实写出；被要求可补多种子（GPU 约 1–2 天）', {})]),
        ('-', [('"没比 AMG / BDDC"', {'bold': True}), ('→ 已写明；属于后续"精确求解器加速"方向', {})]),
    ], fill=LBLUE, head_color=BLUE, size=15)
    cite(s, '论文第 6.4 节；review_r1/RESPONSE_PREP_SPEEDUP_CN.md')

    # ---------------------------------------------------------------- 23 submission plan
    s = D.slide('投稿计划：CMAME，同时挂 arXiv',
                '投稿计划：投 CMAME，同时挂 arXiv 占优先权。投稿前需要作者信息、各项声明包括 AI 使用声明、通读。'
                '可选的是先给 Nature Computational Science 发一封预投稿咨询，一两周回复，预期会被建议转投专业期刊，几乎不耽误时间。')
    steps = [('1  作者信息与声明', '作者、单位、CRediT、利益冲突、致谢、AI 使用声明（作者本人填写）'),
             ('2  通读定稿', '中英文版、Word 版均已就绪，可分工通读'),
             ('3  （可选）预投稿咨询', 'Nature Computational Science，1–2 周回复；预期建议转专业期刊'),
             ('4  投 CMAME + arXiv', '同时挂预印本，确立 NICE、单侧界、平衡校正的优先权')]
    bw = (12.33 - 3 * 0.45) / 4
    for k, (h, b) in enumerate(steps):
        x = X0 + k * (bw + 0.45)
        card(s, x, 1.5, bw, 2.5, h, [b], fill=LBLUE if k == 3 else LIGHT, head_color=BLUE if k == 3 else RED, size=15, head_size=16)
        if k < 3:
            arrow(s, x + bw + 0.07, 2.55, 0.31, 0.38)
    text(s, X0, 4.4, 12.33, 1.3, [
        [('为什么是 CMAME：', {'bold': True, 'color': RED}),
         ('论文的强项（变分结构、可组装、误差链、严格的精确凝聚核对）在 CMAME 最被看重；冲更高需要新工作（第二胞族、更大规模、实验），放到后续论文。', {})]],
        size=16, fill=LIGHT, anchor='m', margin=0.2)

    # ---------------------------------------------------------------- 24 roadmap
    s = D.slide('后续路线：不比加速比，比"可行与不可行"',
                '后续路线一页概览，细节在路线图 PDF 里。P2 做带误差控制的局部量设计，把后验误差和规模作为支撑；'
                'P3 做跨胞族泛化，加上我们的 B 样条拉扭单胞；冲 NC 的方向是局部量驱动的有限尺寸格栅设计加打印和破坏试验，'
                '卖点是可行与不可行；另外有几篇低投入的并行小篇。')
    items = [('P2  局部量设计', ['带误差控制的局部量（应力）设计', '后验误差估计与自适应校正', '规模（K_PP 多层粗解）作支撑'], LIGHT, RED),
             ('P3  跨胞族泛化', ['一网多族、映射几何、对称内建', 'B 样条拉扭单胞作第二胞族', '有限尺寸的非经典（拉扭）响应'], LIGHT, RED),
             ('NC  有限尺寸设计', ['局部量驱动、上千胞的真实零件', 'NICE 设计 vs 均匀化设计打印对照、破坏试验', '卖点：直接法放不下，我们能算'], LBLUE, BLUE),
             ('并行小篇（低投入）', ['切割胞数据集', 'CutFEM 方法', '均匀化在切割边界失效的定量研究 + 实测'], LIGHT, GRAY)]
    W4 = (12.33 - 3 * 0.3) / 4
    for k, (h, body, f, hc) in enumerate(items):
        card(s, X0 + k * (W4 + 0.3), 1.25, W4, 3.4, h, [('-', b) for b in body], fill=f, head_color=hc, size=15, head_size=16)
    text(s, X0, 4.95, 12.33, 1.25, [
        [('先做共享前置：', {'bold': True, 'color': RED}), ('K_PP 多层粗解、后验估计 + 最坏方向诊断、质量 / 几何刚度凝聚、第二胞族。', {})],
        [('依据：', {'bold': True, 'color': RED}), ('直接法 Cholesky 因子超线性增长（四胞 29–32 → 八胞 66–81 GiB），NICE 每胞约 0.7 GiB、线性增长。', {})],
    ], size=15, space=6, fill=LIGHT, anchor='m', margin=0.2)
    cite(s, '详见 docs/followup/ROADMAP_20261003_CN.pdf')

    # ---------------------------------------------------------------- 25 decisions
    s = D.slide('需要老师拍板的四件事',
                '最后四件事需要您拍板：投稿去向；打印和测试条件；后续优先做哪几个方向、时间窗口多长；算力要不要增租。')
    qs = [('1', '投稿去向', '直接投 CMAME，还是先发 Nat. Comput. Sci. 预投稿咨询？'),
          ('2', '实验条件', '能否打印金属 / 聚合物 TPMS 试件？有无 DIC？有无激光测振？'),
          ('3', '方向与时间', '两年内主线选哪 3–4 个？NC 方向 1 约需 4–6 个月（前置完成后）'),
          ('4', '算力', '是否增租 GPU（P2 规模、P3 新胞族数据都需要）？')]
    for k, (n, h, b) in enumerate(qs):
        y = 1.25 + k * 1.33
        c = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(X0), Inches(y + 0.1), Inches(0.85), Inches(0.85))
        c.fill.solid(); c.fill.fore_color.rgb = rgb(RED); c.line.fill.background(); c.shadow.inherit = False
        tf = c.text_frame; tf.paragraphs[0].alignment = PP_ALIGN.CENTER
        r = tf.paragraphs[0].add_run(); r.text = n; font(r, 24, 'FFFFFF', True)
        text(s, X0 + 1.1, y + 0.02, 11.2, 0.5, [h], size=19, color=RED, bold=True)
        text(s, X0 + 1.1, y + 0.52, 11.2, 0.6, [b], size=17)

    # ---------------------------------------------------------------- backup divider
    s = D.slide('备用页', '以下为备用页，被追问时再翻。')
    text(s, X0, 2.6, 12.33, 1.8, [['B1  排序条件与离散切换'], ['B2  参考解误差与训练成本'], ['B3  映射单胞零样本测试（后续工作前期结果）'],
                                   ['B4  PIML 六篇完整对照']], size=20, color=GRAY, align='c', space=10)

    # ---------------------------------------------------------------- B1
    s = D.slide('B1  排序条件与离散切换',
                '排序 Ŝ 不大于 Ŝ_net 需要精确或非负平移的粗求解，光滑区间从上方界住谱。离散切换：几何变化时活跃单元和保留集会跳变，'
                '所以优化不用 KKT 残差停止，而用目标相对变化。')
    card(s, X0, 1.2, 5.95, 3.4, '校正的排序条件（附录 C、D）', [
        ('-', [('S ⪯ Ŝ ⪯ Ŝ_net', {'math': True})]),
        ('-', '条件：粗求解精确或非负平移；Jacobi 尺度化谱落在 (0, b]'),
        ('-', '光滑区间上端点 b 从上方界住谱；下端点 a = b/30'),
        ('-', '满足时校正不会增大误差；实测中加大预算误差持续下降（Table ST07）'),
    ], fill=LIGHT, size=15)
    card(s, X0 + 6.38, 1.2, 5.95, 3.4, '离散模型切换（第 3.3 节，附录 B.1）', [
        ('-', '几何变化时活跃单元、保留集会跳变：目标函数在设计迭代之间改变'),
        ('-', '因此停止判据用目标相对变化 < 10⁻⁴ 连续 3 次，不用 KKT 残差'),
        ('-', '算例 A 每对相邻迭代至少 3/8 个胞切换；板至少 11/24 个胞'),
        ('-', '残差功始终 ≤5.4×10⁻⁸ 倍柔度'),
    ], fill=LIGHT, size=15)

    # ---------------------------------------------------------------- B2
    s = D.slide('B2  参考解误差与训练成本',
                '参考解本身的离散误差：n 等于 32 相对 64，除 H2 外柔度不超过 0.28%、灵敏度 0.44%。训练一次性约 60 GPU 小时，不计入表 5。')
    table(s, X0, 1.2, 12.33, [
        ['项目', '数值', '出处'],
        ['参考解 n = 32 相对 n = 64（除 H2）', '柔度 ≤0.28%，灵敏度 ≤0.44%', '第 5.2 节，Table ST10'],
        ['参考解 n = 32 相对 n = 64（H2）', '柔度约 1%，灵敏度约 3.6%', '同上'],
        ['ghost 罚系数 10⁻⁵–10⁻³、体积积分加密', '柔度与灵敏度变化 ≤0.12%', '第 5.2 节，补充说明 S1'],
        ['训练几何', '691 个（591 训练 + 100 验证）', 'Table ST01'],
        ['数据生成 / 训练', '约 42 GPU-h / 4.6 h + 3.5 h，总计约 60 GPU-h', '第 5.1 节'],
    ], (4.2, 4.6, 3.5), size=15, row_h=0.62)

    # ---------------------------------------------------------------- B3
    s = D.slide('B3  映射单胞零样本：结构始终成立，拉伸是唯一明显失效',
                'P1 网络不重训直接用在映射胞上：384 对全部保持 Ŝ 不小于 S；共转拉回后扭转、弯曲误差在 0.1% 量级；'
                '拉伸是唯一明显失效。另外最坏方向误差远大于平均值，是后续要查清的问题。')
    table(s, X0, 1.2, 12.33, [
        ['映射', '平均能量误差（共转拉回）', '最差胞', '最坏方向 μ−1 中位数'],
        ['恒等（即 P1）', '0.072%', '0.29%', '0.23%'],
        ['整胞转 30°', '0.074%（直接用 3.72%）', '0.30%', '0.31%'],
        ['扭转 10° / 胞', '0.089%', '0.33%', '0.85%'],
        ['扭转 30° / 胞', '0.27%', '0.88%', '18%'],
        ['弯曲 R/L = 5', '0.097%', '0.43%', '0.27%'],
        ['拉伸 ×0.5（cond J = 2）', '3.9%', '8.9%', '215%'],
    ], (3.6, 3.6, 2.2, 2.9), size=15, row_h=0.55)
    text(s, X0, 5.25, 12.33, 1.3, [
        ('-', '384 对全部保持 Ŝ ⪰ S 与对称半正定；刚体基须在物理坐标里分离'),
        ('-', '最坏方向（只是下界）远大于平均：恒等映射下重切胞 2012 的 μ−1 ≥ 44.5% —— 待查清是否会在装配中被激发'),
    ], size=15, space=6)
    cite(s, '来源：docs/followup/A/full16/SUMMARY.md；STEP0_PLAN_CN.md 第 10 节')

    # ---------------------------------------------------------------- B4
    s = D.slide('B4  PIML 六篇完整对照',
                '六篇和 NICE 在边界表示、灵敏度、理论、校正上的完整对照。')
    table(s, X0, 1.15, 12.33, [
        ['维度', 'EML 2023 / JMPS 2024', 'CMAME 2026', 'CS 2026', 'IJMS 2026', 'NICE'],
        ['边界自由度', '24', '168（Bézier）', '24', '24', '2,679–45,900'],
        ['训练', '监督 / 无数据', '监督', '监督', '监督（迁移）', '能量对数比 + 灵敏度项'],
        ['灵敏度', '场基，忽略 ∂N/∂ρ', '未说明', '未说明', '场基；应力问题反传', '场基 + 完整导数；逐胞报告'],
        ['理论', '刚体不变性', '插值不变性', '误差放大估计', '刚体不变性', 'Ritz 下界、份额界、校正排序'],
        ['校正', '无', '无', '投影（rank-6）', '无（转细网格）', '固定两层网格校正 𝒲'],
        ['参考解', '细网格 / EMsFEM', '细网格（设计未复核）', '同边界子结构', '同边界子结构', '同一离散的精确凝聚'],
    ], (1.6, 2.2, 2.0, 1.8, 2.0, 2.7), size=13, row_h=0.62, hl_rows=())
    cite(s, '来源：GUO_SERIES_COMPARISON_CN.md 第 2 节')

    n = len(D.prs.slides)
    D.total(n)
    D.prs.save(out)
    print(out, n, 'slides')


if __name__ == '__main__':
    build(sys.argv[1], sys.argv[2])
