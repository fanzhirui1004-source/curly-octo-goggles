"""Build the advisor-meeting deck (P1, NICE) on the group's PowerPoint template.

Usage: python3 build_talk.py <template.pptx> <out.pptx>
The template's first slide (title page) is kept and edited; its other slides are removed; every new slide uses the
template's single layout (red title at the top, group footer). Figures come from docs/paper_p1/figures; the cost and
scale charts are native PowerPoint charts. Numbers follow docs/followup/ADVISOR_TALK_OUTLINE_20261003_CN.md, the
manuscript and docs/paper_p1/review_r1/GUO_SERIES_COMPARISON_CN.md. Speaker notes are written on every slide."""
import sys, copy, re
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
LATIN, EA, MATH = 'Arial', 'Microsoft YaHei', 'Times New Roman'
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
        pieces = re.split(r'(_\{[^}]*\}|\^\{[^}]*\})', text) if o.get('math') else [text]
        for piece in pieces:
            if not piece:
                continue
            base = 0
            if piece.startswith('_{'):
                piece, base = piece[2:-1], -25000
            elif piece.startswith('^{'):
                piece, base = piece[2:-1], 30000
            r = p.add_run(); r.text = piece
            font(r, o.get('size', size), o.get('color', color), o.get('bold', bold), o.get('italic', False), o.get('math', False))
            if base:
                r._r.get_or_add_rPr().set('baseline', str(base))


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


def eq(slide, x, y, w, h, lines, size=20, fill=LIGHT, color=DARK):
    """Display formulas: each line a str with _{..} / ^{..} for sub- and superscripts (set in Cambria Math)."""
    return text(slide, x, y, w, h, [[(ln, {'math': True})] for ln in lines], size=size, color=color, align='c',
                anchor='m', fill=fill, space=6, margin=0.15)


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
        '老师好。今天汇报 P1 的定稿情况：问题和相关工作、方法、结果，以及投稿计划和后续方向。'
        '最后有几件事需要您拍板。')

    # ---------------------------------------------------------------- 2 one-page conclusion
    s = D.slide('一页结论：保留完整边界，只近似内部，误差能追到灵敏度',
                '先说结论。我们的学习模型保留每个胞全部盒面和切割带自由度，约两万四千个，但保住了静力凝聚的结构：'
                '对称半正定、刚体核、被精确 Schur 补从下方界住。四个数字：单胞平均能量误差 0.074%；两胞装配柔度 0.28%、'
                '灵敏度 1.5% 以内；八胞分析约为 16 线程直接解的十分之一时间；切割固支板上均匀化低估柔度 27%。'
                '和 PIML 系列的取舍不同：他们缩减边界换规模，我们保留完整边界、只近似内部，换来精度和对灵敏度的可追溯性。目标期刊 CMAME。')
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
        [('与 PIML 系列的不同取舍：', {'bold': True, 'color': BLUE}),
         ('PIML 缩减边界（多为 24 个角点自由度）换规模；本文保留完整边界、只近似内部，换来 0.01% 量级的精度，并说清误差怎么进入灵敏度。', {})]],
        size=17, fill=LBLUE, anchor='m', margin=0.2)
    text(s, X0, 6.05, 12.33, 0.5, [[('状态：', {'bold': True}), ('正文 63 页 + 补充 37 页，可投；目标 ', {}), ('CMAME', {'bold': True, 'color': BLUE}),
                                    ('，同时挂 arXiv', {})]], size=16, color=GRAY)

    # ---------------------------------------------------------------- agenda
    s = D.slide('汇报提纲',
                '汇报分六部分：问题与相关工作；问题设定，也就是离散模型、精确凝聚和四项要求；误差理论；NICE 的构造；数值结果；'
                '最后是投稿计划和后续方向。方法和结果部分会讲得细一些。')
    parts = [('1', '问题与相关工作', '组件模型的几条路线；与 PIML 系列的取舍'),
             ('2', '问题设定', '切割薄壁胞的离散模型；保留自由度与精确凝聚；四项要求'),
             ('3', '误差理论', 'Ritz 恒等式；按能量份额加权的柔度界；灵敏度中的一次项'),
             ('4', 'NICE 的构造', '变分凝聚刚度；误差谱与两层网格校正；几何条件化网络；训练与部署'),
             ('5', '数值结果', '单胞、两胞装配、消融、八胞格栅、成本、厚度优化、规模'),
             ('6', '计划', '局限与审稿准备；投稿；后续路线；需要拍板的事')]
    for k, (n, h, b) in enumerate(parts):
        col, row = k % 2, k // 2
        x, y = X0 + col * 6.33, 1.25 + row * 1.75
        c = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(y + 0.1), Inches(0.8), Inches(0.8))
        c.fill.solid(); c.fill.fore_color.rgb = rgb(RED); c.line.fill.background(); c.shadow.inherit = False
        tf = c.text_frame; tf.paragraphs[0].alignment = PP_ALIGN.CENTER
        r = tf.paragraphs[0].add_run(); r.text = n; font(r, 22, 'FFFFFF', True)
        text(s, x + 1.0, y + 0.02, 5.0, 0.5, [h], size=20, color=RED, bold=True)
        text(s, x + 1.0, y + 0.55, 5.0, 0.9, [b], size=15, color=GRAY)

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
                '学习子结构，比如 PIML 系列，是第四条路，下一页把相关工作放在一起看。')
    W4 = (12.33 - 3 * 0.3) / 4
    cards = [
        ('均匀化', ['依赖尺度分离', '在支撑、载荷、切割胞处失效', [('切割固支板：柔度低估 ', {}), ('27%', {'bold': True, 'color': RED})]]),
        ('全分辨率', ['每次设计迭代改变每个胞，代价每轮全额重付', [('八胞 183–211 万自由度，Cholesky 因子 ', {}), ('66–81 GiB', {'bold': True, 'color': RED})]]),
        ('精确凝聚', ['每个几何一次内部分解', '显式 Schur 补：每个保留自由度一次内部求解', [('一胞中位 2.4×10⁴ 个保留自由度', {'bold': True, 'color': RED})]]),
        ('学习子结构', ['以 PIML 系列为代表', '每个子结构保留少量边界自由度，换来规模', [('边界误差与网络无关，需另行控制', {'bold': True, 'color': RED})]]),
    ]
    for k, (h, body) in enumerate(cards):
        card(s, X0 + k * (W4 + 0.3), 1.3, W4, 3.4, h, [('-', b) for b in body], fill=LIGHT, size=15)
    text(s, X0, 5.0, 12.33, 0.9, [[('我们的位置：', {'bold': True, 'color': BLUE}),
                                    ('走子结构这条中间路线，但把近似放在内部延拓上，保留完整的保留空间；结构由构造保证，不靠训练。', {})]],
         size=17, fill=LBLUE, anchor='m', margin=0.2)

    # ---------------------------------------------------------------- 5 related work
    s = D.slide('相关工作：组件模型的五条路线',
                '先把相关工作放在一张表里。按论文引言的逻辑，组件模型大致有五条路线：均匀化与多尺度有限元；组件降基和端口缩减；'
                '面向格栅的凝聚与区域分解；学习子结构和学习单元，PIML 系列是其中发展最充分的一支；以及把学习用在求解器里的工作。'
                '区分它们的三个问题是：保留什么、组件怎么表示、近似在哪里改进。我们的位置在最下面一行。')
    table(s, X0, 1.1, 12.33, [
        ['路线', '代表工作', '做法与特点', '与本文的关系'],
        ['均匀化 / 多尺度', 'Li et al. 2018；Hou & Wu 1997', '等效性质或多尺度基函数；依赖尺度分离', '切割、支撑处失效（本文板算例 27%）；MsFEM 与本文同用能量形式'],
        ['组件降基 / 端口缩减', 'Huynh et al. 2013；Eftang & Patera 2013；Chasapi et al. 2023', '离线构造降基，可给后验界；基于参数族共享的离散空间', '本文的离散空间随切割变化，网络以每胞的离散几何为条件'],
        ['格栅凝聚与区域分解', 'Wu et al. 2019；Hirschler et al. 2024；Bonilla Moreno et al. 2027；Chen & Li 2026', '超单元插值、ROM-FETI-DP、2D 非贴体 BDDC、非贴体数值形函数', '各自在胞间保持同一离散；本文面对随几何变化的保留集'],
        ['学习子结构 / 学习单元', 'PIML（Huang et al. 2022–2024；Guo et al. 2026a,b 等）；Parish et al. 2024；Jiang H. et al. 2026', '学习形函数或界面刚度；PIML 保留少量边界自由度，规模大', '本文保留完整保留空间，近似只在内部延拓（下一页对比）'],
        ['学习 + 求解器', 'Greenfeld et al. 2019；Heinlein et al. 2021；Zhang E. et al. 2024', '学习多重网格、粗空间或混合求解器；迭代收敛到精确解', '本文的校正改变的是被组装的算子本身'],
        [[('本文 NICE', {'bold': True, 'color': BLUE})], [('—', {})],
         [('完整保留空间 + 几何条件化的线性延拓 + 固定两层网格校正', {'bold': True})],
         [('可组装、单侧界、部署时可改进、误差追到灵敏度', {'bold': True})]],
    ], (1.9, 3.3, 3.6, 3.5), size=12, row_h=0.74, hl_rows=(6,))
    cite(s, '论文第 1 节（引言）与参考文献')

    # ---------------------------------------------------------------- 6 PIML comparison
    s = D.slide('与 PIML 系列的比较：两种不同的取舍',
                'PIML 系列和我们都是学习子结构，但取舍相反。他们每个子结构保留少量边界自由度，换来很小的粗模型，规模做到十亿单元、'
                '应用很广；代价是有一部分与网络无关的边界误差，他们用分区加密、边界增强、过采样来控制，系列里也没有报告灵敏度误差。'
                '我们保留完整的边界，误差只在内部延拓里、可以校正，柔度和灵敏度都到 0.1% 以内；代价是粗模型大，目前规模到 110 胞。'
                '两条路线的交汇点是：他们 CMAME 2026 也试过保留全部边界节点，因为成本没有采用；我们的线性网络表示和固定校正，'
                '正是让完整边界变得可行的部分。')
    card(s, X0, 1.15, 6.0, 3.85, 'PIML：缩减边界，换规模', [
        ('-', '每个子结构保留少量边界自由度（多为 24 个角点；Bézier 为 168）'),
        ('-', '粗模型小：串行 10⁹ 单元；应用广（柔顺机构、应力、复杂域）'),
        ('-', '边界误差与网络无关，靠分区加密、边界增强、过采样控制'),
        ('-', '细网格复核的柔度误差 2.8–20.7%（偏刚）；未报告灵敏度误差'),
    ], fill=LIGHT, size=15)
    card(s, X0 + 6.33, 1.15, 6.0, 3.85, 'NICE：保留完整边界，换精度与可追溯性', [
        ('-', '每胞保留全部盒面与切割带自由度（2,679–45,900）'),
        ('-', '误差只在内部延拓，可由固定校正降低；切割面支撑、加载不需重训'),
        ('-', '八胞格栅：柔度 ≤0.015%，灵敏度 ≤0.14%'),
        ('-', '粗模型大：目前到 110 胞；加速约 10×（16 线程）'),
    ], fill=LBLUE, head_color=BLUE, size=15)
    text(s, X0, 5.2, 12.33, 1.4, [
        [('交汇点：', {'bold': True, 'color': RED}),
         ('Guo et al. (2026a) 也评估过保留全部边界节点（每个子结构 456–1,806 个自由度），加速 7.3–7.5×、位移误差 0.84–3.46%，'
          '因成本未采用。本文对保留位移精确线性的网络表示和固定校正，正是让完整边界可行的部分。', {})],
    ], size=15, fill=LIGHT, anchor='m', margin=0.2)
    cite(s, '来源：Guo et al. (2026a) Table 1；review_r1/GUO_SERIES_COMPARISON_CN.md 第 2、3 节')

    # ---------------------------------------------------------------- setup: discrete model
    s = D.slide('问题设定（一）：切割薄壁 Schwarz-P 胞的离散模型',
                '先交代问题设定。单胞在单位盒里，材料域由 Schwarz-P 型水平集的一条带定义，带宽由八个角点参数三线性插值，'
                '可选一个平面切割。离散用笛卡尔背景网格上的 Q2 单元，每轴 32 个单元，加 ghost 罚项稳定小切割。'
                '要强调的一点是：本文所有误差都相对于这个离散系统的精确解度量，不是相对于连续问题。')
    eq(s, X0, 1.15, 6.4, 1.75, ['φ(x) = Σ_{a} cos(2πx_{a}),   τ(x) = Σ_{c} N_{c}^{Q1}(x) τ_{c}',
                                'Ω(η) = { x ∈ [0,1]^{3} :  |φ(x)| ≤ τ(x),  n·x ≤ b_{cut} }'], size=18)
    text(s, X0, 3.1, 6.4, 3.5, [
        ('-', [('八个角点厚度参数 ', {}), ('τ_{c}', {'math': True}), ('（带参数，非逐点壁厚）；对它们的导数即厚度灵敏度', {})]),
        ('-', '至多一个平面切割，法向背离保留材料'),
        ('-', '背景单元与 Ω 交集测度为正即为激活单元，由区间包络认证'),
        ('-', [('稳定化离散能量 ', {}), ('½ u^{T}Ku', {'math': True}), ('；参考解即该离散系统的平衡解', {'bold': True})]),
    ], size=16, space=8)
    table(s, 7.2, 1.15, 5.63, [
        ['量', '设置'],
        ['位移近似', '连续张量积 Q2 实体单元'],
        ['背景分辨率', '每轴 n = 32 个单元（65 个 Q2 节点）'],
        ['材料', 'E_Y = 1，ν = 0.3（归一化）'],
        ['稳定化', 'ghost 罚项，γ = 10⁻⁴'],
        ['体积积分', '4³ 子胞 + 局部加密；裁剪 Kuhn 四面体'],
        ['厚度差分步长', 'h_c = 10⁻⁵ τ_c，激活集固定'],
        ['参数范围', '角点参数 0.176–0.698'],
    ], (1.6, 4.0), size=14, row_h=0.56)
    cite(s, '论文第 2.1 节，式 (1)，表 1；附录 A')

    # ---------------------------------------------------------------- setup: retained DOFs and exact condensation
    s = D.slide('问题设定（二）：保留自由度与精确静力凝聚',
                '保留自由度包括激活的盒面自由度，以及切割带，也就是被切割面穿过的那一层单元的全部自由度。切割带保留下来，'
                '切割面上的支撑和载荷就能直接施加。精确凝聚是标准的 Schur 补。难点在规模：一个胞中位保留两万三千多个自由度，'
                '显式形成 Schur 补要对每个保留自由度做一次内部求解；而设计每迭代一次，每个胞的几何都变，内部分解都得重做。')
    image(s, FIG / 'F12_field_error_M1.png', X0, 1.05, 4.6, 4.2, crop=(0, 0, 0.5, 0.5))
    text(s, X0, 5.3, 4.6, 1.3, [[('图：M1 的保留盒面自由度（黑）、保留切割带自由度（蓝）与内部自由度（灰）', {'color': GRAY})]], size=12)
    eq(s, 5.4, 1.15, 7.43, 1.65, ['K = [ K_{PP}  K_{PI} ;  K_{IP}  A ],   A = K_{II}',
                                  'E = [ I ; −A^{−1}K_{IP} ],   S = K_{PP} − K_{PI}A^{−1}K_{IP} = E^{T}KE'], size=18)
    text(s, 5.4, 3.0, 7.43, 3.6, [
        ('-', [('保留集 P', {'bold': True}), ('：激活盒面自由度 + 切割带（被切割面穿过的激活单元层）的全部自由度', {})]),
        ('-', '切割带承担切割面上的位移和虚功 → 切割面支撑、载荷可直接施加，无需重训'),
        ('-', [('规模：', {'bold': True}), ('每胞 2,679–45,900 个保留自由度（中位 23,604）', {})]),
        ('-', [('代价：', {'bold': True}), ('显式 S 需每个保留自由度一次内部求解；作用 S 需每个几何一次内部分解；每次设计迭代每个胞都变', {})]),
        ('-', [('假设：', {}), ('K ⪰ 0，A ≻ 0；自由胞恰有 6 个刚体模态', {'math': True})]),
    ], size=15, space=7)
    cite(s, '论文第 2.2–2.3 节，式 (2)–(3)；图：论文 Fig. 6a')

    # ---------------------------------------------------------------- setup: four requirements
    s = D.slide('问题设定（三）：学习组件模型要替代静力凝聚，需满足四项要求',
                '要替代静力凝聚，学习模型需要满足四项要求：可组装，也就是对称半正定且零空间只有刚体模态；对新几何不需要内部分解，'
                '即使切割改变了离散空间；误差能追到柔度和设计需要的局部灵敏度；部署时不重训就能改进。右边一列是本文分别靠什么满足的。')
    table(s, X0, 1.2, 12.33, [
        ['要求', '为什么需要', '本文如何满足'],
        ['① 对称半正定，零空间只含刚体模态', '才能装配：带支撑的装配矩阵正定', '能量形式 Ŝ = FᵀKF + 构造上再现刚体（式 (11)、(17)）'],
        ['② 新几何无需内部分解，离散空间随切割变化也适用', '设计每迭代一次，每个胞都变', '几何条件化网络，对保留位移线性；一个网络服务所有保留集'],
        ['③ 误差可追溯到柔度与局部灵敏度', '设计由局部灵敏度驱动', 'Ritz 恒等式 + 份额加权界 + 灵敏度误差式（第 3 节）'],
        ['④ 部署时可改进，无需重训', '精度不够时可加预算', '固定两层网格校正 𝒲：给定条件下不增大误差（第 4.3 节）'],
    ], (3.6, 3.3, 5.4), size=15, row_h=0.85)
    text(s, X0, 5.65, 12.33, 0.9, [[('关键：', {'bold': True, 'color': BLUE}),
                                    ('①④ 由构造保证，对任何网络输出都成立；②靠网络表示；③ 是本文的误差理论，判定何时可用于设计。', {})]],
         size=16, fill=LBLUE, anchor='m', margin=0.2)
    cite(s, '论文第 1 节第 2 段，第 2.3 节')

    # ---------------------------------------------------------------- theory 1: Ritz identity
    s = D.slide('误差理论（一）：Ritz 恒等式与方向能量误差',
                '第一个关系是 Ritz 恒等式。任何容许延拓 F，只要保留行是单位阵，和精确延拓的差别只在内部，记为 H。内部平衡消去了交叉项，'
                '所以近似凝聚刚度减精确 Schur 补正好等于 H 转置 A H，半正定，没有一阶项。方向能量误差可以用内部残差表示，'
                '残差不需要参考解就能算，只是取 A 逆范数要一次内部求解。这是经典结果，对 PIML 用 N 转置 K N 的形式同样成立。')
    eq(s, X0, 1.2, 12.33, 1.1, ['J_{P}F = I_{p},   H = J_{I}(F − E),   Ŝ = F^{T}KF   ⟹   Ŝ − S = H^{T}AH ⪰ 0'], size=22, fill=LBLUE)
    eq(s, X0, 2.5, 12.33, 1.1, ['ε(q) = q^{T}(Ŝ − S)q / q^{T}Sq = d_{I}^{T}Ad_{I} / q^{T}Sq = r_{I}^{T}A^{−1}r_{I} / q^{T}Sq,   r_{I} = (KFq)_{I}'], size=20)
    W3 = (12.33 - 2 * 0.3) / 3
    card(s, X0, 3.85, W3, 2.0, '单侧', ['近似刚度永远不比精确刚度软：S ⪯ Ŝ；装配柔度偏小（偏刚）'], size=15)
    card(s, X0 + W3 + 0.3, 3.85, W3, 2.0, '二次', ['误差是内部误差的能量，没有一阶项：参考场平衡的直接结果'], size=15)
    card(s, X0 + 2 * (W3 + 0.3), 3.85, W3, 2.0, '可计算', ['内部残差无需参考解即可得到；其 A⁻¹ 范数需每方向一次内部求解'], size=15)
    cite(s, '论文第 3.1 节，式 (4)–(5)；附录 B（经典结果：Fraeijs de Veubeke 1965；Toselli & Widlund 2005）')

    # ---------------------------------------------------------------- theory 2: compliance and energy share
    s = D.slide('误差理论（二）：装配柔度误差按能量份额加权',
                '第二个关系：装配后柔度的低估量，等于重构误差的总能量，分成保留解的变化和内部不平衡两部分。再定义每个胞的能量份额 w，'
                '和它在精确迹处的能量误差 ε，柔度相对误差就被 β 界住，β 是份额加权的能量误差之和。意思是：一个胞局部场不准，'
                '但它承担的能量很少，柔度照样准。比如 U1/x 一个邻胞载荷下，基础网络目标胞份额 0.127%、能量误差 2.31%，'
                'β 给出 0.00295%，观测 0.00283%，但这个胞的灵敏度误差有 5.83%。')
    eq(s, X0, 1.15, 12.33, 1.0, ['C − Ĉ = ‖Û − U‖^{2}_{𝕂} + Σ_{m} ‖H_{m}B_{m}Û‖^{2}_{Am}'], size=21)
    eq(s, X0, 2.3, 12.33, 1.0, ['w_{m} = q_{m}^{T}S_{m}q_{m} / C,   β = Σ_{m} w_{m} ε_{m}(q_{m}),   0 ≤ (C − Ĉ)/C ≤ β/(1+β) ≤ β'], size=21, fill=LBLUE)
    image(s, FIG / 'F10_energy_share.png', X0, 3.45, 5.6, 3.25, crop=(0, 0.04, 0, 0.5))
    card(s, 6.4, 3.5, 6.43, 3.1, '例：U1/x，邻胞面 z 向载荷，基础网络', [
        ('-', '目标胞能量份额 w = 0.127%，局部能量误差 ε = 2.31%'),
        ('-', 'β = wε = 0.00295%；实测柔度误差 0.00283%'),
        ('-', [('同一载荷下目标胞灵敏度误差 ', {}), ('5.83%', {'bold': True, 'color': RED})]),
        ('-', '→ 份额小的胞可以让柔度看起来很准'),
    ], fill=LRED, size=15)
    cite(s, '论文第 3.2 节，式 (6)–(7)；第 5.6 节；图：论文 Fig. 10(a,b)')

    # ---------------------------------------------------------------- theory 3: sensitivity
    s = D.slide('误差理论（三）：灵敏度误差含一次项，能量误差管不住',
                '第三个关系是灵敏度。精确柔度灵敏度是负的 u 转置 K 导数 u；我们用重构场代入，得到场基估计。两者之差有一个线性交叉项和一个二次项。'
                '内部平衡只保证 K u 的内部分量为零，不保证 K 导数乘 u 的内部分量为零，所以线性项一般存在；能量误差减小，灵敏度误差不一定单调减小。'
                '代理柔度的完整导数还多一项，耦合延拓的设计依赖和内部残差；本文报告的是场基估计，它估计的是精确灵敏度。')
    eq(s, X0, 1.15, 12.33, 1.0, ['s_{c} = −u^{T}K_{,c}u,    s̃_{c} = −û^{T}K_{,c}û'], size=21)
    eq(s, X0, 2.3, 12.33, 1.0, ['s̃_{c} − s_{c} = −2 d^{T}K_{,c}u − d^{T}K_{,c}d,    u = Eq,  d = Fq − Eq'], size=21, fill=LRED)
    eq(s, X0, 3.45, 12.33, 1.0, ['Ĉ_{,c} = s̃_{c} − 2 (F_{I,c}q̂)^{T} r_{I}    （完整设计导数，式 (10)）'], size=20)
    text(s, X0, 4.65, 12.33, 1.95, [
        ('-', [('线性项一般不为零：', {'bold': True}), ('(Ku)_{I} = 0，但 (K_{,c}u)_{I} ≠ 0', {'math': True}), ('；二次项因 ', {}), ('K_{,c} ⪰ 0', {'math': True}), (' 非正，交叉项可正可负', {})]),
        ('-', '误差小时线性项占主导：节点力下在六个胞中占线性与二次项之和的 24–72%'),
        ('-', [('导数在离散选择固定的设计区间上成立；', {}), ('激活单元、ghost 面、保留集等任一变化即为离散模型切换', {'bold': True})]),
    ], size=15, space=7)
    cite(s, '论文第 3.3 节，式 (9)–(10)；第 5.4 节；附录 H')

    # ---------------------------------------------------------------- 10 error chain
    s = D.slide('误差链小结：柔度准不等于灵敏度准',
                '误差链有四环。第一，Ritz 恒等式：近似刚度减精确刚度等于 H 转置 A H，半正定，误差二次。第二，装配柔度误差按'
                '各胞能量份额加权，所以份额小的胞可以让柔度看起来很准。第三，灵敏度里有一个一次项，能量误差管不住它。'
                '第四，能量误差分解成 δ 平方乘 κ：约 1% 的位移误差落在刚性方向上，放大成 1.8% 到 35% 的能量误差。'
                '所以判据必须是柔度和灵敏度同时准。')
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
        ('-', '第①环同样适用于以 NᵀKN 形成刚度的学习子结构（边界受限时下界为 LᵀSL），与其细网格复核中柔度偏刚的观察一致'),
        ('-', '第③环此前在学习子结构中没有被量化：本文逐胞报告灵敏度误差'),
    ], size=15, space=4)
    cite(s, '论文第 3 节；附录 B、C、H')

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

    # ---------------------------------------------------------------- method: variational stiffness
    s = D.slide('构造（一）：变分凝聚刚度——必须用完整延拓的转置',
                '构造的第一部分是力学对象。延拓 F 等于校正 𝒲 作用在网络延拓上；凝聚刚度取能量形式 F 转置 K F。'
                '施加这个算子时，必须先算 K F q，再乘完整延拓的转置，包括刚体重构、保留值恢复和校正的转置。'
                '分块写出来第二项是把内部残差的功传回保留自由度，平衡场时为零，否则必须保留，返回的力才是能量的导数。'
                '对称半正定只来自 K，不要求网络里的收集、散布或网格传递对称。')
    eq(s, X0, 1.15, 12.33, 1.0, ['F = 𝒲Ê,   Ŝ = F^{T}KF,   Ŝq = (KFq)_{P} + F_{I}^{T}(KFq)_{I}'], size=22, fill=LBLUE)
    card(s, X0, 2.35, 6.0, 4.25, '算法 1：校正后凝聚刚度的作用', [
        ('-', [('① 延拓：', {'bold': True}), ('网络给出 Ê q；固定保留值，施加前光滑 → 粗网格校正 → 后光滑，得 û = Fq', {})]),
        ('-', [('② 转置：', {'bold': True}), ('计算 y = Kû，返回 Fᵀy：按相反顺序施加校正的转置，再施加学习延拓的转置', {})]),
        ('-', [('③ 装配：', {'bold': True}), ('按式 (3) 装配各胞作用；求解后恢复 F_m B_m Û，用于场和灵敏度', {})]),
    ], size=15)
    card(s, X0 + 6.33, 2.35, 6.0, 4.25, '由此得到的性质（对任何网络输出）', [
        ('-', '对称、半正定：只来自 K = Kᵀ ⪰ 0，不要求学习映射对称'),
        ('-', '零空间只有保留刚体模态（FR_P = R）'),
        ('-', '被精确 Schur 补从下方界住，误差为二次：Ŝ − S = HᵀAH'),
        ('-', '第二项传回内部残差的功：缺了它，返回的力就不是能量的导数'),
    ], fill=LBLUE, head_color=BLUE, size=15)
    cite(s, '论文第 4.1 节，式 (11)–(12)，算法 1；附录 B、E')

    # ---------------------------------------------------------------- method: error spectrum and smoothing
    s = D.slide('构造（二）：网络误差落在哪里——误差谱与 Chebyshev 光滑',
                '校正为什么要两层。把内部误差在 Jacobi 缩放的模态上展开，看误差能量落在哪些模态上。M1 一致面力下，最低 200 个模态'
                '含 24.5% 的误差能量，但只含 4.87% 的精确场能量，而且全部低于光滑区间下端 a 等于 0.173。Chebyshev 光滑在 a 到 b 的区间'
                '上衰减误差，对低于 a 的模态衰减很慢，所以需要互补的粗网格校正。H2 不一样，200 个模态里只有 66 个低于 a，光滑就够了很多。')
    image(s, FIG / 'F03_spectrum.png', X0, 1.05, 6.9, 5.6)
    eq(s, 7.6, 1.15, 5.23, 1.2, ['Av_{j} = λ_{j}Dv_{j},   D = diag(A)', 'd_{I}^{T}Ad_{I} = Σ_{j} λ_{j}c_{j}^{2}'], size=17)
    text(s, 7.6, 2.55, 5.23, 4.05, [
        ('-', [('M1：', {'bold': True}), ('最低 200 个模态含 ', {}), ('24.5%', {'bold': True, 'color': RED}), (' 的误差能量，只含 4.87% 的精确场能量，且全部低于光滑区间下端 a = 0.173', {})]),
        ('-', [('H2：', {'bold': True}), ('200 个模态中只有 66 个低于 a = 0.138', {})]),
        ('-', '光滑区间 [b/30, b]，b 取幂迭代估计的 1.05 倍；谱在 (0, b] 内时光滑在 A 范数下非扩张'),
        ('-', '低于 a 的模态衰减慢 → 交给粗网格校正'),
    ], size=15, space=7)
    cite(s, '论文第 4.2 节，式 (13)；第 5.4 节；图：论文 Fig. 5（基础网络）')

    # ---------------------------------------------------------------- method: coarse correction and ordering
    s = D.slide('构造（三）：粗网格校正与两层循环——排序有保证，幅度靠实测',
                '粗网格校正是在当前内部场加上粗空间的范围上极小化能量，消除的能量正好是 b 转置 A_c 逆 b。主粗空间是 17 的三次方顶点网格上'
                '限制在内部自由度的三线性函数，不动保留值。把粗校正放在两个光滑阶段之间，就是两层循环，也就是方法里的 𝒲。'
                '如果粗求解精确、光滑在 A 范数下不扩张，就有 S 不大于 Ŝ 不大于 Ŝ_net：校正不会让误差变大，柔度向参考解靠近。'
                '排序有保证，减小多少没有保证，要实测；这个排序也不能推广到灵敏度。')
    eq(s, X0, 1.15, 12.33, 1.0, ['û_{I}^{c} = û_{I} − VA_{c}^{−1}V^{T}r_{I},    ‖d_{I}‖^{2}_{A} − ‖d_{I}^{c}‖^{2}_{A} = b_{r}^{T}A_{c}^{−1}b_{r},   A_{c} = V^{T}AV'], size=19)
    eq(s, X0, 2.3, 12.33, 1.0, ['H = Φ_{k}C_{V}Φ_{k}H_{net}    ⟹    S ⪯ Ŝ ⪯ Ŝ_{net}   （粗求解精确，‖Φ_{k}‖_{A} ≤ 1）'], size=20, fill=LBLUE)
    W3 = (12.33 - 2 * 0.3) / 3
    card(s, X0, 3.5, W3, 3.1, '粗空间 Q₁(17)', ['17³ 顶点网格上的三线性向量函数，限制于内部自由度；M1：5,601 个粗自由度对 165,927 个内部自由度', '只作用于内部，保留值不变'], size=14)
    card(s, X0 + W3 + 0.3, 3.5, W3, 3.1, '两层循环 𝒲', ['8 步 Chebyshev → Q₁(17) Galerkin 校正 → 8 步 Chebyshev', '对给定几何是线性、固定的：训练时梯度可穿过它'], size=14)
    card(s, X0 + 2 * (W3 + 0.3), 3.5, W3, 3.1, '保证什么、不保证什么', ['保证：能量误差不增大，柔度向参考解靠近', '不保证：减小幅度（需实测）；灵敏度误差的单调性'], fill=LRED, size=14)
    cite(s, '论文第 4.3 节，式 (14)–(15)；附录 C、D、F.1')

    # ---------------------------------------------------------------- 12 correction
    s = D.slide('构造（三）续：校正效果——网络固定，部署后仍可改进',
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

    # ---------------------------------------------------------------- 11 network
    s = D.slide('构造（四）：几何条件化网络——对保留位移精确线性',
                '网络学习的是内部平衡映射 q 到负 A 逆 K_IP q。先把刚体运动分离出去，网络只延拓变形部分，所以刚体重构和训练精度无关。'
                '几何分支是非线性的：单元矩描述材料分布，节点特征标识保留、切割带、弱支撑和局部刚度，编码后由系数头给出局部相互作用、'
                '网格传递和卷积的权重。位移分支是线性的：把保留位移提升到 32 个通道，用 27 槽模板上的局部相互作用和 65、33、17、9 的'
                'U 形网格层级传播。权重作用在模板、通道和层级上，不对应单个自由度，所以一个约六十万参数的网络服务所有保留集。'
                '最后加回刚体场、恢复保留值，容许性和刚体再现由构造保证。')
    image(s, FIG / 'F11_network_architecture.png', X0, 1.05, 4.3, 5.65)
    eq(s, 5.1, 1.15, 7.73, 0.95, ['Ê q = J_{P}^{T}q + J_{I}^{T}J_{I}[ RC_{R}q + 𝒩_{θ}(η) Π_{P}q ]   ⟹   J_{P}Ê = I_{p},  ÊR_{P} = R'], size=17, fill=LBLUE)
    card(s, 5.1, 2.3, 3.75, 3.2, '几何分支（非线性）', [
        ('-', '单元矩：背景单元内的材料分布'),
        ('-', '节点特征：保留 / 切割带 / 弱支撑 / 局部刚度'),
        ('-', '64 通道编码 + 两轮交换；系数头给出相互作用、网格传递、卷积的权重'),
    ], size=13, head_size=15)
    card(s, 9.08, 2.3, 3.75, 3.2, '位移分支（线性）', [
        ('-', '保留位移提升到 32 个通道'),
        ('-', '27 槽模板上的局部相互作用，4 个加权头'),
        ('-', 'U 形潜在层级 65 / 33 / 17 / 9；弱支撑模板单独处理'),
    ], fill=LBLUE, head_color=BLUE, size=13, head_size=15)
    text(s, 5.1, 5.65, 7.73, 0.95, [
        [('约 6×10⁵ 个参数服务 2,679–45,900 个保留自由度', {'bold': True}),
         ('：权重作用于模板、通道和网格层级，不对应单个自由度；不形成显式矩阵（中位尺寸胞的稠密 Ŝ 需 4.3 GiB）。', {})]],
        size=14, fill=LIGHT, anchor='m', margin=0.15)
    cite(s, '论文第 4.4 节，式 (16)–(17)；图：论文 Fig. 3；附录 G')

    # ---------------------------------------------------------------- method: training
    s = D.slide('构造（五）：训练方向与目标函数——在部署的算子上训练',
                '训练用保留位移方向来考察延拓：多项式和多尺度位移提供空间内容，节点力、一致面力、弹簧支撑和邻胞作用下的响应提供力学方向。'
                '每个方向去掉刚体分量后用参考解归一化。目标函数是预测能量和参考能量之比的对数平均，加一个灵敏度项。'
                '由 Ritz 恒等式，能量项的最小值就对应平衡场。NICE 直接在校正后的延拓上训练，梯度穿过光滑递推和粗校正，'
                '类似求解器在环训练。训练集 591 个几何，基础网络 4.6 小时，NICE 延续训练 3.5 小时。')
    eq(s, X0, 1.15, 12.33, 1.25, ['𝓛(θ) = (1/B) Σ_{j} log( q_{j}^{T}Ŝq_{j} ) + w_{s} (1/|𝒥_{s}|) Σ_{j∈𝒥_{s}} ‖s̃_{j} − s_{j}‖^{2} / ‖s_{j}‖^{2},    q_{j}^{T}Sq_{j} = 1'], size=19, fill=LBLUE)
    card(s, X0, 2.6, 6.0, 3.3, '训练方向', [
        ('-', '规定的多项式位移与多尺度位移'),
        ('-', '平衡节点力、一致面力、弹簧支撑、邻胞作用下的响应'),
        ('-', '去刚体分量，按 qᵀSq = 1 归一化'),
        ('-', '训练中加入能量比较大的方向（刚体补空间上的块搜索）'),
        ('-', '几何增广：立方体 48 种对称'),
    ], size=15)
    card(s, X0 + 6.33, 2.6, 6.0, 3.3, '在部署的算子上训练', [
        ('-', [('能量项：Ritz 原理，最小值对应每个方向上的平衡场', {})]),
        ('-', '灵敏度项：带参考灵敏度标签的方向，w_s = 1'),
        ('-', 'NICE 在 F = 𝒲Ê 上计算损失，梯度穿过光滑与粗校正'),
        ('-', '数据：691 个几何（591 训练 + 100 验证），生成约 42 GPU-h'),
        ('-', '训练：基础网络 4.6 h + NICE 延续 3.5 h（一块 RTX 5090）'),
    ], fill=LBLUE, head_color=BLUE, size=15)
    cite(s, '论文第 4.5 节，式 (18)；第 5.1 节，表 2，表 ST01；附录 G')

    # ---------------------------------------------------------------- method: deployment and lattice solve
    s = D.slide('构造（六）：部署与格栅求解——不形成任何矩阵',
                '部署时，依赖几何的量对每个几何只准备一次：单元矩、网络的几何编码、系数、光滑区间和粗分解。之后每次作用 Ŝ 只需要一次网络前向、'
                '十六步光滑、一次粗求解和对应的转置。延拓和凝聚刚度都不显式形成。格栅层面，在自由保留自由度上用预条件共轭梯度求解，'
                '预条件子是平衡两层的：细层是装配的保留刚度块 K_PP 的分解，每次设计迭代分解一次；粗空间是格栅顶点三线性函数乘六个刚体模态。'
                '灵敏度用恢复的场直接计算。')
    W3 = (12.33 - 2 * 0.45) / 3
    steps = [('每个几何准备一次', ['单元矩、网络几何编码、系数与映射', '光滑区间（幂迭代）、粗分解', '不形成 Ê、Ŝ（中位尺寸胞的稠密 Ŝ 需 4.3 GiB）']),
             ('每次作用 Ŝq', ['一次网络前向（单精度）', '2k 步光滑 + 一次粗求解', '再按相反顺序施加转置']),
             ('格栅求解与灵敏度', ['自由保留自由度上的 PCG（相对残差 10⁻⁶）', '平衡两层预条件：细层 = 𝕂_PP 分解（每次设计迭代一次）；粗空间 = 格栅顶点三线性 × 6 个刚体模态', '恢复各胞场，计算场基灵敏度'])]
    for k, (h, body) in enumerate(steps):
        x = X0 + k * (W3 + 0.45)
        card(s, x, 1.25, W3, 3.45, h, [('-', b) for b in body], fill=LBLUE if k == 2 else LIGHT, head_color=BLUE if k == 2 else RED, size=15)
        if k < 2:
            arrow(s, x + W3 + 0.07, 3.0, 0.31, 0.38)
    text(s, X0, 4.95, 12.33, 1.15, [
        [('内存：', {'bold': True, 'color': RED}), ('学习胞状态每胞 0.25–1.37 GiB（常规精确凝聚的内部 Cholesky 因子 0.60–9.4 GiB）；超出 GPU 预算的胞从 CPU 内存流式读取。', {})],
        [('当前上限：', {'bold': True, 'color': RED}), ('预条件子里 𝕂_PP 的直接分解随装配保留系统增长（第 6.4 节）。', {})],
    ], size=15, fill=LIGHT, anchor='m', margin=0.2, space=4)
    cite(s, '论文第 4.6 节，第 5.8–5.9 节；补充说明 S3、S4.1，表 ST13')

    # ---------------------------------------------------------------- results: validation set and variants
    s = D.slide('验证集、载荷类别与比较的变体',
                '先交代实验设置。80 个验证几何，未切割、轻度、中度、重度切割各 20 个。每个几何用九类保留位移方向考察，'
                '一致面力每个几何 64 个方向，是主要载荷类别；节点力还会加载弱支撑节点，作为压力测试。比较四个变体：基础网络不带校正；'
                '在它基础上延续训练 15000 步得到三个变体：训练里带完整校正的 NICE、只带八步光滑的平滑训练、不带校正的未校正延续。'
                '另外还评估基础网络部署时直接加校正、不重训。')
    table(s, X0, 1.15, 7.2, [
        ['变体', '训练集', '训练中的校正', '评估时的校正'],
        [[('NICE', {'bold': True, 'color': BLUE})], '591', '8 / Q₁(17) / 8', '8 / Q₁(17) / 8'],
        ['平滑训练', '591', '8 步光滑', '8 步光滑'],
        ['未校正（延续）', '591', '无', '无'],
        ['基础网络', '305', '无', '无'],
        ['基础网络（加校正）', '305', '无', '8 / Q₁(17) / 8'],
    ], (2.2, 1.0, 2.0, 2.0), size=14, row_h=0.55, hl_rows=(1,))
    text(s, X0, 4.6, 7.2, 2.0, [
        ('-', '三个延续训练都从基础网络出发，再训练 15,000 步，几何与顺序相同'),
        ('-', '80 个验证几何中 20 个用于检查点选择；另报告其余 60 个的统计'),
    ], size=15, space=6)
    card(s, 8.0, 1.15, 4.83, 4.6, '80 个验证几何', [
        ('-', '未切割、轻度（保留 >2/3）、中度（1/3–2/3）、重度（<1/3）切割各 20 个'),
        ('-', '厚度场：均匀、仿射或混合三线性；角点参数 0.176–0.698'),
        ('-', '九类保留位移方向：多项式、多尺度、节点力、一致面力（主类，每几何 64 个）、弹簧支撑、邻胞诱导……'),
        ('-', '节点力会加载弱支撑节点：稳定化问题的压力测试'),
    ], size=14)
    cite(s, '论文第 5.1 节，表 2，表 ST01–ST03')

    # ---------------------------------------------------------------- results: verification
    s = D.slide('参考解与部署算子的验证',
                '两类验证。参考解：n 等于 32 在七个胞上加密验证，未切割和中度切割胞柔度变化不超过 0.14%、灵敏度 0.23%；'
                '重度切割的 H2 到约 1% 和 3.6%。ghost 罚系数和积分加密的影响不超过 0.12%。部署算子：对称性 9 乘 10 的负 9 次方，'
                '功与能量一致到 5 乘 10 的负 9 次方，刚体模态能量 2 乘 10 的负 11 次方，这些都是构造性质在浮点下的体现。')
    card(s, X0, 1.2, 6.0, 3.6, '离散参考解（n = 32）', [
        ('-', '七个胞加密：U1 → n = 40；M1、M2 → n = 48；H1、H2 等 → n = 64'),
        ('-', '未切割、中度切割：柔度变化 ≤0.14%，灵敏度 ≤0.23%'),
        ('-', [('重度切割 H2：柔度约 1%，灵敏度约 ', {}), ('3.6%', {'bold': True, 'color': RED})]),
        ('-', 'ghost 罚系数 10⁻⁵–10⁻³、体积积分加密：变化 ≤0.12%'),
        ('-', '所有误差都相对于这个离散参考解'),
    ], size=15)
    table(s, X0 + 6.33, 1.2, 6.0, [
        ['部署算子检查（U2、M1、M2、H1、H2）', '一致面力', '节点力'],
        [[('双线性形式对称性 ', {}), ('|q_{i}^{T}Ŝq_{j} − q_{j}^{T}Ŝq_{i}|', {'math': True})], '9×10⁻⁹', '≤3×10⁻⁸'],
        ['返回功与恢复场能量之差', '5×10⁻⁹', '≤2×10⁻⁸'],
        ['刚体模态能量 / 典型变形能', '2×10⁻¹¹', '≤1×10⁻¹⁰'],
        ['部署算子 vs 训练时场', '1.1×10⁻⁷', '—'],
    ], (3.2, 1.4, 1.4), size=14, row_h=0.62)
    text(s, X0 + 6.33, 4.5, 6.0, 1.0, [[('构造性质在浮点算术下得到保持（网络单精度、刚度作用与能量双精度）', {'color': GRAY})]], size=14)
    cite(s, '论文第 5.2 节；表 ST04、ST10；补充说明 S1；图 S01')

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

    # ---------------------------------------------------------------- results: error components
    s = D.slide('误差的组成——落在切割带旁，并被刚性方向放大',
                '误差的组成。空间上，M1 的精确场只有 8% 的能量在离切割面两个单元以内，但基础网络误差能量的 39% 到 43% 在这一层，'
                '紧邻保留的切割带。NICE 把总误差能量降低 113 到 138 倍，这一层的份额减半到 19% 到 22%。放大机制：能量误差恰好等于'
                'δ 平方乘 κ，δ 是加权位移误差，κ 是误差和精确场的 Rayleigh 商之比。基础网络位移误差只有百分之一左右，但 κ 有 124 到 7089，'
                '所以能量误差到 1.8% 到 35%。')
    image(s, FIG / 'F12_field_error_M1.png', X0, 1.05, 6.4, 5.65)
    card(s, 7.15, 1.15, 5.68, 2.45, '空间分布（M1，一致面力）', [
        ('-', '精确场：距切割面两单元内的层只含 8% 的能量（占单元 13%）'),
        ('-', '基础网络误差能量：39–43% 在这一层'),
        ('-', 'NICE：总误差能量降低 113–138 倍，该层份额减半至 19–22%'),
    ], size=14)
    eq(s, 7.15, 3.8, 5.68, 0.75, ['ε = δ^{2}κ'], size=22, fill=LBLUE)
    text(s, 7.15, 4.65, 5.68, 1.95, [
        ('-', 'δ：Jacobi 加权相对位移误差；κ：误差与精确场 Rayleigh 商之比'),
        ('-', [('基础网络（M1、M2、H1、H2）：δ = 0.7–1.3%，κ = 124–7089 → ε = ', {}), ('1.8–35%', {'bold': True, 'color': RED})]),
        ('-', '小位移误差落在刚性方向上 → 大能量误差'),
    ], size=14, space=5)
    cite(s, '论文第 5.4 节；附录 B.3；表 ST04；图：论文 Fig. 6')

    # ---------------------------------------------------------------- results: two-cell setup
    s = D.slide('两胞装配——学习目标胞 + 精确邻胞',
                '装配测试的设置：一个学习目标胞和一个精确、未切割的邻胞，厚度在界面上连续。两种配置 x 和 y，每种六个面载荷：'
                '三个方向的一致面力分别施加在目标胞面和邻胞面上。比较柔度和每个胞的八分量灵敏度向量，用 3% 作为共同参考线。'
                '选择集七个目标胞，再加九个选择之外的胞作独立检验：五个单胞误差最大的和每层随机一个。')
    image(s, FIG / 'F09_assembly_loads.png', X0, 1.05, 12.33, 3.4)
    W3 = (12.33 - 2 * 0.3) / 3
    card(s, X0, 4.6, W3, 1.95, '目标', ['选择集：U1、U2、L1、M1、M2、H1、H3', '留出：9 个胞（5 个误差最大 + 每层 1 个）'], size=14)
    card(s, X0 + W3 + 0.3, 4.6, W3, 1.95, '载荷', ['每配置 6 个面载荷：3 个方向 × 目标面 / 邻胞面', '另测 3 个切割面载荷'], size=14)
    card(s, X0 + 2 * (W3 + 0.3), 4.6, W3, 1.95, '度量', ['柔度误差；每胞 8 分量灵敏度向量误差', '共同参考线 3%；取 6 个载荷和两胞的最大值'], size=14)
    cite(s, '论文第 5.6 节；图：论文 Fig. 8；补充说明 S4.2')

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
                '这一页回答为什么要保留完整边界。用精确胞算子，只把盒面位移限制成 r 次 Bernstein 多项式：r 等于 1 时柔度误差'
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

    # ---------------------------------------------------------------- fine-grid checks
    s = D.slide('细网格复核：每一层都对照了全分辨率的离散解',
                '这一页回答"有没有做细网格复核"。做了，而且口径比只看柔度更严。我们的参考解就是全分辨率的切割有限元离散解：'
                '精确凝聚和整体直接求解在数学上是同一个解，实测柔度一致到 7 乘 10 的负 10 次方。按层级：单胞对照精确 Schur 补；'
                '两胞对照精确凝聚；四胞格栅直接对照整体格栅直接解，也就是把所有内部和保留自由度、近一百万个自由度一起求解，差 0.022% 以内；'
                '八胞对照装配的精确凝聚；优化算例 A 整条路径用精确凝聚孪生跑，板在初始和最终设计处做精确复核。离散本身也做了 n 等于 32 到 64 的加密。'
                '唯一没有复核的是 24 胞以上的规模演示，这在局限里写明了。')
    table(s, X0, 1.15, 12.33, [
        ['层级', '参考解（全分辨率离散解）', '复核的量', '结果'],
        ['单胞（80 个验证几何）', '同一 n = 32 网格上的精确 Schur 补', '方向能量误差', '平均 0.074%'],
        ['两胞装配（32 个配置）', '精确凝聚（目标胞与邻胞）', '柔度 / 每胞 8 分量灵敏度', '≤0.28% / ≤1.5%'],
        [[('四胞格栅（2 个）', {'bold': True})], [('整体格栅直接求解：全部内部 + 保留自由度（88.5 万–95.8 万）一起求解', {'bold': True})], '柔度', [('≤0.022%', {'bold': True, 'color': BLUE})]],
        ['八胞格栅（2 个）', '装配精确凝聚（与整体直接解一致到 7×10⁻¹⁰）', '柔度 / 灵敏度 / 保留解', '≤0.015% / ≤0.14% / ≤0.13%'],
        ['优化算例 A', '精确凝聚孪生优化；迭代 0、12、23 复核', '柔度 / 梯度 / 最终设计', '≤0.028% / ≤0.33% / 角点差 ≤0.0051'],
        ['切割固支板（24 胞）', '三个设计的精确凝聚复核（表 ST21）', '柔度 / 梯度', '≤0.030% / ≤0.30%'],
        ['离散本身', 'n = 32 → 40 / 48 / 64 加密（7 个胞）', '柔度 / 灵敏度', '除 H2 外 ≤0.28% / ≤0.44%'],
    ], (2.4, 4.6, 2.4, 2.9), size=13, row_h=0.56, hl_rows=(3,))
    text(s, X0, 5.85, 12.33, 0.7, [[('说明：', {'bold': True, 'color': RED}),
                                    ('精确凝聚与整体直接求解是同一个离散解；未复核的只有 24 胞以上的规模演示（第 6.4 节已写明）。', {})]],
         size=15, fill=LIGHT, anchor='m', margin=0.2)
    cite(s, '论文第 5.2、5.3、5.6、5.8、5.9、5.10 节；Table 5，Table ST10、ST17、ST21')

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
    s = D.slide('加速比要和精度、基准一起看',
                '审稿人可能拿 PIML 的加速比来比，所以把加速比、精度和基准放在一张表里。各工作的精度层级和基准不同，加速比不能直接比较；'
                '我们 10 倍是对 16 线程 PARDISO，单线程口径 57 到 77 倍，精度在 0.01% 量级。正文不做这个比较，被问到时按此表陈述事实。')
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
    text(s, X0, 5.85, 12.33, 0.7, [[('用途：', {'bold': True, 'color': BLUE}),
                                    ('精度层级与基准不同，加速比不宜直接比较；正文不做此比较，审稿人问到时按此表陈述事实（RESPONSE_PREP_SPEEDUP_CN.md）。', {})]],
         size=15, fill=LIGHT, anchor='m', margin=0.2)
    cite(s, '来源：GUO_SERIES_COMPARISON_CN.md 第 2 节；本文 Table 5、ST12b')

    # ---------------------------------------------------------------- results: optimisation setup
    s = D.slide('厚度优化的设置',
                '优化设置。设计变量是格栅顶点处的角点厚度参数，映射到每个胞的八个角点。目标是单位一致面力下柔度最小，体积不超过初始的 80%。'
                '加载面所在平面上的顶点参数固定，使节点载荷与设计无关。上下界 0.18 到 0.69，每个胞的角点跨度和梯度范数不超过 0.45，'
                '保证每个胞都在训练范围内。每次设计迭代重新生成每个胞的几何和学习子结构，从上一步的解开始 PCG，走一步 MMA，'
                '移动限 5%，不做线搜索。停止判据是目标相对变化连续三次低于 10 的负 4 次方，不用 KKT 残差，因为离散模型在迭代间会切换。')
    eq(s, X0, 1.15, 12.33, 1.05, ['min  C(τ_{g}) = f_{g}^{T}U    s.t.   V(τ_{g}) ≤ V^{*} = 0.8 V(τ^{0}),   0.18 ≤ τ ≤ 0.69'], size=20, fill=LBLUE)
    card(s, X0, 2.4, 6.0, 3.2, '变量与约束', [
        ('-', [('变量：格栅顶点处的角点厚度参数 ', {}), ('τ_{g}', {'math': True}), ('，映射到各胞角点 ', {}), ('τ_{m}(τ_{g})', {'math': True})]),
        ('-', '加载面平面上的顶点参数固定（算例 A 27 个中 9 个，板 74 个中 10 个）→ 载荷与设计无关'),
        ('-', '每胞角点跨度、梯度范数 ≤0.45：保持在训练范围内（[0.175, 0.699]，≤0.47）'),
        ('-', '梯度：各胞场基灵敏度按顶点求和'),
    ], size=15)
    card(s, X0 + 6.33, 2.4, 6.0, 3.2, '每次设计迭代', [
        ('-', '重新生成每个胞的几何与学习子结构'),
        ('-', '从上一步的解开始 PCG 求解格栅'),
        ('-', 'MMA（Svanberg 2007 变体）一步，移动限 5%，无线搜索'),
        ('-', '停止：目标相对变化连续 3 次 < 10⁻⁴；不用 KKT 残差（离散模型在迭代间切换）'),
    ], fill=LBLUE, head_color=BLUE, size=15)
    cite(s, '论文第 5.10 节；表 ST16；补充说明 S6.1')

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
    text(s, X0, 2.6, 12.33, 1.8, [['B1  映射单胞零样本测试（后续工作前期结果）'], ['B2  与 PIML 系列的逐项对照']], size=20, color=GRAY, align='c', space=10)

    # ---------------------------------------------------------------- B3
    s = D.slide('B1  映射单胞零样本：结构始终成立，拉伸是唯一明显失效',
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
    s = D.slide('B2  与 PIML 系列的逐项对照',
                '与 PIML 系列各篇在边界表示、训练、灵敏度、理论、校正上的逐项对照，供追问时查。')
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
