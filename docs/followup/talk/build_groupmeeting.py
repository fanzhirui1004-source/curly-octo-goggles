"""Build the group-meeting deck (P1, NICE) on the group's PowerPoint template.

Usage: python3 build_groupmeeting.py <template.pptx> <out.pptx>
The template is the group template as carried by the advisor deck (P1_NICE_导师汇报_20261008.pptx): its first slide
(title page) is kept and edited, its other slides are removed, and every new slide uses the template's single layout.
Helpers (text boxes, cards, stat callouts, tables, pictures) come from build_talk.py. Display formulas are typeset with
LaTeX (Latin Modern, as in the paper) and inserted as pictures at their natural size. Numbers follow the R7 manuscript
(docs/paper_p1/MANUSCRIPT_CN.md); figures come from docs/paper_p1/figures. Speaker notes are written on every slide."""
import sys, hashlib, subprocess, tempfile
from pathlib import Path
from PIL import Image
from pptx.util import Inches, Pt
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_LABEL_POSITION

sys.path.insert(0, str(Path(__file__).resolve().parent))
import re  # noqa: E402
import build_talk  # noqa: E402
from build_talk import (rgb, font, text, card, stat, image, table, cite, Deck,  # noqa: E402
                        RED, DARK, GRAY, BLUE, LIGHT, LBLUE, X0)


def para_runs(p, spec, size, color, bold):
    """As build_talk.para_runs, but a run with 'ss' also gets _{..} / ^{..} as sub- and superscripts in the body font."""
    if isinstance(spec, str):
        spec = [spec]
    for item in spec:
        txt, o = (item, {}) if isinstance(item, str) else item
        pieces = re.split(r'(_\{[^}]*\}|\^\{[^}]*\})', txt) if (o.get('math') or o.get('ss')) else [txt]
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


build_talk.para_runs = para_runs


def ss(s, **kw):
    return (s, dict(ss=True, **kw))


# ---------------------------------------------------------------------------------------------------------------
# Visual system (flat): the template supplies the red title and the footer; content uses one ink colour, one navy
# for numbers and table heads, the template red only for small heads, bullets and the lead word of a takeaway, and
# flat light panels without shadows.
from pptx.oxml.ns import qn  # noqa: E402
from lxml import etree  # noqa: E402

NAVY, INK, MUTED = '1F3864', '262626', '6B7280'
PANEL, PANEL_B, PANEL_R, RULE = 'F4F6F9', 'EDF2F8', 'FBF1F1', 'D9DDE3'
BLUE = RED                    # lead words of takeaways and secondary heads use the template red
LIGHT, LBLUE = PANEL, PANEL_B
GRAY = MUTED


def flat(shape, fill):
    """Solid fill, no outline, no theme style (hence no shadow)."""
    st = shape._element.find(qn('p:style'))
    if st is not None:
        shape._element.remove(st)
    shape.fill.solid(); shape.fill.fore_color.rgb = rgb(fill); shape.line.fill.background()
    return shape


def box(slide, x, y, w, h, fill=PANEL, radius=0.035):
    b = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    b.adjustments[0] = radius
    return flat(b, fill)


def my_bullet(p, level=0, char='•'):
    pPr = p._p.get_or_add_pPr()
    pPr.set('marL', str(int(Inches(0.24 + 0.24 * level))))
    pPr.set('indent', str(int(-Inches(0.22))))
    for tag in ('a:buClr', 'a:buSzPct', 'a:buFont', 'a:buNone', 'a:buChar', 'a:buAutoNum'):
        el = pPr.find(qn(tag))
        if el is not None:
            pPr.remove(el)
    bc = etree.SubElement(pPr, qn('a:buClr')); etree.SubElement(bc, qn('a:srgbClr')).set('val', RED if level == 0 else MUTED)
    etree.SubElement(pPr, qn('a:buSzPct')).set('val', '70000')
    etree.SubElement(pPr, qn('a:buFont')).set('typeface', 'Arial')
    etree.SubElement(pPr, qn('a:buChar')).set('char', '■' if level == 0 else '–')


build_talk.bullet = my_bullet
_text = build_talk.text


def text(slide, x, y, w, h, paras, size=16, color=INK, line=1.12, fill=None, margin=0.06, **kw):
    if fill:
        box(slide, x, y, w, h, fill)
        fill = None
        margin = max(margin, 0.18)
    return _text(slide, x, y, w, h, paras, size=size, color=color, line=line, fill=fill, margin=margin, **kw)


def card(slide, x, y, w, h, head, body, fill=PANEL, head_color=RED, size=15, head_size=17, name=None):
    box(slide, x, y, w, h, PANEL if fill in (PANEL, 'F2F2F2') else (PANEL_B if fill in (PANEL_B, 'EAF2FB') else fill))
    text(slide, x + 0.2, y + 0.14, w - 0.4, 0.45, [head], size=head_size, color=head_color if head_color != '0070C0' else RED,
         bold=True)
    if body:
        text(slide, x + 0.2, y + 0.62, w - 0.4, h - 0.75, body, size=size, space=6)


def stat(slide, x, y, w, number, label, color=NAVY, nsize=34, lsize=13, fill=PANEL, h=1.45):
    if fill:
        box(slide, x, y, w, h, PANEL)
    text(slide, x + 0.1, y + 0.12, w - 0.2, 0.7, [number], size=nsize, color=NAVY if color == RED else color, bold=True,
         align='c', anchor='m')
    text(slide, x + 0.2, y + 0.84, w - 0.4, h - 0.9, [label], size=lsize, color=MUTED, align='c')


def banner(slide, y, runs, h=0.75, fill=None, size=16):
    box(slide, X0, y, W, h, PANEL_R)
    return _text(slide, X0 + 0.2, y, W - 0.4, h, [runs], size=size, color=INK, anchor='m', margin=0.02, line=1.12)


def cite(slide, s):
    _text(slide, X0, 6.8, W, 0.3, [s], size=10, color=MUTED, margin=0.0)


def _cell_border(cell, sides, color=RULE, w=6350):
    tcPr = cell._tc.get_or_add_tcPr()
    for side in ('a:lnL', 'a:lnR', 'a:lnT', 'a:lnB'):
        old = tcPr.find(qn(side))
        if old is not None:
            tcPr.remove(old)
    for side in ('a:lnL', 'a:lnR', 'a:lnT', 'a:lnB'):
        ln = etree.SubElement(tcPr, qn(side))
        if side in sides:
            ln.set('w', str(w)); sf = etree.SubElement(ln, qn('a:solidFill'))
            etree.SubElement(sf, qn('a:srgbClr')).set('val', color)
        else:
            ln.set('w', '0'); etree.SubElement(ln, qn('a:noFill'))
    # schema order: borders come before the cell fill
    fill = tcPr.find(qn('a:solidFill'))
    if fill is not None:
        tcPr.remove(fill); tcPr.append(fill)


def table(slide, x, y, w, rows, widths, size=13, head_fill=NAVY, row_h=0.4, hl_rows=(), hl_fill=PANEL_R, name=None):
    nr, nc = len(rows), len(rows[0])
    gt = slide.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(w), Inches(row_h * nr))
    tbl = gt.table
    tblPr = tbl._tbl.tblPr
    for k in ('bandRow', 'firstRow'):
        tblPr.set(k, '0')
    st = tblPr.find(qn('a:tableStyleId'))
    if st is not None:
        st.text = '{2D5ABB26-0587-4C30-8999-92F81FD0307C}'       # "No Style, No Grid"
    for j, fw in enumerate(widths):
        tbl.columns[j].width = Inches(w * fw / sum(widths))
    for i, row in enumerate(rows):
        tbl.rows[i].height = Inches(row_h)
        for j, val in enumerate(row):
            c = tbl.cell(i, j)
            c.margin_left = c.margin_right = Inches(0.08); c.margin_top = c.margin_bottom = Inches(0.03)
            c.vertical_anchor = build_talk.MSO_ANCHOR.MIDDLE
            p = c.text_frame.paragraphs[0]; c.text_frame.word_wrap = True
            head = i == 0
            para_runs(p, val, size, 'FFFFFF' if head else INK, head)
            c.fill.solid()
            c.fill.fore_color.rgb = rgb(head_fill if head else (hl_fill if i in hl_rows else 'FFFFFF'))
            _cell_border(c, () if head else ('a:lnB',))
    return gt

FIG = Path(__file__).resolve().parents[2] / 'paper_p1' / 'figures'
W = 12.33                                   # content width (inches)
EQ_DIR = Path(tempfile.mkdtemp(prefix='gm_eq_'))
DPI = 600

PREAMBLE = r"""\documentclass[border=2pt]{standalone}
\usepackage{lmodern}
\usepackage{amsmath,amssymb,bm}
\usepackage[HTML]{xcolor}
\begin{document}
"""


def eq_png(tex, size):
    """Typeset one display formula at `size` pt; return (png path, width in, height in)."""
    key = hashlib.md5(f'{size}|{tex}'.encode()).hexdigest()[:12]
    png = EQ_DIR / f'{key}.png'
    if not png.exists():
        src = EQ_DIR / f'{key}.tex'
        src.write_text(PREAMBLE + r'\fontsize{%g}{%g}\selectfont\color[HTML]{262626}$\displaystyle %s$' % (size, 1.2 * size, tex)
                       + '\n\\end{document}\n', encoding='utf-8')
        r = subprocess.run(['lualatex', '-interaction=nonstopmode', '-halt-on-error', src.name], cwd=EQ_DIR,
                           capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError('LaTeX failed for: ' + tex + '\n' + r.stdout[-1500:])
        subprocess.run(['pdftocairo', '-png', '-transp', '-r', str(DPI), '-singlefile', f'{key}.pdf', key], cwd=EQ_DIR,
                       check=True)
    w, h = Image.open(png).size
    return png, w / DPI, h / DPI


def formula(slide, tex, x, y, w, h, size=22, fill=LIGHT, align='c'):
    """A rounded box (x, y, w, h) holding one formula at its natural size, centred vertically."""
    if fill:
        box(slide, x, y, w, h, fill)
    sz = size
    png, pw, ph = eq_png(tex, sz)
    while (pw > w - 0.2 or ph > h - 0.08) and sz > 16:
        sz -= 1
        png, pw, ph = eq_png(tex, sz)
    if pw > w - 0.2 or ph > h - 0.08:
        raise ValueError(f'formula too large for its box ({pw:.2f} x {ph:.2f} in): {tex}')
    if sz != size:
        print(f'  formula shrunk {size} -> {sz} pt: {tex[:60]}')
    px = x + (w - pw) / 2 if align == 'c' else x + 0.2
    slide.shapes.add_picture(str(png), Inches(px), Inches(y + (h - ph) / 2), Inches(pw), Inches(ph))


def label(head, body, color=RED):
    return [(head, {'bold': True, 'color': color})] + (list(body) if isinstance(body, list) else [(body, {})])


def m(s):
    return (s, {'math': True})


def build(template, out):
    D = Deck(template)

    # ------------------------------------------------------------------ 1 title page (template slide, edited)
    s1 = D.prs.slides[0]
    for sh in s1.shapes:
        if not sh.has_text_frame:
            continue
        t = sh.text_frame.text
        if t.startswith('P1 进展汇报') or t.startswith('组会汇报'):
            r = sh.text_frame.paragraphs[0].runs
            r[0].text = '带平衡校正的神经初始化静力凝聚（NICE）'
            for extra in r[1:]:
                extra.text = ''
            for rr in r:
                rr.font.size = Pt(34)
        elif t.startswith('切割薄壁'):
            r = sh.text_frame.paragraphs[0].runs
            r[0].text = '及其在切割 TPMS 点阵厚度设计中的应用　·　组会汇报　·　2026 年 10 月'
            for extra in r[1:]:
                extra.text = ''
    s1.notes_slide.notes_text_frame.text = (
        '各位老师、同学好。今天汇报 P1 的工作：带平衡校正的神经初始化静力凝聚，简称 NICE，以及这一方法在切割 TPMS 点阵厚度设计中的应用。'
        '先讲问题和思路，再讲方法和误差关系，然后是数值算例，最后是局限和后续工作。')

    # ------------------------------------------------------------------ 2 one-page summary
    s = D.slide('一页概要：保留全部主自由度，只近似内部位移',
                '先给结论。对象是按零件外形切割的 TPMS 点阵的厚度设计。NICE 保留胞元边界面和切割平面单元上的全部自由度，'
                '只近似内部位移：网络由主自由度位移给出内部位移，固定的两重网格修正减小内部不平衡力，凝聚刚度矩阵由修正后位移场的应变能计算。'
                '这样得到的凝聚刚度矩阵对任意网络参数都对称半正定、不低于精确值，误差是内部位移误差的二阶小量。'
                '误差进入柔度和灵敏度的途径不同，所以两者分别检验。右边四个数：单胞元平均能量误差 0.074%；两个八胞元点阵柔度误差低于 0.015%；'
                '含灵敏度的八胞元分析在一块 GPU 上比 16 线程直接求解快约十倍；切割面固支板上均匀化低估初始设计柔度 27%，NICE 最终设计的柔度偏差 0.03%。')
    for k, (h, b) in enumerate([
            ('对象', '按零件外形切割的 TPMS 点阵；厚度设计既要细尺度精度，又要每次迭代快速分析'),
            ('方法', '保留胞元边界面与切割平面单元上的全部自由度；网络给出内部位移，两重网格修正减小内部不平衡力，刚度由应变能计算'),
            ('性质', '对任意网络参数：对称半正定、只有刚体零能模态、不低于精确值；误差为二阶小量'),
            ('误差', '柔度误差按应变能占比加权；灵敏度误差含一阶项，两者分别检验')]):
        y = 1.25 + k * 1.3
        text(s, X0, y, 1.0, 0.5, [h], size=17, color=RED, bold=True)
        text(s, X0 + 0.95, y, 5.2, 1.25, [b], size=16)
    stat(s, 6.85, 1.2, 2.9, '0.074%', '80 个验证几何上的平均能量误差（面力载荷）', fill=LIGHT, h=1.95, nsize=38)
    stat(s, 9.93, 1.2, 2.9, '<0.015%', '两个八胞元点阵的柔度误差；梯度误差在 0.09% 以内', fill=LIGHT, h=1.95, nsize=38)
    stat(s, 6.85, 3.35, 2.9, '≈10×', '八胞元点阵分析：一块 GPU 对 16 线程直接求解', fill=LIGHT, h=1.95, nsize=38)
    stat(s, 9.93, 3.35, 2.9, '27% / 0.03%', '切割面固支板：均匀化的低估 / NICE 的偏差', fill=LIGHT, h=1.95,
         nsize=30)
    text(s, 6.85, 5.55, 5.98, 1.1, [[('数值均相对同一离散模型的精确静力凝聚；', {}), ('论文：', {'bold': True}),
                                      ('正文与补充材料已完成本轮修改，目标期刊 CMAME', {})]], size=14, color=GRAY)

    # ------------------------------------------------------------------ 3 agenda
    s = D.slide('汇报提纲',
                '汇报分六部分：背景与问题；切割胞元与静力凝聚；NICE 方法；误差关系；数值算例；讨论与结论。方法和算例部分讲得细一些。')
    parts = [('1', '背景与问题', '切割点阵的厚度设计；三类方法；学习组件模型'),
             ('2', '切割胞元与静力凝聚', '几何与离散；主自由度与精确静力凝聚'),
             ('3', 'NICE 方法', '应变能计算刚度；网络；两重网格修正；训练'),
             ('4', '误差关系', '柔度误差加权；灵敏度误差含一阶项'),
             ('5', '数值算例', '胞元、装配、消融、点阵、成本、设计、规模'),
             ('6', '讨论与结论', '适用范围与局限；后续工作')]
    for k, (n, h, b) in enumerate(parts):
        col, row = k % 2, k // 2
        x, y = X0 + 0.3 + col * 6.2, 1.45 + row * 1.7
        text(s, x, y - 0.08, 1.1, 0.9, ['0' + n], size=40, color=RED, bold=True)
        text(s, x + 1.2, y, 4.7, 0.5, [h], size=20, color=INK, bold=True)
        text(s, x + 1.2, y + 0.55, 4.7, 0.9, [b], size=15, color=MUTED)

    # ------------------------------------------------------------------ 4 object
    s = D.slide('研究对象：按零件外形切割、在切割处固支的点阵',
                '研究对象是按零件外形切割的 TPMS 点阵。左图是 8×4 个 Schwarz-P 胞元经平面切割后余下的 24 个胞元，其中 8 个被切割，'
                '支撑就作用在切割平面单元上。右图是支撑旁的一个切割胞元：蓝点是胞元边界面上的主自由度节点，棕点是切割平面单元的节点，'
                '72,631 个节点中保留 9,674 个，其余为内部节点，由静力凝聚消去。厚度设计要在每次迭代中重新分析全部胞元，'
                '而驱动设计的往往是切割胞元的局部刚度和厚度灵敏度。')
    image(s, FIG / 'F00_problem.png', X0, 1.05, W, 4.5)
    Wc = (W - 2 * 0.3) / 3
    for k, (h, b) in enumerate([('厚度梯度', '各胞元壁厚由角点参数控制'),
                                ('切割胞元', '支撑和载荷作用在切割平面单元上'),
                                ('设计迭代', '每次迭代都要重新分析全部胞元')]):
        card(s, X0 + k * (Wc + 0.3), 5.65, Wc, 1.05, h, [b], size=15, head_size=16)
    cite(s, '图：论文图 1')

    # ------------------------------------------------------------------ 5 three routes
    s = D.slide('三类分析方法各有瓶颈',
                '分析方法有三类。细尺度有限元离散全部胞元，精度有保证，但每次迭代都要重新组装和求解，八胞元点阵已有一两百万个自由度，'
                '16 线程直接求解要 14 到 17 分钟。均匀化代价低，但周期性边界条件的假设在不满足尺度分离或胞元被切割时不成立，'
                '后面的切割面固支板上，均匀化把初始设计的柔度低估了 27%。静力凝聚消去内部自由度、不引入近似，'
                '但常规实现要对每个胞元分解内部刚度矩阵并形成稠密的凝聚刚度矩阵，设计迭代改变全部胞元，这些分解要全部重做。'
                '本文保留静力凝聚的框架，只把内部位移的求解换成网络加固定修正。')
    cards = [
        ('细尺度有限元模型', ['离散全部胞元，精度有保证', '每次设计迭代都需重新组装和求解',
                          '八胞元点阵已有 183–211 万个自由度'], '14–17 min', '八胞元点阵一次直接求解（16 线程）'),
        ('均匀化方法', ['以等效材料代替胞元，计算代价低', '周期性边界条件假设在不满足尺度分离或胞元被切割时不再成立'],
         '27%', '切割面固支板上低估初始设计柔度'),
        ('精确静力凝聚', ['消去内部自由度，不引入近似', '常规实现须分解每个胞元的内部刚度矩阵，并形成稠密凝聚刚度矩阵',
                     '每次设计迭代全部重做'], '0.60–9.4 GiB', '单个胞元的 Cholesky 因子；八胞元点阵一次分析 16–20 min'),
    ]
    for k, (h, body, num, lab) in enumerate(cards):
        x = X0 + k * (Wc + 0.3)
        card(s, x, 1.25, Wc, 3.95, h, [('-', b) for b in body], size=15)
        text(s, x + 0.1, 3.75, Wc - 0.2, 0.62, [num], size=28, color=RED, bold=True, align='c', anchor='m')
        text(s, x + 0.15, 4.38, Wc - 0.3, 0.75, [lab], size=13, color=GRAY, align='c')
    banner(s, 5.45, [('本文的思路：', {'bold': True, 'color': BLUE}),
                     ('保留静力凝聚的框架和全部主自由度，用网络加固定修正近似内部位移，免去内部刚度矩阵的分解。', {})], h=0.9)
    cite(s, '数据：论文第 1、5.9、5.10 节；表 5；补充表 ST13')

    # ------------------------------------------------------------------ 6 related work and the cut-cell requirement
    s = D.slide('学习组件模型与切割胞元的要求',
                '学习组件模型用神经网络给出单元或子结构的刚度关系。Capuano 和 Rimoli 直接学习节点位移到节点力的映射；'
                '学习子结构方法预测数值形函数，按 N 转置 K N 计算子结构刚度矩阵，已经用于复杂三维设计域和点阵优化。'
                '这类方法用少量边界自由度描述子结构之间传递的位移，装配模型规模小，边界误差通过细化划分、提高插值阶次或过采样来控制。'
                '切割胞元带来一个具体要求：切割面上的支撑和载荷作用在切割平面单元的全部节点上，这些自由度都要保留，'
                '于是主自由度集合的规模和构成随几何变化。本文研究的就是这种情形下怎样近似静力凝聚，同时保持静力凝聚的性质。')
    card(s, X0, 1.2, 7.2, 4.1, '学习组件模型', [
        ('-', 'Capuano & Rimoli (2019)：网络直接给出单元节点位移到节点力的映射'),
        ('-', [('学习子结构（Huang et al. 2022–2024；Guo et al. 2026a,b 等）：预测数值形函数，按 ', {}), m('N^{T}KN'),
               (' 计算子结构刚度矩阵，已用于复杂三维设计域和点阵优化', {})]),
        ('-', '以少量边界自由度描述子结构间传递的位移，装配模型规模小；边界误差通过细化划分、提高插值阶次或过采样控制'),
        ('-', '端口约化（Eftang & Patera 2013 等）同样缩减组件间共享的自由度'),
    ], size=15)
    image(s, FIG / 'F00_problem.png', 7.95, 1.1, 4.88, 3.15, crop=(0.635, 0.03, 0.0, 0.10))
    text(s, 7.95, 4.25, 4.88, 1.25, [
        ('-', '切割面上的支撑和载荷作用于切割平面单元的全部节点，这些自由度须全部保留'),
        ('-', '主自由度集合的规模与构成随几何变化：2,679–45,900 个'),
    ], size=15, space=4)
    banner(s, 5.55, [('本文的问题：', {'bold': True, 'color': BLUE}),
                     ('主自由度集合随切割变化时，如何近似静力凝聚、保持其性质，并可靠地给出柔度和厚度灵敏度。', {})], h=0.9)
    cite(s, '论文第 1、1.1 节；右图：论文图 1(b)，支撑旁的切割胞元及其主自由度节点')

    # ------------------------------------------------------------------ 7 discretisation
    s = D.slide('切割胞元：几何、切割有限元离散与主自由度',
                '每个胞元是 Schwarz-P 型材料层，厚度由八个角点参数三线性插值，至多有一个平面切割；相邻胞元共用公共面的四个角点值，'
                '所以厚度在公共面上连续。离散采用笛卡尔背景网格上的 Q2 实体单元，每轴 32 个单元，用 ghost 罚项稳定，'
                '厚度或切割改变时只重算单元积分，不用重新划分网格。主自由度是胞元边界面上的节点自由度加上切割平面单元的全部自由度，'
                '80 个验证几何中从 2,679 到 45,900 个，中位数约两万四，内部自由度约为其六倍。图中是四个代表性验证胞元。')
    image(s, FIG / 'F08_geometry.png', X0, 1.05, W, 3.55)
    for k, (h, b) in enumerate([
            ('几何', 'Schwarz-P 型材料层；厚度由八个角点参数三线性插值；至多一个平面切割；相邻胞元共用公共面的角点值'),
            ('离散', '笛卡尔背景网格上的 Q2 实体单元，每轴 n = 32；ghost 罚项稳定化；厚度或切割改变时无需重新划分网格'),
            ('主自由度', '胞元边界面上的节点自由度和切割平面单元的全部自由度：2,679–45,900 个（中位数约 2.4 万），内部自由度约为其 6 倍')]):
        card(s, X0 + k * (Wc + 0.3), 4.8, Wc, 1.85, h, [b], size=14, head_size=16)
    cite(s, '图：论文图 4（U1 未切割；M1 中度切割；H1、H2 重度切割）；论文第 2.1–2.3 节，表 1')

    # ------------------------------------------------------------------ 8 static condensation
    s = D.slide('静力凝聚：给定主自由度位移，由内部平衡恢复内部位移',
                '静力凝聚先把主自由度位移 q 当作给定值，由内部平衡恢复内部位移，得到精确位移恢复 Psi q；凝聚刚度矩阵就是 Psi 转置 K Psi，'
                '等于熟悉的 Schur 补。A 是内部刚度块，正定，所以给定 q 就排除了全部零能的内部运动。点阵中 q 由装配方程组确定，'
                '求解后恢复各胞元的内部位移，用于计算灵敏度。精确灵敏度是负 u 转置 K 对 tau 的导数再乘 u，位移恢复对设计的导数不出现，'
                '因为精确位移场满足内部平衡。近似的位移恢复要保住这套结构，需要满足右边三项：线性、在主自由度上等于给定位移、精确再现刚体运动。')
    formula(s, r'u=\Psi q,\quad J_P\Psi=I_p,\quad J_I K\Psi=0\;\Rightarrow\;\Psi_I q=-A^{-1}K_{IP}\,q', X0, 1.2, 7.4, 1.0, size=21)
    formula(s, r'S=\Psi^{T}K\Psi=K_{PP}-K_{PI}A^{-1}K_{IP},\qquad A=K_{II}\succ 0', X0, 2.35, 7.4, 1.0)
    formula(s, r'\mathbb{K}=\sum_m B_m^{T}S_mB_m,\quad \mathbb{K}U=f_g,\quad C=f_g^{T}U,\quad s_c=-u^{T}K_{,c}\,u', X0, 3.5, 7.4, 1.0,
            size=21)
    text(s, X0, 4.7, 7.4, 1.9, [
        ('-', '给定 q 即排除全部零能的内部运动；点阵中 q 由装配方程组确定'),
        ('-', '精确位移场满足内部平衡，位移恢复对设计的导数不进入灵敏度'),
        ('-', '中位数胞元的稠密 S 需 4.2 GiB；每个新几何都要重新分解 A'),
    ], size=15, space=5)
    card(s, 8.2, 1.2, 4.63, 5.4, '近似位移恢复须满足', [
        ('-', '对主自由度位移线性'),
        ('-', '在主自由度上等于给定位移'),
        ('-', '精确再现刚体运动'),
    ], fill=LBLUE, size=15)
    formula(s, r'J_P\widehat\Psi=I_p,\qquad \widehat\Psi R_P=R', 8.4, 3.05, 4.23, 0.85, fill='FFFFFF')
    text(s, 8.35, 4.1, 4.33, 2.4, ['这三项保证凝聚刚度矩阵的结构性质，与网络的训练精度无关（第 3 节）',
                                   '命题 1、3、4 对任何满足这些条件的线性位移恢复算子成立；NICE 由构造满足'], size=15, space=10)
    cite(s, '论文第 2.3、2.4、4.2 节')

    # ------------------------------------------------------------------ 9 overview
    s = D.slide('NICE：网络给出初值，修正减小内部不平衡力',
                '这是 NICE 的总体结构。上半部分是一个胞元：主自由度位移先分出刚体部分，网络以胞元几何为输入，由变形部分预测内部位移，'
                '刚体部分和给定的主自由度位移绕过网络直接恢复。然后在主自由度位移固定的条件下施加一次两重网格修正：Chebyshev 光滑、'
                'Q1 粗网格校正、再光滑。凝聚刚度矩阵由修正后位移场的应变能计算，乘上转置。下半部分是装配和设计：装配求解只需要矩阵和向量的乘积，'
                '不形成位移恢复矩阵和凝聚刚度矩阵；设计迭代中几何改变，网络参数不变，只重算几何编码和修正设置。')
    image(s, FIG / 'F01_method_overview.png', X0, 1.05, 6.95, 5.65)
    text(s, 7.75, 1.2, 5.08, 5.4, [
        label('网络　', '以胞元几何为输入，由主自由度位移预测内部位移；对几何非线性，对主自由度位移严格线性'),
        label('两重网格修正　', 'Chebyshev 光滑和 Q1 粗网格校正；主自由度位移保持不变；对每个几何是固定的线性运算'),
        label('应变能　', '凝聚刚度矩阵由修正后位移场的应变能计算'),
        label('装配求解　', '预条件共轭梯度法只需矩阵–向量乘积：网络、修正、K 及其转置依次作用，不形成位移恢复矩阵和凝聚刚度矩阵'),
        label('设计迭代　', '网络参数不变；几何改变后只重新计算几何编码和修正设置'),
    ], size=16, space=12)
    cite(s, '图：论文图 2')

    # ------------------------------------------------------------------ 10 Ritz identity
    s = D.slide('由应变能计算凝聚刚度矩阵：Ritz 恒等式',
                '凝聚刚度矩阵定义为恢复位移场的应变能，Ŝ 等于 Psi hat 转置 K Psi hat。施加时第二项不能省：内部不平衡力做的功要通过转置传回主自由度，'
                '这样返回的节点力才是应变能的导数。命题 1 是经典的 Ritz 恒等式：只要位移恢复是线性的、在主自由度上等于给定位移、精确再现刚体运动，'
                'Ŝ 减 S 就等于内部误差的能量 H 转置 A H，半正定。交叉项为零是因为精确位移场满足内部平衡，所以误差是二阶的。'
                '能量误差也可以用内部残差表示，残差运行时就能算，但计算残差的 A 逆范数需要一次内部求解。这些性质由构造保证，对任意网络参数成立，'
                '精度则要靠训练和修正。')
    formula(s, r'\widehat S=\widehat\Psi^{T}K\widehat\Psi,\qquad \widehat S q=(K\widehat\Psi q)_P+\widehat\Psi_I^{T}(K\widehat\Psi q)_I',
            X0, 1.2, 6.9, 1.15)
    formula(s, r'\boxed{\;\widehat S-S=H^{T}\!AH\succeq 0\;},\qquad H=J_I(\widehat\Psi-\Psi)', X0, 2.5, 6.9, 1.25, size=24,
            fill=LBLUE)
    formula(s, r'\varepsilon(q)=\frac{d_I^{T}Ad_I}{q^{T}Sq}=\frac{r_I^{T}A^{-1}r_I}{q^{T}Sq},\qquad d_I=Hq,\quad r_I=(K\widehat u)_I',
            X0, 3.9, 6.9, 1.35)
    text(s, X0, 5.4, 6.9, 1.25, [[('命题 1（Ritz 恒等式）', {'bold': True, 'color': RED}),
                                  ('：线性位移恢复在主自由度上等于给定位移、精确再现刚体运动时成立；为经典结果（Fraeijs de Veubeke 1965 等）', {})]],
         size=14, color=GRAY)
    text(s, 7.75, 1.2, 5.08, 5.4, [
        ('-', '凝聚刚度矩阵对称半正定，零能模态仅为刚体运动'),
        ('-', '近似刚度不低于精确值：学习模型偏刚，柔度被低估'),
        ('-', '精确位移场满足内部平衡，交叉项为零：误差为内部位移误差的二阶小量'),
        ('-', '施加时须带上转置：内部不平衡力所做的功传回主自由度'),
        ('-', '以上由构造保证，对任意网络参数成立；精度取决于网络与修正'),
    ], size=16, space=10)
    cite(s, '论文第 3.1 节，式 (4)–(7)，附录 B')

    # ------------------------------------------------------------------ 11 network
    s = D.slide('预测内部位移的网络：几何分支加线性位移分支',
                '网络要逼近的是精确位移恢复的内部部分。主自由度有两千多到四万多个，而且随切割变化，所以既不能给每个主自由度设一个输出，'
                '也不能对每个几何形成位移恢复矩阵。网络分两支：几何分支是非线性的，输入单元矩、体积和节点指标，对每个几何计算一次各种系数；'
                '位移分支是线性的，把去掉刚体部分的主自由度位移提升到 32 个通道，在模板上做局部交互，并经过每轴 65、33、17、9 个位置的 U 形多层网格，'
                '最后转换为内部位移。刚体运动在输入端分离、输出端重构，主自由度位移在输出端重新赋值，所以两项再现性质由构造保证。'
                '权重作用在模板中的局部节点位置、通道和网格层级上，不对应具体自由度，所以约六十万参数的同一个网络适用于各种主自由度集合。')
    image(s, FIG / 'F11_network_architecture.png', 6.2, 1.05, 6.63, 4.3, crop=(0.0, 0.0, 0.0, 0.485))
    text(s, X0, 1.2, 5.45, 5.45, [
        label('难点　', '主自由度 2,679–45,900 个且随切割变化：不能为每个主自由度设一个输出，也不能对每个几何形成位移恢复矩阵'),
        label('几何分支　', '非线性，每个几何算一次：单元矩与节点指标 → 编码与信息交换 → 局部交互、网格间传递和卷积的系数'),
        label('位移分支　', '对 q 线性：提升为 32 个通道 → 模板上的局部交互（27 个局部节点、4 个头）与 U 形多层网格（每轴 65 / 33 / 17 / 9）→ 内部位移'),
        label('构造保证　', '刚体运动在输入端分离、输出端重构；主自由度位移在输出端重新赋值；转置按相反顺序逐个转置'),
    ], size=15, space=10)
    text(s, 6.2, 5.5, 6.63, 1.15, [[('规模无关：', {'bold': True, 'color': BLUE}),
                                    ('权重作用于局部节点位置、特征通道和网格层级，约 6×10⁵ 个参数的同一网络适用于各种主自由度集合', {})]],
         size=15, fill=LBLUE, anchor='m', margin=0.15)
    cite(s, '图：论文图 3(a,b)；论文第 3.2 节，式 (8)、(9)')

    # ------------------------------------------------------------------ 12 two-grid correction
    s = D.slide('两重网格修正：光滑与粗网格校正分工互补',
                '网络之后施加一次两重网格修正，主自由度位移保持不变。Chebyshev 光滑以 Jacobi 预条件，区间取 b 的三十分之一到 b，'
                'b 是最大特征值的幂迭代估计再放大 5%；光滑消除 Rayleigh 商较大的误差分量。光滑衰减很慢的低频分量交给粗网格校正：'
                '17 的立方个顶点上的三线性函数，在粗空间上极小化能量。系数只依赖几何，所以修正后的位移恢复仍对 q 线性并保持再现性质。'
                '命题 2：光滑区间上端点不低于实际最大特征值、粗求解精确时，修正不增大凝聚刚度矩阵的误差，近似刚度夹在精确值和未修正值之间。'
                '右图是 M1 在网络参数固定时各阶段的能量误差：只做光滑从 13.5% 降到 4.68%，加上粗网格校正降到 0.23%，完整循环 0.186%。')
    formula(s, r'\widehat\Psi=\mathcal W\widehat\Psi_{\rm net},\qquad \mathcal W:\ 8\ /\ Q_1(17)\ /\ 8', X0, 1.2, 7.2, 0.95)
    formula(s, r'S\preceq\widehat S\preceq\widehat S_{\rm net}', X0, 2.3, 7.2, 0.95, size=24, fill=LBLUE)
    text(s, X0, 3.4, 7.2, 3.3, [
        ('-', [('Chebyshev 光滑：Jacobi 预条件，区间 [b/30, b]；消除 Rayleigh 商较大的误差分量', {})]),
        ('-', [('粗网格校正：17', {}), m('^{3}'), (' 顶点网格上的三线性函数，在粗空间上极小化能量，处理光滑衰减缓慢的低频分量', {})]),
        ('-', '主自由度位移不变；系数只依赖几何 → 修正后仍对 q 线性，保持两项再现性质；不含可训练参数'),
        ('-', [('命题 2', {'bold': True, 'color': RED}), ('：区间上端点不低于实际最大特征值、粗求解精确（或带非负平移）时，修正不增大凝聚刚度矩阵的误差', {})]),
    ], size=15, space=8)
    image(s, FIG / 'F04_correction.png', 7.95, 1.2, 4.88, 4.6, crop=(0.0, 0.55, 0.5, 0.0))
    text(s, 7.95, 5.85, 4.88, 0.8, ['M1，网络参数固定：各修正阶段后的能量误差（面力载荷）'], size=13, color=GRAY, align='c')
    cite(s, '论文第 3.3、3.4 节，表 1；右图：论文图 8(c)')

    # ------------------------------------------------------------------ 13 training and deployment
    s = D.slide('训练与部署：在部署所用的算子上训练',
                '训练数据是若干组测试位移，用精确凝聚刚度矩阵归一化。损失函数第一项是能量比的对数平均，由命题 1 非负，只有恢复位移场精确时为零；'
                '第二项是八分量厚度灵敏度相对参考值的误差。所以训练计入能量和灵敏度。损失由修正后的算子计算，梯度经过光滑和粗网格校正传到网络参数，'
                '训练和部署用同一个算子。右表是三个变体：基础网络不带修正训练四万步；基础网络加修正沿用其参数，只在评估时加修正；'
                'NICE 在带修正的条件下继续训练一万五千步，几何取自一个更大的集合。部署时每个几何只算一次几何编码和修正设置，'
                '之后每乘一次 Ŝ 都执行网络、修正、K 及其转置。')
    formula(s, r'\mathcal L(\theta)=\frac1B\sum_{j=1}^{B}\log\!\big(q_j^{T}\widehat S q_j\big)'
               r'+w_s\frac{1}{|\mathcal J_s|}\sum_{j\in\mathcal J_s}\frac{\|\tilde{\bm s}_j-\bm s_j\|_2^2}{\|\bm s_j\|_2^2},'
               r'\qquad q_j^{T}Sq_j=1', X0, 1.2, W, 1.25, size=22)
    text(s, X0, 2.6, 5.9, 4.05, [
        ('-', '第一项：能量比的对数平均；由命题 1 非负，仅当恢复位移场精确时为零'),
        ('-', [('第二项：八分量厚度灵敏度相对参考值的误差（', {}), ss('w_{s} = 1'), ('）', {})]),
        ('-', '测试位移：多项式与多尺度位移，集中节点载荷、面力载荷、弹簧支撑下的响应，相邻胞元施加的边界位移'),
        ('-', '损失由修正后的算子计算，梯度经光滑和粗网格校正传到网络参数'),
    ], size=16, space=12)
    table(s, 6.65, 2.65, 6.18, [
        ['变体', '网络参数', '训练修正', '评估修正'],
        ['基础网络', '训练 40,000 步（305 个几何）', '无', '无'],
        ['基础网络加修正', '沿用基础网络', '无', '8/Q1(17)/8'],
        [[('NICE', {'bold': True, 'color': RED})], '继续训练 15,000 步（591 个几何的集合）', '8/Q1(17)/8', '8/Q1(17)/8'],
    ], (1.7, 2.5, 1.05, 1.05), size=13, row_h=0.62, hl_rows=(3,))
    text(s, 6.65, 5.3, 6.18, 1.35, [[('部署：', {'bold': True, 'color': BLUE}),
                                     ('每个几何只算一次几何编码和修正设置；每次乘以 Ŝ 执行网络、修正、K 及其转置；装配方程组用预条件共轭梯度法求解，'
                                      '预条件子在每次设计迭代中分解一次 ', {}), m('K_{PP}')]], size=15, fill=LIGHT, margin=0.15, anchor='m')
    cite(s, '论文第 3.5、3.6 节，式 (13)，表 2')

    # ------------------------------------------------------------------ 14 error relations
    s = D.slide('误差进入柔度与灵敏度的途径不同',
                '第四节推导误差怎样进入设计量。柔度：命题 3，相对柔度误差不超过加权能量误差 beta，权重是各胞元的应变能占比，'
                '能量误差在精确装配解的主自由度位移处计算；所以应变能占比小的胞元即使内部位移不准，柔度也可能准。'
                '灵敏度：命题 4，给定主自由度位移时，灵敏度误差由线性项和二次项组成；线性项是位移误差的一阶小量，'
                '能量误差只能在根号 epsilon 的阶次上控制相对灵敏度误差。装配后主自由度解的误差也进入灵敏度误差。'
                '还要注意 s 波浪是参考灵敏度的估计，不是近似柔度的导数，两者差一个与内部残差有关的项。结论是柔度准不保证灵敏度准，'
                '所以训练计入能量与灵敏度，装配检验同时考察柔度与灵敏度。')
    Wh = (W - 0.3) / 2
    card(s, X0, 1.2, Wh, 4.15, '柔度（命题 3）', [], fill=LIGHT)
    formula(s, r'0\le\frac{C-\widehat C}{C}\le\frac{\beta}{1+\beta}\le\beta,\qquad \beta=\sum_m w_m\,\varepsilon_m', X0 + 0.15, 1.8, Wh - 0.3,
            1.3, fill='FFFFFF')
    text(s, X0 + 0.15, 3.25, Wh - 0.3, 2.0, [
        ('-', [('各胞元在精确主自由度位移下的能量误差，按应变能占比 ', {}), ss('w_{m}'), (' 加权', {})]),
        ('-', '应变能占比小的胞元几乎不影响柔度'),
        ('-', '两胞元装配中，实测柔度误差为 β 的 0.963–1.004 倍'),
    ], size=15, space=6)
    card(s, X0 + Wh + 0.3, 1.2, Wh, 4.15, '灵敏度（命题 4）', [], fill=LIGHT)
    formula(s, r'\tilde s_c-s_c=-2\,d^{T}K_{,c}\,u-d^{T}K_{,c}\,d,\qquad d=\widehat\Psi q-\Psi q', X0 + Wh + 0.45, 1.8, Wh - 0.3, 1.3,
            fill='FFFFFF')
    text(s, X0 + Wh + 0.45, 3.25, Wh - 0.3, 2.0, [
        ('-', '线性项为位移误差的一阶小量；能量误差只能在 √ε 的阶次上控制相对灵敏度误差'),
        ('-', '装配后，主自由度解的误差也进入灵敏度误差'),
        ('-', '基础网络误差较大的 M1、H2 中二次项占主导；误差趋于零时线性项一般将占主导'),
    ], size=15, space=6)
    banner(s, 5.55, [('结论：', {'bold': True, 'color': BLUE}),
                     ('柔度准确不保证灵敏度准确 → 训练计入能量与灵敏度，装配检验同时考察柔度与灵敏度', {})], h=0.75)
    cite(s, '论文第 4 节，式 (15)、(17)；s̃ 为参考灵敏度的估计，不是近似柔度 Ĉ 的导数（式 (18)）')

    # ------------------------------------------------------------------ 15 single-cell accuracy
    s = D.slide('单个胞元：80 个验证几何的平均能量误差 0.074%',
                '先看单个胞元。80 个验证几何按切割程度分四组，每组 20 个，其中 20 个用于检查点选择。左图 (a) 是面力载荷下各组的均值和范围：'
                '灰色是基础网络，绿色是基础网络加修正，橙色是 NICE。NICE 全部 80 个几何的平均能量误差 0.074%，是九类载荷中最大的一类；'
                '基础网络是 6.89%，不重新训练、只在部署时加修正就降到 0.0965%。从未切割到重度切割，NICE 的组均值从 0.016% 增到 0.109%。'
                '5,120 个测试位移中 99% 低于 0.69%，最大 1.24%。检查点选择之外的 60 个几何上是 0.077%，差别不大。')
    image(s, FIG / 'F02_validation.png', X0, 1.1, 8.0, 4.85)
    text(s, X0, 6.0, 8.0, 0.65, ['验证集：未切割、轻度、中度、重度切割各 20 个；20 个用于检查点选择，其余 60 个上的均值为 0.077%'],
         size=14, color=GRAY)
    for k, (num, lab) in enumerate([('0.074%', 'NICE，面力载荷（九类载荷中最大）'),
                                    ('6.89% → 0.0965%', '基础网络 → 只在部署时加修正'),
                                    ('0.016% → 0.109%', '未切割 → 重度切割的分组均值'),
                                    ('99% < 0.69%', '5,120 个测试位移；最大 1.24%')]):
        stat(s, 8.75, 1.1 + k * 1.4, 4.08, num, lab, fill=LIGHT, nsize=26 if k else 32, h=1.3)
    cite(s, '图：论文图 5；论文第 5.1、5.3 节，补充表 ST03')

    # ------------------------------------------------------------------ 16 error sources
    s = D.slide('网络误差的来源：低频模态、切割面附近、刚硬',
                '基础网络的误差有三个特点。第一，在谱上：M1 最低的 200 个 Jacobi 缩放模态含 24.5% 的误差能量，精确内部位移场只有 4.87%，'
                '而且这 200 个模态都低于光滑区间的下端点，光滑衰减很慢，要靠粗网格校正。第二，在空间上：M1 中距切割平面两个单元以内的单元占 13%，'
                '集中了基础网络误差能量的 39 到 43%，网络要把切割平面单元上的位移传进内部。第三，误差比精确解刚硬：'
                '相对位移误差只有 0.7% 到 1.3%，但 Rayleigh 商比精确解高两到三个数量级，能量误差因此达到 1.8% 到 35%。')
    image(s, FIG / 'F03_spectrum.png', X0, 1.1, 6.9, 4.7)
    text(s, X0, 5.85, 6.9, 0.8, ['累积能量占比：深色为误差，浅灰为精确内部位移场；横轴为 Jacobi 缩放内部谱'], size=13, color=GRAY)
    text(s, 7.7, 1.2, 5.13, 5.45, [
        label('谱分布　', 'M1 最低 200 个模态含 24.5% 的误差能量，精确内部位移场仅含 4.87%；这些模态均低于光滑区间下端，由粗网格校正处理'),
        label('空间分布　', 'M1 中距切割平面两个单元以内的单元占 13%，含基础网络误差能量的 39–43%'),
        label('刚硬　', [('相对位移误差 δ 约 0.7–1.3%，Rayleigh 商之比 κ = 124–7089；', {}), m('ε = δ^{2}κ'),
                       ('，能量误差达 1.8–35%', {})]),
        label('灵敏度　', '误差较大的胞元中二次项占主导；误差趋于零时，线性项一般将占主导'),
    ], size=16, space=12)
    cite(s, '图：论文图 6；论文第 5.4 节，图 7，补充表 ST04、ST05')

    # ------------------------------------------------------------------ 17 correction effect
    s = D.slide('修正：不改网络参数，误差降低约两个数量级',
                '这一页固定基础网络的参数，单独看修正。上图 (a)(b) 是五个胞元在 Chebyshev 光滑过程中的能量误差和灵敏度误差：'
                'H2 的误差主要在高频，八步光滑就从 35% 降到 0.2%；M1 的误差有很大一部分在光滑区间以下，八步只降到 4.68%，要靠粗网格校正。'
                '下表是同一修正作用在四种初始场上：零内部位移、图调和初始场、基础网络、NICE。图调和初始场不含可训练参数，'
                '修正后遗留的误差是 NICE 的 5.4 到 265 倍；光滑步数加到八倍，六个胞元中仍有五个高于基础网络。'
                '所以网络和修正互补；误差的降低主要来自修正，NICE 带修正继续训练后再降到基础网络加修正的 1/1.31。')
    image(s, FIG / 'F04_correction.png', X0, 1.1, 7.3, 2.75, crop=(0.0, 0.0, 0.0, 0.5))
    table(s, X0, 3.95, 7.3, [
        ['胞元', '基础网络（无修正）', '零内部位移', '图调和', '基础网络', 'NICE'],
        ['M1', '13.5', '1097', '17.2', '0.186', '0.131'],
        ['M2', '3.64', '324', '6.72', '0.0273', '0.0358'],
        ['H1', '1.81', '39.9', '1.28', '0.0131', '0.0115'],
        ['H2', '35.0', '28.3', '0.0806', '0.0117', '0.0148'],
        ['U2', '0.402', '47.2', '1.23', '0.00424', '0.00465'],
    ], (0.6, 1.8, 1.25, 0.95, 1.2, 0.95), size=13, row_h=0.39)
    text(s, X0, 6.36, 7.3, 0.3, ['后四列为施加同一 8/Q1(17)/8 修正后的平均能量误差（%），面力载荷'], size=12, color=GRAY)
    text(s, 8.1, 1.2, 4.73, 5.45, [
        ('-', 'M1：13.5% → 八步光滑 4.68% → 粗网格校正后八步光滑 0.230% → 8 / Q1(17) / 8 0.186%'),
        ('-', '光滑消除高频分量，低频分量由粗网格校正处理'),
        ('-', '相同修正下，图调和初始场遗留的误差为 NICE 的 5.4–265 倍；光滑工作量增至八倍，六个胞元中仍有五个高于基础网络'),
        ('-', '误差的降低主要来自修正；NICE 带修正继续训练，基础网络加修正的误差为 NICE 的 1.31 倍'),
    ], size=15, space=10)
    cite(s, '图：论文图 8(a,b)；表：论文表 4；论文第 5.3、5.5 节，补充表 ST07')

    # ------------------------------------------------------------------ 18 two-cell assemblies
    s = D.slide('两胞元装配：柔度与灵敏度误差均低于 3% 线',
                '装配后能否得到准确的柔度和局部灵敏度？每个两胞元装配由一个学习的被测胞元和一个精确的未切割相邻胞元组成，'
                '每种构型六个面载荷，分别加在被测胞元和相邻胞元上。七个检查点选择胞元的十四种构型中，NICE 最大柔度误差 0.056%，'
                '最大灵敏度误差 0.67%。另外九个检查点选择之外的胞元，其中五个是单胞元误差最大的几何，十八种构型的最大值是 0.27% 和 1.49%。'
                '不加修正时，基础网络在 U1 和 M1 上超过 3% 灵敏度线，最大 12.2%，M1/x 的柔度误差也有 4.38%。')
    image(s, FIG / 'F05_assembly.png', X0, 1.05, 5.75, 5.65)
    image(s, FIG / 'F09_assembly_loads.png', 6.55, 1.1, 6.28, 2.3)
    text(s, 6.55, 3.55, 6.28, 3.1, [
        ('-', '被测胞元为学习胞元，相邻胞元为精确的未切割胞元；每种构型六个面载荷'),
        ('-', '七个检查点选择胞元、十四种构型：柔度误差 ≤0.056%，灵敏度误差 ≤0.67%'),
        ('-', '另九个胞元、十八种构型（含单胞元误差最大的五个）：≤0.27% 和 ≤1.49%'),
        ('-', '不施加修正时，基础网络的灵敏度误差达 12.2%，M1/x 柔度误差 4.38%'),
    ], size=15, space=8)
    cite(s, '图：论文图 9、图 10；论文第 5.6 节，补充表 ST08')

    # ------------------------------------------------------------------ 19 compliance vs sensitivity
    s = D.slide('柔度准确不保证灵敏度准确',
                '为什么柔度误差可以比灵敏度误差小几个数量级？关键是被测胞元的应变能占比。左图柔度误差和 beta 几乎相等；'
                '中、右图是柔度误差和灵敏度误差随应变能占比的变化。以 U1/x 相邻胞元边界面上的 z 向载荷为例：被测胞元只承担 0.127% 的应变能，'
                '局部能量误差 0.0575%，beta 等于两者之积 7.33 乘 10 的负 5 次方 %，实测柔度误差 7.31，几乎相等。'
                '但被测胞元的灵敏度误差是 0.598%，约为柔度误差的 8,200 倍：一个因子 1/w 约 780 来自归一化，另一个来自根号 epsilon 的阶次。')
    image(s, FIG / 'F10_energy_share.png', X0, 1.05, W, 4.15)
    for k, (num, lab) in enumerate([('0.127%', '被测胞元的应变能占比 w'),
                                    ('0.0000733%', 'β = wε（ε = 0.0575%）'),
                                    ('0.0000731%', '实测柔度误差'),
                                    ('0.598%', '被测胞元灵敏度误差，约为柔度误差的 8,200 倍')]):
        stat(s, X0 + k * (2.9 + 0.243), 5.3, 2.9, num, lab, fill=LIGHT, nsize=24, lsize=13, h=1.4)
    cite(s, '图：论文图 11；例子：U1/x，相邻胞元边界面 z 向面力载荷（论文第 5.6 节，补充表 ST25）')

    # ------------------------------------------------------------------ 20 ablation
    s = D.slide('消融：限制胞元边界面位移，局部灵敏度误差大',
                '保留全部主自由度有什么用？这个消融里两个胞元都用精确凝聚刚度矩阵，只把胞元边界面位移限制为 r 次 Bernstein 多项式，'
                '所以误差全部来自边界表示。所有边界面都受限时，一次多项式的柔度误差 78% 到 85%，r 等于 8 降到 0.5% 到 0.7%，'
                '这时主自由度约为完整表示的四成。但局部灵敏度收敛慢得多：r 等于 8 时，计入相邻胞元边界面载荷，被测胞元灵敏度误差还有 24% 到 64%。'
                '只限制共享界面，r 等于 3 时柔度误差已降到 0.17% 到 0.44%，灵敏度误差仍有 8% 到 21%。'
                '注意这个消融只在胞元尺寸固定时限制边界表示，没有采用学习子结构控制边界误差的那些措施，比如细化分区、增强边界插值或过采样。')
    image(s, FIG / 'F06_bernstein.png', X0, 1.1, 7.6, 3.85)
    banner(s, 5.15, [('结论：', {'bold': True, 'color': BLUE}),
                     ('低次界面即可满足柔度要求；局部灵敏度需要完整的主自由度。该消融在胞元尺寸固定时进行，未采用细化分区、增强边界插值或过采样。', {})],
           h=1.45, size=15)
    text(s, 8.4, 1.2, 4.43, 3.85, [
        ('-', '两个胞元均用精确凝聚刚度矩阵，只把胞元边界面位移限制为 r 次 Bernstein 多项式'),
        ('-', '全部边界面受限：柔度误差 r = 1 时 78–85%，r = 8 时 0.49–0.74%（H1：14,001 对 32,991 个主自由度）'),
        ('-', 'r = 8：被测胞元灵敏度误差 1.3–4.5%；计入相邻胞元载荷后 24–64%'),
        ('-', '仅限制共享界面，r = 3：柔度 0.17–0.44%，灵敏度仍 8–21%'),
    ], size=14, space=7)
    cite(s, '图：论文图 12（H1/x）；论文第 5.7 节，补充说明 S2，补充表 ST11')

    # ------------------------------------------------------------------ 21 eight-cell lattices
    s = D.slide('八胞元点阵：全部为未参与训练的学习胞元',
                '设计中每个胞元都是学习胞元，相邻胞元的误差进入同一个装配解。这里用两个点阵检验：2×2×2 块体，四个胞元被切割；'
                '3×3×1 单层，三个切割、一个角部空缺。十六个胞元专为这一节生成，没有参与训练，也没有参与检查点选择。'
                '面载荷下两个点阵的柔度误差都低于 0.015%，各胞元灵敏度误差不超过 0.144%，点阵梯度与精确梯度相差 0.09% 以内，夹角余弦高于 0.999999。'
                '误差没有在装配中累积：点阵的柔度和梯度误差都低于其中单个胞元的最大误差。式 (15) 的 beta 给出柔度误差的上界，面载荷下超出实测值不到 beta 的 0.4%。')
    for k, (num, lab) in enumerate([('<0.015%', '点阵柔度误差（面载荷）'),
                                    ('≤0.09%', '点阵梯度误差；夹角余弦 > 0.999999'),
                                    ('≤0.144%', '各胞元灵敏度误差（面载荷）'),
                                    ('0.13% / 0.12%', '装配主自由度解的相对差')]):
        col, row = k % 2, k // 2
        stat(s, X0 + col * 3.15, 1.25 + row * 1.85, 2.95, num, lab, fill=LIGHT, nsize=30 if k < 3 else 26, h=1.65)
    text(s, X0, 5.0, 6.1, 0.55, [[('随机载荷下：', {'bold': True}), ('柔度 ≤0.069%，灵敏度 ≤0.454%，梯度 0.22–0.28%', {})]],
         size=14, color=GRAY)
    card(s, 6.85, 1.25, 5.98, 4.1, '点阵与检验', [
        ('-', '2×2×2 块体：8 个互不相同的胞元，4 个切割（剩余体积分数 62% 和 25%）'),
        ('-', '3×3×1 单层：8 个胞元，3 个切割，一个角部空缺'),
        ('-', '十六个胞元专为本节生成，未参与训练和检查点选择'),
        ('-', '一面固支；三个单位面力载荷和三个随机载荷；参考解由精确凝聚刚度矩阵装配'),
    ], size=16, fill=LBLUE)
    banner(s, 5.65, [('误差没有在装配中累积：', {'bold': True, 'color': RED}),
                     ('点阵的柔度和梯度误差均低于其中单个胞元的最大误差；式 (15) 的 β 为柔度误差上界，面载荷下超出实测值不到 β 的 0.4%', {})],
           h=0.95, size=15)
    cite(s, '论文第 5.8 节，表 3，补充表 ST14')

    # ------------------------------------------------------------------ 22 cost
    s = D.slide('计算成本：含灵敏度的八胞元分析约快十倍',
                '成本比较三条路线：整体点阵直接求解，16 线程 PARDISO；常规精确静力凝聚，16 线程；NICE，一块 GPU，含灵敏度。'
                '两个八胞元点阵上 NICE 分别 79 秒和 105 秒，直接求解 868 和 1,042 秒，约十倍；常规精确凝聚更慢。'
                '代数多重网格预条件的共轭梯度法要 637 和 752 秒，而且不含灵敏度。内存上直接求解峰值 71 和 86 GiB，精确凝聚 40 和 42 GiB，'
                'NICE 的 GPU 峰值 9.2 GiB。要说明的是各路线运行在不同处理器上，比较的是各自部署方式下的成本，都不含数据生成和训练的一次性成本。'
                '快的原因有两个：只在主自由度上求解，十几万个未约束主自由度对两百万个自由度；每个胞元只算一次几何编码和修正设置，不分解内部刚度矩阵。')
    cd = CategoryChartData()
    cd.categories = ['四胞元层 z=0', '四胞元层 z=1', '八胞元块体', '八胞元单层']
    cd.add_series('(a) 整体点阵直接求解，16 线程', (484, 349, 868, 1042))
    cd.add_series('(b) 常规精确静力凝聚，16 线程', (589, 632, 950, 1218))
    cd.add_series('(c) NICE，一块 GPU（含灵敏度）', (39.4, 37.3, 79.2, 105.5))
    gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(X0), Inches(1.1), Inches(7.9), Inches(4.9), cd)
    ch = gf.chart
    ch.has_title = True; ch.chart_title.text_frame.text = '一次点阵分析的耗时（s）'
    for r in ch.chart_title.text_frame.paragraphs[0].runs:
        font(r, 15, DARK, True)
    ch.has_legend = True; ch.legend.position = XL_LEGEND_POSITION.BOTTOM; ch.legend.include_in_layout = False
    ch.legend.font.size = Pt(12)
    for ser, col in zip(ch.plots[0].series, ('7F7F7F', 'BFBFBF', RED)):
        ser.format.fill.solid(); ser.format.fill.fore_color.rgb = rgb(col)
    pl = ch.plots[0]; pl.has_data_labels = True; pl.data_labels.font.size = Pt(11)
    pl.data_labels.number_format = '#,##0'; pl.data_labels.number_format_is_linked = False
    pl.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END; pl.gap_width = 60
    va = ch.value_axis; va.has_major_gridlines = True; va.major_gridlines.format.line.color.rgb = rgb('E0E0E0')
    va.tick_labels.font.size = Pt(11); va.format.line.fill.background()
    ch.category_axis.tick_labels.font.size = Pt(12)
    text(s, X0, 6.05, 7.9, 0.65, ['各路线运行于不同处理器（一块 GPU 对 16 个 CPU 线程），比较的是各自部署方式下的成本；均不含数据生成和训练的一次性成本'],
         size=12, color=GRAY)
    stat(s, 8.75, 1.1, 4.08, '79–105 s', 'NICE，八胞元点阵（含灵敏度）', fill=LIGHT, h=1.55, nsize=30)
    stat(s, 8.75, 2.8, 4.08, '9.2 GiB', 'GPU 内存峰值（直接求解 71–86 GiB）', fill=LIGHT, h=1.55, nsize=30)
    stat(s, 8.75, 4.5, 4.08, '637–752 s', 'AMG 预条件共轭梯度法（不含灵敏度）', fill=LIGHT, h=1.55, nsize=30)
    cite(s, '论文第 5.9 节，表 5，补充表 ST12、ST13')

    # ------------------------------------------------------------------ 23 optimisation case A
    s = D.slide('厚度优化算例 A：复现精确静力凝聚得到的设计',
                '优化问题是体积约束下的柔度最小化，体积上限为初始的 80%，参数界限 0.18 到 0.69，每个胞元的角点跨度和梯度范数限制在 0.45 以内，'
                '保证每个胞元都在训练范围内。每次迭代重新生成几何和学习子结构，走一步 MMA，不做线搜索，因为 s 波浪不是近似柔度的导数。'
                '算例 A 是第 5.8 节的块体，对照运行用精确静力凝聚。NICE 24 次迭代停止，对照运行 23 次；两次运行都在材料减少 20% 的同时使柔度降低 2%，'
                '最终角点参数之差至多 0.0051，最终设计的精确柔度相对相差十万分之一量级。三个校核设计上 NICE 柔度至多低 0.028%，梯度误差至多 0.33%，'
                '每个分量的符号都正确。每次迭代 NICE 74 秒，对照运行 469 秒。')
    image(s, FIG / 'F13_optimisation.png', X0, 1.1, 5.6, 5.55, crop=(0.0, 0.0, 0.5, 0.0))
    formula(s, r'\min_{\bm\tau}\ C(\bm\tau)\quad\text{s.t.}\quad V(\bm\tau)\le V^{*}=0.8\,V(\bm\tau^{0}),\quad 0.18\le\tau\le 0.69',
            6.35, 1.2, 6.48, 0.95, size=20)
    text(s, 6.35, 2.35, 6.48, 4.3, [
        ('-', '2×2×2 块体，固支面的对面施加法向面力；加载面所在平面内的 9 个顶点参数固定'),
        ('-', 'MMA，移动限 5%，不做线搜索；角点跨度和梯度范数 ≤ 0.45，使各胞元保持在训练范围内'),
        ('-', 'NICE 24 次迭代，精确静力凝聚的对照运行 23 次；材料减少 20%，柔度降低 2.0%'),
        ('-', [('最终角点参数之差至多 0.0051；最终设计的精确柔度相对相差 ', {}), ss('1.25×10^{-5}')]),
        ('-', '三个校核设计：NICE 柔度至多低 0.028%，梯度误差至多 0.33%，各分量符号一致'),
        ('-', '每次设计迭代：NICE 74 s，对照运行 469 s'),
    ], size=15, space=7)
    cite(s, '图：论文图 14(a)；论文第 2.4、5.10 节，表 6')

    # ------------------------------------------------------------------ 24 clamped plate
    s = D.slide('切割面固支板：均匀化低估初始设计柔度 27%',
                '第二个算例是切割面固支的 24 胞元单层板，8 个胞元被切割，支撑就是切割平面单元。上图是 NICE 的最终设计和相对均匀初始设计的厚度变化。'
                '均匀化模型的弹性张量来自同一离散模型上的周期均匀化，在切割平面上固支；均匀化模型把均匀初始设计的精确柔度低估了 27%，宏观网格加密后仍是 25.8%，'
                '这与单层结构厚度方向没有尺度分离相符。NICE 经精确校核，初始设计柔度偏差 0.011%，最终设计 0.030%，梯度误差 0.30%。'
                '均匀化方法得到的设计比 NICE 设计少用 0.18% 的材料，柔度高 2.44%；优化设计的柔度比同体积均匀设计低 32%。')
    image(s, FIG / 'F14_plate_design3d.png', X0, 1.05, 8.0, 2.95, crop=(0.0, 0.0, 0.0, 0.47))
    text(s, X0, 4.2, 8.0, 2.45, [
        ('-', 'NICE：24 次迭代，柔度 75.59 → 110.49 → 78.26；每次迭代 200 s，共 1.3 h'),
        ('-', '精确校核：初始设计柔度偏差 0.011%、梯度 0.029%；最终设计 0.030%、0.30%，64 个梯度分量符号一致'),
        ('-', '均匀化设计与 NICE 设计的角点参数相差可达 0.26；优化设计的柔度比同体积均匀设计低 32%'),
    ], size=15, space=7)
    stat(s, 8.75, 1.1, 4.08, '27%', '均匀化对初始设计柔度的低估（宏观网格加密后 25.8%）', fill=LIGHT, h=1.6, nsize=34)
    stat(s, 8.75, 2.85, 4.08, '0.030% / 0.30%', 'NICE 最终设计：柔度 / 梯度偏差', fill=LIGHT, h=1.6, nsize=28)
    stat(s, 8.75, 4.6, 4.08, '2.44%', '均匀化设计的精确柔度更高（材料少 0.18%）', fill=LIGHT, h=1.6, nsize=34)
    cite(s, '图：论文图 13(a,b)；论文第 5.10 节，图 14(b)，补充表 ST18、ST21')

    # ------------------------------------------------------------------ 25 scale
    s = D.slide('规模：设计迭代耗时与胞元数成正比',
                '最后是规模。四块板 24、51、88、110 个胞元，比例和切割方式相同，各跑前四次设计迭代。一次迭代的耗时近似与胞元数成正比，'
                '每胞元 10.1 到 10.9 秒，110 胞元时 19.4 分钟，这块板的切割有限元模型有 3270 万个自由度。共轭梯度迭代 138 到 173 次，基本不随规模增长。'
                '24 和 51 胞元板在第一次迭代时做了精确校核，柔度误差低于 0.014%，梯度误差低于 0.05%。'
                '学习胞元在 GPU 上占固定的 4 GiB，GPU 内存随主自由度装配系统增长，所以目前的规模上限来自预条件子里 K_PP 的直接分解。')
    image(s, FIG / 'F16_scale_plates.png', X0, 1.05, 8.2, 4.4)
    text(s, X0, 5.55, 8.2, 1.1, [
        ('-', '24 和 51 胞元板经精确校核：柔度误差 < 0.014%，梯度误差 < 0.05%，梯度分量符号一致'),
        ('-', [('学习胞元在 GPU 上占固定 4 GiB；GPU 内存随主自由度装配系统增长，预条件子直接分解 ', {}), ss('K_{PP}')]),
    ], size=14, space=4)
    stat(s, 8.95, 1.1, 3.88, '10.1–10.9 s', '每胞元每次设计迭代（24–110 胞元）', fill=LIGHT, h=1.6, nsize=30)
    stat(s, 8.95, 2.85, 3.88, '19.4 min', '110 胞元板：3270 万个自由度，162 万个未约束主自由度', fill=LIGHT, h=1.6, nsize=30)
    stat(s, 8.95, 4.6, 3.88, '138–173', '共轭梯度迭代次数', fill=LIGHT, h=1.6, nsize=30)
    cite(s, '图：论文图 16；论文第 5.10 节，补充表 ST20')

    # ------------------------------------------------------------------ 26 table 3
    s = D.slide('全部装配算例：NICE 与精确静力凝聚的偏差',
                '这一页把所有装配算例放在一起，都是 NICE 相对同一离散模型的精确静力凝聚。两胞元装配里只有被测胞元是学习胞元，'
                '其余算例全部是学习胞元。柔度误差最大 0.27%，出现在单胞元误差最大的几何上；全学习的点阵和设计中，柔度误差不超过 0.069%，'
                '梯度误差不超过 0.33%。精确核验限于表里这些算例：88 和 110 胞元板没有算精确参考解。')
    table(s, X0, 1.15, W, [
        ['算例', '每构型胞元数（切割）', '载荷或设计', '柔度误差 (%)', '灵敏度或梯度误差 (%)'],
        ['两胞元装配，七个选择胞元', '2（0 或 1）', '14 个构型，各六个面载荷', '≤ 0.056', '灵敏度 ≤ 0.67'],
        ['两胞元装配，切割面面力载荷', '2（1）', '10 个构型，各三个切割面面力载荷', '≤ 0.10', '灵敏度 ≤ 0.16'],
        ['两胞元装配，九个检查点选择之外的胞元', '2（0 或 1）', '18 个构型，各六个面载荷', '≤ 0.27', '灵敏度 ≤ 1.49'],
        ['全学习胞元点阵，面载荷', '8（4），8（3）', '三个面载荷', '≤ 0.015', '灵敏度 ≤ 0.144；梯度 0.04–0.09'],
        ['全学习胞元点阵，随机载荷', '8（4），8（3）', '三个随机载荷', '≤ 0.069', '灵敏度 ≤ 0.454；梯度 0.22–0.28'],
        ['算例 A，厚度优化', '8（4）', '第 0、12、23 次迭代的设计', '0.011–0.028', '梯度 0.069–0.33'],
        ['切割面固支板', '24（8）', '初始、最终、均匀化及两个均匀设计', '0.011–0.032', '梯度 0.029–0.32'],
        ['规模算例的板', '24（8），51（12）', '初始设计', '0.013，0.011', '梯度 0.045，0.047'],
    ], (3.5, 2.15, 3.0, 1.4, 2.55), size=13, row_h=0.5, hl_rows=(4, 6, 7))
    banner(s, 5.85, [('柔度误差：', {'bold': True, 'color': BLUE}), ('全部 ≤ 0.27%；', {}),
                     ('全学习胞元的点阵与设计：', {'bold': True, 'color': BLUE}), ('柔度 ≤ 0.069%，梯度 ≤ 0.33%', {})], h=0.7)
    cite(s, '论文表 3；两胞元装配中只有被测胞元为学习胞元，其余算例中所有胞元均为学习胞元')

    # ------------------------------------------------------------------ 27 limitations
    s = D.slide('适用范围与局限',
                '局限分两类。网络方面：网络只适用于训练覆盖的设定，Schwarz-P 型胞元、角点参数 0.175 到 0.699、至多一个平面切割、一种离散设置。'
                '对 gyroid 片状胞元直接使用，误差是 Schwarz-P 的 56 到 234 倍；加入十八个 gyroid 胞元继续训练后降低六到十倍，仍是 10 到 23 倍，'
                '所以新胞族需要新的训练数据和网络参数。网络对立方体对称变换不等变。精度与求解方面：结构性质对任意网络参数成立，精度没有这样的保证，'
                '也没有经认证的误差界；学习模型偏刚，柔度不是保守估计；灵敏度是估计值，不是近似柔度的导数；规模受 K_PP 直接分解限制；'
                '变体比较只用了一个随机种子。')
    card(s, X0, 1.2, Wh, 4.0, '网络的适用范围', [
        ('-', 'Schwarz-P 型胞元；角点参数 [0.175, 0.699]，角点跨度与梯度范数 ≤ 0.47'),
        ('-', [('每个胞元至多一个平面切割；一种离散设置（n = 32，E = 1，ν = 0.3，', {}), ss('γ = 10^{-4}'), ('）', {})]),
        ('-', 'gyroid 片状胞元：直接使用误差为 Schwarz-P 的 56–234 倍；加入 18 个 gyroid 胞元继续训练后仍为 10–23 倍 → 新胞族需重新训练'),
        ('-', '对立方体对称变换不等变：第二种取向下平均能量误差相对高 4%（面力载荷 12%）'),
    ], size=15)
    card(s, X0 + Wh + 0.3, 1.2, Wh, 4.0, '精度与求解', [
        ('-', '结构性质对任意网络参数成立，精度没有保证；未给出经认证的误差界'),
        ('-', '学习模型偏刚，柔度被低估，不是保守估计'),
        ('-', '灵敏度为参考灵敏度的估计，不是近似柔度的导数；优化不做线搜索'),
        ('-', [('可分析规模受预条件子中 ', {}), m('K_{PP}'), (' 直接分解的限制；不承载的切割平面单元自由度占未约束主自由度的 37–50%', {})]),
        ('-', '变体比较只用一个随机种子；损失中灵敏度项的作用未单独考察'),
    ], size=15)
    banner(s, 5.45, [('迁移：', {'bold': True, 'color': BLUE}),
                     ('新的胞族、多个切割、曲面边界、其他分辨率和材料需要新的训练数据和网络参数；凝聚刚度矩阵的构造、误差关系与修正可原样沿用。', {})],
           h=1.15)
    cite(s, '论文第 6.3、7 节，补充说明 S7，补充表 ST02、ST03c')

    # ------------------------------------------------------------------ 28 conclusions and next steps
    s = D.slide('结论与后续工作',
                '总结五点。第一，保留全部主自由度、只近似内部位移，切割面上的支撑和载荷直接施加，同一网络适用于随切割变化的主自由度集合。'
                '第二，由应变能计算凝聚刚度矩阵，结构性质由构造保证，对任意网络参数成立。第三，固定的两重网格修正与网络互补。'
                '第四，柔度误差按应变能占比加权，灵敏度误差含一阶项，两者分别检验。第五，数值上八胞元点阵快约十倍，厚度设计复现精确静力凝聚的设计，'
                '切割面固支板上均匀化低估 27%，NICE 偏差 0.03%。后续工作：多胞族训练、对不承载的切割平面单元自由度做约化、改进预条件子以扩大规模。')
    concl = [
        '保留全部主自由度、只近似内部位移：切割面上的支撑和载荷直接施加，同一网络适用于随切割变化的主自由度集合',
        '由应变能计算凝聚刚度矩阵：对称半正定、只有刚体零能模态、不低于精确值，误差为二阶小量；对任意网络参数成立',
        '固定的两重网格修正与网络互补：不改网络参数即可将平均能量误差由 6.89% 降至 0.0965%',
        '柔度误差按应变能占比加权，灵敏度误差含一阶项：两者须分别检验',
        '八胞元点阵分析快约十倍；厚度设计复现精确静力凝聚的设计；切割面固支板上均匀化低估柔度 27%，NICE 偏差 0.03%',
    ]
    for k, c in enumerate(concl):
        y = 1.2 + k * 1.08
        text(s, X0, y + 0.05, 0.7, 0.9, ['0%d' % (k + 1)], size=26, color=RED, bold=True, anchor='m')
        text(s, X0 + 0.8, y, 6.8, 1.0, [c], size=16, anchor='m')
    card(s, 8.35, 1.2, 4.48, 5.45, '后续工作', [
        ('-', '多胞族训练，在留出的胞族上检验网络的可迁移程度'),
        ('-', '对切割平面单元中不共享、不承载的自由度采用约化表示'),
        ('-', '预条件子增加一层或采用迭代粗求解，突破单 GPU 上的规模限制'),
        ('-', '对称等变的网络结构'),
    ], size=16, fill=LBLUE, head_color=BLUE)
    cite(s, '论文第 6、7 节')

    # ------------------------------------------------------------------ 29 closing
    s = D.slide('谢谢！敬请各位老师和同学批评指正', '谢谢大家，欢迎提问。')
    image(s, FIG / 'F14_plate_design3d.png', 2.0, 1.3, 9.33, 4.7, crop=(0.0, 0.0, 0.5, 0.47))
    text(s, X0, 6.1, W, 0.5, ['切割面固支板的 NICE 最终设计（NICE 柔度 78.26，精确柔度 78.29）'], size=15, color=GRAY, align='c')
    cite(s, '图：论文图 13(a)')

    # ------------------------------------------------------------------ backup 1: reference model and deployed operator
    s = D.slide('备用：参考模型与部署算子的验证',
                '备用页：参考模型和部署算子的验证。n 等于 32 的参考模型在七个胞元上加密到 40、48 或 64，除重度切割的 H2 外，'
                '柔度变化不超过 0.28%，灵敏度不超过 0.44%；H2 约 1% 和 3.6%。未切割胞元与贴体二次四面体模型相差 0.05% 到 0.25%。'
                'ghost 罚项系数在 10 的负 5 次方到负 3 次方之间变化，柔度和灵敏度变化不超过 0.12%。部署算子的对称性和功能量恒等式偏差不超过 3 乘 10 的负 8 次方。'
                '光滑区间上端点 b 比收敛的最大特征值高 2.1% 到 5.0%。')
    image(s, FIG / 'S06_reference_verification.png', X0, 1.1, 7.4, 4.7)
    text(s, 8.2, 1.2, 4.63, 5.45, [
        ('-', '网格加密（U1 → n = 40；M1、M2 → 48；H1、H2 等 → 64）：除 H2 外柔度变化 ≤ 0.28%，灵敏度 ≤ 0.44%；H2 约 1% 和 3.6%'),
        ('-', '未切割胞元：与贴体二次四面体模型相差 0.05–0.25%'),
        ('-', [('ghost 罚项系数 ', {}), ss('10^{-5}–10^{-3}'), ('、积分加密：变化 ≤ 0.12%', {})]),
        ('-', [('部署算子的对称性偏差 ≤ ', {}), ss('3×10^{-8}')]),
        ('-', '光滑区间上端点 b 比收敛的最大特征值高 2.1–5.0%（80 个几何）'),
    ], size=14, space=7)
    cite(s, '论文第 5.2 节；补充说明 S1，补充表 ST04、ST10')

    # ------------------------------------------------------------------ backup 2: derivatives and solver residual
    s = D.slide('备用：灵敏度的求导实现与求解残差',
                '备用页，回答两个常见追问。第一，优化中的灵敏度由矩积分的反向模式微分求得，精确校核用矩的中心差分；'
                '反向模式导数是固定裁剪拓扑下离散矩的精确导数，两者吻合到 2 乘 10 的负 8 次方，所以报告的偏差不来自求导方式。'
                '第二，八胞元点阵求解中重新计算的残差停在 10 的负 4 次方到负 3 次方，这与单精度网络的舍入误差相符；'
                '最终迭代步的残差功至多为柔度的 2.5 乘 10 的负 8 次方，所以柔度误差来自凝聚刚度矩阵，不来自求解不完全。')
    card(s, X0, 1.2, Wh, 3.6, '灵敏度的求导实现', [
        ('-', '优化与计时：矩积分的反向模式微分（恢复位移场固定）'),
        ('-', '精确校核：矩的中心差分（下式）；步长研究显示二阶截断误差'),
        ('-', [('反向模式导数为固定裁剪拓扑下离散矩的精确导数，与中心差分吻合至 ', {}), ss('2×10^{-8}')]),
        ('-', '所报灵敏度偏差反映位移近似，与求导方式无关'),
    ], size=15)
    formula(s, r'M_{e\alpha,c}\approx\frac{M_{e\alpha}(\bm\tau+h_ce_c)-M_{e\alpha}(\bm\tau-h_ce_c)}{2h_c},\qquad h_c=10^{-5}\tau_c',
            X0, 5.0, Wh, 1.3, size=20)
    card(s, X0 + Wh + 0.3, 1.2, Wh, 3.6, '求解残差与柔度误差', [
        ('-', [('递推残差达到容差；重新计算的残差停滞于 ', {}), ss('3.6×10^{-4}'), ('（块体）、', {}), ss('3.2×10^{-3}'), ('（单层）', {})]),
        ('-', '这一水平与单精度网络的舍入误差相符；修正改用单精度，变化小于 1%'),
        ('-', [('最终迭代步的残差功至多为柔度的 ', {}), ss('2.5×10^{-8}'), ('（下式）', {})]),
        ('-', '因此所报柔度误差来自凝聚刚度矩阵，而非求解不完全'),
    ], size=15)
    formula(s, r'C-f_g^{T}\bar U=a(\bar u-u,\bar u-u)+\bar U^{T}\rho,\qquad \rho=f_g-\widehat{\mathbb K}\bar U',
            X0 + Wh + 0.3, 5.0, Wh, 1.3, size=20)
    cite(s, '论文第 5.8、5.10 节，附录 H；补充说明 S1、S4.3，补充表 ST14')

    n = len(D.prs.slides)
    D.total(n)
    D.prs.save(out)
    print(out, n, 'slides')


if __name__ == '__main__':
    build(sys.argv[1], sys.argv[2])
