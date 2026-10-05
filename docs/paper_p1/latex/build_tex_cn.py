"""Chinese PDF builds of the P1 manuscript and supplement from MANUSCRIPT_CN.md, APPENDICES_CN.md and
SUPPLEMENTARY_CN.md (translations of the *_EN.md sources). Reuses the pandoc conversion of build_tex.py; only the
Chinese markers differ ('## 摘要', '**关键词：**', '## 附录 A.', '**图 n.**', '**表 n.**', '## 参考文献').
Compile with LuaLaTeX (ctex, Fandol fonts):
  python3 build_tex_cn.py && lualatex main_cn && lualatex main_cn && lualatex supp_cn && lualatex supp_cn"""
import re
from pathlib import Path
import build_tex as B

HERE = Path(__file__).resolve().parent
SRC = B.SRC

PREAMBLE_CN = (HERE / 'preamble.tex').read_text() \
    .replace('\\usepackage[utf8]{inputenc}\n', '') \
    .replace('\\usepackage[T1]{fontenc}\n', '') \
    .replace('\\usepackage{lmodern}\n', '\\usepackage[UTF8,fontset=fandol,heading=false,scheme=chinese]{ctex}\n') \
    + ('\\newunicodechar{⪯}{\\ensuremath{\\preceq}}\\newunicodechar{⪰}{\\ensuremath{\\succeq}}\n'
       '\\newunicodechar{δ}{\\ensuremath{\\delta}}\\newunicodechar{κ}{\\ensuremath{\\kappa}}\n'
       '\\abstracttitle{摘要}\\keywordtitle{关键词}\n\\linespread{1.25}\n')


def strip_numbers(md):
    md = re.sub(r'^(#{2,4}) 附录 ([A-Z])\. ', r'\1 ', md, flags=re.M)
    return B.strip_numbers(md)


def figures(md):
    pat = re.compile(r'^!\[(Figure [^\]]*)\]\((figures/[^)]+)\)\s*\n\s*\n\*\*(图 [0-9A-Z]+)\. ([^\n]*?)\*\*([^\n]*)\n', re.M)

    def rep(m):
        path = re.sub(r'\.(png|svg)$', '.pdf', m.group(2))
        if not (SRC / path).exists():
            path = m.group(2)
        cap = B.pandoc('**' + m.group(4).strip() + '**' + m.group(5)).strip()
        label = 'fig:' + m.group(3).split()[1]
        long = len(cap) > 1500
        h, size = ('0.62', '\\footnotesize ') if long else ('0.8', '')
        return ('\n```{=latex}\n\\begin{figure}[!htbp]\n\\centering\n\\includegraphics[width=\\textwidth,height=' + h + '\\textheight,keepaspectratio]{../' + path + '}\n'
                '\\caption{' + size + cap + '}\\label{' + label + '}\n\\end{figure}\n```\n')
    return pat.sub(rep, md)


def tables(md):
    lines = md.split('\n'); out = []; i = 0
    while i < len(lines):
        m = re.match(r'^\*\*(表 [0-9]+)\.\s*(.*?)\*\*(.*)$', lines[i])
        if m:
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines) and lines[j].lstrip().startswith('|'):
                k = j
                while k < len(lines) and lines[k].lstrip().startswith('|'):
                    k += 1
                out += lines[j:k] + ['', 'Table: ' + (m.group(2) + m.group(3)).strip(), '']
                i = k
                continue
        out.append(lines[i]); i += 1
    return '\n'.join(out)


def shrink(tex, unnumbered_plain):
    def rep(m):
        cols = m.group(1).count('p{')
        size = '\\scriptsize' if cols >= 8 else ('\\footnotesize' if cols >= 6 else '\\small')
        block = '{' + size + '\n' + m.group(0) + '}'
        if unnumbered_plain and '\\caption' not in m.group(0):
            block = '{\\def\\LTcaptype{} % do not increment counter\n' + block + '\n}'
        return block
    return re.sub(r'\\begin\{longtable\}\[\]\{@\{\}(.*?)@\{\}\}.*?\\end\{longtable\}', rep, tex, flags=re.S)


ENDMATTER_CN = r'''\section*{作者贡献声明}
\textcolor{red}{[作者贡献（CRediT）。]}

\section*{利益冲突声明}
\textcolor{red}{[利益冲突声明。]}

\section*{致谢}
\textcolor{red}{[基金与致谢。]}

\section*{生成式人工智能使用声明}
\textcolor{red}{[由作者按期刊现行政策填写。]}
'''


def main_cn():
    ms = (SRC / 'MANUSCRIPT_CN.md').read_text()
    ap = (SRC / 'APPENDICES_CN.md').read_text()
    title = re.search(r'^# (.+)$', ms, re.M).group(1).strip()
    abstract = re.search(r'^## 摘要\s*\n(.*?)\n\*\*关键词：\*\*', ms, re.S | re.M).group(1).strip()
    kw = re.search(r'^\*\*关键词：\*\*\s*(.+)$', ms, re.M).group(1).strip().rstrip('。.')
    body = ms[ms.index('## 1. 引言'):ms.index('## 参考文献')]
    refs = ms[ms.index('## 参考文献') + len('## 参考文献'):]
    refs = refs[:refs.index('## 补充材料')] if '## 补充材料' in refs else refs
    conv = lambda t: B.pandoc(tables(figures(strip_numbers(t))))
    tex = PREAMBLE_CN + '\\begin{document}\n\\begin{frontmatter}\n\\title{' + B.pandoc(title).strip() + '}\n'
    if (HERE / 'authors.tex').exists():
        tex += (HERE / 'authors.tex').read_text() + '\n'
    tex += '\\begin{abstract}\n' + B.pandoc(abstract) + '\\end{abstract}\n'
    tex += '\\begin{keyword}\n' + ' \\sep '.join(B.pandoc(k).strip() for k in kw.split('；')) + '\n\\end{keyword}\n'
    tex += '\\end{frontmatter}\n\n' + conv(body) + '\n' + ENDMATTER_CN + '\n'
    tex += '\n\\appendix\\def\\appendixname{附录 }\n' + conv(ap) + '\n'
    tex += B.references(refs).replace('\\section*{References}', '\\section*{参考文献}') + '\n\\end{document}\n'
    tex = tex.replace('\\hypertarget', '%\\hypertarget')
    tex = tex.replace('\\section{代码与数据可用性}', '\\section*{代码与数据可用性}')
    tex = re.sub(r'\{\\def\\LTcaptype\{(?:none)?\} % do not increment counter\n'
                 r'(\\begin\{longtable\}.*?\\end\{longtable\})\n\}', r'\1', tex, flags=re.S)
    tex = shrink(tex, True)
    tex = B.set_widths(tex, '一次含灵敏度的点阵分析的成本', (.12, .17, .16, .16, .13, .12, .14))
    tex = B.set_widths(tex, '厚度优化算例', (.11, .09, .17, .07, .10, .10, .22, .08))
    (HERE / 'main_cn.tex').write_text(tex)
    print('main_cn.tex', len(tex))


def supp_cn():
    ms = (SRC / 'MANUSCRIPT_CN.md').read_text()
    title = re.search(r'^# (.+)$', ms, re.M).group(1).strip()
    sp = (SRC / 'SUPPLEMENTARY_CN.md').read_text()
    sp = re.sub(r'^# .*\n', '', sp, count=1, flags=re.M)
    tex = PREAMBLE_CN
    tex += ('\\setcounter{secnumdepth}{0}\n\\renewcommand{\\thefigure}{S\\ifnum\\value{figure}<10 0\\fi\\arabic{figure}}\n'
            '\\begin{document}\n\\begin{center}{\\Large 补充材料}\\\\[4pt]{\\Large\\bfseries ' + B.pandoc(title).strip() + '}\\end{center}\n\n')
    tex += B.pandoc(B.tables(figures(sp))) + '\n\\end{document}\n'
    tex = tex.replace('\\hypertarget', '%\\hypertarget')
    i = tex.index('表 ST20. 规模演示')
    s_ = tex.index('\\begin{longtable}', i); e_ = tex.index('\\end{longtable}', s_)
    it = iter((.07, .11, .07, .30, .08, .12, .11, .08))
    tex = tex[:s_] + re.sub(r'\\real\{[0-9.]+\}', lambda m: '\\real{%.3f}' % next(it), tex[s_:e_], count=8) + tex[e_:]
    tex = shrink(tex, False)
    (HERE / 'supp_cn.tex').write_text(tex)
    print('supp_cn.tex', len(tex))


if __name__ == '__main__':
    main_cn(); supp_cn()
