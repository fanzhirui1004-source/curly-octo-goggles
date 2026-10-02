"""Build the elsarticle (CMAME) LaTeX source of the P1 manuscript from MANUSCRIPT_EN.md and APPENDICES_EN.md.

The Markdown files remain the editing source. This script:
  - takes the title, abstract and keywords into the elsarticle front matter;
  - removes the manual section numbers (LaTeX numbers sections and appendices itself);
  - turns an image followed by a bold 'Figure N.' paragraph into a figure with that caption (PDF version of the image);
  - turns a bold 'Table N.' paragraph before a pipe table into that table's caption;
  - converts the body with pandoc (tex_math_single_backslash keeps \\( \\) and \\[ \\tag{} \\] as written);
  - inserts authors.tex (author block) into the front matter and endmatter.tex (declarations) after the main text; the
    manuscript's 'Code and data availability' section is set unnumbered and serves as the data-availability statement;
  - orders the document as main text, declarations, appendices, references (Elsevier convention);
  - sets Table 5 (lattice-level cost comparison) and Table 6 (thickness optimisation cases), eight columns each, on landscape pages;
  - sets the reference list as an unnumbered section in author-year form.
Usage: python3 build_tex.py   (writes main.tex next to this file; compile with pdflatex twice)"""
import re, subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent


def pandoc(md):
    r = subprocess.run(['python3', '-c', 'import pypandoc,sys; sys.stdout.write(pypandoc.convert_text(sys.stdin.read(), "latex", '
                        'format="markdown+tex_math_single_backslash-auto_identifiers-smart", extra_args=["--wrap=preserve","--shift-heading-level-by=-1"]))'],
                       input=md, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr)
    return r.stdout


def strip_numbers(md):
    md = re.sub(r'^(#{2,4}) Appendix ([A-Z])\. ', r'\1 ', md, flags=re.M)             # '## Appendix A. Title'
    md = re.sub(r'^(#{2,4}) [A-Z]\.[0-9]+\. ', r'\1 ', md, flags=re.M)               # '### A.1. Title'
    md = re.sub(r'^(#{2,4}) [0-9]+(\.[0-9]+)*\. ', r'\1 ', md, flags=re.M)           # '## 1. ' / '### 2.1. '
    return md


def figures(md):
    """'![Figure n](figures/x.png)' + blank + '**Figure n. Title.** caption' -> a raw LaTeX figure."""
    pat = re.compile(r'^!\[(Figure [^\]]*)\]\((figures/[^)]+)\)\s*\n\s*\n\*\*(Figure [0-9A-Z]+)\. ([^\n]*?)\*\*([^\n]*)\n', re.M)

    def rep(m):
        path = re.sub(r'\.(png|svg)$', '.pdf', m.group(2))
        if not (SRC / path).exists():
            path = m.group(2)
        cap = pandoc('**' + m.group(4).strip() + '**' + m.group(5)).strip()
        label = 'fig:' + m.group(3).split()[1]
        long = len(cap) > 1500                                             # very long captions: smaller image and type
        h, size = ('0.62', '\\footnotesize ') if long else ('0.8', '')
        return ('\n```{=latex}\n\\begin{figure}[!htbp]\n\\centering\n\\includegraphics[width=\\textwidth,height=' + h + '\\textheight,keepaspectratio]{../' + path + '}\n'
                '\\caption{' + size + cap + '}\\label{' + label + '}\n\\end{figure}\n```\n')
    return pat.sub(rep, md)


def tables(md):
    """'**Table n. Title.** rest' immediately before a pipe table -> pandoc caption line after the table (numbered by
    LaTeX). Only numeric labels: supplementary labels ('Table ST27') stay as their bold paragraph, like the other ST tables."""
    lines = md.split('\n'); out = []; i = 0
    while i < len(lines):
        m = re.match(r'^\*\*(Table [0-9]+)\.\s*(.*?)\*\*(.*)$', lines[i])
        if m:
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines) and lines[j].lstrip().startswith('|'):
                k = j
                while k < len(lines) and lines[k].lstrip().startswith('|'):
                    k += 1
                cap = (m.group(2).rstrip('.') + '.' + m.group(3)).strip()
                out += lines[j:k] + ['', 'Table: ' + cap, '']
                i = k
                continue
        out.append(lines[i]); i += 1
    return '\n'.join(out)


def references(md):
    items = [p.strip() for p in md.strip().split('\n\n') if p.strip()]
    body = pandoc('\n\n'.join(items))
    return ('\\section*{References}\n\\begingroup\\small\\setlength{\\parindent}{0pt}\\setlength{\\parskip}{3pt}'
            '\\everypar{\\hangindent=1.5em\\hangafter=1}\n' + body + '\n\\endgroup\n')


def landscape(tex, caption_start, widths=None, overhang=0.0):
    """Put the (size-wrapped) longtable whose caption starts with caption_start on a landscape page. Relative widths
    may sum to more than one: the table then extends into the right-hand margin of the rotated page."""
    i = tex.index(caption_start)
    s = tex.rfind('\\begin{longtable}', 0, i)
    e = tex.index('\\end{longtable}', i) + len('\\end{longtable}')
    for size in ('{\\scriptsize\n', '{\\footnotesize\n', '{\\small\n'):
        if tex[s - len(size):s] == size:
            s -= len(size); e += 1                                          # include the size group and its brace
            break
    block = tex[s:e]
    if overhang:                                                           # shift left into the margin
        block = block.replace('{\\footnotesize\n', '{\\footnotesize\\setlength{\\LTleft}{-%g\\linewidth}\n' % overhang, 1)
    if widths:                                                             # relative column widths (sum <= 1)
        it = iter(widths)
        block = re.sub(r'\\real\{[0-9.]+\}', lambda m: '\\real{%.3f}' % next(it), block, count=len(widths))
    return tex[:s] + '\\begin{landscape}\n' + block + '\n\\end{landscape}\n' + tex[e:]


def set_widths(tex, caption_start, widths):
    """Relative column widths of the longtable whose caption starts with caption_start (portrait, sum <= 1)."""
    i = tex.index(caption_start)
    s = tex.rfind('\\begin{longtable}', 0, i)
    e = tex.index('\\end{longtable}', i)
    it = iter(widths)
    block = re.sub(r'\\real\{[0-9.]+\}', lambda m: '\\real{%.3f}' % next(it), tex[s:e], count=len(widths))
    return tex[:s] + block + tex[e:]


def main():
    ms = (SRC / 'MANUSCRIPT_EN.md').read_text()
    ap = (SRC / 'APPENDICES_EN.md').read_text()
    title = re.search(r'^# (.+)$', ms, re.M).group(1).strip()
    abstract = re.search(r'^## Abstract\s*\n(.*?)\n\*\*Keywords:\*\*', ms, re.S | re.M).group(1).strip()
    kw = re.search(r'^\*\*Keywords:\*\*\s*(.+)$', ms, re.M).group(1).strip().rstrip('.')
    body = ms[ms.index('## 1. Introduction'):ms.index('## References')]
    refs = ms[ms.index('## References') + len('## References'):]
    refs = refs[:refs.index('## Supplementary material')] if '## Supplementary material' in refs else refs
    conv = lambda t: pandoc(tables(figures(strip_numbers(t))))
    tex = (HERE / 'preamble.tex').read_text()
    tex += '\\begin{document}\n\\begin{frontmatter}\n\\title{' + pandoc(title).strip() + '}\n'
    if (HERE / 'authors.tex').exists():
        tex += (HERE / 'authors.tex').read_text() + '\n'
    tex += '\\begin{abstract}\n' + pandoc(abstract) + '\\end{abstract}\n'
    tex += '\\begin{keyword}\n' + ' \\sep '.join(pandoc(k).strip() for k in kw.split(';')) + '\n\\end{keyword}\n'
    tex += '\\end{frontmatter}\n\n' + conv(body) + '\n'
    if (HERE / 'endmatter.tex').exists():
        tex += (HERE / 'endmatter.tex').read_text() + '\n'
    tex += '\n\\appendix\n' + conv(ap) + '\n' + references(refs) + '\n\\end{document}\n'
    tex = tex.replace('\\section{', '\\section{', ).replace('\\hypertarget', '%\\hypertarget')
    tex = tex.replace('\\section{Code and data availability}', '\\section*{Code and data availability}')  # unnumbered, before the declarations
    # Normalise Pandoc's optional unnumbered-table wrapper before applying our own.
    tex = re.sub(r'\{\\def\\LTcaptype\{(?:none)?\} % do not increment counter\n'
                 r'(\\begin\{longtable\}.*?\\end\{longtable\})\n\}', r'\1', tex, flags=re.S)
    def shrink(m):                                                         # wide tables: smaller type
        cols = m.group(1).count('p{') + m.group(1).count('l') * 0
        size = '\\scriptsize' if cols >= 8 else ('\\footnotesize' if cols >= 6 else '\\small')
        block = '{' + size + '\n' + m.group(0) + '}'
        if '\\caption' not in m.group(0):
            block = '{\\def\\LTcaptype{} % do not increment counter\n' + block + '\n}'
        return block
    tex = re.sub(r'\\begin\{longtable\}\[\]\{@\{\}(.*?)@\{\}\}.*?\\end\{longtable\}', shrink, tex, flags=re.S)
    tex = set_widths(tex, 'Cost of one design iteration of the lattices', (.12, .17, .16, .16, .13, .12, .14))
    tex = landscape(tex, 'Thickness optimisation cases',
                    widths=(.12, .08, .16, .08, .13, .13, .20, .10))
    (HERE / 'main.tex').write_text(tex)
    print('main.tex', len(tex))


if __name__ == '__main__':
    main()
