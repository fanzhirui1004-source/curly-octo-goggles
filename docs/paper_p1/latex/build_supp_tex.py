"""Build the LaTeX source of the supplementary material (supp.tex) from SUPPLEMENTARY_EN.md, with the conversion helpers of
build_tex.py (pandoc with the same options; image + bold 'Figure Sxx.' paragraph -> figure with that caption).

Differences from the main build: no front matter or references (the supplement cites the main text); sections are
unnumbered because the Markdown headings carry their own labels (R1, Table ST01, Supplementary Note S1, ...); figures are
numbered S01, S02, ...; wide pipe tables are set in smaller type as in the main build.
Usage: python3 build_supp_tex.py   (writes supp.tex next to this file; compile with pdflatex twice)"""
import re
from pathlib import Path
import build_tex as B

HERE = Path(__file__).resolve().parent


def main():
    ms = (B.SRC / 'MANUSCRIPT_EN.md').read_text()
    title = re.search(r'^# (.+)$', ms, re.M).group(1).strip()
    sp = (B.SRC / 'SUPPLEMENTARY_EN.md').read_text()
    sp = re.sub(r'^# Supplementary material\s*\n', '', sp, count=1, flags=re.M)
    tex = (HERE / 'preamble.tex').read_text()
    tex += ('\\newunicodechar{δ}{\\ensuremath{\\delta}}\\newunicodechar{κ}{\\ensuremath{\\kappa}}\n')
    tex += ('\\setcounter{secnumdepth}{0}\n\\renewcommand{\\thefigure}{S\\ifnum\\value{figure}<10 0\\fi\\arabic{figure}}\n'
            '\\begin{document}\n\\begin{center}{\\Large Supplementary material for}\\\\[4pt]{\\Large\\bfseries '
            + B.pandoc(title).strip() + '}\\end{center}\n\n')
    tex += B.pandoc(B.tables(B.figures(sp))) + '\n\\end{document}\n'
    tex = tex.replace('\\hypertarget', '%\\hypertarget')

    def widths_after(tex, caption_start, widths):                          # caption paragraph precedes its table
        i = tex.index(caption_start)
        s_ = tex.index('\\begin{longtable}', i)
        e_ = tex.index('\\end{longtable}', s_)
        it = iter(widths)
        block = re.sub(r'\\real\{[0-9.]+\}', lambda m: '\\real{%.3f}' % next(it), tex[s_:e_], count=len(widths))
        return tex[:s_] + block + tex[e_:]
    tex = widths_after(tex, 'Table ST25. Scale demonstration.', (.07, .10, .07, .06, .26, .08, .11, .10, .08))

    def shrink(m):                                                         # wide tables: smaller type (as build_tex)
        cols = m.group(1).count('p{')
        size = '\\scriptsize' if cols >= 8 else ('\\footnotesize' if cols >= 6 else '\\small')
        return '{' + size + '\n' + m.group(0) + '}'
    tex = re.sub(r'\\begin\{longtable\}\[\]\{@\{\}(.*?)@\{\}\}.*?\\end\{longtable\}', shrink, tex, flags=re.S)
    (HERE / 'supp.tex').write_text(tex)
    print('supp.tex', len(tex))


if __name__ == '__main__':
    main()
