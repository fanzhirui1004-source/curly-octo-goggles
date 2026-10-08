"""Advisor-review build: the same text as main/supp (EN/CN) without journal-specific front and end matter.

Run after build_tex.py, build_supp_tex.py and build_tex_cn.py. From main.tex, supp.tex, main_cn.tex and supp_cn.tex it
  - removes the \\journal line and replaces the 'Preprint submitted to ...' footer by a draft note with the date;
  - removes the author block (red placeholders) of the main documents;
  - removes the declarations set after the main text (CRediT, competing interest, acknowledgements, AI use);
and compiles advisor_*.tex here (pdflatex for EN, lualatex for CN, twice each), copying the PDFs to ../advisor_review/.
Usage: python3 build_advisor.py [date text]"""
import re, shutil, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / 'advisor_review'
DATE_EN = sys.argv[1] if len(sys.argv) > 1 else '8 October 2026'
DATE_CN = '2026 年 10 月 8 日'
JOBS = [('main.tex', 'pdflatex', 'Draft for internal review', DATE_EN, 'P1_NICE_advisor_review_EN.pdf'),
        ('supp.tex', 'pdflatex', 'Supplementary material, draft for internal review', DATE_EN, 'P1_NICE_advisor_review_EN_supplement.pdf'),
        ('main_cn.tex', 'lualatex', '内部审阅稿', DATE_CN, 'P1_NICE_advisor_review_CN.pdf'),
        ('supp_cn.tex', 'lualatex', '补充材料，内部审阅稿', DATE_CN, 'P1_NICE_advisor_review_CN_supplement.pdf')]


def advisor(tex, note, date):
    tex, n = re.subn(r'^\\journal\{[^}]*\}\n', '', tex, flags=re.M)
    assert n == 1, 'journal line'
    tex = tex.replace('\\begin{document}', '\\makeatletter\\gdef\\@elsarticlemyfooteralign{L}\\gdef\\@elsarticlemyfooter{' + note + ('，' if date.endswith('日') else ', ') + date + '}\\makeatother\n\\begin{document}', 1)
    i = tex.find('\\author[')
    if i >= 0:                                                             # author block up to the abstract
        j = tex.index('\\begin{abstract}', i)
        tex = tex[:i] + tex[j:]
    for start in ('\\section*{CRediT authorship contribution statement}', '\\section*{作者贡献声明}'):
        i = tex.find(start)
        if i >= 0:
            j = tex.index('\\appendix', i)
            tex = tex[:i] + tex[j:]
    assert 'textcolor{red}' not in tex, 'placeholder left'
    return tex


def main():
    OUT.mkdir(exist_ok=True)
    for src, engine, note, date, pdf in JOBS:
        job = 'advisor_' + src[:-4]
        (HERE / (job + '.tex')).write_text(advisor((HERE / src).read_text(), note, date))
        for _ in range(2):
            subprocess.run([engine, '-interaction=nonstopmode', job + '.tex'], cwd=HERE, capture_output=True)
        log = (HERE / (job + '.log')).read_text(errors='replace')
        errs = [l for l in log.splitlines() if l.startswith('!')]
        print(job, 'errors:', len(errs), 'undefined refs:', log.count('undefined'))
        shutil.copy(HERE / (job + '.pdf'), OUT / pdf)
        for ext in ('.aux', '.log', '.out', '.toc', '.tex', '.pdf'):
            (HERE / (job + ext)).unlink(missing_ok=True)


if __name__ == '__main__':
    main()
