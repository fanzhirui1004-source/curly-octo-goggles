"""Build the advisor notes (Chinese) to PDF: python3 build_notes.py  ->  ../P1_后续工作安排.pdf, ../P1_NICE与PIML对比.pdf"""
import subprocess, pathlib, pypandoc
HERE = pathlib.Path(__file__).resolve().parent
DOCS = [('P1_STUDY_GUIDE_CN.md', 'P1_通读讲义')]
PRE = r'''\documentclass[11pt,a4paper]{ctexart}
\usepackage[margin=1.9cm]{geometry}
\usepackage{amsmath,amssymb,booktabs,longtable,array,calc,hyperref,fancyhdr}
\usepackage[table]{xcolor}
\hypersetup{colorlinks=true,linkcolor=black,urlcolor=blue}
\setlength{\parskip}{4pt}\linespread{1.25}
\pagestyle{fancy}\fancyhf{}\renewcommand{\headrulewidth}{0pt}
\lfoot{\footnotesize P1 通读讲义，2026 年 10 月 6 日}\rfoot{\footnotesize \thepage}
\providecommand{\tightlist}{\setlength{\itemsep}{2pt}\setlength{\parskip}{0pt}}
\ctexset{section/format=\Large\bfseries\raggedright}\setcounter{secnumdepth}{0}
\begin{document}
'''
for md, out in DOCS:
    t = (HERE / md).read_text()
    title, body = t.split('\n', 1)
    tex = pypandoc.convert_text(body, 'latex', format='markdown+tex_math_single_backslash-auto_identifiers',
                                extra_args=['--wrap=preserve', '--shift-heading-level-by=-1'])
    tex = tex.replace('\\begin{longtable}[]', '{\\small\\begin{longtable}[]').replace('\\end{longtable}', '\\end{longtable}}')
    doc = PRE + '\\begin{center}{\\LARGE\\bfseries ' + title.lstrip('# ').strip() + '}\\end{center}\n\\vspace{6pt}\n' + tex + '\n\\end{document}\n'
    (HERE / (out + '.tex')).write_text(doc)
    for _ in range(2):
        subprocess.run(['lualatex', '-interaction=nonstopmode', out + '.tex'], cwd=HERE, capture_output=True)
    log = (HERE / (out + '.log')).read_text(errors='replace')
    print(out, 'errors', sum(l.startswith('!') for l in log.splitlines()))
    (HERE / (out + '.pdf')).replace(HERE / (out + '.pdf'))
    for ext in ('.aux', '.log', '.out', '.tex'):
        (HERE / (out + ext)).unlink(missing_ok=True)
