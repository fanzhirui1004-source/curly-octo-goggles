"""Build the Figure 2 / Figure 3 walkthrough (Chinese) to PDF: python3 build_fig23.py -> P1_图2图3拆解讲义.pdf
Figure parts are crops of figures/F01_method_overview.png and figures/F11_network_architecture.png in fig_parts/."""
import subprocess, pathlib, pypandoc
HERE = pathlib.Path(__file__).resolve().parent
MD, OUT = 'P1_FIG2_FIG3_CN.md', 'P1_图2图3拆解讲义'
PRE = r'''\documentclass[11pt,a4paper]{ctexart}
\usepackage[margin=1.9cm]{geometry}
\usepackage{amsmath,amssymb,booktabs,longtable,array,calc,hyperref,fancyhdr,graphicx}
\usepackage[table]{xcolor}
\hypersetup{colorlinks=true,linkcolor=black,urlcolor=blue}
\setlength{\parskip}{4pt}\linespread{1.25}
\pagestyle{fancy}\fancyhf{}\renewcommand{\headrulewidth}{0pt}
\lfoot{\footnotesize P1 图 2 与图 3 拆解讲义，2026 年 10 月 9 日}\rfoot{\footnotesize \thepage}
\providecommand{\tightlist}{\setlength{\itemsep}{2pt}\setlength{\parskip}{0pt}}
\ctexset{section/format=\Large\bfseries\raggedright,subsection/format=\large\bfseries\raggedright}\setcounter{secnumdepth}{0}
\begin{document}
'''
t = (HERE / MD).read_text()
title, body = t.split('\n', 1)
tex = pypandoc.convert_text(body, 'latex', format='markdown+tex_math_single_backslash+raw_attribute-auto_identifiers',
                            extra_args=['--wrap=preserve', '--shift-heading-level-by=-1'])
tex = tex.replace('\\begin{longtable}[]', '{\\small\\begin{longtable}[]').replace('\\end{longtable}', '\\end{longtable}}')
doc = PRE + '\\begin{center}{\\LARGE\\bfseries ' + title.lstrip('# ').strip() + '}\\end{center}\n\\vspace{6pt}\n' + tex + '\n\\end{document}\n'
(HERE / (OUT + '.tex')).write_text(doc)
for _ in range(2):
    subprocess.run(['lualatex', '-interaction=nonstopmode', OUT + '.tex'], cwd=HERE, capture_output=True)
log = (HERE / (OUT + '.log')).read_text(errors='replace')
print(OUT, 'errors', sum(l.startswith('!') for l in log.splitlines()), 'overfull', log.count('Overfull \\hbox'))
for ext in ('.aux', '.log', '.out', '.tex'):
    (HERE / (OUT + ext)).unlink(missing_ok=True)
