"""Build Word versions of the P1 manuscript and supplement with pandoc (via pypandoc).

  python3 build_docx.py en   -> P1_main_EN.docx, P1_supp_EN.docx   (from MANUSCRIPT_EN.md, APPENDICES_EN.md, SUPPLEMENTARY_EN.md)
  python3 build_docx.py cn   -> P1_main_CN.docx, P1_supp_CN.docx   (from the *_CN.md translations)

Main document order as in the PDF: title, abstract, keywords, main text (incl. code and data availability), appendices,
references. Images are inserted without pandoc's implicit captions; the bold 'Figure n.' / '图 n.' paragraph that follows
each image is the caption. Inline and display TeX math become Word equations."""
import re, sys, pypandoc
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent
FMT = 'markdown+tex_math_single_backslash-auto_identifiers-smart-implicit_figures'


def build(md, out):
    md = re.sub(r'\\tag\{([^}]*)\}', r'\\qquad(\1)', md)              # Word equations have no \tag
    md = re.sub(r'\{\\rm ([^{}]*)\}', r'\\mathrm{\1}', md)             # {\rm x} -> \mathrm{x} (texmath)
    pypandoc.convert_text(md, 'docx', format=FMT, outputfile=str(out),
                          extra_args=['--resource-path', str(SRC), '--wrap=preserve'])
    print(out.name, out.stat().st_size)


def main(lang):
    sfx = 'EN' if lang == 'en' else 'CN'
    ms = (SRC / f'MANUSCRIPT_{sfx}.md').read_text()
    ap = (SRC / f'APPENDICES_{sfx}.md').read_text()
    sp = (SRC / f'SUPPLEMENTARY_{sfx}.md').read_text()
    ref_h = '## References' if lang == 'en' else '## 参考文献'
    i = ms.index(ref_h)
    refs = ms[i:]
    j = refs.find('## Supplementary material') if lang == 'en' else refs.find('## 补充材料')
    refs = refs[:j] if j > 0 else refs
    ap = re.sub(r'^# .*\n', '', ap, count=1, flags=re.M)              # appendices file title: one document title only
    build(ms[:i] + '\n\n' + ap + '\n\n' + refs, HERE / f'P1_main_{sfx}.docx')
    build(sp, HERE / f'P1_supp_{sfx}.docx')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'en')
