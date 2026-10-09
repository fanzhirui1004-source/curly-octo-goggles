"""Assemble the R5 plain-language rewrite into the six source files and run consistency checks.

Usage: python3 assemble_r5.py [--write]
Without --write, the assembled texts go to scratchpad/r5_out/ only; with --write, they replace the files in docs/paper_p1.
"""
import re, sys, json, collections
from pathlib import Path

P1 = Path('/home/user/curly-octo-goggles/docs/paper_p1')
SEC = P1 / 'rewrite_plain' / 'sections'
OUT = Path(__file__).resolve().parent / 'r5_out'
OUT.mkdir(exist_ok=True)

MAIN_UNITS = ['U00', 'U01', 'U02', 'U03', 'U04', 'U05', 'U06', 'U07', 'U08', 'U09', 'U10', 'U11']
APPX = [('A1', 1, 203), ('A2', 204, 339), ('A3', 340, 596)]
SUPP = [('S1', 1, 242), ('S2', 243, 478), ('S3', 479, 626), ('S4', 627, 887)]


def read(p):
    return Path(p).read_text(encoding='utf8')


def unit(uid, lang):
    t = read(SEC / f'{uid}_{lang}.md').strip('\n')
    return t


def assemble_main(lang):
    old = read(P1 / f'MANUSCRIPT_{lang}.md').split('\n')
    refs_head = '## References' if lang == 'EN' else '## 参考文献'
    i = next(k for k, l in enumerate(old) if l.strip() == refs_head)
    body = '\n\n'.join(unit(u, lang) for u in MAIN_UNITS)
    return body + '\n\n' + '\n'.join(old[i:]).rstrip('\n') + '\n'


def splice(fname, chunks, lang):
    old = read(P1 / f'{fname}_{lang}.md').split('\n')
    out, pos = [], 1
    for cid, a, b in chunks:
        assert pos == a, (fname, cid, pos, a)
        new = read(SEC / f'{cid}_{lang}.md').rstrip('\n').split('\n')
        out += new
        pos = b + 1
    out += old[pos - 1:]
    return '\n'.join(out).rstrip('\n') + '\n'


def supp_additions(lang):
    """Collect '## SUPPLEMENT ADDITIONS' blocks from the unit notes (EN and CN parts are separated by headings that
    contain 'EN' / 'CN' or 'English' / 'Chinese'; we return the whole block per unit for manual placement)."""
    blocks = []
    for u in MAIN_UNITS:
        f = SEC / f'{u}_NOTES.md'
        if not f.exists():
            continue
        t = read(f)
        m = re.search(r'^## SUPPLEMENT ADDITIONS\s*$(.*?)(?=^## (?!#)|\Z)', t, re.M | re.S)
        if m and m.group(1).strip():
            blocks.append((u, m.group(1).strip()))
    return blocks


# ----------------------------------------------------------------------------------------------------------- checks
NUM = re.compile(r'(?<![\w.])[-−]?\d[\d,]*(?:\.\d+)?(?:[eE][-−+]?\d+)?')


def numbers(t):
    t = re.sub(r'\]\([^)]*\)', ']', t)                    # drop link targets (DOIs)
    return [n.replace(',', '').replace('−', '-') for n in NUM.findall(t)]


def check_main(en, cn):
    rep = []
    h_en = [l for l in en.split('\n') if l.startswith('#')]
    h_cn = [l for l in cn.split('\n') if l.startswith('#')]
    rep.append(f'headings EN {len(h_en)} CN {len(h_cn)}')
    for a, b in zip(h_en, h_cn):
        na = re.match(r'#+ ([0-9.]+)', a); nb = re.match(r'#+ ([0-9.]+)', b)
        if (na and nb and na.group(1) != nb.group(1)) or (a.count('#') != b.count('#')):
            rep.append(f'  heading mismatch: {a!r} / {b!r}')
    tags = [int(x) for x in re.findall(r'\\tag\{(\d+)\}', en.split('## References')[0])]
    rep.append(f'tags EN {tags}')
    tags_cn = [int(x) for x in re.findall(r'\\tag\{(\d+)\}', cn.split('## 参考文献')[0])]
    if tags != tags_cn:
        rep.append(f'  tags CN differ: {tags_cn}')
    if tags != list(range(1, len(tags) + 1)):
        rep.append('  tags not sequential')
    body = en.split('## References')[0]
    for m in re.finditer(r'Eqs?\. \((\d+)\)', body):
        if int(m.group(1)) > max(tags):
            rep.append(f'  dangling Eq. ({m.group(1)})')
    secs = set(re.findall(r'^#+ ([0-9]+(?:\.[0-9]+)?)\. ', body, re.M))
    for m in re.finditer(r'Sections? ([0-9]+(?:\.[0-9]+)?)', body):
        if m.group(1) not in secs:
            rep.append(f'  dangling Section {m.group(1)}')
    props = re.findall(r'\*\*Proposition (\d+)', body)
    rep.append(f'propositions in order: {props}')
    for m in re.finditer(r'Propositions? (\d+)', body):
        if m.group(1) not in props:
            rep.append(f'  dangling Proposition {m.group(1)}')
    tabs = re.findall(r'^\*\*Table (\d+)\.', body, re.M)
    figs = re.findall(r'^\*\*Figure (\d+)\.', body, re.M)
    rep.append(f'tables {tabs}  figures {figs}')
    for m in re.finditer(r'Tables? (\d+)', body):
        if m.group(1) not in tabs:
            rep.append(f'  dangling Table {m.group(1)}')
    # abstract words
    ab = re.search(r'^## Abstract\s*\n(.*?)\n\*\*Keywords', en, re.S | re.M).group(1)
    rep.append(f'abstract words {len(ab.split())}')
    # old terms
    OLD_EN = ['box face', 'box-face', 'cut band', 'cut-band', 'energy share', 'trial field', 'improvab', 'division of',
              'geometry-conditioned', 'field-based', 'consistent traction', 'work-conjugate', 'nonexpansive', 'latent',
              'slot', 'stratum', 'strata', 'target cell', 'Uncorrected', 'Smoothing-trained', 'extension']
    OLD_CN = ['胞元边界面', '切割带', '能量份额', '试探场', '可改进性', '分工', '几何条件化', '场基', '一致面力', '功共轭',
              '非扩张', '潜空间', '槽位', '分层', '目标胞元', '未校正', '平滑训练', '延拓', '保留自由度', '保留位移']
    for w in OLD_EN:
        k = len(re.findall(re.escape(w), body, re.I))
        if k:
            rep.append(f'  old EN term {w!r}: {k}')
    bcn = cn.split('## 参考文献')[0]
    for w in OLD_CN:
        k = bcn.count(w)
        if k:
            rep.append(f'  old CN term {w!r}: {k}')
    # numbers EN vs CN by section
    def split_sec(t):
        parts = re.split(r'^(#{2,3} [0-9]+(?:\.[0-9]+)?\. .*)$', t, flags=re.M)
        return parts
    pe, pc = split_sec(body), split_sec(bcn)
    if len(pe) == len(pc):
        for k in range(1, len(pe), 2):
            ne = collections.Counter(numbers(pe[k + 1])); nc = collections.Counter(numbers(pc[k + 1]))
            d1 = ne - nc; d2 = nc - ne
            if d1 or d2:
                rep.append(f'  numbers differ in {pe[k][:60]!r}: EN-only {dict(d1)} CN-only {dict(d2)}')
            pa = len([x for x in pe[k + 1].split('\n\n') if x.strip()]); pb = len([x for x in pc[k + 1].split('\n\n') if x.strip()])
            if pa != pb:
                rep.append(f'  paragraph count differs in {pe[k][:60]!r}: EN {pa} CN {pb}')
    else:
        rep.append(f'  section split mismatch EN {len(pe)} CN {len(pc)}')
    # sentence length
    sents = [s for s in re.split(r'(?<=[.;:])\s+(?=[A-Z])', re.sub(r'\\\[.*?\\\]', '', body, flags=re.S)) if s.strip() and not s.startswith(('|', '!', '#'))]
    long = [s for s in sents if len(s.split()) > 40]
    rep.append(f'EN sentences >40 words: {len(long)}')
    # citations
    cited = set(re.findall(r'\[([^\]]+?\(\d{4}[a-z]?\))\]\(', body))
    refs = en.split('## References')[1]
    rep.append(f'distinct cited works {len(cited)}')
    return rep


if __name__ == '__main__':
    en, cn = assemble_main('EN'), assemble_main('CN')
    (OUT / 'MANUSCRIPT_EN.md').write_text(en); (OUT / 'MANUSCRIPT_CN.md').write_text(cn)
    for f, ch in (('APPENDICES', APPX), ('SUPPLEMENTARY', SUPP)):
        for lang in ('EN', 'CN'):
            if all((SEC / f'{c}_{lang}.md').exists() for c, _, _ in ch):
                txt = splice(f, ch, lang)
                if f == 'SUPPLEMENTARY':                    # new Note S8 goes before the supplementary figures
                    head = '## Supplementary figures' if lang == 'EN' else '## 补充图'
                    s8 = read(SEC / f'S8_{lang}.md').rstrip('\n')
                    assert txt.count('\n' + head + '\n') == 1, head
                    txt = txt.replace('\n' + head + '\n', '\n' + s8 + '\n\n' + head + '\n')
                (OUT / f'{f}_{lang}.md').write_text(txt)
    print('\n'.join(check_main(en, cn)))
    adds = supp_additions('EN')
    print('supplement additions from', [u for u, _ in adds])
    if '--write' in sys.argv:
        for f in OUT.glob('*.md'):
            (P1 / f.name).write_text(f.read_text())
        print('written to', P1)
