"""Revision 2 restructure of the P1 manuscript (one-off; refuses to run twice).

Reorders the sections of MANUSCRIPT_EN.md to: 1 Introduction, 2 Discrete cut cells (unchanged), 3 error relations (old 4),
4 NICE (old 3.2, 5.1, 5.2, 3.1, 3.3 + 5.4, 5.3), 5 numerical examples (old 6), 6 discussion (old 7), 7 conclusions (old 8);
renumbers the main-text equations (\\tag{n}) and every reference to them, and the section references, in the manuscript,
the appendices and the supplement.  Writes latex/restructure_r2.log with every changed line.
Usage: python3 restructure_r2.py        (run from anywhere; edits the three Markdown files in place)"""
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent
FILES = ['MANUSCRIPT_EN.md', 'APPENDICES_EN.md', 'SUPPLEMENTARY_EN.md']
MARK = '<!-- restructured r2 -->'

# old -> new main-text equation numbers
EQ = {1: 1, 2: 2, 3: 3, 4: 15, 5: 16, 6: 10, 7: 11, 8: 17, 9: 4, 10: 5, 11: 6, 12: 7, 13: 8, 14: 9,
      15: 12, 16: 13, 17: 14, 18: 18}
assert sorted(EQ.values()) == list(range(1, 19))
# old -> new section numbers (strings)
SEC = {'3': '4', '3.1': '4.4', '3.2': '4.1', '3.3': '4.5', '4': '3', '4.1': '3.1', '4.2': '3.2', '4.3': '3.3',
       '5': '4', '5.1': '4.2', '5.2': '4.3', '5.3': '4.6', '5.4': '4.5', '6': '5', '7': '6', '8': '7'}
SEC.update({f'6.{k}': f'5.{k}' for k in range(1, 12)})
SEC.update({f'7.{k}': f'6.{k}' for k in range(1, 6)})

log = []


def blocks(md):
    """Split into (heading, body) blocks at '## ' / '### ' headings; the text before the first heading is key ''."""
    out, key, buf = [], '', []
    for line in md.split('\n'):
        if re.match(r'^#{2,3} ', line):
            out.append((key, '\n'.join(buf))); key, buf = line, []
        else:
            buf.append(line)
    out.append((key, '\n'.join(buf)))
    return out


def reorder(md):
    B = dict(blocks(md))
    order = list(dict(blocks(md)).keys())
    get = lambda prefix: next(k for k in order if k.startswith(prefix))
    g = lambda prefix: B[get(prefix)]
    new = []
    new.append(('', B['']))
    new.append((get('## Abstract'), g('## Abstract')))
    new.append((get('## 1. Introduction'), g('## 1. Introduction')))
    for p in ('## 2. ', '### 2.1', '### 2.2', '### 2.3'):
        new.append((get(p), g(p)))
    # 3: error relations (old 4)
    new.append(('## 3. Error of an approximate extension in analysis and design', g('## 4. ')))
    new.append(('### 3.1. Variational energy error', g('### 4.1')))
    new.append(('### 3.2. Assembled compliance and energy participation', g('### 4.2')))
    new.append(('### 3.3. Field-based sensitivity and the complete design derivative', g('### 4.3')))
    # 4: NICE (old 3 intro, 3.2, 5.1, 5.2, 3.1, 3.3 + 5.4, 5.3)
    new.append(('## 4. Neural-initialised condensation with equilibrium correction', g('## 3. ')))
    new.append(('### 4.1. Variational condensed stiffness', g('### 3.2')))
    intro5 = g('## 5. ').strip('\n')
    new.append(('### 4.2. Error spectrum and polynomial relaxation', '\n' + intro5 + '\n' + g('### 5.1')))
    new.append(('### 4.3. Interior coarse correction', g('### 5.2')))
    new.append(('### 4.4. Geometry-conditioned multilevel displacement extension', g('### 3.1')))
    new.append(('### 4.5. Training directions and objective', g('### 3.3').rstrip('\n') + '\n\n' + g('### 5.4').strip('\n') + '\n\n'))
    new.append(('### 4.6. Energy-consistent operator application', g('### 5.3')))
    # 5: numerical examples (old 6)
    new.append(('## 5. Numerical examples', g('## 6. ')))
    for k in range(1, 12):
        key = get(f'### 6.{k}.')
        new.append((key.replace(f'### 6.{k}.', f'### 5.{k}.'), B[key]))
    new.append(('## 6. Discussion', g('## 7. ')))
    for k in range(1, 6):
        key = get(f'### 7.{k}.')
        new.append((key.replace(f'### 7.{k}.', f'### 6.{k}.'), B[key]))
    new.append(('## 7. Conclusions', g('## 8. ')))
    for p in ('## Code and data', '## References', '## Supplementary material'):
        new.append((get(p), g(p)))
    used = {k for k, _ in new}
    missing = [k for k in order if k not in used and k not in ('## 3. Geometry-conditioned neural displacement extension',
               '## 4. Extension error and mechanical response', '## 5. Internal equilibrium correction', '## 6. Numerical examples',
               '## 7. Discussion', '## 8. Conclusions') and not re.match(r'### [34567]\.', k)]
    assert not missing, missing
    return '\n'.join((k + '\n' + b) if k else b for k, b in new)


def renumber_eqs(text, tags):
    """Equation references (and \\tag{n} if tags) through placeholders."""
    ph = lambda n: f'\x00EQ{EQ[int(n)]}\x00'
    if tags:
        text = re.sub(r'\\tag\{(\d{1,2})\}', lambda m: '\\tag{' + ph(m.group(1)) + '}', text)
    lead = r'(Eqs?\. |Equations? |identity |form |functional )'
    # chains: Eq. (13) and (14); Eqs. (16)–(17); Eqs. (13), (14) and (18)
    def chain(m):
        return m.group(1) + re.sub(r'\((\d{1,2})\)', lambda mm: '(' + ph(mm.group(1)) + ')', m.group(2))
    text = re.sub(lead + r'(\(\d{1,2}\)(?:(?:,? and |–|, )\(\d{1,2}\))*)', chain, text)
    text = re.sub(r'\(Eq\. (\d{1,2})\)', lambda m: '(Eq. ' + ph(m.group(1)) + ')', text)
    text = re.sub(r'Eq\. (\d{1,2})(?=[ ,;)])', lambda m: 'Eq. ' + ph(m.group(1)), text)
    return text.replace('\x00EQ', '').replace('\x00', '')


def renumber_secs(text):
    def span(m):
        s = m.group(0)
        return re.sub(r'\d+(?:\.\d+)?', lambda mm: '\x00' + SEC.get(mm.group(0), mm.group(0)) + '\x00', s)
    text = re.sub(r'Sections? \d+(?:\.\d+)?(?:(?:, |,? and | to |–)\d+(?:\.\d+)?)*', span, text)
    return text.replace('\x00', '')


def main():
    ms = (SRC / 'MANUSCRIPT_EN.md').read_text()
    if MARK in ms:
        raise SystemExit('already restructured')
    for f in FILES:
        old = (SRC / f).read_text()
        new = reorder(old) if f == 'MANUSCRIPT_EN.md' else old
        new = renumber_eqs(new, tags=(f == 'MANUSCRIPT_EN.md'))
        new = renumber_secs(new)
        if f == 'MANUSCRIPT_EN.md':
            new = new.rstrip('\n') + '\n\n' + MARK + '\n'
        (SRC / f).write_text(new)
        o, n = old.split('\n'), new.split('\n')
        log.append(f'=== {f}: {len(o)} -> {len(n)} lines')
        # line-level report of changed references (same text order for the two non-reordered files)
        if f != 'MANUSCRIPT_EN.md':
            for i, (a, b) in enumerate(zip(o, n)):
                if a != b:
                    log.append(f'{i + 1}: {b[:300]}')
    # manuscript: report every line containing a reference, after the change
    new = (SRC / 'MANUSCRIPT_EN.md').read_text().split('\n')
    for i, line in enumerate(new):
        if re.search(r'Eqs?\. |Equations? \(|Sections? \d|\\tag\{|identity \(|form \(|functional \(', line):
            for m in re.finditer(r'.{0,60}(Eqs?\. \(?\d|Equations? \(\d|Sections? \d[\d.]*|\\tag\{\d+\}|identity \(\d|form \(\d|functional \(\d).{0,40}', line):
                log.append(f'M{i + 1}: {m.group(0)}')
    (HERE / 'restructure_r2.log').write_text('\n'.join(log) + '\n')
    print('\n'.join(log[:40])); print(f'... {len(log)} log lines -> latex/restructure_r2.log')


if __name__ == '__main__':
    main()
