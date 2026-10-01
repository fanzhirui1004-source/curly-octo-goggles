"""Revision 3 restructure of the P1 manuscript (one-off; refuses to run twice).

Main text: the residual-work identity, old Eq. (18) in Section 4.6, moves to the end of Section 3.2 and becomes Eq. (8)
(old 8-17 -> 9-18); Section 5.7 is merged into Section 5.6 and 5.8-5.11 become 5.7-5.10; the dangling reference to
Section 6.5 becomes 6.4; the r2 marker is removed.
Appendices: Appendix J is dissolved: J.1 -> B.1 (assumptions), J.2 -> end of B.2, J.3 -> C.1, J.6 -> C.2, J.8 -> D.1,
J.7 -> G.4 (its last paragraph -> end of B.3), J.4 -> H.1, J.5 -> H.2, J.9 -> Supplementary Note S8 (merged with the
definition of example 1); old B.1 -> B.3; Appendix I is retitled and loses its last paragraph; equation tags and every
equation/appendix reference are renumbered in all three files.
Supplement: section R2 is deleted.
Writes latex/restructure_r3.log.  Usage: python3 restructure_r3.py"""
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent
FILES = ['MANUSCRIPT_EN.md', 'APPENDICES_EN.md', 'SUPPLEMENTARY_EN.md']
log = []

# main-text equations: old -> new
EQ = {n: n for n in range(1, 8)}
EQ.update({n: n + 1 for n in range(8, 18)})
EQ[18] = 8
assert sorted(EQ.values()) == list(range(1, 19))
# main-text sections: old -> new
SEC = {'5.7': '5.6', '5.8': '5.7', '5.9': '5.8', '5.10': '5.9', '5.11': '5.10', '6.5': '6.4'}
# appendix equation tags: old -> new
AEQ = {'J.1': 'B.7', 'B.7': 'B.8', 'B.8': 'B.9', 'B.9': 'B.10', 'J.2': 'C.5', 'J.9': 'D.6', 'J.8': 'G.4',
       'J.3': 'H.7', 'J.4': 'H.8', 'J.5': 'H.9', 'J.6': 'H.10', 'J.7': 'H.11', 'J.10': 'S8.1'}
# appendix sections: old -> new
ASEC = {'J.1': 'Appendix B.1', 'J.2': 'Appendix B.2', 'J.3': 'Appendix C.1', 'J.4': 'Appendix H.1',
        'J.5': 'Appendix H.2', 'J.6': 'Appendix C.2', 'J.7': 'Appendix G.4', 'J.8': 'Appendix D.1',
        'J.9': 'Supplementary Note S8', 'B.1': 'Appendix B.3'}


def take(text, start, end, what):
    """Remove text[start_marker : end_marker) and return (rest, removed)."""
    i = text.index(start)
    j = text.index(end, i + len(start)) if end is not None else len(text)
    log.append(f'-- moved {what}: {len(text[i:j])} chars')
    return text[:i] + text[j:], text[i:j]


def insert_after(text, anchor, block, what):
    i = text.index(anchor) + len(anchor)
    log.append(f'-- inserted {what} after {anchor[:60]!r}')
    return text[:i] + block + text[i:]


def insert_before(text, anchor, block, what):
    i = text.index(anchor)
    log.append(f'-- inserted {what} before {anchor[:60]!r}')
    return text[:i] + block + text[i:]


def sub1(text, old, new, what):
    assert text.count(old) == 1, (what, text.count(old))
    log.append(f'-- {what}')
    return text.replace(old, new)


# ------------------------------------------------------------------------------------------------ renumbering
def renumber_main_eqs(text, tags):
    ph = lambda n: f'\x00EQ{EQ[int(n)]}\x00'
    if tags:
        text = re.sub(r'\\tag\{(\d{1,2})\}', lambda m: '\\tag{' + ph(m.group(1)) + '}', text)
    lead = r'(Eqs?\. |Equations? |identity |form |functional )'

    def chain(m):
        return m.group(1) + re.sub(r'\((\d{1,2})\)', lambda mm: '(' + ph(mm.group(1)) + ')', m.group(2))
    text = re.sub(lead + r'(\(\d{1,2}\)(?:(?:,? and |–|, )\(\d{1,2}\))*)', chain, text)
    text = re.sub(r'\(Eqs?\. (\d{1,2})\)', lambda m: m.group(0)[:m.group(0).index(' ') + 1] + ph(m.group(1)) + ')', text)
    text = re.sub(r'Eq\. (\d{1,2})(?=[ ,;)])', lambda m: 'Eq. ' + ph(m.group(1)), text)
    return text.replace('\x00EQ', '').replace('\x00', '')


def renumber_main_secs(text):
    def span(m):
        return re.sub(r'\d+\.\d+', lambda mm: SEC.get(mm.group(0), mm.group(0)), m.group(0))
    return re.sub(r'Sections? \d+(?:\.\d+)?(?:(?:, |,? and | to |–)\d+(?:\.\d+)?)*', span, text)


def renumber_app(text):
    # equation tags and parenthesised / Eq.-prefixed equation references, single pass
    text = re.sub(r'\\tag\{([A-J]\.\d+)\}', lambda m: '\\tag{' + AEQ.get(m.group(1), m.group(1)) + '}', text)
    text = re.sub(r'\(([A-J]\.\d+)\)', lambda m: '(' + AEQ.get(m.group(1), m.group(1)) + ')', text)
    text = re.sub(r'(Eqs?\. )([A-J]\.\d+)(?=[ ,;)])', lambda m: m.group(1) + AEQ.get(m.group(2), m.group(2)), text)
    # appendix section references
    text = text.replace('Appendices J.8 and J.9', 'Appendix D.1 and Supplementary Note S8')
    text = re.sub(r'Appendix ([BJ]\.\d+)', lambda m: ASEC.get(m.group(1), 'Appendix ' + m.group(1)), text)
    return text


# ------------------------------------------------------------------------------------------------ main text
def main_text(ms):
    # 1. residual-work identity: 4.6 -> end of 3.2
    ms, blk = take(ms, 'At the assembled level, the energy relation also separates approximation error from incomplete equilibrium.',
                   '\n## 5. Numerical examples', 'residual-work identity (old Eq. 18)')
    blk = blk.replace('At the assembled level, the energy relation also separates approximation error from incomplete equilibrium.',
                      'The same energy relation separates the operator error from the error of an incomplete assembled solve.')
    blk = blk.rstrip('\n') + '\n\n'
    ms = re.sub(r'\n{3,}## 5\. Numerical examples', '\n\n## 5. Numerical examples', ms)
    anchor = 'Small participation can thus attenuate a substructure\'s effect on compliance even when its local field remains inaccurate (Appendix J.3).\n\n'
    ms = insert_after(ms, anchor, blk, 'residual-work identity at the end of 3.2')
    # 2. Section 5.7 into 5.6
    ms = sub1(ms, '### 5.6. Compliance and local sensitivity after assembly',
              '### 5.6. Compliance, local sensitivity and energy share after assembly', 'retitle 5.6')
    ms = sub1(ms, '### 5.7. Energy participation\n\n', '', 'drop heading 5.7')
    for old, new in (('5.8', '5.7'), ('5.9', '5.8'), ('5.10', '5.9'), ('5.11', '5.10')):
        ms = sub1(ms, f'### {old}. ', f'### {new}. ', f'heading {old} -> {new}')
    # 3. r2 marker
    ms = sub1(ms, '\n\n<!-- restructured r2 -->\n', '\n', 'drop r2 marker')
    return ms


# ------------------------------------------------------------------------------------------------ appendices
def appendices(ap):
    # cut Appendix J into its subsections
    ap, J = take(ap, '## Appendix J. Further variational and mechanical analysis', None, 'Appendix J')
    ap = ap.rstrip('\n') + '\n'
    parts = re.split(r'\n(?=### J\.\d\. )', J)
    subs = {re.match(r'### (J\.\d)\.', p).group(1): p for p in parts[1:]}
    assert sorted(subs) == [f'J.{k}' for k in range(1, 10)], sorted(subs)
    body = lambda k: subs[k].split('\n', 1)[1].strip('\n')
    # J.7: last paragraph (softness of eps = delta^2 kappa vs Jacobi spectrum) goes to B.3
    j7 = body('J.7')
    cut = j7.index('The identity \\(\\varepsilon=\\delta^2\\kappa\\) in Eq. (B.8)')
    j7_main, j7_last = j7[:cut].rstrip('\n'), j7[cut:].strip('\n')
    # J.2: opening sentence now inside Appendix B
    j2 = body('J.2').replace(
        'Appendix B derives \\(\\widehat S-S=H^TAH\\) and \\(r_I=(KFq)_I=AHq\\) from the expansion in Eq. (B.1), in which internal stationarity \\(J_IKE=0\\) removes both cross terms.',
        'The expansion (B.1), in which internal stationarity \\(J_IKE=0\\) removes both cross terms, gives \\(\\widehat S-S=H^TAH\\) and \\(r_I=(KFq)_I=AHq\\).')
    assert j2 != body('J.2')
    # Appendix B: B.1 assumptions, B.2 identity and kernel (+ J.2), B.3 old B.1 (+ last paragraph of J.7)
    ap = sub1(ap, '## Appendix B. Variational identity, rigid kernel, and directional norms\n\n',
              '## Appendix B. Variational identity, rigid kernel, and directional norms\n\n### B.1. Assumptions\n\n' + body('J.1') +
              '\n\n### B.2. Variational identity, complete transpose and rigid kernel\n\n', 'B.1 assumptions (J.1) and B.2 heading')
    ap = sub1(ap, '### B.1. Displacement magnitude and directional stiffness',
              '\x00J2\x00### B.3. Displacement magnitude and directional stiffness', 'old B.1 -> B.3')
    ap = ap.replace('\x00J2\x00', j2 + '\n\n')
    ap = insert_before(ap, '## Appendix C. Assembly and compliance ordering', j7_last + '\n\n', 'J.7 last paragraph at the end of B.3')
    # Appendix C: C.1 (J.3), C.2 (J.6)
    ap = insert_before(ap, '## Appendix D. Polynomial smoothing and coarse projections',
                       '### C.1. Compliance error as the reconstructed error energy\n\n' + body('J.3') +
                       '\n\n### C.2. Inexact assembled solves\n\n' + body('J.6') + '\n\n', 'C.1 (J.3), C.2 (J.6)')
    # Appendix D: D.1 (J.8)
    ap = insert_before(ap, '## Appendix E. Transpose of the complete extension',
                       '### D.1. Corrections, orderings and approximate coarse inverses\n\n' + body('J.8') + '\n\n', 'D.1 (J.8)')
    # Appendix G: G.4 (J.7 without its last paragraph)
    ap = insert_before(ap, '## Appendix H. Sensitivity identities and design intervals',
                       '### G.4. Direction coverage, symmetry and spectra\n\n' + j7_main + '\n\n', 'G.4 (J.7)')
    # Appendix H: H.1 (J.4), H.2 (J.5)
    ap = insert_before(ap, '## Appendix I. Residual lower diagnostics',
                       '### H.1. Sensitivity error bounds\n\n' + body('J.4') +
                       '\n\n### H.2. The complete design derivative and the sign of the thickness derivative\n\n' + body('J.5') + '\n\n',
                       'H.1 (J.4), H.2 (J.5)')
    # Appendix I
    ap = sub1(ap, '## Appendix I. Residual lower diagnostics', '## Appendix I. A computable lower bound on the energy error', 'retitle I')
    ap = sub1(ap, '\n\nThe reduced-boundary ablation of Section 5.8 restricts the retained box-face displacements instead of the interior; its restricted Galerkin system and conditions are given in Supplementary Note S4.',
              '', 'drop last paragraph of I')
    ap = re.sub(r'\n{3,}', '\n\n', ap).rstrip('\n') + '\n'
    return ap, body('J.9')


# ------------------------------------------------------------------------------------------------ supplement
def supplement(sp, j9):
    sp, _ = take(sp, '## R2. Further diagnostic observations', '## Table ST01.', 'supplement R2 (deleted)')
    s8_head = '## Supplementary Note S8. Definition of the illustrative matrix example\n\n'
    i = sp.index(s8_head)
    j = sp.index('## Supplementary Note S9.', i)
    old = sp[i + len(s8_head):j].strip('\n')
    old = old.replace('The re-equilibration example (example 1 of Appendix J.9) uses', 'Example 1 uses')
    old = old.replace('in examples 2 and 3 of Appendix J.9', 'in examples 2 and 3 above')
    j9 = j9.replace('Supplementary Note S8 gives the matrices, parameter sequence, and saved slope values for this illustrative algebraic example.',
                    'Section S8.1 gives the matrices, parameter sequence, and saved slope values.')
    new = ('## Supplementary Note S8. Illustrative matrix examples\n\n' + j9 + '\n\n### S8.1. Matrices of example 1\n\n' + old + '\n\n')
    log.append('-- Note S8 = J.9 examples + definition of example 1 (S8.1)')
    return sp[:i] + new + sp[j:]


def main():
    ms = (SRC / 'MANUSCRIPT_EN.md').read_text()
    if '### 5.7. Energy participation' not in ms or '## Appendix J.' not in (SRC / 'APPENDICES_EN.md').read_text():
        raise SystemExit('already restructured (r3)')
    old = {f: (SRC / f).read_text() for f in FILES}
    ms = main_text(old['MANUSCRIPT_EN.md'])
    ap, j9 = appendices(old['APPENDICES_EN.md'])
    sp = supplement(old['SUPPLEMENTARY_EN.md'], j9)
    new = {'MANUSCRIPT_EN.md': ms, 'APPENDICES_EN.md': ap, 'SUPPLEMENTARY_EN.md': sp}
    for f in FILES:
        t = new[f]
        t = renumber_main_eqs(t, tags=(f == 'MANUSCRIPT_EN.md'))
        t = renumber_main_secs(t)
        t = renumber_app(t)
        new[f] = t
        (SRC / f).write_text(t)
        log.append(f'=== {f}: {len(old[f].splitlines())} -> {len(t.splitlines())} lines')
    # every reference after the change, for inspection
    for f in FILES:
        for i, line in enumerate(new[f].split('\n')):
            for m in re.finditer(r'.{0,50}(Eqs?\. \(?[A-JS]?\.?\d|Equations? \(\d|Sections? \d[\d.]*|\\tag\{[^}]+\}|identity \(\d|form \(\d|functional \(\d|Appendi(?:x|ces) [A-J][.\d]*|Note S8).{0,30}', line):
                log.append(f'{f[0]}{i + 1}: {m.group(0)}')
    (HERE / 'restructure_r3.log').write_text('\n'.join(log) + '\n')
    print('\n'.join(l for l in log if l.startswith(('--', '==='))))
    print(f'{len(log)} log lines -> latex/restructure_r3.log')


if __name__ == '__main__':
    main()
