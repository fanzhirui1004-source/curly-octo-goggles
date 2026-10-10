"""R6 check: compare edited units (EN and CN) with their versions at the R6 baseline commit.

Usage: python3 check_r6.py U04 [U05 A1 S8 ...]

For each unit and language it reports
  - paragraph counts (must be unchanged, and EN must equal CN);
  - numbers, citation links and inline/display mathematics that were added or removed (must be empty, apart from
    enumeration markers, which are ignored, and changes listed in the unit's R6 notes);
  - headings, image lines, caption starts, table rows and equation tags (must be unchanged);
  - occurrences of colloquial or banned words (old -> new);
  - sentence statistics (for information only).
"""
import collections
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEC = HERE / 'sections'
REPO = HERE.parent.parent.parent
BASE = 'cdd11e4'                                        # last commit before the R6 revision of the units

NUM = re.compile(r'(?<![\w.])[-−]?\d[\d,]*(?:\.\d+)?(?:[eE][-−+]?\d+)?')
CN_WORDS = ['做法', '有的', '却', '多得多', '反过来', '那部分', '得不偿失', '发展得', '有多', '为何', '什么', '怎么',
            '可以看出', '下面', '合在一起', '很', '非常', '值得注意的是', '需要指出的是', '这正是', '正是', '换言之',
            '由此可见', '显著', '大幅', '优异', '首次', '——', '？', '，即', '称为']
EN_WORDS = ['that is,', 'novel', 'significant', 'remarkabl', 'It is worth', 'In other words', 'clearly', 'obviously',
            'first ', ' — ', '?', 'called']


def base_text(name):
    r = subprocess.run(['git', '-C', str(REPO), 'show', f'{BASE}:docs/paper_p1/rewrite_plain/sections/{name}'],
                       capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


def paragraphs(t):
    return [p for p in re.split(r'\n\s*\n', t) if p.strip()]


def numbers(t):
    t = re.sub(r'\]\([^)]*\)', ']', t)                  # link targets (DOIs)
    t = re.sub(r'（\d+）', '', t)                         # CN enumeration markers
    return collections.Counter(n.replace(',', '').replace('−', '-') for n in NUM.findall(t))


def links(t):
    return collections.Counter(re.findall(r'\]\((https?://[^)\s]+)\)', t))


def maths(t):
    inl = re.findall(r'\\\((.+?)\\\)', t, re.S)
    disp = re.findall(r'\\\[(.+?)\\\]', t, re.S)
    norm = lambda s: re.sub(r'\s+', '', s)
    return collections.Counter(norm(x) for x in inl + disp)


def structure(t):
    out = []
    for l in t.split('\n'):
        if re.match(r'#{1,4} ', l) or l.startswith('!['):
            out.append(l.strip())
        m = re.match(r'\*\*((?:Figure|Table|Algorithm|Proposition|Remark|图|表|算法|命题|注)\s*[0-9A-Z.]+)', l)
        if m:
            out.append('CAP ' + re.sub(r'\s+', '', m.group(1)))
    out.append('table rows %d' % sum(1 for l in t.split('\n') if l.lstrip().startswith('|')))
    out.append('tags ' + ','.join(re.findall(r'\\tag\{([^}]*)\}', t)))
    return out


def sentence_stats(t, lang):
    body = '\n\n'.join(p for p in paragraphs(t) if not p.lstrip().startswith(('|', '!', '#', '\\[', '**Figure', '**Table',
                                                                                  '**图', '**表', '$$')))
    body = re.sub(r'\\\[.*?\\\]', '', body, flags=re.S)
    body = re.sub(r'\\\((.+?)\\\)', 'X', body, flags=re.S)
    body = re.sub(r'\[([^\]]+)\]\([^)]*\)', 'R', body)
    if lang == 'CN':
        s = [x for x in re.split(r'[。；]', body) if len(x.strip()) > 1]
        n = [len(re.sub(r'\s', '', x)) for x in s]
        short = sum(1 for k in n if k <= 30)
        return f'{len(n)} sentences, mean {sum(n) / max(len(n), 1):.1f} chars, <=30 chars {short / max(len(n), 1):.0%}'
    s = [x for x in re.split(r'(?<=[.;])\s+(?=[A-Z(])', body) if x.strip()]
    n = [len(x.split()) for x in s]
    short = sum(1 for k in n if k <= 15)
    return f'{len(n)} sentences, mean {sum(n) / max(len(n), 1):.1f} words, <=15 words {short / max(len(n), 1):.0%}'


def diff(a, b):
    return dict(a - b), dict(b - a)


def check(uid):
    rep, para = [], {}
    for lang in ('EN', 'CN'):
        name = f'{uid}_{lang}.md'
        old, new = base_text(name), (SEC / name).read_text()
        if old is None:
            rep.append(f'  {name}: no baseline'); continue
        po, pn = paragraphs(old), paragraphs(new)
        para[lang] = len(pn)
        changed = sum(1 for a, b in zip(po, pn) if a != b) if len(po) == len(pn) else None
        rep.append(f'  {name}: paragraphs {len(po)} -> {len(pn)}' + ('' if changed is None else f', changed {changed}')
                   + ('   <-- PARAGRAPH COUNT CHANGED' if len(po) != len(pn) else ''))
        for label, f in (('numbers', numbers), ('links', links), ('maths', maths)):
            rem, add = diff(f(old), f(new))
            if rem or add:
                rep.append(f'    {label} DIFF: removed {rem} added {add}')
        so, sn = structure(old), structure(new)
        if so != sn:
            rep.append(f'    structure DIFF: {[x for x in so if x not in sn]} -> {[x for x in sn if x not in so]}')
        words = CN_WORDS if lang == 'CN' else EN_WORDS
        cnt = [(w, old.count(w), new.count(w)) for w in words if old.count(w) or new.count(w)]
        if cnt:
            rep.append('    words (old->new): ' + ', '.join(f'{w!r} {a}->{b}' for w, a, b in cnt))
        rep.append(f'    sentences old: {sentence_stats(old, lang)}')
        rep.append(f'    sentences new: {sentence_stats(new, lang)}')
    if para.get('EN') != para.get('CN'):
        rep.append(f'  EN/CN PARAGRAPH MISMATCH: {para}')
    return rep


if __name__ == '__main__':
    for u in sys.argv[1:]:
        print(f'== {u}')
        print('\n'.join(check(u)))
