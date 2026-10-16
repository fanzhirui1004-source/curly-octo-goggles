"""Local self-test of exact_trace against a verbatim Python port of the frozen algorithms (Fraction backend):
max pivot with combinations, max pivot without combinations, first-pivot replay, and the direct-sum reverse
elimination (augmented rows), on random sparse rational rows with dependent rows and cancellations."""
import subprocess, random, sys
from fractions import Fraction as F
from heapq import heapify, heappop, heappush
ONE = F(1)
def _subtract(row, basis, factor):
    get = row.get
    for col, value in basis.items():
        current = get(col)
        updated = -factor * value if current is None else current - factor * value
        if updated: row[col] = updated
        else: row.pop(col, None)
def _reduce(row, basis, pivots, positions, combination=None, combinations=None):
    pending = [positions[c] for c in row if c in positions]; scheduled = set(pending); heapify(pending)
    while pending:
        index = heappop(pending); pivot = pivots[index]
        if pivot not in row: continue
        factor = row[pivot]; previous = basis[index]
        _subtract(row, previous, factor)
        if combination is not None: _subtract(combination, combinations[index], factor)
        for column in previous:
            ni = positions.get(column)
            if ni is not None and ni > index and ni not in scheduled and column in row:
                heappush(pending, ni); scheduled.add(ni)
def maxpivot(rows):
    basis, pivots, selected, combinations, positions = [], [], [], [], {}
    for index, original in enumerate(rows):
        row, comb = dict(original), {index: ONE}
        _reduce(row, basis, pivots, positions, comb, combinations)
        if not row: continue
        pivot = min(row, key=lambda c: (-abs(row[c]), c)); scale = row[pivot]
        positions[pivot] = len(pivots); pivots.append(pivot)
        basis.append({c: v/scale for c, v in row.items()}); combinations.append({c: v/scale for c, v in comb.items()}); selected.append(index)
    return selected, basis, combinations, pivots
def firstpivot(rows):
    basis, pivots, selected, positions = [], [], [], {}
    for index, original in enumerate(rows):
        row = dict(original); _reduce(row, basis, pivots, positions)
        if not row: continue
        pivot = min(row); scale = row[pivot]
        positions[pivot] = len(pivots); pivots.append(pivot)
        basis.append({c: v/scale for c, v in row.items()}); selected.append(index)
    return selected
def augmented_of(rows, pivots, dimension):
    augmented = [{**row, dimension+i: ONE} for i, row in enumerate(rows)]
    positions = {pivot: i for i, pivot in enumerate(pivots)}
    dependents = [[] for _ in rows]
    for i, row in enumerate(rows):
        for column, value in row.items():
            target = positions.get(column)
            if target is not None and target > i: dependents[target].append(i)
    for i in reversed(range(len(rows))):
        pivot = pivots[i]
        for j in dependents[i]:
            if pivot in augmented[j]: _subtract(augmented[j], augmented[i], augmented[j][pivot])
    return augmented
def parse(line):
    t = line.split(); return {int(t[1 + 2*i]): F(t[2 + 2*i]) for i in range(int(t[0]))}
def run(mode, rows, ncol):
    inp = f'{mode} {ncol} {len(rows)}\n' + ''.join('%d %s\n' % (len(r), ' '.join('%d %s' % (c, v) for c, v in r.items())) for r in rows)
    return subprocess.run(['./exact_trace'], input=inp, capture_output=True, text=True, check=True).stdout.split('\n')
for seed in range(20):
    random.seed(seed)
    ncol = random.randint(10, 60); rows = []
    for k in range(random.randint(20, 200)):
        if rows and random.random() < 0.3:   # dependent row: combination of earlier rows
            a, b = random.sample(range(len(rows)), 2) if len(rows) > 1 else (0, 0)
            fa, fb = F(random.randint(-5, 5), random.randint(1, 4)), F(random.randint(-5, 5), random.randint(1, 4))
            r = dict(rows[a]); _subtract(r, rows[b], -fb); r = {c: v * fa for c, v in r.items() if v * fa}
            if not r: r = {random.randrange(ncol): F(1)}
        else:
            r = {c: F(random.randint(-9, 9), random.randint(1, 7)) for c in random.sample(range(ncol), random.randint(1, 8))}
            r = {c: v for c, v in r.items() if v} or {0: F(1)}
        rows.append(r)
    sel, basis, comb, piv = maxpivot(rows)
    aug = augmented_of(basis, piv, ncol)
    for mode in ('max', 'maxnc'):
        out = run(mode, rows, ncol)
        n = int(out[0].split()[1]); pairs = [tuple(map(int, l.split())) for l in out[1:1 + n]]
        assert [p[0] for p in pairs] == sel and [p[1] for p in pairs] == piv, (seed, mode, 'selection/pivots')
        i = 1 + n; assert out[i] == 'basis'; assert [parse(out[i + 1 + j]) for j in range(n)] == basis, (seed, mode, 'basis')
        i += 1 + n
        if mode == 'max':
            assert out[i] == 'combinations'; assert [parse(out[i + 1 + j]) for j in range(n)] == comb, (seed, 'combinations'); i += 1 + n
        assert out[i] == 'augmented'; assert [parse(out[i + 1 + j]) for j in range(n)] == aug, (seed, mode, 'augmented')
    out = run('first', rows, ncol); n = int(out[0].split()[1])
    assert [int(l.split()[0]) for l in out[1:1 + n]] == firstpivot(rows), (seed, 'first')
print('self-test ok on 20 random systems (max, maxnc, first, augmented)')
