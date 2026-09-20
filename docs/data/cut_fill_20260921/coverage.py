"""What does the cut family actually cover, and what is structurally missing?

The question is whether the cut dataset is too SMALL or too NARROW - different hypotheses with
different remedies.  n is already measured flat (8 = 28 = 59 give the same held-out median), which
speaks against size.  Narrowness has never been measured.  Report the geometry of the family over
every packet on disk, used and unused, and state plainly which directions are absent.
"""
import json, os, collections
from fractions import Fraction
import numpy as np

R = '/root/autodl-tmp/CUTFEM_INGEST_R38/dataset_independent_20260910'
recs = [v for v in json.load(open('/root/_packet_class.json')) if 'error' not in v]
cut = [v for v in recs if v['cut'] and v.get('plane')]
print('cut packets with a plane on disk:', len(cut))

def comps(v):
    return [Fraction(x) for x in v['plane']]

rows = []
for v in cut:
    a, b, c, d = comps(v)
    n = np.array([float(a), float(b), float(c)])
    rows.append(dict(batch=v['batch'], in_manifest=v['in_manifest'], q=v.get('q'),
                     tau_mean=v['tau_mean'], a=float(a), b=float(b), c=float(c), d=float(d),
                     zero_components=int(a == 0) + int(b == 0) + int(c == 0),
                     unit=(n / np.linalg.norm(n)).tolist()))

print()
print('=== the third component ===')
print('  distinct values of c:', collections.Counter(r['c'] for r in rows).most_common(5))
print('  distinct values of a:', collections.Counter(r['a'] for r in rows).most_common(5))
print('  zero-component count:', collections.Counter(r['zero_components'] for r in rows).most_common())
generic = [r for r in rows if r['zero_components'] == 0]
print('  planes with ALL THREE components non-zero: %d of %d' % (len(generic), len(rows)))

print()
print('=== b and offset, in and out of the manifest ===')
for tag, grp in (('in manifest', [r for r in rows if r['in_manifest']]),
                 ('NOT in manifest', [r for r in rows if not r['in_manifest']])):
    if not grp:
        continue
    b = np.array([r['b'] for r in grp]); d = np.array([r['d'] for r in grp])
    print('  %-16s n=%3d  b [%.4f, %.4f] med %.4f | offset [%.4f, %.4f] med %.4f' % (
        tag, len(grp), b.min(), b.max(), np.median(b), d.min(), d.max(), np.median(d)))

print()
print('=== how evenly is (b, offset) covered?  in-domain cut packets only ===')
for tag, grp in (('the 139 in use', [r for r in rows if r['in_manifest'] and (r['tau_mean'] or 9) <= 0.45]),
                 ('all 239 on disk', [r for r in rows if (r['tau_mean'] or 9) <= 0.45])):
    if len(grp) < 3:
        continue
    P = np.array([[r['b'], r['d']] for r in grp])
    lo, hi = P.min(0), P.max(0)
    Pn = (P - lo) / np.maximum(hi - lo, 1e-12)
    D = np.linalg.norm(Pn[:, None, :] - Pn[None, :, :], axis=2)
    np.fill_diagonal(D, np.inf)
    nn = D.min(1)
    # largest empty circle, sampled: the worst-covered point of the unit square
    g = np.stack(np.meshgrid(np.linspace(0, 1, 120), np.linspace(0, 1, 120), indexing='ij'), -1).reshape(-1, 2)
    hole = np.linalg.norm(g[:, None, :] - Pn[None, :, :], axis=2).min(1).max()
    print('  %-16s n=%3d  nearest neighbour (normalised): min %.4f med %.4f max %.4f | largest hole %.4f' % (
        tag, len(grp), nn.min(), np.median(nn), nn.max(), hole))
    # uniformity: how far the marginals are from uniform
    for j, name in enumerate(('b', 'offset')):
        u = np.sort(Pn[:, j]); k = np.arange(1, len(u) + 1) / len(u)
        ks = float(np.max(np.abs(u - k)))
        print('      marginal %-6s Kolmogorov distance from uniform %.4f' % (name, ks))

json.dump(rows, open('/root/_cut_coverage.json', 'w'))
