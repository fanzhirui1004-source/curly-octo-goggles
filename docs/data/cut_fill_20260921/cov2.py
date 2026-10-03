"""Coverage in coordinates where every point is physically reachable.

A first pass measured a "largest hole" of 0.39 in the (b, offset) box, which is not a sampling gap:
the plane (1, b, 0) . y = d only meets the unit cell for d in (0, 1 + b), so the region d > 1 + b is
empty by construction.  Re-measure with the depth fraction s = d / (1 + b) in (0, 1), where the
domain IS the unit square, and report the same statistics.  Also report the fraction of the cell the
plane actually removes, which is what the physics sees.
"""
import json
import numpy as np

rows = json.load(open('/root/_cut_coverage.json'))
for r in rows:
    r['s'] = r['d'] / (1.0 + r['b'])          # depth fraction, reachable iff in (0, 1)
print('cut packets:', len(rows))
bad = [r for r in rows if not (0 < r['s'] < 1)]
print('planes with s outside (0,1):', len(bad), [round(r['s'], 3) for r in bad[:6]])

def report(tag, grp):
    if len(grp) < 3:
        return
    P = np.array([[r['b'], r['s']] for r in grp])
    D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
    np.fill_diagonal(D, np.inf)
    nn = D.min(1)
    g = np.stack(np.meshgrid(np.linspace(0, 1, 160), np.linspace(0, 1, 160), indexing='ij'), -1).reshape(-1, 2)
    hole = np.linalg.norm(g[:, None, :] - P[None, :, :], axis=2).min(1).max()
    ks = []
    for j in range(2):
        u = np.sort(P[:, j]); k = np.arange(1, len(u) + 1) / len(u)
        ks.append(float(np.max(np.abs(u - k))))
    expected = np.sqrt(1.0 / len(grp))
    print('  %-22s n=%3d | nn med %.4f (uniform would be ~%.4f) | largest hole %.4f | KS b %.3f s %.3f' % (
        tag, len(grp), np.median(nn), expected, hole, ks[0], ks[1]))
    # where is the hole?
    idx = int(np.argmax(np.linalg.norm(g[:, None, :] - P[None, :, :], axis=2).min(1)))
    print('       worst-covered point: b=%.3f  s=%.3f' % (g[idx][0], g[idx][1]))
    # quadrant counts, to show the imbalance plainly
    q = np.zeros((4, 4), dtype=int)
    for b_, s_ in P:
        q[min(3, int(b_ * 4)), min(3, int(s_ * 4))] += 1
    print('       4x4 cells of (b, s), rows = b quartile:')
    for i in range(4):
        print('         ', ' '.join('%4d' % v for v in q[i]))

print()
print('=== coverage of (b, depth fraction) ===')
report('the 139 in use', [r for r in rows if r['in_manifest'] and (r['tau_mean'] or 9) <= 0.45])
report('all 239 in domain', [r for r in rows if (r['tau_mean'] or 9) <= 0.45])
report('all 603 on disk', rows)
print()
print('=== what a uniform sample of the same size would give, for reference ===')
rng = np.random.default_rng(0)
for n in (139, 239):
    hs = []
    for _ in range(5):
        P = rng.random((n, 2))
        g = np.stack(np.meshgrid(np.linspace(0, 1, 160), np.linspace(0, 1, 160), indexing='ij'), -1).reshape(-1, 2)
        hs.append(np.linalg.norm(g[:, None, :] - P[None, :, :], axis=2).min(1).max())
    print('  n=%3d uniform: largest hole %.4f +- %.4f' % (n, np.mean(hs), np.std(hs)))
