"""Design the geometries to add, by farthest-point insertion against what we already have.

The cut family's failure mode in (b, s) - b the plane's second normal component, s = d / (1 + b) the
depth fraction, the coordinates in which every point is physically reachable - is not that there are
too few points but that they are clumped and leave a hole.  Measured over the 239 in-domain packets:
nearest-neighbour median 0.0254 where a uniform sample of that size would give 0.0647, and a largest
empty circle of 0.2041 where uniform gives 0.129.  So the points are both tighter than uniform and
leave a bigger gap than uniform - the signature of independent random draws at small n.

Adding more independent random draws does not fix that: the 100 unused packets have the same
distribution and adding them leaves the largest hole at 0.2041, unchanged.

Farthest-point (maximin) insertion does fix it, and it is the right rule because it minimises exactly
the quantity that is bad: each new point is placed at the position of the fine grid that is farthest
from every point already there, existing ones included, so nothing already paid for is wasted.
"""
import json
import numpy as np

rows = json.load(open('/root/_cut_coverage.json'))
for r in rows:
    r['s'] = r['d'] / (1.0 + r['b'])
existing_domain = np.array([[r['b'], r['s']] for r in rows if (r['tau_mean'] or 9) <= 0.45])
existing_all = np.array([[r['b'], r['s']] for r in rows])
print('existing in-domain cut geometries: %d | all cut: %d' % (len(existing_domain), len(existing_all)))

G = 200
g = np.stack(np.meshgrid(np.linspace(0.005, 0.995, G), np.linspace(0.005, 0.995, G), indexing='ij'), -1).reshape(-1, 2)

def largest_hole(P):
    return float(np.min(np.linalg.norm(g[:, None, :] - P[None, :, :], axis=2), axis=1).max())

def uniform_reference(n, reps=5, seed=0):
    rng = np.random.default_rng(seed)
    return float(np.mean([largest_hole(rng.random((n, 2))) for _ in range(reps)]))

for base_name, base in (('in-domain only', existing_domain), ('all cut on disk', existing_all)):
    P = base.copy()
    dist = np.linalg.norm(g[:, None, :] - P[None, :, :], axis=2).min(1)
    print()
    print('=== seeding from the %d %s points ===' % (len(base), base_name))
    print('  start: largest hole %.4f (uniform at n=%d would be %.4f)' % (
        largest_hole(P), len(P), uniform_reference(len(P))))
    added = []
    marks = {}
    for k in range(1, 1201):
        i = int(np.argmax(dist))
        p = g[i]
        added.append(p.tolist())
        d_new = np.linalg.norm(g - p, axis=1)
        dist = np.minimum(dist, d_new)
        if k in (50, 100, 200, 300, 400, 600, 800, 1000, 1200):
            n_tot = len(base) + k
            marks[k] = (float(dist.max()), uniform_reference(n_tot, reps=3, seed=k))
    for k, (hole, unif) in marks.items():
        print('   +%-5d (total %4d): largest hole %.4f  | uniform of that size %.4f  | ratio %.2f' % (
            k, len(base) + k, hole, unif, hole / unif))
    if base_name == 'in-domain only':
        json.dump(dict(seeded_from=len(base), points=added), open('/root/_fill_points.json', 'w'))
