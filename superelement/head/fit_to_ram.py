"""Choose a presented set whose labels fit the container's page cache.

The first attempt presented 226 seats.  Their fp64 packed labels are 251 GiB against
a 90 GiB cgroup limit, so the page cache thrashed and the step time went from 0.15 s
(single seat) to 1.79 s and rising -- 30 hours for the planned budget, all of it
scattered reads, not arithmetic.

This keeps the same seeded split and simply takes the largest prefix of the presented
list, in a size-balanced order, that fits a byte budget.  The unseen set is untouched:
it is only ever read during evaluation, which is ten seats.
"""
import argparse, json
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--plan', type=Path, required=True)
ap.add_argument('--manifest', type=Path, required=True)
ap.add_argument('--output', type=Path, required=True)
ap.add_argument('--budget-gib', type=float, default=50.0)
a = ap.parse_args()

plan = json.loads(a.plan.read_text())
rows = {int(r['seat']): r for r in json.loads(a.manifest.read_text())}


def nbytes(seat):
    d = int(rows[seat]['q']) - 6
    return (d * (d + 1) // 2) * 8


# interleave small and large so the retained set keeps the q spread of the original
ordered = sorted(plan['presented'], key=nbytes)
weave = []
lo, hi = 0, len(ordered) - 1
while lo <= hi:
    weave.append(ordered[lo]); lo += 1
    if lo <= hi:
        weave.append(ordered[hi]); hi -= 1

budget = a.budget_gib * 2**30
kept, total = [], 0
for s in weave:
    if total + nbytes(s) > budget:
        continue
    kept.append(s); total += nbytes(s)
kept = sorted(kept)

# evaluation must stay inside what is prepared, and must keep both sides balanced,
# so the presented evaluation seats are re-picked from the retained set, spread across
# its q range under the same cost cap the original plan used.
cap = 15000
eligible = sorted((s for s in kept if int(rows[s]['q']) <= cap), key=lambda s: int(rows[s]['q']))
want = len(plan['eval_unseen']) - 2
if not eligible:
    eligible = sorted(kept, key=lambda s: int(rows[s]['q']))[:1]
take = min(want, len(eligible))
index = np.linspace(0, len(eligible) - 1, take).round().astype(int)
eval_presented = sorted({eligible[i] for i in index})
eval_seats = sorted(set(eval_presented) | set(plan['eval_unseen']))
prepared = sorted(set(kept) | set(plan['unseen']))

q = {s: int(rows[s]['q']) for s in prepared}
out = dict(budget_gib=a.budget_gib,
           presented=kept, unseen=plan['unseen'], prepared=prepared,
           eval_presented=eval_presented, eval_unseen=plan['eval_unseen'],
           eval_seats=eval_seats,
           presented_bytes=total, presented_gib=total / 2**30,
           dropped_from_presented=sorted(set(plan['presented']) - set(kept)),
           dropped_count=len(plan['presented']) - len(kept),
           q_presented_median=int(np.median([q[s] for s in kept])),
           q_unseen_median=int(np.median([q[s] for s in plan['unseen']])),
           q_presented_range=[int(min(q[s] for s in kept)), int(max(q[s] for s in kept))],
           scope='same seeded split as SPLIT_PLAN.json, presented set truncated to fit the '
                 'container page cache; the unseen set and the evaluation seats are unchanged '
                 'except where an evaluation seat was dropped from the presented side')
a.output.write_text(json.dumps(out, indent=1))
print(json.dumps(dict(presented=len(kept), unseen=len(plan['unseen']), prepared=len(prepared),
                      presented_gib=round(total / 2**30, 1),
                      eval_seats=eval_seats, dropped=out['dropped_count'],
                      q_presented_range=out['q_presented_range'],
                      q_presented_median=out['q_presented_median'])))
print('SEATS=' + ' '.join(map(str, prepared)))
print('TRAIN_SEATS=' + ' '.join(map(str, kept)))
print('EVAL_SEATS=' + ' '.join(map(str, eval_seats)))
