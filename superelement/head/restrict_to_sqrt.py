"""Restrict the generalisation split to seats whose A^{1/2} label actually exists.

If the label build stopped early (disk), the sqrt arm must not be launched on seats
it cannot read.  Dropped seats are named, not silently skipped, because a quieter
split is a different experiment.
"""
import argparse, json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--plan', type=Path, required=True)
ap.add_argument('--manifest', type=Path, required=True)
ap.add_argument('--output', type=Path, required=True)
a = ap.parse_args()

plan = json.loads(a.plan.read_text())
rows = {int(r['seat']): r for r in json.loads(a.manifest.read_text())}


def has_label(seat):
    r = rows.get(seat)
    if r is None:
        return False
    ref = Path(r['reference'])
    return (ref / 'M_UPPER.npy').exists() and (ref / 'SQRT_RESULT.json').exists()


presented = [s for s in plan['presented'] if has_label(s)]
unseen = [s for s in plan['unseen'] if has_label(s)]
seats = sorted(presented + unseen)
ev = [s for s in plan['eval_seats'] if s in set(seats)]
dropped = dict(presented=[s for s in plan['presented'] if s not in set(presented)],
               unseen=[s for s in plan['unseen'] if s not in set(unseen)],
               eval=[s for s in plan['eval_seats'] if s not in set(ev)])
out = dict(presented=presented, unseen=unseen, seats=seats, eval_seats=ev, dropped=dropped,
           dropped_counts={k: len(v) for k, v in dropped.items()},
           comparable_to_chol_arm=not any(dropped.values()),
           scope='same split as the chol arm minus any seat whose A^{1/2} label is missing')
(a.output / 'SPLIT_PLAN_SQRT.json').write_text(json.dumps(out, indent=1))
lines = [f"SEATS={' '.join(map(str, seats))}",
         f"TRAIN_SEATS={' '.join(map(str, presented))}",
         f"EVAL_SEATS={' '.join(map(str, ev))}"]
(a.output / 'plan_sqrt.txt').write_text('\n'.join(lines) + '\n')
print(json.dumps(dict(seats=len(seats), presented=len(presented), unseen=len(unseen),
                      eval_seats=len(ev), dropped=out['dropped_counts'],
                      comparable=out['comparable_to_chol_arm'])))
