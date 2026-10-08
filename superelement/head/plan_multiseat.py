"""Choose the presented/unseen split for the generalisation run and print the launch.

Every eps_op this project has quoted came from at most 26 geometries, so it has
never measured generalisation at all.  This withholds a fixed fraction of the
newly built train labels from the optimizer and from the conditioning aggregate,
and evaluates on them.  The frozen holdout is untouched and stays untouched.
"""
import argparse, json
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--unseen-fraction', type=float, default=0.15)
    ap.add_argument('--eval-presented', type=int, default=4)
    ap.add_argument('--eval-unseen', type=int, default=6)
    ap.add_argument('--eval-max-q', type=int, default=15000)
    ap.add_argument('--seed', type=int, default=20260917)
    args = ap.parse_args()

    rows = json.loads(args.manifest.read_text())
    train = sorted((int(r['seat']), int(r['q'])) for r in rows if r['split'] == 'train')
    seats = [s for s, _ in train]
    q = dict(train)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(seats))
    cut = int(round(args.unseen_fraction * len(seats)))
    unseen = sorted(seats[i] for i in order[:cut])
    presented = sorted(seats[i] for i in order[cut:])

    # evaluation costs a full dense inference plus a d x d eigendecomposition, so
    # only small seats are evaluated, and the same rule picks from both sides.
    def pick(pool, count):
        # Spread across the eligible q range instead of taking the smallest, so the
        # score is not a score on tiny cells only.  The cap is a cost cap: the
        # evaluation is a dense inference plus a d x d eigendecomposition.
        small = sorted((s for s in pool if q[s] <= args.eval_max_q), key=lambda s: q[s])
        if not small:
            raise ValueError('NO_ELIGIBLE_EVALUATION_SEAT')
        count = min(count, len(small))
        index = np.linspace(0, len(small) - 1, count).round().astype(int)
        return sorted({small[i] for i in index})

    eval_presented = pick(presented, args.eval_presented)
    eval_unseen = pick(unseen, args.eval_unseen)
    plan = dict(seed=args.seed, manifest=str(args.manifest),
                train_rows=len(seats), unseen_fraction=args.unseen_fraction,
                presented=presented, unseen=unseen,
                eval_presented=eval_presented, eval_unseen=eval_unseen,
                eval_seats=sorted(eval_presented + eval_unseen),
                eval_max_q=args.eval_max_q,
                q_presented_median=int(np.median([q[s] for s in presented])),
                q_unseen_median=int(np.median([q[s] for s in unseen])),
                q_of_eval_seats={str(s): q[s] for s in sorted(eval_presented + eval_unseen)},
                frozen_holdout_untouched=True,
                scope='unseen seats are train-split labels withheld from this run; '
                      'they are not the frozen 6-seat holdout and are not a test score')
    args.output.write_text(json.dumps(plan, indent=1))
    print(json.dumps(dict(train_rows=len(seats), presented=len(presented), unseen=len(unseen),
                          eval_seats=plan['eval_seats'],
                          q_presented_median=plan['q_presented_median'],
                          q_unseen_median=plan['q_unseen_median'])))
    print('SEATS=' + ' '.join(map(str, seats)))
    print('TRAIN_SEATS=' + ' '.join(map(str, presented)))
    print('EVAL_SEATS=' + ' '.join(map(str, plan['eval_seats'])))


if __name__ == '__main__':
    main()
