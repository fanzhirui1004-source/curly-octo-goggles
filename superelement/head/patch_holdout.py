"""Let one run train on a subset and evaluate on seats it never saw.

--seats prepares every seat.  --train-seats (default: all of them) chooses which
ones the training loop and the conditioning statistics may use.  --eval-seats
chooses which ones are evaluated, and may name seats outside --train-seats: those
are a genuine unseen set, prepared but never presented and never calibrated on.
--calibrate-pairs bounds the per-seat draw, which matters once there are hundreds.
"""
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]) / 'stage_cutfem_m4'


def edit(path, old, new, count=1):
    text = path.read_text()
    n = text.count(old)
    if n != count:
        raise SystemExit(f'PATCH_ANCHOR {path.name}: expected {count}, found {n}\n---\n{old}\n---')
    path.write_text(text.replace(old, new))
    print(f'  ok {path.name}: {old.splitlines()[0][:66]}')


run = ROOT / 'run.py'

edit(run, """    ap.add_argument('--eval-seats',type=int,nargs='*',default=None,""",
"""    ap.add_argument('--train-seats',type=int,nargs='*',default=None,
        help='subset of --seats presented to the optimizer and used for conditioning; the rest are prepared but unseen')
    ap.add_argument('--calibrate-pairs',type=int,default=32768,
        help='per-seat off-diagonal draws for the conditioning aggregate')
    ap.add_argument('--eval-seats',type=int,nargs='*',default=None,""")

edit(run, """        rows=selected_rows(args.manifest,args.seats,args.known_328)
        samples=[prepare(r,args) for r in rows]
        write(args.output/'INPUTS.json',[s['record'] for s in samples])
        conditioning,stats=calibrate(samples,args.patch_size)""",
"""        rows=selected_rows(args.manifest,args.seats,args.known_328)
        samples=[prepare(r,args) for r in rows]
        write(args.output/'INPUTS.json',[s['record'] for s in samples])
        if args.train_seats is None:train=samples
        else:
            wanted=set(args.train_seats)
            if not wanted<=set(args.seats):raise ValueError('TRAIN_SEAT_NOT_PREPARED')
            train=[s for s in samples if int(s['seat']) in wanted]
        if not train:raise ValueError('EMPTY_TRAINING_SET')
        presented=set(int(s['seat']) for s in train)
        unseen=[int(s['seat']) for s in samples if int(s['seat']) not in presented]
        write(args.output/'SPLIT.json',dict(prepared=[int(s['seat']) for s in samples],
            presented=sorted(presented),never_presented=unseen,
            conditioning_source='presented seats only',
            scope='unseen seats are train-split labels withheld from this run; they are not the frozen holdout'))
        conditioning,stats=calibrate(train,args.patch_size,per_bucket=args.calibrate_pairs)""")

edit(run, """            tick=sync();sample=samples[(step-1)%len(samples)];ctx=sample['context'];d=sample['d']""",
"""            tick=sync();sample=train[(step-1)%len(train)];ctx=sample['context'];d=sample['d']""")

edit(run, """        evaluated_seats='all prepared seats' if args.eval_seats is None else args.eval_seats,""",
"""        evaluated_seats='all prepared seats' if args.eval_seats is None else args.eval_seats,
        presented_seats='all prepared seats' if args.train_seats is None else args.train_seats,
        calibrate_pairs=args.calibrate_pairs,""")

print('HOLDOUT_PATCH_COMPLETE')
