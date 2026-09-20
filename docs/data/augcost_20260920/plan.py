"""Pre-register the augmentation-dose experiment BEFORE it runs.

Augmentation has never been applied to a cut cell in any multi-seat arm.  Every arm that recorded
`augmented_seats` recorded zero of them for its cut seats (cutonly_noaug 0 of 32, wide_cut 0 of 73,
ncut_02/04/08/16/59 0, cutmix_fullaug 28 of 64 and those 28 are its box seats), because they all ran
`--augment --augment-full-only`.  The only evidence that augmenting a cut cell hurts is one n = 1
pair, solo_100079 against noaug_100079, 3.130 % against 12.726 %.

So the question is open and this is the experiment: at a fixed pool, capacity, schedule and step
budget, does rotating cut cells help or hurt?  Three doses plus a seed control, and the seed control
is what says whether a difference is resolvable at all -- with one seed per arm it would not be.

The held-out set is the twelve seats already pre-registered in HELDOUT_H2.json, so the numbers land
on the same axis as the H2 result (n = 8 median 96.2 %, n = 28 median 87.9 %) and the training pool
is built to exclude them.
"""
import json, os
import numpy as np

MANIFEST = '/root/autodl-tmp/CLAUDE_LABELS_20260917/V2_LABELS.json'
HELD = [100001, 100003, 100009, 245, 100014, 100016, 100020, 100022, 100023, 100025, 230, 100029]
# The pool is CUTONLY_NOAUG's own 28 cut seats, not a fresh selection.  Two reasons: DOSE_00 is then
# a re-run of a configuration that already exists, so a disagreement with it is a bug signal; and
# n = 28 is the H2 arm whose held-out median was 87.9 % on exactly the twelve seats held out here.
# Sorting the design domain by seat id and taking the first sixteen, as a first draft did, would have
# taken only the low-numbered seat family.
POOL = [129, 139, 185, 198, 224, 100002, 100006, 100008, 100010, 100013, 100015, 100017, 100019,
        100024, 100028, 100030, 100032, 100035, 100039, 100041, 100046, 100048, 100050, 100056,
        100059, 100068, 100070, 100072]
STEPS = 60000

desc = {r['seat']: r for r in json.load(open('/root/_seat_desc.json'))}
rows = json.load(open(MANIFEST))
candidates = []
for r in rows:
    seat = int(r['seat'])
    d = desc.get(seat, {})
    if (d.get('n_kind1') or 0) <= 0:                       # cut cells only
        continue
    if (d.get('tau_mean') or 9) > 0.45:                    # the design domain
        continue
    if not os.path.exists(os.path.join(r['reference'], 'MQ_UPPER.npy')):
        continue
    if seat in HELD:
        continue
    candidates.append(seat)
candidates.sort()
# Nine of CUTONLY_NOAUG's 28 seats are outside the design domain (tau_mean > 0.45), so the pool here
# is its 19 in-domain seats.  They are dropped rather than kept because tau is the design variable and
# the optimiser works the thin-wall end; an arm trained partly on tau_mean > 0.45 is not measuring the
# family the application asks for.
dropped = [s_ for s_ in POOL if s_ not in candidates]
pool = [s_ for s_ in POOL if s_ in candidates]
if len(pool) < 12:
    raise SystemExit(f'POOL_TOO_SMALL {pool}')
held_ok = [s for s in HELD if s in desc]
plan = dict(
    schema='AUGMENTATION_DOSE_V1', declared='2026-09-20',
    question='at a fixed pool, capacity, schedule and step budget, does rotating CUT cells help or hurt?',
    why_open=('every multi-seat cut arm ran --augment --augment-full-only, which leaves cut cells in '
              'the identity frame: augmented_seats was 0 of 32 (cutonly_noaug), 0 of 73 (wide_cut), '
              '0 for ncut_02/04/08/16/59, and 28 of 64 for cutmix_fullaug where the 28 are its box '
              'seats.  The only contrary evidence is the n=1 pair solo_100079 / noaug_100079.'),
    train_pool=pool, train_pool_size=len(pool),
    train_pool_source='the in-domain subset of CUTONLY_NOAUG\'s 28 cut seats, so DOSE_00 is that configuration restricted to tau_mean <= 0.45',
    dropped_from_that_pool=dropped,
    dropped_reason='tau_mean > 0.45, outside the design domain the optimiser works',
    held_out=held_ok, held_out_source='HELDOUT_H2.json, pre-registered 2026-09-20',
    presented_control=100032,
    arms=[dict(name='DOSE_00', flags=['--augment', '--augment-full-only'], group_elements=1, seed=20260920,
               note='identical code path to every historical cut arm: cut cells stay in the identity frame'),
          dict(name='DOSE_00B', flags=['--augment', '--augment-full-only'], group_elements=1, seed=20260921,
               note='SEED CONTROL.  Its distance from DOSE_00 is the resolution of this experiment; '
                    'any dose difference smaller than that is not a finding'),
          dict(name='DOSE_24', flags=['--augment', '--proper-only'], group_elements=24, seed=20260920,
               note='the 24 proper rotations reach the cut cells'),
          dict(name='DOSE_48', flags=['--augment'], group_elements=48, seed=20260920,
               note='all 48, so improper elements and the z-tilted planes the dataset does not contain')],
    matched=['pool', 'steps', 'capacity', 'warmup', 'schedule', 'pairs per bucket', 'near_cells',
             'label residency', 'bf16', 'evaluation set', 'metric'],
    steps=STEPS,
    metric=dict(primary='median over the 12 held-out seats of each seat median absolute face-load '
                        'response error, assembly-free, on the final checkpoint, via sweep_checkpoints',
                secondary=['frac_within_3pct per seat', 'the presented seat 100032 for the '
                           'presented-to-held-out gap', 'factor relative error']),
    decision_rule=('let R = |DOSE_00 - DOSE_00B| on the primary metric.  A dose is called better or '
                   'worse only if it differs from DOSE_00 by more than R.  If DOSE_24 and DOSE_48 are '
                   'both worse by more than R, rotating cut cells costs more than it buys at this '
                   'budget and the historical --augment-full-only choice was right for the wrong '
                   'reason.  If the ordering is monotone the other way, every cut arm so far has been '
                   'leaving the one measured generalisation lever unused.  A result inside R is '
                   'reported as unresolved at this budget, not as a null.'),
    forbidden=['widening the acceptance gate', 'jitter', 'pseudo-inverse', 'spectral clipping',
               'deleting directions', 'reading the held-out set before the arms finish',
               'changing the pool or the metric after seeing any arm'],
    decode=dict(inference_chunk=8192,
                why='the only chunk measured to reproduce the reference element for element; 32768 '
                    'differs by 6.2e-7 relative on the factor'))
json.dump(plan, open('/root/autodl-tmp/CLAUDE_AUGCOST_20260920/PREREGISTRATION.json', 'w'), indent=1)
print(json.dumps(dict(pool=pool, pool_size=len(pool), dropped=dropped, held=held_ok,
                      candidates=len(candidates))))
