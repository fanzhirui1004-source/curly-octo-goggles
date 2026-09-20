"""Generate the cut geometries the dataset is missing, under a trace-dimension budget.

The cut family in use is not too small so much as clumped and censored.  Two measurements set
this up.  First, the teacher's own sampler is uniform in retained box volume -- across all 607
cut packets the ten volume deciles hold 52 to 66 each -- so the generator is not what produced
the gaps.  Second, at fixed thickness a deep cut carries about 1.7x the dofs of a shallow one
(1.61-1.76 across five tau bands), and a steeper plane cuts a larger polygon and so carries
more cut coordinates.  The q ceiling that decides what we can label is therefore a filter on
depth and angle, and the surviving subset leans shallow and shallow-angled: in-domain the
volume deciles run 52 at the bottom to 17 at the top.

So the fix is not to sample harder in the same way and hope.  It is to invert the order:
choose the coverage first, then spend the thickness budget on whatever is left.  For each
target (angle, depth) this bisects the thickness mean until the predicted trace dimension lands
in the requested band, using the calibrated screen in `cut_budget` (82 ms a shot, median error
1.5 % on cut cells).  A target that no admissible thickness can reach is reported as
unreachable rather than quietly dropped, because a silently shrinking design is how the current
coverage happened in the first place.

Nothing here re-derives the thickness contract.  The eight corner values are built by the
frozen teacher's own construction -- one centred polynomial in x, y, z, xy, xz, yz, xyz scaled
as a whole -- and handed to the frozen `Thickness`, which enforces corners in [0.1755, 0.8775],
span <= 0.47 and |grad tau| <= 0.47.  The plane is (1, tan theta, 0) with theta in [0, 45), the
one family `GradedContract` accepts: a normal with all three components nonzero raises
GRADED_SINGLE_ZERO_COMPONENT_PLANE_FAMILY.

Splits are assigned to mother fields before any geometry is generated, so the held-out set is
fixed before it can be chosen to flatter a result.

    python -m superelement.objective.fill_cut_family --count 300 --output DIR \
        --q-budget 19500 --holdout-fraction 0.2
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from fractions import Fraction
from itertools import product
from pathlib import Path

import numpy as np


LOWER = Fraction('0.1755')
UPPER = Fraction('0.8775')
# The frozen corner bounds are the image of a VOLUME FRACTION range, measured on a 400^3 grid of
# the Schwarz-P sheet |phi| <= tau: tau = 0.1755 gives rho = 0.1001, tau = 0.8775 gives
# rho = 0.5026, and UPPER/LOWER is 5.0000 exactly.  So [0.1755, 0.8775] is rho in [0.10, 0.50],
# and a cap stated in volume fraction is the natural way to narrow it.
MEASURED_VOLUME_FRACTION = {'0.1755': 0.1001, '0.8775': 0.5026}
CORNER_ORDER = tuple(product((0, 1), repeat=3))
MAX_ANGLE_DEGREES = 45.0


def _decimal(value):
    return Fraction(f'{value:.9f}')


def _seeded(*parts):
    digest = hashlib.sha256('|'.join(map(str, parts)).encode()).digest()
    return random.Random(int.from_bytes(digest, 'big'))


def draw_shape(eta, rng, frozen):
    """The centred polynomial, drawn once and independent of the thickness mean.

    Keeping the shape fixed while the mean moves is what makes the budget search a bisection:
    the trace dimension is then monotone in the mean.  Drawing a fresh shape at every step
    instead -- which is what an earlier version of this did -- destroys that monotonicity and
    the search wanders.
    """
    shape_values, shape_gradient_squared, direction = frozen
    linear = [x * math.sqrt(3 * float(1 - eta)) for x in direction(rng, 3)]
    mixed = [x * math.sqrt(float(eta)) * weight
             for x, weight in zip(direction(rng, 4), (3, 3, 3, math.sqrt(27)))]
    coefficients = tuple(_decimal(x) for x in linear + mixed)
    shape = shape_values(coefficients)
    gradient = math.sqrt(float(shape_gradient_squared(coefficients)))
    if gradient <= 0 or max(shape) <= 0 or min(shape) >= 0:
        return None
    return dict(coefficients=coefficients, shape=shape, gradient=gradient)


def corners_at_mean(mean, amplitude_fraction, drawn, upper=UPPER):
    """Eight corner values at this mean, scaled to the frozen contract's own headroom."""
    shape, gradient = drawn['shape'], drawn['gradient']
    span_max, gradient_max = Fraction('0.47'), Fraction('0.47')
    headroom = min((upper - mean) / max(shape), (mean - LOWER) / (-min(shape)),
                   span_max / (max(shape) - min(shape)),
                   Fraction(str(float(gradient_max) / gradient)))
    scale = Fraction(math.floor(headroom * amplitude_fraction * 10 ** 9), 10 ** 9)
    if scale <= 0:
        return None
    return tuple(mean + scale * value for value in shape)


def offset_for_depth(b, s):
    """The plane x + b y = d at depth fraction s = d/(1+b)."""
    return Fraction(s) * (1 + Fraction(b))


def volume_fraction_curve(source, resolution=256):
    """rho(tau) for the Schwarz-P sheet, and a self-check against the frozen bounds.

    The self-check is the point: it asserts that the bounds really are the volume-fraction range
    they are being treated as, so a cap expressed in rho cannot silently mean something else.
    """
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from cctpms.geometry.implicit import phi_p
    axis = (np.arange(resolution) + .5) / resolution
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing='ij'), axis=-1).reshape(-1, 3)
    magnitude = np.abs(np.asarray(phi_p(grid, 1.0), dtype=float))

    def rho(tau):
        return float((magnitude <= float(tau)).mean())

    for bound, expected in MEASURED_VOLUME_FRACTION.items():
        if abs(rho(bound) - expected) > 2e-3:
            raise ValueError(f'VOLUME_FRACTION_CURVE_DISAGREES_AT_{bound} '
                             f'{rho(bound):.4f} vs {expected:.4f}')
    return rho


def tau_at_volume_fraction(rho, target, steps=60):
    low, high = 0.0, 3.0
    for _ in range(steps):
        middle = (low + high) / 2
        if rho(middle) < target:
            low = middle
        else:
            high = middle
    return Fraction(f'{(low + high) / 2:.6f}')


def offset_for_volume(b, volume):   # replaced at run time by the frozen sampler's own inverse
    raise NotImplementedError('OFFSET_FOR_VOLUME_COMES_FROM_THE_FROZEN_SAMPLER')


def retained_volume(b, d):
    b, d = Fraction(b), Fraction(d)
    if d <= 0:
        return Fraction(0)
    if d >= 1 + b:
        return Fraction(1)
    if b == 0:
        return d
    if d < b:
        return d * d / (2 * b)
    if d <= 1:
        return d - b / 2
    return 1 - (1 + b - d) ** 2 / (2 * b)


def existing_coverage(manifest, dataset, q_ceiling):
    """The (angle, retained volume) points already labelable, in the coordinates coverage uses."""
    rows = json.loads(Path(manifest).read_text())
    rows = rows['rows'] if isinstance(rows, dict) and 'rows' in rows else rows
    points = []
    for row in rows:
        sample = json.loads((Path(row['packet']) / 'SAMPLE.json').read_text())
        plane = sample['geometry'].get('cut_plane')
        if not plane:
            continue
        d = float(Fraction(str(plane[3])))
        if d >= 2:
            continue
        if int(sample['full_trace_dimension']) > q_ceiling:
            continue
        b = float(Fraction(str(plane[1])))
        points.append((math.degrees(math.atan(b)) / MAX_ANGLE_DEGREES,
                       float(retained_volume(Fraction(str(plane[1])), Fraction(str(plane[3]))))))
    return points


class MaximinFill:
    """Farthest-point insertion in the (angle, depth) square, with replacement.

    Greedy maximin is what a coverage repair wants -- each new point goes where the current set
    is emptiest, so the largest hole shrinks monotonically rather than on average -- but it makes
    replacement essential rather than optional.  Maximin picks the edges and corners first,
    because those are farthest from the existing cloud, and the edges are exactly where a target
    turns out to be unbuildable.  Dropping such a target instead of replacing it therefore throws
    away the picks that were covering the biggest holes: measured on a 300-point run, 27 drops at
    the shallow edge left the largest hole at 0.1195 when the insertion order had earned 0.0373.
    So `reject` blanks the neighbourhood of a target that cannot be built and the next `take`
    goes elsewhere, which is the frozen sampler's own replacement policy -- a new case in the
    same stratum -- applied to a stratum that is a region rather than a bin.

    The second coordinate is RETAINED VOLUME, not the depth fraction s = d/(1+b).  Using s was
    an error and the rejections showed it: equal steps in s are wildly unequal in retained
    material once the plane is tilted, because at b = 1 the point s = 0.033 is a corner wedge of
    volume 0.0005 while at b = 0 it is a slab of volume 0.033.  A maximin fill in (angle, s)
    therefore spends itself on a band of the square that holds no material at all -- measured,
    274 rejections out of 574 attempts, every one of them at s <= 0.122, with the screen
    returning q = 0 or a few hundred.  Retained volume is the comparable measure, it is what the
    frozen sampler stratifies on, and it is monotone in cost, so covering it uniformly spreads
    the trace-dimension budget uniformly too.  Both tails are still excluded: volume 0 is an
    empty cell and volume 1 an uncut one.
    """

    def __init__(self, existing, volume_low, volume_high, resolution=181, seed=20260921):
        grid = np.linspace(0.0, 1.0, resolution)
        points = np.stack(np.meshgrid(grid, grid, indexing='ij'), axis=-1).reshape(-1, 2)
        self.candidates = points[(points[:, 1] >= volume_low) & (points[:, 1] <= volume_high)]
        self.spacing = 1.0 / (resolution - 1)
        self.live = np.ones(len(self.candidates), dtype=bool)
        chosen = np.array(existing, dtype=float).reshape(-1, 2)
        if len(chosen):
            self.best = np.min(((self.candidates[:, None, :] - chosen[None, :, :]) ** 2).sum(axis=2), axis=1)
        else:
            self.best = np.full(len(self.candidates), np.inf)
        self.rng = np.random.default_rng(seed)

    def take(self):
        if not self.live.any():
            return None, None
        masked = np.where(self.live, self.best, -1.0)
        top = np.flatnonzero(masked >= masked.max() - 1e-15)
        index = int(top[self.rng.integers(len(top))])
        return index, (float(self.candidates[index, 0]), float(self.candidates[index, 1]))

    def accept(self, index):
        point = self.candidates[index]
        self.best = np.minimum(self.best, ((self.candidates - point) ** 2).sum(axis=1))
        self.live[index] = False

    def reject(self, index, radius_cells=2):
        """Blank this target and its immediate neighbourhood; the region is unbuildable."""
        point = self.candidates[index]
        near = ((self.candidates - point) ** 2).sum(axis=1) <= (radius_cells * self.spacing) ** 2
        self.live[near] = False


def fit_thickness_to_budget(b, d, ceiling, rng, frozen, estimate, upper, attempts=12):
    """The thickest admissible field whose predicted trace dimension still fits the ceiling.

    The budget is a ceiling, not a target.  Treating it as a target is what an earlier version
    did, and it is wrong twice over: it forces a shallow cut to carry a thick wall it does not
    need, and it declares the deep, steep corner unreachable when in fact a thin wall reaches it
    perfectly well.  Taking the thickest field that fits instead puts as much material as the
    budget allows into every cell, and it produces the thickness-versus-depth trade-off by
    itself -- thin walls where the cut is deep, thick walls where it is shallow -- which is the
    curve the in-domain packets already trace out (tau_mean reaches 0.868 only at shallow cuts).

    Monotonicity in the mean is what makes this a bisection, so the shape is drawn once, before
    the search, and only the mean moves.
    """
    amplitude = Fraction(1 + int(rng.random() * 999999), 1000000) * Fraction(2, 3)
    eta = Fraction(0) if rng.random() < 0.5 else Fraction(1 + int(rng.random() * 999999), 3000000)
    drawn = draw_shape(eta, rng, frozen)
    if drawn is None:
        return None, []
    normal = ('1', str(b), '0')
    lo, hi = LOWER + Fraction(1, 200), upper - Fraction(1, 200)
    trace, best = [], None

    def probe(mean):
        corners = corners_at_mean(mean, amplitude, drawn, upper)
        if corners is None:
            return None, None
        return corners, estimate(corners, normal, str(d))

    corners, q = probe(lo)
    if corners is None or q is None:
        return None, trace
    trace.append(dict(step=-1, mean=float(lo), q=round(q, 1)))
    if q > ceiling:
        # even the thinnest admissible wall busts the budget: a genuine exclusion, not a miss
        return dict(corners=corners, coefficients=drawn['coefficients'], mean=lo, q=q,
                    amplitude=amplitude, eta=eta, fits=False), trace
    best = dict(corners=corners, coefficients=drawn['coefficients'], mean=lo, q=q,
                amplitude=amplitude, eta=eta, fits=True)
    corners, q = probe(hi)
    if corners is not None and q is not None:
        trace.append(dict(step=-2, mean=float(hi), q=round(q, 1)))
        if q <= ceiling:
            return dict(corners=corners, coefficients=drawn['coefficients'], mean=hi, q=q,
                        amplitude=amplitude, eta=eta, fits=True), trace
    for step in range(attempts):
        mid = (lo + hi) / 2
        corners, q = probe(mid)
        if corners is None or q is None:
            hi = mid
            continue
        trace.append(dict(step=step, mean=float(mid), q=round(q, 1)))
        if q <= ceiling:
            lo = mid
            if q > best['q']:
                best = dict(corners=corners, coefficients=drawn['coefficients'], mean=mid, q=q,
                            amplitude=amplitude, eta=eta, fits=True)
        else:
            hi = mid
    return best, trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--count', type=int, default=300)
    ap.add_argument('--q-budget', type=float, default=19500.,
                    help='ceiling on the PREDICTED trace dimension; the thickest field that fits wins')
    ap.add_argument('--q-floor', type=float, default=2000.,
                    help='a cell with fewer coordinates than this retains almost no material')
    ap.add_argument('--volume-low', type=float, default=.02)
    ap.add_argument('--volume-high', type=float, default=.98)
    ap.add_argument('--volume-fraction-max', type=float, default=.40,
                    help='cap on the cell volume fraction; the frozen bound 0.8775 is rho = 0.5026, '
                         'and 92.4 %% of the packets above rho = 0.40 are past the label ceiling anyway')
    ap.add_argument('--q-ceiling', type=int, default=23000,
                    help='what counts as already labelable, for the existing-coverage baseline')
    ap.add_argument('--holdout-fraction', type=float, default=.2)
    ap.add_argument('--seed', type=int, default=20260921)
    ap.add_argument('--batch', default='F300')
    ap.add_argument('--sub', type=int, default=4)
    ap.add_argument('--manifest', type=Path,
                    default=Path('/root/autodl-tmp/CLAUDE_LABELS_20260917/V2_LABELS.json'))
    ap.add_argument('--dataset', type=Path,
                    default=Path('/root/autodl-tmp/CUTFEM_INGEST_R38/dataset_independent_20260910'))
    ap.add_argument('--source', type=Path,
                    default=Path('/root/autodl-tmp/CUTFEM_INGEST_R38/source_independent_6624dc8_20260910'))
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(a.source)); sys.path.insert(0, str(a.source / 'src'))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    # The private helpers ARE the contract; importing them keeps this from drifting away from
    # the construction that produced the existing dataset.
    from stage_cutfem_graded.sampling import _shape_values, _shape_gradient_squared, _direction
    from stage_cutfem_graded.thickness import Thickness
    from stage_cutfem_graded.dataset import field_class
    from stage_cutfem_cluster.common import validate_case
    from stage_cutfem_graded.sampling import offset_for_volume as frozen_offset_for_volume
    globals()['offset_for_volume'] = frozen_offset_for_volume
    from superelement.objective.cut_budget import estimate_trace_dimension
    frozen = (_shape_values, _shape_gradient_squared, _direction)

    def estimate(corners, normal, offset):
        return estimate_trace_dimension([str(c) for c in corners], normal, offset,
                                        n=32, sub=a.sub, source=a.source / 'src')

    t0 = time.time()
    rho = volume_fraction_curve(a.source / 'src')
    upper = min(UPPER, tau_at_volume_fraction(rho, a.volume_fraction_max))
    print(json.dumps(dict(phase='thickness_bound', volume_fraction_max=a.volume_fraction_max,
                          tau_upper=str(upper), frozen_tau_upper=str(UPPER),
                          frozen_volume_fraction=MEASURED_VOLUME_FRACTION)), flush=True)
    existing = existing_coverage(a.manifest, a.dataset, a.q_ceiling)
    fill = MaximinFill(existing, a.volume_low, a.volume_high, seed=a.seed)
    print(json.dumps(dict(phase='plan', existing=len(existing), requested=a.count,
                          candidates=int(fill.live.sum()),
                          largest_hole_before=float(math.sqrt(fill.best.max())))), flush=True)

    split_rng = random.Random(a.seed ^ 0x5f5f)
    rows, unreachable, traces = [], [], []
    index = -1
    while len(rows) < a.count:
        slot, target = fill.take()
        if slot is None:
            break
        index += 1
        u, volume = target
        theta = u * MAX_ANGLE_DEGREES
        b = _decimal(math.tan(math.radians(theta)))
        d = offset_for_volume(b, _decimal(volume))
        s = float(d) / (1 + float(b))
        rng = _seeded(a.seed, a.batch, index, 'thickness')
        best, trace = fit_thickness_to_budget(b, d, a.q_budget, rng, frozen, estimate, upper)
        traces.append(dict(index=index, angle_degrees=theta, volume=volume, depth=s, steps=trace))
        if best is None or not best['fits'] or best['q'] < a.q_floor:
            reason = ('NO_ADMISSIBLE_THICKNESS_FIELD' if best is None else
                      'THINNEST_ADMISSIBLE_WALL_EXCEEDS_BUDGET' if not best['fits'] else
                      'RETAINS_ALMOST_NO_MATERIAL')
            unreachable.append(dict(index=index, angle_degrees=theta, volume=volume, depth=s,
                                    best_q=None if best is None else best['q'], reason=reason))
            fill.reject(slot)
            continue
        fill.accept(slot)
        thickness = Thickness(best['corners'])
        mother = f'{a.batch}_MOTHER_{a.seed}_{index:04d}_r0'
        split = 'holdout' if split_rng.random() < a.holdout_fraction else 'train'
        row = dict(case_id=f'{a.batch}_{index:04d}_r0', normal=['1', str(b), '0'], offset=str(d),
                   tau_corners=[str(v) for v in thickness.corners],
                   mother_field_id=mother, split=split,
                   thickness_class=field_class(thickness))
        validate_case(row)
        row['design'] = dict(angle_degrees=theta, target_angle_unit=u, depth_fraction=s,
                             target_volume=volume, retained_volume=float(retained_volume(b, d)),
                             predicted_trace_dimension=best['q'],
                             thickness_mean=float(best['mean']),
                             volume_fraction_at_mean=rho(best['mean']),
                             span=float(max(thickness.corners) - min(thickness.corners)),
                             maximum_gradient_norm=math.sqrt(float(thickness.maximum_gradient_squared)),
                             amplitude_fraction=str(best['amplitude']),
                             mixed_variance_fraction=str(best['eta']),
                             centered_shape_coefficients=[str(c) for c in best['coefficients']])
        rows.append(row)
        if len(rows) % 25 == 0:
            print(json.dumps(dict(done=len(rows), of=a.count, unreachable=len(unreachable),
                                  elapsed=round(time.time() - t0, 1))), flush=True)

    got = [(math.degrees(math.atan(float(Fraction(r['normal'][1])))) / MAX_ANGLE_DEGREES,
            r['design']['retained_volume']) for r in rows]
    combined = np.array(existing + got, dtype=float)
    grid = np.linspace(0, 1, 181)
    probe = np.stack(np.meshgrid(grid, grid, indexing='ij'), axis=-1).reshape(-1, 2)
    # Measure the hole over the region the fill is allowed to reach.  Probing the whole square
    # instead makes the number meaningless: the excluded degenerate edges dominate it and it
    # cannot improve however well the interior is covered.
    probe = probe[(probe[:, 1] >= a.volume_low) & (probe[:, 1] <= a.volume_high)]
    def largest_hole(points):
        if not len(points):
            return None
        return float(np.sqrt(np.min(((probe[:, None, :] - points[None, :, :]) ** 2).sum(axis=2), axis=1)).max())
    qs = np.array([r['design']['predicted_trace_dimension'] for r in rows])
    summary = dict(batch=a.batch, seed=a.seed, requested=a.count, generated=len(rows),
                   unreachable=len(unreachable), existing_points=len(existing),
                   q_budget=a.q_budget, q_floor=a.q_floor,
                   volume_fraction_max=a.volume_fraction_max, tau_upper=str(upper),
                   volume_range=[a.volume_low, a.volume_high],
                   unreachable_reasons={r: sum(1 for x in unreachable if x['reason'] == r)
                                        for r in sorted({x['reason'] for x in unreachable})},
                   predicted_q=dict(min=float(qs.min()), median=float(np.median(qs)),
                                    max=float(qs.max())) if len(qs) else None,
                   largest_hole_existing=largest_hole(np.array(existing, dtype=float)),
                   largest_hole_combined=largest_hole(combined),
                   candidates_left=int(fill.live.sum()),
                   splits={k: sum(1 for r in rows if r['split'] == k) for k in ('train', 'holdout')},
                   thickness_class={k: sum(1 for r in rows if r['thickness_class'] == k)
                                    for k in ('AFFINE', 'MIXED', 'UNIFORM_ANCHOR')},
                   seconds=time.time() - t0)
    (a.output / 'CASES.json').write_text(json.dumps(rows, indent=1))
    (a.output / 'UNREACHABLE.json').write_text(json.dumps(unreachable, indent=1))
    (a.output / 'SEARCH_TRACES.json').write_text(json.dumps(traces, indent=1))
    (a.output / 'SUMMARY.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == '__main__':
    main()
