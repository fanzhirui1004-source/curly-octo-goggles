"""Generate the cut geometries the dataset is missing, under a trace-dimension budget.

The cut family in use is not too small so much as clumped and censored.  Two measurements set
this up.  First, the teacher's own sampler is uniform in retained box volume -- across all 607
cut packets the ten volume deciles hold 52 to 66 each -- so the generator is not what produced
the gaps.  Second, at fixed thickness a deep cut carries about 1.7x the dofs of a shallow one
(1.61-1.76 across five tau bands), and a steeper plane cuts a larger polygon and so carries
more cut coordinates.  The q ceiling that decides what we can label is therefore a filter on
depth and angle, and the surviving subset leans shallow and shallow-angled: in-domain the
volume deciles run 52 at the bottom to 17 at the top.

So the fix is not to sample harder in the same way and hope.  It is to choose the coverage
directly, in all three directions that matter: the plane's ANGLE, the RETAINED VOLUME it leaves,
and the cell's VOLUME FRACTION.  Density has to be its own axis rather than a consequence of the
other two.  An earlier version made it a consequence -- it bisected the thickness mean until the
predicted trace landed in a band, then took the thickest field that fit -- and that collapsed the
axis: of 300 designed cells, over half came out pinned at the density ceiling and the whole design
spanned rho in [0.214, 0.397] with nothing thin at all.  Since tau is the design variable the
deployed network has to differentiate, a fill that holds tau almost constant is close to worthless
however well it covers angle and depth.  **There is no search over thickness left in here**: the
mean follows from the requested volume fraction, and only the shape of the grading is drawn.

The trace-dimension budget is then a constraint on that three-dimensional box rather than a knob
inside it.  Each target is screened in 82 ms (`cut_budget`, median error 1.5 % on cut cells), and a
target the budget cannot afford is reported as unreachable and REPLACED rather than quietly
dropped, because a silently shrinking design is how the current coverage happened in the first
place.  What comes back is a map of which (angle, depth, density) combinations this card can label.

Nothing here re-derives the thickness contract.  The eight corner values are built by the
frozen teacher's own construction -- one centred polynomial in x, y, z, xy, xz, yz, xyz scaled
as a whole -- and handed to the frozen `Thickness`, which enforces corners in [0.1755, 0.8775],
span <= 0.47 and |grad tau| <= 0.47.  The plane is (1, tan theta, 0) with theta in [0, 45), the
one family `GradedContract` accepts: a normal with all three components nonzero raises
GRADED_SINGLE_ZERO_COMPONENT_PLANE_FAMILY.

Splits are assigned to mother fields before any geometry is generated, so the held-out set is
fixed before it can be chosen to flatter a result.

    python -m superelement.objective.fill_cut_family --count 300 --output DIR \
        --q-budget 19500 --volume-fraction-min 0.10 --volume-fraction-max 0.40
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

    The shape is drawn once and the mean is imposed separately, so a cell's grading and its
    density are independent.  An earlier version drew a fresh shape at every step of a bisection
    over the mean, which destroyed the monotonicity that bisection depended on; there is no
    bisection left, but the separation is still what makes a requested density mean anything.
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
    """Eight corner values at this mean, scaled to the frozen contract's own headroom.

    At EITHER end of the density axis one headroom term collapses -- (mean - LOWER) at the bottom,
    (upper - mean) at the top -- and no nonzero scale survives the frozen contract, so the cell can
    only be UNGRADED there.  Returning nothing made both ends unreachable, and the affordability map
    read 0 of 99 at rho = 0.10 and again at rho = 0.40 for that reason alone, not for want of
    budget; 107 of 522 targets failed the same way.  A uniform field is a class the frozen dataset
    already recognises (`field_class` calls it UNIFORM_ANCHOR), so that is what is emitted instead.
    """
    shape, gradient = drawn['shape'], drawn['gradient']
    span_max, gradient_max = Fraction('0.47'), Fraction('0.47')
    headroom = min((upper - mean) / max(shape), (mean - LOWER) / (-min(shape)),
                   span_max / (max(shape) - min(shape)),
                   Fraction(str(float(gradient_max) / gradient)))
    scale = Fraction(math.floor(headroom * amplitude_fraction * 10 ** 9), 10 ** 9)
    if scale <= 0:
        return (mean,) * 8
    return tuple(mean + scale * value for value in shape)


def volume_fraction_curve(source, resolution=256):
    """rho(tau) for the Schwarz-P sheet and its inverse, both from one sorted sample.

    Sorting |phi| once turns both directions into a lookup: rho(tau) is a searchsorted and
    tau(rho) is an order statistic.  The obvious alternative -- bisecting rho(tau) per call --
    costs a full pass over the sample every step and is why an earlier version spent minutes
    inverting the same curve three hundred times.

    The self-check is the point of returning them together: it asserts that the frozen corner
    bounds really are the volume-fraction range they are being treated as, so a bound expressed
    in rho cannot silently come to mean something else.
    """
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from cctpms.geometry.implicit import phi_p
    axis = (np.arange(resolution) + .5) / resolution
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing='ij'), axis=-1).reshape(-1, 3)
    magnitude = np.sort(np.abs(np.asarray(phi_p(grid, 1.0), dtype=float)))
    count = len(magnitude)

    def rho(tau):
        return float(np.searchsorted(magnitude, float(tau), side='right')) / count

    def inverse(target):
        index = min(count - 1, max(0, int(round(float(target) * count)) - 1))
        return Fraction(f'{magnitude[index]:.6f}')

    for bound, expected in MEASURED_VOLUME_FRACTION.items():
        if abs(rho(bound) - expected) > 2e-3:
            raise ValueError(f'VOLUME_FRACTION_CURVE_DISAGREES_AT_{bound} '
                             f'{rho(bound):.4f} vs {expected:.4f}')
    rho.inverse = inverse
    return rho


def tau_at_volume_fraction(rho, target):
    return rho.inverse(target)


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


def existing_coverage(manifest, dataset, q_ceiling, rho, rho_low, rho_high):
    """The (angle, retained volume, density) points already labelable, in coverage coordinates."""
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
        corners = [float(Fraction(str(c))) for c in sample['geometry']['tau_corners']]
        density = rho(sum(corners) / len(corners))
        points.append((math.degrees(math.atan(b)) / MAX_ANGLE_DEGREES,
                       float(retained_volume(Fraction(str(plane[1])), Fraction(str(plane[3])))),
                       (density - rho_low) / (rho_high - rho_low)))
    return points


def affordable_map(resolution, volume_low, volume_high, density_low, density_high,
                   rho, lower_tau, upper, q_floor, q_budget, estimate, frozen, seed):
    """Which (angle, depth, density) combinations this budget can label at all.

    The three-dimensional box is not uniformly affordable and the emptiness is not a sampling
    accident: a deep cut at high density busts the trace budget however the grading is chosen,
    because depth and density both push the trace up.  Measuring coverage over the whole box
    therefore reports a hole that no amount of sampling can close, exactly as measuring the
    two-dimensional coverage over the degenerate edges did.  So the region is mapped first, on a
    coarse grid with a nearly ungraded field, and both the fill and the coverage metric are then
    confined to it.

    The map is an UPPER BOUND on affordability, not a guarantee.  It probes with a nearly uniform
    field while the fill builds with a grading amplitude drawn up to two thirds of the frozen
    headroom, and grading raises the trace on net -- thickening a corner adds cells faster than
    thinning the opposite one removes them.  So a target inside the mapped region can still bust the
    budget, and 49 of them did on the 300-cell run.  That is handled, because such a target is
    replaced like any other; it is recorded here so nobody reads the mapped fraction as a success
    rate.

    The budget carries deliberate headroom over the hard ceiling: the screen's p95 error is 6.2 %
    on cut cells, so a target screened at 19500 lands under 20700 in the worst case, against a
    measured card limit of q = 22433.
    """
    axis = np.linspace(0.0, 1.0, resolution)
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing='ij'), axis=-1).reshape(-1, 3)
    keep = (grid[:, 1] >= volume_low) & (grid[:, 1] <= volume_high)
    grid = grid[keep]
    flat = Fraction(1, 1000000)          # amplitude so small the field is effectively uniform
    rows = []
    for u, volume, density_unit in grid:
        theta = float(u) * MAX_ANGLE_DEGREES
        b = _decimal(math.tan(math.radians(theta)))
        d = offset_for_volume(b, _decimal(float(volume)))
        density = density_low + float(density_unit) * (density_high - density_low)
        mean = max(lower_tau, min(upper, tau_at_volume_fraction(rho, density)))
        built = build_at_target(b, d, mean, q_budget, _seeded(seed, 'map', len(rows)),
                               frozen, estimate, upper, amplitude=flat)
        q = None if built is None else built['q']
        rows.append(dict(angle_unit=float(u), volume=float(volume), density_unit=float(density_unit),
                         density=density, q=q,
                         affordable=bool(q is not None and q_floor <= q <= q_budget)))
    return grid, np.array([r['affordable'] for r in rows], dtype=bool), rows


class MaximinFill:
    """Farthest-point insertion in the (angle, retained volume, density) box, with replacement.

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
    frozen sampler stratifies on, and it is monotone in cost.  Both tails are still excluded:
    volume 0 is an empty cell and volume 1 an uncut one.

    The third coordinate is the cell's VOLUME FRACTION, on its own axis.  rho is very nearly
    affine in tau over the frozen window -- measured, 0.1755, 0.3502, 0.5250, 0.6994 map to
    0.1001, 0.200, 0.300, 0.400 -- so a uniform grid in rho is a uniform grid in the thickness
    the network differentiates, and stating the bound in rho is what makes it mean a density.
    """

    def __init__(self, existing, volume_low, volume_high, resolution=41, seed=20260921,
                 affordable_grid=None, affordable_mask=None):
        grid = np.linspace(0.0, 1.0, resolution)
        points = np.stack(np.meshgrid(grid, grid, grid, indexing='ij'), axis=-1).reshape(-1, 3)
        points = points[(points[:, 1] >= volume_low) & (points[:, 1] <= volume_high)]
        if affordable_grid is not None:
            nearest = np.argmin(((points[:, None, :] - affordable_grid[None, :, :]) ** 2).sum(axis=2), axis=1)
            points = points[affordable_mask[nearest]]
        if not len(points):
            raise ValueError('NO_AFFORDABLE_CANDIDATE_REMAINS')
        self.candidates = points
        self.spacing = 1.0 / (resolution - 1)
        self.live = np.ones(len(self.candidates), dtype=bool)
        chosen = np.array(existing, dtype=float).reshape(-1, 3)
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
        return index, tuple(float(v) for v in self.candidates[index])

    def accept(self, index):
        point = self.candidates[index]
        self.best = np.minimum(self.best, ((self.candidates - point) ** 2).sum(axis=1))
        self.live[index] = False

    def reject(self, index, radius_cells=2):
        """Blank this target and its immediate neighbourhood; the region is unbuildable."""
        point = self.candidates[index]
        near = ((self.candidates - point) ** 2).sum(axis=1) <= (radius_cells * self.spacing) ** 2
        self.live[near] = False


def build_at_target(b, d, mean, ceiling, rng, frozen, estimate, upper, attempts=6, amplitude=None):
    """The field at a REQUESTED thickness mean, and what the screen says its trace will be.

    There is no search over thickness any more, because thickness is a coverage axis rather than
    a free parameter: the mean comes from the requested volume fraction.  The only freedom left is
    the shape of the grading, and the loop exists solely because a drawn shape can leave no
    admissible scale at this mean (the frozen headroom collapses), in which case another shape is
    drawn.  The budget then either affords this (angle, depth, density) or it does not, and saying
    which is the useful output.
    """
    for attempt in range(attempts):
        drawn_amplitude = (amplitude if amplitude is not None
                           else Fraction(1 + int(rng.random() * 999999), 1000000) * Fraction(2, 3))
        eta = Fraction(0) if rng.random() < 0.5 else Fraction(1 + int(rng.random() * 999999), 3000000)
        drawn = draw_shape(eta, rng, frozen)
        if drawn is None:
            continue
        corners = corners_at_mean(mean, drawn_amplitude, drawn, upper)
        if corners is None:
            continue
        q = estimate(corners, ('1', str(b), '0'), str(d))
        return dict(corners=corners, coefficients=drawn['coefficients'], mean=mean, q=q,
                    amplitude=drawn_amplitude, eta=eta, fits=q <= ceiling, attempts=attempt + 1)
    return None


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
                    help='top of the density axis; the frozen bound 0.8775 is rho = 0.5026, and '
                         '92.4 %% of the packets above rho = 0.40 are past the label ceiling anyway')
    ap.add_argument('--volume-fraction-min', type=float, default=.10,
                    help='bottom of the density axis; the frozen bound 0.1755 is rho = 0.1001')
    ap.add_argument('--map-resolution', type=int, default=11,
                    help='coarse grid on which the affordable region is mapped before filling')
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
    lower_tau = max(LOWER, tau_at_volume_fraction(rho, a.volume_fraction_min))
    print(json.dumps(dict(phase='density_axis', volume_fraction=[a.volume_fraction_min, a.volume_fraction_max],
                          tau_range=[str(lower_tau), str(upper)], frozen_tau=[str(LOWER), str(UPPER)],
                          frozen_volume_fraction=MEASURED_VOLUME_FRACTION)), flush=True)
    map_grid, map_mask, map_rows = affordable_map(
        a.map_resolution, a.volume_low, a.volume_high, a.volume_fraction_min, a.volume_fraction_max,
        rho, lower_tau, upper, a.q_floor, a.q_budget, estimate, frozen, a.seed)
    print(json.dumps(dict(phase='affordable_map', probed=len(map_grid),
                          affordable=int(map_mask.sum()),
                          affordable_fraction=round(float(map_mask.mean()), 4),
                          seconds=round(time.time() - t0, 1))), flush=True)
    existing = existing_coverage(a.manifest, a.dataset, a.q_ceiling, rho,
                                 a.volume_fraction_min, a.volume_fraction_max)
    fill = MaximinFill(existing, a.volume_low, a.volume_high, seed=a.seed,
                       affordable_grid=map_grid, affordable_mask=map_mask)
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
        u, volume, density_unit = target
        theta = u * MAX_ANGLE_DEGREES
        b = _decimal(math.tan(math.radians(theta)))
        d = offset_for_volume(b, _decimal(volume))
        s = float(d) / (1 + float(b))
        density = a.volume_fraction_min + density_unit * (a.volume_fraction_max - a.volume_fraction_min)
        mean = max(lower_tau, min(upper, tau_at_volume_fraction(rho, density)))
        rng = _seeded(a.seed, a.batch, index, 'thickness')
        best = build_at_target(b, d, mean, a.q_budget, rng, frozen, estimate, upper)
        traces.append(dict(index=index, angle_degrees=theta, volume=volume, density=density,
                           thickness_mean=float(mean),
                           q=None if best is None else best['q'],
                           fits=None if best is None else best['fits']))
        if best is None or not best['fits'] or best['q'] < a.q_floor:
            reason = ('NO_ADMISSIBLE_THICKNESS_FIELD' if best is None else
                      'TRACE_EXCEEDS_BUDGET_AT_THIS_DENSITY' if not best['fits'] else
                      'RETAINS_ALMOST_NO_MATERIAL')
            unreachable.append(dict(index=index, angle_degrees=theta, volume=volume, depth=s,
                                    density=density, best_q=None if best is None else best['q'],
                                    reason=reason))
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
                             target_volume_fraction=density,
                             predicted_trace_dimension=best['q'],
                             thickness_mean=float(best['mean']),
                             volume_fraction_at_mean=rho(best['mean']),
                             span=float(max(thickness.corners) - min(thickness.corners)),
                             uniform=bool(thickness.uniform),
                             maximum_gradient_norm=math.sqrt(float(thickness.maximum_gradient_squared)),
                             amplitude_fraction=(None if thickness.uniform else str(best['amplitude'])),
                             mixed_variance_fraction=(None if thickness.uniform else str(best['eta'])),
                             # A uniform cell reached its density with scale 0, so the drawn shape
                             # was discarded; recording it would suggest it shaped the field.
                             centered_shape_coefficients=(
                                 None if thickness.uniform
                                 else [str(c) for c in best['coefficients']]))
        rows.append(row)
        if len(rows) % 25 == 0:
            print(json.dumps(dict(done=len(rows), of=a.count, unreachable=len(unreachable),
                                  elapsed=round(time.time() - t0, 1))), flush=True)

    span = a.volume_fraction_max - a.volume_fraction_min
    got = [(math.degrees(math.atan(float(Fraction(r['normal'][1])))) / MAX_ANGLE_DEGREES,
            r['design']['retained_volume'],
            (r['design']['volume_fraction_at_mean'] - a.volume_fraction_min) / span) for r in rows]
    combined = np.array(existing + got, dtype=float)
    # Measure the hole over the region the fill is allowed to reach, which is the affordable
    # region and not the whole box.  Probing everything makes the number meaningless: the
    # unaffordable deep-and-dense corner dominates it and no amount of sampling closes it.
    probe = fill.candidates
    def largest_hole(points):
        if not len(points):
            return None
        return float(np.sqrt(np.min(((probe[:, None, :] - points[None, :, :]) ** 2).sum(axis=2), axis=1)).max())
    qs = np.array([r['design']['predicted_trace_dimension'] for r in rows])
    densities = np.array([r['design']['volume_fraction_at_mean'] for r in rows])
    summary = dict(batch=a.batch, seed=a.seed, requested=a.count, generated=len(rows),
                   unreachable=len(unreachable), existing_points=len(existing),
                   q_budget=a.q_budget, q_floor=a.q_floor,
                   volume_fraction=[a.volume_fraction_min, a.volume_fraction_max],
                   tau_range=[str(lower_tau), str(upper)],
                   achieved_volume_fraction=dict(
                       min=float(densities.min()), median=float(np.median(densities)),
                       max=float(densities.max())) if len(densities) else None,
                   coverage_space='(angle/45deg, retained volume, volume fraction), 3-D maximin',
                   affordable_map=dict(resolution=a.map_resolution, probed=len(map_grid),
                                       affordable=int(map_mask.sum()),
                                       fraction=float(map_mask.mean())),
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
    (a.output / 'AFFORDABLE_MAP.json').write_text(json.dumps(map_rows, indent=1))
    (a.output / 'SUMMARY.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == '__main__':
    main()
