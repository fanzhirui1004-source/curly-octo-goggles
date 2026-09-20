"""How many interface coordinates a candidate cell will have, before any assembly.

Everything that limits this project runs on `q`, the trace dimension: the teacher's own cost,
the 32 GB card that builds `M_q`, and the `q^2` bytes each label occupies.  Until now `q` was
only ever discovered *after* paying for the geometry, so the dataset's in-domain subset is
whatever the cost ceiling happened to admit rather than what was designed.  Measured on the
607 cut packets, that censoring is not neutral: at a fixed thickness a deep cut carries about
1.7x the dofs of a shallow one (1.61-1.76 across five tau bands), so the ceiling throws away
deep cuts and the surviving coverage is clumped toward shallow.

Two routines, at two prices.

`trace_nodes_from_topology` is exact.  The teacher's stage-1 topology emits one patch per
active cell face that lies on the box boundary -- and only the box boundary, because the
certified macro plane adds no trace face -- so the trace is the union of the Q2 node patches
those faces carry.  Verified against `full_trace_dimension` on packets 0257 (q = 19038) and
0258 (q = 27744): ratio 1.0000 on both.  It costs nothing, because stage 1 runs anyway; a
worker can compile the topology, read `q`, and abandon a candidate that busts the budget
before the expensive assembly starts.

That identity is exact for an UNCUT cell, where the teacher records
`box_coordinates_no_macro_cut_face` and the trace is nothing but the box coefficients
(verified on packets 0257 and 0258, ratio 1.0000 on both).  A genuine cut cell is different
and the identity does not hold there: `compile_trace` adds one exact functional per sextic
triangle point on every MACRO_CUT_FACE polygon and then keeps a maximal independent subset,
so the cut contribution is the RANK of those functionals, not a node count.  Packet 0267 is
the proof that no node count can work: its trace, 28308, exceeds the 24732 that the whole box
boundary would carry.  The rank needs the assembled body's node list, so an exact `q` before
assembly is not available at any price.

`estimate_trace_dimension` is therefore a calibrated screen, not an identity.  It counts two
things by plain sampling -- the Q2 nodes on box faces that carry material on the retained
side, and the cells the cut plane passes through that carry material -- and combines them
with coefficients fitted to every packet whose trace dimension is known.  It is a design aid
for choosing what to generate; the gate that decides is the teacher itself, which can read
`q` off its own stage-2 record and abandon a candidate before the expensive condensation.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np


BOX_FACE = {'BOX_XMIN': (0, 0), 'BOX_XMAX': (0, 1), 'BOX_YMIN': (1, 0),
            'BOX_YMAX': (1, 1), 'BOX_ZMIN': (2, 0), 'BOX_ZMAX': (2, 1)}


def _q2_face_nodes(cell, axis, side, n):
    """The nine Q2 node ids on one face of one background cell, on the (2n+1)^3 node grid."""
    grid = 2 * n + 1
    base = [2 * int(c) for c in cell]
    spans = [range(base[d], base[d] + 3) for d in range(3)]
    spans[axis] = range(base[axis] + 2 * side, base[axis] + 2 * side + 1)
    return [(a * grid + b) * grid + c for a in spans[0] for b in spans[1] for c in spans[2]]


def trace_nodes_from_topology(topology, n):
    """Exact trace node count from a compiled topology.  No assembly, no approximation."""
    nodes = set()
    other = set()
    for patch in topology['patches']:
        face = BOX_FACE.get(patch['tag'])
        if face is None:
            other.add(patch['tag'])
            continue
        nodes.update(_q2_face_nodes(patch['parent'], face[0], face[1], n))
    if other:
        raise ValueError(f'UNEXPECTED_TRACE_FACE_TAGS {sorted(other)}')
    return len(nodes)


def trace_dimension_from_topology(topology, n):
    return 3 * trace_nodes_from_topology(topology, n)


def _tau_at(corners, points):
    """Trilinear thickness on the unit cell from its eight corner values, corner order 000..111."""
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    v = np.asarray(corners, dtype=float).reshape(2, 2, 2)
    wx = np.stack((1 - x, x)); wy = np.stack((1 - y, y)); wz = np.stack((1 - z, z))
    out = np.zeros(points.shape[0])
    for i in range(2):
        for j in range(2):
            for k in range(2):
                out += v[i, j, k] * wx[i] * wy[j] * wz[k]
    return out


def face_and_cut_counts(tau_corners, normal, offset, n=32, sub=4, source=None):
    """The two cheap counts the trace dimension is built from.

    Returns (box_face_nodes, cut_cells): the Q2 nodes on box faces whose cell carries material
    on the retained side, and the number of material cells the plane passes through.  Sampling
    on a (sub+1)^3 grid per cell can only miss a sliver, never invent one, so both counts rise
    monotonically with `sub` toward the certified values.
    """
    if source is not None and str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from cctpms.geometry.implicit import phi_p

    corners = [float(Fraction(str(c))) for c in tau_corners]
    plane = np.array([float(Fraction(str(v))) for v in normal])
    d0 = float(Fraction(str(offset)))
    h = 1.0 / n
    t = np.linspace(0.0, 1.0, sub + 1)
    ox, oy, oz = np.meshgrid(t, t, t, indexing='ij')
    off = np.stack((ox.ravel(), oy.ravel(), oz.ravel()), axis=1) * h

    def material_and_side(cells):
        points = ((cells * h)[:, None, :] + off[None, :, :]).reshape(-1, 3)
        phi = np.asarray(phi_p(points, 1.0), dtype=float)
        inside = phi * phi - _tau_at(corners, points) ** 2 <= 0.0
        g = points @ plane
        return (inside.reshape(len(cells), -1), (g <= d0).reshape(len(cells), -1),
                (g >= d0).reshape(len(cells), -1))

    nodes = set()
    for axis in range(3):
        for side in (0, 1):
            u, v = [d for d in range(3) if d != axis]
            iu, iv = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
            cells = np.zeros((n * n, 3), dtype=np.int64)
            cells[:, u] = iu.ravel(); cells[:, v] = iv.ravel()
            cells[:, axis] = 0 if side == 0 else n - 1
            inside, below, _ = material_and_side(cells)
            for cell in cells[(inside & below).any(axis=1)]:
                nodes.update(_q2_face_nodes(cell, axis, side, n))

    if d0 >= 2.0:
        return len(nodes), 0
    # cells the plane passes through: cheap bounding-box straddle first, then material
    grid = np.stack(np.meshgrid(*(np.arange(n),) * 3, indexing='ij'), axis=-1).reshape(-1, 3)
    lo = grid * h
    corner_offsets = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)]) * h
    g_corners = (lo[:, None, :] + corner_offsets[None, :, :]) @ plane
    straddle = (g_corners.min(axis=1) <= d0) & (g_corners.max(axis=1) >= d0)
    candidates = grid[straddle]
    if len(candidates) == 0:
        return len(nodes), 0
    inside, below, above = material_and_side(candidates)
    live = (inside & below).any(axis=1) & (inside & above).any(axis=1)
    return len(nodes), int(live.sum())


# Fitted on every packet with a known trace dimension; see CALIBRATION.json beside the run that
# produced them.  An uncut cell has cut_cells = 0 and the box term alone must carry it, so the
# intercept is pinned at zero and only the two slopes are free.
# Fitted on all 920 packets at sub=4 (docs/data/cut_budget_20260921/): median |error| 1.5 % on
# the 607 cut packets and 0.54 % on the 313 uncut ones, p95 6.2 % and 2.1 %, worst 15.9 %.  The
# box slope lands at 0.979 because the sampled cell set is a ~2 % superset of the certified one,
# and the cut slope at 8.44 independent functionals per cut cell, against the 9 a Q2 face carries.
CALIBRATION = dict(box=0.9789077391570264, cut=8.443926144628383, fitted=True,
                   packets=920, sub=4, cut_median_abs=0.0149, cut_p95_abs=0.0623)


def estimate_trace_dimension(tau_corners, normal, offset, n=32, sub=4, source=None,
                             calibration=None):
    box_nodes, cut_cells = face_and_cut_counts(tau_corners, normal, offset, n=n, sub=sub, source=source)
    c = calibration or CALIBRATION
    return 3.0 * (c['box'] * box_nodes + c['cut'] * cut_cells)


def main():
    ap = argparse.ArgumentParser(description='validate the cheap screen against every known packet')
    ap.add_argument('--dataset', type=Path,
                    default=Path('/root/autodl-tmp/CUTFEM_INGEST_R38/dataset_independent_20260910'))
    ap.add_argument('--source', type=Path,
                    default=Path('/root/autodl-tmp/CUTFEM_INGEST_R38/source_independent_6624dc8_20260910/src'))
    ap.add_argument('--sub', type=int, default=4)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)

    samples = sorted(a.dataset.glob('*/payload/packet/SAMPLE.json'))
    if a.limit:
        samples = samples[:a.limit]
    rows = []
    t0 = time.time()
    for path in samples:
        sample = json.loads(path.read_text())
        geometry = sample['geometry']
        plane = geometry.get('cut_plane')
        normal = tuple(map(str, plane[:3])) if plane else ('1', '0', '0')
        offset = str(plane[3]) if plane else '2'
        n = int(sample['n'])
        truth = int(sample['full_trace_dimension'])
        t1 = time.time()
        box_nodes, cut_cells = face_and_cut_counts(geometry['tau_corners'], normal, offset,
                                                   n=n, sub=a.sub, source=a.source)
        rows.append(dict(case=path.parts[-4], n=n, cut=float(Fraction(offset)) < 2, truth=truth,
                         box_nodes=box_nodes, cut_cells=cut_cells,
                         seconds=time.time() - t1,
                         tau_mean=float(np.mean([float(Fraction(str(c))) for c in geometry['tau_corners']]))))
        if len(rows) % 100 == 0:
            print(json.dumps(dict(done=len(rows), of=len(samples), elapsed=round(time.time() - t0, 1))), flush=True)

    # least squares through the origin on the two counts, then report what it actually does
    A = np.array([[r['box_nodes'], r['cut_cells']] for r in rows], dtype=float)
    y = np.array([r['truth'] / 3.0 for r in rows], dtype=float)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    calibration = dict(box=float(coef[0]), cut=float(coef[1]), fitted=True, packets=len(rows), sub=a.sub)
    for r in rows:
        r['predicted'] = 3.0 * (coef[0] * r['box_nodes'] + coef[1] * r['cut_cells'])
        r['relative'] = r['predicted'] / r['truth'] - 1.0
    rel = np.array([r['relative'] for r in rows])
    cut = np.array([r['cut'] for r in rows])
    summary = dict(packets=len(rows), sub=a.sub, calibration=calibration,
                   seconds_per_packet=float(np.median([r['seconds'] for r in rows])),
                   median_relative=float(np.median(rel)), mean_relative=float(rel.mean()),
                   p05=float(np.quantile(rel, .05)), p95=float(np.quantile(rel, .95)),
                   max_abs_relative=float(np.abs(rel).max()),
                   within_2pct=int((np.abs(rel) <= .02).sum()), within_5pct=int((np.abs(rel) <= .05).sum()),
                   cut_packets=int(cut.sum()), box_packets=int((~cut).sum()),
                   cut_median_abs=float(np.median(np.abs(rel[cut]))) if cut.any() else None,
                   cut_p95_abs=float(np.quantile(np.abs(rel[cut]), .95)) if cut.any() else None,
                   box_median_abs=float(np.median(np.abs(rel[~cut]))) if (~cut).any() else None,
                   box_p95_abs=float(np.quantile(np.abs(rel[~cut]), .95)) if (~cut).any() else None)
    (a.output / 'ROWS.json').write_text(json.dumps(rows, indent=1))
    (a.output / 'CALIBRATION.json').write_text(json.dumps(calibration, indent=1))
    (a.output / 'SUMMARY.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == '__main__':
    main()
