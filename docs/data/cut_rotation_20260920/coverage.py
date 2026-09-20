"""The cross-seat coverage of the coordinate-descriptor space, in the teacher's basis and in the
nodal pullback, for the SAME cut seats.

`cross.py` found no cross-seat ambiguity but something sharper: box coordinates of a NEW seat land
essentially on top of coordinates already seen (relative aggregate distance 0.0016, target agreeing
to 3e-4), while cut coordinates of a new seat sit 0.135 away -- 85x further -- in a space where the
target moves about 1.0 relative over that distance.  Box generalises because a box coordinate is a
background node, and nodes are a finite repeating vocabulary shared by every seat; a cut coordinate
is a pivot-chosen functional, continuously parameterised by where the plane cuts, so it recurs
nowhere.

The nodal pullback turns every cut coordinate into a background node.  The prediction is therefore
sharp and cheap to test: under the pullback the cut coordinates' cross-seat nearest-neighbour
distance should collapse from 0.135 towards the box value.  This measures exactly that, on the same
seats, with the same descriptor code and the same metric.

    python coverage.py <seat> <seat> ...
"""
import json, pathlib, sys
import numpy as np
from scipy.spatial import cKDTree

MANIFEST = '/root/autodl-tmp/CLAUDE_LABELS_20260917/V2_LABELS.json'
sys.path.insert(0, '/root/_inject_src')
from superelement.equi.context import compile_equi_inputs, compile_from_trace


def aggregates(ctx):
    return np.concatenate([np.asarray(ctx['node_scalar']).reshape(ctx['count'], -1),
                           np.asarray(ctx['node_vector']).reshape(ctx['count'], -1),
                           np.asarray(ctx['node_tensor']).reshape(ctx['count'], -1),
                           np.asarray(ctx['faces']).reshape(ctx['count'], -1),
                           np.asarray(ctx['pos']).reshape(ctx['count'], -1)], axis=1)


def both_contexts(seat):
    """The teacher-basis context, and the nodal-pullback context on the same cell's support."""
    row = [r for r in json.load(open(MANIFEST)) if int(r['seat']) == int(seat)][0]
    cache = dict(np.load(row['trace_cache'], allow_pickle=False))
    meta = json.loads((pathlib.Path(row['trace_cache']).parent / 'INPUT.json').read_text())['metadata']
    n = int(meta['n']); top = 2 * n
    teacher = compile_equi_inputs(cache, meta)
    # the pullback's coordinates: every background node the trace basis actually touches, each a
    # single-node functional with unit coefficient -- exactly a box coordinate's description
    indptr = np.asarray(cache['indptr']); indices = np.asarray(cache['indices'])
    nodes = np.asarray(cache['background_nodes'])
    used = np.unique(indices)
    support_int = np.column_stack(np.unravel_index(nodes[used], (top + 1,) * 3)).astype(np.int64)
    material = meta['material']; gp = meta['gp']
    nodal = compile_from_trace(np.arange(len(used) + 1), support_int, np.ones(len(used)),
                              np.zeros(len(used)), np.asarray(cache['tau_corners'], dtype=np.float64),
                              np.asarray(cache['cut_plane'], dtype=np.float64), n,
                              [material['E'], material['nu'], gp['gamma'], 1. / n])
    kind = np.asarray(teacher['kind'])
    return dict(seat=int(seat), teacher_agg=aggregates(teacher), teacher_kind=kind,
                nodal_agg=aggregates(nodal), nodal_count=len(used),
                teacher_cut=int((kind > 0).sum()), teacher_box=int((kind == 0).sum()))


def coverage(blocks, key, select=None):
    """For every coordinate, the distance to the nearest coordinate of a DIFFERENT seat.

    Built one seat at a time against the pool of all other seats, so the answer cannot be limited
    by a k-nearest cutoff: an earlier version asked for 24 neighbours and every one of them came
    from the coordinate's own seat, which is itself the point being measured here.
    """
    per = []
    for b in blocks:
        M = b[key]
        per.append(M[select(b)] if select is not None else M)
    got = []
    for i, M in enumerate(per):
        other = np.concatenate([p for j, p in enumerate(per) if j != i])
        dist, _ = cKDTree(other).query(M, k=1)
        got.append(dist / np.maximum(np.linalg.norm(M, axis=1), 1e-300))
    got = np.concatenate(got)
    return dict(coordinates=int(sum(len(M) for M in per)),
                cross_seat_nearest_relative=dict(min=float(got.min()), p05=float(np.quantile(got, .05)),
                                                 median=float(np.median(got)), p95=float(np.quantile(got, .95)),
                                                 max=float(got.max())))


if __name__ == '__main__':
    seats = [int(s) for s in sys.argv[1:]]
    blocks = [both_contexts(s) for s in seats]
    print(json.dumps(dict(seats=seats, teacher_cut=[b['teacher_cut'] for b in blocks],
                          teacher_box=[b['teacher_box'] for b in blocks],
                          nodal=[b['nodal_count'] for b in blocks])), flush=True)
    out = dict(
        teacher_basis_cut_rows=coverage(blocks, 'teacher_agg', lambda b: b['teacher_kind'] > 0),
        teacher_basis_box_rows=coverage(blocks, 'teacher_agg', lambda b: b['teacher_kind'] == 0),
        nodal_pullback_all_coordinates=coverage(blocks, 'nodal_agg'))
    for k, v in out.items():
        print(json.dumps({k: v}), flush=True)
    json.dump(out, open('/root/_coverage.json', 'w'), indent=1)
