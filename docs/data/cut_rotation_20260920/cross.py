"""Is the cut target a function of the model's input ACROSS geometries, and how does it scale?

`injectivity.py` refuted only the strongest form of the hypothesis -- that two cut coordinates of
ONE cell can be identical to the model -- on the smallest cut seat.  Three gaps were left open and
this probe closes them:

  1. the ambiguity that would explain the measured 28x presented-to-held-out gap is the CROSS-seat
     one: a coordinate of seat A and a coordinate of seat B that the model cannot tell apart but
     whose operator differs.  Same-cell separation says nothing about it;
  2. the small seat has 142 cut coordinates; a large one has 4805 packed into the same 32^3 field
     resolution, so the nearest-neighbour distance must shrink with the count, and the scaling is
     what matters, not one seat's value;
  3. `separable` is not `learnable`: the quantity is the local ratio
     ||target_i - target_j|| / ||descriptor_i - descriptor_j||, and the two channels the model has
     order cut against box in OPPOSITE directions, so both are reported, never combined by hand.

Per coordinate: the two channels the model is handed, kept apart --

  * `aggregate`  the explicit per-coordinate features, exactly as the model receives them
                 (already divided by the pipeline's NODE_*_SCALE), plus position and face flags;
  * `pullback`   the weight vector `w_i` over the 32^3 cells with `w_i . F = sum_a c_ia F(x_a)`,
                 which is the whole of the learned-volume-field channel;

and as the target the coordinate's own diagonal 3x3 block of the trace operator, which is a local
observable and is therefore comparable across seats of different q.  Scale-free: every distance is
relative, and the target is compared both raw and after dividing by each seat's median block norm,
because a seat with thinner walls is globally softer and that is legitimate information the model
has from tau.

    python cross.py <n_cut> <n_box>
"""
import json, pathlib, sys
import numpy as np
from scipy.spatial import cKDTree

MANIFEST = '/root/autodl-tmp/CLAUDE_LABELS_20260917/V2_LABELS.json'
sys.path.insert(0, '/root/_inject_src')
from superelement.equi.context import compile_equi_inputs


def trilinear_weights(points, n):
    u = np.asarray(points, dtype=np.float64) * n - .5
    lo = np.floor(u).astype(np.int64)
    frac = u - lo
    idx = np.empty((len(u), 8), dtype=np.int64)
    wgt = np.empty((len(u), 8), dtype=np.float64)
    for corner in range(8):
        bits = ((corner >> 2) & 1, (corner >> 1) & 1, corner & 1)
        cell = np.clip(lo + np.array(bits), 0, n - 1)
        w = np.ones(len(u))
        for axis in range(3):
            w *= frac[:, axis] if bits[axis] else (1. - frac[:, axis])
        idx[:, corner] = np.ravel_multi_index(cell.T, (n, n, n))
        wgt[:, corner] = w
    return idx, wgt


def diagonal_blocks(path, count):
    """The (count, 6) upper parts of each coordinate's own 3x3 block, read straight from the packed file."""
    v = np.load(path, mmap_mode='r')
    q = 3 * count
    if q * (q + 1) // 2 != v.size:
        raise ValueError(f'PACKED_UPPER {path} {v.size} for q={q}')

    def offset(r, c):                                     # np.triu_indices order, row major
        return r * q - r * (r - 1) // 2 + (c - r)

    i = np.arange(count)
    picks = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
    out = np.empty((count, 6))
    for k, (a, b) in enumerate(picks):
        out[:, k] = v[offset(3 * i + a, 3 * i + b)]
    return out


def load(seat):
    row = [r for r in json.load(open(MANIFEST)) if int(r['seat']) == int(seat)][0]
    cache = dict(np.load(row['trace_cache'], allow_pickle=False))
    meta = json.loads((pathlib.Path(row['trace_cache']).parent / 'INPUT.json').read_text())['metadata']
    ctx = compile_equi_inputs(cache, meta)
    n = ctx['n']
    agg = np.concatenate([np.asarray(ctx['node_scalar']).reshape(ctx['count'], -1),
                          np.asarray(ctx['node_vector']).reshape(ctx['count'], -1),
                          np.asarray(ctx['node_tensor']).reshape(ctx['count'], -1),
                          np.asarray(ctx['faces']).reshape(ctx['count'], -1),
                          np.asarray(ctx['pos']).reshape(ctx['count'], -1)], axis=1)
    idx, wgt = trilinear_weights(ctx['support_pos'], n)
    rows = np.asarray(ctx['support_rows'])
    contrib = wgt * np.asarray(ctx['support_coefficients'])[:, None]
    y = diagonal_blocks(str(pathlib.Path(row['packet']) / 'S_UPPER.npy'), ctx['count'])
    return dict(seat=int(seat), n=n, kind=np.asarray(ctx['kind']), agg=agg, y=y,
                w_idx=idx, w_wgt=contrib, w_rows=rows, count=ctx['count'])


def sparse_w(entry, take):
    """w restricted to the chosen coordinates, as a dict of {coordinate: {cell: weight}}."""
    keep = np.zeros(entry['count'], dtype=bool); keep[take] = True
    mask = keep[entry['w_rows']]
    rows = entry['w_rows'][mask]
    cells = entry['w_idx'][mask].ravel()
    vals = entry['w_wgt'][mask].ravel()
    rep = np.repeat(rows, 8)
    out = {}
    for r, c, v in zip(rep, cells, vals):
        d = out.setdefault(int(r), {})
        d[int(c)] = d.get(int(c), 0.) + float(v)
    return out


def wdist(a, b):
    keys = set(a) | set(b)
    s = sum((a.get(k, 0.) - b.get(k, 0.)) ** 2 for k in keys)
    na = np.sqrt(sum(v * v for v in a.values()))
    return float(np.sqrt(s) / max(na, 1e-300))


def analyse(entries, family, want_cross):
    aggs, ys, tags, ws = [], [], [], []
    for e in entries:
        take = np.flatnonzero(e['kind'] > 0 if family == 'cut' else e['kind'] == 0)
        if len(take) == 0:
            continue
        w = sparse_w(e, take)
        scale = float(np.median(np.linalg.norm(e['y'][take], axis=1)))
        for i in take:
            aggs.append(e['agg'][i]); ys.append(e['y'][i]); tags.append((e['seat'], int(i), scale))
            ws.append(w[int(i)])
    A = np.asarray(aggs); Y = np.asarray(ys)
    An = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-300)
    tree = cKDTree(A)
    k = 8
    dist, nbr = tree.query(A, k=k + 1)
    records = []
    for i in range(len(A)):
        for slot in range(1, k + 1):
            j = int(nbr[i, slot])
            if j == i:
                continue
            cross = tags[i][0] != tags[j][0]
            if want_cross and not cross:
                continue
            if not want_cross and cross:
                continue
            da = float(dist[i, slot] / max(np.linalg.norm(A[i]), 1e-300))
            dy = float(np.linalg.norm(Y[i] - Y[j]) / max(np.linalg.norm(Y[i]), 1e-300))
            dys = float(np.linalg.norm(Y[i] / tags[i][2] - Y[j] / tags[j][2]) /
                        max(np.linalg.norm(Y[i] / tags[i][2]), 1e-300))
            records.append((da, dy, dys, i, j))
    records.sort(key=lambda r: r[0])
    out = dict(family=family, scope='cross_seat' if want_cross else 'same_seat',
               coordinates=len(A), pairs=len(records))
    if not records:
        out['status'] = 'NO_PAIRS'; return out
    da = np.array([r[0] for r in records]); dy = np.array([r[1] for r in records])
    dys = np.array([r[2] for r in records])
    ratio = dy / np.maximum(da, 1e-300)
    out.update(aggregate_distance=dict(min=float(da.min()), p05=float(np.quantile(da, .05)),
                                       median=float(np.median(da))),
               target_over_aggregate_ratio=dict(median=float(np.median(ratio)),
                                                p95=float(np.quantile(ratio, .95)),
                                                max=float(ratio.max())),
               target_relative_at_the_closest_20_pairs=dict(
                   median=float(np.median(dy[:20])), max=float(dy[:20].max())),
               target_relative_scalefree_at_the_closest_20_pairs=dict(
                   median=float(np.median(dys[:20])), max=float(dys[:20].max())))
    worst = []
    for da_, dy_, dys_, i, j in records[:6]:
        worst.append(dict(seat_i=tags[i][0], seat_j=tags[j][0], aggregate_relative=da_,
                          target_relative=dy_, target_relative_scalefree=dys_,
                          pullback_relative=wdist(ws[i], ws[j])))
    out['closest_pairs'] = worst
    return out


if __name__ == '__main__':
    n_cut, n_box = int(sys.argv[1]), int(sys.argv[2])
    desc = {r['seat']: r for r in json.load(open('/root/_seat_desc.json'))}
    rows = json.load(open(MANIFEST))
    cut, box = [], []
    for r in rows:
        s = int(r['seat']); d = desc.get(s, {})
        if (d.get('tau_mean') or 9) > 0.45:
            continue
        (cut if (d.get('n_kind1') or 0) > 0 else box).append(s)
    cut = cut[:n_cut]; box = box[:n_box]
    print(json.dumps(dict(cut_seats=cut, box_seats=box)), flush=True)
    out = []
    for family, seats in (('cut', cut), ('box', box)):
        entries = [load(s) for s in seats]
        print(json.dumps(dict(family=family, counts=[e['count'] for e in entries])), flush=True)
        for want_cross in (True, False):
            rec = analyse(entries, family, want_cross)
            out.append(rec)
            print(json.dumps(rec), flush=True)
        # gap 2: how does same-seat nearest-neighbour separation scale with the coordinate count?
        scaling = []
        for e in entries:
            take = np.flatnonzero(e['kind'] > 0 if family == 'cut' else e['kind'] == 0)
            if len(take) < 2:
                continue
            A = e['agg'][take]
            d, _ = cKDTree(A).query(A, k=2)
            rel = d[:, 1] / np.maximum(np.linalg.norm(A, axis=1), 1e-300)
            scaling.append(dict(seat=e['seat'], coordinates=int(len(take)),
                                nearest_relative_min=float(rel.min()),
                                nearest_relative_median=float(np.median(rel))))
        out.append(dict(family=family, scaling=scaling))
        print(json.dumps(dict(family=family, scaling=scaling)), flush=True)
        del entries
    json.dump(out, open('/root/_cross.json', 'w'), indent=1)
