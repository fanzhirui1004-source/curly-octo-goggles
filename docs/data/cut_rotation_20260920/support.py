"""Route A size probe.  The pullback of the trace form to nodal displacements on the cut support
is invariant to the teacher's pivot basis (q'_c = V q_c gives C' = V C, S'_cc = V^-T S_cc V^-1, so
C'^T S'_cc C' = C^T S_cc C and S'_bc C' = S_bc C), hence exactly covariant under the cubic group.
The only cost is the size of the nodal support.  Measure it from the frozen trace caches alone.
"""
import json, os
import numpy as np

L = json.load(open('/root/autodl-tmp/CLAUDE_LABELS_20260917/V2_LABELS.json'))
desc = {r['seat']: r for r in json.load(open('/root/_seat_desc.json'))}
rows = []
for e in L:
    seat = int(e['seat'])
    d = desc.get(seat, {})
    if not d.get('n_kind1'):
        continue
    tc = e.get('trace_cache')
    if not tc or not os.path.exists(tc):
        continue
    z = np.load(tc, allow_pickle=True)
    kind = np.asarray(z['kind'])
    indptr = np.asarray(z['indptr'])
    indices = np.asarray(z['indices'])
    nb = int((kind == 0).sum())
    nc = int(len(kind) - nb)
    box_nodes = set(indices[indptr[0]:indptr[nb]].tolist())
    cut_idx = indices[indptr[nb]:indptr[len(kind)]]
    cut_nodes = set(cut_idx.tolist())
    nnz = np.diff(indptr[nb:len(kind) + 1])
    total_nodes = len(box_nodes | cut_nodes)
    r = dict(seat=seat, tau_mean=d.get('tau_mean'), b=(d.get('cut_plane') or [None, None])[1],
             q_teacher=3 * len(kind), nb=nb, nc=nc,
             n_box_nodes=len(box_nodes), n_cut_nodes=len(cut_nodes),
             cut_nodes_new=len(cut_nodes - box_nodes), n_nodes_routeA=total_nodes,
             q_routeA=3 * total_nodes,
             growth=round(3 * total_nodes / (3 * len(kind)), 4),
             kernel_dim=3 * (len(cut_nodes - box_nodes) + len(box_nodes & cut_nodes) - nc) if False else 3 * (total_nodes - nb - nc),
             nnz_mean=round(float(nnz.mean()), 2), nnz_max=int(nnz.max()),
             background_nodes=int(len(np.asarray(z['background_nodes']))))
    rows.append(r)
    print(json.dumps(r), flush=True)
json.dump(rows, open('/root/_routeA_support.json', 'w'))
g = np.array([r['growth'] for r in rows])
q = np.array([r['q_routeA'] for r in rows])
qt = np.array([r['q_teacher'] for r in rows])
print()
print('cut seats with a trace cache:', len(rows))
print('q_teacher  min %d med %d max %d' % (qt.min(), np.median(qt), qt.max()))
print('q_routeA   min %d med %d max %d' % (q.min(), np.median(q), q.max()))
print('growth     min %.3f med %.3f max %.3f' % (g.min(), np.median(g), g.max()))
print('dense storage growth (square): min %.2f med %.2f max %.2f' % ((g**2).min(), np.median(g**2), (g**2).max()))
