import sys, numpy as np, scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
FIX = '/tmp/claude-0/-home-user-curly-octo-goggles/76f53c89-e2a9-52bb-8817-9460fd780b76/scratchpad/fixdl/FIX/'
for case in sys.argv[1:]:
    z = np.load(FIX + case + '/NETDATA.npz'); en = z['elem_nodes'].astype(np.int64); vf = z['moments'][:, 0] * 32 ** 3
    N = len(z['grid']); port = z['is_port']; box = z['is_box']
    print(case, 'elements', len(en), 'vf quantiles', np.quantile(vf, [0, .1, .5, .9]).round(4))
    # elements connected if they share a FACE (9 nodes) -- material continuity requires the shared face to carry material;
    # proxy: both elements vf > thr. Node-sharing (edge/vertex) contact is not load-bearing continuity.
    cells = z['elem_cells'].astype(np.int64); key = {tuple(c): i for i, c in enumerate(cells)}
    pairs = []
    for i, c in enumerate(cells):
        for d in range(3):
            o = c.copy(); o[d] += 1; j = key.get(tuple(o))
            if j is not None: pairs.append((i, j))
    pairs = np.array(pairs)
    for thr in [0.0, 1e-4, 1e-3, 1e-2, 0.05]:
        m = vf > thr
        pp = pairs[m[pairs[:, 0]] & m[pairs[:, 1]]]
        A = sp.coo_matrix((np.ones(len(pp)), (pp[:, 0], pp[:, 1])), shape=(len(en), len(en)))
        nc, lab = connected_components(A, directed=False)
        labs = lab[m]; u, cnt = np.unique(labs, return_counts=True)
        # does the component touch a box port node?
        tb = [bool(box[en[(lab == l) & m]].any()) for l in u]
        vol = [float(vf[(lab == l) & m].sum() / 32 ** 3) for l in u]
        order = np.argsort(-np.array(vol))
        print('  thr %.0e: %d material components; (elements, volume, touches box face):' % (thr, len(u)),
              [(int(cnt[k]), round(vol[k], 5), tb[k]) for k in order[:8]])
