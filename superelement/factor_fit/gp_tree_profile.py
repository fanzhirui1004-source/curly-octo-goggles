#!/usr/bin/env python3
"""Layered elimination cost on a real production GP teacher's coupling graph.

The earlier profile ran on small no-GP bodies whose trace was 53-80% of their coordinates.
This one runs on seat 0328 as the dataset actually built it: 231,192 dofs, 218,394 interior,
12,798 trace (5.5%), 7,478 active Q2 cells and 18,092 ghost-penalty faces, rebuilt from the
packet's own geometry at its source commit and verified against the node and trace sets the
packet ships.

Two element types carry the coupling, and the ghost faces are the reason this cannot be read
off cell adjacency alone:
  - each active cell contributes a dense 81-dof block over its 27 Q2 nodes
  - each ghost face contributes a dense 135-dof block over the 45 nodes of its two cells,
    coupling pairs that lie in no single element

A node of the bisection tree owns a cell set C.  Its elements are the cells in C plus the ghost
faces whose BOTH cells are in C; a face straddling the cut is not internal and its dofs stay on
the boundary.  A dof is private to C when every element touching it anywhere is internal to C.
Trace dofs are never eliminated.  Separator, front, flops and storage follow as before.
"""
import argparse, json, time
from pathlib import Path
import numpy as np
from scipy import sparse

R = Path('/root/autodl-tmp/GP_TREE_PROFILE/runs/seat0328')


def build_incidence():
    build = json.load(open(R/'BUILD.json'))
    n = int(build['n']); dim = int(build['body_dofs'])
    nodes = np.load(R/'nodes.npy'); cell_ijk = np.load(R/'cell_indices.npy')
    faces = np.load(R/'ghost/faces.npy')                       # (F, 3): owner, neighbour, axis
    fdofs_local = np.load(R/'ghost/dofs.npy')                  # (F, 135) into global_support
    support = np.load(R/'ghost/global_support.npy')
    fdofs = support[fdofs_local]                               # (F, 135) global dofs
    trace = np.load(R/'trace.npy')

    # cell -> its 27 Q2 nodes -> 81 dofs
    pos = {int(v): i for i, v in enumerate(nodes)}
    off = np.stack(np.meshgrid(*[np.arange(3)]*3, indexing='ij'), -1).reshape(-1, 3)   # 27 x 3
    lat = 2*cell_ijk[:, None, :] + off[None, :, :]                                     # (C,27,3)
    ids = np.ravel_multi_index(lat.reshape(-1, 3).T, (2*n+1,)*3).reshape(len(cell_ijk), 27)
    local = np.vectorize(pos.__getitem__)(ids)
    cdofs = (3*local[:, :, None] + np.arange(3)).reshape(len(cell_ijk), 81)

    nc, nf = len(cdofs), len(fdofs)
    rows = np.concatenate([np.repeat(np.arange(nc), 81), np.repeat(nc + np.arange(nf), 135)])
    cols = np.concatenate([cdofs.ravel(), fdofs.ravel()])
    inc = sparse.csr_matrix((np.ones(len(rows), np.int8), (rows, cols)), shape=(nc+nf, dim))
    inc.data[:] = 1
    return dict(inc=inc, nc=nc, nf=nf, dim=dim, trace=trace, faces=faces,
                centroid=cell_ijk.astype(float) + 0.5, build=build)


def profile(S, leaf):
    inc, nc, nf = S['inc'], S['nc'], S['nf']
    faces, trace, dim = S['faces'], S['trace'], S['dim']
    total = np.asarray(inc.sum(axis=0)).ravel().astype(np.int64)     # global element count per dof
    is_trace = np.zeros(dim, bool); is_trace[trace] = True
    fa, fb = faces[:, 0].astype(np.int64), faces[:, 1].astype(np.int64)
    levels = {}

    def rec(cells, depth):
        inside = np.zeros(nc, bool); inside[cells] = True
        internal_faces = np.flatnonzero(inside[fa] & inside[fb])
        rowsel = np.concatenate([cells, nc + internal_faces])
        sub = inc[rowsel]
        cnt_all = np.asarray(sub.sum(axis=0)).ravel().astype(np.int64)
        touched = np.flatnonzero(cnt_all > 0)
        priv = touched[(cnt_all[touched] == total[touched]) & (~is_trace[touched])]
        if len(cells) <= leaf:
            child = np.empty(0, np.int64)
        else:
            axis = int(np.argmax(S['centroid'][cells].max(0) - S['centroid'][cells].min(0)))
            order = cells[np.argsort(S['centroid'][cells, axis], kind='stable')]
            h = len(order)//2
            child = np.union1d(rec(np.sort(order[:h]), depth+1), rec(np.sort(order[h:]), depth+1))
        sep = np.setdiff1d(priv, child, assume_unique=False)
        bnd = np.setdiff1d(touched, priv, assume_unique=False)
        e, b = len(sep), len(bnd)
        levels.setdefault(depth, []).append(dict(cells=int(len(cells)), e=int(e), b=int(b),
            faces=int(len(internal_faces)), flops=e**3/3 + e*e*b + e*b*b, front=(e+b)**2*8))
        return priv

    rec(np.arange(nc, dtype=np.int64), 0)
    return levels


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--leaf', type=int, default=8)
    ap.add_argument('--out', default='/root/autodl-tmp/GP_TREE_PROFILE/TREE_PROFILE_GP_0328.json')
    a = ap.parse_args(); t0 = time.time()
    S = build_incidence()
    b = S['build']
    print('seat 328  dofs %d  interior %d  trace %d (%.2f%%)  cells %d  ghost faces %d  (%.0f s)'
          % (b['body_dofs'], b['interior_dofs'], b['trace_dofs'], 100*b['trace_fraction'],
             b['cells'], S['nf'], time.time()-t0), flush=True)
    levels = profile(S, a.leaf)
    tot = sum(x['flops'] for L in levels.values() for x in L)
    ni, tr = b['interior_dofs'], b['trace_dofs']
    mono = ni**3/3 + ni*ni*tr + ni*tr**2
    print('\ntree total %.3e flops   monolithic dense %.3e   ratio %.1fx' % (tot, mono, mono/tot), flush=True)
    print('peak front %.3f GiB' % (max(x['front'] for L in levels.values() for x in L)/2**30), flush=True)
    print('\ndepth  nodes      e range          b range         faces      flops      share')
    for d in sorted(levels, key=int):
        L = levels[d]; fl = sum(x['flops'] for x in L)
        es = [x['e'] for x in L]; bs = [x['b'] for x in L]
        print('%-6d %-10d [%6d..%6d]  [%6d..%6d]  %-9d %.3e  %5.1f%%'
              % (d, len(L), min(es), max(es), min(bs), max(bs),
                 sum(x['faces'] for x in L), fl, 100*fl/tot), flush=True)
    out = dict(schema='GP_TREE_PROFILE_V1', seat=328, leaf=a.leaf, build=b,
               total_flops=tot, monolithic_dense_flops=mono, ratio=mono/tot,
               peak_front_bytes=max(x['front'] for L in levels.values() for x in L),
               levels={str(k): v for k, v in levels.items()}, seconds=time.time()-t0,
               provenance='rebuilt on a profiling host; not certified dataset evidence')
    Path(a.out).write_text(json.dumps(out, indent=1))
    print('\nwritten', a.out, '(%.0f s)' % out['seconds'], flush=True)


if __name__ == '__main__': main()
