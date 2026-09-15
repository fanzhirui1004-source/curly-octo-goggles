#!/usr/bin/env python3
"""Where does an elimination tree's work actually sit?

The proposal is to have a network emit already-condensed sub-domain operators and assemble them up
a tree with exact Schur eliminations, instead of emitting the complete factor. The mathematics is
standard substructuring and we have already checked the depth-1 case numerically (S1 + S2 vs the
direct condensation, 8e-16 on four bodies). The open question is cost: a tree does not remove fill,
and if the top merges dominate then replacing leaves with a network buys little.

This profiles the tree symbolically - no numerics, no learning. Cells are bisected recursively on
the widest geometric axis. At a tree node T with children T1, T2 the eliminated set is

    sep(T) = private(T) \\ (private(T1) u private(T2))

where private(T) are the non-trace coordinates touched only by cells inside T. Its dense front is
sep(T) u boundary(T). Trace coordinates are never eliminated, so the root front necessarily
contains the whole trace - that part is irreducible for a complete-trace deliverable.

Flops to eliminate e columns against a boundary of b: e^3/3 + e^2 b + e b^2.
Front storage: (e+b)^2 doubles.
"""
import argparse, json, math
from pathlib import Path
import numpy as np
from scipy import sparse


def load(root):
    root = Path(root)
    dofs = np.load(root/'body/dofs.npy'); ids = np.load(root/'body/canonical_ids.npy')
    bank = np.load(root/'body/canonical_G.npy'); nodes = np.load(root/'body/NODES.npy')
    P = sparse.load_npz(root/'GEOMETRY/ORIGINAL_FROM_TRACE_FREE.npz')
    q = sparse.load_npz(root/'GEOMETRY/TRACE_FROM_ORIGINAL.npz').shape[0]
    nc = len(dofs); n = P.shape[1]
    r = np.concatenate([np.repeat(np.arange(81)+81*e, 81) for e in range(nc)])
    c = np.concatenate([np.tile(dofs[e], 81) for e in range(nc)])
    v = np.concatenate([np.abs(bank[ids[e]]).ravel() for e in range(nc)])
    G = sparse.csr_matrix((v, (r, c)), shape=(81*nc, P.shape[0])) @ abs(P)
    G = (G > 0).astype(np.int8).tocsr()
    cell_col = sparse.csr_matrix((np.ones(81*nc, np.int8), (np.repeat(np.arange(nc), 81), np.zeros(81*nc, int))), shape=(nc, 1))
    inc = sparse.csr_matrix((np.ones(nc*81, np.int8), (np.repeat(np.arange(nc), 81), np.arange(81*nc))), shape=(nc, 81*nc)) @ G
    inc = (inc > 0).tocsr()                                            # (cells x coordinates)
    xyz = np.array(np.unravel_index(nodes, (65,)*3)).T/64.
    cent = np.array([xyz[np.unique(dofs[e]//3)].mean(0) for e in range(nc)])
    return inc, q, n, nc, cent


def profile(inc, q, n, nc, cent, leaf):
    total = np.asarray(inc.sum(axis=0)).ravel()                        # cells touching each coordinate
    levels = {}
    def rec(cells, depth):
        sub = inc[cells]
        touched = np.flatnonzero(np.asarray(sub.sum(axis=0)).ravel() > 0)
        cnt = np.asarray(sub.sum(axis=0)).ravel()[touched]
        priv = touched[(touched >= q) & (cnt == total[touched])]        # interior to this subtree
        if len(cells) <= leaf:
            child_priv = np.empty(0, np.int64)
        else:
            axis = int(np.argmax(cent[cells].max(0) - cent[cells].min(0)))
            order = cells[np.argsort(cent[cells, axis])]; h = len(order)//2
            p1 = rec(np.sort(order[:h]), depth+1); p2 = rec(np.sort(order[h:]), depth+1)
            child_priv = np.union1d(p1, p2)
        sep = np.setdiff1d(priv, child_priv)
        bnd = np.setdiff1d(touched, priv)
        e, b = len(sep), len(bnd)
        levels.setdefault(depth, []).append(dict(cells=len(cells), e=e, b=b,
            flops=e**3/3 + e*e*b + e*b*b, front=(e+b)**2*8))
        return priv
    rec(np.arange(nc), 0)
    return levels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', nargs='+', required=True); ap.add_argument('--leaf', type=int, default=8)
    ap.add_argument('--out', required=True)
    a = ap.parse_args(); out = {}
    for path in a.inputs:
        name = Path(path).name
        inc, q, n, nc, cent = load(path)
        lv = profile(inc, q, n, nc, cent, a.leaf)
        tot_f = sum(r['flops'] for rs in lv.values() for r in rs)
        peak = max(r['front'] for rs in lv.values() for r in rs)
        root = lv[0][0]
        print('\n### %s   cells %d  coordinates %d  trace %d  interior %d  leaf %d cells' % (name, nc, n, q, n-q, a.leaf))
        print('%6s %6s %8s %9s %9s %12s %10s %8s' % ('depth', 'nodes', 'cells/nd', 'elim med', 'bnd med', 'flops', '% of all', 'max front'))
        for d in sorted(lv):
            rs = lv[d]; f = sum(r['flops'] for r in rs)
            print('%6d %6d %8.0f %9.0f %9.0f %12.4g %9.1f%% %7.2fG'
                  % (d, len(rs), np.median([r['cells'] for r in rs]), np.median([r['e'] for r in rs]),
                     np.median([r['b'] for r in rs]), f, 100*f/tot_f, max(r['front'] for r in rs)/2**30))
        print('  root: eliminates %d against a boundary of %d; its front is %d^2 = %.2f GiB'
              % (root['e'], root['b'], root['e']+root['b'], root['front']/2**30))
        print('  root share of flops %.1f%%;  leaves (deepest level) share %.1f%%'
              % (100*root['flops']/tot_f, 100*sum(r['flops'] for r in lv[max(lv)])/tot_f))
        print('  monolithic reference: eliminating all %d interior at once against q=%d costs %.4g flops, front %.2f GiB'
              % (n-q, q, (n-q)**3/3 + (n-q)**2*q + (n-q)*q*q, n*n*8/2**30))
        print('  tree total %.4g flops, peak front %.2f GiB' % (tot_f, peak/2**30))
        out[name] = dict(cells=nc, coordinates=n, trace=q, leaf=a.leaf, total_flops=tot_f, peak_front_bytes=peak,
                         root=root, levels={str(d): [dict(r) for r in lv[d]] for d in lv})
    Path(a.out).write_text(json.dumps(out, indent=1)); print('\nwritten', a.out)


if __name__ == '__main__': main()
