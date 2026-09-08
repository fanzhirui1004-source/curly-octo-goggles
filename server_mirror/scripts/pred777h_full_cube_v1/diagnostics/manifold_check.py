#!/usr/bin/env python3
"""Is the tet mesh a valid 3-manifold-with-boundary triangulation?  Checks the two things a CGAL Triangulation_3
cannot represent: a boundary vertex whose incident boundary triangles do not form ONE fan (a pinched vertex), and an
interior vertex whose link is not a sphere; plus the Euler characteristic of the boundary surface per component."""
from __future__ import annotations
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOTD = Path("/root/autodl-tmp/_claude_diag/tonight")


def read_medit(path):
    V = []; T = []
    with open(path) as fh:
        it = iter(fh)
        for line in it:
            s = line.strip()
            if s == "Vertices":
                n = int(next(it)); V = np.asarray([[float(x) for x in next(it).split()[:3]] for _ in range(n)])
            elif s == "Tetrahedra":
                n = int(next(it)); T = np.asarray([[int(x) for x in next(it).split()[:4]] for _ in range(n)]) - 1
    return V, T


for tag in sys.argv[1:]:
    V, T = read_medit(ROOTD / tag / "mesh.mesh")
    faces = np.sort(np.vstack([T[:, [0, 1, 2]], T[:, [0, 1, 3]], T[:, [0, 2, 3]], T[:, [1, 2, 3]]]), axis=1)
    u, cnt = np.unique(faces, axis=0, return_counts=True)
    B = u[cnt == 1]                                    # boundary triangles
    # boundary edge manifoldness
    E = np.sort(np.vstack([B[:, [0, 1]], B[:, [1, 2]], B[:, [0, 2]]]), axis=1)
    ue, ce = np.unique(E, axis=0, return_counts=True)
    bad_edges = int((ce != 2).sum())
    # boundary vertex fans: incident boundary triangles must be connected through shared edges into ONE cycle
    inc = defaultdict(list)
    for i, f in enumerate(B):
        for v in f: inc[int(v)].append(i)
    pinched = 0; examples = []
    for v, tris in inc.items():
        # graph on tris: adjacent if they share an edge containing v
        edge_to_tris = defaultdict(list)
        for t in tris:
            f = B[t]; others = [int(x) for x in f if x != v]
            for o in others: edge_to_tris[o].append(t)
        adj = defaultdict(set)
        for o, ts in edge_to_tris.items():
            for a in ts:
                for b in ts:
                    if a != b: adj[a].add(b)
        seen = set(); stack = [tris[0]]; seen.add(tris[0])
        while stack:
            a = stack.pop()
            for b in adj[a]:
                if b not in seen: seen.add(b); stack.append(b)
        if len(seen) != len(tris):
            pinched += 1
            if len(examples) < 5: examples.append((v, V[v].round(5).tolist(), len(tris), len(seen)))
    nb = len(np.unique(B)); chi = nb - len(ue) + len(B)
    # interior vertices: link must be a sphere: for each vertex, V-E+F of the link == 2 <=> (#tets - #facets + #edges around v) ... use: link is sphere iff  ntets - nfacets_inc + nedges_inc = 2? Simpler: check via tets around v: link vertices L, link faces = tets, link edges = facets containing v.
    print(f"=== {tag} ===  vertices {len(V)}, tets {len(T)}, boundary tris {len(B)}, boundary vertices {nb}")
    print(f"  boundary edges with != 2 triangles: {bad_edges}")
    print(f"  pinched boundary vertices (incident triangles not one fan): {pinched}   examples {examples}")
    print(f"  boundary surface Euler characteristic V-E+F = {chi}  (2 per sphere component; 0 per torus)")
    # interior vertex links
    tet_inc = defaultdict(list)
    for i, t in enumerate(T):
        for v in t: tet_inc[int(v)].append(i)
    bset = set(np.unique(B).tolist()); bad_links = 0
    for v, tets in tet_inc.items():
        if v in bset: continue
        Lf = len(tets)                                   # link faces
        Le = set(); Lv = set()
        for t in tets:
            others = tuple(sorted(int(x) for x in T[t] if x != v)); Lv.update(others)
            for a in range(3):
                for b in range(a + 1, 3): Le.add((others[a], others[b]))
        if len(Lv) - len(Le) + Lf != 2: bad_links += 1
    print(f"  interior vertices whose link is not a sphere: {bad_links}")
