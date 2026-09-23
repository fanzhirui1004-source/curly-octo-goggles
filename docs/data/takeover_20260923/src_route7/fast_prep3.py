"""Route 7, CPU stage of the fast exact pipeline: geometry parameters -> body-lite files for the GPU box encoder.

  fast local support (fast_topology, vectorized filter + frozen per-cell fallback; no global connectedness proof, by
  the user's decision; the face adjacency of the active cells is checked instead)
  -> nodes (frozen node_ids) -> element DOF map (27 Q2 nodes, node-major xyz)
  -> ghost faces (frozen face rule; full-cell test vectorized with frozen fallback)
Writes NODES.npy, CELL_INDICES.npy, dofs.npy, GP_FACES.npy, PREP.json to <out>/<case>/, plus the fixed ghost face
templates GP_TEMPLATES_n<n>.npz once. No trace coordinates (P) are built: the box encoder condenses directly onto the
box nodes. Where frozen products exist they are compared exactly (COVER_G body, packet GP_FACES).
Usage: fast_prep3.py <workers> <out_dir> <case> [<case> ...]
"""
import json, sys, time
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).resolve().parent))
T = Path('/root/autodl-tmp/CLAUDE_TAKEOVER_20260923'); R = Path('/root/autodl-tmp/CUTFEM_FRESH_GP_20260921/packets')


def main(workers, out, case):
    import fast_topology
    from stage_cutfem_graded.contract import from_case
    from stage_cutfem_q2.space import OFFSETS
    ctx = json.loads((R / case / 'FRESH_CONTEXT.json').read_text())
    n = int(ctx['n'])
    row = dict(case=case, workers=workers)
    t0 = time.perf_counter()
    c = from_case(ctx['case'], n)
    topo, stats = fast_topology.compile_support_fast(c, workers, global_proof=False)
    row['topology_seconds'] = time.perf_counter() - t0; row['topology_stats'] = stats
    t = time.perf_counter()
    cells = np.asarray(topo['active_parent_cells'], dtype=np.int32)
    lookup = {tuple(map(int, x)): k for k, x in enumerate(cells)}
    ei, ej = [], []
    for k, x in enumerate(cells):
        for axis in range(3):
            y = x.copy(); y[axis] += 1
            nb = lookup.get(tuple(map(int, y)))
            if nb is not None:
                ei.append(k); ej.append(nb)
    G = sparse.coo_matrix((np.ones(len(ei)), (ei, ej)), shape=(len(cells), len(cells)))
    ncomp, _ = connected_components(G, directed=False)
    row['active_cell_components'] = int(ncomp)
    if ncomp != 1:
        raise ValueError(f'ACTIVE_CELLS_NOT_FACE_CONNECTED:{ncomp}')
    ids = np.ravel_multi_index((2 * cells[:, None, :].astype(np.int64) + np.asarray(OFFSETS)[None]).transpose(2, 0, 1), (2 * n + 1,) * 3)
    nodes = np.unique(ids)
    local = np.searchsorted(nodes, ids)
    dofs = (3 * local[:, :, None] + np.arange(3)).reshape(len(cells), 81)
    row['nodes_dofs_seconds'] = time.perf_counter() - t
    t = time.perf_counter()
    full, fstats = fast_topology.full_flags_fast(c, cells, workers)
    faces = []
    for owner in range(len(cells)):
        ijk = cells[owner]
        for axis in range(3):
            other = ijk.copy(); other[axis] += 1
            nbr = lookup.get(tuple(map(int, other)))
            if nbr is not None and not (full[owner] and full[nbr]):
                faces.append((owner, nbr, axis))
    faces = np.asarray(faces, dtype=np.int32)
    row['faces_seconds'] = time.perf_counter() - t; row['faces_stats'] = fstats
    row['total_seconds'] = time.perf_counter() - t0
    d = Path(out) / case; d.mkdir(parents=True, exist_ok=True)
    np.save(d / 'NODES.npy', nodes); np.save(d / 'CELL_INDICES.npy', cells); np.save(d / 'dofs.npy', dofs)
    np.save(d / 'GP_FACES.npy', faces)
    tpl = Path(out) / f'GP_TEMPLATES_n{n}.npz'
    if not tpl.exists():
        from stage_cutfem_gp.kernel import stencil, face_factor
        np.savez(tpl, canonical=np.stack([face_factor(a, 1 / n) for a in range(3)]),
                 offsets=np.stack([stencil(a)[0] for a in range(3)]))
    # exact comparisons with frozen products where they exist
    body = T / 'COVER_G' / 'runs' / (case + '_G') / 'body'
    if body.exists():
        row['same_nodes'] = bool(np.array_equal(nodes, np.load(body / 'NODES.npy')))
        row['same_cells'] = bool(np.array_equal(cells, np.load(body / 'CELL_INDICES.npy')))
        row['same_dofs'] = bool(np.array_equal(dofs, np.load(body / 'dofs.npy')))
    ref_faces = R / case / 'CONTEXT' / 'GP_FACES.npy'
    if ref_faces.exists():
        rf = np.load(ref_faces)
        row['same_faces'] = bool(rf.shape == faces.shape and np.array_equal(rf, faces))
    ref_cells = R / case / 'CONTEXT' / 'CELL_INDICES.npy'
    if ref_cells.exists():
        row['same_cells_packet'] = bool(np.array_equal(np.load(ref_cells), cells))
    ref_nodes = R / case / 'CONTEXT' / 'ACTIVE_NODE_IDS.npy'
    if ref_nodes.exists():
        row['same_nodes_packet'] = bool(np.array_equal(np.load(ref_nodes), nodes))
    row.update(cells=int(len(cells)), nodes=int(len(nodes)), faces=int(len(faces)))
    (d / 'PREP.json').write_text(json.dumps(row, indent=2))
    print(json.dumps(row), flush=True)


if __name__ == '__main__':   # the frozen pool uses spawn, which re-imports this file in every worker
    workers, out = int(sys.argv[1]), sys.argv[2]
    for case in sys.argv[3:]:
        main(workers, out, case)
