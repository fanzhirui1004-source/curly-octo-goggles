"""Route 7, CPU stage of the design-loop superset: one sparsity pattern valid for every design whose thickness corners
stay within (1 +- delta) of a centre design, so small topology changes need no new cuDSS plan.

The material {|phi| <= tau(x)} grows with every corner (trilinear weights are nonnegative), so
  cells: active(tau) is contained in active(tau_c (1 + delta)),
  faces: a ghost face (both cells active, not both interval-full) of any design in the band is an adjacent pair of
         active(tau_c (1 + delta)) that is not full-full at tau_c (1 - delta) (full cells only grow with tau).
Writes the superset body-lite (NODES.npy, CELL_INDICES.npy, dofs.npy, GP_FACES.npy of the superset) to
<out>/<case>_sup<delta>/. Box nodes stay per design (fast_prep3); the encoder rebuilds a cell whose box set changes.
Usage: fast_superset.py <workers> <out> <delta> <case> [<case> ...]
"""
import json, sys, time
from fractions import Fraction
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
R = Path('/root/autodl-tmp/CUTFEM_FRESH_GP_20260921/packets')


def main(workers, out, delta, case):
    import fast_topology
    from stage_cutfem_graded.contract import from_case
    from stage_cutfem_q2.space import OFFSETS
    ctx = json.loads((R / case / 'FRESH_CONTEXT.json').read_text())
    n = int(ctx['n'])
    t0 = time.perf_counter()
    def scaled(f):
        row = dict(ctx['case']); row['tau_corners'] = [str(Fraction(v) * f) for v in ctx['case']['tau_corners']]
        return from_case(row, n)
    d = Fraction(delta)
    c_plus, c_minus = scaled(1 + d), scaled(1 - d)
    topo, stats = fast_topology.compile_support_fast(c_plus, workers, global_proof=False)
    cells = np.asarray(topo['active_parent_cells'], dtype=np.int32)
    lookup = {tuple(map(int, x)): k for k, x in enumerate(cells)}
    ids = np.ravel_multi_index((2 * cells[:, None, :].astype(np.int64) + np.asarray(OFFSETS)[None]).transpose(2, 0, 1), (2 * n + 1,) * 3)
    nodes = np.unique(ids)
    dofs = (3 * np.searchsorted(nodes, ids)[:, :, None] + np.arange(3)).reshape(len(cells), 81)
    full_minus, _ = fast_topology.full_flags_fast(c_minus, cells, workers)
    faces = []
    for owner in range(len(cells)):
        for axis in range(3):
            other = cells[owner].copy(); other[axis] += 1
            nbr = lookup.get(tuple(map(int, other)))
            if nbr is not None and not (full_minus[owner] and full_minus[nbr]):
                faces.append((owner, nbr, axis))
    faces = np.asarray(faces, dtype=np.int32)
    dd = Path(out) / f'{case}_sup{delta.replace("/", "_")}'; dd.mkdir(parents=True, exist_ok=True)
    np.save(dd / 'NODES.npy', nodes); np.save(dd / 'CELL_INDICES.npy', cells); np.save(dd / 'dofs.npy', dofs)
    np.save(dd / 'GP_FACES.npy', faces)
    row = dict(case=case, delta=delta, cells=int(len(cells)), nodes=int(len(nodes)), faces=int(len(faces)),
               seconds=time.perf_counter() - t0, topology_stats=stats)
    (dd / 'SUPERSET.json').write_text(json.dumps(row, indent=2))
    print(json.dumps(row), flush=True)


if __name__ == '__main__':
    workers, out, delta = int(sys.argv[1]), sys.argv[2], sys.argv[3]
    for case in sys.argv[4:]:
        main(workers, out, delta, case)
