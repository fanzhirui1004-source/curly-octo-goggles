"""Can the frozen trace cache be recompiled from a packet alone?

Self-test: recompile seat 0253's TRACE_CACHE from its packet and compare every
array against the frozen file the 38 existing labels are bound to.  If they
match, the same path turns the other 882 already-condensed packets into labels.
"""
import argparse, json, time, sys
from pathlib import Path
import numpy as np


def case_row(sample, case_id):
    g = sample['geometry']
    plane = g['cut_plane']
    return dict(case_id=case_id, normal=tuple(plane[:3]), offset=plane[3],
                tau_corners=tuple(g['tau_corners']))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--packet', type=Path, required=True)
    ap.add_argument('--case-id', required=True)
    ap.add_argument('--expect', type=Path, default=None)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    from stage_cutfem_graded.contract import from_case
    from stage_cutfem_graded.topology import compile_topology
    from stage_cutfem_multiconstraint.complete_trace import compile_trace
    from stage_cutfem_q2.space import node_ids
    sys.path.insert(0, str(Path(__file__).parent))
    from full_factor_geometry import trace_cache

    sample = json.loads((args.packet / 'SAMPLE.json').read_text())
    n = int(sample['n'])
    report = dict(packet=str(args.packet), n=n, case_id=args.case_id,
                  full_trace_dimension=sample.get('full_trace_dimension'))

    row = case_row(sample, args.case_id)
    t = time.perf_counter(); c = from_case(row, n); report['from_case_seconds'] = time.perf_counter() - t

    t = time.perf_counter()
    topology = compile_topology(c, workers=args.workers)
    report['compile_topology_seconds'] = time.perf_counter() - t
    report['topology_status'] = topology['status']
    report['component_count'] = topology['component_count']
    if topology['status'] != 'TRUE_TOPOLOGY_CERTIFIED' or topology['component_count'] != 1:
        report['status'] = 'TOPOLOGY_FAILURE'
        (args.output / 'REPLAY.json').write_text(json.dumps(report, indent=1))
        print(json.dumps(report, indent=1)); return

    cells = np.asarray(topology['active_parent_cells'], dtype=np.int32)
    t = time.perf_counter()
    nodes = np.unique(np.stack([node_ids(n, i) for i in cells]))
    report['node_ids_seconds'] = time.perf_counter() - t
    report['active_cells'] = int(len(cells)); report['body_dofs'] = int(3 * len(nodes))

    body = dict(contract=c, nodes=nodes, topology=topology)
    t = time.perf_counter()
    compiled = compile_trace(body, coordinate_convention='max_pivot_geometric_residuals_v2')
    report['compile_trace_seconds'] = time.perf_counter() - t

    t = time.perf_counter()
    cache = trace_cache(body, compiled)
    report['trace_cache_seconds'] = time.perf_counter() - t
    np.savez(args.output / 'TRACE_CACHE.npz', **cache)
    report['arrays'] = {k: [list(np.shape(v)), str(np.asarray(v).dtype)] for k, v in cache.items()}

    if args.expect is not None:
        with np.load(args.expect, allow_pickle=False) as frozen:
            keys_frozen = sorted(frozen.files); keys_new = sorted(cache)
            report['key_sets_equal'] = keys_frozen == keys_new
            report['frozen_only'] = [k for k in keys_frozen if k not in keys_new]
            report['new_only'] = [k for k in keys_new if k not in keys_frozen]
            equal = {}
            for k in keys_frozen:
                if k not in cache:
                    equal[k] = 'MISSING'; continue
                a = np.asarray(frozen[k]); b = np.asarray(cache[k])
                if a.shape != b.shape or a.dtype != b.dtype:
                    equal[k] = f'SHAPE_OR_DTYPE {a.shape}{a.dtype} vs {b.shape}{b.dtype}'
                elif a.dtype.kind in 'fc':
                    equal[k] = bool(np.array_equal(a, b)) or f'max_abs_diff={float(np.abs(a-b).max())}'
                else:
                    equal[k] = bool(np.array_equal(a, b))
            report['all_arrays_equal'] = equal
            report['bit_exact'] = all(v is True for v in equal.values()) and report['key_sets_equal']
    report['status'] = 'REPLAY_COMPLETE'
    (args.output / 'REPLAY.json').write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
