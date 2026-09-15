#!/usr/bin/env python3
"""Rebuild one production GP cell's pre-condensation system, for elimination-cost profiling.

The shipped packets carry only the condensed Schur complement S (12798 x 12798), so the coupling
graph that a nested-dissection tree would actually run on does not exist on disk.  This rebuilds
it from the packet's own geometry through the production chain at the packet's source commit:

    contract -> topology -> Q2 body (K_upper) -> unit ghost penalty -> K = K_body + gamma*K_ghost

and checks the rebuilt node set and trace set against the ones the packet ships in TRACE.npz, so
the rebuild is verified rather than trusted.

This runs on a host registered for profiling, not the host that produced the dataset.  It is a
cost measurement, not certified dataset evidence, and the authorization flags in the profiling
config are deliberately left false.
"""
import json, sys, time
from pathlib import Path
import numpy as np
from scipy import sparse

P = '/root/autodl-tmp/CUTFEM_INGEST_R38/source_independent_6624dc8_20260910'
sys.path.insert(0, P)
PKT = Path('/root/autodl-tmp/CUTFEM_INGEST_R38/dataset_independent_20260910/'
           'B1024_N32_INDEPENDENT_20260910_0328/payload/packet')
OUT = Path('/root/autodl-tmp/GP_TREE_PROFILE/runs/seat0328')


def main():
    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    sample = json.load(open(PKT/'SAMPLE.json'))
    geo = sample['geometry']; cut = geo['cut_plane']
    n = int(sample['n']); gamma = float(sample['gp']['gamma'])
    row = dict(case_id='B1024_0328_r0', normal=tuple(cut[:3]), offset=cut[3],
               tau_corners=tuple(geo['tau_corners']))
    print('n %d  gamma %g  normal %s  offset %s' % (n, gamma, cut[:3], cut[3]), flush=True)

    from stage_cutfem_graded.contract import from_case
    from stage_cutfem_graded.topology import compile_topology
    from stage_cutfem_multiconstraint.body import build as build_body
    from stage_cutfem_gp.assembly import build as build_ghost
    from stage_cutfem_runtime.config import CONFIG
    W = CONFIG['parallelism']['assembly_workers']

    contract = from_case(row, n)
    print('contract built (%.1f s)' % (time.time()-t0), flush=True)
    topology = compile_topology(contract, workers=W)
    print('topology compiled (%.1f s)' % (time.time()-t0), flush=True)
    body = build_body(contract, OUT/'body', workers=W, topology=topology)
    K_body = body['K_upper']
    print('body: %d dofs, %d nodes, %d cells, upper nnz %d (%.1f s)'
          % (K_body.shape[0], len(body['nodes']), len(body['cell_indices']),
             K_body.nnz, time.time()-t0), flush=True)

    ghost = build_ghost(body, OUT/'ghost', workers=W)
    K_gp = ghost['K_upper']
    print('ghost: upper nnz %d, faces %s (%.1f s)'
          % (K_gp.nnz, ghost['record'].get('face_count', '?'), time.time()-t0), flush=True)

    upper = (K_body + gamma*K_gp).tocsr(); upper.sort_indices()
    print('combined upper nnz %d (body %d, ghost adds %d new entries)'
          % (upper.nnz, K_body.nnz, upper.nnz - K_body.nnz), flush=True)

    from stage_cutfem_assembly.registry import box_ports, vector_maps
    faces, scalar = box_ports(body)
    trace = np.sort(vector_maps(scalar).ravel())
    print('trace dofs rebuilt: %d' % len(trace), flush=True)

    # verify against what the packet ships
    tr = np.load(PKT/'TRACE.npz')
    keys = list(tr.keys()); print('TRACE.npz keys:', keys, flush=True)
    checks = {}
    if 'background_nodes' in keys:
        bn = tr['background_nodes']
        mine = np.asarray(body['nodes'], dtype=np.int64)
        checks['background_nodes'] = dict(packet=int(bn.size), rebuilt=int(mine.size),
                                          identical=bool(bn.size == mine.size and np.array_equal(np.sort(bn.ravel()), np.sort(mine))))
    for k in ('boundary', 'box_boundary_original'):
        if k in keys:
            b = np.sort(tr[k].ravel())
            checks[k] = dict(packet=int(b.size), rebuilt=int(trace.size),
                             identical=bool(b.size == trace.size and np.array_equal(b, trace)))
    print('verification:', json.dumps(checks, indent=1), flush=True)

    sparse.save_npz(OUT/'K_upper_gp.npz', upper.tocsr())
    np.save(OUT/'trace.npy', trace)
    np.save(OUT/'cell_indices.npy', np.asarray(body['cell_indices']))
    np.save(OUT/'nodes.npy', np.asarray(body['nodes']))
    rec = dict(schema='GP_PRECONDENSATION_V1', seat=328, n=n, gamma=gamma,
               body_dofs=int(K_body.shape[0]), nodes=int(len(body['nodes'])),
               cells=int(len(body['cell_indices'])), body_upper_nnz=int(K_body.nnz),
               ghost_upper_nnz=int(K_gp.nnz), combined_upper_nnz=int(upper.nnz),
               new_entries_from_gp=int(upper.nnz - K_body.nnz),
               trace_dofs=int(len(trace)), interior_dofs=int(K_body.shape[0]-len(trace)),
               trace_fraction=float(len(trace))/K_body.shape[0],
               ghost_record={k: v for k, v in ghost['record'].items() if isinstance(v, (int, float, str))},
               verification=checks, seconds=time.time()-t0,
               provenance='rebuilt on a profiling host; not certified dataset evidence')
    (OUT/'BUILD.json').write_text(json.dumps(rec, indent=1))
    print('written', OUT/'BUILD.json', '(%.1f s)' % rec['seconds'], flush=True)


if __name__ == '__main__': main()
