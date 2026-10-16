"""v3_plate.py <census.json> <opt_design arguments...>: one NICE analysis of design iteration 0 of an opt_design layout
(opt_design.analyse_fast: the route of the ST20 scale runs), with the v3_census instrumentation. Bodies of iteration 0
are generated (fast_prep4, --workers) into <root>/body when missing; no MMA step, no state file. Writes
<root>/analysis0.json (compliance, PCG iterations, true residual, phase times, peaks) and <root>/analysis0.npz
(per-cell sensitivities S, volumes, dV, lattice order)."""
import os, sys, json, time, resource
from pathlib import Path
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.environ.get('SRC', '/root/autodl-tmp/OPL/src_v3')
sys.path.insert(0, SRC); sys.path.insert(0, HERE)
OUT = sys.argv[1]
sys.argv = [os.path.join(SRC, 'opt_design.py')] + sys.argv[2:]
import v3_census as VC                                                  # instrumentation patches (imports lat_scale)
VC.OUT = OUT
import opt_design as OD                                                 # parses sys.argv (opt_design arguments)
import numpy as np
import torch
import lat_multi as LM

_ml_init = LM.MultiLattice.__init__
_done = [False]


def ml_init(self, *a, **k):
    if not _done[0]:
        _done[0] = True
        VC.cell_census()                                                # after the front end of analyse_fast
    return _ml_init(self, *a, **k)


LM.MultiLattice.__init__ = ml_init


def main():
    os.chdir(SRC)
    A = OD.A
    L = json.loads(Path(A.layout).read_text())
    base = [c['case'] for c in L['cells']]
    positions = [tuple(int(v) for v in c['position']) for c in L['cells']]
    taus0 = [[float(x) for x in c['tau_corners']] for c in L['cells']]
    vid, nv, tv0, dmax, vkeys = OD.RC.vertex_map(positions, taus0)
    corners = [tv0[vid[m]] for m in range(len(base))]
    cases = OD.write_packets(base, corners, 0)
    t = time.perf_counter()
    nb = OD.make_bodies(cases)
    t_body = time.perf_counter() - t
    name, h = OD.LH.holder_for(A.model)
    VC.point('start_analysis')
    torch.cuda.reset_peak_memory_stats()
    t = time.perf_counter()
    res = OD.analyse_fast(cases, positions, h)
    wall = time.perf_counter() - t
    VC.point('end_analysis')
    out = dict(cells=len(cases), C=res['C'], pcg=res['pcg'], true_residual=res['true_residual'], Ut_rho_rel=res['Ut_rho_rel'],
               times=dict(res['times'], bodies_s=t_body, analysis_wall_s=wall), bodies_new=nb, streamed=res.get('streamed'),
               gpu_peak_gb=torch.cuda.max_memory_allocated() / 2 ** 30,
               host_peak_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20,
               env={k: v for k, v in os.environ.items() if k.startswith('OPL_') or k.startswith('PYTORCH')})
    root = Path(A.root)
    (root / 'analysis0.json').write_text(json.dumps(out, indent=1, default=float))
    np.savez(root / 'analysis0.npz', S=res['S'], vol=res['vol'], dvol=res['dvol'], order=np.asarray(res['order']))
    VC.REC['analysis'] = out
    VC.flush()
    print(json.dumps(out, default=float), flush=True)


if __name__ == '__main__':
    main()
