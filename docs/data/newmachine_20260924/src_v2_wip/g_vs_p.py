"""G (gyroid) zero-shot lattice vs its P twin: per load compliance and sensitivity errors of the learned lattice (A3),
exact PCG iterations, from lat_hetero JSONs. Usage: g_vs_p.py <P.json> <G.json> [out.json]"""
import sys, json
import numpy as np


def summ(path):
    d = json.load(open(path)); (name, L), = d['lattices'].items()
    ex = L['exact']; r = dict(file=path.split('/')[-1], lattice=name, exact_iterations=ex['iterations'], exact_compliance=ex['compliance'])
    for k, v in L.items():
        if isinstance(v, dict) and k != 'exact' and 'compliance_rel_err' in v:
            se = np.asarray(v['sens_rel_err'])                                  # cells x loads (as stored)
            r[k] = dict(iterations=v.get('iterations'), seconds=v.get('seconds'), true_residual=v.get('true_residual'),
                        compliance_rel_err=v['compliance_rel_err'], sens_rel_err_max_per_load=se.max(0).tolist() if se.ndim == 2 else se.tolist(),
                        compliance_max=float(np.max(v['compliance_rel_err'])), sens_max=float(se.max()),
                        compliance_consistent_max=float(np.max(v['compliance_rel_err'][:3])), sens_consistent_max=float(se[:, :3].max()) if se.ndim == 2 else None)
    return r


P, G = summ(sys.argv[1]), summ(sys.argv[2])
out = dict(P=P, G=G)
print(json.dumps(out, indent=1))
if len(sys.argv) > 3:
    json.dump(out, open(sys.argv[3], 'w'), indent=1)
