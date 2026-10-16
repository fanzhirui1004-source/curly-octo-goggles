"""Where does the lattice-gate sensitivity error come from? Continuous-neighbour gate (lat_full --nb-mode explicit, set
'test': test cell learned, neighbour exact), per configuration, the 8-corner sensitivity error of the test cell split into
  full       learned lattice solution X_hat, learned field       (= the gate)
  field_only exact lattice solution U, learned field at the exact port data q  (interior reconstruction error alone)
  sol_only   learned X_hat, exact field at the learned port data q_hat         (lattice-solution error alone)
plus the port-data error ||q_hat - q||_S / ||q||_S (S = exact Schur complement of the test cell) and eps = q^T S_hat q /
q^T S q - 1 at the exact q. Usage: gate_decomp.py <out.json> <ckpt> <case>[,<case>...] [--configs x,y]"""
import json, sys, time, argparse
from pathlib import Path
import numpy as np
import torch
import models as MD                                                    # noqa: F401  first: applies OPL_CONV_FP32
import trainlib as TL
import diag_cert as DC
import lat_full as LF

dev, dt = TL.dev, TL.dt


def sens_err(ref, res):
    s0, s1 = ref['sens'][0], res['sens'][0]                                         # test cell: 8 x loads
    return (np.linalg.norm(s1 - s0, axis=0) / np.linalg.norm(s0, axis=0)).tolist()


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('ckpt'); ap.add_argument('cases'); ap.add_argument('--configs', default='x,y')
    ap.add_argument('--maxit', type=int, default=400)
    a = ap.parse_args(argv)
    import os
    import lattice3 as LT
    import ops as OP
    os.environ['LAT_LOADS'] = 'consistent'
    os.environ['LAT_SKIP_NBR_SENS'] = '1'
    ck = DC.load_ckpt(a.ckpt)
    body = ck['cfg']['body']
    rec = dict(ckpt=a.ckpt, results=[])
    for case in a.cases.split(','):
        Ct, _ = LT.prepared(case, body)
        opt, _ = LF.fast_op(ck, case, body, ck['cfg']['data'], Ct)
        for conf in a.configs.split(','):
            t0 = time.perf_counter()
            lat = LT.build(case, f'{case}_nbm{conf}', conf, body, log=lambda s_: None)
            ex = [OP.ExactOp(cd['cell'], cd['T']) for cd in lat.cells]
            ref = lat.reference()
            res = lat.evaluate([opt, ex[1]], maxit=a.maxit)
            f_only = lat._measure([opt, ex[1]], ref['U'])
            s_only = lat._measure(ex, res['U'])
            q, qh = lat.gather(ref['U'], 0), lat.gather(res['U'], 0)
            with torch.no_grad():
                Sq = ex[0].apply(q); dq = qh - q
                qerr = torch.sqrt((dq * ex[0].apply(dq)).sum(0) / (q * Sq).sum(0)).cpu().numpy()
                eps = ((q * opt.apply(q)).sum(0) / (q * Sq).sum(0) - 1).cpu().numpy()
            r = dict(case=case, config=conf, loads=lat.labels, gate=lat.gate.tolist(),
                     compliance_rel_err=np.abs(res['compliance'] / ref['compliance'] - 1).tolist(),
                     sens_full=sens_err(ref, res), sens_field_only=sens_err(ref, f_only), sens_sol_only=sens_err(ref, s_only),
                     q_err_S=qerr.tolist(), eps=eps.tolist(), pcg=res.get('pcg_iterations'), seconds=time.perf_counter() - t0)
            g = np.asarray(lat.gate, bool)
            print(json.dumps(dict(case=case, config=conf, **{k: round(float(np.max(np.asarray(r[k])[g])), 4) for k in
                                                               ('compliance_rel_err', 'sens_full', 'sens_field_only', 'sens_sol_only',
                                                                'q_err_S', 'eps')})), flush=True)
            rec['results'].append(r)
            Path(a.out).write_text(json.dumps(rec, indent=1))
            del lat, ex; DC.free_mem()
        del opt; LT._CACHE.clear(); DC.free_mem()


if __name__ == '__main__':
    main(sys.argv[1:])
