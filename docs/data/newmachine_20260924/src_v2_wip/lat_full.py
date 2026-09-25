"""A7: the lattice gate with BOTH cells learned (each through fastnet with its own checkpoint) next to the one-learned-cell
gate, on the same lattice and exact reference (lattice3 / evalnet conventions: configurations x and y, traction-consistent
loads, gate = compliance and 8-corner sensitivities of every cell <= 3%).
Operator sets per configuration: 'test' (test learned, neighbour exact: the current gate, = evalnet FASTNET=1),
'nbr' (test exact, neighbour learned), 'both' (both learned: deployment).
Per load and cell c at the exact lattice solution U (q_c = the cell's port data):
  eps_c = q_c^T S_hat_c q_c / q_c^T S_c q_c - 1,  w_c = q_c^T S_c q_c / C,  C = F^T U
  0 <= (C - C_hat) / C <= sum over learned c of w_c eps_c        (C_hat = max_V 2 F^T V - V^T K_hat V >= C - sum_c q_c^T
                                                                   (S_hat_c - S_c) q_c at V = U; K_hat >= K gives C_hat <= C)
so the compliance errors of the learned cells add up with the same sign (no cancellation between cells), which the
'both' vs 'test' + 'nbr' comparison measures (additivity = err_both / (err_test + err_nbr)).
Pairs (step-1 gate-passing runs, PROGRESS / MORNING_REPORT_20260924_CN.md): r2 = (mgno2_r2_d2 on 0020 r2, mgno_full_c on
0020 FULL), r1 = (mgno_r1_e on 0020 r1, mgno_full_c on 0020 FULL).
Usage: lat_full.py <out.json> [--pairs r2,r1] [--configs x,y] [--sets test,nbr,both] [--s1 DIR] [--r2-run ...] (--help)
       --custom <test ckpt>:<test case>:<nbr ckpt>:<nbr case>[,...] adds pairs (e.g. a step-2 checkpoint on a held-out cut
       cell and its family FULL parent; the neighbour case defaults to the family FULL parent when left empty)
Recommended env as evalnet runs: LAT_CPU=1 (dense lattice factor on the host).
Convolution precision (INVARIANTS-2): OPL_CONV_FP32=1 -> true fp32 convolutions (models reads it at import, imported first
here); the mode in force is recorded in the output JSON ('conv').
"""
import json, sys, os, time, argparse
from pathlib import Path
import numpy as np
import torch
import models as MD                                                    # noqa: F401  first: applies OPL_CONV_FP32
import trainlib as TL
import diag_cert as DC

dev, dt = TL.dev, TL.dt


@torch.no_grad()
def cell_metrics(lat, ref, ops, learned):
    """eps (cells x loads) at the exact lattice port data (0 for exact cells), w (cells x loads) and the bound
    sum_c w_c eps_c (loads). ops: one operator per cell (field / apply); learned: bool per cell."""
    C = np.asarray(ref['compliance'], float)
    E = np.asarray(ref['energy'], float)                                            # cells x loads: q_c^T S_c q_c
    eps = np.zeros_like(E)
    for i, (op, l_) in enumerate(zip(ops, learned)):
        if l_:
            q = lat.gather(ref['U'], i)
            eps[i] = ((q * op.apply(q)).sum(0).cpu().numpy()) / E[i] - 1
    w = E / C[None, :]
    return eps, w, (w * eps).sum(0)


def summarize(cmp_, eps, w, bound, gate):
    ce = np.asarray(cmp_['compliance_rel_err']); sv = np.asarray(cmp_['sens_vec_rel_err'])
    g = np.asarray(gate, bool)
    return dict(gate_compliance_max=cmp_['gate_compliance_max'], gate_sens_max=cmp_['gate_sens_max'], gate_pass=cmp_['gate_pass'],
                gate_sens_max_per_cell=np.nanmax(sv[:, g], 1).tolist(), compliance_rel_err=ce.tolist(),
                sens_vec_rel_err=sv.tolist(), eps=eps.tolist(), w=w.tolist(), bound=bound.tolist(),
                bound_holds=bool((ce <= bound * (1 + 1e-6) + 1e-9).all()),
                eps_gate_max_per_cell=eps[:, g].max(1).tolist(), cut_compliance_max=cmp_.get('cut_compliance_max'),
                cut_sens_max=cmp_.get('cut_sens_max'), pcg_iterations=cmp_.get('pcg_iterations'))


def run_lattice(lat, ops_learned, ops_exact, sets=('test', 'nbr', 'both'), maxit=400, log=print):
    """Reference once, then every operator set on the same lattice. ops_*: [test op, neighbour op]."""
    ref = lat.reference() if getattr(lat, 'ref', None) is None else lat.ref
    masks = dict(test=(True, False), nbr=(False, True), both=(True, True))
    out = {}
    for s in sets:
        m = masks[s]
        ops = [ops_learned[i] if m[i] else ops_exact[i] for i in range(2)]
        t = time.perf_counter()
        res = lat.evaluate(ops, maxit=maxit)
        cmp_ = lat.compare(res)
        eps, w, bound = cell_metrics(lat, ref, ops, m)
        out[s] = summarize(cmp_, eps, w, bound, lat.gate)
        out[s]['seconds'] = time.perf_counter() - t
        log(dict(set=s, gate_compliance_max=cmp_['gate_compliance_max'], gate_sens_max=cmp_['gate_sens_max'],
                 gate_pass=cmp_['gate_pass'], bound_holds=out[s]['bound_holds'], pcg=cmp_.get('pcg_iterations')))
    if all(k in out for k in ('test', 'nbr', 'both')):
        a = np.asarray(out['both']['compliance_rel_err']); b = np.asarray(out['test']['compliance_rel_err']) + np.asarray(out['nbr']['compliance_rel_err'])
        out['additivity'] = (a / np.maximum(b, 1e-300)).tolist()
        out['both_over_test_gate_compliance'] = out['both']['gate_compliance_max'] / max(out['test']['gate_compliance_max'], 1e-300)
    out['loads'], out['gate'] = lat.labels, lat.gate.tolist()
    out['energy_share'] = (np.asarray(ref['energy']) / np.asarray(ref['compliance'])[None, :]).tolist()
    return out


def fast_op(ck, case, body, data, C):
    import evalnet as EN
    import fastnet as FN
    geo = TL.Geo(case, body, data, neumann=False, log=lambda s_: None, cell=C, load_banks=False)
    model, miss = DC.net_for(ck, geo)
    return EN.FastOp(FN.FastNet(model, geo)), miss


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('out')
    ap.add_argument('--pairs', default='r2,r1'); ap.add_argument('--configs', default='x,y'); ap.add_argument('--sets', default='test,nbr,both')
    ap.add_argument('--s1', default=DC.S1)
    ap.add_argument('--r2-run', default=DC.RUNS['r2']); ap.add_argument('--r1-run', default=DC.RUNS['r1'])
    ap.add_argument('--full-run', default=DC.RUNS['full'])
    ap.add_argument('--r2-case', default=DC.R2); ap.add_argument('--r1-case', default=DC.R1); ap.add_argument('--full-case', default=DC.FULL)
    ap.add_argument('--custom', default='', help='tckpt:tcase:nckpt:ncase,...')
    ap.add_argument('--maxit', type=int, default=400)
    args = ap.parse_args(argv)
    import lattice3 as LT
    import ops as OP
    os.environ['LAT_LOADS'] = 'consistent'
    s1 = Path(args.s1)
    spec = {k: (str(s1 / r / 'best.pt'), c, str(s1 / args.full_run / 'best.pt'), args.full_case)
            for k, r, c in (('r2', args.r2_run, args.r2_case), ('r1', args.r1_run, args.r1_case))}
    pairs = [p for p in args.pairs.split(',') if p]
    for i, c in enumerate(x for x in args.custom.split(',') if x):
        tck, tcase, nck, ncase = c.split(':')
        spec[f'custom{i}'] = (tck, tcase, nck, ncase or DC.family_full(tcase)); pairs.append(f'custom{i}')
    rec = dict(args=vars(args), results=[], conv=TL.conv_precision())
    log = lambda d: print(json.dumps(d), flush=True)
    for p in pairs:
        tckp, tcase, nckp, ncase = spec[p]
        trun, nrun = Path(tckp).parent.name, Path(nckp).parent.name
        ckt, ckn = DC.load_ckpt(tckp), DC.load_ckpt(nckp)
        Ct, _ = LT.prepared(tcase, ckt['cfg']['body']); Cn, _ = LT.prepared(ncase, ckn['cfg']['body'])
        opt, mt = fast_op(ckt, tcase, ckt['cfg']['body'], ckt['cfg']['data'], Ct)
        opn, mn = fast_op(ckn, ncase, ckn['cfg']['body'], ckn['cfg']['data'], Cn)
        for conf in args.configs.split(','):
            t = time.perf_counter()
            lat = LT.build(tcase, ncase, conf, ckt['cfg']['body'])
            ex = [OP.ExactOp(cd['cell'], cd['T']) for cd in lat.cells]
            r = run_lattice(lat, [opt, opn], ex, tuple(args.sets.split(',')), args.maxit,
                            log=lambda d: log(dict(pair=p, config=conf, **d)))
            r.update(pair=p, config=conf, test_cell=dict(case=tcase, run=trun, ckpt=tckp, missing_keys=mt,
                                                         ckpt_conv_fp32=ckt['cfg'].get('conv_fp32')),
                     nbr_cell=dict(case=ncase, run=nrun, ckpt=nckp, missing_keys=mn, ckpt_conv_fp32=ckn['cfg'].get('conv_fp32')),
                     seconds=time.perf_counter() - t, conv_tf32=TL.conv_precision()['conv_tf32'])
            rec['results'].append(r)
            Path(args.out).write_text(json.dumps(rec, indent=1))
            del lat; DC.free_mem()
        del opt, opn; DC.free_mem()
    return rec


if __name__ == '__main__':
    main()
