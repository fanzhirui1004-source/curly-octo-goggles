"""Per-cell correction setup time on the GPU: smoothing interval (40 power steps) and Q1(17) coarse setup (probing,
Jacobi scaling, Cholesky), synchronised, after the cell is assembled. Usage: time_setup.py <out.json> <case>[,...] [--body DIR]"""
import sys, json, time, argparse
from pathlib import Path
import torch
import trainlib as TL
import teacher as TE


def main(argv):
    ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('cases'); ap.add_argument('--body', default='/root/autodl-tmp/OPL/S0')
    a = ap.parse_args(argv)
    rec = {}
    for case in a.cases.split(','):
        C = TE.Cell(case, a.body, log=lambda s_: None); C.assemble()
        r = {}
        for rep in range(2):                                              # second repetition: warm kernels
            C.__dict__.pop('_tail_bounds', None)
            for k in [k for k in list(C.__dict__) if k.startswith('_c')]:
                delattr(C, k)
            torch.cuda.synchronize(); t = time.perf_counter(); TL.tail_bounds(C, 30.0); torch.cuda.synchronize(); t1 = time.perf_counter()
            TL.coarse_setup(C, 'Q1_17'); torch.cuda.synchronize(); t2 = time.perf_counter()
            r = dict(interval_s=t1 - t, coarse_s=t2 - t1, total_s=t2 - t, coarse_cols=int(C._cV.shape[1]), shift=float(C._c_shift))
        rec[case] = r
        print(json.dumps(dict(case=case, **r)), flush=True)
        C._free(); del C; torch.cuda.empty_cache()
    Path(a.out).write_text(json.dumps(rec, indent=1))


if __name__ == '__main__':
    main(sys.argv[1:])
