"""P1 cost baseline: the conventional route on the host CPU -- CutFEM assembly followed by static condensation with a sparse
direct solver (MKL PARDISO), per cell:
  topology    fast_prep4 seconds and workers, from the body's PREP.json (shared by every route)
  setup       teacher.Cell construction on the host (body arrays, ghost-penalty matrix, element patterns)
  assembly    element moments (cut-cell integration) + stiffness K, host
  factor      PARDISO LU of the scaled interior block K_II (analysis + numerical factorisation), peak / permanent memory
              reported by PARDISO (iparm 15-17, kB)
  query       S q = (K E q)_P for B = 1, 16, 64 random retained vectors (3 timed repetitions after one warm-up)
  explicit S  the dense condensed matrix S = K_PP - K_PI K_II^-1 K_IP column block by column block (256 columns per solve);
              timed until --s-budget seconds, then extrapolated linearly in the number of columns (flag 'extrapolated');
              memory 8 np^2 bytes
Threads: MKL_NUM_THREADS / OMP_NUM_THREADS as set (recorded).
Usage: bench_cpu.py <out.json> <case>[,...] [--body /root/autodl-tmp/OPL/S0] [--s-budget 900]
Needs OPL_DEV=cpu (diag_sens environment)."""
import diag_sens as DS                                                   # first: CPU environment
import os, sys, json, time, argparse, gc
from pathlib import Path
import numpy as np
import torch
import trainlib as TL
import teacher as TE
import box_encode as BX

dt = TL.dt
BX.dev = TL.dev


def factor(C):
    import scipy.sparse as sp, pypardiso
    C._free()
    pm = torch.zeros(C.nb, dtype=torch.bool); pm[C.P] = True
    new = torch.full((C.nb,), -1, dtype=torch.long); new[C.I] = torch.arange(C.ni)
    ru, cu = C.ru.long(), C.cu.long()
    sel = torch.nonzero(~pm[ru] & ~pm[cu]).squeeze(1)
    rA, cA, vA = new[ru[sel]], new[cu[sel]], C.vals[sel]
    sA = torch.zeros(C.ni, dtype=dt); sA[rA[rA == cA]] = 1 / torch.sqrt(vA[rA == cA])
    U = sp.csc_matrix(((vA * sA[rA] * sA[cA]).numpy(), (rA.numpy(), cA.numpy())), shape=(C.ni, C.ni))
    A = (U + sp.triu(U, 1).T).tocsr()
    ps = pypardiso.PyPardisoSolver()
    t = time.perf_counter(); ps.factorize(A); tf = time.perf_counter() - t
    try:
        ip = ps.get_iparms()
        mem = dict(peak_analysis_kB=int(ip.get(15, 0)), permanent_kB=int(ip.get(16, 0)), factor_kB=int(ip.get(17, 0)))
    except Exception as e:                                                          # noqa: BLE001
        mem = dict(error=repr(e)[:120])

    class _S:
        def solve(self, r):
            return torch.as_tensor(ps.solve(A, np.ascontiguousarray(r.numpy())), dtype=dt).reshape(r.shape)

        def free(self):
            ps.free_memory(everything=True)
    C.sA, C.sol_I, C.fp32 = sA, _S(), False
    return tf, mem


def timed(fn, reps=3, warm=1):
    for _ in range(warm):
        fn()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t) / reps


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('cases')
    ap.add_argument('--body', default='/root/autodl-tmp/OPL/S0'); ap.add_argument('--s-budget', type=float, default=900.0)
    a = ap.parse_args(argv)
    rec = dict(threads=dict(MKL=os.environ.get('MKL_NUM_THREADS'), OMP=os.environ.get('OMP_NUM_THREADS'), torch=torch.get_num_threads()),
               cpu=os.cpu_count(), per_case={})
    if Path(a.out).exists():
        rec = json.loads(Path(a.out).read_text())
    for case in a.cases.split(','):
        if case in rec['per_case']:
            continue
        r = dict(loadavg_start=os.getloadavg())
        prep = Path(a.body) / case / 'PREP.json'
        if prep.exists():
            p = json.loads(prep.read_text())
            r['topology'] = dict(seconds=p.get('total_seconds'), workers=p.get('workers'))
        t = time.perf_counter(); C = TE.Cell(case, a.body, log=lambda s_: None); r['setup_s'] = time.perf_counter() - t
        t = time.perf_counter(); C.assemble(); r['assembly_s'] = time.perf_counter() - t
        r.update(dofs=int(C.nb), ports=int(C.np_), interior=int(C.ni), elements=int(len(C.cells)))
        r['factor_s'], r['factor_mem'] = factor(C)
        g = torch.Generator().manual_seed(0)
        r['query_s'] = {}
        for B in (1, 16, 64):
            q = torch.randn((C.np_, B), dtype=dt, generator=g)
            r['query_s'][str(B)] = timed(lambda: C.apply(q))
        # explicit dense S, column blocks of 256
        npt = C.np_; blk = 256; done = 0
        t = time.perf_counter()
        while done < npt and time.perf_counter() - t < a.s_budget:
            m = min(blk, npt - done)
            E = torch.zeros((npt, m), dtype=dt); E[done + torch.arange(m), torch.arange(m)] = 1
            _ = C.apply(E)
            done += m
        el = time.perf_counter() - t
        r['explicit_S'] = dict(columns_done=done, columns=npt, seconds_done=el, seconds=el * npt / done,
                               extrapolated=bool(done < npt), dense_GB=8 * npt ** 2 / 2 ** 30)
        C._free(); C = None; gc.collect()
        rec['per_case'][case] = r
        print(json.dumps(dict(case=case, **{k: v for k, v in r.items() if k != 'factor_mem'})), flush=True)
        Path(a.out).write_text(json.dumps(rec, indent=1))


if __name__ == '__main__':
    main(sys.argv[1:])
