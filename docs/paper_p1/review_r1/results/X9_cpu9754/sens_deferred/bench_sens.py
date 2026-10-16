"""CPU speed of cpu_sens.cell_sens on one cell for several row batches and checkpoint settings (same result).
Usage: bench_sens.py <case> <batches comma> [body]"""
import diag_sens as DS                                                   # noqa: F401  first: CPU environment
import sys, time, json
import numpy as np
import torch
import trainlib as TL
import teacher as TE
import box_encode as BX
import moments_ad as MA
import cpu_sens as CS
BX.dev = TL.dev
case = sys.argv[1]; batches = [int(b) for b in sys.argv[2].split(',')]
body = sys.argv[3] if len(sys.argv) > 3 else '/root/autodl-tmp/OPL/S4/body'
C = TE.Cell(case, body, log=lambda s_: None); C.assemble()
DS.cpu_factor(C)
q = torch.as_tensor(np.random.default_rng(0).standard_normal((C.np_, 3)), dtype=torch.float64)
u = C.extend(q).numpy()
Lc = CS.light(C)
cells, taus, nrm, off = MA.cell_rows(Lc)
nf = int((~MA.full_rows(cells, Lc.n, taus, nrm, off, s=Lc.s, surface=Lc.surface)).sum())
out = dict(case=case, elements=len(Lc.cells), rows_not_full=nf, threads=torch.get_num_threads(), runs=[])
ref = None
for ck in ((False,) if __import__('os').environ.get('OPL_SENS_NOCKPT') == '1' else (True, False)):
    MA.CHECKPOINT = ck
    for b in batches:
        t = time.perf_counter(); s, tt = CS.cell_sens(Lc, u, batch=b); dt = time.perf_counter() - t
        ref = s if ref is None else ref
        out['runs'].append(dict(checkpoint=ck, batch=b, seconds=dt, rel_vs_first=float(np.linalg.norm(s - ref) / np.linalg.norm(ref))))
        print(json.dumps(out['runs'][-1]), flush=True)
print(json.dumps(out))
