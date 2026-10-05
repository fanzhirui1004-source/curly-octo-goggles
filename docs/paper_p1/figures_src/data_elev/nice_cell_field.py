"""ELEV 2026-10-05, figure data only (no change to any run directory).
NICE lattice solve of the final design (k = 23) of the plate supported on its cut (FINAL2/cplateN), by the deployment route
of opt_design.analyse_fast (same modules, flags and preconditioner; no sensitivities), with a small resident budget so that
the learned cells are streamed from host memory.  Then, for every cut cell: the NICE recovered field u_hat = F_m B_m U_hat,
the exact field u = E_m q_m from the stored exact retained displacements of the CPU exact check (exact_cplate/cplateN_k023/q),
and the NICE extension of the exact q (F_m q_m) for reference; relative differences in the energy norm of the cell's K and in
the Euclidean norm of the nodal displacement vector.
Outputs (render dir): nice_q_k023.npz (B_m U_hat per cell), nicefield_<case>.npz, nice_solve.json; the exact fields and\nthe differences: exact_cell_field.py.
"""
import os, sys, json, time, gc
from pathlib import Path
import numpy as np

RUN = Path('/root/autodl-tmp/OPL/S1/V2/R1/FINAL2/cplateN')
EXQ = Path('/root/autodl-tmp/OPL/S1/V2/R1/OPT/exact_cplate/cplateN_k023/q')
LAYOUT = '/root/autodl-tmp/OPL/S1/V2/R1/OPT/layouts/plate841.json'
OUT = Path('/root/autodl-tmp/OPL/ELEV_20261005/render'); OUT.mkdir(parents=True, exist_ok=True)
TMP = OUT / 'tmp'; TMP.mkdir(exist_ok=True)
RES_GB = float(os.environ.get('ELEV_RES_GB', '2.0'))
K = 23

import torch
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
torch.cuda.set_per_process_memory_fraction(float(os.environ.get('ELEV_FRAC', '0.33')))
sys.path.insert(0, '/root/autodl-tmp/OPL/src_v2'); os.chdir('/root/autodl-tmp/OPL/src_v2')
import models as MD                                                     # noqa: F401  first: OPL_CONV_FP32
import teacher as TE
import trainlib as TL
import fastnet as FN
import evalnet as EN
import bench_deploy as BD
import lat_multi as LM
import lat_precond as PR
import lat_hetero as LH
import fastidx as FI
import stream_ops as SO
import lat_scale as LS
FI.ON = True; FN.FUSED = True
dev, dt = TE.dev, TE.dt
assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
log = lambda **d: print(json.dumps(d, default=float), flush=True)

H = [json.loads(l) for l in open(RUN / 'history.jsonl')]
h_k = [h for h in H if h.get('k') == K][-1]
L = json.loads(Path(LAYOUT).read_text())
pos_of = {c['case']: tuple(int(v) for v in c['position']) for c in L['cells']}
cases = h_k['cases']
positions = [pos_of[c.rsplit('_o', 1)[0]] for c in cases]
BODY = RUN / 'body'
t0 = time.perf_counter()
name, hold = LH.holder_for('A3=/root/autodl-tmp/OPL/S1/V2/A3_2grid/best.pt')
SO.init()
Cs, ops, kpp_host = {}, {}, {}
for case in cases:
    C = TE.Cell(case, str(BODY), log=lambda s_: None, deploy=True)
    C.assemble_deploy(C.taus0)
    BD.netdata(C, str(BODY), TMP / case)
    geo = TL.Geo(case, str(BODY), TMP, neumann=False, log=lambda s_: None, cell=C, load_banks=False)
    if hold.model is not None:
        hold.model.caches.pop(case, None)
    model = hold.add(geo)
    op = EN.FastOp(FN.FastNet(model, geo))
    with torch.no_grad():
        op.apply(torch.zeros((C.np_, 1), dtype=dt, device=dev))
    kpp_host[case] = tuple(x.cpu() for x in C._kpp_cache)
    C._kpp_cache = kpp_host[case]; C.diag3 = None
    if torch.cuda.memory_allocated() / 2 ** 30 < RES_GB:
        ops[case] = LS._Resident(op)
    else:
        ops[case] = SO.StreamedOp(op, C, [op.fast, geo, C])
    Cs[case] = C
    del geo, op
    gc.collect(); torch.cuda.empty_cache()
log(event='prep', s=time.perf_counter() - t0, alloc_gb=torch.cuda.memory_allocated() / 2 ** 30,
    peak_gb=torch.cuda.max_memory_allocated() / 2 ** 30,
    streamed=sum(isinstance(o, SO.StreamedOp) for o in ops.values()))
lay = {}
for case, p in zip(cases, positions):
    g = LM.from_teacher(Cs[case])
    g._kpp_fn = (lambda cs: (lambda: tuple(x.to(dev) for x in kpp_host[cs])))(case)
    lay[tuple(p)] = g
lat = LM.MultiLattice(lay, clamp=('cut', ''), load=('y', 'min'), loads='consistent', n_random=0, device=dev, max_cols=16,
                      log=lambda s_: None)
order = [g.case for g in lat.geoms]
olist = [ops[c] for c in order]
SO.link([o for o in olist if isinstance(o, SO.StreamedOp)])
F = lat.F[:, ['xyz'.index('x')]].contiguous()
for g in lat.geoms:
    g._kpp = None
kpp = lat.assemble_kpp()
for g in lat.geoms:
    g._kpp = None
QF = OUT / f'nice_q_k{K:03d}.npz'
SOLVED = QF.exists()
t = time.perf_counter()
if SOLVED:
    z = np.load(QF); qh = {c: z[c] for c in z.files}; Chat = float('nan'); tres = float('nan'); r = dict(iterations=-1)
    log(event='solve_reused', file=str(QF))
else:
  fac = PR.Factory(lat, olist, shared={'kpp_triplets': kpp, 'kpp_triplets_s': 0.0}, kpp_backend='auto', log=lambda d: None)
  pc, _, _ = fac.build('bnn:kpp:q1r')
  del kpp
  log(event='precond', s=time.perf_counter() - t, peak_gb=torch.cuda.max_memory_allocated() / 2 ** 30)
  t = time.perf_counter()
  r = PR.pcg(lat, olist, pc, F=F, tol=1e-6, maxit=3000)
  LS._free_prec(fac, pc); del fac, pc
  X = r['X']
  with torch.no_grad():
      rho = F - lat.matvec(olist, X)
      Chat = float((F * X).sum()); tres = float(rho.norm() / F.norm()); Ur = float((X * rho).sum())
  log(event='solve', s=time.perf_counter() - t, pcg=int(r['iterations']), C_hat=Chat, C_hat_run=h_k['C'],
      rel_vs_run=Chat / h_k['C'] - 1, true_residual=tres, Ut_rho_rel=Ur / Chat, peak_gb=torch.cuda.max_memory_allocated() / 2 ** 30)
  qh = {}
  with torch.no_grad():
      for i, c in enumerate(order):
          qh[c] = lat.gather(X, i).detach().cpu().numpy()
  np.savez(OUT / f'nice_q_k{K:03d}.npz', **qh)


# NICE fields of the cut cells (streamed operators: activate one at a time)
targets = list(order)                                                   # every cell (exact fields: exact_cell_field.py)
fields = {}
for c in targets:
    i = order.index(c)
    o = olist[i]
    qe = np.load(EXQ / f'{c}.npy')
    with o.active():
        with torch.no_grad():
            u_hat = o.field(torch.as_tensor(qh[c], dtype=dt, device=dev)).to(dt).cpu().numpy()
            u_nq = o.field(torch.as_tensor(qe, dtype=dt, device=dev)).to(dt).cpu().numpy()
    fields[c] = (u_hat, u_nq, qe)
    np.savez(OUT / f'nicefield_{c}.npz', u_hat=u_hat, u_nq=u_nq, q_hat=qh[c], q_exact=qe)
SO.park_all()
for o in olist:
    o.release()
for c in order:
    hold.model.caches.pop(c, None)
del olist, ops, lat, F, Cs, hold
gc.collect(); torch.cuda.empty_cache()
log(event='nice_fields', n=len(fields), peak_gb=torch.cuda.max_memory_allocated() / 2 ** 30)

(OUT / 'nice_solve.json').write_text(json.dumps(dict(design=f'cplateN k={K}', C_hat=Chat, C_hat_run=h_k['C'],
    pcg=int(r['iterations']), true_residual=tres, order=order), indent=1, default=float))
log(event='done', s=time.perf_counter() - t0)
