"""ELEV 2026-10-05, figure data only.  After nice_cell_field.py: for every cell of the final plate design (cplateN, k = 23)
the exact field u = E_m q_m (assembled cell matrix, interior factor, teacher.Cell.extend; q_m = B_m U from the stored solve of
the CPU exact check, exact_cplate/cplateN_k023/q) against the NICE recovered field u_hat = F_m B_m U_hat (nicefield_<case>.npz)
and against the NICE extension of the exact q (F_m q_m).  Relative differences in the energy norm of the cell's K and in the
Euclidean norm of the nodal displacements (all DOFs, interior, retained); plate totals over all cells.
Host route (OPL_DEV=cpu, interior factor diag_sens.cpu_factor by MKL PARDISO), as the workers of exact_check_cpu.py.
Outputs: cellfield_<case>.npz (nodal fields, float32) for the cut cells, cellfield_summary.json.
"""
import os, sys, json, time, gc
from pathlib import Path
import numpy as np
os.environ.update(OPL_DEV='cpu', CUDA_VISIBLE_DEVICES='', OPL_GP_CACHE='0')   # host route of exact_check_cpu.py (workers)
sys.path.insert(0, '/root/autodl-tmp/OPL/src_v2'); os.chdir('/root/autodl-tmp/OPL/src_v2')
import diag_sens as DS                                                  # noqa: F401  first (hides the GPU, host defaults)
import trainlib as TL
import box_encode as BX
BX.dev = TL.dev
import torch
import teacher as TE
dev, dt = TE.dev, TE.dt
RUN = Path('/root/autodl-tmp/OPL/S1/V2/R1/FINAL2/cplateN'); BODY = RUN / 'body'
OUT = Path('/root/autodl-tmp/OPL/ELEV_20261005/render')
S = json.loads((OUT / 'nice_solve.json').read_text())
L = json.loads(Path('/root/autodl-tmp/OPL/S1/V2/R1/OPT/layouts/plate841.json').read_text())
info = {c['case']: c for c in L['cells']}
log = lambda **d: print(json.dumps(d, default=float), flush=True)
summ = dict(S, load='consistent traction of unit resultant on the end face y = min, direction x',
            clamp='every cut-band DOF of every cut cell', cells={})
tot = dict(eE=0.0, eH=0.0, err=0.0, err_loc=0.0, n2=0.0, d2=0.0)
t0 = time.perf_counter()
for c in S['order']:
    z = np.load(OUT / f'nicefield_{c}.npz')
    C2 = TE.Cell(c, str(BODY), log=lambda s_: None); C2.assemble(); DS.cpu_factor(C2)
    with torch.no_grad():
        qe = torch.as_tensor(z['q_exact'], dtype=dt, device=dev)
        u_ex = C2.extend(qe)
        uh = torch.as_tensor(z['u_hat'], dtype=dt, device=dev).reshape(u_ex.shape)
        un = torch.as_tensor(z['u_nq'], dtype=dt, device=dev).reshape(u_ex.shape)
        e = lambda v: float((v * (C2.K @ v)).sum())
        d1, d2 = uh - u_ex, un - u_ex
        eE, eH, err, errl = e(u_ex), e(uh), max(e(d1), 0.0), max(e(d2), 0.0)
        I, P = C2.I, C2.P
        mag_e, mag_h = u_ex.reshape(-1, 3).norm(dim=1), uh.reshape(-1, 3).norm(dim=1)
        base = info[c.rsplit('_o', 1)[0]]
        rec = dict(position=base['position'], kind=base['kind'], retained=base['retained'], nodes=int(len(C2.nodes)),
                   ports=int(C2.np_), interior=int(C2.ni), energy_exact=eE, energy_nice=eH,
                   rel_energy_norm=float(np.sqrt(err / eE)), rel_l2=float(d1.norm() / u_ex.norm()),
                   rel_l2_interior=float(d1[I].norm() / u_ex[I].norm()), rel_l2_retained=float(d1[P].norm() / u_ex[P].norm()),
                   rel_max_magnitude=float((mag_h - mag_e).abs().max() / mag_e.max()),
                   max_magnitude_exact=float(mag_e.max()), max_magnitude_nice=float(mag_h.max()),
                   local_rel_energy_norm=float(np.sqrt(errl / eE)), local_rel_l2=float(d2.norm() / u_ex.norm()),
                   q_rel_l2=float((uh[P] - u_ex[P]).norm() / u_ex[P].norm()),
                   q_exact_consistency=float((u_ex[P] - qe).norm() / qe.norm()))
        tot['eE'] += eE; tot['eH'] += eH; tot['err'] += err; tot['err_loc'] += errl
        tot['n2'] += float(u_ex.norm()) ** 2; tot['d2'] += float(d1.norm()) ** 2
        summ['cells'][c] = rec
        if base['kind'] != 'FULL':
            np.savez_compressed(OUT / f'cellfield_{c}.npz', nodes=C2.nodes.astype(np.int32), is_box=C2.is_box, is_cut=C2.is_cut,
                                u_nice=uh.reshape(-1, 3).cpu().numpy().astype(np.float32),
                                u_exact=u_ex.reshape(-1, 3).cpu().numpy().astype(np.float32),
                                u_nice_exactq=un.reshape(-1, 3).cpu().numpy().astype(np.float32),
                                taus=np.asarray(C2.taus0), normal=np.asarray(C2.normal), offset=C2.offset)
    log(event='cell', case=c, s=time.perf_counter() - t0, **{k: v for k, v in rec.items() if k != 'position'})
    C2._free(); del C2, u_ex, uh, un, d1, d2, qe
    gc.collect()
summ['plate'] = dict(rel_energy_norm=float(np.sqrt(tot['err'] / tot['eE'])), rel_l2=float(np.sqrt(tot['d2'] / tot['n2'])),
                     local_rel_energy_norm=float(np.sqrt(tot['err_loc'] / tot['eE'])),
                     energy_exact_sum=tot['eE'], energy_nice_sum=tot['eH'])
(OUT / 'cellfield_summary.json').write_text(json.dumps(summ, indent=1, default=float))
log(event='done', s=time.perf_counter() - t0, plate=summ['plate'])
