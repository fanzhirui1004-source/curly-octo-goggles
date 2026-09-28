"""Revision round 1, E14: element-level positive semidefiniteness of the numerical stiffness and of its design
derivative on the detailed cells (extends arch_analysis_0926/work/mech2/psd.py, which sampled 300 elements of one
training cell, to EVERY partially filled element). New script.

Per cell (teacher.Cell at its own tau, fixed body):
  K_e      = sum_m M_em T_m (81 x 81) for every element with 0 < volume fraction < 1 (and, as a control, 64 full ones);
  K_e,c    = sum_m dM_em,c T_m for the 8 corners c, with dM from
             'fd'  the central difference of Cell.dmoments (Eq. H.6, h = --rel tau_c; default 1e-5, the value used for all
                   reference sensitivities), and
             'ad'  the exact derivative of the discrete moments within the fixed clip topology (moments_ad.moments_rows
                   differentiated in forward mode, one tangent per corner; fallback: reverse mode, 125 passes);
           and the uniform-thickening derivative sum_c K_e,c.
  Reported: min over elements of lambda_min / max|lambda| (quantiles 0, 1e-3, 1e-2, 0.5), the number of elements with
  lambda_min < -1e-12, -1e-10, -1e-8 and -1e-6 times max|lambda|, the worst element (index, volume fraction, corner),
  ||dM_ad - dM_fd|| / ||dM_fd|| per corner, and the moments of moments_rows against Cell.moments (same integrator).
Element-wise PSD implies global PSD because K_,c = sum_e P_e^T K_e,c P_e (the ghost term does not depend on tau).
Usage: r1_psd.py <out.json> <case>[,<case>...] [--body /root/autodl-tmp/OPL/S0] [--rel 1e-5] [--no-ad] [--chunk 4096]
"""
import json, time, argparse, os
from pathlib import Path
import numpy as np
import torch
import teacher as TE
import moments_ad as MA
import r1x3_common as RC

dev, dt = TE.dev, TE.dt


def eig_stats(Mrows, Tm, chunk, ids, vf, corner=None):
    """Mrows (E, 125) coefficient rows (moments or moment derivatives) -> eigenvalue ratios lambda_min / max|lambda|."""
    Tf = Tm.reshape(125, 81 * 81)
    rmin = torch.empty(len(Mrows), dtype=dt, device=Mrows.device)
    rmax_neg = torch.empty(len(Mrows), dtype=dt, device=Mrows.device)
    for lo in range(0, len(Mrows), chunk):
        K = (Mrows[lo:lo + chunk] @ Tf).reshape(-1, 81, 81)
        K = 0.5 * (K + K.transpose(1, 2))
        w = torch.linalg.eigvalsh(K)
        mx = w.abs().amax(1).clamp_min(1e-300)
        rmin[lo:lo + chunk] = w[:, 0] / mx
        rmax_neg[lo:lo + chunk] = w[:, -1] / mx
    r = rmin.cpu().numpy()
    k = int(np.argmin(r)) if len(r) else -1
    out = dict(n=int(len(r)), q0=float(r.min()) if len(r) else None,
               q1e_3=float(np.quantile(r, 1e-3)) if len(r) else None, q1e_2=float(np.quantile(r, 1e-2)) if len(r) else None,
               q50=float(np.median(r)) if len(r) else None,
               n_below={},
               worst=dict(element=int(ids[k]), vf=float(vf[k]), ratio=float(r[k])) if len(r) else None,
               max_over_max=float(rmax_neg.cpu().numpy().max()) if len(r) else None)
    if corner is not None:
        out['corner'] = corner
    for t in ('1e-12', '1e-10', '1e-8', '1e-6'):
        out['n_below'][t] = int((r < -float(t)).sum())
    return out


def dM_ad(C, rows):
    """(8, len(rows), 125) exact derivative of the discrete moments (fixed clip topology), forward mode; and the moments."""
    import torch.autograd.forward_ad as fwAD
    MA.CHECKPOINT = False
    cells = np.asarray(C.cells)[rows.cpu().numpy()]
    E = len(cells)
    taus = torch.as_tensor(np.asarray(C.taus, float), dtype=dt, device=dev)[None].expand(E, 8).contiguous()
    nrm, off = MA.plane_rows(C.normal, C.offset, E, dev)
    sf = getattr(C, 'surface', 'P')
    out, M0 = [], None
    for c in range(8):
        tan = torch.zeros_like(taus); tan[:, c] = 1.0
        with fwAD.dual_level():
            td = fwAD.make_dual(taus, tan)
            Md = MA.moments_rows(cells, C.n, td, nrm, off, s=C.s, levels=C.levels, surface=sf)
            p, tg = fwAD.unpack_dual(Md)
            out.append(tg.detach().clone() if tg is not None else torch.zeros_like(p))
            if M0 is None:
                M0 = p.detach().clone()
    return torch.stack(out), M0


def dM_ad_reverse(C, rows, batch=512):
    """Fallback: reverse mode, one pass per moment (125 per row batch)."""
    cells = np.asarray(C.cells)[rows.cpu().numpy()]
    E = len(cells)
    taus = torch.as_tensor(np.asarray(C.taus, float), dtype=dt, device=dev)[None].expand(E, 8).contiguous()
    nrm, off = MA.plane_rows(C.normal, C.offset, E, dev)
    G = torch.eye(125, dtype=dt, device=dev)
    out = torch.zeros((8, E, 125), dtype=dt, device=dev); M0 = torch.zeros((E, 125), dtype=dt, device=dev)
    for b0 in range(0, E, batch):
        b1 = min(E, b0 + batch)
        Gb = G[None].expand(b1 - b0, 125, 125).contiguous()
        M, V = MA.moments_vjp(cells[b0:b1], C.n, taus[b0:b1], nrm[b0:b1], off[b0:b1], Gb, s=C.s, levels=C.levels,
                              batch=b1 - b0, surface=getattr(C, 'surface', 'P'))
        out[:, b0:b1] = V.permute(1, 0, 2); M0[b0:b1] = M
    return out, M0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('cases')
    ap.add_argument('--body', default='/root/autodl-tmp/OPL/S0'); ap.add_argument('--rel', type=float, default=1e-5)
    ap.add_argument('--no-ad', action='store_true'); ap.add_argument('--chunk', type=int, default=4096)
    ap.add_argument('--max-elems', type=int, default=0, help='smoke tests: limit the partially filled elements')
    a = ap.parse_args()
    res = dict(args=vars(a), env=RC.env_flags(), cells={}, t0=time.strftime('%F %T'))
    for case in a.cases.split(','):
        t = time.perf_counter()
        C = TE.Cell(case, a.body, log=lambda s_: None)
        C.assemble()
        h3 = (1.0 / C.n) ** 3
        vf = (C.M[:, 0] / h3).cpu().numpy()
        part = np.flatnonzero((vf > 1e-12) & (vf < 1 - 1e-12))
        full = np.flatnonzero(vf >= 1 - 1e-12)
        if a.max_elems:
            part = part[:a.max_elems]
        rows = torch.as_tensor(part, device=dev)
        rec = dict(elements=int(len(vf)), partial=int(len(part)), full=int(len(full)), taus=list(C.taus), n=C.n)
        rec['K_e_partial'] = eig_stats(C.M[rows], C.Tm, a.chunk, part, vf[part])
        ctl = full[:64]
        rec['K_e_full_control'] = eig_stats(C.M[torch.as_tensor(ctl, device=dev)], C.Tm, a.chunk, ctl, vf[ctl]) if len(ctl) else None
        dM = C.dmoments(rel=a.rel)[:, rows]                                     # 8 x P x 125
        dfull = float(C.dM[:, torch.as_tensor(full, device=dev)].abs().max()) if len(full) else 0.0
        rec['dM_fd_full_elements_absmax'] = dfull
        rec['fd'] = [eig_stats(dM[c], C.Tm, a.chunk, part, vf[part], corner=c) for c in range(8)]
        rec['fd_uniform'] = eig_stats(dM.sum(0), C.Tm, a.chunk, part, vf[part])
        if not a.no_ad:
            t1 = time.perf_counter()
            try:
                dA, M0 = dM_ad(C, rows); rec['ad_mode'] = 'forward'
            except Exception as e:                                               # noqa: BLE001
                rec['ad_forward_error'] = repr(e)[:300]
                dA, M0 = dM_ad_reverse(C, rows); rec['ad_mode'] = 'reverse'
            rec['ad_seconds'] = time.perf_counter() - t1
            rec['ad_moments_vs_cell'] = float((M0 - C.M[rows]).norm() / C.M[rows].norm())
            rec['ad_vs_fd_rel'] = [float((dA[c] - dM[c]).norm() / dM[c].norm().clamp_min(1e-300)) for c in range(8)]
            rec['ad'] = [eig_stats(dA[c], C.Tm, a.chunk, part, vf[part], corner=c) for c in range(8)]
            rec['ad_uniform'] = eig_stats(dA.sum(0), C.Tm, a.chunk, part, vf[part])
            del dA, M0
        rec['worst_fd'] = min(r_['q0'] for r_ in rec['fd'])
        rec['worst_ad'] = min(r_['q0'] for r_ in rec['ad']) if 'ad' in rec else None
        rec['seconds'] = time.perf_counter() - t
        print(json.dumps(dict(event='CELL', case=case, partial=rec['partial'], K_e_min=rec['K_e_partial']['q0'],
                              worst_fd=rec['worst_fd'], worst_ad=rec['worst_ad'], ad_mode=rec.get('ad_mode'),
                              ad_vs_fd=rec.get('ad_vs_fd_rel'), seconds=rec['seconds'], gpu=RC.gpu_gb()), default=float), flush=True)
        res['cells'][case] = rec
        del C, dM; RC.free()
        RC.dump(a.out, res)
    res['t1'] = time.strftime('%F %T')
    RC.dump(a.out, res)


if __name__ == '__main__':
    main()
