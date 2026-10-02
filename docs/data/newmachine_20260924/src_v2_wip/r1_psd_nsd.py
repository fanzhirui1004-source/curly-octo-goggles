"""Follow-up to r1_psd.py (Table ST15): for every partially filled element whose thickness derivative has
lambda_min / max|lambda| below -1e-6, also report lambda_max / max|lambda|, to decide whether the element derivative is
negative semidefinite (lambda_max <= tol * max|lambda|) or indefinite. Same cells, body, derivatives (exact derivative of
the discrete moments at fixed clipping topology, forward mode, and the production central difference) and element
matrices K_e,c = sum_a dM_ea,c T_a as r1_psd.py. New script; r1_psd.py is imported unchanged.
Usage: r1_psd_nsd.py <out.json> <case>[,<case>...] [--body /root/autodl-tmp/OPL/S0] [--rel 1e-5] [--tol 1e-12]
"""
import json, time, argparse
import numpy as np
import torch
import teacher as TE
import r1x3_common as RC
import r1_psd as P

dev, dt = TE.dev, TE.dt


def classify(Mrows, Tm, tol, chunk=4096):
    Tf = Tm.reshape(125, 81 * 81)
    lo_r, hi_r = [], []
    for lo in range(0, len(Mrows), chunk):
        K = (Mrows[lo:lo + chunk] @ Tf).reshape(-1, 81, 81)
        K = 0.5 * (K + K.transpose(1, 2))
        w = torch.linalg.eigvalsh(K)
        mx = w.abs().amax(1).clamp_min(1e-300)
        lo_r.append((w[:, 0] / mx).cpu()); hi_r.append((w[:, -1] / mx).cpu())
    rmin, rmax = torch.cat(lo_r).numpy(), torch.cat(hi_r).numpy()
    neg = rmin < -1e-6
    nsd = neg & (rmax <= tol)
    return dict(n=int(len(rmin)), below_1e6=int(neg.sum()), nsd=int(nsd.sum()), indefinite=int((neg & ~nsd).sum()),
                ratio_minus1=int((rmin <= -1 + 1e-9).sum()), ratio_minus1_nsd=int(((rmin <= -1 + 1e-9) & (rmax <= tol)).sum()),
                max_lmax_ratio_among_negative=float(rmax[neg].max()) if neg.any() else None,
                min_lmax_ratio_among_negative=float(rmax[neg].min()) if neg.any() else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('cases')
    ap.add_argument('--body', default='/root/autodl-tmp/OPL/S0'); ap.add_argument('--rel', type=float, default=1e-5)
    ap.add_argument('--tol', type=float, default=1e-12)
    a = ap.parse_args()
    res = dict(args=vars(a), cells={}, t0=time.strftime('%F %T'))
    for case in a.cases.split(','):
        C = TE.Cell(case, a.body, log=lambda s_: None); C.assemble()
        h3 = (1.0 / C.n) ** 3
        vf = (C.M[:, 0] / h3).cpu().numpy()
        part = np.flatnonzero((vf > 1e-12) & (vf < 1 - 1e-12))
        rows = torch.as_tensor(part, device=dev)
        dM = C.dmoments(rel=a.rel)[:, rows]
        dA, _ = P.dM_ad(C, rows)
        rec = dict(partial=int(len(part)),
                   fd=[classify(dM[c], C.Tm, a.tol) for c in range(8)], fd_uniform=classify(dM.sum(0), C.Tm, a.tol),
                   ad=[classify(dA[c], C.Tm, a.tol) for c in range(8)], ad_uniform=classify(dA.sum(0), C.Tm, a.tol))
        for k in ('fd', 'ad'):
            rec[k + '_corners_total'] = {f: sum(r[f] for r in rec[k]) for f in ('below_1e6', 'nsd', 'indefinite', 'ratio_minus1', 'ratio_minus1_nsd')}
        res['cells'][case] = rec
        print(json.dumps(dict(case=case, ad=rec['ad_corners_total'], ad_uniform={k: rec['ad_uniform'][k] for k in ('below_1e6', 'nsd', 'indefinite')},
                              fd=rec['fd_corners_total'])), flush=True)
        del C, dM, dA; RC.free()
        RC.dump(a.out, res)
    res['t1'] = time.strftime('%F %T'); RC.dump(a.out, res)


if __name__ == '__main__':
    main()
