#!/usr/bin/env python3
"""Why is the assembled sensitivity error pinned near 20% across five different arms?

eps_op spans 18x and g spans 240x across the arms while the measured sensitivity sits
at 20-39% in four of five.  So the question is which directions carry that error, and
whether they are the SAME directions for architecturally different arms.

Mode indices are not comparable across arms, because each arm has its own whitened
eigenbasis.  The true operator's eigenbasis is shared, so everything here is expressed
in it:  A = R_*^T R_*,  A e_j = lambda_j e_j,  lambda ascending (softest first).

Per arm, the comparable statistic is the predicted relative stiffness of true mode j,

    r_j = (e_j^T A_hat e_j) / lambda_j ,

which is 1 when the arm gets mode j exactly right.  The exact first-order decompositions
of the two physical errors, for a displacement x with a = E^T x, are

    energy      x^T(A_hat - A)x / x^T A x   =  sum_j a_j^2 lambda_j (r_j - 1) / sum a_j^2 lambda_j
    compliance  driven by 1/r_j - 1 on the same weights,

so the load-energy distribution over true modes, a_j^2 lambda_j, is reported alongside.
If log r_j correlates strongly between arms, the failure is shared and lives in the
encoder or the sampler rather than in the target or the loss.
"""
import argparse, gc, json, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, '/root/cutfem_neural_a_20260910/source_14301bc56')
sys.path.insert(0, '/root/cutfem_neural_a_20260910/superelement_v0')
sys.path.insert(0, '/root/autodl-tmp/NEURAL_SCHUR')
import v1_scaled as V
from stage_cutfem_neural_a.elimination_reference import load_upper_factor

DEV = V.DEV
F64 = torch.float64


def unpack(path, d):
    p = np.load(path, mmap_mode='r', allow_pickle=False)
    if p.shape != (d * (d + 1) // 2,) or p.dtype != np.float64:
        raise ValueError(f'FP64_UPPER_REQUIRED {path}')
    host = np.zeros((d, d), dtype=np.float64)
    off = 0
    for row in range(d):
        host[row, row:] = p[off:off + d - row]
        off += d - row
    return torch.from_numpy(host).to(DEV)


def build_Ahat(path, kind, d):
    U = unpack(path, d)
    if kind == 'chol':
        out = U.T @ U
    elif kind == 'sqrt':
        M = U + torch.triu(U, 1).T
        del U
        out = M @ M
        del M
    else:
        raise ValueError(kind)
    out = 0.5 * (out + out.T)
    gc.collect(); torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seat', type=int, default=253)
    ap.add_argument('--arms', required=True, help='name:kind:path, comma separated')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--top', type=int, default=32)
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    recs = json.load(open('/root/autodl-tmp/CUTFEM_SUPERELEMENT_LABELS/V1_LABELS.json'))
    rec = [r for r in recs if int(r['seat']) == a.seat][0]
    label = V.Label(rec, 0.2, 0.03, 10.0)
    g = label.to_gpu(need_A=False, need_Z=False, z_dtype=torch.float32)
    d, q = label.d, label.q
    quotient = g['data'].quotient
    Rstar = load_upper_factor(Path(rec['reference']) / 'R_UPPER.npy', d, DEV)
    A = Rstar.T @ Rstar
    A = 0.5 * (A + A.T)
    ijk = np.asarray(label.ijk)
    nn = len(ijk)
    report = dict(seat=a.seat, d=int(d), q=int(q),
                  frame='eigenbasis of the true A, eigenvalues ascending (softest first)',
                  statistic='r_j = (e_j^T A_hat e_j) / lambda_j; 1 is exact')

    # ---- two-cell assembly, exact solve, to get the displacement each module carries
    xhi, xlo = int(ijk[:, 0].max()), int(ijk[:, 0].min())
    key = {(int(r[0]), int(r[1]), int(r[2])): i for i, r in enumerate(ijk)}
    shift = xhi - xlo
    gid_A = np.arange(nn); gid_B = np.empty(nn, dtype=np.int64); nxt = nn
    for i, r in enumerate(ijk):
        if int(r[0]) == xlo and (xhi, int(r[1]), int(r[2])) in key:
            gid_B[i] = key[(xhi, int(r[1]), int(r[2]))]
        else:
            gid_B[i] = nxt; nxt += 1
    N_nodes = nxt; N = 3 * N_nodes
    dof_A = torch.as_tensor((gid_A[:, None] * 3 + np.arange(3)).ravel(), device=DEV)
    dof_B = torch.as_tensor((gid_B[:, None] * 3 + np.arange(3)).ravel(), device=DEV)
    xyz = np.zeros((N_nodes, 3)); xyz[gid_A] = ijk; xyz[gid_B] = ijk + np.array([shift, 0, 0])
    xyz = xyz / 64.0
    P = torch.as_tensor(xyz, dtype=F64, device=DEV)
    Nrb = torch.zeros(N, 6, dtype=F64, device=DEV)
    for k in range(3):
        Nrb[k::3, k] = 1.0
    c = P - P.mean(0)
    Nrb[0::3, 3] = -c[:, 1]; Nrb[1::3, 3] = c[:, 0]
    Nrb[1::3, 4] = -c[:, 2]; Nrb[2::3, 4] = c[:, 1]
    Nrb[0::3, 5] = c[:, 2];  Nrb[2::3, 5] = -c[:, 0]
    Nrb, _ = torch.linalg.qr(Nrb)
    endL = torch.as_tensor(np.flatnonzero(xyz[:, 0] <= xyz[:, 0].min() + 1e-9), device=DEV)
    endR = torch.as_tensor(np.flatnonzero(xyz[:, 0] >= xyz[:, 0].max() - 1e-9), device=DEV)
    loads = {}
    for name, comp in (('compression_x', 0), ('shear_y', 1), ('shear_z', 2)):
        f = torch.zeros(N, dtype=F64, device=DEV)
        f[endL * 3 + comp] = -1.0 / len(endL); f[endR * 3 + comp] = +1.0 / len(endR)
        loads[name] = f
    f = torch.zeros(N, dtype=F64, device=DEV)
    wl = P[endL, 1] - P[endL, 1].mean(); wr = P[endR, 1] - P[endR, 1].mean()
    f[endL * 3 + 0] = -wl / wl.abs().sum(); f[endR * 3 + 0] = wr / wr.abs().sum()
    loads['bending_xy'] = f
    for k in loads:
        loads[k] = loads[k] - Nrb @ (Nrb.T @ loads[k])
        loads[k] = loads[k] / loads[k].norm()

    left = quotient.lift(A.T.contiguous())
    S = quotient.lift(left.T.contiguous())
    S = 0.5 * (S + S.T)
    del left
    K = torch.zeros(N, N, dtype=F64, device=DEV)
    K[dof_A.unsqueeze(1), dof_A.unsqueeze(0)] += S
    K[dof_B.unsqueeze(1), dof_B.unsqueeze(0)] += S
    K = 0.5 * (K + K.T)
    del S
    sc = float(K.diagonal().mean())
    K.addmm_(Nrb, Nrb.T, alpha=sc)
    L = torch.linalg.cholesky(K)
    del K
    gc.collect(); torch.cuda.empty_cache()
    xq = {}
    for name, fv in loads.items():
        u = torch.cholesky_solve(fv.unsqueeze(1), L).squeeze(1)
        u = u - Nrb @ (Nrb.T @ u)
        xq[name] = quotient(u[dof_A].unsqueeze(1)).squeeze(1)      # module A, quotient coords
    del L, Nrb
    gc.collect(); torch.cuda.empty_cache()

    # ---- the shared frame
    tick = time.time()
    lam, E = torch.linalg.eigh(A)
    report['eigh_seconds'] = time.time() - tick
    report['lambda_min'] = float(lam[0]); report['lambda_max'] = float(lam[-1])
    del A
    gc.collect(); torch.cuda.empty_cache()

    # load energy over true modes, a_j^2 lambda_j, and its decile profile
    decile = (torch.arange(d, device=DEV) * 10 // d).clamp(max=9)
    energy = {}
    for name, x in xq.items():
        aj = E.T @ x
        w = (aj.square() * lam)
        w = w / w.sum()
        energy[name] = [float(w[decile == k].sum()) for k in range(10)]
    report['load_energy_by_true_stiffness_decile'] = energy
    print('load energy over true-stiffness deciles (softest decile first):', flush=True)
    for name, v in energy.items():
        print('   %-14s %s' % (name, ' '.join('%.4f' % t for t in v)), flush=True)

    arms = {}
    logr = {}
    for spec in a.arms.split(','):
        name, kind, path = spec.split(':', 2)
        if not Path(path).exists():
            print(f'   skipping {name}: missing {path}', flush=True)
            continue
        Ahat = build_Ahat(path, kind, d)
        r = ((E * (Ahat @ E)).sum(dim=0)) / lam            # e_j^T A_hat e_j / lambda_j
        del Ahat
        gc.collect(); torch.cuda.empty_cache()
        logr[name] = r.clamp_min(1e-300).log()
        row = dict(kind=kind, path=path,
                   r_min=float(r.min()), r_max=float(r.max()),
                   r_median=float(r.median()),
                   nonpositive_r=int((r <= 0).sum()),
                   r_median_by_decile=[float(r[decile == k].median()) for k in range(10)],
                   fraction_outside_10pct_by_decile=[float(((r[decile == k] - 1).abs() > .1).double().mean())
                                                     for k in range(10)],
                   loads={})
        for lname, x in xq.items():
            aj = E.T @ x
            w = aj.square() * lam
            tot = float(w.sum())
            energy_err = float((w * (r - 1)).sum() / tot)
            inv = (1.0 / r.clamp_min(1e-300) - 1.0)
            wc = aj.square() / lam
            compliance_err = float((wc * inv).sum() / float(wc.sum()))
            contrib = (wc * inv) / float(wc.sum())
            order = contrib.abs().argsort(descending=True)[:a.top]
            row['loads'][lname] = dict(
                first_order_energy_error=energy_err,
                first_order_compliance_error=compliance_err,
                top_modes=[int(i) for i in order.cpu()],
                top_contributions=[float(contrib[i]) for i in order],
                top_share_of_total=float(contrib[order].abs().sum() / contrib.abs().sum()),
                compliance_error_by_decile=[float(contrib[decile == k].sum()) for k in range(10)])
        arms[name] = row
        print('\n%s  r median %.4g  r range [%.4g, %.4g]  nonpositive %d'
              % (name, row['r_median'], row['r_min'], row['r_max'], row['nonpositive_r']), flush=True)
        print('   r median by decile   %s' % ' '.join('%8.4g' % v for v in row['r_median_by_decile']), flush=True)
        print('   frac >10%% off       %s' % ' '.join('%8.3f' % v for v in row['fraction_outside_10pct_by_decile']), flush=True)
        for lname in xq:
            L2 = row['loads'][lname]
            print('   %-14s 1st-order compliance %9.4f  energy %9.4f  top-%d carry %.3f'
                  % (lname, L2['first_order_compliance_error'], L2['first_order_energy_error'],
                     a.top, L2['top_share_of_total']), flush=True)
        report['arms'] = arms
        (a.output / 'FLOOR_PROBE.json').write_text(json.dumps(report, indent=1))

    # ---- do different arms fail on the same true modes?
    names = sorted(logr)
    corr = {}
    for i, p in enumerate(names):
        for qn in names[i + 1:]:
            x1 = logr[p] - logr[p].mean(); x2 = logr[qn] - logr[qn].mean()
            corr[f'{p} vs {qn}'] = float((x1 @ x2) / (x1.norm() * x2.norm()))
    report['log_r_correlation_between_arms'] = corr
    print('\ncorrelation of log r_j between arms (1 = identical failure pattern):', flush=True)
    for k, v in sorted(corr.items(), key=lambda t: -t[1]):
        print('   %-44s %.4f' % (k, v), flush=True)
    report['status'] = 'FLOOR_PROBE_COMPLETE'
    report['seconds'] = time.time() - t0
    (a.output / 'FLOOR_PROBE.json').write_text(json.dumps(report, indent=1))
    print('\ndone (%.0f s)' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
