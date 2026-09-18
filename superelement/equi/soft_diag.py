"""Why does the physics error track g?  Decompose it into load-energy weights.

`g = max_i max(mu_i, 1/mu_i)` is an unweighted worst case over the pencil (Ahat, A*); the
assembled compliance error is, to first order, a load-energy-weighted average over the same
spectrum:

    -(chat - c)/c = u^T Khat u / u^T K u - 1 = <mu>_w - 1,   w_i = the true solution's energy
                                                            share in pencil mode i.

The two therefore agree only when the modes that attain the extremes also carry load.  This
script measures both sides on an already-accepted seat and reports

  * the recomputed spectrum (g, mu_min, mu_max) as a cross-check on the acceptance table;
  * w at argmax mu and at argmin mu -- the load share of the modes that set g;
  * the first-order predictor <mu>_w - 1 against the measured signed compliance error;
  * g_live: the sandwich restricted to modes carrying at least 0.1 % of the load energy;
  * g_soft(k): the same sandwich restricted to the k softest eigenvectors of A*, which is
    basis-free and boundary-condition-free, unlike w, so it could serve as a gate;
  * the load energy share inside that soft subspace, and how localized the soft modes are
    (a localized soft mode is a void-pocket near-mechanism).

    python -m superelement.equi.soft_diag --seat 100186 --pred <A_PRED_UPPER.npy> --output DIR
"""
from __future__ import annotations

import argparse, gc, json, sys, time
from pathlib import Path
import numpy as np
import torch

F64 = torch.float64


def _free(dev):
    gc.collect()
    if dev.type == 'cuda':
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', type=Path, default=Path('/root/autodl-tmp/CLAUDE_LABELS_20260917/V2_LABELS.json'))
    ap.add_argument('--seat', type=int, required=True)
    ap.add_argument('--pred', type=Path, required=True, help='A_PRED_UPPER.npy (d x d upper Cholesky of Ahat)')
    ap.add_argument('--accept', type=Path, default=None, help='ASSEMBLE_TWO.json for the measured errors')
    ap.add_argument('--source', type=Path, default=Path('/root/autodl-tmp/CLAUDE_SQRTHEAD_20260917/src_v5'))
    ap.add_argument('--softk', default='16,64,256,1024')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(a.source))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from stage_cutfem_m4.quotient import RigidQuotient
    from superelement.equi.context import compile_equi_inputs, rigid_span_residual
    from superelement.equi.assemble_two import unpack, dense_S, rigid_trace, pick_glue
    torch.set_num_threads(a.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device(a.device)
    t0 = time.time()

    row = [r for r in json.loads(a.manifest.read_text()) if int(r['seat']) == a.seat][0]
    ref = Path(row['reference'])
    receipt = json.loads((ref / 'RESULT.json').read_text())
    d = int(receipt['dimension']); q = d + 6
    cache = dict(np.load(row['trace_cache'], allow_pickle=False))
    meta = json.loads((Path(row['trace_cache']).parent / 'INPUT.json').read_text())['metadata']
    n = int(meta['n'])
    ctx = compile_equi_inputs(cache, meta)
    span = rigid_span_residual(cache, n)
    if span > 1e-12:
        raise ValueError(f'RIGID_IS_NOT_THE_TRACE_PULLBACK {span:.3e}')
    kind = np.asarray(cache['kind']).astype(np.int64)
    pos = ctx['pos']; count = ctx['count']
    if 3 * count != q:
        raise ValueError('DIMENSION_BINDING')
    axis, pair, n_lo, n_hi = pick_glue(kind, pos)
    report = dict(seat=a.seat, d=d, q=q, n=n, coordinates=count, kind1=int((kind == 1).sum()),
                  box_only=bool(ctx['box_only']), glue_axis=int(axis), shared_coordinates=len(pair),
                  pred=str(a.pred))

    # ---- assembly, identical to assemble_two.py -------------------------------------------
    gid_A = np.arange(count); gid_B = np.empty(count, dtype=np.int64); nxt = count
    for i in range(count):
        if i in pair:
            gid_B[i] = pair[i]
        else:
            gid_B[i] = nxt; nxt += 1
    N_nodes = nxt; N = 3 * N_nodes
    report.update(assembled_coordinates=int(N_nodes), assembled_dofs=int(N))
    dof_A = torch.as_tensor((gid_A[:, None] * 3 + np.arange(3)).ravel(), device=dev)
    dof_B = torch.as_tensor((gid_B[:, None] * 3 + np.arange(3)).ravel(), device=dev)
    offset = np.zeros(3); offset[axis] = 1.0
    P = np.zeros((N_nodes, 3)); P[gid_A] = pos; P[gid_B] = pos + offset
    rb_A = rigid_trace(cache, n, np.zeros(3)); rb_B = rigid_trace(cache, n, offset)
    Nrb = np.zeros((N, 6))
    Nrb[(gid_A[:, None] * 3 + np.arange(3)).ravel()] = rb_A
    Nrb[(gid_B[:, None] * 3 + np.arange(3)).ravel()] = rb_B
    Nrb = torch.as_tensor(Nrb, dtype=F64, device=dev)
    Nrb, _ = torch.linalg.qr(Nrb)
    Pt = torch.as_tensor(P, dtype=F64, device=dev)
    kindN = np.zeros(N_nodes, dtype=np.int64); kindN[gid_A] = kind; kindN[gid_B] = kind
    endL = np.flatnonzero((kindN == 0) & (P[:, axis] <= P[:, axis].min() + 1e-9))
    endR = np.flatnonzero((kindN == 0) & (P[:, axis] >= P[:, axis].max() - 1e-9))
    tv = [j for j in range(3) if j != axis]
    loads = {}
    for name, comp in ((f'axial_{"xyz"[axis]}', axis), (f'shear_{"xyz"[tv[0]]}', tv[0]),
                       (f'shear_{"xyz"[tv[1]]}', tv[1])):
        f = torch.zeros(N, dtype=F64, device=dev)
        f[torch.as_tensor(endL * 3 + comp, device=dev)] = -1.0 / len(endL)
        f[torch.as_tensor(endR * 3 + comp, device=dev)] = +1.0 / len(endR)
        loads[name] = f
    f = torch.zeros(N, dtype=F64, device=dev)
    wl = Pt[torch.as_tensor(endL, device=dev), tv[0]]; wl = wl - wl.mean()
    wr = Pt[torch.as_tensor(endR, device=dev), tv[0]]; wr = wr - wr.mean()
    f[torch.as_tensor(endL * 3 + axis, device=dev)] = -wl / wl.abs().sum().clamp_min(1e-300)
    f[torch.as_tensor(endR * 3 + axis, device=dev)] = wr / wr.abs().sum().clamp_min(1e-300)
    loads[f'bending_{"xyz"[axis]}{"xyz"[tv[0]]}'] = f
    for k in loads:
        loads[k] = loads[k] - Nrb @ (Nrb.T @ loads[k])
        loads[k] = loads[k] / loads[k].norm()
    names = list(loads)
    report['load_names'] = names

    quotient = RigidQuotient(torch.from_numpy(np.asarray(cache['rigid'], dtype=np.float64)).to(dev),
                             torch.from_numpy(np.asarray(cache['order'])).to(dev))

    # ---- exact solve: the true displacement, hence the load-energy weights ----------------
    Rstar = unpack(ref / 'R_UPPER.npy', d, dev)
    Astar = Rstar.T @ Rstar; Astar = .5 * (Astar + Astar.T)
    Sstar = dense_S(Astar, quotient)
    del Astar; _free(dev)
    K = torch.zeros((N, N), dtype=F64, device=dev)
    K[dof_A.unsqueeze(1), dof_A.unsqueeze(0)] += Sstar
    K[dof_B.unsqueeze(1), dof_B.unsqueeze(0)] += Sstar
    del Sstar; _free(dev)
    scale = float(K.diagonal().abs().mean())
    K.addmm_(Nrb, Nrb.T, alpha=scale)
    root = torch.linalg.cholesky(K)
    F = torch.stack([loads[k] for k in names], dim=1)
    U = torch.cholesky_solve(F, root)
    del root; _free(dev)
    resid = float(((K @ U - F).norm(dim=0) / F.norm(dim=0)).max())
    cexact = (F * U).sum(dim=0)
    del K, F; _free(dev)
    report['solve_residual'] = resid
    report['c_exact'] = [float(v) for v in cexact]

    xA = quotient(U[dof_A]); xB = quotient(U[dof_B])              # d x nloads, quotient traces
    del U; _free(dev)
    YA = Rstar @ xA; YB = Rstar @ xB                              # whitened: energy = ||Y||^2
    eA = (YA * YA).sum(dim=0); eB = (YB * YB).sum(dim=0)
    report['cell_energy_fraction_A'] = [float(v) for v in (eA / (eA + eB))]
    report['energy_vs_compliance'] = [float(v) for v in ((eA + eB) / cexact)]

    # ---- A*'s own spectrum, the soft subspace, and its localization ------------------------
    Astar = Rstar.T @ Rstar; Astar = .5 * (Astar + Astar.T)
    lam, Pv = torch.linalg.eigh(Astar)                            # ascending
    del Astar; _free(dev)
    report['A_eig_min'] = float(lam.min()); report['A_eig_max'] = float(lam.max())
    report['A_cond'] = float(lam.max() / lam.min().clamp_min(1e-300))
    ks = sorted(int(v) for v in a.softk.split(','))
    kmax = min(max(ks), d)
    ca = Pv.T @ xA; cb = Pv.T @ xB
    en = lam[:, None] * (ca * ca + cb * cb)
    en = en / en.sum(dim=0, keepdim=True)
    soft = {str(k): dict(dim=int(k), energy_share=[float(v) for v in en[:k].sum(dim=0)]) for k in ks}
    del ca, cb, en; _free(dev)
    Plow = Pv[:, :32].contiguous()
    lift = quotient.lift(Plow)                                     # q x 32
    nodal = lift.reshape(count, 3, 32).pow(2).sum(dim=1)
    nodal = nodal / nodal.sum(dim=0, keepdim=True)
    part = 1.0 / nodal.pow(2).sum(dim=0)
    report['soft32_participation_ratio'] = [float(v) for v in part]
    report['soft32_participation_min'] = float(part.min())
    report['soft32_top_coordinate_share'] = float(nodal.max())
    del Plow, lift, nodal, part; _free(dev)
    Ysoft = Rstar @ Pv[:, :kmax].contiguous()                      # d x kmax, whitened soft basis
    lam_soft = lam[:kmax].clone()
    del Pv, lam; _free(dev)

    # ---- the pencil ----------------------------------------------------------------------
    Rh = unpack(a.pred, d, dev)
    Ahat = Rh.T @ Rh; Ahat = .5 * (Ahat + Ahat.T); del Rh; _free(dev)
    W = torch.linalg.solve_triangular(Rstar.T, Ahat, upper=False, left=True)
    del Ahat; _free(dev)
    Z = torch.linalg.solve_triangular(Rstar.T, W.T.contiguous(), upper=False, left=True).T.contiguous()
    del W, Rstar; _free(dev)
    Z = .5 * (Z + Z.T)
    mu, V = torch.linalg.eigh(Z)
    del Z; _free(dev)
    mu = mu.clamp_min(1e-300)
    g = float(torch.maximum(mu, 1.0 / mu).max())
    report.update(g=g, mu_min=float(mu.min()), mu_max=float(mu.max()),
                  frac_within_3pct=float(((mu - 1).abs() <= 0.03).double().mean()))

    C = V.T @ Ysoft                                                # d x kmax
    del Ysoft; _free(dev)
    cA = V.T @ YA; cB = V.T @ YB
    del V, YA, YB; _free(dev)
    wtot = cA * cA + cB * cB
    wtot = wtot / wtot.sum(dim=0, keepdim=True)
    imax = int(torch.argmax(mu)); imin = int(torch.argmin(mu))
    report['mu_max_index'] = imax; report['mu_min_index'] = imin
    report['w_at_mu_max'] = [float(v) for v in wtot[imax]]
    report['w_at_mu_min'] = [float(v) for v in wtot[imin]]
    mean_mu = (wtot * mu[:, None]).sum(dim=0)
    mean_inv = (wtot / mu[:, None]).sum(dim=0)
    report['mean_mu_w'] = [float(v) for v in mean_mu]
    report['mean_invmu_w'] = [float(v) for v in mean_inv]
    report['first_order_predictor'] = [float(-(v - 1)) for v in mean_mu]
    bad = (mu - 1).abs() > 0.03
    report['energy_share_outside_3pct'] = [float(v) for v in wtot[bad].sum(dim=0)]
    order = torch.argsort(wtot.mean(dim=1), descending=True)[:8]
    report['top_energy_modes'] = [dict(index=int(i), mu=float(mu[i]), w=float(wtot[i].mean())) for i in order]
    share = wtot.mean(dim=1)
    for thr in (1e-3, 1e-4):
        live = share >= thr
        if bool(live.any()):
            lm = mu[live]
            report[f'g_live_{thr:g}'] = float(torch.maximum(lm, 1.0 / lm).max())
            report[f'n_live_{thr:g}'] = int(live.sum())

    # g_soft(k): generalized eigenvalues of (Pk^T Ahat Pk, Pk^T A* Pk) = (Ck^T diag(mu) Ck,
    # diag(lam[:k])), since Pk^T A* Pk is diagonal in A*'s own eigenbasis.
    for k in ks:
        kk = min(k, kmax)
        Ck = C[:, :kk]
        Bk = Ck.T @ (mu[:, None] * Ck)
        Bk = .5 * (Bk + Bk.T)
        inv = lam_soft[:kk].clamp_min(1e-300).rsqrt()
        Hk = inv[:, None] * Bk * inv[None, :]
        ev = torch.linalg.eigvalsh(.5 * (Hk + Hk.T)).clamp_min(1e-300)
        soft[str(k)].update(g_soft=float(torch.maximum(ev, 1.0 / ev).max()),
                            mu_min=float(ev.min()), mu_max=float(ev.max()))
        del Ck, Bk, Hk, ev
    report['soft'] = soft

    if a.accept is not None and a.accept.exists():
        acc = json.loads(a.accept.read_text())
        report['measured_signed_compliance_rel'] = {
            nm: acc['predicted'][nm]['compliance'] / acc['exact'][nm]['compliance'] - 1.0
            for nm in names if nm in acc['exact']}

    np.savez_compressed(a.output / 'SPECTRUM.npz', mu=mu.cpu().numpy(),
                        weights=wtot.cpu().numpy().astype(np.float32))
    report['seconds'] = time.time() - t0
    report['status'] = 'SOFT_DIAG_COMPLETE'
    (a.output / 'SOFT_DIAG.json').write_text(json.dumps(report, indent=1))
    keys = ('seat', 'g', 'mu_min', 'mu_max', 'w_at_mu_max', 'w_at_mu_min', 'first_order_predictor',
            'measured_signed_compliance_rel', 'energy_share_outside_3pct', 'soft32_participation_min',
            'soft', 'seconds')
    print(json.dumps({k: report[k] for k in keys if k in report}, indent=1), flush=True)


if __name__ == '__main__':
    main()
