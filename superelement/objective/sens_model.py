#!/usr/bin/env python3
"""What compliance and sensitivity error does the TRAINED model actually cause?

E1 measured the sensitivity error of a *constructed* module at eps_op 0.152 (0.37%),
and E1b measured Codex's modules (41-48%).  Everything between is a two-point
extrapolation, and the whole "is line 1/3 alive" argument currently rests on it.

This closes that gap by running the same two-module assembly and the same exact
adjoint on the factors the two stage-1 arms actually predicted, so the number is
measured rather than interpolated:

    chol arm   A_hat = R_hat^T R_hat   from R_PRED_UPPER.npy   (eps_op 215.72)
    sqrt arm   A_hat = M_hat M_hat     from M_PRED_UPPER.npy   (eps_op  48.03)

Design variable: a uniform stiffness multiplier rho_m per module, the standard
lattice-cell knob, with the exact adjoint dc/drho_m = -u_m^T S_m u_m.  Both arms are
compared against the same exact operator on the same loads.  No repair anywhere.
"""
import argparse, json, sys, time
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


def unpack_upper(path, d):
    p = np.load(path, mmap_mode='r', allow_pickle=False)
    if p.shape != (d * (d + 1) // 2,) or p.dtype != np.float64:
        raise ValueError(f'FP64_UPPER_REQUIRED {path} {p.shape} {p.dtype}')
    host = np.zeros((d, d), dtype=np.float64)
    off = 0
    for row in range(d):
        host[row, row:] = p[off:off + d - row]
        off += d - row
    return torch.from_numpy(host).to(DEV)


def dense_S(A, quotient):
    left = quotient.lift(A.T.contiguous())
    S = quotient.lift(left.T.contiguous())
    return 0.5 * (S + S.T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seat', type=int, default=253)
    ap.add_argument('--chol-factor', type=Path, required=True)
    ap.add_argument('--sqrt-factor', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
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
    report = dict(seat=a.seat, d=int(d), q=int(q), trace_nodes=int(nn),
                  design='uniform stiffness multiplier per module; dc/drho_m = -u_m^T S_m u_m',
                  scope='two glued cells; the operators are what the stage-1 arms predicted')
    print(f'seat {a.seat}  d {d}  q {q}  trace nodes {nn}', flush=True)

    xhi, xlo = int(ijk[:, 0].max()), int(ijk[:, 0].min())
    key = {(int(r[0]), int(r[1]), int(r[2])): i for i, r in enumerate(ijk)}
    shift = xhi - xlo
    gid_A = np.arange(nn)
    gid_B = np.empty(nn, dtype=np.int64)
    nxt = nn
    shared = 0
    for i, r in enumerate(ijk):
        if int(r[0]) == xlo and (xhi, int(r[1]), int(r[2])) in key:
            gid_B[i] = key[(xhi, int(r[1]), int(r[2]))]
            shared += 1
        else:
            gid_B[i] = nxt
            nxt += 1
    N_nodes = nxt
    N = 3 * N_nodes
    report.update(shared_nodes=int(shared), assembled_dofs=int(N))
    print(f'glued on {shared} shared nodes -> {N_nodes} nodes, {N} assembled trace dofs', flush=True)
    dof_A = torch.as_tensor((gid_A[:, None] * 3 + np.arange(3)).ravel(), device=DEV)
    dof_B = torch.as_tensor((gid_B[:, None] * 3 + np.arange(3)).ravel(), device=DEV)

    xyz = np.zeros((N_nodes, 3))
    xyz[gid_A] = ijk
    xyz[gid_B] = ijk + np.array([shift, 0, 0])
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
        f[endL * 3 + comp] = -1.0 / len(endL)
        f[endR * 3 + comp] = +1.0 / len(endR)
        loads[name] = f
    f = torch.zeros(N, dtype=F64, device=DEV)
    wl = P[endL, 1] - P[endL, 1].mean()
    wr = P[endR, 1] - P[endR, 1].mean()
    f[endL * 3 + 0] = -wl / wl.abs().sum()
    f[endR * 3 + 0] = wr / wr.abs().sum()
    loads['bending_xy'] = f
    for k in loads:
        loads[k] = loads[k] - Nrb @ (Nrb.T @ loads[k])
        loads[k] = loads[k] / loads[k].norm()

    def solve_with(Ahat):
        S = dense_S(Ahat, quotient)
        K = torch.zeros(N, N, dtype=F64, device=DEV)
        K[dof_A.unsqueeze(1), dof_A.unsqueeze(0)] += S
        K[dof_B.unsqueeze(1), dof_B.unsqueeze(0)] += S
        K = 0.5 * (K + K.T)
        sc = float(K.diagonal().mean())
        # rank-6 rigid regulariser added in place: K + sc Nrb Nrb^T without a
        # second and third N x N temporary, which would not fit beside the
        # generalisation run currently on this GPU.
        K.addmm_(Nrb, Nrb.T, alpha=sc)
        L = torch.linalg.cholesky(K)
        del K
        torch.cuda.empty_cache()
        out = {}
        for name, fv in loads.items():
            u = torch.cholesky_solve(fv.unsqueeze(1), L).squeeze(1)
            u = u - Nrb @ (Nrb.T @ u)
            sens = {}
            for mod, dofm in (('A', dof_A), ('B', dof_B)):
                um = u[dofm]
                sens[mod] = float(-(um @ (S @ um)))
            out[name] = dict(u=u, compliance=float(fv @ u), sens=sens)
        del L, S
        torch.cuda.empty_cache()
        return out

    def eps_op_of(Ahat):
        X = torch.linalg.solve_triangular(Rstar, Ahat, upper=True, left=False)
        W = torch.linalg.solve_triangular(Rstar, X.T.contiguous(), upper=True, left=False).T.contiguous()
        del X
        W = 0.5 * (W + W.T)
        mu = torch.linalg.eigvalsh(W)
        del W
        torch.cuda.empty_cache()
        return float((mu - 1).abs().max()), float(mu.min()), float(mu.max())

    print('\nexact arm...', flush=True)
    ex = solve_with(A)
    report['exact'] = {k: dict(compliance=v['compliance'], sens=v['sens']) for k, v in ex.items()}
    identity = {}
    for k, v in ex.items():
        tot = v['sens']['A'] + v['sens']['B']
        identity[k] = abs(tot + v['compliance']) / abs(v['compliance'])
        print('   %-16s c %.8e   dc/drho_A %.6e   dc/drho_B %.6e   identity %.3e'
              % (k, v['compliance'], v['sens']['A'], v['sens']['B'], identity[k]), flush=True)
    report['adjoint_identity_relative'] = identity
    if max(identity.values()) > 1e-8:
        raise ValueError('ADJOINT_IDENTITY_FAILED')
    del A
    torch.cuda.empty_cache()

    arms = {}
    for head, path in (('chol', a.chol_factor), ('sqrt', a.sqrt_factor)):
        if not path.exists():
            print(f'   skipping {head}: {path} missing', flush=True)
            continue
        U = unpack_upper(path, d)
        if head == 'chol':
            Ahat = U.T @ U
        else:
            M = U + torch.triu(U, 1).T
            del U
            Ahat = M @ M
            del M
        Ahat = 0.5 * (Ahat + Ahat.T)
        torch.cuda.empty_cache()
        eps, mn, mx = eps_op_of(Ahat)
        ap = solve_with(Ahat)
        del Ahat
        torch.cuda.empty_cache()
        row = dict(head=head, factor=str(path), eps_op=eps, mu_min=mn, mu_max=mx, loads={})
        print(f'\n   {head}: eps_op {eps:.6g}  mu_min {mn:.4g}  mu_max {mx:.6g}', flush=True)
        for name in loads:
            ue, ce = ex[name]['u'], ex[name]['compliance']
            uh, ch = ap[name]['u'], ap[name]['compliance']
            du = float((uh - ue).norm() / ue.norm())
            dc = abs(ch - ce) / abs(ce)
            ds = {m: abs(ap[name]['sens'][m] - ex[name]['sens'][m]) / abs(ex[name]['sens'][m])
                  for m in ('A', 'B')}
            row['loads'][name] = dict(displacement_rel=du, compliance_rel=dc, sensitivity_rel=ds,
                                      compliance_predicted=ch, compliance_exact=ce)
            print('      %-16s displ %.3e  compliance %.3e  dc/drho_A %.3e  dc/drho_B %.3e'
                  % (name, du, dc, ds['A'], ds['B']), flush=True)
        row['worst_sensitivity_rel'] = max(max(v['sensitivity_rel'].values()) for v in row['loads'].values())
        row['worst_compliance_rel'] = max(v['compliance_rel'] for v in row['loads'].values())
        print('      -> worst compliance %.3e   worst sensitivity %.3e   at eps_op %.6g'
              % (row['worst_compliance_rel'], row['worst_sensitivity_rel'], eps), flush=True)
        arms[head] = row
        report['arms'] = arms
        report['seconds'] = time.time() - t0
        (a.output / 'SENS_MODEL.json').write_text(json.dumps(report, indent=1))

    report['status'] = 'TRAINED_MODEL_SENSITIVITY_COMPLETE'
    report['seconds'] = time.time() - t0
    (a.output / 'SENS_MODEL.json').write_text(json.dumps(report, indent=1))
    print('\ndone (%.0f s)' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
