#!/usr/bin/env python3
"""E1 - the missing experiment: what module accuracy do the OPTIMISATION SENSITIVITIES need?

ASSEMBLY_TOLERANCE_20260916.md established that on a two-cell strip the compliance
error stays <= 0.33% even for a module that fails +-10%, and named the gap:

  "Topology optimisation uses d(compliance)/d(design), not compliance alone, and the
   derivative is FIRST-order in the operator error. The adjoint on this same strip is
   the missing experiment and it is what should set the gate."

The user has now set the contract at the task level - only the assembled lattice's
compliance and optimisation sensitivities must be right - so this experiment sets the
gate, and everything else is measured against it.

Design variables: a uniform stiffness multiplier rho_m on each module (the standard
SIMP/homogenisation knob for a lattice cell). For K(rho) = sum_m rho_m P_m^T S_m P_m
and compliance c = f^T u with K u = f,

        dc/drho_m = - u_m^T S_m u_m      (exact adjoint; K symmetric)

An optimiser only ever has the approximate model, so it computes
-uhat_m^T Shat_m uhat_m and we compare that against the exact value. This is the error
the optimiser actually suffers, not a linearisation of it.

Reported per (module, load): the relative sensitivity error, alongside the
displacement error (first order) and the compliance error (second order) so the three
can be read on one axis.
"""
import json, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, '/root/cutfem_neural_a_20260910/source_14301bc56')
sys.path.insert(0, '/root/cutfem_neural_a_20260910/superelement_v0')
sys.path.insert(0, '/root/autodl-tmp/NEURAL_SCHUR')
import v1_scaled as V
from stage_cutfem_neural_a.elimination_reference import load_upper_factor

DEV = V.DEV; F64 = torch.float64
OUT = Path('/root/autodl-tmp/CLAUDE_SENSNET_20260917'); OUT.mkdir(exist_ok=True)


def dense_S(A, quotient, q):
    left = quotient.lift(A.T.contiguous())
    S = quotient.lift(left.T.contiguous())
    return 0.5 * (S + S.T)


def main():
    t0 = time.time()
    recs = json.load(open('/root/autodl-tmp/CUTFEM_SUPERELEMENT_LABELS/V1_LABELS.json'))
    rec = [r for r in recs if int(r['seat']) == 253][0]
    rec.setdefault('reference', '/root/autodl-tmp/CUTFEM_SUPERELEMENT_LABELS/REFERENCE_0253')
    label = V.Label(rec, 0.2, 0.03, 10.0)
    g = label.to_gpu(need_A=False, need_Z=False, z_dtype=torch.float32)
    d, q = label.d, label.q
    quotient = g['data'].quotient
    R = load_upper_factor(Path(rec['reference']) / 'R_UPPER.npy', d, DEV)
    A = R.T @ R
    ijk = np.asarray(label.ijk); nn = len(ijk)
    print('seat 253   d %d   q %d   trace nodes %d' % (d, q, nn), flush=True)

    xhi, xlo = int(ijk[:, 0].max()), int(ijk[:, 0].min())
    key = {(int(r[0]), int(r[1]), int(r[2])): i for i, r in enumerate(ijk)}
    shift = xhi - xlo
    gid_A = np.arange(nn); gid_B = np.empty(nn, dtype=np.int64); nxt = nn; shared = 0
    for i, r in enumerate(ijk):
        if int(r[0]) == xlo and (xhi, int(r[1]), int(r[2])) in key:
            gid_B[i] = key[(xhi, int(r[1]), int(r[2]))]; shared += 1
        else:
            gid_B[i] = nxt; nxt += 1
    N_nodes = nxt; N = 3 * N_nodes
    print('glued on %d shared nodes -> %d nodes, %d assembled trace dofs' % (shared, N_nodes, N), flush=True)
    dof_A = torch.as_tensor((gid_A[:, None] * 3 + np.arange(3)).ravel(), device=DEV)
    dof_B = torch.as_tensor((gid_B[:, None] * 3 + np.arange(3)).ravel(), device=DEV)

    xyz = np.zeros((N_nodes, 3)); xyz[gid_A] = ijk; xyz[gid_B] = ijk + np.array([shift, 0, 0])
    xyz = xyz / 64.0
    P = torch.as_tensor(xyz, dtype=F64, device=DEV)
    Nrb = torch.zeros(N, 6, dtype=F64, device=DEV)
    for a in range(3): Nrb[a::3, a] = 1.0
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
    wl = (P[endL, 1] - P[endL, 1].mean()); wr = (P[endR, 1] - P[endR, 1].mean())
    f[endL * 3 + 0] = -wl / wl.abs().sum(); f[endR * 3 + 0] = wr / wr.abs().sum()
    loads['bending_xy'] = f
    for k in loads:
        loads[k] = loads[k] - Nrb @ (Nrb.T @ loads[k]); loads[k] = loads[k] / loads[k].norm()

    def solve_with(Ahat):
        """Assemble, solve, and take the exact adjoint sensitivity of each design variable."""
        S = dense_S(Ahat, quotient, q)
        K = torch.zeros(N, N, dtype=F64, device=DEV)
        K[dof_A.unsqueeze(1), dof_A.unsqueeze(0)] += S
        K[dof_B.unsqueeze(1), dof_B.unsqueeze(0)] += S
        K = 0.5 * (K + K.T)
        sc = float(K.diagonal().mean())
        L = torch.linalg.cholesky(K + sc * (Nrb @ Nrb.T)); del K; torch.cuda.empty_cache()
        out = {}
        for name, fv in loads.items():
            u = torch.cholesky_solve(fv.unsqueeze(1), L).squeeze(1)
            u = u - Nrb @ (Nrb.T @ u)
            # dc/drho_m = - u_m^T S_m u_m, with u_m the module's own trace restriction
            sens = {}
            for mod, dofm in (('A', dof_A), ('B', dof_B)):
                um = u[dofm]
                sens[mod] = float(-(um @ (S @ um)))
            out[name] = dict(u=u, compliance=float(fv @ u), sens=sens)
        del L, S; torch.cuda.empty_cache()
        return out

    print('\nexact arm...', flush=True)
    ex = solve_with(A)
    for k, v in ex.items():
        print('   %-16s c %.8e   dc/drho_A %.6e   dc/drho_B %.6e'
              % (k, v['compliance'], v['sens']['A'], v['sens']['B']), flush=True)
    # consistency: for a uniform multiplier on BOTH modules, sum_m dc/drho_m = -c
    for k, v in ex.items():
        tot = v['sens']['A'] + v['sens']['B']
        rel = abs(tot + v['compliance']) / abs(v['compliance'])
        print('   identity check %-16s (sum dc/drho + c)/c = %.3e %s'
              % (k, rel, 'OK' if rel < 1e-8 else 'FAIL'), flush=True)

    # The approximate modules are now CODEX'S ACTUAL PREDICTIONS at 40000 steps,
    # not a compression of the teacher. This asks the only question that matters under
    # a task-level contract: is the network already good enough for the assembled
    # compliance and the optimisation sensitivities, even at eps_op ~ 264?
    RUN = Path('/root/autodl-tmp/CUTFEM_M4_MULTI_20260917/RUN_TRAIN32_ACCEL_V1')
    Rinv = torch.linalg.solve_triangular(R, torch.eye(d, dtype=F64, device=DEV), upper=True)
    rows = {}
    def arms():
        for arm in ('baseline', 'relative'):
            p = RUN / arm / f'EVAL_040000_0253' / 'R_PRED_UPPER.npy'
            yield arm, load_upper_factor(p, d, DEV), int(d * (d + 1) // 2)
    for tol, Rhat, params in arms():
        X = Rhat @ Rinv; H = X.T @ X; del X
        mu = torch.linalg.eigvalsh(0.5 * (H + H.T)); del H; torch.cuda.empty_cache()
        eps_op = float((mu - 1).abs().max()); del mu
        ap = solve_with(Rhat.T @ Rhat)
        row = dict(arm=tol, eps_op=eps_op, params=int(params), loads={})
        print('\n   arm %s   module eps_op %.4e   numbers %d' % (tol, eps_op, params), flush=True)
        for name in loads:
            ue, ce = ex[name]['u'], ex[name]['compliance']
            uh, ch = ap[name]['u'], ap[name]['compliance']
            du = float((uh - ue).norm() / ue.norm()); dc = abs(ch - ce) / abs(ce)
            ds = {m: abs(ap[name]['sens'][m] - ex[name]['sens'][m]) / abs(ex[name]['sens'][m])
                  for m in ('A', 'B')}
            row['loads'][name] = dict(displacement_rel=du, compliance_rel=dc, sensitivity_rel=ds)
            print('      %-16s  displ %.3e   compliance %.3e   dc/drho_A %.3e   dc/drho_B %.3e'
                  % (name, du, dc, ds['A'], ds['B']), flush=True)
        worst = max(max(v['sensitivity_rel'].values()) for v in row['loads'].values())
        row['worst_sensitivity_rel'] = worst
        print('      -> worst sensitivity error %.3e at module eps_op %.4e' % (worst, eps_op), flush=True)
        rows[str(tol)] = row
        (OUT / 'RESULT.json').write_text(json.dumps(dict(
            seat=253, d=int(d), q=int(q), assembled_dofs=int(N), shared_nodes=int(shared),
            design='uniform stiffness multiplier per module; dc/drho_m = -u_m^T S_m u_m',
            modules='Codex RUN_TRAIN32_ACCEL_V1 predicted factors at 40000 steps, seat 0253',
            exact={k: dict(compliance=v['compliance'], sens=v['sens']) for k, v in ex.items()},
            rows=rows, seconds=time.time() - t0), indent=1))
        del Rhat; torch.cuda.empty_cache()
    print('\ndone (%.0f s)' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
