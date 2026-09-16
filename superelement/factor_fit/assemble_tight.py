#!/usr/bin/env python3
"""Tighter tolerances for the two-cell assembly, to settle one disputed reading.

The first run's compliance column was non-monotone (1.3e-3, 3.3e-3, 2.5e-3, 1.0e-3 for compression)
and I read that as a construction floor near 1e-3.  The audit rightly says that is not shown:
error matrices at different tolerances can change direction and sign, and compliance need not
be monotone in the tolerance.  A floor is a claim about the exact-limit behaviour, so test the
limit: at module eps_op of 3.5e-3 and 8e-4, does the compliance error keep falling below 1e-3,
or does it stall?  If it keeps falling there is no floor and the first reading is withdrawn.

Original header follows.

What module accuracy does the ASSEMBLED structure actually need?

The +-3%-per-mode gate was asserted, never derived.  The deliverable is an assembled lattice under
realistic loads, so the tolerance should come from there.  BRIEFING section 3 records that
assembly of EXACT module Schur complements reproduces a direct FE solve to 1e-12, i.e. condensation
itself composes; it says nothing about how a module error propagates, which is the open question.

Construction.  Two copies of seat 0328 side by side in x.  The trace lives entirely on the outer
faces (4266 nodes, 100% on some face), the -x face has 696 nodes and the +x face 716, and the -x
(y,z) set is a strict subset of the +x one -- so the geometry is not periodic across its own
boundary and 20 of the +x nodes have no partner.  They are simply left free: the result is a
genuine two-body elasticity problem bonded on 696 shared nodes, which is what an error-propagation
question needs.  It is not a reproduction of any particular lattice, and is not claimed to be.

Both arms use S = B^T A B with the SAME B, so every difference is attributable to Ahat vs A and
none to the quotient map.  Loads are self-equilibrated face tractions (compression, shear, bending)
applied to the outer faces only.  The assembled operator keeps a 6-dimensional rigid null space,
which is removed by projection rather than by clamping nodes, so no artificial stiffness is added.
"""
import json, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, '/root/cutfem_neural_a_20260910/source_14301bc56')
sys.path.insert(0, '/root/cutfem_neural_a_20260910/superelement_v0')
sys.path.insert(0, '/root/autodl-tmp/NEURAL_SCHUR')
import v1_scaled as V
from stage_cutfem_neural_a.elimination_reference import load_upper_factor

DEV = V.DEV
F64 = torch.float64


def dense_S(A, quotient, q):
    """S = B^T A B on the full trace, via the quotient's own lift."""
    d = A.shape[0]
    left = quotient.lift(A.T.contiguous())      # (q, d) = (B^T A^T)  -> A symmetric so B^T A
    S = quotient.lift(left.T.contiguous())      # (q, q) = B^T A B
    return 0.5 * (S + S.T)


def main():
    t0 = time.time()
    recs = json.load(open('/root/autodl-tmp/CUTFEM_SUPERELEMENT_LABELS/V1_LABELS.json'))
    rec = [r for r in recs if int(r['seat']) == 328][0]
    rec.setdefault('reference', '/root/autodl-tmp/CUTFEM_SUPERELEMENT_LABELS/REFERENCE_0328')
    label = V.Label(rec, 0.2, 0.03, 10.0)
    g = label.to_gpu(need_A=False, need_Z=False, z_dtype=torch.float32)
    d, q = label.d, label.q
    quotient = g['data'].quotient
    R = load_upper_factor(Path(rec['reference']) / 'R_UPPER.npy', d, DEV)
    A = R.T @ R
    ijk = np.asarray(label.ijk)
    nn = len(ijk)
    print('seat 328   d %d   q %d   trace nodes %d' % (d, q, nn), flush=True)

    # ---- glue two copies along x ------------------------------------------------------------
    xhi, xlo = int(ijk[:, 0].max()), int(ijk[:, 0].min())
    key = {(int(r[0]), int(r[1]), int(r[2])): i for i, r in enumerate(ijk)}
    # cell A occupies x in [xlo, xhi]; cell B is A shifted by (xhi - xlo)
    shift = xhi - xlo
    gid_A = np.arange(nn)
    gid_B = np.empty(nn, dtype=np.int64)
    nxt = nn
    shared = 0
    for i, r in enumerate(ijk):
        src = (int(r[0]) + shift, int(r[1]), int(r[2]))
        # a node of B coincides with a node of A when B's shifted position is an A node
        j = key.get(src)
        if int(r[0]) == xlo and (xhi, int(r[1]), int(r[2])) in key:
            gid_B[i] = key[(xhi, int(r[1]), int(r[2]))]; shared += 1
        else:
            gid_B[i] = nxt; nxt += 1
    N_nodes = nxt
    N = 3 * N_nodes
    print('glued on %d shared nodes -> %d nodes, %d assembled trace dofs' % (shared, N_nodes, N), flush=True)
    dof_A = torch.as_tensor((gid_A[:, None] * 3 + np.arange(3)).ravel(), device=DEV)
    dof_B = torch.as_tensor((gid_B[:, None] * 3 + np.arange(3)).ravel(), device=DEV)

    # global node coordinates, for the loads and the rigid null space
    xyz = np.zeros((N_nodes, 3))
    xyz[gid_A] = ijk
    xyz[gid_B] = ijk + np.array([shift, 0, 0])
    xyz = xyz / 64.0
    P = torch.as_tensor(xyz, dtype=F64, device=DEV)

    # rigid null space of the ASSEMBLED body: 3 translations + 3 rotations
    Nrb = torch.zeros(N, 6, dtype=F64, device=DEV)
    for a in range(3):
        Nrb[a::3, a] = 1.0
    c = P - P.mean(0)
    Nrb[0::3, 3] = -c[:, 1]; Nrb[1::3, 3] = c[:, 0]
    Nrb[1::3, 4] = -c[:, 2]; Nrb[2::3, 4] = c[:, 1]
    Nrb[0::3, 5] = c[:, 2];  Nrb[2::3, 5] = -c[:, 0]
    Nrb, _ = torch.linalg.qr(Nrb)

    # ---- loads: self-equilibrated tractions on the two end faces ----------------------------
    endL = torch.as_tensor(np.flatnonzero(xyz[:, 0] <= xyz[:, 0].min() + 1e-9), device=DEV)
    endR = torch.as_tensor(np.flatnonzero(xyz[:, 0] >= xyz[:, 0].max() - 1e-9), device=DEV)
    loads = {}
    for name, comp in (('compression_x', 0), ('shear_y', 1), ('shear_z', 2)):
        f = torch.zeros(N, dtype=F64, device=DEV)
        f[endL * 3 + comp] = -1.0 / len(endL)
        f[endR * 3 + comp] = +1.0 / len(endR)
        loads[name] = f
    f = torch.zeros(N, dtype=F64, device=DEV)     # bending: linear in y on the two ends
    wl = (P[endL, 1] - P[endL, 1].mean()); wr = (P[endR, 1] - P[endR, 1].mean())
    f[endL * 3 + 0] = -wl / wl.abs().sum(); f[endR * 3 + 0] = wr / wr.abs().sum()
    loads['bending_xy'] = f
    for k in loads:                                # project out rigid components
        loads[k] = loads[k] - Nrb @ (Nrb.T @ loads[k])
        loads[k] = loads[k] / loads[k].norm()

    def assemble_and_solve(Ahat, tag):
        S = dense_S(Ahat, quotient, q)
        K = torch.zeros(N, N, dtype=F64, device=DEV)
        K[dof_A.unsqueeze(1), dof_A.unsqueeze(0)] += S
        K[dof_B.unsqueeze(1), dof_B.unsqueeze(0)] += S
        del S; torch.cuda.empty_cache()
        K = 0.5 * (K + K.T)
        sc = float(K.diagonal().mean())
        K = K + sc * (Nrb @ Nrb.T)                 # regularise ONLY the rigid subspace
        L = torch.linalg.cholesky(K)
        del K; torch.cuda.empty_cache()
        out = {}
        for name, fv in loads.items():
            u = torch.cholesky_solve(fv.unsqueeze(1), L).squeeze(1)
            u = u - Nrb @ (Nrb.T @ u)
            out[name] = (u, float(fv @ u))
        del L; torch.cuda.empty_cache()
        return out

    print('\nassembling exact arm...', flush=True)
    ex = assemble_and_solve(A, 'exact')
    for k, (u, cmp_) in ex.items():
        print('   %-16s compliance %.8e' % (k, cmp_), flush=True)

    # ---- the approximate modules: hierarchical factor construction at several tolerances ----
    from hmat_factor_lib import build_factor_approximations
    Rinv = torch.linalg.solve_triangular(R, torch.eye(d, dtype=F64, device=DEV), upper=True)
    results = {}
    for tol, Rhat, params in build_factor_approximations(label, R, DEV, tols=(3e-4, 1e-4, 3e-5, 1e-5)):
        X = Rhat @ Rinv
        H = X.T @ X; del X
        mu = torch.linalg.eigvalsh(0.5 * (H + H.T)); del H; torch.cuda.empty_cache()
        eps_op = float((mu - 1).abs().max()); del mu
        Ahat = Rhat.T @ Rhat
        ap = assemble_and_solve(Ahat, 'tol %g' % tol)
        row = dict(tol=tol, eps_op=eps_op, params=int(params))
        print('\n   tol %.0e   module eps_op %.4e   params %d (%.2f%% of dense)'
              % (tol, eps_op, params, 100*params/(d*(d+1)/2)), flush=True)
        for name, (u, cmp_) in ap.items():
            ue, ce = ex[name]
            du = float((u - ue).norm() / ue.norm())
            dc = abs(cmp_ - ce) / abs(ce)
            row[name] = dict(displacement_rel=du, compliance_rel=dc)
            print('      %-16s  displacement %.4e   compliance %.4e' % (name, du, dc), flush=True)
        results[str(tol)] = row
        del Ahat, Rhat; torch.cuda.empty_cache()
    Path('/root/autodl-tmp/NEURAL_SCHUR/ASSEMBLY_TIGHT.json').write_text(json.dumps(
        dict(seat=328, d=int(d), q=int(q), assembled_dofs=int(N), shared_nodes=int(shared),
             exact_compliance={k: v[1] for k, v in ex.items()}, rows=results,
             seconds=time.time()-t0), indent=1))
    print('\nwritten ASSEMBLY_TIGHT.json (%.0f s)' % (time.time()-t0), flush=True)


if __name__ == '__main__':
    main()
