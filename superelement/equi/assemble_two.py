"""Two glued cells, exact adjoint, for a FULL or a CUT cell: the acceptance test.

The contract is that only the assembled lattice's compliance and its design sensitivities have
to be right, so this is the measurement that counts.  `superelement/objective/sens_model.py`
does it for a full cell by gluing two copies on matching background nodes; a cut cell's trace
is not a set of nodes, so that harness cannot see it.  Here the same experiment is built from
the signed CSR:

  * a coordinate with kind 0 is one background node (nnz 1) on a box face that survived the
    macro cut, so two copies glue on it exactly as before;
  * a coordinate with kind 1 is a functional on the macro cut surface -- the lattice's free
    surface.  It is never shared, and it carries load rather than continuity;
  * the assembly's six rigid modes are the CSR pullback of the global rigid field, per cell,
    which is exact (`context.rigid_pullback`) and reduces to the nodal construction when every
    functional is one node.  That the two cells agree on every shared coordinate is checked,
    not assumed: it is the statement that the glue is consistent.

The design variable is a uniform stiffness multiplier per module, as in the original harness,
with the exact adjoint dc/drho_m = -u_m^T S_m u_m, and the adjoint identity
(sum_m dc/drho_m + c)/c is reported.

Validated by reproducing sens_model.py's numbers on seat 0253, which is box-only.

    python -m superelement.equi.assemble_two --seat 253 --mq <MQ_PRED_UPPER.npy> --output DIR
    python -m superelement.equi.assemble_two --seat 253 --a-factor <A_PRED_UPPER.npy> --output DIR
"""
from __future__ import annotations

import argparse, gc, json, sys, time
from pathlib import Path
import numpy as np
import torch

F64 = torch.float64


def unpack(path, d, device):
    p = np.load(path, mmap_mode='r', allow_pickle=False)
    if p.shape != (d * (d + 1) // 2,) or p.dtype != np.float64:
        raise ValueError(f'FP64_UPPER_REQUIRED {path} {p.shape} {p.dtype}')
    out = np.zeros((d, d), dtype=np.float64); off = 0
    for row in range(d):
        out[row, row:] = p[off:off + d - row]; off += d - row
    return torch.from_numpy(out).to(device)


def dense_S(A, quotient):
    """S = B^T A B on the full trace, from A in the quotient."""
    left = quotient.lift(A.T.contiguous())
    S = quotient.lift(left.T.contiguous())
    return .5 * (S + S.T)


def trace_operator(cache):
    from scipy import sparse
    return sparse.csr_matrix((np.asarray(cache['coefficients'], dtype=np.float64),
                              np.asarray(cache['indices']), np.asarray(cache['indptr'])),
                             shape=(len(cache['kind']), len(cache['background_nodes'])))


def rigid_trace(cache, n, offset):
    """(L (x) I3) applied to the global rigid field of this cell's background nodes, shifted."""
    top = 2 * int(n)
    nodes = np.asarray(cache['background_nodes'], dtype=np.int64)
    xyz = np.column_stack(np.unravel_index(nodes, (top + 1,) * 3)) / top + np.asarray(offset, dtype=np.float64)
    field = np.zeros((3 * len(xyz), 6))
    for d in range(3):
        field[d::3, d] = 1.0
        field[:, 3 + d] = np.cross(np.eye(3)[d], xyz).ravel()
    L = trace_operator(cache)
    out = np.zeros((3 * L.shape[0], 6))
    for c in range(3):
        out[c::3] = L @ field[c::3]
    return out


def pick_glue(kind, pos, tol=1e-9):
    """The axis whose opposite faces share the most kind-0 coordinates, and the pairing.

    Two copies of the cell are placed a unit apart along that axis, so a coordinate of copy B
    on the low face coincides with one of copy A on the high face.  Matching is on the exact
    transverse position of the single background node, which is what makes the glue exact.
    """
    best = None
    for axis in range(3):
        other = [j for j in range(3) if j != axis]
        lo = np.flatnonzero((kind == 0) & (np.abs(pos[:, axis] - 0.) <= tol))
        hi = np.flatnonzero((kind == 0) & (np.abs(pos[:, axis] - 1.) <= tol))
        if not len(lo) or not len(hi):
            continue
        key = {(round(float(pos[i, other[0]]), 12), round(float(pos[i, other[1]]), 12)): i for i in hi}
        pair = {int(i): key[(round(float(pos[i, other[0]]), 12), round(float(pos[i, other[1]]), 12))]
                for i in lo if (round(float(pos[i, other[0]]), 12), round(float(pos[i, other[1]]), 12)) in key}
        if best is None or len(pair) > len(best[1]):
            best = (axis, pair, len(lo), len(hi))
    if best is None or len(best[1]) < 3:
        raise ValueError('NO_GLUEABLE_FACE_PAIR: the cut removed every opposite face pair')
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', type=Path, default=Path('/root/autodl-tmp/CLAUDE_LABELS_20260917/V2_LABELS.json'))
    ap.add_argument('--seat', type=int, required=True)
    ap.add_argument('--mq', type=Path, default=None, help='predicted MQ_PRED_UPPER.npy (q x q, symmetric)')
    ap.add_argument('--a-factor', type=Path, default=None, help='predicted A_PRED_UPPER.npy (d x d upper Cholesky of A_hat)')
    ap.add_argument('--source', type=Path, default=Path('/root/autodl-tmp/CLAUDE_SQRTHEAD_20260917/src_v5'))
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    if (a.mq is None) == (a.a_factor is None):
        raise ValueError('GIVE_EXACTLY_ONE_OF_MQ_OR_A_FACTOR')
    a.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(a.source))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from stage_cutfem_m4.quotient import RigidQuotient
    from superelement.equi.context import compile_equi_inputs, rigid_span_residual
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
    pos = ctx['pos']
    count = ctx['count']
    if 3 * count != q:
        raise ValueError('DIMENSION_BINDING')
    axis, pair, n_lo, n_hi = pick_glue(kind, pos)
    report = dict(seat=a.seat, d=d, q=q, n=n, coordinates=count, kind1=int((kind == 1).sum()),
                  box_only=bool(ctx['box_only']), glue_axis=int(axis), low_face=int(n_lo), high_face=int(n_hi),
                  shared_coordinates=len(pair), rigid_span_residual=span,
                  design='uniform stiffness multiplier per module; dc/drho_m = -u_m^T S_m u_m',
                  scope='two copies of one cell glued on its kind-0 coordinates; kind-1 coordinates are free surface')
    print(json.dumps({k: report[k] for k in ('seat', 'q', 'coordinates', 'kind1', 'box_only', 'glue_axis',
                                             'shared_coordinates', 'rigid_span_residual')}), flush=True)

    # assembled coordinate ids: copy A keeps 0..count-1, copy B reuses A's id where they glue
    gid_A = np.arange(count)
    gid_B = np.empty(count, dtype=np.int64)
    nxt = count
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
    P = np.zeros((N_nodes, 3))
    P[gid_A] = pos
    P[gid_B] = pos + offset

    # the assembly's rigid modes, from each cell's exact CSR pullback; the two must agree
    # wherever they glue, which is the statement that the glue is consistent
    rb_A = rigid_trace(cache, n, np.zeros(3))
    rb_B = rigid_trace(cache, n, offset)
    Nrb = np.zeros((N, 6))
    Nrb[(gid_A[:, None] * 3 + np.arange(3)).ravel()] = rb_A
    shared_dofs = np.array([(pair[i], i) for i in sorted(pair)], dtype=np.int64)
    if len(shared_dofs):
        rows_a = (shared_dofs[:, 0][:, None] * 3 + np.arange(3)).ravel()
        rows_b = (shared_dofs[:, 1][:, None] * 3 + np.arange(3)).ravel()
        mismatch = np.abs(rb_A[rows_a] - rb_B[rows_b]).max() / max(np.abs(rb_A).max(), 1e-300)
        report['glue_rigid_mismatch'] = float(mismatch)
        if mismatch > 1e-12:
            raise ValueError(f'GLUE_IS_INCONSISTENT_ON_THE_RIGID_MODES {mismatch:.3e}')
    Nrb[(gid_B[:, None] * 3 + np.arange(3)).ravel()] = rb_B
    Nrb = torch.as_tensor(Nrb, dtype=F64, device=dev)
    Nrb, _ = torch.linalg.qr(Nrb)

    # loads on the extreme kind-0 coordinates along the glue axis
    Pt = torch.as_tensor(P, dtype=F64, device=dev)
    kindN = np.zeros(N_nodes, dtype=np.int64); kindN[gid_A] = kind; kindN[gid_B] = kind
    endL = np.flatnonzero((kindN == 0) & (P[:, axis] <= P[:, axis].min() + 1e-9))
    endR = np.flatnonzero((kindN == 0) & (P[:, axis] >= P[:, axis].max() - 1e-9))
    if not len(endL) or not len(endR):
        raise ValueError('NO_LOADABLE_END')
    report.update(load_left=int(len(endL)), load_right=int(len(endR)))
    tv = [j for j in range(3) if j != axis]
    loads = {}
    for name, comp in ((f'axial_{"xyz"[axis]}', axis), (f'shear_{"xyz"[tv[0]]}', tv[0]), (f'shear_{"xyz"[tv[1]]}', tv[1])):
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
    # The load has to be orthogonal to the six rigid modes or the singular system has no solution,
    # but projecting the WHOLE vector puts generalised force on the kind-1 coordinates, because Nrb
    # is non-zero there -- so the test stops being the free-cut-surface configuration it claims to
    # be.  Projecting inside the kind-0 block instead keeps the cut components exactly zero and is
    # still exactly rigid-orthogonal, since Nrb^T f = Nrb[box]^T f[box] when f vanishes off the box.
    box_dofs = torch.as_tensor(np.flatnonzero(np.repeat(kindN == 0, 3)), device=dev)
    Qb, _ = torch.linalg.qr(Nrb[box_dofs])
    if int(torch.linalg.matrix_rank(Qb)) != 6:
        raise ValueError('BOX_RESTRICTION_OF_THE_RIGID_TRACE_IS_RANK_DEFICIENT')
    cut_mask = torch.ones(N, dtype=torch.bool, device=dev); cut_mask[box_dofs] = False
    old_leak = new_leak = orthogonality = 0.
    for k in loads:
        f = loads[k]
        if bool(cut_mask.any()) and float(f[cut_mask].abs().max()) != 0.:
            raise ValueError('LOAD_WAS_NOT_BUILT_ON_KIND_0_COORDINATES')
        was = f - Nrb @ (Nrb.T @ f)                              # what the old code produced
        fb = f[box_dofs]; fb = fb - Qb @ (Qb.T @ fb)
        g = torch.zeros_like(f); g[box_dofs] = fb
        loads[k] = g / g.norm()
        if bool(cut_mask.any()):
            new_leak = max(new_leak, float(loads[k][cut_mask].abs().max()))
            old_leak = max(old_leak, float(was[cut_mask].norm() / was.norm()))
        orthogonality = max(orthogonality, float((Nrb.T @ loads[k]).abs().max()))
    report.update(load_cut_component_old_projection=old_leak, load_cut_component=new_leak,
                  load_rigid_orthogonality=orthogonality,
                  load_scope='built on kind-0 coordinates and rigid-projected INSIDE the kind-0 '
                             'block, so the cut surface carries exactly zero generalised force')

    quotient = RigidQuotient(torch.from_numpy(np.asarray(cache['rigid'], dtype=np.float64)).to(dev),
                             torch.from_numpy(np.asarray(cache['order'])).to(dev))

    def assemble(S):
        K = torch.zeros((N, N), dtype=F64, device=dev)
        K[dof_A.unsqueeze(1), dof_A.unsqueeze(0)] += S
        K[dof_B.unsqueeze(1), dof_B.unsqueeze(0)] += S
        scale = float(K.diagonal().abs().mean())
        K.addmm_(Nrb, Nrb.T, alpha=scale)                   # regularise the six rigid modes only
        # S is symmetrised and the scatter-add is symmetric, as is the rank-6 update, so K is
        # exactly symmetric: symmetrising would cost another N x N buffer for nothing.
        return K, scale

    def solve(S):
        K, scale = assemble(S)
        asym = float((K - K.T).abs().max())
        if asym != 0.0:
            raise ValueError(f'ASSEMBLED_OPERATOR_NOT_SYMMETRIC {asym:.3e}')
        root = torch.linalg.cholesky(K)
        out = {}
        eps = float(torch.finfo(F64).eps)
        for name, f in loads.items():
            u = torch.cholesky_solve(f[:, None], root)[:, 0]
            res = float((K @ u - f).norm() / f.norm())
            # This is a DIRECT solve, so `res` is its backward error, about kappa(K) * eps -- a
            # conditioning readout, not a convergence failure, and no stopping rule applies to it.
            # One step of iterative refinement with the factor already in hand recovers what the
            # conditioning allows; both residuals are reported so neither is mistaken for the other.
            u = u + torch.cholesky_solve((f - K @ u)[:, None], root)[:, 0]
            refined = float((K @ u - f).norm() / f.norm())
            c = float(f @ u)
            uA = u[dof_A]; uB = u[dof_B]
            sens = dict(A=float(-(uA @ (S @ uA))), B=float(-(uB @ (S @ uB))))
            out[name] = dict(compliance=c, sens=sens, solve_residual=refined,
                             direct_backward_error=res, condition_estimate=res / eps,
                             solve='dense Cholesky plus one step of iterative refinement',
                             adjoint_identity_relative=float((sens['A'] + sens['B'] + c) / c))
        del K, root
        gc.collect()
        if dev.type == 'cuda':
            torch.cuda.empty_cache()
        return out

    Rstar = unpack(ref / 'R_UPPER.npy', d, dev)
    Astar = Rstar.T @ Rstar; Astar = .5 * (Astar + Astar.T)
    Sstar = dense_S(Astar, quotient)
    del Astar
    gc.collect()
    report['exact'] = solve(Sstar)

    if a.mq is not None:
        M = unpack(a.mq, q, dev); M = M + torch.triu(M, 1).T
        Mhat = quotient(quotient(M).T.contiguous()); del M
        gc.collect()
        Minv = torch.linalg.inv(Mhat); del Mhat
        Ahat = Minv @ Minv; del Minv
        source = str(a.mq)
    else:
        Rh = unpack(a.a_factor, d, dev)
        Ahat = Rh.T @ Rh; del Rh
        source = str(a.a_factor)
    Ahat = .5 * (Ahat + Ahat.T)
    report['e_A'] = float((Ahat - Rstar.T @ Rstar).norm() / (Rstar.T @ Rstar).norm())
    del Rstar
    gc.collect()
    Shat = dense_S(Ahat, quotient); del Ahat
    gc.collect()
    predicted = solve(Shat)
    report['predicted'] = predicted
    report['source'] = source
    rel = {}
    for name in loads:
        e, p = report['exact'][name], predicted[name]
        rel[name] = dict(compliance_rel=abs(p['compliance'] / e['compliance'] - 1),
                         sensitivity_rel={m: abs(p['sens'][m] / e['sens'][m] - 1) for m in ('A', 'B')})
    report['relative'] = rel
    report['worst_compliance_rel'] = max(v['compliance_rel'] for v in rel.values())
    report['worst_sensitivity_rel'] = max(max(v['sensitivity_rel'].values()) for v in rel.values())
    report['worst_exact_adjoint_identity'] = max(abs(v['adjoint_identity_relative']) for v in report['exact'].values())
    report['seconds'] = time.time() - t0
    report['status'] = 'ASSEMBLE_TWO_COMPLETE'
    (a.output / 'ASSEMBLE_TWO.json').write_text(json.dumps(report, indent=1))
    print(json.dumps(dict(worst_compliance_rel=report['worst_compliance_rel'],
                          worst_sensitivity_rel=report['worst_sensitivity_rel'],
                          e_A=report['e_A'],
                          worst_exact_adjoint_identity=report['worst_exact_adjoint_identity'],
                          seconds=round(report['seconds'], 1))), flush=True)


if __name__ == '__main__':
    main()
