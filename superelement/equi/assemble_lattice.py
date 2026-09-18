"""An nx x ny x nz lattice of copies of one FULL cell, exact adjoint per module: the contract itself.

assemble_two.py glues two copies on one face pair.  Here copies sit at every integer offset of a
grid and glue on all three face pairs, so a node can be shared by up to eight modules.  The
contract is that the assembled compliance and every module's design sensitivity are right to
3 %; the load-energy diagnosis predicts the compliance error to be a law-of-large-numbers
cancellation of per-mode errors (robust for smooth loads) and the sensitivity error to be its
half-domain fluctuation ~ rms / sqrt(n_eff / 2).  A lattice spreads the energy over more modules
and more modes, so both predictions are testable here at the scale that matters.

Dense float64 on the host (2 x 2 x 2 of a 6016-coordinate cell is ~108k dofs, 93 GB per operator;
the box has 754 GB).  With --grid 2 1 1 along ASSEMBLE_TWO's glue axis it must reproduce
assemble_two.py's numbers, which is the validation gate (--check-two).

    python -m superelement.equi.assemble_lattice --seat 347 --a-factor A_PRED_UPPER.npy --grid 2 2 2 --output DIR
"""
from __future__ import annotations

import argparse, gc, json, sys, time
from pathlib import Path
import numpy as np
import torch

F64 = torch.float64


def face_pairs(kind, pos, axis, tol=1e-9):
    """lo-face coordinate -> hi-face coordinate of the same cell with the same transverse position."""
    other = [j for j in range(3) if j != axis]
    lo = np.flatnonzero((kind == 0) & (np.abs(pos[:, axis]) <= tol))
    hi = np.flatnonzero((kind == 0) & (np.abs(pos[:, axis] - 1.) <= tol))
    key = {(round(float(pos[i, other[0]]), 12), round(float(pos[i, other[1]]), 12)): int(i) for i in hi}
    return {int(i): key[k] for i in lo
            if (k := (round(float(pos[i, other[0]]), 12), round(float(pos[i, other[1]]), 12))) in key}, len(lo), len(hi)


class UnionFind:
    def __init__(self, n):
        self.p = np.arange(n)

    def find(self, i):
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]; i = self.p[i]
        return i

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', type=Path, default=Path('/root/autodl-tmp/CLAUDE_LABELS_20260917/V2_LABELS.json'))
    ap.add_argument('--seat', type=int, required=True)
    ap.add_argument('--a-factor', type=Path, required=True, help='A_PRED_UPPER.npy (d x d upper Cholesky of A_hat)')
    ap.add_argument('--grid', type=int, nargs=3, default=[2, 2, 2])
    ap.add_argument('--check-two', type=Path, default=None, help='ASSEMBLE_TWO.json to reproduce with --grid 2 1 1 along its axis')
    ap.add_argument('--solver', choices=['dense', 'pcg'], default='pcg')
    ap.add_argument('--rtol', type=float, default=1e-12); ap.add_argument('--max-iterations', type=int, default=4000)
    ap.add_argument('--source', type=Path, default=Path('/root/autodl-tmp/CLAUDE_SQRTHEAD_20260917/src_v5'))
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(a.source)); sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from stage_cutfem_m4.quotient import RigidQuotient
    from superelement.equi.context import compile_equi_inputs, rigid_span_residual
    from superelement.equi.assemble_two import unpack, dense_S, rigid_trace
    torch.set_num_threads(a.threads)
    dev = torch.device('cpu'); t0 = time.time()

    row = [r for r in json.loads(a.manifest.read_text()) if int(r['seat']) == a.seat][0]
    ref = Path(row['reference']); receipt = json.loads((ref / 'RESULT.json').read_text())
    d = int(receipt['dimension']); q = d + 6
    cache = dict(np.load(row['trace_cache'], allow_pickle=False))
    meta = json.loads((Path(row['trace_cache']).parent / 'INPUT.json').read_text())['metadata']; n = int(meta['n'])
    ctx = compile_equi_inputs(cache, meta)
    if not ctx['box_only']:
        raise ValueError('FULL_CELLS_ONLY: a cut cell has no face pairs on the cut side')
    if rigid_span_residual(cache, n) > 1e-12:
        raise ValueError('RIGID_IS_NOT_THE_TRACE_PULLBACK')
    kind = np.asarray(cache['kind']).astype(np.int64); pos = ctx['pos']; count = int(ctx['count'])
    if 3 * count != q:
        raise ValueError('DIMENSION_BINDING')
    grid = [int(v) for v in a.grid]
    copies = [(i, j, k) for i in range(grid[0]) for j in range(grid[1]) for k in range(grid[2])]
    cidx = {c: m for m, c in enumerate(copies)}
    pairs = {axis: face_pairs(kind, pos, axis) for axis in range(3)}
    report = dict(seat=a.seat, q=q, n=n, coordinates=count, grid=grid, modules=len(copies),
                  face_pairs={axis: dict(shared=len(p[0]), low=p[1], high=p[2]) for axis, p in pairs.items()},
                  a_factor=str(a.a_factor))

    # global node ids by union-find over (copy, local node): copy c's HIGH face node glues to
    # copy c+e_axis's LOW face node at the same transverse position
    uf = UnionFind(len(copies) * count)
    for (i, j, k), m in cidx.items():
        for axis in range(3):
            nb = [i, j, k]; nb[axis] += 1; nb = tuple(nb)
            if nb not in cidx:
                continue
            m2 = cidx[nb]
            for lo, hi in pairs[axis][0].items():
                uf.union(m * count + hi, m2 * count + lo)
    roots = np.array([uf.find(i) for i in range(len(copies) * count)])
    _, gid = np.unique(roots, return_inverse=True)
    gid = gid.reshape(len(copies), count)
    N_nodes = int(gid.max()) + 1; N = 3 * N_nodes
    report.update(assembled_coordinates=N_nodes, assembled_dofs=N)
    print(json.dumps({k: report[k] for k in ('seat', 'grid', 'modules', 'assembled_coordinates', 'assembled_dofs', 'face_pairs')}), flush=True)
    dofs = [torch.as_tensor((gid[m][:, None] * 3 + np.arange(3)).ravel()) for m in range(len(copies))]
    P = np.zeros((N_nodes, 3)); kindN = np.zeros(N_nodes, dtype=np.int64)
    for m, c in enumerate(copies):
        P[gid[m]] = pos + np.asarray(c, dtype=np.float64); kindN[gid[m]] = kind

    # rigid modes: each module's exact CSR pullback of the global rigid field; consistent on glue
    Nrb = np.zeros((N, 6)); worst = 0.
    for m, c in enumerate(copies):
        rb = rigid_trace(cache, n, np.asarray(c, dtype=np.float64))
        rows_ = dofs[m].numpy()
        filled = np.abs(Nrb[rows_]).sum(axis=1) > 0
        if filled.any():
            worst = max(worst, float(np.abs(Nrb[rows_][filled] - rb[filled]).max() / max(np.abs(rb).max(), 1e-300)))
        Nrb[rows_] = rb
    report['glue_rigid_mismatch'] = worst
    if worst > 1e-12:
        raise ValueError(f'GLUE_IS_INCONSISTENT_ON_THE_RIGID_MODES {worst:.3e}')
    Nrb, _ = torch.linalg.qr(torch.as_tensor(Nrb, dtype=F64))

    # loads on the two extreme faces along x (the longest axis by construction of --grid)
    axis = int(np.argmax(grid)); tv = [j for j in range(3) if j != axis]
    endL = np.flatnonzero((kindN == 0) & (P[:, axis] <= P[:, axis].min() + 1e-9))
    endR = np.flatnonzero((kindN == 0) & (P[:, axis] >= P[:, axis].max() - 1e-9))
    loads = {}
    for name, comp in ((f'axial_{"xyz"[axis]}', axis), (f'shear_{"xyz"[tv[0]]}', tv[0]), (f'shear_{"xyz"[tv[1]]}', tv[1])):
        f = torch.zeros(N, dtype=F64); f[torch.as_tensor(endL * 3 + comp)] = -1.0 / len(endL); f[torch.as_tensor(endR * 3 + comp)] = 1.0 / len(endR)
        loads[name] = f
    f = torch.zeros(N, dtype=F64)
    wl = P[endL, tv[0]] - P[endL, tv[0]].mean(); wr = P[endR, tv[0]] - P[endR, tv[0]].mean()
    f[torch.as_tensor(endL * 3 + axis)] = torch.as_tensor(-wl / max(np.abs(wl).sum(), 1e-300))
    f[torch.as_tensor(endR * 3 + axis)] = torch.as_tensor(wr / max(np.abs(wr).sum(), 1e-300))
    loads[f'bending_{"xyz"[axis]}{"xyz"[tv[0]]}'] = f
    for k in loads:
        loads[k] = loads[k] - Nrb @ (Nrb.T @ loads[k]); loads[k] = loads[k] / loads[k].norm()
    report.update(load_axis=axis, load_left=int(len(endL)), load_right=int(len(endR)))
    names = list(loads); Fm = torch.stack([loads[k] for k in names], dim=1)

    quotient = RigidQuotient(torch.from_numpy(np.asarray(cache['rigid'], dtype=np.float64)),
                             torch.from_numpy(np.asarray(cache['order'])))

    rb_basis = None

    def solve_pcg(S):
        """K is never formed.  K u = sum_m scatter_m(S gather_m(u)) + scale Nrb (Nrb^T u), and the
        preconditioner is additive Schwarz with one exact local solve, reused by every module.

        A dense K costs N^2 doubles, and torch's CPU indexing caps a tensor at 2^31 elements, so
        the dense path stops at N = 46340 -- two cells of a 4176-coordinate seat, not a lattice.
        The local solve is shared because the cell's rigid trace spans the SAME six-dimensional
        space at every integer offset (shifting the origin mixes the rotations into translations,
        which are in the space already), so S + sigma Pi_rigid is one factorisation for all
        modules.
        """
        nonlocal rb_basis
        diag = torch.zeros(N, dtype=F64)
        sd = S.diagonal()
        for m in range(len(copies)):
            diag.index_add_(0, dofs[m], sd)
        scale = float(diag.abs().mean())
        if rb_basis is None:
            rb, _ = torch.linalg.qr(torch.from_numpy(np.asarray(cache['rigid'], dtype=np.float64)))
            rb_basis = rb
        sigma = float(S.diagonal().abs().mean())
        local = S + sigma * (rb_basis @ rb_basis.T)
        root = torch.linalg.cholesky(.5 * (local + local.T)); del local; gc.collect()

        def apply(u):
            out = torch.zeros_like(u)
            for m in range(len(copies)):
                out.index_add_(0, dofs[m], S @ u[dofs[m]])
            return out + scale * (Nrb @ (Nrb.T @ u))

        def precondition(r):
            out = torch.zeros_like(r)
            for m in range(len(copies)):
                out.index_add_(0, dofs[m], torch.cholesky_solve(r[dofs[m]].unsqueeze(1), root).squeeze(1))
            return out

        out = {}; tick = time.time(); iterations = {}
        for j, name in enumerate(names):
            f = Fm[:, j]
            u = torch.zeros(N, dtype=F64); r = f.clone(); z = precondition(r); p = z.clone()
            rz = float(r @ z); fn = float(f.norm()); it = 0
            for it in range(1, a.max_iterations + 1):
                Kp = apply(p); alpha = rz / float(p @ Kp)
                u += alpha * p; r -= alpha * Kp
                if float(r.norm()) / fn <= a.rtol:
                    break
                z = precondition(r); rz_new = float(r @ z)
                p = z + (rz_new / rz) * p; rz = rz_new
            resid = float((apply(u) - f).norm() / fn)
            c = float(f @ u)
            sens = [float(-(u[dofs[m]] @ (S @ u[dofs[m]]))) for m in range(len(copies))]
            iterations[name] = it
            out[name] = dict(compliance=c, sens=sens, solve_residual=resid, iterations=it,
                             adjoint_identity_relative=float((sum(sens) + c) / c),
                             cholesky_seconds=time.time() - tick)
            if resid > 100 * a.rtol:
                raise ValueError(f'PCG_DID_NOT_CONVERGE {name} residual {resid:.3e} in {it} iterations')
        del root; gc.collect()
        return out

    def solve_dense(S):
        K = torch.zeros((N, N), dtype=F64)
        for m in range(len(copies)):
            K.index_put_((dofs[m][:, None], dofs[m][None, :]), S, accumulate=True)
        scale = float(K.diagonal().abs().mean()); K.addmm_(Nrb, Nrb.T, alpha=scale)
        asym = float((K - K.T).abs().max())
        if asym != 0.0:
            raise ValueError(f'ASSEMBLED_OPERATOR_NOT_SYMMETRIC {asym:.3e}')
        tick = time.time(); root = torch.linalg.cholesky(K); del K; gc.collect()
        U = torch.linalg.solve_triangular(root, Fm, upper=False, left=True)
        U = torch.linalg.solve_triangular(root.T, U, upper=True, left=True)
        resid = float(((root @ (root.T @ U) - Fm).norm(dim=0) / Fm.norm(dim=0)).max()); del root; gc.collect()
        out = {}
        for j, name in enumerate(names):
            u = U[:, j]; c = float(Fm[:, j] @ u)
            sens = [float(-(u[dofs[m]] @ (S @ u[dofs[m]]))) for m in range(len(copies))]
            out[name] = dict(compliance=c, sens=sens, solve_residual=resid,
                             adjoint_identity_relative=float((sum(sens) + c) / c), cholesky_seconds=time.time() - tick)
        return out

    if a.solver == 'dense' and N > 46340:
        raise ValueError(f'DENSE_SOLVER_EXCEEDS_TORCH_INDEX_LIMIT N={N}; use --solver pcg')
    solve = solve_dense if a.solver == 'dense' else solve_pcg
    report['solver'] = a.solver

    Rstar = unpack(ref / 'R_UPPER.npy', d, dev); Astar = Rstar.T @ Rstar; Astar = .5 * (Astar + Astar.T)
    Sstar = dense_S(Astar, quotient); del Astar; gc.collect()
    report['exact'] = solve(Sstar); del Sstar; gc.collect()
    print(json.dumps(dict(phase='exact', seconds=round(time.time() - t0, 1),
                          chol=round(report['exact'][names[0]]['cholesky_seconds'], 1))), flush=True)
    Rh = unpack(a.a_factor, d, dev); Ahat = Rh.T @ Rh; del Rh; Ahat = .5 * (Ahat + Ahat.T)
    Shat = dense_S(Ahat, quotient); del Ahat; gc.collect()
    report['predicted'] = solve(Shat); del Shat; gc.collect()

    rel = {}
    for name in names:
        e, p = report['exact'][name], report['predicted'][name]
        rel[name] = dict(compliance_rel=p['compliance'] / e['compliance'] - 1.0,
                         sensitivity_rel=[p['sens'][m] / e['sens'][m] - 1.0 for m in range(len(copies))],
                         split_exact=[s / sum(e['sens']) for s in e['sens']],
                         split_predicted=[s / sum(p['sens']) for s in p['sens']])
    report['relative'] = rel
    report['worst_compliance_rel'] = max(abs(v['compliance_rel']) for v in rel.values())
    report['worst_sensitivity_rel'] = max(max(abs(x) for x in v['sensitivity_rel']) for v in rel.values())
    report['worst_split_rel'] = max(max(abs(sp / se - 1.0) for sp, se in zip(v['split_predicted'], v['split_exact'])) for v in rel.values())
    report['worst_exact_adjoint_identity'] = max(abs(v['adjoint_identity_relative']) for v in report['exact'].values())
    if a.check_two is not None:
        two = json.loads(a.check_two.read_text())
        # assemble_two reports |compliance_rel|; this reports it signed, so compare magnitudes
        diffs = {nm: abs(abs(rel[nm]['compliance_rel']) - two['relative'][nm]['compliance_rel']) for nm in names if nm in two['relative']}
        diffs.update({f'{nm}/exact_compliance': abs(report['exact'][nm]['compliance'] / two['exact'][nm]['compliance'] - 1.0)
                      for nm in names if nm in two['exact']})
        report['check_two'] = dict(max_compliance_rel_difference=max(diffs.values()), per_load=diffs)
    report['seconds'] = time.time() - t0; report['status'] = 'ASSEMBLE_LATTICE_COMPLETE'
    (a.output / 'ASSEMBLE_LATTICE.json').write_text(json.dumps(report, indent=1))
    print(json.dumps(dict(worst_compliance_rel=report['worst_compliance_rel'], worst_sensitivity_rel=report['worst_sensitivity_rel'],
                          worst_split_rel=report['worst_split_rel'], adjoint=report['worst_exact_adjoint_identity'],
                          check_two=report.get('check_two', {}).get('max_compliance_rel_difference'), seconds=round(report['seconds'], 1))), flush=True)


if __name__ == '__main__':
    main()
