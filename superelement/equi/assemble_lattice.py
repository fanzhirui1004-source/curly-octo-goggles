"""An nx x ny x nz lattice of copies of one FULL cell, exact adjoint per module: the contract itself.

assemble_two.py glues two copies on one face pair.  Here copies sit at every integer offset of a
grid and glue on all three face pairs, so a node can be shared by up to eight modules.  The
contract is that the assembled compliance and every module's design sensitivity are right to
3 %; the load-energy diagnosis predicts the compliance error to be a law-of-large-numbers
cancellation of per-mode errors (robust for smooth loads) and the sensitivity error to be its
half-domain fluctuation ~ rms / sqrt(n_eff / 2).  A lattice spreads the energy over more modules
and more modes, so both predictions are testable here at the scale that matters.

Float64 on the host.  The default solver is matrix-free, because a dense operator is impossible
here twice over: torch's CPU indexing caps a tensor at 2^31 elements (N = 46340, two cells of a
4176-coordinate seat), and the container's cgroup ceiling is 90 GiB, not the 754 GB that `free`
reports for the host.  With --grid 2 1 1 along ASSEMBLE_TWO's glue axis it must reproduce
assemble_two.py's numbers, which is the validation gate (--check-two).

    python -m superelement.equi.assemble_lattice --seat 347 --a-factor A_PRED.npy --grid 2 2 2 --output DIR
    python -m superelement.equi.assemble_lattice --seat 100034 347 --a-factor CUT.npy FULL.npy --grid 2 1 1 --output DIR
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
    ap.add_argument('--seat', type=int, nargs='+', required=True,
                    help='one seat, or one per copy in x-major order: neighbours may be different cells')
    ap.add_argument('--a-factor', type=Path, nargs='+', required=True,
                    help='A_PRED_UPPER.npy per distinct seat, in the order the seats first appear')
    ap.add_argument('--grid', type=int, nargs=3, default=[2, 2, 2])
    ap.add_argument('--check-two', type=Path, default=None, help='ASSEMBLE_TWO.json to reproduce with --grid 2 1 1 along its axis')
    ap.add_argument('--solver', choices=['dense', 'pcg'], default='pcg')
    ap.add_argument('--rtol', type=float, default=1e-12); ap.add_argument('--max-iterations', type=int, default=4000)
    ap.add_argument('--replace-every', type=int, default=200,
                    help='recompute the TRUE residual F - K U every this many iterations and restart the '
                         'search direction from it.  The recursive residual R <- R - alpha K P drifts away '
                         'from the true one on an ill-conditioned predicted operator: the mixed three-cell '
                         'run of 2026-09-20 stopped with a recursive residual under 1e-9 and a true residual '
                         'of 1.863e-7, and CONTROL passed the same item only at 5.343e-8 against a 1e-7 line. '
                         'Termination now requires a freshly computed TRUE residual under --rtol, so this '
                         'tightens the gate; 0 disables replacement and restores the old recursive-only test.')
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

    manifest = json.loads(a.manifest.read_text())
    grid = [int(v) for v in a.grid]
    copies = [(i, j, k) for i in range(grid[0]) for j in range(grid[1]) for k in range(grid[2])]
    cidx = {c: m for m, c in enumerate(copies)}
    seats = [int(v) for v in a.seat]
    if len(seats) == 1:
        seats = seats * len(copies)
    if len(seats) != len(copies):
        raise ValueError(f'GIVE_ONE_SEAT_OR_ONE_PER_COPY {len(seats)} vs {len(copies)}')
    order = list(dict.fromkeys(seats))                     # distinct seats, first appearance
    if len(a.a_factor) != len(order):
        raise ValueError(f'GIVE_ONE_A_FACTOR_PER_DISTINCT_SEAT {len(a.a_factor)} vs {len(order)}')
    factor_of = dict(zip(order, a.a_factor))
    cell = {}
    n = None
    for seat in order:
        row = [r for r in manifest if int(r['seat']) == seat][0]
        ref = Path(row['reference']); receipt = json.loads((ref / 'RESULT.json').read_text())
        d = int(receipt['dimension']); q = d + 6
        cache = dict(np.load(row['trace_cache'], allow_pickle=False))
        meta = json.loads((Path(row['trace_cache']).parent / 'INPUT.json').read_text())['metadata']
        if n is None:
            n = int(meta['n'])
        elif int(meta['n']) != n:
            raise ValueError('MIXED_BACKGROUND_GRIDS')
        ctx = compile_equi_inputs(cache, meta)
        if rigid_span_residual(cache, n) > 1e-12:
            raise ValueError(f'RIGID_IS_NOT_THE_TRACE_PULLBACK seat {seat}')
        kind = np.asarray(cache['kind']).astype(np.int64)
        if 3 * int(ctx['count']) != q:
            raise ValueError('DIMENSION_BINDING')
        cell[seat] = dict(ref=ref, d=d, q=q, cache=cache, ctx=ctx, kind=kind, pos=ctx['pos'],
                          count=int(ctx['count']), box_only=bool(ctx['box_only']),
                          nodes=np.asarray(cache['background_nodes'], dtype=np.int64))
    report = dict(seats=seats, distinct=order, grid=grid, modules=len(copies), n=n,
                  per_seat={str(k): dict(q=v['q'], coordinates=v['count'], box_only=v['box_only'],
                                         cut_functionals=int((v['kind'] == 1).sum())) for k, v in cell.items()},
                  a_factor={str(k): str(v) for k, v in factor_of.items()},
                  graded=bool(len(order) > 1))

    # Global ids by union-find over (copy, local coordinate).  Two neighbouring modules glue on a
    # coordinate that is ONE background node (kind 0) lying on their shared face, matched by its
    # exact position in the lattice.  That works when the neighbours are different cells too: the
    # glue set is then the INTERSECTION of the two faces' surviving nodes, and a node only one side
    # kept is free surface, which is the physical situation at a graded interface.  Cut-surface
    # functionals (kind 1) are never shared.
    offsets = np.cumsum([0] + [cell[s]['count'] for s in seats])
    total_local = int(offsets[-1])
    uf = UnionFind(total_local)
    key = []
    for m, (c, seat) in enumerate(zip(copies, seats)):
        v = cell[seat]
        pos_world = v['pos'] + np.asarray(c, dtype=np.float64)
        key.append({(round(float(x), 9), round(float(y), 9), round(float(z), 9)): int(offsets[m]) + i
                    for i, (x, y, z) in enumerate(pos_world) if v['kind'][i] == 0})
    shared_count = 0
    for m, (c, seat) in enumerate(zip(copies, seats)):
        for axis in range(3):
            nb = list(c); nb[axis] += 1; nb = tuple(nb)
            if nb not in cidx:
                continue
            m2 = cidx[nb]
            for k3, idx in key[m].items():
                if abs(k3[axis] - (c[axis] + 1)) > 1e-9:
                    continue
                other = key[m2].get(k3)
                if other is not None:
                    uf.union(idx, other); shared_count += 1
    roots = np.array([uf.find(i) for i in range(total_local)])
    _, gid_flat = np.unique(roots, return_inverse=True)
    gid = [gid_flat[offsets[m]:offsets[m + 1]] for m in range(len(copies))]
    N_nodes = int(gid_flat.max()) + 1; N = 3 * N_nodes
    report.update(assembled_coordinates=N_nodes, assembled_dofs=N, glued_coordinate_pairs=shared_count)
    # Surviving kind-0 coordinates per face, per seat: a cut plane can remove a whole face, and then
    # the modules along that axis are not connected at all.  In the graded set every cut seat has an
    # empty high-x face, so a cut cell glues along y or z, not x.
    report['face_population'] = {str(seat): {'xyz'[ax]: [int(((cell[seat]['kind'] == 0) & (np.abs(cell[seat]['pos'][:, ax] - side) < 1e-9)).sum())
                                                         for side in (0., 1.)] for ax in range(3)} for seat in order}
    if shared_count == 0 and len(copies) > 1:
        raise ValueError(f"MODULES_ARE_NOT_CONNECTED: no shared face coordinate. face_population "
                         f"{json.dumps(report['face_population'])}; pick an axis whose faces survive the cut")
    print(json.dumps({k: report[k] for k in ('distinct', 'grid', 'modules', 'graded', 'assembled_coordinates',
                                             'assembled_dofs', 'glued_coordinate_pairs', 'per_seat',
                                             'face_population')}), flush=True)
    dofs = [torch.as_tensor((gid[m][:, None] * 3 + np.arange(3)).ravel()) for m in range(len(copies))]
    P = np.zeros((N_nodes, 3)); kindN = np.zeros(N_nodes, dtype=np.int64)
    for m, (c, seat) in enumerate(zip(copies, seats)):
        P[gid[m]] = cell[seat]['pos'] + np.asarray(c, dtype=np.float64); kindN[gid[m]] = cell[seat]['kind']

    # rigid modes: each module's exact CSR pullback of the global rigid field; consistent on glue
    Nrb = np.zeros((N, 6)); worst = 0.
    for m, (c, seat) in enumerate(zip(copies, seats)):
        rb = rigid_trace(cell[seat]['cache'], n, np.asarray(c, dtype=np.float64))
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

    for seat in order:
        cache_s = cell[seat]['cache']
        cell[seat]['quotient'] = RigidQuotient(torch.from_numpy(np.asarray(cache_s['rigid'], dtype=np.float64)),
                                               torch.from_numpy(np.asarray(cache_s['order'])))
        rb, _ = torch.linalg.qr(torch.from_numpy(np.asarray(cache_s['rigid'], dtype=np.float64)))
        cell[seat]['rb_basis'] = rb

    def solve_pcg(S):
        """K is never formed, and every module and load goes through S and the factor together.

        K u = sum_m scatter_m(S gather_m(u)) + scale Nrb (Nrb^T u), preconditioned by additive
        Schwarz with ONE exact local solve shared by all modules: the cell's rigid trace spans the
        same six-dimensional space at every integer offset (shifting the origin mixes rotations
        into translations, which are in the space already), so S + sigma Pi_rigid factorises once.

        A dense K costs N^2 doubles and torch's CPU indexing caps a tensor at 2^31 elements, so the
        dense path stops at N = 46340 -- two cells of a 4176-coordinate seat, not a lattice.  The
        loop that matters is not flops but bandwidth: S and its factor are 1.25 GB at q = 12528, so
        a naive per-module loop re-reads them n_modules times per iteration.  Gathering every module
        and every load into one (q, modules x loads) block makes each iteration read them ONCE, so
        the cost is nearly independent of the lattice size.  Validated against the dense solver on
        2 x 1 x 1 to 2.1e-14 on both the exact compliances and the relative errors.
        """
        L = len(names)
        groups = {seat: [m for m in range(len(copies)) if seats[m] == seat] for seat in order}
        stack = {seat: torch.stack([dofs[m] for m in groups[seat]]) for seat in order}
        flat = {seat: stack[seat].reshape(-1) for seat in order}
        diag = torch.zeros(N, dtype=F64)
        for m, seat in enumerate(seats):
            diag.index_add_(0, dofs[m], S[seat].diagonal())
        scale = float(diag.abs().mean())
        root = {}
        tick = time.time()
        for seat in order:
            sd = S[seat].diagonal()
            rb = cell[seat]['rb_basis']
            local = S[seat] + float(sd.abs().mean()) * (rb @ rb.T)
            root[seat] = torch.linalg.cholesky(.5 * (local + local.T)); del local
            gc.collect()
        chol_seconds = time.time() - tick

        def block(V, seat):                                            # (N, L) -> (q, |group| * L)
            g = len(groups[seat])
            return V[stack[seat]].permute(1, 0, 2).reshape(cell[seat]['q'], g * L)

        def unblock(B, seat, into):
            g = len(groups[seat]); q_ = cell[seat]['q']
            into.index_add_(0, flat[seat], B.reshape(q_, g, L).permute(1, 0, 2).reshape(g * q_, L))

        def apply(V):
            out = torch.zeros_like(V)
            for seat in order:
                unblock(S[seat] @ block(V, seat), seat, out)
            return out + scale * (Nrb @ (Nrb.T @ V))

        def precondition(V):
            out = torch.zeros_like(V)
            for seat in order:
                unblock(torch.cholesky_solve(block(V, seat), root[seat]), seat, out)
            return out

        F0 = Fm.clone()
        fn = F0.norm(dim=0)
        U = torch.zeros((N, L), dtype=F64); R = F0.clone()
        Z = precondition(R); P = Z.clone(); rz = (R * Z).sum(dim=0)
        iterations = 0; replacements = 0; recursive = float('nan')

        def true_residual():
            return (apply(U) - F0).norm(dim=0) / fn

        for iterations in range(1, a.max_iterations + 1):
            KP = apply(P)
            alpha = rz / (P * KP).sum(dim=0).clamp_min(1e-300)
            U = U + alpha * P; R = R - alpha * KP
            recursive = float((R.norm(dim=0) / fn).max())
            due = a.replace_every > 0 and iterations % a.replace_every == 0
            if recursive <= a.rtol or due:
                # The recursive residual is an estimate; only a freshly computed F - K U can stop the
                # loop.  Replacing R breaks conjugacy with the old P, so the direction restarts.
                if a.replace_every > 0:
                    R = F0 - apply(U); replacements += 1
                    if float((R.norm(dim=0) / fn).max()) <= a.rtol:
                        break
                    Z = precondition(R); rz = (R * Z).sum(dim=0); P = Z.clone()
                    continue
                if recursive <= a.rtol:
                    break
            Z = precondition(R); rz_new = (R * Z).sum(dim=0)
            P = Z + (rz_new / rz.clamp_min(1e-300)) * P; rz = rz_new
        KU = apply(U)
        resid = (KU - F0).norm(dim=0) / fn
        # Normwise backward error, with ||K u|| in place of ||K|| ||u||.  ||K u|| <= ||K|| ||u||, so
        # the denominator is smaller and eta is an upper bound.  ||r|| / ||f|| cannot go below about
        # kappa(K) * eps, so on a predicted cut operator (kappa ~ 1e6..1e10) a fixed relative-residual
        # line is partly a conditioning readout; eta says whether the solve itself is at its floor.
        eta = (KU - F0).norm(dim=0) / (KU.norm(dim=0) + fn).clamp_min(1e-300)
        floor = 10. * (N ** .5) * float(torch.finfo(F64).eps)
        if float(resid.max()) > 100 * a.rtol:
            raise ValueError(f'PCG_DID_NOT_CONVERGE residual {float(resid.max()):.3e} in {iterations} iterations '
                             f'(recursive {recursive:.3e}, {replacements} true-residual replacements, '
                             f'backward error {float(eta.max()):.3e} against a floor of {floor:.3e})')
        energy = torch.zeros((len(copies), L), dtype=F64)              # per module, for the adjoint
        for seat in order:
            B = block(U, seat); E = (B * (S[seat] @ B)).sum(dim=0).reshape(len(groups[seat]), L)
            for i, m in enumerate(groups[seat]):
                energy[m] = E[i]
        del root; gc.collect()
        out = {}
        for j, name in enumerate(names):
            c = float(F0[:, j] @ U[:, j])
            sens = [float(-energy[m, j]) for m in range(len(copies))]
            out[name] = dict(compliance=c, sens=sens, solve_residual=float(resid[j]), iterations=iterations,
                             recursive_residual=recursive, true_residual_replacements=replacements,
                             adjoint_identity_relative=float((sum(sens) + c) / c), cholesky_seconds=chol_seconds)
        return out

    def solve_dense(S):
        K = torch.zeros((N, N), dtype=F64)
        for m, seat in enumerate(seats):
            K.index_put_((dofs[m][:, None], dofs[m][None, :]), S[seat], accumulate=True)
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
            sens = [float(-(u[dofs[m]] @ (S[seats[m]] @ u[dofs[m]]))) for m in range(len(copies))]
            out[name] = dict(compliance=c, sens=sens, solve_residual=resid,
                             adjoint_identity_relative=float((sum(sens) + c) / c), cholesky_seconds=time.time() - tick)
        return out

    if a.solver == 'dense' and N > 46340:
        raise ValueError(f'DENSE_SOLVER_EXCEEDS_TORCH_INDEX_LIMIT N={N}; use --solver pcg')
    solve = solve_dense if a.solver == 'dense' else solve_pcg
    report['solver'] = a.solver

    def operators(which):
        out = {}
        for seat in order:
            v = cell[seat]
            if which == 'exact':
                R = unpack(v['ref'] / 'R_UPPER.npy', v['d'], dev); A = R.T @ R; del R
            else:
                R = unpack(Path(factor_of[seat]), v['d'], dev); A = R.T @ R; del R
            A = .5 * (A + A.T)
            out[seat] = dense_S(A, v['quotient']); del A
            gc.collect()
        return out

    Se = operators('exact'); report['exact'] = solve(Se); del Se; gc.collect()
    print(json.dumps(dict(phase='exact', seconds=round(time.time() - t0, 1),
                          chol=round(report['exact'][names[0]]['cholesky_seconds'], 1))), flush=True)
    Sp = operators('predicted'); report['predicted'] = solve(Sp); del Sp; gc.collect()

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
