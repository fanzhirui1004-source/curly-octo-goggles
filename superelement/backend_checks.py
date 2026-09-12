"""Pre-checks of the representation class, no optimization, no network (CPU float64).

truncation : Cholesky factor of the (rigid-regularized) teacher S in the packet order, truncated to the physical band of radius r;
             generalized spectrum of the truncated product against the teacher. Counts the modes a banded factor damages,
             split into over-stiff (fixable by a low-rank softening) and over-soft (only L itself can fix), and the residual
             after removing the k stiffest over-stiff modes (the best a rank-k softening could do on top of that L).
coverage   : energy fraction of the teacher's softest eigenvectors inside the coarse face space span(P) (whitened metric),
             for several coarse strides and material-pruning thresholds.
"""
import argparse, json, os, sys, time
import numpy as np, torch
from pathlib import Path
sys.path.insert(0, '/root/cutfem_neural_a_20260910/source_14301bc56'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v0_superelement as V0, v1_superelement as V1
CPU = torch.device('cpu'); V0.DEV = CPU; V1.DEV = CPU
from stage_cutfem_neural_a.data import load_teacher_dense, write_json
from stage_cutfem_neural_a.dense_fit import FactorSample, json_finite
from stage_cutfem_neural_a.elimination_reference import load_upper_factor, quotient_dense
N = 65


def tri(R, B, upper): return torch.linalg.solve_triangular(R, B, upper=upper)


def load(seat, labels):
    record = next(r for r in json.load(open(labels)) if int(r['seat']) == seat)
    sample = FactorSample(record, 2026091197); data = sample.data; cache = sample.cache
    ijk = np.stack(np.unravel_index(cache['background_nodes'][cache['indices']], (N, N, N)), axis=1); q = 3 * len(ijk); d = data.dimension
    w = np.clip(V0.support_weights(ijk, N, cache['tau_corners'], cache['cut_plane']), 1e-4, 1.0)
    R = load_upper_factor(Path(record['reference']) / 'R_UPPER.npy', d, CPU)
    rigid = torch.from_numpy(cache['rigid']).double()
    return dict(record=record, sample=sample, data=data, cache=cache, ijk=ijk, q=q, d=d, w=w, R=R, rigid=rigid, quotient=data.quotient)


def sliver_subspace(L, w_thr=1e-2):
    """Orthonormal basis (whitened metric) of the displacements supported on sliver DOFs (w < w_thr)."""
    q = L['q']; mask = torch.from_numpy(np.repeat(L['w'] < w_thr, 3)); n_s = int(mask.sum())
    Es = torch.zeros((q, n_s), dtype=torch.float64); Es[mask.nonzero().squeeze(1), torch.arange(n_s)] = 1.0
    Qs, _ = torch.linalg.qr(L['R'] @ L['quotient'](Es)); return Qs


def shifted_spectrum(L, Ahat, c=10.0, w_thr=1e-2):
    """Generalized spectrum of (A_hat + E, A + E), E = c S0 max(0, w_thr - w) per DOF: the ghost-penalty band metric."""
    e = torch.from_numpy(np.repeat(c * V1.S0 * np.maximum(0.0, w_thr - L['w']), 3)).double()
    Ye = L['quotient'](torch.diag(e.sqrt())); Es = Ye @ Ye.T; del Ye
    Re = torch.linalg.cholesky(L['R'].T @ L['R'] + Es, upper=True)
    Y = tri(Re.T, Ahat + Es, False); We = tri(Re.T, Y.T.contiguous(), False); del Y, Es, Re
    return torch.linalg.eigvalsh(0.5 * (We + We.T)).numpy()


def whitened_spectrum(L, Ahat, vectors=False):
    Y = tri(L['R'].T, Ahat, False); W = tri(L['R'].T, Y.T.contiguous(), False); del Y; W = 0.5 * (W + W.T)
    if vectors:
        mu, V = torch.linalg.eigh(W); return mu.numpy(), V
    return torch.linalg.eigvalsh(W).numpy()


def stats(mu, taus=(0.03, 0.1, 0.3), ranks=(512, 1500, 3000)):
    out = dict(mu_min=float(mu.min()), mu_max=float(mu.max()), divergence_per_mode=float(np.mean(mu - np.log(np.maximum(mu, 1e-300)) - 1)) if (mu > 0).all() else float('inf'))
    for t in taus:
        out[f'over_stiff_{t}'] = int((mu > 1 + t).sum()); out[f'over_soft_{t}'] = int((mu < 1 - t).sum())
    stiff = np.sort(mu[mu > 1])[::-1]
    for k in ranks:                                                      # best rank-k softening: remove the k stiffest modes
        rest = stiff[k:] if len(stiff) > k else np.array([1.0])
        out[f'residual_mu_max_after_softening_{k}'] = float(rest.max())
    return out


def check_truncation(L, radii, out):
    q, ijk = L['q'], L['ijk']
    S, _ = load_teacher_dense(L['record']['packet'], q, CPU)
    Nr = L['rigid']; s = float(S.diagonal().median())
    S_reg = S + s * (Nr @ torch.linalg.solve(Nr.T @ Nr, Nr.T)); del S
    t0 = time.time(); Rs = torch.linalg.cholesky(S_reg, upper=True); del S_reg; print(json.dumps(dict(stage='cholesky', seconds=time.time() - t0)), flush=True)
    xyz = torch.from_numpy(ijk / (N - 1)).double()
    A_full, _ = quotient_dense(Rs.T @ Rs, L['quotient']); mu_ref = whitened_spectrum(L, A_full); del A_full
    res = dict(seat=int(L['record']['seat']), q=q, d=L['d'], regularized_reference=stats(mu_ref), factor_frobenius=float(torch.linalg.matrix_norm(Rs)), radii={})
    print(json.dumps(dict(stage='reference', **res['regularized_reference'])), flush=True)
    node = torch.arange(q) // 3; Qs = sliver_subspace(L)
    for r in radii:
        ni, nj = V0.pairs_within(xyz, xyz, r, upper=True)
        keep = torch.zeros((len(ijk), len(ijk)), dtype=torch.bool); keep[ni, nj] = True; keep[nj, ni] = True
        mask = keep[node[:, None], node[None, :]]; del keep
        Rb = torch.where(mask, Rs, torch.zeros((), dtype=torch.float64)); del mask
        dropped = float(torch.linalg.matrix_norm(Rs - Rb) / torch.linalg.matrix_norm(Rs))
        Ab, _ = quotient_dense(Rb.T @ Rb, L['quotient']); del Rb
        mu, V = whitened_spectrum(L, Ab, vectors=True)
        damaged = np.abs(mu - 1) > 0.1; sf = (Qs.T @ V[:, torch.from_numpy(damaged)]).square().sum(0).numpy() if damaged.any() else np.zeros(0); del V
        mu_e = shifted_spectrum(L, Ab); del Ab
        res['radii'][str(r)] = dict(dropped_factor_fraction=dropped, band_entries=int(9 * len(ni) - 3 * len(ijk)), **stats(mu),
                                    damaged_modes_0_1=int(damaged.sum()), damaged_sliver_dominated=int((sf > 0.5).sum()), damaged_sliver_fraction_mean=float(sf.mean()) if len(sf) else 0.0,
                                    shifted=stats(mu_e))
        print(json.dumps(dict(stage='truncation', radius=r, **res['radii'][str(r)])), flush=True)
    write_json(out / f"TRUNCATION_{int(L['record']['seat']):04d}.json", json_finite(res))


def graph_basis(L, n_modes, weighted=True):
    """Scalar eigenvectors of the material face-graph Laplacian (fine nodes adjacent on the same face), kron I3: a coarse basis
    that is smooth along thin material curves but resolves the thickness. Returns (q x 3 n_modes)."""
    import scipy.sparse as sp, scipy.sparse.linalg as sla
    ijk, w = L['ijk'], L['w']; n = len(ijk); faces = V0.face_ids(ijk, N)
    xyz = torch.from_numpy(ijk.astype(np.float64)); ni, nj = V0.pairs_within(xyz, xyz, 1.5, upper=True); ni, nj = ni.numpy(), nj.numpy()
    keep = (ni != nj) & (faces[ni] == faces[nj]); ni, nj = ni[keep], nj[keep]
    wt = np.minimum(w[ni], w[nj]) if weighted else np.ones(len(ni))
    Adj = sp.coo_matrix((np.concatenate((wt, wt)), (np.concatenate((ni, nj)), np.concatenate((nj, ni)))), shape=(n, n)).tocsr()
    Lap = sp.diags(np.asarray(Adj.sum(1)).ravel()) - Adj + 1e-9 * sp.eye(n)
    vals, vecs = sla.eigsh(Lap.tocsc(), k=n_modes, sigma=-1e-6, which='LM')
    return torch.from_numpy(np.kron(vecs[:, np.argsort(vals)], np.eye(3)))


def check_coverage(L, strides, prunes, counts, out, graph_modes=()):
    R, quotient, ijk, q, d = L['R'], L['quotient'], L['ijk'], L['q'], L['d']
    A = R.T @ R; t0 = time.time(); lam, Phi = torch.linalg.eigh(A); del A; print(json.dumps(dict(stage='teacher_eigh', seconds=time.time() - t0, lam_min=float(lam[0]), lam_max=float(lam[-1]))), flush=True)
    m = max(counts); Psi = R @ Phi[:, :m] / lam[:m].sqrt()[None, :]      # whitened, orthonormal
    Qs = sliver_subspace(L); sliver = (Qs.T @ Psi).square().sum(0).numpy(); lam_np = lam[:m].numpy()
    Nr = L['rigid']; Pn = torch.eye(q, dtype=torch.float64) - Nr @ torch.linalg.solve(Nr.T @ Nr, Nr.T)
    bands = [(0, 1e-6), (1e-6, 1e-5), (1e-5, 1e-4), (1e-4, 1e-3)]
    res = dict(seat=int(L['record']['seat']), q=q, d=d, teacher_eigenvalues=dict(min=float(lam[0]), q10=float(lam[len(lam) // 10]), median=float(lam[len(lam) // 2]), max=float(lam[-1]),
               below_1e_6=int((lam < 1e-6).sum()), below_1e_5=int((lam < 1e-5).sum()), below_1e_4=int((lam < 1e-4).sum())),
               sliver_dominated_by_band={f'{lo:g}..{hi:g}': dict(modes=int(((lam_np >= lo) & (lam_np < hi)).sum()), sliver_dominated=int(((lam_np >= lo) & (lam_np < hi) & (sliver > 0.5)).sum())) for lo, hi in bands},
               spaces={})
    print(json.dumps(dict(stage='teacher_modes', **res['sliver_dominated_by_band'])), flush=True)
    for stride in strides:
        P = torch.from_numpy(np.kron(V0.coarse_interpolation(ijk, N, stride), np.eye(3)))
        for prune in prunes:
            colnorm = P.abs().sum(0); keepc = colnorm >= prune * colnorm.max(); Pk = Pn @ P[:, keepc]
            Qc, _ = torch.linalg.qr(R @ quotient(Pk)); k = Qc.shape[1]
            frac = (Qc.T @ Psi).square().sum(0).numpy()
            entry = dict(columns=int(Pk.shape[1]), rank=int(k))
            for c in counts:
                f = frac[:c]; entry[f'softest_{c}'] = dict(mean=float(f.mean()), min=float(f.min()), below_0_9=int((f < 0.9).sum()), below_0_99=int((f < 0.99).sum()))
            for lo, hi in bands:                                          # material-dominated modes per eigenvalue band
                sel = (lam_np >= lo) & (lam_np < hi) & (sliver <= 0.5)
                if sel.any(): f = frac[sel]; entry[f'material_{lo:g}..{hi:g}'] = dict(modes=int(sel.sum()), mean=float(f.mean()), min=float(f.min()), below_0_9=int((f < 0.9).sum()))
            res['spaces'][f'stride{stride}_prune{prune}'] = entry
            print(json.dumps(dict(stage='coverage', stride=stride, prune=prune, **{k_: v for k_, v in entry.items()})), flush=True)
    for nm in graph_modes:
        for weighted in (True, False):
            P = graph_basis(L, nm, weighted); Pk = Pn @ P
            Qc, _ = torch.linalg.qr(R @ quotient(Pk)); k = Qc.shape[1]
            frac = (Qc.T @ Psi).square().sum(0).numpy()
            entry = dict(columns=int(Pk.shape[1]), rank=int(k))
            for c in counts:
                f = frac[:c]; entry[f'softest_{c}'] = dict(mean=float(f.mean()), min=float(f.min()), below_0_9=int((f < 0.9).sum()), below_0_99=int((f < 0.99).sum()))
            for lo, hi in bands:
                sel = (lam_np >= lo) & (lam_np < hi) & (sliver <= 0.5)
                if sel.any(): f = frac[sel]; entry[f'material_{lo:g}..{hi:g}'] = dict(modes=int(sel.sum()), mean=float(f.mean()), min=float(f.min()), below_0_9=int((f < 0.9).sum()))
            res['spaces'][f'graph{nm}_{"w" if weighted else "u"}'] = entry
            print(json.dumps(dict(stage='coverage', basis='graph', modes=nm, weighted=weighted, **entry)), flush=True)
    write_json(out / f"COVERAGE_{int(L['record']['seat']):04d}.json", json_finite(res))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', required=True, choices=['truncation', 'coverage']); ap.add_argument('--seat', type=int, required=True)
    ap.add_argument('--labels', default='/root/autodl-tmp/CUTFEM_SUPERELEMENT_LABELS/V1_LABELS.json'); ap.add_argument('--out', default='/root/autodl-tmp/CUTFEM_BACKEND_CHECKS_20260912')
    ap.add_argument('--radii', default='0.05,0.1,0.2,0.35'); ap.add_argument('--strides', default='4,2'); ap.add_argument('--prunes', default='0,0.05,0.2'); ap.add_argument('--counts', default='200,500,1000,1500,2400')
    ap.add_argument('--threads', type=int, default=12); ap.add_argument('--graph-modes', default='')
    args = ap.parse_args(); torch.set_num_threads(args.threads); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    L = load(args.seat, args.labels)
    if args.check == 'truncation': check_truncation(L, [float(x) for x in args.radii.split(',')], out)
    else: check_coverage(L, [int(x) for x in args.strides.split(',') if x], [float(x) for x in args.prunes.split(',')], [int(x) for x in args.counts.split(',')], out,
                         graph_modes=[int(x) for x in args.graph_modes.split(',') if x])


if __name__ == '__main__':
    main()
