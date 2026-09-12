"""Exact whitened spectrum of a learned super-element against its teacher (probe-free evaluation).

  W = R*^-T A_hat R*^-1,  mu = eig(W)  (d generalized eigenvalues of the student against the teacher)
  D = sum(mu - log mu - 1)            log-det divergence  (= 2 KL between the two elastic Gibbs measures)
  mu_min, mu_max                       spectral-equivalence constants: every response error is bounded by them
Every mode is also classified by where its displacement lives (sliver nodes / coarse face space / other),
so the divergence can be attributed without putting any coarse space into the training objective.
Runs on the CPU in float64 (the GPU may be busy with training); the student is materialized once.
"""
import argparse, json, math, os, sys, time
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


def logdet_rigid(N_r, solve_S):
    """log det(N^T S^-1 N) - log det(N^T N): the rigid-quotient correction of log det (B S B^T) for an orthonormal quotient B."""
    U = solve_S(N_r)
    return float(torch.logdet(N_r.T @ U) - torch.logdet(N_r.T @ N_r))


def student_v1(run, ckpt, record):
    proto = json.load(open(Path(run) / 'PROTOCOL.json')); off_scale = proto.get('off_scale', 1.0)
    label = V1.Label(record, proto['r_near'], proto['decay'], off_scale)
    sd = torch.load(ckpt, map_location='cpu')['net']
    vw = proto.get('volume_width', 0) if any(k.startswith('volume.') for k in sd) else 0
    net = V1.GeometryNet(label.images.shape[1], proto['width'], proto['rank'], proto['hidden'], off_scale=off_scale, volume_width=vw)
    net.load_state_dict(sd); net.eval()
    g = dict(images=label.images, theta=label.theta, mem_node=label.mem_node, mem_face=label.mem_face, mem_pix=label.mem_pix, volume=label.volume, xyz=label.xyz_nodes,
             Q=torch.eye(3, dtype=torch.float64), face_of_node=label.face_of_node, pair_i=label.pair_i, pair_j=label.pair_j, pair_dx=label.pair_dx, pair_decay=label.pair_decay,
             band_index=label.band_index, band_index_t=label.band_index_t, band_perm=label.band_perm, band_crow=label.band_crow, w=torch.from_numpy(label.w).double())
    with torch.no_grad():
        values, M = net(g, g['w'].log()); op = V1.Operator(g, values, M); S = op.materialize()
        diag = values[label.band_index[0] == label.band_index[1]]
        G = M.T @ M; G.diagonal().add_(1.0)
        L = torch.sparse_coo_tensor(label.band_index, values, (label.q, label.q)).to_dense()
        def solve_S(Y):
            Z = tri(L.T, Y, False); Z = Z + M @ (M.T @ Z); return tri(L, Z, True)
        rigid = torch.from_numpy(label.sample.cache['rigid']).double()
        closed = dict(logdet_S=2 * float(diag.log().sum()) - float(torch.logdet(G)), rigid_term=logdet_rigid(rigid, solve_S))
        del L
    return S, label.data, label.sample, label.ijk, label.w, closed, dict(schema='V1', seat=label.seat, q=label.q, d=label.d, off_scale=off_scale, rank=proto['rank'], r_near=proto['r_near'])


def student_v0(run, ckpt):
    proto = json.load(open(Path(run) / 'PROTOCOL.json'))
    records = json.load(open('/root/autodl-tmp/CUTFEM_NEURAL_DENSE_20260911_P01/INPUT_MANIFEST.json'))['samples']
    record = next(r for r in records if int(r['seat']) == proto['seat'])
    sample = FactorSample(record, 2026091197); data = sample.data; cache = sample.cache
    ijk = np.stack(np.unravel_index(cache['background_nodes'][cache['indices']], (N, N, N)), axis=1); q = 3 * len(ijk)
    w = np.clip(V0.support_weights(ijk, N, cache['tau_corners'], cache['cut_plane']), 1e-4, 1.0)
    xyz_face = torch.from_numpy(ijk / (N - 1)).double()
    model = V0.ContractionSuperElement(xyz_face, torch.zeros((0, 3), dtype=torch.float64), proto['r_near'], proto['rho'], proto['rank'], torch.zeros(q, dtype=torch.float64), 1.0, proto['seed'],
                                       decay=proto.get('decay', 0.03), off_scale=proto['off_scale'], gi_scale=proto['gi_scale'])
    model.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=False)['model']); model.eval()
    with torch.no_grad():
        S = model.materialize(); M = model.Ucol(); G = M.T @ M; G.diagonal().add_(1.0)
        L = model.sparse_gg().to_dense()
        def solve_S(Y):
            Z = tri(L.T, Y, False); Z = Z + M @ (M.T @ Z); return tri(L, Z, True)
        rigid = torch.from_numpy(cache['rigid']).double()
        closed = dict(logdet_S=2 * float(model.gg_logdiag.sum()) - float(torch.logdet(G)), rigid_term=logdet_rigid(rigid, solve_S))
        del L
    return S, data, sample, ijk, w, closed, dict(schema='V0', seat=proto['seat'], q=q, d=data.dimension, off_scale=proto['off_scale'], rank=proto['rank'], r_near=proto['r_near'])


def whitened_subspace(R, quotient, E):
    """Orthonormal basis (in the teacher energy metric) of the displacement subspace spanned by the columns of E (q x m)."""
    Qs, _ = torch.linalg.qr(R @ quotient(E)); return Qs


def bucket_report(mu, terms, key, edges, names):
    out = {}
    for lo, hi, name in zip(edges[:-1], edges[1:], names):
        sel = (key >= lo) & (key < hi)
        if sel.any():
            m = mu[sel]; out[name] = dict(modes=int(sel.sum()), divergence=float(terms[sel].sum()), mu_min=float(m.min()), mu_median=float(np.median(m)), mu_max=float(m.max()),
                                         below_0_5=int((m < 0.5).sum()), above_2=int((m > 2).sum()), forward_max=float(np.abs(m - 1).max()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True); ap.add_argument('--checkpoint', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--labels', default='/root/autodl-tmp/CUTFEM_SUPERELEMENT_LABELS/V1_LABELS.json'); ap.add_argument('--seat', type=int, default=None)
    ap.add_argument('--threads', type=int, default=12); ap.add_argument('--k', type=int, default=4); ap.add_argument('--sliver-w', type=float, default=1e-2)
    args = ap.parse_args(); torch.set_num_threads(args.threads); t0 = time.perf_counter()
    proto = json.load(open(Path(args.run) / 'PROTOCOL.json'))
    if proto['schema'] == 'CUTFEM_SUPERELEMENT_V0':
        S, data, sample, ijk, w, closed, info = student_v0(args.run, args.checkpoint)
    else:
        record = next(r for r in json.load(open(args.labels)) if int(r['seat']) == args.seat)
        S, data, sample, ijk, w, closed, info = student_v1(args.run, args.checkpoint, record)
    q, d = info['q'], info['d']; quotient = data.quotient
    print(json.dumps(dict(stage='student', seconds=time.perf_counter() - t0, **info)), flush=True)
    Ahat, cons = quotient_dense(S, quotient); del S
    R = load_upper_factor(sample.reference / 'R_UPPER.npy', d, CPU)
    logdet_A = 2 * float(R.diagonal().log().sum())
    chol, code = torch.linalg.cholesky_ex(Ahat); logdet_Ahat_chol = 2 * float(chol.diagonal().log().sum()) if int(code) == 0 else float('nan'); del chol
    Y = tri(R.T, Ahat, False); del Ahat; W = tri(R.T, Y.T.contiguous(), False); del Y; W = 0.5 * (W + W.T)
    print(json.dumps(dict(stage='whitened', seconds=time.perf_counter() - t0, cholesky_info=int(code))), flush=True)
    mu, V = torch.linalg.eigh(W); del W
    print(json.dumps(dict(stage='eigh', seconds=time.perf_counter() - t0)), flush=True)
    mu_np = mu.numpy(); pos = mu_np > 0
    terms = np.where(pos, mu_np - np.log(np.maximum(mu_np, 1e-300)) - 1.0, np.inf)
    # where every mode lives, measured in the teacher energy metric (Euclidean fractions of the lifted displacement are dominated by the soft teacher modes)
    U = tri(R, V, True)                                                          # displacements u_i = R*^-1 v_i (quotient coordinates, ||.|| = lifted norm)
    stiffness = 1.0 / U.square().sum(0).numpy()                                  # u^T A u / u^T u of every mode: the teacher stiffness the mode sees
    n_s = int((w < args.sliver_w).sum()); sliver_dofs = torch.from_numpy(np.repeat(w < args.sliver_w, 3))
    Es = torch.zeros((q, 3 * n_s), dtype=torch.float64); Es[sliver_dofs.nonzero().squeeze(1), torch.arange(3 * n_s)] = 1.0
    Qs = whitened_subspace(R, quotient, Es); sliver = (Qs.T @ V).square().sum(0).numpy(); del Qs, Es
    P3 = torch.from_numpy(np.kron(V0.coarse_interpolation(ijk, N, 4), np.eye(3)))
    Qc = whitened_subspace(R, quotient, P3); coarse = (Qc.T @ V).square().sum(0).numpy(); del Qc, P3
    node = quotient.lift(U[:, np.concatenate((np.argsort(mu_np)[:args.k], np.argsort(mu_np)[::-1][:args.k]))]).reshape(-1, 3, 2 * args.k).square().sum(1)      # (n, 2k) lifted node norms of the extreme modes
    part = node.sum(0).square() / node.square().sum(0); del U
    cls = np.where(sliver > 0.5, 'sliver', np.where(coarse > 0.5, 'coarse', 'other'))
    groups = {}
    for name in ('sliver', 'coarse', 'other'):
        sel = cls == name
        if sel.any():
            m = mu_np[sel]; groups[name] = dict(modes=int(sel.sum()), divergence=float(terms[sel].sum()), mu_min=float(m.min()), mu_median=float(np.median(m)), mu_max=float(m.max()),
                                               below_0_5=int((m < 0.5).sum()), above_2=int((m > 2).sum()), forward_max=float(np.abs(m - 1).max()))
    edges = [-np.inf, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, np.inf]
    stiff_buckets = bucket_report(mu_np, terms, stiffness, edges, ['<1e-6', '1e-6..1e-5', '1e-5..1e-4', '1e-4..1e-3', '1e-3..1e-2', '>=1e-2'])
    soft_share = float(terms[mu_np < 1].sum() / terms.sum()) if np.isfinite(terms.sum()) else float('nan')
    def modes(idx, offset):
        rows = []
        for c, i in enumerate(idx):
            j = int(node[:, offset + c].argmax())
            rows.append(dict(mu=float(mu_np[i]), term=float(terms[i]), teacher_stiffness=float(stiffness[i]), sliver_fraction=float(sliver[i]), coarse_fraction=float(coarse[i]), participation_nodes=float(part[offset + c]),
                             peak_node=dict(ijk=[int(v) for v in ijk[j]], w=float(w[j]), fraction=float(node[j, offset + c] / node[:, offset + c].sum()))))
        return rows
    order = np.argsort(mu_np)
    logdet_gap = float(np.log(np.maximum(mu_np, 1e-300)).sum()) if pos.all() else float('nan')
    res = dict(run=str(args.run), checkpoint=str(args.checkpoint), **info, cholesky_info=int(code), congruence=cons,
               mu_min=float(mu_np.min()), mu_max=float(mu_np.max()), mu_quantiles={str(p): float(np.quantile(mu_np, p)) for p in (0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999)},
               below_0_5=int((mu_np < 0.5).sum()), below_0_9=int((mu_np < 0.9).sum()), above_1_1=int((mu_np > 1.1).sum()), above_2=int((mu_np > 2).sum()),
               divergence=float(terms.sum()), divergence_per_mode=float(terms.mean()), whitened_rms=float(np.sqrt(np.mean((mu_np - 1) ** 2))),
               forward_bound=float(np.abs(mu_np - 1).max()), inverse_bound=float(np.abs(1 / mu_np - 1).max()) if pos.all() else float('inf'),
               logdet_gap_eigen=logdet_gap, logdet_gap_cholesky=logdet_Ahat_chol - logdet_A, logdet_gap_closed=closed['logdet_S'] + closed['rigid_term'] - logdet_A, closed=closed,
               groups=groups, stiffness_buckets=stiff_buckets, divergence_share_oversoft=soft_share, softest=modes(order[:args.k], 0), stiffest=modes(order[::-1][:args.k], args.k), sliver_node_fraction=float((w < args.sliver_w).mean()), seconds=time.perf_counter() - t0)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True); tag = f"{Path(args.run).name}_{Path(args.checkpoint).stem}_{info['seat']:04d}"
    write_json(out / f'SPECTRUM_{tag}.json', json_finite(res))
    np.savez(out / f'SPECTRUM_{tag}.npz', mu=mu_np, sliver=sliver, coarse=coarse, stiffness=stiffness)
    print(json.dumps(dict(stage='done', tag=tag, mu_min=res['mu_min'], mu_max=res['mu_max'], D=res['divergence'], D_per_mode=res['divergence_per_mode'], forward=res['forward_bound'], inverse=res['inverse_bound'],
                          logdet_gap=dict(eig=res['logdet_gap_eigen'], chol=res['logdet_gap_cholesky'], closed=res['logdet_gap_closed']), groups={k: (v['modes'], round(v['divergence'], 3), round(v['mu_min'], 4), round(v['mu_max'], 3)) for k, v in groups.items()}, stiffness={k: (v['modes'], round(v['divergence'], 3), round(v['mu_min'], 4), round(v['mu_max'], 3)) for k, v in stiff_buckets.items()}, oversoft_share=round(soft_share, 3),
                          seconds=res['seconds'])), flush=True)


if __name__ == '__main__':
    main()
