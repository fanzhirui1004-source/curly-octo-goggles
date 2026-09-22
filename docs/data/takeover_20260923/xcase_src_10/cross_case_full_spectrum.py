"""Stage B: complete spectrum of the fixed Chebyshev strain hierarchy for a nominated case/depth.

Differs from Codex's full_spectrum_chebyshev.py only in how columns are produced: the dense internal
extension H (interior x d) does not fit a 32 GB GPU for FULL/moderate cells, so every column of the
whitened operator is taken directly from the model's original complete boundary action
(extension + exact two-history adjoint + energy readout), in batches. Codex already cross-checked this
direct action against the energy Gram to 1e-9 on the heavy case. All d physical directions are kept.
"""
import argparse, json, os, resource, sys, threading, time, traceback
from pathlib import Path
import numpy as np
from scipy import linalg, sparse
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cross_case_chebyshev as X  # noqa: E402


def main(args):
    torch.set_num_threads(args.threads)
    from threadpoolctl import threadpool_limits
    threadpool_limits(args.threads)
    import psutil
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    proc, stop = psutil.Process(), threading.Event()
    finite = Path(args.finite)
    fres = json.loads((finite / 'RESULT.json').read_text())
    if fres['case'] != args.case or fres['selected_full_spectrum_depth'] != args.depth:
        raise ValueError('DEPTH_NOT_THE_NOMINATED_MINIMUM')

    def event(kind, **data):
        row = dict(event=kind, elapsed=round(time.monotonic() - start, 3), **data)
        print(json.dumps(row, allow_nan=False), flush=True)
        with (out / 'PROGRESS.jsonl').open('a') as f:
            f.write(json.dumps(row, allow_nan=False) + '\n')

    def guard():
        while not stop.wait(1.0):
            rss, gpu = proc.memory_info().rss, torch.cuda.memory_allocated()
            if time.monotonic() - start > args.wall or rss > args.rss_gib * 2**30 or gpu > args.gpu_gib * 2**30:
                X.write(out / 'RESOURCE_STOP.json', dict(seconds=time.monotonic() - start, rss=rss, gpu=gpu))
                os._exit(79)
    threading.Thread(target=guard, daemon=True).start()

    sys.path.insert(0, str(X.HS / 'source_chebyshev_full_01'))
    sys.path.insert(0, str(X.MN / 'source_v1'))
    runner = X.load('xc_run_mechanics', X.FROZEN['run_mechanics'])
    cheb = X.load('xc_chebyshev_strain_network', X.FROZEN['chebyshev_strain_network'])
    hier = X.load('xc_hierarchy_strain_network', HERE / 'hierarchy_strain_network_xcase.py')
    case = args.case
    compiled = X.asset_dir(case, args.asset_root) / 'compiled'
    target = X.ROOT / 'targets' / (case + '_v1')
    adir = finite / 'STRAIN_ASSETS'
    ameta = json.loads((finite / 'STRAIN_ASSETS.json').read_text())
    for name, digest in ameta['output_sha256'].items():
        if X.sha(adir / name) != digest:
            raise ValueError('STRAIN_ASSET_SHA:' + name)
    device = torch.device('cuda:0')
    tt = lambda x: torch.as_tensor(x, dtype=torch.float64, device=device)
    A, C, D = [sparse.load_npz(compiled / (n + '.npz')).tocsr() for n in ['A', 'C', 'D']]
    local, bound0, _ = runner.factors_and_bound(A)
    h = {n: sparse.load_npz(adir / (n + '.npz')).tocsr() for n in ['P1', 'A1', 'P12', 'A2']}
    b1 = np.load(adir / 'BLOCKS_LEVEL1.npz')
    ub, ui = runner.orthonormalize_rigid_pair(tt(np.load(compiled / 'Q_RIGID.npy')),
                                              tt(np.load(compiled / 'INTERIOR_RIGID.npy')))
    Ag = runner.tensor_sparse(A, device)
    mats = {n: runner.tensor_sparse(h[n], device) for n in h}
    pair_record = None
    if args.pair_theta is None:
        fine_idx, fine_F, omega0 = torch.arange(A.shape[0], device=device).reshape(-1, 3), tt(local), 1 / bound0
    else:
        pidx, pF, pbound, pair_record = X.pair_blocks(A, args.pair_theta)
        frec = json.loads((finite / 'PAIR_BLOCKS.json').read_text())
        if frec != json.loads(json.dumps(pair_record)):
            raise ValueError('PAIR_BLOCKS_DIFFER_FROM_FINITE_RUN')
        fine_idx, fine_F, omega0 = torch.as_tensor(pidx, dtype=torch.long, device=device), tt(pF), 1 / pbound
    core = hier.load_frozen_core(X.FROZEN['mechanics_network'], X.sha(X.FROZEN['mechanics_network']))
    correction = hier.HierarchyCorrection(Ag, mats['P1'], mats['A1'], mats['P12'], mats['A2'], fine_idx, fine_F,
        torch.as_tensor(b1['block_indices'], dtype=torch.long, device=device), tt(b1['block_factors']),
        tt(np.load(adir / 'COARSE_FULL_FACTOR.npy')), omega0, ameta['omega1'], mode='vcycle')
    base = core.MechanicsNetwork(Ag, runner.tensor_sparse(C, device), runner.tensor_sparse(D, device), ub, ui,
                                 tt(local), (), layers=1, step_scale=1.0, learnable=False, share_layers=True)
    base = hier.install_hierarchy(base, correction, cycles=1)
    model = cheb.ChebyshevStrainNetwork(base, cycles=args.depth)
    R = tt(np.load(target / 'REFERENCE_RQ.npy'))
    quotient = runner.Quotient(np.load(target / 'RIGID_Q.npy'), device)
    d = R.shape[0]
    X.write(out / 'PROTOCOL.json', dict(case=case, depth=args.depth, physical_dimension=d, batch=args.batch,
            definition='G[:,j] = R^-T B S_hat B^T R^-1 e_j via original complete boundary action', pair_theta=args.pair_theta,
            finite_nomination=str(finite / 'RESULT.json'), finite_nomination_sha256=X.sha(finite / 'RESULT.json'),
            port_source_sha256={p.name: X.sha(p) for p in HERE.glob('*.py')},
            modes_removed=0, jitter=False, clipping=False, pseudoinverse=False, training=False))
    G = np.lib.format.open_memmap(out / 'G_RAW.npy', mode='w+', dtype=np.float64, shape=(d, d), fortran_order=True)
    t0 = time.perf_counter()
    with torch.no_grad():
        for lo in range(0, d, args.batch):
            hi = min(d, lo + args.batch)
            eye = R.new_zeros((d, hi - lo))
            eye[torch.arange(lo, hi, device=device), torch.arange(hi - lo, device=device)] = 1
            q = quotient.lift(torch.linalg.solve_triangular(R, eye, upper=True))
            col = torch.linalg.solve_triangular(R.T, quotient.reduce(model(q)), upper=False)
            if not torch.isfinite(col).all():
                raise ValueError('NONFINITE_ACTION')
            G[:, lo:hi] = col.cpu().numpy()
            if lo == 0 or hi == d or (hi // args.batch) % 8 == 0:
                event('COLUMNS', done=hi, total=d, seconds=time.perf_counter() - t0,
                      gpu_peak=torch.cuda.max_memory_allocated())
    G.flush()
    action_seconds = time.perf_counter() - t0
    num = den = 0.0
    S = np.lib.format.open_memmap(out / 'G_SYMMETRIC.npy', mode='w+', dtype=np.float64, shape=(d, d), fortran_order=True)
    for lo in range(0, d, 256):
        hi = min(d, lo + 256)
        row, tr = np.array(G[lo:hi, :]), np.array(G[:, lo:hi].T)
        num += float(np.square(row - tr).sum()); den += float(np.square(row).sum())
        S[lo:hi, :] = (row + tr) / 2
    asym = float(np.sqrt(num / den))
    S.flush()
    event('SYMMETRY', relative_frobenius=asym)
    if asym > 1e-10:
        raise ValueError('ASYMMETRY_EXCEEDS_1e-10')
    t = time.perf_counter()
    w, V = linalg.eigh(np.array(S, order='F'), driver='evd', check_finite=False, overwrite_a=True)
    eig_seconds = time.perf_counter() - t
    np.save(out / 'ALL_STIFFNESS_EIGENVALUES.npy', w)
    ext = V[:, [0, -1]].copy()
    np.save(out / 'EXTREME_WHITE_EIGENVECTORS.npy', ext)
    top = np.argsort(w)[-32:]
    np.save(out / 'TOP32_WHITE_EIGENVECTORS.npy', V[:, top])
    del V
    with torch.no_grad():
        z = tt(ext)
        qe = quotient.lift(torch.linalg.solve_triangular(R, z, upper=True))
        act = torch.linalg.solve_triangular(R.T, quotient.reduce(model(qe)), upper=False)
        vals = tt(w[[0, -1]])
        resid = (torch.linalg.vector_norm(act - z * vals, dim=0) / (vals.abs() * torch.linalg.vector_norm(z, dim=0))).cpu().tolist()
    mu = w
    res = dict(status='FULL_SPECTRUM_FINISHED', case=case, depth=args.depth, physical_dimension=d,
               mu_min=float(mu.min()), mu_max=float(mu.max()),
               outside_work_gate=int(np.sum((mu < .9) | (mu > 1.1))),
               outside_target_gate=int(np.sum((mu < .97) | (mu > 1.03))),
               work_gate=bool(mu.min() >= .9 and mu.max() <= 1.1),
               target_gate_spectrum=bool(mu.min() >= .97 and mu.max() <= 1.03),
               D_over_d=float(np.mean(mu - 1 - np.log(mu))), D_gate=1e-4,
               asymmetry=asym, extreme_direct_action_relative_residual=resid,
               action_seconds=action_seconds, eig_seconds=eig_seconds, seconds=time.monotonic() - start,
               peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
               cuda_peak_bytes=torch.cuda.max_memory_allocated(), scientific_pass=False,
               scope='Complete boundary spectrum with exact FE local matrices; not a learned-coefficient result')
    res['target_gate'] = bool(res['target_gate_spectrum'] and res['D_over_d'] < 1e-4)
    X.write(out / 'RESULT.json', res)
    stop.set()
    event('COMPLETE', **{k: res[k] for k in ('mu_min', 'mu_max', 'D_over_d', 'work_gate', 'target_gate')})


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', required=True)
    ap.add_argument('--finite', required=True)
    ap.add_argument('--depth', type=int, required=True)
    ap.add_argument('--asset-root', default=None)
    ap.add_argument('--output', required=True)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--pair-theta', type=float, default=None)
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--wall', type=float, default=5 * 3600)
    ap.add_argument('--rss-gib', type=float, default=48)
    ap.add_argument('--gpu-gib', type=float, default=28)
    a = ap.parse_args()
    try:
        main(a)
    except BaseException as exc:
        o = Path(a.output)
        if o.is_dir() and not (o / 'FAILURE.json').exists():
            X.write(o / 'FAILURE.json', dict(error=repr(exc), traceback=traceback.format_exc()))
        raise
