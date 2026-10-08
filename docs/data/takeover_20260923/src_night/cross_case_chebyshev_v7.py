"""Cross-case port of Codex's fixed two-history Chebyshev strain hierarchy (stage A: finite witness).

Same algorithm as HIERARCHY_STRAIN_20260922_01/CHEBYSHEV_01, applied unchanged to other cases:
  * rigid6 4h/8h patch hierarchy from HIERARCHICAL_MECHANICS source_assets_01/prepare_hierarchy.py
  * symmetric-strain enrichment from HIERARCHY_STRAIN source_assets_01/build_strain_hierarchy.py
  * V-cycle + fixed Chebyshev recurrence (DESIGN_A=0.01, upper 1) from source_chebyshev_full_01
  * finite evaluation exactly as evaluate_chebyshev_strain.py: 7 original probes + adjoint/variational
    checks, 16 validation energy directions, 20-step Krylov witness (seed 92219), original-action replay.
Frozen modules are imported read-only from their original paths. The only deviation: the coarsest
dense-factor size guard (800) is raised to 8192 in a copied hierarchy module, because FULL/moderate
cells have 224-354 8h patches. Nothing trained. No mode removal, jitter, clipping or pseudoinverse.
"""
import argparse, hashlib, importlib.util, json, os, resource, socket, sys, threading, time, traceback
from pathlib import Path
import numpy as np
from scipy import sparse
import torch

ROOT = Path('/root/autodl-tmp/CUTFEM_FRESH_GP_20260921')
DG = ROOT / 'diagnostics'
MN = DG / 'MECHANICS_NETWORK_20260922_01'
HM = DG / 'HIERARCHICAL_MECHANICS_20260922_01'
HS = DG / 'HIERARCHY_STRAIN_20260922_01'
HERE = Path(__file__).resolve().parent
RUNS = {'fresh_train_0003_d0_v0': 'HEAVY_FREE_01', 'fresh_train_0003_full': 'FULL_FREE_02',
        'fresh_train_0013_d0_v1': 'MODERATE_FREE_01'}
DEPTHS = (16, 32, 64)
SEED, ITERS = 92219, 20
FROZEN = {
    'prepare_hierarchy': HM / 'source_assets_01/prepare_hierarchy.py',
    'build_strain_hierarchy': HS / 'source_assets_01/build_strain_hierarchy.py',
    'chebyshev_strain_network': HS / 'source_chebyshev_full_01/chebyshev_strain_network.py',
    'evaluate_chebyshev_strain': HS / 'source_chebyshev_01/evaluate_chebyshev_strain.py',
    'mechanics_network': MN / 'source_v1/mechanics_network.py',
    'run_mechanics': MN / 'source_v1/run_mechanics.py',
    'evaluate_mechanics': MN / 'source_eval_03/evaluate_mechanics.py',
}
# Codex's own heavy-case finite result, used only to confirm this port reproduces it.
HEAVY_CONTROL = HS / 'CHEBYSHEV_01'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 2**20), b''):
            h.update(b)
    return h.hexdigest()


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def pair_blocks(A, theta, max_nodes=4):
    """Fine smoother blocks: nodes joined by node-block strength >= theta share one exactly inverted block.

    s_ij = ||A_ij||_F / sqrt(||A_ii||_F ||A_jj||_F). Union-find over strong edges; clusters above max_nodes
    are refused (recorded), never truncated. Returns right-padded int64 indices [blocks, 3*max_nodes] (-1 pad)
    and upper inverse-Cholesky factors, the same format Codex's BlockJacobi takes, plus a record.
    """
    from scipy import linalg
    from scipy.sparse.csgraph import connected_components
    Ab = sparse.csr_matrix(A).tobsr(blocksize=(3, 3))
    nb = Ab.shape[0] // 3
    norms = np.sqrt(np.square(Ab.data).sum(axis=(1, 2)))
    rows = np.repeat(np.arange(nb), np.diff(Ab.indptr))
    cols = Ab.indices
    diag = np.zeros(nb)
    on = rows == cols
    diag[rows[on]] = norms[on]
    s = norms / np.sqrt(diag[rows] * diag[cols])
    keep = (rows != cols) & (s >= theta)
    G = sparse.csr_matrix((np.ones(int(keep.sum())), (rows[keep], cols[keep])), shape=(nb, nb))
    n, labels = connected_components(G, directed=False)
    sizes = np.bincount(labels)
    if sizes.max() > max_nodes:
        raise ValueError(f'STRONG_CLUSTER_TOO_LARGE:{int(sizes.max())}')
    width = 3 * int(sizes.max())
    order = np.argsort(labels, kind='stable')
    ptr = np.r_[0, np.cumsum(sizes)]
    indices = np.full((n, width), -1, dtype=np.int64)
    factors = np.zeros((n, width, width))
    Acsr = sparse.csr_matrix(A)
    flist = []
    for k in range(n):
        nodes = np.sort(order[ptr[k]:ptr[k + 1]])
        dofs = (3 * nodes[:, None] + np.arange(3)).ravel()
        block = Acsr[dofs][:, dofs].toarray()
        chol = linalg.cholesky(block, lower=True, check_finite=True)
        F = linalg.solve_triangular(chol.T, np.eye(len(dofs)), lower=False)
        indices[k, :len(dofs)] = dofs
        factors[k, :len(dofs), :len(dofs)] = F
        flist.append((dofs, F))
    r_, c_, v_ = [], [], []
    for dofs, F in flist:
        r_.append(np.repeat(dofs, len(dofs))); c_.append(np.tile(dofs, len(dofs))); v_.append(F.ravel())
    Fg = sparse.csr_matrix((np.concatenate(v_), (np.concatenate(r_), np.concatenate(c_))), shape=A.shape)
    bound = float(np.asarray(abs((Fg.T @ Acsr @ Fg).tocsr()).sum(axis=1)).max())
    record = dict(theta=theta, blocks=int(n), nodes=int(nb), size_histogram=np.bincount(sizes).tolist(),
                  merged_nodes=int((sizes[labels] > 1).sum()), row_sum_bound=bound, omega0=1 / bound)
    return indices, factors, bound, record


def asset_dir(case, asset_root):
    """Codex's three registered cases, or a case prepared by the J1 asset step (same frozen compiler)."""
    if asset_root is None:
        if case not in RUNS:
            raise ValueError('UNREGISTERED_CASE_NEEDS_ASSET_ROOT')
        return MN / 'assets' / case
    return Path(asset_root) / case


def chebyshev_coefficients_interval(cycles, design_a, upper=1.0):
    """Codex's stable Chebyshev ratio recurrence with the lower design end as a parameter."""
    center = (upper + design_a) / 2; radius = (upper - design_a) / 2; sigma = center / radius
    alpha, beta = [1 / center], [0.0]; rho = 1 / sigma
    for _ in range(1, cycles):
        nxt = 1 / (2 * sigma - rho); alpha.append(2 * nxt / radius); beta.append(rho * nxt); rho = nxt
    return alpha, beta


def lanczos_lower(Aop, Bop, n, device, iters, seed=20260923):
    """PCG on B A from a random right-hand side; Ritz values of the Lanczos tridiagonal (upper estimates of
    the extreme low eigenvalues of B A)."""
    g = torch.Generator(device=device).manual_seed(seed)
    b = torch.randn((n, 1), dtype=torch.float64, device=device, generator=g)
    x = torch.zeros_like(b); r = b.clone(); z = Bop(r); p = z.clone(); rz = (r * z).sum()
    al, be = [], []
    for _ in range(iters):
        Ap = Aop(p); alpha = rz / (p * Ap).sum(); x = x + alpha * p; r = r - alpha * Ap
        z = Bop(r); rzn = (r * z).sum(); beta = rzn / rz
        al.append(float(alpha)); be.append(float(beta)); rz = rzn; p = z + beta * p
    k = len(al); Tm = np.zeros((k, k))
    for j in range(k):
        Tm[j, j] = 1 / al[j] + (be[j - 1] / al[j - 1] if j > 0 else 0)
        if j + 1 < k:
            Tm[j, j + 1] = Tm[j + 1, j] = np.sqrt(be[j]) / al[j]
    return np.linalg.eigvalsh(Tm)


def main(args):
    torch.set_num_threads(args.threads)
    from threadpoolctl import threadpool_limits
    threadpool_limits(args.threads)
    import psutil
    case = args.case
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    proc = psutil.Process()
    stop = threading.Event()

    def event(kind, **data):
        row = dict(event=kind, elapsed=round(time.monotonic() - start, 3), **data)
        print(json.dumps(row, allow_nan=False), flush=True)
        with (out / 'PROGRESS.jsonl').open('a') as f:
            f.write(json.dumps(row, allow_nan=False) + '\n')

    def guard():
        while not stop.wait(1.0):
            rss = proc.memory_info().rss
            gpu = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            el = time.monotonic() - start
            if el > args.wall or rss > args.rss_gib * 2**30 or gpu > args.gpu_gib * 2**30:
                write(out / 'RESOURCE_STOP.json', dict(seconds=el, rss=rss, gpu=gpu))
                os._exit(79)
    threading.Thread(target=guard, daemon=True).start()

    frozen_sha = {k: sha(p) for k, p in FROZEN.items()}
    port_sha = {p.name: sha(p) for p in HERE.glob('*.py')}
    # Module search paths the frozen files expect.
    sys.path.insert(0, str(HS / 'source_chebyshev_full_01'))
    sys.path.insert(0, str(MN / 'source_v1'))
    prep = load('xc_prepare_hierarchy', FROZEN['prepare_hierarchy'])
    strain = load('xc_build_strain_hierarchy', FROZEN['build_strain_hierarchy'])
    runner = load('xc_run_mechanics', FROZEN['run_mechanics'])
    evaluator = load('xc_evaluate_mechanics', FROZEN['evaluate_mechanics'])
    cheb = load('xc_chebyshev_strain_network', FROZEN['chebyshev_strain_network'])
    evalcheb = load('xc_evaluate_chebyshev_strain', FROZEN['evaluate_chebyshev_strain'])
    hier = load('xc_hierarchy_strain_network', HERE / 'hierarchy_strain_network_xcase.py')

    asset = asset_dir(case, args.asset_root)
    compiled = asset / 'compiled'
    target = ROOT / 'targets' / (case + '_v1')
    refdir = MN / RUNS[case] if args.asset_root is None else Path(args.asset_root) / (case + '_REFERENCE')
    reference_path = refdir / 'REFERENCE_STRUCTURE.npz'
    zfile = refdir / 'VALIDATION_Z.npy'
    meta = json.loads((asset / 'RESULT.json').read_text())
    if meta['status'] != 'PASS':
        raise ValueError('UNQUALIFIED_ASSET')
    expected = dict(meta['output_sha256'])
    sup = asset / 'ASSET_SUPPLEMENT_01.json'
    if sup.exists():
        extra = json.loads(sup.read_text())
        expected.update(extra['output_sha256'])
    verified = {}
    for name in ['A.npz', 'C.npz', 'D.npz', 'Q_RIGID.npy', 'INTERIOR_RIGID.npy', 'INTERIOR_POINTS.npy',
                 'QUALIFICATION_PROBES.npz']:
        p = compiled / name
        verified[str(p)] = sha(p)
        key = 'compiled/' + name
        if key in expected and verified[str(p)] != expected[key]:
            raise ValueError('ASSET_SHA:' + name)
    manifest = json.loads((target / 'MANIFEST.json').read_text())
    for name in ['REFERENCE_RQ.npy', 'RIGID_Q.npy']:
        p = target / name
        verified[str(p)] = sha(p)
        if verified[str(p)] != manifest[name]:
            raise ValueError('TARGET_SHA:' + name)
    for p in [reference_path, zfile]:
        verified[str(p)] = sha(p)
    write(out / 'START.json', dict(case=case, pid=os.getpid(), host=socket.gethostname(),
                                   device=torch.cuda.get_device_name(), torch=torch.__version__,
                                   frozen_source_sha256={str(FROZEN[k]): v for k, v in frozen_sha.items()},
                                   port_source_sha256=port_sha, input_sha256=verified,
                                   deviation='coarsest dense-factor size guard 800 -> 8192; optional strong-pair fine smoother blocks (--pair-theta); optional --asset-root', pair_theta=args.pair_theta, asset_root=args.asset_root,
                                   depths=list(args.depths), witness_seed=SEED, witness_iterations=ITERS,
                                   threads=args.threads, wall=args.wall, rss_gib=args.rss_gib,
                                   gpu_gib=args.gpu_gib, training=False))

    A, C, D = [sparse.load_npz(compiled / (n + '.npz')).tocsr() for n in ['A', 'C', 'D']]
    points = np.load(compiled / 'INTERIOR_POINTS.npy', allow_pickle=False)
    ref = np.load(reference_path, allow_pickle=False)
    o4, o8 = ref['level0_owner'], ref['level1_owner']
    event('LOADED', interior=A.shape[0], boundary=D.shape[0], A_nnz=int(A.nnz),
          patches4=int(len(np.unique(o4))), patches8=int(len(np.unique(o8))))

    t = time.perf_counter()
    rigid = prep.build_hierarchy(A, points, o4, o8, 'rigid6')
    rigid_seconds = time.perf_counter() - t
    event('RIGID6_BUILT', dims=rigid['result']['dimensions'], seconds=rigid_seconds)
    t = time.perf_counter()
    h = strain.build_hierarchy(A, points, o4, o8, rigid['P1'], rigid['P2direct'], rigid['P12'],
                               rigid['BLOCKS_LEVEL1'], rigid['BLOCKS_LEVEL2'], prep.block_data,
                               max_coarsest=hier.COARSEST_LIMIT)
    strain_seconds = time.perf_counter() - t
    adir = out / 'STRAIN_ASSETS'
    adir.mkdir()
    for name in ('P1', 'P12', 'P2direct', 'A1', 'A2'):
        sparse.save_npz(adir / (name + '.npz'), h[name], compressed=False)
    for name in ('BLOCKS_LEVEL1', 'BLOCKS_LEVEL2'):
        np.savez(adir / (name + '.npz'), **h[name])
    np.save(adir / 'COARSE_FULL_FACTOR.npy', h['COARSE_FULL_FACTOR'], allow_pickle=False)
    asset_record = dict(dimensions=h['dimensions'], rigid6_dimensions=rigid['result']['dimensions'],
                        checks=h['checks'], omega1=h['omega1'],
                        block_level1_scaled_abs_row_bound=h['block_level1_scaled_abs_row_bound'],
                        block_level2_scaled_abs_row_bound=h['block_level2_scaled_abs_row_bound'],
                        rigid6_seconds=rigid_seconds, strain_seconds=strain_seconds,
                        output_sha256={p.name: sha(p) for p in adir.iterdir()})
    if case == 'fresh_train_0003_d0_v0':
        codex = json.loads((HS / 'ASSETS_01/RESULT.json').read_text())
        same = {}
        for name, digest in codex['output_sha256'].items():
            mine = adir / name
            if mine.exists():
                if name.endswith('.npz'):
                    a, b = np.load(mine), np.load(HS / 'ASSETS_01' / name)
                    same[name] = all(k in b.files and np.array_equal(a[k], b[k]) for k in a.files)
                elif name.endswith('.npy'):
                    same[name] = bool(np.array_equal(np.load(mine), np.load(HS / 'ASSETS_01' / name)))
        asset_record['codex_heavy_assets_array_equal'] = same
        asset_record['codex_dimensions'] = codex['dimensions']
        asset_record['codex_omega1'] = codex['omega1']
    write(out / 'STRAIN_ASSETS.json', asset_record)
    event('STRAIN_BUILT', dims=h['dimensions'], seconds=strain_seconds,
          codex_equal=asset_record.get('codex_heavy_assets_array_equal'))

    device = torch.device('cuda:0')
    tt = lambda x: torch.as_tensor(x, dtype=torch.float64, device=device)
    t = time.perf_counter()
    local, bound0, error0 = runner.factors_and_bound(A)
    Ag, Cg, Dg = [runner.tensor_sparse(x, device) for x in [A, C, D]]
    ub, ui = runner.orthonormalize_rigid_pair(tt(np.load(compiled / 'Q_RIGID.npy')),
                                              tt(np.load(compiled / 'INTERIOR_RIGID.npy')))
    mats = {n: runner.tensor_sparse(h[n], device) for n in ['P1', 'A1', 'P12', 'A2']}
    pair_record = None
    if args.pair_theta is None:
        fine_idx, fine_F, omega0 = torch.arange(A.shape[0], device=device).reshape(-1, 3), tt(local), 1 / bound0
    else:
        pidx, pF, pbound, pair_record = pair_blocks(A, args.pair_theta)
        write(out / 'PAIR_BLOCKS.json', pair_record)
        fine_idx, fine_F, omega0 = torch.as_tensor(pidx, dtype=torch.long, device=device), tt(pF), 1 / pbound
    core = hier.load_frozen_core(FROZEN['mechanics_network'], frozen_sha['mechanics_network'])
    correction = hier.HierarchyCorrection(Ag, mats['P1'], mats['A1'], mats['P12'], mats['A2'], fine_idx, fine_F,
        torch.as_tensor(h['BLOCKS_LEVEL1']['block_indices'], dtype=torch.long, device=device),
        tt(h['BLOCKS_LEVEL1']['block_factors']), tt(h['COARSE_FULL_FACTOR']), omega0, h['omega1'], mode='vcycle')
    base = core.MechanicsNetwork(Ag, Cg, Dg, ub, ui, tt(local), (), layers=1, step_scale=1.0,
                                 learnable=False, share_layers=True)
    base = hier.install_hierarchy(base, correction, cycles=1)
    R = tt(np.load(target / 'REFERENCE_RQ.npy'))
    quotient = runner.Quotient(np.load(target / 'RIGID_Q.npy'), device)
    zval = tt(np.load(zfile))
    qval = quotient.lift(torch.linalg.solve_triangular(R, zval, upper=True))
    probes = np.load(compiled / 'QUALIFICATION_PROBES.npz')
    probe_t = {k: tt(probes[key]) for k, key in [('q', 'q'), ('zref', 'z'), ('reference_force', 'reference_force')]}
    torch.cuda.synchronize()
    event('MODEL_READY', setup_seconds=time.perf_counter() - t, omega0=omega0, omega1=h['omega1'],
          fine_inverse_error=error0, cuda_bytes=torch.cuda.memory_allocated(),
          physical_dimension=int(R.shape[0]), validation_directions=int(zval.shape[1]))

    design = dict(design_a=0.01, source='codex_fixed_constant')
    depths = list(args.depths)
    if args.adaptive:
        # Encode-time step: estimate the slow end of B A once, choose the Chebyshev interval and depth.
        t_l = time.perf_counter()
        with torch.no_grad():
            rv = lanczos_lower(lambda v: torch.sparse.mm(Ag, v), base.corrections[0], Ag.shape[0], device, args.lanczos)
        a_design = float(args.safety * rv.min())
        depth = int(np.ceil(np.log(2 / args.tolerance) / (2 * np.sqrt(a_design))))
        design = dict(design_a=a_design, source='lanczos', ritz_min=float(rv.min()), ritz_smallest5=rv[:5].tolist(),
                      lanczos_iterations=args.lanczos, safety=args.safety, tolerance=args.tolerance,
                      chosen_depth=depth, lanczos_seconds=time.perf_counter() - t_l)
        depths = [depth]
        event('ADAPTIVE_DESIGN', **design)
    write(out / 'CHEBYSHEV_DESIGN.json', design)
    rows = []
    for depth in depths:
        torch.cuda.reset_peak_memory_stats()
        model = cheb.ChebyshevStrainNetwork(base, cycles=depth)
        if design['source'] != 'codex_fixed_constant':
            ca, cb = chebyshev_coefficients_interval(depth, design['design_a'])
            model.chebyshev_alpha.copy_(torch.tensor(ca, dtype=torch.float64))
            model.chebyshev_beta.copy_(torch.tensor(cb, dtype=torch.float64))
        algebra = evalcheb.complete_algebra(model, probe_t, evaluator)
        with torch.no_grad():
            values = 2 * model.energy(qval)
        witness, white = evaluator.spectral_witness(model, R, quotient.lift, quotient.reduce,
                                                    seed=SEED, maxiter=ITERS, return_vector=True)
        replay, vectors = evalcheb.replay_witness(model, R, quotient, white)
        np.save(out / f'WITNESS_WHITE_{depth:03d}.npy', white.cpu().numpy())
        refE = np.asarray(algebra['reference_energy'], dtype=float)
        predE = np.asarray(algebra['predicted_energy'], dtype=float)
        positive = bool(np.isfinite(refE).all() and np.isfinite(predE).all() and (refE > 0).all())
        original7 = (predE / refE).tolist() if positive else []
        ratios = list(values.cpu().tolist()) + list(replay['actual_rayleigh']) + original7
        finite = all(v is not None and np.isfinite(v) for v in ratios)
        eligible = bool(positive and finite and all(.97 <= v <= 1.03 for v in ratios)
                        and all(algebra['numerical_checks'].values()))
        row = dict(case=case, cycles=depth, algebra=algebra, independent16_energy_ratio=values.cpu().tolist(),
                   spectral_witness=witness, original_action_witness_replay=replay,
                   original7_energy_ratio=original7, found_min=min(ratios) if finite else None,
                   found_max=max(ratios) if finite else None,
                   eligible_for_separate_full_spectrum=eligible,
                   cuda_peak_bytes=torch.cuda.max_memory_allocated(), full_spectrum=False, D_over_d=None,
                   scientific_pass=False, training=False)
        write(out / f'CYCLES_{depth:03d}.json', row)
        rows.append(row)
        event('DEPTH_DONE', cycles=depth, witness=replay['actual_rayleigh'][0],
              random16_max=float(values.max()), original7=original7,
              action7_seconds=algebra['action_panel_seconds'], witness_seconds=witness['seconds'],
              numerical_ok=all(algebra['numerical_checks'].values()), eligible=eligible)
    control = None
    if case == 'fresh_train_0003_d0_v0' and args.pair_theta is None:
        control = {}
        for depth in args.depths:
            c = json.loads((HEAVY_CONTROL / f'CYCLES_{depth:03d}.json').read_text())
            mine = next(r for r in rows if r['cycles'] == depth)
            control[depth] = dict(codex_witness=c['original_action_witness_replay']['actual_rayleigh'][0],
                                  port_witness=mine['original_action_witness_replay']['actual_rayleigh'][0],
                                  codex_random16_max=max(c['independent16_energy_ratio']),
                                  port_random16_max=max(mine['independent16_energy_ratio']))
    eligible_depths = [r['cycles'] for r in rows if r['eligible_for_separate_full_spectrum']]
    result = dict(status='CROSS_CASE_FINITE_COMPLETE', case=case, pair_blocks=pair_record,
                  hierarchy_dimensions=h['dimensions'],
                  summary=[dict(cycles=r['cycles'], witness=r['original_action_witness_replay']['actual_rayleigh'][0],
                                found_max=r['found_max'], found_min=r['found_min'],
                                action7_seconds=r['algebra']['action_panel_seconds'],
                                eligible=r['eligible_for_separate_full_spectrum']) for r in rows],
                  selected_full_spectrum_depth=min(eligible_depths) if eligible_depths else None, chebyshev_design=design,
                  heavy_reproduction_control=control, seconds=time.monotonic() - start,
                  peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                  scientific_pass=False, full_spectrum=False, D_over_d=None,
                  scope='Finite witness only. Exact FE local matrices. Full spectrum is a separate stage.')
    write(out / 'RESULT.json', result)
    write(out / 'SHA256SUMS.json', {p.name: sha(p) for p in out.iterdir() if p.is_file()})
    stop.set()
    event('COMPLETE', **{k: result[k] for k in ('selected_full_spectrum_depth', 'seconds')})


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', required=True)
    ap.add_argument('--asset-root', default=None)
    ap.add_argument('--depths', type=int, nargs='+', default=list(DEPTHS))
    ap.add_argument('--adaptive', action='store_true')
    ap.add_argument('--lanczos', type=int, default=80)
    ap.add_argument('--safety', type=float, default=0.5)
    ap.add_argument('--tolerance', type=float, default=1e-3)
    ap.add_argument('--output', required=True)
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--pair-theta', type=float, default=None)
    ap.add_argument('--wall', type=float, default=7200)
    ap.add_argument('--rss-gib', type=float, default=48)
    ap.add_argument('--gpu-gib', type=float, default=28)
    a = ap.parse_args()
    try:
        main(a)
    except BaseException as exc:
        o = Path(a.output)
        if o.is_dir() and not (o / 'FAILURE.json').exists():
            write(o / 'FAILURE.json', dict(error=repr(exc), traceback=traceback.format_exc()))
        raise
