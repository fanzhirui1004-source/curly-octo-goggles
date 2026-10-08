"""The tau-derivative gate, prediction side: does the NETWORK's compliance carry the teacher's
tau sensitivity?

The teacher's side is already measured (`superelement/objective/tau_gate.py`, recorded in
docs/HEAD_AB_AND_TAU_20260917.md): for seat 0328, five admitted operators at tau_corners scaled
by 1+eps, eps in {0, +-1e-4, +-1e-3}, the platen observable trace(H^-1) has
d ln(trace H^-1)/d eps = -2.2784 with a Richardson ratio central(1e-4)/central(1e-3) = 0.999997.
The teacher's derivative exists and is clean even though one cut cell is born between eps = 0 and
+1e-4 and takes mu_max from 1 to 1.598.

What this script adds is the network's side of the same five points, on two observables:

  platen      trace(H^-1) from the predicted M_q through `equi.platen`, the teacher's own fixture
  compliance  c(f) = ||M_q f||^2 for a family of smooth face, smooth global and point loads,
              which is the contract's quantity and needs no assembly

and the truth for the second observable, which is one triangular solve per operator:
              c(f) = ||R^-T (B f)||^2                      (A = R^T R, B B^T = I, B rigid = 0)

Writing chat{c}(eps) = c(eps) (1 + err(eps)), the relative error of the finite-difference
derivative is   err + (d err / d eps) / (d ln c / d eps)   with d ln c / d eps = -2.2784, so the
test measures how fast the model's relative error moves with tau. A 3 % derivative needs the
model's relative error to change by less than 1.4e-4 across the 2e-3 span of the h = 1e-3 pair.

Why reusing the base trace cache with scaled tau_corners is exact, not an approximation: the
teacher's own per-path caches record `all_arrays_equal` false for `indices`, `background_nodes`
and `rigid`, but the featurizer only ever uses the composition `background_nodes[indices]`, which
is bit-identical across all five geometries, and `rigid` moves only within its own span (two of
six columns, a shift of the rotation reference point), so the Householder quotient B agrees to
1e-15. Everything else the featurizer reads is bit-identical except tau_corners itself.

    python tau_predict.py --run <TRAIN_OUTPUT_DIR> --output <DIR>
"""
import argparse, gc, json, sys, time
from pathlib import Path
import numpy as np
import torch

FROZEN = Path('/root/autodl-tmp/CUTFEM_SUPERELEMENT_LABELS/REFERENCE_0328')
PATHS = Path('/root/autodl-tmp/CUTFEM_FULL_FACTOR_LOCAL_GEOMETRY_20260913T2330/ANALYSIS_R1')
CASES = [('-1/1000', -1e-3, PATHS / 'PATH_1' / 'R_UPPER.npy'),
         ('-1/10000', -1e-4, PATHS / 'PATH_2' / 'R_UPPER.npy'),
         ('0', 0.0, FROZEN / 'R_UPPER.npy'),
         ('1/10000', 1e-4, PATHS / 'PATH_3' / 'R_UPPER.npy'),
         ('1/1000', 1e-3, PATHS / 'PATH_4' / 'R_UPPER.npy')]
CACHES = {-1e-3: 'PATH_1_H0', -1e-4: 'PATH_2_H0', 0.0: None, 1e-4: 'PATH_3_H0', 1e-3: 'PATH_4_H0'}
GEOM = Path('/root/autodl-tmp/CUTFEM_FULL_FACTOR_LOCAL_GEOMETRY_20260913T2330/GEOMETRY_R1')


def sha256(path, chunk=1 << 22):
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, nargs='+', required=True,
                    help='training output dirs; the last CHECKPOINT_*.pt of each is used, and if '
                         'more than one is given their predicted M_q are also averaged')
    ap.add_argument('--src', type=Path, default=Path('/root/autodl-tmp/CLAUDE_EQUI_20260918/src'))
    ap.add_argument('--source', type=Path, default=Path('/root/autodl-tmp/CLAUDE_SQRTHEAD_20260917/src_v5'))
    ap.add_argument('--device', default='cuda:0'); ap.add_argument('--eval-device', default='cpu')
    ap.add_argument('--inference-chunk', type=int, default=8192)
    ap.add_argument('--threads', type=int, default=6)
    ap.add_argument('--per-face', type=int, default=16); ap.add_argument('--n-global', type=int, default=32)
    ap.add_argument('--n-point', type=int, default=32); ap.add_argument('--seed', type=int, default=20260919)
    ap.add_argument('--replicate', action='store_true',
                    help='predict eps=0 twice and report the reproducibility of chat{c} and trace(H^-1)')
    ap.add_argument('--keep-mq', action='store_true', help='keep the packed predicted M_q per epsilon')
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(a.source)); sys.path.insert(0, str(a.src))
    from stage_cutfem_m4.quotient import RigidQuotient
    from superelement.equi import train_equi as T
    from superelement.equi.context import compile_equi_inputs
    from superelement.equi.model import to_torch
    from superelement.equi.model import EquiModel, GroupTables
    from superelement.equi.platen import require_box_only, operator_from_mq, platen_response
    from superelement.equi import cubic_group as CG
    from stage_cutfem_m4.factors import Conditioning
    torch.set_num_threads(a.threads)
    # The test amplifies any relative error on chat{c} by 1/(2 h |dlnc/deps|) ~ 300 at h=1e-3 and
    # ~3000 at h=1e-4, so reduced precision anywhere is fatal.  matmul TF32 is off in
    # train_equi.main but a standalone script does not inherit it, and cudnn TF32 defaults ON --
    # the volume encoder is Conv3d, so that path matters.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.is_autocast_enabled():
        raise ValueError('AUTOCAST_MUST_BE_OFF')
    dev = torch.device(a.device); out_dev = torch.device(a.eval_device)

    receipt = json.loads((FROZEN / 'RESULT.json').read_text())
    d = int(receipt['dimension'])
    base_cache_path = FROZEN / 'input' / 'TRACE_CACHE.npz'
    bind = dict(base_factor_sha256=sha256(FROZEN / 'R_UPPER.npy'), base_trace_sha256=sha256(base_cache_path),
                recorded_factor_sha256=receipt['factor_sha256'], recorded_trace_sha256=receipt['trace_sha256'])
    if bind['base_factor_sha256'] != bind['recorded_factor_sha256'] or bind['base_trace_sha256'] != bind['recorded_trace_sha256']:
        raise ValueError('FROZEN_REFERENCE_BINDING')
    base = dict(np.load(base_cache_path, allow_pickle=False))
    require_box_only(base)
    meta = json.loads((FROZEN / 'input' / 'INPUT.json').read_text())['metadata']
    base_ctx = compile_equi_inputs(base, meta)
    q = 3 * base_ctx['count']
    if q != d + 6:
        raise ValueError(f'DIMENSION {q} {d}')
    n = base_ctx['n']
    report = dict(seat=328, q=q, d=d, n=n, split='holdout', bindings=bind,
                  base_tau_corners=np.asarray(base['tau_corners'], dtype=float).tolist(),
                  cut_plane=np.asarray(base['cut_plane'], dtype=float).tolist(),
                  scope='tau derivative of the assembled-free observables, network vs teacher',
                  teacher_recorded=dict(dlnc_deps=-2.2784, richardson_central_ratio=0.999997,
                                        source='docs/HEAD_AB_AND_TAU_20260917.md, tau_gate.py'))

    # ---- the load family, built once from the tau-free part of the base context -------------
    rng = np.random.default_rng(a.seed)
    loads = T.smooth_face_loads(base_ctx, a.per_face, rng) + T.smooth_global_loads(base_ctx, a.n_global, rng)
    names = ['face'] * (len(loads) - a.n_global) + ['global'] * a.n_global
    picks = rng.choice(base_ctx['count'], size=a.n_point, replace=False)
    F = T.load_vectors(loads, q)
    P = np.zeros((a.n_point, q))
    for j, c0 in enumerate(picks):
        P[j, 3 * c0 + (j % 3)] = 1.0
    F = np.concatenate((F, P), axis=0)
    names += ['point'] * a.n_point
    Ft = torch.from_numpy(F).to(dev)
    report['loads'] = dict(n=int(F.shape[0]), face=names.count('face'), glob=names.count('global'),
                           point=names.count('point'), seed=a.seed)

    # ---- truth: c(f) = ||R^-T (B f)||^2, one triangular solve per operator ------------------
    Qt = RigidQuotient(torch.from_numpy(base['rigid']).to(dev), torch.from_numpy(base['order']).to(dev))
    Bf = Qt(Ft.T.contiguous())
    rigid_leak = float((Ft @ torch.from_numpy(base['rigid']).to(dev)).norm())
    truth = {}
    for name, eps, path in CASES:
        t0 = time.perf_counter()
        R = T.unpack(path, d, dev)
        Y = torch.linalg.solve_triangular(R.T, Bf, upper=False, left=True)
        truth[eps] = (Y * Y).sum(dim=0).cpu().numpy()
        del R, Y; gc.collect()
        if dev.type == 'cuda':
            torch.cuda.empty_cache()
        print(json.dumps(dict(phase='truth', epsilon=name, c_min=float(truth[eps].min()),
                              c_max=float(truth[eps].max()), seconds=round(time.perf_counter() - t0, 1))), flush=True)
    del Bf
    if min(float(truth[e].min()) for e in truth) <= 0:
        raise ValueError('NONPOSITIVE_EXACT_COMPLIANCE')
    report['load_rigid_projection_norm'] = rigid_leak

    # ---- the models -------------------------------------------------------------------------
    models = []
    for run in a.run:
        cks = sorted(run.glob('CHECKPOINT_*.pt'))
        if not cks:
            raise ValueError(f'NO_CHECKPOINT {run}')
        ck = torch.load(cks[-1], map_location=dev, weights_only=False)
        pm = ck['protocol']['model']
        volume_in = int(to_torch(base_ctx, dev)['volume'].shape[0])
        model = EquiModel(n=n, volume_in=volume_in, near_radius=float(ck['near_radius']),
                          state_dim=pm['state'], width=pm['width'], depth=pm['depth'],
                          volume_channels=pm['volume_channels'], segment_samples=pm['segment_samples']).to(dev)
        model.load_state_dict(ck['model']); model.eval()
        models.append(dict(name=run.name, checkpoint=cks[-1].name, step=int(ck['step']),
                           model=model, cond=Conditioning(**ck['conditioning']),
                           canonical=bool(ck['protocol'].get('canonical', False)),
                           parameters=sum(p.numel() for p in model.parameters())))
        print(json.dumps(dict(phase='model', run=run.name, checkpoint=cks[-1].name, step=int(ck['step']),
                              parameters=models[-1]['parameters'], width=pm['width'], state=pm['state'],
                              depth=pm['depth'], canonical=models[-1]['canonical'])), flush=True)
    tables = GroupTables(n, dev, len(base_ctx['vol_scalar']))
    # pin the presentation frame once: a positive scale cannot change the geometry's frame, but a
    # float tie in the lexicographic argmax could, which would inject the model's equivariance
    # error into the derivative
    g_view = T.view_frame(dict(ctx=base_ctx), 0, models[0]['canonical'])
    report['view_frame'] = int(g_view)
    report['models'] = [{k: m[k] for k in ('name', 'checkpoint', 'step', 'parameters', 'canonical')} for m in models]

    # ---- prediction at each epsilon ---------------------------------------------------------
    rows = []
    for name, eps, path in CASES:
        cache = dict(base)
        cache['tau_corners'] = np.asarray(base['tau_corners'], dtype=np.float64) * (1.0 + eps)
        ctx = compile_equi_inputs(cache, meta)
        # the context must be exactly affine in the tau scale; everything tau-free must be identical
        for key in ('pos', 'faces', 'kind', 'support_pos', 'support_coefficients', 'plane', 'known'):
            if not np.array_equal(np.asarray(ctx[key]), np.asarray(base_ctx[key])):
                raise ValueError(f'TAU_FREE_FEATURE_MOVED {key}')
        ctx_t = to_torch(ctx, dev)
        row = dict(epsilon=name, eps=eps, tau_corners=cache['tau_corners'].tolist(), per_model={})
        Msum = None
        for m in models:
            t0 = time.perf_counter()
            sample = dict(q=q, count=ctx['count'], ctx=ctx, ctx_t=ctx_t)
            M = T.predict_full_q(m['model'], sample, tables, m['cond'], g_view, a.inference_chunk, out_dev)
            secs = time.perf_counter() - t0
            Msum = M.clone() if Msum is None else Msum + M
            ch = ((Ft.to(out_dev) @ M) ** 2).sum(dim=1).cpu().numpy()
            mq = a.output / f'MQ_{m["name"]}_{eps:+.0e}.npy'
            T.pack(M, mq)
            A, Q2 = operator_from_mq(mq, cache, q, a.device)
            pl = platen_response(A, Q2, cache, a.device)
            del A, Q2, M; gc.collect()
            if dev.type == 'cuda':
                torch.cuda.empty_cache()
            if not a.keep_mq:
                mq.unlink()
            row['per_model'][m['name']] = dict(c_hat=ch.tolist(), trace_H_inverse=pl['trace_H_inverse'],
                                               free_equilibrium_relative_max=max(pl['free_equilibrium_relative']),
                                               inference_seconds=round(secs, 1))
            print(json.dumps(dict(phase='predict', epsilon=name, model=m['name'],
                                  trace_H_inverse=pl['trace_H_inverse'],
                                  c_rel_median=float(np.median(ch / truth[eps] - 1)),
                                  seconds=round(time.perf_counter() - t0, 1))), flush=True)
        if len(models) > 1:
            M = Msum / len(models)
            ch = ((Ft.to(out_dev) @ M) ** 2).sum(dim=1).cpu().numpy()
            mq = a.output / f'MQ_ensemble_{eps:+.0e}.npy'
            T.pack(M, mq)
            A, Q2 = operator_from_mq(mq, cache, q, a.device)
            pl = platen_response(A, Q2, cache, a.device)
            del A, Q2, M; gc.collect()
            if not a.keep_mq:
                mq.unlink()
            row['per_model']['ensemble'] = dict(c_hat=ch.tolist(), trace_H_inverse=pl['trace_H_inverse'],
                                                free_equilibrium_relative_max=max(pl['free_equilibrium_relative']))
            print(json.dumps(dict(phase='predict', epsilon=name, model='ensemble',
                                  trace_H_inverse=pl['trace_H_inverse'],
                                  c_rel_median=float(np.median(ch / truth[eps] - 1)))), flush=True)
        del Msum, ctx_t; gc.collect()
        if dev.type == 'cuda':
            torch.cuda.empty_cache()
        row['c_true'] = truth[eps].tolist()
        rows.append(row)
    report['load_names'] = names
    report['cases'] = rows
    if a.replicate:
        cache = dict(base)
        ctx = compile_equi_inputs(cache, meta); ctx_t = to_torch(ctx, dev)
        rep = {}
        for m in models:
            sample = dict(q=q, count=ctx['count'], ctx=ctx, ctx_t=ctx_t)
            M = T.predict_full_q(m['model'], sample, tables, m['cond'], g_view, a.inference_chunk, out_dev)
            ch = ((Ft.to(out_dev) @ M) ** 2).sum(dim=1).cpu().numpy()
            mq = a.output / 'MQ_replicate.npy'
            T.pack(M, mq)
            A, Q2 = operator_from_mq(mq, cache, q, a.device)
            pl = platen_response(A, Q2, cache, a.device)
            del A, Q2, M; gc.collect()
            mq.unlink()
            first = eps_of0 = [r for r in rows if r['eps'] == 0.0][0]['per_model'][m['name']]
            c0 = np.asarray(first['c_hat'])
            rep[m['name']] = dict(max_relative_compliance_difference=float(np.abs(ch / c0 - 1).max()),
                                  bitwise_identical_compliance=bool(np.array_equal(ch, c0)),
                                  trace_H_inverse_relative_difference=float(pl['trace_H_inverse'] / first['trace_H_inverse'] - 1))
            print(json.dumps(dict(phase='replicate', model=m['name'], **rep[m['name']])), flush=True)
        report['replication'] = rep

    # ---- the finite-difference table --------------------------------------------------------
    eps_of = {r['eps']: r for r in rows}
    who = list(rows[0]['per_model'].keys())

    def slopes(value):
        out = {}
        for h in (1e-4, 1e-3):
            left = (value(0.0) - value(-h)) / h
            right = (value(h) - value(0.0)) / h
            central = (value(h) - value(-h)) / (2 * h)
            out[f'h={h:g}'] = dict(left=np.asarray(left).tolist(), right=np.asarray(right).tolist(),
                                   central=np.asarray(central).tolist())
        return out

    def logslope(value):
        v0 = np.asarray(value(0.0))
        out = {}
        for h in (1e-4, 1e-3):
            out[f'h={h:g}'] = dict(
                left=float(np.mean((v0 - np.asarray(value(-h))) / h / v0)),
                right=float(np.mean((np.asarray(value(h)) - v0) / h / v0)),
                central=float(np.mean((np.asarray(value(h)) - np.asarray(value(-h))) / (2 * h) / v0)))
        return out

    tab = dict(platen={}, compliance={})
    tab['platen']['truth_trace_H_inverse'] = {r['epsilon']: None for r in rows}
    for w in who:
        tr = lambda e, w=w: eps_of[e]['per_model'][w]['trace_H_inverse']
        tab['platen'][w] = dict(values={r['epsilon']: tr(r['eps']) for r in rows}, log_slopes=logslope(tr))
    ct = lambda e: np.asarray(eps_of[e]['c_true'])
    tab['compliance']['truth'] = dict(log_slopes=logslope(ct))
    for w in who:
        ch = lambda e, w=w: np.asarray(eps_of[e]['per_model'][w]['c_hat'])
        rel0 = ch(0.0) / ct(0.0) - 1
        der = {}
        for h in (1e-4, 1e-3):
            dt = (ct(h) - ct(-h)) / (2 * h)
            dp = (ch(h) - ch(-h)) / (2 * h)
            der[f'h={h:g}'] = dict(
                central_relative_median=float(np.median(dp / dt - 1)),
                central_relative_p90=float(np.quantile(np.abs(dp / dt - 1), .9)),
                central_relative_max=float(np.abs(dp / dt - 1).max()),
                frac_within_3pct=float(np.mean(np.abs(dp / dt - 1) <= .03)),
                sign_agreement=float(np.mean(np.sign(dp) == np.sign(dt))))
            for kind in ('face', 'global', 'point'):
                sel = np.array([x == kind for x in names])
                if sel.any():
                    der[f'h={h:g}'][kind] = dict(
                        median=float(np.median(np.abs(dp[sel] / dt[sel] - 1))),
                        max=float(np.abs(dp[sel] / dt[sel] - 1).max()),
                        frac_within_3pct=float(np.mean(np.abs(dp[sel] / dt[sel] - 1) <= .03)))
        tab['compliance'][w] = dict(value_relative=dict(median=float(np.median(rel0)),
                                                        p90=float(np.quantile(np.abs(rel0), .9)),
                                                        max=float(np.abs(rel0).max())),
                                    log_slopes=logslope(ch), derivative=der)
    report['table'] = tab
    report['status'] = 'TAU_PREDICT_COMPLETE'
    (a.output / 'TAU_PREDICT.json').write_text(json.dumps(report, indent=1))
    print(json.dumps(dict(status=report['status'], platen={w: tab['platen'][w]['log_slopes'] for w in who},
                          compliance_derivative={w: tab['compliance'][w]['derivative'] for w in who},
                          compliance_value={w: tab['compliance'][w]['value_relative'] for w in who}), indent=1), flush=True)


if __name__ == '__main__':
    main()
