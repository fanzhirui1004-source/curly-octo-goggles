"""Local CPU tests of the models v2 options (B1 bounded coefficients, B2 features, fringe_soft) on the fixture.
  0. D_FULL.npz (written next to models.py from the fixture's teacher Tm if absent, else checked against it); full-cube
     moments vs the fixture's full elements; cube symmetry of D_full.
  1. flags off: geometry outputs, field and parameter gradients bit-identical to the original s0/src/models.py.
  2. bounded=True, A = inf: bit-identical; _SoftClip gradcheck.
  3. A calibrated on this geometry (calib_b1.calibrate and the CLI): saturation <= 1e-3, max relative field change and
     energy excess before / after reported.
  4. feat_v2 + load_compat: bit-identical at load, strictness, gradient reaches the new input columns.
  5. B2 features: finite, ranges, rho ~ 0 in full solid, invariant under all 48 cube-group elements (synthetic rotation of
     the keys the features read) and on the fixture's real rotated packets (fresh_train_0021_cover01_r1_rot*).
  6. fringe_soft: masks / fades, linear in q.
  7. fastnet.FastNet with all flags on vs the model path (field, adjoint, s_hat) <= 1e-5 relative.
Usage: OPL_DEV=cpu python3 t_models_v2.py"""
import os, sys, json, time, itertools, tempfile, importlib.util
from pathlib import Path
os.environ.setdefault('OPL_DEV', 'cpu')
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fixture as FX
import models as MD
import fastnet as FN
import calib_b1 as CB

ORIG = HERE.parent / 'src' / 'models.py'
CKS = ('mgno2_r2_d2_best.pt', 's2v1_snap_10000.pt')
f64 = torch.float64
REP = {}


def check(name, ok, **info):
    REP[name] = dict(ok=bool(ok), **info)
    print(json.dumps({name: REP[name]}, default=float), flush=True)
    if not ok:
        raise AssertionError(name)


def rel(a, b):
    a, b = a.to(f64), b.to(f64)
    return float(((a - b).norm(dim=0) / b.norm(dim=0).clamp_min(1e-300)).max())


def orig_models():
    spec = importlib.util.spec_from_file_location('models_orig', ORIG)
    mo = importlib.util.module_from_spec(spec); spec.loader.exec_module(mo)
    mo.dev = torch.device('cpu')                                   # the deployed file hard-codes cuda:0
    return mo


def build(mod, g, ck, **over):
    cfg = ck['cfg']
    args = dict(cfg.get('model_args', {})); args.update(over)
    return mod.build(cfg['model'], [g], **args)


def cube_group():
    out = []
    for p in itertools.permutations(range(3)):
        for s in itertools.product((1, -1), repeat=3):
            R = np.zeros((3, 3), int)
            for i in range(3):
                R[i, p[i]] = s[i]
            out.append(R)
    return out


def slot_perm(R, offs):
    """new slot s holds the old slot t with R (o_t - 1) + 1 = o_s (v_rot / O_h convention)."""
    img = (offs - 1) @ R.T + 1
    return np.array([int(np.flatnonzero((img == offs[s]).all(1))[0]) for s in range(27)])


def rotate_nd(nd, R):
    """The keys node_feats_v2 reads, rotated as the O_h view does: grid, elem_cells (cell_img), elem_nodes slots,
    diag3 -> R D R^T, normal -> R n, offset -> offset + (sum n' - sum n) / 2; node flags unchanged."""
    n = int(nd['n'])
    out = dict(nd)
    out['grid'] = ((nd['grid'].astype(np.int64) - n) @ R.T + n).astype(np.int16)
    out['elem_cells'] = (((2 * nd['elem_cells'].astype(np.int64) + 1 - n) @ R.T + n - 1) // 2).astype(np.int32)
    out['elem_nodes'] = nd['elem_nodes'][:, slot_perm(R, MD.d_full_table()['offsets'])]
    out['diag3'] = np.einsum('ij,njk,lk->nil', R, nd['diag3'], R)
    nn_ = R @ nd['normal']
    out['normal'], out['offset'] = nn_, np.asarray(float(nd['offset']) + (nn_.sum() - nd['normal'].sum()) / 2)
    return out


def feat_diff(fa, fb, ia=None, ib=None):
    return {k: float((fa[k][ia] - fb[k][ib]).abs().max()) if ia is not None else float((fa[k] - fb[k]).abs().max()) for k in fa}


def t0_dfull(g):
    nd = g.nd; n = int(nd['n'])
    offs = nd['grid'][nd['elem_nodes'][0]].astype(np.int64) - 2 * nd['elem_cells'][0].astype(np.int64)
    D = MD.d_full_from_Tm(g.C.Tm, n)
    p = HERE / 'D_FULL.npz'
    if not p.exists():
        np.savez(p, D=D, n=np.asarray(n), offsets=offs, source=np.asarray(f'teacher Tm of {g.case} (fixture), full-cube moments'))
        print(json.dumps(dict(event='WROTE', path=str(p))), flush=True)
    T = MD.d_full_table()
    check('dfull_file', np.abs(T['D'] - D).max() <= 1e-14 * np.abs(D).max() and np.array_equal(T['offsets'], offs) and int(T['n']) == n,
          max_abs=float(np.abs(T['D'] - D).max()))
    vf = nd['moments'][:, 0] * n ** 3
    full = vf > 1 - 1e-12
    Mf = MD.full_moments(n)
    check('full_moments_vs_fixture', full.sum() > 0 and np.abs(nd['moments'][full] - Mf).max() <= 1e-12 * np.abs(Mf).max(),
          full_elements=int(full.sum()), max_rel=float(np.abs(nd['moments'][full] - Mf).max() / np.abs(Mf).max()))
    Ke = np.einsum('m,mij->ij', Mf, g.C.Tm.numpy())
    err = max(float(np.abs(D - np.einsum('ij,sjk,lk->sil', R, D[slot_perm(R, offs)], R)).max()) for R in cube_group())   # D[s] = R D[perm s] R^T
    check('dfull_cube_symmetry', err <= 1e-12 * np.abs(D).max(), max_abs=err, rigid_modes=int((np.abs(np.linalg.eigvalsh(Ke)) < 1e-12).sum()))


def t1_flags_off(g, MO, Q):
    out = {}
    for name in CKS:
        ck = FX.checkpoint(name)
        mo, mn = build(MO, g, ck), build(MD, g, ck)
        mo.load_state_dict(ck['model'], strict=False); info = MD.load_compat(mn, ck['model'])
        mo.eval(); mn.eval()
        with torch.no_grad():
            go, gn = mo.geometry(g.case), mn.geometry(g.case)
            geq = all(torch.equal(go[k], gn[k]) for k in ('ab', 'fab', 'xab', 'ge', 'gn')) and \
                all(torch.equal(a, b) for l in range(len(go['rw'])) for a, b in zip(go['rw'][l], gn['rw'][l])) and \
                all(torch.equal(a, b) for a, b in zip(go['gates'], gn['gates']))
            uo, un = g.field(mo, Q), g.field(mn, Q)
        # parameter gradients of an energy-like loss (autograd through the whole q path). The CPU backward accumulates
        # in a thread-dependent order (run-to-run ~1e-6) unless deterministic algorithms are on: compare bitwise with them
        # on; if an op has no deterministic kernel (warn_only), fall back to <= 3x the old model's own run-to-run noise.
        qq = Q[:, :2].float()
        grads = []
        torch.use_deterministic_algorithms(True, warn_only=True)
        try:
            for m in (mo, mo, mn):
                m.zero_grad(set_to_none=True)
                with torch.enable_grad():
                    u = g.field(m, qq)
                    (u.double() * (g.C.K @ u.double())).sum().backward()
                grads.append({k: p.grad.clone() for k, p in m.named_parameters() if p.grad is not None})
        finally:
            torch.use_deterministic_algorithms(False)
        gd = lambda a_, b_: max(float((a_[k] - b_[k]).norm() / a_[k].norm().clamp_min(1e-30)) for k in a_)
        same_keys = grads[0].keys() == grads[2].keys()
        noise, diff = gd(grads[0], grads[1]), gd(grads[0], grads[2])
        geq2 = same_keys and (diff == 0 if noise == 0 else diff <= 3 * noise)
        check(f'flags_off_identical[{name}]', geq and torch.equal(uo, un) and geq2, geometry_equal=geq, field_equal=torch.equal(uo, un),
              grads_max_rel_diff=diff, grads_old_repeat_noise=noise, n_grads=len(grads[0]), compat=info, sparse=bool(mn.sparse))
        out[name] = un
    # MGNO v0 (no v0 checkpoint in the fixture): the original's random weights, flags off
    torch.manual_seed(5)
    mo = MO.build('mgno', [g]).eval()
    mn = MD.build('mgno', [g]).eval()
    info = MD.load_compat(mn, mo.state_dict())
    with torch.no_grad():
        eq = torch.equal(g.field(mo, Q), g.field(mn, Q))
    check('flags_off_identical[mgno_v0_random]', eq and info == dict(padded=[], kept=[]))
    return out


def t2_bounded_inf(g, Q, ref):
    for name in CKS:
        ck = FX.checkpoint(name)
        m = build(MD, g, ck, bounded=True)
        info = MD.load_compat(m, ck['model']); m.eval()
        with torch.no_grad():
            u = g.field(m, Q)
        sat = m.sat_stats()
        check(f'bounded_inf_identical[{name}]', torch.equal(u, ref[name]) and sat['max'] == 0, kept=info['kept'], sat_max=sat['max'])
    x = torch.randn(5, 3, 2, 4, dtype=f64, requires_grad=True)
    A = torch.tensor([[0.3, float('inf'), 1.0, 2.0], [float('inf')] * 4], dtype=f64).view(1, 1, 2, 4)
    ok1 = torch.autograd.gradcheck(lambda x_: MD._SoftClip.apply(x_, A, True), (x,))
    ok2 = torch.autograd.gradcheck(lambda x_: MD._SoftClip.apply(x_, A.clamp_max(5.0), False), (x,))
    y = MD._SoftClip.apply(x, A, True)
    fin = torch.isfinite(A).expand_as(x)
    check('softclip', ok1 and ok2 and torch.equal(y[~fin], x[~fin]) and bool((y[fin].abs() <= A.expand_as(x)[fin]).all()))


def energies(g, m, per=4):
    out = {}
    with torch.no_grad():
        for c in g.classes:
            Qc = g.banks['val'][c][:, :per]
            u = g.field(m, Qc)
            out[c] = float(((u.double() * (g.C.K @ u.double())).sum(0) - 1).mean())
    return out


def t3_calibrated(g, Q, ref):
    res = {}
    for name in CKS:
        ck = FX.checkpoint(name)
        m = build(MD, g, ck, bounded=True)
        MD.load_compat(m, ck['model']); m.eval()
        bounds, stats, per_geo = CB.calibrate(m, [g], mult=2.5, log=lambda s_: None)
        e0 = energies(g, m)
        m.set_bounds(bounds)
        with torch.no_grad():
            u = g.field(m, Q)
        sat = m.sat_stats()
        e1 = energies(g, m)
        r = rel(u, ref[name])
        check(f'calibrated[{name}]', np.isfinite(r) and sat['max'] <= 1.01e-3 and all(np.isfinite(list(e1.values()))),
              max_rel_field_change=r, sat_max=sat['max'], excess_before=e0, excess_after=e1,
              A_ab=[float(bounds['ab'].min()), float(bounds['ab'].max())], w_max=bounds['rw'].tolist())
        res[name] = bounds
    # saturation counts against a direct count on the captured raw values (A = p50: about half of each group saturates)
    m.set_bounds({k: torch.tensor(v['p50']) for k, v in stats.items()})
    m._rec = {}
    with torch.no_grad():
        m.geometry(g.case)
    raw, m._rec = m._rec, None
    sat = m.sat_stats()
    err = 0.0
    for k in ('ab', 'fab', 'xab'):
        A = getattr(m, 'bnd_' + k)
        direct = (raw[k].abs() > A.view(1, 1, *A.shape, 1)).double().mean((0, 1, 4))
        err = max(err, float((direct - torch.tensor(sat[k], dtype=f64)).abs().max()))
    direct_rw = torch.tensor([[float((raw['rw'][l][j] > m.bnd_rw[l, j]).double().mean()) for j in range(2)] for l in range(m.levels)], dtype=f64)
    err = max(err, float((direct_rw - torch.tensor(sat['rw'], dtype=f64)).abs().max()))
    m.track_sat = False
    with torch.no_grad():
        m.geometry(g.case)
    check('sat_stats_counts', err <= 1e-6 and 0.3 < sat['max'] <= 0.6 and m.sat_stats()['max'] == 0, max_abs_err=err, sat_max_at_p50=sat['max'])
    # the CLI on a one-geometry split from the fixture NETDATA gives the same bounds
    with tempfile.TemporaryDirectory() as td:
        sp = Path(td) / 'split.json'; sp.write_text(json.dumps(dict(train=[FX.SMALL])))
        out = Path(td) / 'b1.pt'
        CB.main([str(FX.FIX / CKS[1]), str(sp), str(out), '--data', str(FX.FIX), '--mult', '2.5'])
        z = torch.load(out, weights_only=False)
        js = json.loads(out.with_suffix('.json').read_text())
        same = all(torch.equal(z['bounds'][k], res[CKS[1]][k]) for k in res[CKS[1]])
        m = build(MD, g, FX.checkpoint(CKS[1]), bounded=True, bounds=str(out))
        same2 = all(torch.equal(getattr(m, 'bnd_' + k), z['bounds'][k]) for k in z['bounds'])
        check('calib_cli', same and same2 and set(js['stats']) == {'ab', 'fab', 'xab', 'rw'}, groups=list(js['stats']))
    return res


def t4_feat_v2(g, Q, ref):
    for name in CKS:
        ck = FX.checkpoint(name)
        m = build(MD, g, ck, feat_v2=True)
        info = MD.load_compat(m, ck['model']); m.eval()
        with torch.no_grad():
            u = g.field(m, Q)
        m2 = build(MD, g, ck, feat_v2=True, bounded=True)
        MD.load_compat(m2, ck['model']); m2.eval()
        with torch.no_grad():
            u2 = g.field(m2, Q)
        check(f'feat_v2_load_identical[{name}]', torch.equal(u, ref[name]) and torch.equal(u2, ref[name]) and info['padded'] == ['node_in.0.weight'],
              padded=info['padded'], node_in=list(m.node_in[0].weight.shape))
    ck = FX.checkpoint(CKS[0])
    m = build(MD, g, ck, feat_v2=True)
    errs = []
    for bad in ({**ck['model'], 'extra.w': torch.zeros(1)}, {k: v for k, v in ck['model'].items() if k != 'W_in'},
                {**ck['model'], 'W_out': torch.zeros(3, 3)}):
        try:
            MD.load_compat(m, bad); errs.append(False)
        except (KeyError, ValueError):
            errs.append(True)
    old = build(MD, g, ck)
    try:
        MD.load_compat(old, m.state_dict()); errs.append(False)                     # a widened state into an old model
    except ValueError:
        errs.append(True)
    MD.load_compat(m, ck['model']); m.train()
    m.zero_grad(set_to_none=True)
    u = g.field(m, Q[:, :2].float())
    (u.double() * (g.C.K @ u.double())).sum().backward()
    gw = m.node_in[0].weight.grad[:, -MD.NF2:]
    check('load_compat_strict_and_new_cols_train', all(errs) and bool(torch.isfinite(gw).all()) and float(gw.abs().max()) > 0,
          raised=errs, new_col_grad_max=float(gw.abs().max()))


def t5_features(g):
    nd = g.nd
    f = MD.node_feats_v2(nd, device='cpu')
    fin = all(bool(torch.isfinite(v).all()) for v in f.values())
    en = nd['elem_nodes'].astype(np.int64)
    vf = nd['moments'][:, 0] * int(nd['n']) ** 3
    allfull = np.ones(len(nd['grid']), bool)
    np.logical_and.at(allfull, en.reshape(-1), np.repeat(vf > 1 - 1e-12, 27))
    touched = np.zeros(len(nd['grid']), bool); touched[en.reshape(-1)] = True
    solid = allfull & touched
    rho_solid = float(np.abs(f['rho'].numpy()[solid]).max()) if solid.any() else float('nan')
    ranges = dict(s=(float(f['s'].min()), float(f['s'].max())), hop=sorted(set(f['hop'].tolist())),
                  plane=(float(f['plane'].min()), float(f['plane'].max())), box=(float(f['box'].min()), float(f['box'].max())))
    ok = fin and 0 <= ranges['s'][0] and ranges['s'][1] <= 1 and set(ranges['hop']) <= set(range(7)) and \
        -6 <= ranges['plane'][0] and ranges['plane'][1] <= 6 and 0 <= ranges['box'][0] and ranges['box'][1] <= 6 and \
        bool((f['hop'][torch.as_tensor(nd['is_port'])] == 0).all()) and rho_solid < 1e-2
    check('features_finite_ranges', ok, ranges=ranges, solid_nodes=int(solid.sum()), max_abs_rho_solid=rho_solid,
          weak_old=float(nd['weak'].mean()), s_gt_half=float((f['s'] > 0.5).float().mean()), s_gt_001=float((f['s'] > 0.01).float().mean()))
    # FULL (no plane) convention
    ndf = dict(nd); ndf['normal'] = np.zeros(3); ndf['offset'] = np.asarray(0.0)
    check('features_full_plane', bool((MD.node_feats_v2(ndf, device='cpu')['plane'] == 6).all()))
    # all 48 cube-group elements on a synthetic rotation of the keys the features read
    worst = {k: 0.0 for k in f}
    for R in cube_group():
        fr = MD.node_feats_v2(rotate_nd(nd, R), device='cpu')
        for k, v in feat_diff(fr, f).items():
            worst[k] = max(worst[k], v)
    check('features_cube_invariant_synthetic', max(worst.values()) <= 1e-9, max_abs_diff=worst)
    # the fixture's real rotated packets (independent teacher runs on the rotated geometry): match nodes by the grid image
    base = FX.netdata(FX.ROT_BASE)
    fb = MD.node_feats_v2(base, device='cpu')
    n = int(base['n'])
    for rn, R in FX.ROTS.items():
        R = np.asarray(R)
        rot = FX.netdata(f'{FX.ROT_BASE}_rot{rn}')
        fr = MD.node_feats_v2(rot, device='cpu')
        img = np.ravel_multi_index((((base['grid'].astype(np.int64) - n) @ R.T) + n).T, (2 * n + 1,) * 3)
        pos = np.searchsorted(rot['node_ids'], img)
        okm = bool(np.array_equal(rot['node_ids'][np.minimum(pos, len(pos) - 1)], img))
        d = feat_diff(fb, fr, torch.arange(len(pos)), torch.as_tensor(pos))
        # rounding of the independently assembled diag3 moves s only near its threshold: bound the log-ratio difference
        check(f'features_rot_packet[{rn}]', okm and d['hop'] == 0 and d['box'] == 0 and d['plane'] <= 1e-9 and d['rho'] <= 1e-6 and d['s'] <= 1e-4,
              max_abs_diff=d)


def t6_fringe_soft(g, bounds):
    ck = FX.checkpoint(CKS[0])
    try:
        build(MD, g, ck, fringe_soft=True); raised = False
    except ValueError:
        raised = True
    m = build(MD, g, ck, bounded=True, feat_v2=True, fringe_soft=True, bounds=bounds[CKS[0]])
    MD.load_compat(m, ck['model']); m.eval()
    c = m.caches[g.case]
    old = build(MD, g, ck).caches[g.case]
    s = c.s_weak
    ok = raised and bool((c.el_fade > 0.01).all()) and bool((c.el_fade <= 1).all()) and bool((c.gp_fade > 0.01).all()) and \
        int(c.el_fringe.sum()) == len(c.el_fade) and int(c.gp_fringe.sum()) == len(c.gp_fade) and \
        bool(torch.equal(c.el_fringe, (s[c.en] > 0.01).any(1)))
    gen = torch.Generator().manual_seed(0)
    q1 = torch.randn((g.np_, 3), dtype=f64, generator=gen); q2 = torch.randn((g.np_, 3), dtype=f64, generator=gen)
    with torch.no_grad():
        lin = rel(g.field(m, q1 + 2 * q2), g.field(m, q1) + 2 * g.field(m, q2))
        fin = bool(torch.isfinite(g.field(m, g.banks['val']['force'][:, :4])).all())
    check('fringe_soft', ok and lin <= 1e-5 and fin, needs_feat_v2_raised=raised, linear_rel=lin,
          el_fringe=[int(old.el_fringe.sum()), int(c.el_fringe.sum()), len(c.en)], gp_fringe=[int(old.gp_fringe.sum()), int(c.gp_fringe.sum()), len(c.fn)],
          el_fade_mean=float(c.el_fade.mean()), gp_fade_mean=float(c.gp_fade.mean()),
          old_mask_in_new=[bool((c.el_fringe | ~old.el_fringe).all()), bool((c.gp_fringe | ~old.gp_fringe).all())])


def t7_fastnet(g, bounds):
    gen = torch.Generator().manual_seed(1)
    Q = torch.cat([torch.cat([g.banks['val'][c][:, :1] for c in g.classes], 1).to(f64), torch.randn((g.np_, 3), dtype=f64, generator=gen)], 1)
    for name in CKS:
        ck = FX.checkpoint(name)
        m = build(MD, g, ck, bounded=True, feat_v2=True, fringe_soft=True, bounds=bounds[name])
        MD.load_compat(m, ck['model']); m.eval()
        with torch.no_grad():                                           # non-trivial weights on the new columns
            m.node_in[0].weight[:, -MD.NF2:] = 0.05 * torch.randn(m.node_in[0].weight.shape[0], MD.NF2, generator=gen)
        fast = FN.FastNet(m, g)
        with torch.no_grad():
            u_ref = g.field(m, Q)
        r_field = rel(fast.field(Q), u_ref)
        V = torch.randn((g.nb, Q.shape[1]), dtype=torch.float32, generator=gen)
        lhs = (fast.field(Q) * V).sum(0).to(f64); rhs = (Q.float() * fast.field_T(V)).sum(0).to(f64)
        r_adj = float(((lhs - rhs).abs() / (fast.field(Q).norm(dim=0) * V.norm(dim=0)).to(f64)).max())
        r_shat = rel(fast.s_hat(Q), g.s_hat_apply(m, Q))
        qd = torch.randn((g.np_, 3), generator=gen); y = torch.randn((g.nb, 3), generator=gen)
        qq = qd.clone().requires_grad_(True)
        with torch.enable_grad():
            g_ref = torch.autograd.grad(m(g, qq), qq, grad_outputs=y)[0]
        r_T = rel(fast.ext_T(y), g_ref)
        check(f'fastnet_flags_on[{name}]', max(r_field, r_adj, r_shat, r_T) <= 1e-5, field_rel=r_field, adjoint_identity_rel=r_adj,
              s_hat_rel=r_shat, model_T_rel=r_T, sparse_model=bool(m.sparse), sat_max=m.sat_stats()['max'])


def t7b_fastnet_v0(g):
    """MGNO v0 (no checkpoint in the fixture: random weights) with bounded (finite A, one group left at inf) + feat_v2."""
    torch.manual_seed(3)
    m = MD.build('mgno', [g], bounded=True, feat_v2=True).eval()
    bnd = dict(ab=torch.full((2, 8), 0.05), rw=torch.tensor([[0.8, float('inf')], [0.8, 1.2], [0.8, 1.2]]))
    bnd['ab'][1, 3] = float('inf')
    m.set_bounds(bnd)
    gen = torch.Generator().manual_seed(2)
    Q = torch.randn((g.np_, 4), dtype=f64, generator=gen)
    fast = FN.FastNet(m, g)
    with torch.no_grad():
        u_ref = g.field(m, Q)
    r_field = rel(fast.field(Q), u_ref)
    r_shat = rel(fast.s_hat(Q), g.s_hat_apply(m, Q))
    sat = m.sat_stats()
    check('fastnet_flags_on[mgno_v0_random]', max(r_field, r_shat) <= 1e-5 and m._bst == dict(ab='mixed', rw='mixed'),
          field_rel=r_field, s_hat_rel=r_shat, sat_ab=sat['ab'], bst=m._bst)


def main():
    t0 = time.perf_counter()
    torch.manual_seed(0)
    g = FX.small()
    MO = orig_models()
    t0_dfull(g)
    Q = torch.cat([g.banks['val'][c][:, :1] for c in g.classes], 1)
    ref = t1_flags_off(g, MO, Q)
    t2_bounded_inf(g, Q, ref)
    bounds = t3_calibrated(g, Q, ref)
    t4_feat_v2(g, Q, ref)
    t5_features(g)
    t6_fringe_soft(g, bounds)
    t7_fastnet(g, bounds)
    t7b_fastnet_v0(g)
    print(json.dumps(dict(ALL_PASSED=all(r['ok'] for r in REP.values()), n=len(REP), seconds=time.perf_counter() - t0)), flush=True)


if __name__ == '__main__':
    main()
