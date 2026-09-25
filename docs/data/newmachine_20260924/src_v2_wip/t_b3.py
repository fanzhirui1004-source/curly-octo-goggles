"""Local CPU tests of B3 (stiffness-share scattering, models b3=True) on the fixture (fresh_train_0010, MGNO2 r2 d2).
  0. B3_TABLE.npz (written next to models.py from the fixture's teacher Tm and GP_TEMPLATES_n32.npz if absent, else checked
     against them); the tables reproduce the assembled diagonal exactly: nd['diag3'] = sum_e K_e[s, s] + gamma sum_f G_f[j, j]
     (gamma fitted, one scalar); full_moments . Tmd = D_FULL; the fn stencil lookup vs a brute-force dictionary lookup.
  1. b3=True with lambda = 0 (load_compat of the old checkpoint): field bit-identical to b3=False, dense and sparse paths.
  2. lambda = 0.7: sparse vs dense <= 1e-5, the field changes, the gradient reaches every lambda; pi sums to 1 per covered
     node over the element and over the ghost-face hyperedge sets (and over the fringe subsets at weak nodes); pi of an O_h
     view (oh.view, rotated nd) equals the base pi incidence by incidence.
  3. fastnet.FastNet vs the autograd model at lambda = 0.7 (field, adjoint identity, s_hat, model^T) <= 1e-5, also with
     feat_v2 + fringe_soft (masked + faded fringe layers), and MGNO v0 (random weights, element layers only).
  4. load_compat: old checkpoint -> b3 model keeps the zero lambdas (and is identical, see 1); b3 state round trip is exact;
     a b3 state into an old model raises.
Usage: OPL_DEV=cpu python3 t_b3.py"""
import os, sys, json, time
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
import oh as OH

CK = 'mgno2_r2_d2_best.pt'
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


def build(g, ck, **over):
    args = dict(ck['cfg'].get('model_args', {})); args.update(over)
    m = MD.build(ck['cfg']['model'], [g], **args)
    info = MD.load_compat(m, ck['model'])
    return m.eval(), info


def set_lam(m, v):
    with torch.no_grad():
        for k in ('b3_le', 'b3_lf', 'b3_lx'):
            getattr(m, k).fill_(v)


def t0_table(g):
    nd = g.nd; n = int(nd['n'])
    offs = nd['grid'][nd['elem_nodes'][0]].astype(np.int64) - 2 * nd['elem_cells'][0].astype(np.int64)
    tpl = dict(np.load(FX.FIX / f'GP_TEMPLATES_n{n}.npz'))
    T0 = MD.b3_table_from(g.C.Tm, tpl, n, offs)
    p = HERE / 'B3_TABLE.npz'
    if not p.exists():
        np.savez(p, **T0, source=np.asarray(f'teacher Tm of {g.case} (fixture) and GP_TEMPLATES_n{n}.npz'))
        print(json.dumps(dict(event='WROTE', path=str(p))), flush=True)
    T = MD.b3_table()
    same = all(np.array_equal(T[k], T0[k]) for k in T0)
    dfull = np.einsum('m,msij->sij', MD.full_moments(n), T['Tmd'])
    D = MD.d_full_table()['D']
    # exact diagonal: element blocks (all 27 slots of every element) + gamma x ghost blocks (all 45 template nodes of every face)
    N = len(nd['grid'])
    en = torch.as_tensor(nd['elem_nodes'].astype(np.int64))
    Kd = MD.elem_stiff_blocks(nd, device='cpu')
    Dn = torch.zeros((N, 3, 3), dtype=f64).index_add_(0, en.reshape(-1), Kd.reshape(-1, 3, 3))
    f = nd['gp_faces'].astype(np.int64); M_ = 2 * n + 1
    gpos = 2 * nd['elem_cells'].astype(np.int64)[f[:, 0]][:, None, :] + T['gp_offsets'][f[:, 2]]
    key = lambda x: (x[..., 0] * M_ + x[..., 1]) * M_ + x[..., 2]
    ids = key(nd['grid'].astype(np.int64)); o = np.argsort(ids)
    loc = o[np.searchsorted(ids[o], key(gpos))]
    Gn = torch.zeros((N, 3, 3), dtype=f64).index_add_(0, torch.as_tensor(loc.reshape(-1)),
                                                        torch.as_tensor(T['gp_blk'][f[:, 2]].reshape(-1, 3, 3)))
    d3 = torch.as_tensor(nd['diag3'])
    r, gg = (d3 - Dn).reshape(-1), Gn.reshape(-1)
    gamma = float(r @ gg / (gg @ gg))
    res = float((r - gamma * gg).norm() / d3.norm())
    # stencil lookup in gp_stiff_blocks vs a brute-force (offset -> template node) dictionary
    m, _ = build(g, FX.checkpoint(CK), b3=True)
    c = m.caches[g.case]
    Gb = MD.gp_stiff_blocks(nd, c.fn, c.gp_owner, c.gp_axis, device='cpu')
    oo = (c.grid[c.fn] - 2 * torch.as_tensor(nd['elem_cells'].astype(np.int64))[c.gp_owner][:, None, :]).numpy()
    lut = [{tuple(T['gp_offsets'][a, j]): j for j in range(T['gp_offsets'].shape[1])} for a in range(3)]
    ax = c.gp_axis.numpy()
    ref = np.stack([np.stack([T['gp_blk'][ax[i], lut[ax[i]][tuple(oo[i, s])]] for s in range(27)]) for i in range(len(ax))])
    check('b3_table', same and np.abs(dfull - D).max() <= 1e-13 * np.abs(D).max() and res <= 1e-12 and
          np.array_equal(Gb.numpy(), ref), table_matches_teacher=same, dfull_max_abs=float(np.abs(dfull - D).max()),
          diag3_resid_rel=res, gamma_fit=gamma, body_only_resid_rel=float(r.norm() / d3.norm()),
          pi_el_range=[float(c.pi_el.min()), float(c.pi_el.max())], pi_gp_range=[float(c.pi_gp.min()), float(c.pi_gp.max())])


def t1_identity(g, ck, Q):
    m0, _ = build(g, ck)
    m1, info = build(g, ck, b3=True)
    out = {}
    with torch.no_grad():
        for sp in (False, True):
            m0.sparse = m1.sparse = sp
            u0, u1 = g.field(m0, Q), g.field(m1, Q)
            out['sparse' if sp else 'dense'] = dict(equal=bool(torch.equal(u0, u1)), rel=rel(u1, u0))
    m0.sparse = m1.sparse = False
    n0 = sum(p.numel() for p in m0.parameters()); n1 = sum(p.numel() for p in m1.parameters())
    check('lambda0_identical', all(v['rel'] <= 1e-6 for v in out.values()) and not any(k.startswith('b3_') for k in m0.state_dict()),
          **out, params_old=n0, params_b3=n1, b3_params=n1 - n0, compat=info)
    return m1


def pi_sums(hn, pi, N):
    s = torch.zeros(N, dtype=f64).index_add_(0, hn.reshape(-1), pi.reshape(-1))
    cov = torch.bincount(hn.reshape(-1), minlength=N) > 0
    return float((s[cov] - 1).abs().max()), s, cov


def by_node(hn, pi):
    """pi keyed by (hyperedge, node): rows sorted by node id (slot orders differ between views)."""
    o = torch.argsort(hn, 1)
    return torch.gather(hn, 1, o), torch.gather(pi, 1, o)


def t2_lambda(g, ck, m, Q):
    c = m.caches[g.case]
    with torch.no_grad():
        u0 = g.field(m, Q)
    set_lam(m, 0.7)
    with torch.no_grad():
        m.sparse = False; ud = g.field(m, Q)
        m.sparse = True; us = g.field(m, Q)
    m.sparse = False
    r_sd, r_ch = rel(us, ud), rel(ud, u0)
    # gradient reaches every lambda (energy-like loss, dense autograd path; then the sparse path)
    grads = {}
    for sp in (False, True):
        m.sparse = sp
        m.zero_grad(set_to_none=True)
        with torch.enable_grad():
            u = g.field(m, Q[:, :2].float())
            (u.double() * (g.C.K @ u.double())).sum().backward()
        grads['sparse' if sp else 'dense'] = {k: getattr(m, k).grad.clone() for k in ('b3_le', 'b3_lf', 'b3_lx')}
    m.sparse = False
    gd, gs = grads['dense'], grads['sparse']
    g_ok = all(bool((v != 0).all()) and bool(torch.isfinite(v).all()) for v in gd.values())
    g_sd = max(float((gs[k] - gd[k]).norm() / gd[k].norm()) for k in gd)
    check('lambda07_sparse_dense_grad', r_sd <= 1e-5 and r_ch > 1e-4 and g_ok and g_sd <= 1e-4, sparse_vs_dense_rel=r_sd,
          field_change_vs_lambda0_rel=r_ch, grad_sparse_vs_dense_rel=g_sd,
          grad_abs_min={k: float(v.abs().min()) for k, v in gd.items()})
    # partition of unity of pi over each hyperedge set; fringe subsets (full-set pi restricted) at weak nodes
    e_el, _, _ = pi_sums(c.en, c.pi_el, c.N)
    e_gp, _, _ = pi_sums(c.fn, c.pi_gp, c.N)
    _, s_x, _ = pi_sums(c.en[c.el_fringe], c.pi_el[c.el_fringe], c.N)
    _, s_xf, _ = pi_sums(c.fn[c.gp_fringe], c.pi_gp[c.gp_fringe], c.N)
    wk = c.weak
    e_x = float((s_x[wk] - 1).abs().max()) if bool(wk.any()) else 0.0
    wf = wk & (torch.bincount(c.fn.reshape(-1), minlength=c.N) > 0)
    e_xf = float((s_xf[wf] - 1).abs().max()) if bool(wf.any()) else 0.0
    fac_ok = bool(torch.equal(c.b3_el, (c.pi_el * c.deg.double()[c.en] - 1).float()))
    # O_h view: pi from the rotated nd equals the base pi incidence by incidence
    errs = []
    for k in (5, 29, 47):
        v = OH.view(m, g, k)
        cv = m.caches[v.case]
        for hb, pb, hv, pv in ((c.en, c.pi_el, cv.en, cv.pi_el), (c.fn, c.pi_gp, cv.fn, cv.pi_gp)):
            nb_, qb = by_node(hb, pb); nv_, qv = by_node(hv, pv)
            errs.append(float((qb - qv).abs().max()) if torch.equal(nb_, nv_) else float('inf'))
        OH.drop(m, v)
    check('pi_partition_and_views', max(e_el, e_gp) <= 1e-12 and max(e_x, e_xf) <= 1e-12 and fac_ok and max(errs) <= 1e-12,
          elem_sum_err=e_el, face_sum_err=e_gp, fringe_elem_sum_err_weak=e_x, fringe_face_sum_err_weak=e_xf,
          weak_nodes=int(wk.sum()), view_pi_max_abs=max(errs), fold_factor_ok=fac_ok,
          fringe_subset_sum_range=[float(s_x[s_x > 0].min()), float(s_x.max())])


def t3_fastnet(g, ck, Q, **over):
    gen = torch.Generator().manual_seed(1)
    m, _ = build(g, ck, b3=True, **over)
    with torch.no_grad():                                               # distinct lambdas per layer around 0.7
        for k in ('b3_le', 'b3_lf', 'b3_lx'):
            p = getattr(m, k); p.copy_(0.7 + 0.2 * torch.randn(p.shape, generator=gen))
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
    tag = ','.join(k for k, v in over.items() if v) or 'plain'
    check(f'fastnet_lambda07[{tag}]', max(r_field, r_adj, r_shat, r_T) <= 1e-5, field_rel=r_field, adjoint_identity_rel=r_adj,
          s_hat_rel=r_shat, model_T_rel=r_T)


def t3b_mgno_v0(g, Q):
    """MGNO v0 (element layers only; random weights) with b3: FastNet field and s_hat at lambda = 0.7."""
    torch.manual_seed(3)
    m = MD.build('mgno', [g], b3=True).eval()
    with torch.no_grad():
        m.b3_le.fill_(0.7)
    fast = FN.FastNet(m, g)
    with torch.no_grad():
        u_ref = g.field(m, Q)
    r_field, r_shat = rel(fast.field(Q), u_ref), rel(fast.s_hat(Q), g.s_hat_apply(m, Q))
    check('fastnet_lambda07[mgno_v0_random]', max(r_field, r_shat) <= 1e-5, field_rel=r_field, s_hat_rel=r_shat)


def t4_compat(g, ck):
    m, info = build(g, ck, b3=True)
    kept_ok = sorted(info['kept']) == ['b3_le', 'b3_lf', 'b3_lx'] and all(float(getattr(m, k).abs().max()) == 0 for k in info['kept'])
    set_lam(m, 0.3)
    sd = {k: v.clone() for k, v in m.state_dict().items()}
    m2, info2 = build(g, ck, b3=True)
    MD.load_compat(m2, sd)
    rt = all(torch.equal(sd[k], v) for k, v in m2.state_dict().items())
    old, _ = build(g, ck)
    try:
        MD.load_compat(old, sd); raised = False
    except KeyError:
        raised = True
    check('load_compat', kept_ok and rt and raised, kept=info['kept'], padded=info['padded'], b3_round_trip=rt,
          b3_state_into_old_raises=raised)


def main():
    t0 = time.perf_counter()
    torch.manual_seed(0)
    g = FX.small()
    ck = FX.checkpoint(CK)
    gen = torch.Generator().manual_seed(0)
    Q = torch.cat([torch.cat([g.banks['val'][c][:, :1] for c in g.classes], 1).to(f64)[:, :2],
                   torch.randn((g.np_, 2), dtype=f64, generator=gen)], 1)
    t0_table(g)
    m = t1_identity(g, ck, Q)
    t2_lambda(g, ck, m, Q)
    t3_fastnet(g, ck, Q)
    t3_fastnet(g, ck, Q, feat_v2=True, fringe_soft=True)
    t3b_mgno_v0(g, Q)
    t4_compat(g, ck)
    print(json.dumps(dict(ALL_PASSED=all(r['ok'] for r in REP.values()), n=len(REP), seconds=time.perf_counter() - t0)), flush=True)


if __name__ == '__main__':
    main()
