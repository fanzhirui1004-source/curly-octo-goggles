"""Local CPU tests of oh.py (B6 O_h views) on the fixture.
  1. group: 48 distinct signed permutations, ELEMS[0] = I, closed under composition and inverse, index_of; rotate_nd is a
     group action on NETDATA (rotate_nd(rotate_nd(nd, A), B) == rotate_nd(nd, B A), exact) for all 48 x 48 pairs.
  2. rotate_nd(base nd, R) vs the NETDATA of the independently prepared rotated packets fresh_train_0021_cover01_r1_rot*
     (teacher run on the rotated geometry), after matching nodes (v_rot.node_map) and elements (cell_img): grid, node ids,
     elem_cells, elem_nodes slot contents (exact), moments (<= 1e-10 of the full-element volume), gp_faces as ORIENTED
     triples (owner, nbr, axis) (exact), diag3 (<= 1e-9 of max |diag3|), normal (exact), offset (<= 1e-14), taus
     (exact), node flags (exact); plus a negative control (R^T in place of R must not match).
  3. model level on the small geometry (both fixture checkpoints, and r2 with feat_v2 + fringe_soft): identity view ==
     Geo.field bit-for-bit (field and s_hat_apply); on all 48 views (step-2 snapshot; 8 views for the step-1 checkpoint,
     which is zero-shot here): port values exact (bitwise), linear in q, rigid part exact (step-2 snapshot), energy >=
     exact energy of the fixture's exact extensions, network output == that of the materialized rotated geometry (nodes /
     elements / faces in packet order, i.e. what a rotated packet's NETDATA would be); per-view mean energy error
     (orientation sensitivity, reported); evaluate() and S_hat symmetry on a view; drop().
  4. MGCache / MGNO2 face setup (and feat_v2 + fringe_soft caches) built from rotate_nd(0021 base) vs built from the
     rotated packets' own NETDATA, after node / element / face matching: element slots, element and node features, grid
     transfer (as sets), face stencils slot by slot, fringe masks and fades, geometry-path outputs, network output.
  5. operator meaning of the moment rule: element stiffness from rotated moments with the teacher's Tm == the rotated
     element stiffness (slot and component permutation), all 48 elements, <= 1e-12.
  6. smoke run of t_oh_remote.part_a / part_b on CPU against a synthetic rotated packet of the small geometry (packet
     node / element order, K = Pi K Pi^T, dM with corners / elements / moments mapped): energies, fields and sensitivities
     of the view vs the packet.
Usage: OPL_DEV=cpu python3 t_oh.py"""
import os, sys, json, time, types
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
import trainlib as TL
import oh

CKS = ('mgno2_r2_d2_best.pt', 's2v1_snap_10000.pt')
f64 = torch.float64
NODE_KEYS = ('node_ids', 'grid', 'is_box', 'is_cut', 'is_port', 'weak', 'diag3')
REP = {}


def check(name, ok, **info):
    REP[name] = dict(ok=bool(ok), **info)
    print(json.dumps({name: REP[name]}, default=float), flush=True)
    if not ok:
        raise AssertionError(name)


def rel(a, b):
    a, b = a.to(f64), b.to(f64)
    return float((a - b).norm() / b.norm().clamp_min(1e-300))


def cellkey(c, n=32):
    c = np.asarray(c, dtype=np.int64)
    return (c[:, 0] * n + c[:, 1]) * n + c[:, 2]


def inv(p):
    q = np.empty_like(p); q[p] = np.arange(len(p))
    return q


def match(v, r):
    """Node / element / face maps view -> packet: packet[nm[a]] = view[a], packet[em[e]] = view[e], packet face fm[f]."""
    nm = np.searchsorted(r['node_ids'], v['node_ids'])
    okn = bool((nm < len(r['node_ids'])).all() and np.array_equal(r['node_ids'][np.minimum(nm, len(nm) - 1)], v['node_ids']))
    kr, kv = cellkey(r['elem_cells']), cellkey(v['elem_cells'])
    ro = np.argsort(kr); em = ro[np.minimum(np.searchsorted(kr[ro], kv), len(kr) - 1)]
    oke = bool(np.array_equal(kr[em], kv))
    E = len(kr)
    fk = lambda f, e: (e[f[:, 0]].astype(np.int64) * E + e[f[:, 1]]) * 3 + f[:, 2]
    fr, fv = fk(r['gp_faces'], np.arange(E)), fk(v['gp_faces'], em)
    fo = np.argsort(fr); fm = fo[np.minimum(np.searchsorted(fr[fo], fv), len(fr) - 1)]
    okf = bool(len(fr) == len(fv) and np.array_equal(fr[fm], fv) and len(np.unique(fm)) == len(fm))
    return nm, em, fm, okn, oke, okf


def materialize(nd):
    """nd put in packet order (nodes by node id, elements by cell key, faces by (owner, axis)), and the maps."""
    no = np.argsort(nd['node_ids']); nm = inv(no)
    eo = np.argsort(cellkey(nd['elem_cells'])); em = inv(eo)
    out = dict(nd)
    for k in NODE_KEYS:
        out[k] = nd[k][no]
    out['elem_cells'], out['moments'] = nd['elem_cells'][eo], nd['moments'][eo]
    out['elem_nodes'] = nm[nd['elem_nodes'][eo]].astype(nd['elem_nodes'].dtype)
    f = nd['gp_faces']
    f2 = np.stack([em[f[:, 0]], em[f[:, 1]], f[:, 2]], 1)
    out['gp_faces'] = f2[np.lexsort((f2[:, 2], f2[:, 0]))].astype(f.dtype)
    return out


def light(case, nd):
    return types.SimpleNamespace(case=case, nd=nd)


def to_packet(qd, is_port_v, is_port_r, nm):
    """Port DOF block in view port order -> packet port order (node permutation only; components already rotated)."""
    pv = np.flatnonzero(is_port_v); pr = np.flatnonzero(is_port_r)
    pos = np.searchsorted(pr, nm[pv])
    B = qd.shape[1]
    out = torch.zeros_like(qd).reshape(-1, 3, B)
    out[torch.as_tensor(pos)] = qd.reshape(-1, 3, B)
    return out.reshape(-1, B)


def model_variants(g):
    """(name, model): the step-1 r2 checkpoint (dense) and the step-2 v1 snapshot (sparse layers) as built, plus the
    feat_v2 + fringe_soft variant (load_compat) to exercise node_feats_v2 and the soft fringe caches."""
    out = [(ck, FX.model_for(g, ck)) for ck in CKS]
    ck = FX.checkpoint(CKS[0]); args = dict(ck['cfg']['model_args'], feat_v2=True, fringe_soft=True)
    m = MD.build(ck['cfg']['model'], [g], **args); MD.load_compat(m, ck['model'])
    out.append(('r2+feat_v2+fringe_soft', m.eval()))
    return out


# ------------------------------------------------------------------------------------------------------------------ 1
def t1_group(nd):
    E = oh.ELEMS
    flat = {tuple(R.reshape(-1)) for R in E}
    signed = all(sorted(np.abs(R).sum(0)) == [1, 1, 1] and sorted(np.abs(R).sum(1)) == [1, 1, 1] and set(np.unique(R)) <= {-1, 0, 1}
                 for R in E)
    closed = all(tuple((A @ B).reshape(-1)) in flat for A in E for B in E)
    inverse = all(tuple(R.T.reshape(-1)) in flat for R in E)
    idx = all(oh.index_of(R) == k for k, R in enumerate(E))
    try:
        oh.index_of(np.eye(3) * 2); bad = False
    except ValueError:
        bad = True
    dets = [round(float(np.linalg.det(R))) for R in E]
    check('group', len(E) == 48 and len(flat) == 48 and np.array_equal(E[0], np.eye(3, dtype=int)) and signed and closed and inverse
          and idx and bad and dets.count(1) == 24, proper=dets.count(1))
    t = time.perf_counter()
    base = {k: oh.rotate_nd(nd, A) for k, A in enumerate(E)}
    worst_off, exact = 0.0, True
    for a, A in enumerate(E):
        for B in E:
            two = oh.rotate_nd(base[a], B); one = base[oh.index_of(B @ A)]
            for k in nd:
                if k == 'offset':
                    worst_off = max(worst_off, abs(float(two[k]) - float(one[k])))
                elif not (np.array_equal(two[k], one[k]) and np.asarray(two[k]).dtype == np.asarray(one[k]).dtype):
                    exact = False
                    print(json.dumps(dict(event='ACTION_MISMATCH', key=k, A=A.tolist(), B=B.tolist())), flush=True)
    ident = oh.rotate_nd(nd, E[0])
    same = all(np.array_equal(ident[k], nd[k]) and np.asarray(ident[k]).dtype == np.asarray(nd[k]).dtype for k in nd)
    check('group_action_on_nd', exact and worst_off <= 1e-14 and same, pairs=48 * 48, max_offset_diff=worst_off,
          identity_exact=same, seconds=time.perf_counter() - t)
    try:
        oh.rotate_nd(dict(nd, extra=np.zeros(1)), E[1]); raised = False
    except KeyError:
        raised = True
    check('unknown_key_raises', raised)


# ------------------------------------------------------------------------------------------------------------------ 2
def t2_packets(base):
    sc_m = float(np.abs(base['moments'][:, 0]).max())                        # full-element volume
    sc_d = float(np.abs(base['diag3']).max())
    for rn, R in FX.ROTS.items():
        R = np.asarray(R)
        r = FX.netdata(f'{FX.ROT_BASE}_rot{rn}')
        v = oh.rotate_nd(base, R)
        nm, em, fm, okn, oke, okf = match(v, r)
        img = oh.node_map(base['node_ids'], R)
        cimg = oh.cell_img(base['elem_cells'], R)
        dm = np.abs(v['moments'] - r['moments'][em]).max(1) / sc_m
        full = base['moments'][:, 0] / sc_m > 1 - 1e-12
        dd = np.abs(v['diag3'] - r['diag3'][nm])
        dnode = np.linalg.norm(dd.reshape(-1, 9), axis=1) / np.linalg.norm(r['diag3'][nm].reshape(-1, 9), axis=1)
        flags = {k: bool(np.array_equal(v[k], r[k][nm])) for k in ('is_box', 'is_cut', 'is_port', 'weak')}
        info = dict(nodes=okn, elements=oke, oriented_faces=okf, faces=len(fm),
                    node_map=bool(np.array_equal(img, v['node_ids'])), cell_img=bool(np.array_equal(cimg, v['elem_cells'])),
                    grid=bool(np.array_equal(v['grid'], r['grid'][nm])), node_ids=bool(np.array_equal(v['node_ids'], r['node_ids'][nm])),
                    elem_cells=bool(np.array_equal(v['elem_cells'], r['elem_cells'][em])),
                    elem_slots=bool(np.array_equal(nm[v['elem_nodes']], r['elem_nodes'][em])),
                    moments_max=float(dm.max()), moments_full_max=float(dm[full].max()), full_elements=int(full.sum()),
                    diag3_max=float(dd.max() / sc_d), diag3_node_rel_max=float(dnode.max()),
                    normal=bool(np.array_equal(v['normal'], r['normal'])), offset_diff=float(abs(v['offset'] - r['offset'])),
                    taus=bool(np.array_equal(v['taus'], r['taus'])), flags=flags, n=int(v['n']) == int(r['n']))
        ok = okn and oke and okf and all(info[k] for k in ('node_map', 'cell_img', 'grid', 'node_ids', 'elem_cells', 'elem_slots',
                                                          'normal', 'taus', 'n')) and all(flags.values()) and \
            info['moments_max'] <= 1e-10 and info['moments_full_max'] <= 1e-14 and info['diag3_max'] <= 1e-9 and info['offset_diff'] <= 1e-14
        check(f'nd_vs_packet[{rn}]', ok, **info)
    # negative control: the inverse element in place of R (general has order 3, so R^T != R)
    R = np.asarray(FX.ROTS['general'])
    r = FX.netdata(f'{FX.ROT_BASE}_rotgeneral')
    w = oh.rotate_nd(base, R.T)
    nm, em, fm, okn, oke, okf = match(w, r)
    check('negative_control[general,R^T]', not (okn and oke and okf), nodes=okn, elements=oke, faces=okf)
    # and a wrong moment rule (no signs) is caught on the inversion packet
    R = np.asarray(FX.ROTS['invert'])
    r = FX.netdata(f'{FX.ROT_BASE}_rotinvert')
    v = oh.rotate_nd(base, R)
    nm, em, fm, *_ = match(v, r)
    src, sgn = oh.moment_map(R)
    bad = float(np.abs(base['moments'][:, src] - r['moments'][em]).max() / sc_m)
    check('negative_control[invert,unsigned moments]', bad > 1e-3, max_diff=bad)


# ------------------------------------------------------------------------------------------------------------------ 3
def t3_model_small(g):
    Q = torch.cat([g.banks['val'][c][:, :2] for c in g.classes], 1)                  # 10 unit-energy directions
    Ustar = torch.cat([torch.as_tensor(g.fix_ustar_val[c][:, :2]) for c in g.classes], 1).to(f64)
    e_star = TL.energy(Ustar, g.C.K)                                                 # exact energies (= 1 up to fp32 q)
    gen = torch.Generator().manual_seed(0)
    q1, q2 = torch.randn((g.np_, 2), generator=gen, dtype=f64), torch.randn((g.np_, 2), generator=gen, dtype=f64)
    crig = torch.randn((6, 2), generator=gen, dtype=f64)
    qr = g.RP @ crig
    for name, m in model_variants(g):
        with torch.no_grad():
            u0 = g.field(m, Q)
            v0 = oh.view(m, g, 0)
            same_f = torch.equal(v0.field(m, Q), u0)
            rig0 = rel(g.field(m, qr), g.RA @ crig)
        nt = torch.get_num_threads(); torch.set_num_threads(1)           # the CPU backward is bitwise reproducible
        same_s = torch.equal(v0.s_hat_apply(m, Q[:, :4]), g.s_hat_apply(m, Q[:, :4]))   # only single-threaded
        torch.set_num_threads(nt)
        oh.drop(m, v0)
        check(f'identity_view_bitwise[{name}]', same_f and same_s and v0.case not in m.caches, field=same_f, s_hat=same_s)
        # the step-2 snapshot (trained on the pool) on all 48 views; the single-geometry step-1 checkpoint (zero-shot here:
        # e_hat ~ 1e10, it amplifies the fp32 rounding of the rigid split) on 8 of them, rigid part only reported
        sane = name == CKS[1]
        ks = range(48) if sane else (0, 1, 7, 13, 22, 29, 38, 47)
        worst = dict(port=True, lin=0.0, rigid=0.0, rigid_base=rig0, ub=np.inf, equiv=0.0)
        mean_err = {}
        t = time.perf_counter()
        for k in ks:
            v = oh.view(m, g, k)
            with torch.no_grad():
                u = v.field(m, Q)
                worst['port'] &= torch.equal(u[g.P], Q.to(torch.float32))
                a, b = 0.7, -1.3
                ul = v.field(m, a * q1 + b * q2); ur = a * v.field(m, q1) + b * v.field(m, q2)
                worst['lin'] = max(worst['lin'], rel(ul, ur))
                worst['rigid'] = max(worst['rigid'], rel(v.field(m, qr), g.RA @ crig))
                e = TL.energy(u, g.C.K)
                worst['ub'] = min(worst['ub'], float((e / e_star).min()))
                mean_err[k] = float((e / e_star - 1).mean())
                # the view's network == the network on the rotated geometry in packet order
                mat = materialize(v.nd)
                lg = light(f'{g.case}@mat{k}', mat); m.add_geo(lg)
                nm = np.searchsorted(mat['node_ids'], v.nd['node_ids'])
                qd = oh.rot_dofs(q1.to(torch.float32), *v.oh_ps)
                a_ = m(v, qd)
                b_ = m(lg, to_packet(qd, v.nd['is_port'], mat['is_port'], nm)).reshape(-1, 3, qd.shape[1])[torch.as_tensor(nm)].reshape(a_.shape)
                worst['equiv'] = max(worst['equiv'], rel(a_, b_))
                m.drop_geo(lg.case)
            oh.drop(m, v)
        me = np.array(list(mean_err.values()))
        ok = worst['port'] and worst['lin'] <= 1e-5 and worst['ub'] >= 1 - 1e-6 and worst['equiv'] <= 1e-5 and (not sane or worst['rigid'] <= 1e-5) \
            and not any('@' in c for c in m.caches)
        check(f'views[{name}]', ok, views=len(mean_err), **worst, mean_rel_energy_err=dict(identity=mean_err[0], min=float(me.min()),
              median=float(np.median(me)), max=float(me.max()), worst_view=int(list(mean_err)[int(me.argmax())])),
              seconds=time.perf_counter() - t)
    # evaluate() through a view (energies with the original K, sensitivities with the original dM); S_hat symmetric
    m = FX.model_for(g, CKS[1])
    ev0 = g.evaluate(m, 'val', chunk=8)
    v = oh.view(m, g, 29)
    ev = v.evaluate(m, 'val', chunk=8)
    fin = all(np.isfinite(list(d.values())).all() for d in ev.values())
    S6 = Q[:, :6].to(f64).T @ v.s_hat_apply(m, Q[:, :6])
    asym = float((S6 - S6.T).abs().max() / S6.abs().max())
    check('view_evaluate_and_s_hat', fin and asym <= 1e-5 and v.case not in (oh.drop(m, v), m.caches)[1], s_hat_asym=asym,
          evaluate={c: dict(mean=d['mean'], mean_identity=ev0[c]['mean'], sens_mean=d.get('sens_mean'),
                            sens_mean_identity=ev0[c].get('sens_mean')) for c, d in ev.items()})


# ------------------------------------------------------------------------------------------------------------------ 4
def t4_caches(g, base):
    for name, m in model_variants(g):
        if name == CKS[1]:
            continue                                                             # same caches as CKS[0]; forward below
        for rn, R in FX.ROTS.items():
            R = np.asarray(R); k = oh.index_of(R)
            r = FX.netdata(f'{FX.ROT_BASE}_rot{rn}')
            gv, gr = light(f'{FX.ROT_BASE}@oh{k}', oh.rotate_nd(base, R)), light(f'{FX.ROT_BASE}_rot{rn}', r)
            m.add_geo(gv); m.add_geo(gr)
            cv, cr = m.caches[gv.case], m.caches[gr.case]
            nm, em, fm, okn, oke, okf = match(gv.nd, r)
            NM, EM, FM = map(torch.as_tensor, (nm, em, fm))
            info = dict(match=okn and oke and okf)
            info['elem_slots'] = bool(torch.equal(NM[cv.en], cr.en[EM]))
            info['deg'] = bool(torch.equal(cv.deg, cr.deg[NM]))
            info['efeat'] = float((cv.efeat - cr.efeat[EM]).abs().max())
            info['nfeat'] = float((cv.nfeat - cr.nfeat[NM]).abs().max())
            tr = True
            for l, (tv, tr_) in enumerate(zip(cv.trans, cr.trans)):
                iv = NM[tv['i']] if l == 0 else tv['i']
                sv = torch.stack([iv.to(f64), tv['v'].to(f64), tv['w'].to(f64)], 1)
                sr = torch.stack([tr_['i'].to(f64), tr_['v'].to(f64), tr_['w'].to(f64)], 1)
                sv, sr = sv[np.lexsort(sv.T.numpy()[::-1])], sr[np.lexsort(sr.T.numpy()[::-1])]
                tr &= tv['n_dst'] == tr_['n_dst'] and tv['m'] == tr_['m'] and torch.equal(tv['dense_idx'], tr_['dense_idx']) and torch.equal(sv, sr)
            info['transfer'] = bool(tr)
            info['face_ends'] = bool(torch.equal(EM[cv.gp_owner], cr.gp_owner[FM]) and torch.equal(EM[cv.gp_nbr], cr.gp_nbr[FM])
                                     and torch.equal(cv.gp_axis, cr.gp_axis[FM]))
            info['face_slots'] = bool(torch.equal(NM[cv.fn], cr.fn[FM]))
            info['fdeg'] = bool(torch.equal(cv.fdeg, cr.fdeg[NM]))
            info['fringe'] = bool(torch.equal(cv.el_fringe, cr.el_fringe[EM]) and torch.equal(cv.gp_fringe, cr.gp_fringe[FM]))
            tol = dict(efeat=1e-4, nfeat=1e-5)
            if hasattr(cv, 'nfeat2'):
                info['nfeat2'] = float((cv.nfeat2 - cr.nfeat2[NM]).abs().max())

                def full(mask, fade):
                    z = torch.zeros(len(mask)); z[mask] = fade
                    return z
                info['fades'] = max(float((full(cv.el_fringe, cv.el_fade) - full(cr.el_fringe, cr.el_fade)[EM]).abs().max()),
                                    float((full(cv.gp_fringe, cv.gp_fade) - full(cr.gp_fringe, cr.gp_fade)[FM]).abs().max()))
                tol.update(nfeat2=1e-4, fades=1e-4)
            with torch.no_grad():
                pv, pr = m.geometry(gv.case), m.geometry(gr.case)
                gd = dict(ab=rel(pv['ab'], pr['ab'][EM]), fab=rel(pv['fab'], pr['fab'][FM]), xab=rel(pv['xab'], pr['xab'][EM]),
                          rw0=rel(torch.stack(pv['rw'][0], 1), torch.stack(pr['rw'][0], 1)[NM]),
                          gates=max(rel(a, b) for a, b in zip(pv['gates'], pr['gates'])))
                info['geometry_rel'] = gd
                gen = torch.Generator().manual_seed(1)
                qd = torch.randn((3 * int(gv.nd['is_port'].sum()), 3), generator=gen)
                a_ = m(gv, qd)
                b_ = m(gr, to_packet(qd, gv.nd['is_port'], r['is_port'], nm)).reshape(-1, 3, 3)[NM].reshape(a_.shape)
                info['forward_rel'] = rel(a_, b_)
                if name == CKS[0]:                                               # the sparse-layer checkpoint on the same caches
                    ms = FX.model_for(g, CKS[1]); ms.caches[gv.case], ms.caches[gr.case] = cv, cr
                    a_ = ms(gv, qd)
                    b_ = ms(gr, to_packet(qd, gv.nd['is_port'], r['is_port'], nm)).reshape(-1, 3, 3)[NM].reshape(a_.shape)
                    info['forward_rel_sparse_ckpt'] = rel(a_, b_)
            ok = info['match'] and all(info[k] for k in ('elem_slots', 'deg', 'transfer', 'face_ends', 'face_slots', 'fdeg', 'fringe')) \
                and all(info[k] <= t_ for k, t_ in tol.items()) and max(gd.values()) <= 1e-4 and info['forward_rel'] <= 1e-4 \
                and info.get('forward_rel_sparse_ckpt', 0.0) <= 1e-4
            check(f'caches_vs_packet[{name},{rn}]', ok, **info)
            m.drop_geo(gv.case); m.drop_geo(gr.case)


# ------------------------------------------------------------------------------------------------------------------ 5
def t5_element_stiffness(g):
    """Element stiffness from rotated moments with the teacher's Tm == the rotated element stiffness (all 48 elements):
    K'_e[(t, i), (t', j)] = s_i s_j K_e[(sp_t, p_i), (sp_t', p_j)], K_e = sum_m M_em Tm_m (the operator-level meaning of the
    moment rule; the network reads moments only as features)."""
    nd, Tm = g.nd, g.C.Tm.to(f64)
    Ke = torch.einsum('em,mij->eij', torch.as_tensor(nd['moments']), Tm).reshape(-1, 27, 3, 27, 3)
    offs = nd['grid'][nd['elem_nodes'][0]].astype(np.int64) - 2 * nd['elem_cells'][0].astype(np.int64)
    worst = 0.0
    for R in oh.ELEMS:
        v = oh.rotate_nd(nd, R)
        p, s = oh.perm_sign(R)
        sp = torch.as_tensor(oh.slot_perm(R, offs)); P_, S_ = torch.as_tensor(p), torch.as_tensor(s, dtype=f64)
        K2 = torch.einsum('em,mij->eij', torch.as_tensor(v['moments']), Tm).reshape(-1, 27, 3, 27, 3)
        img = Ke[:, sp][:, :, P_][:, :, :, sp][:, :, :, :, P_] * (S_[:, None] * S_[None, :])[None, None, :, None, :]
        worst = max(worst, float((K2 - img).abs().max() / Ke.abs().max()))
    check('element_stiffness_equivariance', worst <= 1e-12, max_rel=worst, elements=int(Ke.shape[0]))


# ------------------------------------------------------------------------------------------------------------------ 6
class _PermK:
    """K of the synthetic rotated geometry, K' = Pi K Pi^T with (Pi x)[nm[a]] = R x[a] (node image nm, components R)."""

    def __init__(self, K, nm, p, s):
        self.K, self.nm, self.ps, self.pst = K, torch.as_tensor(nm), (p, s), (np.argsort(p), s[np.argsort(p)])

    def __matmul__(self, x):
        B = x.shape[1]
        xb = oh.rot_dofs(x.reshape(-1, 3, B)[self.nm].reshape(-1, B), *self.pst)
        out = torch.zeros_like(x).reshape(-1, 3, B)
        out[self.nm] = oh.rot_dofs(self.K @ xb, *self.ps).reshape(-1, 3, B)
        return out.reshape(-1, B)


def synthetic_packet(g, R, case):
    """What a rotated packet's trainlib.Geo would be (nodes, elements, ports in packet order; K = Pi K Pi^T exactly; dofs,
    Tm and dM for sens_hat with corners, elements and moments mapped), built from the small geometry."""
    import ops as OP
    p, s = oh.perm_sign(R)
    v = oh.rotate_nd(g.nd, R)
    mat = materialize(v)
    nm = np.searchsorted(mat['node_ids'], v['node_ids'])
    em = np.argsort(np.argsort(cellkey(v['elem_cells'])))
    n = int(g.nd['n'])
    ports = mat['node_ids'][mat['is_port']]
    C = types.SimpleNamespace(n=n, nodes=mat['node_ids'], port_node_ids=ports, K=_PermK(g.C.K, nm, p, s),
                              dofs=torch.as_tensor(3 * mat['elem_nodes'].astype(np.int64)[:, :, None] + np.arange(3)).reshape(-1, 81))
    r = types.SimpleNamespace(case=case, nd=mat, C=C, np_=3 * len(ports), nb=3 * len(mat['node_ids']), Tm32=g.Tm32)
    r.P = torch.nonzero(torch.as_tensor(np.repeat(mat['is_port'], 3))).squeeze(1)
    ctr = (np.stack(np.unravel_index(ports, (2 * n + 1,) * 3), 1) / (2 * n)).mean(0)
    r.RP, r.RA = OP.rigid_raw(ports, n, ctr), OP.rigid_raw(mat['node_ids'], n, ctr)
    r.RPpinv = torch.linalg.pinv(r.RP)
    src, sgn = oh.moment_map(R)
    dM = torch.zeros_like(g.dM32)
    dM[torch.as_tensor(oh.corner_perm(R))[:, None], torch.as_tensor(em)[None, :]] = g.dM32[:, :, torch.as_tensor(src)] * torch.as_tensor(sgn, dtype=torch.float32)
    r.dM32 = dM
    r.field = types.MethodType(TL.Geo.field, r)
    r.sens_hat = types.MethodType(TL.Geo.sens_hat, r)
    return r


def t6_remote_smoke(g):
    """t_oh_remote.part_a / part_b on CPU: base = the small geometry (two classes, 8 val columns), rotated packet =
    synthetic_packet (so the remote script's node / port / corner maps and criteria run once before they meet the GPU)."""
    import copy
    import t_oh_remote as TR
    TR.NTRAIN = 0
    old = os.environ.get('SENS_REASSOC'); os.environ['SENS_REASSOC'] = '1'           # same map, 8 instead of 125 contractions
    gs = copy.copy(g); gs.classes = ['force', 'support']
    gs.banks = {'val': {c: g.banks['val'][c][:, :8] for c in gs.classes}}
    m = FX.model_for(g, CKS[1])
    rec, chk = [], TR.check
    TR.check = lambda name, ok, **info: (rec.append(dict(info, ok=ok)), chk(name, ok, **info))
    try:
        for rn in ('general', 'invert'):
            R = np.asarray(FX.ROTS[rn]); k = oh.index_of(R)
            rot = synthetic_packet(g, R, f'{g.case}_synrot{rn}')
            m.add_geo(rot)
            v = oh.view(m, gs, k)
            TR.part_a(m, gs, v, rot, f'synthetic_{rn}', R, chunk=8)
            oh.drop(m, v); m.drop_geo(rot.case)
        TR.part_b(m, gs, chunk=8, ks=(0, 12, 29))
    finally:
        os.environ.pop('SENS_REASSOC') if old is None else os.environ.__setitem__('SENS_REASSOC', old)
        TR.check = chk
    check('remote_script_smoke', not TR.FAIL and len(rec) == 3 and all(r['ok'] and r.get('sens_ok', True) and r.get('sens_checked', True)
                                                                        for r in rec), failed=list(TR.FAIL))


def main():
    t0 = time.perf_counter()
    g = FX.small()
    base = FX.netdata(FX.ROT_BASE)
    t1_group(g.nd)
    t2_packets(base)
    t3_model_small(g)
    t4_caches(g, base)
    t5_element_stiffness(g)
    t6_remote_smoke(g)
    print(json.dumps(dict(event='ALL_PASSED', tests=len(REP), seconds=time.perf_counter() - t0)), flush=True)


if __name__ == '__main__':
    main()
