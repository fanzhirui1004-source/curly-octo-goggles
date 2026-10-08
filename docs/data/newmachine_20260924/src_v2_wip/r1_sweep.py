"""Revision round 1, E13 (I-31): one-corner thickness sweeps of the NICE surrogate across discrete switches. New script.

A sweep varies ONE corner tau_c of one cell over tau_c0 (1 + 0.10 t), t in [-1, 1] ('k', --coarse points) and over the
finer window tau_c0 (1 + 0.005 t) ('f', --fine points). Every point is a NEW geometry: a derived packet (FRESH_CONTEXT
with that corner changed, SAMPLE copied) and a NEW body from fast_prep4 (so element activation, ghost faces and the
retained set follow the design, unlike the fixed-body derivative of r1_pairs / r1_lat). Per point:
  learned cell rebuilt from scratch (moments, K, NETDATA, Geo, fresh model cache, fastnet, correction caches of this K);
  assembled solve with the other cells unchanged: 'pair' = the gate configuration (lattice3.Lattice, consistent loads
  recomputed at this design: 3 test-face + 3 neighbour-face gate loads and, for a cut test cell, 3 cut-surface loads;
  neighbour exact through its dense S), PCG to 1e-10 preconditioned by the EXACT dense lattice factor of the base design
  (mapped by absolute node key; DOFs absent from the base get the inverse diagonal); 'lattice' = lat_multi.MultiLattice
  (clamp y min, three consistent loads on y max, as Section 6.9), bnn:kpp:q1r, --lat-tol;
  C_hat = f^T U_bar, J = C_hat + U_bar^T rho, true residual; s_tilde_c = -u^T K_,c u (u = recovered field, K_,c by central
  moment differences h = 1e-5 tau_c on this body); switch fingerprint: active elements, ghost faces, box / cut port nodes
  (count + hash), weak-flag node ids, fringe hyperedge counts, coarse Cholesky shift, Chebyshev endpoint b.
  At reference points (--ref-every on the coarse grid, --ref-fine-every on the fine grid): exact C(tau), s_c(tau) (teacher
  interior factor, same solve), the test cell's energy error eps at the exact traces and beta = w eps.
Points are visited in a coarse-to-fine interleaved order, so a run stopped by --deadline still covers the whole range.
Subcommands:
  plan  <root> <spec> ...   derived packets + the ordered point list (<root>/<tag>/plan.json); prints the body case list
  run   <root> <tag> [--deadline EPOCH]   GPU sweep (waits for each body; skips a point whose body failed)
  summary <root>            per sweep: jumps between consecutive points, switch attribution, reference jumps, beta C
spec = tag:kind:case:config_or_layout:corner   (corner: an integer, 'max' or 'med' from the E3 records, 'lat' = the
       largest-|s| corner of a CUT cell at a vertex no other cell shares, else that cell's largest corner)
"""
import json, os, sys, time, hashlib, argparse, shutil, glob
from fractions import Fraction
from pathlib import Path
import numpy as np

SEL_LEVELS = [('k', 50), ('f', 25), ('k', 20), ('f', 10), ('k', 10), ('f', 5), ('k', 5), ('f', 2), ('k', 2), ('f', 1), ('k', 1)]
S0, S3P, S4 = '/root/autodl-tmp/OPL/S0', '/root/autodl-tmp/OPL/S3/packets', '/root/autodl-tmp/OPL/S4'
X3 = '/root/autodl-tmp/OPL/S1/V2/R1/X3'


def order(nk, nf):
    ck, cf = nk // 2, nf // 2
    seen, out = set(), [('k', ck)]
    seen.add(('k', ck))
    for g, s in SEL_LEVELS:
        n, c = (nk, ck) if g == 'k' else (nf, cf)
        for i in range(n):
            if (i - c) % s == 0 and (g, i) not in seen and not (g == 'f' and i == cf):
                seen.add((g, i)); out.append((g, i))
    for i in range(nk):
        if ('k', i) not in seen:
            out.append(('k', i))
    for i in range(nf):
        if ('f', i) not in seen and i != cf:
            out.append(('f', i))
    return out


def packet_root(case):
    for r in [p for p in os.environ.get('OPL_PACKETS_EXTRA', '').split(':') if p] + ['/root/autodl-tmp/CUTFEM_FRESH_GP_20260921/packets']:
        if (Path(r) / case / 'FRESH_CONTEXT.json').exists():
            return Path(r) / case
    raise FileNotFoundError(case)


def pick_corner(kind, case, conf, how):
    if how not in ('max', 'med', 'lat'):
        return int(how), case, 'given'
    try:
        if kind == 'pair':
            r = json.loads(Path(f'{X3}/pairs/pair_{case}.json').read_text())
            c = r['configs'][conf]
            s = np.asarray(c['sens_exact'], float)[:, np.asarray(c['gate_mask'], bool)]
            o = np.argsort(-np.linalg.norm(s, axis=1))
            return int(o[0] if how == 'max' else o[len(o) // 2]), case, 'E3 exact s'
        L = json.loads(Path(conf).read_text())
        z = np.load(f'{X3}/lat64_222.npz')
        S = z[f"{L['name']}__S"]                                                      # cells x 8 x loads (layout order)
        from element_polyref import CUBE
        cnt = {}
        for c in L['cells']:
            for k in range(8):
                v = tuple(int(a) + int(b) for a, b in zip(c['position'], CUBE[k])); cnt[v] = cnt.get(v, 0) + 1
        best = None
        for i, c in enumerate(L['cells']):
            if c['kind'] != 'CUT':
                continue
            g = np.linalg.norm(S[i][:, :3], axis=1)
            for k in range(8):
                v = tuple(int(a) + int(b) for a, b in zip(c['position'], CUBE[k]))
                if cnt[v] == 1 and g[k] >= 0.1 * g.max() and (best is None or g[k] > best[0]):
                    best = (g[k], k, c['case'], 'unshared vertex')
        if best is None:
            i = max((i for i, c in enumerate(L['cells']) if c['kind'] == 'CUT'), key=lambda i: np.linalg.norm(S[i][:, :3]))
            g = np.linalg.norm(S[i][:, :3], axis=1)
            best = (g.max(), int(np.argmax(g)), L['cells'][i]['case'], 'cell corner (shared vertex; per-cell variation)')
        return int(best[1]), best[2], best[3]
    except Exception as e:                                                              # noqa: BLE001
        return (4 if how == 'max' else 0), (case if kind == 'pair' else json.loads(Path(conf).read_text())['cells'][2]['case']), f'fallback ({e!r})'[:200]


def plan(root, specs, nk0, nf0):
    root = Path(root); (root / 'packets').mkdir(parents=True, exist_ok=True)
    allc = []
    for spec in specs:
        parts = spec.split(':')
        tag, kind, case, conf, how = parts[:5]
        nk, nf = (int(parts[5]), int(parts[6])) if len(parts) > 6 else (nk0, nf0)
        c, cell, why = pick_corner(kind, case, conf, how)
        pk = packet_root(cell)
        ctx = json.loads((pk / 'FRESH_CONTEXT.json').read_text())
        t0 = Fraction(ctx['case']['tau_corners'][c])
        pts = []
        for g, i in order(nk, nf):
            n, rel = (nk, 0.10) if g == 'k' else (nf, 0.005)
            t = -1 + 2 * i / (n - 1)
            tau = t0 if abs(t) < 1e-15 else (t0 * (1 + Fraction(rel) * Fraction(t).limit_denominator(10 ** 6))).limit_denominator(10 ** 9)
            pc = f'{cell}_x3{tag}{g}{i:03d}'
            d = root / 'packets' / pc
            if not (d / 'FRESH_CONTEXT.json').exists():
                d.mkdir(parents=True, exist_ok=True)
                cx = json.loads((pk / 'FRESH_CONTEXT.json').read_text())
                cx['case']['tau_corners'][c] = str(tau)
                cx.setdefault('provenance', {})['r1_sweep_of'] = cell
                (d / 'FRESH_CONTEXT.json').write_text(json.dumps(cx, indent=1))
                shutil.copy(pk / 'SAMPLE.json', d / 'SAMPLE.json')
            pts.append(dict(grid=g, i=i, t=t, rel=rel, tau=float(tau), case=pc))
            allc.append(pc)
        (root / tag).mkdir(exist_ok=True)
        (root / tag / 'plan.json').write_text(json.dumps(dict(tag=tag, kind=kind, case=case, cell=cell, conf=conf, corner=c, why=why,
                                                             tau0=float(t0), nk=nk, nf=nf, points=pts), indent=1))
        print(json.dumps(dict(event='PLAN', tag=tag, kind=kind, cell=cell, conf=conf, corner=c, why=why, tau0=float(t0), n=len(pts))),
              file=sys.stderr, flush=True)
    print('\n'.join(allc))


# ======================================================================================== GPU part
def _h(a):
    return hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()[:12]


def run(root, tag, deadline, lat_tol):
    import torch
    import models as MD                                                          # noqa: F401  first: OPL_CONV_FP32
    import teacher as TE, trainlib as TL, fastnet as FN, evalnet as EN, bench_deploy as BD, ops as OP
    import lattice3 as LT, lat_multi as LM, lat_precond as PR, lat_hetero as LH, diag_cert as DC
    import r1x3_common as RC
    dev, dt = TE.dev, TE.dt
    root = Path(root); P = json.loads((root / tag / 'plan.json').read_text())
    body, tmp = str(root / 'body'), root / 'tmp'
    ck = DC.load_ckpt('/root/autodl-tmp/OPL/S1/V2/A3_2grid/best.pt')
    kind, c = P['kind'], P['corner']
    os.environ['LAT_LOADS'] = 'consistent'
    log = lambda d: print(json.dumps(RC.tojson(d), default=float), flush=True)
    out_j, out_z = root / tag / 'sweep.json', root / tag / 'sweep.npz'
    rec = dict(plan={k: v for k, v in P.items() if k != 'points'}, env=RC.env_flags(), points=[], t0=time.strftime('%F %T'))
    vec = {}

    def learned(C):
        BD.netdata(C, body, tmp / C.case)
        geo = TL.Geo(C.case, body, tmp, neumann=False, log=lambda s_: None, cell=C, load_banks=False)
        model = DC.net_for(ck, geo)[0]
        op = EN.FastOp(FN.FastNet(model, geo))
        with torch.no_grad():
            op.apply(torch.zeros((C.np_, 1), dtype=dt, device=dev))
        cc = model.caches[C.case]
        fp = dict(active=len(C.cells), active_h=_h(C.cells), faces=int(len(np.load(Path(body) / C.case / 'GP_FACES.npy'))),
                  faces_h=_h(np.load(Path(body) / C.case / 'GP_FACES.npy')), ports=int(C.np_), ports_h=_h(C.port_node_ids),
                  cut_nodes=int(C.is_cut.sum()), weak=int(geo.nd['weak'].sum()),
                  el_fringe=int(cc.el_fringe.sum()) if getattr(cc, 'el_fringe', None) is not None else None,
                  gp_fringe=int(cc.gp_fringe.sum()) if getattr(cc, 'gp_fringe', None) is not None else None,
                  shift=float(getattr(C, '_c_shift', 0.0) or 0.0), b=float(C._tail_bounds[2]))
        return op, model, geo, fp, C.nodes[np.asarray(geo.nd['weak'], bool)]

    def dmc(C):
        h = 1e-5 * C.taus[c]
        tp = list(C.taus); tp[c] += h; tm = list(C.taus); tm[c] -= h
        C.dM = ((C.moments(tp) - C.moments(tm)) / (2 * h))[None]

    def body_ready(pc, wait=900):
        f = Path(body) / pc / 'PREP.json'
        t = time.time()
        while not f.exists() and time.time() - t < wait:
            if (Path(body) / (pc + '.failed')).exists():
                return False
            time.sleep(5)
        return f.exists()

    # ------------------------------------------------------------------ fixed parts
    if kind == 'pair':
        conf = P['conf']
        Cn = TE.Cell(f"{P['case']}_nbm{conf}", S0, log=lambda s_: None); Cn.assemble()
        Tn = LT.dense_T(Cn, S0, host=True).to(dev)
        Cb = TE.Cell(P['case'], S0, log=lambda s_: None); Cb.assemble()
        Tb = LT.dense_T(Cb, S0, host=True)
        off = (-1, 0, 0) if conf == 'x' else (0, -1, 0)
        latb = LT.Lattice([dict(label='test', cell=Cb, offset=(0, 0, 0), T=Tb), dict(label='nbr', cell=Cn, offset=off, T=Tn)], conf, log=lambda s_: None)
        ldev = torch.device('cpu') if os.environ.get('LAT_CPU') else dev
        nf = len(latb.free)
        K = torch.zeros((nf, nf), dtype=dt, device=ldev)
        for i, cd in enumerate(latb.cells):
            f = latb.gather_idx[i]; keep = torch.nonzero(f >= 0).squeeze(1); fc = f[keep]
            T = cd['T']
            for r0 in range(0, len(keep), 2048):
                rows = keep[r0:r0 + 2048]
                K.index_put_((f[rows][:, None].to(ldev), fc[None, :].to(ldev)), T[rows.to(T.device)].to(ldev, dt)[:, keep.to(ldev)], accumulate=True)
        info = torch.empty((), dtype=torch.int32, device=ldev)
        torch.linalg.cholesky_ex(K, out=(K, info))
        L0 = K.tril_(); del K, Tb, Cb; RC.free()

        def keys(lat):
            out = {}
            for i, cd in enumerate(lat.cells):
                C_ = cd['cell']
                g = np.stack(np.unravel_index(C_.port_node_ids, (2 * C_.n + 1,) * 3), 1) + 2 * C_.n * np.asarray(cd['offset'])
                priv = np.repeat(C_.port_is_cut & ~C_.port_is_box, 3)
                gg = np.repeat(g, 3, 0); comp = np.tile(np.arange(3), len(g))
                f = lat.gather_idx[i].cpu().numpy()
                for r in np.flatnonzero(f >= 0):
                    k = (cd['label'], int(C_.port_node_ids[r // 3]), int(comp[r])) if priv[r] else (int(gg[r, 0]), int(gg[r, 1]), int(gg[r, 2]), int(comp[r]))
                    out[int(f[r])] = k
            return out
        kb = {v: k for k, v in keys(latb).items()}

        def make_prec(lat, C):
            kt = keys(lat)
            nf_ = len(lat.free)
            src = np.full(nf_, -1, np.int64)
            for j, k in kt.items():
                src[j] = kb.get(k, -1)
            m = torch.as_tensor(src >= 0, device=dev); si = torch.as_tensor(src[src >= 0], device=dev)
            dg = torch.ones(nf_, dtype=dt, device=dev)
            f0 = lat.gather_idx[0]; kp = torch.nonzero(f0 >= 0).squeeze(1)
            dg[f0[kp]] = C.dK[C.P][kp].to(dt)
            dinv = 1 / dg

            def pc(R):
                Z = dinv[:, None] * R
                Rb = torch.zeros((nf, R.shape[1]), dtype=dt, device=ldev)
                Rb[si.to(ldev)] = R[m].to(ldev)
                Zb = torch.cholesky_solve(Rb, L0).to(dev)
                Z[m] = Zb[si]
                return Z
            return pc, int((~m).sum())
        nop = OP.ExactOp(Cn, Tn)
    else:
        L = json.loads(Path(P['conf']).read_text())
        fixed = {}
        for cl in L['cells']:
            if cl['case'] == P['cell']:
                continue
            C_ = TE.Cell(cl['case'], S4 + '/body', log=lambda s_: None); C_.assemble()
            BD.netdata(C_, S4 + '/body', tmp / C_.case)
            geo = TL.Geo(C_.case, S4 + '/body', tmp, neumann=False, log=lambda s_: None, cell=C_, load_banks=False)
            op_ = EN.FastOp(FN.FastNet(DC.net_for(ck, geo)[0], geo))
            with torch.no_grad():
                op_.apply(torch.zeros((C_.np_, 1), dtype=dt, device=dev))
            de = LH.DenseExactOp(C_, S4 + '/body')
            Td = de.T.to(dev)
            de.apply = (lambda q, Td=Td: Td @ q.to(dt))                               # dense exact S on the device
            fixed[cl['case']] = (C_, op_, de, LM.from_teacher(C_))
        pos = {cl['case']: tuple(cl['position']) for cl in L['cells']}
    t_start = time.time()
    prevX = None
    for pt in P['points']:
        if deadline and time.time() > deadline:
            rec['stopped'] = 'deadline'; break
        pc = pt['case']
        if not body_ready(pc):
            rec['points'].append(dict(pt, error='NO_BODY')); continue
        t = time.perf_counter()
        row = dict(pt)
        try:
            C = TE.Cell(pc, body, log=lambda s_: None); C.assemble()
            op, model, geo, fp, weak_ids = learned(C)
            row['fp'] = fp; vec[f'weak_{pt["grid"]}{pt["i"]:03d}'] = weak_ids
            dmc(C)
            ref_pt = (pt['grid'] == 'k' and (pt['i'] - P['nk'] // 2) % A.ref_every == 0) or \
                     (pt['grid'] == 'f' and (pt['i'] - P['nf'] // 2) % A.ref_fine_every == 0)
            with torch.no_grad():
                if kind == 'pair':
                    lat = LT.Lattice([dict(label='test', cell=C, offset=(0, 0, 0), T=None), dict(label='nbr', cell=Cn, offset=off, T=Tn)],
                                     conf, log=lambda s_: None)
                    prec, nnew = make_prec(lat, C)
                    F = lat.F
                    mv = lambda Pm, o=op: lat.matvec([o, nop], Pm)
                    X, it, rr = RC.pcg_fixed(mv, prec, F, tol=1e-10, maxit=300)
                    rho = F - mv(X)
                    q = lat.gather(X, 0); u = op.field(q).to(dt)
                    row.update(labels=lat.labels, gate=lat.gate.tolist(), unmapped_dofs=nnew, pcg=it, rec_res=rr)
                    idx_test = 0
                else:
                    lay = {pos[k]: v[3] for k, v in fixed.items()}
                    lay[pos[P['cell']]] = LM.from_teacher(C)
                    lat = LM.MultiLattice(lay, clamp=('y', 'min'), load=('y', 'max'), loads='consistent', n_random=0, device=dev,
                                          max_cols=16, log=lambda s_: None)
                    ops_ = [fixed[g.case][1] if g.case in fixed else op for g in lat.geoms]
                    fac = PR.Factory(lat, ops_, kpp_backend='auto', log=lambda d: None)
                    pcn, _, _ = fac.build('bnn:kpp:q1r')
                    r = PR.pcg(lat, ops_, pcn, tol=lat_tol, maxit=3000)
                    fac.free(); del pcn
                    X = r['X']; F = lat.F
                    rho = F - lat.matvec(ops_, X)
                    idx_test = [g.case for g in lat.geoms].index(pc)
                    q = lat.gather(X, idx_test); u = op.field(q).to(dt)
                    row.update(labels=lat.labels, pcg=r['iterations'], rec_res=r['residual'])
                Ch = (F * X).sum(0)
                row.update(C_hat=Ch, J=Ch + (X * rho).sum(0), true_res=rho.norm(dim=0) / F.norm(dim=0),
                           s_tilde=C.sens(u)[0], e_test_act=(q * op.apply(q)).sum(0))
                if ref_pt:
                    ex = BD.SparseExactOp(C)
                    if kind == 'pair':
                        mve = lambda Pm: lat.matvec([ex, nop], Pm)
                        Xe, ite, rre = RC.pcg_fixed(mve, prec, F, tol=1e-10, maxit=300)
                    else:
                        eops = [fixed[g.case][2] if g.case in fixed else ex for g in lat.geoms]
                        fac = PR.Factory(lat, eops, kpp_backend='auto', log=lambda d: None)
                        pce, _, _ = fac.build('bnn:kpp:q1r')
                        re_ = PR.pcg(lat, eops, pce, tol=1e-10, maxit=3000); fac.free(); del pce
                        Xe, ite = re_['X'], re_['iterations']
                    Ce = (F * Xe).sum(0)
                    qe = lat.gather(Xe, idx_test)
                    ue = ex.field(qe)
                    Eex = (qe * ex.apply(qe)).sum(0); Ehat = (qe * op.apply(qe)).sum(0)
                    row.update(ref=dict(C=Ce, s=C.sens(ue)[0], pcg=ite, eps=Ehat / Eex - 1, w=Eex / Ce, beta=(Ehat - Eex) / Ce,
                                        err=(Ce - Ch) / Ce))
                    C._free(); del ex, ue
            row['seconds'] = time.perf_counter() - t
            log(dict(event='POINT', tag=tag, grid=pt['grid'], i=pt['i'], tau=pt['tau'], C_hat=row['C_hat'][:3], s_tilde=row['s_tilde'][:3],
                     pcg=row['pcg'], fp=fp, ref=row.get('ref', {}).get('err'), seconds=row['seconds']))
            del op, model, geo, C, lat, X, rho, q, u
        except Exception as e:                                                            # noqa: BLE001
            import traceback
            row['error'] = traceback.format_exc()[-1500:]
            log(dict(event='POINT_ERROR', tag=tag, case=pc, error=repr(e)[:300]))
        rec['points'].append(row)
        RC.free()
        if len(rec['points']) % 5 == 0:
            RC.dump(out_j, rec); np.savez(out_z, **vec)
    rec['t1'] = time.strftime('%F %T'); rec['wall_s'] = time.time() - t_start
    RC.dump(out_j, rec); np.savez(out_z, **vec)
    log(dict(event='DONE', tag=tag, points=len(rec['points']), wall_s=rec['wall_s']))


def summary(root):
    out = {}
    for f in sorted(glob.glob(str(Path(root) / '*' / 'sweep.json'))):
        r = json.loads(Path(f).read_text()); tag = r['plan']['tag']
        pts = [p for p in r['points'] if 'C_hat' in p]
        if len(pts) < 3:
            out[tag] = dict(n=len(pts)); continue
        pts.sort(key=lambda p: p['tau'])
        tau = np.asarray([p['tau'] for p in pts]); Ch = np.asarray([p['C_hat'] for p in pts]); st = np.asarray([p['s_tilde'] for p in pts])
        g = np.asarray(pts[0].get('gate', [True] * Ch.shape[1]), bool)
        keysw = ('active_h', 'faces_h', 'ports_h', 'weak', 'el_fringe', 'gp_fringe', 'shift')
        jumps = []
        for a, b in zip(pts[:-1], pts[1:]):
            dtau = b['tau'] - a['tau']
            ca, cb = np.asarray(a['C_hat']), np.asarray(b['C_hat'])
            pred = 0.5 * (np.asarray(a['s_tilde']) + np.asarray(b['s_tilde'])) * dtau
            sw = [k for k in keysw if a['fp'].get(k) != b['fp'].get(k)]
            jumps.append(dict(tau=[a['tau'], b['tau']], dC_rel=((cb - ca) / ca)[g], nonsmooth_rel=np.abs(cb - ca - pred)[g] / np.abs(ca[g]),
                              switch=sw, geometric=[k for k in sw if k in ('active_h', 'faces_h', 'ports_h')]))
        def mx(sel, key):
            v = [np.max(j[key]) for j in jumps if sel(j)]
            return float(max(v)) if v else None
        refs = [p for p in pts if 'ref' in p]
        rj = []
        for a, b in zip(refs[:-1], refs[1:]):
            dtau = b['tau'] - a['tau']
            Ca, Cb = np.asarray(a['ref']['C']), np.asarray(b['ref']['C'])
            pred = 0.5 * (np.asarray(a['ref']['s']) + np.asarray(b['ref']['s'])) * dtau
            ha, hb = np.asarray(a['C_hat']), np.asarray(b['C_hat'])
            predh = 0.5 * (np.asarray(a['s_tilde']) + np.asarray(b['s_tilde'])) * dtau
            geo_ = [k for k in ('active_h', 'faces_h', 'ports_h') if a['fp'].get(k) != b['fp'].get(k)]
            rj.append(dict(tau=[a['tau'], b['tau']], ref_nonsmooth_rel=float((np.abs(Cb - Ca - pred) / np.abs(Ca))[g].max()),
                           hat_nonsmooth_rel=float((np.abs(hb - ha - predh) / np.abs(ha))[g].max()), geometric=geo_))
        out[tag] = dict(plan=r['plan'], n=len(pts), stopped=r.get('stopped'), errors=sum(1 for p in r['points'] if 'error' in p),
                        tau_range=[float(tau.min()), float(tau.max())],
                        max_jump_rel=mx(lambda j: True, 'dC_rel'), max_nonsmooth_rel=mx(lambda j: True, 'nonsmooth_rel'),
                        max_nonsmooth_rel_switch=mx(lambda j: j['switch'], 'nonsmooth_rel'),
                        max_nonsmooth_rel_geometric=mx(lambda j: j['geometric'], 'nonsmooth_rel'),
                        max_nonsmooth_rel_noswitch=mx(lambda j: not j['switch'], 'nonsmooth_rel'),
                        n_intervals=len(jumps), n_switch=sum(1 for j in jumps if j['switch']),
                        n_geometric=sum(1 for j in jumps if j['geometric']),
                        switch_kinds={k: sum(1 for j in jumps if k in j['switch']) for k in keysw},
                        ref=dict(n=len(refs), max_err=float(max(np.max(np.abs(np.asarray(p['ref']['err'])[g])) for p in refs)) if refs else None,
                                 max_beta=float(max(np.max(np.asarray(p['ref']['beta'])[g]) for p in refs)) if refs else None,
                                 jumps=rj),
                        top_jumps=sorted([dict(tau=j['tau'], nonsmooth=float(np.max(j['nonsmooth_rel'])), switch=j['switch']) for j in jumps],
                                         key=lambda d: -d['nonsmooth'])[:10])
    Path(root, 'E13_SUMMARY.json').write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: {kk: v.get(kk) for kk in ('n', 'max_jump_rel', 'max_nonsmooth_rel_switch', 'max_nonsmooth_rel_noswitch')}
                      for k, v in out.items()}, default=float))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd'); ap.add_argument('root'); ap.add_argument('rest', nargs='*')
    ap.add_argument('--coarse', type=int, default=201); ap.add_argument('--fine', type=int, default=101)
    ap.add_argument('--deadline', type=float, default=0); ap.add_argument('--lat-tol', type=float, default=1e-9)
    ap.add_argument('--ref-every', type=int, default=10); ap.add_argument('--ref-fine-every', type=int, default=25)
    A = ap.parse_args()
    if A.cmd == 'plan':
        plan(A.root, A.rest, A.coarse, A.fine)
    elif A.cmd == 'run':
        run(A.root, A.rest[0], A.deadline, A.lat_tol)
    elif A.cmd == 'summary':
        summary(A.root)
