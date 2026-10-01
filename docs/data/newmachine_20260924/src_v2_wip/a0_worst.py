"""Direction A, step 0d: converged worst directions mu = lambda_max(S~^+ S_hat) on mapped cells (new script, default paths
unchanged; runs on the GPU server).

Why: a0_eval.worst() / trainlib.Geo.adversarial() run an UNSHIFTED block power iteration on S~^+ S_hat, keeping only the
last block. Its eigenvalues are 1 + eps_i with eps_i small and clustered, so the per-step factor ~ (1+eps_9)/(1+eps_1) is
close to 1 and the stopping rule fires early: in full16, 268 of 768 records have mu - 1 below the largest of the 64
sampled force_c errors and 190 below their mean. Ritz values remain valid LOWER bounds (Courant-Fischer).

Here: block Krylov with Rayleigh-Ritz over the whole accumulated subspace (S~-orthonormal basis), stopped on the residual
bound |mu - lambda| <= ||S_hat x - mu S~ x||_{S~^-1} relative to mu - 1. Per block step: one S_hat application (through
the network, as Geo.s_hat_apply) and one Neumann solve, as before. S~ X is known without solves (S~ S~^+ f = Pi f).
Optionally the same on the subspace of loads restricted to box ports ("box-load worst": cut-band ports carry no external
load in a lattice whose cut surfaces are free), and, for --detail cases, where the worst directions live (port nodes:
cut band / box / weak; error energy near weakly supported nodes) and how much of them the force_c bank spans.

Data: reuses the NETDATA / banks of an a0_eval run directory (<data>/<map>/<case>), so the evaluated directions are the
ones of full16. Variant V0R; --corot 1 adds the co-rotated Jacobi preconditioner (corot_smooth.py) to the correction.
Usage: a0_worst.py <out_dir> <ckpt> <body_dir> <data_dir> <cases (comma)> <maps.json> --maps id,twist30 [--corot 1]
       [--k 8] [--maxit 40] [--box 1] [--detail case1,case2] [--means 1]"""
import argparse, gc, json, sys, time
from pathlib import Path
import numpy as np
import torch

import models as MD                                                       # first: applies OPL_CONV_FP32
import trainlib as TL
import a0_eval as AE
import mapped_cell as MC
import corot_smooth as CR
import a_ucond as AU
import weak_patch as WP

dev, dt = AE.dev, AE.dt


def _eorth(V, X, passes=2, drop=1e-10):
    """Euclidean orthonormalisation of the block X against V (orthonormal) and within itself (exact, no drift); columns
    whose norm falls below drop x their norm before projection are discarded."""
    n0 = X.norm(dim=0).clamp_min(1e-300)
    for _ in range(passes):
        if V is not None:
            X = X - V @ (V.T @ X)
    Qx, Rx = torch.linalg.qr(X)
    keep = Rx.diagonal().abs() > drop * n0.max()
    X = Qx[:, keep]
    if V is not None and X.shape[1]:
        X = X - V @ (V.T @ X); X = torch.linalg.qr(X)[0]
    return X


def krylov(g, model, C, k=8, maxit=40, box=None, seed=0, log=None):
    """Phase A (Neumann factor): block Krylov subspace span{X0, T X0, T^2 X0, ...} of T = S~^+ R S_hat (box: mask (np,)
    restricting the LOADS to box ports, self-equilibrated there; x = S~^+ f), Euclidean-orthonormal basis V (exact; an
    S~-orthonormal recurrence drifts because T = I + O(eps) amplifies solve rounding by ~1/eps per step). maxit block steps
    (one S_hat application and one Neumann solve each). Ritz values come from exact_rr() with the exact S~ V.
    Returns dict(V, HV, it)."""
    if box is not None:
        Qb = C.Q * box[:, None]; Pb = torch.linalg.pinv(Qb)
        R = lambda f: f * box[:, None] - Qb @ (Pb @ (f * box[:, None]))
    else:
        R = lambda f: f
    gen = torch.Generator(device=dev).manual_seed(seed)
    X = C.neumann(R(torch.randn((g.np_, k), dtype=dt, device=dev, generator=gen)))
    V = HV = None
    it = 0
    for it in range(1, maxit + 1):
        X = _eorth(V, X)
        if X.shape[1] == 0:
            break
        HX = g.s_hat_apply(model, X)
        V = X if V is None else torch.cat([V, X], 1); HV = HX if HV is None else torch.cat([HV, HX], 1)
        if log and (it % 10 == 0 or it == 1):
            log(f'  it {it} basis {V.shape[1]}')
        X = C.neumann(R(HX))
    return dict(V=V, HV=HV, it=it)


def exact_rr(V, HV, SVe, k, drop=1e-12):
    """Phase B: Rayleigh-Ritz of (S_hat, S~) on span V with the exact S~V (interior factor). Ritz values are lower bounds
    of the top eigenvalues (Courant-Fischer) whatever the quality of V. Returns mu (k), X, SX (S~-normalised), min_ritz."""
    G = V.T @ SVe; G = 0.5 * (G + G.T)
    ev, U = torch.linalg.eigh(G)
    keep = ev > drop * ev.max()
    W = U[:, keep] / torch.sqrt(ev[keep])[None, :]
    H = W.T @ (V.T @ HV) @ W; H = 0.5 * (H + H.T)
    lam, Z = torch.linalg.eigh(H)
    o = torch.argsort(lam, descending=True); lam, Z = lam[o], Z[:, o]
    C_ = W @ Z[:, :k]
    return dict(mu=[float(x) for x in lam[:k]], X=V @ C_, SX=SVe @ C_, min_ritz=float(lam[-1]), rank=int(keep.sum()))


def exact_apply(C, V, chunk=64):
    return torch.cat([C.apply(V[:, j:j + chunk]) for j in range(0, V.shape[1], chunk)], 1)


def detail(g, model, C, X, SX, d, bank):
    """Where the worst directions live. X (np, m) S~-normalised top Ritz vectors; d: NETDATA dict; bank: (np, B) force_c."""
    nodes = C.nodes; onport = np.isin(nodes, C.port_node_ids)
    is_cut = torch.as_tensor(d['is_cut'][onport], device=dev); is_box = torch.as_tensor(d['is_box'][onport], device=dev)
    weak_all = torch.as_tensor(d['weak'], device=dev); weak = weak_all[torch.as_tensor(onport, device=dev)]
    out = []
    U = torch.cat([g.field(model, X[:, j:j + 4]).to(dt) for j in range(0, X.shape[1], 4)], 1)        # corrected extension
    E = C.extend(X)                                                                                  # exact extension
    Dd = U - E
    KD = C.K @ Dd
    en = (Dd * KD).reshape(-1, 3, X.shape[1]).sum(1)                                                 # nodal error energy
    # nodes within 2 grid spacings of a weak node (grid coordinates)
    grid = torch.as_tensor(d['grid'].astype(np.int64), device=dev); M = 2 * C.n + 1
    vol = torch.zeros((1, 1, M, M, M), dtype=torch.float32, device=dev)
    vol[0, 0, grid[weak_all, 0], grid[weak_all, 1], grid[weak_all, 2]] = 1
    vol = torch.nn.functional.max_pool3d(vol, 5, stride=1, padding=2)                                 # Chebyshev dist <= 2
    near = vol[0, 0, grid[:, 0], grid[:, 1], grid[:, 2]] > 0
    Sb = C.apply(bank.to(dt))                                                                         # S~ q_b
    Gb = bank.to(dt).T @ Sb; Gb = 0.5 * (Gb + Gb.T)
    evb, Ub = torch.linalg.eigh(Gb); keep = evb > 1e-10 * evb.max()
    for j in range(X.shape[1]):
        x = X[:, j]; x3 = x.reshape(-1, 3).norm(dim=1) ** 2; tot = float(x3.sum())
        f = SX[:, j].reshape(-1, 3).norm(dim=1) ** 2; ftot = float(f.sum())                          # load producing x
        top = torch.sort(x3, descending=True).values
        b = Sb.T @ x                                                                                  # <q_b, x>_S
        cb = (Ub[:, keep].T @ b) / torch.sqrt(evb[keep])
        e = en[:, j]; etot = float(e.sum())
        out.append(dict(
            ritz_minus_1=float((x * g.s_hat_apply(model, x[:, None])[:, 0]).sum() - 1),
            err_energy=etot,
            disp_frac_cut=float(x3[is_cut].sum() / tot), disp_frac_box=float(x3[is_box].sum() / tot),
            disp_frac_weak=float(x3[weak].sum() / tot),
            disp_top10_nodes=float(top[:10].sum() / tot), disp_participation_nodes=float(tot ** 2 / (x3 ** 2).sum()),
            load_frac_cut=float(f[is_cut].sum() / ftot), load_frac_box=float(f[is_box].sum() / ftot),
            err_frac_weak_nodes=float(e[weak_all].sum() / etot), err_frac_near_weak=float(e[near].sum() / etot),
            err_top10_nodes=float(torch.sort(e, descending=True).values[:10].sum() / etot),
            frac_nodes_near_weak=float(near.float().mean()),
            bank_span_cos2=float((cb ** 2).sum()), bank_best_single_cos=float(b.abs().max())))
    return out


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('ckpt'); ap.add_argument('body'); ap.add_argument('data'); ap.add_argument('cases')
    ap.add_argument('mapsjson'); ap.add_argument('--maps', default='id')
    ap.add_argument('--weakpatch', type=int, default=0); ap.add_argument('--wp_dil', type=int, default=2); ap.add_argument('--wp_seed', default='cutport')
    ap.add_argument('--uckpt', default='')                                   # trainA checkpoint (stretch inputs)
    ap.add_argument('--corot', type=int, default=1); ap.add_argument('--k', type=int, default=8)
    ap.add_argument('--maxit', type=int, default=40); ap.add_argument('--tol', type=float, default=1e-2)
    ap.add_argument('--box', type=int, default=1); ap.add_argument('--detail', default='')
    ap.add_argument('--means', type=int, default=1)
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    specs = {m['name']: m['spec'] for m in json.loads(Path(a.mapsjson).read_text())}
    res_path = out / 'WORST.jsonl'
    done = set()
    if res_path.exists():
        done = {(r['case'], r['map']) for r in map(json.loads, res_path.read_text().splitlines()) if 'error' not in r}
    detail_cases = set(c for c in a.detail.split(',') if c)
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False); cfg = ck['cfg']
    model = None
    if a.corot:
        CR.install()
    log = lambda s: print(s, flush=True)
    for mname in a.maps.split(','):                                              # map-major: identity first
        for case in [c for c in a.cases.split(',') if c]:
            if (case, mname) in done:
                continue
            rec = dict(case=case, map=mname, weakpatch=a.weakpatch, wp_dil=a.wp_dil, uckpt=a.uckpt, corot=a.corot, k=a.k, tol=a.tol)
            t0 = time.perf_counter(); C = g = None
            try:
                torch.cuda.reset_peak_memory_stats()
                C = MC.MappedCell(case, a.body, specs[mname], log=lambda s_: None); C.assemble()
                g = AE.MappedGeo(case, a.body, Path(a.data) / mname, neumann=False, log=lambda s_: None, cell=C)
                g.case = f'{case}@{mname}'
                if model is None:
                    model = MD.build(cfg['model'], [g], **dict(cfg.get('model_args', {}))).to(dev)
                    MD.load_compat(model, ck['model']); model.eval()
                    if a.uckpt:
                        AU.attach(model); model.load_state_dict(torch.load(a.uckpt, map_location=dev, weights_only=False)['model'])
                        model.caches.pop(g.case, None); AU.prepare(g, C); model.add_geo(g); AU.set_ufeat(model, g, C)
                else:
                    if a.uckpt:
                        AU.prepare(g, C)
                    model.add_geo(g)
                    if a.uckpt:
                        AU.set_ufeat(model, g, C)
                wrap = AE._Wrap(model)
                g.set_variant('V0R', wrap)
                if a.corot:
                    CR.set_corot(C, AE.nodal_rotations(C))
                if a.weakpatch:
                    rec['weakpatch'] = WP.setup(C, g.nd, dil=a.wp_dil, seed=a.wp_seed); WP.wrap_geo(g)
                if a.means:
                    with torch.no_grad():
                        rec['means'] = {c: v for c, v in AE.evaluate(g, model, 'V0R').items() if c in ('force_c', 'face_c', 'grf')}
                C.factor(neumann=True, interior=False)
                t = time.perf_counter(); log(f'{case} {mname}: full')
                rf = krylov(g, model, C, k=a.k, maxit=a.maxit, log=log)
                rec['full'] = dict(it=rf['it'], basis=int(rf['V'].shape[1]), seconds_A=time.perf_counter() - t)
                if a.box:
                    t = time.perf_counter(); log(f'{case} {mname}: box loads')
                    box = torch.as_tensor(np.repeat(C.port_is_box, 3), device=dev).to(dt)
                    rb = krylov(g, model, C, k=a.k, maxit=a.maxit, box=box, log=log)
                    rec['box'] = dict(it=rb['it'], basis=int(rb['V'].shape[1]), seconds_A=time.perf_counter() - t)
                C._free()
                C.factor(neumann=False)                                         # phase B: exact S~ V, exact Ritz values
                t = time.perf_counter()
                SVe = exact_apply(C, rf['V'])
                ef = exact_rr(rf['V'], rf['HV'], SVe, a.k)
                half = rf['V'].shape[1] // 2 // a.k * a.k                       # convergence check: first half of the basis
                eh = exact_rr(rf['V'][:, :half], rf['HV'][:, :half], SVe[:, :half], 1) if half >= a.k else None
                rec['full'].update(mu=ef['mu'], min_ritz=ef['min_ritz'], rank=ef['rank'],
                                   mu_half_basis=eh['mu'][0] if eh else None, seconds_B=time.perf_counter() - t)
                del SVe
                if a.box:
                    eb = exact_rr(rb['V'], rb['HV'], exact_apply(C, rb['V']), a.k)
                    rec['box'].update(mu=eb['mu'], min_ritz=eb['min_ritz'], rank=eb['rank'])
                    del rb
                else:
                    eb = None
                if case in detail_cases:
                    d = np.load(Path(a.data) / mname / case / 'NETDATA.npz')
                    with torch.no_grad():
                        rec['detail'] = detail(g, model, C, ef['X'][:, :4], ef['SX'][:, :4], d, g.banks['val']['force_c'])
                        if a.box:
                            rec['detail_box'] = detail(g, model, C, eb['X'][:, :4], eb['SX'][:, :4], d, g.banks['val']['force_c'])
                            np.save(out / f'XB_{case}_{mname}.npy', eb['X'][:, :4].cpu().numpy())
                    np.save(out / f'X_{case}_{mname}.npy', ef['X'][:, :4].cpu().numpy())
                del rf, ef, eb
                C._free()
                model.caches.pop(g.case, None)
                rec['gpu_peak_gb'] = torch.cuda.max_memory_allocated() / 2 ** 30
            except Exception as e:
                import traceback
                rec['error'] = repr(e)[:400]; rec['trace'] = traceback.format_exc()[-2000:]
                try:
                    C._free(); model.caches.pop(g.case, None)
                except Exception:
                    pass
            rec['seconds'] = time.perf_counter() - t0
            with open(res_path, 'a') as f:
                f.write(json.dumps(rec) + '\n')
            log(json.dumps(dict(case=case, map=mname, s=round(rec['seconds'], 1), err=rec.get('error'),
                                mu=rec.get('full', {}).get('mu', [None])[:2], it=rec.get('full', {}).get('it'),
                                mu_half=rec.get('full', {}).get('mu_half_basis'),
                                box=rec.get('box', {}).get('mu', [None])[:1],
                                fc=rec.get('means', {}).get('force_c', {}).get('mean'))))
            del C, g
            gc.collect(); torch.cuda.empty_cache()
    print('DONE', flush=True)


if __name__ == '__main__':
    main(sys.argv[1:])
