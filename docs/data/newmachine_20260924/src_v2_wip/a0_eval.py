"""Direction A, step 0: zero-shot evaluation of the P1 network (A3) on isoparametrically mapped cells (default off; new
script, existing files unchanged).

Per (cell, map):
  1. MappedCell (mapped_cell.py): K~ with the mapped stiffness and physical-coordinate ghost penalty, physical rigid bases;
     interior and Neumann factors (fp64).
  2. Direction banks at unit exact energy of K~ (rigid part removed with the physical rigid basis), loads and displacement
     fields evaluated at the PHYSICAL port positions:
       force   nodal plane-wave / patch forces on box ports (10% also on the cut band), q = S~^+ f       (prep_data)
       macro   polynomial displacement fields of degree 1-3                                              (prep_data)
       grf     multiscale random displacement fields                                                      (prep_data)
       force_c consistent self-equilibrated tractions on the box faces (10% also on the cut), surface measure of the
               mapped faces (Nanson: det(J) |J^-T n|), traction fields sampled at the physical quadrature points (prep_geo2)
       face_c  the same on one box face at a time                                                         (prep_geo2)
     Splits: train 8 / val VAL / test 8 (only val is evaluated).
  3. NETDATA from K~ (diag3, weak flags from the mapped stiffness; moments, grid and masks of the reference cell) and
     PORTS.json, written to <out>/data/<map>/<case>.
  4. Variants on the same banks (A3 weights unchanged; correction = A3's tail(8) - Q1_17 coarse - tail(8) on K~):
       V0     network on the reference grid, rigid part split off with the PHYSICAL rigid modes (exact)
       V0R    V0 plus a co-rotational pull-back: the non-rigid port values are rotated into the local reference frame
              (R = polar factor of dx/dX at each node), the network extends them, the field is rotated back
       V0ref  network with the reference-coordinate rigid split of P1 (diagnostic: what an unaware deployment does)
       Conly  no network: rigid part plus zero, then the same correction (what the correction alone achieves)
       V0Rc   (not in the default list) V0R with the co-rotated Jacobi preconditioner in the correction (corot_smooth.py):
              exactly equivariant under rigid rotations, identical to V0R on the identity map
     Every variant is a linear extension read out in energy form with K~, so S_hat >= S~ (checked: min energy excess).
  5. Metrics: per class mean / p90 / max / min of e = u^T K~ u - 1; worst direction mu = top Ritz value of (S_hat, S~)
     (the Neumann and interior factors are held one at a time, which bounds the GPU memory of full cells)
     (block power iteration, worst()) for V0 and V0R; energy of the extended physical rigid modes (rigid_energy); map
     statistics (min det J, max cond J); rigid checks; timings. (V0 and V0ref coincide on rigid-free banks when the network
     reproduces reference rigid motions; they differ on the physical rigid modes, which rigid_energy measures.)
Usage: a0_eval.py <out_dir> <ckpt> <body_dir> <cases (comma)> <maps.json> [--val 64] [--worst 1] [--variants V0,V0R,V0ref,Conly]
maps.json: list of {"name": ..., "spec": {...}} (mapped_cell.make_map)."""
import argparse, gc, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
import torch

import models as MD                                                       # first: applies OPL_CONV_FP32
import trainlib as TL
import teacher as TE
import prep_data as PD
import prep_geo2 as PG2
import mapped_cell as MC
import corot_smooth as CR                                                 # V0Rc only (default off)

dev, dt = TE.dev, TE.dt


# ------------------------------------------------------------------------------------------------------ mapped tractions
def _face_factor(C, P, normal_ref):
    """Physical surface measure per reference area and physical positions at reference surface points P (m, 3):
    det(J_X) |J_X^-T n| (Nanson) with J_X = dx_h/dX of the element containing each point."""
    n = C.n
    c = np.minimum(np.floor(P * n).astype(np.int64), n - 1)
    t = torch.as_tensor(2 * (P * n - c) - 1, dtype=dt, device=dev)
    o = np.array(list(np.ndindex(3, 3, 3)))
    M = 2 * n + 1
    g = 2 * c[:, None, :] + o[None]
    ids = (g[..., 0] * M + g[..., 1]) * M + g[..., 2]
    loc = np.searchsorted(C.nodes, ids)
    xa = torch.as_tensor(C.xyz[loc], dtype=dt, device=dev)                       # (m, 27, 3)
    N, dN = MC.q2_basis(t, o - 1)
    Jx = torch.einsum('mai,maj->mij', xa, dN) * (2 * n)                          # dx/dX
    nr = torch.as_tensor(np.array(np.broadcast_to(normal_ref, P.shape)), dtype=dt, device=dev)
    v = torch.linalg.solve(Jx.transpose(1, 2), nr[..., None]).squeeze(-1)        # J^-T n
    fac = torch.linalg.det(Jx) * v.norm(dim=1)
    x = torch.einsum('ma,mai->mi', N, xa)
    return fac, x


class MappedTraction(PG2.Traction):
    """prep_geo2.Traction on a MappedCell: quadrature weights times the physical surface measure, traction fields sampled
    at the physical quadrature points. Same quadrature points and nodal shares as the reference cell."""

    def __init__(self, C, axis=None, value=None, cut=False, pts=6):
        super().__init__(C, axis, value, cut, pts)
        if self.m == 0:
            return
        P = self.X.cpu().numpy()
        if cut:
            nr = np.asarray(C.normal, float); nr = nr / np.linalg.norm(nr)
        else:
            nr = np.eye(3)[axis]
        fac, x = _face_factor(C, P, nr)
        A = self.A.to_sparse_coo().coalesce()
        i = A.indices()
        self.A = torch.sparse_coo_tensor(i, A.values() * fac[i[1]], A.shape).coalesce().to_sparse_csr()
        self.X = x
        self.area = float((fac * (self.area / self.m)).sum())


# ------------------------------------------------------------------------------------------------------ banks
def class_gen(case, mname, cls):
    s = int(hashlib.sha256(f'{case}|{mname}|{cls}'.encode()).hexdigest()[:8], 16)
    return torch.Generator(device=dev).manual_seed(s)


def finish(C, q):
    q = q - C.Q @ (C.Q.T @ q)
    e = torch.cat([(q[:, j:j + 64] * C.apply(q[:, j:j + 64])).sum(0) for j in range(0, q.shape[1], 64)])
    return q / torch.sqrt(e)[None, :], e


def raw_banks(C, case, mname, total, classes):
    """Direction banks before normalisation; the force classes need the Neumann factor only."""
    onport = np.isin(C.nodes, C.port_node_ids)
    X = torch.as_tensor(C.xyz[onport], dtype=dt, device=dev)
    isbox = torch.as_tensor(C.port_is_box, device=dev)
    out, info = {}, {}
    PG2.Traction = MappedTraction                                               # box_tractions / force_c / face_c
    for cls in classes:
        t = time.perf_counter(); gen = class_gen(case, mname, cls)
        if cls == 'force':
            nw = total - total // 4
            f = torch.cat([PD.plane_waves(X, nw, 0.5, 8.0, gen), PD.patches(X, total - nw, gen)], 2)
            f = f[:, :, torch.randperm(total, device=dev, generator=gen)]
            cutload = torch.rand(total, device=dev, generator=gen) < (0.1 if C.is_cut.any() else 0.0)
            f = f * torch.where(cutload[None, None, :], torch.ones_like(isbox, dtype=dt)[:, None, None], isbox.to(dt)[:, None, None])
            F = f.reshape(-1, total)
            q = torch.cat([C.neumann(F[:, j:j + 64]) for j in range(0, total, 64)], 1)
        elif cls == 'macro':
            q = PD.polys(X, total, gen).reshape(-1, total)
        elif cls == 'grf':
            q = PD.plane_waves(X, total, 0.5, 24.0, gen).reshape(-1, total)
        elif cls == 'force_c':
            q, _ = PG2.force_c(C, PG2.box_tractions(C), total, gen)
        elif cls == 'face_c':
            q, _ = PG2.face_c(C, PG2.box_tractions(C), total, gen)
        else:
            raise ValueError(cls)
        out[cls] = q
        info[cls] = dict(seconds=time.perf_counter() - t)
    return out, info


def finish_banks(C, raw, info):
    """Unit exact energy of K~ (needs the interior factor only)."""
    out = {}
    for cls, q in raw.items():
        q, e = finish(C, q)
        out[cls] = q
        info[cls]['energy_raw_quantiles'] = np.quantile(e.cpu().numpy(), [0, .5, 1]).tolist()
    return out


def write_data(C, d, banks, splits, body_dir):
    d.mkdir(parents=True, exist_ok=True)
    for cls, q in banks.items():
        lo = 0
        for name, m in splits:
            np.save(d / f'{name}_{cls}.npy', q[:, lo:lo + m].T.contiguous().to(torch.float32).cpu().numpy()); lo += m
    (d / 'PORTS.json').write_text(json.dumps(dict(port_node_ids=C.port_node_ids.tolist())))
    N = len(C.nodes)                                                            # prep_data.netdata with the mapped K~
    diag3 = torch.zeros((N, 3, 3), dtype=dt, device=dev)
    ru, cu = C.ru.long(), C.cu.long()
    same = (ru // 3) == (cu // 3)
    r, c, v = ru[same], cu[same], C.vals[same]
    diag3.index_put_((r // 3, r % 3, c % 3), v, accumulate=True)
    off = r != c
    diag3.index_put_((c[off] // 3, c[off] % 3, r[off] % 3), v[off], accumulate=True)
    nrm = diag3.reshape(N, 9).norm(dim=1)
    weak = (nrm < 0.01 * nrm.median()).cpu().numpy()
    np.savez(d / 'NETDATA.npz', node_ids=C.nodes, grid=np.stack(np.unravel_index(C.nodes, (2 * C.n + 1,) * 3), 1).astype(np.int16),
             elem_cells=C.cells, elem_nodes=(C.dofs[:, ::3] // 3).cpu().numpy().astype(np.int32),
             moments=C.M.cpu().numpy(), gp_faces=np.load(Path(body_dir) / C.case / 'GP_FACES.npy'),
             is_box=C.is_box, is_cut=C.is_cut, is_port=np.isin(C.nodes, C.port_node_ids), weak=weak,
             diag3=diag3.cpu().numpy(), taus=np.asarray(C.taus0), normal=np.asarray(C.normal if C.normal else [0, 0, 0]),
             offset=np.asarray(C.offset if C.offset is not None else 0.0), n=C.n)
    return dict(weak_nodes=int(weak.sum()))


# ------------------------------------------------------------------------------------------------------ variants
def nodal_rotations(C):
    """Polar factor R (N, 3, 3) of dx/dX at every node (averaged over the elements sharing the node)."""
    xi = torch.as_tensor(C.xi_nodes, dtype=dt, device=dev)
    _, dN = MC.q2_basis(xi, C.xi_nodes)                                          # (27 points, 27, 3)
    N = len(C.nodes)
    acc = torch.zeros((N, 3, 3), dtype=dt, device=dev); cnt = torch.zeros(N, dtype=dt, device=dev)
    nodes_e = C.dofs[:, ::3] // 3
    for lo in range(0, len(nodes_e), 8192):
        J = torch.einsum('eai,paj->epij', C.xe[lo:lo + 8192], dN) * (2 * C.n)   # J at the element's own nodes
        ne = nodes_e[lo:lo + 8192]
        acc.index_add_(0, ne.reshape(-1), J.reshape(-1, 3, 3)); cnt.index_add_(0, ne.reshape(-1), torch.ones(ne.numel(), dtype=dt, device=dev))
    J = acc / cnt[:, None, None]
    U, _, Vh = torch.linalg.svd(J)
    return U @ Vh


class MappedGeo(TL.Geo):
    """Geo on a MappedCell with a variant-dependent field()."""

    def set_variant(self, variant, model_wrap):
        C = self.C
        self.variant, self.model_wrap = variant, model_wrap
        onport = np.isin(C.nodes, C.port_node_ids)
        if variant == 'V0ref':
            g = np.stack(np.unravel_index(C.port_node_ids, (2 * C.n + 1,) * 3), 1) / (2 * C.n)
            import ops as OP
            self.RP = OP.rigid_raw(C.port_node_ids, C.n, g.mean(0)); self.RA = OP.rigid_raw(C.nodes, C.n, g.mean(0))
        else:
            ctr = C.xyz[onport].mean(0)
            self.RP = MC.rigid_raw_xyz(C.xyz[onport], ctr); self.RA = MC.rigid_raw_xyz(C.xyz, ctr)
        self.RPpinv = torch.linalg.pinv(self.RP)
        if variant in ('V0R', 'V0Rc'):
            R = nodal_rotations(C).to(torch.float32)
            self.Rall = R; self.Rport = R[torch.as_tensor(onport, device=dev)]

    def field(self, model, q):
        q32 = q.to(torch.float32)
        c = self.RPpinv.to(torch.float32) @ q32
        qd = q32 - self.RP.to(torch.float32) @ c
        if self.variant == 'Conly':
            u = torch.zeros((self.nb, q.shape[1]), dtype=torch.float32, device=dev)
        elif self.variant in ('V0R', 'V0Rc'):
            k = qd.shape[1]
            ql = torch.einsum('nji,njk->nik', self.Rport, qd.reshape(-1, 3, k)).reshape(-1, k)     # R^T q
            ul = model(self, ql)
            u = torch.einsum('nij,njk->nik', self.Rall, ul.reshape(-1, 3, k)).reshape(-1, k)       # R u
        else:
            u = model(self, qd)
        u = u + self.RA.to(torch.float32) @ c
        u = u.index_copy(0, self.P, q32)
        return TL.wrap(self.C, u, self.model_wrap)


class _Wrap:                                                                   # correction settings of the model
    def __init__(self, m):
        self.smooth_k = getattr(m, 'smooth_k', 0)
        self.coarse_space = getattr(m, 'coarse_space', None)
        self.smooth_alpha = getattr(m, 'smooth_alpha', 30.0)


def worst(g, model, k=8, max_iters=25, rtol=1e-2, seed=0):
    """Top Ritz value mu of (S_hat, S~) by block power iteration with S~-orthonormalisation (Geo.adversarial, one step per
    call), stopped when mu - 1 changes by less than rtol relative (Geo.worst_ratio tests mu itself, which stops at once when
    mu - 1 is ~1e-4)."""
    gen = torch.Generator(device=dev).manual_seed(seed)
    X, prev, mu, it = None, None, float('nan'), 0
    for it in range(1, max_iters + 1):
        X, ritz = g.adversarial(model, k=k, iters=1, gen=gen, start=X)
        mu = float(ritz[0])
        if prev is not None and abs(mu - prev) <= rtol * abs(mu - 1):
            break
        prev = mu
    return mu, it


def rigid_energy(g, model):
    """Energy of the extensions of the six orthonormal PHYSICAL port rigid modes (exactly zero for an extension that
    reproduces the mapped rigid motions), relative to the mean exact energy of unit-norm force_c / force bank directions."""
    C = g.C
    u = g.field(model, C.Q)
    e = TL.energy(u, C.K)
    cls = 'force_c' if 'force_c' in g.classes else g.classes[0]
    Q = g.banks['val'][cls].to(dt)
    scale = float((1.0 / (Q * Q).sum(0)).mean())                                # banks at unit energy: q^T S q / |q|^2
    return float(e.max()) / scale


def evaluate(g, model, variant, chunk=16):
    out = {}
    for cls in g.classes:
        Q = g.banks['val'][cls]
        e = torch.cat([TL.energy(g.field(model, Q[:, j:j + chunk]), g.C.K) - 1 for j in range(0, Q.shape[1], chunk)]).cpu().numpy()
        out[cls] = dict(mean=float(e.mean()), p90=float(np.quantile(e, .9)), max=float(e.max()), min=float(e.min()))
    return out


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('ckpt'); ap.add_argument('body'); ap.add_argument('cases'); ap.add_argument('maps')
    ap.add_argument('--val', type=int, default=64); ap.add_argument('--worst', type=int, default=1)
    ap.add_argument('--variants', default='V0,V0R,V0ref,Conly')
    ap.add_argument('--classes', default='force,macro,grf,force_c,face_c')
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    maps = json.loads(Path(a.maps).read_text())
    variants = a.variants.split(','); classes = a.classes.split(',')
    splits = (('train', 8), ('val', a.val), ('test', 8)); total = sum(m for _, m in splits)
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False); cfg = ck['cfg']
    model = None
    res_path = out / 'RESULTS.jsonl'
    done = set()
    if res_path.exists():
        done = {(r['case'], r['map']) for r in map(json.loads, res_path.read_text().splitlines()) if 'error' not in r}
    for case in [c for c in a.cases.split(',') if c]:
        for m in maps:
            if (case, m['name']) in done:
                continue
            rec = dict(case=case, map=m['name'], spec=m['spec'])
            t0 = time.perf_counter()
            try:
                torch.cuda.reset_peak_memory_stats()
                C = MC.MappedCell(case, a.body, m['spec'], log=lambda s_: None)
                t = time.perf_counter(); C.assemble(); rec['assemble_s'] = time.perf_counter() - t
                rec['map_stats'] = C.map_stats
                X = torch.linalg.qr(torch.randn((C.nb, 6), dtype=dt, device=dev, generator=torch.Generator(device=dev).manual_seed(0)))[0]
                rec['rigid_phys_over_random'] = float((C.K @ C.Qall).norm() / (C.K @ X).norm())
                del X
                # one sparse factor on the GPU at a time: Neumann (force banks, worst direction) and interior (energies)
                t = time.perf_counter(); C.factor(neumann=True, interior=False); rec['factor_neumann_s'] = time.perf_counter() - t
                t = time.perf_counter(); raw, rec['banks'] = raw_banks(C, case, m['name'], total, classes)
                C._free()
                t2 = time.perf_counter(); C.factor(neumann=False); rec['factor_interior_s'] = time.perf_counter() - t2
                banks = finish_banks(C, raw, rec['banks']); del raw
                rec['banks_s'] = time.perf_counter() - t
                d = out / 'data' / m['name'] / case
                rec['netdata'] = write_data(C, d, banks, splits, a.body); del banks
                g = MappedGeo(case, a.body, out / 'data' / m['name'], neumann=False, log=lambda s_: None, cell=C)
                g.case = f'{case}@{m["name"]}'                                  # own network cache per map
                if model is None:
                    model = MD.build(cfg['model'], [g], **dict(cfg.get('model_args', {}))).to(dev)
                    MD.load_compat(model, ck['model']); model.eval()
                else:
                    model.add_geo(g)
                wrap = _Wrap(model)
                rec['variants'] = {}
                for v in variants:                                             # energies: no factor needed
                    t = time.perf_counter()
                    g.set_variant(v, wrap)
                    if v == 'V0Rc':
                        CR.install(); CR.set_corot(C, nodal_rotations(C))
                    elif getattr(C, '_cr_Minv', None) is not None:
                        CR.clear_corot(C)
                    with torch.no_grad():
                        r = evaluate(g, model, v)
                        r['rigid_energy_rel'] = rigid_energy(g, model)
                    r['seconds'] = time.perf_counter() - t
                    rec['variants'][v] = r
                C._free()
                if a.worst:                                                    # worst direction: Neumann factor only
                    t = time.perf_counter(); C.factor(neumann=True, interior=False); rec['factor_neumann2_s'] = time.perf_counter() - t
                    for v in ('V0', 'V0R', 'V0Rc'):
                        if v in rec['variants']:
                            t = time.perf_counter(); g.set_variant(v, wrap)
                            if v == 'V0Rc':
                                CR.set_corot(C, nodal_rotations(C))
                            elif getattr(C, '_cr_Minv', None) is not None:
                                CR.clear_corot(C)
                            mu, it = worst(g, model)
                            rec['variants'][v].update(worst_mu=mu, worst_iters=it, worst_seconds=time.perf_counter() - t)
                    C._free()
                rec['tail_bounds'] = list(getattr(C, '_tail_bounds', ()) or ())
                model.caches.pop(g.case, None)
                rec['gpu_peak_gb'] = torch.cuda.max_memory_allocated() / 2 ** 30
                C._free(); del g, C
            except Exception as e:                                               # record and continue with the next map
                import traceback
                rec['error'] = repr(e)[:400]; rec['trace'] = traceback.format_exc()[-1500:]
                try:                                                           # release the sparse factors on every path
                    C._free(); model.caches.pop(g.case, None); del g, C
                except Exception:
                    pass
            rec['seconds'] = time.perf_counter() - t0
            rec['conv'] = TL.conv_precision()
            with open(res_path, 'a') as f:
                f.write(json.dumps(rec) + '\n')
            brief = {v: {c: round(100 * x['mean'], 4) for c, x in r.items() if isinstance(x, dict)}
                     for v, r in rec.get('variants', {}).items()}
            print(json.dumps(dict(case=case, map=m['name'], seconds=round(rec['seconds'], 1), error=rec.get('error'),
                                  mean_pct=brief, worst={v: r.get('worst_mu') for v, r in rec.get('variants', {}).items()})), flush=True)
            gc.collect(); torch.cuda.empty_cache()
    print('DONE', flush=True)


if __name__ == '__main__':
    main(sys.argv[1:])
