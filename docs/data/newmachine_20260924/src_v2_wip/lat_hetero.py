"""P1 item 9: heterogeneous lattices (gen_hlat.py layouts) with EVERY cell learned, against the exact lattice.
Lattice: lat_multi.MultiLattice, clamped on y = min, traction-consistent unit loads x / y / z on y = max (+ n_random random
loads); PCG with one deployable preconditioner (--prec, lat_precond.Factory) to --tol on all loads.
  exact     bench_deploy.SparseExactOp per cell (GPU interior factor fp32 + fp64 refinement): reference solution U, compliance,
            per-cell fields E_c q_c, 8-corner sensitivities s_c = -u_c^T K_c,tau u_c and energies q_c^T S_c q_c
  learned   one run per --model (name=ckpt[;json model_args override]): every cell through fastnet (FastOp); compliance error
            per load, per-cell sensitivity error ||s_hat - s|| / ||s|| from the recovered learned fields at the learned
            solution, eps_c = q_c^T (S_hat_c - S_c) q_c / q_c^T S_c q_c at the exact traces and the bound sum_c w_c eps_c,
            relative solution distance, PCG iterations / seconds, recomputed residual
Gate-style summary on the consistent loads: max compliance error, max over cells of the sensitivity error.
Usage: lat_hetero.py <out.json> <layout.json>[,...] --model A3=<ckpt> [--model 'B2grid=<ckpt>;{"smooth_k": 8, ...}']
       [--prec bnn:kpp:q1r] [--exact-host | --exact-dense] [--tol 1e-10] [--maxit 3000] [--n-random 3] [--body /root/autodl-tmp/OPL/S4/body]"""
import json, time, gc, os, argparse
from pathlib import Path
import numpy as np
import torch
import models as MD                                                    # noqa: F401  first: applies OPL_CONV_FP32
import teacher as TE
import trainlib as TL
import fastnet as FN
import evalnet as EN
import bench_deploy as BD
import lat_multi as LM
import lat_precond as PR

dev, dt = TE.dev, TE.dt


def holder_for(spec):
    name, rest = spec.split('=', 1)
    ck, ov = (rest.split(';', 1) + [None])[:2]
    h = BD.ModelHolder(ck)
    if ov:
        h.ck['cfg'] = dict(h.ck['cfg'], model_args=dict(h.ck['cfg'].get('model_args', {}), **json.loads(ov)))
    return name, h


def free():
    gc.collect(); torch.cuda.empty_cache()


def exact_op(C):
    """Interior factor on the GPU after releasing the torch cache (cuDSS allocates outside torch); fp64 retry."""
    free()
    try:
        return BD.SparseExactOp(C)
    except Exception:                                                        # noqa: BLE001  cuDSS ALLOC_FAILED
        C._free(); free()
        C.factor(neumann=False, fp32=False)
        return BD.SparseExactOp(C)


class _HostSPD:
    """Interior K_II factor held by MKL PARDISO in host memory (fp64); right-hand sides move to the host and back."""

    def __init__(self, A):
        import pypardiso
        self.A, self.ps = A, pypardiso.PyPardisoSolver()
        self.ps.factorize(A)

    def solve(self, r):
        x = self.ps.solve(self.A, np.ascontiguousarray(r.detach().to('cpu', torch.float64).numpy()))
        return torch.as_tensor(x, device=r.device).to(r.dtype).reshape(r.shape)

    def free(self):
        self.ps.free_memory(everything=True)


def exact_op_host(C):
    """--exact-host: the same scaled interior block as teacher.Cell.factor, factorised on the host, so that the exact
    factors of all cells of a lattice can be held at once (the GPU holds one or two)."""
    import scipy.sparse as sp
    C._free(); free()
    P, I = C.P.cpu(), C.I.cpu()
    pm = torch.zeros(C.nb, dtype=torch.bool); pm[P] = True
    new = torch.full((C.nb,), -1, dtype=torch.long); new[I] = torch.arange(C.ni)
    ru, cu, vals = C.ru.long().cpu(), C.cu.long().cpu(), C.vals.cpu().to(torch.float64)
    sel = torch.nonzero(~pm[ru] & ~pm[cu]).squeeze(1)
    rA, cA, vA = new[ru[sel]], new[cu[sel]], vals[sel]
    sA = torch.zeros(C.ni, dtype=torch.float64); sA[rA[rA == cA]] = 1 / torch.sqrt(vA[rA == cA])
    U = sp.csr_matrix(((vA * sA[rA] * sA[cA]).numpy(), (rA.numpy(), cA.numpy())), shape=(C.ni, C.ni))
    A = (U + sp.triu(U, 1).T).tocsr()
    C.sA, C.sol_I, C.fp32 = sA.to(dev, dt), _HostSPD(A), False
    return BD.SparseExactOp(C)


class DenseExactOp:
    """--exact-dense: exact S of a cell as a dense host matrix (T64.npy from make_T_gpu.py / make_T_cpu.py), computed one cell
    at a time beforehand, so that the assembled exact solve holds no factorisation; fields are recovered afterwards cell by cell."""

    def __init__(self, C, body):
        from pathlib import Path
        pd = Path(body) / (C.case + '_portview')
        assert np.array_equal(np.load(pd / 'BOX_NODES.npy'), C.port_node_ids), C.case
        self.C, self.T = C, torch.from_numpy(np.load(pd / 'T64.npy'))

    def apply(self, q):
        return (self.T @ q.detach().to('cpu', torch.float64)).to(q.device, q.dtype)


def _move(C, device, min_bytes=1 << 24):
    """Move the cell's large tensors (stiffness, correction caches, moments) between host and device."""
    for k, v in list(vars(C).items()):
        if torch.is_tensor(v) and v.device != device:
            big = v.layout != torch.strided or v.numel() * v.element_size() >= min_bytes
            if big:
                setattr(C, k, v.to(device))


class ParkedOp:
    """--park: the learned operator of a cell whose large tensors live on the host between uses; they move to the
    device for each call (or for a 'with op.active():' block), so that only one parked cell occupies the device at a time.
    The arithmetic is unchanged; only where the tensors wait between calls."""

    def __init__(self, op, C, resident=False):
        self.op, self.C, self.resident, self.depth = op, C, resident, 0

    def active(self):
        import contextlib
        me = self

        @contextlib.contextmanager
        def ctx():
            if not me.resident and me.depth == 0:
                _move(me.C, dev)
            me.depth += 1
            try:
                yield
            finally:
                me.depth -= 1
                if not me.resident and me.depth == 0:
                    _move(me.C, torch.device('cpu')); torch.cuda.empty_cache()
        return ctx()

    def apply(self, q):
        with self.active():
            return self.op.apply(q)

    def field(self, q):
        with self.active():
            return self.op.field(q)


def solve(lat, ops, spec, tol, maxit, kpp=None):
    fac = PR.Factory(lat, ops, shared={} if kpp is None else {'kpp_triplets': kpp, 'kpp_triplets_s': 0.0},
                     kpp_backend='auto', log=lambda d: None)
    pc, st, _ = fac.build(spec)
    r = PR.solve(lat, ops, pc, tol=tol, maxit=maxit)
    out = dict(iterations=r['iterations'], seconds=r['seconds'], residual=r['residual'], setup_s=st['total_s'],
               true_residual=PR.true_residual(lat, ops, r['X']))
    X = r['X']
    fac.free(); del pc, r
    return X, out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('layouts'); ap.add_argument('--model', action='append', default=[])
    ap.add_argument('--prec', default='bnn:kpp:q1r'); ap.add_argument('--tol', type=float, default=1e-10)
    ap.add_argument('--maxit', type=int, default=3000); ap.add_argument('--n-random', type=int, default=3)
    ap.add_argument('--body', default='/root/autodl-tmp/OPL/S4/body')
    ap.add_argument('--exact-host', action='store_true', help='exact interior factors by host PARDISO (default: GPU)')
    ap.add_argument('--exact-dense', action='store_true', help='exact reference from cached dense S per cell (make_T_*.py)')
    ap.add_argument('--max-cols', type=int, default=64, help='columns per operator call (lower for large learned cells)')
    ap.add_argument('--lean', action='store_true', help='learned cells with lean() storage (element matrices, float64 arithmetic unchanged)')
    ap.add_argument('--deploy', action='store_true', help='learned cells in deployment mode: lean(deploy=True), float32 correction, fused tail transpose, float32 coarse solve')
    ap.add_argument('--park', action='store_true', help='keep learned cells on the host between operator calls')
    ap.add_argument('--resident', type=int, default=0, help='with --park: number of cells kept on the device')
    a = ap.parse_args(argv)
    if a.deploy:
        os.environ['OPL_TAILT_FUSED'] = '1'; os.environ['OPL_COARSE_FP32'] = '1'
    log = lambda d: print(json.dumps(d, default=float), flush=True)
    res = dict(args=vars(a), gpu=torch.cuda.get_device_name(0), conv=TL.conv_precision(), lattices={})
    BD.TMP.mkdir(parents=True, exist_ok=True)
    for lp in a.layouts.split(','):
        L = json.loads(Path(lp).read_text())
        rec = dict(layout=L['name'], cells=[c['case'] for c in L['cells']], kinds=[c['kind'] for c in L['cells']])
        res['lattices'][L['name']] = rec
        t = time.perf_counter()
        Cs, lay = [], {}
        for c in L['cells']:
            C = TE.Cell(c['case'], a.body, log=lambda s_: None); C.assemble()
            Cs.append(C); lay[tuple(c['position'])] = LM.from_teacher(C)
            if a.park:                                                          # one assembled cell on the device at a time
                _move(C, torch.device('cpu')); free()
        lat = LM.MultiLattice(lay, clamp=('y', 'min'), load=('y', 'max'), loads='consistent', n_random=a.n_random, device=dev,
                              max_cols=a.max_cols, log=lambda s_: None)
        order = [lay_g.case for lay_g in lat.geoms]
        Cmap = {C.case: C for C in Cs}
        rec.update(setup_s=time.perf_counter() - t, **{k: v for k, v in lat.info().items() if k not in ('positions', 'cases')})
        ncons = lat.F.shape[1] - a.n_random
        kpp = lat.assemble_kpp() if a.park else None                         # exact K_PP once, all cells on the device
        # exact reference
        if a.exact_dense:
            ops = [DenseExactOp(Cmap[c], a.body) for c in order]
        else:
            ops = [(exact_op_host if a.exact_host else exact_op)(Cmap[c]) for c in order]
        X, st = solve(lat, ops, a.prec, a.tol, a.maxit, kpp)
        comp = (lat.F * X).sum(0).cpu().numpy()
        S, E = [], []
        for i, c in enumerate(order):
            C = Cmap[c]; q = lat.gather(X, i)
            if a.park:
                _move(C, dev, min_bytes=0)
            if a.exact_dense:                                                   # one interior factor at a time
                exact_op_host(C)
            u = C.extend(q.to(dt)); C.dmoments()
            S.append(C.sens(u).cpu()); E.append((u * (C.K @ u)).sum(0).cpu())
            del u
            if a.exact_dense:
                C._free(); free()
            if a.park:
                _move(C, torch.device('cpu')); free()
        rec['exact'] = dict(st, compliance=comp.tolist(), energy_share=(torch.stack(E) / torch.as_tensor(comp)).tolist())
        log(dict(event='EXACT', lattice=L['name'], **st, compliance=comp.tolist()))
        Xref = X.cpu()
        for C in Cs:
            C._free()
        del ops; free()
        for spec in a.model:
            name, h = holder_for(spec)
            t = time.perf_counter()
            ops = []
            if a.park:
                for C in Cs:
                    _move(C, torch.device('cpu'))
                free()
            for j, c in enumerate(order):
                C = Cmap[c]
                if a.park:
                    _move(C, dev)
                BD.netdata(C, a.body, BD.TMP / c)
                geo = TL.Geo(c, a.body, BD.TMP, neumann=False, log=lambda s_: None, cell=C, load_banks=False)
                op = EN.FastOp(FN.FastNet(h.add(geo), geo))
                if a.deploy and not getattr(C, '_lean', False):
                    C.lean(deploy=True); free()
                elif a.lean and not getattr(C, '_lean', False):
                    C.lean(); free()
                if a.park:
                    op.apply(torch.zeros((C.np_, 1), dtype=dt, device=dev))            # builds the correction caches on C
                    op = ParkedOp(op, C, resident=j < a.resident)
                    if not op.resident:
                        _move(C, torch.device('cpu')); free()
                ops.append(op)
            prep = time.perf_counter() - t
            X, st = solve(lat, ops, a.prec, a.tol, a.maxit, kpp)
            ch = (lat.F * X).sum(0).cpu().numpy()
            cerr = np.abs(ch - comp) / np.abs(comp)
            serr, eps = [], []
            import contextlib
            for i, c in enumerate(order):
                C = Cmap[c]
                with (ops[i].active() if a.park else contextlib.nullcontext()):
                    if a.park:
                        _move(C, dev, min_bytes=0)                              # sens reads the small tensors too
                    u = ops[i].field(lat.gather(X, i)).to(dt)
                    sh = C.sens(u).cpu(); del u
                    serr.append(((sh - S[i]).norm(dim=0) / S[i].norm(dim=0)).numpy())
                    qe = lat.gather(Xref.to(dev), i)
                    eps.append(((qe * ops[i].apply(qe)).sum(0).cpu() / E[i] - 1).numpy())
            serr, eps = np.stack(serr), np.stack(eps)
            w = np.asarray(torch.stack(E)) / comp[None]
            row = dict(st, prep_s=prep, compliance_rel_err=cerr.tolist(), sens_rel_err=serr.tolist(), eps=eps.tolist(),
                       bound=(w * eps).sum(0).tolist(),
                       solution_rel_err=float((X.cpu() - Xref).norm() / Xref.norm()),
                       gate_compliance_max=float(cerr[:ncons].max()), gate_sens_max=float(serr[:, :ncons].max()),
                       gate_sens_max_per_cell=serr[:, :ncons].max(1).tolist(), random_compliance_max=float(cerr[ncons:].max()) if a.n_random else None)
            rec[name] = row
            log(dict(event='LEARNED', lattice=L['name'], model=name, gate_compliance_max=row['gate_compliance_max'],
                     gate_sens_max=row['gate_sens_max'], iterations=st['iterations'], seconds=st['seconds'], prep_s=prep,
                     true_residual=st['true_residual']))
            for c in order:
                h.model.caches.pop(c, None)
            del ops, h; free()
            if a.park:
                for C in Cs:
                    for k in ('_cV', '_cL', '_c_space', '_tail_bounds'):          # correction caches are per model settings
                        if hasattr(C, k):
                            delattr(C, k)
                free()
            Path(a.out).write_text(json.dumps(res, indent=1, default=float))
        for C in Cs:
            C._free()
        del Cs, Cmap, lat; free()
        Path(a.out).write_text(json.dumps(res, indent=1, default=float))


if __name__ == '__main__':
    main()
