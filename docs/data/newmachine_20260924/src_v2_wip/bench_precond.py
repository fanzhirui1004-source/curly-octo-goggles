"""Deployable preconditioners for multi-cell lattice solves (lat_multi + lat_precond) with exact and learned cell operators.

For every lattice and every preconditioner: PCG (deflated PCG for 'defl:*') to a relative residual of tol (default 1e-8,
max 3000 iterations) on all load columns at once, with
  exact    bench_deploy.SparseExactOp per DISTINCT case (one fp32 interior factor + fp64 refinement, shared by all cells of
           that case: one batched apply per case)
  learned  fastnet.FastNet (evalnet.FastOp) per distinct case (skipped when <ckpt> is 'exact')
Reported per run: iterations, converged, seconds, seconds per iteration, setup seconds (breakdown: K_PP assembly /
factorisation, coarse basis, A Z operator applications, coarse eigendecomposition), seconds per preconditioner application,
recursive and true final residual, compliance per load and its relative error against the exact-operator reference
(the exact run with the smallest true residual), relative 2-norm distance of the solution to that reference.
Per lattice / operator set: free DOFs, loads, matvec seconds, coarse space sizes, peak memory.
Lattices (--lattice, repeatable; default: the four below):
  pair:<x|y>:<test case>:<nbr case>     lattice3's two-cell configuration (neighbour at -x / -y, its far face clamped,
                                         traction-consistent loads on the plane y = 0 / x = 0; + random loads); the gluing
                                         is cross-checked against lattice3.Lattice (same free DOFs and numbering)
  <nx>x<ny>x<nz>:<case>                  block of one case, clamped on x = min, traction on x = max
  <nx>x<ny>x<nz>:<case>+cut:<cut case>   the block plus one layer of cut cells on the side the cut normal points to
                                         (lat_multi.cut_layer); clamp / load on the min / max faces of the first axis
                                         other than the cut axis (both faces then cross FULL and cut cells)
Usage: bench_precond.py <ckpt | exact> <out.json> [--lattice SPEC ...] [--precs a,b,...] [--tol 1e-8] [--maxit 3000]
       [--max-seconds S] [--n-random 3] [--max-cols 64] [--kpp-backend auto|cudss|cholmod|splu] [--body DIR]
Convolution precision (INVARIANTS-2): OPL_CONV_FP32=1 -> true fp32 convolutions (models reads it at import).
"""
import json, time, gc, os, re, argparse
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
LATTICES = ['pair:x:fresh_train_0020_cover01_r2:fresh_train_0020_full', '2x2x2:fresh_train_0020_full',
            '3x3x3:fresh_train_0020_full', '2x2x1:fresh_train_0020_full+cut:fresh_train_0020_cover01_r2']
PRECS = ['jacobi', 'kpp', 'add:jac:q1r', 'add:jac:pu6', 'bnn:kpp:q1r', 'bnn:kpp:pu12', 'defl:jac:q1r', 'defl:kpp:q1r']
CELLS = {}


def free_mem():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def cell(case, body):
    """teacher.Cell (assembled) and its CellGeom, once per case."""
    if case not in CELLS:
        C = TE.Cell(case, body, log=lambda s_: None)
        C.assemble()
        CELLS[case] = (C, LM.from_teacher(C))
    return CELLS[case]


def parse_lattice(spec, body):
    m = re.match(r'^pair:([xy]):([^:]+):([^:]+)$', spec)
    if m:
        conf, t, nb = m.groups()
        off = (-1, 0, 0) if conf == 'x' else (0, -1, 0)
        lay = {(0, 0, 0): cell(t, body)[1], off: cell(nb, body)[1]}
        return lay, (conf, 'min'), ('y' if conf == 'x' else 'x', 'min'), dict(kind='pair', config=conf, offset=off)
    m = re.match(r'^(\d+)x(\d+)x(\d+):([^+]+)(?:\+cut:(.+))?$', spec)
    if not m:
        raise ValueError(f'LATTICE_SPEC:{spec}')
    shape = tuple(int(m.group(i)) for i in (1, 2, 3))
    full, cutc = m.group(4), m.group(5)
    if cutc is None:
        return LM.block_layout(shape, cell(full, body)[1]), ('x', 'min'), ('x', 'max'), dict(kind='block', shape=shape)
    lay, ax, side = LM.cut_layer(shape, cell(full, body)[1], cell(cutc, body)[1])
    a = [c for c in 'xyz' if c != ax][0]
    return lay, (a, 'min'), (a, 'max'), dict(kind='block+cut', shape=shape, cut_axis=ax, cut_side=side)


def lattice3_check(lat, desc, body):
    """Same free DOFs / numbering as lattice3's two-cell lattice (T is not needed to build it)."""
    import lattice3 as LT
    os.environ.pop('LAT_LOADS', None)
    cells = [dict(label=l_, cell=G.cell, offset=p, T=None) for l_, p, G in zip(('test', 'nbr'), lat.positions, lat.geoms)]
    l3 = LT.Lattice(cells, desc['config'], log=lambda s_: None)
    same = len(l3.free) == lat.nfree and all(torch.equal(a.cpu(), b.cpu()) for a, b in zip(lat.gather_idx, l3.gather_idx))
    return dict(lattice3_free_dofs=int(len(l3.free)), lattice3_same_numbering=bool(same))


def exact_ops(lat):
    opmap = {}
    for G in lat.geoms:
        if G.case not in opmap:
            opmap[G.case] = BD.SparseExactOp(G.cell)
    return opmap


def learned_ops(lat, holder, body):
    opmap = {}
    for G in lat.geoms:
        if G.case not in opmap:
            C = G.cell
            BD.netdata(C, body, BD.TMP / C.case)
            geo = TL.Geo(C.case, body, BD.TMP, neumann=False, log=lambda s_: None, cell=C, load_banks=False)
            opmap[G.case] = EN.FastOp(FN.FastNet(holder.add(geo), geo))
    return opmap


def compare(row, X, ref):
    c = np.asarray(row['compliance'])
    row['compliance_rel_err'] = (np.abs(c - ref['compliance']) / np.abs(ref['compliance'])).tolist()
    row['compliance_rel_err_max'] = float(max(row['compliance_rel_err']))
    Xr = ref['X'].to(X.device)
    row['solution_rel_err'] = float((X - Xr).norm() / Xr.norm())


def run_opset(lat, oname, opmap, args, shared, ref, rec, out, log):
    ops = [opmap[G.case] for G in lat.geoms]
    sols = []
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    _, mv = BD.timed(lambda: lat.matvec(ops, lat.F), reps=2, warm=1)
    rec.setdefault('matvec_s', {})[oname] = mv
    log(dict(event='MATVEC', operators=oname, seconds=mv, columns=int(lat.F.shape[1]) * len(ops)))
    fac = PR.Factory(lat, ops, shared=shared, kpp_backend=args.kpp_backend, log=lambda d: log(dict(operators=oname, **d)))
    for spec in args.precs.split(','):
        row = dict(operators=oname, preconditioner=spec)
        try:
            pc, st, desc = fac.build(spec)
            row.update(setup_s=st['total_s'], setup=st, **desc)
            f = pc if isinstance(pc, PR.KppFine) else getattr(pc, 'fine', None)
            if isinstance(f, PR.KppFine):
                row.update(kpp_backend=f.backend, kpp_nnz_upper=f.nnz_upper)
            row['apply_s'] = PR.apply_cost(pc, lat.F)
            r = PR.solve(lat, ops, pc, tol=args.tol, maxit=args.maxit, max_seconds=args.max_seconds)
            X = r['X']
            row.update(iterations=r['iterations'], converged=r['residual'] < args.tol, seconds=r['seconds'],
                       s_per_iter=r['seconds'] / max(r['iterations'], 1), residual=r['residual'],
                       true_residual=PR.true_residual(lat, ops, X), history_every50=r['history'][::50])
            c = (lat.F * X).sum(0).cpu().numpy()
            row['compliance'] = c.tolist()
            if oname == 'exact':                                          # compared once the reference is final
                sols.append((row, X.cpu()))
                if ref.get('true_residual') is None or row['true_residual'] < ref['true_residual']:
                    ref.update(true_residual=row['true_residual'], compliance=c, X=X.cpu(), preconditioner=spec)
            elif ref.get('compliance') is not None:
                compare(row, X, ref)
            if torch.cuda.is_available():
                row['peak_alloc_GB'] = torch.cuda.max_memory_allocated() / 2 ** 30
            del X, r
        except Exception as e:
            import traceback
            row.update(error=repr(e)[:300], trace=traceback.format_exc()[-1500:])
        rec['runs'].append(row)
        log(dict(event='RUN', lattice=rec['spec'], **{k: v for k, v in row.items()
                                                       if k not in ('compliance', 'compliance_rel_err', 'history_every50', 'trace', 'setup')}))
        Path(args.out).write_text(json.dumps(out, indent=1, default=float))
        free_mem()
    fac.free()
    for row, X in sols:                                                   # exact rows against the final reference
        compare(row, X, ref)
    Path(args.out).write_text(json.dumps(out, indent=1, default=float))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('ckpt'); ap.add_argument('out')
    ap.add_argument('--lattice', action='append', default=None)
    ap.add_argument('--precs', default=','.join(PRECS))
    ap.add_argument('--tol', type=float, default=1e-8); ap.add_argument('--maxit', type=int, default=3000)
    ap.add_argument('--max-seconds', type=float, default=None, help='wall-clock cap per PCG run')
    ap.add_argument('--n-random', type=int, default=3); ap.add_argument('--max-cols', type=int, default=64)
    ap.add_argument('--kpp-backend', default='auto'); ap.add_argument('--body', default=BD.BODY)
    args = ap.parse_args(argv)
    lattices = args.lattice or LATTICES
    log = lambda d: print(json.dumps(d, default=float), flush=True)
    out = dict(args=vars(args), gpu=torch.cuda.get_device_name(0), conv=TL.conv_precision(), lattices=[])
    holder = None if args.ckpt == 'exact' else BD.ModelHolder(args.ckpt)
    BD.TMP.mkdir(parents=True, exist_ok=True)
    for spec in lattices:
        rec = dict(spec=spec, runs=[])
        out['lattices'].append(rec)
        try:
            t = time.perf_counter()
            lay, clamp, load, desc = parse_lattice(spec, args.body)
            rec['cells_s'] = time.perf_counter() - t
            t = time.perf_counter()
            lat = LM.MultiLattice(lay, clamp=clamp, load=load, loads='consistent', n_random=args.n_random, device=dev,
                                  max_cols=args.max_cols, log=lambda s_: None)
            rec.update(desc, lattice_s=time.perf_counter() - t, **lat.info())
            if desc['kind'] == 'pair':
                try:
                    rec.update(lattice3_check(lat, desc, args.body))
                except Exception as e:
                    rec['lattice3_check_error'] = repr(e)[:200]
            log(dict(event='LATTICE', **{k: v for k, v in rec.items() if k not in ('runs', 'positions', 'cases')}))
            shared, ref = {}, {}
            opmap = exact_ops(lat)
            run_opset(lat, 'exact', opmap, args, shared, ref, rec, out, log)
            for op in opmap.values():
                op.C._free()
            del opmap; free_mem()
            if holder is not None:
                t = time.perf_counter()
                opmap = learned_ops(lat, holder, args.body)
                BD.sync(); rec['learned_setup_s'] = time.perf_counter() - t
                run_opset(lat, 'learned', opmap, args, shared, ref, rec, out, log)
                for G in lat.geoms:
                    holder.model.caches.pop(G.case, None) if hasattr(holder.model, 'caches') else None
                del opmap
            rec['reference'] = dict(preconditioner=ref.get('preconditioner'), true_residual=ref.get('true_residual'),
                                    compliance=None if ref.get('compliance') is None else ref['compliance'].tolist())
            for k in ('kpp',):
                if k in shared:
                    shared[k].free()
            del lat, shared, ref
        except Exception as e:
            import traceback
            rec.update(error=repr(e)[:300], trace=traceback.format_exc()[-2000:])
            log(dict(event='LATTICE_FAIL', spec=spec, error=rec['error'], trace=rec['trace']))
        Path(args.out).write_text(json.dumps(out, indent=1, default=float))
        for C, _ in CELLS.values():
            C._free()
        CELLS.clear(); free_mem()
    Path(args.out).write_text(json.dumps(out, indent=1, default=float))
    import shutil
    shutil.rmtree(BD.TMP, ignore_errors=True)


if __name__ == '__main__':
    main()
