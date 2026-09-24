"""Deployment speed benchmark (task 67): what one design iteration costs per cell with the exact CutFEM teacher and with the
learned operator, and what a lattice solve costs, at the accuracy each route actually delivers. One process on the GPU,
every stage timed between synchronisations, memory as the change of free device memory (cuDSS allocates outside torch).

Per cell (the cases on the command line):
  shared   setup (topology, GP cache) | element moments | assembly of K (needed by both routes: the learned route's
           variational readout S_hat = E_hat^T K E_hat uses the exact K)
  exact    interior factorization fp64 and fp32 (+ 3 fp64 refinement steps in the solves) | reaction S q = (K E q)_P for
           B = 1, 16, 64 directions
  learned  network input data (diag blocks, weak flags) + trainlib.Geo | model cache (MGCache, GP-face setup) | fastnet
           freeze (geometry path, sparse layer matrices) | S_hat q for B = 1, 16, 64 | certificate m = 0 and 8 (B = 16) |
           energy ratio q^T S_hat q / q^T S q on the benchmark directions
Lattice (first case + its family FULL parent, config x, traction-consistent gate loads + cut-surface loads, solved together):
  conjugate gradients to a relative residual of 1e-8, operators: learned (both cells learned: the deployment case) and exact
  (per-cell interior factor: exact domain decomposition); preconditioners: 'ideal' = the dense exact lattice factor (what
  the acceptance gate uses; not deployable), 'none', 'jacobi' = inverse diagonal of the assembled port stiffness K_PP
  (deployable, no factorization); iterations, seconds, seconds per iteration, gate metrics against the exact reference.
Usage: bench_deploy.py <ckpt> <out.json> <case> [<case> ...]
"""
import sys, json, time, gc, os, shutil
from pathlib import Path
import numpy as np
import torch
import teacher as TE
import trainlib as TL
import models as MD
import fastnet as FN
import lattice3 as LT
import evalnet as EN

dev, dt = TE.dev, TE.dt
BODY = '/root/autodl-tmp/OPL/S0'
TMP = Path('/root/autodl-tmp/OPL/S1/V2/bench_tmp')


def sync():
    torch.cuda.synchronize()


def timed(fn, reps=1, warm=0):
    for _ in range(warm):
        fn()
    sync(); t = time.perf_counter()
    for _ in range(reps):
        out = fn()
    sync()
    return out, (time.perf_counter() - t) / reps


def free_gb():
    sync(); gc.collect(); torch.cuda.empty_cache()
    return torch.cuda.mem_get_info()[0] / 2 ** 30


def netdata(C, body, d):
    """prep_data.netdata without its sys.argv dependence (same arrays), plus PORTS.json: what trainlib.Geo reads."""
    N = len(C.nodes)
    diag3 = torch.zeros((N, 3, 3), dtype=dt, device=dev)
    ru, cu = C.ru.long(), C.cu.long()
    same = (ru // 3) == (cu // 3)
    r, c, v = ru[same], cu[same], C.vals[same]
    diag3.index_put_((r // 3, r % 3, c % 3), v, accumulate=True)
    off = r != c
    diag3.index_put_((c[off] // 3, c[off] % 3, r[off] % 3), v[off], accumulate=True)
    nrm = diag3.reshape(N, 9).norm(dim=1)
    weak = (nrm < 0.01 * nrm.median()).cpu().numpy()
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / 'NETDATA.npz', node_ids=C.nodes, grid=np.stack(np.unravel_index(C.nodes, (2 * C.n + 1,) * 3), 1).astype(np.int16),
             elem_cells=C.cells, elem_nodes=(C.dofs[:, ::3] // 3).cpu().numpy().astype(np.int32),
             moments=C.M.cpu().numpy(), gp_faces=np.load(Path(body) / C.case / 'GP_FACES.npy'),
             is_box=C.is_box, is_cut=C.is_cut, is_port=np.isin(C.nodes, C.port_node_ids), weak=weak,
             diag3=diag3.cpu().numpy(), taus=np.asarray(C.taus0), normal=np.asarray(C.normal if C.normal else [0, 0, 0]),
             offset=np.asarray(C.offset if C.offset is not None else 0.0), n=C.n)
    (d / 'PORTS.json').write_text(json.dumps(dict(case=C.case, port_node_ids=C.port_node_ids.tolist(),
                                                  port_is_box=C.port_is_box.tolist(), port_is_cut=C.port_is_cut.tolist())))


def directions(C, B, seed=0):
    """Smooth-ish rigid-free port directions (plane waves, like the grf bank class)."""
    import prep_data as PD
    gen = torch.Generator(device=dev).manual_seed(seed)
    X = torch.as_tensor(np.stack(np.unravel_index(C.port_node_ids, (2 * C.n + 1,) * 3), 1) / (2 * C.n), dtype=dt, device=dev)
    q = PD.plane_waves(X, B, 0.5, 8.0, gen).reshape(-1, B)
    return q - C.Q @ (C.Q.T @ q)


class ModelHolder:
    def __init__(self, ckpt):
        self.ck = torch.load(ckpt, map_location=dev, weights_only=False)
        self.model = None

    def add(self, geo):
        cfg = self.ck['cfg']
        if self.model is None:
            self.model = MD.build(cfg['model'], [geo], **cfg.get('model_args', {})).to(dev)
            (MD.load_compat(self.model, self.ck['model']) if hasattr(MD, 'load_compat') else self.model.load_state_dict(self.ck['model'], strict=False))
            self.model.eval()
        else:
            self.model.add_geo(geo)
        return self.model


def per_cell(case, holder, log):
    rec = dict(case=case)
    f0 = free_gb()
    C, rec['setup_s'] = timed(lambda: TE.Cell(case, BODY, log=lambda s_: None))
    _, rec['moments_s'] = timed(lambda: C.moments(C.taus0))
    _, rec['assemble_s'] = timed(lambda: C.assemble())
    rec.update(dofs=int(C.nb), ports=int(C.np_), interior=int(C.ni), elements=int(len(C.cells)), K_nnz_upper=int(C.vals.numel()))
    rec['K_GB'] = (C.vals.numel() * (8 + 4 + 8 + 4 + 4 + 4)) / 2 ** 30                 # vals, cu, ru, Ut vals, col_t, tperm
    Q = directions(C, 64)
    # ---------------------------------------------------------------- exact route
    for prec, fp32 in (('fp64', False), ('fp32', True)):
        fb = free_gb()
        _, rec[f'factor_{prec}_s'] = timed(lambda: C.factor(neumann=False, fp32=fp32))
        rec[f'factor_{prec}_GB'] = fb - free_gb()
        for B in (1, 16, 64):
            _, rec[f'exact_{prec}_Sq_B{B}_s'] = timed(lambda: C.apply(Q[:, :B]), reps=3, warm=1)
        if prec == 'fp64':
            SQ = C.apply(Q)
            e_ex = (Q * SQ).sum(0)
        C._free()
    # ---------------------------------------------------------------- learned route
    d = TMP / case
    _, rec['netdata_s'] = timed(lambda: netdata(C, BODY, d))
    geo, rec['geo_s'] = timed(lambda: TL.Geo(case, BODY, TMP, neumann=False, log=lambda s_: None, cell=C, load_banks=False))
    fb = free_gb()
    model, rec['model_cache_s'] = timed(lambda: holder.add(geo))
    fast, rec['freeze_s'] = timed(lambda: FN.FastNet(model, geo))
    rec['learned_state_GB'] = fb - free_gb()
    for B in (1, 16, 64):
        _, rec[f'learned_Sq_B{B}_s'] = timed(lambda: fast.s_hat(Q[:, :B]), reps=3, warm=1)
    e_hat = (Q * fast.s_hat(Q).to(dt)).sum(0)
    ratio = (e_hat / e_ex).cpu().numpy()
    rec['energy_ratio'] = dict(mean=float(ratio.mean()), min=float(ratio.min()), max=float(ratio.max()))
    try:
        import cert as CE
        ce = CE.from_geo(geo)
        U = fast.field(Q[:, :16]).to(dt)
        for m in (0, 8):
            _, rec[f'cert_m{m}_B16_s'] = timed(lambda: ce.lower(U, m), reps=3, warm=1)
    except Exception as e:
        rec['cert_error'] = repr(e)[:200]
    rec['total_free_GB_used'] = f0 - free_gb()
    log(json.dumps(dict(event='CELL', **rec)))
    model.caches.pop(case, None)
    del fast, geo, C, Q
    gc.collect(); torch.cuda.empty_cache()
    return rec


class SparseExactOp:
    """Exact cell operator without the dense port matrix: S q = (K E q)_P through the cell's interior factor."""

    def __init__(self, C):
        self.C = C
        if C.sol_I is None:
            C.factor(neumann=False, fp32=True)

    def apply(self, q):
        return self.C.apply(q)

    def field(self, q):
        return self.C.extend(q)


def cg(lat, ops, prec, tol=1e-8, maxit=3000):
    X = torch.zeros_like(lat.F); R = lat.F.clone()
    Z = prec(R); P = Z.clone()
    rz = (R * Z).sum(0); r0 = R.norm(dim=0)
    sync(); t = time.perf_counter(); it = 0
    for it in range(1, maxit + 1):
        AP = lat.matvec(ops, P)
        alpha = rz / (P * AP).sum(0)
        X += alpha * P; R -= alpha * AP
        if (R.norm(dim=0) / r0).max() < tol:
            break
        Z = prec(R)
        rz_new = (R * Z).sum(0)
        P = Z + (rz_new / rz) * P; rz = rz_new
    sync()
    return X, it, time.perf_counter() - t, float((R.norm(dim=0) / r0).max())


def lattice(case, holder, log):
    import re
    fam = re.match(r'^(fresh_[a-z]+_\d{4})_', case).group(1)
    nbr = fam + '_full' if not case.endswith('_full') else case
    os.environ['LAT_LOADS'] = 'consistent'; os.environ['LAT_CPU'] = '1'
    t = time.perf_counter()
    lat = LT.build(case, nbr, 'x', BODY, log=lambda s_: None)
    ref = lat.reference()
    rec = dict(case=case, nbr=nbr, free_dofs=int(len(lat.free)), loads=int(lat.F.shape[1]), reference_s=time.perf_counter() - t)
    # deployable Jacobi: inverse diagonal of the assembled port stiffness (sum over the cells sharing a DOF)
    dg = torch.zeros(len(lat.free), dtype=dt, device=dev)
    for i, cd in enumerate(lat.cells):
        C = cd['cell']
        lat.scatter_add(dg[:, None], C.dK[C.P][:, None], i)
    jac = 1 / dg
    precs = dict(ideal=lat._psolve, none=lambda R: R, jacobi=lambda R: jac[:, None] * R)
    # learned operators on both cells (deployment)
    geos, fasts = [], []
    for cd in lat.cells:
        C = cd['cell']
        netdata(C, BODY, TMP / C.case)
        g = TL.Geo(C.case, BODY, TMP, neumann=False, log=lambda s_: None, cell=C, load_banks=False)
        m = holder.add(g)
        geos.append(g); fasts.append(FN.FastNet(m, g))
    learned = [EN.FastOp(f) for f in fasts]
    rec['runs'] = []
    for oname in ('learned', 'exact_dd'):
        # exact factors only after the learned runs: lattice3._measure frees a cell's factor after its sensitivities
        ops = learned if oname == 'learned' else [SparseExactOp(cd['cell']) for cd in lat.cells]
        for pname, pr in precs.items():
            X, it, sec, res = cg(lat, ops, pr)
            out = lat._measure(ops, X) if oname == 'learned' else None
            row = dict(operators=oname, preconditioner=pname, iterations=it, seconds=sec, s_per_iter=sec / max(it, 1), residual=res)
            if out is not None:
                cmp_ = lat.compare(dict(out, pcg_iterations=it, pcg_residual=res, seconds=sec))
                row.update(gate_compliance_max=cmp_['gate_compliance_max'], gate_sens_max=cmp_['gate_sens_max'])
            else:
                c = (lat.F * X).sum(0).cpu().numpy()
                row['compliance_rel_err_max'] = float(np.max(np.abs(c - ref['compliance']) / np.abs(ref['compliance'])))
            rec['runs'].append(row)
            log(json.dumps(dict(event='LATTICE_RUN', case=case, **row)))
    for cd in lat.cells:
        cd['cell']._free()
    return rec


def main(argv):
    ckpt, out, cases = argv[0], argv[1], argv[2:]
    TMP.mkdir(parents=True, exist_ok=True)
    holder = ModelHolder(ckpt)
    log = lambda s_: print(s_, flush=True)
    rec = dict(ckpt=ckpt, gpu=torch.cuda.get_device_name(0), cells=[], lattice=None)
    for c in cases:
        try:
            rec['cells'].append(per_cell(c, holder, log))
        except Exception as e:
            import traceback
            log(json.dumps(dict(event='CELL_FAIL', case=c, error=repr(e)[:300], trace=traceback.format_exc()[-1200:])))
        Path(out).write_text(json.dumps(rec, indent=1))
    try:
        rec['lattice'] = lattice(cases[0], holder, log)
    except Exception as e:
        import traceback
        log(json.dumps(dict(event='LATTICE_FAIL', error=repr(e)[:300], trace=traceback.format_exc()[-1500:])))
    Path(out).write_text(json.dumps(rec, indent=1))
    shutil.rmtree(TMP, ignore_errors=True)


if __name__ == '__main__':
    main(sys.argv[1:])
