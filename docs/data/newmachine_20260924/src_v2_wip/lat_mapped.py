"""Direction A, step 0c: lattices of affinely mapped cells, fine-scale exact vs learned vs homogenised (new script; default
paths unchanged; runs on the GPU server; P1 bodies are only read).

One cell kind (an uncut periodic Schwarz-P cell of uniform thickness, homog bodies) under one constant map A (x = A X),
repeated as an nx x ny x nz block (lat_multi.MultiLattice: ports glued by reference grid key, which is the physical
neighbour relation for a constant A). Clamp y = min (non-private DOFs fixed), traction-consistent unit loads x / y / z
on y = max (lattice3 weights on the reference face; for a diagonal A the physical weights differ by one constant per face
and the loads are normalised to unit total force, so they are the physical uniform tractions) + n_random random loads.
Cell operators are DENSE port matrices (one per cell kind, shared by all cells, applied as one GEMM per PCG step):
  exact      S~ = K_PP - K_PI K_II^-1 K_IP (interior factor, fp64), the fine-scale reference
  learned    S_hat = E_hat^T K~ E_hat for the fields of a0_budget (V0R + co-rotated Jacobi; cN = N correction cycles,
             suffix w = + weak-node patch), assembled column by column through the autograd adjoint (Geo.s_hat_apply)
PCG (lat_precond, --prec) to --tol on every load. Per lattice and field: relative compliance error on every load, relative
solution distance, iterations. The dense form measures ACCURACY only (deployment cost: P1 benchmarks, matrix-free).
Homogenised macro model (diagonal A only, so the lattice is a box): Q1 hexahedra on the physical box (--m elements per cell
and axis), constant anisotropic tensor, same clamp, uniform unit traction on the load face; tensors
  homog_exact  periodic cell problem on the mapped cell (homog_mapped.periodic_CH), i.e. homogenisation done right
  homog_rot    reference tensor rotated by the polar factor (stretch ignored; = reference for diagonal A)
  homog_push   push-forward of the reference tensor by A
(reference tensor = this script's identity-map C^H if 'id' runs first, else --p1 homog_cells.json). Their compliance errors
against the exact lattice show the scale-separation / boundary-layer error of homogenisation at finite cell counts.
Usage: lat_mapped.py <out.json> <ckpt> <body_dir> <case> <maps.json[,...]> --maps id,stry2 --shapes 2x2x2,3x3x3
       [--fields c1,c2w,c4w] [--work <dir for NETDATA>] [--prec bnn:kpp:q1r] [--tol 1e-10] [--m 4,8 (+ Richardson
       from the last two when they differ by 2x)] [--p1 homog_cells.json]
       lat_mapped.py --selftest      (CPU: macro solver against the closed form for nu = 0)"""
import argparse, gc, json, sys, time
from pathlib import Path
import numpy as np


# ------------------------------------------------------------------------------------------------ macro model (CPU)
def _q1_grad(h):
    """(8 gauss points, 8 nodes, 3) physical gradients of the Q1 shape functions on a box element of size h (3,);
    node order (i, j, k) in {0,1}^3 as 4i + 2j + k; gauss weights all equal to prod(h) / 8."""
    g = np.array([-1, 1]) / np.sqrt(3)
    xi = np.array([[(a + 1) / 2 for a in (gx, gy, gz)] for gx in g for gy in g for gz in g])        # (8, 3) in [0, 1]^3
    nodes = np.array([[(c >> 2) & 1, (c >> 1) & 1, c & 1] for c in range(8)], float)
    G = np.zeros((8, 8, 3))
    for q in range(8):
        for a in range(8):
            f = [nodes[a, d] * xi[q, d] + (1 - nodes[a, d]) * (1 - xi[q, d]) for d in range(3)]
            s = [2 * nodes[a, d] - 1 for d in range(3)]
            G[q, a] = [s[0] * f[1] * f[2] / h[0], f[0] * s[1] * f[2] / h[1], f[0] * f[1] * s[2] / h[2]]
    return G, nodes.astype(int)


def _bmat(dN):
    """(6, 24) strain-displacement matrix, Voigt xx, yy, zz, yz, xz, xy with engineering shear (homog_mapped.VOIGT)."""
    B = np.zeros((6, 24))
    for a in range(8):
        dx, dy, dz = dN[a]
        B[0, 3 * a] = dx; B[1, 3 * a + 1] = dy; B[2, 3 * a + 2] = dz
        B[3, 3 * a + 1] = dz; B[3, 3 * a + 2] = dy
        B[4, 3 * a] = dz; B[4, 3 * a + 2] = dx
        B[5, 3 * a] = dy; B[5, 3 * a + 1] = dx
    return B


def macro_compliance(CH, L, N, clamp=('y', 'min'), load=('y', 'max')):
    """Compliances (3,) of the box [0, L] (N elements per axis, constant 6 x 6 tensor CH) clamped on one face, with a uniform
    traction of unit total force in x, y, z on the load face (consistent nodal forces)."""
    import scipy.sparse as sp
    AX = {'x': 0, 'y': 1, 'z': 2}
    L, N = np.asarray(L, float), np.asarray(N, int)
    h = L / N
    G, loc = _q1_grad(h)
    Ke = sum(_bmat(G[q]).T @ CH @ _bmat(G[q]) for q in range(8)) * np.prod(h) / 8
    M = N + 1
    nid = lambda i, j, k: (i * M[1] + j) * M[2] + k
    I, J, K = np.meshgrid(np.arange(N[0]), np.arange(N[1]), np.arange(N[2]), indexing='ij')
    I, J, K = I.ravel(), J.ravel(), K.ravel()
    en = np.stack([nid(I + loc[a, 0], J + loc[a, 1], K + loc[a, 2]) for a in range(8)], 1)        # (E, 8)
    ed = (3 * en[:, :, None] + np.arange(3)).reshape(len(en), 24)
    rows = np.repeat(ed, 24, 1).ravel(); cols = np.tile(ed, (1, 24)).ravel()
    ndof = 3 * int(M.prod())
    Kg = sp.csr_matrix((np.tile(Ke.ravel(), len(en)), (rows, cols)), shape=(ndof, ndof))
    gi = np.stack(np.unravel_index(np.arange(int(M.prod())), tuple(M)), 1)                          # node grid indices
    ca, la = AX[clamp[0]], AX[load[0]]
    fixed_n = gi[:, ca] == (0 if clamp[1] == 'min' else N[ca])
    on_load = gi[:, la] == (0 if load[1] == 'min' else N[la])
    w = np.ones(len(gi))                                                                            # trapezoid face weights
    for d in range(3):
        if d != la:
            w *= np.where((gi[:, d] == 0) | (gi[:, d] == N[d]), 0.5, 1.0) * h[d]
    w = np.where(on_load, w, 0.0); w /= w.sum()
    free = np.flatnonzero(np.repeat(~fixed_n, 3))
    Kf = Kg[free][:, free].tocsc()
    F = np.zeros((ndof, 3))
    for d in range(3):
        F[d::3, d] = w
    Ff = F[free]
    try:
        import pypardiso
        U = pypardiso.spsolve(Kf.tocsr(), Ff)
    except ImportError:
        import scipy.sparse.linalg as spla
        U = spla.splu(Kf).solve(Ff)
    return (Ff * U).sum(0), dict(dofs=len(free))


def selftest():
    """nu = 0 isotropic box, clamped y = min, unit force in y on y = max: no Poisson coupling, so the clamp does not
    restrain anything and the Q1 solution is exact: compliance = L_y / (E L_x L_z)."""
    E = 3.0
    CH = np.diag([E, E, E, E / 2, E / 2, E / 2])
    L = [1.5, 2.0, 0.75]
    c, _ = macro_compliance(CH, L, [3, 4, 2])
    exact = L[1] / (E * L[0] * L[2])
    print(json.dumps(dict(compliance_y=float(c[1]), exact=exact, rel=float(abs(c[1] / exact - 1)))))
    assert abs(c[1] / exact - 1) < 1e-10
    # mesh refinement of a shear load converges (monotone decrease of compliance for a conforming discretisation)
    cs = [macro_compliance(CH, L, [3 * r, 4 * r, 2 * r])[0][0] for r in (1, 2, 4)]
    print(json.dumps(dict(shear_compliance_refinement=[float(x) for x in cs])))
    assert cs[0] < cs[1] < cs[2]
    print('SELFTEST_OK')


# ------------------------------------------------------------------------------------------------ fine scale (GPU)
class DenseOp:
    """Cell operator held as a dense fp64 port matrix (shared by all cells of the kind: one GEMM per lattice matvec)."""

    def __init__(self, S):
        self.S = S

    def apply(self, q):
        return self.S @ q.to(self.S.dtype)


def _dense(apply, np_, chunk, dev, dt):
    import torch
    S = torch.empty((np_, np_), dtype=dt, device=dev)
    for j in range(0, np_, chunk):
        m = min(chunk, np_ - j)
        E = torch.zeros((np_, m), dtype=dt, device=dev)
        E[j + torch.arange(m, device=dev), torch.arange(m, device=dev)] = 1
        S[:, j:j + m] = apply(E)
    S = 0.5 * (S + S.T)
    return S


def main(argv):
    if argv and argv[0] == '--selftest':
        selftest(); return
    import torch
    import models as MD                                                   # first: applies OPL_CONV_FP32
    import trainlib as TL
    import a0_eval as AE
    import a0_budget as AB
    import mapped_cell as MC
    import corot_smooth as CR
    import weak_patch as WP
    import homog_mapped as HM
    import lat_multi as LM
    import lat_precond as PR
    dev, dt = AE.dev, AE.dt

    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('ckpt'); ap.add_argument('body'); ap.add_argument('case'); ap.add_argument('mapsjson')
    ap.add_argument('--maps', default='id,stry2'); ap.add_argument('--shapes', default='2x2x2,3x3x3')
    ap.add_argument('--fields', default='c1,c2w,c4w'); ap.add_argument('--work', default='/root/autodl-tmp/OPL/A0/work_lat')
    ap.add_argument('--prec', default='bnn:kpp:q1r'); ap.add_argument('--tol', type=float, default=1e-10)
    ap.add_argument('--maxit', type=int, default=3000); ap.add_argument('--n-random', type=int, default=2)
    ap.add_argument('--m', default='4,8'); ap.add_argument('--p1', default='')
    ap.add_argument('--wp_dil', type=int, default=2); ap.add_argument('--wp_seed', default='cutweakbox')
    ap.add_argument('--chunk', type=int, default=16, help='columns per learned adjoint application')
    ap.add_argument('--fast', type=int, default=0, help='learned dense operators through mapped_fast.MappedFastOp (64 columns)')
    a = ap.parse_args(argv)
    log = lambda d: print(json.dumps(d, default=float), flush=True)
    specs = {}
    for f in a.mapsjson.split(','):
        specs.update({m['name']: m['spec'] for m in json.loads(Path(f).read_text())})
    shapes = [tuple(int(v) for v in s.split('x')) for s in a.shapes.split(',')]
    fields = [f for f in a.fields.split(',') if f]
    ref = None
    if a.p1:
        for c in json.loads(Path(a.p1).read_text())['cells']:
            if c['case'] == a.case:
                ref = np.asarray(c['CH'])
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False); cfg = ck['cfg']
    model = None
    CR.install()
    res = dict(args=vars(a), conv=TL.conv_precision(), maps={})
    for mname in a.maps.split(','):
        spec = specs[mname]
        A = np.eye(3) if spec['kind'] == 'identity' else np.asarray(spec['A'], float)
        rec = res['maps'][mname] = dict(A=A.tolist(), lattices={})
        t0 = time.perf_counter()
        C = MC.MappedCell(a.case, a.body, spec, log=lambda s_: None); C.assemble()
        rec['map_stats'] = C.map_stats; rec['ports'] = int(C.np_); rec['dofs'] = int(C.nb)
        # homogenised tensors
        CH, _ = HM.periodic_CH(C); CH = CH / np.linalg.det(A)
        if mname == 'id' and ref is None:
            ref = CH
        U_, S_, Vh = np.linalg.svd(A); R = U_ @ Vh
        tens = dict(homog_exact=CH)
        if ref is not None:
            tens.update(homog_rot=HM.rotate(ref, R), homog_push=HM.push(ref, A))
        rec['CH'] = {k: v.tolist() for k, v in tens.items()}
        rec['kappa'] = float(S_[0] / S_[-1])
        # lattices (consistent loads need the assembled cell) and the shared K_PP
        geom = LM.from_teacher(C); geom.kpp()
        lats = {s: LM.MultiLattice(LM.block_layout(s, geom), clamp=('y', 'min'), load=('y', 'max'), loads='consistent',
                                   n_random=a.n_random, device=dev, log=lambda s_: None) for s in shapes}
        kpps = {s: lat.assemble_kpp() for s, lat in lats.items()}
        for s, lat in lats.items():
            rec['lattices']['x'.join(map(str, s))] = dict(free_dofs=int(lat.nfree), labels=lat.labels, fields={})
        rec['setup_s'] = time.perf_counter() - t0

        def run(name, S):
            op = DenseOp(S)
            for s, lat in lats.items():
                ops = [op] * len(lat.geoms)
                fac = PR.Factory(lat, ops, shared={'kpp_triplets': kpps[s], 'kpp_triplets_s': 0.0}, kpp_backend='auto',
                                 log=lambda d: None)
                pc, st, _ = fac.build(a.prec)
                r = PR.solve(lat, ops, pc, tol=a.tol, maxit=a.maxit)
                X = r['X']
                row = dict(iterations=r['iterations'], seconds=r['seconds'], residual=r['residual'],
                           true_residual=PR.true_residual(lat, ops, X), compliance=(lat.F * X).sum(0).cpu().numpy().tolist())
                fac.free(); del pc
                lr = rec['lattices']['x'.join(map(str, s))]
                if name == 'exact':
                    lr['_X'] = X
                else:
                    ce = np.asarray(lr['fields']['exact']['compliance'])
                    row['compliance_rel_err'] = (np.asarray(row['compliance']) / ce - 1).tolist()
                    row['solution_rel_err'] = float((X - lr['_X']).norm() / lr['_X'].norm())
                lr['fields'][name] = row
                log(dict(event='LAT', map=mname, shape=s, field=name, it=row['iterations'],
                         err=row.get('compliance_rel_err'), res=row['true_residual']))
                del X, r
            gc.collect(); torch.cuda.empty_cache()

        # exact
        t = time.perf_counter()
        C.factor(neumann=False, fp32=False)
        S = _dense(C.apply, C.np_, 512, dev, dt)
        C._free()
        rec['exact_dense_s'] = time.perf_counter() - t
        run('exact', S)
        Sx = S; del S
        # learned
        d = Path(a.work) / mname
        AE.write_data(C, d / a.case, {}, (), a.body)
        g = AB.BudgetGeo(a.case, a.body, d, neumann=False, log=lambda s_: None, cell=C, load_banks=False)
        g.case = f'{a.case}@{mname}'
        if model is None:
            model = MD.build(cfg['model'], [g], **dict(cfg.get('model_args', {}))).to(dev)
            MD.load_compat(model, ck['model']); model.eval()
        else:
            model.add_geo(g)
        wrap = AE._Wrap(model)
        CR.set_corot(C, AE.nodal_rotations(C))
        rec['cell_eps'] = {}
        for fname in fields:
            base, cyc = AB.FIELDS[fname]
            if fname.endswith('w') and getattr(C, '_wp', 'unset') == 'unset':
                rec['weakpatch'] = WP.setup(C, g.nd, dil=a.wp_dil, seed=a.wp_seed)
            g.set_budget(base, cyc, wrap, wp=fname.endswith('w'))
            t = time.perf_counter()
            if a.fast:
                import mapped_fast as MF
                mop = MF.MappedFastOp(g, model, cyc, wrap, patch=fname.endswith('w'))
                Sh = _dense(mop.s_hat, C.np_, 64, dev, dt)
                del mop
            else:
                Sh = _dense(lambda E: g.s_hat_apply(model, E), C.np_, a.chunk, dev, dt)
            rec.setdefault('learned_dense_s', {})[fname] = time.perf_counter() - t
            # cell-level: lambda_max of (S_hat, S~) restricted away from rigid modes is not formed; record the
            # energy excess at the exact lattice traces instead (per lattice, mean over cells)
            run(fname, Sh)
            for s, lat in lats.items():
                lr = rec['lattices']['x'.join(map(str, s))]
                X = lr['_X']; ex, sx = [], []
                for i in range(len(lat.geoms)):
                    q = lat.gather(X, i)
                    ex.append((q * (Sh @ q)).sum(0)); sx.append((q * (Sx @ q)).sum(0))
                ex, sx = torch.stack(ex), torch.stack(sx)
                lr['fields'][fname]['cell_eps_max'] = (ex / sx.clamp_min(1e-300) - 1).max(0).values.cpu().numpy().tolist()
                lr['fields'][fname]['bound'] = ((ex - sx).sum(0) / sx.sum(0)).cpu().numpy().tolist()
            del Sh; gc.collect(); torch.cuda.empty_cache()
        model.caches.pop(g.case, None)
        # homogenised macro model (diagonal maps: the lattice is a box)
        if np.allclose(A, np.diag(np.diag(A))):
            for s, lat in lats.items():
                lr = rec['lattices']['x'.join(map(str, s))]
                ce = np.asarray(lr['fields']['exact']['compliance'][:3])
                L = np.asarray(s) * np.diag(A)
                ms = sorted(int(v) for v in a.m.split(','))
                for k, T in tens.items():
                    cs = []
                    for m in ms:
                        t = time.perf_counter()
                        c, info = macro_compliance(T, L, np.asarray(s) * m)
                        cs.append(c)
                        lr['fields'][f'{k}_m{m}'] = dict(compliance=c.tolist(), compliance_rel_err=(c / ce - 1).tolist(),
                                                         seconds=time.perf_counter() - t, **info)
                        log(dict(event='MACRO', map=mname, shape=s, field=f'{k}_m{m}', err=(c / ce - 1).tolist()))
                    if len(ms) >= 2 and ms[-1] == 2 * ms[-2]:                  # Q1 energy error O(h^2): Richardson
                        c = cs[-1] + (cs[-1] - cs[-2]) / 3
                        lr['fields'][f'{k}_rich'] = dict(compliance=c.tolist(), compliance_rel_err=(c / ce - 1).tolist())
                        log(dict(event='MACRO', map=mname, shape=s, field=f'{k}_rich', err=(c / ce - 1).tolist()))
        for lr in rec['lattices'].values():
            lr.pop('_X', None)
        del Sx, lats, kpps, geom, g
        try:
            WP.free(C)
        except Exception:
            pass
        C._free(); del C; gc.collect(); torch.cuda.empty_cache()
        Path(a.out).write_text(json.dumps(res, indent=1, default=float))
    print('DONE', flush=True)


if __name__ == '__main__':
    main(sys.argv[1:])
