"""PIML-style condensation in the continuous-neighbour lattice gate (no network: exact interior extension, so only the
boundary-kinematics assumption is measured). Every cell's box-face port displacement is restricted to a polynomial
space of order p over the cell (tensor Bernstein, control points shared between neighbours on common faces; p = 1 is
the EMsFEM / PIML linear boundary assumption on 8 corner nodes, p = 3 a cubic (Bezier) boundary), cut-band ports stay
free (generous to the baseline). The lattice is solved by Galerkin on that space: K_c = G^T K G with the exact lattice
stiffness K (assembled from the exact cell Schur complements), c = K_c^-1 G^T F, U = G c; compliance and 8-corner
sensitivities (exact extension of the constrained trace) are compared with the exact lattice as in lat_full.
Usage: piml_gate.py <out.json> <case>[,...] [--configs x,y] [--orders 1,2,3,5,8]   (OPL_DEV=cpu LAT_CPU=1 recommended;
needs the cached dense port operators <body>/<case>_portview/T64.npy from earlier gate runs)"""
import diag_sens as DS                                                   # first: CPU env when OPL_DEV=cpu
import sys, os, json, time, argparse
from math import comb
from pathlib import Path
import numpy as np
import torch
import trainlib as TL
import teacher as TE
import lattice3 as LT
import ops as OP

dt = TL.dt
if TL.dev.type == 'cpu':                                                 # exact interior factor by PARDISO on the host
    TE.Cell.factor = lambda self, neumann=True, fp32=False, **kw: DS.cpu_factor(self)
    import box_encode as BX                                              # ghost faces for bodies without a GP cache
    BX.dev = TL.dev


def bern(p, t):
    return np.stack([comb(p, i) * t ** i * (1 - t) ** (p - i) for i in range(p + 1)], -1)


def build_G(lat, p, mode='all'):
    """(n_free x n_ctrl) map from global control coefficients to the lattice free DOFs; cut-band DOFs identity.
    mode 'all': every box-face DOF of every cell on the polynomial space (a PIML-style lattice); 'interface': only the
    DOFs on the shared test/neighbour face, every other DOF free (isolates the interface-kinematics assumption).
    A lattice DOF shared by both cells gets its row once (the two cells' polynomials coincide there)."""
    n2 = 2 * lat.n
    rows, cols, vals = [], [], []
    ckey = {}
    clamped_ctrl = set()
    seen = set()
    poly_rows = set()
    for ci, ((gg, comp, priv), k) in enumerate(zip(lat.pos, lat.idx)):
        off = 2 * lat.n * np.asarray(lat.cells[ci]['offset'])
        f = lat.fmap[k]
        loc = (gg - off) / n2                                            # local coordinates in [0, 1]
        box = ~priv
        if mode == 'interface':
            ax = 'xy'.index(lat.config)
            box = box & (gg[:, ax] == 0)                                  # the shared face (test x_ax = 0 = neighbour's far side)
        W = [bern(p, np.clip(loc[:, a], 0, 1)) for a in range(3)]
        for i in range(p + 1):
            for j in range(p + 1):
                for l in range(p + 1):
                    w = W[0][:, i] * W[1][:, j] * W[2][:, l]
                    sel = np.flatnonzero(box & (np.abs(w) > 1e-14))
                    if not len(sel):
                        continue
                    gcp = (int(off[0] // n2 * p + i), int(off[1] // n2 * p + j), int(off[2] // n2 * p + l))
                    for r in sel:
                        key = gcp + (int(comp[r]),)
                        if key not in ckey:
                            ckey[key] = len(ckey)
                        if f[r] >= 0:
                            if (f[r], ckey[key]) not in seen:             # shared DOF: one row entry, not one per cell
                                seen.add((f[r], ckey[key])); rows.append(f[r]); cols.append(ckey[key]); vals.append(w[r])
                        else:
                            clamped_ctrl.add(ckey[key])                   # control point touching the clamped face
        poly_rows.update(int(x) for x in f[box & (f >= 0)])
        pr = np.flatnonzero((f >= 0) & (priv | ~box))                       # cut-band (and, in interface mode, off-face) DOFs: free
        for r in pr:
            key = ('free', int(f[r]))
            if key in ckey:
                continue
            ckey[key] = len(ckey); rows.append(f[r]); cols.append(ckey[key]); vals.append(1.0)
    import scipy.sparse as sp
    drop = {ckey[k] for k in ckey if isinstance(k, tuple) and len(k) == 2 and k[0] == 'free' and k[1] in poly_rows}
    if drop:
        keep_e = [i for i, c_ in enumerate(cols) if c_ not in drop]
        rows = [rows[i] for i in keep_e]; cols = [cols[i] for i in keep_e]; vals = [vals[i] for i in keep_e]
    G = sp.coo_matrix((vals, (rows, cols)), shape=(len(lat.free), len(ckey))).tocsr()
    # a control point with nonzero weight on a clamped node must vanish (polynomial = 0 on the clamped face)
    keep = np.setdiff1d(np.arange(len(ckey)), np.asarray(sorted(clamped_ctrl | drop), dtype=int))
    G = G[:, keep].tocsc()
    G.sum_duplicates()
    names = {v: k for k, v in ckey.items()}
    is_free = np.array([isinstance(names[c], tuple) and names[c][0] == 'free' for c in keep])
    other_rows = np.array([names[c][1] for c in keep[is_free]], dtype=np.int64)
    Bcols = np.flatnonzero(~is_free)
    Gp = G[:, Bcols].tocsr()
    poly_rows = np.flatnonzero(np.diff(Gp.indptr) > 0)
    if len(np.intersect1d(poly_rows, other_rows)) or len(poly_rows) + len(other_rows) != G.shape[0]:
        raise AssertionError('G split: polynomial and free rows must partition the free DOFs')
    return G, other_rows, poly_rows, Gp[poly_rows]


def lattice_K(lat, inv=None):
    """Dense lattice stiffness on the free DOFs, rows/columns in the order inv (inv[free index] = position)."""
    nf = len(lat.free)
    K = torch.zeros((nf, nf), dtype=dt)
    inv = torch.arange(nf) if inv is None else torch.as_tensor(inv)
    for i, cd in enumerate(lat.cells):
        f = lat.gather_idx[i].cpu(); keep = torch.nonzero(f >= 0).squeeze(1); fc = inv[f[keep]]
        T = cd['T']
        for r0 in range(0, len(keep), 2048):
            rows = keep[r0:r0 + 2048]
            K.index_put_((inv[f[rows]][:, None], fc[None, :]), T[rows][:, keep].to(dt), accumulate=True)
    return K


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('cases'); ap.add_argument('--configs', default='x,y')
    ap.add_argument('--orders', default='1,2,3,5,8'); ap.add_argument('--mode', default='all'); ap.add_argument('--body', default='/root/autodl-tmp/OPL/S0')
    a = ap.parse_args(argv)
    os.environ['LAT_LOADS'] = 'consistent'
    rec = dict(orders=a.orders, mode=a.mode, results=[])
    for case in a.cases.split(','):
        for conf in a.configs.split(','):
            t0 = time.perf_counter()
            lat = LT.build(case, f'{case}_nbm{conf}', conf, a.body, log=lambda s_: None)
            orders = [int(x) for x in a.orders.split(',')]
            Gs = {p: build_G(lat, p, a.mode) for p in orders}
            other, poly = Gs[orders[0]][1], Gs[orders[0]][2]                   # the split depends on the mode only
            for p in orders:
                if not (np.array_equal(Gs[p][1], other) and np.array_equal(Gs[p][2], poly)):
                    raise AssertionError('split differs between orders')
            perm = np.concatenate([other, poly]); no = len(other); nf = len(perm)
            inv = np.empty(nf, dtype=np.int64); inv[perm] = np.arange(nf)
            K = lattice_K(lat, inv)                                           # [other | polynomial] order, one dense copy
            Fp = lat.F.cpu()[torch.as_tensor(perm)]
            blocks = {}
            for p in orders:
                B = torch.as_tensor(Gs[p][3].toarray(), dtype=dt)              # polynomial rows x control coefficients
                Aoc = K[:no, no:] @ B if no else torch.zeros((0, B.shape[1]), dtype=dt)
                Acc = B.T @ (K[no:, no:] @ B)
                rsum = np.asarray(Gs[p][0].sum(1)).ravel()
                if (rsum > 1 + 1e-9).any():
                    raise AssertionError(f'G_ROW_SUM>1: max {rsum.max()}')
                blocks[p] = (B, Aoc, Acc, int(Gs[p][0].shape[1]), int((np.abs(rsum - 1) > 1e-9).sum()))
            info = torch.empty((), dtype=torch.int32)
            torch.linalg.cholesky_ex(K, out=(K, info))                        # the leading block is chol(K_oo)
            if int(info) != 0:
                raise ValueError(f'LATTICE_CHOLESKY_INFO:{int(info)}')
            L = K.tril_(); del K
            unperm = lambda X: X[torch.as_tensor(inv)]
            U0 = unperm(torch.cholesky_solve(Fp, L))
            ref = lat._measure([OP.ExactOp(cd['cell'], cd['T']) for cd in lat.cells], U0.to(TL.dev))
            lat.ref = ref
            Loo = L[:no, :no]
            sols = {}
            for p in orders:
                B, Aoc, Acc, nctrl, nbad = blocks[p]
                Fo, Fi = Fp[:no], Fp[no:]
                if no:
                    Z = torch.cholesky_solve(Aoc, Loo); zF = torch.cholesky_solve(Fo, Loo)
                    Sc = Acc - Aoc.T @ Z; rhs = B.T @ Fi - Aoc.T @ zF
                else:
                    Sc, rhs = Acc, B.T @ Fi
                c = torch.linalg.solve(Sc, rhs)
                xo = (zF - Z @ c) if no else torch.zeros((0, Fp.shape[1]), dtype=dt)
                sols[p] = (unperm(torch.cat([xo, B @ c])), nctrl, nbad)
            del L, Loo, blocks
            ex = [OP.ExactOp(cd['cell'], cd['T']) for cd in lat.cells]
            for e in ex:
                e.keep_factor = False                                     # free each cell factor after use (memory)
            r = dict(case=case, config=conf, loads=lat.labels, gate=lat.gate.tolist(), free=int(len(lat.free)),
                     ref_compliance=np.asarray(ref['compliance']).tolist(), orders={})
            for p in orders:
                Up, nctrl, nbad = sols[p]
                res = lat._measure(ex, Up.to(TL.dev))
                cmp_ = lat.compare(res)
                g = np.asarray(lat.gate, bool)
                sv = np.asarray(cmp_['sens_vec_rel_err'])[0]
                ce = np.asarray(cmp_['compliance_rel_err'])
                tf = [i for i, l in enumerate(lat.labels) if l.startswith('test_face')]
                r['orders'][str(p)] = dict(ctrl_dofs=nctrl, compliance_rel_err=ce.tolist(), sens_vec_rel_err_test=sv.tolist(),
                                           gate_compliance_max=float(ce[g].max()), gate_sens_max=float(np.nanmax(sv[g])),
                                           test_face_sens_max=float(np.nanmax(sv[tf])), test_face_compliance_max=float(ce[tf].max()))
                print(json.dumps(dict(mode=a.mode, rowsum_lt1=nbad, case=case, config=conf, p=p, ctrl=nctrl, free=int(len(lat.free)),
                                      comp_max=round(100 * float(ce[g].max()), 2), sens_max=round(100 * float(np.nanmax(sv[g])), 2),
                                      test_face_comp=round(100 * float(ce[tf].max()), 2), test_face_sens=round(100 * float(np.nanmax(sv[tf])), 2))),
                      flush=True)
            r['seconds'] = time.perf_counter() - t0
            rec['results'].append(r)
            Path(a.out).write_text(json.dumps(rec, indent=1))
            del lat, sols; LT._CACHE.clear()
            import gc; gc.collect()


if __name__ == '__main__':
    main(sys.argv[1:])
