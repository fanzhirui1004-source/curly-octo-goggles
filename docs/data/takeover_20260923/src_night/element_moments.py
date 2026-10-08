"""J4a: what exactly is a cut element's stiffness, and can it be written with nonnegative weights? CPU only.

For every active cell of one case:
 1. recover the 125 tensor moments M_abc = int xi^a eta^b zeta^c dx (a,b,c <= 4) from the archived
    V = int N_i N_j dx (Q2 Lagrange basis built on the element's own 27 node coordinates, so no node-order
    assumption), and check them against the archived 27 moments (a,b,c <= 2);
 2. rebuild K_e = sum_m M_m T_m with fixed templates T_m from isotropic elasticity (E, nu from the packet)
    and compare with the archived element_K;
 3. fit nonnegative weights w >= 0 on a fixed Gauss-Lobatto grid of p^3 points so that the grid reproduces the
    125 moments (NNLS); a positive grid measure gives K_fit = sum_q w_q B_q^T C B_q, symmetric PSD with the
    exact rigid kernel by construction; report moment residual and the element energy-norm deviation
    max |eig(K_e^+/2 (K_fit - K_e) K_e^+/2)|.
Archive members are read straight from the tar (sha256-checked); nothing archived is modified.
"""
import argparse, hashlib, io, json, tarfile, time
from itertools import product
from pathlib import Path
import numpy as np
from numpy.polynomial import legendre
from scipy.optimize import nnls

ROOT = Path('/root/autodl-tmp/CUTFEM_FRESH_GP_20260921')
L1D = {-1: np.array([0., -.5, .5]), 0: np.array([1., 0., -1.]), 1: np.array([0., .5, .5])}  # coefficients of 1, x, x^2
DL1D = {k: np.array([v[1], 2 * v[2]]) for k, v in L1D.items()}  # derivative: coefficients of 1, x


def members(case, names):
    src = ROOT / 'source_archives' / case
    inv = json.loads((src / 'INVENTORY.json').read_text())
    g = next(s for s in inv['stages'] if s.endswith('_G'))
    out = {}
    with tarfile.open(src / 'SOURCE.tar', 'r:') as t:
        for n in names:
            name = f'{g}/body/{n}'
            raw = t.extractfile(t.getmember(name)).read()
            if hashlib.sha256(raw).hexdigest() != inv['files'][name]['sha256']:
                raise ValueError('ARCHIVE_SHA:' + name)
            out[n] = np.load(io.BytesIO(raw))
    return out


def local_coordinates(case, n, arrays):
    nodes = arrays['NODES.npy']; dofs = arrays['dofs.npy']; cells = arrays['CELL_INDICES.npy']
    if not (np.all(dofs % 3 == np.tile(np.arange(3), 27)) and np.all(dofs.reshape(-1, 27, 3) // 3 == (dofs.reshape(-1, 27, 3)[:, :, :1] // 3))):
        raise ValueError('ELEMENT_DOFS_NOT_NODE_MAJOR')
    body_node = dofs[:, ::3] // 3
    grid = np.stack(np.unravel_index(nodes[body_node], (2 * n + 1,) * 3), axis=-1)  # (E,27,3) in half-cell units
    xi = grid - 2 * cells[:, None, :] - 1  # in {-1,0,1}
    if not np.all(np.isin(xi, (-1, 0, 1))) or not all(len({tuple(r) for r in e}) == 27 for e in xi[:50]):
        raise ValueError('ELEMENT_NODES_NOT_THE_27_Q2_NODES')
    return xi


def poly_product(a, b):
    return np.convolve(a, b)


def pattern_operators(xi_nodes, lam, mu, n):
    """For one node ordering: V-coefficients (729,125) and stiffness templates (125,81,81)."""
    N = [[L1D[int(xi_nodes[i, d])] for d in range(3)] for i in range(27)]
    DN = [[DL1D[int(xi_nodes[i, d])] for d in range(3)] for i in range(27)]
    Lv = np.zeros((27, 27, 5, 5, 5))
    for i in range(27):
        for j in range(27):
            px, py, pz = (poly_product(N[i][d], N[j][d]) for d in range(3))
            Lv[i, j, :len(px), :len(py), :len(pz)] = np.einsum('a,b,c->abc', px, py, pz)
    # grad N_i . e_d in xi units times (2n) for physical derivative
    def dpoly(i, d):
        out = []
        for e in range(3):
            out.append(DN[i][e] if e == d else N[i][e])
        return out
    G = np.zeros((27, 3, 27, 3, 5, 5, 5))  # int dN_i/dx_d dN_j/dx_e -> monomial coefficients
    for i in range(27):
        for d in range(3):
            pi = dpoly(i, d)
            for j in range(27):
                for e in range(3):
                    pj = dpoly(j, e)
                    p = [poly_product(pi[k], pj[k]) for k in range(3)]
                    G[i, d, j, e, :len(p[0]), :len(p[1]), :len(p[2])] = np.einsum('a,b,c->abc', *p) * (2 * n) ** 2
    T = np.zeros((27, 3, 27, 3, 5, 5, 5))
    for p_, q_ in product(range(3), repeat=2):
        # C_pdqe = lam d_pd d_qe + mu (d_pq d_de + d_pe d_dq)
        T[:, p_, :, q_] += lam * G[:, p_, :, q_]
        if p_ == q_:
            T[:, p_, :, q_] += mu * sum(G[:, d, :, d] for d in range(3))
        T[:, p_, :, q_] += mu * G[:, q_, :, p_]
    return Lv.reshape(729, 125), T.reshape(81, 81, 125).transpose(2, 0, 1)


def gll(p):
    x = np.r_[-1, legendre.Legendre.basis(p - 1).deriv().roots(), 1]
    return np.sort(x)


def main(a):
    t0 = time.perf_counter()
    out = Path(a.output); out.mkdir(parents=True, exist_ok=False)
    ctx = json.loads((ROOT / 'packets' / a.case / 'FRESH_CONTEXT.json').read_text())
    n = int(ctx['n']); E = float(ctx['material']['E']); nu = float(ctx['material']['nu'])
    lam = E * nu / ((1 + nu) * (1 - 2 * nu)); mu = E / (2 * (1 + nu))
    arr = members(a.case, ['NODES.npy', 'dofs.npy', 'CELL_INDICES.npy', 'V_data.npy', 'cell_moments.npy', 'element_K.npy'])
    xi = local_coordinates(a.case, n, arr)
    # V_data is the reduced row factor of the volume Gram (stage_cutfem_gp.shift_body.factors): Gram = V^T V.
    V = np.einsum('eki,ekj->eij', arr['V_data.npy'], arr['V_data.npy']); M27 = arr['cell_moments.npy']; K = arr['element_K.npy']
    ne = len(V)
    keys, inverse = np.unique(xi.reshape(ne, -1), axis=0, return_inverse=True)
    inverse = inverse.ravel()
    M = np.empty((ne, 125)); vres = np.empty(ne); kres = np.empty(ne)
    ops = {}
    for k in range(len(keys)):
        Lv, T = pattern_operators(keys[k].reshape(27, 3), lam, mu, n)
        sel = np.flatnonzero(inverse == k)
        sol, *_ = np.linalg.lstsq(Lv, V[sel].reshape(len(sel), -1).T, rcond=None)
        M[sel] = sol.T
        vres[sel] = np.linalg.norm(Lv @ sol - V[sel].reshape(len(sel), -1).T, axis=0) / np.linalg.norm(V[sel].reshape(len(sel), -1), axis=1)
        Kr = np.einsum('em,mij->eij', M[sel], T)
        kres[sel] = np.linalg.norm((Kr - K[sel]).reshape(len(sel), -1), axis=1) / np.linalg.norm(K[sel].reshape(len(sel), -1), axis=1)
        ops[k] = T
    idx27 = [a_ * 25 + b_ * 5 + c_ for a_, b_, c_ in product(range(3), repeat=3)]
    m27 = np.linalg.norm(M[:, idx27] - M27, axis=1) / np.linalg.norm(M27, axis=1)
    vol = M[:, 0]; full = vol * (n ** 3) > 1 - 1e-12
    rec = dict(case=a.case, elements=int(ne), node_orderings=int(len(keys)), full_cells=int(full.sum()),
               v_residual_max=float(vres.max()), moments27_residual_max=float(m27.max()),
               K_template_residual_median=float(np.median(kres)), K_template_residual_max=float(kres.max()),
               volume_fraction_quantiles=np.quantile(vol * n ** 3, [0, .01, .1, .5, .9, 1]).tolist())
    print(json.dumps(dict(event='MOMENTS', **rec, seconds=time.perf_counter() - t0)), flush=True)
    np.savez(out / 'MOMENTS125.npz', moments=M, cells=arr['CELL_INDICES.npy'], v_residual=vres, K_residual=kres,
             ordering=inverse, orderings=keys)
    # NNLS positive-grid representability
    rng = np.random.default_rng(1)
    sample = np.arange(ne) if ne <= a.max_elements else np.sort(rng.choice(ne, a.max_elements, replace=False))
    rec['nnls'] = {}
    for p in a.orders:
        x = gll(p); pts = np.array(list(product(x, x, x)))
        mono = np.array([pts[:, 0] ** i * pts[:, 1] ** j * pts[:, 2] ** k for i, j, k in product(range(5), repeat=3)])  # (125, P)
        mres = np.empty(len(sample)); kdev = np.empty(len(sample)); t = time.perf_counter()
        W = np.zeros((len(sample), len(pts)))
        for s, e in enumerate(sample):
            scale = M[e, 0]
            w, _ = nnls(mono / 1.0, M[e] / scale, maxiter=50 * len(pts))
            W[s] = w * scale
            mres[s] = np.linalg.norm(mono @ w - M[e] / scale) / np.linalg.norm(M[e] / scale)
            Kf = np.einsum('m,mij->ij', mono @ W[s], ops[inverse[e]])
            ev, U = np.linalg.eigh(K[e]); keep = ev > ev.max() * 1e-12
            Wh = U[:, keep] / np.sqrt(ev[keep])
            kdev[s] = np.abs(np.linalg.eigvalsh(Wh.T @ (Kf - K[e]) @ Wh)).max()
        r = dict(points=int(len(pts)), moment_residual_median=float(np.median(mres)), moment_residual_max=float(np.max(mres)),
                 energy_deviation_median=float(np.median(kdev)), energy_deviation_q99=float(np.quantile(kdev, .99)),
                 energy_deviation_max=float(kdev.max()), below_1e_6=float(np.mean(kdev < 1e-6)),
                 below_1e_3=float(np.mean(kdev < 1e-3)), seconds=time.perf_counter() - t)
        rec['nnls'][p] = r
        print(json.dumps(dict(event='NNLS', order=p, **r)), flush=True)
        np.savez(out / f'NNLS_GLL{p}.npz', sample=sample, weights=W, moment_residual=mres, energy_deviation=kdev)
    rec['seconds'] = time.perf_counter() - t0
    (out / 'RESULT.json').write_text(json.dumps(rec, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--case', required=True); p.add_argument('--output', required=True)
    p.add_argument('--orders', type=int, nargs='*', default=[5, 6, 8])
    p.add_argument('--max-elements', type=int, default=1500)
    main(p.parse_args())
