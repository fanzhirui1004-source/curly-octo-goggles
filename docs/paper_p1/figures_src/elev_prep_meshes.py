"""Data preparation for Figure 1 (F00_problem) and Figure F14_plate_design3d: surface meshes of the plate supported on its cut
(Section 5.10) and of its cut cell (3, 2, 0), written to figures_src/data_elev/ (the figure scripts only read these files).

Geometry (Eq. (1)), in internal lattice coordinates x in [0, 4] (short side), y in [0, 8] (long side), z in [0, 1]:
  material  |phi(x)| <= tau(x),  phi = sum_a cos(2 pi x_a)  (periodic, so one global field serves every cell),
            tau trilinear in each cell from the corner thickness parameters at the lattice vertices (shared by the cells
            meeting there; design vector 'tv' of history.jsonl on the vertices 'vkeys' of meta.json),
  cut       n . x <= b_global (plate841.json), the material beyond it is removed; the eight cells beyond the cut are absent.
Surfaces by marching cubes of g = max(|phi| - tau, n . x - b) on a grid of R points per cell axis, padded outside the plate so
that the walls are capped on the outer faces, then decimated (fast_simplification, quadric edge collapse); tau is evaluated
again at the decimated vertices, and faces are flagged as cut-plane section (all three vertices on the plane) or as box-face
section (all three on one outer face of the plate or of the cell).
Field of the cut cell: nodal displacements of the Q2 model (nodes on the 65^3 grid of half the element size, x-fastest index
order (x, y, z)) interpolated trilinearly on that grid at the surface vertices; written for the NICE field F_m B_m U_hat, the
exact field E_m B_m U and the NICE extension of the exact retained displacements.

Inputs (repository): review_r1/results/X6_final/{plate841.json, cplateN/meta.json, cplateN/history.jsonl};
data_elev/nodes_320_o000.npz (node ids, box-face and cut-band flags of cell (3, 2, 0) at the uniform start, from
body/plate841_320_o000 of the run), data_elev/cellfield_plate841_320_o023.npz and cellfield_summary.json (server script
data_elev/nice_cell_field.py).
Outputs: data_elev/plate_design.json, plate_k000.npz, plate_k023.npz, plate_ghost.npz, cell320_k000.npz, cell320_k023.npz.
Usage: python3 elev_prep_meshes.py        (needs scikit-image and fast-simplification)
"""
import json
from pathlib import Path
import numpy as np
from skimage import measure
import fast_simplification as FSIMP

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data_elev'
X6 = HERE.parent / 'review_r1' / 'results' / 'X6_final'
CELL = (3, 2, 0)                                                         # cut cell next to the clamp, closest to the load
CASE = 'plate841_320'


def design():
    L = json.loads((X6 / 'plate841.json').read_text())
    M = json.loads((X6 / 'cplateN' / 'meta.json').read_text())
    H = [json.loads(l) for l in open(X6 / 'cplateN' / 'history.jsonl')]
    h0, hk = [h for h in H if h['k'] == 0][-1], H[-1]
    assert hk['k'] == 23
    D = dict(shape=L['shape'], normal=L['normal'], b_global=L['b_global'],
             cells=[dict(position=c['position'], kind=c['kind'], retained=c['retained']) for c in L['cells']],
             vkeys=M['vkeys'], fixed=[bool(f) for f in M['fixed']], tv0=h0['tv'], tv_final=hk['tv'], k_final=hk['k'],
             C0=h0['C'], C_final=hk['C'], cell=list(CELL), case=CASE,
             source='review_r1/results/X6_final: plate841.json, cplateN/meta.json, cplateN/history.jsonl (k = 0 and k = 23)')
    (DATA / 'plate_design.json').write_text(json.dumps(D, indent=1))
    return D


def vertex_grid(D, tv):
    nx, ny, nz = D['shape']
    T = np.full((nx + 1, ny + 1, nz + 1), np.nan)
    for (i, j, k), v in zip(D['vkeys'], tv):
        T[i, j, k] = v
    # vertices of no cell (beyond the cut): nearest defined value, only so that interpolation is defined (masked anyway)
    I = np.argwhere(np.isnan(T)); V = np.argwhere(~np.isnan(T))
    for p in I:
        q = V[np.argmin(((V - p) ** 2).sum(1))]; T[tuple(p)] = T[tuple(q)]
    return T


def tau_at(P, T):
    """Trilinear interpolation of the vertex grid T at points P (N, 3) in lattice coordinates."""
    n = np.array(T.shape) - 1
    i0 = np.clip(np.floor(P).astype(int), 0, n - 1)
    f = np.clip(P - i0, 0.0, 1.0)
    out = np.zeros(len(P))
    for c in np.ndindex(2, 2, 2):
        w = np.prod(np.where(np.array(c) == 1, f, 1 - f), axis=1)
        out += w * T[i0[:, 0] + c[0], i0[:, 1] + c[1], i0[:, 2] + c[2]]
    return out


def phi(P):
    return np.cos(2 * np.pi * P).sum(-1)


def mc(G, h, origin):
    Gp = np.pad(G, 1, constant_values=50.0)                              # caps on the outer faces
    v, f, _, _ = measure.marching_cubes(Gp, level=0.0, spacing=(h,) * 3)
    v = v - h + np.asarray(origin)
    return v, f.astype(np.int64)


def decimate(v, f, target):
    if len(f) <= target:
        return v, f
    v2, f2 = FSIMP.simplify(v.astype(np.float64), f.astype(np.int64), target_reduction=1 - target / len(f))
    return np.asarray(v2), np.asarray(f2, np.int64)


def flags(v, f, n, b, lo, hi, tol):
    tri = v[f]
    on_cut = np.all(np.abs(tri @ n - b) < tol, axis=1)
    on_box = np.zeros(len(f), bool)
    for a in range(3):
        for w in (lo[a], hi[a]):
            on_box |= np.all(np.abs(tri[:, :, a] - w) < tol, axis=1)
    return on_cut & ~on_box, on_box


def plate_mesh(D, tv, R, target, keep=+1):
    """keep = +1: the plate; keep = -1: the material of the full 4 x 8 x 1 lattice beyond the cut (for context)."""
    nx, ny, nz = D['shape']
    n, b = np.asarray(D['normal']), D['b_global']
    T = vertex_grid(D, tv)
    xs, ys, zs = (np.linspace(0, m, m * R + 1) for m in (nx, ny, nz))
    h = xs[1] - xs[0]
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    P = np.stack([X, Y, Z], -1).reshape(-1, 3)
    G = (np.abs(phi(P)) - tau_at(P, T)).reshape(X.shape)
    G = np.maximum(G, keep * (n[0] * X + n[1] * Y + n[2] * Z - b))
    if keep > 0:                                                         # only the cells of the layout
        have = np.zeros((nx, ny, nz), bool)
        for c in D['cells']:
            have[tuple(c['position'])] = True
        ci = [np.clip(np.floor(A_ - 1e-12).astype(int), 0, m - 1) for A_, m in ((X, nx), (Y, ny), (Z, nz))]
        cj = [np.clip(np.floor(A_ + 1e-12).astype(int), 0, m - 1) for A_, m in ((X, nx), (Y, ny), (Z, nz))]
        inside = have[ci[0], ci[1], ci[2]] | have[cj[0], cj[1], cj[2]]
        G = np.where(inside, G, np.maximum(G, 1.0))
        # every cell meeting the kept half-space is in the layout (so the mask removes nothing the cut keeps)
        miss = [(i, j, k) for i in range(nx) for j in range(ny) for k in range(nz)
                if not have[i, j, k] and n[0] * i + n[1] * j + min(0.0, n[2]) < b - 1e-9]
        print('cells meeting the kept half-space but absent from the layout:', miss)
    v, f = mc(G, h, (0, 0, 0))
    v = np.clip(v, 0, [nx, ny, nz])
    nf0 = len(f)
    v, f = decimate(v, f, target)
    cut, box = flags(v, f, n, b, (0, 0, 0), (nx, ny, nz), 1.5e-3)
    print(f'plate keep={keep:+d} R={R}: {nf0} -> {len(f)} triangles, {cut.sum()} on the cut, {box.sum()} on outer faces')
    return dict(v=v.astype(np.float32), f=f.astype(np.int32), tau=tau_at(v, T).astype(np.float32), on_cut=cut, on_box=box)


def cell_mesh(D, tv, R, target, field=None):
    """Cut cell CELL in local coordinates [0, 1]^3; optional nodal fields interpolated at the vertices."""
    n, b = np.asarray(D['normal']), D['b_global'] - float(np.dot(D['normal'], CELL))
    T = vertex_grid(D, tv)
    g = np.linspace(0, 1, R + 1); h = g[1]
    X, Y, Z = np.meshgrid(g, g, g, indexing='ij')
    P = np.stack([X, Y, Z], -1).reshape(-1, 3)
    G = (np.abs(phi(P)) - tau_at(P + np.asarray(CELL), T)).reshape(X.shape)
    G = np.maximum(G, n[0] * X + n[1] * Y + n[2] * Z - b)
    v, f = mc(G, h, (0, 0, 0)); v = np.clip(v, 0, 1)
    nf0 = len(f)
    v, f = decimate(v, f, target)
    cut, box = flags(v, f, n, b, (0, 0, 0), (1, 1, 1), 1.5e-3)
    out = dict(v=v.astype(np.float32), f=f.astype(np.int32), tau=tau_at(v + np.asarray(CELL), T).astype(np.float32),
               on_cut=cut, on_box=box, normal=n, offset=b)
    # removed part of the cell (beyond the cut), for context
    Gr = np.maximum((np.abs(phi(P)) - tau_at(P + np.asarray(CELL), T)).reshape(X.shape), -(n[0] * X + n[1] * Y - b))
    vr, fr = mc(Gr, h, (0, 0, 0)); vr = np.clip(vr, 0, 1)
    vr, fr = decimate(vr, fr, target // 3)
    out.update(v_removed=vr.astype(np.float32), f_removed=fr.astype(np.int32))
    if field is not None:
        for k, U in field.items():
            if not k.startswith('_'):
                out[k] = interp_nodal(field['_nodes'], U, v)
    print(f'cell {CELL} R={R}: {nf0} -> {len(f)} triangles, {cut.sum()} on the cut, {box.sum()} on the box faces')
    return out


def interp_nodal(nodes, U, P):
    """Trilinear interpolation of nodal vectors U (N, 3) given on the 65^3 node grid (ids, (x, y, z) order) at P (local)."""
    m = 65
    A = np.full((m, m, m, U.shape[1]), np.nan)
    ijk = np.stack(np.unravel_index(nodes, (m,) * 3), 1)
    A[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = U
    s = P * (m - 1)
    i0 = np.clip(np.floor(s).astype(int), 0, m - 2); fr = np.clip(s - i0, 0, 1)
    num = np.zeros((len(P), U.shape[1])); den = np.zeros(len(P))
    for c in np.ndindex(2, 2, 2):
        w = np.prod(np.where(np.array(c) == 1, fr, 1 - fr), axis=1)
        a = A[i0[:, 0] + c[0], i0[:, 1] + c[1], i0[:, 2] + c[2]]
        ok = ~np.isnan(a[:, 0])
        num[ok] += w[ok, None] * a[ok]; den[ok] += w[ok]
    miss = den < 1e-12
    if miss.any():                                                       # no node around: nearest node
        from scipy.spatial import cKDTree
        tr = cKDTree(ijk / (m - 1)); _, j = tr.query(P[miss])
        num[miss] = U[j]; den[miss] = 1.0
    print(f'  interpolation: {int((den < .999).sum())} of {len(P)} vertices with incomplete node stencils, {int(miss.sum())} nearest')
    return (num / den[:, None]).astype(np.float32)


def main():
    D = design()
    np.savez_compressed(DATA / 'plate_k000.npz', **plate_mesh(D, D['tv0'], 40, 110_000))
    np.savez_compressed(DATA / 'plate_k023.npz', **plate_mesh(D, D['tv_final'], 40, 110_000))
    np.savez_compressed(DATA / 'plate_ghost.npz', **plate_mesh(D, D['tv0'], 20, 25_000, keep=-1))
    np.savez_compressed(DATA / 'cell320_k000.npz', **cell_mesh(D, D['tv0'], 80, 30_000))
    z = np.load(DATA / f'cellfield_{CASE}_o023.npz')
    fld = dict(_nodes=z['nodes'], u_nice=z['u_nice'], u_exact=z['u_exact'], u_nice_exactq=z['u_nice_exactq'])
    np.savez_compressed(DATA / 'cell320_k023.npz', **cell_mesh(D, D['tv_final'], 80, 40_000, fld))
    for p in sorted(DATA.glob('*')):
        print(f'{p.name}: {p.stat().st_size / 1e6:.2f} MB')


if __name__ == '__main__':
    main()
