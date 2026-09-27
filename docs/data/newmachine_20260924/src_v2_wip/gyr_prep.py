"""Surface-generic body builder (not frozen): the files of fast_prep4 for any sheet family of surfaces.py.

The frozen body pipeline (fast_prep4 -> stage_cutfem_graded support certification) has the Schwarz P level set built
in. This builder defines the body from the discrete geometry of the moment integrator instead:
  active cells   the material volume of the cell (polyref_torch_fast.cell_moments, the same s / levels as the teacher,
                 so the same piecewise-linear psi that defines the element moments) is positive; face-connected
                 components other than the largest are dropped (recorded)
  full cells     every level-0 sub-cube corner has all three constraints >= 1e-9 (moments_ad.full_rows), i.e. the
                 integrator takes the cell as full; ghost faces join active neighbours unless both are full (the frozen
                 face rule)
  box patches    a boundary face of an active cell carries material if one of samples^2 points on it (the continuous
                 level set, plane included) is strictly inside
  cut patches    the macro plane section of an active cell carries material (samples^2 points on the plane section);
                 a plane that is a box face (FULL packets) has no cut patch, as in fast_prep4
Nodes, DOF map, face list and file formats are those of fast_prep4. With surface 'P' the output can be compared with
the frozen bodies (gyr_prep_check.py): the rules above are the discrete counterparts of the frozen certified ones.
Usage: gyr_prep.py <out_dir> <case> [<case> ...] [--samples 33]"""
import json, sys, time, os
from itertools import product
from fractions import Fraction
from pathlib import Path
import numpy as np
import torch
from scipy import sparse
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).resolve().parent))
import polyref_torch_fast as PT
import moments_ad as MA
import surfaces as SF

OFFSETS = np.asarray(list(product(range(3), repeat=3)))
R = Path('/root/autodl-tmp/CUTFEM_FRESH_GP_20260921/packets')
dev, dt = torch.device('cuda'), torch.float64


def packet_dir(case):
    pk = R / case
    if pk.exists():
        return pk
    for r_ in [Path(x) for x in os.environ.get('OPL_PACKETS_EXTRA', '').split(':') if x]:
        if (r_ / case).exists():
            return r_ / case
    raise FileNotFoundError(case)


def tau_at(taus, pts):
    """trilinear thickness (corner index 4x + 2y + z) at points (..., 3) in the unit cell."""
    x, y, z = pts[..., 0], pts[..., 1], pts[..., 2]
    out = 0.0
    for c in range(8):
        bx, by, bz = (c >> 2) & 1, (c >> 1) & 1, c & 1
        out = out + taus[c] * (x if bx else 1 - x) * (y if by else 1 - y) * (z if bz else 1 - z)
    return out


def inside(pts, taus, normal, offset, surface):
    f = SF.f_np(pts, surface)
    t = tau_at(taus, pts)
    ok = (t - f > 0) & (t + f > 0)
    if normal is not None:
        ok &= (offset - pts @ np.asarray(normal)) > 0
    return ok


def box_plane(normal, offset):
    return normal is not None and sum(v != 0 for v in normal) == 1 and abs(offset) in (0.0, 1.0)


def main(out, case, samples=33):
    t0 = time.perf_counter()
    ctx = json.loads((packet_dir(case) / 'FRESH_CONTEXT.json').read_text())
    cs = ctx['case']; n = int(ctx['n'])
    surface = SF.surface_of(cs)
    taus = [float(Fraction(v)) for v in cs['tau_corners']]
    normal = None if cs.get('normal') is None else [float(Fraction(v)) for v in cs['normal']]
    offset = None if normal is None else float(Fraction(cs['offset']))
    s, levels = 4, 1                                                    # teacher.Cell defaults
    grid = np.asarray(list(product(range(n), repeat=3)), dtype=np.int64)
    if normal is not None:                                              # cells entirely beyond the plane: no material
        corners = (grid[:, None, :] + np.asarray(list(product((0, 1), repeat=3)))[None]) / n
        kept = ((corners @ np.asarray(normal)) - offset < 0).any(1)
    else:
        kept = np.ones(len(grid), dtype=bool)
    cand = grid[kept]
    M = PT.cell_moments(cand, n, taus, normal, offset, s, levels=levels, surface=surface)
    vol = M[:, 0]
    cells = cand[vol > 0]
    row = dict(case=case, surface=surface, n=n, candidates=int(len(cand)), positive=int(len(cells)),
               volume_fraction=float(vol.sum()))                          # moments carry the physical measure
    # face-connected components: keep the largest
    lookup = {tuple(map(int, x)): k for k, x in enumerate(cells)}
    ei, ej = [], []
    for k, x in enumerate(cells):
        for axis in range(3):
            y = x.copy(); y[axis] += 1
            nb = lookup.get(tuple(map(int, y)))
            if nb is not None:
                ei.append(k); ej.append(nb)
    G = sparse.coo_matrix((np.ones(len(ei)), (ei, ej)), shape=(len(cells), len(cells)))
    ncomp, lab = connected_components(G, directed=False)
    if ncomp > 1:
        big = np.bincount(lab).argmax()
        row['dropped_components'] = int(ncomp - 1); row['dropped_cells'] = int((lab != big).sum())
        cells = cells[lab == big]
        lookup = {tuple(map(int, x)): k for k, x in enumerate(cells)}
    cells = cells.astype(np.int32)
    ids = np.ravel_multi_index((2 * cells[:, None, :].astype(np.int64) + OFFSETS[None]).transpose(2, 0, 1), (2 * n + 1,) * 3)
    nodes = np.unique(ids)
    local = np.searchsorted(nodes, ids)
    dofs = (3 * local[:, :, None] + np.arange(3)).reshape(len(cells), 81)
    # full cells and ghost faces
    E = len(cells)
    taus_t = torch.as_tensor(np.asarray(taus, float), dtype=dt, device=dev)[None].expand(E, 8).contiguous()
    nrm_t, off_t = MA.plane_rows(normal, offset, E, dev)
    full = MA.full_rows(cells, n, taus_t, nrm_t, off_t, s=s, surface=surface).cpu().numpy()
    faces = []
    for owner in range(E):
        ijk = cells[owner]
        for axis in range(3):
            other = ijk.copy(); other[axis] += 1
            nbr = lookup.get(tuple(map(int, other)))
            if nbr is not None and not (full[owner] and full[nbr]):
                faces.append((owner, nbr, axis))
    faces = np.asarray(faces, dtype=np.int32)
    # box patches (sampled)
    u = (np.arange(samples) + 0.5) / samples
    A, B = np.meshgrid(u, u, indexing='ij')
    box_ids = []
    for k, ijk in enumerate(cells):
        for axis, side in product(range(3), (0, 1)):
            coord = ijk[axis] + side
            if coord not in (0, n):
                continue
            o = [d for d in range(3) if d != axis]
            pts = np.zeros((samples * samples, 3))
            pts[:, axis] = coord / n
            pts[:, o[0]] = (ijk[o[0]] + A.reshape(-1)) / n; pts[:, o[1]] = (ijk[o[1]] + B.reshape(-1)) / n
            if not inside(pts, taus, normal, offset, surface).any():
                continue
            g = [np.arange(2 * ijk[d], 2 * ijk[d] + 3) for d in range(3)]
            g[axis] = np.asarray([2 * ijk[axis] + (2 if side else 0)])
            p3 = np.stack(np.meshgrid(*g, indexing='ij'), -1).reshape(-1, 3)
            box_ids.append(np.ravel_multi_index(p3.T, (2 * n + 1,) * 3))
    box_nodes = np.unique(np.concatenate(box_ids)) if box_ids else np.zeros(0, dtype=np.int64)
    # cut patches (sampled on the plane section)
    cut = []
    if normal is not None and not box_plane(normal, offset):
        nv = np.asarray(normal)
        ax = int(np.argmax(np.abs(nv))); o = [d for d in range(3) if d != ax]
        for k, ijk in enumerate(cells):
            cor = (ijk[None] + np.asarray(list(product((0, 1), repeat=3)))) / n
            sd = cor @ nv - offset
            if sd.min() > 0 or sd.max() < 0:
                continue
            pts = np.zeros((samples * samples, 3))
            pts[:, o[0]] = (ijk[o[0]] + A.reshape(-1)) / n; pts[:, o[1]] = (ijk[o[1]] + B.reshape(-1)) / n
            pts[:, ax] = (offset - pts[:, o[0]] * nv[o[0]] - pts[:, o[1]] * nv[o[1]]) / nv[ax]
            inb = (pts[:, ax] > ijk[ax] / n) & (pts[:, ax] < (ijk[ax] + 1) / n)
            if not inb.any():
                continue
            f = SF.f_np(pts[inb], surface); t = tau_at(taus, pts[inb])
            if ((t - f > 0) & (t + f > 0)).any():
                cut.append(k)
    cut_cells = np.asarray(cut, dtype=np.int64)
    cut_nodes = np.unique(ids[cut_cells]) if len(cut_cells) else np.zeros(0, dtype=np.int64)
    d = Path(out) / case; d.mkdir(parents=True, exist_ok=True)
    np.save(d / 'NODES.npy', nodes); np.save(d / 'CELL_INDICES.npy', cells); np.save(d / 'dofs.npy', dofs)
    np.save(d / 'GP_FACES.npy', faces); np.save(d / 'BOX_NODES.npy', box_nodes)
    np.save(d / 'CUT_NODES.npy', cut_nodes); np.save(d / 'CUT_CELLS.npy', cut_cells)
    (d / 'CUT_PATCHES.json').write_text(json.dumps([dict(parent=[int(v) for v in cells[k]], tag='MACRO_CUT_FACE') for k in cut]))
    row.update(cells=int(E), nodes=int(len(nodes)), faces=int(len(faces)), full_cells=int(full.sum()),
               box_nodes=int(len(box_nodes)), cut_cells=int(len(cut_cells)), cut_nodes=int(len(cut_nodes)),
               seconds=time.perf_counter() - t0,
               builder='gyr_prep (discrete-geometry rules, not the frozen certification)', samples=samples)
    (d / 'PREP.json').write_text(json.dumps(row, indent=2))
    print(json.dumps(row), flush=True)


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    smp = 33
    if '--samples' in sys.argv:
        smp = int(sys.argv[sys.argv.index('--samples') + 1]); args.remove(str(smp))
    out = args[0]
    for case in args[1:]:
        main(out, case, smp)
