"""Surface-generic body builder (not frozen): the files of fast_prep4 for any sheet family of surfaces.py.

The frozen body pipeline (fast_prep4 -> stage_cutfem_graded support certification) has the Schwarz P level set built
in. This builder defines the body from the discrete geometry of the moment integrator instead:
  active cells   the material volume of the cell (polyref_torch_fast.cell_moments, the same s / levels as the teacher,
                 so the same piecewise-linear psi that defines the element moments) is positive, or the frozen witness
                 search (3-D, continuous level set) finds positive-measure material in it; face-connected
                 components other than the largest are dropped (recorded); with --layout <lattice.json> the
                 components are those of the whole lattice (cells of neighbouring lattice cells joined across the
                 shared box face), so that material disconnected inside one cell but connected through a neighbour
                 is kept on both sides and the box patches of neighbours match
  full cells     the frozen rule: no cell corner beyond the macro plane and both constraints strictly negative over
                 the cell by an interval enclosure of f (surfaces.f_range_box) and of the trilinear thickness (corner
                 min / max), margin 1e-10; ghost faces join active neighbours unless both are full (the frozen face rule)
  box patches    a boundary face of an active cell carries positive-area material: the frozen witness strategy on the
                 continuous level set (interval exclusion, strictly interior centre witness, 2 x 2 subdivision up to
                 depth 16; the plane is ignored when it is itself a box face, as in fast_prep4)
  cut patches    the same witness search on the macro plane section of the cells the plane crosses or touches; a
                 plane that is a box face (FULL packets) has no cut patch
Nodes, DOF map, face list and file formats are those of fast_prep4. With surface 'P' the output can be compared with
the frozen bodies (gyr_prep_check.py): the rules above are the discrete counterparts of the frozen certified ones.
Usage: gyr_prep.py <out_dir> <case> [<case> ...]
       gyr_prep.py <out_dir> --layout <lattice.json>        (all cells of the layout, lattice components)"""
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


def witness(items, taus, surface, max_depth=16, cap=4_000_000):
    """Positive-area material on 2-D patches by the frozen strategy (enclosure exclusion, strict-interior witness at the
    centre, 2 x 2 subdivision). items: list of (u0, u1, v0, v1, emb) with emb(u, v) -> (points (..., 3), valid mask)
    and emb.box(u0, u1, v0, v1) -> (lo (N, 3), hi (N, 3), ok (N,)) a box enclosing the patch piece. Returns a bool per
    item (True: a witness was found; False: excluded everywhere or undecided at max_depth, counted)."""
    found = np.zeros(len(items), dtype=bool)
    undecided = 0
    by_emb = {}
    for k, it in enumerate(items):
        by_emb.setdefault(id(it[4]), []).append(k)
    for ks in by_emb.values():
        emb = items[ks[0]][4]
        R = np.asarray([items[k][:4] for k in ks], dtype=float)        # u0 u1 v0 v1
        own = np.asarray(ks)
        for depth in range(max_depth + 1):
            if not len(R):
                break
            lo, hi, ok = emb.box(own, R[:, 0], R[:, 1], R[:, 2], R[:, 3])
            flo, fhi = SF.f_range_box(lo, hi, surface)
            cor = np.stack([np.where(np.asarray(b, bool), hi, lo) for b in product((0, 1), repeat=3)])
            tv = tau_at(taus, cor); tlo, thi = tv.min(0), tv.max(0)
            excl = ~ok | (flo - thi > 0) | (-fhi - thi > 0) | emb.excluded(own, lo, hi)
            keep = ~excl & ~found[own]
            R, own = R[keep], own[keep]
            if not len(R):
                break
            uc, vc = (R[:, 0] + R[:, 1]) / 2, (R[:, 2] + R[:, 3]) / 2
            pts, valid = emb.point(own, uc, vc)
            f = SF.f_np(pts, surface); t = tau_at(taus, pts)
            hit = valid & (t - f > 0) & (t + f > 0) & emb.strict(own, pts)
            found[own[hit]] = True
            keep = ~found[own]
            R, own = R[keep], own[keep]
            if depth == max_depth or not len(R):
                undecided += len(np.unique(own)) if depth == max_depth else 0
                break
            if 4 * len(R) > cap:
                undecided += len(np.unique(own)); break
            um, vm = (R[:, 0] + R[:, 1]) / 2, (R[:, 2] + R[:, 3]) / 2
            R = np.concatenate([np.stack([R[:, 0], um, R[:, 2], vm], 1), np.stack([um, R[:, 1], R[:, 2], vm], 1),
                                np.stack([R[:, 0], um, vm, R[:, 3]], 1), np.stack([um, R[:, 1], vm, R[:, 3]], 1)])
            own = np.concatenate([own] * 4)
    return found, undecided


def witness3(boxes_lo, boxes_hi, taus, normal, offset, surface, max_depth=24, cap=4_000_000):
    """Positive-measure material in 3-D boxes (the frozen local-support strategy: enclosure exclusion, strictly interior
    centre witness, 2 x 2 x 2 subdivision). Returns found (bool per box) and the undecided count."""
    N = len(boxes_lo)
    found = np.zeros(N, dtype=bool)
    lo, hi, own = np.asarray(boxes_lo, float), np.asarray(boxes_hi, float), np.arange(N)
    nv = None if normal is None else np.asarray(normal)
    oct8 = np.asarray(list(product((0, 1), repeat=3)), float)
    undecided = 0
    for depth in range(max_depth + 1):
        if not len(lo):
            break
        flo, fhi = SF.f_range_box(lo, hi, surface)
        cor = np.stack([np.where(np.asarray(b, bool), hi, lo) for b in product((0, 1), repeat=3)])
        tv = tau_at(taus, cor); thi = tv.max(0)
        excl = (flo - thi > 0) | (-fhi - thi > 0)
        if nv is not None:
            excl |= ((cor @ nv) - offset >= 0).all(0)
        keep = ~excl & ~found[own]
        lo, hi, own = lo[keep], hi[keep], own[keep]
        if not len(lo):
            break
        c = (lo + hi) / 2
        f = SF.f_np(c, surface); t = tau_at(taus, c)
        hit = (t - f > 0) & (t + f > 0)
        if nv is not None:
            hit &= offset - c @ nv > 0
        found[own[hit]] = True
        keep = ~found[own]
        lo, hi, own = lo[keep], hi[keep], own[keep]
        if not len(lo):
            break
        if depth == max_depth or 8 * len(lo) > cap:
            undecided += len(np.unique(own)); break
        h = (hi - lo) / 2
        lo = (lo[:, None, :] + oct8[None] * h[:, None, :]).reshape(-1, 3)
        hi = lo + np.repeat(h, 8, 0)
        own = np.repeat(own, 8)
    return found, undecided


class FaceEmb:
    """Box faces: item k lies in the plane x_axis = coord / n; u, v are the other two coordinates."""

    def __init__(self, axis, coord, n, normal, offset):
        self.axis, self.coord, self.n = axis, float(coord), n
        self.o = [d for d in range(3) if d != axis]
        self.normal = None if normal is None else np.asarray(normal); self.offset = offset

    def _pts(self, own, u, v):
        p = np.zeros(u.shape + (3,))
        p[..., self.axis] = self.coord / self.n; p[..., self.o[0]] = u; p[..., self.o[1]] = v
        return p

    def box(self, own, u0, u1, v0, v1):
        lo, hi = self._pts(own, u0, v0), self._pts(own, u1, v1)
        return lo, hi, np.ones(len(u0), bool)

    def excluded(self, own, lo, hi):
        if self.normal is None:
            return np.zeros(len(lo), bool)
        cor = np.stack([np.where(np.asarray(b, bool), hi, lo) for b in product((0, 1), repeat=3)])
        return ((cor @ self.normal) - self.offset >= 0).all(0)              # entirely beyond the plane

    def point(self, own, u, v):
        return self._pts(own, u, v), np.ones(len(u), bool)

    def strict(self, own, pts):
        return np.ones(len(pts), bool) if self.normal is None else (self.offset - pts @ self.normal > 0)


class CutEmb:
    """Macro plane sections: item k is the section of cell k; u, v are the two coordinates other than the dominant
    normal axis; the third follows from the plane and must lie in the cell."""

    def __init__(self, cells, n, normal, offset):
        self.cells, self.n, self.nv, self.offset = np.asarray(cells), n, np.asarray(normal), offset
        self.ax = int(np.argmax(np.abs(self.nv))); self.o = [d for d in range(3) if d != self.ax]

    def _third(self, u, v):
        return (self.offset - u * self.nv[self.o[0]] - v * self.nv[self.o[1]]) / self.nv[self.ax]

    def box(self, own, u0, u1, v0, v1):
        w = np.stack([self._third(a, b) for a in (u0, u1) for b in (v0, v1)])
        wlo, whi = w.min(0), w.max(0)
        clo, chi = self.cells[own, self.ax] / self.n, (self.cells[own, self.ax] + 1) / self.n
        wlo, whi = np.maximum(wlo, clo), np.minimum(whi, chi)
        lo = np.zeros((len(u0), 3)); hi = np.zeros((len(u0), 3))
        lo[:, self.o[0]], hi[:, self.o[0]] = u0, u1; lo[:, self.o[1]], hi[:, self.o[1]] = v0, v1
        lo[:, self.ax], hi[:, self.ax] = wlo, whi
        return lo, hi, wlo <= whi

    def excluded(self, own, lo, hi):
        return np.zeros(len(lo), bool)

    def point(self, own, u, v):
        p = np.zeros(u.shape + (3,))
        p[..., self.o[0]] = u; p[..., self.o[1]] = v; p[..., self.ax] = self._third(u, v)
        c = self.cells[own, self.ax] / self.n
        return p, (p[..., self.ax] > c) & (p[..., self.ax] < c + 1 / self.n)

    def strict(self, own, pts):
        return np.ones(len(pts), bool)


def box_plane(normal, offset):
    return normal is not None and sum(v != 0 for v in normal) == 1 and abs(offset) in (0.0, 1.0)


def main(out, case, samples=65, select=None):
    """select: None (largest face-connected component of this cell), 'active' (return the active cells, no files), or
    a callable cells -> keep mask (lattice components)."""
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
    # cells whose discrete volume vanishes but which carry positive-measure material (frozen rule: a certified witness
    # in the continuous geometry) are active too, with zero element moments (stabilised by the ghost penalty)
    zero = np.flatnonzero(vol <= 0)
    extra, und_act = witness3(cand[zero] / n, (cand[zero] + 1) / n, taus, normal, offset, surface)
    act = vol > 0
    act[zero[extra]] = True
    cells = cand[act]
    row = dict(case=case, surface=surface, n=n, candidates=int(len(cand)), positive=int((vol > 0).sum()),
               zero_volume_active=int(extra.sum()), witness_undecided_active=int(und_act),
               volume_fraction=float(vol.sum()))                          # moments carry the physical measure
    if select == 'active':
        return cells
    if callable(select):
        keep = select(cells)
        row['dropped_components'] = 'lattice'; row['dropped_cells'] = int((~keep).sum())
        cells = cells[keep]
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
    if ncomp > 1 and callable(select):
        row['cell_components'] = int(ncomp)                             # connected through neighbours only
    elif ncomp > 1:
        big = np.bincount(lab).argmax()
        row['dropped_components'] = int(ncomp - 1); row['dropped_cells'] = int((lab != big).sum())
        cells = cells[lab == big]
        lookup = {tuple(map(int, x)): k for k, x in enumerate(cells)}
    cells = cells.astype(np.int32)
    ids = np.ravel_multi_index((2 * cells[:, None, :].astype(np.int64) + OFFSETS[None]).transpose(2, 0, 1), (2 * n + 1,) * 3)
    nodes = np.unique(ids)
    local = np.searchsorted(nodes, ids)
    dofs = (3 * local[:, :, None] + np.arange(3)).reshape(len(cells), 81)
    # full cells (frozen rule, interval enclosure) and ghost faces
    E = len(cells)
    lo_b, hi_b = cells / n, (cells + 1) / n
    flo, fhi = SF.f_range_box(lo_b, hi_b, surface)
    cor = np.stack([(cells + np.asarray(b)) / n for b in product((0, 1), repeat=3)])      # 8 x E x 3
    tv = tau_at(taus, cor)
    tlo = tv.min(0)
    full = (fhi - tlo < -1e-10) & (-flo - tlo < -1e-10)
    if normal is not None and not box_plane(normal, offset):
        full &= ((cor @ np.asarray(normal)) - offset <= 0).all(0)
    faces = []
    for owner in range(E):
        ijk = cells[owner]
        for axis in range(3):
            other = ijk.copy(); other[axis] += 1
            nbr = lookup.get(tuple(map(int, other)))
            if nbr is not None and not (full[owner] and full[nbr]):
                faces.append((owner, nbr, axis))
    faces = np.asarray(faces, dtype=np.int32)
    # box patches and cut patches: frozen witness strategy (enclosure exclusion, centre witness, subdivision)
    bp = box_plane(normal, offset)
    pn, po = (None, None) if bp else (normal, offset)
    items, meta = [], []
    embs = {}
    for k, ijk in enumerate(cells):
        for axis, side in product(range(3), (0, 1)):
            coord = int(ijk[axis]) + side
            if coord not in (0, n):
                continue
            key = (axis, coord)
            if key not in embs:
                embs[key] = FaceEmb(axis, coord, n, pn, po)
            o = [d for d in range(3) if d != axis]
            items.append(((ijk[o[0]]) / n, (ijk[o[0]] + 1) / n, (ijk[o[1]]) / n, (ijk[o[1]] + 1) / n, embs[key]))
            meta.append((k, axis, side))
    found, und_box = witness(items, taus, surface)
    box_ids = []
    for (k, axis, side), ok in zip(meta, found):
        if not ok:
            continue
        ijk = cells[k]
        g = [np.arange(2 * ijk[d], 2 * ijk[d] + 3) for d in range(3)]
        g[axis] = np.asarray([2 * ijk[axis] + (2 if side else 0)])
        p3 = np.stack(np.meshgrid(*g, indexing='ij'), -1).reshape(-1, 3)
        box_ids.append(np.ravel_multi_index(p3.T, (2 * n + 1,) * 3))
    box_nodes = np.unique(np.concatenate(box_ids)) if box_ids else np.zeros(0, dtype=np.int64)
    cut = []
    und_cut = 0
    if normal is not None and not bp:
        nv = np.asarray(normal)
        cor = (cells[:, None, :] + np.asarray(list(product((0, 1), repeat=3)))[None]) / n
        sd = cor @ nv - offset
        crossed = np.flatnonzero((sd.min(1) <= 0) & (sd.max(1) >= 0))
        emb = CutEmb(cells, n, normal, offset)
        o = emb.o
        citems = [(cells[k, o[0]] / n, (cells[k, o[0]] + 1) / n, cells[k, o[1]] / n, (cells[k, o[1]] + 1) / n, emb) for k in crossed]
        # the item index must be the cell index for CutEmb (own -> cells[own]): remap
        emb.cells = cells[crossed]
        cfound, und_cut = witness(citems, taus, surface)
        cut = [int(crossed[i]) for i in np.flatnonzero(cfound)]
    cut_cells = np.asarray(cut, dtype=np.int64)
    cut_nodes = np.unique(ids[cut_cells]) if len(cut_cells) else np.zeros(0, dtype=np.int64)
    d = Path(out) / case; d.mkdir(parents=True, exist_ok=True)
    np.save(d / 'NODES.npy', nodes); np.save(d / 'CELL_INDICES.npy', cells); np.save(d / 'dofs.npy', dofs)
    np.save(d / 'GP_FACES.npy', faces); np.save(d / 'BOX_NODES.npy', box_nodes)
    np.save(d / 'CUT_NODES.npy', cut_nodes); np.save(d / 'CUT_CELLS.npy', cut_cells)
    (d / 'CUT_PATCHES.json').write_text(json.dumps([dict(parent=[int(v) for v in cells[k]], tag='MACRO_CUT_FACE') for k in cut]))
    row.update(cells=int(E), nodes=int(len(nodes)), faces=int(len(faces)), full_cells=int(full.sum()),
               box_nodes=int(len(box_nodes)), cut_cells=int(len(cut_cells)), cut_nodes=int(len(cut_nodes)),
               witness_undecided_box=int(und_box), witness_undecided_cut=int(und_cut),
               seconds=time.perf_counter() - t0,
               builder='gyr_prep (discrete-geometry rules, not the frozen certification)', samples=samples)
    (d / 'PREP.json').write_text(json.dumps(row, indent=2))
    print(json.dumps(row), flush=True)


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--') and not a.endswith('.json')]
    smp = 65
    if '--samples' in sys.argv:
        smp = int(sys.argv[sys.argv.index('--samples') + 1]); args.remove(str(smp))
    out = args[0]
    if '--layout' in sys.argv:
        lay = json.loads(Path(sys.argv[sys.argv.index('--layout') + 1]).read_text())
        pos = {c['case']: np.asarray(c['position'], dtype=np.int64) for c in lay['cells']}
        act = {c: main(out, c, smp, select='active') for c in pos}
        n = int(json.loads((packet_dir(next(iter(pos))) / 'FRESH_CONTEXT.json').read_text())['n'])
        glob = np.concatenate([pos[c][None] * n + act[c] for c in pos])            # lattice cell coordinates
        owner = np.concatenate([[c] * len(act[c]) for c in pos])
        lookup = {tuple(map(int, x)): k for k, x in enumerate(glob)}
        ei, ej = [], []
        for k, x in enumerate(glob):
            for axis in range(3):
                y = x.copy(); y[axis] += 1
                nb = lookup.get(tuple(map(int, y)))
                if nb is not None:
                    ei.append(k); ej.append(nb)
        G = sparse.coo_matrix((np.ones(len(ei)), (ei, ej)), shape=(len(glob), len(glob)))
        ncomp, lab = connected_components(G, directed=False)
        big = np.bincount(lab).argmax()
        keepset = {c: set(map(tuple, (glob[(owner == c) & (lab == big)] - pos[c] * n).tolist())) for c in pos}
        print(json.dumps(dict(layout=lay['name'], lattice_components=int(ncomp), lattice_cells=int(len(glob)),
                              dropped=int((lab != big).sum()))), flush=True)
        for c in pos:
            main(out, c, smp, select=lambda cells, c=c: np.asarray([tuple(map(int, x)) in keepset[c] for x in cells], dtype=bool))
    else:
        for case in args[1:]:
            main(out, case, smp)
