"""V step, part 2: is the pipeline equivariant under the cube group, i.e. do rotated geometries give the permuted
masks and the rotated operator on the SAME fixed grid? For base case B and rotation R (x' = c + R (x - c) on the
node grid, c = n):
  - active cells, nodes, box nodes, GP faces of the rotated case == images of the base sets (exact integers),
  - one local node ordering per case (element stiffness = moments x one fixed template),
  - box operator: T_rot == Pi T_B Pi^T with Pi = node permutation (x) R on the displacement components,
    reported as relative Frobenius difference and max relative energy difference over random and soft directions,
  - cut band: background nodes of active cells crossed by the cut plane (count; all on the fixed grid by construction).
Usage: v_rot.py <body_dir> <out_json> <base_case> [<base_case> ...]
"""
import json, sys, time, gc
from pathlib import Path
from fractions import Fraction
import numpy as np
import torch
sys.path.insert(0, '/root/autodl-tmp/OPL/src')
import precision_study as PS
import make_rot as MR

dev, dt = PS.dev, PS.dt
N = 32


def load(body, case):
    d = Path(body) / case
    return {k: np.load(d / f'{k}.npy') for k in ('NODES', 'CELL_INDICES', 'GP_FACES', 'BOX_NODES')}


def grid(ids):
    return np.stack(np.unravel_index(ids, (2 * N + 1,) * 3), 1)


def node_map(ids, r):
    g = grid(ids) - N
    g2 = g @ np.asarray(r).T + N
    return np.ravel_multi_index(g2.T, (2 * N + 1,) * 3)


def cell_img(cells, r):
    c = 2 * cells + 1 - N                                     # cell centers in node-grid units, centered
    c2 = c @ np.asarray(r).T + N
    return (c2 - 1) // 2


def face_set(cells, faces):
    """faces as unordered pairs of cell keys."""
    k = (cells[:, 0].astype(np.int64) * N + cells[:, 1]) * N + cells[:, 2]
    a, b = k[faces[:, 0]], k[faces[:, 1]]
    return set(zip(np.minimum(a, b).tolist(), np.maximum(a, b).tolist()))


def cut_band(case, cells):
    ctx = json.loads((PS.SE.ROOT / 'packets' / case / 'FRESH_CONTEXT.json').read_text())['case']
    if ctx['kind'] != 'CUT':
        return 0
    nrm = np.array([float(Fraction(v)) for v in ctx['normal']]); off = float(Fraction(ctx['offset']))
    lo = cells / N; corners = lo[:, None, :] + np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)])[None] / N
    s = corners @ nrm - off
    crossed = (s.min(1) < 0) & (s.max(1) > 0)
    return int(crossed.sum())


def main(body, out, bases):
    rec = {}
    for base in bases:
        B = load(body, base)
        is_full = base.endswith('_full')
        TB, idB, _, stB = PS.schur_T(base, body, 'fp64', host=is_full)
        TB = TB.to(dev) if not is_full else TB
        rows = {'box': int(len(idB)), 'cut_band_cells': cut_band(base, B['CELL_INDICES'])}
        for name, r in MR.ROTS.items():
            case = f'{base}_rot{name}'
            Rr = load(body, case)
            row = {}
            row['cells_equal'] = bool(np.array_equal(np.unique(cell_img(B['CELL_INDICES'], r), axis=0),
                                                     np.unique(Rr['CELL_INDICES'], axis=0)))
            row['nodes_equal'] = bool(np.array_equal(np.sort(node_map(B['NODES'], r)), Rr['NODES']))
            row['box_nodes_equal'] = bool(np.array_equal(np.sort(node_map(B['BOX_NODES'], r)), Rr['BOX_NODES']))
            ci = cell_img(B['CELL_INDICES'], r)
            row['gp_faces_equal'] = face_set(ci, B['GP_FACES']) == face_set(Rr['CELL_INDICES'], Rr['GP_FACES'])
            try:
                TR, idR, _, stR = PS.schur_T(case, body, 'fp64', host=is_full)
                row['one_node_ordering'] = True
            except ValueError as e:
                row['one_node_ordering'] = str(e); rows[name] = row; continue
            # Pi: base box dof (a, k) -> rotated box dof (m(a), i) with coefficient R[i, k]
            m = node_map(idB, r)
            pos = np.searchsorted(idR, m)
            if not (pos < len(idR)).all() or not np.array_equal(idR[np.minimum(pos, len(idR) - 1)], m):
                row['operator'] = 'BOX_SETS_DIFFER'; rows[name] = row; continue
            nb = len(idB)
            Rt = torch.as_tensor(np.asarray(r, dtype=np.float64), device=dev)
            TRd = TR.to(dev); TB4 = TB.to(dev).reshape(nb, 3, nb, 3)
            inv = torch.as_tensor(np.argsort(pos), device=dev)       # rotated node p <- base node inv[p]
            g = torch.Generator(device=dev).manual_seed(0)
            Q = torch.randn((3 * nb, 64), dtype=dt, device=dev, generator=g)
            small = 3 * nb < 15000
            if small:
                ev, V = torch.linalg.eigh(TRd)
                Q = torch.cat([Q, V[:, 6:38]], 1); del V, ev          # 32 softest deformation directions
            num = torch.zeros(Q.shape[1], dtype=dt, device=dev); den = torch.zeros_like(num)
            dn = tn = 0.0
            for p0 in range(0, nb, 256):
                rr = torch.arange(p0, min(p0 + 256, nb), device=dev)
                blk = TB4[inv[rr]][:, :, inv, :]                    # (r, 3, nb, 3) in rotated node order
                blk = torch.einsum('ik,rkcl->ricl', Rt, blk)
                blk = torch.einsum('ricl,jl->ricj', blk, Rt).reshape(len(rr) * 3, 3 * nb)
                d = TRd[3 * p0:3 * p0 + blk.shape[0]] - blk
                dn += float((d ** 2).sum()); tn += float((blk ** 2).sum())
                qs = Q[3 * p0:3 * p0 + blk.shape[0]]
                num += (qs * (d @ Q)).sum(0); den += (qs * (blk @ Q)).sum(0)
                del blk, d
            row['rel_frobenius'] = (dn / tn) ** 0.5
            row['max_rel_energy_diff'] = float((num.abs() / den).max())
            row['soft_directions_checked'] = bool(small)
            del TB4
            row['schur_seconds'] = stR.get('factorize_schur')
            rows[name] = row
            print(json.dumps({'base': base, 'rot': name, **row}), flush=True)
            del TR, TRd, Q
            gc.collect(); torch.cuda.empty_cache()
        rec[base] = rows
        del TB
        gc.collect(); torch.cuda.empty_cache()
    Path(out).write_text(json.dumps(rec, indent=2))
    print('DONE', flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
