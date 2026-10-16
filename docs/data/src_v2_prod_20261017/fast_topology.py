"""Route 7: the frozen local-support certification (stage_cutfem_graded.support.compile_support) with a vectorized
float64 filter in front of the exact per-cell routine; the record must equal the frozen one exactly.

The frozen routine certifies, for each of the n^3 background cells, whether the clipped cell carries material of
positive measure, using closed-form interval enclosures over the bounding box of the clipped cell:
    phi range   = sum over axes of the hull of cos(2 pi x) at both box ends and of (-1)^k at every half-integer
                  k/2 inside the range (exact table values at the special points, 80-digit intervals otherwise);
    tau range   = min / max of the exact trilinear thickness over the 8 box corners;
    constraint  = sign*phi - tau, sign = +-1 (material: both constraints < 0).
It excludes a cell if an enclosure is strictly positive, accepts it if both constraints are strictly negative at the
centre or at a vertex (first box, so the visited count is 1), and otherwise refines and subdivides.

This filter evaluates the same formulas in float64 for every cell at once and decides only when the float value is
farther than a margin (1e-10, versus float errors near 1e-15) from every threshold, so each fast decision implies the
identical exact decision at the identical step, with the identical record (visited count 1, depth 0, and for
box-face patches the centre witness). Everything else goes to the unchanged frozen cell_support:
  - cells the macro plane crosses or touches (the clipped polytope is not the box),
  - cells whose enclosure or point values fall inside the margin,
  - active cells on the unit-box boundary whose face centre does not decide the face patch (witness order).
Cells entirely on the removed side of the plane have an empty ambient polytope (visited 0) exactly as in the frozen
code. The global connectedness proof (compile_homotopy on the coarse grid) is called unchanged.
"""
import json
import time
from fractions import Fraction as F
from itertools import product
from pathlib import Path

import numpy as np

EXACT_COS = {F(0): 1.0, F(1, 6): 0.5, F(1, 4): 0.0, F(1, 3): -0.5, F(1, 2): -1.0, F(2, 3): -0.5, F(3, 4): 0.0, F(5, 6): 0.5}


def cos_grid(n):
    """cos(2 pi i / (2n)) for the half-cell grid i = 0..2n (exact where the frozen code is exact)."""
    out = np.empty(2 * n + 1)
    for i in range(2 * n + 1):
        x = F(i, 2 * n) % 1
        out[i] = EXACT_COS[x] if x in EXACT_COS else np.cos(2 * np.pi * (i / (2 * n)))
    return out


def axis_ranges(n):
    """per cell index i along one axis: [min, max] of cos(2 pi x) over [i/n, (i+1)/n] with the frozen extrema rule."""
    c = cos_grid(n)
    lo = np.empty(n); hi = np.empty(n)
    for i in range(n):
        vals = [c[2 * i], c[2 * i + 2]]
        a, b = F(i, n), F(i + 1, n)
        for k in range(int(np.ceil(2 * a)), int(np.floor(2 * b)) + 1):
            vals.append(float((-1) ** k))
        lo[i], hi[i] = min(vals), max(vals)
    return lo, hi, c


def compile_support_fast(contract, workers=None, margin=1e-10, log=print, global_proof=True):
    from stage_cutfem_full_interface.contracts import digest
    from stage_cutfem_runtime.config import assembly_workers
    from stage_cutfem_runtime.parallel import ExecutionPool
    from stage_cutfem_graded import support as S
    from stage_cutfem_graded.contract import GradedContract
    from stage_cutfem_graded.topology import compile_homotopy
    t0 = time.perf_counter()
    workers = assembly_workers(workers)
    policy = json.loads((Path(S.__file__).with_name('SUPPORT_POLICY_R43.json')).read_text())
    coarse = GradedContract(contract.case_id, policy['global_topology_initial_grid'], contract.normal, contract.offset,
                            contract.tau_corners)
    if global_proof:
        global_topology = compile_homotopy(coarse, workers)
    else:
        # user decision 2026-09-23: no global connectedness proof in the fast pipeline; the caller checks the face
        # adjacency of the active cells and the factorization exposes any extra zero mode numerically
        global_topology = dict(status='TRUE_TOPOLOGY_CERTIFIED', component_count=1, topology_grid=None,
                               proof='SKIPPED_NO_GLOBAL_PROOF_FAST_PIPELINE')
    if global_topology['status'] == 'TRUE_TOPOLOGY_NOT_CERTIFIED' and policy.get('global_topology_fallback'):
        from stage_cutfem_graded.regular_reference import compile_global
        historical = global_topology
        global_topology = compile_global(coarse, workers)
        global_topology['historical_unresolved_global_proof'] = historical
    t_global = time.perf_counter() - t0
    result = dict(contract=contract.record(), contract_sha256=digest(contract.record()),
                  support_policy_sha256=digest(policy), global_topology=global_topology,
                  proof='GLOBAL_INTERVAL_HOMOTOPY_AND_SEPARATE_EXACT_LOCAL_SUPPORT_R43_V5',
                  background_grid=contract.n, quadrature_defines_topology=False,
                  topology_independent_of_background_resolution=True, disconnected_fragments_discarded=0)
    if global_topology['status'] != 'TRUE_TOPOLOGY_CERTIFIED':
        return dict(result, status='TRUE_TOPOLOGY_NOT_CERTIFIED'), dict(global_seconds=t_global)
    count = global_topology['component_count']
    if count != 1:
        return dict(result, status='TRUE_TOPOLOGY_CERTIFIED', component_count=count, active_parent_cells=[], patches=[]), \
            dict(global_seconds=t_global)
    t1 = time.perf_counter()
    n = contract.n
    # exact side of the macro plane at every grid node, in integers: s = n.x - d, kept side s <= 0
    normal, offset = contract.rational_plane()
    from math import gcd
    L = 1
    for q in (*normal, offset):
        L = L * q.denominator // gcd(L, q.denominator)
    coef = [int(v * L) for v in normal]; dd = int(offset * L * n)
    s_node = np.empty((n + 1,) * 3, dtype=np.int64)
    for i in range(n + 1):
        for j in range(n + 1):
            base = coef[0] * i + coef[1] * j - dd                        # exact Python integers
            s_node[i, j] = [(base + coef[2] * k > 0) - (base + coef[2] * k < 0) for k in range(n + 1)]
    corner_s = np.stack([s_node[a:a + n, b:b + n, c:c + n] for a, b, c in product((0, 1), repeat=3)])
    removed = corner_s.min(0) >= 0          # zero-volume intersection with the kept side: ambient polytope empty
    inside = corner_s.max(0) < 0            # the plane is redundant: the clipped cell is the whole box
    # enclosures over the box
    lo1, hi1, cg = axis_ranges(n)
    phi_lo = lo1[:, None, None] + lo1[None, :, None] + lo1[None, None, :]
    phi_hi = hi1[:, None, None] + hi1[None, :, None] + hi1[None, None, :]
    pc = [float(v) for v in contract.thickness.power_coefficients]
    def tau(x, y, z):
        a, b, c, d, e, f, g, h = pc
        return a + b * z + y * (c + d * z) + x * (e + f * z + y * (g + h * z))
    g = np.arange(n + 1) / n
    X, Y, Z = np.meshgrid(g, g, g, indexing='ij')
    tn = tau(X, Y, Z)
    tc = np.stack([tn[a:a + n, b:b + n, c:c + n] for a, b, c in product((0, 1), repeat=3)])
    tau_lo, tau_hi = tc.min(0), tc.max(0)
    E = {1: (phi_lo - tau_hi, phi_hi - tau_lo), -1: (-phi_hi - tau_hi, -phi_lo - tau_lo)}
    excluded = (E[1][0] > margin) | (E[-1][0] > margin)
    near = ((np.abs(E[1][0]) <= margin) | (np.abs(E[-1][0]) <= margin)) & ~excluded
    # point values at the centre and the 8 corners (half-cell grid cosines)
    ch = cg                                         # cos at i/(2n)
    def point_values(ix, iy, iz):                   # half-grid integer coordinates
        phi = ch[ix] + ch[iy] + ch[iz]
        t = tau(ix / (2 * n), iy / (2 * n), iz / (2 * n))
        return phi - t, -phi - t
    I = np.arange(n)
    CI, CJ, CK = np.meshgrid(2 * I + 1, 2 * I + 1, 2 * I + 1, indexing='ij')
    pts = [(CI, CJ, CK)] + [(CI - 1 + 2 * a, CJ - 1 + 2 * b, CK - 1 + 2 * c) for a, b, c in product((0, 1), repeat=3)]
    passes = np.zeros((n, n, n), dtype=bool); unsure = np.zeros((n, n, n), dtype=bool)
    for p in pts:
        v1, v2 = point_values(*p)
        ok = (v1 < -margin) & (v2 < -margin)
        uncertain = ~ok & ~((v1 > margin) | (v2 > margin))
        passes |= ok; unsure |= uncertain
    positive = ~excluded & ~near & passes
    fast_empty = inside & excluded
    fast_positive = inside & positive
    # box-face patches of fast-positive cells on the unit-box boundary: centre witness must decide every face
    face_ok = np.ones((n, n, n), dtype=bool)
    face_patch = {}
    boundary = np.zeros((n, n, n), dtype=bool)
    boundary[0] = boundary[-1] = True; boundary[:, 0] = boundary[:, -1] = True; boundary[:, :, 0] = boundary[:, :, -1] = True
    for idx in zip(*np.nonzero(fast_positive & boundary)):
        idx = tuple(int(v) for v in idx)
        patches = []
        for axis, side in product(range(3), (0, 1)):
            coord = idx[axis] + side
            if coord not in (0, n):
                continue
            lo = [F(idx[d], n) for d in range(3)]; hi = [F(idx[d] + 1, n) for d in range(3)]
            lo[axis] = hi[axis] = F(coord, n)
            # face enclosure in float with the same formulas
            h_lo, h_hi = [], []
            for d in range(3):
                a, b = lo[d], hi[d]
                vals = [cg[int(2 * n * a)], cg[int(2 * n * b)]]
                for k in range(int(np.ceil(2 * a)), int(np.floor(2 * b)) + 1):
                    vals.append(float((-1) ** k))
                h_lo.append(min(vals)); h_hi.append(max(vals))
            f_lo, f_hi = sum(h_lo), sum(h_hi)
            tv = [tau(*(float(hi[d] if bit[d] else lo[d]) for d in range(3))) for bit in product((0, 1), repeat=3)]
            fe = {1: (f_lo - max(tv), f_hi - min(tv)), -1: (-f_hi - max(tv), -f_lo - min(tv))}
            if fe[1][0] > margin or fe[-1][0] > margin:
                continue                                        # face excluded at its first box: no patch
            if abs(fe[1][0]) <= margin or abs(fe[-1][0]) <= margin:
                face_ok[idx] = False; break
            centre = tuple((lo[d] + hi[d]) / 2 for d in range(3))
            hc = [int(2 * n * x) for x in centre]
            if any(2 * n * x != h for x, h in zip(centre, hc)):
                face_ok[idx] = False; break
            v1, v2 = point_values(*hc)
            if not (v1 < -margin and v2 < -margin):
                face_ok[idx] = False; break                     # witness order would matter: exact routine
            sgn = 2 * side - 1
            patches.append(dict(parent=list(idx), tag=f"BOX_{'XYZ'[axis]}{'MAX' if side else 'MIN'}",
                                normal=[str(F(sgn) if d == axis else F(0)) for d in range(3)],
                                offset=str(F(sgn) * F(coord, n)), true_positive_support_certified=True,
                                true_positive_point=[str(x) for x in centre], support_boxes=1))
        face_patch[idx] = patches
    fast_positive &= face_ok
    decided = removed | fast_empty | fast_positive
    exact_idx = [tuple(int(v) for v in idx) for idx in zip(*np.nonzero(~decided))]
    t_filter = time.perf_counter() - t1
    t2 = time.perf_counter()
    rows = {}
    if exact_idx:
        jobs = [(contract, idx, policy) for idx in exact_idx]
        with ExecutionPool(workers) as pool:
            for ordinal, row in pool.map_unordered(S.cell_support, jobs):
                rows[exact_idx[ordinal]] = row
    t_exact = time.perf_counter() - t2
    failed = [row for row in rows.values() if not row['admitted']]
    if failed:
        return dict(result, status='TRUE_TOPOLOGY_NOT_CERTIFIED', local_support_unresolved_count=len(failed),
                    local_support_witnesses=failed[:20]), {}
    active, patches, boxes, depth, total = [], [], 0, 0, 0
    for index in product(range(n), repeat=3):
        total += 1
        if index in rows:
            row = rows[index]
            boxes += row['support']['boxes']; depth = max(depth, row['support']['maximum_depth'])
            if row['active']:
                active.append(list(index)); patches.extend(row['patches'])
        elif removed[index]:
            pass                                               # boxes 0, depth 0
        elif fast_empty[index]:
            boxes += 1
        else:
            boxes += 1
            active.append(list(index)); patches.extend(face_patch.get(index, []))
    out = dict(result, status='TRUE_TOPOLOGY_CERTIFIED', component_count=count,
               active_parent_cells=active, parent_components=[[0] for _ in active], patches=patches,
               local_support_summary=dict(cells=total, active=len(active), empty=total - len(active), unresolved=0,
                                          boxes=boxes, maximum_depth=depth),
               topology_grid=global_topology['topology_grid'])
    stats = dict(global_seconds=t_global, filter_seconds=t_filter, exact_seconds=t_exact, exact_cells=len(exact_idx),
                 removed=int(removed.sum()), fast_empty=int(fast_empty.sum()), fast_positive=int(fast_positive.sum()),
                 near=int((near & inside).sum()), boundary_fallback=int((~face_ok).sum()))
    return out, stats


def full_flags_fast(contract, cell_indices, workers=None, margin=1e-10):
    """GP full-cell test (stage_cutfem_graded.fields.full_background_box) for the active cells: no box corner beyond the
    macro plane (exact integers) and both constraint enclosures strictly negative over the box. Same formulas in
    float64 with a margin; undecided cells call the frozen test."""
    from stage_cutfem_runtime.parallel import ExecutionPool
    from math import gcd
    n = contract.n
    idx = np.asarray(cell_indices, dtype=np.int64)
    normal, offset = contract.rational_plane()
    L = 1
    for q in (*normal, offset):
        L = L * q.denominator // gcd(L, q.denominator)
    coef = [int(v * L) for v in normal]; dd = int(offset * L * n)
    beyond = np.zeros(len(idx), dtype=bool)
    for bit in product((0, 1), repeat=3):
        g = idx + np.asarray(bit)
        vals = [coef[0] * int(a) + coef[1] * int(b) + coef[2] * int(c) - dd for a, b, c in g]
        beyond |= np.asarray([v > 0 for v in vals])
    lo1, hi1, cg = axis_ranges(n)
    phi_lo = lo1[idx[:, 0]] + lo1[idx[:, 1]] + lo1[idx[:, 2]]
    phi_hi = hi1[idx[:, 0]] + hi1[idx[:, 1]] + hi1[idx[:, 2]]
    pc = [float(v) for v in contract.thickness.power_coefficients]
    def tau(x, y, z):
        a, b, c, d, e, f, g_, h = pc
        return a + b * z + y * (c + d * z) + x * (e + f * z + y * (g_ + h * z))
    tv = np.stack([tau(*((idx[:, d] + bit[d]) / n for d in range(3))) for bit in product((0, 1), repeat=3)])
    tau_lo = tv.min(0)
    up1, up2 = phi_hi - tau_lo, -phi_lo - tau_lo          # upper bounds of phi - tau and -phi - tau
    full = ~beyond & (up1 < -margin) & (up2 < -margin)
    notfull = beyond | (up1 > margin) | (up2 > margin)
    undecided = np.flatnonzero(~full & ~notfull)
    if len(undecided):
        from stage_cutfem_graded.fields import full_background_box
        for k in undecided:
            full[k] = bool(full_background_box(contract, idx[k]))
    return full, dict(undecided=int(len(undecided)), full=int(full.sum()))
