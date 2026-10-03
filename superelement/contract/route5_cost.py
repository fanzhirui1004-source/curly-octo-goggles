#!/usr/bin/env python3
"""Route 5 costing: is forming S at all worth it inside a topology-optimisation loop?

The question (user, 2026-09-17): a Schur complement is DENSE, and density is exactly
what keeps it out of topology optimisation. So why condense at all - why not run the
fine problem directly?

GPT Pro's decision 1 lands in the same place: "允许隐式" - deliver apply/energy, do not
materialise S.

Every input below is either measured in this project or an explicitly labelled
assumption with a range. Nothing is asserted without a source.
"""
import json
from pathlib import Path

# ---------------------------------------------------------------- measured inputs
M = dict(
    fine_dofs_per_cell=231192,        # seat 0328, BRIEFING_20260915
    interface_dofs_per_cell=12798,    # q for seat 0328
    teacher_seconds_per_cell=(39, 95),# TEACHER_COST_20260917: two receipts
    teacher_assembly_seconds=0.6,     # HANDOFF_20260913, seat 0257
    dense_S_numbers=12798 * 12799 // 2,
    hmat_numbers_at_gate=8_581_236,   # THE_REAL_GATE: eps_op 0.152
    hmat_numbers_at_3pct=12_229_217,  # FIRST_PASS: eps_op 0.026
    predict_dense_flops=5.6747e13,    # Codex SPEED_ARCHITECTURE_AUDIT, per cell
    gpu_tf32_peak=4.2e14,             # RTX 5090 TF32 tensor, spec
    gpu_fp64_effective=1.0e12,        # ASSUMPTION: sustained fp64 for a matrix-free
                                      # Q2 hex operator with sum factorisation.
                                      # 5090 fp64 is weak; this is deliberately
                                      # conservative and is the number to check first.
)
# ------------------------------------------------------------------- assumptions
A = dict(
    matvec_flops_per_dof=1000,   # ASSUMPTION: Q2 hex, sum-factorised operator apply.
                                 # Literature range 500-2000 for 3D Q2.
    mg_matvec_equivalents=(30, 150),  # ASSUMPTION: V-cycles x work per cycle, for a
                                      # preconditioned CutFEM solve. Wide on purpose:
                                      # cut-cell conditioning is the open risk.
    interface_iterations=(20, 80),    # ASSUMPTION: preconditioned interface iterations
    geometry_seconds_per_cell=(0.05, 5.0),  # THE DECISIVE UNKNOWN. The fine route must
                                   # re-cut and re-quadrature every cell every design
                                   # iteration; the predicted-S route needs only the 12
                                   # geometry numbers (8 tau corners + 4 cut-plane) and
                                   # never meshes. The teacher's receipt bundles
                                   # "geometry and factor preparation ~29 s" so the
                                   # geometry part alone is NOT separately measured.
                                   # Low end: incremental update on a fixed background
                                   # grid, only cells near the moving interface.
                                   # High end: full re-cut per cell.
    N_cells=(100, 1000, 10000),
    M_iterations=(100, 500),
)


def fmt(s):
    if s < 1: return f'{s*1e3:.0f} ms'
    if s < 90: return f'{s:.1f} s'
    if s < 5400: return f'{s/60:.1f} min'
    if s < 3600*48: return f'{s/3600:.1f} h'
    return f'{s/86400:.1f} d'


def route_fine(N):
    """Never form S. Matrix-free fine solve over the whole lattice."""
    dofs = M['fine_dofs_per_cell'] * N
    matvec = dofs * A['matvec_flops_per_dof'] / M['gpu_fp64_effective']
    lo = matvec * A['mg_matvec_equivalents'][0]
    hi = matvec * A['mg_matvec_equivalents'][1]
    mem = dofs * 8 * 10 / 2**30                       # ~10 work vectors
    geo = (A['geometry_seconds_per_cell'][0] * N, A['geometry_seconds_per_cell'][1] * N)
    return dict(dofs=dofs, matvec_seconds=matvec, solve=(lo, hi), memory_GiB=mem,
                geometry=geo, total=(lo + geo[0], hi + geo[1]))


def route_condensed(N, numbers_per_cell, emit_flops_per_cell, label):
    """Predict every cell's S each design iteration, assemble, solve the interface."""
    emit = emit_flops_per_cell / M['gpu_tf32_peak'] * N
    store = numbers_per_cell * 8 * N / 2**30
    # one interface matvec: 2 flops per stored number per cell
    mv = 2 * numbers_per_cell * N / M['gpu_fp64_effective']
    lo = emit + mv * A['interface_iterations'][0]
    hi = emit + mv * A['interface_iterations'][1]
    return dict(label=label, emit_seconds=emit, storage_GiB=store,
                matvec_seconds=mv, solve=(lo, hi))


if __name__ == '__main__':
    out = dict(measured=M, assumptions=A, rows=[])
    print('ROUTE 5 COSTING - per topology-optimisation iteration\n')
    print('Assumptions that carry the result (check these first):')
    print(f"  sustained fp64 for matrix-free apply : {M['gpu_fp64_effective']:.1e} FLOP/s")
    print(f"  flops per dof per matvec             : {A['matvec_flops_per_dof']}")
    print(f"  multigrid matvec-equivalents         : {A['mg_matvec_equivalents']}")
    print(f"  interface iterations                 : {A['interface_iterations']}\n")
    hdr = f"{'N cells':>9}{'route':>26}{'memory':>12}{'emit':>12}{'per TO iter':>22}"
    print(hdr); print('-' * len(hdr))
    for N in A['N_cells']:
        f = route_fine(N)
        print(f"{N:>9}{'fine: solve only':>26}{f['memory_GiB']:>10.1f} G{'-':>12}"
              f"{fmt(f['solve'][0]) + ' - ' + fmt(f['solve'][1]):>22}")
        print(f"{'':>9}{'fine: + re-cut geometry':>26}{'':>12}"
              f"{fmt(f['geometry'][0]) + '-' + fmt(f['geometry'][1]):>12}"
              f"{fmt(f['total'][0]) + ' - ' + fmt(f['total'][1]):>22}")
        for numbers, flops, label in (
                (M['dense_S_numbers'], M['predict_dense_flops'], 'predicted dense S'),
                (M['hmat_numbers_at_gate'],
                 M['predict_dense_flops'] * M['hmat_numbers_at_gate'] / M['dense_S_numbers'],
                 'predicted H-matrix S @0.15')):
            c = route_condensed(N, numbers, flops, label)
            print(f"{'':>9}{label:>26}{c['storage_GiB']:>10.1f} G{fmt(c['emit_seconds']):>12}"
                  f"{fmt(c['solve'][0]) + ' - ' + fmt(c['solve'][1]):>22}")
            out['rows'].append(dict(N=N, **{k: v for k, v in c.items()}))
        out['rows'].append(dict(N=N, label='fine matrix-free', **f))
        print()
    print('Whole optimisation run (N = 1000):')
    print(f"{'route':>28}{'M=100':>16}{'M=500':>16}")
    f = route_fine(1000)
    print(f"{'fine: solve only':>28}"
          f"{fmt(f['solve'][0]*100) + '-' + fmt(f['solve'][1]*100):>16}"
          f"{fmt(f['solve'][0]*500) + '-' + fmt(f['solve'][1]*500):>16}")
    print(f"{'fine: + re-cut geometry':>28}"
          f"{fmt(f['total'][0]*100) + '-' + fmt(f['total'][1]*100):>16}"
          f"{fmt(f['total'][0]*500) + '-' + fmt(f['total'][1]*500):>16}")
    for numbers, flops, label in (
            (M['dense_S_numbers'], M['predict_dense_flops'], 'predicted dense S'),
            (M['hmat_numbers_at_gate'],
             M['predict_dense_flops'] * M['hmat_numbers_at_gate'] / M['dense_S_numbers'],
             'predicted H-matrix S @0.15')):
        c = route_condensed(1000, numbers, flops, label)
        print(f"{label:>28}{fmt(c['solve'][0]*100) + '-' + fmt(c['solve'][1]*100):>16}"
              f"{fmt(c['solve'][0]*500) + '-' + fmt(c['solve'][1]*500):>16}")
    print(f"\n  teacher, no network, N=1000, M=1: "
          f"{fmt(M['teacher_seconds_per_cell'][0]*1000)} - {fmt(M['teacher_seconds_per_cell'][1]*1000)}")
    Path('/home/user/curly-octo-goggles/docs/data/ROUTE5_COST.json').write_text(
        json.dumps(out, indent=1, default=str))
