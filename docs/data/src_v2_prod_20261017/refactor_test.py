"""Route 7: cost of re-encoding a cell inside a design loop.

1. Fresh encode (as box_encode.py): elements, ghost from faces, direct box condensation, cuDSS plan + factorization.
2. Design step with unchanged topology (thickness corners scaled by 1 + eps; active cells and faces unchanged): new
   element values into the same sparsity pattern, the ghost matrix reused (it depends only on the faces), cuDSS
   refactorization without a new plan. Checked against a fresh encode of the perturbed design.
3. Single-precision factorization with double-precision iterative refinement: time and accuracy of T q.
Usage: refactor_test.py <case> <body_dir> <out_dir>
"""
import json, sys, time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import encode_r1 as E
import box_encode as BX

dev, dt = E.dev, E.dt


def sync():
    torch.cuda.synchronize()


def scaled_upper(A):
    i, v = A.indices(), A.values()
    dg = torch.zeros(A.shape[0], dtype=dt, device=dev).index_add_(0, i[0][i[0] == i[1]], v[i[0] == i[1]])
    s = 1 / torch.sqrt(dg)
    up = i[0] <= i[1]
    U = torch.sparse_coo_tensor(i[:, up], v[up] * s[i[0][up]] * s[i[1][up]], A.shape).coalesce().to_sparse_csr()
    return s, U


def build_K(case, body_dir, eps, G=None):
    E.TAU_EPS = str(eps)
    T = E.Timer()
    Kb, nb, ctx = E.elements_polyref(case, 4, 4, T, 1)
    if G is None:
        G = BX.ghost_faces_gpu(body_dir, case, int(ctx['n']), nb)
    gamma = float(json.loads((BX.PACKETS / case / 'SAMPLE.json').read_text())['gp']['gamma'])
    K = (Kb + gamma * G).coalesce()
    return K, G, nb, ctx, T.rows


def main(case, body_dir, out):
    import os
    os.environ['G_BODY_DIR'] = str(Path(body_dir) / case)
    import element_moments as EM
    import cross_case_chebyshev as X
    sys.path.insert(0, str(E.MN / 'source_v1'))
    runner = X.load('xc_run_mechanics', X.FROZEN['run_mechanics'])
    rec = dict(case=case)
    # warm-up of libraries (imports, cuDSS handles) on a first full encode, then time a second one
    sync(); t = time.perf_counter()
    K, G, nb, ctx, rows = build_K(case, body_dir, 0)
    n = int(ctx['n'])
    nodes = EM.members(case, ['NODES.npy'])['NODES.npy']
    grid = np.stack(np.unravel_index(nodes, (2 * n + 1,) * 3), axis=1)
    b_mask = torch.as_tensor(np.repeat(np.any((grid == 0) | (grid == 2 * n), axis=1), 3), device=dev)
    A, C, D, bi, ii = BX.split(K, b_mask)
    del K; torch.cuda.empty_cache()
    big = A.shape[0] > 100000                     # large cells: no second factorization held for the comparison
    R = torch.as_tensor(BX.rigid_fields(grid / (2 * n)), dtype=dt, device=dev)
    ub, ui = runner.orthonormalize_rigid_pair(R[bi], R[ii])
    T0 = E.Timer()
    model = E.DirectModel(A, C, D, ub, ui, T0)
    sync(); rec['first_encode_seconds'] = time.perf_counter() - t
    rec['first_stage_seconds'] = dict(rows, **T0.rows)
    # 2. design step: same pattern, reuse ghost matrix and cuDSS plan
    q = torch.randn((D.shape[0], 16), dtype=dt, device=dev)
    for eps in ('1/1000000', '1/10000'):
        pattern = (A.indices().clone(), C.indices().clone(), D.indices().clone()) if not big else None
        model.A = model.C = model.D = model.Ct = None
        del A, C, D
        torch.cuda.empty_cache()
        sync(); t = time.perf_counter()
        T1 = E.Timer()
        with T1('R1_elements'):
            K2, _, _, _, rows2 = build_K(case, body_dir, eps, G=G)
        with T1('R2_split'):
            A2, C2, D2, _, _ = BX.split(K2, b_mask)
            del K2
        same_pattern = None if pattern is None else bool(torch.equal(A2.indices(), pattern[0]) and torch.equal(C2.indices(), pattern[1])
                                                         and torch.equal(D2.indices(), pattern[2]))
        del pattern
        with T1('R3_values'):
            s2, U2 = scaled_upper(A2)
            same_csr = bool(torch.equal(U2.crow_indices().int(), model.U.crow_indices()) and torch.equal(U2.col_indices().int(), model.U.col_indices()))
            model.U.values().copy_(U2.values())        # in place: cuDSS keeps its pointers, the plan stays valid
        with T1('R4_refactor'):
            model.solver.factorize()
        model.A, model.C, model.D, model.s = A2, C2, D2, s2
        model.Ct = C2.t().coalesce()
        sync(); step_seconds = time.perf_counter() - t
        with torch.no_grad():
            y_re = model(q)
        # reference: fresh encode of the perturbed design (small cells only; memory)
        if not big:
            Tf = E.Timer()
            fresh = E.DirectModel(A2, C2, D2, ub, ui, Tf)
            with torch.no_grad():
                y_fr = fresh(q)
            del fresh; torch.cuda.empty_cache()
            vs_fresh = float((y_re - y_fr).norm() / y_fr.norm())
        else:
            vs_fresh = None
        rec[f'step_{eps}'] = dict(seconds=step_seconds, stages=dict(rows2, **T1.rows), same_pattern=same_pattern, same_csr=same_csr,
                                  refactor_vs_fresh=vs_fresh)
        print(json.dumps({k: v for k, v in rec.items() if k.startswith('step') or k.startswith('first')}, default=float)[-600:], flush=True)
        A, C, D = A2, C2, D2
    # 3. single-precision factor + double-precision refinement
    torch.cuda.empty_cache()
    from nvmath.sparse.advanced import (DirectSolver, DirectSolverOptions, DirectSolverMatrixType, DirectSolverMatrixViewType)
    import os as _os
    lib = _os.environ.get('CUDSS_MT')
    opts = DirectSolverOptions(sparse_system_type=DirectSolverMatrixType.SPD, sparse_system_view=DirectSolverMatrixViewType.UPPER,
                               **(dict(multithreading_lib=lib) if lib else {}))
    U32 = torch.sparse_csr_tensor(model.U.crow_indices(), model.U.col_indices(), model.U.values().float(), size=model.U.shape)
    w = 8
    b32 = torch.zeros((w, A.shape[0]), dtype=torch.float32, device=dev).T
    s32 = DirectSolver(U32, b32, options=opts)
    if lib:
        s32.plan_config.host_nthreads = 16
    torch.cuda.empty_cache()
    sync(); t = time.perf_counter(); s32.plan(); sync(); plan32 = time.perf_counter() - t
    sync(); t = time.perf_counter(); s32.factorize(); sync(); fac32 = time.perf_counter() - t
    sync(); t = time.perf_counter(); model.solver.factorize(); sync(); fac64 = time.perf_counter() - t
    # scaled system: (s A s) y = s r, x = s y ; refine in double precision
    Asc = torch.sparse_coo_tensor(A.indices(), A.values() * model.s[A.indices()[0]] * model.s[A.indices()[1]], A.shape).coalesce()
    def solve32(r):
        b = torch.zeros((w, A.shape[0]), dtype=torch.float32, device=dev).T
        b[:, :r.shape[1]] = r.float()
        s32.reset_operands(b=b)
        return s32.solve()[:, :r.shape[1]].double()
    rhs = torch.randn((A.shape[0], 8), dtype=dt, device=dev)
    ref = model.solve_raw(rhs)
    hist = []
    sync(); t = time.perf_counter()
    rs = model.s[:, None] * rhs
    y = solve32(rs)
    for it in range(4):
        r = rs - torch.sparse.mm(Asc, y)
        y = y + solve32(r)
        x = model.s[:, None] * y
        hist.append(float((x - ref).norm() / ref.norm()))
    sync(); refine_seconds = time.perf_counter() - t
    sync(); t = time.perf_counter(); model.solve_raw(rhs); sync(); solve64 = time.perf_counter() - t
    rec['fp32'] = dict(plan_seconds=plan32, factor_seconds=fac32, factor64_seconds=fac64, refine_history=hist,
                       refine4_seconds_8rhs=refine_seconds, solve64_seconds_8rhs=solve64)
    print(json.dumps(rec['fp32']), flush=True)
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / f'REFACTOR_{case}.json').write_text(json.dumps(rec, indent=2, default=float))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])
