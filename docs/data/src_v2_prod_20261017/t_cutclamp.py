"""CPU tests of the opt-in cut-band clamp (lat_multi.MultiLattice clamp=('cut', ''), opt_design.py --clamp cut), on the
synthetic Q1 cells of t_lat_precond.py (ports = box nodes + private interior 'cut-band' nodes + box nodes flagged cut).
Checks: (1) the default face clamp is unchanged against the pre-change lat_multi (lat_multi.py.bak_cplate_20261002):
same free DOFs, numbering, gathers and loads; (2) the cut clamp fixes exactly the union of the cells' cut-band DOFs (a
shared box node is clamped if any cell holds it in its band) and nothing else; (3) the assembled operator on the free DOFs
equals the restriction of an independently assembled dense lattice matrix and is SPD; (4) every preconditioner of
t_lat_precond.SPECS stays symmetric positive definite and PCG with bnn:kpp:q1r reaches the dense solution; (5) the
consistent / uniform loads vanish on clamped DOFs (the load face is away from the band).
Usage: OPL_DEV=cpu python t_cutclamp.py
"""
import os
os.environ.setdefault('OPL_DEV', 'cpu')
import sys
import json
import importlib.util
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lat_multi as LM
import lat_precond as PR
import t_lat_precond as TP

dt, CPU = torch.float64, torch.device('cpu')


def _old_lm():
    import importlib.machinery
    ld = importlib.machinery.SourceFileLoader('lat_multi_old', str(HERE / 'lat_multi.py.bak_cplate_20261002'))
    m = importlib.util.module_from_spec(importlib.util.spec_from_loader('lat_multi_old', ld)); ld.exec_module(m)
    return m


def _cells(shape=(3, 2, 1), m=4, contrast=1.0, seeds=(0, 1, 2)):
    kinds = [TP.synth_cell(m, s, contrast, n_priv=3, n_box_cut=6) for s in seeds]
    lo = {p: kinds[(p[0] + 2 * p[1] + p[2]) % len(kinds)][:2] for p in np.ndindex(*shape)}
    return lo, kinds


def test_default_unchanged():
    OLD = _old_lm()
    lo, _ = _cells()
    out = {}
    for clamp, load in ((('x', 'min'), ('x', 'max')), (('y', 'min'), ('y', 'max')), (('x', 'max'), ('y', 'min'))):
        for loads in ('uniform',):
            lay = {p: g for p, (g, _) in lo.items()}
            a = LM.MultiLattice(lay, clamp=clamp, load=load, loads=loads, n_random=1, device=CPU, log=lambda s_: None)
            b = OLD.MultiLattice(lay, clamp=clamp, load=load, loads=loads, n_random=1, device=CPU, log=lambda s_: None)
            assert a.N == b.N and np.array_equal(a.free, b.free) and torch.equal(a.F, b.F)
            assert all(torch.equal(x, y) for x, y in zip(a.gather_idx, b.gather_idx))
            assert np.array_equal(a.priv, b.priv)
            out[f'{clamp}->{load}'] = dict(free=a.nfree, clamped=a.n_clamped)
    return out


def _dense_full(lat, lo, kinds):
    """Independent dense assembly of the lattice operator on ALL glued DOFs (no clamp) from the cells' Schur blocks."""
    A = torch.zeros(lat.N, lat.N, dtype=dt)
    for i, (p, (G, op)) in enumerate(lo.items()):
        g = torch.as_tensor(lat.idx[i])
        A[g[:, None], g[None, :]] += op.S
    return A


def test_cut_clamp_set_and_operator():
    lo, kinds = _cells()
    lay = {p: g for p, (g, _) in lo.items()}
    lat = LM.MultiLattice(lay, clamp=('cut', ''), load=('y', 'max'), loads='uniform', n_random=0, device=CPU,
                          log=lambda s_: None)
    ops = [op for (_, op) in lo.values()]
    # (2) clamped set = union of cut-band DOFs, by an independent loop over cells and port nodes
    want = np.zeros(lat.N, bool)
    for i, (p, (G, _)) in enumerate(lo.items()):
        for r in range(G.nport):
            if G.cut[r // 3]:
                want[lat.idx[i][r]] = True
    got = np.ones(lat.N, bool); got[lat.free] = False
    assert np.array_equal(got, want) and lat.n_clamped == int(want.sum())
    shared = sum(int(G.cut[~G.priv].sum()) for G, _, _ in kinds)
    assert shared > 0                                                    # the test covers cut-flagged box nodes
    # (3) operator = restriction of the dense assembly; SPD
    Afull = _dense_full(lat, lo, kinds)
    fr = torch.as_tensor(lat.free)
    A = TP.dense_A(lat, ops)
    rel = float((A - Afull[fr[:, None], fr[None, :]]).norm() / A.norm())
    ev = torch.linalg.eigvalsh((A + A.T) / 2)
    assert rel < 1e-13 and float(ev.min()) > 0
    # (5) loads vanish on clamped DOFs: F lives on the free DOFs only; total force preserved (unit per column)
    assert torch.allclose(lat.F.sum(0), torch.ones(3, dtype=dt))
    return dict(N=lat.N, free=lat.nfree, clamped=lat.n_clamped, private=int(sum(G.priv.sum() for G, _, _ in kinds)),
                restriction_rel=rel, A_min_eig=float(ev.min()), A_cond=float(ev.max() / ev.min()))


def test_cut_clamp_preconditioners_pcg():
    lo, kinds = _cells((3, 3, 1), contrast=1.5)
    lay = {p: g for p, (g, _) in lo.items()}
    lat = LM.MultiLattice(lay, clamp=('cut', ''), load=('y', 'min'), loads='uniform', n_random=0, device=CPU,
                          log=lambda s_: None)
    ops = [op for (_, op) in lo.values()]
    A = TP.dense_A(lat, ops)
    I = torch.eye(lat.nfree, dtype=dt)
    fac = PR.Factory(lat, ops)
    out = {}
    for spec in TP.SPECS:
        pc, st, _ = fac.build(spec)
        if isinstance(pc, PR.Deflated):
            Pm = pc.coarse.proj_T(I); S = Pm.T @ A @ pc(A @ Pm)
            out[spec] = dict(A_sym=float((S - S.T).norm() / S.norm())); assert out[spec]['A_sym'] < 1e-9
            continue
        M = pc(I)
        sym = float((M - M.T).norm() / M.norm()); ev = torch.linalg.eigvalsh((M + M.T) / 2)
        out[spec] = dict(sym=sym, min_eig=float(ev.min()))
        assert sym < 1e-10 and float(ev.min()) > 0, (spec, out[spec])
    F = lat.F[:, :2].contiguous()
    X_ref = torch.linalg.solve(A, F)
    pc, _, _ = fac.build('bnn:kpp:q1r')
    r = PR.pcg(lat, ops, pc, F=F, tol=1e-12, maxit=2000)
    err = float((r['X'] - X_ref).norm() / X_ref.norm())
    assert err < 1e-9
    out['pcg'] = dict(iterations=int(r['iterations']), sol_rel_err=err,
                      compliance=[float(x) for x in (F * X_ref).sum(0)])
    return out


if __name__ == '__main__':
    res, ok = {}, True
    for name in ('test_default_unchanged', 'test_cut_clamp_set_and_operator', 'test_cut_clamp_preconditioners_pcg'):
        try:
            res[name] = globals()[name](); print(name, 'PASS', json.dumps(res[name], default=float)[:600], flush=True)
        except Exception as e:
            ok = False; print(name, 'FAIL', repr(e), flush=True)
    print('ALL', 'PASS' if ok else 'FAIL')
    sys.exit(0 if ok else 1)
