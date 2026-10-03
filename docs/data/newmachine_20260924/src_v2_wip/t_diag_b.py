"""Tests of diag_b (A5 diag_cert.py, A6 diag_mu.py, A7 lat_full.py, A8 diag_fringe.py).
Local (CPU): A5 and A8 end to end on the fixture small geometry (fresh_train_0010_cover01_r1), the factorization-free
pieces of A6 / A7 on the fixture and on toy two-cell lattices built through lattice3.build. Test-only stand-ins:
teacher.SPDSolver -> scipy LU (fp64) and teacher.sync -> no-op (the teacher's own factor / extend / neumann / apply paths
then run on the CPU); lattice3.prepared -> toy cells.
Remote (GPU, not run here; after the v1 training): the GPU-only checks with explicit pass criteria, see remote().
Usage: OPL_DEV=cpu python3 t_diag_b.py            (local)
       python3 t_diag_b.py --remote [--s1 DIR] [--only a5,a6,a7,a8] [--out FILE]   (remote GPU, LAT_CPU=1 recommended)"""
import os, sys, json, time, tempfile, warnings
REMOTE = '--remote' in sys.argv
if not REMOTE:
    os.environ['OPL_DEV'] = 'cpu'
os.environ.pop('LAT_LOADS', None)
warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np
import torch
import scipy.sparse as sp
import scipy.linalg as sla
from scipy.sparse.linalg import splu
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fixture as FX
import teacher as TE
import trainlib as TL
import lattice3 as LT
import ops as OP
import diag_cert as DC
import diag_fringe as DF
import diag_mu as DM
import lat_full as LF

dt = torch.float64
FAIL, T0 = [], time.perf_counter()
if not REMOTE:
    torch.set_num_threads(1)                                                         # CPU sparse products: 6x faster than 4 contending threads here


def check(name, ok, **info):
    print(json.dumps(dict(test=name, ok=bool(ok), **{k: (round(v, 8) if isinstance(v, float) else v) for k, v in info.items()}),
                     default=str), flush=True)
    if not ok:
        FAIL.append(name)


# ------------------------------------------------------------------------------------------ test-only stand-ins
class SciSPD:
    """teacher.SPDSolver interface (upper CSR, SPD) on scipy's sparse LU (fp64); an fp32 solver (fdt) rounds its solution to
    fp32, so the callers' fp64 iterative refinement (teacher.Cell.extend, diag_mu.dirichlet_solver) is exercised."""

    def __init__(self, crow, col, vals, n, w=16, threads=16, fdt=None):
        self.fdt = vals.dtype if fdt is None else fdt
        U = sp.csr_matrix((vals.double().cpu().numpy(), col.long().cpu().numpy(), crow.long().cpu().numpy()), shape=(n, n))
        self.lu = splu((U + U.T - sp.diags(U.diagonal())).tocsc())

    def solve(self, r):
        x = torch.as_tensor(self.lu.solve(r.double().cpu().numpy()))
        return x.to(self.fdt).to(r.dtype)

    def free(self):
        pass


if not REMOTE:
    TE.SPDSolver = SciSPD
    TE.sync = lambda: None
try:
    import cert as CE
    CERT = 'cert.py'
except ImportError:                                                                  # minimal stand-in (Jacobi / Krylov Galerkin)
    CERT = 'stand-in'

    class _Cert:
        def __init__(self, C, P, I):
            self.C, self.I, self.nb = C, I, C.nb
            d = torch.zeros(C.nb, dtype=dt); d[:] = C.vals[C.diag].to(dt)
            self.d = d[I]

        def kii(self, w):
            x = torch.zeros((self.nb, w.shape[1]), dtype=dt); x[self.I] = w
            return (self.C @ x)[self.I]

        def lower(self, u, m=0):
            r = (self.C @ u.to(dt))[self.I]
            out = []
            for b in range(r.shape[1]):
                W = [r[:, b:b + 1] / self.d[:, None]]
                for _ in range(m):
                    W.append(self.kii(W[-1]) / self.d[:, None])
                W = torch.linalg.qr(torch.cat(W, 1))[0]
                G = W.T @ self.kii(W); bb = W.T @ r[:, b]
                out.append(bb @ torch.linalg.lstsq(G, bb).solution)
            return torch.stack(out)

        def mu_lower(self, u, m=8):
            e = (u.to(dt) * (self.C @ u.to(dt))).sum(0); lb = self.lower(u, m)
            return e / (e - lb), lb / (e - lb)

    class CE:
        Cert = _Cert

        @staticmethod
        def from_geo(geo):
            return _Cert(geo.C, geo.P, geo.I)
    sys.modules['cert'] = CE


# ------------------------------------------------------------------------------------------ toy cells (n = 32 grid, coarse nodes)
class ToyCell:
    """teacher.Cell look-alike for lattice3: m^3 nodes on the 65^3 grid spanning the cube, ports = box nodes, dense K with
    exactly the 6 rigid modes as null space, exact extension E and port Schur complement T."""

    def __init__(self, seed, m=5, n=32):
        g1 = np.linspace(0, 2 * n, m).astype(np.int64)
        G = np.stack(np.meshgrid(g1, g1, g1, indexing='ij'), -1).reshape(-1, 3)
        ids = np.ravel_multi_index(G.T, (2 * n + 1,) * 3); o = np.argsort(ids); G, ids = G[o], ids[o]
        self.n, self.nodes = n, ids
        box = (G == 0).any(1) | (G == 2 * n).any(1)
        self.is_box, self.is_cut = box, np.zeros(len(ids), bool)
        self.port_node_ids, self.port_is_box, self.port_is_cut = ids[box], np.ones(box.sum(), bool), np.zeros(box.sum(), bool)
        pm = torch.as_tensor(np.repeat(box, 3))
        self.P, self.I = torch.nonzero(pm).squeeze(1), torch.nonzero(~pm).squeeze(1)
        self.nb, self.np_, self.ni = 3 * len(ids), len(self.P), len(self.I)
        self.Q = TE.rigid_basis(self.port_node_ids, n); Qa = TE.rigid_basis(ids, n)
        rng = np.random.default_rng(seed)
        A = torch.as_tensor(rng.standard_normal((self.nb, self.nb)))
        Pi = torch.eye(self.nb, dtype=dt) - Qa @ Qa.T
        self.Kd = Pi @ (A @ A.T / self.nb + 0.5 * torch.eye(self.nb, dtype=dt)) @ Pi
        self.Kd = 0.5 * (self.Kd + self.Kd.T)
        KII, KIP = self.Kd[self.I][:, self.I], self.Kd[self.I][:, self.P]
        self.Eint = -torch.linalg.solve(KII, KIP)
        self.T = self.Kd[self.P][:, self.P] + self.Kd[self.P][:, self.I] @ self.Eint
        self.T = 0.5 * (self.T + self.T.T)
        B = torch.as_tensor(rng.standard_normal((8, self.nb, 3)))
        self.A8 = B @ B.transpose(1, 2)
        self.sol_I = 'dense'

    def __matmul__(self, x):
        return self.Kd @ x

    def extend(self, q):
        x = torch.zeros((self.nb, q.shape[1]), dtype=dt); x[self.P] = q; x[self.I] = self.Eint @ q
        return x

    def factor(self, **kw):
        self.sol_I = 'dense'

    def _free(self):
        self.sol_I = None

    def sens(self, u):
        return -torch.einsum('ik,cij,jk->ck', u, self.A8, u)


class ToyNet:
    """A 'learned' extension of a toy cell: E_hat = E + J N (I - Q Q^T) (rigid part exact), variational S_hat = E_hat^T K E_hat."""

    def __init__(self, cell, eta, seed):
        g = torch.Generator().manual_seed(seed)
        N = eta * torch.randn((cell.ni, cell.np_), dtype=dt, generator=g) @ (torch.eye(cell.np_, dtype=dt) - cell.Q @ cell.Q.T)
        self.E = torch.zeros((cell.nb, cell.np_), dtype=dt); self.E[cell.P] = torch.eye(cell.np_, dtype=dt); self.E[cell.I] = cell.Eint + N
        self.S = self.E.T @ cell.Kd @ self.E; self.S = 0.5 * (self.S + self.S.T)

    def field(self, q):
        return self.E @ q

    def apply(self, q):
        return self.S @ q


def top_eig(A, B, basis=None):
    if basis is not None:
        A, B = basis.T @ A @ basis, basis.T @ B @ basis
    return sla.eigh(A.numpy(), B.numpy(), eigvals_only=True)[::-1]


def toy_lattice(conf, cells):
    LT.prepared = lambda case, body: (cells[case], cells[case].T)                    # lattice3.build's offsets and gluing unchanged
    return LT.build('toyT', 'toyN', conf, '')


# ======================================================================================== remote (GPU) checks
def remote(argv):
    """GPU checks with explicit pass criteria (known numbers: PROGRESS_20260924_CN.md, r2 d2 arm).
    a5  diag_cert on mgno2_r2_d2 / its own geometry: no bound violation, lb0 <= lb8 <= lb16 (nested Galerkin spaces),
        bank error of the own geometry in [0.3%, 5%] (d2 val bank 0.6-2.1%: right checkpoint / geometry pairing)
    a6  diag_mu pieces on r2 d2: block_ritz converges (< maxit), mu >= 1.25 (train1's unconverged estimate 1.31 is a lower
        bound), exact Ritz within 2% of the fp32-factor Ritz value, mu_-F <= mu (1 + 1e-3) for x and y,
        Dirichlet-F factor (fp64): q_F = 0 and |(S q)_G - y_G| / |y_G| <= 1e-6 (S from the interior factor); the fp32 +
        refinement residual is reported
    a7  lat_full.run_lattice on (r2 d2, FULL v0), config y: the 'test' set reproduces the gate of the d2 arm
        (compliance 1.29%, sensitivity max 3.10%, both +-0.05 points), 0 <= compliance error <= sum_c w_c eps_c in every
        set, energy shares sum to 1
    a8  diag_fringe.run_geo on the first val geometry of SPLIT.json: exact candidate excess <= 1e-10 and sensitivity
        error <= 1e-8, unit exact energy of the solved fields (|u*^T K u* - 1| <= 1e-5), affine / root maps reproduce
        affine / tri-quadratic fields to 1e-9"""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--remote', action='store_true'); ap.add_argument('--s1', default=DC.S1)
    ap.add_argument('--only', default='a5,a6,a7,a8'); ap.add_argument('--out', default='T_DIAG_B_REMOTE.json')
    a = ap.parse_args(argv)
    only = a.only.split(','); rec = {}
    import cert as CE_
    s1 = Path(a.s1)
    ck_r2 = DC.load_ckpt(s1 / DC.RUNS['r2'] / 'best.pt')
    if 'a5' in only:
        g_ = DC.load_geo(DC.R2)
        m_, miss = DC.net_for(ck_r2, g_)
        raw = DC.cert_case(g_, DC.field_fn(m_, g_), CE_.from_geo(g_), (0, 8, 16))
        viol = sum(DC.summarize(rw, (0, 8, 16))['m'][str(m)]['violations'] for rw in raw.values() for m in (0, 8, 16))
        mono = all(bool((rw['lb16'] >= rw['lb8'] * (1 - 1e-6)).all() and (rw['lb8'] >= rw['lb0'] * (1 - 1e-6)).all()) for rw in raw.values())
        tm = {c: float(rw['dtrue'].mean()) for c, rw in raw.items()}
        check('R_A5', viol == 0 and mono and miss == 0 and all(0.003 <= v <= 0.05 for v in tm.values()), violations=viol, monotone=mono,
              true_mean=tm, eff16={c: float(np.median(rw['lb16'] / rw['dtrue'])) for c, rw in raw.items()},
              seconds={c: rw['seconds'] for c, rw in raw.items()})
        rec['a5'] = dict(true_mean=tm); del g_, m_; DC.free_mem()
    if 'a6' in only or 'a7' in only:
        import evalnet as EN_
        import fastnet as FN_
        cfg = ck_r2['cfg']
        C_, _ = LT.prepared(DC.R2, cfg['body'])
        g_ = TL.Geo(DC.R2, cfg['body'], cfg['data'], neumann=False, log=lambda s_: None, cell=C_, load_banks=False)
        op_ = EN_.FastOp(FN_.FastNet(DC.net_for(ck_r2, g_)[0], g_))
    if 'a6' in only:
        X0_ = torch.randn((C_.np_, 16), dtype=dt, device=DC.dev, generator=torch.Generator(device=DC.dev).manual_seed(0))
        C_.factor(neumann=True, interior=False, fp32_neumann=True)
        X_, r_, SH_, h_ = DM.block_ritz(op_.apply, C_.neumann, DM.rigid_proj(C_.Q), X0_, 1e-4, 60)
        C_._free()
        blocks, muF, dres = {}, {}, {}
        for conf, ax in DM.F_AXIS.items():
            fp_ = DM.face_ports(C_, ax)
            y_ = torch.randn((C_.np_, 4), dtype=dt, device=DC.dev); y_[fp_] = 0
            solve_, free_ = DM.dirichlet_solver(C_, fp_, fp32=False)                     # fp64 factor: construction check
            qd64 = solve_(y_); free_(); DC.free_mem()
            solve_, free_ = DM.dirichlet_solver(C_, fp_)                                  # fp32 + refinement: what main() uses
            qd32 = solve_(y_)
            blocks[conf] = DM.block_ritz(op_.apply, solve_, DM.zero_proj(fp_), X0_, 1e-4, 60)
            free_(); DC.free_mem()
            C_.factor(neumann=False, fp32=True)
            G_ = torch.ones(C_.np_, dtype=torch.bool, device=DC.dev); G_[fp_] = False
            rel_ = lambda q__: float((C_.apply(q__)[G_] - y_[G_]).norm() / y_[G_].norm())
            dres[conf] = (float(qd64[fp_].abs().max()), rel_(qd64), rel_(qd32))
            muF[conf] = float(DM.exact_ritz(blocks[conf][0], C_.apply(blocks[conf][0]), blocks[conf][2])[0])
            C_._free()
        C_.factor(neumann=False, fp32=True)
        mu_ex = float(DM.exact_ritz(X_, C_.apply(X_), SH_)[0]); C_._free()
        check('R_A6', len(h_) < 60 and mu_ex >= 1.25 and abs(mu_ex - float(r_[0])) <= 0.02 * mu_ex
              and all(v <= mu_ex * (1 + 1e-3) for v in muF.values()) and all(z == 0 and e_ <= 1e-6 for z, e_, _ in dres.values()),
              mu=float(r_[0]), mu_exact=mu_ex, iterations=len(h_), mu_F=muF, dirichlet=dres)
        rec['a6'] = dict(mu=mu_ex, mu_F=muF)
    if 'a7' in only:
        ck_f = DC.load_ckpt(s1 / DC.RUNS['full'] / 'best.pt')
        Cn_, _ = LT.prepared(DC.FULL, ck_f['cfg']['body'])
        opn_, _ = LF.fast_op(ck_f, DC.FULL, ck_f['cfg']['body'], ck_f['cfg']['data'], Cn_)
        os.environ['LAT_LOADS'] = 'consistent'
        lat_ = LT.build(DC.R2, DC.FULL, 'y', cfg['body'])
        out_ = LF.run_lattice(lat_, [op_, opn_], [OP.ExactOp(cd['cell'], cd['T']) for cd in lat_.cells])
        os.environ.pop('LAT_LOADS', None)
        t_ = out_['test']
        check('R_A7', abs(t_['gate_compliance_max'] - 0.0129) <= 5e-4 and abs(t_['gate_sens_max'] - 0.0310) <= 5e-4
              and all(out_[s]['bound_holds'] for s in ('test', 'nbr', 'both')) and np.allclose(np.asarray(out_['both']['w']).sum(0), 1, atol=1e-8),
              test=(t_['gate_compliance_max'], t_['gate_sens_max']), both=(out_['both']['gate_compliance_max'], out_['both']['gate_sens_max']),
              nbr=(out_['nbr']['gate_compliance_max'], out_['nbr']['gate_sens_max']), additivity=out_.get('additivity'))
        rec['a7'] = out_; del lat_; DC.free_mem()
    if 'a8' in only:
        case = json.loads(Path(DC.SPLIT).read_text())['val'][0]
        g_ = DC.load_geo(case)
        tp = DF.Topo(g_.nd); Xg = tp.g / 64.0
        iA_, jA_, wA_, _ = tp.affine(tp.W); iR_, jR_, wR_, has_, _ = tp.root(tp.W)
        aff_ = torch.as_tensor(np.stack([1 + Xg @ np.array([.3, -.2, .5]), Xg @ np.array([.1, .7, -.4]), 2 - Xg[:, 2]], 1).reshape(-1, 1))
        qd_ = torch.as_tensor(np.stack([Xg[:, 0] ** 2 * Xg[:, 1], Xg[:, 1] * Xg[:, 2] ** 2, Xg[:, 0] * Xg[:, 2]], 1).reshape(-1, 1))
        e_aff = float((DF.apply_map(iA_, jA_, wA_, aff_, tp.W) - aff_[DF.dofs(tp.W)]).abs().max())
        e_quad = float((DF.apply_map(iR_, jR_, wR_, qd_, tp.W).reshape(-1, 3)[has_] - qd_.reshape(-1, 3)[tp.W[has_]]).abs().max())
        g_.C.factor(neumann=False, fp32=True)
        r8 = DF.run_geo(g_, g_.C.extend, n_train=64, log=lambda s_: None)
        g_.C._free()
        ok = 'classes' in r8
        ex = max(r8['classes'][c]['exact']['excess']['max'] for c in r8['classes']) if ok else None
        exs = max(r8['classes'][c]['exact']['sens']['max'] for c in r8['classes']) if ok else None
        en = max(r8['classes'][c]['check']['energy_minus_1_absmax'] for c in r8['classes']) if ok else None
        check('R_A8', ok and ex <= 1e-10 and exs <= 1e-8 and en <= 1e-5 and e_aff <= 1e-9 and e_quad <= 1e-9, case=case, exact_excess=ex,
              exact_sens=exs, energy=en, affine_repro=e_aff, root_repro=e_quad, worst=r8.get('worst_class_mean'), decision=r8.get('decision'))
        rec['a8'] = r8
    Path(a.out).write_text(json.dumps(dict(failed=FAIL, **rec), indent=1, default=str))
    print(json.dumps(dict(event='REMOTE_DONE', failed=FAIL)), flush=True)
    return 1 if FAIL else 0


if REMOTE:
    sys.exit(remote(sys.argv[1:]))

# ======================================================================================== fixture
geo = FX.small()
C = geo.C
print(json.dumps(dict(fixture=geo.case, dofs=geo.nb, ports=geo.np_, cert=CERT)), flush=True)

# ------------------------------------------------------------------------------------------ teacher paths with the stand-in
C.factor(neumann=False, fp32=True)
us = C.extend(geo.banks['val']['force'].to(dt))
check('stand_in_extend_matches_fixture_ustar', float((us - geo.fix_ustar_val['force']).norm() / geo.fix_ustar_val['force'].norm()) < 1e-8,
      rel=float((us - geo.fix_ustar_val['force']).norm() / geo.fix_ustar_val['force'].norm()))
Sq = C.apply(geo.banks['val']['force'].to(dt))
check('stand_in_apply_matches_fixture_react', float((Sq - geo.fix_react_val['force']).norm() / geo.fix_react_val['force'].norm()) < 1e-8)
C._free()

# ======================================================================================== A5 diag_cert
m_s2 = FX.model_for(geo, 's2v1_snap_10000.pt'); m_r2 = FX.model_for(geo, 'mgno2_r2_d2_best.pt')
ct = CE.from_geo(geo)
f_s2 = DC.field_fn(m_s2, geo, True)
u = f_s2(geo.banks['val']['force'][:, :8].to(dt))
mu_lb, eps_lb = ct.mu_lower(u, 8)
lb = ct.lower(u, 8)
e = (u * (C @ u)).sum(0)
check('A5_eps_from_matches_cert_mu_lower', np.allclose(DC.eps_from(lb.numpy(), e.numpy()), eps_lb.numpy(), rtol=1e-9, atol=1e-14))
check('A5_fast_field_equals_geo_field', float(((u - DC.field_fn(m_s2, geo, False)(geo.banks['val']['force'][:, :8].to(dt))).norm(dim=0)
                                                / u.norm(dim=0)).max()) < 1e-4)

# synthetic 'own-geometry' fields: exact extension + interior error at 2% energy (rough and smooth)
gen = torch.Generator().manual_seed(0)
xyz = torch.as_tensor(geo.nd['grid'], dtype=dt).repeat_interleave(3, 0) / 64


def own_field(kind, eps=0.02):
    def fld(q):
        us_ = C.extend(q)
        d = torch.zeros_like(us_)
        if kind == 'rough':
            d[geo.I] = torch.randn((len(geo.I), q.shape[1]), dtype=dt, generator=gen)
        else:
            kv = torch.randn((3, q.shape[1]), dtype=dt, generator=gen)                    # one smooth wave per column (per-DOF coordinates)
            ph = torch.rand((1, q.shape[1]), dtype=dt, generator=gen) * 6.3
            d[geo.I] = torch.sin(2 * np.pi * xyz[geo.I] @ kv + ph)
        return us_ + d * torch.sqrt(eps / (d * (C @ d)).sum(0))[None, :]
    return fld


C.factor(neumann=False, fp32=True)
ms = (0, 8, 16)
rows, raws = [], []
for part, tag, fld in (('own', 'rough2pct', own_field('rough')), ('own', 'smooth2pct', own_field('smooth')),
                       ('zeroshot', 'mgno2_r2_d2', DC.field_fn(m_r2, geo)), ('step2', 's2v1_10k', f_s2)):
    t = time.perf_counter()
    raw = DC.cert_case(geo, fld, ct, ms)
    for c, rw in raw.items():
        r = dict(key=f'{part}|{tag}|{geo.case}|{c}', part=part, tag=tag, case=geo.case, cls=c, **DC.summarize(rw, ms))
        rows.append(r); raws.append(rw)
    viol = sum(r['m'][str(m)]['violations'] for r in rows[-len(raw):] for m in ms)
    check(f'A5_cert_case_{tag}', viol == 0, violations=viol, seconds=time.perf_counter() - t,
          true_mean={c: round(rows[-len(raw) + i]['true_mean'], 4) for i, c in enumerate(raw)},
          eff16={c: round(rows[-len(raw) + i]['m']['16']['eff_median'], 3) for i, c in enumerate(raw)},
          eff0={c: round(rows[-len(raw) + i]['m']['0']['eff_median'], 3) for i, c in enumerate(raw)})
C._free()
own_true = [r['true_mean'] for r in rows if r['part'] == 'own']
check('A5_own_fields_at_2pct', max(abs(x - 0.02) for x in own_true) < 1e-6, own_true_max=max(own_true))
ana, dec = DC.analyse(rows, raws, ms, hi=0.3)
for m in ms:
    a = ana[str(m)]
    check(f'A5_analyse_m{m}', a['rule']['feasible'] and a['criteria']['all_bad_flagged'] and a['criteria']['no_own_flagged']
          and a['directions_at_t']['flagged_true_below_t'] == 0 and a['rule']['t_low'] < a['rule']['t'] <= a['rule']['t_high'],
          t=a['rule']['t'], t_low=a['rule']['t_low'], t_high=a['rule']['t_high'], spearman=a['spearman_directions'],
          spearman_cases=a['spearman_cases'], hit=a['directions_at_t']['hit_rate'], alarm=a['directions_at_t']['alarm_rate'])
check('A5_decision', 'best_m' in dec and 'learned_head_needed' in dec, **dec)
# infeasible rule: an own case above a bad case -> grid choice, reported as infeasible
bad_rows = [dict(r, part='own') if r['tag'] == 's2v1_10k' else r for r in rows]
fr = DC.flag_rule(bad_rows, 8, hi=0.3)
check('A5_flag_rule_infeasible_branch', (not fr['feasible']) and fr['own_flagged'] + len(fr['missed']) > 0, t=fr['t'])

# job plumbing and main() end to end (geometry / checkpoint loaders patched to the fixture)
with tempfile.TemporaryDirectory() as td:
    split = Path(td) / 'SPLIT.json'; split.write_text(json.dumps(dict(val=['fresh_train_0003_d0_v0', geo.case], train=[], test=[])))
    args = DC.parser().parse_args([str(Path(td) / 'o.json'), '--split', str(split), '--val-max', '2'])
    J = DC.jobs(args)
    exp = [('own', 'mgno_r1_e', DC.R1), ('own', 'mgno_full_c', DC.FULL), ('own', 'mgno2_r2_d2', DC.R2)]
    check('A5_jobs_defaults', len(J) == 3 + 7 + 2 and all(any(j[0] == p and j[1] == r and j[3] == c for j in J) for p, r, c in exp)
          and sum(j[1] == 'mgno2_r2_c' for j in J) == 3 and all(j[2].endswith('/best.pt') for j in J if j[0] != 'step2')
          and [j[2] for j in J if j[0] == 'step2'][0] == '/root/autodl-tmp/OPL/S1/s2_full/last.pt'
          and len({j[3] for j in J}) == len(list(dict.fromkeys(j[3] for j in J))), jobs=len(J))
    orders = [j[3] for j in J]
    pos_ = {c: [i for i, x in enumerate(orders) if x == c] for c in set(orders)}
    check('A5_jobs_grouped_by_case', all(max(v_) - min(v_) + 1 == len(v_) for v_ in pos_.values()))
    DC.load_geo = lambda case, *a, **k: geo
    DC.load_ckpt = lambda p: FX.checkpoint('mgno2_r2_d2_best.pt' if 'mgno2_r2' in str(p) else 's2v1_snap_10000.pt')
    t = time.perf_counter()
    rec = DC.main([str(Path(td) / 'o.json'), '--parts', 'own,zeroshot,step2', '--split', str(split), '--val-cases', geo.case,
                   '--zs', 'mgno2_r2_c:fresh_train_0031_cover01_r2,mgno_r1_e:fresh_train_0031_cover01_r1'])
    z = np.load(Path(td) / 'o.npz')
    nviol = sum(r['m'][m]['violations'] for r in rec['cases'] for m in r['m'])
    check('A5_main_end_to_end', len(rec['cases']) == 6 * 5 and 'analysis' in rec and nviol == 0 and len(z.files) == 6 * 5 * 4,
          cases=len(rec['cases']), npz=len(z.files), seconds=time.perf_counter() - t, decision=rec.get('decision'))

# ======================================================================================== A8 diag_fringe
topo = DF.Topo(geo.nd)
check('A8_topology', len(topo.W) == int((geo.nd['weak'] & ~geo.nd['is_port']).sum()) and topo.vf.max() <= 1 and (topo.vf == 1).any(),
      W=len(topo.W), E=topo.E, vf_ge_half=float((topo.vf >= .5).mean()))
iA, jA, wA, infoA = topo.affine(topo.W)
iR, jR, wR, has, infoR = topo.root(topo.W)
X = topo.g / 64.0
rng = np.random.default_rng(1)
aff = torch.as_tensor(np.stack([rng.standard_normal() + X @ rng.standard_normal(3) for _ in range(3)], 1).reshape(-1, 1))
v = DF.apply_map(iA, jA, wA, aff, topo.W)
check('A8_affine_reproduces_affine_fields', float((v - aff[DF.dofs(topo.W)]).abs().max()) < 1e-10, **infoA)
quad = np.stack([X[:, 0] ** 2 - X[:, 1] * X[:, 2], X[:, 0] * X[:, 1] * X[:, 2] ** 2, (X[:, 1] ** 2) * X[:, 2] + X[:, 0]], 1)
quad = torch.as_tensor(quad.reshape(-1, 1))
v = DF.apply_map(iR, jR, wR, quad, topo.W).reshape(-1, 3)[has]
check('A8_root_reproduces_triquadratics', float((v - quad.reshape(-1, 3)[topo.W[has]]).abs().max()) < 1e-10,
      **{k: v_ for k, v_ in infoR.items()})
grp, gi, gj, ginfo = topo.groups(1, 32)
check('A8_groups_partition_W', len(grp) == len(topo.W) and np.bincount(grp).max() <= 32 and np.bincount(grp).min() >= 1
      and not topo.weak[gj].any(), **ginfo)
Xr = torch.randn((200, 30), dtype=dt, generator=gen); M0 = torch.randn((30, 6), dtype=dt, generator=gen)
M1, lam = DF.ridge_fit(Xr, Xr @ M0, (1e-12,))
Xs = torch.randn((20, 50), dtype=dt, generator=gen); Ys = torch.randn((20, 4), dtype=dt, generator=gen)
M2, _ = DF.ridge_fit(Xs, Ys, (1e-12,))
M3, lam3 = DF.ridge_fit(Xr, Xr @ M0 + 0.1 * torch.randn((200, 6), dtype=dt, generator=gen))
check('A8_ridge_fit', float((M1 - M0).norm() / M0.norm()) < 1e-6 and float((Xs @ M2 - Ys).norm() / Ys.norm()) < 1e-6 and lam3 in (1e-8, 1e-6, 1e-4, 1e-2),
      primal=float((M1 - M0).norm() / M0.norm()), dual=float((Xs @ M2 - Ys).norm() / Ys.norm()))
check('A8_decide', DF.decide(dict(root=.002, affine=.01, ridge=1)) == 'hard_continuation' and DF.decide(dict(root=.01, affine=.05, ridge=1)) ==
      'gated_continuation' and DF.decide(dict(root=.1, affine=.05, ridge=.01)) == 'learned_local_map' and
      DF.decide(dict(root=.1, affine=.05, ridge=.1)) == 'non_local_stop')
with tempfile.TemporaryDirectory() as td:
    t = time.perf_counter()
    rec = DF.main([str(Path(td) / 'f.json'), '--cases', geo.case, '--n-train', '16'])                 # C.factor / extend via the stand-in
    g0 = rec['geometries'][0]
    ok = 'classes' in g0
    if ok:
        ex = max(g0['classes'][c]['exact']['excess']['max'] for c in g0['classes'])
        exs = max(g0['classes'][c]['exact']['sens']['max'] for c in g0['classes'])
        chk = max(g0['classes'][c]['check']['energy_minus_1_absmax'] for c in g0['classes'])
        res_ = max(g0['classes'][c]['check']['resid_W_rel_max'] for c in g0['classes'])
        lab = max(g0['classes'][c]['check']['sens_hat_vs_label']['max'] for c in g0['classes'])
        fin = all(np.isfinite(g0['classes'][c][k]['excess']['mean']) for c in g0['classes'] for k in DF.CANDS)
        order = all(g0['classes'][c]['zero']['excess']['mean'] > g0['classes'][c]['exact']['excess']['mean'] for c in g0['classes'])
        ok = ex < 1e-12 and exs < 1e-9 and chk < 1e-6 and res_ < 1e-8 and lab < 1e-2 and fin and order
        check('A8_main_end_to_end', ok, seconds=time.perf_counter() - t, exact_excess_max=ex, exact_sens_max=exs, resid=res_,
              sens_hat_vs_label=lab, decision=g0['decision'], worst={k: round(v_, 6) for k, v_ in g0['worst_class_mean'].items()},
              force={k: round(g0['classes']['force'][k]['excess']['mean'], 6) for k in DF.CANDS},
              force_sens={k: round(g0['classes']['force'][k]['sens']['mean'], 5) for k in DF.CANDS},
              root=g0['root'], ridge=g0['ridge'])
    else:
        check('A8_main_end_to_end', False, rec=g0)
# fixture ustar path of run_geo (no solve for the val fields) agrees with the solved path
C.factor(neumann=False, fp32=True)
r1 = DF.run_geo(geo, C.extend, n_train=16, ustar=geo.fix_ustar_val, log=lambda s_: None)
C._free()
check('A8_run_geo_fixture_ustar', abs(r1['worst_class_mean']['affine'] - g0['worst_class_mean']['affine']) <= 1e-6 * max(g0['worst_class_mean']['affine'], 1e-12),
      a=r1['worst_class_mean']['affine'], b=g0['worst_class_mean']['affine'])

# ======================================================================================== A6 diag_mu (pieces)
g = np.stack(np.unravel_index(C.port_node_ids, (65,) * 3), 1)
for conf, ax in DM.F_AXIS.items():
    fp = DM.face_ports(C, ax).numpy()
    nodes = fp[::3] // 3
    check(f'A6_face_ports_{conf}', len(fp) == 3 * int((C.port_is_box & (g[:, ax] == 0)).sum()) and (g[nodes, ax] == 0).all()
          and C.port_is_box[nodes].all() and (fp.reshape(-1, 3) - fp.reshape(-1, 3)[:, :1] == np.arange(3)).all(), dofs=len(fp))
# block_ritz == trainlib.Geo.adversarial (same math, one S_hat product per iteration); Neumann factor via the stand-in
C.factor(neumann=True, interior=False, fp32_neumann=True)
X0 = torch.randn((geo.np_, 4), dtype=dt, generator=torch.Generator().manual_seed(3))
m_s2d = FX.model_for(geo, 's2v1_snap_10000.pt', sparse=False)                      # dense autograd path (same weights)
shat_ag = lambda X_: geo.s_hat_apply(m_s2d, X_)
Xa, ra = geo.adversarial(m_s2d, k=4, iters=3, start=X0.clone())
Xb, rb, SHb, hb = DM.block_ritz(shat_ag, C.neumann, DM.rigid_proj(C.Q), X0.clone(), tol=0, maxit=3)
check('A6_block_ritz_equals_geo_adversarial', float(((ra - rb).abs() / ra.abs()).max()) < 1e-5, ritz=rb.tolist(),
      rel=float(((ra - rb).abs() / ra.abs()).max()))                                   # fp32 network: evaluation order only
import evalnet as EN
import fastnet as FN
op_s2 = EN.FastOp(FN.FastNet(m_s2, geo))
Xf, rf, _, _ = DM.block_ritz(op_s2.apply, C.neumann, DM.rigid_proj(C.Q), X0.clone(), tol=0, maxit=3)
check('A6_block_ritz_fastnet_vs_autograd', float(((rf - rb).abs() / rb.abs()).max()) < 1e-4, rel=float(((rf - rb).abs() / rb.abs()).max()))
Xm, rm, SHm, hm = DM.block_ritz(op_s2.apply, C.neumann, DM.rigid_proj(C.Q), X0.clone(), tol=1e-5, maxit=80)
check('A6_block_ritz_converges', len(hm) < 80 and float(rm[0]) >= float(rf[0]) * (1 - 1e-6), mu=float(rm[0]), it=len(hm))
C._free()
# Dirichlet-F factor: S_GG q_G = y_G, q_F = 0 (checked with the exact interior extension)
fp = DM.face_ports(C, 0)
y = torch.randn((geo.np_, 3), dtype=dt, generator=gen); y[fp] = 0
qs = {}
for key, kw in (('fp64', dict(fp32=False)), ('fp32_refined', {}), ('fp32_raw', dict(refine=0))):
    solve, free = DM.dirichlet_solver(C, fp, **kw)
    qs[key] = solve(y); free()
C.factor(neumann=False, fp32=True)
G_ = torch.ones(geo.np_, dtype=torch.bool); G_[fp] = False
rel = {k: float((C.apply(q_)[G_] - y[G_]).norm() / y[G_].norm()) for k, q_ in qs.items()}
check('A6_dirichlet_solver', all(float(q_[fp].abs().max()) == 0 for q_ in qs.values()) and rel['fp64'] < 1e-8
      and rel['fp32_refined'] < 1e-8 and rel['fp32_raw'] > 10 * rel['fp32_refined'], **rel)
C._free()
solve, free = DM.dirichlet_solver(C, fp)
XF, rF, SHF, hF = DM.block_ritz(op_s2.apply, solve, DM.zero_proj(fp), X0.clone(), tol=1e-5, maxit=80)
free()
C.factor(neumann=False, fp32=True)
exF, exA = DM.exact_ritz(XF, C.apply(XF), SHF), DM.exact_ritz(Xm, C.apply(Xm), SHm)
C._free()
check('A6_exact_ritz_fixture', float((exF[:4] - rF[:4]).abs().max() / rF[0]) < 1e-6 and float((exA[:4] - rm[:4]).abs().max() / rm[0]) < 1e-6
      and float(XF[fp].abs().max()) == 0, mu=float(exA[0]), mu_F=float(exF[0]), it_F=len(hF))

# toy pencils: block_ritz against dense generalized eigenvalues
tc, tn = ToyCell(1), ToyCell(2)
nt, nn = ToyNet(tc, 0.03, 11), ToyNet(tn, 0.02, 12)
Tp = torch.linalg.pinv(tc.T, hermitian=True, rtol=1e-12)
comp = torch.linalg.eigh(torch.eye(tc.np_, dtype=dt) - tc.Q @ tc.Q.T)[1][:, 6:]                   # rigid-free basis
mu_dense = top_eig(nt.S, tc.T, comp)
X0t = torch.randn((tc.np_, 16), dtype=dt, generator=torch.Generator().manual_seed(5))
Xt, rt, SHt, ht = DM.block_ritz(nt.apply, lambda f: Tp @ f, DM.rigid_proj(tc.Q), X0t, tol=1e-13, maxit=2000)
check('A6_block_ritz_toy_mu', abs(float(rt[0]) - mu_dense[0]) / mu_dense[0] < 1e-9, mu=float(rt[0]), dense=float(mu_dense[0]), it=len(ht))
check('A6_exact_ritz_toy', abs(float(DM.exact_ritz(Xt, tc.T @ Xt, SHt)[0]) - mu_dense[0]) / mu_dense[0] < 1e-9)
res_conf = {}
for conf, ax in DM.F_AXIS.items():
    lat = toy_lattice(conf, dict(toyT=tc, toyN=tn))
    fpt = DM.face_ports(tc, ax)
    shared = np.flatnonzero(np.isin(lat.idx[0], lat.idx[1]))
    check(f'A6_shared_face_is_{conf}0_face (lattice3.build)', np.array_equal(shared, fpt.numpy()), dofs=len(shared))
    Gm = torch.ones(tc.np_, dtype=torch.bool); Gm[fpt] = False
    SGG = tc.T[Gm][:, Gm]
    muF_dense = sla.eigh(nt.S[Gm][:, Gm].numpy(), SGG.numpy(), eigvals_only=True)[::-1]

    def solveF(y_, SGG=SGG, Gm=Gm):
        out = torch.zeros_like(y_); out[Gm] = torch.linalg.solve(SGG, y_[Gm]); return out
    XFt, rFt, SHFt, hFt = DM.block_ritz(nt.apply, solveF, DM.zero_proj(fpt), X0t, tol=1e-13, maxit=2000)
    check(f'A6_block_ritz_toy_mu_F_{conf}', abs(float(rFt[0]) - muF_dense[0]) / muF_dense[0] < 1e-9 and muF_dense[0] <= mu_dense[0] + 1e-12,
          mu_F=float(rFt[0]), dense=float(muF_dense[0]), mu=float(mu_dense[0]))
    # lattice: the bound with mu_-F <= mu_eff <= mu (mu_eff from the dense pencil of the assembled lattice)
    ref = lat.reference()
    ex_t, ex_n = OP.ExactOp(tc, tc.T), OP.ExactOp(tn, tn.T)
    res = lat.evaluate([nt, ex_n], tol=1e-13)
    lt = DM.lattice_terms(lat, ref, res, nt, ex_t, ex_n)
    KL = lat.L @ lat.L.T
    fidx = lat.gather_idx[0]; keep = fidx >= 0
    Gg = torch.zeros((tc.np_, KL.shape[0]), dtype=dt); Gg[torch.nonzero(keep).squeeze(1), fidx[keep]] = 1
    D = Gg.T @ (nt.S - tc.T) @ Gg
    mu_eff = 1 + sla.eigh(D.numpy(), KL.numpy(), eigvals_only=True)[-1]
    bt = DM.bound_table(lt['eps_q'], mu_dense[0], muF_dense[0], lt['r'], lt['r_hat'], None, lat.gate)
    bt_eff = DM.bound_table(lt['eps_q'], mu_eff, mu_eff, lt['r'], lt['r_hat'], None, lat.gate)
    check(f'A6_lattice_bound_{conf}', muF_dense[0] - 1e-9 <= mu_eff <= mu_dense[0] + 1e-9 and bt['r_le_bound_hi'] and bt['r_hat_le_bound_hi']
          and bt_eff['r_hat_le_bound_hi'] and bt['implied_le_mu'] and all(a <= b + 1e-12 for a, b in zip(lt['r'], lt['r_hat']))
          and max(bt['implied_mu_eff']) <= mu_eff + 1e-9,
          mu=float(mu_dense[0]), mu_eff=float(mu_eff), mu_F=float(muF_dense[0]), eps_q=[round(x, 4) for x in lt['eps_q']],
          r_hat=[round(x, 4) for x in lt['r_hat']], bound_lo=[round(x, 4) for x in bt['bound_lo']], bound_hi=[round(x, 4) for x in bt['bound_hi']],
          spearman=bt['spearman_gate'])
    # A7: both learned, one learned, the rigorous sum rule 0 <= (C - C_hat) / C <= sum_c w_c eps_c
    out = LF.run_lattice(lat, [nt, nn], [ex_t, ex_n], log=lambda d: None)
    E0 = (lat.gather(ref['U'], 0) * nt.apply(lat.gather(ref['U'], 0))).sum(0) / torch.as_tensor(ref['energy'][0]) - 1
    wsum = np.asarray(out['both']['w']).sum(0)
    direct = lat.compare(lat.evaluate([nt, ex_n]))
    check(f'A7_run_lattice_{conf}', all(out[s]['bound_holds'] for s in ('test', 'nbr', 'both')) and np.allclose(wsum, 1, atol=1e-9)
          and np.allclose(out['test']['eps'][0], E0.numpy(), rtol=1e-9) and np.allclose(out['test']['eps'][1], 0)
          and np.allclose(out['test']['compliance_rel_err'], direct['compliance_rel_err'], rtol=1e-6)
          and all(0 < a < 1.5 for a in out['additivity']) and all(np.all(np.asarray(out[s]['compliance_rel_err']) > 0) for s in ('test', 'nbr', 'both')),
          compl_both=out['both']['gate_compliance_max'], compl_test=out['test']['gate_compliance_max'], compl_nbr=out['nbr']['gate_compliance_max'],
          additivity=[round(a, 3) for a in out['additivity']], bound_both=[round(b, 4) for b in out['both']['bound']],
          sens_both=out['both']['gate_sens_max_per_cell'], sens_test=out['test']['gate_sens_max_per_cell'])
    res_conf[conf] = out
# bound_table on hand numbers: eps_q = 0.04, mu = 1.5 -> bound_hi = sqrt(0.04 / 3)
bt = DM.bound_table([0.04], 1.5, 1.2, [0.1], [0.1])
check('A6_bound_table_numbers', abs(bt['bound_hi'][0] - np.sqrt(0.04 / 3)) < 1e-12 and abs(bt['bound_lo'][0] - np.sqrt(0.04 / 6)) < 1e-12
      and abs(bt['implied_mu_eff'][0] - 1 / (1 - 0.01 / 0.04)) < 1e-12 and bt['r_le_bound_hi'])
# lat_full.main plumbing (custom pair, consistent-load branch of lattice3 with stub face weights: the real ones need the
# polyhedral integrator, remote only)
LT.face_traction_weights = lambda cell, axis, value, pts_per_elem=6: \
    (np.stack(np.unravel_index(cell.nodes, (65,) * 3), 1)[:, axis] == int(value) * 64).astype(float)
LT.prepared = lambda case, body: (dict(toyT=tc, toyN=tn)[case], dict(toyT=tc, toyN=tn)[case].T)
DC.load_ckpt = lambda p_: dict(cfg=dict(body='', data=''), path=str(p_))
LF.fast_op = lambda ck, case, body, data, C_: (dict(toyT=nt, toyN=nn)[case], 0)
with tempfile.TemporaryDirectory() as td:
    rec = LF.main([str(Path(td) / 'l.json'), '--pairs', '', '--custom', '/x/runT/best.pt:toyT:/x/runN/best.pt:toyN', '--configs', 'x,y'])
    rr = rec['results']
    check('A7_main_plumbing', len(rr) == 2 and all(r_['pair'] == 'custom0' and r_['test_cell']['run'] == 'runT' and r_['nbr_cell']['case'] == 'toyN'
                                                    and all(r_[s_]['bound_holds'] for s_ in ('test', 'nbr', 'both'))
                                                    and all(l_.endswith('_cons') for l_ in r_['loads']) for r_ in rr),
          compl_both=[r_['both']['gate_compliance_max'] for r_ in rr], additivity=[max(r_['additivity']) for r_ in rr])
os.environ.pop('LAT_LOADS', None)
check('family_full', DC.family_full('fresh_train_0020_cover01_r2') == 'fresh_train_0020_full'
      and DC.family_full('fresh_development_0002_full') == 'fresh_development_0002_full')

print(json.dumps(dict(event='DONE', failed=FAIL, seconds=round(time.perf_counter() - T0, 1))), flush=True)
sys.exit(1 if FAIL else 0)
