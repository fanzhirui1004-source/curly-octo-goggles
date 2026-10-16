"""Revision round 1 (E3 / E4): shared helpers for the complete-derivative check and the instrumented assembled solves.
New file; imports the existing modules read-only and changes none of their defaults.

Rebuilder      the learned operator of ONE cell rebuilt from scratch at a given corner vector tau on a private teacher Cell
               (same body = same active elements, as Cell.assemble(taus) / Cell.dmoments): moments and K at tau, network
               input data (NETDATA: moments, diag3, weak flags of this tau) in a private directory, trainlib.Geo, a fresh
               model cache, fastnet freeze, and the equilibrium-correction caches of this K (Chebyshev interval by the
               seeded power iteration, i.e. a deterministic function of tau; Galerkin coarse factor with its shift).
               fix_bounds: keep the Chebyshev interval of a reference design instead (diagnostic of the b(tau) term).
               Switch indicators per build: weak-flag vector, fringe hyperedge sets, coarse Cholesky shift, interval,
               number of elements with positive volume.
grad_metrics   metrics of an approximate gradient against the exact one (per load column): relative 2-norm error, cosine,
               per-variable relative error (median / p95 / max) with a floor 1e-3 max|g|, sign agreement, max component
               error over components >= 5% of the largest (the gate's componentwise measure).
vertex_map     lattice cell corners -> shared lattice vertices (position + CUBE[c]); tau consistency across cells.
env_flags      precision / environment flags recorded in every output (OPL_*, FUSED_HYPER, LAT_CPU, TF32 switches).
"""
import os, json, time, gc
from pathlib import Path
import numpy as np
import torch
import models as MD                                                    # noqa: F401  first: applies OPL_CONV_FP32
import teacher as TE
import trainlib as TL
import fastnet as FN
import evalnet as EN
import bench_deploy as BD

dev, dt = TE.dev, TE.dt
CORR_KEYS = ('_cV', '_cL', '_c_space', '_c_shift', '_c_seconds', '_tail_bounds', '_cL32', '_cAi32', '_cV32', '_cV32t',
             '_cL32_keep', '_Kc', 'dM')


def free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def env_flags():
    keys = sorted(k for k in os.environ if k.startswith('OPL_') or k in ('FUSED_HYPER', 'LAT_CPU', 'SENS_REASSOC', 'LAT_LOADS',
                                                                      'PYTORCH_CUDA_ALLOC_CONF', 'OMP_NUM_THREADS'))
    out = {k: os.environ[k] for k in keys}
    out.update(conv=TL.conv_precision(), matmul_allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
               cudnn_allow_tf32=bool(torch.backends.cudnn.allow_tf32), torch=torch.__version__,
               gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
               network='fp32 (true fp32 convolutions when OPL_CONV_FP32=1)',
               correction=('fp32 (OPL_COARSE_FP32=1 with a deploy/lean fp32 correction stiffness)'
                           if os.environ.get('OPL_COARSE_FP32') == '1' else 'fp64'),
               field_return='fp32 (trainlib.wrap returns the network dtype)', energies='fp64')
    return out


def gpu_gb():
    return dict(alloc=torch.cuda.memory_allocated() / 2 ** 30, peak=torch.cuda.max_memory_allocated() / 2 ** 30,
                free=torch.cuda.mem_get_info()[0] / 2 ** 30) if torch.cuda.is_available() else {}


class Rebuilder:
    def __init__(self, case, body, tmp_root, model_fn, log=print):
        self.case, self.body, self.tmp = case, body, Path(tmp_root)
        self.C = TE.Cell(case, body, log=lambda s_: None)
        self.model_fn, self.log = model_fn, log
        self.ref_info = None
        self.fixed_bounds = None
        self.n_builds, self.seconds = 0, 0.0

    def build(self, taus, fix_bounds=False):
        """-> (FastOp at taus, info). The previous operator must have been released by the caller."""
        t0 = time.perf_counter()
        C = self.C
        for k in CORR_KEYS:
            if k in vars(C):
                delattr(C, k)
        free()
        C.assemble(list(taus))
        d = self.tmp / self.case
        BD.netdata(C, self.body, d)
        geo = TL.Geo(self.case, self.body, self.tmp, neumann=False, log=lambda s_: None, cell=C, load_banks=False)
        model = self.model_fn(geo)
        if fix_bounds:
            if self.fixed_bounds is None:
                raise RuntimeError('fix_bounds without reference interval')
            C._tail_bounds = self.fixed_bounds
        fast = FN.FastNet(model, geo)
        op = EN.FastOp(fast)
        with torch.no_grad():
            op.apply(torch.zeros((C.np_, 1), dtype=dt, device=dev))            # correction caches of this K
        info = self._info(geo, model)
        info['seconds'] = time.perf_counter() - t0
        self.n_builds += 1; self.seconds += info['seconds']
        op._geo, op._model = geo, model                                          # released by release()
        return op, info

    def release(self, op):
        try:
            op._model.caches.pop(self.case, None)
        except Exception:                                                        # noqa: BLE001
            pass
        op.fast = None; op._geo = None; op._model = None
        free()

    def _info(self, geo, model):
        C = self.C
        c = model.caches.get(self.case)
        inf = dict(weak=np.asarray(geo.nd['weak'], bool), shift=float(getattr(C, '_c_shift', float('nan')) or 0.0),
                   tail=[float(v) for v in C._tail_bounds[1:]] if getattr(C, '_tail_bounds', None) else None,
                   active=int((C.M[:, 0] > 0).sum()))
        for k in ('el_fringe', 'gp_fringe'):
            v = getattr(c, k, None) if c is not None else None
            inf[k] = None if v is None else v.detach().cpu().numpy().astype(bool)
        return inf

    def set_reference(self, info):
        self.ref_info = info
        self.fixed_bounds = (30.0, info['tail'][0], info['tail'][1]) if info['tail'] else None

    def switches(self, info):
        """Discrete-switch indicators relative to the reference build."""
        r = self.ref_info
        out = dict(weak_flips=int((info['weak'] != r['weak']).sum()), shift=info['shift'], shift_ref=r['shift'],
                   shift_changed=bool(info['shift'] != r['shift']), active=info['active'], active_ref=r['active'],
                   tail_lmax=info['tail'][1] if info['tail'] else None)
        for k in ('el_fringe', 'gp_fringe'):
            a, b = info[k], r[k]
            out[k + '_flips'] = None if a is None or b is None else (int((a != b).sum()) if a.shape == b.shape else -1)
        out['any_switch'] = bool(out['weak_flips'] or out['shift_changed'] or out['active'] != out['active_ref']
                                 or (out.get('el_fringe_flips') or 0) or (out.get('gp_fringe_flips') or 0))
        return out


def energy_cols(op, Q):
    """q^T S_hat q per column (fp64) and the recovered field u (fp64 copy of the fp32 field)."""
    with torch.no_grad():
        e = (Q * op.apply(Q)).sum(0)
        u = op.field(Q)
    return e, u


def grad_metrics(g_ex, g_ap, floor_rel=1e-3):
    """g_*: (nvar, nload) numpy. Per load column."""
    g_ex, g_ap = np.asarray(g_ex, float), np.asarray(g_ap, float)
    out = dict(rel=[], cos=[], comp_med=[], comp_p95=[], comp_max=[], sign_agree=[], comp5_max=[], abs_max_over_norm=[])
    for j in range(g_ex.shape[1]):
        a, b = g_ex[:, j], g_ap[:, j]
        na = np.linalg.norm(a)
        out['rel'].append(float(np.linalg.norm(b - a) / na) if na > 0 else float('nan'))
        out['cos'].append(float(a @ b / (na * np.linalg.norm(b))) if na > 0 and np.linalg.norm(b) > 0 else float('nan'))
        fl = floor_rel * np.abs(a).max()
        r = np.abs(b - a) / np.maximum(np.abs(a), fl)
        out['comp_med'].append(float(np.median(r))); out['comp_p95'].append(float(np.quantile(r, 0.95)))
        out['comp_max'].append(float(r.max()))
        big = np.abs(a) >= fl
        out['sign_agree'].append(float((np.sign(a[big]) == np.sign(b[big])).mean()) if big.any() else float('nan'))
        m = np.abs(a) >= 0.05 * np.abs(a).max()
        out['comp5_max'].append(float(np.max(np.abs(b[m] / a[m] - 1))) if m.any() else float('nan'))
        out['abs_max_over_norm'].append(float(np.abs(b - a).max() / na) if na > 0 else float('nan'))
    return out


def vertex_map(positions, taus_by_cell):
    """positions: list of (i, j, k); taus_by_cell: list of 8-lists. -> (vid (ncell, 8) int, nvert, tau per vertex,
    max |tau difference| between cells sharing a vertex)."""
    from element_polyref import CUBE
    keys, vid, tv, dmax = {}, np.zeros((len(positions), 8), np.int64), [], 0.0
    for i, p in enumerate(positions):
        for c in range(8):
            v = tuple(int(a) + int(b) for a, b in zip(p, CUBE[c]))
            if v not in keys:
                keys[v] = len(keys); tv.append(float(taus_by_cell[i][c]))
            else:
                dmax = max(dmax, abs(tv[keys[v]] - float(taus_by_cell[i][c])))
            vid[i, c] = keys[v]
    return vid, len(keys), np.asarray(tv), dmax, [list(k) for k in keys]


def aggregate(G, vid, nvert):
    """G: (ncell, 8, nload) -> (nvert, nload): sum over the cell corners at each vertex (shared design variable)."""
    out = np.zeros((nvert, G.shape[2]))
    for i in range(G.shape[0]):
        for c in range(8):
            out[vid[i, c]] += G[i, c]
    return out


def tojson(x):
    if isinstance(x, dict):
        return {str(k): tojson(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [tojson(v) for v in x]
    if isinstance(x, np.ndarray):
        return tojson(x.tolist())
    if torch.is_tensor(x):
        return tojson(x.detach().cpu().numpy())
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def dump(path, obj):
    Path(path).write_text(json.dumps(tojson(obj), indent=1, default=float))


def pcg_fixed(lat_matvec, psolve, F, tol=1e-10, maxit=400):
    """lattice3.Lattice.evaluate's loop (same preconditioner and stopping rule) without the measurement pass."""
    X = torch.zeros_like(F); R = F.clone()
    Z = psolve(R); P = Z.clone()
    rz = (R * Z).sum(0); r0 = R.norm(dim=0)
    it = 0
    for it in range(1, maxit + 1):
        AP = lat_matvec(P)
        alpha = rz / (P * AP).sum(0)
        X += alpha * P; R -= alpha * AP
        if (R.norm(dim=0) / r0).max() < tol:
            break
        Z = psolve(R)
        rz_new = (R * Z).sum(0)
        P = Z + (rz_new / rz) * P; rz = rz_new
    return X, it, float((R.norm(dim=0) / r0).max())
