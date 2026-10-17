"""v3_census.py <census.json> <lat_scale arguments...>: lat_scale.main() with instrumentation (monkeypatches only; the
computation is unchanged). Run from the source directory (SRC, default /root/autodl-tmp/OPL/src_v3).
Records
  cells     per cell after the front end: device bytes by attribute path (resident cells), stream bytes by attribute path
            (streamed cells), device residue of streamed cells (CUDA tensors left on the device: < MIN_BYTES or inside
            tuples, plus fastidx index caches), host bytes (pinned buffer, K_PP host triplets, geo.nd, Cell numpy arrays);
  points    torch allocated / reserved / peak, whole device used (mem_get_info), RSS at phase points;
  kpp       device memory and seconds of the K_PP factor (cuDSS: plan, factorize);
  coarse    bytes of the lattice coarse space (SparseCoarse Z, ZT, AZ, AZT, T, TT);
  pcg       per PCG iteration: matvec / preconditioner seconds (synchronised);
  timeline  matvec calls TL_FROM..TL_TO: per cell CUDA-event times (wait for the prefetch, compute) and CPU issue time,
            per prefetch the side-stream copy (+ unpack) time."""
import os, sys, json, time, gc, types
SRC = os.environ.get('SRC', '/root/autodl-tmp/OPL/src_v3')
sys.path.insert(0, SRC)
import lat_scale as LS                                                  # imports models first (OPL_CONV_FP32)
import numpy as np
import torch
import stream_ops as SO
import lat_precond as PR
import lat_multi as LM
import teacher as TE

GiB = 2 ** 30
OUT, ARGV = (sys.argv[1], sys.argv[2:]) if __name__ == '__main__' else (None, None)   # v3_plate sets OUT
TL_FROM, TL_TO = int(os.environ.get('TL_FROM', '20')), int(os.environ.get('TL_TO', '22'))
REC = dict(points=[], cells=[], kpp=[], coarse=[], pcg=None, timeline=[], prefetch=[])
SKIP = (torch.nn.Module, types.ModuleType, type, types.FunctionType, types.MethodType, types.BuiltinFunctionType)


def rss():
    for l in open('/proc/self/status'):
        if l.startswith('VmRSS'):
            return int(l.split()[1]) * 1024 / GiB


def pools():
    """Reserved / allocated GiB per allocator pool (the default pool and each CUDA-graph pool), from the allocator's
    snapshot (OPL_V3_CENSUS_POOLS=1)."""
    out = {}
    for seg in torch.cuda.memory_snapshot():
        key = str(tuple(seg.get('segment_pool_id', (0, 0))))
        d = out.setdefault(key, dict(reserved=0.0, allocated=0.0, segments=0))
        d['reserved'] += seg['total_size'] / GiB
        d['allocated'] += seg.get('allocated_size', 0) / GiB
        d['segments'] += 1
    return {k: {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()} for k, v in out.items()}


def point(name, **kw):
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    if os.environ.get('OPL_V3_CENSUS_POOLS') == '1':
        kw['pools'] = pools()
    REC['points'].append(dict(name=name, alloc=torch.cuda.memory_allocated() / GiB, reserved=torch.cuda.memory_reserved() / GiB,
                              peak=torch.cuda.max_memory_allocated() / GiB, device_used=(total - free) / GiB, rss=rss(), **kw))


def parts(t):
    if t.layout == torch.sparse_csr:
        return [t.crow_indices(), t.col_indices(), t.values()]
    if t.layout == torch.sparse_coo:
        return [t._indices(), t._values()]
    return [t]


def walk(roots, want):
    """roots: [(label, obj)]. Returns (bytes by label path for tensors matching want(t), path of every container id)."""
    out, seen_obj, seen_store, cpath = {}, set(), set(), {}
    stack = list(roots)
    while stack:
        path, o = stack.pop()
        if id(o) in seen_obj:
            continue
        seen_obj.add(id(o)); cpath[id(o)] = path
        if isinstance(o, dict):
            items = [(f'{path}[{k}]' if isinstance(k, str) else f'{path}[]', v) for k, v in o.items()]
        elif isinstance(o, (list, tuple)):
            items = [(f'{path}[]', v) for v in o]
        else:
            try:
                items = [(f'{path}.{k}', v) for k, v in vars(o).items()]
            except TypeError:
                continue
        for p, v in items:
            if torch.is_tensor(v):
                for t in parts(v):
                    if want(t):
                        s = t.untyped_storage()
                        if s.data_ptr() not in seen_store:
                            seen_store.add(s.data_ptr())
                            out[p] = out.get(p, 0) + s.nbytes()
                fi = getattr(v, '_fi_flat', None)
                if isinstance(fi, dict):
                    for b, ft in fi.items():
                        if want(ft) and ft.untyped_storage().data_ptr() not in seen_store:
                            seen_store.add(ft.untyped_storage().data_ptr())
                            out[p + f'._fi_flat{b}'] = out.get(p + f'._fi_flat{b}', 0) + ft.untyped_storage().nbytes()
            elif isinstance(v, np.ndarray):
                if want(v):
                    out[p] = out.get(p, 0) + v.nbytes
            elif isinstance(v, SKIP) or isinstance(v, SO._Slot):
                continue
            elif isinstance(v, (dict, list, tuple)) or hasattr(v, '__dict__'):
                stack.append((p, v))
    return out, cpath


_OPS = []
_so_init = SO.StreamedOp.__init__


def so_init(self, op, cell, roots):
    _so_init(self, op, cell, roots)
    self._v3_roots = list(roots)
    _OPS.append(self)


SO.StreamedOp.__init__ = so_init
_res_init = LS._Resident.__init__


def res_init(self, op):
    _res_init(self, op)
    _OPS.append(self)


LS._Resident.__init__ = res_init


def top(d, n=40):
    return {k: round(v / 2 ** 20, 3) for k, v in sorted(d.items(), key=lambda kv: -kv[1])[:n]}


def cell_census():
    point('after_front_end')
    tot = dict(resident_dev=0, stream=0, residue=0, pinned=0, kpp_host=0, nd_host=0, cell_np=0)
    for o in _OPS:
        if isinstance(o, SO.StreamedOp):
            fast, geo, C = o._v3_roots
            rec = dict(case=C.case, kind='streamed', total=o.total)
            res, cpath = walk([('net', fast), ('geo', geo), ('C', C)], lambda t: torch.is_tensor(t) and t.is_cuda)
            by = {}
            done = set()
            for cont, k, s in o.slots:
                lab = cpath.get(id(cont), '?') + (f'.{k}' if not isinstance(cont, (list, tuple)) else '[]')
                for key in s.keys:
                    if key in done:
                        continue
                    done.add(key)
                    by[lab] = by.get(lab, 0) + o.layout[key][1]
            rec['stream_by_attr_MiB'] = top(by)
            rec['stream_bytes'] = sum(by.values())
            rec['residue_by_attr_MiB'] = top(res, 25)
            rec['residue_bytes'] = sum(res.values())
            tot['stream'] += rec['stream_bytes']; tot['residue'] += rec['residue_bytes']; tot['pinned'] += o.total
        else:
            fast = o.op.fast; geo = fast.geo; C = geo.C
            rec = dict(case=C.case, kind='resident')
            dv, _ = walk([('net', fast), ('geo', geo), ('C', C)], lambda t: torch.is_tensor(t) and t.is_cuda)
            rec['device_by_attr_MiB'] = top(dv)
            rec['device_bytes'] = sum(dv.values())
            tot['resident_dev'] += rec['device_bytes']
        kp = getattr(C, '_kpp_cache', None)
        disk = kp is not None and type(kp).__name__ == 'Stored'                 # kppstore (OPL_V3_KPP_DISK): a scratch file
        rec['kpp_host_bytes'] = 0 if disk else (sum(t.numel() * t.element_size() for t in kp) if kp is not None else 0)
        rec['kpp_disk_bytes'] = kp.nbytes if disk else 0
        rec['kpp_nnz'] = (kp.n if disk else int(kp[0].numel())) if kp is not None else 0
        nd = getattr(geo, 'nd', None) or {}
        rec['nd_host_bytes'] = sum(v.nbytes for v in nd.values() if isinstance(v, np.ndarray))
        rec['nd_by_key_MiB'] = {k: round(v.nbytes / 2 ** 20, 3) for k, v in nd.items() if isinstance(v, np.ndarray) and v.nbytes > 2 ** 18}
        cn, _ = walk([('C', C)], lambda t: isinstance(t, np.ndarray) or (torch.is_tensor(t) and not t.is_cuda))
        rec['cell_host_bytes'] = sum(v for k, v in cn.items() if '_kpp_cache' not in k)
        rec['cell_host_by_attr_MiB'] = top({k: v for k, v in cn.items() if '_kpp_cache' not in k}, 12)
        rec.update(nb=int(C.nb), np=int(C.np_), ni=int(C.ni), elements=int(len(C.cells)),
                   nc=int(C._cV32.shape[1]) if getattr(C, '_cV32', None) is not None else getattr(C, '_c_nc', None),
                   inv_err=getattr(C, '_c_inv_err', None), tri_err=getattr(C, '_c_tri_err', None))
        tot['kpp_host'] += rec['kpp_host_bytes']; tot['nd_host'] += rec['nd_host_bytes']; tot['cell_np'] += rec['cell_host_bytes']
        REC['cells'].append(rec)
    REC['cell_totals_GiB'] = {k: v / GiB for k, v in tot.items()}
    gc.collect()
    flush()


_census = SO.cuda_census
_ncall = [0]


def census_hook(top_=12):
    _ncall[0] += 1
    if _ncall[0] == 1:
        cell_census()
    else:
        point('end')
    return _census(top_)


SO.cuda_census = census_hook

# ---------------------------------------------------------------- preconditioner pieces
_sc_init = PR.SparseCoarse.__init__


def sc_init(self, lat, ops, kind, rcond=1e-10, batch=True):
    point('before_coarse')
    t = time.perf_counter()
    _sc_init(self, lat, ops, kind, rcond, batch)
    point('after_coarse')
    b = {}
    for nm in ('Z', 'ZT', 'AZ', 'AZT', 'T', 'TT'):
        v = getattr(self, nm, None)
        if v is not None:
            b[nm] = sum(x.numel() * x.element_size() for x in parts(v)) / GiB
    if getattr(self, 'azb', None):
        b['azb'] = sum(sum(x.numel() * x.element_size() for x in blk) for blk in self.azb) / GiB
    REC['coarse'].append(dict(seconds=time.perf_counter() - t, bytes_GiB=b, desc=self.desc, setup=self.setup))


PR.SparseCoarse.__init__ = sc_init
_kf_init = PR.KppFine.__init__


def kf_init(self, lat, kpp=None, backend='auto', holder=None):
    point('before_kpp_factor')
    f0 = torch.cuda.mem_get_info()[0]
    if holder is not None:
        _kf_init(self, lat, kpp, backend, holder=holder)
    else:
        _kf_init(self, lat, kpp, backend)
    torch.cuda.synchronize()
    f1 = torch.cuda.mem_get_info()[0]
    point('after_kpp_factor')
    REC['kpp'].append(dict(nnz_upper=self.nnz_upper, n=int(lat.nfree), device_delta_GiB=(f0 - f1) / GiB, setup=self.setup,
                           mem_est=getattr(self.fac, 'mem_est', None)))


PR.KppFine.__init__ = kf_init

# ---------------------------------------------------------------- SPDSolver phases (plan / factorize memory)
_spd_init = TE.SPDSolver.__init__


def spd_init(self, crow, col, vals, n, w=16, threads=16, fdt=None):
    torch.cuda.synchronize(); f0 = torch.cuda.mem_get_info()[0]; t0 = time.perf_counter()
    _spd_init(self, crow, col, vals, n, w, threads, fdt)
    torch.cuda.synchronize(); f1 = torch.cuda.mem_get_info()[0]
    REC.setdefault('spd', []).append(dict(n=int(n), nnz=int(vals.numel()), dtype=str(self.fdt), seconds=time.perf_counter() - t0,
                                          device_delta_GiB=(f0 - f1) / GiB))


TE.SPDSolver.__init__ = spd_init

# ---------------------------------------------------------------- PCG timing and timeline
_pcg = PR.pcg
_MV = [0]
_TL_ON = [False]


def pcg(lat, ops, prec, **kw):
    tm, tp = [], []

    def prec_w(R):
        torch.cuda.synchronize(); t = time.perf_counter()
        y = prec(R)
        torch.cuda.synchronize(); tp.append(time.perf_counter() - t)
        return y
    orig = lat.matvec

    def mv(ops_, U, batch=True, max_cols=None):
        _MV[0] += 1
        _TL_ON[0] = TL_FROM <= _MV[0] <= TL_TO
        torch.cuda.synchronize(); t = time.perf_counter()
        y = orig(ops_, U, batch, max_cols)
        torch.cuda.synchronize(); tm.append(time.perf_counter() - t)
        if _TL_ON[0]:
            _collect_timeline(_MV[0])
        _TL_ON[0] = False
        return y
    point('before_pcg')
    torch.cuda.reset_peak_memory_stats()
    lat.matvec = mv
    try:
        r = _pcg(lat, ops, prec_w, **kw)
    finally:
        lat.matvec = orig
    point('after_pcg')
    REC['pcg'] = dict(iterations=r['iterations'], seconds=r['seconds'], matvec_s=tm, prec_s=tp,
                      matvec_mean=float(np.mean(tm)), prec_mean=float(np.mean(tp)))
    flush()
    return r


PR.pcg = pcg
LS.PR.pcg = pcg

_EV = []                                                                # (cell index, case, e0, e_bound, e1, cpu_s)
_PF = []                                                                # (case, e0, e1) on the side stream
_ab = LM.MultiLattice.apply_blocks


def apply_blocks(self, ops, blocks, batch=True, max_cols=None):
    if not _TL_ON[0]:
        return _ab(self, ops, blocks, batch, max_cols)
    mc = max_cols or self.max_cols
    outs = [None] * len(blocks)
    for op, cells in self.groups(ops, batch):
        cells = [i for i in cells if blocks[i] is not None and blocks[i].shape[1] > 0]
        if not cells:
            continue
        X = torch.cat([blocks[i] for i in cells], 1)
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        op._v3_bound = None
        e0.record(); t = time.perf_counter()
        if mc is None or X.shape[1] <= mc:
            Y = op.apply(X)
        else:
            Y = torch.cat([op.apply(X[:, c0:c0 + mc]) for c0 in range(0, X.shape[1], mc)], 1)
        cpu = time.perf_counter() - t
        e1.record()
        case = op.C.case if isinstance(op, SO.StreamedOp) else op.op.fast.geo.C.case
        _EV.append((cells[0], case, e0, getattr(op, '_v3_bound', None), e1, cpu, isinstance(op, SO.StreamedOp)))
        c0 = 0
        for i in cells:
            k = blocks[i].shape[1]
            outs[i] = Y[:, c0:c0 + k]; c0 += k
    return outs


LM.MultiLattice.apply_blocks = apply_blocks
_bind = SO.StreamedOp._bind


def bind(self):
    _bind(self)
    if _TL_ON[0]:
        e = torch.cuda.Event(enable_timing=True); e.record(); self._v3_bound = e


SO.StreamedOp._bind = bind
_prefetch = SO.StreamedOp.prefetch


def prefetch(self, stream):
    if not _TL_ON[0] or self.loaded is not None:
        return _prefetch(self, stream)
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record(stream)
    t = time.perf_counter()
    _prefetch(self, stream)
    cpu = time.perf_counter() - t
    e1.record(stream)
    _PF.append((self.C.case, e0, e1, cpu, self.total))


SO.StreamedOp.prefetch = prefetch


def _collect_timeline(k):
    torch.cuda.synchronize()
    rows = []
    for i, case, e0, eb, e1, cpu, streamed in _EV:
        rows.append(dict(cell=i, case=case, streamed=streamed, gpu_ms=e0.elapsed_time(e1),
                         wait_ms=e0.elapsed_time(eb) if eb is not None else 0.0, cpu_ms=1000 * cpu))
    pf = [dict(case=c, copy_ms=a.elapsed_time(b), cpu_ms=1000 * cpu, bytes=nb) for c, a, b, cpu, nb in _PF]
    REC['timeline'].append(dict(matvec=k, cells=rows))
    REC['prefetch'].append(dict(matvec=k, prefetches=pf))
    _EV.clear(); _PF.clear()


def flush():
    if OUT is None:
        return
    with open(OUT, 'w') as f:
        json.dump(REC, f, indent=1, default=float)


if __name__ == '__main__':
    os.chdir(SRC)
    point('start')
    t0 = time.perf_counter()
    LS.main(ARGV)
    REC['wall_s'] = time.perf_counter() - t0
    point('final')
    flush()
