"""Streamed learned cell operators (P2, step 2): the read-only per-cell state of a learned operator (cell tensors, the
frozen network state, the model's per-geometry cache, correction caches) lives once in pinned host memory; for every use
it is copied to the device on a side stream, one cell ahead of the computation (double buffer), and dropped afterwards
(never copied back: nothing in it changes during a solve). The arithmetic is unchanged.

  op = stream_ops.StreamedOp(FastOp, cell, roots)          # after a warm-up application (builds the correction caches)
  stream_ops.link(ops)                                     # prefetch order = list order (the lattice's operator order)
  ops[i].apply(q), ops[i].field(q), with ops[i].active(): ...
  op.release()                                             # slots back to ordinary host tensors (e.g. before the next model)

Collected: every CUDA tensor of at least MIN_BYTES reachable from the given roots through attributes of plain objects,
lists, tuples and dicts (torch modules, python modules, callables and foreign library objects are not entered: module
parameters are shared by all cells). A slot already owned by another StreamedOp stays resident. Sparse CSR / COO tensors
are stored as pinned components (shared components, e.g. index arrays of two CSR matrices, once) and rebuilt on the device.

OPL_STREAM_FP32=1 (default off; scale demonstration only): float64 parts are STORED in the host buffer as float32 and
converted back to float64 on the device after each copy, so the arithmetic precision of every operation is unchanged
but the stored state is rounded to single precision (halves the host memory and copy volume of the float64 parts;
its effect on the solution must be checked against the float64 store).  bytes_by_dtype reports the stored layout."""
import os
import time
import types
import weakref
import numpy as np
import contextlib
import torch
import cellgraph as CG

MIN_BYTES = int(os.environ.get('OPL_STREAM_MIN', '0'))                # src_v3: 0 (src_v2: 1 MiB; smaller tensors stayed on the
                                                                        # device for every streamed cell, ~15-30 MB per cell)
STORE_FP32 = os.environ.get('OPL_STREAM_FP32', '0') == '1'
STORE_PACK = os.environ.get('OPL_STREAM_PACK', '0') == '1'
# src_v3 S25 (OPL_V3_SPILL=1): around the K_PP planning and factorisation (spill_guard) the pinned states of enough parked
# cells are written to files in OPL_V3_SPILL_DIR and freed, and read back afterwards (the same bytes): the host peak of
# cuDSS's analysis (~71 bytes per K_PP nonzero for ~15 s, measured: 8.6 GiB on the 192-cell plate) no longer needs a
# reserve in the front end's budget (hostmem.Budget).
SPILL = os.environ.get('OPL_V3_SPILL', '0') == '1'
SPILL_DIR = os.environ.get('OPL_V3_SPILL_DIR', '')
SPILL_BPNZ = float(os.environ.get('OPL_V3_SPILL_BPNZ', '80'))          # host peak per K_PP (upper) nonzero, bytes
SPILL_FORCE = float(os.environ.get('OPL_V3_SPILL_FORCE_GB', '0')) * 2 ** 30   # check only: spill at least this much
_ALL = weakref.WeakSet()                                                # every StreamedOp (spill_guard)
GSTAGES = int(os.environ.get('OPL_V3_GRAPH_STAGES', '4'))              # S26: device stage buffers of graphed streamed cells
_GST = []                                                               # stage dicts {'buf': uint8 tensor, 'free': Event}
PACK_MIN = 1 << 24                                                      # square matrices of at least 16 MB
_PB = 512                                                               # rows per (un)packing block


def _tril_blocks(n):
    for r0 in range(0, n, _PB):
        r1 = min(n, r0 + _PB)
        yield r0, r1, (r1 * (r1 + 1) - r0 * (r0 + 1)) // 2


def _pack_tril(x, out):
    """Row-major lower triangle (incl. diagonal) of the square x into the 1-D out."""
    n, pos = x.shape[0], 0
    for r0, r1, cnt in _tril_blocks(n):
        m = torch.arange(r1, device=x.device)[None, :] <= torch.arange(r0, r1, device=x.device)[:, None]
        out[pos:pos + cnt].copy_(x[r0:r1, :r1][m])
        pos += cnt


def _unpack_tril(seg, n, sym):
    """Inverse of _pack_tril on seg's device; sym: mirror the strict lower triangle (symmetric matrix). Integer scatter
    indices (src_v3; src_v2 assigned through a boolean mask, which synchronises the host with the copy stream)."""
    y = torch.zeros((n, n), dtype=seg.dtype, device=seg.device)
    pos, yf, d = 0, y.view(-1), seg.device
    for r0, r1, cnt in _tril_blocks(n):
        r = torch.arange(r0, r1, device=d)
        rr = torch.repeat_interleave(r, r + 1, output_size=cnt)
        start = (r * (r + 1)) // 2 - (r0 * (r0 + 1)) // 2
        cc = torch.arange(cnt, device=d) - torch.repeat_interleave(start, r + 1, output_size=cnt)
        yf[rr * n + cc] = seg[pos:pos + cnt]
        pos += cnt
    if sym:
        y += torch.tril(y, -1).T
    return y


def _encoding(x):
    """Storage mode of one part: None (as is), 'f32' (OPL_STREAM_FP32: float64 stored as float32, rounding), and with
    OPL_STREAM_PACK (exact): 'tril' / 'sym' (lower-triangular / symmetric square matrix: lower triangle only), 'i32'
    (int64 whose values fit in int32)."""
    if STORE_PACK and x.dim() == 2 and x.shape[0] == x.shape[1] and x.is_floating_point() and x.layout == torch.strided \
            and x.numel() * x.element_size() >= PACK_MIN:
        if torch.equal(x, torch.tril(x)):
            return 'tril'
        if torch.equal(x, x.T):
            return 'sym'
    if STORE_PACK and x.dtype == torch.int64 and x.numel() and int(x.min()) >= -2 ** 31 and int(x.max()) < 2 ** 31:
        return 'i32'
    if STORE_FP32 and x.dtype == torch.float64:
        return 'f32'
    return None
REGISTER = True                                                         # page-lock the host buffer (async copies)
dev = torch.device('cuda')
_STREAM = [None]
_OWNED = {}                                                             # (id(container), key) -> StreamedOp
_LIVE = set()                                                           # ops whose state is on the device
_SKIP_MOD = ('torch', 'numpy', 'scipy', 'triton', 'cupy', 'builtins')


def _parts(t):
    if t.layout == torch.sparse_csr:
        return [t.crow_indices(), t.col_indices(), t.values()]
    if t.layout == torch.sparse_coo:
        t = t.coalesce()
        return [t._indices(), t._values()]
    return [t]


def _nbytes(t):
    return sum(x.numel() * x.element_size() for x in _parts(t))


class _Slot:
    """Placeholder left in a container while the op's state is off the device (any use before loading fails loudly)."""
    __slots__ = ('layout', 'shape', 'keys')

    def __init__(self, t, keys):
        self.layout, self.shape, self.keys = t.layout, t.shape, keys

    def build(self, p):
        x = [p[k] for k in self.keys]
        if self.layout == torch.sparse_csr:
            return torch.sparse_csr_tensor(x[0], x[1], x[2], size=self.shape)
        if self.layout == torch.sparse_coo:
            return torch.sparse_coo_tensor(x[0], x[1], size=self.shape, is_coalesced=True)
        return x[0]


def _enter_obj(v):
    if isinstance(v, (dict, list, tuple)):
        return True
    if isinstance(v, (torch.nn.Module, types.ModuleType, type, _Slot)) or callable(v) or not hasattr(v, '__dict__'):
        return False
    return type(v).__module__.split('.')[0] not in _SKIP_MOD


def _walk(root, handles, seen):
    """Collect (container, key, tensor) for large CUDA tensors reachable from root."""
    stack = [root]
    while stack:
        o = stack.pop()
        if id(o) in seen:
            continue
        seen.add(id(o))
        if isinstance(o, dict):
            items = list(o.items())
        elif isinstance(o, (list, tuple)):
            items = list(enumerate(o))
        else:
            items = list(vars(o).items())
        for k, v in items:
            if torch.is_tensor(v):
                if v.is_cuda and _nbytes(v) >= MIN_BYTES and not getattr(v, '_opl_shared', False):
                    handles.append((o, k, v))
            elif _enter_obj(v):
                stack.append(v)


def _set(o, k, v):
    if isinstance(o, (dict, list)):
        o[k] = v
    else:
        setattr(o, k, v)


class StreamedOp:
    def __init__(self, op, cell, roots):
        self.op, self.C = op, cell
        handles, seen = [], set()
        for r in roots:
            _walk(r, handles, seen)
        self.skipped_tuple = sum(isinstance(o, tuple) for o, _, _ in handles)
        handles = [h for h in handles if not isinstance(h[0], tuple)]
        self.skipped_owned = sum((id(o), k) in _OWNED for o, k, _ in handles)
        handles = [h for h in handles if (id(h[0]), h[1]) not in _OWNED]
        src, slots, byid = {}, [], {}
        for o, k, t in handles:
            if id(t) not in byid:
                keys = []
                for x in _parts(t):
                    key = (x.data_ptr(), x.dtype, tuple(x.shape), tuple(x.stride()))
                    src.setdefault(key, x)
                    keys.append(key)
                byid[id(t)] = _Slot(t, keys)
            slots.append((o, k, byid[id(t)]))
            _OWNED[(id(o), k)] = self
        self.slots = slots
        # one exact-size host buffer per op, page-locked by cudaHostRegister (the caching pinned allocator rounds every
        # block up to a power of two); every part is a 512-byte aligned view of it; one copy per prefetch
        self.layout, self.orig, self.enc, off = {}, {}, {}, 0
        for key, x in src.items():
            mode, sdt, shp = _encoding(x), x.dtype, tuple(x.shape)
            if mode == 'f32':
                sdt = torch.float32
            elif mode == 'i32':
                sdt = torch.int32
            elif mode in ('tril', 'sym'):
                shp = (x.shape[0] * (x.shape[0] + 1) // 2,)
            if mode:
                self.enc[key] = (mode, x.dtype, tuple(x.shape))
                self.orig[key] = x.dtype
            nb = int(np.prod(shp)) * torch.empty((), dtype=sdt).element_size()
            self.layout[key] = (off, nb, sdt, shp)
            off += (nb + 511) // 512 * 512
        self.total = max(off, 512)
        self.buf = torch.empty(self.total, dtype=torch.uint8)
        self.registered = False
        if self.total and REGISTER:                                     # src_v3: page-locked BEFORE the copies (device
            err = torch.cuda.cudart().cudaHostRegister(self.buf.data_ptr(), self.total, 0)   # to pageable memory runs at
            self.registered = int(err) == 0                                                 # ~1.5 GB/s: 0.35 s per cell)
        for key, x in src.items():
            if self.enc.get(key, (None,))[0] in ('tril', 'sym'):
                _pack_tril(x, self._view(self.buf, key))
            else:
                self._view(self.buf, key).copy_(x.contiguous())
        self.pinned = {key: self._view(self.buf, key) for key in src}
        del src
        self.bytes = sum(v[1] for v in self.layout.values())
        self.bytes_by_dtype = {}
        for v in self.layout.values():
            self.bytes_by_dtype[str(v[2])] = self.bytes_by_dtype.get(str(v[2]), 0) + v[1]
        self.bytes_rounded = sum(self.layout[k][1] for k, e in self.enc.items() if e[0] == 'f32')
        self.bytes_unpacked = sum(int(np.prod(e[2])) * torch.empty((), dtype=e[1]).element_size() for e in self.enc.values())
        self.enc_counts = {m: sum(e[0] == m for e in self.enc.values()) for m in ('f32', 'i32', 'tril', 'sym')}
        self.n_tensors = len(byid)
        self.nxt, self.loaded, self.ev, self.depth = None, None, None, 0
        self.copies, self.spilled = 0, None
        _ALL.add(self)
        del handles, byid
        self._park()
        torch.cuda.empty_cache()

    def _unregister(self):
        if getattr(self, 'registered', False) and getattr(self, 'buf', None) is not None:
            torch.cuda.cudart().cudaHostUnregister(self.buf.data_ptr())
        self.registered = False

    def __del__(self):
        try:
            self._unregister()
        except Exception:
            pass

    def spill(self, path, chunk=1 << 28):
        """src_v3 S25: write the pinned host buffer to path and free it (parked op); unspill() reads it back. Returns the
        bytes freed. The file's pages are dropped from the page cache (charged to the container's memory limit)."""
        if self.depth or self.buf is None or self.spilled is not None:
            return 0
        self._park()
        mv = memoryview(self.buf.numpy())
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            pos = 0
            while pos < self.total:
                pos += os.write(fd, mv[pos:min(self.total, pos + chunk)])
            os.fsync(fd)
            if hasattr(os, 'posix_fadvise'):
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
        del mv
        self._unregister()
        self.pinned, self.buf, self.spilled = {}, None, path
        return self.total

    def unspill(self, chunk=1 << 28):
        """src_v3 S25: a new page-locked buffer with the spilled bytes; the file is removed."""
        if self.spilled is None:
            return 0
        path = self.spilled
        buf = torch.empty(self.total, dtype=torch.uint8)
        if REGISTER:
            err = torch.cuda.cudart().cudaHostRegister(buf.data_ptr(), self.total, 0)
            self.registered = int(err) == 0
        mv = memoryview(buf.numpy())
        fd = os.open(path, os.O_RDONLY)
        try:
            pos = 0
            while pos < self.total:
                n = os.readv(fd, [mv[pos:min(self.total, pos + chunk)]])
                if n <= 0:
                    raise IOError(f'spill file {path} short: {pos} of {self.total} bytes')
                pos += n
            if hasattr(os, 'posix_fadvise'):
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
        del mv
        os.remove(path)
        self.buf, self.spilled = buf, None
        self.pinned = {key: self._view(buf, key) for key in self.layout}
        return self.total

    def _view(self, buf, key):
        o, nb, dty, shp = self.layout[key]
        return buf[o:o + nb].view(dty).view(shp)

    def _park(self):
        for o, k, s in self.slots:
            _set(o, k, s)
        self.loaded = self.dbuf = None
        _LIVE.discard(self)

    def prefetch(self, stream):
        if self.loaded is not None:
            return
        with torch.cuda.stream(stream):
            dbuf = torch.empty(self.total, dtype=torch.uint8, device=dev)
            dbuf.copy_(self.buf, non_blocking=self.registered)
            p = {k: self._view(dbuf, k) for k in self.layout}
            for k, (mode, odt, shp) in self.enc.items():                # stored form -> working tensor
                p[k] = _unpack_tril(p[k], shp[0], mode == 'sym') if mode in ('tril', 'sym') else p[k].to(odt)
            ev = torch.cuda.Event(); ev.record(stream)
        self.loaded, self.ev, self.dbuf = p, ev, dbuf
        self.copies += 1
        _LIVE.add(self)

    def _bind(self):
        cur = torch.cuda.current_stream()
        cur.wait_event(self.ev)
        self.dbuf.record_stream(cur)
        for k in self.orig:
            self.loaded[k].record_stream(cur)
        built = {}
        for o, k, s in self.slots:
            if id(s) not in built:
                built[id(s)] = s.build(self.loaded)
            _set(o, k, built[id(s)])

    def _enter(self):
        if self.depth == 0:
            for o in list(_LIVE):                                       # at most this cell and the next one on the device
                if o is not self and o is not self.nxt and o.depth == 0:
                    o._park()
            self.prefetch(_STREAM[0])
            self._bind()
            if self.nxt is not None:
                self.nxt.prefetch(_STREAM[0])                           # the next cell's copy overlaps this compute
        self.depth += 1

    def _exit(self):
        self.depth -= 1
        if self.depth == 0:
            self._park()

    @contextlib.contextmanager
    def active(self):
        self._enter()
        try:
            yield
        finally:
            self._exit()

    def apply(self, q):
        if CG.ON and self.depth == 0 and self.registered and self.buf is not None:   # src_v3 S26: CUDA graph
            y = self._apply_staged(q) if GSTAGES > 0 else \
                CG.apply(self, q, self._graph_body, fp=(self.buf.data_ptr(), self.total))
            if y is not None:
                if self.loaded is not None:                             # a state prefetched for an eager use is
                    self._park()                                        # not read by the graph
                return y
        with self.active():
            return self.op.apply(q)

    def _apply_staged(self, q):
        """src_v3 S26 (OPL_V3_GRAPH_STAGES = K > 0): the graph reads the raw state from stage buffer gstage mod K; before
        each replay the state is copied there on the copy stream once the stage's previous replay has finished."""
        if not CG.eligible(q):
            return None
        st = _stage(getattr(self, 'gstage', 0) % GSTAGES, self.total)
        raw = st['buf'][:self.total]
        cur = torch.cuda.current_stream()
        init()
        cs = _STREAM[0]

        def pre():
            with torch.cuda.stream(cs):
                if st['free'] is not None:
                    cs.wait_event(st['free'])
                raw.copy_(self.buf, non_blocking=True)
                ev = torch.cuda.Event()
                ev.record(cs)
            cur.wait_event(ev)

        def post():
            ev = torch.cuda.Event()
            ev.record(cur)
            st['free'] = ev
        return CG.apply(self, q, lambda qs: self._graph_body(qs, raw), fp=(self.buf.data_ptr(), self.total, raw.data_ptr()),
                        pre=pre, post=post)

    def _graph_body(self, qs, raw=None):
        """src_v3 S26: the apply as cellgraph captures it: the raw state (raw: a stage buffer; None: copied from the
        page-locked buffer into the graph's memory pool) decoded (as prefetch does), bound, the operator applied, the
        placeholders put back."""
        if raw is None:
            raw = torch.empty(self.total, dtype=torch.uint8, device=dev)
            raw.copy_(self.buf, non_blocking=True)
        p = {k: self._view(raw, k) for k in self.layout}
        for k, (mode, odt, shp) in self.enc.items():
            p[k] = _unpack_tril(p[k], shp[0], mode == 'sym') if mode in ('tril', 'sym') else p[k].to(odt)
        built = {}
        for o, k, s in self.slots:
            if id(s) not in built:
                built[id(s)] = s.build(p)
            _set(o, k, built[id(s)])
        try:
            return self.op.apply(qs)
        finally:
            for o, k, s in self.slots:
                _set(o, k, s)

    def field(self, q):
        with self.active():
            return self.op.field(q)

    def release(self, to=None):
        """Put ordinary (non-pinned) host tensors (to=None) or device tensors (to=device) back into the slots and forget
        the op."""
        if self.depth:
            raise RuntimeError('release inside active()')
        host = {}
        for k, x in self.pinned.items():
            mode, odt, shp = self.enc.get(k, (None, x.dtype, None))
            if mode in ('tril', 'sym'):
                host[k] = _unpack_tril(x if to is None else x.to(to), shp[0], mode == 'sym')
            else:
                host[k] = x.to(dtype=odt, copy=True) if to is None else x.to(to, dtype=odt)
        built = {}
        for o, k, s in self.slots:
            if id(s) not in built:
                built[id(s)] = s.build(host)
            _set(o, k, built[id(s)])
            _OWNED.pop((id(o), k), None)
        self.loaded = self.dbuf = None
        _LIVE.discard(self)
        self._unregister()
        CG.drop(self)
        self.pinned, self.slots, self.buf = {}, [], None
        self.nxt = self.op = self.C = None                              # the prefetch ring must not keep released ops alive


def drop(op):
    """Forget a StreamedOp whose cells are discarded (src_v3): unregister and free the pinned buffer without restoring the
    slots (release() first copies every part back into new host tensors, a transient host peak of the whole state)."""
    if op.depth:
        raise RuntimeError('drop inside active()')
    for o, k, s in op.slots:
        _OWNED.pop((id(o), k), None)
    op.loaded = op.dbuf = None
    _LIVE.discard(op)
    op._unregister()
    CG.drop(op)
    op.pinned, op.slots, op.buf = {}, [], None
    op.nxt = op.op = op.C = None


def link(ops):
    """Prefetch ring in list order (S26: ring position i -> graph stage buffer i mod OPL_V3_GRAPH_STAGES)."""
    if _STREAM[0] is None:
        _STREAM[0] = torch.cuda.Stream()
    for i, (a, b) in enumerate(zip(ops, ops[1:] + ops[:1])):
        a.nxt = b
        a.gstage = i
    return ops


def _stage(j, nbytes):
    """S26: stage buffer j (as large as the largest streamed state; grown if needed: graphs reading the old one are
    captured again, their fingerprint holds its address)."""
    while len(_GST) <= j:
        _GST.append({'buf': None, 'free': None})
    st = _GST[j]
    if st['buf'] is None or st['buf'].numel() < nbytes:
        need = max([nbytes] + [o.total for o in list(_ALL) if o.buf is not None])
        if st['buf'] is not None:
            torch.cuda.synchronize()
        st['buf'], st['free'] = torch.empty(need, dtype=torch.uint8, device=dev), None
    return st


def free_stages():
    """S26: the stage buffers back to the allocator (after a solve)."""
    if _GST:
        torch.cuda.synchronize()
    _GST.clear()


def park_all():
    """Drop every prefetched (not active) state from the device."""
    for o in list(_LIVE):
        if o.depth == 0:
            o._park()


@contextlib.contextmanager
def spill_guard(need_bytes):
    """src_v3 S25 (OPL_V3_SPILL=1): context around a host-memory peak of need_bytes (the K_PP planning and factorisation:
    SPILL_BPNZ bytes per nonzero). If the RSS plus the peak would pass OPL_V3_HOST_TOP of the host limit, every state
    is parked (after a device synchronisation: no copy from a pinned buffer in flight) and the largest parked cells are
    written to OPL_V3_SPILL_DIR and freed until it fits; all of them are read back on exit. Yields a dict of what was done
    (seconds as *_t: they are part of the timed factorisation). No cell operator may be applied inside."""
    info = {}
    spilled = []
    if SPILL:
        import hostmem as HM
        avail = HM.top_fraction() * HM.limit_bytes() - HM.rss_bytes()
        short = max(need_bytes - avail, SPILL_FORCE)
        info.update(spill_need_GiB=round(need_bytes / 2 ** 30, 2), spill_avail_GiB=round(avail / 2 ** 30, 2))
        if short > 0:
            torch.cuda.synchronize()
            park_all()
            d = SPILL_DIR or os.path.join(os.environ.get('OPL_V3W', '/root/autodl-tmp/OPL/V3W'), 'spill')
            os.makedirs(d, exist_ok=True)
            t0, done = time.perf_counter(), 0
            for o in sorted([o for o in _ALL if o.buf is not None and o.depth == 0], key=lambda o: -o.total):
                if done >= short:
                    break
                n = o.spill(os.path.join(d, f'cell_{os.getpid()}_{id(o):x}.bin'))
                if n:
                    spilled.append(o); done += n
            HM.trim()
            info.update(spill_cells=len(spilled), spill_GiB=round(done / 2 ** 30, 2),
                        spill_write_t=round(time.perf_counter() - t0, 2), spill_rss_GiB=round(HM.rss_bytes() / 2 ** 30, 2))
    try:
        yield info
    finally:
        if spilled:
            t0 = time.perf_counter()
            for o in spilled:
                o.unspill()
            info['spill_read_t'] = round(time.perf_counter() - t0, 2)


def cuda_census(top=12):
    """Live CUDA tensors reachable by the garbage collector, grouped by (dtype, shape): (GB, count) (diagnostics)."""
    import gc
    agg, seen = {}, set()
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda and obj.layout == torch.strided:
                key = (obj.untyped_storage().data_ptr())
                if key in seen:
                    continue
                seen.add(key)
                k = (str(obj.dtype), tuple(obj.shape))
                g, n = agg.get(k, (0.0, 0))
                agg[k] = (g + obj.untyped_storage().nbytes() / 1e9, n + 1)
        except Exception:
            pass
    return sorted(((round(v[0], 4), v[1], str(k)) for k, v in agg.items()), reverse=True)[:top]


def init():
    if _STREAM[0] is None:
        _STREAM[0] = torch.cuda.Stream()
