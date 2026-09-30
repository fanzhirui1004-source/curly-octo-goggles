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
import types
import numpy as np
import contextlib
import torch

MIN_BYTES = 1 << 20
STORE_FP32 = os.environ.get('OPL_STREAM_FP32', '0') == '1'
STORE_PACK = os.environ.get('OPL_STREAM_PACK', '0') == '1'
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
    """Inverse of _pack_tril on seg's device; sym: mirror the strict lower triangle (symmetric matrix)."""
    y = torch.zeros((n, n), dtype=seg.dtype, device=seg.device)
    pos = 0
    for r0, r1, cnt in _tril_blocks(n):
        m = torch.arange(r1, device=seg.device)[None, :] <= torch.arange(r0, r1, device=seg.device)[:, None]
        y[r0:r1, :r1][m] = seg[pos:pos + cnt]
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
                if v.is_cuda and _nbytes(v) >= MIN_BYTES:
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
        for key, x in src.items():
            if self.enc.get(key, (None,))[0] in ('tril', 'sym'):
                _pack_tril(x, self._view(self.buf, key))
            else:
                self._view(self.buf, key).copy_(x.contiguous())
        self.registered = False
        if self.total and REGISTER:
            err = torch.cuda.cudart().cudaHostRegister(self.buf.data_ptr(), self.total, 0)
            self.registered = int(err) == 0
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
        self.copies = 0
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
        with self.active():
            return self.op.apply(q)

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
        self.pinned, self.slots, self.buf = {}, [], None
        self.nxt = self.op = self.C = None                              # the prefetch ring must not keep released ops alive


def link(ops):
    """Prefetch ring in list order."""
    if _STREAM[0] is None:
        _STREAM[0] = torch.cuda.Stream()
    for a, b in zip(ops, ops[1:] + ops[:1]):
        a.nxt = b
    return ops


def park_all():
    """Drop every prefetched (not active) state from the device."""
    for o in list(_LIVE):
        if o.depth == 0:
            o._park()


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
