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
are stored as pinned components (shared components, e.g. index arrays of two CSR matrices, once) and rebuilt on the device."""
import types
import contextlib
import torch

MIN_BYTES = 1 << 20
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
        self.pinned, slots, byid = {}, [], {}
        for o, k, t in handles:
            if id(t) not in byid:
                keys = []
                for x in _parts(t):
                    key = (x.data_ptr(), x.dtype, tuple(x.shape), tuple(x.stride()))
                    if key not in self.pinned:
                        self.pinned[key] = x.cpu().pin_memory()
                    keys.append(key)
                byid[id(t)] = _Slot(t, keys)
            slots.append((o, k, byid[id(t)]))
            _OWNED[(id(o), k)] = self
        self.slots = slots
        self.bytes = sum(x.numel() * x.element_size() for x in self.pinned.values())
        self.n_tensors = len(byid)
        self.nxt, self.loaded, self.ev, self.depth = None, None, None, 0
        self.copies = 0
        del handles, byid
        self._park()
        torch.cuda.empty_cache()

    def _park(self):
        for o, k, s in self.slots:
            _set(o, k, s)
        self.loaded = None
        _LIVE.discard(self)

    def prefetch(self, stream):
        if self.loaded is not None:
            return
        with torch.cuda.stream(stream):
            p = {k: x.to(dev, non_blocking=True) for k, x in self.pinned.items()}
            ev = torch.cuda.Event(); ev.record(stream)
        self.loaded, self.ev = p, ev
        self.copies += 1
        _LIVE.add(self)

    def _bind(self):
        cur = torch.cuda.current_stream()
        cur.wait_event(self.ev)
        for x in self.loaded.values():
            x.record_stream(cur)
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
        host = {k: (x.clone() if to is None else x.to(to)) for k, x in self.pinned.items()}
        built = {}
        for o, k, s in self.slots:
            if id(s) not in built:
                built[id(s)] = s.build(host)
            _set(o, k, built[id(s)])
            _OWNED.pop((id(o), k), None)
        self.loaded = None
        _LIVE.discard(self)
        self.pinned, self.slots = {}, []
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
