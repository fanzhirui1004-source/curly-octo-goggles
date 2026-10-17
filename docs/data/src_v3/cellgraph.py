"""CUDA graphs for the narrow cell applications (src_v3 S26, OPL_V3_GRAPH=1; stream_ops.StreamedOp.apply,
lat_scale._Resident.apply).

The PCG matvec applies every cell's S_hat to one column. A full plate cell's S_hat is ~1400 kernel launches with ~20 ms
of GPU work; lat_multi issues the cells round robin on two streams so that one cell's small kernels run alongside the
next one's, but the CPU's launches (~16-18 ms per cell) then bound the matvec. Here each cell's narrow apply is captured
once as a CUDA graph (torch.cuda.CUDAGraph) and replayed: one launch per cell, the same kernels with the same arguments
(the scatters' atomics leave the run-to-run rounding noise they have anyway; scripts/t_graph.py: replay against eager
3.7e-8, eager against eager 3.7e-8).

  - One graph per (cell, width, stream), captured on the cell's OPL_V3_GRAPH_WARM-th (1) apply of that width after
    eager ones (the kernels compiled, the caches built). The first apply's result is eager.
  - Graphs replayed on one stream share one memory pool (torch.cuda.graph_pool_handle): their intermediates are dead
    after each replay, so sequential replays may reuse the same memory. The pool is as large as the largest cell's
    temporaries (with a streamed cell: its state too). They are also captured on one capture stream per replay stream
    (_cap_stream: the cuBLAS workspace a captured GEMM uses belongs to the capture stream).
  - A streamed cell (StreamedOp, page-locked host state): the graph begins with the host-to-device copy of the state
    from its page-locked buffer into the pool, then the decoding (stream_ops' packed forms) and S_hat, so a replay needs
    no device buffer at a fixed address; the copy of one cell runs alongside the other streams' computation. If the
    host buffer moves (S25 spill / unspill) the graph is captured again.
  - A resident cell (_Resident): the graph reads the cell's device tensors in place (they do not move after the front
    end).
  - Not graphed (eager as before): applies inside autograd, wider than OPL_V3_GRAPH_MAXB (4) columns (the coarse A Z:
    16-column chunks, bound by the GPU), a cell whose capture failed (STATS['failed'], logged once)."""
import os
import time
import torch

ON = os.environ.get('OPL_V3_GRAPH', '0') == '1'
MAXB = int(os.environ.get('OPL_V3_GRAPH_MAXB', '4'))
WARM = int(os.environ.get('OPL_V3_GRAPH_WARM', '1'))
_POOLS = {}                                                             # stream handle -> graph pool handle
_CAPS = {}                                                              # stream handle -> its capture stream
STATS = dict(captures=0, recaptures=0, replays=0, failed=0, capture_s=0.0, errors=[])


class _G:
    __slots__ = ('g', 'qs', 'out', 'fp')


def _cap_stream(cur):
    """One capture stream per replay stream: cuBLAS keeps a workspace per (handle, stream) and a captured GEMM keeps the
    capture stream's; graphs replayed at the same time on different streams must not share it (stream-K GEMMs
    synchronise through it: two concurrent replays sharing one hung in scripts/t_graph.py)."""
    s = _CAPS.get(cur.cuda_stream)
    if s is None:
        s = _CAPS[cur.cuda_stream] = torch.cuda.Stream()
    return s


def eligible(q):
    return ON and q.is_cuda and q.dim() == 2 and q.shape[1] <= MAXB and not torch.is_grad_enabled()


def apply(owner, q, body, fp=None):
    """S_hat q through owner's graph for (width, dtype, current stream), or None (the caller applies eagerly).
    body(qs) -> output: the computation to capture (reads qs, a static input); fp: what the graph's addresses depend on
    besides the owner (a changed fp: capture again)."""
    if not eligible(q):
        return None
    cur = torch.cuda.current_stream()
    key = (q.shape[1], q.dtype, cur.cuda_stream)
    graphs = owner.__dict__.setdefault('_cg', {})
    G = graphs.get(key)
    if G is False:                                                      # its capture failed: eager
        return None
    if G is not None and G.fp != fp:
        del graphs[key]
        G = None
        STATS['recaptures'] += 1
    if G is None:
        warm = owner.__dict__.setdefault('_cg_warm', {})
        n = warm.get(key, 0)
        if n < WARM:
            warm[key] = n + 1
            return None
        G = _capture(q, body, cur, fp)
        graphs[key] = G if G is not None else False
        if G is None:
            return None
    G.qs.copy_(q)
    G.g.replay()
    STATS['replays'] += 1
    return G.out.clone()


def _capture(q, body, cur, fp):
    t0 = time.perf_counter()
    qs = torch.empty_like(q)                                            # static input (outside the pool)
    pool = _POOLS.get(cur.cuda_stream)
    if pool is None:
        pool = _POOLS[cur.cuda_stream] = torch.cuda.graph_pool_handle()
    g = torch.cuda.CUDAGraph()
    cap = _cap_stream(cur)
    cap.wait_stream(cur)
    try:
        with torch.cuda.stream(cap):
            g.capture_begin(pool=pool)
            try:
                out = body(qs)
            finally:
                g.capture_end()
    except Exception as e:                                              # noqa: BLE001 (capture failed: eager from now)
        STATS['failed'] += 1
        if len(STATS['errors']) < 5:
            STATS['errors'].append(repr(e)[:300])
        torch.cuda.synchronize()
        return None
    cur.wait_stream(cap)
    G = _G()
    G.g, G.qs, G.out, G.fp = g, qs, out, fp
    STATS['captures'] += 1
    STATS['capture_s'] += time.perf_counter() - t0
    return G


def drop(owner):
    """Forget owner's graphs (e.g. when its state is released)."""
    owner.__dict__.pop('_cg', None)
    owner.__dict__.pop('_cg_warm', None)


def release(owners):
    """All graphs of owners dropped and their pools' memory returned to the device (after a solve: the sensitivity
    phase that follows applies the cells eagerly)."""
    for o in owners:
        drop(o)
    _POOLS.clear()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def pool_reserve_gib(nstreams):
    """Device memory to keep free for the graph pools (hostmem.resident_auto): OPL_V3_RGB_GRAPH GiB (1.75: a full plate
    cell's streamed state with its decoded parts plus S_hat's temporaries with recomputed slot weights) per stream."""
    return float(os.environ.get('OPL_V3_RGB_GRAPH', '1.75')) * nstreams if ON else 0.0


def summary():
    return dict(on=ON, maxb=MAXB, pools=len(_POOLS), **{k: (round(v, 2) if isinstance(v, float) else v) for k, v in STATS.items()})
