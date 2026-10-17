"""CUDA graphs for the narrow cell applications (src_v3 S26, default on, OPL_V3_GRAPH=0: off; stream_ops.StreamedOp.apply,
lat_scale._Resident.apply). Measured on the 51-cell plate (results/v3z_S51, v3z4_S51): solve ~148-154 -> 141.1 s (two
streams), 138.6 s (four), PCG 168, compliance within the run-to-run noise; hlat222 unchanged (eight resident cells,
bound by the GPU).

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
  - A streamed cell (StreamedOp, page-locked host state), OPL_V3_GRAPH_STAGES = K > 0 (default 4): the raw state is
    copied on stream_ops' copy stream into one of K fixed device stage buffers (cell i of the prefetch ring: stage
    i mod K), after that stage's previous replay has finished, and the graph decodes it (stream_ops' packed forms) and
    applies S_hat; so the copies run ahead of the computation as with the eager prefetch. K = 0: the graph begins with
    the host-to-device copy into its pool itself (copy and computation then alternate on each stream: the 51-cell plate
    was slower than eager, 159 vs 148-154 s). If the host buffer moves (S25 spill / unspill) the graph is captured again.
  - A resident cell (_Resident): the graph reads the cell's device tensors in place (they do not move after the front
    end).
  - Before a capture, if less than OPL_V3_GRAPH_LOWMEM_GB (3) GiB of device memory is free, the allocator's cached
    blocks are returned to the device first (cudaFree is not allowed during a capture).
  - Not graphed (eager as before): inputs that require gradients, wider than OPL_V3_GRAPH_MAXB (4) columns (the coarse A Z:
    16-column chunks, bound by the GPU), a cell whose capture failed (STATS['failed'], logged once)."""
import os
import time
import torch

ON = os.environ.get('OPL_V3_GRAPH', '1') == '1'
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
    """A narrow apply whose input needs no gradient (PCG runs with autograd enabled, on tensors without gradients; the
    cells' weights are detached copies)."""
    return ON and q.is_cuda and q.dim() == 2 and q.shape[1] <= MAXB and not q.requires_grad


def apply(owner, q, body, fp=None, pre=None, post=None):
    """S_hat q through owner's graph for (width, dtype, current stream), or None (the caller applies eagerly).
    body(qs) -> output: the computation to capture (reads qs, a static input); fp: what the graph's addresses depend on
    besides the owner (a changed fp: capture again); pre() / post(): called on the current stream right before / after
    each replay (e.g. a streamed cell's copy into its stage buffer, the stage's release)."""
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
    if pre is not None:
        pre()
    G.g.replay()
    if post is not None:
        post()
    STATS['replays'] += 1
    return G.out.clone()


LOWMEM = float(os.environ.get('OPL_V3_GRAPH_LOWMEM_GB', '3')) * 2 ** 30


def _capture(q, body, cur, fp):
    t0 = time.perf_counter()
    if torch.cuda.mem_get_info()[0] < LOWMEM:                           # the allocator cannot return cached blocks to the
        torch.cuda.empty_cache()                                        # device during a capture (on the 259-cell plate
        STATS['emptied'] = STATS.get('emptied', 0) + 1                  # 252 captures ran out of memory without this)
    qs = torch.empty_like(q)                                            # static input (outside the pool)
    pool = _POOLS.get(cur.cuda_stream)
    if pool is None:
        pool = _POOLS[cur.cuda_stream] = torch.cuda.graph_pool_handle()
    g = torch.cuda.CUDAGraph()
    cap = _cap_stream(cur)
    cap.wait_stream(cur)
    try:
        with torch.cuda.stream(cap), torch.no_grad():
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
    """All graphs of owners dropped and their pools' memory (and stream_ops' stage buffers) returned to the device
    (after a solve: the sensitivity phase that follows applies the cells eagerly)."""
    for o in owners:
        drop(o)
    _POOLS.clear()
    import stream_ops as SO
    SO.free_stages()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def pool_reserve_gib(nstreams):
    """Device memory to keep free for the graphs (hostmem.resident_auto): per stream OPL_V3_RGB_GRAPH GiB (1.3: S_hat's
    temporaries of a full plate cell with recomputed slot weights and its decoded parts; with OPL_V3_GRAPH_STAGES=0 the
    pool also holds the cell's state: 2.0), plus the stage buffers of stream_ops (0.65 GiB each: a full cell's state)."""
    if not ON:
        return 0.0
    import stream_ops as SO
    per = float(os.environ.get('OPL_V3_RGB_GRAPH', '1.3' if SO.GSTAGES else '2.0'))
    return per * nstreams + 0.65 * SO.GSTAGES


def summary():
    return dict(on=ON, maxb=MAXB, pools=len(_POOLS), **{k: (round(v, 2) if isinstance(v, float) else v) for k, v in STATS.items()})
