"""src_v3 diagnostic (OPL_V3_PROF_ITER=k, default off): a device-level picture of one PCG iteration's operator
application (lat_precond.pcg profiles the matvec of iteration k with torch.profiler; nothing else changes).

Summary (recorded in the solve record as 'prof'):
  wall_ms             host time of the profiled call (synchronised before and after)
  window_ms           first device event to last
  kernel_sum_ms       sum of kernel durations; kernel_busy_ms: union of kernel intervals over all streams (device busy)
  concurrency         kernel_sum / kernel_busy (> 1: kernels of different streams overlap)
  memcpy_* / memset_* the same for copies (copy engines) and fills; h2d_GB: bytes host -> device
  idle_ms             window minus the union of kernels, copies and fills
  per_stream          per CUDA stream: kernels, kernel ms (sum), copies ms, busy ms
  top_kernels         name, ms, count
  cpu_api             CUDA runtime / driver calls on the host: name -> ms, count (graph launches, waits, syncs)
OPL_V3_PROF_TRACE=1 also keeps the chrome trace (gzip) next to OPL_V3_PROF_OUT."""
import os, json, time, tempfile, gzip, shutil
import torch

ITER = int(os.environ.get('OPL_V3_PROF_ITER', '0') or 0)
OUT = os.environ.get('OPL_V3_PROF_OUT')
TRACE = os.environ.get('OPL_V3_PROF_TRACE', '0') == '1'


def _union(iv):
    tot, cs, ce = 0.0, None, None
    for s, e in sorted(iv):
        if cs is None or s > ce:
            if cs is not None:
                tot += ce - cs
            cs, ce = s, e
        elif e > ce:
            ce = e
    if cs is not None:
        tot += ce - cs
    return tot


def summarize(trace_path, wall_s, top=25):
    ev = json.load(open(trace_path))
    ev = ev['traceEvents'] if isinstance(ev, dict) else ev
    K, Mc, Ms, api = [], [], [], {}
    per = {}
    h2d = 0.0
    for e in ev:
        if e.get('ph') != 'X':
            continue
        cat = e.get('cat', '')
        ts, dur = float(e.get('ts', 0)), float(e.get('dur', 0))
        args = e.get('args', {}) or {}
        if cat in ('kernel', 'gpu_memcpy', 'gpu_memset'):
            sid = str(args.get('stream', '?'))
            p = per.setdefault(sid, dict(kernels=0, kernel_ms=0.0, memcpy_ms=0.0, iv=[]))
            p['iv'].append((ts, ts + dur))
            if cat == 'kernel':
                K.append((ts, ts + dur, e.get('name', '')))
                p['kernels'] += 1; p['kernel_ms'] += dur / 1e3
            elif cat == 'gpu_memcpy':
                Mc.append((ts, ts + dur))
                p['memcpy_ms'] += dur / 1e3
                if 'HtoD' in e.get('name', ''):
                    h2d += float(args.get('bytes', 0) or 0)
            else:
                Ms.append((ts, ts + dur))
        elif cat in ('cuda_runtime', 'cuda_driver'):
            n = e.get('name', '')
            a = api.setdefault(n, [0.0, 0])
            a[0] += dur / 1e3; a[1] += 1
    dev = [(s, e) for s, e, _ in K] + Mc + Ms
    if not dev:
        return dict(wall_ms=round(1e3 * wall_s, 2), note='no device events')
    t0, t1 = min(s for s, _ in dev), max(e for _, e in dev)
    kb = _union([(s, e) for s, e, _ in K])
    ks = sum(e - s for s, e, _ in K)
    by = {}
    for s, e, n in K:
        b = by.setdefault(n, [0.0, 0])
        b[0] += (e - s) / 1e3; b[1] += 1
    out = dict(wall_ms=round(1e3 * wall_s, 2), window_ms=round((t1 - t0) / 1e3, 2), kernels=len(K),
               kernel_sum_ms=round(ks / 1e3, 2), kernel_busy_ms=round(kb / 1e3, 2),
               concurrency=round(ks / kb, 3) if kb > 0 else None,
               memcpy_n=len(Mc), memcpy_sum_ms=round(sum(e - s for s, e in Mc) / 1e3, 2), memcpy_busy_ms=round(_union(Mc) / 1e3, 2),
               h2d_GB=round(h2d / 1e9, 3), memset_sum_ms=round(sum(e - s for s, e in Ms) / 1e3, 2),
               kernel_and_memcpy_overlap_ms=round((kb + _union(Mc) - _union([(s, e) for s, e, _ in K] + Mc)) / 1e3, 2),
               idle_ms=round(((t1 - t0) - _union(dev)) / 1e3, 2),
               per_stream={k: dict(kernels=v['kernels'], kernel_ms=round(v['kernel_ms'], 2), memcpy_ms=round(v['memcpy_ms'], 2),
                                   busy_ms=round(_union(v['iv']) / 1e3, 2)) for k, v in per.items()},
               top_kernels=[(n[:80], round(v[0], 3), v[1]) for n, v in sorted(by.items(), key=lambda kv: -kv[1][0])[:top]],
               cpu_api={n: (round(v[0], 2), v[1]) for n, v in sorted(api.items(), key=lambda kv: -kv[1][0])[:12]})
    return out


class Probe:
    """with Probe() as p: <device work>  ->  p.summary (dict)."""

    def __enter__(self):
        from torch.profiler import profile, ProfilerActivity
        torch.cuda.synchronize()
        self.p = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
        self.p.__enter__()
        self.t = time.perf_counter()
        return self

    def __exit__(self, *a):
        torch.cuda.synchronize()
        wall = time.perf_counter() - self.t
        self.p.__exit__(*a)
        d = tempfile.mkdtemp(prefix='gpuprof_')
        try:
            tp = os.path.join(d, 'trace.json')
            self.p.export_chrome_trace(tp)
            try:
                self.summary = summarize(tp, wall)
            except Exception as e:                                          # noqa: BLE001 (diagnostic only)
                self.summary = dict(wall_ms=round(1e3 * wall, 2), error=repr(e)[:300])
            if OUT:
                with open(OUT, 'w') as f_:
                    json.dump(self.summary, f_, indent=1)
                if TRACE:
                    with open(tp, 'rb') as fi, gzip.open(OUT + '.trace.json.gz', 'wb') as fo:
                        shutil.copyfileobj(fi, fo)
        finally:
            shutil.rmtree(d, ignore_errors=True)
        return False
