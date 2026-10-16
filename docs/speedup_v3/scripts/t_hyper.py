"""Hyperedge gather / scatter kernels on a real cell's layers: fused_hyper (one program per hyperedge, BLK columns) vs
tiled variants (EB hyperedges x BC columns per program). Gather: bitwise comparison; scatter: comparison on a
collision-free subset (no atomic races: exact comparison of the per-row sums) and on the full layer (to rounding).
usage: python t_hyper.py BODY_DIR:CASE [--B 1,3,16]"""
import os, sys, json, time, argparse, itertools
SRC = os.environ.get('SRC', '/root/autodl-tmp/OPL/src_v3'); sys.path.insert(0, SRC)
os.environ.setdefault('OPL_TAILT_FUSED', '1'); os.environ.setdefault('OPL_COARSE_FP32', '1')
os.environ.setdefault('OPL_COARSE_ELEM', '1'); os.environ.setdefault('OPL_TET_TRITON', '1')
import models as MD  # noqa
import torch
import triton
import triton.language as tl
import teacher as TE, trainlib as TL, fastnet as FN, bench_deploy as BD, fastidx as FI
import fused_hyper as FH


@triton.jit
def _gather2(src, hn, wgt, deg, out, Eh, BF, H: tl.constexpr, S: tl.constexpr, EB: tl.constexpr, BC: tl.constexpr,
             DEG: tl.constexpr):
    e = tl.program_id(0) * EB + tl.arange(0, EB)
    me = e < Eh
    cols = tl.program_id(1) * BC + tl.arange(0, BC)
    cm = cols < BF
    hh = tl.arange(0, H)
    acc = tl.zeros((EB, H, BC), dtype=tl.float32)
    for s in tl.static_range(S):
        n = tl.load(hn + e * S + s, mask=me, other=0).to(tl.int64)
        w = tl.load(wgt + (e[:, None] * S + s) * H + hh[None, :], mask=me[:, None], other=0.0)
        if DEG:
            w = w / tl.load(deg + n, mask=me, other=1.0)[:, None]
        x = tl.load(src + n[:, None] * BF + cols[None, :], mask=me[:, None] & cm[None, :], other=0.0)
        acc += w[:, :, None] * x[:, None, :]
    ptr = out + (hh[None, :, None].to(tl.int64) * Eh + e[:, None, None]) * BF + cols[None, None, :]
    tl.store(ptr, acc, mask=me[:, None, None] & cm[None, None, :])


@triton.jit
def _scatter2(src, hn, wgt, deg, out, Eh, BF, H: tl.constexpr, S: tl.constexpr, EB: tl.constexpr, BC: tl.constexpr,
              DEG: tl.constexpr):
    e = tl.program_id(0) * EB + tl.arange(0, EB)
    me = e < Eh
    cols = tl.program_id(1) * BC + tl.arange(0, BC)
    cm = cols < BF
    hh = tl.arange(0, H)
    z = tl.load(src + (hh[None, :, None].to(tl.int64) * Eh + e[:, None, None]) * BF + cols[None, None, :],
                mask=me[:, None, None] & cm[None, None, :], other=0.0)
    for s in tl.static_range(S):
        n = tl.load(hn + e * S + s, mask=me, other=0).to(tl.int64)
        w = tl.load(wgt + (e[:, None] * S + s) * H + hh[None, :], mask=me[:, None], other=0.0)
        if DEG:
            w = w / tl.load(deg + n, mask=me, other=1.0)[:, None]
        tl.atomic_add(out + n[:, None] * BF + cols[None, :], tl.sum(w[:, :, None] * z, axis=1), mask=me[:, None] & cm[None, :])


def gather2(src, hn, w, deg, H, use_deg, EB, BC, nw):
    Eh, S = hn.shape; BF = src.shape[1]
    out = torch.empty((H, Eh, BF), dtype=torch.float32, device=src.device)
    _gather2[(triton.cdiv(Eh, EB), triton.cdiv(BF, BC))](src, hn, w, deg, out, Eh, BF, H=H, S=S, EB=EB, BC=BC, DEG=use_deg,
                                                        num_warps=nw)
    return out


def scatter2(src, hn, w, deg, N, use_deg, EB, BC, nw):
    H, Eh, BF = src.shape
    out = torch.zeros((N, BF), dtype=torch.float32, device=src.device)
    _scatter2[(triton.cdiv(Eh, EB), triton.cdiv(BF, BC))](src, hn, w, deg, out, Eh, BF, H=H, S=hn.shape[1], EB=EB, BC=BC,
                                                         DEG=use_deg, num_warps=nw)
    return out


def bench(fn, rep=50):
    fn(); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(rep):
        fn()
    torch.cuda.synchronize(); return 1e6 * (time.perf_counter() - t) / rep


ap = argparse.ArgumentParser(); ap.add_argument('spec'); ap.add_argument('--B', default='1,3,16')
a = ap.parse_args()
FN.FUSED = True; FI.ON = True
body, case = a.spec.split(':')
h = BD.ModelHolder('/root/autodl-tmp/OPL/S1/V2/A3_2grid/best.pt'); BD.TMP.mkdir(parents=True, exist_ok=True)
C = TE.Cell(case, body, log=lambda s_: None, deploy=True); C.assemble_deploy()
BD.netdata(C, body, BD.TMP / case)
geo = TL.Geo(case, body, BD.TMP, neumann=False, log=lambda s_: None, cell=C, load_banks=False)
f = FN.FastNet(h.add(geo), geo)
layers = dict(elem=f.elem[0], face=f.face[0], fringe=f.fringe[0] if len(f.fringe) else None)
F = f.elem[0].W.shape[-1]
rep = {}
g = torch.Generator(device='cuda'); g.manual_seed(0)
for name, L in layers.items():
    if L is None or L.Eh == 0:
        continue
    N, H, Eh = L.N, L.H, L.Eh
    # collision-free subset: hyperedges whose 27 nodes no earlier kept hyperedge touches
    hn_cpu = L.hn.cpu(); used = torch.zeros(N, dtype=torch.bool); keep = []
    for e in range(Eh):
        r = hn_cpu[e]
        if not used[r].any():
            used[r] = True; keep.append(e)
    keep = torch.tensor(keep, device='cuda')
    hn_cf, a_cf, b_cf = L.hn[keep].contiguous(), L.a[keep].contiguous(), L.b[keep].contiguous()
    for B in [int(x) for x in a.B.split(',')]:
        BF = B * F
        X = torch.randn((N, BF), dtype=torch.float32, device='cuda', generator=g)
        Zs = torch.randn((H, Eh, BF), dtype=torch.float32, device='cuda', generator=g)
        Zcf = Zs[:, keep].contiguous()
        r = dict(Eh=Eh, N=N, cf=int(len(keep)))
        g0 = FH.gather(X, L.hn, L.a, L.deg, H, FH.BLK, False)
        s0 = FH.scatter(Zs, L.hn, L.b, L.deg, N, FH.BLK, True)
        s0cf = FH.scatter(Zcf, hn_cf, b_cf, L.deg, N, FH.BLK, True)
        r['orig_us'] = (round(bench(lambda: FH.gather(X, L.hn, L.a, L.deg, H, FH.BLK, False)), 1),
                        round(bench(lambda: FH.scatter(Zs, L.hn, L.b, L.deg, N, FH.BLK, True)), 1))
        best = {}
        for EB, BC, nw in itertools.product((1, 2, 4, 8, 16), (32, 64, 128), (1, 2, 4)):
            if BC > max(32, triton.next_power_of_2(BF)) or EB * H * BC > 8192:
                continue
            try:
                g1 = gather2(X, L.hn, L.a, L.deg, H, False, EB, BC, nw)
                s1cf = scatter2(Zcf, hn_cf, b_cf, L.deg, N, True, EB, BC, nw)
                s1 = scatter2(Zs, L.hn, L.b, L.deg, N, True, EB, BC, nw)
            except Exception as ex:                                                   # noqa
                r.setdefault('fail', []).append([EB, BC, nw, str(ex)[:80]]); continue
            tg = bench(lambda: gather2(X, L.hn, L.a, L.deg, H, False, EB, BC, nw))
            tsc = bench(lambda: scatter2(Zs, L.hn, L.b, L.deg, N, True, EB, BC, nw))
            ok = dict(g_bit=bool(torch.equal(g1, g0)), s_cf_bit=bool(torch.equal(s1cf, s0cf)),
                      s_rel=float((s1 - s0).abs().max() / s0.abs().max()))
            for kind, t in (('g', tg), ('s', tsc)):
                if kind not in best or t < best[kind][0]:
                    best[kind] = (round(t, 1), EB, BC, nw, ok)
        r['best'] = best
        rep[f'{name}_B{B}'] = r
        print(json.dumps({f'{name}_B{B}': r}), flush=True)
print(json.dumps(dict(case=case, F=F, rep=rep)), flush=True)
