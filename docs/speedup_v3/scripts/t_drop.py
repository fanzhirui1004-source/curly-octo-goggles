"""Items 1b / 1c (src_v3 OPL_V3_DROP, OPL_V3_HN32, OPL_V3_TSHARE) on a full plate cell: S_hat (B = 1, 3) and the float64 K
product with the state as before against (i) the network built with int32 node lists and shared transfer indices and
(ii) after the moments and the ghost-face DOF arrays were dropped (as slim_deploy does): largest relative difference to
the result's largest value, next to the same code's rerun difference (atomics); bytes of the node lists, transfer
matrices, moments and face arrays before / after; the fallbacks after the drop (moments formed again, face array formed
again) against the dropped originals (bitwise); S_hat time (eager, warm) before / after.
usage: python t_drop.py BODY_DIR:CASE [--rep 20]"""
import os, sys, json, argparse, time
SRC = os.environ.get('SRC', '/root/autodl-tmp/OPL/src_v3'); sys.path.insert(0, SRC)
os.environ.setdefault('OPL_TAILT_FUSED', '1'); os.environ.setdefault('OPL_COARSE_FP32', '1')
os.environ.setdefault('OPL_COARSE_ELEM', '1'); os.environ.setdefault('OPL_TET_TRITON', '1')
import models as MD  # noqa: F401
import torch
import teacher as TE, trainlib as TL, fastnet as FN, bench_deploy as BD, fastidx as FI, evalnet as EN

ap = argparse.ArgumentParser(); ap.add_argument('spec'); ap.add_argument('--rep', type=int, default=20)
a = ap.parse_args()
FN.FUSED = True; FI.ON = True
body, case = a.spec.split(':')
h = BD.ModelHolder('/root/autodl-tmp/OPL/S1/V2/A3_2grid/best.pt'); BD.TMP.mkdir(parents=True, exist_ok=True)
C = TE.Cell(case, body, log=lambda s_: None, deploy=True); C.assemble_deploy()
BD.netdata(C, body, BD.TMP / case)
geo = TL.Geo(case, body, BD.TMP, neumann=False, log=lambda s_: None, cell=C, load_banks=False)
dev = 'cuda'
f64, f32 = torch.float64, torch.float32


def ev(fn, rep):
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    fn(); torch.cuda.synchronize()
    s.record()
    for _ in range(rep):
        fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / rep                                      # ms


def ubytes(ts):
    seen, n = set(), 0
    for t in ts:
        for x in ([t.crow_indices(), t.col_indices(), t.values()] if t.layout == torch.sparse_csr else [t]):
            st = x.untyped_storage()
            if st.data_ptr() not in seen:
                seen.add(st.data_ptr()); n += st.nbytes()
    return n


res = dict(case=case)
model = h.add(geo)
hn = lambda f: [L.hn for L in f._hypers()]
tr = lambda f: f.R + f.Rt + f.P + f.Pt
g = torch.Generator(device=dev); g.manual_seed(0)
qs = {B: torch.randn((C.np_, B), dtype=f64, device=dev, generator=g) for B in (1, 3)}
x64 = torch.randn((C.nb, 3), dtype=f64, device=dev, generator=g)
FN.HN32 = FN.TSHARE = False                                             # the network as before (int64 node lists)
f0 = FN.FastNet(model, geo)
op0 = EN.FastOp(f0)
res['MiB'] = dict(hn_before=round(ubytes(hn(f0)) / 2 ** 20, 2), transfers_before=round(ubytes(tr(f0)) / 2 ** 20, 2))
base, base2, ms0 = {}, {}, {}
with torch.no_grad():
    for B, q in qs.items():                                             # (before f1 converts the shared cache's lists)
        base[B] = op0.apply(q).clone(); base2[B] = op0.apply(q).clone()
        ms0[B] = round(ev(lambda: op0.apply(q), a.rep), 3)
del op0, f0
FN.HN32 = FN.TSHARE = True
f1 = FN.FastNet(model, geo)
op1 = EN.FastOp(f1)
res['MiB'].update(hn_after=round(ubytes(hn(f1)) / 2 ** 20, 2), transfers_after=round(ubytes(tr(f1)) / 2 ** 20, 2))
res['hn_dtypes'] = sorted({str(t.dtype) for t in hn(f1)})
with torch.no_grad():
    for B, q in qs.items():
        y0, y0b = base[B], base2[B]
        y1 = op1.apply(q).clone()
        nrm = y0.abs().max()
        res[f'B{B}_net'] = dict(rel_hn32=float((y1 - y0).abs().max() / nrm), rel_rerun=float((y0b - y0).abs().max() / nrm),
                                equal=bool(torch.equal(y1, y0)), ms_before=ms0[B], ms_after=round(ev(lambda: op1.apply(q), a.rep), 3))
        print(json.dumps({f'B{B}_net': res[f'B{B}_net']}), flush=True)
    k0 = C.Kx(x64).clone()
    M0 = C.M.clone()
    it0 = C.GF64.__dict__.get('_iTc')
    it0 = it0.clone() if it0 is not None else None
    res['MiB']['moments'] = round(M0.numel() * 8 / 2 ** 20, 2)
    res['MiB']['face_iTc'] = round(it0.numel() * 4 / 2 ** 20, 2) if it0 is not None else None
    res['park_faces'] = bool(C.park_faces())
    res['drop_moments'] = bool(C.drop_moments())
    res['after_drop'] = dict(M_kept=C.__dict__.get('_M') is not None, iTc_kept=C.GF64.__dict__.get('_iTc') is not None,
                             volume_equal=C.volume() == float(M0[:, 0].sum()))
    k1 = C.Kx(x64).clone()
    res['Kx64_equal'] = bool(torch.equal(k0, k1))
    for B, q in qs.items():
        y2 = op1.apply(q).clone()
        res[f'B{B}_after_drop'] = dict(rel=float((y2 - base[B]).abs().max() / base[B].abs().max()),
                                       ms=round(ev(lambda: op1.apply(q), a.rep), 3))
    t = time.perf_counter()
    M2 = C.M                                                            # fallbacks: formed again
    res['fallback_M'] = dict(equal=bool(torch.equal(M2, M0)), rel=float((M2 - M0).abs().max() / M0.abs().max()),
                             s=round(time.perf_counter() - t, 3), kept_after=C.__dict__.get('_M') is not None)
    if it0 is not None:
        it2 = C.GF64.iTc
        res['fallback_iTc_equal'] = bool(it2 is not None and torch.equal(it2.to(it0.device), it0))
print(json.dumps(res), flush=True)
