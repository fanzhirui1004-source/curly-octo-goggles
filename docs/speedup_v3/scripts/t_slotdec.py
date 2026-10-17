"""P17 (OPL_V3_SLOTDEC): the slot weights recomputed by slotdec (split first layer, fused Triton heads) against the stored
ones (models' geometry path) on a full plate cell: per layer type the largest relative difference of a and b (to the
layer's largest |value|), the time of one recompute (slotdec.assign) against today's (S5/S19: _slot_tensors +
_assign_slots), its transient device memory, the per-cell state kept (gn, P) against the slot weights, and S_hat
(eager, B = 1, 3) with the recomputed weights against the stored ones, next to the stored path's rerun difference.
usage: python t_slotdec.py BODY_DIR:CASE [--rep 20]"""
import os, sys, json, argparse
SRC = os.environ.get('SRC', '/root/autodl-tmp/OPL/src_v3'); sys.path.insert(0, SRC)
os.environ.setdefault('OPL_TAILT_FUSED', '1'); os.environ.setdefault('OPL_COARSE_FP32', '1')
os.environ.setdefault('OPL_COARSE_ELEM', '1'); os.environ.setdefault('OPL_TET_TRITON', '1')
os.environ['OPL_V3_SLOTDEC'] = '0'                                      # build in the stored mode, switch below
import models as MD  # noqa
import torch
import teacher as TE, trainlib as TL, fastnet as FN, bench_deploy as BD, fastidx as FI
import slotdec as SD

ap = argparse.ArgumentParser(); ap.add_argument('spec'); ap.add_argument('--rep', type=int, default=20)
a = ap.parse_args()
FN.FUSED = True; FI.ON = True
body, case = a.spec.split(':')
h = BD.ModelHolder('/root/autodl-tmp/OPL/S1/V2/A3_2grid/best.pt'); BD.TMP.mkdir(parents=True, exist_ok=True)
C = TE.Cell(case, body, log=lambda s_: None, deploy=True); C.assemble_deploy()
BD.netdata(C, body, BD.TMP / case)
geo = TL.Geo(case, body, BD.TMP, neumann=False, log=lambda s_: None, cell=C, load_banks=False)
f = FN.FastNet(h.add(geo), geo)
buf = torch.zeros(64 * 2 ** 20, dtype=torch.float32, device='cuda')
dev = 'cuda'


def ev(fn, rep):
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    fn(); torch.cuda.synchronize()
    s.record()
    for _ in range(rep):
        fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / rep                                     # ms


def cold(fn, rep):
    return ev(lambda: (buf.add_(1.0), fn()), rep) - ev(lambda: buf.add_(1.0), rep)


nbytes = lambda ts: sum(t.numel() * t.element_size() for t in ts)
res = dict(case=case, slot_MiB=round(f.slot_bytes() / 2 ** 20, 1))
groups = dict(elem=f.elem, face_full=f.face[:f.L_pre + f.L_post], face_fringe=f.face[f.L_pre + f.L_post:], fringe=f.fringe)
with torch.no_grad():
    stored = {k: [(hl.a.clone(), hl.b.clone()) for hl in v] for k, v in groups.items()}
    g = torch.Generator(device=dev); g.manual_seed(0)
    qs = {B: torch.randn((C.np_, B), dtype=torch.float64, device=dev, generator=g) for B in (1, 3)}
    s_store = {B: f.s_hat(qs[B]).clone() for B in qs}
    s_store2 = {B: f.s_hat(qs[B]).clone() for B in qs}
    t_store = {B: round(ev(lambda: f.s_hat(qs[B]), a.rep), 3) for B in qs}
    f._rc = f._rc_src                                                   # today's recompute (bitwise the stored values)
    res['old_recompute_ms'] = dict(warm=round(ev(lambda: f._assign_slots(*f._slot_tensors()), a.rep), 3),
                                   cold=round(cold(lambda: f._assign_slots(*f._slot_tensors()), a.rep), 3))
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); m0 = torch.cuda.memory_allocated()
    f._assign_slots(*f._slot_tensors()); torch.cuda.synchronize()
    res['old_recompute_peak_MiB'] = round((torch.cuda.max_memory_allocated() - m0) / 2 ** 20, 1)
    f._rc = None
    ok = f.use_slotdec()
    res['use_slotdec'] = ok
    st = f._sd
    res['dec_state_MiB'] = round(nbytes([st['gn'], st['Pab'], st['Px'], st['Pf']]) / 2 ** 20, 1)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); m0 = torch.cuda.memory_allocated()
    SD.assign(f); torch.cuda.synchronize()
    res['dec_peak_MiB'] = round((torch.cuda.max_memory_allocated() - m0) / 2 ** 20, 1)
    diffs = {}
    for k, v in groups.items():
        da = db = 0.0
        for (a0, b0), hl in zip(stored[k], v):
            da = max(da, float((hl.a - a0).abs().max() / a0.abs().max()))
            db = max(db, float((hl.b - b0).abs().max() / b0.abs().max()))
        diffs[k] = dict(a=da, b=db)
    res['rel_diff'] = diffs
    res['dec_recompute_ms'] = dict(warm=round(ev(lambda: SD.assign(f), a.rep), 3), cold=round(cold(lambda: SD.assign(f), a.rep), 3))
    for hl in f._hypers():
        hl.drop_ab()
    for B in qs:
        s_dec = f.s_hat(qs[B])
        nrm = s_store[B].abs().max()
        res[f's_hat_B{B}'] = dict(rel_dec_store=float((s_dec - s_store[B]).abs().max() / nrm),
                                  rel_store_rerun=float((s_store2[B] - s_store[B]).abs().max() / nrm),
                                  l2_dec_store=float((s_dec - s_store[B]).norm() / s_store[B].norm()),
                                  ms_store=t_store[B], ms_dec=round(ev(lambda: f.s_hat(qs[B]), a.rep), 3))
print(json.dumps(res), flush=True)
