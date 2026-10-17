"""P17 parts: device time of slotdec's pieces on a full plate cell (Q GEMM; each head kernel: element ab, element fringe
xab, face full, face fringe) for rows per program BR in 32 / 64 / 128 and 4 / 8 warps; the floor estimate (bytes
written, multiply-adds). usage: python t_slotdec2.py BODY_DIR:CASE"""
import os, sys, json
SRC = os.environ.get('SRC', '/root/autodl-tmp/OPL/src_v3'); sys.path.insert(0, SRC)
os.environ.setdefault('OPL_TAILT_FUSED', '1'); os.environ.setdefault('OPL_COARSE_FP32', '1')
os.environ.setdefault('OPL_COARSE_ELEM', '1'); os.environ.setdefault('OPL_TET_TRITON', '1')
os.environ['OPL_V3_SLOTDEC'] = '1'
import models as MD  # noqa
import torch
import teacher as TE, trainlib as TL, fastnet as FN, bench_deploy as BD, fastidx as FI
import slotdec as SD
FN.FUSED = True; FI.ON = True
body, case = sys.argv[1].split(':')
h = BD.ModelHolder('/root/autodl-tmp/OPL/S1/V2/A3_2grid/best.pt'); BD.TMP.mkdir(parents=True, exist_ok=True)
C = TE.Cell(case, body, log=lambda s_: None, deploy=True); C.assemble_deploy()
BD.netdata(C, body, BD.TMP / case)
geo = TL.Geo(case, body, BD.TMP, neumann=False, log=lambda s_: None, cell=C, load_banks=False)
f = FN.FastNet(h.add(geo), geo)


def ev(fn, rep=20):
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    fn(); torch.cuda.synchronize()
    s.record()
    for _ in range(rep):
        fn()
    e.record(); torch.cuda.synchronize()
    return round(1000 * s.elapsed_time(e) / rep, 1)                    # us


m, c, st = f.model, f.c, f._sd
H, L = m.H, m.L_pre + m.L_post
Lx, Lg = max(m.n_fringe, 1), L + m.n_fringe
res = dict(case=case, sd=f._sd is not None, assign_us=ev(lambda: SD.assign(f)))
with torch.no_grad():
    res['Q_gemm_us'] = ev(lambda: st['gn'] @ st['Wn'].t())
    Q = st['gn'] @ st['Wn'].t(); q_ld = Q.shape[1]; Cn = st['Pab'].shape[1]
    Qab, Qx, Qf = Q[:, :Cn], Q[:, Cn:2 * Cn], Q[:, 2 * Cn:]
    E, nf = c.en.shape[0], c.fn.shape[0]
    bnd = lambda k: getattr(m, 'bnd_' + k).reshape(-1).float().contiguous()
    lamv = lambda k: getattr(m, 'b3_' + k).detach().float().contiguous()
    nfull = min(L, len(f.face))
    jobs = dict(elem=lambda br, nw: SD._run(st, 'ab', st['Pab'], Qab, q_ld, c.en, None, E, L, 0, len(f.elem), bnd('ab'), lamv('le'), c.b3_el, m.bound_knee, True, H, br, nw),
                face=lambda br, nw: SD._run(st, 'fab', st['Pf'], Qf, q_ld, c.fn, None, nf, Lg, 0, nfull, bnd('fab'), lamv('lf'), c.b3_gp, m.bound_knee, True, H, br, nw),
                face_fr=lambda br, nw: SD._run(st, 'fab', st['Pf'], Qf, q_ld, c.fn, f._fr_gp, f._fr_gp.numel(), Lg, nfull, len(f.face) - nfull, bnd('fab'), lamv('lf'), c.b3_gp, m.bound_knee, True, H, br, nw),
                xab=lambda br, nw: SD._run(st, 'xab', st['Px'], Qx, q_ld, c.en, f._fr_el, f._fr_el.numel(), Lx, 0, len(f.fringe), bnd('xab'), lamv('lx'), c.b3_el, m.bound_knee, True, H, br, nw))
    rows = dict(elem=E * 27, face=nf * 27, face_fr=f._fr_gp.numel() * 27, xab=f._fr_el.numel() * 27)
    cols = dict(elem=64, face=64, face_fr=32, xab=32)
    for k, fn in jobs.items():
        r = dict(rows=rows[k], cols=cols[k], MB_out=round(rows[k] * cols[k] * 4 / 1e6, 1), GFMA=round(rows[k] * 64 * cols[k] / 1e9, 3))
        r['cuda'] = ev(lambda: fn(None, None))
        oc, ot = fn(None, None), fn(64, 4)
        r['cuda_vs_triton'] = max(float((oc[0] - ot[0]).abs().max() / ot[0].abs().max()), float((oc[1] - ot[1]).abs().max() / ot[1].abs().max()))
        for br in (32, 64):
            for nw in (4,):
                try:
                    r[f'br{br}_nw{nw}'] = ev(lambda: fn(br, nw))
                except Exception as ex:                                 # noqa: BLE001
                    r[f'br{br}_nw{nw}'] = repr(ex)[:120]
        res[k] = r
        print(json.dumps({k: r}), flush=True)
print(json.dumps(res), flush=True)
