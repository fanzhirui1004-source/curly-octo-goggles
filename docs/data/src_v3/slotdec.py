"""src_v3 P17 (OPL_V3_SLOTDEC): fastnet's slot weights (the a, b of every hyperedge layer) recomputed on the GPU in every
application from small per-cell inputs, instead of being stored and streamed (they are about 2/3 of a plate cell's
streamed state: 273 of 410 MiB).

Each slot head is Linear(136, 64) -> GELU -> Linear(64, O) on z = [x_e; gn[n]; slot_s] (x: the element embedding ge, or
the ghost-face embedding gf = face_in(...); n = hn[e, s]). Its first layer splits exactly as
    W z + b = W_x x_e + W_n gn[n] + (W_s slot_s + b) = P[e] + Q[n] + R[s],
P = x W_x^T per hyperedge (kept per cell, Eh x 64), Q = gn W_n^T per node (one GEMM per application for the three heads,
from gn, N x 64 kept per cell), R = slot W_s^T + b (27 x 64, per model), so the 136-wide input (E 27 x 136 floats) is
never formed. One Triton kernel per head and row set then forms u = (P[e] + Q[n]) + R[s], GELU (erf, as nn.GELU), the
second layer (float32 'ieee' dot: no TF32), the scale 0.2 and the bounded model's soft clip (models._SoftClip, knee),
the B3 fold of b (b (1 + lambda c.b3[e, s])), and writes each layer's a and b (Eh x 27 x H, contiguous) directly; the
fringe layers are computed on their hyperedge subsets only. Not bitwise the stored weights: the first layer's sum is
reassociated (rounding level; category B)."""
import os
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

ON = os.environ.get('OPL_V3_SLOTDEC', '1') == '1'
BR = int(os.environ.get('OPL_V3_SLOTDEC_BR', '64'))                      # rows per program


@triton.jit
def _head(P, Q, q_ld, R, idx, emap, W2, b2, cols, Abnd, lam, fac, Ao, Bo, n_rows, layer0, knee,
          CH: tl.constexpr, OS: tl.constexpr, BO: tl.constexpr, LH: tl.constexpr, H: tl.constexpr, NL: tl.constexpr,
          BR: tl.constexpr, CLIP: tl.constexpr, B3: tl.constexpr, EMAP: tl.constexpr):
    """Rows r < n_rows of the row set (r = 27 e' + s; e = emap[e'] with EMAP): u = (P[e] + Q[idx[27 e + s]]) + R[s],
    h = GELU(u); for the OS selected output columns cols (original column o = (ab LH + l) H + hh of the head's
    (2, LH, H) output): y = 0.2 (h W2[o] + b2[o]), soft clip with bound Abnd[o // H], then a (ab = 0) to
    Ao[l - layer0, r, hh], b (ab = 1, times 1 + lam[l] fac[27 e + s] with B3) to Bo[l - layer0, r, hh]."""
    r = tl.program_id(0) * BR + tl.arange(0, BR)
    mr = r < n_rows
    es = r // 27
    s = r - 27 * es
    if EMAP:
        e = tl.load(emap + es, mask=mr, other=0).to(tl.int64)
    else:
        e = es.to(tl.int64)
    rf = e * 27 + s                                                         # row of the full hyperedge set
    n = tl.load(idx + rf, mask=mr, other=0).to(tl.int64)
    c = tl.arange(0, CH)
    m2 = mr[:, None]
    u = tl.load(P + e[:, None] * CH + c[None, :], mask=m2, other=0.0)
    u = u + tl.load(Q + n[:, None] * q_ld + c[None, :], mask=m2, other=0.0)
    u = u + tl.load(R + s[:, None] * CH + c[None, :], mask=m2, other=0.0)
    h = 0.5 * u * (1.0 + libdevice.erf(u * 0.7071067811865476))
    if B3:
        fr = tl.load(fac + rf, mask=mr, other=0.0)
    for j in tl.static_range(OS // BO):
        oc = tl.load(cols + j * BO + tl.arange(0, BO))                       # original columns of this chunk
        w = tl.load(W2 + oc[None, :] * CH + c[:, None])                      # (CH, BO) = W2[oc]^T
        y = tl.dot(h, w, input_precision='ieee') + tl.load(b2 + oc)[None, :]
        x = y * 0.2
        g = oc // H
        if CLIP:
            a = tl.load(Abnd + g)[None, :]
            t = a * knee
            ax = tl.abs(x)
            den = a - t
            th = libdevice.tanh(libdevice.div_rn(ax - t, den))
            v = t + den * th
            wv = tl.where(x > 0, v, tl.where(x < 0, -v, 0.0 * v))
            x = tl.where(ax > t, wv, x)
        ab = g // LH
        l = g - LH * ab
        hh = oc - H * g
        isb = (ab == 1)[None, :]
        if B3:
            fold = 1.0 + tl.load(lam + l)[None, :] * fr[:, None]
            x = tl.where(isb, x * fold, x)
        dst = ((l - layer0)[None, :].to(tl.int64) * n_rows + r[:, None]) * H + hh[None, :]
        ok = m2 & ((l >= layer0) & (l < layer0 + NL))[None, :]
        tl.store(Ao + dst, x, mask=ok & (ab == 0)[None, :])
        tl.store(Bo + dst, x, mask=ok & isb)


def _split(head, Cx):
    """(W_x, W_n, W_s + b as R-maker, second layer) of a slot head Sequential(Linear, GELU, Linear)."""
    lin1, act, lin2 = head[0], head[1], head[2]
    if not (isinstance(lin1, torch.nn.Linear) and isinstance(act, torch.nn.GELU) and act.approximate == 'none'
            and isinstance(lin2, torch.nn.Linear) and len(head) == 3):
        raise ValueError('SLOTDEC: unexpected slot head')
    W = lin1.weight.detach().float()
    return W[:, :Cx], W[:, Cx:2 * Cx], W[:, 2 * Cx:], lin1.bias.detach().float(), \
        lin2.weight.detach().float().contiguous(), lin2.bias.detach().float().contiguous()


def prepare(fast):
    """Per-cell inputs (kept on the FastNet, moved / streamed with it): gn, P per head; per-model tables (shared)."""
    m, c = fast.model, fast.c
    ge, gn = fast._rc_src['ge'].float(), fast._rc_src['gn'].float()
    Cg = ge.shape[1]
    with torch.no_grad():
        heads = {}
        for name, mod in (('ab', m.slot_head), ('xab', m.slot_head_x), ('fab', m.fslot_head)):
            Wx, Wn, Ws, b1, W2, b2 = _split(mod, Cg)
            heads[name] = dict(Wn=Wn, R=(m.slot if name != 'fab' else m.fslot).detach().float() @ Ws.t() + b1,
                               W2=W2, b2=b2)
            heads[name]['Wx'] = Wx
        Wn_cat = torch.cat([heads[k]['Wn'] for k in ('ab', 'xab', 'fab')], 0).contiguous()   # (192, Cg)
        ax = torch.nn.functional.one_hot(c.gp_axis, 3).to(torch.float32)
        gf = m.face_in(torch.cat([ge[c.gp_owner], ge[c.gp_nbr], ax], 1)).float()
        st = dict(gn=gn.contiguous(), Wn=Wn_cat,
                  Pab=(ge @ heads['ab']['Wx'].t()).contiguous(), Px=(ge @ heads['xab']['Wx'].t()).contiguous(),
                  Pf=(gf @ heads['fab']['Wx'].t()).contiguous())
    tabs = {}
    for k in ('ab', 'xab', 'fab'):
        hd = heads[k]
        tabs[k] = dict(R=hd['R'].contiguous(), W2=hd['W2'], b2=hd['b2'])
    st['tabs'] = tabs
    return st


_CUDA_ERR = []


def _cuda_ok():
    """The CUDA head usable (hyperk's library builds; a failure is remembered: the Triton head then)."""
    if _CUDA_ERR:
        return False
    try:
        import hyperk as HK
        HK._lib()
        return True
    except Exception as ex:                                                 # noqa: BLE001
        _CUDA_ERR.append(repr(ex)[:800])
        return False


def _cols(O, H, LH, l0, nl, device):
    """Original output columns of layers [l0, l0 + nl) of a (2, LH, H) head output, in order."""
    cs = [((ab * LH + l) * H + hh) for ab in (0, 1) for l in range(l0, l0 + nl) for hh in range(H)]
    return torch.tensor(cs, dtype=torch.int32, device=device)


NW = int(os.environ.get('OPL_V3_SLOTDEC_NW', '4'))                      # warps per program
CUDA = os.environ.get('OPL_V3_SLOTDEC_CUDA', '1') == '1'                # the CUDA head (hyper_kernel.cu), else Triton


def _run(st, name, P, Q, q_ld, idx, emap, n_e, LH, l0, nl, A, lam, fac, knee, clip, H, br=None, nw=None):
    """Layers [l0, l0 + nl) of head name over n_e hyperedges (emap: their indices in the full set, or None)."""
    t = st['tabs'][name]
    dev = P.device
    key = (name, l0, nl)
    cache = st.setdefault('_cols', {})
    if key not in cache:
        cache[key] = _cols(t['W2'].shape[0], H, LH, l0, nl, dev)
    cols = cache[key]
    OS = cols.numel()
    BO = 32 if OS % 32 == 0 else 16
    n_rows = n_e * 27
    Ao = torch.empty((nl, n_rows, H), dtype=torch.float32, device=dev)
    Bo = torch.empty_like(Ao)
    if n_rows:
        b3 = lam is not None
        if CUDA and br is None and OS in (32, 64) and _cuda_ok() and (emap is None or emap.dtype == idx.dtype):
            import hyperk as HK
            dummy = t['b2']
            rc = HK._lib().sd_head(P.data_ptr(), Q.data_ptr(), int(q_ld), t['R'].data_ptr(), idx.data_ptr(),
                                   emap.data_ptr() if emap is not None else None, t['W2'].data_ptr(), t['b2'].data_ptr(),
                                   cols.data_ptr(), OS, (A if clip else dummy).data_ptr(), (lam if b3 else dummy).data_ptr(),
                                   (fac if b3 else dummy).data_ptr(), Ao.data_ptr(), Bo.data_ptr(), n_rows, int(l0), int(LH),
                                   float(knee), int(bool(clip)), int(b3), int(idx.dtype == torch.int32),
                                   torch.cuda.current_stream(P.device).cuda_stream)
            if rc != 0:
                raise RuntimeError(f'SD_HEAD:{rc}')
            return Ao, Bo
        br_ = br or BR
        _head[(triton.cdiv(n_rows, br_),)](
            P, Q, q_ld, t['R'], idx, emap if emap is not None else idx, t['W2'], t['b2'], cols,
            A if clip else t['b2'], lam if b3 else t['b2'], fac if b3 else t['b2'], Ao, Bo, n_rows, l0, float(knee),
            CH=P.shape[1], OS=OS, BO=BO, LH=LH, H=H, NL=nl, BR=br_, CLIP=clip, B3=b3, EMAP=emap is not None,
            num_warps=nw or NW)
    return Ao, Bo


def assign(fast):
    """Every hyperedge layer's a, b recomputed (P17) and set on the layers (views of per-head buffers)."""
    m, c, st = fast.model, fast.c, fast._sd
    H, L = m.H, m.L_pre + m.L_post
    Lx = max(m.n_fringe, 1)
    Lg = L + m.n_fringe
    b3 = getattr(m, 'b3', False)
    clip = bool(m.bounded)
    if clip and any(m._bst[k] != 'all' for k in ('ab', 'xab', 'fab')):
        raise NotImplementedError('SLOTDEC: mixed / off clip states')
    if clip and m.bound_knee <= 0:
        raise NotImplementedError('SLOTDEC: knee 0 (plain tanh)')
    if getattr(m, 'fringe_soft', False):
        raise NotImplementedError('SLOTDEC: fringe_soft')
    with torch.no_grad():
        Q = st['gn'] @ st['Wn'].t()                                          # (N, 192): the three heads' node parts
        q_ld = Q.shape[1]
        Cn = st['Pab'].shape[1]
        Qab, Qx, Qf = Q[:, :Cn], Q[:, Cn:2 * Cn], Q[:, 2 * Cn:]
        E, nf = c.en.shape[0], c.fn.shape[0]
        bnd = lambda k: getattr(m, 'bnd_' + k).reshape(-1).float().contiguous()
        lamv = lambda k: getattr(m, 'b3_' + k).detach().float().contiguous() if b3 else None
        Ae, Be = _run(st, 'ab', st['Pab'], Qab, q_ld, c.en, None, E, L, 0, len(fast.elem), bnd('ab'), lamv('le'),
                      c.b3_el if b3 else None, m.bound_knee, clip, H)
        for l, hl in enumerate(fast.elem):
            hl.set_ab(Ae[l].view(E, 27, H), Be[l].view(E, 27, H))
        nfull = min(L, len(fast.face))
        Af, Bf = _run(st, 'fab', st['Pf'], Qf, q_ld, c.fn, None, nf, Lg, 0, nfull, bnd('fab'), lamv('lf'),
                      c.b3_gp if b3 else None, m.bound_knee, clip, H)
        for g in range(nfull):
            fast.face[g].set_ab(Af[g].view(nf, 27, H), Bf[g].view(nf, 27, H))
        if len(fast.face) > nfull:
            fi = fast._fr_gp
            nfr = fi.numel()
            Ar, Brr = _run(st, 'fab', st['Pf'], Qf, q_ld, c.fn, fi, nfr, Lg, nfull, len(fast.face) - nfull, bnd('fab'),
                           lamv('lf'), c.b3_gp if b3 else None, m.bound_knee, clip, H)
            for g in range(nfull, len(fast.face)):
                fast.face[g].set_ab(Ar[g - nfull].view(nfr, 27, H), Brr[g - nfull].view(nfr, 27, H))
        if len(fast.fringe):
            xi = fast._fr_el
            nxe = xi.numel()
            Ax, Bx = _run(st, 'xab', st['Px'], Qx, q_ld, c.en, xi, nxe, Lx, 0, len(fast.fringe), bnd('xab'), lamv('lx'),
                          c.b3_el if b3 else None, m.bound_knee, clip, H)
            for l, hl in enumerate(fast.fringe):
                hl.set_ab(Ax[l].view(nxe, 27, H), Bx[l].view(nxe, 27, H))
