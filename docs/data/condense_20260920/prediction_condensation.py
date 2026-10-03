"""Does condensing the free cut surface out make the PREDICTED operator better or worse?

For each (arm, seat) with an exported A_PRED_UPPER.npy: build the trace operator S = B^T A B for
teacher and prediction with the same frozen quotient, condense the kind-1 (cut) coordinates out,
and compare the pencil of the condensed pair against the pencil of the full pair.
"""
import json, sys, time, numpy as np, torch
from pathlib import Path
from scipy.linalg import cho_factor, cho_solve
SRC = Path('/root/autodl-tmp/CLAUDE_SQRTHEAD_20260917/src_v5')
sys.path.insert(0, str(SRC)); sys.path.insert(0, '/root/autodl-tmp/CLAUDE_EQUI_20260918/src')
from stage_cutfem_m4.quotient import RigidQuotient
from superelement.equi.context import compile_equi_inputs
torch.backends.cuda.matmul.allow_tf32 = False
MAN = json.loads(Path('/root/autodl-tmp/CLAUDE_LABELS_20260917/V2_LABELS.json').read_text())
BY = {int(r['seat']): r for r in MAN}
W = Path('/root/autodl-tmp/CLAUDE_EQUI_20260918')
dev = torch.device('cuda:0')

def upper(path, d):
    p = np.load(path, mmap_mode='r')
    out = np.zeros((d, d)); off = 0
    for row in range(d):
        out[row, row:] = p[off:off + d - row]; off += d - row
    return out

def dense_S(A, quot):
    left = quot.lift(A.T.contiguous()); S = quot.lift(left.T.contiguous())
    return .5 * (S + S.T)

def pencil(That, Tstar, drop=6):
    w, V = np.linalg.eigh(Tstar)
    keep = V[:, drop:]; wk = w[drop:]
    Wm = keep / np.sqrt(wk)
    mu = np.linalg.eigvalsh(Wm.T @ That @ Wm)
    return mu

print(f"{'arm':>9} {'seat':>7} {'nbox':>6} {'ncut':>6} {'relS':>7} {'relT':>7} "
      f"{'muS_min':>9} {'muS_max':>10} {'muT_min':>9} {'muT_max':>10} {'outS':>6} {'outT':>6} {'sec':>5}")
for arm, seat in [(a, s) for a, s in eval(sys.argv[1])]:
    t0 = time.time()
    ev = W / arm / f'EVAL_{seat:04d}' / 'A_PRED_UPPER.npy'
    if not ev.exists():
        print(f"{arm:>9} {seat:>7}  MISSING {ev}"); continue
    row = BY[seat]; ref = Path(row['reference'])
    d = int(json.loads((ref / 'RESULT.json').read_text())['dimension']); q = d + 6
    cache = dict(np.load(row['trace_cache'], allow_pickle=False))
    meta = json.loads((Path(row['trace_cache']).parent / 'INPUT.json').read_text())['metadata']
    ctx = compile_equi_inputs(cache, meta)
    kind = np.asarray(cache['kind']).astype(np.int64)
    quot = RigidQuotient(torch.from_numpy(np.asarray(cache['rigid'], dtype=np.float64)).to(dev),
                         torch.from_numpy(cache['order']).to(dev))
    Rt = torch.from_numpy(upper(ref / 'R_UPPER.npy', d)).to(dev)
    Astar = Rt.T @ Rt
    Up = torch.from_numpy(upper(ev, d)).to(dev)
    Ahat = Up.T @ Up
    Ss = dense_S(Astar, quot).cpu().numpy(); del Astar
    Sh = dense_S(Ahat, quot).cpu().numpy(); del Ahat, Rt, Up
    torch.cuda.empty_cache()
    box = np.repeat(kind == 0, 3); cut = ~box
    ib = np.flatnonzero(box); ic = np.flatnonzero(cut)
    relS = float(np.linalg.norm(Sh - Ss) / np.linalg.norm(Ss))
    muS = pencil(Sh, Ss)
    def condense(S):
        Scc = np.array(S[np.ix_(ic, ic)], order='F'); Scb = np.array(S[np.ix_(ic, ib)])
        cf = cho_factor(Scc, lower=True, check_finite=False)
        return 0.5 * ((S[np.ix_(ib, ib)] - Scb.T @ cho_solve(cf, Scb, check_finite=False)) +
                      (S[np.ix_(ib, ib)] - Scb.T @ cho_solve(cf, Scb, check_finite=False)).T)
    Ts = condense(Ss); Th = condense(Sh)
    relT = float(np.linalg.norm(Th - Ts) / np.linalg.norm(Ts))
    muT = pencil(Th, Ts)
    outS = float(np.mean((muS < .9) | (muS > 1.1))); outT = float(np.mean((muT < .9) | (muT > 1.1)))
    print(f"{arm:>9} {seat:>7} {ib.size:>6} {ic.size:>6} {relS:>7.4f} {relT:>7.4f} "
          f"{muS.min():>9.2e} {muS.max():>10.3e} {muT.min():>9.2e} {muT.max():>10.3e} "
          f"{outS:>6.3f} {outT:>6.3f} {time.time()-t0:>5.1f}", flush=True)
    del Ss, Sh, Ts, Th
