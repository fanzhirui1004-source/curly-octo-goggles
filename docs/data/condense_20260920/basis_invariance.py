"""Is the condensed operator invariant to a change of basis in the eliminated block, numerically?

The teacher picks a different basis for the cut-surface residual subspace on a rotated cell
(witness_d: 144 of 1436 rows shared) but the SAME subspace (1.6e-15).  Algebraically
S_BC V (V^T S_CC V)^-1 V^T S_CB = S_BC S_CC^-1 S_CB for any invertible V, so the condensed
operator cannot see the basis.  This checks it in float64 at realistic conditioning.
"""
import json, os, sys, time, numpy as np
from scipy.linalg import cho_factor, cho_solve
L=json.load(open('/root/autodl-tmp/CLAUDE_LABELS_20260917/V2_LABELS.json')); BY={int(e['seat']):e for e in L}
def unpack_upper(path,q):
    v=np.load(path, mmap_mode='r'); M=np.zeros((q,q)); M[np.triu_indices(q)]=np.asarray(v,dtype=np.float64)
    M+=M.T; M[np.diag_indices(q)]*=0.5; return M
def condense(SBB,SBC,SCC):
    cf=cho_factor(np.array(SCC,order='F'),lower=True,check_finite=False)
    T=SBB-SBC@cho_solve(cf,SBC.T,check_finite=False); return 0.5*(T+T.T)
print(f"{'seat':>7} {'ncut':>6} {'cond(V)':>9} {'rel dT':>10} {'sec':>6}")
for seat in [int(x) for x in sys.argv[1:]]:
    t0=time.time(); e=BY[seat]; q=int(e['q'])
    nc=3*int(json.load(open(os.path.join(e['packet'],'SAMPLE.json')))['additional_scalar_cut_coordinates']); nb=q-nc
    S=unpack_upper(os.path.join(e['packet'],'S_UPPER.npy'), q)
    SBB=np.array(S[:nb,:nb]); SBC=np.array(S[:nb,nb:]); SCC=np.array(S[nb:,nb:]); del S
    T=condense(SBB,SBC,SCC)
    rng=np.random.default_rng(seat)
    Q,_=np.linalg.qr(rng.standard_normal((nc,nc)))
    d=np.exp(rng.uniform(np.log(1.0), np.log(25.0), nc))          # cond(V) = 25, the teacher's own basis cond is ~5
    V=Q*d
    T2=condense(SBB, SBC@V, V.T@SCC@V)
    print(f"{seat:>7} {nc:>6} {float(d.max()/d.min()):>9.2f} {float(np.linalg.norm(T2-T)/np.linalg.norm(T)):>10.3e} {time.time()-t0:>6.1f}", flush=True)
