"""How few factor entries can represent S to the gate at all?

Keep the k largest-magnitude entries of the TEACHER's factor (pivots always kept,
since a Cholesky factor with a zero pivot is not a factor) and measure eps_op(k).
This is a pure representability question with the network removed entirely: it
upper-bounds eps_op for any k-entry magnitude-sparse factor. Hierarchical low rank
does better at equal k (12.23e6 numbers -> eps_op 2.6e-2 on seat 0328), so read this
as the order of magnitude, not the optimum.

Cost frame: the measured emission rate is 3.97e7 numbers/s, so a 1 ms budget allows
~4e4 numbers and a 10 ms budget ~4e5.
"""
import json,gc,time
from pathlib import Path
import numpy as np, torch
OUT=Path('/root/autodl-tmp/CLAUDE_TRUNC_20260917'); OUT.mkdir(exist_ok=True)
REF=Path('/root/autodl-tmp/CUTFEM_SUPERELEMENT_LABELS'); DEV='cuda:0'

def unpack(path,d):
    p=np.load(path,mmap_mode='r',allow_pickle=False)
    R=torch.zeros((d,d),dtype=torch.float64,device=DEV); off=0
    for row in range(d):
        n=d-row; R[row,row:]=torch.from_numpy(np.ascontiguousarray(p[off:off+n])).to(DEV); off+=n
    return R

def eps_op(Rh,Rs,d):
    Mt=torch.linalg.solve_triangular(Rs.T,Rh.T,upper=False); M=Mt.T.contiguous(); del Mt
    W=M.T@M; del M; gc.collect(); torch.cuda.empty_cache()
    mu=torch.linalg.eigvalsh((W+W.T)*.5); del W; gc.collect(); torch.cuda.empty_cache()
    mn,mx=float(mu.min()),float(mu.max()); return max(abs(mn-1),abs(mx-1)),mn,mx

if __name__=='__main__':
    def dim(p):
        n=np.load(p,mmap_mode='r',allow_pickle=False).shape[0]
        return int((-1+(1+8*n)**.5)/2+.5)
    seat=253; d=dim(REF/f'REFERENCE_{seat:04d}'/'R_UPPER.npy')
    Rs=unpack(REF/f'REFERENCE_{seat:04d}'/'R_UPPER.npy',d)
    strict=torch.triu(torch.ones((d,d),dtype=torch.bool,device=DEV),1)
    vals=Rs[strict].abs()
    total=int(strict.sum())+d
    out=dict(seat=seat,d=d,dense_entries=total,emission_rate_numbers_per_second=3.974e7,cases=[])
    print(f'd={d} dense entries={total}',flush=True)
    for k in (1e4,3e4,1e5,3e5,1e6,3e6,1e7,3e7):
        k=int(k); koff=max(k-d,1)
        if koff>=vals.numel(): thr=0.0
        else: thr=float(torch.kthvalue(vals.float().cpu(),vals.numel()-koff).values)
        keep=strict&(Rs.abs()>thr)
        Rt=torch.where(keep,Rs,torch.zeros((),dtype=torch.float64,device=DEV))
        Rt=Rt+torch.diag_embed(torch.diagonal(Rs))
        kept=int(keep.sum())+d
        t=time.time(); v,mn,mx=eps_op(Rt,Rs,d); del Rt,keep; gc.collect(); torch.cuda.empty_cache()
        ms=kept/3.974e7*1e3
        out['cases'].append(dict(target_k=k,kept=kept,fraction_of_dense=kept/total,
                                 eps_op=v,mu_min=mn,mu_max=mx,emission_ms=ms))
        print(f'  k={kept:>10} ({kept/total:7.3%} of dense, {ms:8.2f} ms to emit)  '
              f'eps_op={v:12.5g}  mu=[{mn:.4g},{mx:.4g}]  ({time.time()-t:.0f}s)',flush=True)
        (OUT/'RESULT.json').write_text(json.dumps(out,indent=1))
    print('done')
