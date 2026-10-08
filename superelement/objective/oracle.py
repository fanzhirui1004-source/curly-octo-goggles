"""E2: required vs achieved per-entry accuracy, on the real factors.

Required: perturb the TEACHER's factor with independent multiplicative relative
noise of size eps and measure eps_op(eps). Achieved: the actual relative per-entry
error of the prediction. If achieved >> required, no amount of training of this
representation reaches the gate, and the conclusion is structural.
"""
import json,gc,time
from pathlib import Path
import numpy as np, torch
OUT=Path('/root/autodl-tmp/CLAUDE_ORACLE_20260917'); OUT.mkdir(exist_ok=True)
RUN=Path('/root/autodl-tmp/CUTFEM_M4_MULTI_20260917/RUN_TRAIN32_ACCEL_V1')
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
    upper=torch.triu(torch.ones((d,d),dtype=torch.bool,device=DEV))
    out={'seat':seat,'d':d,'required':[],'achieved':{}}
    g=torch.Generator(device=DEV).manual_seed(20260917)
    for e in (1e-6,1e-5,1e-4,1e-3,1e-2,1e-1):
        noise=torch.randn(Rs.shape,generator=g,device=DEV,dtype=torch.float64)
        Rp=torch.where(upper,Rs*(1+e*noise),torch.zeros((),dtype=torch.float64,device=DEV))
        piv=torch.diagonal(Rp)
        if bool((piv<=0).any()): Rp=Rp-torch.diag_embed(piv)+torch.diag_embed(piv.abs().clamp_min(1e-30))
        v,mn,mx=eps_op(Rp,Rs,d); del Rp,noise; gc.collect(); torch.cuda.empty_cache()
        out['required'].append(dict(entry_relative_noise=e,eps_op=v,mu_min=mn,mu_max=mx))
        print(f"  required: entry noise {e:.0e} -> eps_op {v:10.4g}  mu=[{mn:.4g},{mx:.4g}]",flush=True)
    for arm in ('baseline','relative'):
        Rh=unpack(RUN/arm/f'EVAL_040000_{seat:04d}'/'R_PRED_UPPER.npy',d)
        rel=torch.where(upper&(Rs.abs()>0),(Rh-Rs).abs()/Rs.abs().clamp_min(1e-300),
                        torch.zeros((),dtype=torch.float64,device=DEV))
        vals=rel[upper]
        out['achieved'][arm]=dict(
            median=float(vals.median()),q90=float(vals.quantile(.9)),q99=float(vals.quantile(.99)),
            mean=float(vals.mean()),
            frobenius_relative=float((Rh-Rs).norm()/Rs.norm()))
        print(f"  achieved {arm}: median {out['achieved'][arm]['median']:.4g} "
              f"q90 {out['achieved'][arm]['q90']:.4g} q99 {out['achieved'][arm]['q99']:.4g} "
              f"||dR||/||R|| {out['achieved'][arm]['frobenius_relative']:.4g}",flush=True)
        del Rh,rel,vals; gc.collect(); torch.cuda.empty_cache()
    (OUT/'RESULT.json').write_text(json.dumps(out,indent=1)); print('done')
