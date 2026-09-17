"""Achieved per-entry accuracy, and whether the error is COHERENT or independent.

Required (measured): eps_op = 104 * eps for independent multiplicative per-entry
noise eps. Work gate 0.10 needs eps ~ 1e-3; target gate 0.03 needs eps ~ 2.9e-4.

Two questions here:
  (a) what per-entry relative error does the network actually achieve?
  (b) is that error INDEPENDENT or SYSTEMATIC? Randomly re-sign every entry of the
      actual error E = R_hat - R_star, keeping every magnitude. Independent error is
      unchanged in distribution, so eps_op should not move. A large drop means the
      error is coherent, and coherence - not magnitude - is the binding constraint.
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

def quantiles(v,qs,cap=8_000_000,seed=1):
    g=torch.Generator(device=v.device).manual_seed(seed)
    s=v if v.numel()<=cap else v[torch.randint(0,v.numel(),(cap,),generator=g,device=v.device)]
    return {f'q{int(q*100)}':float(torch.quantile(s.float(),q)) for q in qs}

if __name__=='__main__':
    def dim(p):
        n=np.load(p,mmap_mode='r',allow_pickle=False).shape[0]
        return int((-1+(1+8*n)**.5)/2+.5)
    seat=253; d=dim(REF/f'REFERENCE_{seat:04d}'/'R_UPPER.npy')
    Rs=unpack(REF/f'REFERENCE_{seat:04d}'/'R_UPPER.npy',d)
    upper=torch.triu(torch.ones((d,d),dtype=torch.bool,device=DEV))
    prev=json.loads((OUT/'RESULT.json').read_text()) if (OUT/'RESULT.json').exists() else {}
    out=dict(prev); out.setdefault('seat',seat); out.setdefault('d',d); out['achieved']={}; out['coherence']={}
    g=torch.Generator(device=DEV).manual_seed(4242)
    for arm in ('baseline','relative'):
        Rh=unpack(RUN/arm/f'EVAL_040000_{seat:04d}'/'R_PRED_UPPER.npy',d)
        E=torch.where(upper,Rh-Rs,torch.zeros((),dtype=torch.float64,device=DEV))
        rel=torch.where(upper&(Rs.abs()>0),E.abs()/Rs.abs().clamp_min(1e-300),
                        torch.zeros((),dtype=torch.float64,device=DEV))[upper]
        a=dict(quantiles(rel,(.5,.9,.99)),
               mean=float(rel.mean()),
               frobenius_relative=float(E.norm()/Rs.norm()))
        base,mn,mx=eps_op(Rh,Rs,d); a['eps_op']=base
        a['implied_independent_eps']=base/104.0
        out['achieved'][arm]=a
        print(f"  achieved {arm}: per-entry rel median {a['q50']:.4g} q90 {a['q90']:.4g} "
              f"q99 {a['q99']:.4g} | ||E||/||R|| {a['frobenius_relative']:.4g} | eps_op {base:.4g}",flush=True)
        del rel; gc.collect(); torch.cuda.empty_cache()
        trials=[]
        for k in range(3):
            sign=(torch.randint(0,2,E.shape,generator=g,device=DEV,dtype=torch.int8).to(torch.float64)*2-1)
            Rr=torch.where(upper,Rs+E*sign,torch.zeros((),dtype=torch.float64,device=DEV))
            piv=torch.diagonal(Rr)
            flipped_pivots=int((piv<=0).sum())
            Rr=Rr-torch.diag_embed(piv)+torch.diag_embed(torch.where(piv>0,piv,torch.diagonal(Rh)))
            v,rmn,rmx=eps_op(Rr,Rs,d); trials.append(dict(eps_op=v,mu_min=rmn,mu_max=rmx,
                                                          pivots_restored=flipped_pivots))
            del Rr,sign,piv; gc.collect(); torch.cuda.empty_cache()
            print(f"    resigned trial {k}: eps_op {v:.4g}  (vs {base:.4g} as predicted)",flush=True)
        out['coherence'][arm]=dict(as_predicted=base,resigned=trials,
            mean_resigned=float(np.mean([t['eps_op'] for t in trials])),
            coherence_factor=base/float(np.mean([t['eps_op'] for t in trials])))
        print(f"  => {arm}: coherence factor {out['coherence'][arm]['coherence_factor']:.3g}"
              f"  (1 = error already behaves independently; >>1 = systematic)",flush=True)
        del Rh,E; gc.collect(); torch.cuda.empty_cache()
        (OUT/'RESULT.json').write_text(json.dumps(out,indent=1))
    print('done')
