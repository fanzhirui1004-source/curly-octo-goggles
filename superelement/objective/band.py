"""E1: inside the off-diagonal, where does the gate error live - near or far?

Uses Codex's own patch structure: patch_size=32 contiguous node groups in compiler
order, so same-patch <=> r//32 == c//32, exactly their bucket definition. Replaces
selected off-diagonal blocks with the teacher's and re-measures eps_op.
"""
import json, gc, time
from pathlib import Path
import numpy as np, torch
OUT=Path('/root/autodl-tmp/CLAUDE_BAND_20260917'); OUT.mkdir(exist_ok=True)
RUN=Path('/root/autodl-tmp/CUTFEM_M4_MULTI_20260917/RUN_TRAIN32_ACCEL_V1')
REF=Path('/root/autodl-tmp/CUTFEM_SUPERELEMENT_LABELS'); DEV='cuda:0'; PATCH=32

def unpack(path,d):
    p=np.load(path,mmap_mode='r',allow_pickle=False)
    R=torch.zeros((d,d),dtype=torch.float64,device=DEV); off=0
    for row in range(d):
        n=d-row; R[row,row:]=torch.from_numpy(np.ascontiguousarray(p[off:off+n])).to(DEV); off+=n
    return R

def spectrum(Rh,Rs,d):
    Mt=torch.linalg.solve_triangular(Rs.T,Rh.T,upper=False); M=Mt.T.contiguous(); del Mt
    W=M.T@M; tr=float(torch.diagonal(W).sum()/d)
    ld=float(2*(torch.log(torch.diagonal(Rh)).sum()-torch.log(torch.diagonal(Rs)).sum()))
    del M; gc.collect(); torch.cuda.empty_cache()
    mu=torch.linalg.eigvalsh((W+W.T)*.5); del W; gc.collect(); torch.cuda.empty_cache()
    mn,mx=float(mu.min()),float(mu.max())
    return dict(mu_min=mn,mu_max=mx,eps_op=max(abs(mn-1),abs(mx-1)),D_closed=tr-ld/d-1.0)

def masks(d):
    idx=torch.arange(d,device=DEV); node=idx//3; patch=node//PATCH
    r=node[:,None]; c=node[None,:]; p=patch[:,None]; q=patch[None,:]
    upper=torch.triu(torch.ones((d,d),dtype=torch.bool,device=DEV),1)
    same=(p==q)&upper; cross=(p!=q)&upper; dist=(c-r).abs()
    return upper,same,cross,dist

if __name__=='__main__':
    def dim(p):
        n=np.load(p,mmap_mode='r',allow_pickle=False).shape[0]
        return int((-1+(1+8*n)**.5)/2+.5)
    out={}
    for seat in (253,403):
        for arm in ('baseline','relative'):
            d=dim(REF/f'REFERENCE_{seat:04d}'/'R_UPPER.npy')
            Rs=unpack(REF/f'REFERENCE_{seat:04d}'/'R_UPPER.npy',d)
            Rh=unpack(RUN/arm/f'EVAL_040000_{seat:04d}'/'R_PRED_UPPER.npy',d)
            upper,same,cross,dist=masks(d)
            cases={'as_predicted':None,'teacher_same_patch':same,'teacher_cross_patch':cross}
            for w in (32,128,512,2048):
                cases[f'teacher_cross_within_{w}']=cross&(dist<=w*3)
                cases[f'teacher_cross_beyond_{w}']=cross&(dist>w*3)
            res={}
            for name,m in cases.items():
                t=time.time(); R=Rh if m is None else torch.where(m,Rs,Rh)
                res[name]=spectrum(R,Rs,d); res[name]['blocks_replaced']=0 if m is None else int(m.sum())
                res[name]['seconds']=time.time()-t
                if m is not None: del R
                gc.collect(); torch.cuda.empty_cache()
                print(f"  {arm:<9}{seat:04d} {name:<28} eps_op={res[name]['eps_op']:10.4g} "
                      f"D={res[name]['D_closed']:.5g} n={res[name]['blocks_replaced']:>11}",flush=True)
            out[f'{arm}_{seat:04d}']=res
            (OUT/'RESULT.json').write_text(json.dumps(out,indent=1))
            del Rs,Rh,upper,same,cross,dist; gc.collect(); torch.cuda.empty_cache()
    print('done')
