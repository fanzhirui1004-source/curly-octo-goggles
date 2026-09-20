"""H3: is part of the cut family structurally unlearnable, and is T even defined?

A cut that detaches a fragment gives S_CC a six-dimensional nullspace per detached fragment, and
then T = S_BB - S_BC S_CC^-1 S_CB does not exist.  Sweep every cut seat: build S_CC from the packed
upper triangle without materialising S, Cholesky it (which fails outright if it is not SPD), and
estimate kappa(S_CC) from the factor with a LAPACK triangular condition estimator.  Teacher-side
only; no training, no label written.
"""
import json, os, sys, time, numpy as np
from scipy.linalg import cholesky, lapack
L=json.load(open('/root/autodl-tmp/CLAUDE_LABELS_20260917/V2_LABELS.json'))
desc={r['seat']:r for r in json.load(open('/root/_seat_desc.json'))}
OUT='/root/autodl-tmp/CLAUDE_CONDENSE_20260920/H3_SCC.jsonl'
done=set()
if os.path.exists(OUT):
    done={json.loads(l)['seat'] for l in open(OUT) if l.strip()}

def cut_block(path, q, nb):
    """rows/cols nb..q-1 of the symmetric matrix stored as a row-major upper triangle."""
    v=np.load(path, mmap_mode='r')
    n=q-nb; M=np.zeros((n,n))
    off=nb*q - nb*(nb-1)//2                                  # start of row nb
    for r in range(nb, q):
        seg=q-r
        M[r-nb, r-nb:]=v[off:off+seg]
        off+=seg
    M+=M.T; M[np.diag_indices(n)]*=0.5
    return M

seats=[int(e['seat']) for e in L if (desc.get(int(e['seat']),{}).get('n_kind1') or 0)>0]
seats=[s for s in seats if s not in done]
print(f'{len(seats)} cut seats to sweep', flush=True)
BY={int(e['seat']):e for e in L}
for seat in seats:
    t0=time.time(); e=BY[seat]; q=int(e['q'])
    try:
        nc=3*int(json.load(open(os.path.join(e['packet'],'SAMPLE.json')))['additional_scalar_cut_coordinates'])
        nb=q-nc
        Scc=cut_block(os.path.join(e['packet'],'S_UPPER.npy'), q, nb)
        d=np.diag(Scc).copy()
        rec=dict(seat=seat, q=q, n_box=nb, n_cut=nc, diag_min=float(d.min()), diag_max=float(d.max()))
        try:
            R=cholesky(Scc, lower=False, check_finite=False)
            piv=np.abs(np.diag(R))
            # kappa_1(R) from LAPACK's estimator, squared -> kappa(S_CC)
            rcond=lapack.dtrcon(R, uplo='U', norm='1')[0]
            rec.update(spd=True, pivot_min=float(piv.min()), pivot_max=float(piv.max()),
                       pivot_ratio=float(piv.max()/piv.min()),
                       kappa_scc=float(1.0/rcond)**2 if rcond>0 else float('inf'),
                       near_null=int((piv < 1e-8*piv.max()).sum()))
        except Exception as ex:
            rec.update(spd=False, cholesky_error=f'{type(ex).__name__}: {ex}')
        rec['seconds']=round(time.time()-t0,1)
    except Exception as ex:
        rec=dict(seat=seat, error=f'{type(ex).__name__}: {ex}', seconds=round(time.time()-t0,1))
    with open(OUT,'a') as f: f.write(json.dumps(rec)+'\n')
    print(json.dumps(rec), flush=True)
