#!/usr/bin/env python3
"""Per-direction signal-to-storage-noise for a delivered no-GP label.

E_i = ||R z_i||^2                                   stored energy of direction i
N_i = sum_r (0.5*eps*sum_j |R_rj z_j|)^2            what FP64 storage of R cannot resolve
S/N = E_i/N_i                                       times the value exceeds its own floor

S/N <= 1 means the stored energy is indistinguishable from the rounding of the
factor that produced it.  The achievable relative energy error is about 1/(S/N).

Directions are the recorded production screen directions when present, otherwise
regenerated with the registered convention (seed 2026091407, 12 power iterations,
two soft starts then two random) so the two cases are comparable.

Reads only.  R is streamed row by row and never materialised as a dense matrix.
"""
import argparse,json
from pathlib import Path
import numpy as np

EPS=np.finfo(np.float64).eps
NAMES=['SOFT_POWER_0','SOFT_POWER_1','RANDOM_0','RANDOM_1']


class Packed:
    """Row-major upper R is column-major packed lower R.T, with no conversion."""
    def __init__(self,path,d=None):
        self.packed=np.load(path,mmap_mode='r',allow_pickle=False)
        if self.packed.dtype!=np.float64 or self.packed.ndim!=1:
            raise ValueError('PACKED_COMPLETE_FP64_ROOT')
        n=self.packed.shape[0]
        self.d=d if d else int((np.sqrt(8*n+1)-1)//2)
        if self.packed.shape!=(self.d*(self.d+1)//2,):raise ValueError('PACKED_TRIANGLE_SHAPE')
        idx=np.arange(self.d,dtype=np.int64)
        self.diagonal=np.asarray(self.packed[idx*(2*self.d-idx+1)//2])
        if not np.isfinite(self.diagonal).all() or np.any(self.diagonal<=0):
            raise ValueError('POSITIVE_DIAGONAL_REQUIRED')
    def _row(self,i,off):
        return np.asarray(self.packed[off:off+self.d-i],dtype=np.float64)
    def action(self,x,*,transpose=False,inverse=False):
        """R x, R^T x, R^-1 x or R^-T x, streaming the packed rows once."""
        d=self.d;b=np.array(x,dtype=np.float64,copy=True)
        if b.shape!=(d,):raise ValueError('PACKED_VECTOR_DIMENSION')
        if not inverse and not transpose:            # y_i = sum_{j>=i} R_ij x_j
            y=np.empty(d);off=0
            for i in range(d):
                y[i]=self._row(i,off)@b[i:];off+=d-i
        elif not inverse:                            # y = R^T x, accumulate by rows
            y=np.zeros(d);off=0
            for i in range(d):
                y[i:]+=self._row(i,off)*b[i];off+=d-i
        elif transpose:                              # solve R^T y = b, forward
            y=np.empty(d);off=0
            for i in range(d):
                row=self._row(i,off);y[i]=b[i]/row[0]
                b[i+1:]-=row[1:]*y[i];off+=d-i
        else:                                        # solve R y = b, backward
            y=np.empty(d);off=len(self.packed)
            for i in range(d-1,-1,-1):
                off-=d-i;row=self._row(i,off)
                y[i]=(b[i]-row[1:]@y[i+1:])/row[0]
        if not np.isfinite(y).all():raise ValueError('NONFINITE_ROOT_ACTION_NO_REPAIR')
        return y
    def stream(self,Z):
        """One pass over the rows: returns ||R z||^2 and the storage floor per column."""
        d=self.d;A=np.abs(Z);y=np.empty((d,Z.shape[1]));a=np.empty((d,Z.shape[1]));off=0
        for i in range(d):
            row=np.asarray(self.packed[off:off+d-i],dtype=np.float64)
            y[i]=row@Z[i:];a[i]=np.abs(row)@A[i:];off+=d-i
        return (y*y).sum(0),((0.5*EPS*a)**2).sum(0)


def registered_directions(root,iterations=12):
    """The production screen convention, reproduced exactly."""
    rng=np.random.default_rng(2026091407);directions=[];inverse=[]
    for _ in range(2):
        v=rng.normal(size=root.d);v/=np.linalg.norm(v)
        for _ in range(iterations):
            w=root.action(v,transpose=True,inverse=True);w/=np.linalg.norm(w)
            v=root.action(w,inverse=True);v/=np.linalg.norm(v)
        directions.append(v);inverse.append(float(1./np.linalg.norm(root.action(v))))
    v=rng.normal(size=root.d);v/=np.linalg.norm(v)
    for _ in range(iterations):
        w=root.action(v);w/=np.linalg.norm(w);v=root.action(w,transpose=True);v/=np.linalg.norm(v)
    largest=float(np.linalg.norm(root.action(v)))
    for _ in range(2):
        v=rng.normal(size=root.d);directions.append(v/np.linalg.norm(v))
    return np.column_stack(directions),dict(inverse_norm_estimates=inverse,
        largest_singular_value_estimate=largest,
        condition_number_lower_estimate=largest*max(inverse),
        condition_estimate_not_certified_bound=True)


def analyse(root_path,directions=None,d=None,label=None):
    root=Packed(root_path,d);extra={}
    if directions and Path(directions).exists():
        Z=np.load(directions,allow_pickle=False);origin='recorded_production_screen'
        if Z.ndim!=2 or Z.shape[0]!=root.d:raise ValueError('RECORDED_DIRECTION_SHAPE')
    else:
        Z,extra=registered_directions(root);origin='regenerated_registered_convention'
    norm=np.linalg.norm(Z,axis=0)
    if not np.all(np.isfinite(norm)) or np.any(norm<=0):raise ValueError('DEGENERATE_DIRECTION')
    E,N=root.stream(np.ascontiguousarray(Z/norm,dtype=np.float64))
    sn=np.divide(E,N,out=np.full(len(E),np.inf),where=N>0)
    names=NAMES[:Z.shape[1]] if Z.shape[1]<=len(NAMES) else ['DIR_%d'%i for i in range(Z.shape[1])]
    return dict(schema='NO_GP_DIRECTION_STORAGE_FLOOR_V1',label=label,d=int(root.d),
        direction_origin=origin,directions=names,
        stored_energy=[float(v) for v in E],storage_noise_floor=[float(v) for v in N],
        signal_to_storage_noise=[float(v) for v in sn],
        expected_relative_energy_error=[float(1./v) if v>0 else float('inf') for v in sn],
        min_signal_to_storage_noise=float(np.min(sn)),
        diagonal_ratio=float(root.diagonal.max()/root.diagonal.min()),
        definition='N_i = sum_r (0.5*eps*sum_j |R_rj z_j|)^2 ; E_i = ||R z_i||^2 on the unit direction',
        eps=float(EPS),acceptance_threshold_not_applied_here=True,**extra)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',required=True);p.add_argument('--directions')
    p.add_argument('--d',type=int);p.add_argument('--out');p.add_argument('--label')
    a=p.parse_args()
    r=analyse(a.root,a.directions,a.d,a.label)
    if a.out:Path(a.out).write_text(json.dumps(r,indent=1))
    print(json.dumps({k:r[k] for k in ('label','d','direction_origin','signal_to_storage_noise')}))
