"""Exact two-pass gradient for a complete symmetric factor, with bounded decoder memory.

One evaluation of the energy-response loss needs the WHOLE q x q factor, not a subset, because the
loss applies the operator twice against dense probe loads.  Holding the decoder graph for all
q(q+1)/2 blocks is what made the term unaffordable.  This is the exact same gradient computed in two
passes: the forward builds the matrix chunk by chunk under `no_grad`, the loss is differentiated with
respect to the matrix alone, and the matrix adjoint is then pushed back through the decoder one chunk
at a time, each chunk's graph discarded before the next.  The encoder is differentiated exactly once,
through detached features that collect the adjoints, and the caller must hand in the SAME
graph-bearing encoding both passes used, because a second encoder forward can round its support
index_add differently.

So it is exact, not approximate.  What it is not is cheap: a full gradient of this term was measured
at 29.0 s for seat 100032 and 37.9 s for 100046 (forward 5.3 / 6.9 s, parameter VJP 23.7 / 30.9 s,
peak 4.6 / 5.9 GiB), which rules it out of an inner training loop at any realistic step count and
makes it a diagnostic or a rarely-applied term.

The chunk size is not free either: the block algebra is exact at any chunk, but the decoder's kernels
tile by batch size and reduce in a different order, so chunk 8192 reproduces the reference element for
element while chunk 32768 is faster and differs by 4.6e-7 relative on the factor and 3.7e-5 per
eigenvalue.  Use `train_equi.VERIFIED_DECODE_CHUNKS`.

Provenance: the algebra is Codex's, from `CLAUDE_HANDOFF_CUT_20260920`; it is landed here unchanged
so that the measured costs above still describe the code that runs.
"""
from __future__ import annotations
import numpy as np
import torch


def pair_chunks(count, chunk, device):
    total=count*(count+1)//2
    for lo in range(0,total,chunk):
        ids=np.arange(lo,min(lo+chunk,total),dtype=np.int64)
        r=np.floor((np.sqrt(8*ids+1)-1)/2).astype(np.int64)
        c=ids-r*(r+1)//2
        yield torch.as_tensor(r,device=device),torch.as_tensor(c,device=device)


def indices(r,c):
    v=torch.arange(3,device=r.device)
    return 3*r[:,None,None]+v[None,:,None],3*c[:,None,None]+v[None,None,:]


def symmetric_blocks(raw,r,c):
    diag=r==c
    out=raw.double().clone()
    out[diag]=out[diag]+torch.tril(out[diag],-1).transpose(1,2)
    return out


def block_pullback(matrix_gradient,r,c):
    i,j=indices(r,c)
    g=matrix_gradient[i,j]+matrix_gradient[j,i]
    diag=r==c
    gd=g[diag]
    gd=torch.tril(gd)
    k=torch.arange(3,device=r.device)
    gd[:,k,k]*=.5
    g[diag]=gd
    return g


@torch.no_grad()
def complete_factor(model,ctx,conditioning,chunk,*,encoded=None):
    enc=model.encode(ctx) if encoded is None else encoded
    count=int(ctx['count'])
    m=torch.empty((3*count,3*count),device=ctx['pos'].device,dtype=torch.float64)
    for r,c in pair_chunks(count,chunk,m.device):
        blocks=symmetric_blocks(model.decode(enc,ctx,r,c,conditioning),r,c)
        i,j=indices(r,c)
        m[i,j]=blocks
        m[j,i]=blocks
    return m


def complete_factor_vjp(model,ctx,conditioning,matrix_gradient,chunk,*,encoded):
    """Accumulate exact VJP into parameter .grad; caller owns zero_grad/optimizer.

    Decoder graphs are discarded per chunk. Detached encoded features collect
    adjoints; the encoder is differentiated exactly once afterwards. The caller
    must pass the SAME graph-bearing encoding used in complete_factor, because
    GPU support index_add can round differently on separate encoder forwards.
    """
    live=encoded
    detached={k:v.detach().requires_grad_(v.requires_grad) for k,v in live.items()}
    for r,c in pair_chunks(int(ctx['count']),chunk,ctx['pos'].device):
        prediction=model.decode(detached,ctx,r,c,conditioning)
        gradient=block_pullback(matrix_gradient,r,c).to(prediction.dtype)
        torch.autograd.backward(prediction,gradient)
    outputs=[];gradients=[]
    for k,v in live.items():
        if v.requires_grad and detached[k].grad is not None:
            outputs.append(v);gradients.append(detached[k].grad)
    torch.autograd.backward(outputs,gradients)


def energy_response_loss(m,quotient,rstar,full_force,teacher_x):
    """Squared teacher-energy norm of deployed displacement error, same quotient."""
    middle=quotient.lift(quotient(m@full_force))
    predicted_x=quotient(m@middle)
    residual=rstar@(predicted_x-teacher_x)
    teacher_energy=(rstar@teacher_x).square().sum(0)
    if bool((teacher_energy<=0).any()):
        raise ValueError("NONPOSITIVE_TEACHER_PROBE_ENERGY")
    errors=residual.square().sum(0)/teacher_energy
    return errors.mean(),errors
