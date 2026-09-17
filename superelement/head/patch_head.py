"""Add --head {chol,sqrt} to a copy of the M4 dev tree.

chol is the existing code path, byte-for-byte: every branch is guarded so that
head=='chol' reaches exactly the statements it reaches today.  sqrt swaps the
label file (M_UPPER.npy), assembles the prediction symmetrically instead of
triangularly, and uses a general (non-triangular) divergence.  The model, the
sampler, the bucket loss and the whitening reference R_* are untouched.
"""
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]) / 'stage_cutfem_m4'


def edit(path, old, new, count=1):
    text = path.read_text()
    n = text.count(old)
    if n != count:
        raise SystemExit(f'PATCH_ANCHOR {path.name}: expected {count} occurrences, found {n}\n---\n{old}\n---')
    path.write_text(text.replace(old, new))
    print(f'  ok {path.name}: {old.splitlines()[0][:70]}')


# ---------------------------------------------------------------- mechanics
mech = ROOT / 'mechanics.py'
mech.write_text(mech.read_text() + '''

# --- head-aware additions (symmetric square-root head) ---------------------
# Nothing above this line is modified.  For head=='chol' these helpers forward
# to the existing functions unchanged.


def validate_symmetric_root(M):
    if M.ndim != 2 or M.shape[0] != M.shape[1] or M.dtype != torch.float64:
        raise ValueError("Expected a square float64 symmetric root")
    if not bool(torch.isfinite(M).all()):
        raise ValueError("Nonfinite root; no repair")
    if bool((M.diagonal() <= 0).any()):
        raise ValueError("Expected a strictly positive diagonal")
    if float((M - M.T).abs().max()) != 0.0:
        raise ValueError("Expected an exactly symmetric root")


def validate_head(value, head):
    if head == "chol":
        validate_factor(value)
    elif head == "sqrt":
        validate_symmetric_root(value)
    else:
        raise ValueError("UNKNOWN_HEAD")


def symmetric_solve(M, force):
    return torch.linalg.solve(M, torch.linalg.solve(M, force))


def head_solve(value, force, head):
    return solve(value, force) if head == "chol" else symmetric_solve(value, force)


def general_divergence(C):
    """D, trace and log determinant gap of W = C^T C for a non-triangular C.

    The triangular closed form in divergence() is unavailable here, so the log
    determinant comes from slogdet.  Near the teacher identity this subtraction
    cancels; the spectral value computed from mu is the one that is quoted.
    """
    sign, gap = torch.linalg.slogdet(C)
    if float(sign) <= 0:
        raise ValueError("SINGULAR_OR_ORIENTATION_REVERSING_RELATIVE_FACTOR")
    gap = 2 * gap
    trace = C.square().sum()
    D = trace - gap - C.shape[0]
    return dict(D=D, D_per_mode=D / C.shape[0], trace=trace, logdet_gap=gap)


def head_divergence(value, reference, return_relative=False, head="chol"):
    if head == "chol":
        return divergence(value, reference, return_relative)
    C = relative_factor(value, reference)
    result = general_divergence(C)
    if return_relative:
        result["C"] = C
    return result
''')
print('  ok mechanics.py: appended head helpers')

# ---------------------------------------------------------------------- run
run = ROOT / 'run.py'

edit(run, """def prepare(row,args):
    reference=Path(row['reference']);factor=reference/'R_UPPER.npy';cache_path=Path(row['trace_cache'])
    receipt=json.loads((reference/'RESULT.json').read_text())
    bindings=dict(factor_sha256=sha256(factor),trace_sha256=sha256(cache_path),
                  teacher_sha256=sha256(Path(row['packet'])/'S_UPPER.npy'))
    if any(bindings[k]!=receipt[k] for k in bindings):raise ValueError('FROZEN_REFERENCE_BINDING')""",
"""def prepare(row,args):
    head=getattr(args,'head','chol')
    reference=Path(row['reference']);factor=reference/'R_UPPER.npy';cache_path=Path(row['trace_cache'])
    receipt=json.loads((reference/'RESULT.json').read_text())
    bindings=dict(factor_sha256=sha256(factor),trace_sha256=sha256(cache_path),
                  teacher_sha256=sha256(Path(row['packet'])/'S_UPPER.npy'))
    if any(bindings[k]!=receipt[k] for k in bindings):raise ValueError('FROZEN_REFERENCE_BINDING')
    label=factor
    if head=='sqrt':
        label=reference/'M_UPPER.npy';sqrt_receipt=json.loads((reference/'SQRT_RESULT.json').read_text())
        if (sqrt_receipt['r_sha256']!=bindings['factor_sha256'] or sqrt_receipt['m_sha256']!=sha256(label)
            or int(sqrt_receipt['d'])!=int(receipt['dimension'])
            or sqrt_receipt['read_blocks_roundtrip']!=0.0):raise ValueError('FROZEN_SQRT_LABEL_BINDING')
        bindings['sqrt_label_sha256']=sqrt_receipt['m_sha256']""")

edit(run, """    record.update(bindings=bindings,factor=str(factor),trace_cache=str(cache_path),
        source_receipt=str(reference/'RESULT.json'),split=row['split'],seat=row['seat'])
    return dict(seat=row['seat'],split=row['split'],d=record['d'],factor=factor,
        record=record,context=ctx,cache=cache,quotient=Q.to('cuda:0'))""",
"""    record.update(bindings=bindings,factor=str(label),whitener=str(factor),head=head,trace_cache=str(cache_path),
        source_receipt=str(reference/'RESULT.json'),split=row['split'],seat=row['seat'])
    return dict(seat=row['seat'],split=row['split'],d=record['d'],factor=label,whitener=factor,head=head,
        record=record,context=ctx,cache=cache,quotient=Q.to('cuda:0'))""")

edit(run, """def predict_full(model,sample,conditioning,chunk):
    tick=sync();model.eval();ctx=sample['context'];nodes=sample['d']//3""",
"""def predict_full(model,sample,conditioning,chunk,head='chol'):
    tick=sync();model.eval();ctx=sample['context'];nodes=sample['d']//3""")

edit(run, """        R[3*c[:,None,None]+b,3*r[:,None,None]+a]=pred.double()
    F.validate_factor(R)
    return R,dict(seconds=sync()-tick,blocks=total,scalar_factor_entries=sample['d']*(sample['d']+1)//2,
                  inference_chunk_pairs=chunk,complete_factor=True,all_entries_directly_predicted=True)""",
"""        if head=='chol':R[3*c[:,None,None]+b,3*r[:,None,None]+a]=pred.double()
        else:R[3*r[:,None,None]+a,3*c[:,None,None]+b]=pred.double()
    if head=='sqrt':
        # Every write above landed weakly below the diagonal: off-diagonal node
        # blocks have r>c, and the head zeroes the strict upper of a diagonal
        # block.  Mirroring is therefore exact, not a symmetrization repair.
        if int(torch.count_nonzero(torch.triu(R,1)))!=0:raise ValueError('SYMMETRIC_ASSEMBLY_WROTE_ABOVE_DIAGONAL')
        R=R+torch.tril(R,-1).T
    F.validate_head(R,head)
    return R,dict(seconds=sync()-tick,blocks=total,scalar_factor_entries=sample['d']*(sample['d']+1)//2,
                  inference_chunk_pairs=chunk,complete_factor=True,all_entries_directly_predicted=True,
                  head=head,assembly='upper triangular transpose' if head=='chol' else 'symmetric mirror of lower blocks')""")

edit(run, """    out.mkdir(exist_ok=False);tick=sync()
    R,inference=predict_full(model,sample,conditioning,args.inference_chunk)
    write(out/'INFERENCE.json',inference)
    packed=np.lib.format.open_memmap(out/'R_PRED_UPPER.npy',mode='w+',dtype=np.float64,shape=(len(R)*(len(R)+1)//2,))""",
"""    out.mkdir(exist_ok=False);tick=sync();head=sample.get('head','chol')
    R,inference=predict_full(model,sample,conditioning,args.inference_chunk,head)
    write(out/'INFERENCE.json',inference)
    pred_name='R_PRED_UPPER.npy' if head=='chol' else 'M_PRED_UPPER.npy'
    packed=np.lib.format.open_memmap(out/pred_name,mode='w+',dtype=np.float64,shape=(len(R)*(len(R)+1)//2,))""")

edit(run, """    reference=load_upper(sample['factor'],sample['d'],R.device)
    e_factor=float((R-reference).norm()/reference.norm())
    A=R.T@R;Astar=reference.T@reference;e_A=float((A-Astar).norm()/Astar.norm());del A,Astar
    div=F.divergence(R,reference,True);C=div.pop('C');W=C.T@C;del C""",
"""    label=load_upper(sample['factor'],sample['d'],R.device)
    e_factor=float((R-label).norm()/label.norm())
    if head=='chol':reference=label
    else:del label;gc.collect();torch.cuda.empty_cache();reference=load_upper(sample['whitener'],sample['d'],R.device)
    A=R.T@R;Astar=reference.T@reference;e_A=float((A-Astar).norm()/Astar.norm());del A,Astar
    div=F.head_divergence(R,reference,True,head);C=div.pop('C');W=C.T@C;del C""")

edit(run, """        numerical_gate_pass=bool(positive and symmetry<=1e-10 and float(res.max())<=1e-10 and loggap<=1e-10 and dgap<=1e-10),""",
"""        head=head,divergence_consistency_tolerance='absolute 1e-10' if head=='chol' else 'relative 1e-8; slogdet cancels against a large trace',
        numerical_gate_pass=bool(positive and symmetry<=1e-10 and float(res.max())<=1e-10
            and loggap<=(1e-10 if head=='chol' else 1e-8*max(1.,abs(float(eiglog))))
            and dgap<=(1e-10 if head=='chol' else 1e-8*max(1.,abs(float(ds))))),""")

edit(run, """        factor_file='R_PRED_UPPER.npy',factor_sha256=sha256(out/'R_PRED_UPPER.npy'))""",
"""        head=head,factor_file=pred_name,factor_sha256=sha256(out/pred_name))""")

edit(run, """    ap.add_argument('--seed',type=int,default=20260917);ap.add_argument('--source-sha',required=True)""",
"""    ap.add_argument('--head',choices=['chol','sqrt'],default='chol')
    ap.add_argument('--seed',type=int,default=20260917);ap.add_argument('--source-sha',required=True)""")

edit(run, """        model=dict(state=192,width=768,depth=4,full_scalar_Cholesky=True),""",
"""        model=dict(state=192,width=768,depth=4,full_scalar_Cholesky=args.head=='chol'),
        head=args.head,
        head_definition='chol: A_hat = R^T R with R upper triangular. sqrt: A_hat = M M with M symmetric positive definite.',
        head_is_the_only_variable='encoder, decoder, sampler, bucket loss, optimizer, schedule and seed are identical across heads',
        acceptance_unchanged='eps_op = max|mu-1| with mu = eig(R_*^-T A_hat R_*^-1); R_* is the Cholesky reference for both heads',""")

print('PATCH_COMPLETE')
