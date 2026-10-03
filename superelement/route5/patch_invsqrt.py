"""--head invsqrt: symmetric assembly (route 3) on the inverse target (route 5).

The head emits lower blocks of a symmetric M as for 'sqrt'; the label is MINV_UPPER.npy,
the packed upper triangle of A^{-1/2}; A_hat^-1 = M M.  Evaluation goes through
eval_inverse.py with --symmetric, since nu = eig((M R_*^T)^T (M R_*^T)) = 1/mu holds for
any symmetric M exactly as it does for the triangular L.
"""
import sys
from pathlib import Path
ROOT = Path(sys.argv[1]) / 'stage_cutfem_m4'


def edit(path, old, new):
    t = path.read_text(); n = t.count(old)
    if n != 1: raise SystemExit(f'PATCH_ANCHOR {path.name}: found {n}\n{old}')
    path.write_text(t.replace(old, new)); print('  ok', path.name, old.splitlines()[0][:60])


run = ROOT / 'run.py'
edit(run, "ap.add_argument('--head',choices=['chol','sqrt','inverse'],default='chol')",
          "ap.add_argument('--head',choices=['chol','sqrt','inverse','invsqrt'],default='chol')")
edit(run, """    if head=='sqrt':
        label=reference/'M_UPPER.npy';sqrt_receipt=json.loads((reference/'SQRT_RESULT.json').read_text())""",
"""    if head=='invsqrt':
        label=reference/'MINV_UPPER.npy';inv=json.loads((reference/'INVSQRT_RESULT.json').read_text())
        if (inv['r_sha256']!=bindings['factor_sha256'] or inv['m_sha256']!=sha256(label)
            or int(inv['d'])!=int(receipt['dimension']) or inv['read_blocks_roundtrip']!=0.0):raise ValueError('FROZEN_INVSQRT_LABEL_BINDING')
        bindings['invsqrt_label_sha256']=inv['m_sha256']
    if head=='sqrt':
        label=reference/'M_UPPER.npy';sqrt_receipt=json.loads((reference/'SQRT_RESULT.json').read_text())""")
edit(run, "        if head=='chol':R[3*c[:,None,None]+b,3*r[:,None,None]+a]=pred.double()\n        else:R[3*r[:,None,None]+a,3*c[:,None,None]+b]=pred.double()\n    if head=='sqrt':",
          "        if head=='chol':R[3*c[:,None,None]+b,3*r[:,None,None]+a]=pred.double()\n        else:R[3*r[:,None,None]+a,3*c[:,None,None]+b]=pred.double()\n    if head in ('sqrt','invsqrt'):")
edit(run, "        window=(-20.,20.) if args.head=='inverse' else (-20.,5.)",
          "        window=(-20.,20.) if args.head in ('inverse','invsqrt') else (-20.,5.)")
edit(run, "        if args.head=='inverse':\n            # the pivots",
          "        if args.head in ('inverse','invsqrt'):\n            # the pivots")
mech = ROOT / 'mechanics.py'
edit(mech, '    elif head == "sqrt":\n        validate_symmetric_root(value)',
           '    elif head in ("sqrt", "invsqrt"):\n        validate_symmetric_root(value)')
print('INVSQRT_PATCH_COMPLETE')
