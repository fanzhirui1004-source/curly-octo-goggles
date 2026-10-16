import json, glob, os
from fractions import Fraction
R='/root/autodl-tmp/CUTFEM_FRESH_GP_20260921/packets'
fam={}
for d in sorted(glob.glob(R+'/*')):
    c=os.path.basename(d)
    try: ctx=json.load(open(d+'/FRESH_CONTEXT.json'))
    except Exception: continue
    cs=ctx['case']
    if 'coverage_round' not in cs and cs.get('kind')!='FULL': continue
    f=cs['family_id']; fam.setdefault(f,{})
    t=[float(Fraction(v)) for v in cs['tau_corners']]
    key=('FULL' if cs.get('kind')=='FULL' else 'r%d'%cs['coverage_round'])
    fam[f][key]=(c, cs.get('retained_macro_volume_target'), cs.get('theta_degrees'), cs.get('field_kind'), round(min(t),3), round(max(t),3), cs.get('split'))
for f,v in fam.items():
    if 'r1' in v or 'r2' in v:
        print(f, v.get('FULL',(None,))[0] is not None, v.get('r1',('-',None,None))[1:3], v.get('r2',('-',None,None))[1:3], (v.get('r1') or v.get('r2'))[3:])
print('FULL kinds', sum(1 for f,v in fam.items() if 'FULL' in v))
