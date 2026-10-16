import json, glob, os
T = '/root/autodl-tmp/CLAUDE_TAKEOVER_20260923'
st = {}
for f in sorted(glob.glob(T + '/COVER_G/runs/*_G_VERIFY.json')):
    st[os.path.basename(f)[:-len('_G_VERIFY.json')]] = json.load(open(f)).get('status')
print('COVER_G verify', len(st), {k: v for k, v in st.items() if v != 'MATCH'})
tg = [c for c in st if os.path.exists('/root/autodl-tmp/CUTFEM_FRESH_GP_20260921/targets/%s_v1/REFERENCE_RQ.npy' % c)]
print('with target', len(tg), 'without', sorted(set(st) - set(tg)))
ci = [d for d in os.listdir(T + '/COVER_INPUTS') if os.path.exists(T + '/COVER_INPUTS/%s/RESULT.json' % d)]
print('COVER_INPUTS', len(ci))
print(sorted(os.listdir(T)))
