import json,glob
for f in sorted(glob.glob('res_n*.json')):
    o=json.load(open(f)); v=o['val']; t=o['train']
    row=[]
    for ty in ['full','light','medium','heavy']:
        row.append(f"{ty[:3]} F {100*v[f'{ty}|force|0']:.2f}/{100*t.get(f'{ty}|force|0',float('nan')):.2f} k8 {100*v[f'{ty}|force|8']:.3f} G {100*v[f'{ty}|grf|0']:.2f}/{100*t.get(f'{ty}|grf|0',float('nan')):.2f}")
    sh=' '.join(f"{ty[:1]}:{v.get(f'{ty}|force|share_high',float('nan')):.2f}/{v.get(f'{ty}|force|share_low',float('nan')):.2f}" for ty in ['full','medium','heavy'])
    print(f"{o['tag']:9s} N{o['Ntr']:4d} st{o['steps']:6d} k{o['k_train']} {o['secs']:.0f}s | "+' | '.join(row)+' | hi/lo '+sh)
