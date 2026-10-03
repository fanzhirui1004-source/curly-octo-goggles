import json,glob
print('total sweeps T = k_train + k_post; val force (grf) mean energy excess %, train-geo force in []')
for f in sorted(glob.glob('res_n*.json')):
    o=json.load(open(f)); v=o['val']; t=o['train']; kt=o['k_train']
    out=[]
    for ty in ['full','medium','heavy','light']:
        s=[]
        for T in (0,4,8,12):
            kp=T-kt
            if kp in (0,1,4,8): s.append(f"T{T}:{100*v[f'{ty}|force|{kp}']:.2f}({100*v[f'{ty}|grf|{kp}']:.2f})")
        s.append(f"[tr {100*t[f'{ty}|force|0']:.2f}]")
        out.append(ty[:3]+' '+' '.join(s))
    print(f"{o['tag']:8s} N{o['Ntr']} st{o['steps']} :: "+' || '.join(out))
