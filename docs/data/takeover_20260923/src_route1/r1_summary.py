"""Print route 1 depth results: witness and the seven original probe energy ratios per depth."""
import json, sys, glob, os
for d in sys.argv[1:]:
    for f in sorted(glob.glob(os.path.join(d, '*', 'RESULT.json'))):
        r = json.load(open(f))
        name = os.path.basename(os.path.dirname(f))
        extra = dict(dims=r.get('dimensions'), ritz=r.get('ritz_min'), split=r.get('block_split'))
        s = r.get('slow_mode_oracle')
        if s: extra['slow'] = dict(m=s['m'], k=s['k'], theta_first=[round(x, 5) for x in s['theta_first'][:3]], theta_last=s['theta_last'], below=s['below_cut'])
        print(f'## {os.path.basename(d)}/{name} d1={r.get("degree1")} d2={r.get("degree2")} enc={r["encode_seconds"]:.1f}s', json.dumps(extra))
        for row in r['depths']:
            o7 = row['original7_energy_ratio']
            w = row['witness'][0] if isinstance(row['witness'], list) else row['witness']
            print(f'   depth {row["depth"]:3d} witness {w:10.4f}  orig7 min {min(o7):.5f} max {max(o7):.5f}  q1 {row["query_seconds"]["1"]*1e3:.1f}ms q7 {row["query_seconds"]["7"]*1e3:.1f}ms')
