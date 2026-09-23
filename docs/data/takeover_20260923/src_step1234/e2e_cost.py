"""Collect one end-to-end preparation measurement into COST.json. Usage: e2e_cost.py <case dir>"""
import json, sys
from pathlib import Path
o = Path(sys.argv[1])
st = {k: json.loads((o / f'STAGE_{k}.json').read_text()) for k in ('G', 'INPUTS', 'ENCODE')}
inp = json.loads((o / 'inputs' / 'RESULT.json').read_text())
enc = json.loads((o / 'encode' / 'ENCODE.json').read_text())
rec = dict(case=o.name, exits={k: v['exit'] for k, v in st.items()},
           wall_seconds={k: v['wall_seconds'] for k, v in st.items()},
           total_wall_seconds=sum(v['wall_seconds'] for v in st.values()),
           peak_rss_gib={k: v['peak_rss_gib'] for k, v in st.items()}, peak_gpu_mib=st['ENCODE']['peak_gpu_mib'],
           inputs_breakdown={k: inp[k] for k in ('load_seconds', 'trace_seconds', 'face_select_seconds', 'ghost_scatter_seconds', 'write_seconds')},
           encode_in_process_seconds=enc['encode_seconds'], encode_stages=enc['stage_seconds'],
           query_seconds=enc['query_seconds'], boundary=enc['boundary'], interior=enc['interior'])
(o / 'COST.json').write_text(json.dumps(rec, indent=2))
print(json.dumps(rec))
