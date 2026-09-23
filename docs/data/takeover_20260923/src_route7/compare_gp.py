"""Route 7 check: parallel face selection against the packets' published GP_FACES, bit for bit; frozen timing on
the first case for reference. Usage: compare_gp.py <case> [<case> ...]"""
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, '/root/autodl-tmp/CLAUDE_TAKEOVER_20260923/xcase_src_11')
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gp_check_body import load_body
from stage_cutfem_gp import assembly
import fast_gp

T = Path('/root/autodl-tmp/CLAUDE_TAKEOVER_20260923'); R = Path('/root/autodl-tmp/CUTFEM_FRESH_GP_20260921/packets')
out = T / 'R7_03'; out.mkdir(exist_ok=True)
for k, case in enumerate(sys.argv[1:]):
    body = load_body(T / 'COVER_G' / 'runs' / (case + '_G'))
    t = time.perf_counter(); faces, record = fast_gp.select_faces(body); tf = time.perf_counter() - t
    t = time.perf_counter(); G = fast_gp.ghost_matrix(body, faces); tg = time.perf_counter() - t
    ref = np.load(R / case / 'CONTEXT' / 'GP_FACES.npy')
    row = dict(case=case, faces=len(faces), same_faces=bool(np.array_equal(np.asarray(faces), ref)), select_seconds=tf, matrix_seconds=tg)
    if k == 0:
        t = time.perf_counter(); faces0, _ = assembly.select_faces(body); row['frozen_select_seconds'] = time.perf_counter() - t
        row['same_as_frozen_call'] = bool(np.array_equal(np.asarray(faces0), np.asarray(faces)))
    (out / (case + '.json')).write_text(json.dumps(row))
    print(json.dumps(row), flush=True)
