"""Check of trainlib.coarse_columns (src_v3 S22): the kept coarse columns (vertex coordinates, slots) equal those of
coarse_prolong on deployment cells; times of both.
usage: python t_coarsecols.py BODY_DIR:CASE1,CASE2,..."""
import os, sys, json, time
SRC = os.environ.get('SRC', '/root/autodl-tmp/OPL/src_v3'); sys.path.insert(0, SRC)
os.environ['OPL_TAILT_FUSED'] = '1'; os.environ['OPL_COARSE_FP32'] = '1'; os.environ['OPL_COARSE_ELEM'] = '1'
import models as MD  # noqa
import numpy as np
import teacher as TE, trainlib as TL

body, cases = sys.argv[1].split(':')
for case in cases.split(','):
    C = TE.Cell(case, body, log=lambda s_: None, deploy=True); C.assemble_deploy()
    space = 'Q1_17'
    t = time.perf_counter(); V, vc0, sl0 = TL.coarse_prolong(C, space, with_meta=True); t0 = time.perf_counter() - t
    t = time.perf_counter(); vc1, sl1, nc1 = TL.coarse_columns(C, space); t1 = time.perf_counter() - t
    print(json.dumps(dict(case=case, space=space, nc=int(V.shape[1]), nc_cols=int(nc1), equal=bool(np.array_equal(vc0, vc1) and np.array_equal(sl0, sl1)),
                          prolong_s=round(t0, 4), columns_s=round(t1, 4))), flush=True)
    del C
