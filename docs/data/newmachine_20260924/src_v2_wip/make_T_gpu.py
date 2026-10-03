"""Dense exact port operator T = S of one cell on the GPU (lattice3.dense_T: cuDSS Schur mode, fp64), cached as
<body>/<case>_portview/T64.npy; a cell per process so that the device is otherwise empty. make_T_cpu.py is the host fallback.
Usage: make_T_gpu.py <body> <case>"""
import sys, time, json
import models as MD                                                    # noqa: F401  first: applies OPL_CONV_FP32
import teacher as TE
import lattice3 as LT

body, case = sys.argv[1], sys.argv[2]
t = time.perf_counter()
C = TE.Cell(case, body, log=lambda s_: None)
T = LT.dense_T(C, body, host=True)
print(json.dumps(dict(case=case, ports=int(T.shape[0]), seconds=time.perf_counter() - t, route='gpu')), flush=True)
