. /root/autodl-tmp/cpu9754/jobenv.sh
{ date; hostname; uname -a; lscpu; nproc; cat /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory.high /sys/fs/cgroup/cpu.max /sys/fs/cgroup/cpu.stat;
  free -g; cat /proc/loadavg; $PY -c "import numpy, scipy, torch; print(numpy.__version__, scipy.__version__, torch.__version__, torch.get_num_threads())";
  ls /root/autodl-tmp/pylib_cpu /root/autodl-tmp/mklenv/lib/python3.12/site-packages 2>/dev/null | grep dist-info; } > $H/env_host.txt 2>&1
$PY -c "import json, platform, pardiso_direct as PD; r = PD.env_record(); r['machine'] = platform.node(); print(json.dumps(r, indent=1))" > $H/env_host.json
MKL_VERBOSE=1 $PY -c "
import numpy as np, scipy.sparse as sp, pardiso_direct as PD
A = sp.random(300, 300, density=0.05, random_state=0); A = (A @ A.T + 300 * sp.identity(300)).tocsr(); U = sp.triu(A).tocsr()
P = PD.Pardiso(U, 2, PD.tuned_iparm()); P.phase(11); P.phase(22); P.solve(np.ones(300)); P.release()
import ctypes; L = PD.lib(); a = np.ones((64, 64)); c = np.zeros((64, 64))
L.cblas_dgemm.argtypes = [ctypes.c_int] * 6 + [ctypes.c_double, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_double, ctypes.c_void_p, ctypes.c_int]
L.cblas_dgemm(101, 111, 111, 64, 64, 64, 1.0, a.ctypes.data, 64, a.ctypes.data, 64, 0.0, c.ctypes.data, 64)
" > $H/mkl_verbose.txt 2>&1
$PY pardiso_direct.py > $H/pardiso_selftest.json
$PY lat_cond_cpu.py --selftest > $HC/selftest.json
