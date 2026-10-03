#!/usr/bin/env python3
"""Set the project's own runtime environment, then exec the build.

stage_cutfem_runtime.environment.runtime_environment() is the project's convention for
PYTHONPATH, LD_LIBRARY_PATH and the MKL handle.  The algoim quadrature backend dlopens
libmkl_rt.so.3 inside worker processes, so LD_LIBRARY_PATH has to be in place before exec,
not set from inside Python.  This computes it with the project's own function and re-execs.
"""
import os, sys, json
from pathlib import Path

P = '/root/autodl-tmp/CUTFEM_INGEST_R38/source_independent_6624dc8_20260910'
sys.path.insert(0, P)
from stage_cutfem_runtime.config import CONFIG
from stage_cutfem_runtime.environment import runtime_environment

names = CONFIG['runtime_names']
extra, bindings = runtime_environment(P, names)
print('runtime names:', names, flush=True)
for k in ('LD_LIBRARY_PATH', 'PYPARDISO_MKL_RT', 'CUTFEM_RUNTIME_ROOT'):
    print('  %s = %s' % (k, extra.get(k, '<unset>')), flush=True)
env = dict(os.environ)
env.update({k: v for k, v in extra.items() if isinstance(v, str)})
target = sys.argv[1]
os.execve(sys.executable, [sys.executable, target], env)
