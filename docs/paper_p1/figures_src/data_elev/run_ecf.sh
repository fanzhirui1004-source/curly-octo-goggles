#!/bin/bash
# ELEV 2026-10-05: exact cell fields on the host (environment of the exact checks in chain_final_route2.sh), 8 threads.
source /root/autodl-tmp/OPL/S1/env_gpu.sh
D=/root/autodl-tmp/CUTFEM_INGEST_R38/environment/runtime/r13_pardiso_v1/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$D PYTHONPATH=$PYTHONPATH:/root/autodl-tmp/pylib_cpu PYPARDISO_MKL_RT=$D/libmkl_rt.so.3
export MKL_NUM_THREADS=8 OMP_NUM_THREADS=8 OPL_GP_CACHE=0 OPL_CONV_FP32=1
export OPL_PACKETS_EXTRA=/root/autodl-tmp/OPL/S1/V2/R1/FINAL2/cplateN/packets:/root/autodl-tmp/OPL/S4/packets:/root/autodl-tmp/OPL/S3/packets:/root/autodl-tmp/OPL/S1/V2/R1/OPT/plate_packets
R=/root/autodl-tmp/OPL/ELEV_20261005/render
taskset -c 0-7 $PY -u $R/exact_cell_field.py > $R/ecf.log 2>&1; echo "RC=$?" >> $R/ecf.log
