# CPU host (westb, AMD EPYC 9654): installed as /root/autodl-tmp/OPL/S1/V2/R1/env_cpu.sh on that machine
M=/root/autodl-tmp/mklenv/lib
export CUDA_VISIBLE_DEVICES= LD_LIBRARY_PATH=$M PYPARDISO_MKL_RT=$M/libmkl_rt.so.3 PYTHONPATH=/root/autodl-tmp/pylib_cpu OPL_DEV=cpu OPL_PACKETS_EXTRA=/root/autodl-tmp/OPL/S4/packets:/root/autodl-tmp/OPL/S3/packets
PY=/root/miniconda3/bin/python
