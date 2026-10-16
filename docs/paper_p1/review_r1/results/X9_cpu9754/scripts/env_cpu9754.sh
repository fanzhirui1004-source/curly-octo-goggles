# CPU environment of the EPYC 9754 campaign (same versions as the EPYC 9654 runs: Python 3.12.3, NumPy 2.3.2, SciPy 1.18.1,
# PyTorch 2.8.0+cu128, MKL 2026.1.0 runtime in mklenv, pypardiso 0.4.7 in pylib_cpu). PY: cpuenv on the 9754 machine,
# the image's Python (same versions) on the GPU machine where the scripts are tested.
M=/root/autodl-tmp/mklenv/lib
export CUDA_VISIBLE_DEVICES= LD_LIBRARY_PATH=$M PYPARDISO_MKL_RT=$M/libmkl_rt.so.3 OPL_DEV=cpu OPL_GP_CACHE=0
export PYTHONPATH=/root/autodl-tmp/cpu9754:/root/autodl-tmp/OPL/src_v2:/root/autodl-tmp/pylib_cpu OPL_PACKETS_EXTRA=/root/autodl-tmp/OPL/S4/packets:/root/autodl-tmp/OPL/S3/packets
if [ -x /root/autodl-tmp/cpuenv/bin/python ]; then PY=/root/autodl-tmp/cpuenv/bin/python; else PY=/root/miniconda3/bin/python; fi
