D=/root/autodl-tmp/CUTFEM_DEPENDENCIES_20260924/root/autodl-tmp/CUTFEM_INGEST_R38/environment/runtime/r13_pardiso_v1/lib
export CUDA_VISIBLE_DEVICES= LD_LIBRARY_PATH=$D PYPARDISO_MKL_RT=$D/libmkl_rt.so.3 PYTHONPATH=/root/autodl-tmp/pylib_cpu OPL_DEV=cpu OPL_PACKETS_EXTRA=/root/autodl-tmp/OPL/S4/packets:/root/autodl-tmp/OPL/S3/packets
PY=/root/autodl-tmp/gpuenv/bin/python
