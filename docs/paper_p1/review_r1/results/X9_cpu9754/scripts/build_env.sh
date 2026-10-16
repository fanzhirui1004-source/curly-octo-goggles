#!/bin/bash
# Python environment of the CPU campaign on the AMD EPYC 9754 machine: the versions of the earlier EPYC 9654 runs
# (Python 3.12.3, NumPy 2.3.2, SciPy 1.18.1, PyTorch 2.8.0+cu128; MKL 2026.1.0 runtime in mklenv, pypardiso in pylib_cpu).
set -x
D=/root/autodl-tmp
/root/miniconda3/bin/python -m venv $D/cpuenv
$D/cpuenv/bin/pip install -q --upgrade pip
$D/cpuenv/bin/pip install numpy==2.3.2 scipy==1.18.1 torch==2.8.0
$D/cpuenv/bin/pip install --prefix $D/mklenv --no-deps mkl==2026.1.0 intel-openmp==2026.1.2 tbb==2023.1.0 tcmlib==1.5.0 umf==1.1.0
$D/cpuenv/bin/python -c "import numpy, scipy, torch; print('VERSIONS', numpy.__version__, scipy.__version__, torch.__version__)"
ls -la $D/mklenv/lib | head -40
echo ENV_DONE
