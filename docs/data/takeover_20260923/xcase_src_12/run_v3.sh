T=/root/autodl-tmp/CLAUDE_TAKEOVER_20260923; S=$T/xcase_src_12; O=$T/ELEMENT_TOL_02; mkdir -p $O
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4; PY=/root/cutfem_neural_a_20260910/env/bin/python
cd $S; $PY -u element_tolerance_v3.py --case fresh_train_0003_d0_v0 --families moment moment_vol --eps 1e-4 1e-3 1e-2 --work $T/ELEMENT_WORK --output $O/train3_moment > $O/train3_moment.log 2>&1
$PY -u element_tolerance_v3.py --case fresh_train_0003_d0_v0 --families nnls8 nnls12 --eps 0 0.03 --work $T/ELEMENT_WORK --output $O/train3_nnls > $O/train3_nnls.log 2>&1
echo done >> $O/status
