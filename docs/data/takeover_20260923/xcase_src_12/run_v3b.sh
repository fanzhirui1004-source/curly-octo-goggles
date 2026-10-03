T=/root/autodl-tmp/CLAUDE_TAKEOVER_20260923; S=$T/xcase_src_12; O=$T/ELEMENT_TOL_02
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4; PY=/root/cutfem_neural_a_20260910/env/bin/python; cd $S
until grep -q done $O/status 2>/dev/null; do sleep 20; done
for sub in plane tpms full; do $PY -u element_tolerance_v3.py --case fresh_train_0003_d0_v0 --subset $sub --families moment spectral --eps 1e-4 0.03 --work $T/ELEMENT_WORK --output $O/train3_sub_$sub > $O/train3_sub_$sub.log 2>&1; done
echo subdone >> $O/status
