T=/root/autodl-tmp/CLAUDE_TAKEOVER_20260923; S=$T/xcase_src_08
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6
for c in fresh_development_0003_d0_v0 fresh_development_0006_d0_v0 fresh_development_0000_d0_v0; do
  /root/miniconda3/bin/python3.12 $S/element_tolerance.py --case $c --work $T/ELEMENT_WORK --output $T/ELEMENT_TOL_01/$c > $T/ELEMENT_TOL_01/$c.log 2>&1 < /dev/null
  echo "$c exit $?" >> $T/ELEMENT_TOL_01/j2b.status
done
