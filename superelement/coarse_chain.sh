#!/bin/sh
# Waits for the old free-coefficient tests to finish, then runs the coarse-compliance back-end tests and their exact spectra (plain and E-shifted).
O=/root/autodl-tmp/CUTFEM_SPECTRUM_20260912; P=/root/cutfem_neural_a_20260910/env/bin/python; V=/root/autodl-tmp/CUTFEM_SUPERELEMENT_20260912_V1G
cd /root/cutfem_neural_a_20260910/superelement_v0
while [ ! -f $O/FREE_TESTS_DONE ]; do sleep 20; done
for CFG in "328 0.2" "415 0.2" "328 0.1"; do
  set -- $CFG; SEAT=$1; R=$2; TAG=$(echo $R | tr -d .)
  OUT=/root/autodl-tmp/CUTFEM_SUPERELEMENT_20260912_COARSE_${SEAT}_r${TAG}
  ./run_free_coarse.sh $SEAT $OUT 3000 $V/CHECKPOINT_003000.pt --r-near $R
  $P spectrum_eval.py --run $OUT --checkpoint $OUT/CHECKPOINT_003000.pt --seat $SEAT --out $O --threads 12 --shift-c 10 > $O/coarse_${SEAT}_r${TAG}_spectrum.log 2>&1
done
echo done > $O/COARSE_TESTS_DONE
