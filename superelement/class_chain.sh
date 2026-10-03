#!/bin/sh
# Back-end class tests on 0328 after the old free-coefficient tests: unbounded free M (rank 1024) at band 0.2 rank 1024, band 0.2 rank 2048, band 0.1 rank 1024 (band width was shown not to matter by the truncation check).
O=/root/autodl-tmp/CUTFEM_SPECTRUM_20260912; P=/root/cutfem_neural_a_20260910/env/bin/python; V=/root/autodl-tmp/CUTFEM_SUPERELEMENT_20260912_V1G
cd /root/cutfem_neural_a_20260910/superelement_v0
while [ ! -f $O/FREE_TESTS_DONE ]; do sleep 20; done
for CFG in "328 0.2 1024" "328 0.2 2048" "328 0.1 1024"; do
  set -- $CFG; SEAT=$1; R=$2; K=$3; TAG=r$(echo $R | tr -d .)_k$K
  OUT=/root/autodl-tmp/CUTFEM_SUPERELEMENT_20260912_CLASS_${SEAT}_${TAG}
  ./run_free.sh $SEAT $OUT 6000 $V/CHECKPOINT_003000.pt --model free_unbounded --rank $K --r-near $R
  $P spectrum_eval.py --run $OUT --checkpoint $OUT/CHECKPOINT_006000.pt --seat $SEAT --out $O --threads 12 --shift-c 10 > $O/class_${SEAT}_${TAG}_spectrum.log 2>&1
done
echo done > $O/CLASS_TESTS_DONE
