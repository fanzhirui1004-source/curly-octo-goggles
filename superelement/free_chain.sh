#!/bin/sh
# Capacity tests after the V1G pause: free coefficients on 0328 and 0415 from V1G/3000, then exact spectra.
V=/root/autodl-tmp/CUTFEM_SUPERELEMENT_20260912_V1G; O=/root/autodl-tmp/CUTFEM_SPECTRUM_20260912; P=/root/cutfem_neural_a_20260910/env/bin/python
cd /root/cutfem_neural_a_20260910/superelement_v0
for SEAT in 328 415; do
  OUT=/root/autodl-tmp/CUTFEM_SUPERELEMENT_20260912_FREE_$SEAT
  ./run_free.sh $SEAT $OUT 3000 $V/CHECKPOINT_003000.pt
  $P spectrum_eval.py --run $OUT --checkpoint $OUT/CHECKPOINT_003000.pt --seat $SEAT --out $O --threads 12 > $O/free_${SEAT}_spectrum.log 2>&1
done
echo done > $O/FREE_TESTS_DONE
