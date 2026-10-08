#!/bin/sh
# Back-end class tests on 0328 on the two-GPU machine: GPU0 runs band 0.2 rank 1024 then band 0.1 rank 1024; GPU1 runs band 0.2 rank 2048.
O=/root/autodl-tmp/CUTFEM_SPECTRUM_20260912; P=/root/cutfem_neural_a_20260910/env/bin/python; V=/root/autodl-tmp/CUTFEM_SUPERELEMENT_20260912_V1G
cd /root/cutfem_neural_a_20260910/superelement_v0
run() {  # gpu seat radius rank
  OUT=/root/autodl-tmp/CUTFEM_SUPERELEMENT_20260912_CLASS_$2_r$(echo $3 | tr -d .)_k$4
  rm -rf $OUT
  CUDA_VISIBLE_DEVICES=$1 ./run_free.sh $2 $OUT 6000 $V/CHECKPOINT_003000.pt --model free_unbounded --rank $4 --r-near $3
  $P spectrum_eval.py --run $OUT --checkpoint $OUT/CHECKPOINT_006000.pt --seat $2 --out $O --threads 12 --shift-c 10 > $O/class_$2_r$(echo $3 | tr -d .)_k$4_spectrum.log 2>&1
}
(run 0 328 0.2 1024; run 0 328 0.1 1024; echo done > $O/CLASS_GPU0_DONE) &
(run 1 328 0.2 2048; echo done > $O/CLASS_GPU1_DONE) &
wait
echo done > $O/CLASS_TESTS_DONE
