#!/bin/sh
# Waits for V1G's step-3000 evaluation, pauses V1G (records the true stop step), then runs the two free-coefficient capacity tests
# and their exact spectra. Decision on resuming V1G is taken afterwards from the spectra, not automatically.
V=/root/autodl-tmp/CUTFEM_SUPERELEMENT_20260912_V1G; O=/root/autodl-tmp/CUTFEM_SPECTRUM_20260912; P=/root/cutfem_neural_a_20260910/env/bin/python
cd /root/cutfem_neural_a_20260910/superelement_v0
while [ ! -f $V/EVALUATION_003000.json ]; do sleep 15; done
sleep 3
PID=$(pgrep -f "v1_superelement.py .*CUTFEM_SUPERELEMENT_20260912_V1G")
STEP=$(tail -1 $V/HISTORY.jsonl | $P -c "import sys,json; print(json.loads(sys.stdin.read())['step'])")
kill $PID; sleep 8; pgrep -f "CUTFEM_SUPERELEMENT_20260912_V1G" > /dev/null && kill -9 $PID
echo "{\"stopped_after_step\": $STEP, \"resumable_checkpoint\": \"CHECKPOINT_003000.pt\", \"reason\": \"paused after the step-3000 evaluation for the free-coefficient capacity test\"}" > $V/STOP.json
sleep 5
for SEAT in 328 415; do
  OUT=/root/autodl-tmp/CUTFEM_SUPERELEMENT_20260912_FREE_$SEAT
  ./run_free.sh $SEAT $OUT 3000 $V/CHECKPOINT_003000.pt
  $P spectrum_eval.py --run $OUT --checkpoint $OUT/CHECKPOINT_003000.pt --seat $SEAT --out $O --threads 12 > $O/free_${SEAT}_spectrum.log 2>&1
done
echo done > $O/FREE_TESTS_DONE
