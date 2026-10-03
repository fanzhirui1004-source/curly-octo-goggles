D=/root/autodl-tmp/CLAUDE_AUGCOST_20260920
PY=/root/cutfem_neural_a_20260910/env/bin/python
M=/root/autodl-tmp/CLAUDE_LABELS_20260917/V2_LABELS.json
cd $D/src; export PYTHONPATH=$D/src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SHA=$(find superelement/equi -name '*.py' | sort | xargs sha256sum | sha256sum | cut -c1-16)
POOL="129 139 185 224 100006 100010 100013 100015 100017 100019 100024 100028 100032 100035 100041 100046 100050 100056 100068"
HELD="100001 100003 100009 245 100014 100016 100020 100022 100023 100025 230 100029"
echo "source_sha=$SHA pool=$(echo $POOL|wc -w) held=$(echo $HELD|wc -w) $(date -Is)"
arm() { NAME=$1; SEED=$2; shift 2
  if [ -f $D/$NAME/RESULT.json ]; then echo "skip $NAME"; else
    echo "=== train $NAME seed=$SEED flags=$* $(date -Is)"; rm -rf $D/$NAME
    python3 /root/_fadv.py >/dev/null 2>&1
    $PY -u -m superelement.equi.train_equi --manifest $M --seats $POOL --train-seats $POOL \
      --eval-seats 100032 --steps 60000 --warmup 1000 --calibrate-pairs 4096 --pairs-per-bucket 8192 \
      --eval-device cpu --probe-rotations 2 --threads 6 --checkpoint-every 8192 --ram-labels --bf16 \
      --seed $SEED --source-sha $SHA --output $D/$NAME "$@" > $D/$(echo $NAME | tr A-Z a-z).log 2>&1
    echo "--- train $NAME rc=$? $(date -Is)"
    head -1 $D/$(echo $NAME | tr A-Z a-z).log
  fi
  if [ -f $D/GATE/$NAME/RESULT.json ]; then echo "skip gate $NAME"; else
    python3 /root/_fadv.py >/dev/null 2>&1
    echo "=== gate $NAME $(date -Is)"; rm -rf $D/GATE/$NAME
    $PY -u -m superelement.equi.sweep_checkpoints --run $D/$NAME --seats $HELD 100032 --last 1 \
      --output $D/GATE/$NAME > $D/gate_$(echo $NAME | tr A-Z a-z).log 2>&1
    echo "--- gate $NAME rc=$? $(date -Is)"
  fi
}
arm DOSE_00  20260920 --augment --augment-full-only
arm DOSE_00B 20260921 --augment --augment-full-only
arm DOSE_24  20260920 --augment --proper-only
arm DOSE_48  20260920 --augment
echo "AUGCOST_DONE $(date -Is)"
