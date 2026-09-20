D=/root/autodl-tmp/CLAUDE_CUT_ROT_20260920
T=/root/autodl-tmp/CUTFEM_INGEST_R38/source_independent_6624dc8_20260910
PY=/root/autodl-tmp/CUTFEM_INGEST_R38/environment/.venv/bin/python
. $D/ENV.sh
SEAT=100000
SHA=$(sha256sum $D/src/superelement/objective/cut_rotation_witness.py | cut -c1-16)
python3 /root/_fadv.py 2>&1 | tail -3
for L in BASE_REPLAY G01 G20 G10; do
  rm -rf $D/ROT_${SEAT}_NUM/${L}_NUMERIC; mkdir -p $D/ROT_${SEAT}_NUM
  echo "=== numeric $L (cgroup $(awk '{printf "%.1f", $1/1073741824}' /sys/fs/cgroup/memory.current) GiB)"
  $PY -u -m superelement.objective.cut_rotation_witness numeric --geometry $D/ROT_$SEAT/$L \
    --output $D/ROT_${SEAT}_NUM/${L}_NUMERIC --source $T --source-sha $SHA --workers 8 2>&1 | tail -3
  python3 /root/_fadv.py 2>&1 | tail -1
done
echo "--- numeric done ---"
rm -rf $D/COMPARE_$SEAT
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  $PY -u -m superelement.objective.cut_rotation_witness compare --seat $SEAT \
  --root $D/ROT_$SEAT --numeric-root $D/ROT_${SEAT}_NUM --output $D/COMPARE_$SEAT \
  --source $T --source-sha $SHA 2>&1 | tail -20
echo "ROT_WITNESS_DONE $(date -Is)"
