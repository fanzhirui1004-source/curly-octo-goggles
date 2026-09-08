#!/bin/bash
# Production run of the sheet fixed-port label route.
#   - one code state for the whole run (the git hash is in every receipt)
#   - production preset, carrier 1/32, HXT at one generation thread
#   - a MESH_FAIL is retried ONCE with a 8 % smaller interior size, then recorded as unproducible
#   - the cooperative memory budget serialises the Schur stages under the cgroup's 128 GiB
R=/root/autodl-tmp/cut_control_tpms_v1_full_cube_single_cell_v1
P=/root/autodl-tmp/_claude_diag/population
OUT=/root/autodl-tmp/_claude_diag/production
mkdir -p $OUT/labels $OUT/budget
cd $R
export PYTHONPATH=src

run_one() {
  c=$1
  [ -f $OUT/labels/$c/LABEL_RECEIPT.json ] && { echo "$c skip" >> $OUT/DONE.txt; return; }
  rm -rf $OUT/labels/$c
  s=$(date +%s)
  OMP_NUM_THREADS=5 timeout 9000 python3 scripts/pred777h_full_cube_v1/produce_sheet_label.py \
    --case-id $c --geometry-manifest $P/cells/$c/geometry_material_manifest.json \
    --output-dir $OUT/labels/$c --resolution production --workers 5 --threads 5 \
    --memory-budget-dir $OUT/budget --memory-budget-gb 80 > $OUT/labels/$c.log 2>&1
  rc=$?
  if [ $rc -ne 0 ] && grep -q "MESH_FAIL" $OUT/labels/$c.log; then
    mv $OUT/labels/$c.log $OUT/labels/$c.attempt1.log; rm -rf $OUT/labels/$c
    OMP_NUM_THREADS=5 timeout 9000 python3 scripts/pred777h_full_cube_v1/produce_sheet_label.py \
      --case-id $c --geometry-manifest $P/cells/$c/geometry_material_manifest.json \
      --output-dir $OUT/labels/$c --resolution production --size-max 0.0368 --workers 5 --threads 5 \
      --memory-budget-dir $OUT/budget --memory-budget-gb 80 > $OUT/labels/$c.log 2>&1
    rc=$?; echo "$c retry exit=$rc seconds=$(( $(date +%s) - s ))" >> $OUT/DONE.txt; return
  fi
  echo "$c exit=$rc seconds=$(( $(date +%s) - s ))" >> $OUT/DONE.txt
}
export -f run_one; export OUT P R
git rev-parse HEAD > $OUT/GIT_HEAD.txt; git status --short | grep -v "^??" > $OUT/GIT_DIRTY.txt
sort $P/CASES.txt > $OUT/CASES.txt
wc -l < $OUT/CASES.txt > $OUT/TOTAL.txt
cat $OUT/CASES.txt | xargs -P 10 -I{} bash -c 'run_one {}'
echo "PRODUCTION_DONE" >> $OUT/DONE.txt
