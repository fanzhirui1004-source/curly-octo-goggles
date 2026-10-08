export BANK_SPLITS=512,64,64
D=/root/autodl-tmp/OPL/S2/data; mkdir -p $D
C="fresh_train_0031_full fresh_train_0031_cover01_r1 fresh_train_0031_cover01_r2 fresh_train_0031_d0_v1 fresh_train_0031_d1_v2"
t0=$(date +%s); $PY -u prep_geo.py $S $D $C > $S1/pilot_prep_geo.log 2>&1; rc=$?
t1=$(date +%s); echo "PILOT_GPU rc $rc total $((t1-t0)) s" > $S1/pilot_gpu.done
du -sh $D/* >> $S1/pilot_gpu.done
