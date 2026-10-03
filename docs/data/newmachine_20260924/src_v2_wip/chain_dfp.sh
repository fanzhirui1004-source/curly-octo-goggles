#!/bin/bash
# Data-free pilot (2026-09-28): DF, CTRL, DF+W arms (10k steps each, random init, label-free selection), each followed by
# the 80-cell newval2 evaluation. Status lines in $O/chain_p1.status, prefix DFP; final marker DF_PILOT_DONE_20260928.
source /root/autodl-tmp/OPL/S1/env_gpu.sh
export OPL_PACKETS_EXTRA=/root/autodl-tmp/OPL/S3/packets OPL_GP_CACHE=0
cd /root/autodl-tmp/OPL/src_v2
O=/root/autodl-tmp/OPL/S1/V2; ST=$O/chain_p1.status
VAL=$(python3 -c "import json;print(','.join(json.load(open('/root/autodl-tmp/OPL/S2/SPLIT_ARMS.json'))['val_s3']))")
W='{"smooth_k": 8, "smooth_alpha": 30.0, "coarse_space": "Q1_17"}'
st() { echo "$(date +%T) DFP $*" >> $ST; }
train() { st "TRAIN_START $1"; env FUSED_HYPER=1 OPL_CONV_FP32=1 SENS_REASSOC=1 $PY -u train3.py $O/$1.json > $O/$1.log 2>&1; st "TRAIN_END $1 rc=$?"; }
nv() { st "NEWVAL_START $2"; ( if [ -n "$3" ]; then export OPL_MODEL_ARGS_OVERRIDE="$3"; fi; export OPL_CONV_FP32=1; $PY -u eval_views.py $O/$1/best.pt $O/$2.json --views 0 --cases $VAL --body /root/autodl-tmp/OPL/S0 --data /root/autodl-tmp/OPL/S2/data_v2 ) > $O/$2.log 2>&1; st "NEWVAL_END $2 rc=$?"; }
st "CHAIN_START chain_dfp.sh pid=$$"
train DFP_DF
nv DFP_DF newval2_DFP_DF
nv DFP_DF newval2_DFP_DF_W "$W"
train DFP_CTRL
nv DFP_CTRL newval2_DFP_CTRL
nv DFP_CTRL newval2_DFP_CTRL_W "$W"
train DFP_DFW
nv DFP_DFW newval2_DFP_DFW
echo "$(date +%T) DF_PILOT_DONE_20260928" >> $ST
