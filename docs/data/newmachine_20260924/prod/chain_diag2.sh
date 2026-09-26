# replaces chain_arms.sh after A0 training and chain_diag.sh: A0 post (newval + gates), then the gate decomposition of
# v2L1 (decides the next architecture step), diag_sens A0, then arm A1_sens3 (train + post), then CHAIN_ARMS_1_DONE
source /root/autodl-tmp/OPL/S1/env_gpu.sh
export OPL_PACKETS_EXTRA=/root/autodl-tmp/OPL/S3/packets OPL_GP_CACHE=0
cd /root/autodl-tmp/OPL/src_v2
O=/root/autodl-tmp/OPL/S1/V2
run() { n=$1; shift; echo "$(date +%T) START $n" >> $O/chain_arms.status; "$@" > $O/$n.log 2>&1; echo "$(date +%T) END $n rc=$?" >> $O/chain_arms.status; }
cfg() {
  python3 - "$1" "$2" <<'PY'
import json, sys
name, ov = sys.argv[1], json.loads(sys.argv[2])
c = json.load(open('/root/autodl-tmp/OPL/S1/V2/v2L1.json'))
c.update(out=f'/root/autodl-tmp/OPL/S1/V2/{name}', init='/root/autodl-tmp/OPL/S1/V2/v2L1/best.pt', split='/root/autodl-tmp/OPL/S2/SPLIT_ARMS.json',
         steps=15000, eval_every=7500, select_min_step=7500, eval_views=[0, 17], val_max=40, seed=0)
for k, v in ov.items():
    if k == 'model_args':
        c['model_args'] = dict(c['model_args'], **v)
    else:
        c[k] = v
json.dump(c, open(f'/root/autodl-tmp/OPL/S1/V2/{name}.json', 'w'), indent=1)
PY
}
post() {
  VAL=$(python3 -c "import json;print(','.join(json.load(open('/root/autodl-tmp/OPL/S2/SPLIT_ARMS.json'))['val_s3']))")
  OPL_CONV_FP32=1 run newval_$1 $PY -u eval_views.py $O/$1/best.pt $O/newval_$1.json --views 0 --cases $VAL --body /root/autodl-tmp/OPL/S0 --data /root/autodl-tmp/OPL/S2/data
  for c in fresh_val_2000_full fresh_val_2001_full fresh_val_2005_d1_v0 fresh_val_2006_d0_v1 fresh_val_2003_d1_v1 fresh_val_2004_d0_v2 fresh_val_2010_d0_v0 fresh_val_2002_d0_v0; do
    for conf in x y; do
      OPL_CONV_FP32=1 FUSED_HYPER=1 run gate_$1_${c}_$conf $PY -u lat_full.py $O/gate_$1_${c}_$conf.json --pairs '' --custom "$O/$1/best.pt:$c:$O/$1/best.pt:" --nb-mode explicit --sets test --configs $conf
    done
  done
  echo "$(date +%T) ARM_DONE $1" >> $O/chain_arms.status
}
while pgrep -f "train3.py $O/A0_ctrl.json" > /dev/null; do sleep 60; done
echo "$(date +%T) TRAINED A0_ctrl (chain_diag2)" >> $O/chain_arms.status
post A0_ctrl
for c in fresh_val_2000_full fresh_val_2003_d1_v1 fresh_val_2006_d0_v1; do
  for conf in x y; do
    OPL_CONV_FP32=1 FUSED_HYPER=1 run decomp_v2L1_${c}_$conf $PY -u gate_decomp.py $O/decomp_v2L1_${c}_$conf.json $O/v2L1/best.pt $c --configs $conf
  done
done
OPL_CONV_FP32=1 run diag_sens_A0 $PY -u diag_sens.py $O/A0_ctrl/best.pt $O/diag_sens_A0.json fresh_val_2005_d1_v0,fresh_val_2006_d0_v1,fresh_val_2003_d1_v1,fresh_val_2000_full,fresh_val_2001_full
echo "$(date +%T) CHAIN_DIAG2_DIAG_DONE" >> $O/chain_arms.status
if [ ! -f $O/SKIP_A1 ]; then
  cfg A1_sens3 '{"sens_w": 3.0}'
  env FUSED_HYPER=1 OPL_CONV_FP32=1 SENS_REASSOC=1 bash -c "cd /root/autodl-tmp/OPL/src_v2 && $PY -u train3.py $O/A1_sens3.json" > $O/A1_sens3.log 2>&1
  echo "$(date +%T) TRAINED A1_sens3 rc=$?" >> $O/chain_arms.status
  post A1_sens3
fi
echo CHAIN_ARMS_1_DONE >> $O/chain_arms.status
