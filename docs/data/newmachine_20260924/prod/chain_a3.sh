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
# A3_2grid: v2L1 + physics wrapper tail(8) -> Galerkin coarse correction on Q1_17 -> tail(8), trained end-to-end (views too)
cfg A3_2grid '{"model_args": {"smooth_k": 8, "smooth_alpha": 30.0, "coarse_space": "Q1_17"}}'
env FUSED_HYPER=1 OPL_CONV_FP32=1 SENS_REASSOC=1 bash -c "cd /root/autodl-tmp/OPL/src_v2 && $PY -u train3.py $O/A3_2grid.json" > $O/A3_2grid.log 2>&1
echo "$(date +%T) TRAINED A3_2grid rc=$?" >> $O/chain_arms.status
post A3_2grid
# A2b_tail8: the tail now also on the O_h views (A2_tail8 trained without it: every training view bypassed Geo.field)
cfg A2b_tail8 '{"model_args": {"smooth_k": 8, "smooth_alpha": 30.0}}'
env FUSED_HYPER=1 OPL_CONV_FP32=1 SENS_REASSOC=1 bash -c "cd /root/autodl-tmp/OPL/src_v2 && $PY -u train3.py $O/A2b_tail8.json" > $O/A2b_tail8.log 2>&1
echo "$(date +%T) TRAINED A2b_tail8 rc=$?" >> $O/chain_arms.status
post A2b_tail8
echo CHAIN_A3_DONE >> $O/chain_arms.status
