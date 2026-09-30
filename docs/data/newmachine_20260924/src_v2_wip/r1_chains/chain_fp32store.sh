#!/bin/bash
# Section 6.11 step 6a: float32 storage of the streamed float64 state (OPL_STREAM_FP32) - accuracy and memory on hlat222,
# every cell streamed (--resident-gb 0), the configuration of the 92-cell scale run (sens ad, summed objective, sparse coarse
# space, fastidx, coarse templates), one design iteration; float64 store vs float32 store.
# Tag FP32S; marker FP32S_DONE_20260930 (EXIT trap).
ST=/root/autodl-tmp/OPL/S1/V2/chain_p1.status; R=/root/autodl-tmp/OPL/S1/V2/R1/SCALE; TAG=FP32S; RC=0
mkdir -p $R
st() { echo "$(date +%T) $TAG $*" >> $ST; }
fin() { st "END rc=$RC"; echo "$(date +%T) FP32S_DONE_20260930" >> $ST; }
trap fin EXIT
run() { local n=$1 to=$2; shift 2; st "START $n"; local t=$(date +%s)
  timeout $to "$@" > $R/$n.log 2>&1; local r=$?; [ $r != 0 ] && RC=$r; st "END $n rc=$r $(( $(date +%s) - t ))s"; }
pgrep -f "opt_design.py|lat_scale.py" >/dev/null && { st "GPU busy - not started"; exit 1; }
source /root/autodl-tmp/OPL/S1/env_gpu.sh
export OPL_PACKETS_EXTRA=/root/autodl-tmp/OPL/S4/packets:/root/autodl-tmp/OPL/S3/packets OPL_GP_CACHE=0 OPL_CONV_FP32=1
unset OPL_MODEL_ARGS_OVERRIDE FUSED_HYPER SENS_REASSOC LAT_CPU; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /root/autodl-tmp/OPL/src_v2
ARGS="/root/autodl-tmp/OPL/S4/hlat222.json --body /root/autodl-tmp/OPL/S4/body --tol 1e-6 --levels 1e-2,1e-3,1e-4 --iters 1 --sens ad --sens-obj sum --sparse-coarse --fastidx --coarse-tpl --resident-gb 0"
for s in 0 1; do
  OPL_STREAM_FP32=$s run store$s 3600 $PY -u lat_scale.py $R/fp32store_s$s.json $ARGS --save-sens $R/fp32store_s$s.npz
done
$PY - <<'P' >> $R/fp32store_compare.txt 2>&1
import json, numpy as np
R='/root/autodl-tmp/OPL/S1/V2/R1/SCALE'
a, b = (json.load(open(f'{R}/fp32store_s{s}.json')) for s in (0, 1))
za, zb = (np.load(f'{R}/fp32store_s{s}.npz') for s in (0, 1))
ca, cb = za['compliance_it0'], zb['compliance_it0']
sa, sb = za['sens_ad_it0'], zb['sens_ad_it0']
print('compliance f64', ca, 'f32', cb, 'rel', (cb - ca) / ca)
print('sens rel (global)', float(np.linalg.norm(sb - sa) / np.linalg.norm(sa)), 'max per cell', float(max(np.linalg.norm(sb[i] - sa[i]) / np.linalg.norm(sa[i]) for i in range(len(sa)))))
for x, n in ((a, 'f64'), (b, 'f32')):
    it = x['iterations'][0]
    bd = {}
    for c in it['per_cell']:
        for k, v in c.get('stream_bytes_by_dtype', {}).items(): bd[k] = bd.get(k, 0) + v
    print(n, 'stream_GB', round(it['stream_GB_total'], 3), 'by dtype GB', {k: round(v / 1e9, 3) for k, v in bd.items()}, 'rss', it['host_rss_GB_after_front_end'],
          'peak', round(it['peak_device_GB'], 2), 'pcg', it['solve']['iterations'], 'phases', {k: round(v, 1) for k, v in it['phases'].items()})
P
cat $R/fp32store_compare.txt >> $R/store1.log
