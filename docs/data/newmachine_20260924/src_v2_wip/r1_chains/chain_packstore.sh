#!/bin/bash
# Section 6.11 step 6b: exact packed storage (OPL_STREAM_PACK: triangular/symmetric matrices, int32 indices) on hlat222,
# every cell streamed (--resident-gb 0), the configuration of the 92-cell scale run (sens ad, summed objective, sparse coarse
# space, fastidx, coarse templates), one design iteration; compared with the unpacked run fp32store_s0.
# Tag PACKS; marker PACKS_DONE_20260930 (EXIT trap).
ST=/root/autodl-tmp/OPL/S1/V2/chain_p1.status; R=/root/autodl-tmp/OPL/S1/V2/R1/SCALE; TAG=PACKS; RC=0
mkdir -p $R
st() { echo "$(date +%T) $TAG $*" >> $ST; }
fin() { st "END rc=$RC"; echo "$(date +%T) PACKS_DONE_20260930" >> $ST; }
trap fin EXIT
run() { local n=$1 to=$2; shift 2; st "START $n"; local t=$(date +%s)
  timeout $to "$@" > $R/$n.log 2>&1; local r=$?; [ $r != 0 ] && RC=$r; st "END $n rc=$r $(( $(date +%s) - t ))s"; }
pgrep -f "opt_design.py|lat_scale.py" >/dev/null && { st "GPU busy - not started"; exit 1; }
source /root/autodl-tmp/OPL/S1/env_gpu.sh
export OPL_PACKETS_EXTRA=/root/autodl-tmp/OPL/S4/packets:/root/autodl-tmp/OPL/S3/packets OPL_GP_CACHE=0 OPL_CONV_FP32=1
unset OPL_MODEL_ARGS_OVERRIDE FUSED_HYPER SENS_REASSOC LAT_CPU; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /root/autodl-tmp/OPL/src_v2
ARGS="/root/autodl-tmp/OPL/S4/hlat222.json --body /root/autodl-tmp/OPL/S4/body --tol 1e-6 --levels 1e-2,1e-3,1e-4 --iters 1 --sens ad --sens-obj sum --sparse-coarse --fastidx --coarse-tpl --resident-gb 0"
OPL_STREAM_PACK=0 run base2 3600 $PY -u lat_scale.py $R/basestore2.json $ARGS --save-sens $R/basestore2.npz
OPL_STREAM_PACK=1 run pack1 3600 $PY -u lat_scale.py $R/packstore.json $ARGS --save-sens $R/packstore.npz
$PY - <<'P' > $R/packstore_compare.txt 2>&1
import json, numpy as np
R='/root/autodl-tmp/OPL/S1/V2/R1/SCALE'
a, b = json.load(open(f'{R}/fp32store_s0.json')), json.load(open(f'{R}/packstore.json'))
za, zb = np.load(f'{R}/fp32store_s0.npz'), np.load(f'{R}/packstore.npz')
print('compliance base', za['compliance_it0'], 'packed', zb['compliance_it0'], 'bitwise equal', bool(np.array_equal(za['compliance_it0'], zb['compliance_it0'])),
      'max rel', float(np.max(np.abs(zb['compliance_it0'] - za['compliance_it0']) / np.abs(za['compliance_it0']))))
print('sens bitwise equal', bool(np.array_equal(za['sens_ad_it0'], zb['sens_ad_it0'])), 'rel', float(np.linalg.norm(zb['sens_ad_it0'] - za['sens_ad_it0']) / np.linalg.norm(za['sens_ad_it0'])))
zc = np.load(f'{R}/basestore2.npz')
print('run-to-run (unpacked twice): compliance max rel', float(np.max(np.abs(zc['compliance_it0'] - za['compliance_it0']) / np.abs(za['compliance_it0']))),
      'sens rel', float(np.linalg.norm(zc['sens_ad_it0'] - za['sens_ad_it0']) / np.linalg.norm(za['sens_ad_it0'])), 'bitwise', bool(np.array_equal(zc['compliance_it0'], za['compliance_it0'])))
for x, n in ((a, 'base'), (b, 'packed')):
    it = x['iterations'][0]
    print(n, 'stream_GB', round(it['stream_GB_total'], 4), 'rss', it['host_rss_GB_after_front_end'], 'peak', round(it['peak_device_GB'], 2), 'pcg', it['solve']['iterations'],
          'solve', round(it['phases']['solve'], 1), 'sens', round(it['phases']['sens_total'], 1), 'total', round(it['phases']['iteration_total'], 1))
    for c in it['per_cell']:
        print('   ', c['case'], round(c['stream_bytes'] / 1e9, 4), c.get('stream_bytes_enc_counts'), round(c.get('stream_bytes_encoded_unpacked', 0) / 1e9, 4))
P
