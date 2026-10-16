# ST12b: route (a) single thread (iparm(24) = iparm(25) = 0, ordering as tuned; METIS ordering if that fails), as chain_r1_cpu.sh
. /root/autodl-tmp/cpu9754/jobenv.sh
export MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
$PY - $IPT $H1 <<'PY'
import json, sys
t = json.load(open(sys.argv[1])); ip = dict(t['iparm']); ip['24'] = 0; ip['25'] = 0
json.dump(dict(iparm=ip, derived_from=sys.argv[1], note='single thread: iparm(24)=0, iparm(25)=0; ordering as tuned'), open(sys.argv[2] + '/iparm_1thread.json', 'w'), indent=1)
ip2 = dict(ip); ip2['2'] = 2
json.dump(dict(iparm=ip2, derived_from=sys.argv[1], note='single thread, fallback ordering iparm(2)=2 (METIS)'), open(sys.argv[2] + '/iparm_1thread_metis.json', 'w'), indent=1)
PY
$PY -u lat_direct_cpu2.py $H1/lat1_hlat222_chol.json $S4/hlat222.json --mtypes 2 --iparm-file $H1/iparm_1thread.json --margin-gib 4
if ! grep -q '"factor_s"' $H1/lat1_hlat222_chol.json 2>/dev/null; then
  $PY -u lat_direct_cpu2.py $H1/lat1_hlat222_chol_metis.json $S4/hlat222.json --mtypes 2 --iparm-file $H1/iparm_1thread_metis.json --margin-gib 4
fi
