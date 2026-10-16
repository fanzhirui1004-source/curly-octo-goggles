# Summaries with the scripts of the EPYC 9654 runs (unchanged): per-cell (ST13) and lattice (Table 5 (a), ST12b) tables, route (b)
. /root/autodl-tmp/cpu9754/jobenv.sh
$PY r1_host_summary.py $H $H/host_summary.json --one $H1
$PY r1_cond_summary.py $HC
