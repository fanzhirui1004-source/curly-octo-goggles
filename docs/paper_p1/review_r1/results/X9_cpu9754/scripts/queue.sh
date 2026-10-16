#!/bin/bash
# Sequential job runner of the EPYC 9754 CPU campaign: runs the lexicographically first job of pending/ (one at a time,
# never two timed runs together), waits while an rsync transfer is in progress, records start / end / exit code /
# seconds / load average / CPU throttling in status.txt. Jobs can be added to pending/ at any time; touch STOP to end.
Q=/root/cpu9754_queue; mkdir -p $Q/pending $Q/running $Q/done $Q/failed $Q/logs
ST=$Q/status.txt
st() { echo "$(date +%F_%T) $*" >> $ST; }
thr() { awk '/throttled_usec/{print $2}' /sys/fs/cgroup/cpu.stat; }
st "RUNNER start pid=$$ host=$(hostname)"
while [ ! -f $Q/STOP ]; do
  j=$(ls $Q/pending 2>/dev/null | sort | head -1)
  if [ -z "$j" ]; then sleep 20; continue; fi
  while pgrep -x rsync > /dev/null; do sleep 30; done
  mv $Q/pending/$j $Q/running/$j
  t0=$(date +%s); h0=$(thr)
  st "START $j load=$(cut -d' ' -f1-3 /proc/loadavg)"
  bash $Q/running/$j > $Q/logs/${j%.sh}.log 2>&1; rc=$?
  st "END $j rc=$rc $(( $(date +%s) - t0 ))s load=$(cut -d' ' -f1-3 /proc/loadavg) throttled_s=$(( ($(thr) - h0) / 1000000 ))"
  if [ $rc = 0 ]; then mv $Q/running/$j $Q/done/$j; else mv $Q/running/$j $Q/failed/$j; fi
done
st "RUNNER stop"
