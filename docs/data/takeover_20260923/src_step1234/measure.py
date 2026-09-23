"""Run one command and record wall time, peak resident memory of the whole process tree and peak GPU memory in use
(nvidia-smi, whole device), sampled every 0.2 s. Usage: measure.py <out.json> -- <command...>"""
import json, subprocess, sys, time
import psutil
out = sys.argv[1]; cmd = sys.argv[sys.argv.index('--') + 1:]
t0 = time.perf_counter()
p = subprocess.Popen(cmd)
proc = psutil.Process(p.pid)
peak_rss = peak_gpu = 0
def gpu():
    try:
        return int(subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                                  capture_output=True, text=True, timeout=5).stdout.split()[0])
    except Exception:
        return 0
base_gpu = gpu()
while p.poll() is None:
    try:
        rss = proc.memory_info().rss + sum(c.memory_info().rss for c in proc.children(recursive=True))
    except psutil.Error:
        rss = 0
    peak_rss = max(peak_rss, rss); peak_gpu = max(peak_gpu, gpu())
    time.sleep(0.2)
rec = dict(command=cmd, exit=p.returncode, wall_seconds=time.perf_counter() - t0, peak_rss_gib=peak_rss / 2**30,
           peak_gpu_mib=peak_gpu, gpu_mib_before=base_gpu)
json.dump(rec, open(out, 'w'), indent=2)
print(json.dumps(rec))
sys.exit(p.returncode)
