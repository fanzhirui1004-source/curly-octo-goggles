# The box is a 90 GiB container, and that is what broke the night's throughput

2026-09-19. `free` on this machine reports the host's 754 GB. The container's own limit is
`/sys/fs/cgroup/memory.max = 96 636 764 160` bytes, i.e. **90 GiB**, and
`memory.current` sat at 94.5 GB with every lane running. Every sizing decision that assumed
754 GB was wrong.

## What it does to a training arm

The labels are read through `np.load(mmap_mode='r')`. One step samples 8192 near plus 8192 far
pairs plus the complete diagonal, so `read_blocks` performs on the order of 73 728 scattered
8-byte reads over a 660 MB file. Those land on ~73 728 distinct 4 KB pages, and the kernel's
readahead fetches 128 KB per fault, so a step whose useful payload is 5 MB can drive hundreds of
megabytes to gigabytes of I/O unless the label file is already resident.

With 45.8 GB of other anonymous memory resident there was no room for that page cache, and the
arm sat in state `D` — uninterruptible I/O wait:

| condition | wall per step |
|---|---|
| lanes competing, 45.8 GB of other anon memory | 1.67 s |
| after `renice` of the other lanes | 1.28 s |
| after pinning each lane to disjoint cores | 0.68 s, then back to 1.9 s |
| after a sequential `dd` warm of all 75.7 GiB of labels | 1.67 s (worse: the warm competed for I/O) |
| **after KILLING the other lanes (anon 53 GB -> 17 GB, cache 39 GB -> 64 GB)** | **0.068 s** |

`renice`, `taskset` and cache warming all failed because the constraint was neither CPU nor
readahead order. `SIGSTOP` was actively counterproductive: a stopped process releases no memory,
and six stopped processes held 45.8 GB of the 90 GiB.

## The rule

**One training arm at a time**, with at most one label builder beside it:

* an arm needs 0.66 GiB of page cache per prepared seat -- 64 seats is 42 GiB, 115 seats is
  76 GiB and does not fit even alone;
* a label build at q = 13 000 needs 16 GiB, at q = 18 000 it needs 23 GiB;
* a matrix-free lattice run needs 2 x 1.25 GiB per distinct seat plus the vectors.

`/root/_serial.sh` enforces the ordering and `/root/_status.sh` prints each arm's wall-per-step,
where anything above 0.3 s means the rule is being broken.

## What this retroactively explains

* three `build_mq_label` CUDA OOMs and the `PEAK_BUFFERS` correction from 7 to 12 -- the host
  headroom that was assumed was never there;
* the four 2 x 2 x 2 dense lattice runs that died with no output (that one was also torch's 2^31
  indexing cap, now a checked precondition);
* why the 115-seat cut/full arm had to be cut to 64 seats.

## Cheap fixes not yet made

* `posix_fadvise(POSIX_FADV_RANDOM)` on the label files would stop the 128 KB readahead per
  fault, which is the largest single waste and would relax the seat-count limit;
* a `--ram-labels` flag would copy each packed label into the process once, sequentially, turning
  random faulting into one streaming read -- affordable up to about 60 seats.
