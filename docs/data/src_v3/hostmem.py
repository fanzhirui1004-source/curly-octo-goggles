"""Host-memory budget for the per-cell state of a lattice analysis (src_v3).

The streamed cells live in page-locked host memory, so the number of cells an analysis can hold is bounded by the host
memory the process may use: on the AutoDL container that is the cgroup limit (memory.max, 90 GiB), not the machine's RAM.
Two per-cell items trade memory for time (both give the same results):
  - the float64 partial-element matrices (teacher.Cell.store64, S9): ~0.2 GB per full plate cell, saves ~4 ms per S_hat;
  - the hyperedge slot weights (fastnet slots, S5): ~0.24 GB per full plate cell; without them they are recomputed in
    every application (bitwise the same, ~+9 ms per S_hat).
Budget.decide (one call per cell, in front-end order, after the cell's warm-up) keeps both while the projected host use
of the whole analysis (the current RSS plus, for this and every remaining cell, the average state per cell so far plus
the two items) stays below OPL_V3_HOST_FRAC (0.85) of the limit; otherwise it drops the float64 matrices, then the slot
weights. The plan is redone for every cell (the cells processed first get the items). 0.85: the RSS peak of an analysis
is ~1.1 x the RSS at the end of its front end (the K_PP assembly, the lattice, cuDSS, the sensitivities; measured on the
51-162-cell plates), so the peak stays below ~0.94 of the limit."""
import os

GiB = 2 ** 30


def limit_bytes():
    """The host memory this process may use: the cgroup limit if there is one, else the physical RAM."""
    ram = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
    for p in ('/sys/fs/cgroup/memory.max', '/sys/fs/cgroup/memory/memory.limit_in_bytes'):
        try:
            v = open(p).read().strip()
            if v and v != 'max':
                return min(int(v), ram)
        except (OSError, ValueError):
            pass
    return ram


def rss_bytes():
    try:
        for line in open('/proc/self/status'):
            if line.startswith('VmRSS:'):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


class Budget:
    def __init__(self, n_cells, frac=None, first_cell_bytes=0.55 * GiB):
        self.n, self.done, self.extra = int(n_cells), 0, 0
        self.frac = float(os.environ.get('OPL_V3_HOST_FRAC', '0.85')) if frac is None else frac
        self.limit = limit_bytes()
        self.rss0 = rss_bytes()
        self.first = first_cell_bytes
        self.log = []

    def decide(self, ke64_bytes, slot_bytes, cell_bytes=None):
        """(keep the slot weights, keep the float64 matrices) for the next cell. The remaining host budget (the limit's
        fraction minus the current RSS minus the bare state of this and every remaining cell, cell_bytes each: this cell's
        device footprint without the two items, or the average host growth per cell so far) is planned anew for every
        cell: slot weights first (they save more time per byte), for as many of the remaining cells as fit, then the
        float64 matrices; the cells processed first get them."""
        rss = rss_bytes()
        n_rest = self.n - self.done                                      # this cell and the ones after it
        if cell_bytes is not None:
            other = cell_bytes
        else:
            other = (rss - self.rss0 - self.extra) / self.done if self.done else self.first
        free = self.frac * self.limit - rss - n_rest * other
        ns = n_rest if slot_bytes <= 0 else max(0, min(n_rest, int(free // slot_bytes)))
        free -= ns * max(slot_bytes, 0)
        if ns < n_rest:                                                  # slot weights for every remaining cell first
            nk = 0
        else:
            nk = n_rest if ke64_bytes <= 0 else max(0, min(n_rest, int(free // ke64_bytes)))
        keep = (ns >= 1, nk >= 1)
        self.log.append((self.done, keep, round(rss / GiB, 2)))
        return keep

    def account(self, kept_bytes):
        self.extra += kept_bytes
        self.done += 1

    def summary(self):
        ks = [k for _, k, _ in self.log]
        return dict(limit_GiB=round(self.limit / GiB, 1), frac=self.frac, cells=len(ks),
                    slots_kept=sum(k[0] for k in ks), float64_kept=sum(k[1] for k in ks))
