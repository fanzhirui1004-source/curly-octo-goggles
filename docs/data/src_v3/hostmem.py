"""Host-memory budget for the per-cell state of a lattice analysis (src_v3).

The streamed cells live in page-locked host memory, so the number of cells an analysis can hold is bounded by the host
memory the process may use: on the AutoDL container that is the cgroup limit (memory.max, 90 GiB), not the machine's RAM.
Two per-cell items trade memory for time (both give the same results):
  - the float64 partial-element matrices (teacher.Cell.store64, S9): ~0.09 GB per full plate cell, saves ~4.1 ms per S_hat;
  - the hyperedge slot weights (fastnet slots, S5): ~0.29 GB per full plate cell; without them they are recomputed in
    every application (bitwise the same, +4.8 ms per S_hat with the fused tail of S19, +9.4 ms without).
Budget.decide (one call per cell, in front-end order, after the cell's warm-up) keeps both while the projected host use
of the whole analysis (the current RSS plus, for this and every remaining cell, its bare state plus the two items) stays
below the target; otherwise it drops the slot weights, then the float64 matrices (S14c: the float64 matrices save more
time per byte; OPL_V3_BUDGET_ORDER=slots: the S14 order). The plan is redone for every cell (the cells processed first
get the items).
Target (src_v3 S14b): OPL_V3_HOST_TOP (0.95) of the limit minus a reserve for the host peak of the K_PP factorisation
that follows the front end: OPL_V3_HOST_TRANSIENT (4; 0 with OPL_V3_SPILL=1, S25: parked cells are written to disk
for that peak instead, stream_ops.spill_guard) x the cells' K_PP triplet bytes on the host (projected from the
cells so far). Measured: the peak is ~71 bytes per assembled K_PP nonzero above the RSS at the end of the front end on the
135-, 162- and 192-cell plates (3.4-3.7 x the triplet bytes; the 192-cell run of S14b with 3 x reached 87.3 of 90 GiB,
since the triplets freed before cuDSS plans (S16) stayed in the heap; S16 now also trims the heap). OPL_V3_HOST_FRAC (S14, if set): the earlier rule, a fixed fraction of the limit for the front end."""
import os

GiB = 2 ** 30
# S14c: which item the budget gives out first. Time saved per byte, full plate cell (results/shat/prof1_v3h, prof1_rc,
# results/tests12/slotfuse.log): float64 matrices 4.1 ms per S_hat for 90 MB (46 ms/GB), slot weights 4.8 ms for 286 MB
# (17 ms/GB; 9.4 ms, 33 ms/GB, without the fused tail of S19).
ORDER = os.environ.get('OPL_V3_BUDGET_ORDER', 'ke64')
SPILL = os.environ.get('OPL_V3_SPILL', '0') == '1'                     # S25: the K_PP peak is met by spilling cells


def top_fraction():
    return float(os.environ.get('OPL_V3_HOST_TOP', '0.95'))


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


def trim():
    """glibc malloc_trim(0): return the free pages of the heap to the system (freed host tensors below the mmap
    threshold otherwise stay in the process's RSS). No-op elsewhere."""
    try:
        import ctypes
        ctypes.CDLL('libc.so.6').malloc_trim(0)
    except (OSError, AttributeError):
        pass


def rss_bytes():
    try:
        for line in open('/proc/self/status'):
            if line.startswith('VmRSS:'):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


class Budget:
    def __init__(self, n_cells, frac=None, first_cell_bytes=0.55 * GiB, top=None, transient=None):
        self.n, self.done, self.extra, self.kpp = int(n_cells), 0, 0, 0
        f = os.environ.get('OPL_V3_HOST_FRAC')
        self.frac = (float(f) if f else None) if frac is None else frac
        self.top = top_fraction() if top is None else top
        self.tx = float(os.environ.get('OPL_V3_HOST_TRANSIENT', '0' if SPILL else '4')) if transient is None else transient
        self.limit = limit_bytes()
        self.rss0 = rss_bytes()
        self.first = first_cell_bytes
        self.reserve = 0.0
        self.log = []

    def decide(self, ke64_bytes, slot_bytes, cell_bytes=None, kpp_bytes=0):
        """(keep the slot weights, keep the float64 matrices) for the next cell. The remaining host budget (the target
        minus the current RSS minus the bare state of this and every remaining cell, cell_bytes each: this cell's device
        footprint without the two items, or the average host growth per cell so far) is planned anew for every cell: the
        item that saves more time per byte first (ORDER), for as many of the remaining cells as fit, then the other; the
        cells processed first get them. kpp_bytes: this cell's K_PP triplets on the host (the reserve)."""
        rss = rss_bytes()
        n_rest = self.n - self.done                                      # this cell and the ones after it
        if cell_bytes is not None:
            other = cell_bytes
        else:
            other = (rss - self.rss0 - self.extra) / self.done if self.done else self.first
        self.kpp += kpp_bytes
        kpp_avg = self.kpp / (self.done + 1)
        if self.frac is not None:                                        # S14 rule
            target = self.frac * self.limit
        else:                                                            # the remaining cells' host triplets (this
            self.reserve = self.tx * kpp_avg * self.n                    # cell's are in the RSS) and the reserve
            target = self.top * self.limit - self.reserve - (n_rest - 1) * kpp_avg
        free = target - rss - n_rest * other

        def fit(nbytes, avail):
            return n_rest if nbytes <= 0 else max(0, min(n_rest, int(avail // nbytes)))
        if ORDER == 'slots':                                             # S14: slot weights first
            ns = fit(slot_bytes, free)
            nk = fit(ke64_bytes, free - ns * max(slot_bytes, 0)) if ns == n_rest else 0
        else:                                                            # S14c: the float64 matrices first
            nk = fit(ke64_bytes, free)
            ns = fit(slot_bytes, free - nk * max(ke64_bytes, 0)) if nk == n_rest else 0
        keep = (ns >= 1, nk >= 1)
        self.log.append((self.done, keep, round(rss / GiB, 2)))
        return keep

    def account(self, kept_bytes):
        self.extra += kept_bytes
        self.done += 1

    def summary(self):
        ks = [k for _, k, _ in self.log]
        return dict(limit_GiB=round(self.limit / GiB, 1), frac=self.frac, top=self.top, transient_x=self.tx,
                    reserve_GiB=round(self.reserve / GiB, 2), cells=len(ks),
                    slots_kept=sum(k[0] for k in ks), float64_kept=sum(k[1] for k in ks))


def resident_auto(n_cells):
    """src_v3 S23 (--resident-gb auto): device memory (GiB) the front end may fill with resident cells, so that the PCG
    phase still fits: total - (OPL_V3_RGB_FIXED 3.5 GiB + OPL_V3_RGB_PERCELL 0.054 GiB per cell: coarse A Z blocks,
    Z, the K_PP factor, lattice vectors; measured on the 51-192-cell plates) - OPL_V3_RGB_MARGIN (3 GiB). Resident cells
    take no host memory (no pinned copy), so on a host-bound container they leave room for more cells or items; a
    3-D lattice's K_PP factor is larger per cell than a plate's (set OPL_V3_RGB_PERCELL for it)."""
    import torch
    tot = torch.cuda.get_device_properties(0).total_memory / GiB
    fixed = float(os.environ.get('OPL_V3_RGB_FIXED', '3.5'))
    per = float(os.environ.get('OPL_V3_RGB_PERCELL', '0.054'))
    margin = float(os.environ.get('OPL_V3_RGB_MARGIN', '3'))
    return max(0.0, tot - fixed - per * n_cells - margin)
