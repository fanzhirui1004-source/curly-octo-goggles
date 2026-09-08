#!/usr/bin/env python3
"""Per-machine production queue for sheet labels: many jobs in flight, memory budgeted at the Schur stage.

Runs produce_sheet_label.py for every case of a population list, skipping cases whose output directory already
holds a receipt (idempotent restarts), with `--jobs` processes at a time.  Concurrency is set by the cores, not
by the memory: the jobs share one MemoryBudget so only as many Schur stages run as fit under `--budget-gb`, while
the single-threaded surface and port stages of the other jobs keep the remaining cores busy.
"""
import argparse, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=Path, required=True, help="text file, one case id per line")
    ap.add_argument("--cells", type=Path, required=True, help="directory holding <case>/geometry_material_manifest.json")
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=14, help="processes in flight (surface stages are single-threaded)")
    ap.add_argument("--threads", type=int, default=8, help="threads per job for Gmsh, the assembly and Pardiso")
    ap.add_argument("--budget-gb", type=float, default=110.0, help="memory for the Schur stages in flight")
    ap.add_argument("--resolution", default="production")
    ap.add_argument("--extra", default="", help="extra arguments passed to produce_sheet_label.py")
    a = ap.parse_args()
    cases = [c.strip() for c in a.cases.read_text().splitlines() if c.strip()]
    a.output_root.mkdir(parents=True, exist_ok=True); budget = a.output_root / "_memory_budget"
    todo = [c for c in cases if not (a.output_root / c / "LABEL_RECEIPT.json").exists()]
    print(f"{len(cases)} cases, {len(cases) - len(todo)} already labelled, {len(todo)} to run, {a.jobs} jobs x {a.threads} threads, budget {a.budget_gb} GB", flush=True)

    def run(case: str) -> tuple[str, int, float]:
        out = a.output_root / case; log = a.output_root / f"{case}.log"
        if out.exists():                                   # a directory without a receipt: an interrupted job
            subprocess.run(["rm", "-rf", str(out)], check=False)
        cmd = [sys.executable, str(ROOT / "scripts/pred777h_full_cube_v1/produce_sheet_label.py"), "--case-id", case,
               "--geometry-manifest", str(a.cells / case / "geometry_material_manifest.json"), "--output-dir", str(out),
               "--resolution", a.resolution, "--workers", str(a.threads), "--threads", str(a.threads),
               "--memory-budget-dir", str(budget), "--memory-budget-gb", str(a.budget_gb)] + a.extra.split()
        env = {"OMP_NUM_THREADS": str(a.threads), "MKL_NUM_THREADS": str(a.threads), "PYTHONPATH": str(ROOT / "src")}
        import os
        env = {**os.environ, **env}; t0 = time.perf_counter()
        with open(log, "w") as fh:
            rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=str(ROOT)).returncode
        return case, rc, time.perf_counter() - t0

    t_all = time.perf_counter(); done = 0; failed = []
    with ThreadPoolExecutor(max_workers=a.jobs) as pool:
        futures = [pool.submit(run, c) for c in todo]
        for fut in as_completed(futures):
            case, rc, secs = fut.result(); done += 1
            status = "?"
            rp = a.output_root / case / "LABEL_RECEIPT.json"
            if rp.exists():
                import json
                status = json.loads(rp.read_text()).get("status", "?")
            if rc != 0 or status not in ("PASS", "EMPTY", "GEOMETRY_DEGENERATE"):
                failed.append(case)
            elapsed = time.perf_counter() - t_all
            print(f"[{done}/{len(todo)}] {case} rc={rc} {status} {secs:.0f}s  elapsed {elapsed/60:.1f} min  rate {done / elapsed * 3600:.0f}/h", flush=True)
    print(f"done: {done} run, {len(failed)} failed: {failed}", flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
