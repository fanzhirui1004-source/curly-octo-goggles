# Where the no-GP factor job actually spends its memory

Question: the coverage envelope is `3000 < q <= 23127`, which admits 488 of the
1024 identities on a 32 vCPU / 64 GB node. Is the cap set by the machine, or by
how we account for the machine?

Answer: by how we account for it. The peak is real but it is in the dense tail,
not the QR, and the admission forecast that draws the line is systematically
about 1.5x the peak that is actually measured.

## Method

No new instrumentation was needed for the decomposition. Every delivered label
already records, in `payload/evidence/numeric/`:

| record | field | meaning |
|---|---|---|
| `RESULT.json` | `peak_rss_kib` | measured whole-process peak |
| `RESULT.json` | `timings` | per-stage wall clock |
| `factor/RESULT.json` | `sparse.native_peak_bytes` | SPQR's own peak during the QR |
| `factor/RESULT.json` | `sparse.native_retained_bytes` | the R that survives the QR |
| `factor/RESULT.json` | `tail.buffer_bytes` | the q*q long-double tail buffer |
| `factor/MEMORY_ADMISSION.json` | `stage_forecast_bytes` | what the gate predicted |

`measured_peak: false` on the admission record: the gate admits on forecasts.
`sparse.native_peak_bytes` is the one measured number inside it.

## The decomposition

The stage order in `no_gp_same_qr_pilot.py::one_qr_session` is

```
runtime.factor(F)            -> native QR working set + native R
del F
physical_tail(qr.R, ni)      -> + H, a q*q long double buffer (16 B/entry)
tail_runtime.transform(H)    -> in place, same buffer
save_root_rows(root)         -> + the packed FP64 writer
del root, H
probes(packed)               -> R + packed root
recover_coordinates(qr.R)    -> R + four right-hand sides
```

The full sparse R is held across all of it, because the four backsolves come
last (`full_sparse_R_retained_until_backsolve: true`). During the tail only the
trace block of R is read; the interior part is resident and unused.

**seat 52** (q = 12828, body 98049 dofs, R 98049^2 with 267M nonzeros)

| component | GiB | live during |
|---|---|---|
| SPQR native peak | 6.155 | sparse_qr |
| SPQR native retained R | 3.993 | QR end -> backsolve end |
| long-double tail buffer q*q | 2.452 | tail_load -> storage |
| packed FP64 root writer | 0.613 | storage |

- QR stage: 6.155 + sparse input ~= 6.3
- tail stage: 3.993 + 2.452 + 0.613 = **7.058**
- measured `peak_rss_kib` = 7 337 024 KiB = **6.997 GiB**

**seat 85** (q = 23022, body 258021 dofs, R 258021^2 with 959M nonzeros)

- QR stage: 21.96 + input
- tail stage: 14.31 + 7.90 + 1.97 = **24.18**
- measured = **24.23 GiB**

Two seats an order of magnitude apart, and in both the tail-stage sum lands on
the measured peak within 1% while the QR-stage estimate falls below it. The peak
is the dense tail.

seat 52 stage timings (s): ordering 3.7, symbolic 2.0, sparse_qr 76.2,
tail_load_unscale 6.9, tail_transform 16.3, storage 4.2, screen 17.6,
four_rhs_backsolve 45.8.

## The forecast is 1.5x the measurement, consistently

`stage_forecast_bytes.native_numeric` against SPQR's measured `native_peak_bytes`,
for every label that has both:

| seat | q | body dofs | R nnz (M) | measured GiB | forecast GiB | ratio |
|---|---|---|---|---|---|---|
| 0010 | 5433 | 8967 | 23 | 0.67 | 1.1 | 1.59 |
| 0034 | 9765 | 17103 | 84 | 2.25 | 3.6 | 1.61 |
| 0013 | 11022 | 70788 | 212 | 4.88 | 7.7 | 1.57 |
| 0139 | 11304 | 89850 | 231 | 5.32 | 8.1 | 1.51 |
| 0009 | 13698 | 93078 | 288 | 6.62 | 10.0 | 1.51 |
| 0048 | 14586 | 141771 | 385 | 8.86 | 12.7 | 1.44 |
| 0026 | 15549 | 61746 | 286 | 6.56 | 11.0 | 1.68 |
| 0046 | 16164 | 242142 | 643 | 14.71 | 20.3 | 1.38 |
| 0016 | 17466 | 132246 | 480 | 11.00 | 16.1 | 1.46 |
| 0021 | 18342 | 51180 | 312 | 8.07 | 13.1 | 1.62 |
| 0124 | 19221 | 173559 | 579 | 13.19 | 17.8 | 1.35 |
| 0001 | 19446 | 126885 | 487 | 11.18 | 16.4 | 1.47 |
| 0007 | 20076 | 100644 | 474 | 10.88 | 17.4 | 1.60 |
| 0028 | 20886 | 183429 | 737 | 16.88 | 25.5 | 1.51 |
| 0032 | 21633 | 256698 | 899 | 20.61 | 32.0 | 1.55 |
| 0018 | 21636 | 194820 | 870 | 19.88 | 30.9 | 1.56 |
| 0085 | 23022 | 258021 | 959 | 21.96 | 33.7 | 1.54 |

median 1.54, range 1.35-1.68, no trend with q. On top of this the gate adds an
explicit `workspace_margin_bytes` of 3 GiB. seat 85 was admitted against a
forecast-with-margin of 36.7 GiB on a 56 GiB reservation, and then measured
24.2 GiB.

## What to change, in order of value per unit of risk

1. **Recalibrate the forecast against these 17 measured points.** Pure
   bookkeeping, no numerical change. At the measured scaling, 56 GiB reaches
   q ~ 33-35k, which is roughly 820-900 of the 1024 identities instead of 488.
   Note that 23127 is a hard upper bound on the *argument* in
   `no_gp_same_qr_pilot.py` (`not 3000 < q <= args.max_q <= 23127`), so raising
   the envelope requires re-versioning the pilot, not just a config edit.

2. **Release the interior of R across the dense tail.** The tail reads only the
   trace block; for seat 85 that is 1.97 GiB out of 14.31 GiB retained. Spilling
   the interior and restoring it for the four backsolves moves the tail stage to
   roughly (trace block + tail buffer + packed) ~ 11.9 GiB and makes the QR the
   peak again: 24.2 -> ~22.0 GiB for seat 85. Costs one write and one read of
   ~15 GB. Changes `complete_sparse_R_written`, which is a recorded contract
   property, so it needs registering rather than just editing.

3. **Substructuring**, if 1 and 2 together still do not reach 1024. Partition
   cells into groups, eliminate each group's private interior by QR, stack the
   reduced roots, and finish on the trace. Orthogonal transforms preserve energy
   exactly, so this keeps the no-subtraction property that the whole formulation
   exists to protect. Peak becomes the larger of one group and the final trace
   triangle.

## Open

- An instrumented rerun of seat 52 (50 ms whole-tree RSS sampling, stage markers
  from artifact creation) is in flight to confirm the stage attribution
  directly. Its symbolic analysis is bit-identical to production (same fronts,
  same `stack_entries`, same `flop_upper_bound`), so the memory plan is the same
  run; only the arithmetic is slower.
- That rerun is ~25x slower than production at identical flop count, which means
  production bound a different BLAS than this environment does.
  `M2_NATIVE_RUNTIME_R1/lib` ships a reference `libblas.so.3` alongside
  `libspqr.so.2`, and `libmkl_rt.so.3` lives in a different directory, so the
  QR's throughput depends on `LD_LIBRARY_PATH` order. Worth pinning explicitly;
  it is a throughput lever, not a memory one.
