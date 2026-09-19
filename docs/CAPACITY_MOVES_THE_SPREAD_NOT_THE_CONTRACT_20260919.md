# Capacity moves the spread, not the contract; an ensemble recovers it

2026-09-19. Box-only cells, the same 12 evaluation seats (8 unseen). Data in
`docs/data/wide_box_20260919/`.

**Correction first.** Earlier notes today compared against `S4_60K` as "the plain-loss baseline".
`S4_60K` is the tilt+tail loss arm (9.4 % worst compliance). The plain-loss baseline is
`MULTI_AUGMENT` (48 presented / 8 unseen, `ACCEPT_MULTI`): worst compliance 2.087 % (seat 100186),
0.17-1.7 % elsewhere, g 2.4-2.9 with three hard seats at 4.1 / 18.4 / 41.3.

## WIDE_BOX: width 1536 / state 256 / depth 5, 31.0 M parameters (3.2x), 56 seats, 60k steps

| | MULTI_AUGMENT | WIDE_BOX |
|---|---|---|
| final loss | 0.0099 | 0.0035 |
| g, typical / hard seats | 2.4-2.9 / 4.1, 18.4, 41.3 | 2.2-2.5 / 5.4, 8.9, 11.5 |
| factor_relative | 0.028-0.046 | 0.022-0.042 |
| worst assembled compliance | 2.09 % (1 seat > 3 %) | 3.97 % (3 seats > 3 %: 244, 100151, 100186 s) |
| face-load signed mean | mixed sign, -1.4..+1.1 % | positive on 10/12 seats, +0.3..+4.7 % |

Every operator metric improves; the contract metric does not. The bias/halfvar expansion
(`soft_diag` on 244, 347, 100054, 403; `SOFTDIAG_WIDE/*/SOFT_DIAG.json`) shows why:

| seat, load | bias base -> wide | halfvar base -> wide | measured % base -> wide |
|---|---|---|---|
| 244 axial_x | +0.025 -> -0.009 | 0.028 -> 0.020 | 0.01 -> 2.72 |
| 244 shear_z | +0.019 -> -0.019 | 0.024 -> 0.019 | 0.37 -> 3.79 |
| 347 axial_z | +0.020 -> +0.002 | 0.023 -> 0.018 | -0.04 -> 1.36 |
| 100054 axial_y | +0.031 -> +0.034 | 0.036 -> 0.026 | -0.18 -> -1.25 |
| 403 shear_y | +0.020 -> +0.012 | 0.023 -> 0.021 | 0.30 -> 0.88 |

The wide model cuts the spread (halfvar) by about a quarter, but the systematic over-stiffness
(bias) that used to cancel it shrinks more, to zero or past it. The net error, which is
`-bias + halfvar` to second order, is then the halfvar itself, 1.5-2 %, plus whatever negative
bias remains. The baseline's 0.2-1 % was a cancellation; the wide model's 1-4 % is the honest
spread. This is the prediction of `THE_LEVERAGE_IS_NOT_THE_LOSS_20260919.md`, now measured.

## Ensemble: the mean of the two models' predicted M_q, no training

| seat | worst c: base / wide / ens | worst s: base / wide / ens |
|---|---|---|
| 120 | 1.721 / 1.248 / **0.569** | 2.459 / 1.723 / **0.890** |
| 244 | 1.002 / 3.790 / 2.068 | 1.403 / 4.380 / 2.663 |
| 347 | 0.174 / 2.014 / 0.928 | 0.589 / 2.320 / 1.137 |
| 403 | 1.055 / 0.877 / **0.590** | 1.090 / 1.699 / 1.159 |
| 941 | 1.149 / 0.883 / **0.660** | 2.498 / 1.730 / **1.997** |
| 100051 | 1.412 / 1.431 / **1.188** | 1.610 / 1.633 / **1.478** |
| 100054 | 0.682 / 1.246 / 0.850 | 0.974 / 2.194 / 1.565 |
| 100077 | 1.545 / 1.614 / **1.442** | 2.090 / 1.792 / 1.935 |
| 100102 | 1.164 / 1.200 / **0.799** | 1.261 / 1.720 / **1.121** |
| 100151 | 1.361 / 3.967 / 2.120 | 2.304 / 4.487 / 2.339 |
| 100167 | 0.615 / 2.173 / 1.130 | 0.742 / 2.246 / 1.139 |
| 100186 | 2.087 / 2.375 / 2.253 | 7.789 / 7.257 / 7.552 |
| worst | 2.09 / 3.97 / 2.25 | 7.79 / 7.26 / 7.55 |

The ensemble beats the wide model on every seat, beats the baseline on 6 of 12 for compliance,
and keeps 11 of 12 inside the contract; the face-load median falls to 0.4-1.4 % on all seats but
244 (2.07 %). Seat 100186's sensitivity (7.3-7.8 %) is the same in all three runs: an
energy-split error on that seat, not model variance, and the one item still outside the contract.

## What this changes

* Capacity is not the lever for the contract metric on box cells; the spread is, and the
  spread's transferable measure is halfvar (rms of the weighted log spectrum), 0.24 -> 0.20.
* Averaging independently trained predictors cuts the spread cheaply. Two models cost two
  forward passes; the budget question is inference time, not training.
* A per-model scalar calibration cannot fix this: the residual bias has both signs across seats
  (244 negative, 100054 positive).
* Seat 100186's sensitivity is a separate, structural item.
