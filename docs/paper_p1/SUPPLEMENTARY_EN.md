# Supplementary results

## R1. Predictor and geometry key

| Scientific label | Numerical role | Archived run identifier |
| --- | --- | --- |
| P0 | Earlier uncorrected predictor | c_oh |
| B | Baseline for fixed-weight corrections | v2L1 |
| C | Continued uncorrected predictor | A0_ctrl |
| S8 | Separate continued weights with eight smoothing steps | A2_tail8 |
| D | Earlier predictor for deployment costs | c_ctrl |

The S8 predictor has its own learned weights. Applying eight steps to B in the fixed-weight study is a distinct comparison.

| Cell label | Geometry stratum | Archived geometry identifier |
| --- | --- | --- |
| U1 | Uncut | fresh_val_2000_full |
| U2 | Uncut | fresh_val_2001_full |
| M1 | Moderately cut | fresh_val_2003_d1_v1 |
| H1 | Heavily cut | fresh_val_2005_d1_v0 |
| M2 | Moderately cut | fresh_val_2006_d0_v1 |
| H2 | Heavily cut | fresh_val_2010_d0_v0 |

The x/y suffix identifies the neighbouring-cell configuration. The deployment geometries use the G1–G4 labels in Figure 11 and Figure S05. The complete timing tables retain the corresponding abbreviated geometry identifiers for lookup.

| Benchmark label | Abbreviated geometry | Archived geometry identifier |
| --- | --- | --- |
| G1 | 0020-r2 | fresh_train_0020_cover01_r2 |
| G2 | 0020-r1 | fresh_train_0020_cover01_r1 |
| G3 | 0020-FULL | fresh_train_0020_full |
| G4 | 0007-FULL | fresh_train_0007_full |

## R2. Further diagnostic observations

The element-group diagnostic for M1 distributes sensitivity-error contributions across the cut material. Elements with volume fraction below 0.1 account for 15.5% of the absolute contributions, while the 0.5–0.999 group accounts for 49.5%. These groups are disjoint; overlapping retained-node and weak-support classifications are not additive groups. Figure S04 gives the corresponding element populations and contribution shares.

The role of the initial field is particularly clear in M1: after 32 smoothing steps, the dimensionless mean energy excess is 0.029452 from the learned field and 188.76 from a zero interior field, under identical retained displacements. These values are ratios, not percentages.

For the same cell, coarse correction followed by one eight-step smoothing stage gives 0.22991% mean energy excess with the trilinear space. Adding the pre-smoothing stage gives 0.18643%. With eight steps on each side, the quadratic coarse space gives 0.09001%, and linear partition-of-unity enrichment gives 0.036479%. The enriched representation retains 22,404 coefficient columns; this count is not a certified independent-space dimension (Appendix F.1). Table ST04 includes both loading classes and all recorded spaces.

For the baseline predictor in M1/x under the target-face y load, the retained-displacement error is 14.198% in the exact Schur norm and the energy excess at the exact retained displacement is 14.778%. These values accompany the full, field-only and solution-only sensitivity errors; the latter are separate replacements, not additive scalar contributions.

Geometry counts refer to geometries with an available direction-class mean. The aggregate records do not retain the realised direction count for every geometry and class; reported percentiles summarise their respective recorded direction banks.

## Table ST01. Identity-view energy excess by direction class

Entries are geometry-equal mean / 90th percentile / maximum of geometry-level direction means, in percent. The maximum is not a worst individual direction. An em dash denotes a class absent from that model’s result.

| Class | Geometries per evaluated arm | P0 | B | C | S8 |
| --- | --- | --- | --- | --- | --- |
| force | 80 | 7.387 / 19.508 / 109.813 | 5.036 / 12.038 / 70.122 | 4.607 / 10.750 / 62.465 | 0.672 / 1.887 / 4.398 |
| support | 80 | 8.078 / 13.959 / 153.172 | 5.332 / 9.811 / 99.857 | 4.878 / 9.493 / 90.741 | 0.761 / 1.864 / 3.408 |
| face | 80 | 3.203 / 6.296 / 33.992 | 2.486 / 5.209 / 22.139 | 2.233 / 4.045 / 17.635 | 0.218 / 0.435 / 2.638 |
| macro | 80 | 0.938 / 1.533 / 2.669 | 0.842 / 1.448 / 2.636 | 0.827 / 1.446 / 2.584 | 0.255 / 0.487 / 1.148 |
| grf | 80 | 2.125 / 3.191 / 4.747 | 2.038 / 3.104 / 4.582 | 2.009 / 3.075 / 4.550 | 0.311 / 0.709 / 1.172 |
| force_c | 20 | — | 6.624 / 14.185 / 35.202 | 5.788 / 12.731 / 24.848 | 1.428 / 3.982 / 4.723 |
| face_c | 20 | — | 3.838 / 4.553 / 36.337 | 3.160 / 4.184 / 25.525 | 0.510 / 1.187 / 2.241 |
| support_k | 19 | — | 3.350 / 8.702 / 10.045 | 3.177 / 8.032 / 8.873 | 1.062 / 2.784 / 3.411 |
| glued | 15 | — | 6.647 / 11.149 / 38.128 | 6.712 / 10.144 / 42.030 | 1.475 / 3.843 / 3.986 |


### ST01b. Force/support geometry-stratum means (%)

| Stratum | Geometries | P0 force/support | B force/support | C force/support | S8 force/support |
| --- | --- | --- | --- | --- | --- |
| FULL | 20 | 1.052 / 1.815 | 0.908 / 1.413 | 0.879 / 1.332 | 0.110 / 0.278 |
| Light cut (v2) | 20 | 4.593 / 4.142 | 3.239 / 3.049 | 3.021 / 2.832 | 0.665 / 0.777 |
| Middle cut (v1) | 20 | 6.166 / 5.265 | 4.720 / 3.930 | 4.506 / 3.810 | 0.912 / 0.856 |
| Heavy cut (v0) | 20 | 17.738 / 21.089 | 11.278 / 12.938 | 10.021 / 11.540 | 0.999 / 1.133 |


### ST01c. View dependence: identity / view 17 means (%)

| Class | P0 | B |
| --- | --- | --- |
| force | 7.387 / 8.107 | 5.036 / 5.271 |
| support | 8.078 / 8.678 | 5.332 / 5.598 |
| face | 3.203 / 3.223 | 2.486 / 2.417 |
| macro | 0.938 / 1.011 | 0.842 / 0.890 |
| grf | 2.125 / 2.240 | 2.038 / 2.119 |
| force_c | — | 6.624 / 7.493 |
| face_c | — | 3.838 / 4.249 |
| support_k | — | 3.350 / 3.760 |
| glued | — | 6.647 / 7.207 |

## Table ST02. Same-trace sensitivity and spectral diagnostics

### ST02a. Sensitivity estimates

Energy and sensitivity errors are directional means (%). The first-order share is the Frobenius norm of the linear error array divided by the sum of the Frobenius norms of the linear and quadratic arrays (%); each array includes all eight corners and all evaluated directions in that cell and class.

| Arm | Cell | Class | Mean energy excess | Mean sensitivity error | 90th-percentile sensitivity error | First-order share |
| --- | --- | --- | --- | --- | --- | --- |
| B | U1 | force_c | 1.270 | 0.664 | 0.778 | 42.887 |
| B | U1 | force | 0.742 | 1.018 | 1.449 | 55.124 |
| B | U2 | force_c | 0.402 | 0.588 | 0.693 | 35.658 |
| B | U2 | force | 0.506 | 1.008 | 1.772 | 71.565 |
| B | M1 | force_c | 13.512 | 14.128 | 19.184 | 7.164 |
| B | M1 | force | 7.156 | 4.822 | 10.493 | 23.516 |
| B | H1 | force_c | 1.814 | 1.909 | 3.221 | 28.403 |
| B | H1 | force | 1.842 | 0.923 | 1.565 | 35.199 |
| B | M2 | force_c | 3.643 | 5.159 | 7.351 | 9.339 |
| B | M2 | force | 2.175 | 1.103 | 2.019 | 41.390 |
| B | H2 | force_c | 34.954 | 75.072 | 156.803 | 0.786 |
| B | H2 | force | 19.507 | 13.649 | 38.851 | 36.322 |
| C | U1 | force_c | 1.165 | 0.937 | 1.138 | 35.750 |
| C | U1 | force | 0.779 | 0.766 | 0.981 | 59.185 |
| C | U2 | force_c | 0.397 | 0.560 | 0.658 | 35.988 |
| C | U2 | force | 0.515 | 1.062 | 1.773 | 72.929 |
| C | M1 | force_c | 13.009 | 13.275 | 18.119 | 7.456 |
| C | M1 | force | 6.841 | 4.442 | 9.663 | 26.413 |
| C | H1 | force_c | 1.699 | 1.894 | 3.223 | 28.046 |
| C | H1 | force | 1.877 | 1.013 | 1.630 | 34.751 |
| C | M2 | force_c | 3.988 | 5.432 | 7.485 | 9.548 |
| C | M2 | force | 2.105 | 1.357 | 2.192 | 49.454 |


### ST02b. Energy fractions in the lowest 200 generalised interior modes

| Cell | Class | Error fraction: mean / p10 / p90 (%) | Exact-field fraction: mean / p10 / p90 (%) |
| --- | --- | --- | --- |
| U1 | force_c | 24.019 / 20.530 / 27.123 | 8.498 / 8.230 / 8.755 |
| U1 | force | 5.043 / 3.486 / 7.430 | 7.977 / 6.906 / 8.860 |
| M1 | force_c | 24.452 / 20.027 / 28.532 | 4.870 / 4.574 / 5.161 |
| M1 | force | 22.378 / 17.917 / 26.173 | 6.644 / 4.770 / 9.198 |
| M2 | force_c | 18.399 / 14.496 / 22.067 | 4.620 / 3.887 / 5.957 |
| M2 | force | 11.411 / 7.955 / 16.414 | 6.578 / 4.832 / 8.138 |
| H2 | force_c | 15.216 / 12.681 / 19.023 | 4.705 / 1.983 / 6.199 |
| H2 | force | 21.837 / 13.953 / 34.960 | 4.877 / 2.383 / 6.832 |


The two fractions use their respective internal-energy denominators. Their cumulative curves are shown in Figure 6. The archived eigenvalues can also be compared with the interval used in the separate fixed-weight smoothing diagnostic:

| Cell | \(\lambda_1\) | \(\lambda_{200}\) | Diagnostic lower endpoint \(a=b/30\) | Recorded modes below \(a\), out of 200 |
| --- | --- | --- | --- | --- |
| U1 | 0.000402235 | 0.00979835 | 0.174937 | 200 |
| M1 | 0.000574199 | 0.0112575 | 0.173294 | 200 |
| M2 | 0.000935594 | 0.0117352 | 0.174182 | 200 |
| H2 | 0.0213148 | 0.416328 | 0.137632 | 66 |

The spectral and smoothing diagnostics use the same cell identifiers, predictor and specified discrete construction, with matching internal-coordinate counts. The intervals come from the separate smoothing runs: the recorded upper endpoint includes the 1.05 safety factor, and \(a=b/30\). These estimates differ from those of the correction wrapper and are not certified spectral bounds. Individual directional attenuation and stiffness-content hashes were not recorded, so the comparison supports a cell-level spectral interpretation.

## Table ST03. Complete recorded smoothing cases

The retained trace is identical for network and zero interior initialisations. Entries are mean [90th percentile] directional errors (%). A zero interior start sets only the internal displacement to zero. The same five cells and both direction classes are included.

### ST03a. Energy excess

| Cell | Class | Network, k=0 | Network, k=8 | Network, k=32 | Zero interior, k=32 |
| --- | --- | --- | --- | --- | --- |
| U1 | force_c | 1.2704 [1.4799] | 0.41971 [0.51235] | 0.27998 [0.35499] | 2680.9 [3717.6] |
| U1 | force | 0.74211 [0.78789] | 0.1063 [0.12755] | 0.049733 [0.066014] | 409.72 [564.75] |
| M1 | force_c | 13.512 [19.634] | 4.685 [7.5373] | 2.9452 [4.74] | 18876 [26535] |
| M1 | force | 7.1563 [10.711] | 2.3037 [3.4374] | 1.4523 [2.1642] | 10417 [14953] |
| H1 | force_c | 1.814 [2.5294] | 0.25081 [0.33593] | 0.059403 [0.08861] | 186.98 [316.28] |
| H1 | force | 1.842 [3.3328] | 0.19516 [0.42805] | 0.047822 [0.11395] | 386.19 [629.7] |
| M2 | force_c | 3.6428 [5.158] | 1.0382 [1.49] | 0.61088 [0.97633] | 5434.1 [8866.3] |
| M2 | force | 2.1747 [3.2578] | 0.46478 [0.71357] | 0.2508 [0.39958] | 2279.1 [3703.7] |
| H2 | force_c | 34.954 [56.196] | 0.20515 [0.32813] | 0.0033313 [0.006318] | 0.42123 [0.57258] |
| H2 | force | 19.507 [36.537] | 0.22768 [0.34841] | 0.0074193 [0.011726] | 0.20928 [0.34236] |


### ST03b. Field-based sensitivity error

| Cell | Class | Network, k=0 | Network, k=8 | Network, k=32 | Zero interior, k=32 |
| --- | --- | --- | --- | --- | --- |
| U1 | force_c | 0.6639 [0.77847] | 0.52592 [0.69671] | 0.45509 [0.63825] | 1671.3 [2121.5] |
| U1 | force | 1.0177 [1.4489] | 0.23011 [0.35671] | 0.14741 [0.19763] | 93.161 [142.59] |
| M1 | force_c | 14.128 [19.184] | 2.7552 [4.1122] | 1.6339 [2.4573] | 10302 [14185] |
| M1 | force | 4.8221 [10.493] | 0.78839 [1.5521] | 0.5492 [0.84882] | 3203.8 [5893.7] |
| H1 | force_c | 1.9092 [3.2207] | 0.26139 [0.48401] | 0.14959 [0.30252] | 163.15 [223.26] |
| H1 | force | 0.9233 [1.5645] | 0.14313 [0.24897] | 0.051681 [0.082265] | 114.31 [197.95] |
| M2 | force_c | 5.1589 [7.3505] | 0.83675 [1.2052] | 0.5753 [0.78197] | 2952.9 [4584.9] |
| M2 | force | 1.1026 [2.0194] | 0.15273 [0.26638] | 0.08977 [0.16216] | 303.46 [686.58] |
| H2 | force_c | 75.072 [156.8] | 0.77195 [1.4896] | 0.010708 [0.023337] | 1.9551 [2.8917] |
| H2 | force | 13.649 [38.851] | 0.75099 [1.4264] | 0.073808 [0.14745] | 0.69083 [1.2398] |

## Table ST04. Interior coarse-space and smoothing comparisons

### ST04a. Eight steps per smoothing stage

Entries are mean [90th percentile] energy excess (%). C denotes one coarse correction and T one eight-step smoothing stage. All network rows use the same B checkpoint; the zero interior reference uses the same retained values. The size column counts coarse coefficient columns remaining after the support and diagonal-energy screens. For the PU rows, structural dependencies in the generating functions and the absence of an archived rank/solve-accuracy check prevent interpreting this count as an independent-space dimension; see Appendix F.1.

| Cell | Class | Coarse space | Surviving coarse coefficient columns | Net + C | Net + C + T | Net + T + C + T | Zero + T + C + T |
| --- | --- | --- | --- | --- | --- | --- | --- |
| U1 | force_c | Q1_9 | 1,728 | 0.87589 [1.0044] | 0.13597 [0.15856] | 0.096621 [0.11469] | 582.87 [746.75] |
| U1 | force_c | PU_9 | 6,912 | 0.53246 [0.60943] | 0.02491 [0.027969] | 0.015406 [0.017553] | 93.269 [108.72] |
| U1 | force_c | Q1_17 | 7,950 | 0.5889 [0.67776] | 0.037909 [0.043611] | 0.02671 [0.031486] | 133.13 [158.19] |
| U1 | force_c | Q2_17 | 10,452 | 0.45492 [0.52339] | 0.018783 [0.021303] | 0.012546 [0.014527] | 89.035 [103.75] |
| U1 | force_c | PU_17 | 31,800 | 0.27807 [0.32057] | 0.0074368 [0.0088953] | 0.0065266 [0.0079824] | 12.104 [15.996] |
| U1 | force_c | Q1_33 | 40,983 | 0.29109 [0.33079] | 0.0090665 [0.010395] | 0.0078425 [0.0093791] | 21.384 [27.345] |
| U1 | force | Q1_9 | 1,728 | 0.67304 [0.70534] | 0.066091 [0.072088] | 0.038737 [0.043165] | 89.796 [114.12] |
| U1 | force | PU_9 | 6,912 | 0.60974 [0.63812] | 0.050323 [0.053281] | 0.027803 [0.03114] | 17.806 [20.626] |
| U1 | force | Q1_17 | 7,950 | 0.60874 [0.63931] | 0.050277 [0.052953] | 0.027829 [0.031141] | 22.456 [27.314] |
| U1 | force | Q2_17 | 10,452 | 0.58831 [0.61599] | 0.048835 [0.051937] | 0.027186 [0.030536] | 17.159 [19.758] |
| U1 | force | PU_17 | 31,800 | 0.50662 [0.54204] | 0.039204 [0.043384] | 0.021406 [0.025149] | 4.4879 [5.6473] |
| U1 | force | Q1_33 | 40,983 | 0.45524 [0.50058] | 0.031736 [0.035242] | 0.014305 [0.016225] | 4.2175 [5.3685] |
| M1 | force_c | Q1_9 | 1,275 | 7.439 [10.515] | 1.0016 [1.4953] | 0.8125 [1.2317] | 4530.7 [6359.6] |
| M1 | force_c | PU_9 | 5,100 | 4.2489 [5.8755] | 0.16633 [0.22693] | 0.11402 [0.16383] | 729.23 [1070.1] |
| M1 | force_c | Q1_17 | 5,601 | 4.6803 [6.5368] | 0.22991 [0.33235] | 0.18643 [0.27983] | 1097.2 [1600.1] |
| M1 | force_c | Q2_17 | 7,401 | 3.5958 [4.9894] | 0.12522 [0.17305] | 0.09001 [0.12817] | 631.41 [927.81] |
| M1 | force_c | PU_17 | 22,404 | 2.0056 [2.7628] | 0.04841 [0.069908] | 0.036479 [0.051356] | 163.6 [238.41] |
| M1 | force_c | Q1_33 | 27,207 | 1.9826 [2.739] | 0.051622 [0.071595] | 0.047665 [0.069164] | 254.92 [370.44] |
| M1 | force | Q1_9 | 1,275 | 4.17 [6.2314] | 0.52699 [0.81966] | 0.41642 [0.6526] | 2344.7 [3181.6] |
| M1 | force | PU_9 | 5,100 | 2.5741 [3.7062] | 0.12599 [0.1791] | 0.078984 [0.1172] | 384.02 [570.53] |
| M1 | force | Q1_17 | 5,601 | 2.8067 [4.0798] | 0.15789 [0.23819] | 0.11392 [0.1776] | 558.09 [807.53] |
| M1 | force | Q2_17 | 7,401 | 2.2196 [3.1669] | 0.10132 [0.14109] | 0.06647 [0.097108] | 341.7 [492.96] |
| M1 | force | PU_17 | 22,404 | 1.3478 [1.8254] | 0.05649 [0.072894] | 0.035344 [0.049353] | 72.462 [114.85] |
| M1 | force | Q1_33 | 27,207 | 1.2562 [1.7009] | 0.045462 [0.056742] | 0.033306 [0.043036] | 116.6 [179.66] |
| M2 | force_c | Q1_9 | 984 | 2.1915 [3.1082] | 0.20405 [0.29624] | 0.13504 [0.1968] | 1617.3 [2562.3] |
| M2 | force_c | PU_9 | 3,936 | 1.3433 [1.9306] | 0.041654 [0.05888] | 0.021112 [0.029566] | 276.48 [424.28] |
| M2 | force_c | Q1_17 | 4,557 | 1.2521 [1.814] | 0.04538 [0.064972] | 0.027309 [0.038377] | 324.11 [506.68] |
| M2 | force_c | Q2_17 | 5,751 | 1.1722 [1.6942] | 0.032985 [0.047125] | 0.01695 [0.024094] | 238.17 [364.67] |
| M2 | force_c | PU_17 | 18,228 | 0.5528 [0.79721] | 0.012062 [0.017306] | 0.0075972 [0.010715] | 30.656 [44.479] |
| M2 | force_c | Q1_33 | 23,199 | 0.54679 [0.78933] | 0.011767 [0.017181] | 0.0087034 [0.01215] | 53.556 [79.929] |
| M2 | force | Q1_9 | 984 | 1.5947 [2.4459] | 0.16628 [0.28435] | 0.10121 [0.17679] | 552.63 [894.43] |
| M2 | force | PU_9 | 3,936 | 1.1993 [1.7578] | 0.095141 [0.15843] | 0.055364 [0.0926] | 90.85 [143.8] |
| M2 | force | Q1_17 | 4,557 | 1.1608 [1.7224] | 0.10256 [0.169] | 0.061274 [0.10242] | 106.64 [166.08] |
| M2 | force | Q2_17 | 5,751 | 1.0837 [1.6123] | 0.087778 [0.1467] | 0.051133 [0.085148] | 79.888 [127.32] |
| M2 | force | PU_17 | 18,228 | 0.71966 [1.0802] | 0.077675 [0.12553] | 0.044293 [0.07325] | 10.536 [17.558] |
| M2 | force | Q1_33 | 23,199 | 0.61821 [0.89137] | 0.049731 [0.074383] | 0.025119 [0.040226] | 17.939 [28.357] |


### ST04b. Shorter two-sided smoothing sequences

| Cell | Class | Steps on each side | Q1_9 mean [p90] (%) | Q1_17 mean [p90] (%) | PU_9 mean [p90] (%) |
| --- | --- | --- | --- | --- | --- |
| U1 | force_c | 2 | 0.28124 [0.33175] | 0.13385 [0.15343] | 0.11137 [0.12737] |
| U1 | force | 2 | 0.16983 [0.17926] | 0.14219 [0.14565] | 0.14314 [0.14567] |
| M1 | force_c | 2 | 2.3292 [3.4188] | 1.0178 [1.4665] | 0.86128 [1.2212] |
| M1 | force | 2 | 1.2227 [1.8318] | 0.59017 [0.86672] | 0.51585 [0.73952] |
| U1 | force_c | 4 | 0.15339 [0.17997] | 0.054177 [0.062172] | 0.04076 [0.046451] |
| U1 | force | 4 | 0.07691 [0.084069] | 0.059814 [0.062339] | 0.060538 [0.06364] |
| M1 | force_c | 4 | 1.328 [1.9919] | 0.42211 [0.61995] | 0.32643 [0.47049] |
| M1 | force | 4 | 0.68988 [1.0641] | 0.2529 [0.38508] | 0.20707 [0.30643] |

## Table ST05. Assembly sensitivity replacement diagnostics

Maximum errors (%) over the six face loads defining the joint criterion; maxima in separate columns may occur at different loads. Full, field-only, and solution-only values are separate nonlinear replacement diagnostics. The trace error is the relative exact-Schur norm, not its square.

| Arm | Cell | Configuration | Full sensitivity error | Field-only error | Solution-only error | Trace error | Energy excess at exact trace |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S8 | U1 | x | 3.894 | 2.460 | 1.556 | 1.544 | 1.108 |
| S8 | U1 | y | 3.747 | 2.376 | 1.527 | 1.606 | 1.047 |
| S8 | M1 | x | 5.829 | 3.320 | 7.839 | 6.580 | 5.101 |
| S8 | M1 | y | 5.555 | 3.094 | 7.623 | 6.402 | 4.940 |
| S8 | M2 | x | 1.887 | 0.727 | 2.421 | 2.706 | 1.137 |
| B | U1 | x | 5.830 | 2.674 | 3.305 | 2.413 | 2.312 |
| B | U1 | y | 5.390 | 2.603 | 2.966 | 2.322 | 2.056 |
| B | M1 | x | 12.153 | 15.602 | 19.269 | 14.198 | 14.778 |
| B | M1 | y | 11.439 | 15.532 | 18.966 | 14.329 | 14.671 |
| B | M2 | x | 2.539 | 3.371 | 4.841 | 4.434 | 2.352 |

## Table ST06. Cut-traction responses outside the six-load criterion

Maximum relative errors (%) across the three cut-surface traction directions. Sensitivity maxima include both cells. The target is learned and the neighbour exact.

| Arm | Cell | Configuration | Compliance error (%) | Sensitivity error (%) | Status |
| --- | --- | --- | --- | --- | --- |
| B | M1 | x | 2.8796 | 9.9241 | Recorded |
| B | H1 | x | 0.44546 | 2.0656 | Recorded |
| B | H1 | y | 0.65052 | 1.275 | Recorded |
| B | M2 | x | 0.55957 | 3.1552 | Recorded |
| C | M1 | x | 2.7679 | 9.8268 | Recorded |
| C | M1 | y | 6.8704 | 14.053 | Recorded |
| C | H1 | x | 0.39921 | 1.825 | Recorded |
| C | H1 | y | 0.61261 | 1.2 | Recorded |
| C | M2 | x | 0.57448 | 3.2516 | Recorded |
| S8 | M1 | x | 0.91215 | 4.4449 | Recorded |
| S8 | M1 | y | 2.8617 | 6.76 | Recorded |
| S8 | H1 | x | 0.077686 | 0.83947 | Recorded |
| S8 | H1 | y | 0.15791 | 0.37196 | Recorded |
| S8 | M2 | x | 0.20034 | 2.2065 | Recorded |

## Table ST07. Bernstein restriction of the box-face trace

The target H1 and its continuous-thickness neighbour are assembled in configuration x using exact cell operators. Degree r applies to every box face; cut-band coefficients retain their identity representation. Controlled DOFs include these unrestricted cut-band coefficients. All errors are maxima in the stated load set (%); sensitivity columns refer to the target-cell vector.

| r | Controlled DOFs | Target-face compliance (3 loads) | Target-face sensitivity (3 loads) | Six-load compliance (6 loads) | Six-load sensitivity (6 loads) | All-load compliance (9 loads) | All-load sensitivity (9 loads) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 12,888 | 81.713 | 87.320 | 81.713 | 333.566 | 81.713 | 333.566 |
| 2 | 12,939 | 37.528 | 29.960 | 41.962 | 188.527 | 41.962 | 188.527 |
| 3 | 13,026 | 21.660 | 18.637 | 30.364 | 340.008 | 30.364 | 340.008 |
| 5 | 13,308 | 2.784 | 2.890 | 3.860 | 171.306 | 3.860 | 171.306 |
| 8 | 14,001 | 0.609 | 1.520 | 0.735 | 64.041 | 0.778 | 64.041 |


The supported system on the full retained space has 32,991 free DOFs. Six-load and all-load maxima coincide only when the maximizing load belongs to both sets.

## Table ST08. Additional single-cell deployment measurements

### ST08a. Memory and fused-operator consistency

| Cell | K storage (GiB) | fp64 factor delta (GiB) | fp32 factor delta (GiB) | Learned state delta (GiB) | Sparse freeze allocation (GiB) | Fused freeze allocation (GiB) | Fused/sparse action difference |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0020-r2 | 1.394 | 4.748 | 0.340 | 0.697 | 0.626 | 0.173 | 2.496e-05 |
| 0020-r1 | 0.505 | 0.164 | 0.105 | 0.270 | 0.226 | 0.063 | 2.174e-05 |
| 0020-FULL | 2.324 | 5.711 | 0.555 | 1.127 | 1.039 | 0.285 | 3.596e-05 |
| 0007-FULL | 2.707 | 5.074 | 0.652 | 1.371 | 1.231 | 0.341 | 3.297e-05 |


Factor and state deltas use free-device-memory changes; freeze allocations use PyTorch allocated-memory changes. These measures have distinct allocation definitions.

### ST08b. Action accuracy and residual-estimate cost

| Cell | Mean energy ratio | Minimum energy ratio | Maximum energy ratio | Residual-estimate cost, m=0 (ms) | Residual-estimate cost, m=8 (ms) |
| --- | --- | --- | --- | --- | --- |
| 0020-r2 | 1.014780 | 1.007903 | 1.024783 | 15.36 | 95.53 |
| 0020-r1 | 1.018094 | 1.010527 | 1.027321 | 5.65 | 34.20 |
| 0020-FULL | 1.014432 | 1.006684 | 1.025316 | 25.58 | 161.48 |
| 0007-FULL | 1.008199 | 1.003768 | 1.014849 | 30.12 | 195.43 |


Energy ratios use 64 benchmark directions; residual-estimate timings use a batch of 16. These are D diagnostics.

## Table ST09. All recorded preconditioned assembly solves

Times are solve times in seconds; the wall-clock cap is 300 s per run. “Recursive tol.” denotes only the original recursive-residual stopping condition. The explicit residual is shown separately and uses the same operator as the run.

| Assembly | Operator | Preconditioner | Iterations | Solve time (s) | Recursive residual | Recomputed residual | Max. compliance error (%) | Stopping status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pair | exact | jacobi | 1839 | 300.16 | 7.954e-04 | 7.954e-04 | 1.474e-06 | Time limit |
| Pair | exact | kpp | 486 | 80.04 | 9.790e-09 | 9.789e-09 | 8.789e-11 | Recursive tol. |
| Pair | exact | add:jac:q1r | 871 | 142.41 | 9.755e-09 | 9.755e-09 | 3.022e-11 | Recursive tol. |
| Pair | exact | bnn:kpp:q1r | 169 | 27.94 | 8.430e-09 | 8.430e-09 | 0.000e+00 | Recursive tol. |
| Pair | exact | defl:kpp:q1r | 169 | 27.90 | 8.460e-09 | 8.460e-09 | 1.101e-10 | Recursive tol. |
| Pair | learned | jacobi | 2520 | 263.14 | 9.732e-09 | 1.985e-02 | 3.409e+00 | Recursive tol. |
| Pair | learned | kpp | 742 | 78.65 | 9.647e-09 | 2.035e-02 | 3.409e+00 | Recursive tol. |
| Pair | learned | add:jac:q1r | 873 | 91.33 | 9.963e-09 | 2.023e-02 | 3.409e+00 | Recursive tol. |
| Pair | learned | bnn:kpp:q1r | 204 | 21.74 | 9.921e-09 | 2.075e-02 | 3.409e+00 | Recursive tol. |
| Pair | learned | defl:kpp:q1r | 2821 | 300.06 | 1.683e+04 | 1.683e+04 | 3.171e+03 | Time limit |
| 2×2×2 | exact | jacobi | 1130 | 210.77 | 9.990e-09 | 9.990e-09 | 1.789e-11 | Recursive tol. |
| 2×2×2 | exact | kpp | 442 | 82.98 | 9.800e-09 | 9.800e-09 | 0.000e+00 | Recursive tol. |
| 2×2×2 | exact | add:jac:q1r | 349 | 65.54 | 9.843e-09 | 9.843e-09 | 6.017e-12 | Recursive tol. |
| 2×2×2 | exact | bnn:kpp:q1r | 119 | 22.71 | 9.997e-09 | 9.997e-09 | 8.518e-11 | Recursive tol. |
| 2×2×2 | exact | defl:kpp:q1r | 119 | 22.51 | 9.877e-09 | 9.877e-09 | 1.143e-11 | Recursive tol. |
| 2×2×2 | learned | jacobi | 685 | 300.31 | 5.821e-05 | 7.117e-03 | 1.882e+00 | Time limit |
| 2×2×2 | learned | kpp | 504 | 221.74 | 9.614e-09 | 7.267e-03 | 1.881e+00 | Recursive tol. |
| 2×2×2 | learned | add:jac:q1r | 361 | 158.74 | 9.620e-09 | 7.649e-03 | 1.882e+00 | Recursive tol. |
| 2×2×2 | learned | bnn:kpp:q1r | 135 | 59.76 | 9.449e-09 | 7.893e-03 | 1.882e+00 | Recursive tol. |
| 2×2×2 | learned | defl:kpp:q1r | 681 | 300.26 | 3.338e+03 | 3.338e+03 | 1.981e+04 | Time limit |
| 3×3×3 | exact | jacobi | 469 | 300.51 | 1.423e-02 | 1.423e-02 | 7.838e-04 | Time limit |
| 3×3×3 | exact | kpp | 467 | 300.51 | 2.764e-06 | 2.764e-06 | 2.383e-11 | Time limit |
| 3×3×3 | exact | add:jac:q1r | 359 | 253.07 | 9.482e-09 | 9.482e-09 | 0.000e+00 | Recursive tol. |
| 3×3×3 | exact | bnn:kpp:q1r | 122 | 94.63 | 9.866e-09 | 9.866e-09 | 5.850e-11 | Recursive tol. |
| 3×3×3 | exact | defl:kpp:q1r | 122 | 86.38 | 9.759e-09 | 9.759e-09 | 2.707e-11 | Recursive tol. |
| 3×3×3 | learned | jacobi | 205 | 301.13 | 1.379e+00 | 1.379e+00 | 9.076e+00 | Time limit |
| 3×3×3 | learned | kpp | 204 | 300.25 | 2.783e-02 | 2.899e-02 | 1.862e+00 | Time limit |
| 3×3×3 | learned | add:jac:q1r | 196 | 300.60 | 1.335e-04 | 8.718e-03 | 1.861e+00 | Time limit |
| 3×3×3 | learned | bnn:kpp:q1r | 138 | 221.51 | 9.588e-09 | 8.990e-03 | 1.861e+00 | Recursive tol. |
| 3×3×3 | learned | defl:kpp:q1r | 196 | 301.33 | 1.207e+02 | 1.207e+02 | 1.047e+03 | Time limit |

## Table ST10. Legacy two-cell deployment comparison

Both cell operators are learned in this D deployment case. The two cells are 0020-r2 and its FULL parent, with 36,264 free DOFs and nine jointly solved loads. The ideal preconditioner uses the exact assembled factor. Six-load errors refer to the six face loads.

| Operators | Preconditioner | Iterations | Solve time (s) | Recursive residual | Six-load compliance (%) | Six-load sensitivity (%) | All-load compliance for exact operator (%) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| learned | ideal | 13 | 64.78 | 1.767e-09 | 4.563 | 8.485 | — |
| learned | none | 3000 | 460.17 | 9.096e-02 | 4.577 | 8.409 | — |
| learned | jacobi | 2524 | 387.29 | 9.622e-09 | 4.562 | 8.487 | — |
| exact_dd | ideal | 1 | 0.16 | 4.532e-11 | — | — | 1.355e-09 |
| exact_dd | none | 3000 | 488.24 | 9.307e-02 | — | — | 2.201e-02 |
| exact_dd | jacobi | 2149 | 349.78 | 9.933e-09 | — | — | 1.402e-09 |


## Table ST11. Discrete operator and diagnostic definitions

The same retained coordinate convention is used for the learned extension, the variational readout, and assembly. Relative errors are dimensionless and are displayed as percentages unless indicated otherwise.

| Item | Definition or setting |
| --- | --- |
| Geometry | Unit-box P-type thin-wall cells with corner thickness parameters; FULL cells and plane-cut cells in retained-volume strata v0, v1, v2. |
| Elastic discretisation | Isotropic small-strain elasticity on active tensor-product Q2 hexahedra; body stiffness plus the prescribed ghost-penalty contribution. |
| Retained coordinates | All active box-face coefficients and the cut-band retained coefficients; node-major Cartesian displacement order. |
| Internal reference | \(A=K_{II}\); exact interior extension with the retained values prescribed. |
| Background coordinate convention | 32 background elements and 65 Q2 node positions per axis. |
| Standard moment evaluator | \(4^3\) initial subcells per active element, one local refinement of partial subcells, and clipped Kuhn tetrahedra with rule parameter 4. |
| Material and stabilisation parameters | For all validation geometries: \(E_Y=1,\nu=0.3,\gamma=0.0001\). |
| Learned readout | \(\widehat S=F^TKF\), with \(F=\widehat E\) for the uncorrected network. |
| Energy excess | \(\varepsilon(q)=q^T(\widehat S-S)q/(q^TSq)\). Geometry means average directions first; population means weight geometries equally. |
| Compliance error | \(e_C=\lvert\widehat C/C-1\rvert\), with \(C=f_g^TU\). |
| Sensitivity error | \(e_s=\lVert\widetilde{\boldsymbol s}-\boldsymbol s\rVert_2/\lVert\boldsymbol s\rVert_2\); the field-based estimate has one component for each corner thickness parameter. |
| Spectrum | \(Av_j=\lambda_j Dv_j\), \(D=\operatorname{diag}(A)\); cumulative fractions of the internal error or exact internal-field energy, with separate denominators. |
| Fixed correction | Retained values fixed; Chebyshev relaxation and an interior Galerkin coarse correction. Two-sided sequences use k steps before and k steps after the coarse correction. |
| Assembly diagnostic | Learned target cell joined to its exact continuous-thickness neighbour; x and y denote the adjacent-cell configuration. |
| Six-load | Maximum compliance and sensitivity errors over the six face loads are each at most 3%; cut-traction loads are tabulated separately. |
| Iterative residuals | Recursive PCG residual and \(\max_j\lVert f_j-\mathbb K_{\rm run}\widehat U_j\rVert_2/\lVert f_j\rVert_2\), recomputed with the same operator used in that run. |


### Validation geometry domain

The 80 geometries use independently generated thickness fields, with 20 uniform, 30 affine and 30 mixed trilinear fields. The generator constrains every corner parameter to \([0.17520160,0.69933962]\), the corner span to at most 0.47, and the maximum reference-coordinate gradient norm to at most 0.47. In blocks of eight, a uniform-field-equivalent centre volume fraction is stratified between 0.1 and 0.4; nonuniform affine or trilinear shapes are scaled within these constraints. This centre-density parameter is a sampling coordinate, not the material fraction after cutting.

Canonical cut normals are \((\cos\vartheta,\sin\vartheta,0)\). The generator stratifies \(\vartheta\) over the two halves of \((0,\pi/4)\) and the retained macro-box volume \(v_{\mathcal B}\) over thirds of \((0,1)\). Two of every eight validation fields are uncut and the other six occupy the angle–volume strata. Training and validation are drawn from separate random streams; the present table describes the 80-geometry validation set.

| Stratum | Geometries | Generation interval for \(v_{\mathcal B}\) | Observed \(v_{\mathcal B}\) | Observed corner-parameter range |
| --- | ---: | --- | --- | --- |
| Uncut | 20 | 1 | 1 | 0.1762–0.6902 |
| Light cut | 20 | \((2/3,1)\) | 0.6765–0.9996 | 0.1853–0.6972 |
| Moderate cut | 20 | \((1/3,2/3)\) | 0.3437–0.6605 | 0.1867–0.6983 |
| Heavy cut | 20 | \((0,1/3)\) | 0.01326–0.3195 | 0.1768–0.6946 |

The cut volumes refer to the box before intersecting it with the TPMS band. Geometry identifiers and cube-orbit identifiers are unique within this validation set. The selected spectral and assembly cases are identified in the benchmark key; they are reported as diagnostic cases rather than a random sample for population inference.

## Table ST12. Training and evaluation settings

| Arm | Training geometries | Training-time validation geometries | Run budget (steps) | Evaluated step / weights | Evaluation correction | New-validation views | New-validation geometries |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P0 | 148 | 20 | 15,000 | 15,000 / EMA | None | 0, 17 | 80 |
| B | 305 | 40 | 40,000 | 30,000 / EMA | None | 0, 17 | 80 |
| C | 591 | 40 | 15,000 | 15,000 / EMA | None | 0 | 80 |
| S8 | 591 | 40 | 15,000 | 15,000 / EMA | Eight-step smoothing | 0 | 80 |


The P0 row uses the final selected-weight snapshot; the other rows use the best selected-weight snapshot. C and S8 are separate continuation recipes initialized from B. S8 denotes evaluation of its own checkpoint with an eight-step tail. The fixed-checkpoint correction experiments use B. The new-validation set comprises 20 FULL cells and 20, 20, and 20 cells in the light-, middle-, and heavy-cut strata. The original five direction classes each cover 80 geometries; force_c and face_c cover 20 and 20, support_k covers 19, and glued covers 15. 

For B, C and S8, checkpoint selection uses the bias-corrected EMA weights and both validation views 0 and 17, even where the new-validation table reports only view 0. Within a geometry family and view, let \(E_{fv}\) be the mean energy excess averaged over the selection classes, \(S_{fv}\) the mean relative sensitivity-vector error over classes with labels, and \(P_{fv}\) the class-average 90th percentile of directional energy excess. Each class statistic is first averaged over the available geometries in that family. The selection score is

\[
J_{\rm sel}=\frac12\sum_{v\in\{0,17\}}\frac1{|\mathcal F|}
\sum_{f\in\mathcal F}\left(E_{fv}+S_{fv}+\tfrac12P_{fv}\right).
\]

Families and the two views carry equal weight. The eight selection classes are `force`, `support`, `face`, `macro`, `grf`, `force_c`, `face_c` and `support_k`; absent classes are omitted and an absent sensitivity term contributes zero. The percentile term averages within-geometry percentiles rather than pooling all directions. No additional sensitivity-percentile term is used. B is evaluated every 10,000 updates and becomes eligible at update 10,000; C and S8 are evaluated every 7,500 updates and become eligible at update 7,500. Among eligible evaluations, the lowest finite score is selected. This selection criterion differs from the per-batch training loss in Eq. (8) and from the geometry-weighted statistics of the unseen validation set.


## Table ST13. Complete continuous-neighbour assembly results

Maximum relative compliance and field-based sensitivity-vector errors over the six face loads defining the joint criterion. Sensitivity maxima include both cells. The target cell uses the specified learned arm and its neighbour is exact. Cell labels abbreviate the validation identifiers. The joint criterion is 3% for both errors.

| Target cell | Configuration | Arm | Max. compliance error (%) | Max. sensitivity error (%) | PCG iterations | Outcome |
| --- | --- | --- | --- | --- | --- | --- |
| U1 | x | B | 0.465 | 5.830 | 11 | Above criterion |
| U1 | y | B | 0.427 | 5.390 | 11 | Above criterion |
| U2 | x | B | 0.105 | 1.996 | 11 | Pass |
| M1 | x | B | 4.384 | 12.153 | 15 | Above criterion |
| H1 | x | B | 1.180 | 2.512 | 13 | Pass |
| H1 | y | B | 1.379 | 1.820 | 13 | Pass |
| M2 | x | B | 0.571 | 2.539 | 15 | Pass |
| U1 | x | C | 0.401 | 4.885 | 11 | Above criterion |
| U1 | y | C | 0.388 | 4.723 | 11 | Above criterion |
| U2 | x | C | 0.104 | 1.781 | 11 | Pass |
| U2 | y | C | 0.119 | 1.906 | 11 | Pass |
| M1 | x | C | 4.119 | 11.365 | 15 | Above criterion |
| M1 | y | C | 3.458 | 10.929 | 15 | Above criterion |
| H1 | x | C | 1.061 | 2.469 | 13 | Pass |
| H1 | y | C | 1.223 | 1.734 | 13 | Pass |
| M2 | x | C | 0.593 | 2.874 | 15 | Pass |
| U1 | x | S8 | 0.139 | 3.894 | 7 | Above criterion |
| U1 | y | S8 | 0.140 | 3.747 | 7 | Above criterion |
| U2 | x | S8 | 0.031 | 1.115 | 6 | Pass |
| U2 | y | S8 | 0.038 | 1.106 | 6 | Pass |
| M1 | x | S8 | 1.677 | 5.829 | 9 | Above criterion |
| M1 | y | S8 | 1.391 | 5.555 | 9 | Above criterion |
| H1 | x | S8 | 0.178 | 1.216 | 9 | Pass |
| H1 | y | S8 | 0.270 | 0.880 | 8 | Pass |
| M2 | x | S8 | 0.188 | 1.887 | 8 | Pass |


The comparison contains seven B configurations and nine each for C and S8. C and S8 satisfy both 3% criteria in the same five configurations. Missing model/configuration combinations have no row.


## Table ST14. Complete deployment dimensions and cost breakdown

All costs in this table refer to D and an NVIDIA GeForce RTX 5090. Local exact and learned routes share the topology, stiffness construction, and mechanical target. Each operator-action time is for the complete B-column batch, averaged over three synchronized repetitions after one warmup.

### ST14a. Single-cell dimensions

| Cell | Active DOFs | Retained DOFs | Interior DOFs |
| --- | --- | --- | --- |
| 0020-r2 | 177,507 | 24,636 | 152,871 |
| 0020-r1 | 67,224 | 18,858 | 48,366 |
| 0020-FULL | 289,494 | 17,508 | 271,986 |
| 0007-FULL | 404,148 | 25,920 | 378,228 |


### ST14b. Single-cell setup stages

| Cell | Topology/setup (s) | Moment evaluation (s) | Stiffness assembly (s) | Factor fp64 (s) | Factor fp32 (s) | Learned preparation (s) |
| --- | --- | --- | --- | --- | --- | --- |
| 0020-r2 | 1.318 | 0.832 | 0.726 | 2.693 | 1.571 | 0.321 |
| 0020-r1 | 0.419 | 0.237 | 0.398 | 0.644 | 0.470 | 0.065 |
| 0020-FULL | 1.653 | 0.994 | 1.174 | 5.581 | 2.668 | 0.122 |
| 0007-FULL | 1.540 | 0.975 | 1.150 | 8.596 | 3.450 | 0.151 |


Learned preparation sums network-input preparation, geometry object construction, model cache, and operator freezing. Moment evaluation and stiffness assembly are reported as separately timed calls.

### ST14c. Batch action costs (ms)

| Cell | B | Exact fp64 | Exact fp32 + refinement | Learned sparse | Learned fused |
| --- | --- | --- | --- | --- | --- |
| 0020-r2 | 1 | 29.34 | 27.26 | 9.44 | 9.42 |
| 0020-r2 | 16 | 43.20 | 62.23 | 97.77 | 86.08 |
| 0020-r2 | 64 | 129.07 | 138.85 | 366.89 | 322.88 |
| 0020-r1 | 1 | 7.92 | 8.03 | 6.17 | 7.11 |
| 0020-r1 | 16 | 12.91 | 20.62 | 38.23 | 32.72 |
| 0020-r1 | 64 | 35.40 | 41.30 | 162.30 | 146.77 |
| 0020-FULL | 1 | 53.53 | 44.02 | 13.15 | 13.00 |
| 0020-FULL | 16 | 76.75 | 102.68 | 154.38 | 134.66 |
| 0020-FULL | 64 | 233.61 | 227.52 | 575.09 | 499.48 |
| 0007-FULL | 1 | 72.94 | 59.95 | 15.31 | 15.12 |
| 0007-FULL | 16 | 100.17 | 129.28 | 198.45 | 172.86 |
| 0007-FULL | 64 | 315.57 | 303.65 | 741.97 | 645.64 |


### ST14d. Balanced two-level preconditioned assembly solves

| Assembly | Free DOFs | Operator | Iterations | Solve time (s) | Recursive residual | Recomputed residual | Max. compliance error (%) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Pair | 36,264 | exact | 169 | 27.94 | 8.430e-09 | 8.430e-09 | 0.000e+00 |
| Pair | 36,264 | learned | 204 | 21.74 | 9.921e-09 | 2.075e-02 | 3.409 |
| 2×2×2 | 93,696 | exact | 119 | 22.71 | 9.997e-09 | 9.997e-09 | 8.518e-11 |
| 2×2×2 | 93,696 | learned | 135 | 59.76 | 9.449e-09 | 7.893e-03 | 1.882 |
| 3×3×3 | 294,516 | exact | 122 | 94.63 | 9.866e-09 | 9.866e-09 | 5.850e-11 |
| 3×3×3 | 294,516 | learned | 138 | 221.51 | 9.588e-09 | 8.990e-03 | 1.861 |


The block assemblies repeat one FULL cell. Six loads are solved: three consistent face loads and three random loads. The residuals use each run’s own operator; compliance is compared with the recorded exact-operator reference. Additional preconditioners and time-limit outcomes are reported in Table ST09. 

The assembly benchmark records TF32-enabled convolutions. A convolution-precision flag is not stored in the historical single-cell benchmark. Exact factorisation precision and residual refinement are identified explicitly in the action-time columns.


## Table ST15. Matched face-load responses for the smoothed variant

Target cell S8, exact neighbour, configuration x. Each row uses one load and reports both cell sensitivity-vector errors. Energy participation refers to the target in the exact assembled solution. Relative quantities are percentages.

| Target | Load | Compliance error | Target sensitivity error | Neighbour sensitivity error | Target energy participation |
| --- | --- | --- | --- | --- | --- |
| H1 | T-x | 0.178113 | 1.21556 | 0.0755185 | 64.2733 |
| H1 | T-y | 0.0730244 | 0.843527 | 0.0145867 | 40.803 |
| H1 | T-z | 0.0319646 | 0.502716 | 0.00906095 | 27.3735 |
| H1 | N-x | 0.000504597 | 0.442032 | 0.000822323 | 0.217835 |
| H1 | N-y | 0.00082195 | 0.345192 | 0.00205754 | 0.308471 |
| H1 | N-z | 4.06479e-05 | 0.885999 | 5.96007e-05 | 0.0360864 |
| U1 | T-x | 0.13928 | 1.18517 | 0.00831909 | 35.0385 |
| U1 | T-y | 0.0448159 | 1.30571 | 0.00536493 | 11.542 |
| U1 | T-z | 0.0313817 | 1.10598 | 0.00456525 | 11.6406 |
| U1 | N-x | 0.00254548 | 2.37052 | 0.00246372 | 0.355414 |
| U1 | N-y | 0.00351229 | 1.28801 | 0.00481203 | 0.873919 |
| U1 | N-z | 0.00136724 | 3.89419 | 0.00186976 | 0.127467 |
| M1 | T-x | 1.6765 | 4.51959 | 1.37179 | 51.2504 |
| M1 | T-y | 1.41865 | 5.82917 | 0.942721 | 31.068 |
| M1 | T-z | 0.416357 | 2.99719 | 0.137715 | 21.2573 |
| M1 | N-x | 0.00212258 | 0.914003 | 0.0056564 | 0.354364 |
| M1 | N-y | 0.0178155 | 1.05549 | 0.0460078 | 0.742265 |
| M1 | N-z | 0.000598185 | 1.82925 | 0.000963143 | 0.102043 |

## Table ST16. Load-specific compliance weighting for the baseline predictor

U1/x under the neighbour-face z traction. B is used on the target and its neighbour is exact. The bound is evaluated from dimensionless ratios before percentage conversion.

| Target participation (%) | Local energy excess (%) | Product bound (%) | Compliance error (%) | Target sensitivity error (%) |
| --- | --- | --- | --- | --- |
| 0.1274669 | 2.312168 | 0.002947249 | 0.002827394 | 5.82992 |

Figures S02–S05 provide the complementary distributions, initial-field comparisons, coarse-space comparisons, sensitivity diagnostics and iterative-solve comparisons.


## Supplementary Note S1. Global preconditioning and two-cell load configurations

### S1.1. Balanced two-level preconditioner

Let \(\mathbb A\) be the supported assembled operator used in a run, either the exact or learned one. This notation is distinct from the local interior stiffness \(A=K_{II}\). Let
\(\mathbb K_{PP}=\sum_mB_m^TK_{PP,m}B_m\) on the free retained coordinates. The stiffness-block fine action is \(\mathcal B_f=\mathbb K_{PP}^{-1}\); the diagonal alternative uses \(\operatorname{diag}(\mathbb K_{PP})^{-1}\). It does not require local Neumann solves.

For a global retained-coordinate coarse basis \(Z_g\), the ideal coarse action is
\(Q_g=Z_g(Z_g^T\mathbb A Z_g)^{-1}Z_g^T\), after removal of dependent columns. The balanced action recorded as BNN is

\[
\mathcal M^{-1}=Q_g+(I-Q_g\mathbb A)\mathcal B_f(I-\mathbb A Q_g).
\]

The reported q1r basis multiplies macro-grid trilinear vertex functions by three translations and three rotations about each vertex, then restricts the resulting fields to the supported retained coordinates. This global coarse basis differs from the interior basis \(V\) used to correct local extensions.

The supplied implementation forms \(A_g=Z_g^T\mathbb A Z_g\), records its relative asymmetry, symmetrises it, and scales it by its diagonal. It retains positive scaled eigenvalues exceeding \(10^{-10}\) times the largest eigenvalue and uses the corresponding normalised columns \(W\) so that \(Q_g=WW^T\). The stored product \(\mathbb A W\) supplies the two projections. This spectral selection concerns the preconditioner coarse action and leaves the local fine stiffnesses and condensed targets unchanged. The ideal symmetric formula above describes the algorithm; a finite-precision operator may additionally exhibit the action/energy discrepancy discussed in Appendix J.6.

The additive variant applies \(Q_g+\mathcal B_f\). The deflated variants use a coarse initial solution and a projected fine correction. Table ST09 preserves the original variant identifiers and records their individual stopping outcomes. The reference inequality \(\mathbb K_{PP}\succeq\mathbb K\) holds for exact condensation of the specified positive-semidefinite cell matrices; it does not imply \(\mathbb K_{PP}\succeq\widehat{\mathbb K}\) for an arbitrary learned extension.

### S1.2. Two-cell diagnostic configuration

The target cell occupies \([0,1]^3\). In configuration x, the neighbour is translated by \((-1,0,0)\), its far face \(x=-1\) is clamped, and the six face loads act on the plane \(y=0\). In configuration y, the neighbour is translated by \((0,-1,0)\), its far face \(y=-1\) is clamped, and the face loads act on \(x=0\). The six cases comprise the three Cartesian traction directions applied to the target face and the same three directions applied to the neighbour face. The consistent-load implementation integrates the Q2 surface shape functions, normalizes each nodal load to unit resultant before support elimination, and then eliminates clamped entries. Three similarly normalised consistent tractions on the target cut surface are reported separately when present. Non-box cut-band coordinates remain private free variables. The continuous-thickness neighbour shares the prescribed thickness values on the common face.

The pair accuracy calculation uses the Cholesky factor of the exact assembled reference stiffness as the PCG preconditioner. The supplied implementation uses a relative recursive-residual tolerance of \(10^{-10}\), and the reported runs allow at most 400 iterations. The saved result summaries retain the iteration count but omit the residual returned by the solver. These settings concern the pair accuracy comparison; Table ST09 uses the separately specified deployment preconditioners.

The implementation also contains a uniform-nodal-force branch. The comparisons identified as consistent face loads use the integrated branch; the two branches must retain distinct load definitions in any reuse of the records.


## Supplementary Note S2. Definition of the illustrative matrix example

The re-equilibration example in Appendix J.9.1 uses two retained and three internal coordinates. Its matrices are

\[
A=\begin{bmatrix}3&.2&.1\\.2&2&-.1\\.1&-.1&1.4\end{bmatrix},\qquad
S=\begin{bmatrix}1.4&-.2\\-.2&.9\end{bmatrix},\qquad
L=\begin{bmatrix}.2&-.1\\.3&.25\\-.2&.15\end{bmatrix},
\]
\[
K=\begin{bmatrix}S+L^TAL&-L^TA\\-AL&A\end{bmatrix},\quad
E=\begin{bmatrix}I_2\\L\end{bmatrix},\quad
H_0=\begin{bmatrix}.3&-.2\\-.1&.4\\.2&.1\end{bmatrix},\quad
F_t=E+t\begin{bmatrix}0_{2\times2}\\H_0\end{bmatrix}.
\]

The supported three-coordinate assembly uses

\[
B=\begin{bmatrix}1&0&0\\0&1&0\end{bmatrix},\quad
G=B^TSB+\begin{bmatrix}.5&0&-.2\\0&.4&-.1\\-.2&-.1&1.1\end{bmatrix},\quad
f=(1,-.4,.8)^T,
\]
\[
\widehat G_t=G+t^2 B^TH_0^TAH_0B,\qquad
U=G^{-1}f,\quad \widehat U_t=\widehat G_t^{-1}f.
\]

Eight symmetric derivative matrices are generated once with NumPy's default generator and seed 620260926: for each corner, draw a \(5\times5\) standard-normal matrix \(R_c\), then set \(D_c=(R_c+R_c^T)/16\). These matrices provide algebraic sensitivity directions; the positive-semidefinite nested-thickening counterexamples are given separately in Appendix J.9.2–3. The saved sequence is \(t=0.1,0.05,0.025,0.0125,0.00625\). The final two entries give the following log-two slopes:

| Quantity | Saved slope |
| --- | --- |
| Condensed operator error | 2.000000 |
| Retained solution error | 1.999941 |
| Compliance gap | 1.999949 |
| Full local field error | 0.999626 |
| Field-based sensitivity error | 1.001084 |
| Retained-solution error energy | 3.999886 |

These entries reproduce the previously saved algebraic check; they are not TPMS observations or new mechanics experiments.


![Figure S02](figures/S02_distributions.png)

**Figure S02. Distributions of geometry-level directional energy errors.** Each point is one geometry's directional mean in the original orientation; horizontal marks are population medians. The nodal-force, spring-support, single-face-force, polynomial and multiscale classes contain 80 geometries each. Consistent traction, single-face consistent traction, stiffness-scaled support and neighbour-induced displacement classes contain 20, 20, 19 and 15 geometries, respectively. P0 has no records in these four enriched classes. Marker shape and colour identify the predictor; deterministic horizontal offsets separate overlapping observations. All panels use the same logarithmic error range.

![Figure S03A](figures/S03A_smoothing.png)

**Figure S03A. Smoothing from learned and zero internal fields.** (a,b) Mean directional energy excess for consistent-traction and nodal-force responses; (c,d) corresponding field-based sensitivity errors. Both initialisations prescribe the same retained displacement. Solid curves with filled markers start from predictor B; dashed curves with open markers start from zero internal displacement. Zero-start sensitivity is recorded only at 32 steps. All corrections use \(\alpha=30\). The step axis is linear between zero and one and logarithmic thereafter.

![Figure S03B](figures/S03B_coarse_spaces.png)

**Figure S03B. Recorded coarse representations and correction sequences.** Rows correspond to U1, M1 and M2; columns use consistent-traction and nodal-force responses. Six coarse representations are compared under four initialisation and smoothing sequences, with eight steps in each pre- or post-smoothing stage. Dots indicate directional means and caps the 90th percentile. Dashed and dotted references denote B alone and B followed by one smoothing stage. \(Q_1\), \(Q_2\) and PU denote trilinear, quadratic and linearly enriched partition-of-unity generating families. The first label number identifies grid resolution and the lower number counts columns after internal restriction and screening. For the structurally redundant PU family, these counts do not establish an independent-space dimension, and the plotted solve results do not verify exact-projection properties. Appendix F.1 explains the rank and solve conditions; Table ST04 gives all statistics. Coarse updates preserve every retained coordinate.

![Figure S04](figures/S04_sensitivity_diagnostics.png)

**Figure S04. Field-based sensitivity-error diagnostics.** (a) Paired mean energy and sensitivity errors for consistent-traction and nodal-force responses, using six B cells and five C cells. (b) Consistent-traction linear-term norm share \(\|D_1\|_F/(\|D_1\|_F+\|D_2\|_F)\), where \(D_1+D_2\) is the sensitivity-error matrix over all eight design components and evaluated directions. (c,d) Shares of absolute elementwise sensitivity-error contributions and element counts in four mutually exclusive material-volume-fraction groups for B under consistent tractions. Each error group sums absolute contributions over its elements, design components and directions before normalisation by the total. Open markers and hatched bars in (a,b) identify C; its H2 observation is unavailable.

![Figure S05](figures/S05_iterative_solves.png)

**Figure S05. Recorded assembled iterative solves.** Rows show a cell pair and repeated-cell \(2\times2\times2\) and \(3\times3\times3\) arrays; columns show iteration counts, times and relative residuals. Filled markers give recursive residuals and open markers explicitly recomputed residuals using the operator applied in the same run. Crosses identify the 300 s time limit, and the dashed residual reference is \(10^{-8}\). Diag, \(K_{PP}\), Add, Bal and Def denote diagonal, assembled retained-block, additive, balanced two-level and deflated preconditioning, respectively. Here \(K_{PP}\) denotes the fine action \(\mathbb K_{PP}^{-1}\) on the assembled retained system; the precise actions are given in Supplementary Note S1. The learned operator uses predictor D. These historical RTX 5090 runs enabled TF32 convolution. The pair joins G1 and G3, while each array repeats G3.

## Supplementary Note S3. Geometry visualisation

The surfaces in Figure 1 are sampled on a grid with 97 positions per unit-box axis using the eight corner band parameters and cut-plane data of U1, M1, H1 and H2. The displayed percentages describe the retained macro-domain volume, before intersection with the thin-wall material. This surface sampling is used for visualisation; the mechanical discretisation has 32 background elements per axis and continuous Q2 displacement functions.

