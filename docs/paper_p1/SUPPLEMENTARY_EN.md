# Supplementary material

Supplementary Notes, Tables and Figures are numbered in the order in which they appear in this supplement. Variant labels follow the main text (Table 2). The two variants not listed there, the Uncorrected continuation and the Smoothing-trained variant, are evaluated in this supplement. The key below gives each variant's record identifier in the data archive.

## R1. Variant and geometry key

| Label | Numerical role | Record identifier in the data archive |
| --- | --- | --- |
| Base network | Baseline variant; starting network parameters of every continuation and of the correction experiments of Section 5.5 | v2L1 |
| Uncorrected continuation | Base network continued; no correction in training or evaluation | A0_ctrl |
| Smoothing-trained | Base network continued and trained through eight smoothing steps | A2b_tail8 |
| Base network + correction | Base network with NICE's correction applied at deployment, without retraining | B2grid |
| NICE | Principal variant, trained through the complete correction (8 / Q1(17) / 8) | A3_2grid |

| Cell label | Cut-severity group | Geometry identifier in the data archive |
| --- | --- | --- |
| U1 | Uncut | fresh_val_2000_full |
| U2 | Uncut | fresh_val_2001_full |
| M1 | Moderately cut | fresh_val_2003_d1_v1 |
| H1 | Heavily cut | fresh_val_2005_d1_v0 |
| M2 | Moderately cut | fresh_val_2006_d0_v1 |
| H2 | Heavily cut | fresh_val_2010_d0_v0 |
| H3 | Heavily cut | fresh_val_2002_d0_v0 |
| L1 | Lightly cut | fresh_val_2004_d0_v2 |
| W1: validation cell with the largest NICE error (Tables ST08b, ST10) | Moderately cut | fresh_val_2051_d1_v1 |
| W2 (Table ST08b) | Heavily cut | fresh_val_2045_d1_v0 |
| W3: validation cell with the thinnest walls (Tables ST08b, ST10) | Heavily cut | fresh_val_2074_d0_v0 |
| W4 (Table ST08b) | Heavily cut | fresh_val_2021_d1_v0 |
| W5 (Table ST08b) | Lightly cut | fresh_val_2063_d1_v2 |
| RU (Table ST08b) | Uncut | fresh_val_2032_full |
| RL (Table ST08b) | Lightly cut | fresh_val_2047_d1_v2 |
| RM (Table ST08b) | Moderately cut | fresh_val_2078_d0_v1 |
| RH (Table ST08b) | Heavily cut | fresh_val_2053_d1_v0 |

In the validation-cell identifiers, d0 and d1 give the half of \((0,\pi/4)\) that contains the cut angle \(\vartheta\), and v0, v1 and v2 give the heavily, moderately and lightly cut groups, which are defined by the remaining volume. In configuration labels such as U1/x, x and y give the neighbour configuration of Figure 9. The deployment geometries use the G1–G4 labels of Table ST13; the table below gives their identifiers in the data archive.

| Benchmark label | Cut-severity group | Geometry identifier in the data archive |
| --- | --- | --- |
| G1 | Cut | fresh_train_0020_cover01_r2 |
| G2 | Cut | fresh_train_0020_cover01_r1 |
| G3 | Uncut | fresh_train_0020_full |
| G4 | Uncut | fresh_train_0007_full |

## Table ST01. Training and evaluation settings

| Variant | Training set (geometries) | Training-time validation geometries | Training steps | Evaluated training step / parameters | Evaluation correction | Evaluated orientations |
| --- | --- | --- | --- | --- | --- | --- |
| Base network | 305 | 40 | 40,000 | 30,000 / EMA | None | Identity, 17 |
| Uncorrected continuation | 591 | 40 | 15,000 | 15,000 / EMA | None | Identity |
| Smoothing-trained | 591 | 40 | 15,000 | 15,000 / EMA | Eight-step smoothing | Identity |
| Base network + correction | 305 | 40 | — | 30,000 / EMA | 8 / Q1(17) / 8 | Identity |
| NICE | 591 | 40 | 15,000 | 15,000 / EMA | 8 / Q1(17) / 8 | Identity, 17 |

All variants are evaluated on the 80 validation geometries of Table ST02. Orientation 17 is one of the 48 cube-symmetry transformations of Appendix G (Eq. (G.2)), applied to the geometry and to the test displacements; orientation 0 is the identity. The training set of 591 comprises 304 of the base network's 305 geometries (one cell that behaved as a near-mechanism was removed) and 287 further training geometries. Under the geometry replacement described below, each 15,000-step continuation visits at most 153 of its 591 training geometries; the base network visited all 305 of its own over 40,000 training steps. The Uncorrected continuation, the Smoothing-trained variant and NICE use the same seed and split, so they draw their geometries in the same order.

Training settings, common to the variants of Table 2, the Uncorrected continuation and the Smoothing-trained variant:

- Batches of 16 test displacements; the geometry changes at every step. Three geometries of the training set are held on the GPU and one of them is replaced every 100 steps, so a run of \(N_{\rm st}\) training steps visits at most \(N_{\rm st}/100+3\) distinct training geometries.
- Nominal class weights, in the order `force`, `force_c`, `support`, `support_k`, `face`, `face_c`, `macro`, `grf`, `glued`, `adv`: \(0.10,0.10,0.075,0.075,0.10,0.10,0.10,0.125,0.075,0.15\), renormalised over the available classes and assigned by systematic quotas.
- Adam with a one-cycle learning-rate schedule: peak \(3\times10^{-4}\), 5% warm-up, cosine decay and final division factor 100; gradient norm clipped at one.
- Model selection and evaluation use an exponential moving average (EMA) of the network parameters, with decay 0.9997 and zero-initialisation bias correction.
- Eq. (13): \(10^{-12}\) lower clamp inside the logarithm of the energy term; the sensitivity term has unit weight.

The base network was initialised from parameters obtained in three preceding training stages. The data split comprises 691 geometries: 591 training geometries and 100 validation geometries, which include the 80 of Table ST02.

For the base network, the Uncorrected continuation, the Smoothing-trained variant and NICE, checkpoint selection uses the bias-corrected EMA parameters and both the identity orientation (\(v=0\)) and orientation 17 (\(v=17\)), although Table ST03 reports only the identity orientation. A geometry family consists of the geometries of the training-time validation list generated from one thickness field; an independently generated validation cell forms a family of its own. Within a family \(f\) and orientation \(v\), let \(\bar\varepsilon_{fv}\) be the mean energy error averaged over the selection classes and \(\bar e_{s,fv}\) the mean relative sensitivity-vector error over the classes with sensitivity labels. Let \(\varepsilon^{(90)}_{fv}\) be the class average of the 90th percentile of the energy error over the test displacements. Each class statistic is first averaged over the available geometries in that family. The selection score is

\[
J_{\rm sel}=\frac12\sum_{v\in\{0,17\}}\frac1{|\mathcal F|}
\sum_{f\in\mathcal F}\left(\bar\varepsilon_{fv}+\bar e_{s,fv}+\tfrac12\varepsilon^{(90)}_{fv}\right).
\]

Families and the two orientations carry equal weight. The eight selection classes are `force`, `support`, `face`, `macro`, `grf`, `force_c`, `face_c` and `support_k`; absent classes are omitted and an absent sensitivity term contributes zero. Among eligible evaluations, the lowest finite score is selected.

- The base network was scored at 10,000, 20,000, 30,000 and 40,000 training steps (\(J_{\rm sel}\) = 0.1057, 0.1065, 0.0905 and 0.0912); the parameters of step 30,000 were selected. Every continued variant starts from these parameters.
- The Uncorrected continuation and NICE were scored at 7,500 and 15,000 training steps (0.0944 and 0.0906; 0.00281 and 0.00237); in both cases the final parameters scored lower and were kept.
- The Smoothing-trained variant used the same evaluation schedule and orientations; its selection scores were not recorded, and the parameters of its final step, 15,000, are used.
- NICE was designated the principal variant after all variants had been compared on the 80 validation geometries and the two-cell configurations.

Twenty of the 80 validation geometries (6 uncut, 14 cut) belong to the selection list; they include all eight detailed cells of Supplementary Section R1.

## Table ST02. Validation geometry domain

The 80 geometries use independently generated thickness fields, with 20 uniform, 30 affine and 30 mixed trilinear fields. The generator constrains every corner parameter to \([0.17520160,0.69933962]\), the corner span to at most 0.47, and the maximum reference-coordinate gradient norm to at most 0.47. In each block of eight geometries, a uniform-field-equivalent centre volume fraction is drawn by stratified sampling between 0.1 and 0.4; nonuniform affine or trilinear shapes are scaled within these constraints. This centre-density parameter is a sampling coordinate and is not the material fraction after cutting.

Canonical cut normals are \((\cos\vartheta,\sin\vartheta,0)\). The generator draws \(\vartheta\) by stratified sampling over the two halves of \((0,\pi/4)\) and the remaining cell-box volume \(v_{\mathcal B}\) over thirds of \((0,1)\). Two of every eight validation fields are uncut and the other six cover the angle–volume combinations. Training and validation are drawn from separate random streams; the present table describes the 80-geometry validation set.

| Cut-severity group | Geometries | Generation interval for \(v_{\mathcal B}\) | Observed \(v_{\mathcal B}\) | Observed corner-parameter range |
| --- | ---: | --- | --- | --- |
| Uncut | 20 | 1 | 1 | 0.1762–0.6902 |
| Lightly cut | 20 | \((2/3,1)\) | 0.6765–0.9996 | 0.1853–0.6972 |
| Moderately cut | 20 | \((1/3,2/3)\) | 0.3437–0.6605 | 0.1867–0.6983 |
| Heavily cut | 20 | \((0,1/3)\) | 0.01326–0.3195 | 0.1768–0.6946 |

The cut volumes refer to the cell box before it is intersected with the TPMS band. No two validation geometries coincide up to a symmetry of the cube. At \(n=32\), the numbers of retained DOFs of the 80 geometries have a median of 23,604 and range from 2,679 to 45,900; the numbers of active DOFs have a median of 221,442 and range from 2,757 to 439,092. The selected spectral and assembly cases are identified in the geometry key of Supplementary Section R1; they are reported as test cases and do not form a random sample for population inference.

## Table ST03. Energy error by load class (identity orientation)

Entries are mean / maximum over geometries, in percent: each geometry first contributes its mean over the test displacements, and geometries are weighted equally. The maximum is not the worst individual test displacement. "Base network + correction" denotes the base network with NICE's correction applied at deployment, without retraining. The last column restricts NICE to the 60 validation geometries that entered neither training nor checkpoint selection (the other 20 entered checkpoint selection, Table ST01). For NICE under traction loads, the 5,120 individually sampled test displacements of the 80 geometries have a 95th percentile of 0.331%, a 99th percentile of 0.693% and a maximum of 1.24%. For the 3,840 test displacements of the 60 geometries outside checkpoint selection, these values are 0.355%, 0.734% and 1.24%. Under traction loads the mean of Base network + correction is 0.0965% and that of NICE 0.0737%, a ratio of 1.31. Resampling the 80 geometries with replacement (paired, ratio of geometry-equal means, percentile interval) gives a 95% interval of 1.20–1.40.

| Class | Geometries (all / outside selection) | Base network | Uncorrected continuation | Smoothing-trained | Base network + correction | NICE | NICE, outside selection |
| --- | --- | --- | --- | --- | --- | --- | --- |
| force | 80 / 60 | 5.036 / 70.122 | 4.607 / 62.465 | 0.525 / 3.658 | 0.092 / 0.683 | 0.058 / 0.325 | 0.059 / 0.325 |
| support | 80 / 60 | 5.332 / 99.857 | 4.878 / 90.741 | 0.638 / 3.552 | 0.084 / 0.852 | 0.055 / 0.372 | 0.056 / 0.372 |
| face | 80 / 60 | 2.486 / 22.139 | 2.233 / 17.635 | 0.165 / 1.808 | 0.072 / 1.990 | 0.037 / 0.724 | 0.040 / 0.724 |
| macro | 80 / 60 | 0.842 / 2.636 | 0.827 / 2.584 | 0.203 / 0.935 | 0.022 / 0.165 | 0.015 / 0.118 | 0.015 / 0.118 |
| grf | 80 / 60 | 2.038 / 4.582 | 2.009 / 4.550 | 0.251 / 0.933 | 0.029 / 0.180 | 0.025 / 0.143 | 0.025 / 0.143 |
| force_c | 80 / 60 | 6.886 / 48.628 | 6.334 / 41.962 | 1.280 / 7.817 | 0.097 / 0.905 | 0.074 / 0.651 | 0.077 / 0.651 |
| face_c | 80 / 60 | 3.150 / 36.337 | 2.806 / 25.525 | 0.483 / 2.326 | 0.043 / 0.319 | 0.032 / 0.210 | 0.033 / 0.210 |
| support_k | 75 / 56 | 4.177 / 28.817 | 3.941 / 26.175 | 0.938 / 5.246 | 0.077 / 0.632 | 0.057 / 0.468 | 0.058 / 0.468 |
| glued | 75 / 60 | 4.593 / 38.128 | 4.422 / 42.030 | 0.953 / 4.629 | 0.078 / 0.555 | 0.060 / 0.384 | 0.055 / 0.384 |

Under traction loads, NICE's means by cut-severity group on the 60 geometries outside checkpoint selection are 0.020% (uncut), 0.054% (lightly cut), 0.105% (moderately cut) and 0.126% (heavily cut). The five largest geometry means of the 80 (0.651%, 0.540%, 0.487%, 0.303%, 0.287%) all belong to these geometries.

### ST03b. Means by cut-severity group under nodal point loads and traction loads (%)

| Cut-severity group | Geometries | Base network force/force_c | Uncorrected continuation force/force_c | Smoothing-trained force/force_c | Base network + correction force/force_c | NICE force/force_c |
| --- | --- | --- | --- | --- | --- | --- |
| Uncut | 20 | 0.908 / 1.142 | 0.879 / 1.029 | 0.072 / 0.242 | 0.034 / 0.029 | 0.013 / 0.016 |
| Lightly cut (v2) | 20 | 3.239 / 5.801 | 3.021 / 5.325 | 0.523 / 1.414 | 0.088 / 0.096 | 0.059 / 0.079 |
| Moderately cut (v1) | 20 | 4.720 / 7.899 | 4.506 / 7.823 | 0.723 / 1.769 | 0.103 / 0.105 | 0.070 / 0.091 |
| Heavily cut (v0) | 20 | 11.278 / 12.702 | 10.021 / 11.160 | 0.783 / 1.697 | 0.141 / 0.156 | 0.090 / 0.109 |

### ST03c. Orientation dependence of NICE: identity / orientation 17 means (%)

NICE was also evaluated in orientation 17, which entered checkpoint selection (Table ST01), on all 80 geometries with the validation test displacements of Table ST03. The class with stiffness-scaled supports and the class with displacements imposed by a neighbouring cell are not part of this comparison.

| Class | Identity | Orientation 17 |
| --- | --- | --- |
| force | 0.058 | 0.059 |
| support | 0.055 | 0.056 |
| face | 0.037 | 0.034 |
| macro | 0.015 | 0.016 |
| grf | 0.025 | 0.026 |
| force_c | 0.074 | 0.083 |
| face_c | 0.032 | 0.034 |

## Table ST04. Operator verification in the deployed arithmetic

Columns: the maximum relative asymmetry \(\max_{i,j}|G_{ij}-G_{ji}|/\sqrt{|G_{ii}G_{jj}|}\) of \(G=Q^T\widehat SQ\), where the columns of \(Q\) are the first eight test displacements of the cell's traction-load (force_c) validation set. The next three columns give the maximum relative difference between returned work and field energy, the deployed-versus-training field difference and the maximum rigid-body energy ratio, all over the same eight test displacements, which also normalise the rigid-body ratio. The ghost-penalty column gives the ghost-penalty fraction of the exact field energy, averaged over the test displacements (traction loads / nodal point loads). The last two columns give the mean \(\delta\) and \(\kappa\) of Section 5.4 for the base network and for NICE under traction loads, with the Jacobi weighting \(\mathsf W=\operatorname{diag}(0,D)\), \(D=\operatorname{diag}(K_{II})\), of Appendix B.3. Over the first eight nodal-point-load test displacements, the asymmetry, the work–energy difference and the rigid-body energy ratio are at most \(3\times10^{-8}\), \(2\times10^{-8}\) and \(1\times10^{-10}\).

| Cell | Asymmetry | Work–energy difference | Deployed vs training field | Rigid-body energy | Ghost-penalty fraction (traction loads / nodal point loads) | \(\delta\), \(\kappa\): base network (traction loads) | \(\delta\), \(\kappa\): NICE (traction loads) |
|---|---|---|---|---|---|---|---|
| H2 | 6.3e-09 | 4.5e-09 | 4.6e-09 | 1.7e-11 | 0.00048 / 0.49 | 0.71%, 7089 | 0.051%, 582 |
| H1 | 5.6e-09 | 4.0e-09 | 4.2e-08 | 1.6e-12 | 7.9e-05 / 0.81 | 1.23%, 124 | 0.186%, 33 |
| M2 | 8.7e-09 | 3.4e-09 | 9.4e-08 | 2.4e-12 | 3.6e-05 / 0.66 | 1.15%, 294 | 0.156%, 163 |
| M1 | 7.3e-09 | 4.4e-09 | 9.3e-08 | 1.4e-11 | 0.00012 / 0.58 | 1.26%, 866 | 0.179%, 426 |
| U2 | 3.5e-09 | 1.8e-09 | 1.1e-07 | 2.9e-13 | 3.6e-05 / 0.79 | 0.74%, 85 | 0.107%, 47 |

## Table ST05. Sensitivity and spectral diagnostics at fixed retained displacements

### ST05a. Energy and sensitivity errors

Means over the test displacements (%) at fixed retained displacements. \(k\) is the number of Jacobi-preconditioned Chebyshev steps applied to the base network's field (\(a=b/30\); Section 5.5, Figure 8a,b). Smoothed values are tabulated for the base network under traction loads, and — marks entries not tabulated. The first-order fraction, evaluated at \(k=0\), is the Frobenius norm of the linear error array divided by the sum of the Frobenius norms of the linear and quadratic arrays (%). Each array includes all eight corners and all evaluated test displacements in that cell and class.

| Variant | Cell | Class | Energy error, \(k=0\) | Energy error, \(k=8\) / \(32\) | Sensitivity error, \(k=0\) | Sensitivity error, \(k=8\) / \(32\) | First-order fraction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Base network | U1 | force_c | 1.270 | 0.41971 / 0.27998 | 0.664 | 0.52592 / 0.45509 | 42.887 |
| Base network | U1 | force | 0.742 | — | 1.018 | — | 55.124 |
| Base network | U2 | force_c | 0.402 | — | 0.588 | — | 35.658 |
| Base network | U2 | force | 0.506 | — | 1.008 | — | 71.565 |
| Base network | M1 | force_c | 13.512 | 4.685 / 2.9452 | 14.128 | 2.7552 / 1.6339 | 7.164 |
| Base network | M1 | force | 7.156 | — | 4.822 | — | 23.516 |
| Base network | H1 | force_c | 1.814 | 0.25081 / 0.059403 | 1.909 | 0.26139 / 0.14959 | 28.403 |
| Base network | H1 | force | 1.842 | — | 0.923 | — | 35.199 |
| Base network | M2 | force_c | 3.643 | 1.0382 / 0.61088 | 5.159 | 0.83675 / 0.5753 | 9.339 |
| Base network | M2 | force | 2.175 | — | 1.103 | — | 41.390 |
| Base network | H2 | force_c | 34.954 | 0.20515 / 0.0033313 | 75.072 | 0.77195 / 0.010708 | 0.786 |
| Base network | H2 | force | 19.507 | — | 13.649 | — | 36.322 |
| Uncorrected continuation | U1 | force_c | 1.165 | — | 0.937 | — | 35.750 |
| Uncorrected continuation | U1 | force | 0.779 | — | 0.766 | — | 59.185 |
| Uncorrected continuation | U2 | force_c | 0.397 | — | 0.560 | — | 35.988 |
| Uncorrected continuation | U2 | force | 0.515 | — | 1.062 | — | 72.929 |
| Uncorrected continuation | M1 | force_c | 13.009 | — | 13.275 | — | 7.456 |
| Uncorrected continuation | M1 | force | 6.841 | — | 4.442 | — | 26.413 |
| Uncorrected continuation | H1 | force_c | 1.699 | — | 1.894 | — | 28.046 |
| Uncorrected continuation | H1 | force | 1.877 | — | 1.013 | — | 34.751 |
| Uncorrected continuation | M2 | force_c | 3.988 | — | 5.432 | — | 9.548 |
| Uncorrected continuation | M2 | force | 2.105 | — | 1.357 | — | 49.454 |

### ST05b. Energy fractions in the lowest 200 generalised interior modes

Base network, mean over the traction-load test displacements; each fraction uses its own interior-energy denominator (Eq. (10)). Cumulative curves, results for nodal point loads and percentiles over the test displacements are shown in Figure 6.

| Cell | Error fraction (%) | Exact-field fraction (%) |
| --- | --- | --- |
| U1 | 24.019 | 8.498 |
| M1 | 24.452 | 4.870 |
| M2 | 18.399 | 4.620 |
| H2 | 15.216 | 4.705 |

| Cell | \(\lambda_1\) | Lower smoothing endpoint \(a=b/30\) | Modes below \(a\) (of the lowest 200) |
| --- | --- | --- | --- |
| U1 | 0.000402235 | 0.174937 | 200 |
| M1 | 0.000574199 | 0.173294 | 200 |
| M2 | 0.000935594 | 0.174182 | 200 |
| H2 | 0.0213148 | 0.137632 | 66 |

Here \(b\) is the power-iteration estimate, multiplied by 1.05, of the smoothing study of Section 5.5. That study starts the power iteration from its own random vector (Appendix F.2), so its interval differs from the one used by the correction \(\mathcal W\).

## Table ST06. Interior coarse spaces

Mean energy error (%) under traction loads after eight smoothing steps, one coarse-grid correction and eight smoothing steps applied to the base network's field at fixed retained displacements. Coarse columns are those surviving the support screen of Appendix F.1 and the removal of columns whose diagonal energy is at most \(10^{-12}\) of the largest; \(V^TAV\) is formed explicitly and factorised without diagonal shift. The enriched partition-of-unity (PU) spaces carry 12 DOFs per vertex of a \(Q_1\) grid through fields \(\sum_vN_v(x)[a_v+B_v(x-x_v)]\). Their generating functions are linearly dependent, so the column count can exceed the dimension of the spanned space, and the PU entries are numerical observations rather than verified Galerkin projections. NICE uses \(Q_1(17)\).

| Coarse space | Coarse columns: M1 / M2 / U1 | M1 | M2 | U1 |
| --- | --- | ---: | ---: | ---: |
| \(Q_1\), \(9^3\) vertices | 1,275 / 984 / 1,728 | 0.813 | 0.135 | 0.0966 |
| \(Q_1\), \(17^3\) vertices (NICE) | 5,601 / 4,557 / 7,950 | 0.186 | 0.0273 | 0.0267 |
| \(Q_1\), \(33^3\) vertices | 27,207 / 23,199 / 40,983 | 0.0477 | 0.00870 | 0.00784 |
| \(Q_2\), \(17^3\) nodes | 7,401 / 5,751 / 10,452 | 0.0900 | 0.0170 | 0.0125 |
| PU, \(9^3\) vertices | 5,100 / 3,936 / 6,912 | 0.114 | 0.0211 | 0.0154 |
| PU, \(17^3\) vertices | 22,404 / 18,228 / 31,800 | 0.0365 | 0.00760 | 0.00653 |

## Table ST07. Energy error of different starting fields under the same correction

Mean energy error (%) under traction loads (32 validation test displacements, fixed retained displacements) after the correction with \(k\) smoothing steps before and after the \(Q_1(17)\) coarse-grid correction, for three starting fields and six detailed cells. The 8-step entries of M1, M2, H1, H2 and U2 are those of Table 4. Harmonic and zero starting fields receive the exact rigid-body split.

The harmonic starting field recovers the interior displacements from the deformation part of the retained displacements by graph-harmonic interpolation. The graph has the active nodes of the cell as vertices and joins every pair of the 27 nodes of each active element, with the element's material volume as weight (summed over the elements that share a pair); \(L=D_W-W\) is its Laplacian. With the retained values prescribed, each displacement component is recovered separately from \(L_{II}u_I=-L_{IP}q\). As for the zero field, the retained displacement is first split as \(q=R_Pc+(q-R_Pc)\) with \(c=R_P^{+}q\); only the second part is recovered in this way, the rigid field \(Rc\) is added, and the retained values are restored. This recovery has no trainable parameters and uses only the element connectivity and volumes.

| cell | starting field | 8 steps | 16 steps | 32 steps | 64 steps |
|---|---|---|---|---|---|
| H2 | Base network | 0.0117 | 0.00163 | 0.000182 | 3.65e-06 |
| H2 | harmonic | 0.0806 | 0.00482 | 0.000614 | 1.22e-05 |
| H2 | zero | 28.3 | 0.284 | 0.0156 | 0.000315 |
| H1 | Base network | 0.0131 | 0.00487 | 0.0013 | 0.000173 |
| H1 | harmonic | 1.28 | 0.547 | 0.199 | 0.0345 |
| H1 | zero | 39.9 | 10.7 | 2.92 | 0.437 |
| M2 | Base network | 0.0273 | 0.0135 | 0.00691 | 0.00351 |
| M2 | harmonic | 6.72 | 3.53 | 1.52 | 0.485 |
| M2 | zero | 324 | 104 | 38.9 | 14.1 |
| M1 | Base network | 0.186 | 0.12 | 0.0727 | 0.0393 |
| M1 | harmonic | 17.2 | 8.13 | 4.11 | 1.43 |
| M1 | zero | 1.1e+03 | 420 | 185 | 81.8 |
| U1 | Base network | 0.0267 | 0.0176 | 0.0118 | 0.00783 |
| U1 | harmonic | 4.19 | 2.65 | 1.87 | 1.33 |
| U1 | zero | 133 | 47.7 | 22 | 11.9 |
| U2 | Base network | 0.00424 | 0.00202 | 0.0009 | 0.000377 |
| U2 | harmonic | 1.23 | 0.693 | 0.38 | 0.194 |
| U2 | zero | 47.2 | 14 | 5.51 | 2.32 |

On M1, from the base network, the same cycle with 2 and 4 steps per stage gives 1.02% and 0.422%, and a coarse-grid correction followed by a single eight-step stage gives 0.230%. The \(Q_1(17)\) space of M1 has 5,601 coarse columns for 165,927 interior DOFs.
## Table ST08. Complete results of the two-cell assemblies with a continuous-thickness neighbour

Maximum relative errors (%) of the compliance and of the sensitivity vector over the six face loads, compared with the 3% reference. Sensitivity maxima include both cells. The test cell uses the stated learned variant, and its neighbour is exact. Cell labels abbreviate the validation identifiers (Supplementary Section R1).

| Test cell | Configuration | Variant | Max. compliance error (%) | Max. sensitivity error (%) |
| --- | --- | --- | --- | --- |
| U1 | x | Base network | 0.465 | 5.830 |
| U1 | y | Base network | 0.427 | 5.390 |
| U2 | x | Base network | 0.105 | 1.996 |
| M1 | x | Base network | 4.384 | 12.153 |
| H1 | x | Base network | 1.180 | 2.512 |
| H1 | y | Base network | 1.379 | 1.820 |
| M2 | x | Base network | 0.571 | 2.539 |
| U1 | x | Uncorrected continuation | 0.401 | 4.885 |
| U1 | y | Uncorrected continuation | 0.388 | 4.723 |
| U2 | x | Uncorrected continuation | 0.104 | 1.781 |
| U2 | y | Uncorrected continuation | 0.119 | 1.906 |
| M1 | x | Uncorrected continuation | 4.119 | 11.365 |
| M1 | y | Uncorrected continuation | 3.458 | 10.929 |
| H1 | x | Uncorrected continuation | 1.061 | 2.469 |
| H1 | y | Uncorrected continuation | 1.223 | 1.734 |
| M2 | x | Uncorrected continuation | 0.593 | 2.874 |
| L1 | x | Uncorrected continuation | 0.174 | 1.894 |
| L1 | y | Uncorrected continuation | 0.182 | 1.646 |
| U1 | x | Smoothing-trained | 0.112 | 3.155 |
| U1 | y | Smoothing-trained | 0.111 | 3.166 |
| U2 | x | Smoothing-trained | 0.031 | 0.508 |
| U2 | y | Smoothing-trained | 0.038 | 0.782 |
| M1 | x | Smoothing-trained | 1.363 | 4.432 |
| M1 | y | Smoothing-trained | 1.115 | 3.974 |
| H1 | x | Smoothing-trained | 0.144 | 0.636 |
| H1 | y | Smoothing-trained | 0.237 | 0.561 |
| M2 | x | Smoothing-trained | 0.189 | 1.428 |
| M2 | y | Smoothing-trained | 0.213 | 2.545 |
| H3 | x | Smoothing-trained | 0.106 | 1.290 |
| H3 | y | Smoothing-trained | 0.132 | 0.678 |
| U1 | x | Base network + correction | 0.011 | 0.945 |
| U1 | y | Base network + correction | 0.011 | 0.904 |
| U2 | x | Base network + correction | 0.00123 | 0.142 |
| U2 | y | Base network + correction | 0.0014 | 0.158 |
| M1 | x | Base network + correction | 0.079 | 0.350 |
| M1 | y | Base network + correction | 0.066 | 0.339 |
| H1 | x | Base network + correction | 0.010 | 0.139 |
| H1 | y | Base network + correction | 0.013 | 0.105 |
| M2 | x | Base network + correction | 0.00492 | 0.083 |
| M2 | y | Base network + correction | 0.0034 | 0.075 |
| H3 | x | Base network + correction | 0.00609 | 0.132 |
| H3 | y | Base network + correction | 0.00894 | 0.202 |
| L1 | x | Base network + correction | 0.00193 | 0.065 |
| L1 | y | Base network + correction | 0.00181 | 0.087 |
| U1 | x | NICE | 0.00614 | 0.598 |
| U1 | y | NICE | 0.00681 | 0.669 |
| U2 | x | NICE | 0.00121 | 0.082 |
| U2 | y | NICE | 0.00159 | 0.073 |
| M1 | x | NICE | 0.056 | 0.161 |
| M1 | y | NICE | 0.048 | 0.206 |
| H1 | x | NICE | 0.0076 | 0.098 |
| H1 | y | NICE | 0.011 | 0.097 |
| M2 | x | NICE | 0.00595 | 0.136 |
| M2 | y | NICE | 0.00472 | 0.154 |
| H3 | x | NICE | 0.0064 | 0.124 |
| H3 | y | NICE | 0.010 | 0.196 |
| L1 | x | NICE | 0.00202 | 0.101 |
| L1 | y | NICE | 0.00197 | 0.078 |

Combinations without a row were not evaluated. All cells in this table belong to the 20 validation geometries used for checkpoint selection (Section 5.1). Table ST09 gives the responses of the same configurations under cut-surface tractions. For the base network on U1/x under the z traction on the neighbour face (circled in Figure S04(a–c)), the test cell carries 0.1274669% of the exact assembled energy and has a local energy error of 2.312168%. The product \(\beta=w\varepsilon\) is 0.002947249%, the compliance error 0.002827394% and the test-cell sensitivity error 5.82992%.

#### ST08b. Held-out two-cell configurations (NICE)

The cells lie outside checkpoint selection: the five validation geometries with NICE's largest single-cell errors and one randomly chosen evaluable cell per cut-severity group. Entries are maximum relative errors (%), over the six face loads for the compliance and over both cells for the thickness sensitivity. As in Table ST08, the neighbour is exact and the test cell learned. Cell labels are defined in Supplementary Section R1.

| Cell | Selection | Remaining volume | x: compliance | x: sensitivity | y: compliance | y: sensitivity |
| --- | --- | --- | --- | --- | --- | --- |
| W1 | largest error, rank 1 | 0.571 | 0.2706 | 1.485 | 0.2220 | 1.312 |
| W2 | largest error, rank 2 | 0.267 | 0.1982 | 0.815 | 0.1703 | 1.344 |
| W3 | largest error, rank 3 | 0.287 | 0.1587 | 1.329 | 0.2489 | 1.419 |
| W4 | largest error, rank 4 | 0.266 | 0.0827 | 0.526 | 0.0753 | 0.971 |
| W5 | largest error, rank 5 | 0.681 | 0.0967 | 0.434 | 0.0878 | 0.563 |
| RU | random, uncut | 1.000 | 0.0010 | 0.181 | 0.0011 | 0.133 |
| RL | random, light | 0.779 | 0.0038 | 0.102 | 0.0051 | 0.102 |
| RM | random, moderate | 0.661 | 0.0101 | 0.103 | 0.0066 | 0.100 |
| RH | random, heavy | 0.078 | 0.0050 | 0.136 | 0.0066 | 0.218 |


## Table ST09. Responses under cut-surface tractions

Maximum relative errors (%) over the three cut-surface traction directions, for every variant and configuration of Table ST08 with a cut test cell. Combinations without a row were not evaluated. Sensitivity maxima include both cells. The test cell is learned and the neighbour exact. As in the comparison of cut-surface tractions in Section 5.6, values above the common 3% line are in bold.

| Variant | Cell | Configuration | Compliance error (%) | Sensitivity error (%) |
| --- | --- | --- | --- | --- |
| Base network | M1 | x | 2.880 | **9.924** |
| Base network | H1 | x | 0.445 | 2.066 |
| Base network | H1 | y | 0.651 | 1.275 |
| Base network | M2 | x | 0.560 | **3.155** |
| Uncorrected continuation | M1 | x | 2.768 | **9.827** |
| Uncorrected continuation | M1 | y | **6.870** | **14.053** |
| Uncorrected continuation | H1 | x | 0.399 | 1.825 |
| Uncorrected continuation | H1 | y | 0.613 | 1.200 |
| Uncorrected continuation | M2 | x | 0.574 | **3.252** |
| Uncorrected continuation | L1 | x | 0.355 | 1.660 |
| Uncorrected continuation | L1 | y | 1.141 | **3.702** |
| Smoothing-trained | M1 | x | 0.713 | **3.228** |
| Smoothing-trained | M1 | y | 2.417 | **5.191** |
| Smoothing-trained | H1 | x | 0.062 | 0.356 |
| Smoothing-trained | H1 | y | 0.130 | 0.587 |
| Smoothing-trained | M2 | x | 0.200 | 1.664 |
| Smoothing-trained | M2 | y | 1.117 | **3.924** |
| Smoothing-trained | H3 | x | 0.052 | 0.689 |
| Smoothing-trained | H3 | y | 0.237 | 1.354 |
| Base network + correction | M1 | x | 0.049 | 0.217 |
| Base network + correction | M1 | y | 0.124 | 0.305 |
| Base network + correction | H1 | x | 0.0044 | 0.096 |
| Base network + correction | H1 | y | 0.00709 | 0.022 |
| Base network + correction | M2 | x | 0.00468 | 0.053 |
| Base network + correction | M2 | y | 0.020 | 0.091 |
| Base network + correction | H3 | x | 0.00391 | 0.038 |
| Base network + correction | H3 | y | 0.011 | 0.042 |
| Base network + correction | L1 | x | 0.00479 | 0.039 |
| Base network + correction | L1 | y | 0.011 | 0.069 |
| NICE | M1 | x | 0.035 | 0.161 |
| NICE | M1 | y | 0.101 | 0.158 |
| NICE | H1 | x | 0.00338 | 0.061 |
| NICE | H1 | y | 0.00587 | 0.126 |
| NICE | M2 | x | 0.00603 | 0.072 |
| NICE | M2 | y | 0.027 | 0.154 |
| NICE | H3 | x | 0.00313 | 0.125 |
| NICE | H3 | y | 0.00982 | 0.129 |
| NICE | L1 | x | 0.00435 | 0.087 |
| NICE | L1 | y | 0.011 | 0.071 |

## Supplementary Note S1. Verification of the reference and of the thickness derivative

Figure S01 and Table ST10 verify the CutFEM reference under refinement of the background grid and under changes of the ghost-penalty coefficient. Table ST10b compares the reference with a body-fitted discretisation on uncut cells. The last paragraph of this note verifies the numerical thickness derivative.

For U1, M1 and M2, the compliance at \(n=32\) differs from the finest level of Figure S01(a) by at most 0.057% and the thickness sensitivity by at most 0.11%. H1 is clamped on \(x=0\) and loaded on \(y=0\), because its remaining part has no material on its \(z\)-faces. At \(n=32\) it differs from \(n=48\) by up to 0.99% in compliance, for the load normal to the loaded face, and by 0.97% in sensitivity, but from \(n=64\) by only 0.07% and 0.08%. Between \(n=48\) and \(n=64\) the difference is 1.08%, so H1 does not converge monotonically (Figure S01(d), Table ST10). On U1, M1, M2 and H1, varying the ghost-penalty coefficient between \(10^{-5}\) and \(10^{-3}\) changes the compliance by at most 0.053% and the sensitivities by at most 0.12%. At \(\gamma=10^{-3}\), the penalty energy is \(1.2\times10^{-4}\) to \(5.6\times10^{-4}\) of the total. Refining the volume integration changes both quantities by at most 0.010%.

### Table ST10. Refinement of the reference to \(n=64\)

Single cells are clamped on one cell face and loaded on another face by unit traction loads in the three coordinate directions, and they are solved at \(n=24\) to 64 by a sparse direct solver. H2 is loaded instead by a body force with unit resultant, because its only cell face that carries material is the clamped one. Each entry gives the signed compliance deviation from \(n=64\) with the largest magnitude over the three loads, followed after the slash by the largest relative deviation of the thickness sensitivity vector, both in percent. H2 could not be meshed at \(n=56\). Elsewhere in this work, the reference solutions are computed at \(n=32\).

| Cell | \(n=24\) | \(n=32\) | \(n=40\) | \(n=48\) | \(n=56\) |
| --- | --- | --- | --- | --- | --- |
| H1 | -1.05 / 1.06 | +0.07 / 0.08 | +0.46 / 0.44 | +1.08 / 1.06 | +1.40 / 1.37 |
| H2 | -1.58 / 5.60 | -0.93 / 3.59 | -0.52 / 2.16 | -0.27 / 1.24 | — |
| Validation cell with the largest NICE error | -0.32 / 0.57 | -0.14 / 0.23 | +0.08 / 0.09 | -0.14 / 0.15 | -0.02 / 0.03 |
| Validation cell with the thinnest walls | -0.66 / 1.02 | -0.28 / 0.44 | -0.16 / 0.24 | -0.09 / 0.20 | +0.17 / 0.16 |

On uncut cells, the discrete model was also compared with an independent body-fitted discretisation (Table ST10b). This discretisation meshes the same implicit geometry with quadratic tetrahedra and shares no geometry, quadrature or stabilisation code with the discrete model.

### Table ST10b. Comparison with a body-fitted quadratic-tetrahedral discretisation on uncut cells

| Configuration | Loads | CutFEM vs finer body-fitted mesh: max. compliance difference (%) | Body-fitted discretisation: change between its two mesh levels (%) |
| --- | --- | ---: | ---: |
| Four single uncut cells | Six fixture load cases | 0.084–0.101 | 0.130–0.273 |
| One uncut cell | Twelve local tractions on finite areas | 0.253 | 0.502 |
| Assemblies of uncut cells: two cells, 2×2×2, 1×1×4 | Fixture load cases | 0.052–0.090 | 0.108–0.190 |

Figure S01(c) reports a step-refinement study of the numerical thickness derivative on four cells. Relative to the step used in the computations, \(10^{-5}\tau_c\), the sensitivity changes by at most \(2.6\times10^{-7}\) at \(10^{-3}\tau_c\), \(2.5\times10^{-9}\) at \(10^{-4}\tau_c\) and \(1.6\times10^{-10}\) at \(10^{-6}\tau_c\). The hundredfold reduction per decade is the second-order truncation error of the central difference. It is consistent with a derivative that is smooth within \(\pm10^{-3}\tau_c\) at the fixed active set on these cells. On the same cells, the central-difference stiffness derivative agrees to \(4\times10^{-8}\) with differences of the compliance recomputed at the perturbed designs. An element-level check over the partially filled elements of the eight detailed cells shows that the discrete moments do not everywhere preserve the nesting on which Eq. (H.7) relies. The element stiffness matrices are positive semidefinite to rounding. However, the exact derivative of the discrete moments at fixed clipping topology gives element derivative matrices with negative eigenvalues. In the uncut cells, the smallest ratio \(\lambda_{\min}/\max|\lambda|\) is \(-2.3\times10^{-3}\). In four of the six cut cells, between 2 and 11 partially filled elements (of 762 to 6,033) have a uniform-thickening derivative whose most negative eigenvalue exceeds \(10^{-6}\) of its largest eigenvalue in magnitude. These derivatives are indefinite, with a largest eigenvalue of at least \(10^{-3}\) of the largest magnitude. The central difference used in the computations agrees with this exact discrete derivative to \(2\times10^{-8}\).

## Supplementary Note S2. Ablation: Bernstein-restricted cell faces

This note gives the complete results of the ablation summarised in Section 5.7. Every cell uses its exact condensed stiffness matrix (the Schur complement), and only the representation of the retained cell-face displacements is reduced.

### S2.1. Restricted Galerkin system

The restricted space uses tensor-product Bernstein polynomials of degree \(r\) on the selected cell faces of each cell. Its global map \(G_r\) respects the DOFs shared between cells. DOFs of the cut-plane elements that do not lie on the cell faces keep identity columns. With the supports applied, the Galerkin system is

\[
\mathbb K_r=G_r^T\mathbb K G_r,\qquad
f_r=G_r^Tf_g,\qquad U_r=G_r\mathbb K_r^{-1}f_r.
\tag{S2.1}
\]

The comparison evaluates the exact condensed stiffness matrices of the cells within this restricted space of cell-face displacements. The interior is therefore exact, and every error comes from the restricted boundary. Two cases are examined: every cell face restricted, as in a lattice built entirely from such substructures, and only the shared interface restricted.

Boundary restriction and the interior correction \(\mathcal W\) act on different displacement spaces. Exact condensation followed by the restriction \(U=G_ry\) solves over a subspace of the space of retained displacements. By the principle of minimum potential energy, the compliance of this restricted solution does not decrease when the boundary space is enlarged to a space that contains it. The correction keeps every retained DOF fixed and changes the interior displacements of \(u=FBU\). Its ordering holds because the two-grid cycle that defines \(H\) does not increase the energy error (Section 3.4), although two such interior spaces need not be nested. In neither case is a norm of the local sensitivity ordered.

### S2.2. Conditions

- Pairs in configuration x (Figure 9; Supplementary Note S4), each test cell with its exact neighbour of continuous thickness, for U1, M1, M2 and H1; the interface-only case was run for U1, M1 and M2.
- One cell per substructure, the fine-scale face traction loads used throughout, and no oversampling.
- Loads: the three tractions on the test-cell face and the three tractions on the neighbour face.
- The DOFs of the cut-plane elements that are not on the cell faces remain unrestricted, which favours the restricted model (controlled DOFs in Table ST11).
- The comparison concerns accuracy only and does not compare accuracy at equal cost.

### Table ST11. Bernstein restriction of the cell-face displacements

Each test cell and its neighbour of continuous thickness are assembled in configuration x with exact condensed stiffness matrices. Degree r applies to the restricted cell faces; the DOFs of the cut-plane elements keep their identity representation. Controlled DOFs include these unrestricted DOFs of the cut-plane elements. All errors are maxima over the stated load set (%), relative to the solution with all retained DOFs; the sensitivity columns refer to the test-cell vector. Test-face loads are the three unit traction loads on the loaded face of the test cell, and the six-load set adds the three tractions on the neighbour face. The maxima of the two load sets need not occur under the same load.

#### ST11a. Every cell face restricted

| Cell | r | Controlled DOFs | Test-face compliance (3 loads) | Test-face sensitivity (3 loads) | Six-load compliance (6 loads) | Six-load sensitivity (6 loads) |
| --- | --- | --- | --- | --- | --- | --- |
| U1 | 1 | 24 | 76.998 | 89.056 | 82.609 | 203.566 |
| U1 | 2 | 102 | 56.944 | 62.055 | 56.944 | 317.415 |
| U1 | 3 | 240 | 38.907 | 47.849 | 38.907 | 398.644 |
| U1 | 5 | 696 | 3.926 | 3.603 | 3.961 | 114.545 |
| U1 | 8 | 1,830 | 0.485 | 1.467 | 0.485 | 24.435 |
| M1 | 1 | 15,423 | 77.749 | 88.966 | 78.485 | 294.272 |
| M1 | 2 | 15,498 | 53.133 | 62.622 | 53.133 | 402.365 |
| M1 | 3 | 15,627 | 33.998 | 45.584 | 33.998 | 421.766 |
| M1 | 5 | 16,047 | 3.930 | 4.748 | 3.930 | 138.963 |
| M1 | 8 | 17,082 | 0.521 | 1.295 | 0.611 | 39.320 |
| M2 | 1 | 15,648 | 76.234 | 84.796 | 85.269 | 244.968 |
| M2 | 2 | 15,723 | 49.547 | 41.050 | 50.436 | 606.013 |
| M2 | 3 | 15,852 | 29.124 | 19.138 | 33.554 | 747.730 |
| M2 | 5 | 16,272 | 3.413 | 4.275 | 3.634 | 334.483 |
| M2 | 8 | 17,307 | 0.505 | 4.458 | 0.586 | 38.693 |
| H1 | 1 | 12,888 | 81.713 | 87.320 | 81.713 | 333.566 |
| H1 | 2 | 12,939 | 37.528 | 29.960 | 41.962 | 188.527 |
| H1 | 3 | 13,026 | 21.660 | 18.637 | 30.364 | 340.008 |
| H1 | 5 | 13,308 | 2.784 | 2.890 | 3.860 | 171.306 |
| H1 | 8 | 14,001 | 0.608 | 1.520 | 0.735 | 64.044 |

#### ST11b. Only the shared interface restricted

| Cell | r | Controlled DOFs | Test-face compliance (3 loads) | Test-face sensitivity (3 loads) | Six-load compliance (6 loads) | Six-load sensitivity (6 loads) |
| --- | --- | --- | --- | --- | --- | --- |
| U1 | 1 | 25,386 | 1.947 | 3.599 | 1.947 | 85.982 |
| U1 | 2 | 25,401 | 0.667 | 4.905 | 0.667 | 29.133 |
| U1 | 3 | 25,422 | 0.169 | 0.338 | 0.169 | 7.999 |
| U1 | 5 | 25,482 | 0.010 | 0.328 | 0.036 | 2.194 |
| U1 | 8 | 25,617 | 0.001 | 0.003 | 0.002 | 0.359 |
| M1 | 1 | 37,080 | 9.355 | 10.397 | 9.355 | 81.909 |
| M1 | 2 | 37,095 | 1.915 | 4.529 | 1.915 | 36.640 |
| M1 | 3 | 37,116 | 0.440 | 1.106 | 0.440 | 9.954 |
| M1 | 5 | 37,176 | 0.073 | 0.335 | 0.073 | 2.174 |
| M1 | 8 | 37,311 | 0.011 | 0.031 | 0.011 | 0.615 |
| M2 | 1 | 35,196 | 4.228 | 8.397 | 4.228 | 88.562 |
| M2 | 2 | 35,211 | 0.728 | 3.825 | 0.728 | 18.427 |
| M2 | 3 | 35,232 | 0.261 | 1.058 | 0.261 | 21.216 |
| M2 | 5 | 35,292 | 0.030 | 1.004 | 0.030 | 3.366 |
| M2 | 8 | 35,427 | 0.009 | 0.049 | 0.009 | 1.537 |

With all retained DOFs and the supports applied, the systems have 28,206 (U1), 39,996 (M1), 39,120 (M2) and 32,991 (H1) free DOFs.
## Supplementary Note S3. Cost records: whole-lattice direct solution and per-cell condensation

This note defines the three routes of Table 5 and records their phases (Table ST12). It also compares the per-cell cost of conventional condensation and NICE (Table ST13). The four-cell lattices are the two layers \(z=0\) and \(z=1\) of the \(2\times2\times2\) block of Section 5.8, with their corner thickness parameters unchanged. Each layer holds two uncut cells and two cut cells with remaining volume fractions 0.616 and 0.252. All lattices are clamped on the face \(y=\min\) and loaded on the face \(y=\max\) by unit traction loads along the three Cartesian axes. Routes (a) and (b) also solve three random loads.

Route (a), the whole-lattice direct solution, assembles the full cut-cell stiffness of every cell, on its retained and interior degrees of freedom, into one global matrix. The retained degrees of freedom are numbered and coupled exactly as in the learned lattice, with the same clamp, free set and load vectors. The interior degrees of freedom of each cell follow the free retained ones. The matrix is scaled symmetrically by its diagonal and factorised by MKL PARDISO 2026.1 on the CPU. The factorisation is a symmetric positive definite Cholesky factorisation of the upper triangle (mtype 2), with nested-dissection ordering (iparm(2) = 3) and 16 threads. The PARDISO phases are called directly with explicitly set parameters, and all six loads are solved at once. Relative residuals \(\|Ku-f\|/\|f\|\) are recomputed with the unscaled matrix.

Route (b), conventional exact condensation, processes the cells one at a time on the CPU with 16 threads. Cell setup and assembly are as in route (a). PARDISO's Cholesky factorisation with the Schur-complement option (iparm(36)) then gives the dense condensed matrix of the cell on its retained degrees of freedom, after which the cell is released. The condensed lattice system is solved exactly by a block Cholesky factorisation in substructuring order. For each cell, its private retained block is factorised and eliminated with dense kernels. The dense interface matrix over the retained DOFs shared by several cells is then factorised and solved, and the private DOFs are recovered by back-substitution. The condensed cell matrices are held until their elimination.

Route (c) runs one lattice analysis with sensitivities in the deployed implementation of NICE on the GPU. Cell preparation comprises cell construction, stiffness and moment assembly, network input and encoding, and a first application that prepares the correction. The lattice and \(\mathbb K_{PP}\) are then assembled, and the preconditioner is set up (the balanced two-level action of Supplementary Note S4.1). Conjugate gradients solve the three traction loads to a recursive relative residual of \(10^{-6}\); the residuals reached are \(9.0\times10^{-7}\) to \(9.9\times10^{-7}\). Finally, the sensitivities \(\widetilde s_c\) of Eq. (17) for the three loads are obtained by reverse-mode differentiation of the moment integrals at the fixed recovered displacement fields. Route (c) uses the arithmetic of the timed route of Table 1. The network and the smoothing and coarse solve of the correction run in single precision, and the stiffness products of the condensed action in double precision (Appendix F.2). The three routes read the same generated cell geometries; geometry generation is not timed.

Whole-lattice iterative solvers (Table ST12e) use the global matrix, load vectors, clamp and numbering of route (a). These are exported once and solved with PETSc 3.26 [Balay et al. (2026)](https://petsc.org/release/manual/) with 16 MPI processes on the CPU used for routes (a) and (b). Conjugate gradients run on the unscaled system and stop on the unpreconditioned relative residual \(\|Ku-f\|/\|f\|\le10^{-9}\). The iterate is recorded where the residual first falls below \(10^{-3}, 10^{-4}, \dots\), together with its compliance and energy-norm difference from the final iterate. GAMG (smoothed aggregation, PETSc defaults) uses block size 3 and the six rigid-body modes of the DOF positions as near-null space. BoomerAMG uses HMIS coarsening, extended+i interpolation, nodal coarsening, strong threshold 0.5 and its default relaxation. The BDDC configuration considered has exact local solvers and one subdomain per cell. The Dirichlet matrix of a cell is its stiffness on the DOFs it shares with no other cell, and its Neumann matrix is the whole cell stiffness on its free DOFs. Each is factorised once, cell after cell, with the PARDISO configuration of route (a) (symmetric Jacobi scaling, real SPD Cholesky, tuned iparm, 16 threads). The singular Neumann matrices of floating cells are shifted by \(10^{-10}\) on the diagonal of the scaled matrix. The sum of these factorisations is a reference for the local factorisation cost of BDDC with exact local solvers in this configuration. It does not include the treatment of the primal constraints and the null space in the local Neumann problems, the coarse problem or the iterations. With MUMPS for the local and coarse problems on eight processes, BDDC with all local factors resident exceeded 90 GiB during its setup.

### Table ST12. Whole-lattice direct solution, conventional exact condensation and learned route: dimensions, phases and memory

Times in s. Memory in GiB (\(2^{30}\) bytes). PARDISO memory is its permanent plus factorisation storage (iparm(16) + iparm(17)) reported by the analysis phase, with kilobytes taken as 1024 bytes. Totals, iteration counts, peak process memory and the memory of route (c) are given in Table 5.

#### ST12a. Lattice dimensions

| Lattice | Cells (cut) | Cut-plane-element DOFs off the cell faces (fraction of free retained DOFs) |
| --- | --- | --- |
| 2×2×1, z=0 | 4 (2) | 35,829 (46%) |
| 2×2×1, z=1 | 4 (2) | 33,327 (47%) |
| 2×2×2 | 8 (4) | 69,156 (50%) |
| 3×3×1 | 8 (3) | 53,085 (37%) |

Total and free retained DOFs: Table 5.

#### ST12b. Route (a), direct solution on the CPU (Cholesky, 16 threads and one thread): phases (s) and memory (GiB)

| Lattice | Threads | Cells: setup + assembly | Global assembly and scaling | Analysis | Factorisation | Solution, 6 loads | Total | PARDISO memory | Max. relative residual |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 2×2×1, z=0 | 16 | 229.9 | 21.9 | 26.0 | 193.1 | 13.5 | 484.4 | 32.0 | 3.3e-10 |
| 2×2×1, z=0 | 1 | 777.3 | 19.8 | 115.8 | 1,500.0 | 12.1 | 2,425.0 | 31.4 | 4.0e-10 |
| 2×2×1, z=1 | 16 | 184.2 | 17.6 | 20.2 | 116.9 | 10.4 | 349.2 | 28.8 | 4.5e-10 |
| 2×2×1, z=1 | 1 | 685.6 | 18.8 | 106.1 | 1,309.4 | 11.3 | 2,131.3 | 28.3 | 5.2e-10 |
| 2×2×2 | 16 | 379.5 | 36.0 | 53.1 | 368.2 | 31.0 | 867.8 | 66.4 | 1.5e-10 |
| 2×2×2 | 1 | 1,402.1 | 39.3 | 254.5 | 4,140.3 | 27.7 | 5,863.9 | 65.5 | 1.8e-10 |
| 3×3×1 | 16 | 431.3 | 42.3 | 50.4 | 481.5 | 36.0 | 1,041.6 | 80.9 | 2.6e-09 |
| 3×3×1 | 1 | 1,669.9 | 45.4 | 324.4 | 6,059.1 | 32.4 | 8,131.0 | 79.8 | 2.8e-09 |

Total: sum of the phases; the 16-thread totals are those of Table 5. With 32 threads, the whole direct solution takes 378.5, 341.1, 823.4 and 955.5 s instead of 484.4, 349.2, 867.8 and 1,041.6 s, because cell setup and assembly do not speed up.

#### ST12c. Route (b), conventional exact condensation on the CPU (16 threads): phases (s) and memory (GiB)

| Lattice | Cell preparation | Cells: Schur complement | Condensed solve | Condensed cell matrices held (GiB) | Compliance vs. (a) |
| --- | --- | --- | --- | --- | --- |
| 2×2×1, z=0 | 210.6 | 274.3 | 86.8 | 17.8 | 5e-10 |
| 2×2×1, z=1 | 263.4 | 279.2 | 71.4 | 14.9 | 4e-10 |
| 2×2×2 | 393.6 | 392.5 | 133.9 | 32.8 | 2e-10 |
| 3×3×1 | 462.0 | 555.0 | 165.7 | 33.4 | 7e-10 |

Condensed solve: private elimination, interface factorisation and solution, and back-substitution for the six loads. The total of Table 5 also includes the per-cell lattice geometry, the matrix scaling and data movement between phases (17 to 35 s). Compliance vs. (a): largest relative difference over the six loads from the whole-lattice direct solution.

#### ST12d. Route (c), learned route: phases of one lattice analysis with sensitivities (s)

| Lattice | Cell preparation | Lattice and \(\mathbb K_{PP}\) assembly | Preconditioner setup | Conjugate-gradient solve, 3 loads | Sensitivities |
| --- | --- | --- | --- | --- | --- |
| 2×2×1, z=0 | 7.50 | 0.08 | 3.78 | 25.63 | 2.12 |
| 2×2×1, z=1 | 6.49 | 0.08 | 3.39 | 25.23 | 1.87 |
| 2×2×2 | 11.82 | 0.17 | 6.57 | 56.37 | 3.80 |
| 3×3×1 | 12.66 | 0.14 | 7.25 | 80.85 | 4.07 |

Total, iterations and memory: Table 5. The phases sum to 0.2–0.5 s less than the totals; the remainder is bookkeeping between phases. GPU memory is the peak allocated by the process; CPU memory is the resident set size after cell preparation. Learned substructures resident on the GPU (Supplementary Note S4.1): all cells of the four-cell lattices and four cells of each eight-cell lattice.

#### ST12e. Whole-lattice iterative solvers on the CPU (16 MPI processes): setup, conjugate gradients and BDDC factorisations (s)

| Lattice | GAMG: setup | GAMG to \(10^{-4}\): iterations / time per load | GAMG to \(10^{-4}\), with setup: 3 loads / 6 loads | GAMG at \(10^{-4}\): max. compliance / energy difference | BoomerAMG: setup / iterations to \(10^{-4}\) / time per load | BDDC factorisations: Dirichlet + Neumann |
| --- | --- | --- | --- | --- | --- | --- |
| 2×2×2 | 35.3 | 90–134 / 52–72 | 221 / 432 | 4.6e-10 / 2.1e-5 | 88.0 / indefinite preconditioner | 207.9 + 241.3 = 449.2 |
| 3×3×1 | 39.2 | 98–123 / 66–84 | 278 / 478 | 1.4e-10 / 1.2e-5 | 96.6 / 250–260 / 405–481 | 245.3 + 277.9 = 523.1 |

Differences are relative to the iterate at a relative residual of \(10^{-9}\), which GAMG reaches in 235–274 (2×2×2) and 204–220 (3×3×1) iterations. Loads: three traction loads, then three random loads, as in route (a). Cell setup and assembly and the global assembly and scaling are taken from route (a) and are common to all fine-scale routes. They are not included (Table ST12b: 416 and 474 s).

### Table ST13. Per-cell cost of conventional condensation and NICE

Four deployment cells, two cut and two uncut (Supplementary Section R1). Cell preparation: geometry preprocessing plus cell setup, moment integration and stiffness assembly, on the GPU for NICE and on the CPU for the conventional route. Condensation: network encoding plus correction setup (smoothing interval and coarse factorisation) for NICE; symbolic analysis and Cholesky factorisation of the interior by MKL PARDISO (16 threads) for the conventional route. Memory, in GiB: GPU memory held by NICE's network state and correction, or PARDISO's factorisation storage (iparm(17)). Neither includes the cell stiffness \(K\), which both routes hold (last column, GPU storage format).

| Cell | DOFs / retained | Cell preparation (s): NICE / conventional | Condensation (s): NICE / conventional | Memory (GiB): NICE state / PARDISO factor | \(K\) (GiB) |
| --- | --- | --- | --- | --- | --- |
| G1 (cut) | 177,507 / 24,636 | 5.3 / 27.3 | 0.72 / 10.5 | 0.74 / 2.89 | 1.39 |
| G2 (cut) | 67,224 / 18,858 | 3.0 / 10.9 | 0.16 / 1.9 | 0.25 / 0.60 | 0.51 |
| G3 (uncut) | 289,494 / 17,508 | 6.1 / 45.0 | 0.78 / 27.3 | 1.20 / 6.18 | 2.32 |
| G4 (uncut) | 404,148 / 25,920 | 6.5 / 47.0 | 1.09 / 48.9 | 1.37 / 9.42 | 2.71 |

## Supplementary Note S4. Global preconditioning, two-cell configurations and residual records

### S4.1. Balanced two-level preconditioner

Let \(\mathbb A\) be the supported assembled operator used in a run, either the exact or learned one. This notation is distinct from the local interior stiffness \(A=K_{II}\). Let
\(\mathbb K_{PP}=\sum_mB_m^TK_{PP,m}B_m\) on the free retained DOFs. The stiffness-block fine action is \(\mathcal B_f=\mathbb K_{PP}^{-1}\).

For a global coarse basis \(Z_g\) on the retained DOFs, the ideal coarse action is
\(Q_g=Z_g(Z_g^T\mathbb A Z_g)^{-1}Z_g^T\), after removal of dependent columns. The balanced action is

\[
\mathcal M^{-1}=Q_g+(I-Q_g\mathbb A)\mathcal B_f(I-\mathbb A Q_g).
\]

The reported coarse basis multiplies the trilinear lattice-vertex functions by three translations and three rotations about each vertex, then restricts the resulting fields to the supported retained DOFs. This global coarse basis differs from the interior basis \(V\) used to correct the network's interior displacements in each cell. The exact and learned lattice solves of Section 5.8, the learned route (c) of Section 5.9 and the learned solves of Section 5.10 use this balanced action with the stiffness-block fine action and this coarse basis.

The coarse matrix \(A_g=Z_g^T\mathbb A Z_g\) is symmetrised and scaled by its diagonal. Eigenvectors whose scaled eigenvalues are positive and exceed \(10^{-10}\) times the largest are kept, and the corresponding normalised columns \(Y_g\) give \(Q_g=Y_gY_g^T\). The stored product \(\mathbb A Y_g\) supplies the two projections.

On the GPU, \(\mathbb K_{PP}\) is factorised once per design iteration by a sparse Cholesky factorisation with NVIDIA cuDSS. The learned substructures are held on the GPU up to a memory budget for resident learned substructures; beyond it, the state of the remaining cells is held in CPU memory and streamed from there. The resident cells or budget of each run are given in the note to Table ST12d, in Table ST16 and in the caption of Table ST20.

### S4.2. Two-cell test configuration

In both configurations of Figure 9, the six face loads are traction loads along the three Cartesian axes on the loaded face of the test cell and the same three loads on the loaded face of the neighbouring cell. The traction loads are obtained by integrating the Q2 surface shape functions. Each nodal load vector is normalised to unit resultant before support elimination, and clamped entries are then eliminated. Three traction loads on the cut surface of the test cell, normalised in the same way, are reported separately when present. The neighbouring cell is uncut. Where the cut plane of the test cell meets the shared face, the pair is therefore a test configuration and not a physically cut specimen.

The pair solves use conjugate gradients preconditioned by the Cholesky factor of the exact assembled reference stiffness, with a relative recursive-residual tolerance of \(10^{-10}\) and at most 400 iterations.

### S4.3. Residual work, gradient checks and difference quotients of the approximate compliance

The lattices of Section 5.8 and their two \(2\times2\times1\) layers are solved with NICE to a recursive residual of \(10^{-10}\), with the correction in double precision. Three quantities are recorded at the final iterate \(\bar U\). The first is the signed residual work \(\bar U^T\rho\), with \(\rho=f_g-y(\bar U)\) and \(y\) the applied learned action. The second is the action–energy inconsistency \(\omega=\sum_m\omega_m\), \(\omega_m=\bar q_m^Ty_m(\bar q_m)-\bar u_m^TK_m\bar u_m\). Here \(\bar q_m=B_m\bar U\), \(\bar u_m=F_m\bar q_m\) and \(y_m\) is the applied action of cell \(m\), so that \(y(\bar U)=\sum_mB_m^Ty_m(\bar q_m)\) and \(\omega\) is the quantity of Appendix C.2. The third is the dual-norm bound on \(|\bar U^T\rho|\) of Appendix C.2, with \(\rho^T\mathbb K^{-1}\rho\) computed with the exact assembled operator. All quantities are relative to the exact compliance. The recomputed relative residual \(\|f_g-\widehat{\mathbb K}\bar U\|/\|f_g\|\) stagnates at \(3.6\times10^{-4}\) (block) and \(3.2\times10^{-3}\) (layer). Running the correction in single instead of double precision changes this residual of the block by less than 1%. It changes the compliance errors of the block by less than \(3\times10^{-9}\) and those of the layer by less than \(4\times10^{-8}\), and the gradient errors of both by less than \(10^{-6}\).

The gradient metrics compare the sensitivities from NICE with the exact ones after aggregation over the shared lattice vertices, that is, after summing the cells' thickness sensitivity components at each vertex. The aggregated vector is the gradient that an optimiser with shared thickness variables receives; loads are held fixed. For the complete derivative of the approximate compliance, every cell's condensed stiffness was rebuilt at \(\tau\pm h\tau_ce_c\) for each corner. The rebuilt stiffness was applied to the solved retained displacements at fixed \(\widehat q\), which gives the action-energy difference quotient \(D_{\mathrm{act}}\). A rebuild counts as switched when, relative to the unperturbed build, it changes any of the following: a binary node indicator; the set of weak-region element or face stencils (stencils containing a weakly connected node, Appendix G.1); the active element set; or the diagonal shift of the coarse factorisation (Appendix F.1).

#### Table ST14. Residual work, dual-norm bound, gradient and difference quotients of the approximate compliance

Face = the three face traction loads; random = the three random loads. Gradient error: \(\|\widetilde{\boldsymbol s}_g-\boldsymbol s_g\|/\|\boldsymbol s_g\|\) over the shared vertex parameters, in percent; cosine over all six loads. \(D_{\mathrm{act}}\): relative error against the exact vertex gradient under the face loads at step \(h\) (in units of \(\tau_c\)), in percent. Switched rebuilds: number of the 49 rebuilds per cell (three steps, both signs and eight corners, plus one unperturbed) that change a discrete choice, range over cells.

| Lattice | \(\max\lvert\bar U^T\rho\rvert/C\) | Dual-norm bound / \(C\) | \(\max\lvert\omega\rvert/C\) | Gradient error, face / random (%) | Smallest cosine | \(D_{\mathrm{act}}\) at \(h=3\times10^{-3}\) / \(10^{-3}\) / \(3\times10^{-4}\) (%) | Switched rebuilds per cell |
| --- | --- | --- | --- | --- | --- | --- | --- |
| \(2\times2\times1\), \(z=0\) | \(2.3\times10^{-8}\) | \(6.8\times10^{-6}\) | \(5.0\times10^{-9}\) | 0.022–0.048 / 0.22–0.27 | 0.9999998 | 0.29 / 0.65 / 1.53 | 31–48 |
| \(2\times2\times1\), \(z=1\) | \(5.3\times10^{-8}\) | \(7.9\times10^{-6}\) | \(9.5\times10^{-9}\) | 0.067–0.130 / 0.29–0.35 | 0.9999993 | 0.21 / 0.27 / 0.35 | 32–47 |
| \(2\times2\times2\) | \(2.9\times10^{-9}\) | \(4.7\times10^{-6}\) | \(2.4\times10^{-9}\) | 0.052–0.069 / 0.24–0.27 | 0.9999996 | 0.18 / 0.29 / 0.59 | 31–48 |
| \(3\times3\times1\) | \(2.5\times10^{-8}\) | \(1.4\times10^{-5}\) | \(7.0\times10^{-9}\) | 0.041–0.088 / 0.22–0.28 | 0.9999996 | 0.19 / 0.23 / 0.35 | 21–48 |

On the lattices, \(D_{\mathrm{act}}\) grows as the step decreases, so the difference quotient does not resolve the complete derivative of the approximate compliance. The same records were made on the fourteen two-cell configurations of the selection cells (Section 5.6). They give \(|\bar U^T\rho|/C\le1.8\times10^{-8}\), a dual-norm bound of at most \(2.0\times10^{-5}\), \(|\omega|/C\le1.7\times10^{-8}\), thickness-sensitivity errors of 0.07–0.67% with cosines of at least 0.999998, and \(D_{\mathrm{act}}\) errors of 0.05–0.27% at \(h=10^{-3}\) that likewise do not decrease with the step.

The accuracy of the two eight-cell lattices in these runs is reported in Section 5.8. The values below are given for the \(2\times2\times2\) block / the \(3\times3\times1\) layer. The compliance errors under the three face loads are 0.0137, 0.0110 and 0.0094% / 0.0147, 0.0134 and 0.0103%, and at most 0.069% / 0.056% under the random loads. The largest thickness-sensitivity error over all cells is 0.144% / 0.144% under the face loads and 0.454% / 0.426% under the random loads. The relative Euclidean difference between the learned and exact assembled retained solutions over all six loads is 0.130% / 0.123%. The cell energy errors \(\varepsilon_m\) at the exact retained displacements are 0.0061–0.026% / 0.0051–0.052% under the face loads and 0.027–0.123% / 0.023–0.077% under the random loads. The excess of \(\beta\) (Eq. (15)) over the lattice compliance error is 0.13–0.28% / 0.21–0.33% of that error under the face loads and 4.4–5.1% / 4.2–5.0% under the random loads. The exact references, assembled from the dense exact condensed matrices, reach recomputed relative residuals of \(9.1\times10^{-11}\) / \(1.3\times10^{-10}\). The corner parameters of the block range from 0.2516 to 0.5350.

One-corner sweeps test the smoothness of the approximate compliance across these switches. In the two-cell configurations M1/x and U1/y, one corner parameter of the test cell was varied over up to ±10% of its value; this was the corner with the largest or the median sensitivity. Every design had its own regenerated geometry and learned substructure. The compliance change between neighbouring designs was compared with the change predicted by the trapezoidal rule from the sensitivities at both ends. The same comparison was made for the exact discrete model at every tenth design. Every interval on M1/x and 24 of the 26 intervals on U1/y change at least one discrete choice. Most of these changes concern the active elements, ghost faces and retained DOFs, which belong to the discrete model itself. On these intervals the approximate compliance departs from the predicted change by up to 0.25% (M1/x), 0.17% and 0.085% (U1/y) of the compliance, and the exact model departs by the same amount. On M1/x the departures are 0.230, 0.252 and 0.228% against 0.228, 0.249 and 0.225%; on U1/y they are identical to three significant digits. On the two U1/y intervals without a switch the departure is \(7\times10^{-8}\). At the reference designs the error of the approximate compliance stays at its bound \(\beta\) (\(6.0\times10^{-4}\) on M1/x, \(7.2\times10^{-5}\) on U1/y). On the checked intervals the dominant departure of the approximate objective is therefore also present in the exact discrete model, which an optimiser driven by exact condensation meets in the same way.

## Supplementary Note S5. Network settings and parameter counts

Table ST15 complements Appendix G with the coefficient-bound groups, the initialisation and the parameter count of every block.

### Table ST15. Coefficient bounds, initialisation and parameter counts

| Item | Setting |
| --- | --- |
| Coefficient bounds \(a_{\max}\) | Fixed per group: gather and scatter role of each of the 8 element, 12 face and 4 weak-region layers (48 groups), and restriction and prolongation of each of the 3 levels (6 groups), plus the knee \(k_b\): 55 stored values, part of the stored model state (definition in Appendix G) |
| Initialisation | Channel maps \(W_{\ell h}\): \(0.5\,\mathcal N(0,1)/\sqrt{32}\); \(W_{\rm in}\): \(\mathcal N(0,1)/\sqrt3\); \(W_{\rm out}\): \(0.1\,\mathcal N(0,1)/\sqrt{32}\); convolution kernels: \(0.5\,\mathcal N(0,1)/\sqrt{27\cdot32}\); position-embedding tables: \(0.1\,\mathcal N(0,1)\); \(\sigma_\ell=1\); \(\lambda_{\rm mix}=0\); affine layers: PyTorch default. The base network is initialised from the parameters of three preceding training stages (notes to Table ST01), the first of which uses the random initialisation above; the continuations start from the base network's selected parameters |
| Parameter counts, geometry branch | Element encoder 12,288; node encoder 9,024; message passing 49,664; face encoder 12,608; position-embedding tables 432; element, face and weak-region coefficient heads 12,928, 15,008 and 10,848; restriction/prolongation heads 12,870; convolution-coefficient heads 37,440. Total 173,110 |
| Parameter counts, displacement branch | Element channel maps 32,768; face channel maps 49,152; weak-region element channel maps 16,384; convolution kernels 331,776; \(W_{\rm in}\) and \(W_{\rm out}\) 96 each; skip scalars 3; \(\lambda_{\rm mix}\) 24. Total 430,299 |
| Stored values | 603,409 trainable parameters and 55 fixed bound values: 603,464 in total |
| Training settings | Notes to Supplementary Table ST01; seed 0 for every run |
## Supplementary Note S6. Design optimisation records

This note gives the settings (S6.1, Table ST16) and the records of the thickness optimisations of Section 5.10. All NICE runs use one GPU. Compliance is given in the units of Section 5.1 (\(E_Y=1\), unit cell) and memory in GiB (\(2^{30}\) bytes). Times are wall-clock times of complete design iterations and include geometry generation, except at iteration 0, whose geometry was generated beforehand.

### S6.1. Optimiser and constraints

Table ST16 lists the settings. The method of moving asymptotes builds one convex separable approximation per analysis from the estimated gradient. It never tests the decrease of \(\widehat C\) along a search direction, where the inconsistency discussed in Section 4.2 would matter. Neither the gradient norm nor a KKT residual is used for stopping, since the estimate is not the gradient of \(\widehat C\) and \(\widehat C\) changes its discrete model between iterations (S6.4). A design iteration differs from an analysis of Table 5 in that it regenerates every cell's geometry and solves for one load instead of three. For case A, a NICE design iteration takes 74 s on average (Table ST17c), against 79 s for the analysis of the same block in Table 5, which needs no geometry generation but solves for three loads.

### Table ST16. Optimiser, constraints and analysis settings

| Item | Setting |
| --- | --- |
| Design variables | Corner thickness parameters at the lattice vertices \(\boldsymbol\tau_g\); \(\boldsymbol\tau_m=\boldsymbol\tau_m(\boldsymbol\tau_g)\) copies the vertex values to the corners of every cell meeting there. Case A: 27 vertices, 18 free; plate: 74 vertices, 64 free |
| Fixed vertices | The vertices in the plane of the loaded face (9 in case A, 10 in the plate), so that the nodal load obtained by consistent integration of the traction load does not change with the design (Appendix H: no load-derivative term) |
| Objective | Compliance \(\widehat C\) under one face traction load of unit resultant, evaluated as the work \(f_g^T\bar U\) of the final conjugate-gradient iterate and divided by its value at the first iteration |
| Gradient | Sensitivity estimate of Section 5.10, from one reverse-mode pass through the moment integrals at the fixed recovered displacement field; the complete derivative of the approximate compliance, Eq. (18), is not used. Exact twin: exact sensitivities from the exact displacement field with the moment derivatives of Eq. (H.3) |
| Volume constraint | \(V/V^*-1\le0\); \(V\) = sum of the zeroth element moments of all cells (material volume of the discrete model), its derivative from the same reverse pass (exact twin: Eq. (H.3)). \(V^*=0.8\,V(\boldsymbol\tau^0)\): 1.03558 (case A), 3.65526 (plate) |
| Corner-span constraint | \((\tau_a-\tau_b)/0.45-1\le0\) for every ordered pair of distinct corner vertices of every cell (316 ordered pairs, 276 of them involving a free vertex, in case A; 938 and 896 in the plate); pairs of two fixed vertices are constant and are not passed to the optimiser; training limit 0.47 (Table ST02) |
| Gradient-norm constraint | \(\lvert\nabla\tau\rvert^2/0.45^2-1\le0\) at every corner of every cell, from the three edge differences (64 corner stencils in case A, 192 in the plate, each with a free vertex); for a trilinear field the largest gradient norm over the cell is attained at a corner; training limit 0.47 |
| Bounds | \(0.18\le\tau\le0.69\); training range [0.1752, 0.6993] (Table ST02) |
| Optimiser | Method of moving asymptotes [Svanberg (1987)](https://doi.org/10.1002/nme.1620240207), one update per analysis, no line search and without the conservativeness test of its globally convergent variant GCMMA; subproblem solved by the primal–dual interior-point method of Svanberg (2007, Section 5), the barrier parameter reduced tenfold from 1 to \(10^{-7}\) (at each value, Newton steps until the largest residual is below 0.9 of it, at most 200); constants of the standard MMA problem form: \(a_0=1\), \(a_i=0\), \(c_i=1000\), \(d_i=1\) |
| Asymptotes | Initially \(x\pm0.5(x_{\max}-x_{\min})\) (first two iterations), then widened by 1.2 where consecutive steps have the same sign and narrowed by 0.7 where they alternate, kept between 0.01 and 10 variable ranges from \(x\); subproblem bounds at 0.1 of the distance to the asymptotes |
| Move limit | 0.05 of the variable range, 0.0255 per iteration |
| Stopping rule | \(\max\lvert\Delta\tau\rvert<10^{-3}\), or relative objective change below \(10^{-4}\) in three consecutive iterations, or 60 iterations |
| Geometry | Every iteration regenerates the geometry of every cell from its current corner parameters with the geometry generator used for all cells of this study, in parallel processes on the CPU (8 for case A, 12 for the plate), and rebuilds every learned substructure |
| Lattice solve | Preconditioned conjugate gradients with the balanced two-level preconditioner of Supplementary Note S4.1 to a recursive relative residual of \(10^{-6}\) (at most 3,000 iterations), in the arithmetic of the timed route of Table 5, with the coarse-factor pivot rule of Appendix F.1; a run stops if a solve reaches 3,000 iterations or a recomputed relative residual above \(10^{-2}\). Exact twin: dense exact condensed matrices of every cell, assembled solve to \(10^{-10}\) |
| Warm start | From the previous iteration's solution, matched DOF by DOF on absolute grid position, displacement component and the private flag of the cut-plane elements; unmatched DOFs start at zero; the start is scaled by the energy-optimal factor \(f_g^TX_0/(X_0^T\widehat{\mathbb K}X_0)\); the stopping criterion, relative to \(\lVert f_g\rVert\), is unchanged. At iteration 0 of every run the solve started from zero. Exact twin: cold start |
| GPU memory budget for resident learned substructures (Supplementary Note S4.1) | Case A: all cells resident; plate: cells resident while the allocated GPU memory stays below 22 GiB; scale study: 4 GiB (Table ST20) |
| Geometry-generation fallback | If generation fails for some cells, the free vertices of those cells are multiplied by \(1+\epsilon\), \(\epsilon=10^{-4},-10^{-4},10^{-3},-10^{-3},3\times10^{-3},-3\times10^{-3}\) in turn (clipped to the bounds), every cell sharing them is regenerated, and the perturbed design is analysed and continued from. Applied four times over all runs (Table ST19a) |

The designs analysed on the fine scale are all iterations of the runs of Tables ST17 and ST18 and the homogenisation design. Over these designs, the largest corner span is 0.4500, the largest gradient norm 0.4518 and the corner parameters lie in [0.1800, 0.6900], so every analysed cell lay within the training limits.

### S6.2. Case A: NICE optimisation, exact twin and exact checks

Case A is the \(2\times2\times2\) block of Section 5.8, loaded by a traction load in \(y\) on the face \(y=\max\) opposite the clamp. The exact twin runs the same optimisation from the same start, with the same settings, constraints and geometry regeneration, but with exact condensation (Table ST16). The NICE run was checked with the exact model at iterations 0, 12 and 23 (Table ST17b). The twin serves as a verification run and is not used as a cost baseline; the cost comparison of the exact and learned routes is that of Section 5.9.

### Table ST17. Case A: NICE optimisation, exact twin and exact checks

#### ST17a. Iteration history

| Iteration | \(\widehat C\) (NICE) | \(C\), exact twin | NICE vs twin (%) |
| --- | --- | --- | --- |
| 0 | 24.59081 | 24.59356 | −0.011 |
| 1 | 27.14662 | 27.14994 | −0.012 |
| 2 | 30.16595 | 30.17025 | −0.014 |
| 3 | 33.73640 | 33.74275 | −0.019 |
| 4 | 37.85673 | 37.86557 | −0.023 |
| 5 | 35.69282 | 35.70004 | −0.020 |
| 6 | 33.80530 | 33.81157 | −0.019 |
| 7 | 32.05341 | 32.05955 | −0.019 |
| 8 | 30.48399 | 30.49000 | −0.020 |
| 9 | 29.08392 | 29.08994 | −0.021 |
| 10 | 27.85699 | 27.86145 | −0.016 |
| 11 | 26.84866 | 26.85332 | −0.017 |
| 12 | 26.00158 | 26.00632 | −0.018 |
| 13 | 25.28700 | 25.29168 | −0.018 |
| 14 | 24.78587 | 24.79256 | −0.027 |
| 15 | 24.46873 | 24.47568 | −0.028 |
| 16* | 24.24430 | 24.25329 | −0.037 |
| 17 | 24.14971 | 24.15703 | −0.030 |
| 18 | 24.10947 | 24.11624 | −0.028 |
| 19* | 24.08748 | 24.11211 | −0.102 |
| 20 | 24.10488 | 24.11155 | −0.028 |
| 21 | 24.10448 | 24.11117 | −0.028 |
| 22 | 24.10411 | 24.11081 | −0.028 |
| 23 | 24.10377 | — | — |

\* Design perturbed by the geometry-generation fallback (Table ST19a). NICE vs twin: \((\widehat C-C_{\rm twin})/C_{\rm twin}\) at the same iteration index. The designs coincide (to 4e-14) at iterations 0–4, where every free parameter moves by the move limit or to the lower bound while the volume bound is violated; there the column is the approximate compliance error of the same design. From iteration 5 the corner parameters differ, by at most 8.8e-04. Largest \(\lvert\bar U^T\rho\rvert/\widehat C\) over the NICE run: 1.5e-08.

#### ST17b. Exact checks of the NICE run

| Iteration | Exact \(C\) | \(\widehat C\) | Approximate compliance error (%) | Gradient error (%) | Cosine | Component error / \(\max\lvert g\rvert\): median / 95th percentile / max | Sign agreement | Exact solve: PCG iterations / recomputed residual |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 24.59356 | 24.59081 | −0.0112 | 0.069 | 0.9999998 | 2.2e-04 / 5.1e-04 / 6.3e-04 | 1.000 | 178 / 9.1e-11 |
| 12 | 26.00630 | 26.00158 | −0.0182 | 0.170 | 0.9999987 | 4.1e-04 / 1.2e-03 / 1.8e-03 | 1.000 | 244 / 9.0e-11 |
| 23 | 24.11050 | 24.10377 | −0.0279 | 0.329 | 0.9999966 | 3.8e-04 / 2.8e-03 / 4.1e-03 | 1.000 | 262 / 9.9e-11 |

Over the 18 free vertex parameters. Approximate compliance error \((\widehat C-C)/C\); gradient error \(\lVert\widetilde{\boldsymbol s}_g-\boldsymbol s_g\rVert/\lVert\boldsymbol s_g\rVert\); component error \(\lvert\widetilde s_{g,i}-s_{g,i}\rvert/\max_j\lvert s_{g,j}\rvert\).

#### ST17c. Final designs and cost

|  | NICE run | Exact twin |
| --- | --- | --- |
| Exact compliance of the final design | 24.110504 | 24.110806 |
| Final \(\tau\) range; largest corner span; largest gradient norm | 0.1800–0.6239; 0.4439; 0.4500 | 0.1800–0.6236; 0.4436; 0.4500 |
| PCG iterations; recomputed residual | 99–157; 6.2e-05–1.5e-04 | 178–262; 8.5e-11–9.9e-11 |
| Time per iteration, mean (range) (s) | 74 (62–120) | 469 (455–534) |
| Mean phases (s) | geometry 20.8, cell preparation 10.1, lattice and \(\mathbb K_{PP}\) assembly 0.1, preconditioner 5.9, PCG 31.6, sensitivities and volume gradient 5.5, MMA 0.006 | — |

Final designs: exact compliance of the NICE design relative to that of the twin's design −1.25e-05; corner parameters differ by at most 0.0051 (root mean square 0.0011). Iterations and first and last compliances of both runs: Table 6. Mean phases over all iterations; the geometry phase is zero at iteration 0.

### S6.3. Plate supported on its cut and homogenisation design

The plate of Section 5.10 is cut by a plane that runs from the bottom-right corner to the top edge at two cells from the left end. Its eight cut cells have remaining volume fractions 0.917, 0.667, 0.333 and 0.083, two each. The layout is stored with its two in-plane axes interchanged, a cube-symmetry image of the plate, so that the cut normal lies at \(\vartheta=33.69^\circ\), within the range \(0<\vartheta<\pi/4\) of the validation geometries (Section 5.1), rather than at its image, 56.31°. The network is not equivariant under the cube symmetries (Section 6.3). Apart from the DOFs of the clamped cut-plane elements, no DOF is fixed.

For comparison with a homogenisation-based graded design, the effective elasticity tensor \(C^H(\tau)\) of the uncut cell with uniform corner parameter was computed by periodic homogenisation on the same discrete model (\(n=32\), Q2 elements, the stabilised stiffness \(K\) with its ghost penalty). Nodes on opposite faces of the cell are identified, and the fluctuation is periodic with one node fixed. Six unit macroscopic strains give \(C^H_{ij}=u_i^TKu_j\) for the unit cell, and the material volume fraction \(V^H(\tau)\) is the sum of the zeroth element moments. Cubic splines in \(\tau\) interpolate \(C^H_{11}\), \(C^H_{12}\), \(C^H_{44}\) and \(V^H\) between the twelve thicknesses of Table ST18b. The macroscale model meshes the plate with Q1 hexahedra, six per cell and axis (4,464 elements, 5,719 nodes). At every quadrature point it interpolates the local parameter \(\tau(x)\) trilinearly from the corners of its cell and evaluates \(C^H(\tau(x))\) and \(V^H(\tau(x))\). It integrates the elements intersected by the cut with \(4^3\) sub-points, the void part carrying \(10^{-6}\,C^H(0.4)\). It applies the clamp on the plane of the cut, \(u=0\) on the section of the plane with the plate, by a penalty \(\alpha\int u\cdot v\,dA\) with \(\alpha=10^6\,C^H_{11}(0.4)/h_M\) and \(h_M=1/6\). The penalty is integrated with a seven-point rule on the triangles of the section polygon of every intersected element. The load is a uniform traction of unit resultant on the loaded face. It was optimised with adjoint sensitivities and the same MMA settings, constraints, fixed vertices and volume fraction.

In iterations 0–3, while the volume bound is violated, the macroscale and NICE runs move every free parameter by the move limit or to the lower bound and analyse the same designs. There the macroscale compliance lies 27.0–31.8% below the NICE compliance, and along the rest of its path it lies 29–33% below it. At the uniform start, refining the macroscale mesh to 12 elements per cell and axis (35,136 elements, 39,949 nodes) raises the macroscale compliance from 55.192 to 56.088 (+1.6%), 25.8% instead of 27.0% below the exact compliance 75.599. Raising the penalty factor from \(10^6\) to \(10^8\) changes it by \(-8.7\times10^{-6}\) (6 elements) and \(-3.7\times10^{-6}\) (12 elements), relative. The optimisation used the six-element mesh. The fine-scale model fixes every DOF of the cut-plane elements of the cut cells (Table ST21). This support is at least as stiff as one acting on the cut surface alone; it lowers the exact compliance and therefore narrows the macroscale underestimation. For the graded homogenisation design, the macroscale volume model no longer equals the fine-scale material volume exactly (Table ST18c).

### Table ST18. Plate supported on its cut: NICE optimisation and homogenisation design

#### ST18a. Runs

|  | NICE run | Homogenised model | Homogenisation design, NICE analysis |
| --- | --- | --- | --- |
| Start design | uniform 0.40 | uniform 0.40 (macroscale) | final macroscale design |
| Largest compliance (iteration) | 110.493 (4) | 74.557 (4) (macroscale) | — |
| Final \(\tau\) range | 0.180–0.690 | 0.180–0.690 | 0.180–0.690 |
| Largest corner span / gradient norm at the end | 0.450 / 0.450 | 0.450 / 0.450 | 0.450 / 0.450 |
| Time per iteration, mean (range) (s) | 200 (155–270) | 4.1 (macroscale, CPU) | — |
| Sum of iteration times (s) | 4,802 | 95 (macroscale, CPU) | — |
| PCG iterations | 73–122 | — | 127 |
| Recomputed residual | 3.5e-04–5.8e-04 | — | 3.9e-04 |
| \(\max\lvert\bar U^T\rho\rvert/\widehat C\) | 5.2e-08 | — | 5.4e-08 |

Every fine-scale value is a NICE value; exact checks in Table ST21. Iterations and first and last compliances: Table 6. Figure 15a,b draws the corner parameters in the layer \(z=0\); those in the layer \(z=1\) differ from them by at most 1.1e-03 (NICE) and 6.3e-13 (homogenisation design).

#### ST18b. Homogenised law of the uniform-thickness cell

| \(\tau\) | \(V^H\) | \(C^H_{11}\) | \(C^H_{12}\) | \(C^H_{44}\) |
| --- | --- | --- | --- | --- |
| 0.18 | 0.1027 | 0.03996 | 0.02806 | 0.01920 |
| 0.22 | 0.1256 | 0.04992 | 0.03432 | 0.02384 |
| 0.27 | 0.1541 | 0.06307 | 0.04212 | 0.02988 |
| 0.32 | 0.1827 | 0.07712 | 0.04989 | 0.03620 |
| 0.37 | 0.2113 | 0.09218 | 0.05765 | 0.04279 |
| 0.42 | 0.2399 | 0.10838 | 0.06541 | 0.04968 |
| 0.47 | 0.2685 | 0.12585 | 0.07323 | 0.05688 |
| 0.52 | 0.2971 | 0.14471 | 0.08117 | 0.06440 |
| 0.57 | 0.3258 | 0.16511 | 0.08929 | 0.07226 |
| 0.62 | 0.3545 | 0.18720 | 0.09768 | 0.08049 |
| 0.67 | 0.3832 | 0.21116 | 0.10644 | 0.08912 |
| 0.70 | 0.4004 | 0.22653 | 0.11193 | 0.09450 |

\(E_Y=1\), \(\nu=0.3\), unit cell of volume 1, Voigt notation with engineering shear strains; \(C^H_{11}\), \(C^H_{12}\), \(C^H_{44}\) are the entry values of the computed tensor (the macroscale model averages the three symmetric entries of each). Largest relative spreads over the twelve thicknesses: \(C^H_{11},C^H_{22},C^H_{33}\) 3.1e-13, \(C^H_{12},C^H_{13},C^H_{23}\) 6.1e-14, \(C^H_{44},C^H_{55},C^H_{66}\) 1.4e-13; largest normal–shear or shear–shear coupling 7.5e-14 of \(C^H_{11}\); largest relative equilibrium residual of the periodic fluctuations 2.2e-13.

#### ST18c. Homogenisation against the fine-scale NICE optimisation

| Quantity | Macroscale model | NICE | Exact condensation |
| --- | --- | --- | --- |
| Compliance of the uniform design | 55.192 | 75.591 | 75.599 |
| Macroscale prediction error at the uniform design (%) | — | −26.99 | −26.99 |
| Compliance of the homogenisation design | 55.271 | 80.172 | 80.198 |
| Homogenisation design vs NICE design (%) | — | +2.44 | +2.44 |
| Fine-scale \(V/V^*\): NICE design / homogenisation design | — | 1.00000 / 0.99821 | — |
| Corner parameters, NICE vs homogenisation design: largest difference / RMS / correlation | — | 0.262 / 0.057 / 0.946 | — |

Prediction error: \((C_{\rm macro}-C)/C\) at the uniform design \(\tau=0.40\). Corner differences and correlation over the \(24\times8\) cell-corner parameters. The macroscale volume of the uniform design, 4.569075, agrees with the fine-scale material volume 4.569074 to 1.2e-07.

### S6.4. Geometry-generation perturbations and discrete switches

Table ST19a lists the four applications of the geometry-generation fallback of Table ST16; a failure means that the geometry generator could not certify the material patches of an element (Appendix A.1). Table ST19b counts the cells whose discrete description changed between consecutive iterations. Each iteration evaluates the objective on a different discrete model, and the derivatives of Section 4.2 hold only within each model.

### Table ST19. Geometry-generation fallback and discrete switches

#### ST19a. Applied perturbations

| Case | Iteration | Failing cells | Applied \(\epsilon\) | Perturbed vertices | Cells regenerated | Largest \(\lvert\Delta\tau\rvert\) |
| --- | --- | --- | --- | --- | --- | --- |
| A, NICE | 16 | 010 (uncut) | +1e-04 | 4 (of 18) | 8 | 5.8e-05 |
| A, NICE | 19 | 010 (uncut) | +1e-03 | 4 (of 18) | 8 | 5.8e-04 |
| Plate | 5 | 060 (cut, 0.917), 160 (cut, 0.083) | +1e-03 | 12 (of 64) | 5 | 2.7e-04 |
| Plate | 14 | 000 (uncut) | +1e-04 | 4 (of 64) | 4 | 3.6e-05 |

Failing cells by layout index (remaining volume fraction of cut cells): the cells whose generation failed for the design returned by MMA. The applied \(\epsilon\) acts on the design returned by MMA; largest \(\lvert\Delta\tau\rvert\): largest change of a corner parameter by the applied perturbation; no perturbed vertex was clipped to a bound. The exact twin and the analysis of the homogenisation design needed no fallback.

#### ST19b. Cells whose discrete description changed between consecutive iterations

| Case | Active elements; ghost faces | Retained DOFs | Cut-plane-element nodes | Weakly connected nodes | Weak-region element stencils | Weak-region face stencils | Coarse-factor shift | Any of these |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A, NICE (8 cells, 23 steps) | 2 / 8 / 8 | 0 / 8 / 8 | 0 / 4 / 4 | 3 / 8 / 8 | 2 / 8 / 8 | 2 / 8 / 8 | 0 / 0 / 2 | 3 / 8 / 8 |
| Plate (24 cells, 23 steps) | 8 / 18 / 24 | 3 / 18 / 24 | 0 / 3 / 8 | 11 / 18 / 24 | 9 / 18 / 24 | 11 / 18 / 24 | 0 / 0 / 4 | 11 / 18 / 24 |

Minimum / median / maximum, over the steps between consecutive iterations, of the number of cells whose count changed. Counts were taken of active elements and ghost-penalty faces (identical counts), retained DOFs, nodes flagged by the network's binary node indicators as nodes of the cut-plane elements or as weakly connected nodes, element and face stencils selected for the weak-region layers (Appendix G.1), and the diagonal shift of the coarse factorisation (Appendix F.1). A change that leaves a count unchanged is not detected, so the numbers are lower bounds. The exact twin (8 cells, 22 steps), which recorded active elements and retained DOFs only, gives 2 / 8 / 8 and 0 / 8 / 8.

### S6.5. Scale study and exact verification of the plate designs

**Scale study.** Plates were generated with 24, 51, 88 and 110 cells (Table ST20). They have the proportions of the plate of S6.3 (short side : long side 1 : 2, one cell thick), with its planar cut scaled with the plate, and all cut cells have the four remaining volume fractions of S6.3. The plates are clamped on the uncut long side and loaded on the opposite face by a traction load of unit resultant, acting in the plane along the long side. After the cut, this face is two cells wide in the 24-cell plate. The 24-cell plate is the plate of S6.3 with this clamp and load. Each run starts from the uniform \(\tau=0.40\) with \(V^*=0.8\,V(\boldsymbol\tau^0)\) and the settings of Table ST16, and was stopped after four analyses (three MMA updates). Degrees of freedom are those of the initial design: the cut finite-element model of all cells, with nodes on shared faces counted once, and the retained DOFs of the assembled lattice without the clamped face.

**Table ST20. Scale study.** Plates with the proportions and cut of the plate of S6.3, clamped on the uncut long side and loaded in plane on the opposite face, uniform start, four analyses; GPU memory budget for resident learned substructures 4 GiB (Supplementary Note S4.1). DOFs: degrees of freedom of the cut finite-element model / free retained DOFs of the assembled lattice (initial design). Start \(\widehat C\): NICE compliance of the uniform start (first analysis). Time: mean over the analyses (range), with mean phase times, in s, geometry generation included. Residual: largest recomputed relative residual / largest \(|\bar U^T\rho|/\widehat C\). CPU memory: peak resident memory of the main process, which holds the streamed state of the learned cells, per cell (total), GiB; the geometry generation runs in separate processes. GPU memory: largest memory in use on the device, sampled every 30 s with nvidia-smi, GiB; besides the resident learned cells it holds the assembled retained system: \(\mathbb K_{PP}\) and its cuDSS factor in the preconditioner and the vectors of the conjugate gradients.

| Cells / cut | DOFs: cut model / free retained | Design variables | Start \(\widehat C\) | Time per design iteration (s) and phases | PCG iterations | Residual / residual work | CPU memory per cell (total) | GPU memory in use |
| --- | --- | ---: | ---: | --- | --- | --- | --- | ---: |
| 24 / 8 | 6.51 M / 0.388 M | 64 | 94.34 | 242 (229–252): geometry 21, cell preparation 36, preconditioner 21, PCG 144, sensitivities 19 | 138–169 | 3.9e-04 / 2.3e-08 | 0.62 (15.0) | 14.9 |
| 51 / 12 | 14.57 M / 0.780 M | 128 | 89.12 | 547 (539–559): geometry 37, cell preparation 83, preconditioner 45, PCG 334, sensitivities 43 | 156–171 | 6.6e-04 / 3.9e-08 | 0.69 (35.2) | 20.2 |
| 88 / 16 | 25.85 M / 1.305 M | 212 | 86.75 | 962 (955–980): geometry 63, cell preparation 150, preconditioner 79, PCG 586, sensitivities 74 | 158–173 | 9.7e-04 / 6.9e-08 | 0.71 (62.8) | 26.1 |
| 110 / 18 | 32.70 M / 1.618 M | 262 | 85.38 | 1,164 (1,119–1,214): geometry 71, cell preparation 185, preconditioner 99, PCG 698, sensitivities 95 | 142–172 | 1.2e-03 / 7.3e-08 | 0.73 (79.9) | 30.0 |

Exact checks at the first design iteration, with the metrics of Table ST21: 24 cells, \(C=94.353\), \(\widehat C=94.341\), approximate compliance error \(-0.0132\%\), gradient error 0.045%, cosine 0.99999993, component error median / 95th percentile / max 0.011 / 0.048 / 0.059%, exact sign in all 64 components; 51 cells, \(C=89.129\), \(\widehat C=89.119\), \(-0.0113\%\), 0.047%, 0.99999994, 0.009 / 0.034 / 0.063%, all 128 components. The exact reference of the 88- and 110-cell plates was not computed.

**Table ST21. Exact checks of the plate designs.** The uniform start, the final NICE design and the homogenisation design of S6.3, and two uniform designs at the volume bound \(V^*\) for comparison; the intermediate designs were not checked. Exact condensation of the 55 distinct cells with PARDISO's Schur-complement option, verified column-wise against interior solves; assembled solve, with every DOF of the cut-plane elements of the cut cells fixed, to a recursive relative residual of \(10^{-10}\). Approximate compliance error: \(\widehat C/C-1\). Gradient: vertex gradient with exact sensitivities (central moment differences, Eq. (H.3)) against the NICE sensitivity estimate, over the 64 free vertex parameters. Exact solves: 136 / 9.4e-11, 190 / 8.8e-11, 191 / 9.5e-11 conjugate-gradient iterations / recomputed residual, in the order of the rows.

| Design | Exact \(C\) | \(\widehat C\) | Approximate compliance error (%) | Gradient error (%) | Cosine | Component error / \(\max\lvert g\rvert\) (%): median / 95th percentile / max | Sign agreement (64 variables) |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| NICE run, iteration 0 (start) | 75.599 | 75.591 | −0.0105 | 0.029 | 0.99999997 | 0.003 / 0.027 / 0.043 | 1.000 |
| NICE run, iteration 23 (final) | 78.287 | 78.264 | −0.0296 | 0.301 | 0.99999652 | 0.057 / 0.273 / 0.315 | 1.000 |
| Homogenisation design | 80.198 | 80.172 | −0.0317 | 0.320 | 0.99999596 | 0.052 / 0.322 / 0.335 | 1.000 |
| Uniform design at \(V^*\), 64 free parameters (loaded-face parameters 0.40) | 115.357 | 115.335 | −0.0191 | 0.064 | 0.99999995 | — | 1.000 |
| Uniform design at \(V^*\), all 74 parameters | 113.279 | 113.259 | −0.0179 | 0.055 | 0.99999995 | — | 1.000 |

## Supplementary Note S7. Transfer to a second TPMS family: gyroid sheet cells

The construction of Sections 3 and 4 comprises the condensed energy with the transpose of the complete recovery operator, Propositions 1–4 and the correction \(\mathcal W\). It uses the symmetry and positive semidefiniteness of the cell stiffness and the positive definiteness of its interior block, and it does not use the level-set function of Eq. (1). The trained network, in contrast, has seen Schwarz-P cells only (Sections 6.1 and 6.3). This note applies NICE to gyroid sheet cells, first with the network parameters of Table 2 unchanged (zero-shot) and then after a short continuation of training that includes a few gyroid cells. It is not part of the evidence of Section 5.

**Cells.** A gyroid cell replaces the level-set function of Eq. (1) by \(\phi_G(x)=\sin2\pi x_1\cos2\pi x_2+\sin2\pi x_2\cos2\pi x_3+\sin2\pi x_3\cos2\pi x_1\) and keeps the cell box, the trilinear corner parameters, the optional planar cut and the settings of Table 1 (background mesh \(n=32\), material, ghost penalty, retained DOFs and correction 8 / \(Q_1(17)\) / 8). Every gyroid cell is the twin of one P cell and keeps the P cell's position, cut plane and cut-severity group (uncut or cut). Its corner parameters are mapped from the P values so that a uniform-thickness cell keeps its sheet volume fraction (for example 23.3% against 23.2% and 5.8% against 6.2% for two twins). The cells are generated by a version of the geometry stage that takes the level-set function as input; with the P function it reproduces the eight cells of the \(2\times2\times2\) block of Section 5.8 cell by cell. When this version generates a lattice, it keeps the largest face-connected set of active elements of the whole lattice rather than of each cell. It also removes fragments of less than a quarter of an element that have no ghost-penalty support, and it repeats both steps until nothing changes. The retained DOFs of neighbouring cells then coincide on their shared faces except along the cell edges (unmatched DOFs are treated as in Section 2.3). The sixteen single cells are the twins of the validation cells fresh_val_2000 to fresh_val_2015 (geometry identifiers gval_000 to gval_1500 in the data archive). These validation cells include U1, U2, L1, M1, M2 and H1–H3 of R1 and belong to the 20 checkpoint-selection geometries of Table ST01. The lattice is the twin of the \(2\times2\times2\) block (glat222). No gyroid cell entered the training or checkpoint selection of any variant of Table 2 or of the two variants evaluated only in this supplement.

Propositions 1, 3 and 4 and the properties of Section 3.1 hold for these cells as for the P cells. They hold because assumptions (A1)–(A3) concern the stiffness and the assembly, and because the network's displacement recovery reproduces the retained displacements and rigid-body motion by construction. Proposition 2 holds under its spectral condition, which was not checked on the gyroid cells, as for the cells of Sections 5.8–5.10.

**Variants.** *NICE, zero-shot*: NICE with the network parameters of Table 2, unchanged. *NICE, continued* (record identifier A3G_ft): NICE's parameters, trained for a further 8,000 steps. The training set comprises 96 P training cells and 18 gyroid twins of P training cells (cells with at most 420,000 DOFs), and the correction 8 / \(Q_1(17)\) / 8 is inside the training loop as in Section 3.5. Its checkpoint was selected on the P validation list only (selection score 0.00339, against 0.00237 for NICE; Table ST01). The sixteen gyroid twins and the gyroid lattice therefore entered neither its training nor its selection. Both variants apply the correction 8 / \(Q_1(17)\) / 8 at evaluation.

**Single cells.** Each gyroid twin was evaluated in the identity orientation with 64 validation test displacements per load class (Appendix G.2). The energy error of Eq. (7) was averaged over the test displacements of each cell and then over the sixteen cells with equal weights, as in Section 5.1 (Table ST22). Zero-shot, the mean error under traction loads is 4.48% (largest cell 25.6%), 66 times the 0.068% of NICE on the P twins of the same cells. Over the six load and support classes the ratio of the means is 56 to 234; under the imposed polynomial and multiscale displacements it is 14 and 7. The degradation is not confined to a few cells. In every cell, the error under each load and support class is 20 to 1,150 times that of its P twin, and under the imposed displacements 2 to 30 times. Under traction loads, the three gyroid cells with the largest errors are the twins of the three P cells with the largest NICE errors among the sixteen (fresh_val_2007, 2012 and 2015). The correction is the same as on the P cells. The accuracy of the network's interior displacements does not carry over, since it is tied to the family on which the network was trained.

After the continuation, the means under the six load and support classes fall by factors of 5.8 to 10.2. Under traction loads the mean falls from 4.48% to 0.74%, and the largest cell mean from 25.6% to 2.67%. Under the imposed polynomial and multiscale displacements the means fall by factors of 1.6 and 1.2. The means remain 6 to 23 times those of NICE on the P twins (11 times under traction loads). The reduction is uneven across cells: the twin of H2 (fresh_val_2010) now has the largest errors under nodal point loads and traction loads, 4.36% and 2.67% against 4.73% and 3.33% zero-shot. On the 80 P validation cells, the class means of the continued network are 1.1 to 1.7 times those of NICE in the same evaluation (0.026–0.072% against 0.015–0.065%).

**Table ST22. Energy error (%) of the gyroid twins and of the P cells.** Geometry-equal mean, with the largest geometry mean in parentheses, identity orientation. Gyroid twins: the sixteen twins of fresh_val_2000 to fresh_val_2015. P twins: these sixteen P validation cells. P validation: the 80 validation cells of Section 5.1 in one evaluation of both variants.

| Load class | Gyroid twins, NICE, zero-shot | Gyroid twins, NICE, continued | P twins, NICE | P validation, NICE | P validation, NICE, continued |
| --- | ---: | ---: | ---: | ---: | ---: |
| Traction loads (force_c) | 4.48 (25.6) | 0.74 (2.67) | 0.068 (0.28) | 0.065 (0.28)\(^{a}\) | 0.072 (0.30)\(^{a}\) |
| Single-face traction loads (face_c) | 4.30 (27.2) | 0.60 (2.52) | 0.031 (0.17) | 0.029 (0.17)\(^{a}\) | 0.037 (0.21)\(^{a}\) |
| Nodal point loads (force) | 6.10 (14.1) | 0.78 (4.36) | 0.058 (0.18) | 0.058 (0.33) | 0.065 (0.31) |
| Single-face nodal point loads (face) | 7.35 (15.2) | 0.72 (1.59) | 0.031 (0.28) | 0.037 (0.72) | 0.044 (0.47) |
| Spring supports (support) | 5.78 (14.0) | 0.60 (1.54) | 0.053 (0.15) | 0.055 (0.37) | 0.064 (0.28) |
| Stiffness-scaled spring supports (support_k) | 3.01 (17.2) | 0.52 (2.42) | 0.054 (0.22)\(^{b}\) | 0.053 (0.22)\(^{a}\) | 0.059 (0.24)\(^{a}\) |
| Polynomial displacements (macro) | 0.19 (0.89) | 0.12 (0.58) | 0.014 (0.075) | 0.015 (0.12) | 0.026 (0.17) |
| Multiscale displacements (grf) | 0.16 (0.79) | 0.13 (0.68) | 0.024 (0.13) | 0.025 (0.14) | 0.037 (0.21) |

\(^{a}\) Evaluated in this run on the 20 checkpoint-selection geometries only (stiffness-scaled spring supports: 19); Table ST03 gives NICE on all 80 (75) geometries. \(^{b}\) 15 twins; fresh_val_2010 has no stiffness-scaled spring-support test displacements.

**Lattice.** The gyroid twin of the \(2\times2\times2\) block of Section 5.8 has eight cells, four of them cut, and every cell is learned; it has 160,629 free retained DOFs against 139,002 for the P block. It was solved as in Section 5.8, clamped on one face and loaded by the three face loads and three random loads, with the dense exact condensed matrices as reference (Table ST23). Zero-shot, the lattice compliance errors are 0.98–1.53% under the face loads (2.14% under the random loads), and the largest thickness-sensitivity error over the cells is 13.5% (10.6%), against 0.014% (0.069%) and 0.14% (0.45%) for NICE on the P block. With the continued network they are 0.23% (0.35%) and 1.84% (3.57%), and on the P block 0.019% (0.084%) and 0.25% (0.49%). With exact operators, the gyroid lattice needs 414 conjugate-gradient iterations against 183 for the P block; on the gyroid lattice, the learned operators add 20 (zero-shot) and 22 (continued) iterations.

**Table ST23. Lattice compliance and sensitivity errors (%) of the P block and its gyroid twin.** Maxima over the three face loads, and over the three random loads in parentheses; sensitivity error: largest relative error of the eight-component vector over the eight cells. Iterations: preconditioned conjugate gradients (Supplementary Note S4.1) with the learned / the exact condensed stiffness matrices, to a recursive relative residual of \(10^{-10}\). The exact gyroid solution reaches a recomputed relative residual of \(9.6\times10^{-11}\); for the learned gyroid solves, the recomputed residual stagnates at \(2.7\times10^{-4}\) (zero-shot) and \(2.8\times10^{-4}\) (continued), as for the P block in Section 5.8.

| Lattice | Variant | Compliance error | Sensitivity error | Iterations |
| --- | --- | ---: | ---: | --- |
| P block (\(2\times2\times2\)) | NICE | 0.014 (0.069) | 0.14 (0.45) | 186 / 183 |
| P block (\(2\times2\times2\)) | NICE, continued | 0.019 (0.084) | 0.25 (0.49) | 187 / 183 |
| Gyroid block (\(2\times2\times2\)) | NICE, zero-shot | 1.53 (2.14) | 13.5 (10.6) | 434 / 414 |
| Gyroid block (\(2\times2\times2\)) | NICE, continued | 0.23 (0.35) | 1.84 (3.57) | 436 / 414 |

**Reading.** The construction carried over unchanged: the energy form, the transpose of the complete recovery operator, the correction and the assembly were applied to the gyroid cells without modification. The properties that do not depend on the network parameters hold for them as for the P cells. The trained network is specific to its family: zero-shot, the gyroid errors under loads and supports are 56 to 234 times those on the P twins in the mean, and the unchanged correction does not make up the difference. A continuation of 8,000 training steps with eighteen gyroid cells lowers them six- to tenfold, to 6 to 23 times the P level. It brings the gyroid lattice to 0.23% in compliance and 1.84% in sensitivity under the face loads, while the class means of the continued network on the P validation cells are 1.1 to 1.7 times those of NICE. Whether more gyroid data and training reach the P level was not tested. A new TPMS family therefore needs new training data (Section 6.3); the construction itself needs no change.

**Records.** Record identifiers in the data archive (files in `evidence/gcell/`): gval.json (gyroid twins: P twin, cut-severity group and corner parameters); gval_A3.json and gval_A3G.json (single gyroid cells, NICE zero-shot and continued); newval_A3G.json (80 P validation cells, NICE continued; NICE in the same evaluation: `evidence/newval_A3_2grid.json`); lat_hetero_g222.json and lat_hetero_g222_A3G.json (gyroid block, NICE zero-shot and continued); lat_hetero222_A3G.json (P block, NICE continued); g_vs_p_222.json (P and gyroid blocks, NICE).

## Supplementary Note S8. Further continuations of the base network

Besides NICE, two further networks continue the training of the base network for 15,000 training steps. The Smoothing-trained variant applies eight smoothing steps without coarse-grid correction in every training step. The Uncorrected continuation is trained without correction. Table ST24 gives their training and evaluation settings in the format of Table 2 of the main text.

**Table ST24. Training and evaluation settings of the two further continuations.** Columns as in Table 2 of the main text.

| Variant | Network parameters | Training set (geometries) | Correction in training | Correction at evaluation |
| --- | --- | ---: | --- | --- |
| Smoothing-trained | Base network continued for 15,000 training steps | 591 | 8 smoothing steps | 8 smoothing steps |
| Uncorrected continuation | Base network continued for 15,000 training steps | 591 | None | None |

Both continuations use the network architecture of Section 3, with about \(6\times10^5\) trainable parameters (Table ST15). The three continuations, NICE, the Smoothing-trained variant and the Uncorrected continuation, draw from the same set of 591 geometries. This set contains 304 of the 305 training geometries of the base network. The three continuations see the same geometries in the same order, at most 153 of the 591 in their 15,000 training steps (Table ST01).

### S8.1. Single cells

Under traction loads, the means of NICE by cut-severity group are 64 to 102 times lower than those of the Uncorrected continuation. The group means of the Uncorrected continuation rise from 1.03% for uncut to 11.2% for heavily cut cells (Table ST03b). Over all 80 geometries, the mean energy error of the Uncorrected continuation under traction loads is 6.33%, and its largest single-geometry mean is 42%. Continuing the base network without correction therefore changes its mean only from 6.89% to 6.33%. The Smoothing-trained variant reaches a mean of 1.28%, and its largest single-geometry mean is 7.8% (Table ST03).

### S8.2. Two-cell assemblies

The two further continuations were also evaluated in the two-cell assemblies of Section 5.6. The configurations, loads and error measures are those of Section 5.6. Tables ST08 and ST09 list the maxima for every variant and configuration evaluated.

The Uncorrected continuation was evaluated in eleven configurations. It exceeds the 3% sensitivity line in four of them, with sensitivity errors up to 11.4% (Table ST08). On M1 it also exceeds the 3% compliance line, with compliance errors up to 4.1%. The Smoothing-trained variant reduces these errors, but it still exceeds the 3% sensitivity line in the same four configurations, with errors of 3.15–4.43%. Smoothing without the coarse-grid correction leaves the slowly damped error in the low modes described in Section 5.4. This is consistent with the sensitivity errors that remain in U1 and M1. Base network + correction stays below both 3% lines, with a largest sensitivity error of 0.95% on U1/x. The correction therefore brings the assembled responses below the 3% lines.

Under the traction loads on the cut surface of the test cell, the Uncorrected continuation exceeds the 3% line in four of its seven configurations, with sensitivity errors up to 14.1% (Table ST09). Under the same loads, the errors of NICE are at most 0.10% in the compliance and 0.16% in the sensitivity.

Without the complete correction, an accurate compliance does not guarantee an accurate local sensitivity (Table ST25). Under the z traction load on the neighbour face of U1/x, the compliance error of the Smoothing-trained variant is 0.00109%, whereas its test-cell sensitivity error is 3.15%. The test cell carries 0.13% of the exact assembled energy under this load. NICE reduces both errors, and its test-cell sensitivity error under this load is 0.598%. On M1/x under the y traction load on the test-cell face, NICE gives errors of 0.0481% in the compliance and 0.145% in the sensitivity. The Uncorrected continuation gives 3.44% and 11.4% under the same load.

**Table ST25. Compliance and test-cell sensitivity under individual face loads.** T and N identify the loaded face of the test cell or of the neighbouring cell; x, y and z give the traction direction. The last column gives the fraction of the exact assembled energy carried by the test cell.

| Test cell / configuration | Load | Uncorrected continuation: compliance / sensitivity (%) | Smoothing-trained: compliance / sensitivity (%) | NICE: compliance / sensitivity (%) | Energy fraction of the test cell |
| --- | --- | --- | --- | --- | ---: |
| H1/x | T-x | 1.06 / 2.47 | 0.144 / 0.636 | 0.0076 / 0.0133 | 0.643 |
| U1/x | N-z | 0.0023 / 4.89 | 0.00109 / 3.15 | 7.3×10⁻⁵ / 0.598 | 0.0013 |
| M1/x | T-y | 3.44 / 11.4 | 1.14 / 4.43 | 0.0481 / 0.145 | 0.311 |

## Supplementary figures

![Figure S01](figures/S06_reference_verification.png)

**Figure S01. Verification of the CutFEM reference.** Single cells clamped on one cell face and loaded by unit traction loads on another face in the three Cartesian directions. In (a), (b) and (d), solid lines with filled markers give the compliance and dashed lines with open markers the thickness sensitivity, each as the largest relative change over the three loads. (a) Change against the finest background resolution computed for U1 (\(n=40\)) and for M1 and M2 (\(n=48\)); the reference used throughout has \(n=32\). (b) Change when the ghost-penalty coefficient is varied from the value \(10^{-4}\) used throughout. (c) Filled: largest relative change of the sensitivity when the finite-difference step of the moment derivatives is varied from the value \(h_c=10^{-5}\tau_c\) used throughout, at most \(2.6\times10^{-5}\)% at \(10^{-3}\tau_c\) and a hundredfold smaller at \(10^{-4}\tau_c\); open: largest relative difference between central compliance differences and the sensitivity at the step used. (d) Refinement to \(n=64\) of H1, H2 and the validation cells with the largest NICE error (W1) and the thinnest walls (W3), the data of Table ST10; H2 has no \(n=56\) solution.

![Figure S02](figures/S01_distributions.png)

**Figure S02. Distributions of geometry-level energy errors.** Each point is one validation geometry's mean energy error in the identity orientation; horizontal bars are population medians. The classes with nodal point loads, spring supports, single-face nodal point loads, polynomial displacements, multiscale displacements, traction loads and single-face traction loads contain 80 geometries. The class with stiffness-scaled supports and the class with displacements imposed by a neighbouring cell contain 75, the same populations as Table ST03. The five variants of Table ST03 (Base network, Uncorrected continuation, Smoothing-trained, Base network + correction and NICE) use the markers and colours of the main-text figures and of Figure S04; Base network + correction applies the correction to the base network at deployment. Deterministic horizontal offsets separate overlapping observations. All panels share the logarithmic error axis.

![Figure S03](figures/S03_sensitivity_diagnostics.png)

**Figure S03. Sensitivity-error diagnostics.** (a) Paired mean energy and sensitivity errors for responses to traction loads and nodal point loads, using six cells of the base network and five of the Uncorrected continuation. (b) Linear-term norm fraction \(\|D_1\|_F/(\|D_1\|_F+\|D_2\|_F)\) under traction loads, where \(D_1+D_2\) is the sensitivity-error matrix over all eight design components and evaluated test displacements. (c,d) Fractions of absolute elementwise sensitivity-error contributions and of element counts in four mutually exclusive material-volume-fraction groups for the base network under traction loads. Each error group sums absolute contributions over its elements, design components and test displacements before normalisation by the total. Filled markers identify the base network. In (a,b), open markers and hatched bars identify the Uncorrected continuation, which is shown for U1, U2, M1, H1 and M2 (Table ST05a).

![Figure S04](figures/S07_energy_share_variants.png)

**Figure S04. The other four variants follow the relations of Figure 11.** The counterpart of Figure 11 for (a–c) the base network, (d–f) the Uncorrected continuation, (g–i) the Smoothing-trained variant and (j–l) Base network + correction, which applies the correction at deployment. One point per load in the configurations evaluated for each variant (54, 87, 96 and 114 loads; Tables ST08 and ST09); filled markers denote face loads and open markers cut-surface loads. Top row: compliance error against \(\beta\); lines denote equality. The ratio of compliance error to \(\beta\) lies between 0.80 and 0.98, 0.79 and 0.99, 0.92 and 0.99, and 0.96 and 1.006 for the four variants; the six ratios above 1 correspond to at most \(1.3\times10^{-9}\) of the compliance. Middle and bottom rows: compliance error and test-cell sensitivity error against the exact energy fraction \(w\) of the test cell; lines are least-squares fits in logarithmic coordinates, with the slope given in each panel. All panels share the error axis. The circled load in (a–c) is U1/x under the z traction load on the face of the neighbouring cell; its values are given in the note to Table ST08.
