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

