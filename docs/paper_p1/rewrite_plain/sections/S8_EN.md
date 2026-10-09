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

Under the traction loads on the cut face of the test cell, the Uncorrected continuation exceeds the 3% line in four of its seven configurations, with sensitivity errors up to 14.1% (Table ST09). Under the same loads, the errors of NICE are at most 0.10% in the compliance and 0.16% in the sensitivity.

Without the complete correction, an accurate compliance does not guarantee an accurate local sensitivity (Table ST25). Under the z traction load on the neighbour face of U1/x, the compliance error of the Smoothing-trained variant is 0.00109%, whereas its test-cell sensitivity error is 3.15%. The test cell carries 0.13% of the exact assembled energy under this load. NICE reduces both errors, and its test-cell sensitivity error under this load is 0.598%. On M1/x under the y traction load on the test-cell face, NICE gives errors of 0.0481% in the compliance and 0.145% in the sensitivity. The Uncorrected continuation gives 3.44% and 11.4% under the same load.

**Table ST25. Compliance and test-cell sensitivity under individual face loads.** T and N identify the loaded face of the test cell or of the neighbouring cell; x, y and z give the traction direction. The last column gives the fraction of the exact assembled energy carried by the test cell.

| Test cell / configuration | Load | Uncorrected continuation: compliance / sensitivity (%) | Smoothing-trained: compliance / sensitivity (%) | NICE: compliance / sensitivity (%) | Energy fraction of the test cell |
| --- | --- | --- | --- | --- | ---: |
| H1/x | T-x | 1.06 / 2.47 | 0.144 / 0.636 | 0.0076 / 0.0133 | 0.643 |
| U1/x | N-z | 0.0023 / 4.89 | 0.00109 / 3.15 | 7.3×10⁻⁵ / 0.598 | 0.0013 |
| M1/x | T-y | 3.44 / 11.4 | 1.14 / 4.43 | 0.0481 / 0.145 | 0.311 |
