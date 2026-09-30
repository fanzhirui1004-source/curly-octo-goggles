# Draft of Section 6.11 (for insertion into MANUSCRIPT_EN.md)

<!-- Draft of 2026-09-30. Every number is taken from results/X6_opt/FACTS_6_11.json / FACTS_6_11.md or computed from the
archived run records by the script shown under "Number sources". [TBD-EXACT: ...] marks the pending exact checks of the
plate designs, [TBD-SCALE: ...] the pending scale demonstration. Supplementary Note S9 and Tables ST21-ST24 are written
separately. Reference to add to the list: Svanberg, K. (1987). [The method of moving asymptotes—a new method for structural
optimization](https://doi.org/10.1002/nme.1620240207). *International Journal for Numerical Methods in Engineering*, 24(2),
359–373. The figure file name below is provisional. -->

### 6.11. Thickness design optimisation

The corner thickness parameters at the lattice vertices, shared by the cells meeting there through \(\boldsymbol\tau_m=\boldsymbol\tau_m(\boldsymbol\tau_g)\), are optimised to minimise the compliance under one unit consistent face traction subject to the bound \(V\le V^*=0.8\,V(\boldsymbol\tau^0)\) on the material volume \(V\) of the discrete model. The parameters on the loaded face are held fixed, so that the nodal load does not depend on the design and the load-derivative term \(2f_{g,c}^T\widehat U\) of Appendix H does not arise. The optimiser is driven by the field-based estimates of Eq. (13), summed over the cells at each vertex, \(\sum_m(\partial\boldsymbol\tau_m/\partial\boldsymbol\tau_g)^T\widetilde{\boldsymbol s}_m\), not by the complete surrogate derivative of Eq. (14). The bounds \(0.18\le\tau\le0.69\) and limits of 0.45 on each cell's corner span and thickness-gradient norm keep every cell inside the trained domain (Section 7.5). The standard method of moving asymptotes [Svanberg (1987)](https://doi.org/10.1002/nme.1620240207) takes one step per design iteration, with a move limit of 5% of the parameter range and no line search, so the inconsistency between \(\widetilde s_c\) and \(\widehat C_{,c}\) (Section 7.3) never enters a step acceptance. Each design iteration regenerates every cell's geometry and learned operator and solves the lattice as in Section 6.10, starting from the previous solution (Supplementary Note S9, Table ST21).

Case A is the \(2\times2\times2\) block of Section 6.9, loaded normal to the face opposite the clamp and started from its graded design; a twin optimisation with exact condensation starts from the same design (Table 6, Figure 12a). NICE converges after 24 design iterations and exact condensation after 23, both lowering the compliance by 2.0% with 20% less material. The final corner parameters of the two runs differ by at most 0.0051, and the exact compliances of the two final designs by \(1.25\times10^{-5}\) of their value. Exact checks of the NICE run at iterations 0, 12 and 23 give NICE compliances 0.011%, 0.018% and 0.028% below the exact ones, as Eq. (12) requires, and lattice-gradient errors of 0.069%, 0.17% and 0.33%, with cosines of at least 0.999996, per-variable errors of at most 0.41% of the largest exact component (95th percentile at most 0.28%) and the exact sign in every component (Table ST22). Both errors grow from the initial to the final design but remain small against the compliance reduction.

Plates B1 and B2 are a single layer of \(8\times4\) cells cut by a plane from one end of the lower long edge to the upper long edge, two cells from its other end, leaving 16 uncut cells and eight cut cells, two each with retained volumes of 8%, 33%, 67% and 92%; with the short side along \(x\), the cut normal has \(\vartheta=33.7^\circ\), inside the trained range. Clamped along its lower long edge, the plate is loaded on its remaining two-cell upper face, in plane along the long edge (B1) or out of plane, bending the layer (B2), from the uniform start \(\tau=0.40\). NICE lowers the compliance by 7.7% in 23 design iterations (B1) and by 8.3% in 30 (B2), with the bounds and the span and gradient limits active at both optima. [TBD-EXACT: exact compliance, surrogate compliance error and lattice-gradient error of the initial, an intermediate and the final design of B1 and B2.]

For comparison, a homogenised model is optimised with the same variables, constraints, \(V^*\) and optimiser. Its effective tensor, from periodic homogenisation of the uniform-thickness cell on the same discrete model, is evaluated at the local parameter on six trilinear elements per cell and axis, with the cut integrated by finite-cell quadrature (Table ST23). Relative to NICE on the cut geometry, it underestimates the compliance of the initial design by 26.7% under the in-plane load and 36.6% under bending, and that of its own final designs by 24.4% and 35.9%.

The optimisations find local optima only, and the NICE designs depend on the start. Evaluated with NICE, the homogenisation designs have 1.07% higher compliance than B1 and 0.34% lower than B2, at volumes within 0.05% of \(V^*\). Continued with NICE for 12 design iterations under the same \(V^*\), they lost a further 0.998% (XH_y) and 0.528% (XH_z) of compliance, still decreasing slowly, and ended 0.060% above B1 and 0.866% below B2 (Figure 12b–d). These comparisons rest on NICE compliances; [TBD-EXACT: exact compliances of H_y, H_z, XH_y and XH_z and the resulting differences].

The discrete model switched between every two consecutive design iterations of every run (Section 4.3): the numbers of active elements, ghost faces or retained coordinates changed in at least 2 of the 8 cells of case A and 8 of the 24 plate cells, the binary node features in at least 3 and 9, and the coarse-factor shift in at most four cells and not after iteration 14 (Table ST24). The recomputed residual stagnated between \(5.3\times10^{-5}\) and \(6.7\times10^{-3}\), as in Section 6.9, while the signed residual work of Eq. (18) stayed below \(3.1\times10^{-7}\) of the compliance. Where the exact local-support certificate of the geometry generator left an element unresolved in a near-tangent configuration, the free parameters of the failing cells were scaled by \(1\pm10^{-4}\) or, if needed, \(1\pm10^{-3}\), and the perturbed design was continued, in at most two iterations per run (Table ST24).

On one RTX 5090, plates of [TBD-SCALE: N1 to N2 cells (cut cells, free retained coordinates)] were run for [TBD-SCALE: k] design iterations each. Storing the streamed cell state exactly in packed form reduces its host memory by 23% and changes the compliance by at most \(1.1\times10^{-8}\) (Supplementary Note S9). [TBD-SCALE: time and GPU and host memory per design iteration against the number of cells (Figure 12e); time of a complete optimisation, stated as an extrapolation.] The accuracy of NICE at these sizes was not checked against the exact reference.

**Table 6. Thickness optimisation cases.** Compliance under a unit consistent face traction. Start: initial design, whose volume exceeds \(V^*\) by 25% in A, B1 and B2; final: last design iteration. Compliances are NICE values except where marked. Exact verification: exact discrete compliance of the NICE designs, with the amount by which the NICE compliance lies below it in parentheses. Time: mean over the design iterations, including geometry generation for every cell; the NICE runs needed at most 24.3 GiB of GPU and 5.0 GiB of host memory. H: homogenised macroscale model (4,464 trilinear elements). Details: Supplementary Note S9 and Tables ST21–ST24.

| Case | Cells (cut) | Load | Design iterations | Start compliance | Final compliance | Exact verification | Time per design iteration (s) |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: |
| A, NICE | 8 (4) | Normal traction on the face opposite the clamp | 24 | 24.5908 | 24.1038 | 24.5936 (0.011%), 26.0063 (0.018%), 24.1105 (0.028%) at iterations 0, 12, 23 | 141 |
| A, exact condensation | 8 (4) | As A | 23 | 24.5936\(^{a}\) | 24.1108\(^{a}\) | Exact throughout; final design 0.0013% above the NICE design | 469 |
| B1 | 24 (8) | Upper face, in plane along the long edge | 23 | 94.34 | 87.09 | [TBD-EXACT] | 574 |
| B2 | 24 (8) | Upper face, out of plane (bending) | 30 | 1,467.7 | 1,346.4 | [TBD-EXACT] | 639 |
| H_y | Macroscale | As B1 | 23 | 69.18\(^{b}\) (NICE 94.34) | 66.57\(^{b}\) (NICE 88.02) | [TBD-EXACT] | 4.8\(^{c}\) |
| H_z | Macroscale | As B2 | 27 | 930.7\(^{b}\) (NICE 1,467.7) | 859.9\(^{b}\) (NICE 1,341.8) | [TBD-EXACT] | 4.8\(^{c}\) |
| XH_y: NICE from H_y | 24 (8) | As B1 | 12 | 88.02 | 87.14 | [TBD-EXACT] | 545 |
| XH_z: NICE from H_z | 24 (8) | As B2 | 12 | 1,341.8 | 1,334.7 | [TBD-EXACT] | 643 |
| Scale | [TBD-SCALE] | [TBD-SCALE] | [TBD-SCALE] | [TBD-SCALE] | [TBD-SCALE] | Not checked | [TBD-SCALE] |

\(^{a}\) Exact compliance. \(^{b}\) Homogenised macroscale model; in parentheses the NICE compliance of the same design on the cut geometry. \(^{c}\) Macroscale model, excluding the one-off homogenisation of the cell.

![Figure 12](figures/F12_optimisation.png)

**Figure 12. Thickness optimisation with NICE.** (a) Case A: compliance histories of the NICE optimisation and of the twin optimisation with exact condensation, with the exact compliance of the NICE designs at iterations 0, 12 and 23; the compliance first rises while the volume is reduced from \(1.25V^*\) to \(V^*\). (b) Plates B1 (in-plane load) and B2 (bending): NICE compliance histories from the uniform start, the NICE compliance of the homogenisation designs H_y and H_z, and the NICE continuations XH_y and XH_z started from them. (c,d) Corner thickness parameters of the final designs B2 and XH_z on the plate. (e) [TBD-SCALE: time and memory per design iteration against the number of cells.] Plate compliances are NICE values; [TBD-EXACT: exact values of the plate designs].

---

## Proposed replacement sentences

**Abstract, final sentence** (replaces "[Placeholder: quantitative outcome of the Section 6.11 design example.]"):

> In thickness optimisation by the method of moving asymptotes driven by the field-based sensitivities, NICE and exact condensation reach designs of an eight-cell lattice whose exact compliances differ by 0.0013%, with the NICE lattice gradient within 0.33% of the exact one at the initial, an intermediate and the final design; on cut 24-cell plates, a homogenised model underestimates the compliance of the initial design by 27% and 37% relative to NICE, which optimises on the cut geometry directly [TBD-EXACT: exact compliance of the plate designs] [TBD-SCALE: largest lattice optimised on one GPU and its time per design iteration].

**Contribution (iii)** (replaces the bracket "[Placeholder: and a thickness design, Section 6.11]"; the preceding "and" before "a lattice-level cost comparison" becomes a comma):

> …, a lattice-level cost comparison with the whole-lattice direct solution and conventional exact condensation, and thickness optimisations driven by the field-based sensitivities, of an eight-cell lattice, reproducing the design obtained with exact condensation, and of cut 24-cell plates, compared with a homogenised model [TBD-SCALE: and of a plate of N cells] (Section 6.11).

**Section 8, paragraph** (replaces "[Placeholder: conclusion on the design-optimisation example of Section 6.11.]"):

> Driven by the field-based sensitivities, the method of moving asymptotes optimises the corner thickness parameters of NICE lattices although the discrete model switches in every design iteration. In an eight-cell lattice, NICE and exact condensation reach final designs whose corner parameters differ by at most 0.0051 and whose exact compliances differ by 0.0013%; at the initial, an intermediate and the final NICE design, the surrogate compliance error is below 0.03% and the lattice gradient within 0.33% of the exact one, with the exact sign in every component. On cut 24-cell plates, a homogenised model underestimates the compliance of the initial design by 27% under an in-plane load and 37% under bending relative to NICE, which optimises on the cut geometry directly; continuing the homogenisation designs with NICE lowered their compliance by a further 1.0% and 0.5%, so the optimisations find local optima only and the NICE designs depend on the start [TBD-EXACT: exact compliances of the plate designs]. [TBD-SCALE: largest plate optimised on one RTX 5090, its time and memory per design iteration, accuracy not verified at this size.]

---

## Number sources

Keys refer to `results/X6_opt/FACTS_6_11.json` (`F[...]`); "md" to the corresponding line of `FACTS_6_11.md`. Derived values were computed with the script below from the same file (and, for the cut geometry, from the archived layout `results/X6_opt/homog/plate841.json`).

```python
import json, math
D = '/home/user/curly-octo-goggles/docs/paper_p1/review_r1/results/X6_opt/'
F = json.load(open(D + 'FACTS_6_11.json'))
L = json.load(open(D + 'homog/plate841.json'))
# reductions and volume
for r in ('A', 'A_exact_twin', 'B1', 'B2', 'XH_y', 'XH_z'):
    print(r, 100 * (1 - F[r]['C_ratio']), F[r]['V0'] / F[r]['Vstar'], F[r]['V_final'] / F[r]['V0'])
# homogenisation: error of the macro model on its own final design, volume offset of the H designs
for t in ('y', 'z'):
    m, h = F['Hmacro_' + t], F['Hfine_' + t]
    print(t, 100 * (m['C_final_macro'] - h['C_fine']) / h['C_fine'], h['V_fine'] / F['B1']['Vstar'] - 1)
# switch log: cells with a change per design iteration, per group
G = dict(topo=('active', 'faces', 'ports'), feat=('cut_nodes', 'weak', 'el_fringe', 'gp_fringe'), shift=('shift',))
for r in ('A', 'B1', 'B2', 'XH_y', 'XH_z'):
    sw = F[r]['switches_per_iteration']
    for g, keys in G.items():
        n = [max(s[k] for k in keys) for s in sw]
        print(r, g, 'first', n[0], 'min', min(n), 'max', max(n), 'last k with change', max([s['k'] for s, x in zip(sw, n) if x] or [None]))
# cut geometry and retained volumes
print(L['args']['theta_deg'], math.degrees(math.atan2(L['normal'][1], L['normal'][0])),
      sorted({round(c['retained'], 4) for c in L['cells'] if c['kind'] == 'CUT'}), sum(c['kind'] == 'FULL' for c in L['cells']))
# packed storage: streamed state before/after (lines of F['storage']['packstore_compare.txt'])
print(1 - 3.7088 / 4.8285)
```

| Number in the draft | Value used | Source key (FACTS_6_11.json) or derivation |
| --- | --- | --- |
| Bounds 0.18, 0.69; span and gradient limits 0.45; move limit 5% | settings | Run arguments `args.tmin`, `tmax`, `span`, `grad`, `move` in `optA/optA/meta.json` and `plates/*/meta.json` (not in the fact sheet; settings, not results) |
| \(V^*=0.8\,V(\boldsymbol\tau^0)\) | 0.8 | `meta.json` `args.vfrac`; consistent with `A.V0 / A.Vstar` = `B1.V0 / B1.Vstar` = `B2.V0 / B2.Vstar` = 1.25 |
| Start volume 25% above \(V^*\); Figure 12a: volume from \(1.25V^*\) to \(V^*\) | 1.25 → 1.00 | `A/B1/B2.V_rel_trace[0]` = 1.25, `V_rel_final` ≈ 1.000 |
| Figure 12a: compliance first rises | peak at k = 4 (A), 3 (B1), 4 (B2) | `A/B1/B2.C_trace` |
| Case A design iterations: NICE 24, exact 23 | 24; 23 | `A.iterations`, `A_exact_twin.iterations` |
| Case A compliance reduction 2.0% (both runs) | 1.98%; 1.96% | 1 − `A.C_ratio` (0.98019); 1 − `A_exact_twin.C_ratio` (0.98037) |
| 20% less material | 0.8000 | `A.V_final / A.V0` = 1.03558 / 1.29448 |
| Final corner parameters of the two runs differ by at most 0.0051 | 0.00508 | `A_final_designs.tau_maxabs_diff` |
| Exact compliances of the two final designs differ by \(1.25\times10^{-5}\); 0.0013% (Table 6, Abstract, Section 8) | −1.2513e-5 | `A_final_designs.rel_diff` (NICE design lower) |
| NICE compliances 0.011%, 0.018%, 0.028% below the exact ones; "below 0.03%" (Section 8) | −1.118e-4, −1.818e-4, −2.791e-4 | `A_checks.0/12/23.surrogate_err` |
| Lattice-gradient errors 0.069%, 0.17%, 0.33% | 6.879e-4, 1.697e-3, 3.287e-3 | `A_checks.0/12/23.grad_rel_err` |
| Cosines of at least 0.999996 | min 0.9999966 | `A_checks.*.grad_cos` (0.9999998, 0.9999987, 0.9999966) |
| Per-variable errors at most 0.41% of the largest exact component | max 4.06e-3 | `A_checks.23.comp_err_rel_to_max.max` (others 6.29e-4, 1.80e-3) |
| 95th percentile at most 0.28% | max 2.79e-3 | `A_checks.23.comp_err_rel_to_max.p95` (others 5.11e-4, 1.17e-3) |
| Exact sign in every component | 1.000 | `A_checks.*.sign_agreement` |
| Table 6, A NICE: 24.5908 → 24.1038 | 24.590814 / 24.103775 | `A.C0`, `A.C_final` |
| Table 6, A exact checks: 24.5936, 26.0063, 24.1105 | 24.593563, 26.006304, 24.110504 | `A_checks.0/12/23.C_exact` |
| Table 6, A exact condensation: 24.5936 → 24.1108 | 24.593563 / 24.110806 | `A_exact_twin.C0`, `A_exact_twin.C_final` |
| Table 6, time per design iteration: A 141 s, exact condensation 469 s | 140.83; 469.26 | `A.iter_s_mean`, `A_exact_twin.iter_s_mean` |
| Table 6, cells (cut): A 8 (4) | 8 (4) | Section 6.9 of the manuscript; `A.switches_per_iteration[0].cut_nodes` = 4 cells with cut nodes |
| Plates: 16 uncut, 8 cut, two each with retained volumes 8%, 33%, 67%, 92%; 24 cells | 0.0833, 0.3333, 0.6667, 0.9167 | `homog/plate841.json` `cells[*].kind`, `cells[*].retained` (script above) |
| Cut normal \(\vartheta=33.7^\circ\) | 33.690° | `homog/plate841.json` `args.theta_deg`, `normal` (script above) |
| Uniform start \(\tau=0.40\) | 0.4 | `B1.tau0_range` = `B2.tau0_range` = [0.4, 0.4] |
| B1: 7.7% in 23 design iterations; Table 6: 94.34 → 87.09, 574 s | 0.92313; 23; 94.3407 / 87.0889; 573.92 | `B1.C_ratio`, `B1.iterations`, `B1.C0`, `B1.C_final`, `B1.iter_s_mean` |
| B2: 8.3% in 30 design iterations; Table 6: 1,467.7 → 1,346.4, 639 s | 0.91729; 30; 1467.743 / 1346.354; 639.23 | `B2.C_ratio`, `B2.iterations`, `B2.C0`, `B2.C_final`, `B2.iter_s_mean` |
| Bounds and span and gradient limits active at both plate optima | 0.18000 / 0.69000 / 0.44998 / 0.45000 (B1); 0.18000 / 0.69000 / 0.44999 / 0.45000 (B2) | `B1/B2.tau_min_final`, `tau_max_final`, `span_max_final`, `grad_max_final` |
| Six trilinear elements per cell and axis; 4,464 elements (Table 6 caption) | 6; 4464 | `homog_macro.py` default `--m 6`; `Hmacro_y.meta.elements` = `Hmacro_z.meta.elements` = 4464 |
| Homogenised model underestimates the initial compliance by 26.7% (in plane) and 36.6% (bending); 27% and 37% (Abstract, Section 8) | −0.26673; −0.36590 | `comparison.y.homog_prediction_error_initial`, `comparison.z.homog_prediction_error_initial` |
| ... and that of its own final designs by 24.4% and 35.9% | −24.37%; −35.91% | (`Hmacro_y.C_final_macro` − `Hfine_y.C_fine`) / `Hfine_y.C_fine`; same for z (script above) |
| Table 6, H_y and H_z: 23 and 27 iterations; macroscale 69.18 → 66.57 and 930.7 → 859.9 | 69.177 / 66.566; 930.693 / 859.925 | `Hmacro_y/z.iterations`, `C0_macro`, `C_final_macro` |
| Table 6, H time per iteration 4.8 s | 4.75; 4.83 | `Hmacro_y.seconds_per_iteration`, `Hmacro_z.seconds_per_iteration` |
| Table 6, NICE compliance of the H designs: 88.02, 1,341.8 | 88.0193; 1341.784 | `Hfine_y.C_fine`, `Hfine_z.C_fine` (= `comparison.*.C_H_fine`) |
| H designs 1.07% above B1 and 0.34% below B2 (NICE) | 0.010684; −0.003394 | `comparison.y.H_vs_B`, `comparison.z.H_vs_B` |
| Volumes within 0.05% of \(V^*\) | H_y +0.040%, H_z −0.045%; B1, B2, XH_y, XH_z within −0.002% | `Hfine_y.V_fine / B1.Vstar` − 1, `Hfine_z.V_fine / B2.Vstar` − 1; `B1/B2/XH_y/XH_z.V_rel_final` |
| Cross-start: 12 design iterations; a further 0.998% and 0.528%; "1.0% and 0.5%" (Section 8) | 12; −0.009975; −0.005282 | `XH_y.iterations`, `XH_z.iterations`; `comparison.y.X_vs_H`, `comparison.z.X_vs_H` |
| Still decreasing slowly at the end | 87.1546 → 87.1413; 1334.7545 → 1334.6969 | `XH_y.C_trace[-2:]`, `XH_z.C_trace[-2:]` |
| Ended 0.060% above B1 and 0.866% below B2 | 0.000602; −0.008658 | `comparison.y.X_vs_B`, `comparison.z.X_vs_B` |
| Table 6, XH_y: 88.02 → 87.14, 545 s; XH_z: 1,341.8 → 1,334.7, 643 s | 88.0193 / 87.1413, 545.18; 1341.784 / 1334.697, 642.93 | `XH_y/XH_z.C0`, `C_final`, `iter_s_mean` |
| Switches: active elements, ghost faces or retained coordinates changed in at least 2 of 8 (A) and 8 of 24 (plates) cells between consecutive iterations | min 2 (A), 14 (B1), 9 (B2), 13 (XH_y), 8 (XH_z) | `*.switches_per_iteration`, keys `active`, `faces`, `ports` (script above) |
| Switches: binary node features in at least 3 and 9 cells | min 3 (A), 15 (B1), 10 (B2), 16 (XH_y), 9 (XH_z) | `*.switches_per_iteration`, keys `cut_nodes`, `weak`, `el_fringe`, `gp_fringe` |
| Switches: coarse-factor shift in at most four cells, not after iteration 14 | max 4 (B1, B2); last change at k = 14 (A, B1), 7 (B2), 9 (XH_y), none (XH_z) | `*.switches_per_iteration`, key `shift` |
| Recomputed residual between \(5.3\times10^{-5}\) and \(6.7\times10^{-3}\) | 5.25e-5 (A) to 6.72e-3 (B2) | `A.true_residual_range[0]`, `B2.true_residual_range[1]`; all runs lie within |
| Signed residual work below \(3.1\times10^{-7}\) of the compliance | max 3.07e-7 | `B2.Ut_rho_rel_absmax` (A 2.7e-8, B1 2.5e-7, XH_y 4.4e-8, XH_z 1.7e-7) |
| Geometry fallback: \(1\pm10^{-4}\) or \(1\pm10^{-3}\); in at most two iterations per run | A: k = 16 (applied eps 1e-4), k = 19 (1e-3); B1, B2: k = 5 (1e-3); XH_z: k = 2 (1e-4); XH_y: none | `*.body_perturbations_applied`; tried values in `*.body_perturbations` |
| Table 6 caption: at most 24.3 GiB of GPU and 5.0 GiB of host memory (NICE runs) | 24.33; 4.99 | `B1.gpu_peak_gb_max` (= B2); `B1.host_peak_gb_max` (A 15.7/3.5, XH_y 22.4/4.8, XH_z 22.2/4.3) |
| Packed storage: host memory of the streamed state reduced by 23% | 4.8285 → 3.7088 GB (23.2%) | `storage["packstore_compare.txt"]`, lines "base stream_GB" and "packed stream_GB" |
| Packed storage: compliance changed by at most \(1.1\times10^{-8}\) | 1.14e-8 | `storage["packstore_compare.txt"]`, "max rel" |
