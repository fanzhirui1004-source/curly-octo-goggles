# Draft of Section 6.11 (for insertion into MANUSCRIPT_EN.md)

<!-- Draft of 2026-09-30, revised after review. Every number is taken from results/X6_opt/FACTS_6_11.json / FACTS_6_11.md
or computed from the archived run records by the script shown under "Number sources". [TBD-EXACT: ...] marks the pending
exact checks of the plate designs, [TBD-SCALE: ...] the pending scale demonstration. Supplementary Note S9 and Tables
ST21-ST25 are written separately (DRAFT_S9_EN.md). Body length: 841 words counted raw, of which 77 are placeholder text
(764 without); the fills should take at most about 130 words so that the finished section stays below 900. For length,
the packed storage of the streamed cell state is left to Supplementary Note S9.6, which states it with its GPU-memory cost
and the unpacked comparison. The preconditioned conjugate gradients are cited from Section 6.9, where they are defined
(Section 6.10 refers to them there as well). Reference to add to the list: Svanberg, K. (1987).
[The method of moving asymptotes—a new method for structural optimization](https://doi.org/10.1002/nme.1620240207).
*International Journal for Numerical Methods in Engineering*, 24(2), 359–373. No periodic-homogenisation reference is in
references_verified.bib yet; one may be cited after "periodic homogenisation" once it has been verified. The figure is
written by figures_src/fig_opt.py as figures/F13_optimisation.{svg,pdf,png} (F12_* is Figure 7). -->

### 6.11. Thickness design optimisation

The corner thickness parameters at the lattice vertices, shared through \(\boldsymbol\tau_m=\boldsymbol\tau_m(\boldsymbol\tau_g)\), are optimised for minimum compliance under one unit consistent face traction subject to \(V\le V^*=0.8\,V(\boldsymbol\tau^0)\), with \(V\) the material volume of the discrete model. The parameters at the vertices in the plane of the loaded face (9 of 27 in case A, 10 of 74 in the plates) are fixed, which keeps the nodal load independent of the design and removes the load-derivative term \(2f_{g,c}^T\widehat U\) of Appendix H. The optimiser receives the field-based estimates of Eq. (13) summed over the cells at each vertex, \(\sum_m(\partial\boldsymbol\tau_m/\partial\boldsymbol\tau_g)^T\widetilde{\boldsymbol s}_m\), not the complete surrogate derivative of Eq. (14). The bounds \(0.18\le\tau\le0.69\) and limits of 0.45 on each cell's corner span and gradient norm (largest values reached: 0.450 and 0.452) keep every cell inside the trained domain ([0.175, 0.699], at most 0.47; Section 7.5). Each design iteration regenerates every cell's geometry and learned operator, solves the lattice by the preconditioned conjugate gradients of Section 6.9 from the previous solution and takes one step of the method of moving asymptotes [Svanberg (1987)](https://doi.org/10.1002/nme.1620240207), with a move limit of 5% of the parameter range and no line search (Tables ST21 and ST25).

Case A, the \(2\times2\times2\) block of Section 6.9 loaded normal to the face opposite the clamp, is optimised from its graded design with NICE and, as a twin, with exact condensation (Table 6, Figure 12a). NICE stops after 24 design iterations and exact condensation after 23, both lowering the compliance by 2.0% with 20% less material; their final corner parameters differ by at most 0.0051, and the exact compliances of their final designs by \(1.25\times10^{-5}\) of their value. At iterations 0, 12 and 23, the NICE compliance lies 0.011%, 0.018% and 0.028% below the exact one, consistent with Eq. (12), and the lattice gradient has errors of 0.069%, 0.17% and 0.33%, cosines of at least 0.999996, 95th-percentile per-variable errors of at most 0.28% of the largest component and the exact sign in every component (Table ST22).

Plates B1 and B2 are a single layer of \(8\times4\) cells from which a planar cut removes eight, leaving 16 uncut and eight cut cells (Table ST23). With the in-plane axes interchanged, the cut normal (\(\vartheta=33.7^\circ\)) lies in the orientation in which NICE was mainly evaluated (\(0<\vartheta<\pi/4\), Sections 6.1 and 7.5). Clamped along one long edge and started from the uniform \(\tau=0.40\), the plate is loaded on the opposite side face, which the cut leaves two cells wide, in plane along the long edge (B1) or out of plane (B2, bending). NICE lowers the compliance by 7.7% in 23 design iterations (B1) and 8.3% in 30 (B2), both final designs reaching the bounds and the span and gradient limits. [TBD-EXACT: exact compliance, surrogate compliance error, lattice-gradient error, cosine, per-variable 95th percentile/maximum and sign agreement at the initial, an intermediate and the final design of B1 and B2.]

A homogenised model, with the tensor of the uniform-thickness cell from periodic homogenisation on the same discrete model (Table ST23), is optimised with the same variables, bounds and limits, volume fraction 0.8 and optimiser; its volume, from the homogenised material volume fraction \(V^H(\tau)\), equals the discrete material volume at the uniform start to \(1.2\times10^{-7}\). Relative to NICE, it underestimates the compliance of the initial design by 26.7% under the in-plane load and 36.6% in bending, since the single layer offers no separation of scales through its thickness.

Evaluated with NICE, the homogenisation designs Hom-\(y\) and Hom-\(z\) have 1.1% higher compliance than B1 and 0.34% lower than B2, at volumes within 0.05% of \(V^*\). The optimisations are local, and their final designs depend on the start: continuing from Hom-\(y\) and Hom-\(z\) under the same \(V^*\), NICE lowered their compliance by a further 1.0% (X-\(y\)) and 0.53% (X-\(z\)) in 12 design iterations, still improving when stopped; X-\(z\) ended 0.87% below B2, whereas X-\(y\) ended 0.060% above B1, a difference of the order of the surrogate error of case A (Figure 12b–d). [TBD-EXACT: exact compliances of Hom-\(y\), Hom-\(z\), X-\(y\), X-\(z\) and their differences.]

The discrete model switched between all consecutive design iterations (Section 4.3): active elements, ghost faces, retained coordinates or binary node features changed in at least 3 of 8 cells (case A) and 9 of 24 (plates), the coarse-factor shift in at most four cells (Table ST24). The recomputed residual stagnated between \(5.3\times10^{-5}\) and \(6.7\times10^{-3}\), as in Section 6.9, while the signed residual work of Eq. (18) stayed below \(3.1\times10^{-7}\) of the compliance. Where the geometry generator left the exact certification of an element's local material support unresolved (Appendix A.1), the free parameters of the failing cells were scaled by \(1\pm10^{-4}\) or \(1\pm10^{-3}\), at most twice per run (Table ST24).

On one RTX 5090, plates of [TBD-SCALE: N1 to N2 cells (cut cells, free retained coordinates)] were run for [TBD-SCALE: k] design iterations each (Supplementary Note S9). [TBD-SCALE: time and GPU and host memory per design iteration against the number of cells (Figure 12e); time of a complete optimisation, stated as an extrapolation.] NICE's accuracy at these sizes was not verified against the exact reference.

**Table 6. Thickness optimisation cases.** Compliance under a unit consistent face traction. Start: initial design, whose volume exceeds \(V^*\) by 25% in A, B1 and B2; final: last design iteration. Compliances are NICE values except where marked. Exact verification: exact discrete compliance of the NICE designs, with the amount by which the NICE compliance lies below it in parentheses. Time: mean over the design iterations, including geometry generation for every cell. The optimisation runs integrate the moments and their reverse-mode derivatives without the fused integration kernels of the timed route of Table 5, so these times, and those of the exact-condensation twin, a verification run, are not comparable with Table 5 (Supplementary Note S9, Table ST22c). The NICE runs needed at most 24.3 GiB of GPU memory and a peak of 5.0 GiB of host memory (main process; geometry generation excluded). H: homogenised macroscale model (4,464 trilinear elements). Details: Supplementary Note S9 and Tables ST21–ST25.

| Case | Cells (cut) | Load | Design iterations | Start compliance | Final compliance | Exact verification | Time per design iteration (s) |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: |
| A, NICE | 8 (4) | Normal traction on the face opposite the clamp | 24 | 24.5908 | 24.1038 | 24.5936 (0.011%), 26.0063 (0.018%), 24.1105 (0.028%) at iterations 0, 12, 23 | 141 |
| A, exact condensation | 8 (4) | As A | 23 | 24.5936\(^{a}\) | 24.1108\(^{a}\) | Exact throughout; final design 0.0013% above the NICE design | 469 |
| B1 | 24 (8) | Face opposite the clamp, in plane along the long edge | 23 | 94.34 | 87.09 | [TBD-EXACT] | 574 |
| B2 | 24 (8) | Face opposite the clamp, out of plane (bending) | 30 | 1,467.7 | 1,346.4 | [TBD-EXACT] | 639 |
| Hom-\(y\) | Macroscale | As B1 | 23 | 69.18\(^{b}\) (NICE 94.34) | 66.57\(^{b}\) (NICE 88.02) | [TBD-EXACT] | — |
| Hom-\(z\) | Macroscale | As B2 | 27 | 930.7\(^{b}\) (NICE 1,467.7) | 859.9\(^{b}\) (NICE 1,341.8) | [TBD-EXACT] | — |
| X-\(y\): NICE from Hom-\(y\) | 24 (8) | As B1 | 12 | 88.02 | 87.14 | [TBD-EXACT] | 545 |
| X-\(z\): NICE from Hom-\(z\) | 24 (8) | As B2 | 12 | 1,341.8 | 1,334.7 | [TBD-EXACT] | 643 |
| Scale | [TBD-SCALE] | [TBD-SCALE] | [TBD-SCALE] | [TBD-SCALE] | [TBD-SCALE] | Not checked | [TBD-SCALE] |

\(^{a}\) Exact compliance. \(^{b}\) Homogenised macroscale model; in parentheses the NICE compliance of the same design on the cut geometry.

![Figure 12](figures/F13_optimisation.png)

**Figure 12. Thickness optimisation with NICE.** (a) Case A: compliance histories of the NICE optimisation and of the twin optimisation with exact condensation, with the exact compliance of the NICE designs at iterations 0, 12 and 23 (open circles); the compliance first rises while the volume is reduced from \(1.25V^*\) to \(V^*\) (shading, iterations 0–3). Lower strip: NICE compliance relative to the twin run at the same design iteration (line; the designs of the two runs differ) and to the exact compliance of the same design (circles: −0.011%, −0.018%, −0.028%); at iterations 16 and 19 the NICE design had been perturbed by the geometry-generation fallback (Table ST24), and at iteration 19 its volume lay 0.036% above \(V^*\). (b) Plates B1 (in-plane load, left) and B2 (bending, right): compliance divided by \(C_0\), the initial NICE compliance of B1 or B2 at the uniform start (94.34 and 1,467.7). The homogenisation designs Hom-\(y\) and Hom-\(z\) evaluated with NICE (stars) and the NICE continuations X-\(y\) and X-\(z\) started from them are divided by the same \(C_0\) and drawn over their own design iterations. Shading: design iterations with \(V>V^*\) of the runs from the uniform start; the continuations start within 0.05% of \(V^*\). Lower strips: the range marked by the bracket, enlarged. (c,d) Corner thickness parameters of the final designs B2 and X-\(z\), drawn with the long side horizontal, layer \(z=0\) (layer \(z=1\) differs by at most \(7.0\times10^{-4}\) in B2 and \(2.0\times10^{-4}\) in X-\(z\)); circles: design variables; squares: vertices in the plane of the loaded face, held fixed; vertices drawn in the removed region are corners of cut cells. ⊗: load face, traction normal to the plate. (e) [TBD-SCALE: time and memory per design iteration against the number of cells.] Plate compliances are NICE values; [TBD-EXACT: exact values of the plate designs].

---

## Proposed replacement sentences

**Abstract, final sentence** (replaces "[Placeholder: quantitative outcome of the Section 6.11 design example.]"). The Abstract has 253 words without the placeholder, above the 150–250 words typical of the journal (CONSOLIDATED.md, I-55); the sentence below adds 51 words plus the placeholders, the short variant 25. Either needs a matching cut elsewhere in the Abstract if the limit is enforced.

> In thickness optimisation by the method of moving asymptotes, NICE reaches the eight-cell design of exact condensation to within 0.0051 in every corner parameter, with lattice-gradient errors below 0.33%; on cut 24-cell plates, a homogenised model underestimates the compliance of the initial design by 27% and 37% relative to NICE [TBD-EXACT: exact check of the plate designs] [TBD-SCALE: largest lattice optimised on one RTX 5090 and its time per design iteration].

Short variant:

> In thickness optimisation by the method of moving asymptotes, NICE reaches the eight-cell design of exact condensation to within 0.0051 in every corner parameter [TBD-EXACT: plate designs] [TBD-SCALE: largest lattice optimised on one RTX 5090].

**Contribution (iii)** (replaces the bracket "[Placeholder: and a thickness design, Section 6.11]"; the preceding "and" before "a lattice-level cost comparison" becomes a comma):

> …, a lattice-level cost comparison with the whole-lattice direct solution and conventional exact condensation, and thickness optimisations driven by the field-based sensitivities, of an eight-cell lattice, matching the design obtained with exact condensation to within 0.0051 in every corner parameter, and of cut 24-cell plates, compared with a homogenised model [TBD-SCALE: and of a plate of N cells] (Section 6.11).

**Section 8, paragraph** (replaces "[Placeholder: conclusion on the design-optimisation example of Section 6.11.]"):

> Driven by the field-based sensitivities, the method of moving asymptotes optimises the corner thickness parameters of NICE lattices although the discrete model switches in every design iteration. In an eight-cell lattice, NICE and exact condensation reach final designs whose corner parameters differ by at most 0.0051 and whose exact compliances differ by 0.0013%; at the initial, an intermediate and the final NICE design, the surrogate compliance error is below 0.03% and the lattice gradient within 0.33% of the exact one, with the exact sign in every component. On cut 24-cell plates, a homogenised model underestimates the compliance of the initial design by 27% under an in-plane load and 37% under bending relative to NICE, which optimises on the cut geometry directly; continued from the homogenisation design, the NICE optimisation under bending ended 0.87% below the one from the uniform start, so the optimisations are local [TBD-EXACT: exact compliances of the plate designs]. [TBD-SCALE: largest plate optimised on one RTX 5090, its time and memory per design iteration, accuracy not verified at this size.]

---

## Number sources

Keys refer to `results/X6_opt/FACTS_6_11.json` (`F[...]`); "md" to the corresponding line of `FACTS_6_11.md`. Derived values were computed with the script below from the same file and, where stated, from the archived run records in `results/X6_opt` (`history.jsonl` and `meta.json` of each run, `homog/plate841.json`, `homog/H_y/history.jsonl`). Output of the script is given after it.

```python
import json, math, re
D = '/home/user/curly-octo-goggles/docs/paper_p1/review_r1/results/X6_opt/'
F = json.load(open(D + 'FACTS_6_11.json'))
L = json.load(open(D + 'homog/plate841.json'))
RUNS = dict(A='optA/optA', B1='plates/plateB1', B2='plates/plateB2', XH_y='plates/xstartH_y', XH_z='plates/xstartH_z')
hist = lambda p: [h for h in map(json.loads, open(D + p + '/history.jsonl')) if h.get('event') == 'ITER']
# reductions and volume
for r in ('A', 'A_exact_twin', 'B1', 'B2', 'XH_y', 'XH_z'):
    print(r, 100 * (1 - F[r]['C_ratio']), F[r]['V0'] / F[r]['Vstar'], F[r]['V_final'] / F[r]['V0'])
# fixed vertices; largest corner span and gradient norm over every analysed design
for r, p in RUNS.items():
    H = hist(p)
    print(r, 'vertices', F[r]['n_vertices'], 'fixed', F[r]['n_fixed'],
          'span max %.5f grad max %.5f' % (max(h['span_max'] for h in H), max(h['grad_max'] for h in H)),
          'cells with cut nodes at k=0', sum(v['cut_nodes'] > 0 for v in H[0]['fps'].values()))
# homogenisation: error of the macro model on its own final design, volume offset of the H designs,
# macroscale volume against the discrete material volume at the uniform start
for t in ('y', 'z'):
    m, h = F['Hmacro_' + t], F['Hfine_' + t]
    print(t, 100 * (m['C_final_macro'] - h['C_fine']) / h['C_fine'], h['V_fine'] / F['B1']['Vstar'] - 1)
Vm0 = json.loads(open(D + 'homog/H_y/history.jsonl').readline())['V']
print('macro/fine volume at the uniform start - 1:', Vm0 / F['B1']['V0'] - 1)
# switch log: per step, number of cells whose active elements, ghost faces, retained coordinates or binary node
# features (cut-band and weak-support indicators) changed; largest number of cells whose coarse-factor shift changed
base = lambda c: c.rsplit('_o', 1)[0]
K = ('active', 'faces', 'ports', 'cut_nodes', 'weak')
for r, p in RUNS.items():
    H = hist(p); n = []
    for a, b in zip(H[:-1], H[1:]):
        fa = {base(c): v for c, v in a['fps'].items()}; fb = {base(c): v for c, v in b['fps'].items()}
        n.append(sum(any(fa[c][k] != fb[c][k] for k in K) for c in fb))
    print(r, 'cells changed per step: min', min(n), 'max', max(n), 'shift max', max(s['shift'] for s in F[r]['switches_per_iteration']))
# cut geometry and retained volumes
print(L['args']['theta_deg'], math.degrees(math.atan2(L['normal'][1], L['normal'][0])),
      sorted({round(c['retained'], 4) for c in L['cells'] if c['kind'] == 'CUT'}), sum(c['kind'] == 'FULL' for c in L['cells']))
# packed storage, parsed from the archived comparison
S = F['storage']['packstore_compare.txt']
num = lambda pat: float(re.search(pat, S).group(1))
b, pk = num(r'base stream_GB ([\d.]+)'), num(r'packed stream_GB ([\d.]+)')
gb, gp = num(r'base stream_GB .*? peak ([\d.]+)'), num(r'packed stream_GB .*? peak ([\d.]+)')
print('stream reduction', 1 - pk / b, 'peak GPU +GiB', (gp - gb) * 1e9 / 2**30,
      'compliance max rel', num(r'bitwise equal \w+ max rel ([\d.e-]+)'), 'unpacked twice', num(r'compliance max rel ([\d.e-]+) sens'))
```

Output:

```
A 1.980573424345311 1.25 0.799998518154179
A_exact_twin 1.9629410974115968 1.25 0.7999983564873573
B1 7.68678514387926 1.25 0.7999935135599451
B2 8.27050878225326 1.25 0.7999982776095533
XH_y 0.9975242886915314 1.0003988310011571 0.9995833260316811
XH_z 0.5281628218883672 0.9995456914157413 1.0004497173261515
A vertices 27 fixed 9 span max 0.44388 grad max 0.45184 cells with cut nodes at k=0 4
B1 vertices 74 fixed 10 span max 0.44998 grad max 0.45093 cells with cut nodes at k=0 8
B2 vertices 74 fixed 10 span max 0.45000 grad max 0.45133 cells with cut nodes at k=0 8
XH_y vertices 74 fixed 10 span max 0.44999 grad max 0.45012 cells with cut nodes at k=0 8
XH_z vertices 74 fixed 10 span max 0.44999 grad max 0.45003 cells with cut nodes at k=0 8
y -24.37343351787554 0.00039883100115711834
z -35.91177633445161 -0.00045430858425865583
macro/fine volume at the uniform start - 1: 1.2056916620650782e-07
A cells changed per step: min 3 max 8 shift max 2
B1 cells changed per step: min 15 max 24 shift max 4
B2 cells changed per step: min 10 max 24 shift max 4
XH_y cells changed per step: min 16 max 18 shift max 1
XH_z cells changed per step: min 9 max 9 shift max 0
33.690067525979785 33.690067525979785 [0.0833, 0.3333, 0.6667, 0.9167] 16
stream reduction 0.23189396292844566 peak GPU +GiB 0.4004687070846555 compliance max rel 1.1396192123895163e-08 unpacked twice 8.865547484895399e-09
```

| Number in the draft | Value used | Source key (FACTS_6_11.json) or derivation |
| --- | --- | --- |
| Bounds 0.18, 0.69; span and gradient limits 0.45; move limit 5% | settings | Run arguments `args.tmin`, `tmax`, `span`, `grad`, `move` in `optA/optA/meta.json` and `plates/*/meta.json` (settings, not results) |
| Largest values reached: span 0.450, gradient norm 0.452 | 0.45000 (B2); 0.45184 (A, k = 14), 0.45133 (B2) | `span_max`, `grad_max` of every record of each `history.jsonl` (script); training limits 0.47 and [0.175, 0.699]: Section 7.5 |
| Fixed vertices: 9 of 27 (case A), 10 of 74 (plates) | 9/27; 10/74 | `A.n_fixed`, `A.n_vertices`; `B1/B2/XH_y/XH_z.n_fixed`, `n_vertices` |
| \(V^*=0.8\,V(\boldsymbol\tau^0)\); volume fraction 0.8 of the homogenised model | 0.8 | `meta.json` `args.vfrac` (fine scale and `homog/H_y/meta.json`); consistent with `A.V0 / A.Vstar` = `B1.V0 / B1.Vstar` = `B2.V0 / B2.Vstar` = 1.25 |
| Homogenised volume equals the discrete material volume at the uniform start to \(1.2\times10^{-7}\) | 1.2057e-07 | `homog/H_y/history.jsonl` record k = 0 `V` (4.5690745) over `B1.V0` (4.5690740) (script); \(V^*\) of the homogenised model 3.6552596 against `B1.Vstar` 3.6552592 |
| Start volume 25% above \(V^*\); Figure 12a: volume from \(1.25V^*\) to \(V^*\) | 1.25 → 1.00 | `A/B1/B2.V_rel_trace[0]` = 1.25, `V_rel_final` ≈ 1.000 |
| Figure 12a: compliance first rises | peak at k = 4 (A), 3 (B1), 4 (B2) | `A/B1/B2.C_trace` |
| Case A design iterations: NICE 24, exact 23 | 24; 23 | `A.iterations`, `A_exact_twin.iterations` |
| Case A compliance reduction 2.0% (both runs) | 1.98%; 1.96% | 1 − `A.C_ratio` (0.98019); 1 − `A_exact_twin.C_ratio` (0.98037) |
| 20% less material | 0.8000 | `A.V_final / A.V0` = 1.03558 / 1.29448 |
| Final corner parameters of the two runs differ by at most 0.0051 (body, Abstract, contribution (iii), Section 8) | 0.00508 | `A_final_designs.tau_maxabs_diff` |
| Exact compliances of the two final designs differ by \(1.25\times10^{-5}\); 0.0013% (Table 6, Section 8) | −1.2513e-5 | `A_final_designs.rel_diff` (NICE design lower) |
| NICE compliances 0.011%, 0.018%, 0.028% below the exact ones; "below 0.03%" (Section 8); "surrogate error of case A" | −1.118e-4, −1.818e-4, −2.791e-4 | `A_checks.0/12/23.surrogate_err` |
| Lattice-gradient errors 0.069%, 0.17%, 0.33%; "below 0.33%" (Abstract), "within 0.33%" (Section 8) | 6.879e-4, 1.697e-3, 3.287e-3 | `A_checks.0/12/23.grad_rel_err` |
| Cosines of at least 0.999996 | min 0.9999966 | `A_checks.*.grad_cos` (0.9999998, 0.9999987, 0.9999966) |
| 95th-percentile per-variable errors at most 0.28% of the largest exact component | max 2.79e-3 | `A_checks.23.comp_err_rel_to_max.p95` (others 5.11e-4, 1.17e-3); the maxima (at most 4.06e-3) are left to Table ST22 |
| Exact sign in every component | 1.000 | `A_checks.*.sign_agreement` |
| Table 6, A NICE: 24.5908 → 24.1038 | 24.590814 / 24.103775 | `A.C0`, `A.C_final` |
| Table 6, A exact checks: 24.5936, 26.0063, 24.1105 | 24.593563, 26.006304, 24.110504 | `A_checks.0/12/23.C_exact` |
| Table 6, A exact condensation: 24.5936 → 24.1108 | 24.593563 / 24.110806 | `A_exact_twin.C0`, `A_exact_twin.C_final` |
| Table 6, time per design iteration: A 141 s, exact condensation 469 s | 140.83; 469.26 | `A.iter_s_mean`, `A_exact_twin.iter_s_mean` |
| Table 6 caption: optimisation runs without the fused integration kernels of Table 5, times not comparable | case A phases: geometry 14.0 s, front end 20.7 s, sensitivities and volume gradient 46.4 s; Table 5 route 9.7 s and 3.7 s | `A.phase_means` (`bodies_s`, `prep_s`, `sens_s`); `docs/paper_p1/evidence/d5_off.json` (`args.tet_triton` = true, `iterations[0].phases.front_end` 9.73, `sens_total` 3.69); `opt_design.py` has no such option; S9.1 |
| Table 6, cells (cut): A 8 (4) | 8 (4) | Section 6.9 of the manuscript; 4 cells with cut-band nodes in record k = 0 of `optA/optA/history.jsonl` (script) |
| Plates: eight of 32 cells removed; 16 uncut, 8 cut (retained volumes 0.083, 0.333, 0.667, 0.917, two each: Table ST23); 24 cells | 16 FULL + 8 CUT of `shape` 4 × 8 × 1 | `homog/plate841.json` `shape`, `cells[*].kind`, `cells[*].retained` (script) |
| Clamped along one long edge; the opposite side face, loaded, is left two cells wide by the cut | clamp internal `x,min`, load `x,max`; cut normal (0.832, 0.555, 0) through the points (x, y) = (0, 8), a corner, and (4, 2), checked with `b_global` | `plates/*/meta.json` `args.clamp`, `args.load`; `homog/plate841.json` `normal`, `shape` [4, 8, 1], cut cells at internal x = 3, y = 2, 3 (retained 0.667, 0.083); remaining full cells on that face: y = 0, 1 |
| Cut normal \(\vartheta=33.7^\circ\) | 33.690° | `homog/plate841.json` `args.theta_deg`, `normal` (script) |
| Uniform start \(\tau=0.40\) | 0.4 | `B1.tau0_range` = `B2.tau0_range` = [0.4, 0.4] |
| B1: 7.7% in 23 design iterations; Table 6: 94.34 → 87.09, 574 s | 0.92313; 23; 94.3407 / 87.0889; 573.92 | `B1.C_ratio`, `B1.iterations`, `B1.C0`, `B1.C_final`, `B1.iter_s_mean` |
| B2: 8.3% in 30 design iterations; Table 6: 1,467.7 → 1,346.4, 639 s | 0.91729; 30; 1467.743 / 1346.354; 639.23 | `B2.C_ratio`, `B2.iterations`, `B2.C0`, `B2.C_final`, `B2.iter_s_mean` |
| Both final plate designs reach the bounds and the span and gradient limits | 0.18000 / 0.69000 / 0.44998 / 0.45000 (B1); 0.18000 / 0.69000 / 0.44999 / 0.45000 (B2) | `B1/B2.tau_min_final`, `tau_max_final`, `span_max_final`, `grad_max_final` |
| 4,464 trilinear elements (Table 6 caption) | 4464 | `Hmacro_y.meta.elements` = `Hmacro_z.meta.elements` = 4464 (six per cell and axis, `homog_macro.py` default `--m 6`) |
| Homogenised model underestimates the initial compliance by 26.7% (in plane) and 36.6% (bending); 27% and 37% (Abstract, Section 8) | −0.26673; −0.36590 | `comparison.y.homog_prediction_error_initial`, `comparison.z.homog_prediction_error_initial` |
| Table 6, H\(_y\) and H\(_z\): 23 and 27 iterations; macroscale 69.18 → 66.57 and 930.7 → 859.9 | 69.177 / 66.566; 930.693 / 859.925 | `Hmacro_y/z.iterations`, `C0_macro`, `C_final_macro` |
| Table 6, H time per iteration 4.8 s | 4.75; 4.83 | `Hmacro_y.seconds_per_iteration`, `Hmacro_z.seconds_per_iteration` |
| Table 6, NICE compliance of the H designs: 88.02, 1,341.8 | 88.0193; 1341.784 | `Hfine_y.C_fine`, `Hfine_z.C_fine` (= `comparison.*.C_H_fine`) |
| H designs 1.1% above B1 and 0.34% below B2 (NICE) | 0.010684; −0.003394 | `comparison.y.H_vs_B`, `comparison.z.H_vs_B` |
| Volumes within 0.05% of \(V^*\) | H\(_y\) +0.040%, H\(_z\) −0.045%; B1, B2, XH\(_y\), XH\(_z\) within −0.002% | `Hfine_y.V_fine / B1.Vstar` − 1, `Hfine_z.V_fine / B2.Vstar` − 1; `B1/B2/XH_y/XH_z.V_rel_final` |
| Cross-start: stopped after 12 design iterations; a further 1.0% and 0.53% | 12; −0.009975; −0.005282 | `XH_y.iterations`, `XH_z.iterations` (fixed budget, `args.maxit` = 12 is not the stopping rule met; `XH_y.dx_final` = 0.0254, at the move limit); `comparison.y.X_vs_H`, `comparison.z.X_vs_H` |
| Still improving when stopped | 87.1546 → 87.1413; 1334.7545 → 1334.6969 | `XH_y.C_trace[-2:]`, `XH_z.C_trace[-2:]` |
| XH\(_z\) ended 0.87% below B2 (body, Section 8); XH\(_y\) 0.060% above B1 | −0.008658; 0.000602 | `comparison.z.X_vs_B`, `comparison.y.X_vs_B` |
| Table 6, XH\(_y\): 88.02 → 87.14, 545 s; XH\(_z\): 1,341.8 → 1,334.7, 643 s | 88.0193 / 87.1413, 545.18; 1341.784 / 1334.697, 642.93 | `XH_y/XH_z.C0`, `C_final`, `iter_s_mean` |
| Switches: active elements, ghost faces, retained coordinates or binary node features changed in at least 3 of 8 (A) and 9 of 24 (plates) cells between consecutive iterations | min 3 (A), 15 (B1), 10 (B2), 16 (XH\(_y\)), 9 (XH\(_z\)) | per-cell `fps` keys `active`, `faces`, `ports`, `cut_nodes`, `weak` of consecutive records of each `history.jsonl` (script; same cell matching as `facts_6_11.py`, which gives the per-key counts in `*.switches_per_iteration`); binary node features = cut-band and weak-support indicators (Appendix G.1); Table ST24b "Any of these" (A 3, B1 16, B2 10, XH\(_y\) 16, XH\(_z\) 9) additionally counts fringe and shift changes |
| Switches: coarse-factor shift in at most four cells | max 4 (B1, B2) | `*.switches_per_iteration`, key `shift` |
| Recomputed residual between \(5.3\times10^{-5}\) and \(6.7\times10^{-3}\) | 5.25e-5 (A) to 6.72e-3 (B2) | `A.true_residual_range[0]`, `B2.true_residual_range[1]`; all runs lie within |
| Signed residual work below \(3.1\times10^{-7}\) of the compliance | max 3.07e-7 | `B2.Ut_rho_rel_absmax` (A 2.7e-8, B1 2.5e-7, XH\(_y\) 4.4e-8, XH\(_z\) 1.7e-7) |
| Geometry fallback: \(1\pm10^{-4}\) or \(1\pm10^{-3}\); at most twice per run | A: k = 16 (applied eps 1e-4), k = 19 (1e-3); B1, B2: k = 5 (1e-3); XH\(_z\): k = 2 (1e-4); XH\(_y\): none | `*.body_perturbations_applied`; tried values in `*.body_perturbations`; failing cells (uncut cell 010 of A, uncut cell 200 of XH\(_z\)): Table ST24a |
| Table 6 caption: at most 24.3 GiB of GPU memory and a peak of 5.0 GiB of host memory (main process) | 24.33; 4.99 | `B1.gpu_peak_gb_max` (= B2); `B1.host_peak_gb_max` (A 15.7/3.5, XH\(_y\) 22.4/4.8, XH\(_z\) 22.2/4.3); host peak is `ru_maxrss` of the main process (`opt_design.py` line 629), GPU peak `torch.cuda.max_memory_allocated` (line 628) |
| Packed storage: host memory of the streamed state reduced by 23% on the \(2\times2\times2\) block (sentence moved to Supplementary Note S9.6; not in the main text) | 4.8285 → 3.7088 GB (23.2%) | `storage["packstore_compare.txt"]`, "base stream_GB" and "packed stream_GB" (parsed by the script) |
| Packed storage: 0.4 GiB more peak GPU memory (sentence moved to Supplementary Note S9.6; not in the main text) | 6.23 → 6.66 GB, +0.400 GiB | same record, "peak" of the base and packed lines (script) |
| Packed storage: compliance changed by at most \(1.1\times10^{-8}\) of its value; two unpacked evaluations \(8.9\times10^{-9}\) (sentence moved to Supplementary Note S9.6; not in the main text) | 1.14e-8; 8.87e-9 | same record, "max rel" and "run-to-run (unpacked twice): compliance max rel" (script) |
