# Late results (appended as they arrive; re-read this file and smooth_v2L1.json before concluding)

- 2010 heavy smoothing, zero-interior start energy excess by k (force_c): k=0 9907, 1: 3019, 2: 1129, 4: 1122,
  8: 33.6, 16: 0.29, 32: 0.0042. Network start: 0.35 -> k=8 0.0021 (network + 8 sweeps ~16000x better than 8 sweeps
  from zero; ~2x better than 32 sweeps from zero). lmax(D^-1 K_II) = 4.13, interior DOFs 1125 (a tiny cell).

- 2003 MEDIUM cut (9574 level-0 grid nodes; a much larger cell than 2010), force_c, network start, (k, energy excess,
  sens rel error): (0, 13.5%, 14.1%) (1, 9.9%, 8.2%) (2, 7.9%, 5.7%) (4, 6.1%, 4.2%) (8, 4.7%, 2.8%) (16, 3.8%, 2.2%)
  (32, 2.95%, 1.6%). Zero-interior start after 32 sweeps: energy excess 189 (x100%), sens 103 (x100%) — the smoother
  alone is useless on this cell. INTERPRETATION TO TEST: on a realistic-size cell only ~1/3 of the error energy is
  high-frequency (removed by 1-2 sweeps); the remaining ~2/3 is smooth / low-energy and decays slowly with k; but the
  sensitivity error falls faster than the energy error (14.1% -> 2.8% at k=8), consistent with dK/dtau being
  surface-local (sensitivities weight high-frequency, surface-near error).

- RUNNING (CPU): error_spectrum.py — lowest 200 eigenpairs of K_II v = lam D v (shift-invert Lanczos, exact factor);
  share of the network error energy and of the exact field energy in the lowest m modes (m = 1..200), cells 2003,
  2010, 2006, 2000. Results will be appended here and in spectrum_v2L1.json when available.

- 2003 medium, force class, network start (k, energy, sens): (0, 7.2%, 4.8%) (1, 5.1%, 2.4%) (2, 4.0%, 1.5%)
  (4, 3.1%, 1.2%) (8, 2.3%, 0.79%) (16, 1.9%, 0.66%) (32, 1.45%, 0.55%). Zero start k=32: energy 104 (x100%).
  Same pattern: ~30% of error energy removed by 1 sweep, sensitivity error halves per sweep initially.

- 2006 medium, network start (k, energy, sens):
  force_c (0, 3.6%, 5.2%) (1, 2.6%, 2.9%) (2, 2.0%, 1.9%) (4, 1.45%, 1.3%) (8, 1.04%, 0.84%) (16, 0.81%, 0.67%) (32, 0.61%, 0.58%)
  force   (0, 2.2%, 1.1%) (1, 1.4%, 0.39%) (2, 1.0%, 0.34%) (4, 0.72%, 0.21%) (8, 0.46%, 0.15%) (16, 0.35%, 0.12%) (32, 0.25%, 0.09%)
  zero start k=32: energy 54 / 23 (x100%). Energy error falls ~3.5x by k=8; sensitivity error ~6-7x by k=8.
- Production: 700+ cells done.

- 2000 FULL, network start (k, energy, sens):
  force   (0, 0.74%, 1.02%) (1, 0.44%, 0.43%) (2, 0.31%, 0.61%) (4, 0.20%, 0.31%) (8, 0.11%, 0.23%) (16, 0.07%, 0.19%) (32, 0.05%, 0.15%)
  force_c (0, 1.27%, 0.66%) (1, 0.93%, 0.35%) (2, 0.74%, 0.41%) (4, 0.56%, 0.44%) (8, 0.42%, 0.53%) (16, 0.35%, 0.49%) (32, 0.28%, 0.46%)
  (sensitivity is not monotone in k; energy is). zero start k=32: energy 4.1 / 26.8 (x100%).
- 2005 heavy (9% volume), network start (k, energy, sens):
  force   (0, 1.84%, 0.92%) (1, 0.95%, 0.42%) (2, 0.59%, 0.38%) (4, 0.43%, 0.24%) (8, 0.20%, 0.14%) (32, 0.05%, 0.05%)
  force_c (0, 1.81%, 1.91%) (1, 1.16%, 0.84%) (2, 0.76%, 0.56%) (4, 0.49%, 0.41%) (8, 0.25%, 0.26%) (32, 0.06%, 0.15%)
  zero start k=32: energy 3.9 / 1.9 (x100%).
- SUMMARY of smoothing headroom at k=8 (energy excess reduction factor): 2010 heavy 166x/85x; 2005 heavy 7x/9x;
  2006 medium 3.5x/4.8x; 2003 medium 2.9x/3.1x; 2000 FULL 3x/6.7x. The hardest large cell (2003) keeps ~1/3 of its
  error after 8 sweeps -> a smooth / soft component the smoother cannot reach dominates there.

- SPECTRUM 2003 medium (error_spectrum.py; interior pencil K_II v = lam D v, D = diag K_II; lowest 200 modes span
  lam in [5.7e-4, 1.13e-2]; lmax(D^-1 K_II) ~ 4, Chebyshev tail interval [lmax/30, lmax] ~ [0.13, 4]).
  Share of energy in the lowest m modes (mean over 32 val directions):
    force_c error: m=1 0.1%, 5 1.5%, 10 3.2%, 20 5%, 50 15.3%, 100 22.8%, 200 24.5%
    force_c exact interior field: m=10 0.6%, 50 2.5%, 100 4.2%, 200 4.9%
    force   error: m=10 4.5%, 50 15.5%, 100 21.3%, 200 22.4%;  field: m=200 6.6%
  Derived (force_c): total error energy 13.5% of field energy; error energy inside the lowest-200 subspace
  = 0.245*13.5% = 3.3% vs field energy there 4.9% -> within the softest 200 interior modes the network's relative
  energy error is ~67% (the network barely represents the soft interior modes), but those modes carry only ~5% of the
  bank-direction field energy. Spectral budget of the error (force_c): ~25% in the softest 200 modes (lam < 0.011),
  ~1/3 above lam ~0.13 (removed by 8 sweeps, see smoothing), the rest (~40%) in the mid band 0.011 < lam < 0.13.
  (Caveat: the interior field u_I is a harmonic lifting; its interior-mode energy is not the full field energy.)

- SPECTRUM 2010 heavy (1125 interior DOFs): lowest 200 interior modes span lam in [0.021, 0.42] (of lmax ~4.1).
  Error share in the lowest m: force_c m=50 1%, 100 9.1%, 200 15.2%; force m=50 5.7%, 100 15.4%, 200 21.8%.
  Field share: m=200 ~4.8%. -> ~80% of this cell's error sits above lam 0.42, i.e. in the band the 8-sweep tail removes
  (consistent with 35% -> 0.2%). The tiny heavy cut has no very soft interior modes once ports are clamped
  (lam_min 0.021 vs 5.7e-4 for 2003).
- Implementation note: trainlib.smooth_tail / fastnet tail implemented behind model_args smooth_k (default 0 = unchanged);
  unit tests: matches reference Chebyshev bit-for-bit, adjoint identity to 1e-15, linear, gradients correct, ports held.
  Chebyshev guarantees the k-sweep error energy <= initial error energy for every k (given lmax >= true lmax);
  intermediate iterates are not monotone in k.

- SPECTRUM 2006 medium: lowest 200 interior modes lam in [9.4e-4, 1.17e-2]. Error share in lowest m:
  force_c m=5 4.5%, 20 11.6%, 50 16.5%, 200 18.4%; field share m=200 4.6%.  force: m=50 10.1%, 200 11.4%; field 6.6%.
  Derived (force_c): error energy in the lowest-200 subspace = 0.184*3.6% = 0.66% vs field energy there 4.6% ->
  relative energy error inside the soft subspace ~14% (2003: ~67%). Again ~80% of the error is above the 200th mode.

- GATE DECOMPOSITION 2000 FULL, config x (gate_decomp.py, v2L1; loads test_x,y,z, nbr_x,y,z):
  sens_full       2.38 2.24 1.39 | 3.54 1.61 5.83 %
  sens_field_only 1.05 0.94 0.65 | 1.44 0.44 2.67 %   (learned field at the EXACT lattice q)
  sens_sol_only   2.95 2.64 1.76 | 2.37 1.39 3.30 %   (exact field at the LEARNED q_hat)
  q_err_S         2.22 2.04 1.52 | 1.99 1.57 2.41 %   (||q_hat - q||_S / ||q||_S)
  eps             1.39 1.23 0.78 | 1.82 1.14 2.31 %   (operator energy excess at exact q)
  compliance      0.47 0.14 0.09 | 0.01 0.01 0.00 %
  => in the lattice, the error of the lattice SOLUTION (q_hat - q, first order in the operator error eps) contributes
  MORE to the sensitivity error than the field reconstruction; q_err_S ~ 1-1.6 x eps; sensitivity error ~ 1.5-2.5 x eps.
  Rule of thumb: gate sensitivity <= 3% needs eps <~ 1.2-1.5% on the lattice-relevant directions. The field-only part
  is small (<= 2.7%). The worst load is still the neighbour-face z load where the test cell carries 0.1% of the energy.

- GATE DECOMPOSITION 2000 FULL y: sens_full 2.35 2.43 1.62 | 1.24 2.36 5.39; field_only 0.45 0.64 0.48 | 0.31 0.84 2.6;
  sol_only 2.44 2.77 1.79 | 1.25 1.91 2.97; q_err_S 1.96 2.12 1.53 | 1.52 1.88 2.32; eps 1.15 1.3 0.81 | 1.06 1.59 2.06 (%).
  Same picture as x: solution error > field error; worst = neighbour-face z load (0.1% energy share).
- GATE DECOMPOSITION 2003 medium (%; loads test x,y,z | nbr x,y,z | cut traction x,y,z; cut loads not gated):
  x: eps 10.6 14.8 6.8 | 2.2 8.8 1.7 | 5.1 10.9 10.8; q_err_S 11.9 14.2 9.8 | 4.0 10.0 3.1 | 7.3 12.0 12.5
     field_only 10.1 15.6 9.2 | 2.5 8.7 0.4 | 6.0 12.6 13.1; sol_only 13.5 19.3 12.0 | 2.3 5.1 2.1 | 8.4 15.5 17.4
     full 9.2 12.2 6.4 | 0.9 2.6 2.5 | 4.1 9.9 9.4; compliance 4.4 3.7 1.2 | ~0 | 2.6 2.5 2.9
  y: eps 14.7 10.2 6.5 | 8.4 2.1 1.6 | 16.3 14.6 14.3; full 11.4 8.2 6.4 | 2.6 1.2 2.5 | 13.8 13.1 12.9
     field_only 15.5 9.8 8.4 | 8.0 1.9 0.5 | 15.8 12.9 14.5; sol_only 19.0 12.8 11.5 | 4.5 2.2 2.0 | 21.5 20.1 21.5
  => on the failing medium cell both parts are large (10-20%) and partly CANCEL (full < either part); both are driven by
  the same operator error eps ~7-16% on the loaded directions. Nothing but lowering eps on the lattice directions fixes it.

- A0_ctrl (v2L1 + 15k more steps, same architecture) vs v2L1, new-val energy excess % (types: v0 heavy, v1 medium,
  v2 light cut): heavy force 11.3 -> 10.0, force_c 12.6 -> 10.2, glued 16.8 -> 18.2; medium ~unchanged (4.7 -> 4.5);
  light force_c 9.9 -> 8.8; FULL 0.91 -> 0.88. Continuous gates (compliance / sens %): 2000 x 0.40/4.89 (v2L1 0.47/5.83),
  y 0.39/4.72; 2003 x 4.12/11.36, y 3.46/10.93; 2005 pass; 2006 x 0.59/2.87 pass; 2001 pass. -> more steps on the same
  data give only marginal gains. 2010 gate returned NaN (tiny-cell numerical issue in the lattice gate, to investigate).
- DECISION TAKEN: A1_sens3 (sensitivity-weighted loss) stopped; the GPU now trains A2_tail8 = v2L1 + smooth_k 8 tail,
  15k steps, same evaluation as A0 (experiment arm only; adopting a hybrid remains the user's decision).

- A2_tail8 (v2L1 warm start + smooth_k 8 tail, trained end-to-end) at step 7500 vs A0_ctrl (no tail) at step 7500,
  training-time validation mean energy excess over the same 40 val geometries (%):
    force 0.70 vs 4.14 | force_c 1.62 vs 6.67 | glued 1.21 vs 5.06 | support 0.81 vs 4.06 | support_k 1.15 vs 3.70
    face 0.27 vs 2.92 | face_c 0.61 vs 3.34 | grf 0.32 vs 2.03 | macro 0.25 vs 0.85 ; selection score 0.058 vs 0.094
  -> 4-10x lower validation energy error with the trained tail (vs ~3x from the untrained tail on v2L1 fields).
  Step time ~0.29 s (unchanged). Per-type new-val and continuous gates follow after step 15000.

- E3 COARSE GALERKIN (e3_coarse.py, v2L1 fields, untrained wrapper; coarse correction from the residual only,
  deployable; 2003 medium, 165,927 interior DOFs). Energy excess %, force_c (force):
    net 13.5 (7.2) | net+tail8 4.7 (2.3)
    Q1_9   (1,275 coarse dofs): net+coarse 7.4 (4.2); net+coarse+tail8 1.00 (0.53)
    PU_9   (5,100):             4.2 (2.6);           0.17 (0.13)
    Q1_17  (5,601):             4.7 (2.8);           0.23 (0.16)
    Q2_17  (7,401):             3.6 (2.2);           0.13 (0.10)
    PU_17  (22,404):            2.0 (1.3);           0.048 (0.056)
    Q1_33  (27,207):            2.0 (1.3);           0.052 (0.045)
    zero-interior start + tail + coarse + tail: 1.6x10^2 .. 4.5x10^3 % -> the network's initial guess is essential.
  => one two-grid correction (coarse Galerkin from the residual + 8 sweeps) on the WORST medium cell takes
  13.5% -> 0.13-0.23% with 5-7k coarse dofs (60-100x), untrained. Coarse matrices V^T K_II V are 1-27k sparse.
- E3 2006 medium (%): force_c net 3.64 | tail 1.04 | Q1_9+tail 0.20 | PU_9+tail 0.04 | Q1_17+tail 0.05 | Q2_17+tail 0.03
                      force   net 2.17 | tail 0.46 | Q1_9+tail 0.17 | PU_9+tail 0.10 | Q1_17+tail 0.10 | Q2_17+tail 0.09
  2003 force: Q1_9+tail 0.53 | PU_9+tail 0.13 | Q1_17+tail 0.16 | Q2_17+tail 0.10.
- A2_tail8 final training eval (step 15000): force 0.66, force_c 1.50, glued 1.15, support 0.78 (%), score 0.0553.
- E3 2000 FULL (%): force_c net 1.27 | tail 0.42 | Q1_9+tail 0.14 | PU_9+tail 0.02 | Q1_17+tail 0.04 | Q2_17+tail 0.02
                    force   net 0.74 | tail 0.11 | Q1_9+tail 0.07 | PU_9+tail 0.05 | Q1_17+tail 0.05 | Q2_17+tail 0.05
- Production: 900+ cells.

- NEW-VAL (80 unseen independent-field cells), energy excess % as mean / median / max, v2L1 -> A0_ctrl -> A2_tail8:
  FULL   force 0.91/0.73/2.3 -> 0.88 -> 0.11/0.08/0.3;  force_c 0.64 -> 0.60 -> 0.17/0.10/0.4;  glued 0.77 -> 0.71 -> 0.25
  light  force 3.24/2.47/9.9 -> 3.02 -> 0.67/0.45/2.2;  force_c 9.9/11.2/15.2 -> 8.8 -> 3.22/3.82/4.7;  glued 8.3 -> 7.5 -> 3.0/3.8/4.0
  medium force 4.72/3.36/20.9 -> 4.51 -> 0.91/0.60/4.4; force_c 5.23 -> 5.24 -> 1.73/1.31/4.3; glued 3.27 -> 3.30 -> 1.14/0.88/2.4
  heavy  force 11.3/7.1/70.1 -> 10.0 -> 1.00/0.73/3.3;  force_c 12.6 -> 10.2 -> 1.20/0.51/3.2; glued 16.8 -> 18.2 -> 1.52/0.90/2.8
  => trained tail: 5-11x lower means, worst cells 70% -> 3.3% (heavy), 21% -> 4.4% (medium). Light-cut force_c / glued
  stay ~3-4% (the least improved group: large cells with a cut face, consistent with the mid/soft band the tail
  cannot reach; the coarse Galerkin (E3) is the candidate for that remainder).

- CORRECTION (verified): A2_tail8 was NOT trained with the tail. The config has oh = true, so every training geometry
  runs through an O_h view whose field (oh._field) bypassed Geo.field and therefore the tail. A2's step losses equal A0's
  to ~5e-6 relative (same seed, float noise). A2's reported improvements (training-eval val_mean on the identity view,
  new-val with --views 0, continuous gates through FastNet) are therefore "A0 network + UNTRAINED 8-sweep tail at
  evaluation". Fixed: oh._field now applies the tail; A2b_tail8 (really trained with the tail) is queued.
- A2 (= A0 + untrained tail8) continuous gates, compliance / sens %: 2000 x 0.14/3.89 fail (test-face loads <=1.31),
  y 0.14/3.75 fail; 2001 x 0.03/1.11, y 0.04/1.11 pass; 2003 x 1.68/5.83 fail (A0 4.12/11.36), y 1.39/5.56 fail;
  2005 x 0.18/1.22, y 0.27/0.88 pass; 2006 x 0.19/1.89 pass. Decomposition 2000 x (A2): eps 0.41 0.40 0.28 | 0.74 0.41 1.11,
  sens field_only 0.56 0.63 0.63 | 1.45 0.82 2.46, sol_only 0.84 0.80 0.60 | 1.02 0.51 1.56 -> the remaining 2000
  failure (nbr_z, dragged free end) is now FIELD-dominated.

- E3 WITH FEWER SWEEPS (untrained, v2L1; tail(k) + coarse + tail(k), energy excess %, force_c / force):
                 k=2               k=4              k=8
  2003 Q1_17:    1.02 / 0.59       0.42 / 0.25      0.19 / 0.11
  2003 PU_9:     0.86 / 0.52       0.33 / 0.21      0.11 / 0.08
  2003 Q1_9:     2.33 / 1.22       1.33 / 0.69      0.81 / 0.42
  2000 Q1_17:    0.13 / 0.14       0.05 / 0.06      0.03 / 0.03
  Query cost ~ (2k + 1 residual) K products + one coarse solve per field (and the same again for the adjoint).
- A3_2grid (trained through tail8 + Q1_17 + tail8): training-batch energy excess ~1e-3 at step 50 and ~1e-4 (bank
  normalisation floor, sometimes negative) from step ~1500 (A0 at the same steps: 16%, 2.3%, 0.6%). 0.6 s/step.
