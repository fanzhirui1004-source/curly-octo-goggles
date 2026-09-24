# Evidence dossier for the architecture review (2026-09-24 evening)

Repository: /home/user/curly-octo-goggles (branch claude/wizardly-euler-3m9cwx). Read-only for this review.
Key docs: docs/OPERATOR_LEARNING_DESIGN_20260923_CN.md (design, decisions, sections 7, 9, 10, 11),
docs/PROGRESS_20260924_CN.md (all experiment logs with numbers), docs/MORNING_REPORT_20260924_CN.md.
Code (as deployed): docs/data/newmachine_20260924/src_s0/ — models.py (MGNO, MGNO2), sparse_layers.py (training sparse
hyperedge layer), fastnet.py (frozen inference path), trainlib.py (Geo, energy/sensitivity losses, adversarial search),
train1.py (single/few geometry trainer), train2.py (step-2 pool trainer), prep_geo.py / prep_data.py (sampling classes),
teacher.py (exact CutFEM teacher), lattice3.py / evalnet.py (lattice acceptance gate), diag_lat.py, zs_eval.py.
Earlier architecture workflow outputs: docs/data/newmachine_20260924/arch_workflow/ (four design lenses, critiques, the
SGNO synthesis spec that was rejected as main line because it puts exact solves in the forward pass).

## 1. Scientific goal and hard constraints
- Goal: "encode the geometry once, get a reusable structured operator; then for ANY boundary displacement q quickly
  return the full boundary reaction" for TPMS Schwarz-P cells (CutFEM + ghost penalty, gamma = 1e-4, n = 32 background
  grid, Q2 elements), cut by planes at lattice boundaries. Used inside lattice analysis and topology/thickness design.
- Acceptance gate: lattice compliance AND thickness sensitivity (8 corner values per cell, every cell incl. the exact
  neighbour) relative error <= 3%, traction-consistent face loads (x and y configurations: cell glued to its FULL parent,
  far face clamped). Worst-direction mu (largest eigenvalue of S^-1 S_hat) is reported only. Cut-surface-loaded cases
  reported only. Uniform nodal loads reported as a stress test.
- Main line MUST be a fully learned network: no exact factorization / exact solve / exact residual iterations in the
  forward pass. Exact stiffness K appears only in the variational readout S_hat = E_hat^T K E_hat (energy) and in losses.
  Hybrids with exact components are "fallback A", allowed only as a separately labelled arm.
- User preferences: discuss before big actions; low-risk changes; results must not silently change; scope decisions
  below.

## 2. Current architecture (MGNO v0 and MGNO2)
- Operator: extension E_hat: port displacements q (box faces + cut band = all 27 nodes of cut elements) -> field on all
  active Q2 DOFs, exactly LINEAR in q. Wrapper: exact rigid-body split (least squares on ports), exact port values
  (overwrite). Reaction via the variational readout S_hat q = E_hat^T K E_hat q (so energies are one-sided upper bounds:
  q^T S_hat q >= q^T S q; error (E_hat - E)^T K (E_hat - E) is second order in the field error).
- Geometry path (nonlinear, once per geometry per forward): element features from the 125 polynomial moments of the
  cut material in each element (volume fraction, log vf, centroid/inertia moments), node flags (port/box/cut/weak) and
  log |diag stiffness block|, sin/cos of grid coords; 2 rounds of element<->node message passing (Cg = 64); heads emit
  per-(element, slot, head) weights alpha, beta (27 slots, H = 4 heads, per layer), restriction/prolongation weights,
  per-voxel channel gates for the coarse convolutions.
- q path (linear): port embedding W_in (3 -> F = 32) -> 4 fine element-hyperedge layers
  [Z = sum_s alpha X[slot s]; Z W_h (F x F per head); scatter with beta / degree] with port clamping after each ->
  U-Net over vertex grids 33^3 / 17^3 / 9^3 (restriction with geometry weights, 2 conv3d 3x3x3 per level with
  geometry gates, prolongation with skip) -> 4 fine layers -> W_out (F -> 3).
- MGNO2 adds ghost-penalty-face hyperedges (27-slot stencil across each GP face) after every fine layer and 4 extra
  "fringe" layer pairs restricted to elements/faces touching weak (fictitious-fringe) nodes. 0.60 M parameters
  (v0: 0.50 M).
- Training loss: mean log(e_hat) at unit exact energy (e_hat - 1 = mu - 1 in that direction) + sens_w * relative
  squared error of the 8 corner sensitivities (exact labels). Adversarial directions from block power iteration on
  (S_hat, S) with an fp32 Neumann factor, refreshed periodically.
- Sampling classes of q (each normalized to unit exact energy): force-driven (smooth random box-face forces solved
  exactly, 10% also load the cut band), macro (polynomial fields), GRF (multiscale random port fields), support (one box
  face on springs, others loaded), face (self-equilibrated load on one box face, all other ports free), adversarial.

## 3. Step-1 results (fixed geometry, one network per geometry)
Lattice gate, traction-consistent loads, sensitivity incl. neighbour (compliance / test-cell sens / neighbour sens):
| geometry | model | x | y |
|---|---|---|---|
| r1 heavy cut (retained 0.195) | MGNO v0 + sens + support, 30k steps | 2.1% / 2.5% / 1.5% PASS | 1.2% / 2.1% / 0.6% PASS |
| FULL parent | MGNO v0 + sens, 20k | 0.67% / 2.0% / 0.09% PASS | 0.68% / 1.5% / 0.05% PASS |
| r2 medium cut (retained 0.586) | MGNO2 + sens + support + face, warm start 10k (d2) | 1.50% / 2.58% / 1.06% PASS | 1.29% / 3.10% / 0.60% (0.1 pt over) |
Zero-parameter graph-harmonic baseline: compliance 21-58%, sensitivity 15-89%.
Bank-level validation errors (unit exact energy): force 1.8-2.4%, support 1.7-2.2%, face 1.0-1.8%, macro 0.6-0.7%,
GRF 2.1-2.6%; sensitivity errors 0.6-1.5% mean; worst direction mu = 1.31-1.48 (i.e. 31-48% energy error in the worst
direction).
Stress test (uniform nodal loads incl. fictitious-fringe nodes): r1 pass; FULL sens 8.2-8.5%; r2 5.3-6.7%.

## 4. Diagnostics so far
- Block diagnosis (r1 v0): in force-driven directions 36% of the error energy is in the ghost-penalty term, and 45% of
  the body error is in elements with weak (fictitious-fringe) nodes (nodes whose 3x3 diagonal block < 1% of median;
  18-22% of nodes, forming sheets on both sides of the shell). -> MGNO2 (GP-face hyperedges, fringe layers) helped.
- Coverage of q directions drove most gains: support class took r1 y from 5% to 1.2%; face class + gentler warm start
  took r2 y sens from 4.42% to 3.10%. Sensitivity supervision helped x but not y at first.
- Lattice error decomposition (diag_lat.py, MGNO2 r2): for each load, "extension-only" error (network field at exact
  port data) and "interface-only" error (exact field at the network's lattice solution) are each 5-10%, but they have
  opposite signs and largely cancel in the consistent total (2-3%): the network is over-stiff (S_hat >= S) so the
  lattice port displacement is too small, while E_hat q carries too much energy. The failing r2 y load (load on the
  neighbour, test cell carries 0.13% of the energy, dragged almost rigidly) is interface-dominated (4.8% vs 1.0%): the
  deformation part of q is tiny relative to the rigid part.
- Uniform nodal loads on fictitious-fringe port nodes inflate sensitivity error 2-4x (why the gate uses consistent loads).
- Precision floor: the fp32 network evaluation has ~1e-4 relative field error vs an fp64 evaluation of the same
  network; gradient run-to-run noise (atomic accumulation order) is 0.3-0.6% because the loss (log of the small excess
  energy ~0.02) amplifies fp32 noise ~50x.
- Warm-start lesson: re-warming the LR to 1e-3 wrecked a converged model (force error 2.4% -> 10%); 3e-4 improved it.

## 5. Zero-shot generalization of single-geometry networks (bank level, val set)
| trained on -> tested on | force energy error mean / max | sens error |
|---|---|---|
| r2 (MGNO2) -> another family's medium cut | 306% / 961% | 83% |
| r2 (MGNO2) -> another family's heavy cut | 107% / 204% | 47% |
| r2 (MGNO2) -> dev_0000 heavy cut | 407% / 2151% | 131% |
| FULL 0020 (v0) -> FULL 0031 (similar thickness) | 8.5% / 17% | 12% |
| FULL 0020 (v0) -> FULL dev_0002 | 2236% / 6441% | 219% |
| r1 (v0) -> other heavy cuts | 1.2e3-2.8e3 x | 180% |
Errors blow up (energies tens to thousands of times too large) when the geometry differs: the extension puts large
spurious energy somewhere (likely the fictitious fringe / weak regions whose distribution changes with geometry).

## 6. Step-2 data and scope (user decisions)
- 209 existing geometries (40 families: FULL parent + cut variants, some packets are cubic-group rotated variants such
  as *_rotswapxy / *_rotinvert); 206 prepared (3 degenerate slivers with NO interior DOFs excluded). Sizes 19k-430k
  DOFs, median ~200k; 18-22% weak nodes.
- Coverage (user): full-cell volume fraction 0.1-0.4 (thickness chosen by the thickness->volume-fraction map), cut
  angle 0-45 deg in one plane (other orientations by rotation augmentation), cut depth (retained macro box volume)
  full range 0-1. Deferred: general 3D-tilted cut planes; multiple cuts per cell.
- Split by family: train 148 geometries (28 families), val 20 (4 families), test 38 (8 families). Per geometry 5
  classes x 512/64/64 q samples with exact sensitivities.
- No 50/100/150 learning curve (user: wasteful). One run on all 148; data sufficiency judged from the gap between
  training-geometry error (unseen q) and held-out-geometry error, plus stratified errors.

## 7. Efficiency
- Training step (r2, MGNO2, batch 16): originally 0.41 s (network fwd+bwd 88%; fwd alone 0.08 s). Now sparse
  hyperedge training layers (custom autograd, cuSPARSE SDDMM for alpha/beta grads, chunked W-gradient GEMM, deduped
  geometry path) + reassociated sensitivity loss: ~2.7x per step, gradients equal to run-to-run noise.
- On step-2 geometries (bigger): 0.229 s/step at batch 16; peak GPU memory 28.4 GB of 31.4 GB with 3 resident
  geometries; geometry swap + adversarial on entry ~13% overhead.
- Inference (fastnet, frozen geometry, sparse layers, explicit adjoint): reaction S_hat q in 4-10 ms for one direction,
  0.09-0.37 s for 64; memory 0.03-3.8 GB. Geometry encoding ~0.1 s + element moments 0.2-1 s.
- Discarded as lossy: fp16/bf16/TF32. Tried, no gain: torch.compile of geometry path, cudnn.benchmark, 2 geometries per
  step.

## 8. Step-2 training in progress (first ~1200 steps of 60k, warm start from the r2 MGNO2)
- Loss (log e_hat + sens) falls fast: step 50: 0.82 (e_mean 1.41 = 141% excess energy), step 550: 0.40, step 1150:
  0.13 (e_mean 0.14), sens loss 0.15 -> 0.003.
- Worst-direction ratio on geometries at their first entry (network not yet trained on them), in order of entry:
  106, 280, 46, 19, 6.6, 4.7 — falling quickly as training proceeds, i.e. the network is learning cross-geometry
  structure, but early entries show catastrophic worst directions (100x energy).
- First validation (20 held-out geometries + 12 training geometries with unseen q) at step 10k (~40 min away).

## 9. Questions for the review
How should the architecture be designed or strengthened now, for (a) cross-geometry generalization (step 2), (b) the
remaining step-1 accuracy gap (worst direction mu 1.3-1.5; r2 y 3.1%; stress test), (c) the downstream use (lattice
solve, design sensitivities, speed), within the fully-learned constraint. Concrete, testable, prioritized proposals;
state expected gain, risk, cost, and how to test each against the existing harness.

## 10. ADDENDUM (arrived after the review started): step-2 run, first evaluation at step 10k
Run: MGNO2 warm-started from r2 d2, 148 training geometries, pool 3 (DOF cap 700k), swap every 300 steps, batch 16,
LR 1e-3 one-cycle (5% warm-up, cosine to 60k), adversarial on entry. Eval = mean energy excess (e_hat - 1) on the val q
bank (64 per class), per geometry.
- Class means, 20 held-out val geometries: force 0.425, support 0.553, face 0.162, macro 0.058, grf 0.061.
- Class means, 12 TRAINING geometries (unseen q): force 0.460, support 0.540, face 0.108, macro 0.050, grf 0.053.
  -> No train/val gap. BUT the 12 training probes are the first 12 geometries of the order (resident during steps
  0-3300 only), so "train_geo" here measures geometries not visited for 7k-10k steps.
- Step-1 single-geometry levels for comparison: force 0.018-0.024, support 0.017-0.022, face 0.010-0.018, macro 0.006,
  grf 0.021-0.026. Step-2 is 10-25x worse at step 10k.
- Per-geometry spread is huge: FULL parents and large-DOF geometries force 0.04-0.12; thin / small / heavily cut
  (d0_v*, some cover01) and a rotated variant (0021_cover01_r1_rotswapxy: 1.58, 0010_d0_v1: 1.56, 0029_d0_v2: 1.21,
  0003_d1_v1: 1.11) force > 1. Error correlates with low DOF count (thin cells).
- Resident-geometry training loss (log e_hat, batch of the geometry currently in the pool) plateaus at 0.04-0.18
  (e_mean 0.04-0.27) from step 3k to 10k at LR ~1e-3; no downward trend over 7k steps.
- Visit statistics: 60k steps / 300-step swaps = 200 entries for 148 geometries -> each geometry is resident ~1.35
  times for ~900 steps per run; strongly non-stationary; evicted geometries degrade to the unseen level (forgetting).
- Entry worst ratios (ritz of S_hat vs S on entry, before training on it): mostly 1.3-10 after step 3k, but
  outliers: 15388 (0001_cover01_r2 at step 1450: next-50-step loss 258, grad norm 5.8e4, clipped to 1, recovered
  within 50 steps), 420 (0021_d1_v1 at step 2050), 280, 106.
- Engineering found while restarting (fixed): host memory growth (evicted slots kept, pinned cache), GPU cap needs a
  DOF-capped pool (two 400k-DOF geometries do not fit with activations), 5 geometries with NaN support samples
  (fp32 spring factor on nearly empty faces) removed at load.
