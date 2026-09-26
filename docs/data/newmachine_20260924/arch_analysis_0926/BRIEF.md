# Architecture analysis brief (2026-09-26): where does the learned-operator error come from, and what must change?

You are one of several independent analysts. Read this brief first, then the sources it points to. Be quantitative,
cite numbers and file names, and separate what the data SHOWS from what you INFER.

## 1. The problem

- Unit cell: TPMS Schwarz-P, thickness field tau(x) = trilinear in 8 corner values (range ~0.1755 .. upper bound),
  CutFEM on an n = 32 background hex grid (Q1 trilinear elements, 81 DOF each) with ghost penalty (GP). Cut cells: the
  cell is additionally cut by a vertical plane (retained volume fraction from ~1% (heavy) to ~70%+ (light)); the cut
  surface is traction-free or loaded. Heavy cuts leave stubs / thin struts; some are near-mechanisms.
- Ports P = box-face background nodes with material (+ cut-band nodes); interior I = the rest. Exact Schur complement
  S = K_PP - K_PI K_II^-1 K_IP; exact extension E q = [q; -K_II^-1 K_IP q].
- Standing goal: "encode the geometry once, get a reusable structured operator; given any boundary displacement q,
  quickly return the full boundary reaction S q". Used inside lattice (multi-cell) compliance minimisation with
  gradient thickness fields: needs lattice compliance and d(compliance)/d(8 corner taus) per cell.
- Acceptance gate (user-defined): lattice compliance AND 8-corner thickness sensitivity relative error <= 3% per cell,
  traction-consistent loads, with the neighbour included. The "continuous-neighbour gate": the test cell glued to a
  neighbour whose thickness field is continuous across the shared face (neighbour exact, test cell learned),
  configurations x / y (neighbour at -x / -y), loads: traction-consistent unit loads on the test cell's far face (3
  directions) and on the neighbour's far face (3 directions), plus (cut cells, not gated) cut-surface tractions.
  Sensitivity metric per load: ||s_hat - s||_2 / ||s||_2 over the 8 corners of the test cell. Gate = max over gated loads.

## 2. The current model (main line, "fully learned", MGNO2, v2)

- u_hat = N_theta(geometry; q): STRICTLY LINEAR in q (q only passes through linear maps; geometry modulates the
  coefficients through nonlinear encoders). Output = displacement on all background nodes; ports are held at q.
- Operator readout: S_hat = E_hat^T K E_hat with the exact CutFEM K. Identity: S_hat - S = (E_hat - E)^T K (E_hat - E)
  >= 0 (Galerkin orthogonality). So the per-direction "energy excess" e(q) = q^T S_hat q / q^T S q - 1 =
  ||(E_hat - E) q||_K^2 / ||E q||_K^2 (squared relative energy-norm extension error). The model is always too stiff.
- Architecture (code: docs/data/newmachine_20260924/src_v2_wip/models.py, class MGNO2; fastnet.py is the inference
  path): fine hyperedge layers on active elements (element = hyperedge over its 8 nodes), GP-face layers, "fringe"
  layers on a weak-node subset, U-Net over fixed voxel grids 33^3 / 17^3 / 9^3 with conv3d, transfers = trilinear
  restriction/prolongation dicts; geometry features from 125 element moments per element; B1 bounded gains; O_h
  (48-element cubic group) augmentation; ~12M-ish parameters (check). fp32 convolutions (TF32 off).
- Loss: per-direction normalised energy excess over probe banks ("classes"): force, macro, grf, support, face,
  force_c, face_c, support_k, glued (+ adversarial "adv" power-iteration directions); 512/64/64 train/val/test
  directions per class per geometry; plus a sensitivity loss term (sens_w). Training swaps geometries through a GPU
  pool (exact factorizations for the teacher on the fly).
- Data: ~3000 independent-thickness-field cells being produced (2 FULL + 6 cuts per 8-block); v2L1 trained on 305
  training geometries (157 new independent-field cells + old), 40k steps; v2L2 (591 geometries) paused at 10k.
- Design decisions on record (see docs/OPERATOR_LEARNING_DESIGN_20260923_CN.md and docs/ARCH_REVIEW_20260924_CN.md):
  user chose (09-23) a feed-forward fully learned neural operator as the main line; exact K is used only in the energy
  readout and the loss. "Fallback A" = learned iteration with exact residuals r = f - K u; "Fallback B" = fixed
  multigrid skeleton (paired smoother, coarse space, Chebyshev) with a few learned coefficients. The 09-24 SGNO spec
  (exact skeleton + learned coefficients) was NOT adopted as main line ("anti-drift": no exact factorization / solve in
  the forward). Route-1 history: pure Chebyshev needed ~32-64 layers to pass the lattice gate from scratch
  (docs/ROUTES_PROGRESS_20260923_CN.md). The deployment benchmark and a lattice BNN preconditioner (K_PP sparse
  Cholesky + Q1 rigid coarse space; iteration counts flat 2 -> 27 cells) exist.
- Standing constraints: fp16/bf16/TF32 rejected; new behaviour behind flags whose default keeps old results; frozen
  teacher/geometry contracts unchanged; the sealed test set is never read.

## 3. Evidence collected (data files in this directory: pack/*.json)

### 3.1 Per-type energy error on 80 unseen independent-field validation cells (force / support, %, identity view)
| type | c_oh (old data) | v2s | v2L1 (305 geos) |
|---|---|---|---|
| FULL | 1.1 / 1.8 | 1.1 / 2.0 | 0.9 / 1.4 |
| light cut | 4.6 / 4.1 | 4.1 / 3.9 | 3.2 / 3.0 |
| medium cut | 6.2 / 5.3 | 6.3 / 5.3 | 4.7 / 3.9 |
| heavy cut | 17.7 / 21.1 | 16.5 / 19.8 | 11.3 / 12.9 |
(newval_*.json: per_geo -> view -> class -> mean energy excess.) Doubling data cut heavy-cut error by ~36%.
Training-cell fit (trainfit_v2L1.json): heavy training cells mostly 1-10%; one near-mechanism outlier (excluded now).
The worst validation cells are those with the lowest force_c Rayleigh quotients.

### 3.2 Per-element sensitivity-error decomposition (diag_sens_v2L1_cpu.json; v2L1; 32 val directions)
| cell | type | force_c sens / energy | first-order share | force sens / energy | continuous gate |
|---|---|---|---|---|---|
| 2001 | FULL | 0.6% / 0.4% | 0.36 | 1.0% / 0.5% | x pass |
| 2000 | FULL | 0.7% / 1.3% | 0.43 | 1.0% / 0.7% | sens 5.8% fail |
| 2005 | heavy | 1.9% / 1.8% | 0.28 | 0.9% / 1.8% | x,y pass |
| 2006 | medium | 5.2% / 3.6% | 0.09 | 1.1% / 2.2% | x pass |
| 2003 | medium | 14.1% / 13.5% | 0.07 | 4.8% / 7.2% | 4.4 / 12% fail |
| 2010 | heavy | 75% / 35% | 0.01 | 14% / 20% | - |
Sensitivity error D = -(2 e^T A u + e^T A e) (A = dK/dtau_c, e = u_hat - u). When errors are large the second-order
part dominates. Error sits in partially-filled elements (full elements have dK/dtau = 0). Force-class sensitivity
error concentrates in port-adjacent elements (FULL: 13% of elements carry ~83% of |D|).

### 3.3 Continuous-neighbour gate per load (gate_cont_v2L1_*.json; test = test-cell far face loads, nbr = neighbour
far face loads; shr = energy share of the test cell)
- 2000 FULL x: test loads sens 2.4/2.2/1.4%, eps 1.4/1.2/0.8%; nbr loads sens 3.5/1.6/**5.8%**, eps 1.8/1.1/2.3%,
  test-cell energy share on nbr loads = 0.4% / 0.9% / 0.1%. y: similar (max 5.4% on nbr_z, share 0.1%).
  => the FULL-cell "failure" is on loads where the test cell carries ~0.1% of the lattice energy.
- 2003 medium x: test loads eps 10.6/14.8/6.8% -> sens 9.2/12.2/6.4%, compliance 4.4/3.7/1.2% (genuine failure).
- 2005 heavy, 2006 medium, 2001 FULL: pass; eps 0.3-2.4%.
- gate sens error ~ 1-2x eps on the same load.

### 3.4 Topology-aware coarse levels (t_split_v2L1.json): splitting coarse vertices into material-connected
components adds 0 nodes at 33^3 and <= 26 (<= 1.3%) at 17^3 / 9^3 on 4 val cells -> "coarse levels short-circuit
across voids" is NOT an error source. (Flag coarse_split implemented, default off.)

### 3.5 NEW: smoothing headroom (smooth_v2L1.json, smooth_headroom.py). u_k = k steps of a fixed LINEAR smoother
(Jacobi-preconditioned Chebyshev on [lmax/30, lmax] of D^-1 K_II; exact K matvecs; ports held) applied to the network
field; energy-norm contractive; linear in q, so S_hat stays SPD and >= S.
- 2010 heavy, force_c: energy excess k=0 35% -> k=1 12.6% -> 2: 5.9% -> 4: 3.6% -> 8: 0.21% -> 16: 0.02%;
  sensitivity 75% -> 29% -> 19% -> 8.8% -> 0.77% -> 0.08%. From the zero-interior start the same smoother reaches
  0.42% energy / 2.0% sens only after 32 steps (small cell: only ~700 coarse nodes).
- 2010 heavy, force: 19.5% -> 7.5 -> 3.8 -> 2.2 -> 0.23% (k=8); sens 13.7% -> 0.75% (k=8).
- Other cells (2003, 2006, 2000, 2005): being computed; will be appended below when available.

### 3.6 Older history (read the docs): step-1 single-geometry fits of the fully learned MGNO on heavy / medium cuts:
force-direction energy 0.023 / 0.036 after 20k steps, but WORST directions 1.54 / 1.57 (i.e. 54-57% excess) — see
docs/OPERATOR_LEARNING_DESIGN_20260923_CN.md section 10 and docs/PROGRESS_20260924_CN.md. Earlier findings: "the floor
is the softest decile", "over-stiffness concentrated on the softest directions" (E3), soft/stiff Rayleigh quotients
differ by 500-2000x, "error is broad", "two error regimes", "cut-cell accuracy is all cancellation" (docs/*.md).
Architecture reviews: docs/ARCH_REVIEW_20260924_CN.md, docs/data/newmachine_20260924/arch_workflow/*.md (four design
proposals + critiques + V0_SPEC SGNO), docs/data/newmachine_20260924/arch_review2/DOSSIER.md, docs/AUDIT_V2_20260925_CN.md,
docs/ARCHITECTURE_HISTORY_20260923_CN.md, docs/V2_ARMS_SPEED_DEPLOY_20260925_CN.md (latest, sections 10-13).

## 4. Pending / available experiments
- gate_decomp.py (queued on the GPU host after A0): continuous-gate sensitivity error split into
  field-only (learned field at exact lattice q) and solution-only (exact field at learned q_hat), + ||q_hat - q||_S.
- A0_ctrl (v2L1 + 15k steps on 305+ geos, control) training now; A1_sens3 (sens_w 3) queued after.
- GPU host: 1x RTX 5090 32 GB, shared with training. CPU: 16 cores, 750 GB RAM, PARDISO available for exact solves.
- Tools: diag_sens.py, smooth_headroom.py, t_split.py, lat_full.py, eval_views.py, train3.py (configs = JSON).
