# Stage 2b verification: T6 ML ablations and training (I-19, I-26, I-33, I-34, I-43)

Verifier: an adversarial check of the T6 issues in CONSOLIDATED.md. For each issue I tried to refute the concern first.
Method: read-only inspection of the manuscript, `evidence/`, the training code (`docs/data/newmachine_20260924/src_v2_wip/`: `train3.py`, `eval_views.py`, `meta_p1.py`, `dfp_cfg.py`, `chain_dfp.sh`), the server chain scripts (`.../prod/chain_*.sh`) and the contemporaneous project notes (`arch_analysis_0926/BRIEF.md`, `LATE_RESULTS.md`, `WORKFLOW_RESULT.json`, `docs/V2_ARMS_SPEED_DEPLOY_20260925_CN.md`, `docs/AUDIT_V2_20260925_CN.md`, `docs/HANDOVER_PAPER_20260926_CN.md`).
Scratch script: `/tmp/claude-0/.../scratchpad/i19.py`. It reads `newval2_*.json` and computes paired per-geometry ratios with bootstrap CIs over geometries (20,000 resamples). Nothing in the manuscript, evidence or code was changed. I did not use the server.

Abbreviations: MS = MANUSCRIPT_EN.md, APP = APPENDICES_EN.md, SUPP = SUPPLEMENTARY_EN.md. Run IDs: B = v2L1, C = A0_ctrl, S8 = A2_tail8, A2b = A2b_tail8, A3 = A3_2grid, B+W = B2grid, P0 = c_oh.

## Summary

| Issue | Verdict | Action | Priority | Cost (one RTX 5090) |
|---|---|---|---|---|
| I-19 training through the correction is confounded | **CONFIRMED** for attribution: no C+W evaluation exists anywhere. **PARTLY REFUTED** on three sub-points: (1) C and A3 saw the same geometry sequence, so C+W is an exactly matched control; (2) the "extra data" amounts to about 74 geometries not seen by B, not 286; (3) the A3 vs B+W gap is robust across geometries for force/support/face (95–99% of geometries), and existing seed-noise data from an earlier pilot are about 5% relative | (b) C+W evaluation, plus (a) reworded claims. (c) seeds optional | **must** (C+W); should (1 extra seed) | C+W: about 1 GPU-h (80-cell eval 27 min, 14 pairs about 15–30 min), +1 h for the 2×2×2 lattice. One extra seed for A3 and C: about 7 GPU-h |
| I-26 view dependence and selection text | **CONFIRMED** (text inconsistent, A3 has no view-17 number). Mitigations: the final checkpoint won in every continuation; B's view effect is small (1.05× on the basic classes, 1.13× on force_c); W is equivariant by construction | (a) text, plus (b) extraction of view-17 data from the training logs (0 GPU), plus a short eval (view 17 for A3 and B+W) | should | 0 GPU-h (logs); 55 min for view 17 on 80 cells; about 4.5 h for 5 extra views |
| I-33 data-free (pilot running) | DECIDED. The pilot is strictly label-free in its gradients and its selection. The DF_W (post) vs DFW (through) pair is also a clean training-through control for I-19 | (a) write-up after the outcome; templates below | must (§3.3 text now) | 0 beyond the running pilot |
| I-34 w_s and architecture not ablated | **CONFIRMED** for w_s (no clean w_s = 0 arm; A1_sens3 was stopped with no result; the DF pilot arm is confounded). **PARTLY REFUTED** for architecture: Table 3 already shows harmonic/zero starts are 7–290× / 2,400–12,000× worse, and earlier v1-generation B1/B2 component arms were neutral. New: the training logs imply that the sensitivity term is only about 1–2% of A3's objective value | (b) sens_loss share from logs (0 GPU); (c) one A3-protocol run with w_s = 0; architecture: (d) rebuttal plus optional 2 small arms | should (w_s run); could (architecture) | w_s = 0: about 3.8 h training + about 1 h eval. Architecture arms: about 2 h each at 10k steps without W |
| I-43 591 / 305 counts | **CONFIRMED** that the evidence gap is real, but 305 is **correct** per contemporaneous notes (148 legacy + 157 new). `SPLIT_V3.json` was later regenerated, which is why `meta_p1.json` shows 591. Each 15k continuation visited about 153 distinct geometries | (b) extract SPLIT/VIEW events from the server logs (0 GPU), plus (a) replacement wording | should | 0 |

Side finding (affects I-34 cost and MS 355): **"A3's continuation took 2.7 h" understates the run.** `A3_2grid.log` was overwritten by the resumed run (`chain_a3r.sh`, `> $O/A3_2grid.log`). The `meta_p1.json` curve starts at step 4,050 (220 step records), and `max_step_seconds_field` = 9,807 s covers only steps 4,000–15,000 plus one training-time evaluation. At 0.653 s/step plus about 1,900 s per training-time evaluation, the full 15,000-step continuation is about 15,000 × 0.653 + 2 × 1,900 ≈ 13,600 s ≈ 3.8 h. That is about 3.4 h without the second evaluation, and 2.7 h is the time after resume. Fix MS 355 ("about 3.5 h, of which 2.7 h after a restart at step 4,000") or re-measure. The step-1–4,000 segment log is lost unless a copy exists.

---

## I-19: benefit of training through the correction

### Does a C+W evaluation (or any model trained without W, evaluated with W, other than B) exist?

**No.** I searched `evidence/` (all `newval*`, `gate_*`, `lat_*`), `docs/data/**` and the chain scripts.
- `newval2_B2grid.json` (= B+W) is `v2L1/best.pt` evaluated with the W override. It is the only "trained without W, evaluated with W" evaluation.
- `newval*_A0_ctrl.json` and `gate_A0_ctrl_*` are C evaluated **without** W.
- `lat_hetero222_A3.json` was invoked with three models (`A3`, `B2grid=…;{W}`, `C=A0_ctrl/best.pt`), but its `lattices.hlat222` contains only `exact` and `A3`. So no C (or C+W) lattice result was archived.
- The running DF pilot (`chain_dfp.sh`) will produce `newval2_DFP_CTRL_W` and `newval2_DFP_DF_W`. These are trained-without-W models evaluated with W, but from random initialisation at 10k steps (see I-33). They are not C+W.

The confound is therefore **not resolved** by existing data.

### What refutes part of the concern

1. **C is an exactly matched control for A3.** `train3.py:514,723` shuffles the training list once with `np.random.default_rng(seed)`, and every arm uses seed 0 and `SPLIT_ARMS.json`. C, S8, A2b and A3 therefore admit the same geometries in the same order, and the O_h view per entry comes from `ogen`, seeded identically. A3 was resumed with `resume_rng: true`, which restores `qi`, the pool and the RNG states, so its sequence is unchanged. C and A3 differ only in the correction, and through it in the adversarial directions and gradients. APP 460 states this correctly. C+W vs A3 therefore isolates "training through W" exactly, apart from nondeterminism.
2. **The "extra data" is smaller than "591 vs 305" suggests.** Pool 3 with one admission per 100 steps gives 3 + 150 = 153 geometries in a 15k continuation. I reconstructed the order from the insertion order of `meta_p1.json` `splits[SPLIT_ARMS].per_case`, whose first 591 keys are the train list, using numpy 2.4.6 `default_rng(0).shuffle`. Of the 153 geometries, **79 are in the first 305 list entries (B's population, if the ingest appended) and 74 are not**; 41 are legacy-family geometries. *Assumptions:* pool = 3 and swap_every = 100 (APP G.3), and B's 305 = the first 305 entries. Verify both from the logs (I-43).
3. **The A3 vs B+W gap is not a few-geometry artefact.** Paired per-geometry analysis (`newval2_A3_2grid` vs `newval2_B2grid`, view 0):

| Class | n | A3 mean (%) | B+W mean (%) | Ratio of means [95% CI] | Geometric-mean ratio [CI] | Share of geometries where B+W is worse |
|---|---|---|---|---|---|---|
| force_c | 80 | 0.0737 | 0.0965 | 1.31 [1.20, 1.40] | 1.16 [1.10, 1.23] | 0.76 |
| force | 80 | 0.0579 | 0.0917 | 1.59 [1.43, 1.75] | 1.69 [1.58, 1.82] | 0.96 |
| support | 80 | 0.0550 | 0.0840 | 1.53 [1.37, 1.70] | 1.50 [1.42, 1.58] | 0.99 |
| face | 80 | 0.0370 | 0.0725 | 1.96 [1.48, 2.39] | 1.91 [1.76, 2.07] | 0.96 |
| face_c | 80 | 0.0321 | 0.0429 | 1.34 [1.25, 1.41] | 1.17 [1.12, 1.24] | 0.84 |
| support_k | 75 | 0.0568 | 0.0771 | 1.36 [1.23, 1.47] | 1.14 [1.07, 1.21] | 0.67 |
| glued | 75 | 0.0597 | 0.0780 | 1.31 [1.19, 1.41] | 1.12 [1.07, 1.19] | 0.67 |
| macro | 80 | 0.0151 | 0.0216 | 1.43 [1.29, 1.55] | 1.09 [1.03, 1.16] | 0.45 |
| grf | 80 | 0.0247 | 0.0291 | 1.18 [1.11, 1.23] | 1.02 [1.00, 1.05] | 0.33 |
| force_c, 60 non-selection | 60 | 0.0767 | 0.1011 | 1.32 [1.19, 1.43] | 1.18 [1.10, 1.26] | 0.78 |

force_c by stratum (ratio of means): uncut 1.78, light 1.22, moderate 1.15 (geometric-mean CI [0.97, 1.12] includes 1), heavy 1.43. The median per-geometry force_c ratio is only 1.10. The paper's "1.3" is a mean driven by a minority of geometries, while the force/support/face gains are broad.

4. **The raw effect of the continuation is small.** C vs B, no W, same geometries: force_c 1.09 [1.05, 1.13], force 1.09, support 1.09, glued 1.04, macro 1.02, grf 1.01. In the pairs, C improves U1/x sensitivity from 5.83% to 4.89% (1.19×) and M1/x from 12.15% to 11.37% (1.07×). These are much smaller than the A3/B+W factors (1.59 on force; 1.35–2.2 on U1/M1). That is suggestive, but not decisive: the continuation may still improve exactly the components W does not remove.
5. **Seed variability evidence exists, but outside the paper.** `V2_ARMS_SPEED_DEPLOY_20260925_CN.md` §9: in the earlier r2 pilot, p_ctrl seeds 1/2/3 (warm-started 10k continuations) differ by ≤ 0.01 pp in gate compliance and ≤ 0.11 pp in gate sensitivity at about 2.1%, which is about 5% relative. The same document (§1) and AUDIT_V2 M6 record 2–8% nondeterminism in single-step loss between same-seed reruns. This is indicative that seed noise in warm-started continuations is well below the 1.35–2.2 factors, but it is **of the same order as** the moderate-stratum force_c gap and the H1/H3/L1 pair differences. It is not paper evidence: a different model (mgno2_r2) and gate.

### What confirms the concern

- **Pair level.** B+W/A3 sensitivity ratios over the 14 configurations: M1/x 2.18, U2/y 2.18, U2/x 1.73, M1/y 1.65, U1/x 1.58, H1/x 1.41, U1/y 1.35, L1/y 1.11, H1/y 1.08, H3/x 1.07, H3/y 1.03, L1/x 0.64, M2/x 0.61, M2/y 0.49. A3 is lower in 10 of 14 (two-sided sign test p = 0.18; the configurations are also paired within cells). Compliance is lower for A3 in 7 of 14. "Factors of 1.35 to 2.2" (MS 436, 568; §8) selects U1/M1. By the same measure, B+W is 1.6–2.0× better on M2 and 1.6× better on L1/x.
- **Correlation.** Per geometry, the log C-over-B raw gain on force_c correlates with the log A3-over-B+W gain (Spearman 0.51, p = 2e-6). Where the plain continuation helped, A3 also beats B+W, which is consistent with a continuation share in the gap. It is ≈ 0 for force (−0.03).
- **Seeds.** There is only one seed for every arm in the paper (all `seed: 0` in `meta_p1.json`).

### Actions

**(b) C+W evaluation (must; no training).** Run after `DF_PILOT_DONE_20260928`.
```
O=/root/autodl-tmp/OPL/S1/V2; W='{"smooth_k": 8, "smooth_alpha": 30.0, "coarse_space": "Q1_17"}'
VAL=$(python3 -c "import json;print(','.join(json.load(open('/root/autodl-tmp/OPL/S2/SPLIT_ARMS.json'))['val_s3']))")
OPL_MODEL_ARGS_OVERRIDE="$W" OPL_CONV_FP32=1 $PY -u eval_views.py $O/A0_ctrl/best.pt $O/newval2_CW.json \
    --views 0 --cases $VAL --body /root/autodl-tmp/OPL/S0 --data /root/autodl-tmp/OPL/S2/data_v2
```
- Cost: 1,635 s, as `newval2_B2grid`.
- Then run the 14 pair configurations exactly as `gate_B2grid_*`: cells 2000, 2001, 2003, 2005, 2006, 2002, 2004 × {x, y}; `lat_full.py … --nb-mode explicit --sets test`. Use the same override mechanism the server used for `gate_B2grid_*`. Note that the local `src_v2_wip/lat_full.py` does not read `OPL_MODEL_ARGS_OVERRIDE`, so check the server copy or use a checkpoint copy with `cfg.model_args` amended. The B+W gates took 865 s of solve time in total, so allow about 30 min.
- Optional: the lattice, via `lat_hetero.py --model 'CW=$O/A0_ctrl/best.pt;{W}'`, about 55 min for hlat222.
- Analysis: the `i19.py` pairing with C+W in place of B+W. Report the ratios A3/(C+W) and (C+W)/(B+W), with CIs, per class and stratum, plus the 14 pair maxima.
- Decision rule:
  - If A3/(C+W) on force_c has a CI including 1, or ≤ 1.1, then "training through" is not supported: rephrase contribution (ii), §6.3/§6.6/§7.3/§8 and the Abstract.
  - If it is ≈ the A3/(B+W) ratio, the attribution holds for this seed.
  - Otherwise, report the split: (C+W)/(B+W) is the continuation share and A3/(C+W) is the training-through share.

**(c) Seeds (should; at least one extra).**
- Configuration: the A3 and C configurations with `seed: 1`. This gives a different geometry order, views and adversarial draws. `init` stays B.
- Cost: A3 about 3.8 h (see the side finding); C: 15,000 × 0.298 s + 2 × 930 s ≈ 1.8 h. Evals (newval2 A3_s1 and C_s1+W, 2 × 27 min, plus 2 × 14 gates ≈ 1 h) bring it to about 7 GPU-h. A second extra seed doubles this (about 14 GPU-h for three seeds per arm).
- This also serves as the noise floor for the I-34 w_s = 0 arm.
- If seeds are not run, cite the r2-pilot seed spread only after archiving it, and call the factors single-seed.

**(a) Text, in every case.**
- MS 341: "isolating the effect of training through the correction" → "B+W differs from A3 both in training through the correction and in A3's 15,000 further updates; C+W isolates the former (Table ST…)".
- MS 381/436/444/568 and §8: give the ratios with CIs and the M2 reversal. Example for §8: "training through the correction lowers the population mean by a factor of 1.31 (95% bootstrap interval over geometries 1.20–1.40; one training seed) and the largest sensitivity errors on U1 and M1 by 1.35–2.2, whereas on M2 the untrained combination is 1.6–2.0 times more accurate."
- Abstract: "the correction alone brings an extension trained without it within the accuracy reference (B+W: 0.0965%); training through it lowers the error further (0.074%)".

---

## I-26: view dependence of A3; checkpoint selection text

### Facts

- **Checkpoints evaluated** (`meta_p1.json` `log.evals`):
  - B: 10k 0.1057, 20k 0.1065, 30k 0.0905, 40k 0.0912. Selected 30k, 0.7% better than 40k.
  - C: 7.5k 0.0944, 15k 0.0906.
  - S8: 0.0582, 0.0553.
  - A3: 0.00281, 0.00237.

  **For every continuation the final checkpoint (15k) won.** A2b is not in `meta_p1.json`. Its chain config (`chain_a3r.sh` `cfg()`) sets `eval_every = 7500`, `select_min_step = 7500` and `eval_views = [0, 17]`, as for the other arms, so ST12's "A2b is evaluated … at update 15,000" (one checkpoint), R6-10 and ST12's "For B, C, S8 and A3 … views 0 and 17" (A2b omitted) are unverified and probably wrong. Check `A2b_tail8.log` (EVAL events) or add A2b to `meta_p1`.
- P0 uses `last.pt` (15k), not the selected checkpoint. V2 doc §1: the identity-only selection had picked 5k.
- **The pair cells are all selection geometries.** Selection used `val[:40]` = 20 legacy-family validation geometries + `fresh_val_2000`–`2019`. All assembly cells (U1, U2, M1, M2, H1, H2, H3, L1 = 2000–2006, 2010) are in that set. MS 357 discloses "development cases" but not this overlap.
- **B's view effect is small.** `newval_v2L1.json`, views 0 and 17, ratio of means view 17/view 0: force 1.047, support 1.050, face 0.972, macro 1.056, grf 1.040 (n = 80); force_c 1.131, face_c 1.107, support_k 1.122, glued 1.084 (n = 15–20). View 17 is worse on 71–89% of geometries for most classes. P0 is 1.01–1.10.
- **W is equivariant by construction.** Jacobi-scaled Chebyshev commutes with the cube permutations, and the Q1(17) grid maps onto itself under O_h. The only non-equivariant part is the seeded power-iteration start for the interval, which is inflated by 1.05. Any view dependence of A3 therefore comes from the network. Its relative size after W is unknown: W may damp or amplify it relatively.
- **View-17 data for A3 already exist on the server.** The training-time `EVAL` record (`train3.py:1026`) logs `score_views`, `score_identity` and `weights=res`. `res` contains `views['17']` per-geometry values for the 40 selection geometries at 7.5k and 15k, for B, C, S8, A3 (and A2b). `meta_p1.py` kept only the view-0 `val_mean`.

### Actions

- (b, 0 GPU) From `A3_2grid.log`, `A0_ctrl.log`, `A2_tail8.log`, `A2b_tail8.log` and `v2L1.log` (at the selected step), extract the view-0 vs view-17 per-class means on the 40 selection geometries. Note that A3's log lost steps < 4,000, but both of its EVAL events (7.5k, 15k) are in the surviving segment.
- (c-lite, 55 min) `eval_views.py $O/A3_2grid/best.pt newval2_A3_v17.json --views 17 --cases $VAL …`, and the same for B+W with the override. This gives an 80-cell view-17 comparison of A3 and B+W, directly comparable to ST01c. Optional: views 5, 29, 38, 46 (the default set of `eval_views.py`) for both models, about 4.5 h in total, which gives the view-to-view spread R2-12 asks for.
- (a) Replacement wording:
  - **MS 357**: "selection compared two checkpoints per predictor" → "selection compared the checkpoints at 7,500 and 15,000 updates for each continuation, and in every case the final checkpoint scored lower; B's weights were selected from four checkpoints (10,000–40,000 updates), and P0 uses its final weights. All cells used in the assembly examples belong to the 20 selection geometries."
  - **APP 460**: "The weights are selected at step 15,000 on the validation list described in Section 6.1." → "Checkpoints were scored at 7,500 and 15,000 updates on the validation list of Section 6.1 (views 0 and 17); the 15,000-update weights scored lower in every continuation and are reported."
  - **SUPP 386/393**: "For B, C, S8 and A3" → "For B, C, S8, A2b and A3" and delete "A2b is evaluated with its EMA weights at update 15,000", *if the A2b log confirms two evaluations*. Otherwise state A2b's actual protocol.
  - **§6.3 / ST01c**: add the A3 and B+W view-17 rows, or state "A3 was evaluated in the identity view only; B's error changes by factors of 0.97–1.13 across the classes in view 17".

---

## I-33: data-free pilot (decided; how to write it up)

**What the pilot is** (`dfp_cfg.py`, `chain_dfp.sh`):
- Arms and setting: three arms from **random initialisation** (the `init` and the B1 bounds are removed), 10k steps, seed 0, `SPLIT_ARMS` (591 pool; ≈103 geometries visited), label-free selection (`select_by: logE` on macro and grf, views 0 and 17).
- DF: mix macro 0.1 / grf 0.125 only, `sens_w = 0`, `adv_on_load = False`.
- CTRL: the full labelled recipe.
- DFW: DF trained through W.
- Evaluations (newval2, 80 cells, view 0): DF, DF_W (W at evaluation only), CTRL, CTRL_W, DFW.
- Cost estimate from the logged step times: about 2 h per no-W arm (0.30 s/step + 4 training-time evals × about 930 s) and about 4 h for DFW (0.653 s/step + 4 × about 1,900 s). With five evals (about 25 min each), the pilot takes about 10 h.

**Strictness check (refutes a possible objection).** With the log objective, ∇θ log(qᵀŜq/qᵀSq) = ∇θ log(qᵀŜq), independent of the stored normalisation by exact energy. `lf_score` (`train3.py:451`) ranks checkpoints by mean log predicted energy. Per-direction offsets are constant across checkpoints, so the ranking is identical to a label-based log-ratio ranking. Macro and grf directions need no solve, and neither does W (it uses K only). The DF arm therefore uses no exact solve in its gradients or its selection; exact data enter only the reported errors. Say so explicitly, and say that the unit-energy normalisation of the stored banks is used for reporting only.

**Interpretation limits (state in either case).**
1. The pilot arms have a far shorter lineage than A3 (random init, 10k steps, vs c_oh → v2s → B 40k → A3 15k). Compare within the pilot only, never with A3's absolute numbers.
2. DF vs CTRL differs in three factors at once: directions, w_s and the adversarial search. It does not isolate w_s (I-34).
3. Single seed; use the ≈5% seed spread above only as an indicative floor.
4. **DF_W (post) vs DFW (through) is an exactly matched pair** (same data, steps, init, seed and loss). It is a clean second instance of the I-19 question and should be reported together with C+W.

**Write-up if DF ≈ CTRL** (e.g. population force_c within about 1.5× and the corrected pair/lattice sensitivities within the 3% reference):
> "A strictly label-free variant, trained from random initialisation only on solve-free macroscopic and random-field directions, without sensitivity labels or the adversarial search, reaches X% (with the correction applied at evaluation, X_W%) against Y% (Y_W%) for the labelled recipe at the same budget of 10,000 updates (Table S…). At this budget the exact solves are therefore needed for evaluation rather than for training; the gap to the principal predictor reflects its longer training lineage, not the labels."

Then add a sentence in §3.3 and §7 contrasting this with data-free PIML (Huang et al. 2024).

**Write-up if DF ≫ CTRL:**
> "Training without solve-derived directions and sensitivity labels increases the population error from Y% to X% at the same budget, chiefly in the classes that the solve-free directions do not span (force, support, glued: …). The present recipe therefore relies on exact Neumann/pinned solves for 77.5% of the sampled directions and on the eight-corner sensitivity labels; replacing them by solve-free directions is possible in principle, since the energy term is a Ritz principle, but is not competitive at this budget."

Report class-wise results to show where transfer fails. Also report whether DFW narrows the gap, i.e. whether W in the loop compensates for missing labels.

**In both cases:** add the §3.3 statement of which classes need exact solves now (independent of the pilot). Put the pilot table in the supplement with population energy by class and stratum, the 14 pair maxima (only if gates are run for the pilot arms, about 30 min per arm) and the lattice sensitivities (optional, about 1 h per arm).

---

## I-34: sensitivity term and architecture

### Existing runs that differ only in the sensitivity term or the architecture

| Run | Differs from its control in | Status / result | Usable? |
|---|---|---|---|
| A1_sens3 (`chain_diag2.sh`) | `sens_w = 3` vs C | Stopped before any result (LATE_RESULTS: "A1_sens3 stopped") | No |
| p_sl1 vs p_ctrl (r2 pilot, V2 doc §8) | smooth-L1 sensitivity loss (δ = 0.003) vs squared | Gate sensitivity 2.09 → 3.84% (worse) | Form of the term, earlier model; not w_s = 0 |
| c_b1, c_b2 vs c_ctrl (v1, V2 doc §1) | B1 bounded gains; B2 extra geometry features | Neutral (identity 3.9/4.6/2.7 vs 3.9/4.5/2.6–2.7%) | Earlier generation; small components only |
| DFP_DF vs DFP_CTRL (running) | w_s = 0 **and** directions **and** adv | Pending | Confounded for w_s |
| Table 3 (MS 420ff) | zero / harmonic / B start under the same W | Harmonic 7–290× worse, zero 2,400–12,000× worse | Answers "geometry-agnostic or trivial extension" at the extreme |

A clean w_s = 0 arm does **not** exist. The v2 architecture has no component ablation.

**New quantitative hint (0 GPU, from `meta_p1.json` curves).** `batch_loss` = mean log ê + w_s·sens term, and `e_mean` = mean(ê − 1). For A3, where ê − 1 ≈ 10⁻³ so log ê ≈ ê − 1, the per-step median of (loss − e_mean)/loss is 1.7% in the first fifth and 0.8% in the last fifth of the logged segment. That is, **the sensitivity term makes up only about 1–2% of A3's objective value.** Both terms are quadratic in the field error, so their gradient contributions are probably of similar relative size. This supports R2's argument (the term is nearly inert once the energy is small), not R7's. It is not a proof: the direction of the small gradient could still matter. The exact per-step `sens_loss` is logged in the STEP events (`train3.py:1001`) and should be extracted (b, 0 GPU) and reported.

### Minimal ablation spec (c)

- **A3_ws0.** `cfg A3_ws0 '{"model_args": {"smooth_k": 8, "smooth_alpha": 30.0, "coarse_space": "Q1_17"}, "sens_w": 0.0}'`. Everything else is as A3: init B, seed 0, SPLIT_ARMS, 15k steps, eval 7.5k/15k, views 0 and 17. Keep the selection score unchanged, including its sensitivity part, so that only the loss differs. Because of seed 0 it sees A3's geometry sequence, so the comparison is paired.
  - Cost: ≤ 3.8 h. Probably less, since `sens_hat` is skipped when sw = 0.
  - Evals: newval2 (27 min), 14 pairs (about 30 min), hlat222 (about 55 min).
  - Total about 5–6 GPU-h.
- **Comparison.** Population energy by class; the 14 pair sensitivity maxima; the lattice sensitivities. Judge against the A3 seed-1 run of I-19 as the noise floor, or against the ≈5% indicative spread.
- **Decision.** If the |Δ| in the pair/lattice sensitivity maxima is within noise, drop "with an objective that includes thickness sensitivity" from the Abstract and contribution (ii). Keep the term as an implementation detail and report the ablation in one sentence.
- **Architecture** (could; otherwise rebut).
  - Rebuttal: Table 3 bounds the "simple extension" end; a component study is outside the paper's scope. Keep it short.
  - If run: architecture changes cannot warm-start from B, so use the pilot setting with DFP_CTRL as the baseline (random init, 10k steps, same config). Two arms, each changing one `model_args` entry that already exists: `n_fringe: 0` (no weak-region layers) and `F: 16` (half width; report parameter count and application time). About 2 h each without W, or about 4 h through W, plus 2 × 25 min of evals.
  - "Latent hierarchy off" (`levels`) and "non-conditioned fixed coefficients" need a smoke test or code; not minimal.

---

## I-43: training-population counts

### Facts

- **591.** The length of `train` in the current `SPLIT_V3.json` and `SPLIT_ARMS.json`. The two files are identical (`meta_p1.json`; same 691 per-case keys). The list is 148 legacy-family geometries followed by 443 new independent-field cells (fresh_train_2000–2451, in ID order). `meta_p1.py` reads the split file **at meta time** (`Path(sp).read_text()`), not at training time.
- **305.** Not in `evidence/`, but recorded contemporaneously:
  - `arch_analysis_0926/BRIEF.md:40`: "v2L1 trained on 305 training geometries (157 new independent-field cells + old), 40k steps; v2L2 (591 geometries) paused at 10k".
  - `V2_ARMS_SPEED_DEPLOY_20260925_CN.md` §12: "v2L1 (+157 new cells, 40k steps)".
  - `HANDOVER_PAPER_20260926_CN.md:90,119`.
  - 148 + 157 = 305. `ingest_s3.py` writes `S2/SPLIT_V3.json` by default and was re-run as production grew, which explains the 591 that R6-16 found. **305 is correct for B; the paper lacks the archived evidence.**
- **C/A3 on 591.** v2L2 (591) existed before the arms, and SPLIT_ARMS equals the current SPLIT_V3, so 591 is very likely. But `WORKFLOW_RESULT.json` calls A0 "305 geos", BRIEF says "305+", and the handover says "same data". This must be settled from the `SPLIT` event in `A0_ctrl.log` and `A3_2grid.log`.
- **Distinct geometries visited** (pool 3, one admission per 100 steps; see I-19):
  - B by its selected step 30,000: 303 of 305. B's init v2s was trained on the 148 legacy geometries.
  - Each 15k continuation (C, S8, A2b, A3): about 153 of 591, in the same sequence for all four; about 79 in B's 305, about 74 new, 41 legacy.
  - P0 (c_oh, 15k): all 148. `meta_c_oh.json` contains the `SPLIT` event with train = 148, which is the only primary count archived.

### Actions

- (b, 0 GPU, on the server) Run `grep -h '"event": "SPLIT"' $O/{v2L1,A0_ctrl,A2_tail8,A2b_tail8,A3_2grid}.log` to get the train counts at training time. Count the distinct `case` in `"event": "VIEW"` records with step ≤ the selected step, per log. For A3, also take the `visits` dict from `A3_2grid/ckpt.pt`, since the log lost steps < 4,000. Check that all of v2L1's VIEW cases lie in the first 305 entries of the current list; if so, archive that list as B's training set. Add a `split_counts.json` to `evidence/`. Extend `meta_p1.py` to record the SPLIT event and the visit counts.
- (a) Replacement wording:
  - **Table 2 / ST12 column** "Training geometries" → "Training pool / distinct geometries visited": P0 148 / 148; B 305 / 303; C, S8, A2b, A3 591 / 153 (B+W: B's).
  - **MS 341**: "continues B on the larger training population" → "continues B for 15,000 updates, drawing from a pool of 591 geometries (B's 305 and 286 cells produced later); with one pool replacement every 100 updates, each continuation visits about 153 distinct geometries, in the same order for C, S8, A2b and A3, about half of which B did not see."
  - **APP G.3**: add "A run of N updates therefore visits at most N/100 + 3 distinct geometries."
  - **Table 2, P0 role**: "Reference with a smaller training population" → "Earlier predictor (148 legacy geometries, separate training lineage)". The counts do not support a data-scaling reading of P0 → B → C.

---

## Top author decisions

1. **I-19: run C+W now (about 1 GPU-h after the pilot) before touching the Abstract.** It is exactly matched to A3 (same geometry sequence). Pre-commit to the decision rule: if A3/(C+W) ≤ 1.1 on force_c, reframe contribution (ii) as "the correction; training through it is a refinement" and remove the "trained through …" lead from the Abstract. Decide whether to add one extra seed (about 7 GPU-h) or to label all factors "single seed".
2. **I-34: keep or drop the sensitivity term as a claimed contribution.** Either run A3_ws0 (about 5–6 GPU-h, paired with A3) or drop "with an objective that includes thickness sensitivity" from the Abstract and contribution (ii) now. The logs already show that the term is about 1–2% of A3's objective. Architecture: rebut via Table 3 unless the reviewers' small set is wanted (about 5 GPU-h for 2 arms).
3. **Facts to fix regardless of experiments:**
   - 305/591 provenance: extract from the server logs; relabel as pool / visited.
   - Selection text: the final checkpoint always won; A2b's protocol to be checked; pair cells ⊂ selection set.
   - A3 wall time: 2.7 h covers only the post-restart segment; the full run is about 3.5–3.8 h.
   - A3/B+W view 17: 55 min, or state it as a limitation.

---
**Coordinator addendum (server logs, read-only grep, 2026-09-28):** the SPLIT events settle the I-43 count question: `v2L1.log` (B) `{"event": "SPLIT", "train": 305, "val": 40}`; `A0_ctrl.log` (C), `A2b_tail8.log` (A2b) and `A3_2grid.log` (A3) all `{"event": "SPLIT", "train": 591, "val": 40}`. The pool sizes 305 (B) and 591 (continuations) are therefore confirmed at training time; the "A0 = 305 geos" project note is wrong. Distinct-visited counts (≈153 per continuation) remain a reconstruction.
