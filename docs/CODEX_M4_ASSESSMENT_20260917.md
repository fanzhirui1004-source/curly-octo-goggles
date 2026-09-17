# Assessment of the Codex M4 CutFEM baseline (`CUTFEM_M4_BASELINE_20260917`)

Read on 2026-09-17 from `/root/autodl-tmp/CUTFEM_M4_BASELINE_20260917` on the AutoDL box
(`a841046-9982-eae5cdfb.bjb2`). Sources: `source/stage_cutfem_m4/`,
`DELIVERY/reports/m4_cutfem_20260917/{M4_CUTFEM_BASELINE.md,SUMMARY.json}` and the
`evidence/` subtrees. Branch `codex/m4-cutfem-baseline-20260917`, SHAs e4983f0df /
e5ad375e5 / 86f14925b.

## 1. What was built

A complete geometry -> operator pipeline:

- input: native 32^3 ghost-penalty CutFEM geometry of one unit cell;
- network: 5,077,241 parameters (192-dim context, 768x4 decoder trunk, patch-pair
  context, two GRUs, three factor-block heads);
- output: the **complete scalar Cholesky factor** `L` of the quotient operator,
  81,824,028 independent entries;
- reconstruction: `A = L Lt`, then `S_hat = Bt L Lt B` with the frozen Householder
  quotient `B`, `d = 12,792`.

Reported cost: complete-factor inference ~3.25 s on seat 0328 (teacher condensation
31.85 s), whitened spectrum ~7.2 s, 0.127 s/step training, peak 11.045 GiB.

## 2. What it measures

All acceptance gates fail, and not narrowly. `eps_op = max|mu - 1|` over the whitened
spectrum; work gate 0.10, target gate 0.03.

| run / seat | e_A | D/d | mu range | modes <0.9 / >1.1 | 6-D compliance max err |
|---|---|---|---|---|---|
| G1_0328_256 / 0328 | 44.248% | 2.3846 | [0.0120, 1080.12] | 5206 / 6474 | 97.651% |
| G1_0328_3000 / 0328 | 16.610% | 0.8718 | [0.00220, 596.905] | 4938 / 5164 | 95.744% |
| G1_0328_MIXTURE_256 / 0328 | 43.222% | 2.6189 | [0.0126, 880.62] | 5114 / 6562 | 97.432% |
| TRAIN2_256 / 0253 | 46.203% | 4.1046 | [0.0453, 1346.59] | 4645 / 7084 | 98.443% |
| TRAIN2_256 / 0403 | 47.933% | 3.9224 | [0.00421, 761.61] | 6746 / 6549 | 97.145% |

Best run: `eps_op ~ 595.9`, i.e. about 6.0e3x the work gate and 2.0e4x the target gate;
10,102 of 12,792 modes outside +-10%.

Three framings matter:

1. These runs **fit a single known label**. This is not a generalisation number; it is a
   memorisation number, and memorisation is still off by a factor of ~600.
2. The 6-D rigid-fixture compliance check is the most benign test available (six smooth
   global load directions, no soft modes probed). It is 95.7% wrong on the best run.
3. From 256 to 3000 steps `e_A` improves 2.66x (44.2% -> 16.6%) while the 6-D response
   moves 97.651% -> 95.744%, i.e. essentially not at all.

## 3. Why the loss and the gate are decoupled

Their own sampling diagnostic (`evidence/SAMPLING_0328/RESULT.json`) is the mechanism.
Over 262,144 sampled cross-patch pairs:

- blocks at distance [0, 0.0625] are **0.444%** of pairs and **97.09%** of sampled factor
  squared norm;
- the top 1% of blocks carry 99.61%;
- the top 0.1% of blocks carry 94.64%.

So a Frobenius-type loss on `L` is ~97% a statement about the near field. The near field
is the easy part - it is close to local element stiffness and smooth in the geometry. The
gate is decided by the far/soft end of the whitened spectrum, which carries the remaining
~3% (really ~0.4%) of the norm and therefore ~0.4% of the gradient.

This is the amplification ladder of `docs/FIRST_PRINCIPLES_PATH_20260917.md` seen from the
other side: `cond(A) = 7.2254e4`, and the near/far cancellation ratio `rho_J = 1e3..1e4`
means a *systematic* near-vs-far bias `delta` gives `eps_op ~ delta * rho_J`, so any
representation that predicts near and far separately needs relative coherence of
3e-6..3e-5. A norm-weighted loss cannot supply that, because it does not see the far field
at all. Item 3 in section 2 is exactly the predicted signature: the norm-weighted number
improves, the accepted quantity does not.

Conclusion: more training steps will lower `e_A` further and will not move the gate. The
gap is structural, not a budget problem.

## 4. What is genuinely valuable here

- **A working end-to-end geometry -> S pipeline.** First time in this project that
  anything runs all the way from a CutFEM unit cell to an `S_hat` under the frozen
  quotient.
- **SPD by construction.** Predicting `L` and forming `L Lt` puts the prediction in the
  right cone for free. We have spent real effort on this question elsewhere.
- **The cost case works.** 3.25 s versus 31.85 s is ~10x before any optimisation, and the
  bottleneck is assembling 81.8M factor entries, not the network forward pass. A head that
  emits far fewer numbers moves this a long way.
- **Honest acceptance reporting.** They report the full whitened spectrum, the mode counts
  outside the band, and a re-equilibrated 6-D response - not just a Frobenius number - and
  they report them failing. Plus zero-tolerance forward/backward comparison against the
  archived code, 12 contract tests, recorded SHAs, frozen teacher untouched, and explicit
  non-claims in the report.
- **Their report already asks the right question**: "if the factor/Frobenius error keeps
  falling while the response stays put, study response- or energy-auxiliary supervision on
  the same factor head." That is the correct diagnosis, reached independently.

The harness is the expensive part and it is built. The learning target on top of it is the
part that is wrong, and swapping a target is much cheaper than building a harness.

## 5. Recommended changes, keeping their infrastructure

1. **Replace/augment the three-bucket factor MSE with a whitened two-sided extreme-mode
   loss.** Get `mu` at both ends by a few Lanczos/power iterations on
   `H = R^-T A_hat R^-1` and penalise `f(mu) = mu - log mu - 1`. Whitening removes the
   7.2e4 amplification, so near and far enter the gradient on equal footing. Two lessons
   from our own E1-F failure apply directly: do **not** use `relu(mu_max - 1.03)^2` - with
   `mu_max ~ 1e5` it dominates the gradient *direction* and Adam normalises magnitude, not
   direction; and get `mu_min` by power iteration on `H^-1 = M^-1 M^-T`, not on `cI - H`,
   which does not converge and returns plausible-looking garbage silently.
2. **Cheap interim fix**: weight each factor block by `1/||L_block||` so the loss measures
   *relative* per-block error. Their sampling code already computes the per-block norms.
   This alone lifts the far field from ~0.4% to ~50% of the gradient.
3. **Run the two falsifiers before scaling training up.** (A) gamma sweep at 1e-3 / 1e-5 on
   seat 0328: if the labels move when the ghost-penalty parameter moves 100x, the labels
   are not a well-posed 3% target and no network fixes that. (B) near/far coherence oracle
   on the H-matrix that already passes +-3%: measures directly how coherent near vs far
   must be, i.e. whether *any* separate-prediction architecture can reach the gate.
4. **The structural alternative their pipeline is well placed to try**: stop predicting the
   factor. Predict local multiplicative physical fields (per-element modulus/correction)
   and run the condensation exactly. Amplification becomes exactly 1 instead of 7.2254e4,
   the output shrinks from 81.8M numbers to O(elements), and inference cost drops with it.
   Their geometry adapter and the frozen quotient carry over unchanged; only the head
   changes. The in-flight feature work in `cutfem_m4_feature_dev.tar.gz` (`test_features.py`,
   GP face structure / local stiffness features) is aimed at exactly the inputs this route
   needs.

## 6. Verification note

Re-reading the box at the time of writing returns HTTP 403 on the Jupyter API (token
rotated or instance restarted). Every number above was read first-hand from the files
listed at the top of this document earlier in the same session; none is re-derived from
memory of a summary.
