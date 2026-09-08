# Step 1 report: Tet10 label route, carrier-weighted gates (2026-09-02)

Scope: G0 (uncut, q=4614), G2 (near-empty wedge, q=153), G5 (deep cut, q=1476). A0 sizing meshes,
3 optimizer policies x 2 surface-criteria variants (no-perturb meshes rejected by FE: 0-degree dihedral slivers).
Norm: carrier low-frequency subspace (wavelength >= 4H = 0.25), Frobenius relative; also mass-whitened and raw.

## Gate results

| case | element | reproducibility (different meshes) | same mesh, different optimizer | convergence A0->A1 | Tet4 vs Tet10 same mesh (lowfreq) |
|---|---|---|---|---|---|
| G2 | Tet4  | 0.48% PASS | -              | 0.71% PASS | 6.7% (energy p50 +1.7%) |
| G2 | Tet10 | 0.47% PASS | -              | 0.45% PASS | |
| G5 | Tet4  | 1.7% FAIL  | 0.30-0.42%     | 2.9% PASS  | 13.8% (p50 +15.6%, p95 31%) |
| G5 | Tet10 | 1.04% FAIL | 0.05-0.09%     | 2.9% PASS  | |
| G0 | Tet4  | 1.85% FAIL | 0.38-0.65%     | 2.5% PASS  | 17.9% (p50 +18%, p95 35%); on A1 mesh still 14.4% |
| G0 | Tet10 | 1.05% FAIL | 0.05-0.07%     | 3.9% FAIL (A0 softer than A1: p50 -2.0%, p95 4.2%) | |

Thresholds: reproducibility 0.5%, convergence 3%.

## Findings

1. Tet4 is locked by 15-18% (median) on TPMS cells in the assembly-relevant norm, while its own A0->A1 change is only 2.5%:
   level differences under-report the Tet4 error by an order of magnitude. Tet4 labels are not usable.
2. With Tet10 the optimizer policy is no longer a noise source (0.05-0.09%).
3. The remaining 1.0% reproducibility failure is identical across G0/G5 and independent of element and policy:
   it is the non-nested nodal interpolation of fine port nodes onto the carrier (mesh-realization jitter).
   This is the evidence for step 2 (boundary-conforming port meshes, P = identity on port faces).
4. G0 three-level Tet10 study (A0 24.8k / A1 48.9k / A2 100.5k Tet4 nodes; Tet10 A2 = 2.03M dof via Pardiso):
   A0->A1 3.85%, A1->A2 0.82% (lowfreq norm); Richardson: A0 ~4.9%, A1 ~1.0%, A2 ~0.2%.
   Tet4 at A1 is still 13.8% from Tet10 A2 (energy p50 +14%). Full TPMS cells need A1-level Tet10 labels (~1M dof);
   thin/near-empty cells converge at A0. See TET10_CONVERGENCE_A0_A1_A2.json.
5. No-perturb CGAL meshes contain degenerate tets (0 deg dihedral); the perturber is required. Its single boundary pinch
   is absorbed by the bounded topology gate (patch 0001).

## Artifacts

- meshes: tet10_gates/meshes/<case>/<policy_variant>/mesh.mesh (+ A2 for G0)
- dense Schur: tet10_gates/<case>/**/S_{TET4,TET10}_<tag>.npy
- gates: tet10_gates/<case>/TET10_LABEL_GATES.{json,md}
- code: src/.../tet10_label.py, scripts/pred777h_full_cube_v1/run_tet10_label_gates.py, combine_tet10_label_gates.py
- Tet4 baseline archive: baseline_tet4_20260902/
