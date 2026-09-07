# Diagnostics of 2026-09-05 (single-cell label error, null space, sliver artefact)

Evidence scripts behind `STEP8_SLIVER_REPORT.md` and the null-space section of `STEP6_CUT_CARRIER_REPORT.md`.  They read
produced labels under a work directory (`/root/autodl-tmp/_claude_diag/tonight` by default) and are kept for the record,
not as pipeline code.

| script | what it establishes |
|---|---|
| `label_error.py`, `label_error2.py`, `analyse_A.py`, `analyse_A_v2.py` | single-cell error metrics: generalized eigenvalues on the reference range (v1, invalidated by the near-null modes), common support + declared energy floor (v2); the six uniform-strain stiffnesses |
| `analyse_B.py` | production-sample statistics (guards, zero rows, storage) and the external stiffness benchmark |
| `nullspace.py`, `ns_why.py`, `prolong.py` | dim Null(S) = 3 dim null(prolongation) + 3 x components; the extra null modes are pairs of carrier columns whose whole material trace is one shared mesh vertex |
| `decide_tri.py` | exact decision procedure (interval bound on phi, witness search) for "does a carrier patch hold material" |
| `rank_mesh.py` | q is mesh independent, the rank is not |
| `sliver_scan.py`, `top_mode.py` | lambda_max's eigenvector sits on one carrier square in a quarter of the production labels: a sliver, not a port stiffness |
| `sliver_anatomy.py`, `cap_needles.py`, `needle_class.py`, `apex_is_support.py` | every sub-degree tetrahedron stands on a needle cap triangle whose apex is a carrier support point a hair from the chain |
| `grad_bomb.py`, `height_ratio.py` | tetrahedron stiffness-bomb metric area^2 / V and the flatness measure |
| `compare_fixed.py`, `reschur.py` | repaired labels against the old ones and the references |
| `manifold_check.py` | why a CGAL Triangulation_3 cannot hold the material alone (genus-5 boundary) |
| `corner_cut2.py`, `node_kinds.py`, `cutface_width.py` | the cut-face carrier of a vanishing corner cut: the node set saturates at 98 and its two families (grid-line crossings and fan centroids); the population's cut-face widths |
