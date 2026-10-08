# C4 cluster (hybrid learned solvers) - screening notes, 2026-09-26
Closest: GMT (Xing, Liu, Xue, Lu; ACM TOG 2026, SIGGRAPH; doi 10.1145/3811333; arXiv 2604.26518)
  - NN (Point Transformer V3 on sparse GMG hierarchy) predicts fine-level u + per-level corrections -> injected into a full
    GMG V-cycle (8-colour Gauss-Seidel, Galerkin coarse ops K_{l+1}=R K_l P) -> differentiated through, end-to-end
    residual/log-residual loss. TPMS, PSL, truss lattices; strict periodic BCs (homogenization cell problem).
    Mentions spectral bias; "thin struts, high-contrast ... challenging". No condensation / Schur complement.
CGiNS (Xing, Liu, Chen, Tang, Lu; arXiv 2506.17087, 2025): PCG embedded, energy loss, TPMS/PSL/truss, periodic.
HINTS (Zhang..Karniadakis NMI 2024 10.1038/s42256-024-00910-x): DeepONet offline, alternating w/ Jacobi/GS; HINTS-MG;
  Poisson/Helmholtz only.
DeepONet TB precond (Kopanicakova & Karniadakis SISC 2025 10.1137/24M162861X): trunk fns -> Galerkin A_c=RAP + smoother
  (E=(I-M1 A)(I-CA)); diffusion, jumping coeff 1..1e5, Helmholtz. No elasticity.
Nikolopoulos et al IJNME 2024 10.1002/nme.7372: FFNN+CAE initial guess -> POD-2G two-grid refinement (separate training).
Classical analogs for C2/C4: smoothed aggregation (tentative P smoothed by Jacobi), energy-min AMG interpolation
  (Olson-Schroder-Tuminaro SISC 2011 10.1137/100803031), inexact BDDC harmonic extensions (Dohrmann NLA 2007
  10.1002/nla.514; Li-Widlund CMAME 2007).
C1 flags (other cluster): Beatson et al NeurIPS 2020 composable energy surrogates; Kroepfl-Maier-Peterseim 2022
  operator compression 10.1186/s13662-022-03702-y; Bonilla Moreno-Guarino-Antolin CMAME 463 (2027) arXiv 2604.09113
  (non-NN ROM-BDDC for non-periodic level-set lattices); EIFEM preconditioner (Rubio-Ferrer-Hernandez CMAME 2025).
