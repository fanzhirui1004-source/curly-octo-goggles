# Cluster: ML for TPMS/lattice mechanics + graded-lattice TO (screening notes, 2026-09-26)
Local PDFs Xu2025-PIML-lattice-MMC.pdf and Zhang2024-isoparametric-PIML.pdf are 14-byte "404: Not Found" stubs;
assessed from publisher intro/section text previously cached in ../jina_S0263822325004957.txt and ../j_7faeb1.txt.
Key verdicts:
- HIGH (for C2 as worded): Huang et al. JMPS 193 (2024) 105893 data-free PIML -- DeepONet extension u=G(x;rho)u_v from 24 corner DOFs,
  condensed K = N^T K N with exact fine K, min-potential-energy label-free loss. => C2 construction+loss not new per se;
  novelty must be: full boundary trace (exact Schur complement, any boundary displacement), one-sided bound/error identity, cut cells.
- MEDIUM: PIML family (EML 2022/2023, EML 2024 isoparam., CMA 2026 Bezier, CompStruct 2025 lattice-MMC, CompStruct 2026 O_h equivariant,
  PIML-OFEM arXiv 2607.22019), Beatson et al. NeurIPS 2020, De Weer et al. CompMech 2022, Wu-Xia-Wang-Shi CMAME 2019 (ARSP),
  CGiNS arXiv 2506.17087 (sparse multilevel U-Net + PCG-in-the-loop + energy loss, periodic homogenisation).
- LOW: homogenisation surrogates (Shojaee 2023 Schwarz-P PANN; Stollberg 2025; DANN AddMa 2023; Chen JMRT 2024), LatticeGraphNet,
  Tozoni cut-cell CGF 2024 (non-ML), CBN (Li&Hu CMAME 2022, non-ML), ZJU patents (no NN), size-effect 2nd-order homog. (EngComp 2026).
