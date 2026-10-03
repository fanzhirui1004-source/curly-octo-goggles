# Notes: CutFEM/unfitted+ML cluster and patents (screening agent, 2026-09-26)
Raw patent HTML: pat/*.html, CN1192*/CN1196*/CN1199*.html; claims text *_claims.txt.
Key finds:
- CN122046816A (NWPU, filed 2026-02-05, pub 2026-05-15, PENDING): DeepONet(material modulus/density field + node coords) -> EMsFEM shape functions of oversampled substructure;
  K_c = assembled with predicted shape functions + static condensation; label-free minimum-potential-energy loss with random Gaussian boundary displacements
  on random filtered density fields; 2D examples (cantilever, L-beam, MBB TO 8M elements). Closest prior art to C2 construction+loss. Claim 1 = 4 steps
  (partition->NO condensed K->global coarse solve->NO shape-function recovery to fine grid w/ iterations).
- arXiv 2608.02036 (Jiang, Zhan, Zhang, Wang, Aug 2026) Convex Neural Energy Elements: hypernetwork -> PSD condensed boundary operator S(g) of geometry-param. tiles,
  supervised on exact Schur complements; 2D heat/plane-strain with holes, 3D heat; assembled with unseen neighbours. Concurrent to C1 generic.
- arXiv 2604.09113 / CMA 2026 119304 (Bonilla Moreno, Guarino, Antolin): ROM(MDEIM)-BDDC for unfitted p-FEM level-set 2D lattices incl. Schwarz-P; no homogenisation;
  per-cell Schur; future work = NN surrogates in 3D, ROM subdomain matrices as preconditioner.
- CN115408914B (DUT Ningbo, granted 2023): 2D PIML TO; FNN density->EMsFEM shape functions; supervised MSE loss on N and K; claim limited to 2D cantilever/MBB.
- CN121389733A (BUAA, pending): CNN substructure moduli -> numerical shape functions, EMsFEM, level-set robust TO.
- CN119249797B / CN119623208A / CN119918215A (ZJU Li Ming): no ML. Voronoi TO w/ CBN shape functions; chip thermal embedded coarse elements w/ reduced CBN
  + exact boundary->interior map; B-rep cut-cell template integration. No risk.
