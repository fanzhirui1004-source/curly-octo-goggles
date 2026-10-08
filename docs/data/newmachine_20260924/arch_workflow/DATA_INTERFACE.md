# Data interface for the v0 operator network: what the pipeline actually produces, and how the network should consume it

Scope: grounding for architecture questions a, b, c, d, g and k in ARCH_BRIEF.md. Sources: the route-7 pipeline
(`scratchpad/xcase/r7/`: fast_prep3, fast_gp, box_encode, design_encoder, schur_encoder, superset_encoder,
fast_superset, polyref_torch_fast, lattice_pcg, precision_study), `scratchpad/xcase/element_moments.py`,
`element_polyref.py`, the R7_13/R7_14/R7_17 logs in `docs/data/takeover_20260923/`, and the docs the brief lists.

Evidence labels used below:
- **[code]**: read directly in the pipeline source.
- **[log]**: a number from a server log or result JSON in the repository.
- **[recomputed]**: I reimplemented the rule in numpy (`arch/count_topology.py`) and it reproduces the server
  numbers exactly (see §0).
- **[estimate]**: comes from my numpy model: uniform tau, and sampling where a cut plane is involved. Right order
  of magnitude, not exact.
- **[inference]**: follows from the mathematics or the code but was not run.

Not available locally: `stage_cutfem_q2.space.OFFSETS` (the local 27-node order), `stage_cutfem_gp.kernel.stencil` and
`face_factor` (the ghost templates), `fast_topology` internals beyond the file, and any real per-geometry arrays.

---

## 0. Calibration: the topology rules are reproduced exactly

`count_topology.py` rebuilds the following for a FULL cell with uniform tau: active elements, the 27-node element
map, the ghost-face rule, box-port patches and the ghost 45-node stencil. For uniform tau the active and full tests
are exact interval tests, because phi = Σ cos(2πx_d) is separable, so its range on an element is the sum of three
1-D ranges. Comparison with the only two FULL runs whose logs record totals (R7_17, `SCHUR_SETUP` events):

| FULL cell, uniform tau | DOF (log) | DOF (mine) | box DOF (log / mine) | nnz upper of K (log) | nnz upper (mine) |
|---|---|---|---|---|---|
| tau = 2/3 | 439,092 | 439,092 | 29,376 / 29,376 | 92,149,416 | 92,149,416 |
| tau = 1/4 | 255,108 | 255,108 | 15,264 / 15,264 | 70,523,376 | 70,523,376 |

The nnz match is exact. It holds only if all of the following are right: the active-element rule, the "not both
interval-full" ghost-face rule, the 45-node ghost stencil (5 node layers along the face axis × 3 × 3) and the
unique node-pair structure. The graph counts below therefore rest on verified rules. Uniform tau = 0.5 gives box
DOF 23,616, **identical** to the real FULL parent of 0013, and 363k DOF against the real 368k (whose tau is
non-uniform). I use tau = 0.5 as the stand-in for "the FULL parent" throughout.

---

## 1. Arrays that exist per geometry (as produced by fast_prep3 and consumed by the GPU encoders)

### 1.1 Grid and index conventions [code]
- Background: n = 32, so 32³ hex elements with h = 1/32. Q2 node lattice (2n+1)³ = 65³ = 274,625 nodes at x = g/64,
  with g ∈ {0..64}³.
- **Global node id** = `np.ravel_multi_index((gx, gy, gz), (65,65,65))` = (gx·65 + gy)·65 + gz. This is C order
  with z fastest (fast_prep3 l.56, box_encode.ghost_faces_gpu).
- **Element (cell) (i,j,k)** covers node indices 2i..2i+2 in each axis. Its 27 nodes are `2*cell + OFFSETS`, with
  OFFSETS ∈ {0,1,2}³ in a fixed order that is not available locally. `element_moments.local_coordinates` checks that
  every element uses one and the same local order, and the encoders raise `ONE_NODE_ORDERING_EXPECTED` otherwise.
  Local coordinates are ξ = g − 2·cell − 1 ∈ {−1, 0, 1}³.
- Node types by index parity (tau = 0.5 FULL) [recomputed]: 17,436 vertices, 47,640 edge-midpoints, 43,080
  face-centres, 12,880 element centres. Centres equal the element count, since each element owns exactly one.

### 1.2 Files per geometry (`<body_dir>/<case>/`) [code]

| file | shape, dtype | meaning |
|---|---|---|
| `CELL_INDICES.npy` | (E, 3) int32 | ijk of the active elements, in `fast_topology` order. The local element index is the row. |
| `NODES.npy` | (N,) int64, sorted | global ids of the active Q2 nodes: the union of the 27 nodes of each active element. The local node index is the position in this array. |
| `dofs.npy` | (E, 81) | 3·local_node + component. Node-major, xyz interleaved. |
| `GP_FACES.npy` | (F, 3) int32 | (owner, neighbour, axis). Owner and neighbour are local element indices; neighbour = owner + e_axis. |
| `BOX_NODES.npy` | (Nb,) int64, sorted | global ids of the box-port nodes |
| `GP_TEMPLATES_n32.npz` (one per n) | canonical (3, 54, 135), offsets (3, 45, 3) | ghost factor per face axis, and the 45-node stencil offsets from 2·owner_cell |
| `PREP.json` | | timings, counts, and equality checks against the frozen products |

- The GPU encoder adds the **element moments** `M` (E, 125) fp64 from `polyref_torch_fast.cell_moments`:
  M_e,abc = ∫_{Ω_e ∩ material} ξ^a η^b ζ^c dx, with a, b, c ≤ 4, index a·25 + b·5 + c, ξ local in [−1, 1] and dx
  the physical measure, so M_e,000·n³ is the volume fraction. The material is the shell |phi| ≤ tau(x) intersected
  with the half-space offset − n·x ≥ 0. tau is trilinear from 8 corners in lexicographic order (index 4x + 2y + z).
  The integral uses piecewise-linear level sets on 6 Kuhn tetrahedra per sub-cube, s = 4 with one octree level
  (effective resolution 8). The plane is cut exactly and the curved sheet approximately, with element error
  ~1e-4 against the packet.
- No trace coordinates (P) and no dense operators are stored. The operator the lattice needs is T = K_bb −
  K_bi K_ii⁻¹ K_ib on the box nodal DOF (box_encode docstring). It matches the packet's S condensed to box
  coordinates to 9e-16.

### 1.3 How the box port is selected [code, recomputed]
The box port consists of the **3×3 Q2 face nodes of every certified box-face patch with positive-area material**
(`topo['patches']` with tag `BOX_{X,Y,Z}{MIN,MAX}`), united over all patches. A node that lies on the box plane but
touches no positive-area material patch is **not** a port node; it is a free interior unknown. This is the frozen
trace-compiler convention: 0021 would have 156 extra nodes under "all box-plane nodes" (ROUTES_PROGRESS 7.5). For
uniform tau the patch test is the 2-D interval test of 1 + cos(2πy) + cos(2πz) against [−tau, tau]. It reproduces
the logged box counts exactly (§0).

Lattice consequence [inference]: on a shared face, phi is periodic, and tau is bilinear from the 4 shared corner
values. When adjacent cells share their corner tau values, which is the natural choice for a thickness field on
lattice vertices, their box-port node sets on that face coincide exactly. A cut face gives a subset.
`lattice_pcg` glues by global grid key with offset 2n per cell.

### 1.4 How the element stiffness is formed [code, recomputed]
- `pattern_operators(ξ_nodes, λ, μ, n)` returns T (125, 81, 81) with K_e = Σ_m M_e,m T_m, built exactly from Q2
  shape-function polynomial products. Isotropic C, E = 1, ν = 0.3, with a derivative scale of (2n)² folded in.
  Encoders keep the upper triangle `Tm_up` (125 × 3321) and compute `ke = M @ Tm_up` in chunks of 4096 elements,
  followed by one `index_add_` into the sorted upper-pattern value vector (`pos_e`).
- Template facts [recomputed with the real `pattern_operators`]:
  - All 3321 upper entries are generically nonzero, so K_e is dense 81 × 81.
  - T has 256k nonzeros in total, about 2,049 per moment.
  - **rank(T as 125 × 6561) = 121.** The 4 moments with a + b + c ∈ {11, 12}, namely (4,4,3) in 3 permutations and
    (4,4,4), never enter K, because the integrand ∂N_i ∂N_j has total degree ≤ 10. The geometry encoder can drop
    them.
- **Exact signed-quadrature form [recomputed]:** with the 5×5×5 Gauss–Legendre points x_q, weights
  w_e = V⁻¹ M_e, where V_mq = x_q^m and cond(V) = 1.2e4, reproduce all 125 moments exactly. Hence
  K_e = Σ_q w_e,q B_qᵀ C B_q exactly (checked: 1.4e-15). On sampled cut-cell moments, Σ_q |w_e,q| / vol_e ≤ 1.77,
  and 60% of elements have some negative weight. The representation is barely signed, which matches the negative
  NNLS finding. This gives a **strain-form** K apply and energy with only 125 numbers per element (§4).
- Rigid kernel: K_e R_e = 0 exactly in exact arithmetic (by construction); in floating point only to rounding.

### 1.5 How ghost-penalty faces enter K [code, recomputed]
- A face is a ghost face if both adjacent elements are active and **not both** are interval-certified full.
  "Full" is tested with the same closed-form interval test as activity.
- K_ghost = Σ_f P_fᵀ (F_axisᵀ F_axis) P_f, with F_axis ∈ R^{54×135} (NIGHT_REPORT §9). P_f gathers the 45 nodes
  2·owner_cell + offsets[axis]: 5 node layers along the axis, 3 × 3 across. K = K_body + γ K_ghost with γ = 1e-4
  from SAMPLE.json.
- **K_ghost depends on the face list only**, not on the tau values or the moments. Within one topology,
  ∂K_ghost/∂τ = 0 [inference from the code: base values = γ·template]. Across topology changes the face list moves.
- Ghost faces couple node pairs that share no element: the two outer node layers, 4 apart along the axis. This
  **multiplies the pair count by 2.4–3.4** (§2.2).
- 54 = 9 face points × 3 components × 2 derivative orders is my guess for the Q2 normal-derivative jumps
  [inference, template not local]. Linear fields have no jumps, so F·R = 0 [inference].

### 1.6 The design-loop superset [code]
The material grows monotonically in every corner tau. So `fast_superset` builds one pattern (cells ⊆
active(τ_c(1+δ)), faces = adjacent pairs not full-full at τ_c(1−δ)) that stays valid for all designs within ±δ.
`superset_encoder` then:
- scatters each design's own elements,
- subtracts the superset faces the design does not have,
- sets unit diagonals on superset nodes outside every active element.

Result: 1e-14 against a fresh encode, with the plan reused. The **network interface can reuse this idea**: build a
fixed superset graph per design band, and represent topology changes as masks (element-on, face-on, node-on) over
it. The sensitivity is then taken within the current masks.

---

## 2. Proposed network-facing interface

### 2.1 Sizes [recomputed for FULL; estimate for cut]

| quantity | FULL tau=2/3 | FULL tau=0.5 (≈ parent of 0013) | FULL tau=1/4 | cut, tau=0.5, retained 0.148 | cut 0.5 | cut 0.758 |
|---|---|---|---|---|---|---|
| active elements E | 16,016 | 12,880 | 8,464 | 1.7–2.6k | 6.4–6.7k | 9.6–10.1k |
| interval-full elements | 8,848 | 5,520 | 864 | ~45% | ~43% | ~43% |
| active nodes N | 146,364 | 121,036 (real 122,748) | 85,036 | 17–25k (0013 real: 22,585) | 61–65k | 91–96k |
| nodes per element | 9.1 | 9.4 | 10.0 | 9.7–10.0 | 9.5–9.6 | 9.5 |
| box-port nodes / DOF | 9,792 / 29,376 | 7,872 / 23,616 | 5,088 / 15,264 | 1.3–1.9k / 3.8–5.6k | 3.8–4.0k / 11.5–12k | 5.4–6.2k / 16–18k |
| ghost faces F | 21,444 | 21,936 | 20,400 | 2.6–4.0k | 10.9–11.2k | 16.3–17.1k |
| active nodes outside the material | — | 35.5% | 54.5% | 43.8% | — | — |

- Real cases [log]:
  - 0013: 62,961 interior + 4,794 box DOF.
  - 0021: 197,658 + 12,363.
  - FULL parent: 344,628 + 23,616.
- Volume fractions (sampled, FULL tau = 0.5): 10% quantile 0.056, median 0.46, and 144 active elements have no
  sample point inside. The archived exact moments of an older case had a 1% quantile of 9e-6 and a minimum of
  1.8e-9 (ELEMENT_MOMENTS_01) [log]. Tiny-support elements are normal, not exceptional.
- **35–55% of active nodes lie outside the material** [estimate]. They are fictitious-domain DOFs whose values are
  fixed only by tiny body stiffness and γ-weighted ghost terms. The network must output them anyway, because
  every element's energy uses all 27 nodes.

### 2.2 Graph options, counted [recomputed]
A Q2 hex couples all 27 of its nodes: C(27, 2) = **351 unique pairs per element**, and K_e is dense 81 × 81.

| graph | FULL tau=2/3 | FULL tau=0.5 | FULL tau=1/4 |
|---|---|---|---|
| element hyperedges (27 incidences each), total incidences | 432k | 348k | 229k |
| unique node pairs from elements | 4.17M | 3.38M | 2.26M |
| unique node pairs, elements + ghost faces (= K's graph) | 10.14M | 9.48M | 7.78M |
| ghost-face hyperedges (45 incidences each), total incidences | 965k | 987k | 918k |
| node degree, element graph (mean / max) | 57 / 124 | 56 / 124 | 53 / 124 |
| node degree, K graph (mean) | 139 | 157 | 183 |

**Recommendation (a): use hyperedges, not a pairwise graph.**
- K is assembled as Σ_e P_eᵀ K_e P_e + γ Σ_f P_fᵀ G_axis P_f. A layer of the form gather(27 or 45 nodes) →
  per-element or per-face linear map → scatter-add has **exactly K's sparsity**.
  - It moves 348k + 987k incidences per channel. A pairwise graph with K's connectivity would move 9.5M edges, 7×
    more, plus per-edge coefficient storage.
  - It needs no edge list: `dofs` or the element-node map (E × 27 int32 = 1.4 MB) and the face stencil are the
    whole topology.
- It also answers "adjacent but disconnected walls must not talk" in the only sense CutFEM defines. Two nodes
  interact **only through a shared active element or a ghost face**; that is what K does. Pieces that share an
  element are coupled in K too, so the network need not be stricter than K.
- The graph is a subgraph of K's by construction. Any coefficient the geometry path generates sits on an
  element or a face, so it can be initialised from, or compared with, the true K_e.

**Equivalent dense-tensor layout [inference]:**
- Pack the 65³ Q2 grid as 33³ slots × 8 parity sub-lattices: node (2i+a, 2j+b, 2k+c) → slot (i,j,k), channel
  (a,b,c), with a pad at index 32. That is 287,496 slots for 274,625 nodes, 4.7% padding.
- A Q2 element's 27 nodes then fall in a 2×2×2 slot window. Any node's full element stencil (offsets −2..2) falls
  in a 3×3×3 slot window.
- Hyperedge gather and scatter over the dense 32³ element grid become fixed-offset slicing (no index gathers).
  The cost is 32,768 elements instead of 12.9k active, 2.5× the work, with fully regular memory access.
- Dense also suits the coarse levels. At 17³ and 9³ the active fraction is 66–86%, and dense 3-D convolution
  and FFT apply directly there.

### 2.3 Port and mask definitions
- **Box port** (existing, exact): `BOX_NODES`. Per node, a 6-bit face mask. Edge and corner nodes belong to 2–3
  faces.
- **Cut-band port.** The candidate definition, grounded in what a cut-surface traction does: for a traction t on
  Γ_cut ∩ material, the nodal force is f_i = ∫_{Γ∩mat} N_i t dA. Restricted to an oblique plane, every Q2 shape
  function of an element crossed by Γ ∩ mat is generically nonzero. So the smallest node set that carries every
  cut load is
  **Γ-band = all 27 nodes of the active elements whose plane section contains material (positive area).**
  - Alternatives: "straddle band" = all nodes of active elements crossed by the plane, 6–16% larger; and the Q1
    vertices of the Γ-band elements.
  - Measured sizes [estimate: tau = 0.5, 3 normals, retained 0.148 / 0.5 / 0.758]:

    | | Γ-band elements | Γ-band nodes | Γ-band DOF | box DOF of the same cell |
    |---|---|---|---|---|
    | retained 0.148 | 364–498 | 4.9–6.9k | **14.6–20.6k** | 3.8–5.6k |
    | retained 0.5 | 288–785 | 4.0–10.3k | 12.0–30.9k | 11.5–12.0k |
    | retained 0.758 | 450–464 | 6.1–6.3k | 18.4–18.6k | 16.2–18.5k |

  - About 13 unique nodes per band element, because the band is a 2-D layer.
  - 23–58% of Γ-band nodes lie outside the material. 22–28% of them are weak (proxy, §2.4). In the heavy cut
    (0.148), 35% of all weak nodes are Γ-band nodes.
  - Overlap with the box port: 0–484 nodes. A precedence rule is needed; I suggest a node that is on both belongs
    to the box port, and the cut-band mask excludes it.
  - Q1 vertices of the Γ-band: 1,278 nodes for the 0.148 case, 5.4× fewer than the Q2 Γ-band.

  **Consequence for the architecture:**
  - With a Dirichlet-type cut port at full Q2 resolution, a heavy cut cell has a port **3–4× larger than its box
    port**, 30% of all its DOFs.
  - In the lattice, a free cut surface then means these DOFs become cell-local interface unknowns: not shared, zero
    external force. They add up to ~31k unknowns per cut cell to the BDD system and to every local Neumann solve.
  - As a benefit, pinning the band removes many weak sliver nodes from the network's interior problem.
  - The alternative is a force-type (Neumann) input on the same mask: f_cut enters linearly, and a free cut is simply
    f_cut = 0 at no extra cost. It keeps decision 4 (fixed grid, masks, nodal forces) but changes the port type.
    This choice should be made explicitly in step 0. Both are supported by the same arrays.
  - Surface moments of Γ ∩ material, needed to build f from a traction field, are **not computed today**.
    polyref clips the plane exactly, so the same clipping gives them cheaply [inference].
- **Masks to carry as inputs** (all cheap):
  - element-on, element-full, element-cut (straddles the plane), element-in-Γ-band;
  - face-on (ghost);
  - node-on, box (6 bits), Γ-band, node-inside-material;
  - weak indicator (§2.4);
  - superset masks for design bands (§1.6).

### 2.4 Cheap features

Per node, all O(N) or O(E × 27):
- grid index and parity type (4 classes), and x;
- phi(x), tau(x), |phi| − tau, and the signed plane distance offset − n·x, all scaled by h;
- **diagonal 3×3 block of K**: D_i = Σ_e (K_e)_ii + γ Σ_f (G)_ii. The body part comes straight from moments through
  a fixed 125 × 27 × 3 × 3 slice of T; the ghost part is a constant per face axis.
- the weak indicator: log₁₀(‖D_i‖ / median);
- lumped material mass m_i = Σ_e ∫ N_i: linear in the 27 moments with per-axis degree ≤ 2;
- the number of active incident elements (1–8 for a vertex);
- port masks.
- Weak-node proxy [estimate; body-only D_i with sampled moments; ghost not included because its template is not
  local]:
  - ‖D_i‖ < 1% of median in **16.7% (FULL tau 0.5), 22.2% (FULL tau 1/4) and 17.5–18.8% (cut cells)** of nodes.
  - These fractions are close to the 17.6–21.9% measured with the full K in cut cells. Weak nodes are therefore
    **also abundant in FULL cells**: they come from the curved-shell fringe (fictitious nodes), not only from the
    plane. This should be confirmed on the server with the true K diagonal.

Per element:
- the 125 moments, 121 of them effective, normalised by n³ so that M₀₀₀·n³ is the volume fraction and the
  others are O(1);
- equivalently the 125 signed Gauss weights w_e, or the normalised centroid and second moments;
- full, cut and Γ-band flags; tau at the element centre;
- optionally trace(K_e) or a few Rayleigh quotients of K_e on fixed Q2 modes, e.g. the 6 uniform strains:
  ε_kᵀ K_e ε_k = Σ_m M_m (ε_kᵀ T_m ε_k), linear in the moments and cheap.

Per ghost face: axis, the full flags of the two sides, and the volume fractions of the two sides.

Scaling and symmetry [inference]:
- Moments transform under the 48-element cube group by a signed permutation (ξ → −ξ multiplies M_abc by
  (−1)^a; axis permutations permute a, b, c), so group augmentation needs no re-integration.
- Caveat: the Kuhn split in polyref is not symmetric under all 48 elements, so re-integrated moments of a rotated
  geometry differ from permuted moments at the polyhedral-error level (~1e-4). The brief reports 1e-7
  equivariance for the pipeline.

### 2.5 Coarse levels aligned with the Q2 vertices [recomputed counts, inference for operators]

| level | meaning | active vertices FULL 2/3 / 0.5 / 1/4 | cut 0.148 / 0.5 / 0.758 |
|---|---|---|---|
| 65³ | Q2 nodes, h/2 | 146k / 121k / 85k of 274,625 | 17–25k / 61–65k / 91–96k |
| 33³ | Q1 vertices of the same 32³ mesh (even Q2 indices) | 20.6k / 17.4k / 12.9k of 35,937 | 2.6–3.7k / 8.9–9.5k / 13.3–14.0k |
| 17³ | Q1 on 16³ macro-elements | 3.5k / 3.2k / 2.6k of 4,913 | 0.6–0.76k / 1.7–1.9k / 2.5–2.7k |
| 9³ | Q1 on 8³ | 624 / 624 / 576 of 729 | 170–186 / 346–429 / 510–558 |

- The spaces are **nested finite-element spaces**: Q1(32³) ⊂ Q2(32³) is p-coarsening, and Q1(16³) ⊂ Q1(32³) is
  h-coarsening.
  - Prolongations are fixed interpolation stencils. From 33 to 65 they are weights 1, 1/2, 1/4, 1/8 for vertex,
    edge, face and centre nodes; from 17 to 33 and from 9 to 17 they are trilinear.
  - Galerkin coarse operators Pᵀ K P and rigid modes (linear fields) are therefore represented exactly on every
    level.
- Caveat: a coarse vertex's support can span material pieces that are disconnected at the fine level, a
  geometry-agnostic coarsening problem. E2 found that the ghost faces link sliver pieces anyway.
- At 17³ and 9³ a FULL cell is 66–86% active: treat those levels as dense grids with masks, with spectral or FFT
  mixing where wanted.

---

## 3. Memory in fp32 (FULL tau = 0.5: N = 121k, E = 12.9k, F = 21.9k; tau = 2/3 is about 20% larger)

| object | size per FULL cell | × 100 cells | verdict for 100 resident cells |
|---|---|---|---|
| element-node map E × 27 int32, plus face list F × 3 | 1.4 + 0.26 MB | 0.17 GB | yes |
| moments E × 125, fp32 / fp64 | 6.4 / 12.9 MB | 0.6 / 1.3 GB | yes: the canonical stored geometry |
| dM/dτ E × 125 × 8, fp32 | 51.5 MB | 5.2 GB | only if sensitivities are needed for all cells at once; else recompute |
| node features, C = 32 / 64 / 128 channels | 15.5 / 31 / 62 MB | 1.6 / 3.1 / 6.2 GB | yes at C ≤ 64 |
| per-element learned operator 27 × 27 (one scalar per node pair) | 37.6 MB | 3.8 GB | yes |
| per-element C × C mixing, C = 64 | 211 MB | 21 GB | no, use low rank or sharing |
| K_e upper 3321 floats / full 81 × 81 | 171 / 338 MB | 17 / 34 GB | no; rebuild from moments on the fly (§4) |
| pair edge list of K's graph, 9.48M × 2 int32 (directed ×2) | 76 (152) MB | 7.6 (15) GB | avoid; hyperedges need none |
| 3×3 K blocks per pair, element graph / with ghost | 122 / 341 MB | 12 / 34 GB | no |
| CSR of K upper: 86M values + int32 columns | 690 MB | 69 GB | no |
| exact fp32 cuDSS factor [log] | 6.6 GB | — | 2 cells max |
| dense T, fp64 [log] | 4.5 GB | — | — |
| field activations per query column, one Q2 vector field × C_u channels: N × 3 × C_u | 1.45 MB × C_u (C_u = 16: 23 MB) | batched k columns × layers kept for backward in training | the real memory driver when training |

Takeaways:
- **Store per cell only topology, moments (fp64 or fp32) and learned coefficients living on nodes and elements with
  O(10²) numbers each.** This is ≤ 50–100 MB per FULL cell, 5–10 GB for 100 cells, and leaves ~20 GB for batched
  query activations.
- Anything stored per node pair, or as a dense element matrix per element, does not fit 100 cells.

---

## 4. Cost of the exact operations the network may call (FULL tau = 0.5, per query column)

| operation | FLOP | stored input | notes |
|---|---|---|---|
| K_body u with precomputed K_e (batched 81 × 81 GEMV) | 2·81²·E = **0.17 GFLOP** | K_e 338 MB, rebuilt per batch | bandwidth-bound per column; amortised over k columns |
| form K_e from moments (M @ Tm_up) | E·125·3321·2 = **10.7 GFLOP** once per batch | moments | fp32 ~0.1–0.3 ms; fp64 on the 5090 (1/64 rate) ~7–10 ms |
| K_body u in strain form (sum-factorised Q2 gradient at 5³ Gauss points, weights w_e, transpose) | ≈ 29k·E ≈ **0.37 GFLOP**; energy only ≈ 0.2 GFLOP | w_e, 125 per element | no K_e storage; rigid-exact energy (strains of rigid fields vanish) |
| K_ghost u, factor form (gather 135 → 54 jumps → back), 3 shared templates | 4·54·135·F = **0.64 GFLOP** | face list + 3 templates | **dominates** body cost 2–4×; a shared-weight dense GEMM per axis (tensor-core friendly) |
| CSR SpMV (whole K, symmetric) | ~0.34 GFLOP, ~1.4 GB traffic | 690 MB CSR | worst for 100 cells |
| exact energy qᵀŜq = ûᵀKû | one K apply + dot ≈ 0.8 GFLOP | — | fp64 ≈ 0.5 ms/column; fp32 memory-bound µs |
| symmetric reaction Ŝq = Êᵀ K Ê q | 1 forward + 1 K apply + 1 VJP of the linear net | — | about 2× network cost + 0.8 GFLOP |
| sensitivity −uᵀ(∂K/∂τ_c)u, c = 1..8 | per-element energy densities a_e,m = u_eᵀ T_m u_e (or per Gauss point in strain form) ≈ 0.2 GFLOP, then Σ a·∂M/∂τ = 26 MFLOP | ∂M/∂τ (E × 125 × 8) | ghost term has zero τ-derivative within a topology. ∂M/∂τ comes from autograd or forward-mode through polyref (the clipping is piecewise smooth in τ): about one extra moment pass (0.25–1 s per cell), or store it (51 MB) |

Precision [inference]:
- An fp32 u_eᵀ K_e u_e loses accuracy when u_e is nearly rigid on the element, which is exactly the soft-mode
  regime: the error is ≈ ε₃₂·‖K_e‖‖u_e‖².
- The strain form (or subtracting the element rigid part before contracting) makes the rounding relative to the
  strain, not to the displacement.
- Recommended readout: fp32 gather and strain, fp64 accumulation. fp64 end-to-end is also affordable: ~0.5 ms per
  column per FULL cell, about 10 s per design iteration at 2e4 applications.

Budget check against brief §6 [inference]:
- At ~2e4 applications per design iteration and a target of a few ms each, one FULL cell application can afford
  roughly 20–40 fine-level hyperedge layers at C ≈ 32–64. Each layer moves ≈ 2·(348k + 987k)·C·4 B ≈ 0.3–0.7 GB,
  which is memory-bound at ~0.2–0.4 ms.
- Plus a few exact K applies (0.8 GFLOP each).
- Coarse levels (≤ 20k vertices) are negligible.
- This favours putting the fine-level work into element and face hyperedge operators with shared or low-rank
  coefficients, and the global transmission into the 33/17/9 levels.

---

## 5. What was verified vs inferred (summary)
- **Verified in code:** array names, shapes and ordering; box-port rule; element-stiffness formation (moments ×
  template, upper scatter); ghost assembly (face rule, 45-node stencil, fixed templates, γ); superset masking;
  Schur/T semantics; rigid-mode construction; lattice gluing by grid key.
- **Verified by exact reproduction:** FULL DOF, box DOF and nnz for tau = 2/3 and 1/4; template rank 121; exactness
  of the signed 5³ Gauss representation.
- **Estimates** (uniform tau, and sampling for the plane): all cut-cell counts, Γ-band sizes, weak-node fractions
  (body-only proxy), outside-material fractions, volume-fraction quantiles.
- **Inferred:** GP template internals (54 = jumps), ∂K_ghost/∂τ = 0, moment symmetry transforms, dense packing,
  nested coarse prolongations, FLOP and time figures.

Scripts and outputs are in `arch/`:
- `count_topology.py` with `counts_full.json`;
- `count_cut.py` with `counts_cut.json`;
- `weak_proxy.py` with `weak_proxy.json`;
- `gauss_weights.py`;
- `outside.py`.
