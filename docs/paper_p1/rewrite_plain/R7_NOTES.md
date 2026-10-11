# R7 notes: section-by-section revision from the author's annotations

## Working rules (author, 11 October 2026)

- The author annotates the Chinese text section by section, starting with the abstract. Each revised passage is shown
  in Chinese and English before it enters the paper.
- During this revision the PDFs, the advisor set and the Word files are not rebuilt after each change. Edit the unit
  files in `sections/`, run `python3 assemble_r5.py --write`, then commit and push at once, so that nothing is lost
  when the container is reset. Rebuild the outputs once, when the author asks.
- The full texts of the PIML papers (Huang et al. 2023, 2024; Jiang et al. 2026; Guo et al. 2026a, 2026b; Zhang et al.
  2026) are kept in the private repository `fanzhirui1004-source/piml-papers`. This repository is public, so they are
  never committed here.
- The authors are not in the group that developed the learned substructure (PIML) methods. The abstract does not
  comment on the limits of those methods: such a sentence would read as provocative. The neutral comparison in
  Introduction ¶3 and Section 1.1 stays as it is.

## Abstract (committed)

Decisions taken from the author's annotations of the Chinese abstract:

1. Open with the need: fine-scale accuracy and a fast analysis at every design iteration. Do not open with cut cells
   carrying supports and loads; in a fine-scale model they are treated like any other cell.
2. No filler such as "且各胞元的几何互不相同"; write 胞元, not 胞元壁.
3. Homogenisation: state the periodic boundary conditions assumed on the unit cell, the loss of accuracy without
   scale separation, and that cut cells are not periodic.
4. Static condensation in one sentence: it reduces the global system without loss of accuracy, but needs one interior
   factorisation per cell geometry, and every design iteration changes every geometry. No pairwise "与 A 相比……；
   与 B 相比……" comparison.
5. NICE in one sentence, without a "但" contrast: what it saves (no interior factorisation) and how it works (all
   degrees of freedom on cell faces and in cut-plane elements retained; network with the cell geometry as input; one
   fixed two-grid cycle; condensed stiffness from the strain energy of the corrected field).
6. Properties in one sentence ("这一构造保证" / "By construction"). The condition behind them (linear in the
   retained displacements, reproduces rigid-body motions) stays in the body; the English abstract has no room for it.
7. Sensitivities: keep "柔度准确并不保证灵敏度准确", led in by "内部位移误差则以一次项进入厚度灵敏度".

English: 250 words. All numbers as before.

## Introduction (proposal, awaiting the author)

The introductions of the PIML papers (Huang et al. 2023, 2024; Jiang et al. 2026; Guo et al. 2026a; Zhang et al. 2026)
give their own results only in the "this paper" paragraph and do not expand the theory there. Proposed changes:

1. Move the comparison in ¶1 (homogenisation 27%, NICE 0.011%) to the end of the NICE paragraph.
2. In ¶2–¶3 (static condensation, learned component models) tighten the wording only.
3. Merge ¶4–¶6 (method, properties, compliance and sensitivity; about 1,300 Chinese characters) into one paragraph of
   about 500 characters, and leave the details to the contribution list and Sections 3–4, which repeat them now.
