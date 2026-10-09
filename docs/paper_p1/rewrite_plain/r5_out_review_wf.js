export const meta = {
  name: 'p1-r5-final-review',
  description: 'Five-lens review of the assembled R5 manuscript (EN/CN main, appendices, supplement) with per-lens adversarial verification',
  phases: [{ title: 'Review', detail: 'five independent lenses, read-only' }, { title: 'Verify', detail: 'adversarial check of each lens findings' }],
}

const P = '/home/user/curly-octo-goggles/docs/paper_p1'
const OLD = '52136f8~1'
const COMMON = `
Files (current, assembled; read-only for you, do NOT edit any file):
  ${P}/MANUSCRIPT_EN.md, ${P}/MANUSCRIPT_CN.md (main text; CN is what the advisor reads)
  ${P}/APPENDICES_EN.md, ${P}/APPENDICES_CN.md
  ${P}/SUPPLEMENTARY_EN.md, ${P}/SUPPLEMENTARY_CN.md
The previous (pre-rewrite) versions are in git: run e.g. \`git -C /home/user/curly-octo-goggles show ${OLD}:docs/paper_p1/MANUSCRIPT_EN.md\`.
The binding style/terminology brief is ${P}/rewrite_plain/R5_BRIEF.md (read sections 1-5 first). Section 5 of the brief gives the old->new numbering maps.
Evidence for numbers: ${P}/evidence/ and ${P}/review_r1/results/ (JSON/MD), plus the supplementary tables.
Context: this paper was rewritten because the advisor (a leading expert in structural mechanics / FE) found the old text unreadable, full of invented jargon, and AI-sounding. The rewrite must use field-standard terms, formal academic register (not chatty, no slogans, no parallel triads, no aphorisms), one claim per sentence, with every number and condition of the source preserved. Main text keeps only three variants: Base network / Base network + correction / NICE (CN 基础网络 / 基础网络加修正 / NICE); the other two continuations live only in Supplementary Note S8 (Tables ST24, ST25). Figure 16 (scale plates) is new; Figure 15 now has only panels (a,b).
Hard constraints: the paper never mentions repeated measurements, machine load, throttling or shared hosts; no 'first' claims; no PIML numerical comparisons; no GPU/dataset/offline training cost discussion; abstract <= 250 words.

Report only real problems that a careful author would fix before sending to the advisor. For each finding give: file, an EXACT verbatim quote (8-40 words, copied character for character, enough to locate it uniquely), the problem, and a concrete fix as exact replacement text (for EN/CN pairs give the fix for each file as separate findings). Do not report matters of taste. Do not report anything listed in 'already decided' below.
Already decided (do not report): NICE is not expanded in the abstract; the abstract wording 'not small relative to it'; the one-sentence two-grid explanation in introduction paragraph 4; Schur complement mentioned in both 1.1 and 2.2; Vanek et al. cited in both 1.1 and 3.3; CN writes large DOF counts in 万; 'retained displacement vectors' in 3.2; the 'Weak' label of Figure 3; ratio range 71 to 117 (computed from ST03b).`

const LENSES = [
  { key: 'readability', prompt: `LENS: readability for the advisor (Chinese main text first, then English main text). Read MANUSCRIPT_CN.md completely as a senior professor of computational solid mechanics would. Find: (1) terms used before they are defined or never defined in plain words; (2) invented or non-standard jargon, or literal translations from English that a Chinese mechanics professor would not use (prefer standard 中文力学/有限元术语); (3) sentences that are hard to parse, overly long, or chain several claims; (4) AI-sounding patterns (排比, 口号, 格言式结尾, '不仅……而且……' slogans, rhetorical contrasts, 'X 而非 Y' slogans), chatty register; (5) paragraphs whose logic jumps. Then check the same places in MANUSCRIPT_EN.md and report the EN counterpart fix when the EN has the same problem. Limit to the 40 most important findings.` },
  { key: 'alignment', prompt: `LENS: EN/CN alignment of the main text and of Supplementary Note S8. Compare MANUSCRIPT_EN.md and MANUSCRIPT_CN.md paragraph by paragraph (and SUPPLEMENTARY_EN/CN Note S8). Find: content present in one language but missing or different in the other (claims, conditions, numbers, references, citations, units); mistranslations that change meaning; captions that differ. Numbers written as 万 in CN for EN millions are fine. Report each mismatch with fixes for the wrong side.` },
  { key: 'crossrefs', prompt: `LENS: cross-references and numbering across all six files. Check every reference to Section x.y, Eq. (n), Proposition n, Remark, Algorithm, Table n, Figure n, Appendix X.n, Eq. (X.n), Supplementary Note Sn(.m), Table STnn / Supplementary Table STnn, Figure Snn, in both languages. Verify that the target exists AND that it contains what the sentence claims (e.g. 'Section 3.3 shows that ...' must be true of the new Section 3.3). Pay special attention to references in APPENDICES and SUPPLEMENTARY that point into the main text, because the main-text sections, equations, propositions and tables were renumbered (brief section 5): any old number left unmapped or double-mapped is a bug. Also check that main-text figures/tables are numbered in order of first citation and that every figure/table is cited in the text. Report each wrong reference with the corrected text.` },
  { key: 'fidelity', prompt: `LENS: factual fidelity of the main text against the previous version and the evidence. Compare the new MANUSCRIPT_EN.md with the old one (git show ${OLD}:docs/paper_p1/MANUSCRIPT_EN.md) section by section using the brief's mapping (old 3.1->3.1, 3.2->4.1, 3.3->4.2, 4.k->3.k, others unchanged; old Table 5 -> Supplementary Note S8 Table ST25; old Fig 15(c) -> Figure 16). Find: numbers that changed value or precision without justification; conditions or scopes that were dropped or widened (e.g. 'under traction loads', 'at most', 'in every component', 'on these geometries'); claims that became stronger or weaker than the source; statements removed without a home (check SUPPLEMENTARY Note S8 and the appendices before reporting); new claims not supported by the source or the supplement tables. Also verify the new derived numbers: 71 to 117 (Table ST03b), 0.144% (Supplementary Note S4.3), 1.14%/12.7%, 10.7 s per cell in the Figure 16 caption (mean time per cell = total mean time / total cells over the four plates in review_r1/results/X6_final/scale/scale_summary.json). Report each problem with the corrected text.` },
  { key: 'terms', prompt: `LENS: terminology consistency across all six files (EN and CN). Using the term map in brief section 3.1-3.3, grep all six files for old terms and for inconsistent variants of the new ones. Examples to check: retained/master DOFs (主自由度; no 保留自由度/保留位移), displacement recovery (位移恢复; no 延拓/extension in the old sense), cell face (胞元表面; no box face/胞元边界面), cut-plane elements (切割平面单元; no cut band/切割带), energy fraction (应变能占比; no energy share/能量份额), two-grid correction (两重网格修正; no equilibrium correction/平衡校正 except where NICE's name is spelled out), traction loads (面力载荷; no consistent traction/一致面力), nodal point loads (集中节点载荷), test displacements (测试位移; no directions/方向 in that sense), test cell (被测胞元; no target cell/目标胞元), weakly connected (弱连接; no weakly supported/弱支撑), local node position (no slot/槽位), feature channels / grid hierarchy (no latent/潜空间), cut-severity groups (no strata/分层), variant names (Base network + correction / 基础网络加修正; Uncorrected continuation / 未修正延续 and Smoothing-trained / 平滑训练 only in the supplement), configuration (CN 构型 for two-cell configurations), approximate compliance (近似柔度; no surrogate/代理), 'plate supported on its cut' (CN 切割面固支板) as the name of the plate. Also flag the same quantity named two different ways in neighbouring text, and CN terms that differ between main text, appendices and supplement. Report each occurrence that needs changing with the exact replacement.` },
]

const FIND = {
  type: 'object',
  properties: {
    findings: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          file: { type: 'string' },
          quote: { type: 'string' },
          problem: { type: 'string' },
          fix: { type: 'string' },
          severity: { type: 'string', enum: ['high', 'medium', 'low'] },
        },
        required: ['file', 'quote', 'problem', 'fix', 'severity'],
      },
    },
  },
  required: ['findings'],
}
const VER = {
  type: 'object',
  properties: {
    verdicts: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          index: { type: 'integer' },
          confirmed: { type: 'boolean' },
          reason: { type: 'string' },
          file: { type: 'string' },
          quote: { type: 'string' },
          fix: { type: 'string' },
        },
        required: ['index', 'confirmed', 'reason', 'file', 'quote', 'fix'],
      },
    },
  },
  required: ['verdicts'],
}

const results = await pipeline(
  LENS_LIST(),
  l => agent(`${l.prompt}\n${COMMON}`, { label: `review:${l.key}`, phase: 'Review', schema: FIND }),
  (r, l) => {
    if (!r || !r.findings || !r.findings.length) return { lens: l.key, findings: [], verdicts: [] }
    const list = r.findings.map((f, i) => `[${i}] file=${f.file}\nquote=${f.quote}\nproblem=${f.problem}\nfix=${f.fix}\nseverity=${f.severity}`).join('\n\n')
    return agent(`You are an adversarial verifier. Another reviewer (lens: ${l.key}) reported the findings below on the assembled manuscript. For EACH finding: (1) check that the quote exists verbatim in the named file (grep it; if it differs slightly, return the exact verbatim quote you found); (2) decide whether the problem is real and worth fixing before the manuscript goes to the advisor, by reading the surrounding text, the brief, the previous version or the evidence as needed; default to confirmed=false if the claim is wrong, a matter of taste, already decided, or would introduce an error; (3) if confirmed, return a fix that is exact replacement text for the quote (keep everything else in the quote unchanged; keep EN/CN numbers identical; keep markdown/LaTeX intact). Do not edit files.\n${COMMON}\n\nFINDINGS:\n${list}`,
      { label: `verify:${l.key}`, phase: 'Verify', schema: VER })
      .then(v => ({ lens: l.key, findings: r.findings, verdicts: v ? v.verdicts : [] }))
  },
)
function LENS_LIST() { return LENSES }
const out = results.filter(Boolean).map(x => ({
  lens: x.lens,
  n_found: x.findings.length,
  confirmed: (x.verdicts || []).filter(v => v.confirmed).map(v => ({ ...v, problem: (x.findings[v.index] || {}).problem, severity: (x.findings[v.index] || {}).severity })),
  rejected: (x.verdicts || []).filter(v => !v.confirmed).map(v => ({ index: v.index, reason: v.reason, problem: (x.findings[v.index] || {}).problem })),
}))
return out
