# Copy for js/main.js

Rewritten versions of every user-visible string that `js/main.js` renders into the page. The file `js/main.js` itself was not edited; whoever owns it should paste these in. Each entry gives the identifier (block and variable), the current text, and the corrected text. All numbers were checked against the paper (arXiv:2602.02600v3, Tables 1, 2, 3, 4, 13, 14 and Figures 2, 5, 6, 12, 13). Rules applied: no em or en dashes anywhere (write "48 to 62%"), no contractions, no "X, not Y" constructions, no sentence fragments, the paper's own terminology (Attack Success Rate, Refusal Rate, Harmful Remasking Rate, Full Recovery Rate, Internal Recovery Rate, SRI, SRI Guard).

## Must-fix items (factual or typographic)

1. **`gap()` note template uses an en dash.** `fmt(mn(arV)) + '–' + fmt(mx(arV))` renders "48.2%–62.2%" into `#gapNote`. Replace the two `'–'` literals with `' to '` so the note reads "48.2% to 62.2%".
2. **`defenses()` note computes a ratio that contradicts the paper.** `Math.round(lg[0] / sri[0])` yields 311 (LLaDA), 307 (LLaDA-1.5), 378 (Dream), 476 (Qwen-2.5), 467 (LLaMA-3) and 263 (Gemma) "× cheaper". The paper states 150 to 300 times lower cost (Section 4.3, Appendix C.3) and "two orders of magnitude lower inference cost" (Appendix C.3). Drop the computed ratio and use the wording below.
3. **`explorer()` Dream note is wrong about Figure 12(f).** The Dream jailbreak trajectory ends inside the compliance region (about 0.8), so "end far closer to the refusal zone" does not describe the figure. Corrected text below.
4. **`explorer()` model sizes.** Table 11 of the paper lists Qwen-2.5 as 8B while its footnote links the 7B-Instruct checkpoint. Avoid the inconsistency by dropping sizes from the explorer names (Section B.1 of the paper only states that all models are in the 7 to 8B range).
5. **File header comment** (line 1) contains an em dash ("Step-Wise Refusal Dynamics — project page interactions"). It is not user-visible; replace it with a colon for consistency.

## 2. Hero simulator (`sim()`)

Note: another agent is reworking the hero. If the card switches from the hand-drawn schematic to real data from the paper's figures, delete the "schematic" sentences below and cite the figure that supplies the data instead.

### `CAPTIONS.diffusion`
Current: `<strong>Diffusion remasking.</strong> Every step predicts all positions but commits only the most confident ones. Harmful predictions (dashed red) compete for commitment and lose; once refusal tokens are committed (solid green) the rest of the response follows them. Schematic of Figures 1, 2 and 12 in the paper, with harmful content redacted; not measured data.`

Corrected: `<strong>Diffusion remasking.</strong> Every step predicts all positions and commits only the positions selected by the remasking rule. Harmful predictions (dashed red) compete for commitment and stay uncommitted; once refusal tokens are committed (solid green), the rest of the response follows them. This card is a schematic inspired by Figures 1, 2 and 12 of the paper, with harmful content redacted; the curves are illustrative and do not reproduce measured data.`

### `CAPTIONS.ar`
Current: `<strong>Autoregressive sampling.</strong> Each token is committed the moment it is generated. Once the compliance prefix is out, the harmful continuation cannot be revised, and the SRI signal never settles into the refusal zone: incomplete internal recovery that the text alone would not reveal. Schematic; not measured data.`

Corrected: `<strong>Autoregressive sampling.</strong> Each token is committed the moment it is generated. Once the compliance prefix is out, the harmful continuation cannot be revised, and the SRI signal never settles into the refusal zone. This is the incomplete internal recovery that the final text alone would hide. The card is a schematic and does not reproduce measured data.`

### `stateFor()` labels
Keep as they are: `Fully masked`, `Harmful predictions forming · none committed`, `Revising · refusal tokens being committed`, `Recovered · safe final output`, `Compliance prefix committed`, `Harmful tokens committed · cannot be revised`, `Jailbreak succeeded · no recovery`.

### Sparkline labels
Keep: `compliance-aligned`, `refusal-aligned`, the axis labels, and the value prefix `σ = `.

### Play button `aria-label`
Keep: `Play animation` / `Pause animation`.

## 3. Charts

### `hrr()` (Table 1)
- `sub`: keep `diffusion`.
- Tooltips: current `LLaDA · Harmful remasking rate 0.81` and `LLaDA · Full recovery rate 0.63`. Corrected (paper's capitalised terms): `LLaDA · Harmful Remasking Rate 0.81`, `LLaDA · Full Recovery Rate 0.63` (same pattern for Dream and LLaDA-1.5).

### `delta()` (Table 2)
- `attacks`: keep `WildJailbreak`, `Flip Attack`, `PAIR`, `Refusal Suppression`, `Random Search`.
- `name`: current `Δ refusal rate` / `Δ attack success (reduction)`. Corrected: `Δ Refusal Rate` / `Δ Attack Success Rate (reduction)`.
- Tooltip suffix: current `+14 pts`. Corrected: `+14 points`.
- Data check: the arrays are ordered [LLaDA, LLaDA-1.5] and match Table 2 (LLaDA: Wild +14/+18, Flip +33/+36, PAIR +7/+5, RefusalSup +55/+26, Random +21/+16; LLaDA-1.5: Wild +15/+21, Flip +38/+60, PAIR +0/+2, RefusalSup +52/+20, Random +19/+8). No change.

### `gap()` (Table 3)
- `colNames`: keep, or change `Raw harmful` to `Raw harmful prompts`.
- `sub`: keep `autoregressive` / `diffusion`.
- Tooltip: current `LLaDA · All jailbreaks · attack success 18.4%`. Corrected: `LLaDA · All jailbreaks · Attack Success Rate 18.4%` and `... · Refusal Rate 67.4%`.
- `note.innerHTML` template. Current:
  `<strong>{column} · {Refusal rate|Attack success}.</strong> Autoregressive models: {min}–{max}. Diffusion models: {min}–{max}.` followed by ` On raw harmful prompts every model refuses most of the time; the gap opens only under attack.` (raw) or ` Lower is safer.` / ` Higher is safer.`
  Corrected:
  `<strong>{column} · {Refusal Rate|Attack Success Rate}.</strong> Autoregressive models: {min} to {max}. Diffusion models: {min} to {max}.` followed by ` On raw harmful prompts all models exhibit relatively high refusal rates; the gap emerges under jailbreak attacks.` (raw) or ` Lower is safer.` / ` Higher is safer.` (unchanged).
- Table caption. Current: `Table 3 of the paper. RR: refusal rate (higher is safer). ASR: attack success rate (lower is safer).` Corrected: `Table 3 of the paper. RR: Refusal Rate (higher is safer). ASR: Attack Success Rate (lower is safer).`
- Table headers `{column} RR ↑` / `{column} ASR ↓`, row suffixes ` (AR)` / ` (diffusion)`, and the button labels `Full table` / `Hide table`: keep.

### `ablation()` (Table 4)
- Row labels: keep `Text-based signal`, `Static activations (first step)`, `SRI, first layer`, `SRI, middle layer`, `SRI, last layer (ours)`.
- Tooltip `{label} · AUROC 0.920 ± 0.047`: keep.

### `defenses()` (Table 13)
- `D`: keep `Undefended`, `PPL filtering`, `Self-Examine`, `LlamaGuard 3`, `SRI Guard`.
- `small`: keep `ours · internal signal`.
- `labels`: keep `Overhead`, `False positives`, `Jailbreak refusal`, `Jailbreak attack success`.
- Tooltip `{defense} on {model} · {label} {value}`: keep.
- `note.innerHTML` template. Current:
  `<strong>{Model}.</strong> SRI Guard runs at {sri}% overhead against {lg}% for LlamaGuard 3, about {ratio}× cheaper, with a higher refusal rate ({a}% vs {b}%).` or `... with refusal rate {a}% vs {b}% for LlamaGuard 3.` then ` The threshold is the 99th percentile of reconstruction error on held-out harmless prompts; Undefended false positives come from the base model refusing benign prompts on its own.`
  Corrected (one branch is enough, and the ratio is removed):
  `<strong>{Model}.</strong> SRI Guard adds {sri}% overhead against {lg}% for LlamaGuard 3, two orders of magnitude less, and reaches a jailbreak refusal rate of {a}% against {b}% for LlamaGuard 3. The detection threshold is the 99% quantile of the reconstruction error on held-out harmless prompts; the false positives of the undefended model come from the base model refusing benign prompts on its own.`

## 4. SRI explorer (`explorer()`)

### `INFO[*].name`
Current: `LLaMA-3 (8B)`, `Qwen-2.5 (7B)`, `Gemma (7B)`, `LLaDA (8B)`, `LLaDA-1.5 (8B)`, `Dream (7B)`. Corrected: `LLaMA-3`, `Qwen-2.5`, `Gemma`, `LLaDA`, `LLaDA-1.5`, `Dream` (see must-fix item 4).

### `INFO[*].fam`
Keep `Autoregressive` / `Diffusion`.

### `INFO.llama3.note`
Current: `Harmless and refusal references stay flat and well separated; the jailbreak trajectory is volatile and never settles into the refusal zone. LLaMA-3 reaches 59.2% attack success across jailbreaks with near-zero internal recovery (Figure 5). SRI Guard lifts its jailbreak refusal rate from 23.4% to 55.0% at 0.01% overhead.`

Corrected: `The harmless and refusal references stay flat and well separated. The jailbreak trajectory drops into the refusal region within the first few steps and then climbs back into the compliance region, an internal move toward refusal that does not complete while the text-level signal stays flat. LLaMA-3 reaches a 59.2% Attack Success Rate across jailbreaks, its Internal Recovery Rate stays near 0.1 at every threshold (Figure 5), and SRI Guard lifts its jailbreak refusal rate from 23.4% to 55.0% at 0.01% overhead (Table 13).`

### `INFO.qwen25.note`
Current: `The model behind Figure 2 of the paper: a harmful generation dips toward refusal, then drifts back toward compliance while the text keeps complying. Qwen-2.5 has the highest jailbreak attack success of the six (62.2%); SRI Guard cuts it to 40.6% and raises refusal from 11.4% to 47.8%.`

Corrected: `This is the model behind Figure 2 of the paper: the harmful generation descends toward the refusal region, hovers around the 0.5 boundary between steps 5 and 13, and drifts back toward compliance while the text keeps complying. Qwen-2.5 has the highest jailbreak Attack Success Rate of the six models (62.2%); SRI Guard lowers it to 40.6% and raises the refusal rate from 11.4% to 47.8% (Table 13).`

### `INFO.gemma.note`
Current: `Gemma refuses jailbreaks about as often as Dream (46.2% vs 44.4%), yet its attack success is 48.2% against Dream’s 9.4%: refusals and attacks are not two sides of one coin. Its jailbreak SRI signal is noisy across the whole generation.`

Corrected: `Gemma refuses jailbreaks about as often as Dream (46.2% against 44.4%), yet its Attack Success Rate is 48.2% against 9.4% for Dream, so a refusal rate alone does not describe robustness. Its jailbreak SRI signal dips below 0.5 in the first steps and then stays volatile inside the compliance region for the whole generation.`

### `INFO.llada.note`
Current: `The primary diffusion baseline. Harmful intermediate content is revised as remasking proceeds (HRR 0.81, FRR 0.63), and internal recovery rate is far above the autoregressive models at every threshold. Switching its weights to AR sampling drops IRR by up to 0.46 (Figure 6).`

Corrected: `LLaDA is the primary diffusion baseline. Harmful intermediate content is revised as remasking proceeds (HRR 0.81, FRR 0.63), and its Internal Recovery Rate stays far above the autoregressive models at every threshold (Figure 5). Running the same weights with AR sampling lowers IRR by up to 0.46 (Figure 6). In this jailbreak example the signal oscillates around the 0.5 boundary for fifteen steps before settling in the compliance region.`

### `INFO.llada15.note`
Current: `A newer LLaDA variant with the same qualitative behaviour: HRR 0.92, FRR 0.65, and aggregate jailbreak attack success of 21.0% against 48–62% for the autoregressive models. Under AR sampling of the same weights, IRR falls by up to 0.48.`

Corrected: `LLaDA-1.5 is a newer LLaDA variant with the same qualitative behavior: HRR 0.92, FRR 0.65, and an aggregate jailbreak Attack Success Rate of 21.0% against 48 to 62% for the autoregressive models. Under AR sampling of the same weights, IRR falls by up to 0.48 (Figure 6). Its jailbreak trajectory drifts around the 0.5 boundary and never approaches the flat harmless reference.`

### `INFO.dream.note`
Current: `A different diffusion architecture with the strongest recovery statistics of the three (HRR 0.96, FRR 0.73) and the lowest jailbreak attack success in the study, 9.4%. Its jailbreak trajectories are noisier than harmless ones but end far closer to the refusal zone.`

Corrected: `Dream is a different diffusion architecture with the strongest recovery statistics of the three (HRR 0.96, FRR 0.73) and the lowest jailbreak Attack Success Rate in the study, 9.4%. Its jailbreak trajectory hovers around the 0.5 boundary for the first fifteen steps before rising into the compliance region, far from the flat harmless reference.`

### Alt-text templates in `show()`
- `sriImg.alt`. Current: `SRI signal over generation steps for {name} under jailbreak prompts, with harmless and refusal reference trajectories and shaded compliance and refusal regions.` Corrected: `SRI signal over 32 generation steps for {name} under a jailbreak prompt, with flat harmless and refusal reference trajectories and shaded compliance and refusal regions.`
- `ldaImg.alt`. Current: `Two-dimensional LDA projection of SRI space for {name}: harmless, harmful and refusal responses.` Corrected: `Two-dimensional LDA projection of the SRI space for {name}: harmless, harmful and refusal responses occupy distinct but partly overlapping regions.`

## 6. Quick-links rail (`rail()`)
Keep `Link copied`.

## 7. BibTeX (`bib()`)
Keep `Copied`, `Copy BibTeX`, `BibTeX copied`.

## 1. Navigation (`nav()`)
Keep `Open menu` / `Close menu`.
