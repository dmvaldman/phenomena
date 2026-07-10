# A Visual Workspace in Unified Autoregressive Models

*Research plan — probing for a small, causally-privileged, visualizable subspace of the residual stream in unified text/image models, in the style of the Jacobian-lens / global-workspace methodology (Gurnee, Sofroniew, et al. 2026), extended from verbalizable to **visualizable** content.*

---

## 1. Hypothesis

Unified multimodal models trained with a single next-token objective over text *and* discrete image tokens (Emu3 family) maintain a small subspace of the residual stream — the **V-space** — that carries internal *imagery*: visual content that is not part of the input or output, is introspectively reportable, is broadcast by attention despite accounting for little activation variance, and is causally used by downstream reasoning and drawing.

### Target properties

Each property is a falsifiable claim with a designated experiment (§6).

| # | Property | Claim | Text test (Phase 1) | Visual test (Phase 2+) |
|---|----------|-------|---------------------|------------------------|
| P1 | Covert representation | "Think about X while answering Y" loads X-imagery into the V-space at text positions, absent from input tokens and output tokens | T2 | E3 |
| P2 | Introspective report | Asked to describe its thinking, the model reports features of the lens-decoded imagery that appear in neither prompt nor output | T3 | E4 |
| P3 | Broadcast | Attention heads and MLPs disproportionately relay/amplify V-space directions across positions, despite the V-space being a small fraction of variance | T6 | E7 |
| P4 | Causal privilege | The V-space component of a visual concept (~small % of variance) drives drawing/report; the orthogonal ~95% does not; clamping V-space kills the remainder's residual effect | T5 | E5 |
| P5 | Functional use | Unspoken visual intermediates in reasoning live in the V-space; swapping them flips the verbal answer | T4 | E6 |
| P6 | Selectivity | V-space ablation destroys flexible imagery-dependent tasks but spares automatic visual processing | T7 | E8 |
| P7 | Capacity & eviction | The V-space holds few items; new imagery evicts old within a few tokens | T7 | E9 |
| P8 | Cross-modal convergence | Imagery evoked by text and by an input image load overlapping V-space directions (shared hub) | — | E10 |
| P9 | Spontaneous determinacy | Imagined content has prompt-underdetermined details (color, pose) stable *within* a trial across lens/draw/report, varying *across* trials | — | E4b |
| P10 | Ignition (stretch) | Bistable visual inputs produce all-or-none V-space interpretation at a depth band; report tracks the lens trial-by-trial | T1 (structural) | E11 |

**Text-first principle.** Every property that has a text analog is established in text on the same model first (Phase 1, T-series): this validates the estimator and toolchain against the source paper's known results, locates Emu3's workspace band, and produces the per-layer effect maps that the visual experiments then query. No visual lens work starts before the Phase 1 gate (§6) is passed.

**Framing discipline.** Evidence for P1–P10 establishes *functional access* structure ("phenomenal workspace access" in the access-consciousness sense), not phenomenal experience. Write-ups claim the former only.

---

## 2. Models & infrastructure

### Primary model: Emu3 (BAAI, 8B)

Why: single decoder-only transformer, single softmax over a unified vocabulary of 184,622 tokens including 32,768 MoVQGAN image codes (8×8 px per code; 512×512 image = 64×64 grid = 4096 tokens, `[EOL]` row delimiters, `[SOV]…[SOT]…[EOV]` region brackets). Trained **from scratch** on both modalities — shared representations are maximally plausible. 32 layers, d_model = 4096, Llama-2-style, GQA.

Checkpoints (all open):
- `BAAI/Emu3-Chat` (`-hf` variant) — instruction following, image *input*; generation ability degraded (verify empirically, task M0.4).
- `BAAI/Emu3-Gen` (`-hf` variant) — T2I generation; weak instruction following.
- `BAAI/Emu3-Stage1` — pretraining checkpoint; both captioning and 512×512 generation, weak instruction following (use few-shot scaffolds).
- `BAAI/Emu3-VisionTokenizer` — MoVQGAN encoder/decoder (needed everywhere).

Checkpoint strategy: text-workspace replication and image-input work on **Chat**; generation-side lenses on **Stage1** (single model with both capabilities beats stitching Gen+Chat); escalate dream experiments to whichever passes M0.4. Lenses are per-checkpoint — never reuse J matrices across checkpoints.

### Escalation model: Emu3.5 (BAAI, 34B)

One RL-tuned model natively interleaving text reasoning and image generation (131,072 IBQ codes, 16× compression, 64 layers, d=5120, from Qwen3). Use once the method works on Emu3: its interleaved "think-then-draw" behavior is the natural habitat for P1/P2, and workspace effects grow with scale in the source paper. Needs 80GB-class GPU; prefer pure-AR decoding (DiDA off) so the lens story stays clean.

Non-candidates: BAGEL and Skywork UniPic generate via continuous latents (flow matching / MAR-diffusion) — no image vocabulary, no shared unembedding. Revisit only for a Diffusion-Lens-style "decoder lens" replication arm.

### Hardware

- RunPod A100 80GB (~$1.6–2/hr): all Emu3 work incl. Jacobian backward passes. A40/A6000 48GB suffices for forward-only phases.
- ~60GB disk (weights + tokenizer + activation caches).
- Local M2/16GB: analysis, plotting, paper — no model inference.

### Core software tasks

- Hidden-state capture: `output_hidden_states=True` (hf checkpoints) or forward hooks; cache h_ℓ,t as fp16 to disk per experiment.
- Vocab bookkeeping: exact index range of the vision block within the 184,622 vocab; `[SOV] [SOT] [EOV] [EOL] [EOF]` ids; final RMSNorm handle (apply before any unembedding — the "ln_f is crucial" lesson from both logit lens and Diffusion Lens).
- Constrained image sampler: vision-block masking, `[EOL]`/`[EOV]` scheduling, optional CFG — reimplement or lift from BAAI repo; must accept externally supplied/pinned residual states (for patching) and an arbitrary readout layer (for early exit).
- Tokenizer round-trip: image → codes → image sanity check; per-code patch atlas (decode each codebook entry in a neutral context) for rendering bag-of-patches readouts.
- VQA judge harness: local Qwen-VL-class or API VLM answering fixed question batteries about generated images, with a ~10% human-agreement audit (Diffusion Lens methodology; they validated GPT-4V vs 10 annotators).

---

## 3. Lens construction (the J-space machinery)

Reporting convention: sample ~16 of 32 layers evenly, index as % depth.

### L0 — Logit lens baseline (free)

`lens₀(h_ℓ) = softmax(W_U · norm(h_ℓ))`. Read (a) full-vocab and (b) **vision-block-renormalized** distributions. Expect: fine in late layers, degraded early, vision rows possibly suppressed at text positions (trained P(vision|text-context) ≈ 0 — renormalization within the block is mandatory, absolute mass is uninformative).

### L1 — Text J-lens (replication of the paper's object)

$$J_\ell^{text} = \mathbb{E}_{t,\ t' \ge t,\ \text{prompt}}\left[\frac{\partial h_{L,t'}}{\partial h_{\ell,t}}\right]$$

over ~1,000 pretraining-like prompts. Rows of $W_U J_\ell^{text}$ = verbalizability vectors; validates our estimator against the paper's phenomena (Phase 1, T-series).

**Estimator.** Full per-sample Jacobians are unaffordable; use stochastic VJP probing: sample $v \sim \mathcal{N}(0, I_d)$ at target position(s) $t'$, one backward pass yields $v^\top \partial h_{L,t'}/\partial h_{\ell,t}$ **for all source layers ℓ and positions t simultaneously**; accumulate $\hat J_\ell \mathrel{+}= v \otimes (v^\top J)$ across samples ($\mathbb{E}[vv^\top] = I$ makes this unbiased). Budget ~30–50k backward passes ≈ tens of A100-hours. Convergence check: split-half cosine of $\hat J_\ell$; behavioral check: J-lens ≈ logit lens at late layers, diverging earlier.

**Cheap fallback:** tuned-lens-style affine translator per layer (train h_ℓ → h_L). Use only as a diagnostic — it is correlational and known to "skip ahead" past exactly the intermediates we care about.

### L2 — Visual J-lens (the novel object)

$$J_\ell^{vis} = \mathbb{E}_{t \in \text{text},\ t' \in \text{image region},\ \text{doc}}\left[\frac{\partial h_{L,t'}}{\partial h_{\ell,t}}\right]$$

Same estimator, but the expectation runs over **caption→image documents** in Emu3's native format (`[BOS] caption [SOV] meta [SOT] codes… [EOV]`) — manufacture ~5–10k by tokenizing captioned images (LAION/COCO subsets) — with source positions in the text and target positions inside the image region. Rows of $W_U^{vis} J_\ell^{vis}$ (vision rows only) = **visualizability vectors**: what a text-position activation is disposed to make the model *draw*. This sidesteps vision-row suppression the same way the J-lens fixes early-layer coordinate mismatch: it measures causal disposition, not current logits.

Variants to compare: target-position weighting (uniform vs. early-image-tokens-only), per-row-of-grid targets (crude spatial localization), and $J^{vis}$ vs $J^{text}$ subspace angles per layer (do "disposed to say" and "disposed to draw" coordinates converge at the workspace band? — that convergence is itself evidence for a shared hub, P8-adjacent).

### L3 — V-space definition

By analogy to the J-space: the set of sparse non-negative combinations of visualizability vectors. Sparse decomposition by gradient pursuit, k swept in {5, 15, 25, 50} (expect meaningful-active count to set k; paper found ~25 for text). Report: fraction of activation variance in V-space per layer (expect small, <10%), occupancy curves, overlap between V-space and J-space (text) frames.

**Granularity mitigation (important):** single MoVQGAN codes are 8×8 pixel patches — texture/color atoms, not concepts. Precompute a **semantic pooling** of the codebook: embed each code's decoded patch (SigLIP or decoder-feature space), cluster into ~512–2048 visual "morphemes," and report lens readouts at both raw-code and cluster granularity. Concept-level imagery may only be legible at cluster level.

---

## 4. Readout instruments

- **I1 — Per-position signature** (no image produced). 32,768-dim vision-row readout per (layer, text position). Metrics: cosine stability across consecutive positions; against controls; match to **reference histograms** (code/cluster distributions of real photos of the target vs. foil categories, tokenized through MoVQGAN). Cheap; runs at every layer; the workhorse for P1 and band-finding.
- **I2 — Behavioral draw + interventions** (the evidential heavyweight). Append `[SOV] meta [SOT]` after the text turn; full model draws; MoVQGAN decodes; VQA judge scores. Alone it proves nothing (the drawing may read the visible prompt); the claims come from interventions: **transplant** layer-ℓ text-position residuals into a neutral prompt's run (does the car appear with no "car" in context?), **ablate** the V-space component (does the car vanish while the text answer survives?), **steer/swap** along visualizability vectors (car→boat in the drawing without touching the prompt).
- **I2.5 — Frozen-context draw** (Diffusion Lens analog). Pin text-position residuals at their layer-ℓ values for all layers above ℓ (image tokens still get the full stack), then draw. Isolates "how much visual specification exists at depth ℓ of prompt processing" with the drawing machinery intact. Include a blend knob α·h_ℓ + (1−α)·h_L — frozen mid-layer states are off-distribution for the drawer; expect to need on-manifold correction (the ln_f lesson, continued).
- **I3 — Early-exit rollout.** Amputate layers above ℓ; generate the image reading logits at ℓ (constrained sampler still on). "What would it draw if computation stopped at ℓ." Caveat: 4096-step compounding — blur is ambiguous between absent content and rollout instability; always pair with I1 at the same layer.
- **I0 — Teacher-forced crystallization** (calibration only). During a normal generation, per-layer argmax at each image position, decode per layer. Maps which depths carry decodable pixel content at image positions; only exists where an image region exists.

---

## 5. Statistical hygiene (applies to all experiments)

- Every effect: ≥100 trials, matched controls (no-instruction; unrelated-concept; instruction-without-imagery e.g. "think about justice"), multiple paraphrases, multiple seeds.
- VQA judge with fixed question batteries; 10% human audit; report judge–human agreement.
- **Shuffle tests** for all consistency claims: within-trial agreement (lens ↔ draw ↔ report) must beat trial-shuffled agreement.
- Pre-register per-experiment success criteria (in each experiment file) before running at scale.
- Layer curves for everything — the depth profile is data, not nuisance.

---

## 6. Experiments

### Phase 1 — The text workspace in Emu3 (step 1; everything else is gated on this)

Goal: reproduce the source paper's J-space findings in text on Emu3, and produce a **layer-band map** — for each workspace property, the depth range where it holds. Deliverable: the band $[\ell_{on}, \ell_{off}]$ (paper found ~38–92% depth on Claude models) plus per-property effect sizes, all computed with L0/L1 only. No vision anywhere in this phase.

**T0 — Lens build & validation.** L0 everywhere; L1 estimator run to convergence (split-half cosine ≥ ~0.8 in mid layers). Sanity: J-lens ≈ logit lens in the last few layers, diverging earlier; top lens tokens on generic prompts are interpretable in mid layers where logit lens is noise. **Toolchain control:** run T0+T2 on a strong open *text-only* LLM (e.g. Qwen3-8B) first/in parallel — separates "our estimator is broken" from "Emu3's text workspace is weak," since the paper's effects are only documented on much stronger models.

**T1 — Band structure (structural signatures, P10-adjacent).** Layer-wise CKA on lens geometry (look for the sensory → workspace → motor block structure); lens next-token accuracy, excess kurtosis, cross-position autocorrelation, effective dimensionality across depth. These should co-locate an onset/offset; that band is the prior for everything downstream. Include the artificial-ambiguity ignition probe (interpolated embeddings → graded-vs-snap readout) if time permits.

**T2 — Directed modulation (P1 in text).** The core "think about cars while telling me what color the sky is in one word" battery: target-concept lens loading at unrelated positions vs. no-instruction and unrelated-instruction controls; layer curve → band confirmation. "Ignore X" white-bear variant (suppressed below focus but above baseline). Expect weaker effects than the paper's (8B; modulation grows with scale); few-shot scaffolds if instruction following is the bottleneck.

**T3 — Report & introspection (P2 in text).** (a) Spontaneous-choice swap: model picks an item covertly ("choose a sport, don't say it"), lens reads the choice, swap the lens vector, model's later report follows the swap. (b) Injection: steer a concept vector across the user turn; "are you thinking about anything? what?" → injected concept reported without being blurted elsewhere.

**T4 — Unspoken intermediates (P5 in text).** Two-hop tasks ("legs on the animal that spins webs"); intermediate visible in lens, never in text; swapping intermediate flips the answer; intermediate takes effect earlier in depth than the answer (the anti-smuggling control).

**T5 — Causal privilege (P4 in text).** Concept representation → J-space component (sparse decomposition) vs. orthogonal remainder; steer each at matched norm; clamping test. This is the paper's privilege result and our template for E5.

**T6 — Broadcast (P3 in text).** MLP amplification of J-lens vectors vs. controls across depth; relay-head identification (top OV-transmitters of J-space content between positions); targeted relay ablation degrades T2/T3 effects with minimal perplexity change.

**T7 — Selectivity & capacity (P6/P7 in text).** Whole-J-space ablation across the band: shallow tasks (extractive QA, sentiment) unaffected; inference-heavy tasks (analogies, multi-hop, translation) degraded. Word-list capacity (~6 unrelated items vs. whole shared category) and eviction dynamics.

**Phase 1 gate:** T2 modulation significantly above baseline *and* at least one successful T4 intermediate swap, with a coherent band from T1. Pass → Phase 2. Marginal → few-shot scaffolds, then Emu3.5. Fail on both Emu3 and the Qwen3 toolchain control → the estimator is broken, fix before proceeding. Fail on Emu3 only → Emu3.5 becomes the primary model (cost decision point).

### Phase 2 — The visual workspace

**E1 — Visual calibration & AR Diffusion-Lens replication.** I0 crystallization across depth for standard prompts; then the Diffusion Lens batteries (complexity: object / colored object / two colored objects; common vs. rare concepts; coarse-to-fine celebrity/detail refinement) with per-layer VQA curves via I2.5 and I3. Deliverables: the depth band where pixel content is decodable — to be compared against the Phase 1 text band; whether bag-of-concepts → relation-binding and rarity-costs-depth reproduce in a unified AR model (any *difference* from the diffusion text-encoder picture is standalone publishable). *Success: coherent I0 images above some layer; monotone-ish VQA curves.*

**E3 — Covert visual content (P1).** Prompts: "think about {X} while {unrelated task with one-word answer}". Readouts: I1 stability + reference-histogram match at every layer (raw codes and clusters); render via I2 (with transplant control), I2.5, I3 at the bands T1/E1 identified. Key comparison: does the visual signature band coincide with the text workspace band, or does conceptual→pixel expansion happen elsewhere/nowhere? A clean dissociation is a finding, not a failure. *Success: I1 signature for X beats controls (p<.01) at some band; transplanted-context draw shows X above chance by VQA.*

**E4 — Introspective report (P2).** Same trials as E3; after the answer, ask "describe what you were picturing — one sentence." VQA-compare the report against the lens-decoded image (I2 transplant version, to break the visible-prompt confound). **E4b — spontaneous determinacy (P9):** score prompt-underdetermined attributes (color, viewpoint, count) for three-way within-trial agreement (lens image ↔ drawn image ↔ verbal report) vs. shuffled-trial baseline. *Success: within-trial agreement > shuffled at p<.01 on attributes never named in prompt or task output. This is the headline experiment.*

**E5 — Causal privilege (P4).** Take concept representations (mean difference or lens vectors) for imagined objects; decompose into V-space component vs. orthogonal remainder (gradient pursuit, §L3). Steer each into neutral contexts at matched norm; measure draw/report effects; then clamp V-space coordinates and re-test the remainder. *Success: V-space component (few % variance) ≫ remainder on draw/report effect; clamping kills the remainder's residual effect.*

**E6 — Functional use in reasoning (P5).** Tasks with visual intermediates and non-visual answers: color mixing ("the color of a banana mixed with the sky" → green), size/shape comparisons, spatial composition ("the letter formed by two stacked Vs"), legs-on-the-imagined-animal counting. Lens the intermediate; swap it (banana→strawberry in V-space coordinates) and check the verbal answer flips; verify the intermediate takes effect earlier in depth than the answer (the paper's two-hop control). *Success: >30% swap-flip rate with intermediate-before-answer depth ordering.*

**E7 — Broadcast (P3).** (a) MLP amplification: pass visualizability vectors through each layer's MLP; amplification factor vs. random/neuron-basis controls across depth. (b) Relay heads: score every attention head for OV-transmission of V-space content between positions; identify the top-1% relay population; compare with the text-J-space relay population (same heads = shared hub, different = parallel workspaces — either is interesting). (c) Ablate relay heads: covert-imagery report (E4) and transplant-draw (E3) should degrade with minimal next-token perplexity change. *Success: amplification and relay concentration significantly above controls; targeted ablation selectively degrades imagery tasks.*

**E8 — Selectivity (P6).** Image-input tasks on Emu3-Chat, matched pairs: automatic (continue/copy an image region, detect a low-level anomaly) vs. flexible (describe from memory after distraction, reason about an occluded part, draw a modified version). Whole-V-space ablation across the band (excluding tokens about to be output). *Success: flexible tasks degrade ≫ automatic tasks.*

**E9 — Capacity & eviction (P7).** "Think about a red car and a green boat and a yellow hat…" — sweep 1–8 items; per-item I1 signature strength; category-chunking variant (8 vehicles vs. 8 unrelated objects); eviction: "now think about a boat" and track the car signature's decay over subsequent tokens. *Success: any reliable capacity ceiling + eviction curve; compare with the paper's ~6-unrelated-items text result.*

**E10 — Cross-modal convergence (P8).** Same concept evoked three ways: text instruction ("think about a red car"), image input (photo of a red car + unrelated task), and recall ("the object from earlier"). Compare V-space signatures and steering-vector cosines across evocation modes, per layer. Include minimal-pair image edits (color/pose swaps) and test the lens tracks the *specific* seen details at text positions. *Success: cross-modal signature similarity ≫ across-concept similarity in the workspace band.*

### Phase 3 — Stretch

**E11 — Bistable imagery & ignition (P10).** Ambiguous inputs: duck–rabbit-style images (input side), ambiguous descriptions ("a figure that could be a vase or two faces") (text side); interpolated/blended image-token grids (artificial ambiguity, the paper's ignition paradigm). Look for all-or-none V-space interpretation at a band onset and trial-by-trial lens↔report correlation.

**E12 — Emu3.5 escalation.** Port L2/E3/E4 to Emu3.5; exploit native interleaved reasoning (lens the text-reasoning segment *between* images in a multi-step generation); scale comparison of every P1–P8 effect size.

---

## 7. Risks & fallbacks

| Risk | Signal | Fallback |
|------|--------|----------|
| Vision rows dead at text positions | I1 flat even renormalized, all layers | L2 visual J-lens is the designed fix; if J-corrected readout is also flat → imagery is conceptual-only: pivot to I2 transplant-draw as sole visual readout; the dissociation is the paper |
| Code granularity too low-level | I1 signatures track color/texture but not object identity | Cluster-level readouts (§L3); category reference-histograms; lean on I2/I2.5 |
| No checkpoint both chats and draws | M0.4 fails on Chat and Stage1 few-shot | Split experiments across checkpoints (accepting weaker within-trial designs); light SFT to restore generation in Chat (last resort — moves the weights, note in write-up); escalate to Emu3.5 early |
| 8B too weak for "think about X" | T2 modulation ≈ baseline (while Qwen3 toolchain control passes) | Few-shot scaffolds; stronger instructions; Emu3.5 as primary model |
| Frozen/patched states off-manifold, drawer derails | I2.5/I2 outputs garbage even at late layers | Blend knob α; patch narrower layer windows; steer instead of hard-patch |
| Jacobian estimate too noisy | Split-half cosine < ~0.8 at workspace band | More probes; low-rank + diagonal parameterization; restrict targets to early image positions |
| Scooped | — | Lit sweep before each phase write-up; the P2/P9 report-consistency and P4 privilege results are the novel core — prioritize E4/E5 once E3 shows signal |

## 8. Milestones

- **M0 (infra, ~week 1):** pod up; Emu3 checkpoints running; hidden-state capture; vocab/special-token map verified; tokenizer round-trip; constrained sampler with patch/pin/early-exit hooks; **M0.4:** capability audit — can Chat draw? can Stage1 follow few-shot "think about X"? (decides checkpoint strategy). Text-only work (T0–T7) needs none of the image infra beyond hidden-state capture — start it immediately.
- **M1:** T0 lenses validated (incl. Qwen3 toolchain control); T1 band map.
- **M2:** T2–T7 text property battery; **Phase 1 gate verdict** — the workspace band $[\ell_{on}, \ell_{off}]$ and per-property effect sizes in text.
- **M3:** E1 visual calibration; L2 visual lens built; E3 covert-imagery verdict — **the go/no-go for the wild thesis.**
- **M4:** E4/E4b report-consistency result (headline), E5 privilege.
- **M5:** E6–E10 property battery; write-up of whatever pattern emerged.
- **M6:** Emu3.5 escalation + paper.

## 9. Prior-art anchors

- Gurnee, Sofroniew et al. 2026 — *Verbalizable Representations Form a Global Workspace in Language Models* (method + property framework; `papers/global_workspace.md`).
- Toker et al., ACL 2024 — *Diffusion Lens* (per-layer generative readout + VQA evaluation methodology; observational, text-encoder-only, no covert content, no interventions — the gap this project fills).
- Logit lens (nostalgebraist 2020) / tuned lens (Belrose et al. 2023) — baselines and their known failure modes.
- Emu3 / Emu3.5 model papers — `papers/models/`.
- TODO before Phase 2 write-up: systematic scoop-check on unified-model interpretability (Chameleon/Emu3 logit-lens work, "multimodal neurons," VLM-patch-to-text-vocab line — ours is the reverse direction).
