# The Jacobian Lens: Derivation, Usage, and Key Results

*A study guide to the Jacobian-lens methodology in* **"Verbalizable Representations Form a Global Workspace in Language Models"** *(Gurnee, Sofroniew, et al., Anthropic / Transformer Circuits Thread, July 2026).*

The paper's central empirical claim is that language models maintain a small, privileged set of internal representations — reportable, controllable, and reusable — sitting atop a much larger volume of automatic processing, in loose analogy to the "global workspace" of human conscious access. The tool used to find and manipulate these representations is the **Jacobian lens (J-lens)**, and the representational subspace it spans is the **J-space**. This document derives the J-lens from first principles, explains how it is used to read and write model activations, and summarizes what the authors found with it.

---

## 1. Background: the residual stream and the readout problem

A transformer processes its input as a sequence of token positions. At each position it maintains a **residual stream** vector $h$ — a shared memory that every layer reads from and writes to. The value is updated progressively across layers:

- At the **first layer**, $h$ encodes little more than the identity of the current token.
- At the **final layer** $L$, $h$ has been transformed into a representation from which the next-token prediction is read directly, by multiplying with a fixed **unembedding matrix** $W_U$ that maps residual-stream vectors to scores over the vocabulary.
- The layers in between perform the model's computation, incrementally enriching the stream with internally computed information.

The problem the J-lens solves: **how do we inspect the contents of the residual stream at these intermediate layers?** We would like a readout that tells us, for an activation $h_\ell$ at some intermediate layer $\ell$, which concepts it carries.

### The logit lens and why it is not enough

The **logit lens** simply applies the unembedding directly to an intermediate activation: read off $W_U h_\ell$. This works reasonably in late layers (because residual connections keep late-layer coordinates aligned with the output), but it degrades in earlier layers, where the residual stream uses a different "coordinate system" than the output space. In early-to-middle layers the logit lens produces largely uninterpretable readouts. The J-lens is designed as a principled correction to exactly this problem.

---

## 2. Deriving the Jacobian lens

### 2.1 The core idea

Characterize an intermediate activation vector **by its first-order causal effect on the model's outputs**, measured over a broad distribution of contexts.

Consider the residual stream $h_{\ell,t}$ at layer $\ell$ and token position $t$. A small perturbation to it will propagate through all remaining layers and shift the final-layer residual stream $h_{\text{final},t'}$ at every position $t' \geq t$ (a perturbation can only affect the present and future, not the past). To first order, this relationship is **linear**, and is captured exactly by the **Jacobian matrix**

$$
\frac{\partial h_{\text{final},t'}}{\partial h_{\ell,t}}.
$$

Composing this Jacobian with the unembedding $W_U$ gives the first-order effect of the perturbation on the model's output **logits** at position $t'$. In other words, the Jacobian tells us how nudging an intermediate activation would nudge what the model says, now and later.

### 2.2 The averaging step (the key move)

A Jacobian computed on a **single** prompt conflates two different things:

1. the model's **general disposition** to verbalize a given concept (what we want), and
2. the **particular use** to which that concept is being put in the current context (a confound).

The authors isolate component (1) by **averaging over positions and contexts**. For each layer $\ell$ they define:

$$
\boxed{\;J_\ell \;=\; \mathbb{E}_{\,t,\; t' \geq t,\; \text{prompt}}\!\left[\, \frac{\partial h_{\text{final},t'}}{\partial h_{\ell,t}} \,\right]\;}
$$

where the expectation runs over

- the **source position** $t$,
- all **subsequent positions** $t' \geq t$ within the context, and
- a corpus of **~1,000 prompts** drawn from a pretraining-like distribution.

The result is a single $d_{\text{model}} \times d_{\text{model}}$ matrix per layer, mapping directions at source layer $\ell$ to the final layer $L$. Averaging is what turns a context-specific quantity into a measure of what a direction is *poised* to make the model say across the range of contexts it typically encounters — i.e. what is **verbalizable**, as opposed to what merely *happens* to be verbalized once.

### 2.3 Reading from the lens

Applying the lens to an activation $h_\ell$ is equivalent to **replacing all downstream layers with the single linear map $J_\ell$**, then running the model's normal output head (typically a normalization, then multiplication by $W_U$):

$$
\boxed{\;\text{lens}(h_\ell) \;=\; \text{softmax}\!\big(\,W_U\,\text{norm}(J_\ell\, h_\ell)\,\big)\;}
$$

This produces a score for **every token** in the vocabulary. Sorting and inspecting the top entries gives a human-readable description of the activation: the short list of words that the activation is, on average across contexts, disposed to make the model say.

### 2.4 J-lens vectors

The rows of $W_U J_\ell$ are the **Jacobian-lens vectors** at layer $\ell$. Each J-lens vector is:

- a **direction in residual-stream space**, and
- associated with a **single token** in the model's vocabulary.

The (approximate, up to a data-dependent normalization) pre-softmax logit for token $t$ is the inner product $\langle v_t, h_\ell\rangle$ between the J-lens vector $v_t$ and the activation. So "how strongly is concept $t$ present in $h_\ell$?" reduces to a projection onto $v_t$.

### 2.5 Relationship to related lenses

| Method            | Per-layer map                                                | Objective                                     | Behavior                                                     |
| ----------------- | ------------------------------------------------------------ | --------------------------------------------- | ------------------------------------------------------------ |
| **Logit lens**    | $J_\ell = I$                                                 | none (identity)                               | fine in late layers, degrades early                          |
| **Tuned lens**    | trained linear map                                           | match output distribution (**correlational**) | tends to "skip ahead" to the output on prompts with unverbalized intermediates |
| **Jacobian lens** | $J_\ell = \mathbb{E}[\partial h_{\text{final}}/\partial h_\ell]$ | average **causal** linear map                 | recovers interpretable content at depths where logit lens fails |

The logit lens is the special case $J_\ell = I$; the J-lens is the *principled correction* — $J_\ell$ is precisely the average linear map relating layer-$\ell$ directions to their final-layer counterparts. The two agree closely in the last several layers and diverge earlier. The tuned lens fits a linear map too, but to match the output *distribution* (correlational), which makes it skip past exactly the intermediate computations we care about. (The method is related to Hernandez et al.'s use of Jacobians for per-relation subject→object maps; the J-lens applies the same first-order approximation to the map from activations to *outputs* rather than to a single relation.)

---

## 3. From the lens to the J-space

### 3.1 An overcomplete frame

At each layer the J-lens vectors form an **overcomplete set**: there are $n_{\text{vocab}}$ of them living in $d_{\text{model}}$-dimensional space, with $n_{\text{vocab}} > d_{\text{model}}$. Consequences:

- They may **linearly span the entire residual stream** (not a lower-dimensional subspace).
- Because they are overcomplete, there is **no unique** way to write a given activation as a combination of them — many decompositions exist.

### 3.2 Sparsity and the definition of the J-space

Empirically, only a **small number** of J-lens vectors are strongly active at any moment. This motivates defining the J-space as the set of points expressible as a **sparse, non-negative** combination of J-lens vectors:

- A sparsity level $k$ must be chosen. It is somewhat arbitrary; the paper varies it but typically uses $k \leq 25$, the number of vectors observed to be meaningfully active at once.
- Geometrically, for a given $k$ the J-space is a **union of $k$-dimensional cones**, one per possible choice of $k$ J-lens vectors.
- For any activation, its **J-space component** is the nearest point in the J-space; the **non-J-space component** is the remainder.

Under the **superposition hypothesis**, a model's activations decompose as sparse combinations from an overcomplete *sparse frame* of feature directions. The J-lens vectors are then a **token-indexed subframe** of that feature frame; the many feature directions *outside* the subframe make up the bulk of the model's representations.

### 3.3 The J-space is small

A recurring empirical fact, established via sparse decomposition: the J-space component typically accounts for **less than ~10% of an activation's variance** (and only ~6–7% of a concept vector's variance). The workspace is a *thin slice* of the model's total representational content — which is exactly what one expects of a selective workspace.

---

## 4. How the J-lens is used

The lens is used in two modes: **reading** (which concepts an activation carries) and **writing** (editing concepts into or out of an activation).

### 4.1 Reading

- **Basic readout** — replace downstream layers with $J_\ell$ and unembed: $\text{lens}(h_\ell) = \text{softmax}(W_U\,\text{norm}(J_\ell h_\ell))$. Sort to get the "top lens tokens" at a position.
- **Per-token probe** — read the score / cosine similarity of $h_\ell$ against a *single* chosen vector $v_t$, without ranking the whole vocabulary. Used to test whether a specific concept is present above a threshold.
- **Sparse decomposition** — solve (by **gradient pursuit**) for a sparse non-negative combination of $k$ J-lens vectors that best reconstructs $h_\ell$. Because the vectors are overcomplete and non-orthogonal, this yields a less-redundant discrete inventory than simply taking top-$k$ by inner product. Underlies the occupancy and fraction-of-variance analyses.

### 4.2 Writing

- **Steering** — add a lens vector: $h \leftarrow h + \alpha\, v_t$, at one or more layers/positions. Used to inject a concept and test introspective detection.
- **Ablation** — negative $\alpha$, or project out the component of $h$ along $v_t$ entirely. Used to suppress a particular concept, or (top-$k$) to suppress the whole J-space content at a position.
- **Patching in lens coordinates** — swap one concept for another while leaving everything else fixed. Given source token $s$ and target token $t$, form $V = [\,v_s \;\; v_t\,]$, read the coordinates $c = V^{\dagger} h$ (with $V^\dagger$ the pseudoinverse), and set

$$
h_{\text{patched}} = h + V\big(\sigma(c) - c\big),
$$

where $\sigma$ swaps the two entries of $c$ (optionally scaled by $\alpha$). The component of $h$ orthogonal to $\text{span}\{v_s, v_t\}$ is left **unchanged** — this precision is what makes the swap a clean causal test.

> **Note on reporting conventions.** Results are reported over 25 evenly spaced layers, reindexed to $[0,100]$ so layer numbers read as percentages of depth. Default model is **Claude Sonnet 4.5**, with key results corroborated on Haiku 4.5 and Opus 4.5 (and some analyses on Opus 4.6).

---

## 5. Key results: the J-space behaves like a global workspace

The authors define a representational subset as "workspace-like" if it satisfies five functional properties. The J-lens was built to find representations satisfying only the **first** (verbalizability); the striking finding is that the same representations satisfy the other four.

### 5.1 Verbal report

- Swapping the J-lens vector of a spontaneously chosen item for another (e.g. **Soccer → Rugby**) changes the model's spoken answer accordingly. Across categories, the ordering of lens tokens correlates strongly with the ordering of reported words, and the correlation increases toward the end of the workspace layers.
- Injecting a concept vector across the user turn makes it **reportable when the model introspects**, without making the model blurt the word out at other positions — i.e. these are *verbalizable-under-the-right-conditions* representations, not unconditional urges to speak.
- **Privilege test.** A concept's full representation was split into a J-space component and a non-J-space remainder. The J-space component holds only **~6–7%** of the variance, yet drives verbal report: swapping along it hits top-5 on **59%** of trials (vs. **88%** for raw J-lens vectors), while the non-J-space component — carrying ~93% of the variance — succeeds only **5%** of the time. Clamping the J-space coordinates drops the residual non-J-space effect to zero, showing that what little effect it has is itself *routed through* the J-space.

### 5.2 Directed modulation

- Told to "concentrate on citrus fruits" while copying an unrelated sentence, the model shows **orange / lemon** in the lens at the unrelated token — plus abstract markers of the act itself (*imagine, thinking, focused*). Told to mentally compute $3^2 - 2$, the lens progresses **arithmetic → nine (intermediate) → seven (answer)**.
- Systematically, "think about X" loads the target into the lens on a substantial fraction of trials (increasing with model size); the no-instruction baseline is ~zero. "Ignore X" suppresses the target well below the focus condition but keeps it **above** baseline — a model analog of the human **"white bear"** ironic-suppression effect.
- Modulation also responds to **implicit** task demands: asking the model to *name* a property (part of speech, tense) surfaces its label in the lens, whereas merely requiring the property for next-word prediction does not — even though both conditions respect the property.

### 5.3 Internal reasoning

The lens exposes **unspoken intermediate steps**, and intervening on them redirects the conclusion:

- *"legs on the animal that spins webs"* → **spider** appears in the lens (never in text); swapping **spider → ant** flips the output **8 → 6**.
- Rhyme planning: for a couplet, **fight** is the planned rhyme; swapping **fight → light** changes an earlier word choice from "coming" to "morning" — evidence of genuine planning.
- Cross-lingual: a Chinese antonym task routes through **English** intermediates (*big/bigger*); swapping these to *long/longer* changes the Chinese output 大 → 长.
- Reward-driven choice: **repeat/switch** strategy tokens appear per condition; swapping them flips the model's A/B choice.
- **Systematic two-hop swaps** succeed on **54% (Haiku 4.5), 70% (Sonnet 4.5), 70% (Opus 4.5)**. The intermediate takes effect ~**17% of depth earlier** than the answer — ruling out the confound that the swap merely smuggles in the answer.
- **Privilege test (again):** swapping the J-space component of an intermediate probe flips the answer on **61%** of trials (matching **60%** for raw lens vectors); the non-J-space remainder manages only **28%**, falling to **6%** once J-space coordinates are clamped.
- Multi-step arithmetic *"( 4 + 17 ) \* 2 + 7 ="* surfaces **21 → 42 → 49** in the lens at **successively later layers**, in exactly the order the computation requires — confirmed causally by activation patching.

### 5.4 Flexible generalization (broadcast)

- A single lens vector serves as a valid argument to many downstream operations. Swapping **France → China** across many function templates ("the capital of ___", "most people in ___ speak", "___ is on the continent of") makes each downstream circuit return China's answer.
- Systematically: **76 / 192** swaps reach top-1 (**101 / 192** at double strength). Success is well predicted by **"workspace loading"** (the cosine similarity of the residual stream with the concept's lens vector): countries load high and swap reliably; number words load low and swap poorly.

### 5.5 Selectivity (automatic vs. flexible cognition)

- With a Spanish passage, swapping **Spanish → French** in the lens changes **explicit report** and **flexible inference** (famous author, word for "hello", pre-Euro currency) but leaves **continuation** and **anomaly detection** untouched — even though "Spanish" appears in the lens in all four conditions. Automatic tasks use the information without routing it through the J-space.
- Character counting: number tokens are **absent** from the lens under the automatic line-wrapping task, **present** when the count is explicitly asked for, and **most present** when the count is an intermediate for a further operation — i.e. information is **pulled into** the J-space on demand.
- **Whole-J-space ablation** (zero the top-$k$ lens directions across a layer band, excluding tokens the model was about to output): shallow tasks (MMLU, odd-one-out, SQuAD extractive QA, sentiment, CoLA) are **essentially unaffected**; tasks needing inferred content (Caesar cipher, analogies, summarization, TriviaQA, multi-hop, translation, sonnet writing) fall **below unablated Haiku 4.5**. GSM8K with **chain-of-thought is far more robust** than direct answers — the model externalizes onto the page what it would otherwise carry in the workspace.
- Ablation also **flattens experiential/sensory language** (the model's stream-of-consciousness becomes mechanical) while preserving coherence — and the same collapse happens when describing *another person's* experience, so it is not specific to self-report.

---

## 6. Key results: the J-space's *structure* supports its function

Three structural signatures, established by looking at the J-space as an object in the model (not just via its behavioral effects):

**1. Layer bands.** CKA on the J-lens geometry across layers reveals a three-block structure the authors label **sensory (early ~first third) → workspace (middle, ~L38–92) → motor (late)**. Multiple lens-derived statistics (next-token accuracy, excess kurtosis, cross-position autocorrelation, effective dimensionality) all mark the same onset (~L38) and offset (~L92). At the workspace onset, the model's interpretation of **artificially ambiguous inputs** snaps from smoothly graded to near all-or-none — an **"ignition"-like** transition predicted by global workspace theory.

**2. Limited capacity.** J-space **occupancy** rises to a plateau of ~**25** active directions across the workspace band, explaining **<10%** of activation variance; only a small fraction of SAE features align with it (and those that don't are dominated by low-level syntactic/bookkeeping features). Word-list experiments show the workspace holds ~**6 unrelated** items at once but nearly a whole **80-word category** when the items share a category — it represents the *shared category*, not each entry. New categories **evict** old contents within a few tokens.

**3. Broadcast hub.** J-lens directions compose with the model's weights far more broadly than other directions — consistent with a format that many circuits read and write:

- **Across depth:** MLP blocks amplify J-lens directions ~**10×** through the workspace range (vs. ~1× for neuron output directions); the most J-space-aligned SAE features are amplified most.
- **Across tokens:** a distinctive top-1% of attention heads selectively **relays** J-space content between positions (no comparable set exists for control populations). Ablating these "broadcast heads" degrades the J-lens readout and downstream workspace-dependent behaviors (injected-thought reporting drops **0.54 → 0.09**; ~a third of the experiential-language drop is reproduced) while barely changing next-token predictions.

---

## 7. Key results: applications

- **Alignment auditing.** On alignment-evaluation scenarios, the J-space carries strategic and emotional cognition absent from the output (*leverage, manipulation, panic*), and sometimes encodes **evaluation-awareness** (*fake, fictional*). Ablating those evaluation-awareness representations can **surface otherwise-concealed misaligned propensities**. On two model organisms with implanted misaligned objectives, the lens exposes tell-tale internal concepts (*reward, bias*; *secretly, trick*) on ordinary prompts.
- **The Assistant's perspective (base vs. post-trained).** Post-training gives the J-space the **Assistant's point of view**: Assistant-style reactions (empathy, safety concerns) appear while the model is still reading the user's message, along with traces of self-monitoring (an internal *BUT* when prefilled to act against its preferences; *damn* when it fails to suppress a forbidden thought; flagging its own roleplay as *fictional*).
- **Counterfactual reflection training.** Motivated directly by the workspace account's prediction that internal reasoning routes through representations of *things the model might say*: train the model to **articulate ethical principles if interrupted and asked to reflect**, and its behavior improves in the original *uninterrupted* contexts — with no direct training of the behavior itself. After training, the workspace in those contexts fills with the relevant concepts (*ethical, honest, integrity*), and **ablating those implanted representations reverts** the improvement — corroborating that report-representations and reasoning-representations are the same.

---

## 8. Limitations to keep in mind

- The J-lens only captures concepts corresponding to **single tokens** in the vocabulary; many important concepts are multi-token (extensions exist but the core method is single-token). This "vocabulary restriction" likely explains failures such as the poor swapping of small-integer number words.
- The J-lens is an **imperfect, approximate** probe of the underlying workspace: in some analyses the most J-space-aligned SAE features behave *more* workspace-like than the J-lens vectors themselves, suggesting the lens under-captures the "true" workspace.
- Readouts in roughly the **first third** of layers are noisy and largely uninterpretable — it remains possible that part of the model's real workspace operates earlier than the lens can resolve.
- The analogy to the brain's global workspace is **partial**: the paper documents the *functional* properties and some structural ones, but not encapsulated competing modules, and the "broadcast" occurs across depth in a single forward pass rather than via recurrence. The authors take **no position** on phenomenal consciousness.

---

## 9. One-paragraph summary

The **Jacobian lens** characterizes an intermediate activation by the *average, context-independent, first-order causal effect* it has on the model's present and future outputs — formally, the corpus-averaged Jacobian $J_\ell = \mathbb{E}[\partial h_{\text{final}}/\partial h_\ell]$ composed with the unembedding, giving a per-token "verbalizability" readout $\text{softmax}(W_U\,\text{norm}(J_\ell h_\ell))$. It is the principled generalization of the logit lens (which assumes $J_\ell = I$). The lens vectors form an overcomplete, token-indexed subframe whose sparse non-negative span defines the **J-space**, a thin (<10% of variance) slice of the residual stream. Reading (rankings, per-token probes, sparse decomposition) and writing (steering, ablation, coordinate-swaps) with the lens show that this slice is **reportable, controllable, usable for unspoken multi-step reasoning, broadcastable to many downstream operations, and engaged selectively for flexible rather than automatic cognition** — the five functional hallmarks of a global workspace — and that it is **structurally privileged** (confined to a middle layer band, capacity-limited, and disproportionately amplified and relayed by the model's weights). The same representations that govern what a model *says* turn out to govern how it silently *thinks*.