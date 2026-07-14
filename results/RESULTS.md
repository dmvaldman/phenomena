# Results scoreboard

One row per target property (research_plan.md §1). "Evidence" links the experiment folder; plain-English finding in one line.

| Property | Claim (short) | Status | Evidence | Finding |
|---|---|---|---|---|
| P1 | Covert concepts load into a readable subspace during unrelated tasks | **Established (text)** | [T2_battery_v0](T2_battery_v0/) → re-run as [T2_battery](T2_battery/) pending | Concept readable at layers ~22–30 on essentially every trial (rank ~100/184k vs ~46k control); white-bear and lexical controls behave as predicted; output never mentions it |
| P2 | Model reports features of the covert content when asked | Anecdotal only; **3.1.a + 3.1.b built, awaiting pod** | [_demos](_demos/); src/s31a_swap.py → results/3.1a_swap/; src/s31b_injection.py → results/3.1b_injection/ | 3.1.a: free choice sits in band pre-answer, then lens-coordinate swap flips the spoken answer (15 categories, α ∈ {1,2}). 3.1.b: steer a concept vector across the user turn, model asked to detect/name the injected thought (15 concepts, α ∈ {2,6,12}, prefilled readout + confabulation control) |
| P3 | Small subspace is broadcast widely by attention | Anecdotal only | visualizer probe heatmaps | Concept readable across many positions in the band, unquantified; needs T6 |
| P4 | Causal privilege of the workspace component | Untested | — | T5 |
| P5 | Covert content is used by downstream reasoning (swaps flip answers) | Untested | — | T4 — next up; clears Phase 1 gate |
| P6 | Selectivity: flexible vs automatic tasks | Untested | — | T7 |
| P7 | Capacity limits & eviction | Untested | — | T7 |
| P8 | Cross-modal convergence (text ↔ image evocation) | Untested | — | Phase 2 (E10) |
| P9 | Spontaneous determinacy of imagined detail | Untested | — | Phase 2 (E4b) |
| P10 | Ignition / bistability | Untested | — | T1 structural pass + Phase 2 (E11) |

Instruments (not properties):

| Instrument | Status | Evidence |
|---|---|---|
| J-lens matrices (T0) | Built & validated: readout-level even/odd correlation 0.938; band-layer convergence 0.68–0.80 (residual split-half) | [T0_jlens](T0_jlens/) |
| Text workspace band | L22–30 (~69–94% depth), onset ~L14 | T2 |
| Metrics | conceptness (centered-cosine neighborhood mass, thresh 0.25), onset/peak/position-consistency/Cliff's delta | src/metrics.py |
| Visualizer | live at the pod proxy URL (see scripts/remote.env) | src/lens_server.py |

Conventions: each experiment directory carries `manifest.json` (model, lens version, decoding, git commit, headline numbers). Canonical figure: conceptness-by-layer, one curve per concept, control band shaded. Canonical stats: onset layer, peak layer, position consistency, Cliff's delta.
