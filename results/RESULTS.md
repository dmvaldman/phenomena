# Results scoreboard

One row per target property (research_plan.md §1). "Evidence" links the experiment folder; plain-English finding in one line.

| Property | Claim (short) | Status | Evidence | Finding |
|---|---|---|---|---|
| P1 | Covert concepts load into a readable subspace during unrelated tasks | **Established (text)** | [T2_battery](T2_battery/) (conceptness, all positions); [T2_battery_v0](T2_battery_v0/) (ranks, final position) | v1: pooled onset L17 (median), band L18–32, position consistency 0.36, Cliff's δ ≥ 0.9 for 17/20 concepts (weak: cars/dogs/cats). v0: rank ~100/184k vs ~46k control at L22–30; white-bear and lexical controls as predicted; output never mentions the concept |
| P2 | Model reports features of the covert content when asked | **Established (text, causal)** | [3.1a_swap](3.1a_swap/), [3.1b_injection](3.1b_injection/) | 3.1.a: 12/15 free choices in band top-100 pre-answer; lens-coordinate swap flips the spoken word 53% (α1.5) / 60% (α2), failures mostly multi-token words. 3.1.b: injected concept named 53% (user-turn injection only, α32; pure attention relay) and 93–100% (span through prefill); confabulation baseline "dog" ≈ 0%; near-misses are semantic neighbors (train→"Platform", fire→"fire hydrant") |
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
