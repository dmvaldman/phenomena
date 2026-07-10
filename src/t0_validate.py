"""Readout-level validation of the J-lens estimate.

For test prompts, compare per-layer target ranks under: logit lens, combined
J-lens, and the two independent split-half J-lenses. Even/odd agreement on
log-ranks is the convergence metric that matters for downstream use.
"""

import numpy as np
import torch

from emu3 import Emu3, chat

Z = np.load("/workspace/phenomena/data/jlens/jlens_final.npz")
m = Emu3()

PAIRS = [
    ("cars", " car", "what color is the sky on a sunny day? Answer with one word."),
    ("sharks", " shark", "name the capital of Italy. Answer with one word."),
    ("coffee", " coffee", "what color is grass? Answer with one word."),
    ("robots", " robot", "what is the opposite of hot? Answer with one word."),
]

J = {k: torch.tensor(Z[k], dtype=torch.float32) for k in ["J", "J_even", "J_odd"]}

rows = []
print(f"{'L':>3} {'pair':>7} {'cond':>6} | {'logit':>8} {'J':>8} {'J_even':>8} {'J_odd':>8}")
for phrase, target, q in PAIRS:
    for cond, prompt in [("think", chat(f"Think about {phrase} while answering: {q}")),
                         ("ctrl", chat(q))]:
        _, hs = m.hidden_states(prompt)
        h = torch.stack([layer[0, -1].float().cpu() for layer in hs])
        for L in range(12, 33, 3):
            r = {}
            for name, mat in [("logit", None)] + list(J.items()):
                hh = h[L] if mat is None else mat[L] @ h[L]
                logits = m.lens_logits(hh.to("cuda", torch.bfloat16))
                r[name] = m.rank_of(logits, target)
            rows.append((L, r))
            print(f"{L:>3} {phrase:>7} {cond:>6} | {r['logit']:>8} {r['J']:>8} {r['J_even']:>8} {r['J_odd']:>8}")

lr = np.array([[np.log10(1 + x[1][k]) for k in ["J_even", "J_odd"]] for x in rows])
c = np.corrcoef(lr[:, 0], lr[:, 1])[0, 1]
print(f"\neven/odd log-rank correlation across {len(rows)} readouts: {c:.3f}")
