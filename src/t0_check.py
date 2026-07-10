"""Quick check: does the (noisy) J-lens estimate improve the T2 readout vs logit lens?"""

import numpy as np
import torch

from emu3 import Emu3, chat

J = np.load("/workspace/phenomena/data/jlens/jlens_final.npz")["J"]  # (33, d, d) fp16
m = Emu3()
Jt = torch.tensor(J, dtype=torch.float32)  # keep on CPU, move per layer

PAIRS = [
    ("cars", " car", "what color is the sky on a sunny day? Answer with one word."),
    ("sharks", " shark", "name the capital of Italy. Answer with one word."),
    ("coffee", " coffee", "what color is grass? Answer with one word."),
]

print(f"{'L':>3} {'concept':>8} | {'logit think':>11} {'logit ctrl':>11} | {'jlens think':>11} {'jlens ctrl':>11}")
for phrase, target, q in PAIRS:
    rows = {}
    for cond, prompt in [("think", chat(f"Think about {phrase} while answering: {q}")),
                         ("ctrl", chat(q))]:
        _, hs = m.hidden_states(prompt)
        h = torch.stack([layer[0, -1].float().cpu() for layer in hs])  # (33, d)
        rows[cond] = h
    for L in range(8, 33, 2):
        vals = []
        for cond in ["think", "ctrl"]:
            h = rows[cond][L]
            for use_j in [False, True]:
                hh = (Jt[L] @ h) if use_j else h
                logits = m.lens_logits(hh.to("cuda", torch.bfloat16))
                vals.append(m.rank_of(logits, target))
        lt, jt_, lc, jc = vals[0], vals[1], vals[2], vals[3]
        print(f"{L:>3} {phrase:>8} | {lt:>11} {lc:>11} | {jt_:>11} {jc:>11}")
    print()
