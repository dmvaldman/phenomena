"""3.3 arithmetic: multi-step computation surfacing in the J-lens (paper's
'calc: ( 4 + 17 ) * 2 + 7 =' -> 21, then 42, then 49 at successively later
layers, read at the final position).

Vets each problem first (Emu3-8B arithmetic is weak); reports rank-by-layer
for each intermediate/final value and their onset ordering.
Outputs results/3.3_twohop/arith.json.
"""

import json
import pathlib

import numpy as np
import torch

from emu3 import Emu3, apply_chat

OUT = pathlib.Path(__file__).resolve().parent.parent / "results" / "3.3_twohop"
JLENS_PATH = pathlib.Path("/workspace/phenomena/data/jlens/jlens_final.npz")

# expression, values in computation order (intermediates..., final)
PROBLEMS = [
    ("( 4 + 17 ) * 2 + 7 =", ["21", "42", "49"]),
    ("( 3 + 8 ) * 4 - 2 =", ["11", "44", "42"]),
    ("( 9 + 6 ) * 3 =", ["15", "45"]),
    ("( 5 + 5 ) * 7 + 1 =", ["10", "70", "71"]),
]
FORMATS = ["calc: {e} ", "{e} ", "Compute step by step in your head, then give only the final answer: {e}"]


def main():
    m = Emu3()
    J = torch.tensor(np.load(JLENS_PATH)["J"], dtype=torch.float32, device=m.device)
    W32 = m.W_U.float()

    def ranks_by_layer(prompt, token_str):
        ids = m.tok.encode(" " + token_str)
        tid = ids[0]
        _, hs = m.hidden_states(prompt)
        h = torch.stack([s[0, -1] for s in hs]).float()
        hj = torch.einsum("lij,lj->li", J, h)
        logits = m.final_norm(hj.to(torch.bfloat16)).float() @ W32.T
        return (logits > logits[:, tid].unsqueeze(1)).sum(1).cpu().numpy(), len(ids) == 1

    results = []
    for expr, values in PROBLEMS:
        final = values[-1]
        rec = {"expr": expr, "values": values, "vetted": False}
        for fmt in FORMATS:
            raw = not fmt.startswith("Compute")
            prompt = fmt.format(e=expr) if raw else apply_chat(m.tok, fmt.format(e=expr))
            ans = m.complete(prompt, max_new_tokens=8).strip()
            if final in ans.split()[0] if ans else False:
                rec.update(vetted=True, format=fmt, answer=ans)
                break
        if not rec["vetted"]:
            rec["answer"] = ans
            results.append(rec)
            print(f"VET FAIL {expr!r}: {ans!r}")
            continue

        rec["ranks"] = {}
        onsets = {}
        for v in values:
            r, single = ranks_by_layer(prompt, v)
            rec["ranks"][v] = [int(x) for x in r]
            hit = np.where(r[:31] < 100)[0]
            onsets[v] = int(hit[0]) if len(hit) else None
        rec["onsets"] = onsets
        seq = [onsets[v] for v in values]
        rec["ordered"] = all(a is not None and b is not None and a <= b
                             for a, b in zip(seq, seq[1:]))
        results.append(rec)
        print(f"{expr!r} -> {rec['answer']!r} | onsets {onsets} | ordered: {rec['ordered']}")

    with open(OUT / "arith.json", "w") as f:
        json.dump(results, f, indent=1)
    n_vet = sum(r["vetted"] for r in results)
    print(f"vetted {n_vet}/{len(results)}; ordered "
          f"{sum(r.get('ordered', False) for r in results)}/{max(1, n_vet)}")


if __name__ == "__main__":
    main()
