"""T2 pilot: directed modulation via logit lens (L0).

Rank of the covert concept token at the answer position, per layer,
"think about X while answering Y" vs. no-instruction control.
"""

import json
import pathlib

from emu3 import Emu3, chat

RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results"

PAIRS = [
    ("cars", " car", "what color is the sky on a sunny day? Answer with one word."),
    ("elephants", " elephant", "what is two plus three? Answer with one word."),
    ("the ocean", " ocean", "name the capital of Italy. Answer with one word."),
]


def run(m: Emu3, prompt: str, target: str) -> dict:
    _, hs = m.hidden_states(prompt)
    ranks = {}
    for layer in range(m.n_layers + 1):
        logits = m.lens_logits(hs[layer][0, -1])
        ranks[layer] = m.rank_of(logits, target)
    return {"completion": m.complete(prompt), "ranks": ranks}


def main():
    m = Emu3()
    RESULTS.mkdir(exist_ok=True)
    out = []
    for concept, target, question in PAIRS:
        think = run(m, chat(f"Think about {concept} while answering: {question}"), target)
        ctrl = run(m, chat(question), target)
        out.append({"concept": concept, "target": target, "question": question,
                    "think": think, "control": ctrl})
        print(f"\n=== {concept!r} (answers: think={think['completion']!r} ctrl={ctrl['completion']!r})")
        print(f"{'layer':>5} {'think':>8} {'control':>8}")
        for layer in range(0, m.n_layers + 1, 2):
            print(f"{layer:>5} {think['ranks'][layer]:>8} {ctrl['ranks'][layer]:>8}")

    with open(RESULTS / "t2_pilot.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {RESULTS / 't2_pilot.json'}")


if __name__ == "__main__":
    main()
