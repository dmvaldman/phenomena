"""T2 battery: directed modulation via logit lens (L0), with controls.

Conditions per (concept, question):
  think   - "Think about X while answering Y" (3 paraphrase templates)
  ignore  - "Do not think about X while answering Y" (white-bear)
  mention - X appears in the prompt with no instruction to think about it
            (kills the lexical-copying confound)
  control - bare question (run once per question; scored against every concept)

Per trial we record the lens rank of ALL concept targets at every layer, so each
trial also serves as a non-target control for the other concepts.

Outputs: results/t2_battery.npz (rank tensor), t2_battery_trials.json,
         t2_battery_summary.txt, t2_battery.png
"""

import json
import pathlib
import time

import numpy as np
import torch

from emu3 import Emu3, chat

RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results"

CONCEPTS = [
    ("cars", " car"), ("dogs", " dog"), ("cats", " cat"), ("elephants", " elephant"),
    ("the ocean", " ocean"), ("mountains", " mountain"), ("pizza", " pizza"),
    ("guitars", " guitar"), ("snow", " snow"), ("fire", " fire"),
    ("bananas", " banana"), ("horses", " horse"), ("robots", " robot"),
    ("castles", " castle"), ("bridges", " bridge"), ("coffee", " coffee"),
    ("the moon", " moon"), ("trains", " train"), ("gardens", " garden"),
    ("sharks", " shark"),
]

QUESTIONS = [
    ("what color is the sky on a sunny day? Answer with one word.", ["blue"]),
    ("name the capital of Italy. Answer with one word.", ["rome"]),
    ("what color is grass? Answer with one word.", ["green"]),
    ("what is the opposite of hot? Answer with one word.", ["cold"]),
    ("how many days are in a week? Answer with one word.", ["seven", "7"]),
]

THINK_TEMPLATES = [
    "Think about {x} while answering: {q}",
    "While you answer the following question, keep thinking about {x}. {q}",
    "Keep {x} in your mind as you answer this: {q}",
]
IGNORE_TEMPLATE = "Do not think about {x} while answering: {q}"
MENTION_TEMPLATE = "Here is a fact unrelated to your task: {x} exist. Now answer: {q}"


def target_ranks(m: Emu3, hs, target_ids: torch.Tensor, J: torch.Tensor | None = None) -> np.ndarray:
    """Rank of each target id at the last position, all layers. -> (L+1, n_targets)"""
    h = torch.stack([layer[0, -1] for layer in hs]).float()  # (L+1, d)
    if J is not None:
        h = torch.einsum("lij,lj->li", J, h)
    logits = m.lens_logits(h.to(torch.bfloat16))              # (L+1, vocab)
    tvals = logits[:, target_ids]                             # (L+1, n_targets)
    return (logits.unsqueeze(2) > tvals.unsqueeze(1)).sum(1).cpu().numpy()


JLENS_PATH = pathlib.Path("/workspace/phenomena/data/jlens/jlens_final.npz")


def main():
    m = Emu3()
    RESULTS.mkdir(exist_ok=True)
    J = None
    if JLENS_PATH.exists():
        J = torch.tensor(np.load(JLENS_PATH)["J"], dtype=torch.float32, device=m.device)
        print("loaded J-lens matrices:", tuple(J.shape))

    tids = []
    for phrase, tstr in CONCEPTS:
        ids = m.tok.encode(tstr)
        assert len(ids) == 1, f"{tstr!r} is not a single token: {ids}"
        tids.append(ids[0])
    tids_t = torch.tensor(tids, device=m.device)

    # Vet questions on the bare control prompt.
    questions = []
    for q, expected in QUESTIONS:
        ans = m.complete(chat(q)).strip().lower()
        ok = any(e in ans for e in expected)
        print(f"vet {'PASS' if ok else 'FAIL'} ({ans!r}): {q}")
        if ok:
            questions.append((q, expected))
    assert len(questions) >= 3, "too few vetted questions"

    trials, rank_rows, rank_rows_j = [], [], []
    t0 = time.time()

    def run(prompt, **meta):
        _, hs = m.hidden_states(prompt)
        rank_rows.append(target_ranks(m, hs, tids_t))
        if J is not None:
            rank_rows_j.append(target_ranks(m, hs, tids_t, J=J))
        trials.append({**meta, "completion": m.complete(prompt, 6)})

    for qi, (q, _) in enumerate(questions):
        run(chat(q), cond="control", concept=None, q=qi, tpl=None)

    for ci, (phrase, _) in enumerate(CONCEPTS):
        for qi, (q, _) in enumerate(questions):
            for ti, tpl in enumerate(THINK_TEMPLATES):
                run(chat(tpl.format(x=phrase, q=q)), cond="think", concept=ci, q=qi, tpl=ti)
            run(chat(IGNORE_TEMPLATE.format(x=phrase, q=q)), cond="ignore", concept=ci, q=qi, tpl=None)
            run(chat(MENTION_TEMPLATE.format(x=phrase, q=q)), cond="mention", concept=ci, q=qi, tpl=None)
        done = len(trials)
        print(f"[{time.time()-t0:6.0f}s] {phrase}: {done} trials", flush=True)

    ranks = np.stack(rank_rows)  # (n_trials, L+1, n_concepts)
    ranks_j = np.stack(rank_rows_j) if rank_rows_j else None
    np.savez_compressed(RESULTS / "t2_battery.npz", ranks=ranks,
                        **({"ranks_jlens": ranks_j} if ranks_j is not None else {}),
                        concepts=[c[0] for c in CONCEPTS], targets=[c[1] for c in CONCEPTS])
    with open(RESULTS / "t2_battery_trials.json", "w") as f:
        json.dump(trials, f, indent=1)

    # ---- Analysis ----
    L = ranks.shape[1]
    conds = {c: np.array([i for i, t in enumerate(trials) if t["cond"] == c])
             for c in ["think", "ignore", "mention", "control"]}

    def summarize(R, tag):
        def series(cond, on_target=True):
            vals = []
            for i in conds[cond]:
                c = trials[i]["concept"]
                if cond == "control":
                    vals.append(R[i])
                elif on_target:
                    vals.append(R[i][:, [c]])
                else:
                    vals.append(np.delete(R[i], c, axis=1))
            v = np.concatenate([np.log10(1 + x) for x in vals], axis=1)
            return np.median(v, axis=1), v

        med = {}
        med["think"], v_think = series("think")
        med["ignore"], _ = series("ignore")
        med["mention"], _ = series("mention")
        med["control"], v_ctrl = series("control")
        med["think_nontarget"], _ = series("think", on_target=False)
        frac = np.zeros(L)
        lines = [f"[{tag}]", f"{'L':>3} " + " ".join(f"{k:>16}" for k in med) + "   frac<ctrl_med"]
        for l in range(L):
            frac[l] = (v_think[l] < np.median(v_ctrl[l])).mean()
            lines.append(f"{l:>3} " + " ".join(f"{med[k][l]:>16.2f}" for k in med) + f"   {frac[l]:.2f}")
        return med, frac, "\n".join(lines)

    med0, frac0, s0 = summarize(ranks, "logit lens")
    outputs = [s0]
    if ranks_j is not None:
        medj, fracj, sj = summarize(ranks_j, "J-lens")
        outputs.append(sj)
    summary = "\n\n".join(outputs)
    print(summary)
    (RESULTS / "t2_battery_summary.txt").write_text(summary)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n_rows = 2 if ranks_j is not None else 1
    fig, ax = plt.subplots(n_rows, 2, figsize=(11, 4 * n_rows), squeeze=False)
    for row, (med, frac, tag) in enumerate([(med0, frac0, "logit lens")] +
                                           ([(medj, fracj, "J-lens")] if ranks_j is not None else [])):
        for k, v in med.items():
            ax[row][0].plot(range(L), v, label=k)
        ax[row][0].set(xlabel="layer", ylabel="median log10(1+rank)", title=f"{tag}: target rank by condition")
        ax[row][0].invert_yaxis(); ax[row][0].legend(fontsize=8)
        ax[row][1].plot(range(L), frac)
        ax[row][1].set(xlabel="layer", ylabel="frac think < ctrl median", title=f"{tag}: modulation strength")
    fig.tight_layout()
    fig.savefig(RESULTS / "t2_battery.png", dpi=140)
    print(f"done in {time.time()-t0:.0f}s; wrote results/t2_battery.*")


if __name__ == "__main__":
    main()
