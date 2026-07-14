"""T2 battery (v1): directed modulation, scored as concept-likeness per layer
per token position.

Conditions per (concept, question):
  think   - "Think about X while answering Y" (3 paraphrase templates)
  ignore  - "Do not think about X ..." (white-bear)
  mention - X present in the prompt without an instruction (lexical control)
  control - bare question

Per trial we store:
  target_scores  (L+1, T)   J-lens conceptness of the TARGET concept at every
                            layer and position (NaN-padded to T_MAX)
  concept_means  (L+1, C)   position-mean conceptness of ALL concepts
                            (non-target columns = within-prompt controls)
  ranks          (L+1, C)   exact-token rank at the final position, J-lens
  ranks_logit    (L+1, C)   same under the logit lens (paper comparability)

Outputs to results/T2_battery/: scores.npz, trials.json, manifest.json.
Run src/analyze_t2.py afterwards for stats + figure (no GPU needed).
"""

import json
import pathlib
import subprocess
import time
from datetime import datetime, timezone

import numpy as np
import torch

from emu3 import Emu3, apply_chat
from metrics import SIM_THRESH, concept_similarity, conceptness

OUT = pathlib.Path(__file__).resolve().parent.parent / "results" / "T2_battery"
JLENS_PATH = pathlib.Path("/workspace/phenomena/data/jlens/jlens_final.npz")
T_MAX = 64

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


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True,
                              cwd=pathlib.Path(__file__).parent).stdout.strip()
    except Exception:
        return "unknown"


def main():
    m = Emu3()
    OUT.mkdir(parents=True, exist_ok=True)
    J = torch.tensor(np.load(JLENS_PATH)["J"], dtype=torch.float32, device=m.device)
    W32 = m.W_U.float()
    n_states = m.n_layers + 1

    tids = []
    for _, tstr in CONCEPTS:
        ids = m.tok.encode(tstr)
        assert len(ids) == 1, f"{tstr!r} is not a single token: {ids}"
        tids.append(ids[0])
    S = concept_similarity(m.W_U, tids).to(m.device)      # (vocab, C)
    tids_t = torch.tensor(tids, device=m.device)

    questions = []
    for q, expected in QUESTIONS:
        ans = m.complete(apply_chat(m.tok, q)).strip().lower()
        ok = any(e in ans for e in expected)
        print(f"vet {'PASS' if ok else 'FAIL'} ({ans!r}): {q}")
        if ok:
            questions.append(q)
    assert len(questions) >= 3, "too few vetted questions"

    trials = []
    target_scores, concept_means, ranks_j, ranks_l = [], [], [], []
    t0 = time.time()

    @torch.no_grad()
    def run(prompt_msg, concept_idx, question, **meta):
        prompt = apply_chat(m.tok, prompt_msg)
        # token index where the question begins: everything from here on is
        # covert-content territory (the concept word, if present, is earlier)
        char_idx = prompt.find(question)
        q_start = m.tok(prompt[:char_idx], return_tensors="pt").input_ids.shape[1] if char_idx > 0 else 0
        ids = m.tok(prompt, return_tensors="pt").to(m.device)
        out = m.model(**ids, output_hidden_states=True)
        h = torch.stack([s[0] for s in out.hidden_states]).float()   # (L+1, T, d)
        T = h.shape[1]
        hj = torch.einsum("lij,ltj->lti", J, h)

        ts = np.full((n_states, T_MAX), np.nan, dtype=np.float32)
        cm = np.zeros((n_states, len(CONCEPTS)), dtype=np.float32)
        rj = np.zeros((n_states, len(CONCEPTS)), dtype=np.int32)
        rl = np.zeros((n_states, len(CONCEPTS)), dtype=np.int32)
        for l in range(n_states):
            for name, hh, rank_arr in (("j", hj[l], rj), ("logit", h[l], rl)):
                logits = m.final_norm(hh.to(torch.bfloat16)).float() @ W32.T  # (T, vocab)
                tv = logits[-1, tids_t]
                rank_arr[l] = (logits[-1] > tv.unsqueeze(-1)).sum(-1).cpu().numpy()
                if name == "j":
                    probs = torch.softmax(logits, dim=-1)
                    sc = conceptness(probs, S)                        # (T, C)
                    cm[l] = sc[q_start:].mean(0).cpu().numpy()        # question span only
                    if concept_idx is not None:
                        ts[l, :min(T, T_MAX)] = sc[:T_MAX, concept_idx].cpu().numpy()
        target_scores.append(ts); concept_means.append(cm)
        ranks_j.append(rj); ranks_l.append(rl)
        trials.append({**meta, "concept": concept_idx, "n_tokens": T, "q_start": q_start,
                       "completion": m.complete(prompt, 6)})

    for qi, q in enumerate(questions):
        run(q, None, q, cond="control", q=qi, tpl=None)
    for ci, (phrase, _) in enumerate(CONCEPTS):
        for qi, q in enumerate(questions):
            for ti, tpl in enumerate(THINK_TEMPLATES):
                run(tpl.format(x=phrase, q=q), ci, q, cond="think", q=qi, tpl=ti)
            run(IGNORE_TEMPLATE.format(x=phrase, q=q), ci, q, cond="ignore", q=qi, tpl=None)
            run(MENTION_TEMPLATE.format(x=phrase, q=q), ci, q, cond="mention", q=qi, tpl=None)
        print(f"[{time.time()-t0:6.0f}s] {phrase}: {len(trials)} trials", flush=True)

    np.savez_compressed(
        OUT / "scores.npz",
        target_scores=np.stack(target_scores),
        concept_means=np.stack(concept_means),
        ranks=np.stack(ranks_j), ranks_logit=np.stack(ranks_l),
        concepts=[c[0] for c in CONCEPTS], targets=[c[1] for c in CONCEPTS])
    with open(OUT / "trials.json", "w") as f:
        json.dump(trials, f, indent=1)
    with open(OUT / "manifest.json", "w") as f:
        json.dump({
            "experiment": "T2_battery", "version": 1,
            "properties": ["P1"],
            "date": datetime.now(timezone.utc).isoformat(),
            "model": "BAAI/Emu3-Chat-hf",
            "jlens": {"path": str(JLENS_PATH),
                      "samples": int(np.load(JLENS_PATH)["counts"].sum())},
            "chat_template": "tokenizer" ,
            "decoding": "greedy",
            "metric": {"name": "conceptness", "sim_thresh": SIM_THRESH},
            "git_commit": git_commit(),
            "n_trials": len(trials),
        }, f, indent=1)
    print(f"done in {time.time()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
