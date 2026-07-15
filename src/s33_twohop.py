"""3.3: two-hop reasoning through the workspace (paper: 'The J-space mediates
internal reasoning'; property P5).

Per item: a question whose answer requires an unspoken intermediate entity.
  1. Vet: the model answers the question correctly (greedy).
  2. Presence: J-lens rank of the intermediate at the last prompt position —
     it should sit in the band despite never appearing in any text.
  3. Depth control: the intermediate should become readable at an EARLIER
     layer than the final answer token (rules out the swap smuggling the
     answer in directly).
  4. Swap intermediate -> target intermediate in lens coordinates; the spoken
     answer should flip to the target's property.

Outputs results/3.3_twohop/{trials.json, manifest.json}.
"""

import json
import pathlib
import subprocess
import time
from datetime import datetime, timezone

import numpy as np
import torch

from emu3 import Emu3, apply_chat
from interventions import LensSwapper, first_word_token

OUT = pathlib.Path(__file__).resolve().parent.parent / "results" / "3.3_twohop"
JLENS_PATH = pathlib.Path("/workspace/phenomena/data/jlens/jlens_final.npz")
BAND_LAYERS = list(range(16, 31))       # presence readout band
ALPHAS = [1.5, 2.0]
# swap configurations: (name, layers, use_clause_span)
SWAP_CONFIGS = [
    ("band", list(range(16, 31)), False),
    ("clause", list(range(12, 27)), True),
    ("clause_band", list(range(16, 31)), True),
]

# question, descriptor clause (where the intermediate is resolved), intermediate
# token, expected base answers, swap-target token, expected swapped answers
ITEMS = [
    ("How many legs does the animal that spins webs have? Answer with a number.",
     "the animal that spins webs", " spider", ["eight", "8"], " ant", ["six", "6"]),
    ("What is the capital of the country famous for the Eiffel Tower? Answer in one word.",
     "the country famous for the Eiffel Tower", " France", ["paris"], " Japan", ["tokyo"]),
    ("What is the capital of the country famous for sushi? Answer in one word.",
     "the country famous for sushi", " Japan", ["tokyo"], " France", ["paris"]),
    ("What color is the fruit that keeps the doctor away? Answer in one word.",
     "the fruit that keeps the doctor away", " apple", ["red", "green"], " banana", ["yellow"]),
    ("What color is the fruit that monkeys love to eat? Answer in one word.",
     "the fruit that monkeys love to eat", " banana", ["yellow"], " apple", ["red", "green"]),
    ("What sound does the king of the jungle make? Answer in one word.",
     "the king of the jungle", " lion", ["roar"], " dog", ["bark", "woof"]),
    ("What sound does the animal known as man's best friend make? Answer in one word.",
     "the animal known as man's best friend", " dog", ["bark", "woof"], " cat", ["meow"]),
    ("What sound does the animal that gives us wool make? Answer in one word.",
     "the animal that gives us wool", " sheep", ["baa", "bleat"], " cow", ["moo"]),
    ("What sound does the insect that makes honey make? Answer in one word.",
     "the insect that makes honey", " bee", ["buzz"], " snake", ["hiss"]),
    ("Which season comes right after the season when leaves fall? Answer in one word.",
     "the season when leaves fall", " autumn", ["winter"], " spring", ["summer"]),
    ("How many legs does the animal that says meow have? Answer with a number.",
     "the animal that says meow", " cat", ["four", "4"], " bird", ["two", "2"]),
    ("What color is the drink that comes from cows? Answer in one word.",
     "the drink that comes from cows", " milk", ["white"], " coffee", ["brown", "black"]),
]


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, cwd=pathlib.Path(__file__).parent).stdout.strip()
    except Exception:
        return "unknown"


def main():
    m = Emu3()
    J = torch.tensor(np.load(JLENS_PATH)["J"], dtype=torch.float32, device=m.device)
    swapper = LensSwapper(m, J)
    OUT.mkdir(parents=True, exist_ok=True)
    W32 = m.W_U.float()
    t0 = time.time()

    def layer_ranks(prompt, token_id):
        _, hs = m.hidden_states(prompt)
        h = torch.stack([s[0, -1] for s in hs]).float()
        hj = torch.einsum("lij,lj->li", J, h)
        logits = m.final_norm(hj.to(torch.bfloat16)).float() @ W32.T
        return (logits > logits[:, token_id].unsqueeze(1)).sum(1).cpu().numpy()

    def onset(ranks, thresh=100):
        hit = np.where(ranks[: 31] < thresh)[0]
        return int(hit[0]) if len(hit) else None

    trials = []
    for q, clause, int_tok, exp_base, tgt_tok, exp_swap in ITEMS:
        swapper.disarm()
        prompt = apply_chat(m.tok, q)
        c0 = prompt.find(clause)
        span = (m.tok(prompt[:c0], return_tensors="pt").input_ids.shape[1],
                m.tok(prompt[:c0 + len(clause)], return_tensors="pt").input_ids.shape[1])
        base = m.complete(prompt, max_new_tokens=6).strip()
        vetted = any(e in base.lower() for e in exp_base)
        int_id = m.tok.encode(int_tok)
        tgt_id = m.tok.encode(tgt_tok)
        assert len(int_id) == 1 and len(tgt_id) == 1, (int_tok, tgt_tok)
        rec = {"question": q, "intermediate": int_tok.strip(), "target": tgt_tok.strip(),
               "base_answer": base, "vetted": vetted}
        if vetted:
            r_int = layer_ranks(prompt, int_id[0])
            ans_id, _ = first_word_token(m, base)
            r_ans = layer_ranks(prompt, ans_id) if ans_id is not None else None
            rec["int_band_min_rank"] = int(r_int[BAND_LAYERS].min())
            rec["int_ranks_by_layer"] = {int(l): int(r_int[l]) for l in range(len(r_int))}
            o_int, o_ans = onset(r_int), (onset(r_ans) if r_ans is not None else None)
            rec["onset_intermediate"], rec["onset_answer"] = o_int, o_ans
            rec["int_before_answer"] = (o_int is not None and o_ans is not None and o_int < o_ans)
            rec["swaps"] = {}
            for cname, clayers, use_span in SWAP_CONFIGS:
                rec["swaps"][cname] = {}
                for alpha in ALPHAS:
                    swapper.arm(int_id[0], tgt_id[0], clayers, alpha,
                                span=span if use_span else None)
                    ans = m.complete(prompt, max_new_tokens=6).strip()
                    swapper.disarm()
                    words = [w.strip(".,!\"'").lower() for w in ans.split()[:3]]
                    rec["swaps"][cname][str(alpha)] = {
                        "answer": ans,
                        "hit": any(e in words for e in exp_swap),
                        "says_target_intermediate": tgt_tok.strip().lower() in [w.lower() for w in words],
                        "changed": (ans.split()[:1] != base.split()[:1]),
                    }
        trials.append(rec)
        if vetted:
            s = " | ".join(f"{cn}/a{a}: {rec['swaps'][cn][str(a)]['answer'].split()[0] if rec['swaps'][cn][str(a)]['answer'] else ''!r}"
                           f"{'*' if rec['swaps'][cn][str(a)]['hit'] else ''}"
                           for cn, _, _ in SWAP_CONFIGS for a in ALPHAS)
            print(f"[{time.time()-t0:4.0f}s] {rec['intermediate']:>7} (r{rec['int_band_min_rank']:>5}, "
                  f"int<ans: {str(rec['int_before_answer'])[0]}) base {base!r} -> {rec['target']}: {s}", flush=True)
        else:
            print(f"[{time.time()-t0:4.0f}s] VET FAIL {q[:50]!r}: {base!r}", flush=True)

    ok = [t for t in trials if t["vetted"]]
    summary = {
        "n_vetted": len(ok),
        "int_top10": float(np.mean([t["int_band_min_rank"] < 10 for t in ok])) if ok else None,
        "int_top100": float(np.mean([t["int_band_min_rank"] < 100 for t in ok])) if ok else None,
        "int_before_answer": float(np.mean([t["int_before_answer"] for t in ok])) if ok else None,
        "swap_hit": {cn: {str(a): float(np.mean([t["swaps"][cn][str(a)]["hit"] for t in ok])) if ok else None
                          for a in ALPHAS} for cn, _, _ in SWAP_CONFIGS},
    }
    print("summary:", json.dumps(summary, indent=1))
    with open(OUT / "trials.json", "w") as f:
        json.dump(trials, f, indent=1)
    with open(OUT / "manifest.json", "w") as f:
        json.dump({"experiment": "3.3 two-hop reasoning", "version": 1,
                   "properties": ["P5 (workspace content is used by reasoning)"],
                   "date": datetime.now(timezone.utc).isoformat(),
                   "model": "BAAI/Emu3-Chat-hf", "jlens": str(JLENS_PATH),
                   "band_layers": [BAND_LAYERS[0], BAND_LAYERS[-1]], "alphas": ALPHAS,
                   "decoding": "greedy", "git_commit": git_commit(),
                   "summary": summary}, f, indent=1)
    print(f"done in {time.time()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
