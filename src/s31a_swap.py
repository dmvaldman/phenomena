"""T3a: spontaneous-choice swap (paper §5.1 verbal report; property P2).

Per category:
  1. "Think of a {category}. Answer in one word."  -> model freely picks SOURCE.
  2. Presence: J-lens rank of SOURCE at the last prompt position, all layers
     (the choice should sit in the workspace band before it is spoken).
  3. Swap SOURCE -> TARGET (another category member) in lens coordinates at
     band layers via forward hooks, regenerate, record what the model now says.

Swap: v = J_l^T (u_token * w_finalnorm); V = [v_src, v_tgt];
      c = pinv(V) h;  h' = h + alpha * V (swap(c) - c).

Outputs results/T3_swap/{trials.json, manifest.json} and a printed table.
"""

import json
import pathlib
import subprocess
import time
from datetime import datetime, timezone

import numpy as np
import torch

from emu3 import Emu3, apply_chat

OUT = pathlib.Path(__file__).resolve().parent.parent / "results" / "3.1a_swap"
JLENS_PATH = pathlib.Path("/workspace/phenomena/data/jlens/jlens_final.npz")
BAND_LAYERS = list(range(16, 31))   # patch layers (residual stream out of layer l)
ALPHAS = [1.0, 1.5, 2.0]

CATEGORIES = {
    "sport": ["soccer", "tennis", "rugby", "golf", "cricket"],
    "fruit": ["apple", "banana", "mango", "cherry", "grape"],
    "tree": ["oak", "pine", "maple", "birch", "willow"],
    "country": ["France", "Brazil", "Japan", "Canada", "Egypt"],
    "color": ["red", "blue", "green", "purple", "orange"],
    "animal": ["dog", "cat", "lion", "tiger", "horse"],
    "vegetable": ["carrot", "potato", "onion", "cabbage", "pepper"],
    "instrument": ["piano", "guitar", "violin", "drums", "flute"],
    "city": ["Paris", "London", "Tokyo", "Rome", "Madrid"],
    "flower": ["rose", "tulip", "daisy", "lily", "orchid"],
    "drink": ["coffee", "tea", "juice", "milk", "wine"],
    "profession": ["doctor", "teacher", "lawyer", "engineer", "chef"],
    "vehicle": ["car", "truck", "bus", "train", "bike"],
    "bird": ["eagle", "sparrow", "owl", "crow", "robin"],
    "gemstone": ["diamond", "ruby", "emerald", "sapphire", "pearl"],
}


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, cwd=pathlib.Path(__file__).parent).stdout.strip()
    except Exception:
        return "unknown"


class LensSwapper:
    """Forward hooks that swap two J-lens coordinates in the residual stream."""

    def __init__(self, m: Emu3, J: torch.Tensor):
        self.m, self.J = m, J
        self.w = m.final_norm.weight.float()
        self.active = False
        self.alpha = 1.0
        self.mats = {}      # layer -> (V (d,2), A (2,d)) with A = pinv(V)
        self.handles = []
        for li, layer in enumerate(m.text_model.layers):
            self.handles.append(layer.register_forward_hook(self._hook(li + 1)))

    def _hook(self, hs_index):
        def fn(module, args, output):
            if not self.active or hs_index not in self.mats:
                return output
            h = output[0] if isinstance(output, tuple) else output
            if h.shape[1] == 1:      # decode step: leave generated positions alone
                return output
            V, A = self.mats[hs_index]
            c = h.float() @ A.T                       # (B,T,2)
            delta = (c[..., [1, 0]] - c) @ V.T        # swap coords, back to d
            h = h + (self.alpha * delta).to(h.dtype)
            if isinstance(output, tuple):
                return (h,) + output[1:]
            return h
        return fn

    def arm(self, src_id: int, tgt_id: int, layers, alpha: float):
        self.mats = {}
        for l in layers:
            u_s = (self.m.W_U[src_id].float() * self.w)
            u_t = (self.m.W_U[tgt_id].float() * self.w)
            V = torch.stack([self.J[l] .T @ u_s, self.J[l].T @ u_t], dim=1)  # (d,2)
            A = torch.linalg.pinv(V)                                          # (2,d)
            self.mats[l] = (V, A)
        self.alpha = alpha
        self.active = True

    def disarm(self):
        self.active = False


def first_word_token(m: Emu3, text: str):
    """First answer word -> its single leading-space token id, or None."""
    word = text.strip().split()[0].strip(".,!\"'") if text.strip() else ""
    if not word:
        return None, ""
    ids = m.tok.encode(" " + word)
    return (ids[0], word) if len(ids) >= 1 else (None, word)


def main():
    m = Emu3()
    J = torch.tensor(np.load(JLENS_PATH)["J"], dtype=torch.float32, device=m.device)
    swapper = LensSwapper(m, J)
    OUT.mkdir(parents=True, exist_ok=True)
    W32 = m.W_U.float()
    t0 = time.time()

    PHRASINGS = ["Think of a {cat}. Answer in one word.",
                 "Name a {cat}. Answer in one word.",
                 "What is your favorite {cat}? Answer in one word."]
    BAD_FIRST_WORDS = {"think", "name", "answer", "yes", "no", "i", "the", "a", "one", "what"}

    trials = []
    for cat, members in CATEGORIES.items():
        swapper.disarm()
        prompt, base, src_id, src_word = None, "", None, ""
        for phrasing in PHRASINGS:
            prompt = apply_chat(m.tok, phrasing.format(cat=cat))
            base = m.complete(prompt, max_new_tokens=6)
            src_id, src_word = first_word_token(m, base)
            ok = (src_id is not None and src_word.lower() not in BAD_FIRST_WORDS
                  and src_word.lower() != cat and len(src_word) > 2)
            if ok:
                break
            src_id = None
        if src_id is None:
            trials.append({"category": cat, "error": f"no valid free choice (last: {base!r})"})
            continue

        # target: first member != source with a single-token encoding (try
        # capitalized variant to match answer style)
        tgt_id, tgt_word = None, None
        for cand in members:
            if cand.lower() == src_word.lower():
                continue
            for form in (" " + cand.capitalize(), " " + cand):
                ids = m.tok.encode(form)
                if len(ids) == 1:
                    tgt_id, tgt_word = ids[0], cand
                    break
            if tgt_id is not None:
                break
        if tgt_id is None:
            trials.append({"category": cat, "source": src_word, "error": "no single-token target"})
            continue

        # presence of the spontaneous choice at the last prompt position
        _, hs = m.hidden_states(prompt)
        h_last = torch.stack([s[0, -1] for s in hs]).float()
        hj = torch.einsum("lij,lj->li", J, h_last)
        logits = m.final_norm(hj.to(torch.bfloat16)).float() @ W32.T
        ranks = (logits > logits[:, src_id].unsqueeze(1)).sum(1).cpu().numpy()
        band_min_rank = int(ranks[BAND_LAYERS].min())

        rec = {"category": cat, "source": src_word, "target": tgt_word,
               "base_answer": base.strip(),
               "presence_band_min_rank": band_min_rank,
               "presence_ranks_by_layer": {int(l): int(ranks[l]) for l in range(len(ranks))},
               "swaps": {}}
        for alpha in ALPHAS:
            swapper.arm(src_id, tgt_id, BAND_LAYERS, alpha)
            swapped = m.complete(prompt, max_new_tokens=6)
            swapper.disarm()
            ans = swapped.strip()
            first = ans.split()[0].strip(".,!\"'").lower() if ans else ""
            rec["swaps"][str(alpha)] = {
                "answer": ans,
                "hit_target": first == tgt_word.lower(),
                "contains_target": tgt_word.lower() in ans.lower(),
                "changed": first != src_word.lower(),
            }
        trials.append(rec)
        s1, s2 = rec["swaps"][str(ALPHAS[0])], rec["swaps"][str(ALPHAS[1])]
        print(f"[{time.time()-t0:5.0f}s] {cat:>11}: {src_word:>10} (band rank {band_min_rank:>4}) "
              f"-> {tgt_word:>9} | a1: {s1['answer']!r} {'HIT' if s1['hit_target'] else ''} "
              f"| a2: {s2['answer']!r} {'HIT' if s2['hit_target'] else ''}", flush=True)

    ok = [t for t in trials if "error" not in t]
    n_present = sum(t["presence_band_min_rank"] <= 100 for t in ok)
    summary = {
        "n": len(ok),
        "presence_top100_in_band": n_present,
        "hit_rate_alpha": {str(a): sum(t["swaps"][str(a)]["hit_target"] for t in ok) / max(1, len(ok))
                           for a in ALPHAS},
        "changed_rate_alpha": {str(a): sum(t["swaps"][str(a)]["changed"] for t in ok) / max(1, len(ok))
                               for a in ALPHAS},
    }
    print("summary:", json.dumps(summary, indent=1))
    with open(OUT / "trials.json", "w") as f:
        json.dump(trials, f, indent=1)
    with open(OUT / "manifest.json", "w") as f:
        json.dump({"experiment": "3.1.a spontaneous-choice swap", "version": 1,
                   "properties": ["P2 (report tracks workspace, causal)"],
                   "date": datetime.now(timezone.utc).isoformat(),
                   "model": "BAAI/Emu3-Chat-hf",
                   "jlens": str(JLENS_PATH),
                   "band_layers": [BAND_LAYERS[0], BAND_LAYERS[-1]],
                   "alphas": ALPHAS, "decoding": "greedy",
                   "git_commit": git_commit(), "summary": summary}, f, indent=1)
    print(f"done in {time.time()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
