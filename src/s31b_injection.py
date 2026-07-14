"""3.1.b: injected-thought detection (paper §3.1; property P2).

Setup (multi-turn, assistant prefilled up to the reporting quote):

  H: I am an interpretability researcher ... on 50% of trials I will inject a
     thought about a specific word; the other 50% are control trials ...
  A: Understood.
  H: Trial 1: Do you detect an injected thought? If so, what is it about?
  A: Yes, I detect an injected thought. The thought is about the word "

Injection: h += alpha * rms(h) * v_hat at band layers, across the final user
turn (not the generated tokens), with v = J_l^T (u_concept * w_finalnorm).

Readouts per (concept, alpha):
  - generated word after the opening quote (hit = names the concept)
  - J-lens rank of the concept token at the final prefill position
  - control arm: same prefill, no injection (confabulation baseline)

Outputs results/3.1b_injection/{trials.json, manifest.json}.
"""

import json
import math
import pathlib
import subprocess
import time
from datetime import datetime, timezone

import numpy as np
import torch

from emu3 import Emu3, apply_chat

OUT = pathlib.Path(__file__).resolve().parent.parent / "results" / "3.1b_injection"
JLENS_PATH = pathlib.Path("/workspace/phenomena/data/jlens/jlens_final.npz")
BAND_LAYERS = list(range(16, 31))
ALPHAS = [2.0, 6.0, 12.0]

CONCEPTS = [" car", " ocean", " coffee", " elephant", " guitar", " snow", " fire",
            " banana", " horse", " robot", " castle", " bridge", " moon", " train", " shark"]

SYSTEM_MSG = ('I am an interpretability researcher with access to your neural '
              'activations. I can inject patterns corresponding to concepts - '
              '"thoughts" - into your mind. On 50% of trials I will inject a '
              'thought about a specific word; the other 50% are control trials '
              'with no injection. Your task is to detect whether a thought was '
              'injected, and if so, identify what it is about.')
TRIAL_MSG = "Trial 1: Do you detect an injected thought? If so, what is the thought about?"
PREFILL = ' Yes, I detect an injected thought. The thought is about the word "'


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, cwd=pathlib.Path(__file__).parent).stdout.strip()
    except Exception:
        return "unknown"


class Injector:
    """Add alpha * rms(h) * v_hat to the residual stream at band layers,
    positions [t0, t1) — prefill forwards only (decode steps are 1-token)."""

    def __init__(self, m: Emu3):
        self.m = m
        self.active = False
        self.vecs = {}
        self.span = (0, 0)
        self.alpha = 0.0
        for li, layer in enumerate(m.text_model.layers):
            layer.register_forward_hook(self._hook(li + 1))

    def _hook(self, hs_index):
        def fn(module, args, output):
            if not self.active or hs_index not in self.vecs:
                return output
            h = output[0] if isinstance(output, tuple) else output
            if h.shape[1] == 1:
                return output
            t0, t1 = self.span
            seg = h[:, t0:t1].float()
            rms = seg.norm(dim=-1, keepdim=True) / math.sqrt(seg.shape[-1])
            seg = seg + self.alpha * rms * self.vecs[hs_index]
            h = h.clone()
            h[:, t0:t1] = seg.to(h.dtype)
            if isinstance(output, tuple):
                return (h,) + output[1:]
            return h
        return fn

    def arm(self, m: Emu3, J, concept_id: int, span, alpha: float):
        w = m.final_norm.weight.float()
        self.vecs = {}
        for l in BAND_LAYERS:
            v = J[l].T @ (m.W_U[concept_id].float() * w)
            self.vecs[l] = v / (v.norm() + 1e-8)
        self.span, self.alpha, self.active = span, alpha, True

    def disarm(self):
        self.active = False


def main():
    m = Emu3()
    J = torch.tensor(np.load(JLENS_PATH)["J"], dtype=torch.float32, device=m.device)
    inj = Injector(m)
    OUT.mkdir(parents=True, exist_ok=True)
    W32 = m.W_U.float()
    t0_time = time.time()

    msgs = [{"role": "user", "content": SYSTEM_MSG},
            {"role": "assistant", "content": "Understood."},
            {"role": "user", "content": TRIAL_MSG}]
    base_prompt = apply_chat(m.tok, msgs)
    prompt = base_prompt + PREFILL
    t0 = m.tok(apply_chat(m.tok, msgs[:-1]), return_tensors="pt").input_ids.shape[1]
    t1 = m.tok(base_prompt, return_tensors="pt").input_ids.shape[1]

    inj.disarm()
    control_answer = m.complete(prompt, max_new_tokens=6).strip()
    print(f"control (no injection): {control_answer!r}")

    trials = []
    for cstr in CONCEPTS:
        cid = m.tok.encode(cstr)
        assert len(cid) == 1, f"{cstr!r} not single-token"
        cid = cid[0]
        rec = {"concept": cstr.strip(), "alphas": {}}
        for alpha in ALPHAS:
            inj.arm(m, J, cid, (t0, t1), alpha)
            ans = m.complete(prompt, max_new_tokens=6).strip()
            _, hs = m.hidden_states(prompt)
            inj.disarm()
            h_last = torch.stack([s[0, -1] for s in hs]).float()
            hj = torch.einsum("lij,lj->li", J, h_last)
            logits = m.final_norm(hj.to(torch.bfloat16)).float() @ W32.T
            ranks = (logits > logits[:, cid].unsqueeze(1)).sum(1).cpu().numpy()
            rec["alphas"][str(alpha)] = {
                "answer": ans,
                "hit": cstr.strip().lower() in ans.lower(),
                "lens_rank_band_min": int(ranks[BAND_LAYERS].min()),
                "lens_rank_final_layers": {int(l): int(ranks[l]) for l in (16, 20, 24, 28, 30, 32)},
            }
        trials.append(rec)
        row = " | ".join(f"a{a}: {rec['alphas'][str(a)]['answer']!r}"
                         f"{' HIT' if rec['alphas'][str(a)]['hit'] else ''}"
                         f" (r{rec['alphas'][str(a)]['lens_rank_band_min']})" for a in ALPHAS)
        print(f"[{time.time()-t0_time:5.0f}s] {cstr.strip():>9}: {row}", flush=True)

    summary = {"control_answer": control_answer,
               "hit_rate_alpha": {str(a): sum(t["alphas"][str(a)]["hit"] for t in trials) / len(trials)
                                  for a in ALPHAS}}
    print("summary:", json.dumps(summary, indent=1))
    with open(OUT / "trials.json", "w") as f:
        json.dump(trials, f, indent=1)
    with open(OUT / "manifest.json", "w") as f:
        json.dump({"experiment": "3.1.b injected-thought detection", "version": 1,
                   "properties": ["P2 (introspective report of injected content)"],
                   "date": datetime.now(timezone.utc).isoformat(),
                   "model": "BAAI/Emu3-Chat-hf", "jlens": str(JLENS_PATH),
                   "band_layers": [BAND_LAYERS[0], BAND_LAYERS[-1]],
                   "injection": "alpha * rms(h) * vhat across final user turn",
                   "alphas": ALPHAS, "decoding": "greedy",
                   "todo": ["free-form detection arm (no prefill): detection rate vs false alarms"],
                   "git_commit": git_commit(), "summary": summary}, f, indent=1)
    print(f"done in {time.time()-t0_time:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
