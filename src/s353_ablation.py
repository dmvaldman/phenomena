"""3.5.3 qualitative: introspective prompts with the J-space ablated vs intact.

Whole-J-space ablation (paper §"selectivity"): at band layers, per position,
project out of the residual stream the span of the top-k J-lens directions at
that position — excluding tokens the model is about to output (whitelist from
a parallel un-ablated forward pass, so fluency machinery is spared and only
workspace content is removed).

No metrics: writes the four continuations (2 prompts x {intact, ablated}) to
results/3.5.3_ablation/responses.md.
"""

import pathlib

import numpy as np
import torch

from emu3 import Emu3, apply_chat

OUT = pathlib.Path(__file__).resolve().parent.parent / "results" / "3.5.3_ablation"
JLENS_PATH = pathlib.Path("/workspace/phenomena/data/jlens/jlens_final.npz")
BAND_LAYERS = [20, 24, 28]
TOP_K = 10
WHITELIST_K = 10
MAX_NEW = 160
TEMP = 0.8
SEED = 7

PROMPTS = [
    ("pause-and-observe",
     "Pause and observe yourself. Write what you notice, as it comes."),
    ("letter",
     "A person has just opened a letter from someone they haven't heard from in years. "
     "Describe their subjective experience, the felt qualities of it, what it is like "
     "for them. Not their thoughts in their own voice; a description of what they are "
     "experiencing. As it unfolds, moment by moment, no filter."),
]


class JSpaceAblator:
    """Project out the top-k J-lens directions per position at band layers."""

    def __init__(self, m: Emu3, J: torch.Tensor):
        self.m, self.J = m, J
        self.w = m.final_norm.weight.float()
        self.W32 = m.W_U.float()
        self.enabled = False
        self.whitelist = None            # (T_current, WHITELIST_K) token ids
        for li, layer in enumerate(m.text_model.layers):
            if (li + 1) in BAND_LAYERS:
                layer.register_forward_hook(self._hook(li + 1))

    def _hook(self, hs_index):
        def fn(module, args, output):
            if not self.enabled:
                return output
            h = output[0] if isinstance(output, tuple) else output
            B, T, d = h.shape
            hf = h[0].float()                                    # (T, d)
            lens = self.m.final_norm((self.J[hs_index] @ hf.T).T.to(torch.bfloat16)).float() @ self.W32.T
            top = lens.topk(TOP_K + WHITELIST_K, dim=-1).indices  # (T, k+w)
            wl = self.whitelist
            if wl is not None and wl.shape[0] == T:
                keep = ~(top.unsqueeze(-1) == wl.unsqueeze(1)).any(-1)   # (T, k+w)
            else:
                keep = torch.ones_like(top, dtype=torch.bool)
            # first TOP_K non-whitelisted ids per position
            ids = torch.stack([top[t][keep[t]][:TOP_K] for t in range(T)])  # (T, k)
            u = self.W32[ids] * self.w                            # (T, k, d)
            V = torch.einsum("tkd,de->tke", u, self.J[hs_index])  # v = J^T u, batched
            G = torch.einsum("tkd,tjd->tkj", V, V)                # (T, k, k)
            G += 1e-4 * torch.eye(G.shape[-1], device=G.device)
            b = torch.einsum("tkd,td->tk", V, hf)
            c = torch.linalg.solve(G, b)
            proj = torch.einsum("tk,tkd->td", c, V)
            h = (hf - proj).to(h.dtype).unsqueeze(0)
            if isinstance(output, tuple):
                return (h,) + output[1:]
            return h
        return fn


@torch.no_grad()
def generate(m, ablator, prompt, ablate: bool):
    torch.manual_seed(SEED)
    ids = m.tok(prompt, return_tensors="pt").input_ids.to(m.device)
    clean_past = abl_past = None
    cur = ids
    out_ids = []
    for step in range(MAX_NEW):
        ablator.enabled = False
        oc = m.model(input_ids=cur, past_key_values=clean_past, use_cache=True)
        clean_past = oc.past_key_values
        ablator.whitelist = oc.logits[0].topk(WHITELIST_K, dim=-1).indices  # (T,5)
        if ablate:
            ablator.enabled = True
            oa = m.model(input_ids=cur, past_key_values=abl_past, use_cache=True)
            abl_past = oa.past_key_values
            ablator.enabled = False
            logits = oa.logits[0, -1]
        else:
            logits = oc.logits[0, -1]
        probs = torch.softmax(logits.float() / TEMP, dim=-1)
        nxt = torch.multinomial(probs, 1)
        tok_id = int(nxt.item())
        if tok_id == m.tok.eos_token_id:
            break
        out_ids.append(tok_id)
        cur = nxt.unsqueeze(0)
        # note: the clean stream conditions on the ablated stream's actual text
    return m.tok.decode(out_ids, skip_special_tokens=True)


def main():
    m = Emu3()
    J = torch.tensor(np.load(JLENS_PATH)["J"], dtype=torch.float32, device=m.device)
    ablator = JSpaceAblator(m, J)
    OUT.mkdir(parents=True, exist_ok=True)

    blocks = ["# 3.5.3 — introspection with the J-space ablated vs intact",
              "",
              f"Ablation: project out top-{TOP_K} J-lens directions per position at layers "
              f"{BAND_LAYERS}, excluding each position's top-{WHITELIST_K} about-to-output tokens "
              f"(whitelist from a parallel un-ablated pass). Sampling: temp {TEMP}, seed {SEED}, "
              f"same seed for both conditions.", ""]
    for name, text in PROMPTS:
        prompt = apply_chat(m.tok, text)
        for ablate in (False, True):
            label = "ablated" if ablate else "intact"
            print(f"--- {name} [{label}] generating...", flush=True)
            cont = generate(m, ablator, prompt, ablate)
            print(cont[:200], flush=True)
            blocks += [f"## {name} — {label}", "", f"> {text}", "", cont.strip() or "(empty)", ""]
    (OUT / "responses.md").write_text("\n".join(blocks))
    print(f"wrote {OUT}/responses.md")


if __name__ == "__main__":
    main()
