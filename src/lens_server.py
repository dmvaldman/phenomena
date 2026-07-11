"""Interactive J-lens visualizer server (chat-capable).

Run on the pod:  uvicorn lens_server:app --host 0.0.0.0 --port 7860
Access locally via:  ssh -N -L 7860:localhost:7860 <pod>
"""

import pathlib

import numpy as np
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from emu3 import Emu3, apply_chat
from metrics import SIM_THRESH

MAX_TOKENS = 256
MAX_REPLY = 64
JLENS_PATH = pathlib.Path("/workspace/phenomena/data/jlens/jlens_final.npz")
STATIC = pathlib.Path(__file__).resolve().parent / "static"

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
m = Emu3()
J = torch.tensor(np.load(JLENS_PATH)["J"], dtype=torch.float32, device=m.device)
W32 = m.W_U.float()  # (vocab, d) fp32 cached
Wc = W32 - W32.mean(0, keepdim=True)
Wc = Wc / (Wc.norm(dim=1, keepdim=True) + 1e-8)  # centered+normalized, for conceptness


def probe_sim_column(probe_id: int) -> torch.Tensor:
    """(vocab,) thresholded cosine similarity of every token to the probe."""
    s = Wc @ Wc[probe_id]
    s[s < SIM_THRESH] = 0.0
    return s


class ChatRequest(BaseModel):
    messages: list[dict]          # [{role, content}, ...] ending with a user turn
    lens: str = "jlens"
    probe: str = ""
    template: bool = True
    topk: int = 5
    temperature: float = 0.7      # 0 = greedy
    seed: int | None = None       # set for reproducible sampling
    skip_tokens: int = 0          # columns already displayed by earlier turns


@torch.no_grad()
def grid_readout(text: str, n_prompt_tokens: int, lens: str, probe: str | None,
                 topk: int, skip: int = 0):
    ids = m.tok(text, return_tensors="pt").to(m.device)
    ids.input_ids = ids.input_ids[:, :MAX_TOKENS]
    ids.attention_mask = ids.attention_mask[:, :MAX_TOKENS]
    out = m.model(**ids, output_hidden_states=True)
    h = torch.stack([s[0] for s in out.hidden_states]).float()  # (L+1, T, d)
    if lens == "jlens":
        h = torch.einsum("lij,ltj->lti", J, h)
    normed = m.final_norm(h.to(torch.bfloat16)).float()

    n_layers, T, _ = normed.shape
    probe_id, S_probe = None, None
    if probe:
        enc = m.tok.encode(probe)
        if enc:
            probe_id = enc[0]
            S_probe = probe_sim_column(probe_id)

    skip = max(0, min(skip, T))
    grid, probe_ranks, probe_scores = [], [], []
    for l in range(n_layers):
        logits = normed[l] @ W32.T
        probs = torch.softmax(logits, dim=-1)
        top = probs.topk(topk, dim=-1)
        grid.append([[{"tok": m.tok.decode([i]), "p": round(float(p), 4)}
                      for i, p in zip(top.indices[t].tolist(), top.values[t].tolist())]
                     for t in range(skip, T)])
        if probe_id is not None:
            probe_ranks.append((logits > logits[:, probe_id].unsqueeze(1)).sum(1).tolist()[skip:])
            probe_scores.append([round(float(x), 5) for x in (probs @ S_probe).tolist()[skip:]])

    return {
        "tokens": [m.tok.decode([i]) for i in ids.input_ids[0, skip:].tolist()],
        "roles": [0 if t < n_prompt_tokens else 1 for t in range(skip, T)],
        "n_layers": n_layers,
        "skip": skip,
        "total_tokens": T,
        "grid": grid,
        "probe_ranks": probe_ranks or None,
        "probe_scores": probe_scores or None,
        "probe_token": m.tok.decode([probe_id]) if probe_id is not None else None,
        "vocab_size": W32.shape[0],
    }


@app.post("/api/chat")
def api_chat(req: ChatRequest):
    if req.template:
        prompt = apply_chat(m.tok, req.messages)
    else:
        prompt = "\n".join(msg["content"] for msg in req.messages)
    seed = req.seed if req.seed is not None else int(torch.randint(0, 2**31, (1,)).item())
    reply = m.complete(prompt, max_new_tokens=MAX_REPLY,
                       temperature=req.temperature, seed=seed).strip()
    full = prompt + " " + reply
    n_prompt = m.tok(prompt, return_tensors="pt").input_ids.shape[1]
    out = grid_readout(full, n_prompt, req.lens, req.probe or None, req.topk,
                       skip=req.skip_tokens)
    out["reply"] = reply
    out["prompt_used"] = prompt
    out["seed"] = seed
    out["temperature"] = req.temperature
    return out


@app.get("/api/lens")
def api_lens(prompt: str, lens: str = "jlens", probe: str = "", topk: int = 5, template: bool = True):
    text = apply_chat(m.tok, prompt) if template else prompt
    out = grid_readout(text, m.tok(text, return_tensors="pt").input_ids.shape[1], lens, probe or None, topk)
    out["prompt_used"] = text
    return out


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
