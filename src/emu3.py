"""Shared model loading, vocab map, and lens utilities for Emu3."""

import torch
from transformers import AutoTokenizer, Emu3ForConditionalGeneration

CHAT_HF = "BAAI/Emu3-Chat-hf"


class Emu3:
    def __init__(self, model_id: str = CHAT_HF, device: str = "cuda"):
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = Emu3ForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map=device
        ).eval()
        self.device = device

        lm = self.model.model
        self.text_model = getattr(lm, "text_model", lm)
        self.final_norm = self.text_model.norm
        self.W_U = self.model.get_output_embeddings().weight  # (vocab, d)

        vocab = self.tok.get_vocab()
        vis = sorted(i for t, i in vocab.items() if "visual token" in t)
        assert vis == list(range(vis[0], vis[-1] + 1)), "vision block not contiguous"
        self.vis_lo, self.vis_hi = vis[0], vis[-1] + 1  # [lo, hi)
        self.tok_eol = vocab["<|extra_200|>"]
        self.tok_eof = vocab["<|extra_201|>"]
        self.tok_img_start = vocab["<|image start|>"]
        self.tok_img_end = vocab["<|image end|>"]
        self.n_layers = self.text_model.config.num_hidden_layers

    @torch.no_grad()
    def hidden_states(self, prompt: str):
        """Returns (input_ids, tuple of n_layers+1 tensors of shape (1, T, d))."""
        ids = self.tok(prompt, return_tensors="pt").to(self.device)
        out = self.model(**ids, output_hidden_states=True)
        return ids.input_ids, out.hidden_states

    @torch.no_grad()
    def lens_logits(self, h: torch.Tensor) -> torch.Tensor:
        """Logit lens (L0): final norm + unembed. h: (..., d) -> (..., vocab), fp32."""
        return self.final_norm(h).float() @ self.W_U.float().T

    def vision_block(self, logits: torch.Tensor) -> torch.Tensor:
        """Restrict a lens readout to the vision-token block."""
        return logits[..., self.vis_lo : self.vis_hi]

    @torch.no_grad()
    def rank_of(self, logits: torch.Tensor, token_str: str) -> int:
        """Rank (0 = top) of a single-token string in a (vocab,) logit vector."""
        tid = self.tok.encode(token_str)[0]
        return int((logits.argsort(descending=True) == tid).nonzero().item())

    @torch.no_grad()
    def complete(self, prompt: str, max_new_tokens: int = 8,
                 temperature: float = 0.0, top_p: float = 0.9,
                 seed: int | None = None) -> str:
        """Greedy when temperature == 0, otherwise nucleus sampling."""
        ids = self.tok(prompt, return_tensors="pt").to(self.device)
        kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False)
        if temperature > 0:
            if seed is not None:
                torch.manual_seed(seed)
            kwargs = dict(max_new_tokens=max_new_tokens, do_sample=True,
                          temperature=temperature, top_p=top_p)
        gen = self.model.generate(**ids, **kwargs)
        return self.tok.decode(gen[0, ids.input_ids.shape[1] :], skip_special_tokens=True)


def chat(user_msg: str) -> str:
    """Emu3-Chat prompt format (manual fallback)."""
    return f"USER: {user_msg} ASSISTANT:"


def apply_chat(tok, messages) -> str:
    """Expand a message list (or bare user string) through the tokenizer's chat
    template, falling back to the manual USER:/ASSISTANT: format."""
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    try:
        s = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        if s and s.strip():
            return s
    except Exception:
        pass
    parts = []
    for m_ in messages:
        role = "USER" if m_["role"] == "user" else "ASSISTANT"
        parts.append(f"{role}: {m_['content']}")
    return " ".join(parts) + " ASSISTANT:"
