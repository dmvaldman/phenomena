"""Shared activation interventions: lens-coordinate swapping (and helpers)."""

import torch

from emu3 import Emu3


class LensSwapper:
    """Forward hooks that swap two J-lens coordinates in the residual stream.

    Patches prefill forwards only (T > 1); generated positions are left alone.
    v_token = J_l^T (u_token * w_finalnorm); V = [v_src, v_tgt]; coordinates
    c = pinv(V) h are exchanged, the orthogonal complement is untouched.
    """

    def __init__(self, m: Emu3, J: torch.Tensor):
        self.m, self.J = m, J
        self.w = m.final_norm.weight.float()
        self.active = False
        self.alpha = 1.0
        self.spare_last = 0
        self.span = None
        self.mats = {}      # hs_index -> (V (d,2), A (2,d))
        for li, layer in enumerate(m.text_model.layers):
            layer.register_forward_hook(self._hook(li + 1))

    def _hook(self, hs_index):
        def fn(module, args, output):
            if not self.active or hs_index not in self.mats:
                return output
            h = output[0] if isinstance(output, tuple) else output
            if h.shape[1] == 1:      # decode step: leave generated positions alone
                return output
            V, A = self.mats[hs_index]
            c = h.float() @ A.T
            delta = (c[..., [1, 0]] - c) @ V.T
            if self.spare_last:      # leave the final (answer-forming) position alone
                delta[:, -self.spare_last:] = 0
            if self.span is not None:  # patch only these positions (e.g. descriptor clause)
                s0, s1 = self.span
                delta[:, :s0] = 0
                delta[:, s1:] = 0
            h = h + (self.alpha * delta).to(h.dtype)
            if isinstance(output, tuple):
                return (h,) + output[1:]
            return h
        return fn

    def arm(self, src_id: int, tgt_id: int, layers, alpha: float, spare_last: int = 0,
            span=None):
        self.mats = {}
        self.spare_last = spare_last
        self.span = span
        for l in layers:
            u_s = self.m.W_U[src_id].float() * self.w
            u_t = self.m.W_U[tgt_id].float() * self.w
            V = torch.stack([self.J[l].T @ u_s, self.J[l].T @ u_t], dim=1)
            self.mats[l] = (V, torch.linalg.pinv(V))
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
