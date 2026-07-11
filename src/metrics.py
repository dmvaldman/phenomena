"""Concept-likeness metrics and band statistics.

Core quantity: for a concept c with unembedding row u_c, and a lens
distribution p over the vocabulary at some (layer, position),

    conceptness(p, c) = sum_t p(t) * sim(u_c, u_t)

where sim is cosine similarity between mean-centered unembedding rows,
thresholded below SIM_THRESH to zero so the score reflects the concept's
semantic neighborhood rather than diffuse background mass.

Statistics reported per experiment:
  onset layer  - first layer where a trial's score clears the control p95
  peak layer   - argmax layer of the trial's score profile
  consistency  - fraction of token positions clearing control p95 in the band
  effect size  - Cliff's delta, think vs control band scores
"""

import numpy as np
import torch

SIM_THRESH = 0.25


@torch.no_grad()
def concept_similarity(W: torch.Tensor, target_ids: list[int],
                       thresh: float = SIM_THRESH) -> torch.Tensor:
    """(vocab, d) unembedding + target token ids -> (vocab, C) similarity map.

    Rows are mean-centered and L2-normalized before cosine; values below
    `thresh` are zeroed. The target token itself scores 1.
    """
    Wc = W.float() - W.float().mean(0, keepdim=True)
    Wc = Wc / (Wc.norm(dim=1, keepdim=True) + 1e-8)
    S = Wc @ Wc[target_ids].T                      # (vocab, C)
    S[S < thresh] = 0.0
    return S


@torch.no_grad()
def conceptness(probs: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    """probs (..., vocab) x S (vocab, C) -> scores (..., C)."""
    return probs @ S


def onset_and_peak(profile: np.ndarray, control_p95: np.ndarray):
    """profile (L,) trial score per layer; control_p95 (L,) threshold.
    Returns (onset_layer or None, peak_layer)."""
    above = np.where(profile > control_p95)[0]
    onset = int(above[0]) if len(above) else None
    return onset, int(np.argmax(profile))


def position_consistency(pos_scores: np.ndarray, band: slice, thresh: np.ndarray) -> float:
    """pos_scores (L, T) with NaN padding; fraction of band-layer positions
    clearing the per-layer control threshold."""
    x = pos_scores[band]                            # (Lb, T)
    ok = x > thresh[band][:, None]
    valid = ~np.isnan(x)
    return float(ok[valid].mean()) if valid.any() else float("nan")


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta of a vs b (positive = a tends larger). Vectorized."""
    a, b = np.asarray(a)[:, None], np.asarray(b)[None, :]
    return float((np.sign(a - b)).mean())
