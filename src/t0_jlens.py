"""T0: estimate per-layer Jacobian-lens matrices J_l = E[dh_final,t' / dh_l,t].

Stochastic VJP estimator: for random v ~ N(0,I), one backward pass from
(v . h_L[t']) yields g_{l,t} = v^T dh_L[t']/dh_{l,t} for all layers/positions.
E[outer(v, g)] = J (unbiased).  Averaged over source positions t <= t',
sampled target positions t', and a corpus of pretraining-like prompts.

Two independent accumulators (even/odd samples) provide split-half convergence.
Checkpoints to /workspace/phenomena/data/jlens/ .
"""

import pathlib
import time

import numpy as np
import torch

from emu3 import Emu3

DATA = pathlib.Path("/workspace/phenomena/data/jlens")
SEQ_LEN = 512
BATCH = 4
TPRIME_PER_SEQ = 4      # backward passes per sequence
MIN_TPRIME = 64         # skip early positions (BOS-adjacent noise)
N_PROMPTS = 38400       # sequences to consume (=> N_PROMPTS * TPRIME_PER_SEQ samples)
CKPT_EVERY = 200        # batches


def corpus_chunks(tok, n, seq_len):
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
    buf, produced = [], 0
    for row in ds:
        text = row["text"].strip()
        if len(text) < 200:
            continue
        buf.extend(tok.encode(text) + [tok.eos_token_id or 0])
        while len(buf) >= seq_len:
            yield buf[:seq_len]
            buf = buf[seq_len:]
            produced += 1
            if produced >= n:
                return


def main():
    m = Emu3()
    model, tok = m.model, m.tok
    for p in model.parameters():
        p.requires_grad_(False)
    n_states = m.n_layers + 1
    d = m.W_U.shape[1]

    # Residual accumulators (identity control variate): per sample we accumulate
    # outer(v, g - alpha_run * v) and book the alpha_run used; J reconstructs as
    # resid/count + mean(alpha_used) * I.  Removes MC noise on the dominant
    # passthrough component of J.
    acc = [torch.zeros(2, d, d, dtype=torch.float32, device="cuda") for _ in range(n_states)]
    alpha_used = torch.zeros(n_states, 2, dtype=torch.float64)
    alpha_run = torch.zeros(n_states, dtype=torch.float64)
    alpha_n = 0
    counts = torch.zeros(2, dtype=torch.long)
    DATA.mkdir(parents=True, exist_ok=True)
    eye = torch.eye(d, dtype=torch.float32)

    def reconstruct(l, half=None):
        if half is None:
            m_ = (acc[l][0] + acc[l][1]).cpu() / counts.sum()
            a_ = alpha_used[l].sum().item() / counts.sum().item()
        else:
            m_ = acc[l][half].cpu() / counts[half]
            a_ = (alpha_used[l][half] / counts[half]).item()
        return m_ + a_ * eye

    def save(tag="ckpt"):
        J = np.stack([reconstruct(l).numpy().astype(np.float16) for l in range(n_states)])
        Je = np.stack([reconstruct(l, 0).numpy().astype(np.float16) for l in range(n_states)])
        Jo = np.stack([reconstruct(l, 1).numpy().astype(np.float16) for l in range(n_states)])
        np.savez(DATA / f"jlens_{tag}.npz", J=J, J_even=Je, J_odd=Jo,
                 counts=counts.numpy(), n_states=n_states, seq_len=SEQ_LEN)

    def split_half_cosine():
        # Residual part only — the alpha*I control variate is shared between
        # halves and would trivially inflate agreement.
        out = []
        for a in acc:
            x, y = a[0].flatten(), a[1].flatten()
            out.append((x @ y / (x.norm() * y.norm() + 1e-30)).item())
        return out

    gen = corpus_chunks(tok, N_PROMPTS, SEQ_LEN)
    emb = model.get_input_embeddings()
    t0, sample_idx, batch_i = time.time(), 0, 0

    while True:
        chunk_ids = []
        try:
            for _ in range(BATCH):
                chunk_ids.append(next(gen))
        except StopIteration:
            break
        ids = torch.tensor(chunk_ids, device="cuda")
        with torch.enable_grad():
            x = emb(ids).detach().requires_grad_(True)
            out = model(inputs_embeds=x, output_hidden_states=True)
            hs = out.hidden_states  # n_states tensors (B, T, d)
            for k in range(TPRIME_PER_SEQ):
                tprime = torch.randint(MIN_TPRIME, SEQ_LEN, (BATCH,))
                v = torch.randn(BATCH, d, device="cuda")
                scalar = sum((hs[-1][b, tprime[b]].float() * v[b]).sum() for b in range(BATCH))
                grads = torch.autograd.grad(scalar, hs, retain_graph=(k < TPRIME_PER_SEQ - 1))
                half = sample_idx % 2
                for l in range(n_states):
                    a_l = alpha_run[l].item()
                    for b in range(BATCH):
                        g = grads[l][b, : tprime[b] + 1].float().mean(0)
                        vb = v[b]
                        a_s = float((vb @ g) / (vb @ vb))
                        acc[l][half] += torch.outer(vb, g - a_l * vb)
                        alpha_used[l][half] += a_l
                        alpha_run[l] += (a_s - alpha_run[l]) / (alpha_n + b + 1)
                        if l == 0:
                            counts[half] += 1
                alpha_n += BATCH
                sample_idx += 1
        batch_i += 1
        if batch_i % CKPT_EVERY == 0:
            cos = split_half_cosine()
            print(f"[{time.time()-t0:6.0f}s] batch {batch_i}, samples {counts.sum().item()}, "
                  f"split-half cos: L8={cos[8]:.3f} L16={cos[16]:.3f} L24={cos[24]:.3f} L30={cos[30]:.3f}",
                  flush=True)
            save()

    save("final")
    cos = split_half_cosine()
    print("final split-half cosines:", [round(c, 3) for c in cos])
    print(f"done in {time.time()-t0:.0f}s, samples={counts.sum().item()}")


if __name__ == "__main__":
    main()
