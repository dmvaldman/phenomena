"""Analyze T2 battery scores -> stats table + layer-profile figure.

Reads results/T2_battery/{scores.npz, trials.json}; writes stats.txt and
layer_profile.png alongside. No GPU needed.

Control distribution for concept c: conceptness of c measured on every trial
that has nothing to do with c (bare controls + think trials of other
concepts). Onset = first layer a think trial clears that distribution's p95.
"""

import json
import pathlib

import numpy as np

from metrics import cliffs_delta, onset_and_peak, position_consistency

DIR = pathlib.Path(__file__).resolve().parent.parent / "results" / "T2_battery"


def main():
    Z = np.load(DIR / "scores.npz", allow_pickle=True)
    trials = json.load(open(DIR / "trials.json"))
    ts, cm = Z["target_scores"], Z["concept_means"]      # (N,L,T) (N,L,C)
    concepts = list(Z["concepts"])
    N, L, C = cm.shape

    cond = np.array([t["cond"] for t in trials])
    tconc = np.array([-1 if t["concept"] is None else t["concept"] for t in trials])

    # mask out pre-question positions: the concept word literally appears there,
    # and its own token embedding scores high from layer 0 (not covert content)
    for i, t in enumerate(trials):
        ts[i, :, :t.get("q_start", 0)] = np.nan

    # control p95 per (layer, concept): trials unrelated to that concept
    p95 = np.zeros((L, C))
    ctrl_vals = {}
    for c in range(C):
        unrelated = (tconc != c) & np.isin(cond, ["control", "think"])
        vals = cm[unrelated, :, c]                        # (n, L)
        ctrl_vals[c] = vals
        p95[:, c] = np.percentile(vals, 95, axis=0)

    rows, profiles = [], {}
    pooled = dict(onset=[], peak=[], cons=[], delta=[])
    for c, name in enumerate(concepts):
        idx = np.where((cond == "think") & (tconc == c))[0]
        prof = np.nanmean(ts[idx], axis=2)                # (n, L) position-mean
        profiles[name] = np.median(prof, axis=0)
        band_layers = np.where(np.median(prof, axis=0) > p95[:, c])[0]
        band = slice(band_layers.min(), band_layers.max() + 1) if len(band_layers) else slice(0, 0)
        onsets, peaks, cons = [], [], []
        for k, i in enumerate(idx):
            o, pk = onset_and_peak(prof[k], p95[:, c])
            if o is not None:
                onsets.append(o)
            peaks.append(pk)
            if band.stop > band.start:
                cons.append(position_consistency(ts[i], band, p95[:, c]))
        band_mean = prof[:, band].mean(axis=1) if band.stop > band.start else prof.max(axis=1)
        delta = cliffs_delta(band_mean, ctrl_vals[c][:, band].mean(axis=1)
                             if band.stop > band.start else ctrl_vals[c].max(axis=1))
        rows.append((name, len(idx), np.median(onsets) if onsets else float("nan"),
                     len(onsets) / len(idx), np.median(peaks),
                     np.nanmedian(cons) if cons else float("nan"), delta,
                     f"{band.start}-{band.stop - 1}" if band.stop > band.start else "-"))
        pooled["onset"] += onsets
        pooled["peak"] += list(peaks)
        pooled["cons"] += [x for x in cons if not np.isnan(x)]
        pooled["delta"].append(delta)

    hdr = f"{'concept':>12} {'n':>4} {'onset_med':>10} {'frac_onset':>10} {'peak_med':>9} {'pos_consist':>11} {'delta':>7} {'band':>7}"
    lines = [hdr]
    for r in rows:
        lines.append(f"{r[0]:>12} {r[1]:>4} {r[2]:>10.1f} {r[3]:>10.2f} {r[4]:>9.0f} {r[5]:>11.2f} {r[6]:>7.2f} {r[7]:>7}")
    lines.append(f"{'POOLED':>12} {'':>4} {np.median(pooled['onset']):>10.1f} "
                 f"{'':>10} {np.median(pooled['peak']):>9.0f} "
                 f"{np.median(pooled['cons']):>11.2f} {np.mean(pooled['delta']):>7.2f}")
    out = "\n".join(lines)
    print(out)
    (DIR / "stats.txt").write_text(out + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5))
    ctrl_all = np.stack([np.percentile(ctrl_vals[c], [5, 50, 95], axis=0) for c in range(C)])
    ax.fill_between(range(L), ctrl_all[:, 0].min(0), ctrl_all[:, 2].max(0),
                    color="gray", alpha=0.25, label="control p5-p95")
    for name, prof in profiles.items():
        ax.plot(range(L), prof, lw=1, alpha=0.8, label=name)
    ax.set(xlabel="layer", ylabel="conceptness (J-lens)", yscale="log",
           title="T2: covert concept score by layer (median per concept)")
    ax.legend(fontsize=6, ncol=4)
    fig.tight_layout()
    fig.savefig(DIR / "layer_profile.png", dpi=140)
    print(f"wrote {DIR}/stats.txt and layer_profile.png")


if __name__ == "__main__":
    main()
