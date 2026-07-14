"""Test scoreboard: compute one-number metrics with pass thresholds for every
experiment, from the raw files in results/, and write results/TESTS.md.

Run from anywhere, no GPU:  python3 src/report.py
Each metric: (test, name, value, threshold, direction). Diffable in git.
"""

import json
import pathlib
import subprocess
from datetime import datetime, timezone

import numpy as np

R = pathlib.Path(__file__).resolve().parent.parent / "results"
BAND = slice(18, 31)          # workspace band for rank-based metrics
SWAP_BAND = slice(16, 31)     # band used by 3.1.a presence readout

rows = []                     # (test, metric, value, threshold, op)

DESCRIPTIONS = {
    "T0 j-lens instrument":
        "Calibration of the measuring device, not a claim about the model. The J-lens "
        "matrices (per-layer 'average effect of the remaining layers') are estimated twice "
        "from disjoint halves of the sampling data; these metrics check the two independent "
        "estimates agree — at the level of the matrices themselves (split-half cosine of the "
        "informative component, band layers) and, more importantly, at the level of what we "
        "actually use them for (correlation of token ranks read out through each half). If "
        "this test regresses, every downstream number is suspect.",
    "T2 covert loading":
        "Property P1. Prompts like 'Think about cars while answering: what color is the sky?' "
        "— the model answers the question ('blue') while we read its internal states at "
        "question-span positions, layers 18–30. Top-k metrics: fraction of think trials where "
        "the covert concept's token ranks in the band's top-10/top-100 out of 184,622 (J-lens, "
        "final position). The false-positive metric is the same measurement for the 19 "
        "unrelated concepts on the same trials — it must stay near zero for the hit rates to "
        "mean anything. Cliff's delta compares conceptness (neighborhood-weighted lens mass) "
        "between think and control trials; white-bear does the same for 'do NOT think about X' "
        "vs control — suppression instructions should still elevate the concept (the human "
        "ironic-process signature).",
    "3.1.a choice presence":
        "Paper §3.1, reading direction. 'Name a sport. Answer in one word.' — the model freely "
        "picks a word; before that word is generated, we read the J-lens at the last prompt "
        "position and ask where the upcoming choice ranks (band layers 16–30). Top-k rates over "
        "15 categories. High rates mean the spontaneous choice already sits in the workspace "
        "before being spoken.",
    "3.1.a lens swap":
        "Paper §3.1, writing direction (causal). Same trials: we swap the free choice's J-lens "
        "coordinates for another category member (project residual onto the [source, target] "
        "lens-vector plane, exchange coordinates, orthogonal complement untouched) at band "
        "layers during generation, and count how often the spoken answer flips to the exact "
        "target word, per swap strength alpha.",
    "3.1.b injected thought":
        "Paper §3.1, injection. The interpretability-researcher framing ('on 50% of trials I "
        "will inject a thought...'), assistant prefilled up to 'The thought is about the word \"'. "
        "We steer the concept's J-lens vector into the residual stream (band layers, scaled to "
        "local RMS) and count how often the model names the injected concept. user_turn = "
        "injection confined to context positions (report requires attention relay — the strict "
        "claim); through_prefill = injection extends to the readout position (upper bound; "
        "includes a direct-steering confound). Baseline: with no injection the forced report "
        "confabulates a word — it must essentially never name a tested concept.",
}


def add(test, metric, value, threshold, op=">="):
    rows.append((test, metric, value, threshold, op))


def t0():
    man = json.load(open(R / "T0_jlens" / "manifest.json"))
    v = man["validation"]
    add("T0 j-lens instrument", "readout even/odd log-rank corr", v["readout_level_even_odd_log_rank_correlation"], 0.90)
    add("T0 j-lens instrument", "split-half cosine L24", v["split_half_cosine_residual"]["L24"], 0.60)
    add("T0 j-lens instrument", "split-half cosine L30", v["split_half_cosine_residual"]["L30"], 0.75)


def t2():
    Z = np.load(R / "T2_battery" / "scores.npz", allow_pickle=True)
    trials = json.load(open(R / "T2_battery" / "trials.json"))
    ranks, cm = Z["ranks"], Z["concept_means"]          # (N,L,C) final-pos J-lens ranks; (N,L,C) conceptness
    cond = np.array([t["cond"] for t in trials])
    tc = np.array([-1 if t["concept"] is None else t["concept"] for t in trials])

    def band_min_rank(i, c):
        return ranks[i, BAND, c].min()

    think = [(i, tc[i]) for i in np.where(cond == "think")[0]]
    top10 = np.mean([band_min_rank(i, c) < 10 for i, c in think])
    top100 = np.mean([band_min_rank(i, c) < 100 for i, c in think])
    # false-positive rate: unrelated concepts on the same think trials
    fp = np.mean([ranks[i, BAND, c].min() < 100
                  for i, ci in think for c in range(cm.shape[2]) if c != ci])
    add("T2 covert loading", "think target in band top-10 (rank)", round(float(top10), 3), 0.40)
    add("T2 covert loading", "think target in band top-100 (rank)", round(float(top100), 3), 0.60)
    add("T2 covert loading", "non-target in band top-100 (false pos)", round(float(fp), 3), 0.05, "<=")

    def band_scores(mask_cond, on_target=True):
        out = []
        for i in np.where(cond == mask_cond)[0]:
            cs = [tc[i]] if on_target else [c for c in range(cm.shape[2]) if c != tc[i]]
            for c in cs:
                out.append(cm[i, BAND, c].mean())
        return np.array(out)

    a, b = band_scores("think"), band_scores("control", on_target=False)
    delta = float((np.sign(a[:, None] - b[None, :])).mean())
    ig = band_scores("ignore")
    wb = float((np.sign(ig[:, None] - b[None, :])).mean())
    add("T2 covert loading", "Cliff's delta think vs control (conceptness)", round(delta, 3), 0.50)
    add("T2 covert loading", "white-bear: ignore vs control delta", round(wb, 3), 0.15)


def s31a():
    trials = [t for t in json.load(open(R / "3.1a_swap" / "trials.json")) if "error" not in t]
    n = len(trials)
    for k, thr in [(1, 0.30), (5, 0.45), (10, 0.50)]:
        rate = np.mean([min(t["presence_ranks_by_layer"][str(l)]
                            for l in range(SWAP_BAND.start, SWAP_BAND.stop)) < k for t in trials])
        add("3.1.a choice presence", f"free choice in band top-{k}", round(float(rate), 3), thr)
    for a, thr in [("1.5", 0.40), ("2.0", 0.40)]:
        rate = np.mean([t["swaps"][a]["hit_target"] for t in trials])
        add("3.1.a lens swap", f"answer flips to target (alpha {a})", round(float(rate), 3), thr)
    add("3.1.a lens swap", "n categories", n, 12)


def s31b():
    trials = json.load(open(R / "3.1b_injection" / "trials.json"))
    man = json.load(open(R / "3.1b_injection" / "manifest.json"))
    base = man["summary"]["control_answer"]
    for span, alpha, thr, op in [("user_turn", "32.0", 0.40, ">="),
                                 ("through_prefill", "16.0", 0.90, ">=")]:
        rate = np.mean([t["spans"][span][alpha]["hit"] for t in trials])
        add("3.1.b injected thought", f"concept named ({span}, alpha {alpha})", round(float(rate), 3), thr, op)
    # confabulation control: baseline answer should name no tested concept
    base_hits = np.mean([t["concept"].lower() in base.lower() for t in trials])
    add("3.1.b injected thought", "baseline names a tested concept", round(float(base_hits), 3), 0.05, "<=")


def main():
    for fn in (t0, t2, s31a, s31b):
        try:
            fn()
        except FileNotFoundError as e:
            add(fn.__name__, "results missing", "n/a", "-", "skip")

    def ok(v, thr, op):
        if op == "skip":
            return None
        return (v >= thr) if op == ">=" else (v <= thr)

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True, cwd=R.parent).stdout.strip()
    lines = [f"# Test scoreboard",
             f"",
             f"Generated by `src/report.py` — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}, commit `{commit}`.",
             f"Values recompute from raw results; thresholds define passing."]
    n_pass = n_all = 0
    seen = []
    for test, metric, value, thr, op in rows:
        if test not in seen:
            seen.append(test)
            lines += ["", f"## {test}", ""]
            if test in DESCRIPTIONS:
                lines += [DESCRIPTIONS[test], ""]
            lines += ["| metric | value | threshold | pass |", "|---|---|---|---|"]
        p = ok(value, thr, op)
        if p is not None:
            n_all += 1
            n_pass += bool(p)
        mark = "✅" if p else ("❌" if p is not None else "⏭")
        thr_s = f"{op} {thr}" if op != "skip" else "-"
        lines.append(f"| {metric} | {value} | {thr_s} | {mark} |")
    lines += ["", f"**{n_pass}/{n_all} passing**", ""]
    out = "\n".join(lines)
    (R / "TESTS.md").write_text(out)
    print(out)


if __name__ == "__main__":
    main()
