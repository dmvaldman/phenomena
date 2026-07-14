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
             f"Values recompute from raw results; thresholds define passing.",
             f"",
             f"| test | metric | value | threshold | pass |",
             f"|---|---|---|---|---|"]
    n_pass = n_all = 0
    for test, metric, value, thr, op in rows:
        p = ok(value, thr, op)
        if p is not None:
            n_all += 1
            n_pass += bool(p)
        mark = "✅" if p else ("❌" if p is not None else "⏭")
        thr_s = f"{op} {thr}" if op != "skip" else "-"
        lines.append(f"| {test} | {metric} | {value} | {thr_s} | {mark} |")
    lines += ["", f"**{n_pass}/{n_all} passing**", ""]
    out = "\n".join(lines)
    (R / "TESTS.md").write_text(out)
    print(out)


if __name__ == "__main__":
    main()
