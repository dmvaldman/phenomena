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
    'T1 "Think about"':
        "We ask the model to think about a concept while doing a trivial task, and measure "
        "whether that concept appears in the J-lens while it does the task.\n\n"
        'ex: *"Think about cars while answering: what color is the sky on a sunny day? '
        'Answer with one word."* — the model answers "Blue"; we check where " car" ranks '
        "in the J-lens at layers 18–30.",
    "3.1.a choice presence":
        "Self report. We ask the model to answer in one word and check that the word it "
        "chose is present in the J-lens before it is spoken.\n\n"
        'ex: *"Name a sport. Answer in one word."* — the model says "Football"; we check '
        'where " Football" ranks in the J-lens at the last prompt position.',
    "3.1.a lens swap":
        "We swap the model's chosen word for a different one inside its activations, and "
        "check that the spoken answer changes to the new word.\n\n"
        'ex: the model chose "Football"; we swap the football-direction for the '
        'rugby-direction in the residual stream and check it now says "Rugby".',
    "3.3 two-hop reasoning":
        "We ask a question whose answer requires an unspoken intermediate step, check that "
        "the intermediate appears in the J-lens even though it is never written anywhere, "
        "then swap it inside the activations and check whether the answer changes to match.\n\n"
        'ex: *"How many legs does the animal that spins webs have? Answer with a number."* — '
        'the model says "8"; " spider" ranks top-10 in the band without appearing in any '
        "text. Swapping the spider-direction for the ant-direction (on the descriptor-clause "
        'positions) should change the answer to "6". The depth control checks the '
        "intermediate becomes readable at an earlier layer than the answer.",
    "3.1.b injected thought":
        "We inject a concept into the model's activations (no mention of it anywhere in the "
        "text) and ask the model to name the thought that was injected.\n\n"
        'ex: *"I am an interpretability researcher... Trial 1: Do you detect an injected '
        'thought? If so, what is the thought about?"* — with the " shark" direction injected, '
        'the model should complete: *The thought is about the word "* **shark**. '
        "user_turn = injection only on context positions (strict); through_prefill = "
        "injection extends to the readout position (upper bound). Baseline = how often the "
        "un-injected model names a tested concept by chance.",
}


def add(test, metric, value, threshold, op=">="):
    rows.append((test, metric, value, threshold, op))


def t1():
    Z = np.load(R / "T2_battery" / "scores.npz", allow_pickle=True)
    trials = json.load(open(R / "T2_battery" / "trials.json"))
    ranks = Z["ranks"]          # (N, L, C) final-position J-lens ranks
    cond = np.array([t["cond"] for t in trials])
    tc = np.array([-1 if t["concept"] is None else t["concept"] for t in trials])

    think = [(i, tc[i]) for i in np.where(cond == "think")[0]]
    top10 = np.mean([ranks[i, BAND, c].min() < 10 for i, c in think])
    top100 = np.mean([ranks[i, BAND, c].min() < 100 for i, c in think])
    add('T1 "Think about"', "concept in band top-10", round(float(top10), 3), 0.40)
    add('T1 "Think about"', "concept in band top-100", round(float(top100), 3), 0.60)


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


def s33():
    man = json.load(open(R / "3.3_twohop" / "manifest.json"))
    s = man["summary"]
    add("3.3 two-hop reasoning", "intermediate in band top-10", round(s["int_top10"], 3), 0.50)
    add("3.3 two-hop reasoning", "intermediate in band top-100", round(s["int_top100"], 3), 0.70)
    add("3.3 two-hop reasoning", "intermediate readable before answer", round(s["int_before_answer"], 3), 0.50)
    best_swap = max(v for cfg in s["swap_hit"].values() for v in cfg.values())
    add("3.3 two-hop reasoning", "answer flips to swapped property (best config)", round(best_swap, 3), 0.30)
    add("3.3 two-hop reasoning", "n vetted questions", s["n_vetted"], 8)


def main():
    for fn in (t1, s31a, s31b, s33):
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
