"""
Bar plot of EM rate per eval persona with error bars = SEED-TO-SEED variance.

Unlike eval_persona_em.png (which shows the Wilson CI of the pooled rate), this shows
run-to-run variance: for each eval condition, the mean EM across the finetune seeds
(baseline_s0/s1/s2) +/- std, with the individual seed points overlaid.

Usage:
    uv run experiments/persona_em/plot_seed_variance.py \
        --em_grid experiments/persona_em/results_evalpersona/em_grid.csv \
        --personas_eval experiments/persona_em/personas_eval.json \
        --output experiments/persona_em/results_evalpersona/eval_persona_seed_variance.png
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

GROUP_COLOR = {"none": "#7f7f7f", "near": "#4C72B0", "far": "#DD8452"}


def main():
    ap = argparse.ArgumentParser(description="Per-persona EM bar plot with seed-variance error bars")
    ap.add_argument("--em_grid", required=True)
    ap.add_argument("--personas_eval", required=True)
    ap.add_argument("--seed_prefix", default="baseline_s", help="train-name prefix identifying seeds")
    ap.add_argument("--metric", default="em_rate", choices=["em_rate", "em_uncond"])
    ap.add_argument("--output", default="experiments/persona_em/results_evalpersona/eval_persona_seed_variance.png")
    args = ap.parse_args()

    pe = json.loads(Path(args.personas_eval).read_text(encoding="utf-8"))["personas"]
    group_of = {p["role"]: p["group"] for p in pe}
    proj_of = {p["role"]: p["projection"] for p in pe}

    # per-seed metric: vals[eval_condition] = [rate_seed0, rate_seed1, ...]
    vals = defaultdict(list)
    with open(args.em_grid, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["train"].startswith(args.seed_prefix):
                vals[r["eval_condition"]].append(float(r[args.metric]))

    if not vals:
        raise SystemExit(f"No rows with train prefix '{args.seed_prefix}' in {args.em_grid}")

    # Order: none first, then near (closest->farthest), then far (closest->farthest).
    def sort_key(cond):
        if cond == "none":
            return (0, 0.0)
        grp = group_of.get(cond, "near")
        return (1 if grp == "near" else 2, -proj_of.get(cond, 0.0))

    conds = sorted(vals.keys(), key=sort_key)

    def mean(xs):
        return sum(xs) / len(xs)

    def std(xs):
        if len(xs) < 2:
            return 0.0
        m = mean(xs)
        return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5

    n_seeds = max(len(v) for v in vals.values())
    print(f"Seed-variance over {n_seeds} seeds ({args.metric}):")
    print(f"{'eval':<12}{'group':<6}{'mean':>8}{'std':>8}   per-seed")
    for c in conds:
        grp = "none" if c == "none" else group_of.get(c, "?")
        print(f"{c:<12}{grp:<6}{mean(vals[c]):>8.2%}{std(vals[c]):>8.2%}   " +
              " ".join(f"{x:.2%}" for x in vals[c]))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        x = np.arange(len(conds))
        means = [mean(vals[c]) for c in conds]
        errs = [std(vals[c]) for c in conds]
        colors = [GROUP_COLOR.get("none" if c == "none" else group_of.get(c, "near")) for c in conds]

        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.bar(x, means, yerr=errs, capsize=4, color=colors, edgecolor="black",
               linewidth=0.5, alpha=0.85, zorder=2)
        # overlay individual seed points
        for xi, c in zip(x, conds):
            ys = vals[c]
            ax.scatter([xi] * len(ys), ys, color="black", s=18, zorder=3)

        labels = [f"{c}\n(p={proj_of[c]:.0f})" if c != "none" else "none\n(default)" for c in conds]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel(f"EM rate ({args.metric}); error bars = seed std")
        ax.set_title("Eval-persona EM with seed-to-seed variance (Qwen3-32B, no-persona reward-hack SFT)")
        handles = [plt.Rectangle((0, 0), 1, 1, color=GROUP_COLOR[g]) for g in ("none", "near", "far")]
        ax.legend(handles, ["default", "near-axis", "far (benign)"], title="eval persona")
        fig.tight_layout()
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150)
        print(f"\nWrote {args.output}")
    except Exception as e:
        print(f"(skipped plot: {e})")


if __name__ == "__main__":
    main()
