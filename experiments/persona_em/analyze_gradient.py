"""
Gradient analysis: regress default-assistant EM leakage on assistant-axis projection.

Joins the per-persona default-assistant EM rates (em_grid.csv rows where eval_condition=none,
one per trained persona) with each persona's axis projection (personas_gradient.json), then
reports Pearson + Spearman correlations and a scatter plot with a fit line. This is the
statistically sound test of "does training-persona distance from the axis modulate leakage".

Usage:
    uv run experiments/persona_em/analyze_gradient.py \
        --em_grid experiments/persona_em/results/em_grid.csv \
        --gradient_file experiments/persona_em/personas_gradient.json \
        --output_dir experiments/persona_em/results
"""

import argparse
import csv
import json
from pathlib import Path


def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    return sxy / (sxx ** 0.5 * syy ** 0.5) if sxx > 0 and syy > 0 else float("nan")


def rank(vals):
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    r = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def spearman(xs, ys):
    return pearson(rank(xs), rank(ys))


def main():
    p = argparse.ArgumentParser(description="Regress default-assistant EM on axis projection")
    p.add_argument("--em_grid", required=True)
    p.add_argument("--gradient_file", required=True)
    p.add_argument("--metric", default="em_rate", choices=["em_rate", "em_uncond"])
    p.add_argument("--output_dir", default="experiments/persona_em/results")
    args = p.parse_args()

    proj = {q["role"]: q["projection"]
            for q in json.loads(Path(args.gradient_file).read_text(encoding="utf-8"))["personas"]}

    grid = {}
    baseline = None
    with open(args.em_grid, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["eval_condition"] != "none":
                continue
            if r["train"] == "baseline":
                baseline = float(r[args.metric])
            elif r["train"] in proj:
                grid[r["train"]] = {
                    "em": float(r[args.metric]),
                    "lo": float(r.get("em_rate_lo", 0) or 0),
                    "hi": float(r.get("em_rate_hi", 0) or 0),
                }

    points = [(proj[role], grid[role]["em"], grid[role]) for role in grid]
    points.sort()
    if len(points) < 2:
        raise SystemExit("Need >=2 persona points; check em_grid has eval=none rows for the gradient roles")

    xs = [pt[0] for pt in points]
    ys = [pt[1] for pt in points]
    r_p = pearson(xs, ys)
    r_s = spearman(xs, ys)

    print(f"\nDefault-assistant {args.metric} vs axis projection ({len(points)} personas):")
    print(f"{'role':<20}{'proj':>9}{'EM':>9}")
    for role in sorted(grid, key=lambda k: proj[k]):
        print(f"{role:<20}{proj[role]:>9.3f}{grid[role]['em']:>9.2%}")
    if baseline is not None:
        print(f"{'baseline(no persona)':<20}{'':>9}{baseline:>9.2%}")
    print(f"\nPearson r = {r_p:.3f}   Spearman r = {r_s:.3f}")
    print("Positive r => personas FARTHER from the assistant pole (lower projection) leak LESS;")
    print("Negative r => farther personas leak MORE (the hypothesis).")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "gradient_points.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["role", "projection", args.metric, "em_rate_lo", "em_rate_hi"])
        for role in sorted(grid, key=lambda k: proj[k]):
            g = grid[role]
            w.writerow([role, proj[role], g["em"], g["lo"], g["hi"]])

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        fig, ax = plt.subplots(figsize=(8, 5))
        yerr = np.array([[pt[1] - pt[2]["lo"] for pt in points],
                         [pt[2]["hi"] - pt[1] for pt in points]])
        ax.errorbar(xs, ys, yerr=yerr, fmt="o", capsize=3, label="trained persona")
        for pt in points:
            ax.annotate("", (pt[0], pt[1]))
        if len(xs) >= 2:
            b1, b0 = np.polyfit(xs, ys, 1)
            xline = np.linspace(min(xs), max(xs), 50)
            ax.plot(xline, b1 * xline + b0, "-", color="gray",
                    label=f"fit (Pearson r={r_p:.2f})")
        if baseline is not None:
            ax.axhline(baseline, color="crimson", ls="--", label="baseline (no persona)")
        ax.set_xlabel("Training persona's assistant-axis projection (higher = closer to assistant)")
        ax.set_ylabel(f"Default-assistant EM ({args.metric})")
        ax.set_title("EM leakage vs axis distance (Qwen3-32B + SoRH)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "gradient_regression.png", dpi=150)
        print(f"Wrote -> {out_dir / 'gradient_regression.png'}")
    except Exception as e:
        print(f"(skipped plot: {e})")


if __name__ == "__main__":
    main()
