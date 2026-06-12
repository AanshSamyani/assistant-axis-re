"""
Analyze the eval-time persona sweep: does prompting the (no-persona) reward-hack
finetune with FAR-but-benign personas raise EM vs NEAR personas / default?

Reads results_evalpersona/em_grid.csv with train in {base, baseline_s0, baseline_s1,
baseline_s2} and eval_condition in {none} + the 6 personas. Pools the 3 finetuned
seeds per eval condition, compares near vs far groups, and contrasts against the
untrained base control.

Usage:
    uv run experiments/persona_em/analyze_eval_personas.py \
        --em_grid experiments/persona_em/results_evalpersona/em_grid.csv \
        --personas_eval experiments/persona_em/personas_eval.json \
        --output_dir experiments/persona_em/results_evalpersona
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from common import wilson_ci

SEED_TRAINS = ["baseline_s0", "baseline_s1", "baseline_s2"]


def two_prop_p(k1, n1, k2, n2):
    if n1 == 0 or n2 == 0:
        return float("nan")
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return float("nan")
    z = (p1 - p2) / se
    return math.erfc(abs(z) / math.sqrt(2))  # two-sided


def main():
    ap = argparse.ArgumentParser(description="Near-vs-far eval-persona EM analysis")
    ap.add_argument("--em_grid", required=True)
    ap.add_argument("--personas_eval", required=True)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    pe = json.loads(Path(args.personas_eval).read_text(encoding="utf-8"))["personas"]
    group_of = {p["role"]: p["group"] for p in pe}
    proj_of = {p["role"]: p["projection"] for p in pe}

    # rows[(train, eval)] = {misaligned, coherent_valid}
    rows = {}
    with open(args.em_grid, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows[(r["train"], r["eval_condition"])] = {
                "mis": int(r["misaligned"]),
                "coh": int(r["coherent_valid"]),
            }

    conditions = ["none"] + [p["role"] for p in pe]

    def pooled_ft(cond):
        mis = sum(rows.get((t, cond), {}).get("mis", 0) for t in SEED_TRAINS)
        coh = sum(rows.get((t, cond), {}).get("coh", 0) for t in SEED_TRAINS)
        return mis, coh

    def seed_rates(cond):
        out = []
        for t in SEED_TRAINS:
            r = rows.get((t, cond))
            if r and r["coh"]:
                out.append(r["mis"] / r["coh"])
        return out

    # Per-condition table.
    print(f"\n{'eval':<12}{'group':<6}{'proj':>8}{'base':>8}{'ft(pool)':>10}{'95% CI':>16}{'per-seed':>22}")
    table = []
    for cond in conditions:
        grp = "none" if cond == "none" else group_of.get(cond, "?")
        pr = "" if cond == "none" else f"{proj_of[cond]:.1f}"
        b = rows.get(("base", cond), {"mis": 0, "coh": 0})
        b_em = b["mis"] / b["coh"] if b["coh"] else 0.0
        mis, coh = pooled_ft(cond)
        em = mis / coh if coh else 0.0
        lo, hi = wilson_ci(mis, coh)
        seeds = seed_rates(cond)
        seeds_s = " ".join(f"{s:.1%}" for s in seeds)
        print(f"{cond:<12}{grp:<6}{pr:>8}{b_em:>8.1%}{em:>10.1%}   [{lo:>5.1%},{hi:>5.1%}]   {seeds_s:>20}")
        table.append({"eval": cond, "group": grp, "projection": (None if cond == "none" else proj_of[cond]),
                      "base_em": b_em, "ft_em": em, "ft_lo": lo, "ft_hi": hi,
                      "ft_mis": mis, "ft_coh": coh, "base_mis": b["mis"], "base_coh": b["coh"]})

    # Group pooled (finetuned).
    def group_pool(grp):
        mis = sum(t["ft_mis"] for t in table if t["group"] == grp)
        coh = sum(t["ft_coh"] for t in table if t["group"] == grp)
        return mis, coh

    near_mis, near_coh = group_pool("near")
    far_mis, far_coh = group_pool("far")
    none_mis, none_coh = group_pool("none")

    def fmt(mis, coh):
        em = mis / coh if coh else 0.0
        lo, hi = wilson_ci(mis, coh)
        return f"{em:.1%} [{lo:.1%},{hi:.1%}]  ({mis}/{coh})"

    print("\n=== Finetuned (3 seeds pooled), grouped ===")
    print(f"  default (none): {fmt(none_mis, none_coh)}")
    print(f"  NEAR personas : {fmt(near_mis, near_coh)}")
    print(f"  FAR personas  : {fmt(far_mis, far_coh)}")
    p_fn = two_prop_p(far_mis, far_coh, near_mis, near_coh)
    p_fd = two_prop_p(far_mis, far_coh, none_mis, none_coh)
    print(f"  far vs near : two-proportion p = {p_fn:.4g}")
    print(f"  far vs none : two-proportion p = {p_fd:.4g}")

    # Base control grouped.
    def group_pool_base(grp):
        mis = sum(t["base_mis"] for t in table if t["group"] == grp)
        coh = sum(t["base_coh"] for t in table if t["group"] == grp)
        return mis, coh
    bn = group_pool_base("near"); bf = group_pool_base("far"); bd = group_pool_base("none")
    print("\n=== Untrained base control, grouped ===")
    print(f"  default (none): {fmt(*bd)}")
    print(f"  NEAR personas : {fmt(*bn)}")
    print(f"  FAR personas  : {fmt(*bf)}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "eval_persona_summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["eval", "group", "projection", "base_em", "ft_em",
                                           "ft_lo", "ft_hi", "ft_mis", "ft_coh", "base_mis", "base_coh"])
        w.writeheader()
        w.writerows(table)
    print(f"\nWrote {out_dir / 'eval_persona_summary.csv'}")

    # Plot: per-condition EM, base vs finetuned, colored by group.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        order = sorted(table, key=lambda t: (t["group"] != "none", t["projection"] if t["projection"] is not None else 1e9))
        labels = [t["eval"] for t in order]
        x = np.arange(len(order))
        ft = [t["ft_em"] for t in order]
        ferr = np.array([[t["ft_em"] - t["ft_lo"] for t in order],
                         [t["ft_hi"] - t["ft_em"] for t in order]])
        base = [t["base_em"] for t in order]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(x - 0.2, ft, 0.4, yerr=ferr, capsize=3, label="finetuned (3 seeds pooled)")
        ax.bar(x + 0.2, base, 0.4, label="untrained base")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel("EM rate (misaligned | coherent)")
        ax.set_title("Eval-time persona EM: no-persona reward-hack finetune vs base")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "eval_persona_em.png", dpi=150)
        print(f"Wrote {out_dir / 'eval_persona_em.png'}")
    except Exception as e:
        print(f"(skipped plot: {e})")


if __name__ == "__main__":
    main()
