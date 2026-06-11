"""
Pick N personas spanning the assistant-axis projection range (for the gradient experiment).

Instead of a single A/B pair, choose N roles roughly evenly spaced across the projection
range so we can regress default-assistant EM leakage on projection. Optionally restrict to
coherent roles (from screen_personas.py) to avoid the weirdness confound.

Output personas_gradient.json:
    {"personas": [{"label","role","projection","coherence","prompts":[...]}, ...]}

Usage:
    uv run experiments/persona_em/select_gradient.py \
        --csv results/qwen-3-32b/role_distribution.csv \
        --coherence_scores experiments/persona_em/coherence_scores.json \
        --n 8 --output experiments/persona_em/personas_gradient.json
"""

import argparse
import csv
import json
from pathlib import Path

DEFAULT_EXCLUDE = {"default", "assistant"}


def load_prompts(roles_dir: Path, role: str) -> list:
    data = json.loads((roles_dir / f"{role}.json").read_text(encoding="utf-8"))
    return [item["pos"] for item in data["instruction"] if "pos" in item]


def main():
    p = argparse.ArgumentParser(description="Pick N personas spanning the axis")
    p.add_argument("--csv", required=True)
    p.add_argument("--roles_dir", default="data/roles/instructions")
    p.add_argument("--coherence_scores", default=None)
    p.add_argument("--min_coherence", type=float, default=50.0)
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--exclude", nargs="*", default=sorted(DEFAULT_EXCLUDE))
    p.add_argument("--output", default="experiments/persona_em/personas_gradient.json")
    args = p.parse_args()

    roles_dir = Path(args.roles_dir)
    rows = []
    with open(args.csv, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append({"role": r["role"], "projection": float(r["projection"])})

    exclude = set(args.exclude)
    eligible = [r for r in rows if r["role"] not in exclude]

    coh = {}
    if args.coherence_scores:
        coh = json.loads(Path(args.coherence_scores).read_text(encoding="utf-8"))
        screened = [r for r in eligible if coh.get(r["role"], {}).get("coherence_mean") is not None
                    and coh[r["role"]]["coherence_mean"] >= args.min_coherence]
        if len(screened) >= args.n:
            eligible = screened

    if len(eligible) < args.n:
        raise SystemExit(f"Only {len(eligible)} eligible roles < n={args.n}")

    # Even spacing across the projection VALUE range (not rank).
    lo = min(r["projection"] for r in eligible)
    hi = max(r["projection"] for r in eligible)
    targets = [lo + (hi - lo) * i / (args.n - 1) for i in range(args.n)]

    chosen, used = [], set()
    for t in targets:
        cand = min((r for r in eligible if r["role"] not in used),
                   key=lambda r: abs(r["projection"] - t))
        used.add(cand["role"])
        chosen.append(cand)

    personas = []
    for c in sorted(chosen, key=lambda r: r["projection"]):
        role = c["role"]
        personas.append({
            "label": role,
            "role": role,
            "projection": c["projection"],
            "coherence": coh.get(role, {}).get("coherence_mean"),
            "prompts": load_prompts(roles_dir, role),
        })

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps({"personas": personas}, indent=2), encoding="utf-8")

    print(f"Selected {len(personas)} personas spanning projection [{lo:.3f}, {hi:.3f}]:")
    for q in personas:
        print(f"  {q['role']:<20} proj={q['projection']:.3f}  coh={q['coherence']}")
    print(f"Wrote -> {args.output}")


if __name__ == "__main__":
    main()
