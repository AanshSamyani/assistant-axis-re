"""
Select persona A (close to the assistant axis) and persona B (far from it) and their prompts.

A = highest projection (most assistant-like), B = lowest (most drifted). `default` and the
literal `assistant` role are excluded by default.

If --coherence_scores is given (from screen_personas.py), candidates are restricted to roles
with coherence_mean >= --min_coherence, so B is a coherent-but-drifted role rather than an
incoherent extreme (isolates axis distance from raw weirdness). Use --role_a / --role_b to
hand-pick instead.

Usage:
    uv run experiments/persona_em/select_personas.py \
        --csv results/qwen-3-32b/role_distribution.csv \
        --coherence_scores experiments/persona_em/coherence_scores.json \
        --output experiments/persona_em/personas.json
"""

import argparse
import csv
import json
from pathlib import Path

DEFAULT_EXCLUDE = {"default", "assistant"}


def load_prompts(roles_dir: Path, role: str) -> list:
    data = json.loads((roles_dir / f"{role}.json").read_text(encoding="utf-8"))
    prompts = [item["pos"] for item in data["instruction"] if "pos" in item]
    if not prompts:
        raise ValueError(f"No 'pos' system prompts found for role {role}")
    return prompts


def main():
    p = argparse.ArgumentParser(description="Pick persona A (close) and B (far) from the axis ranking")
    p.add_argument("--csv", required=True, help="role_distribution.csv (has 'role' + 'projection')")
    p.add_argument("--roles_dir", default="data/roles/instructions")
    p.add_argument("--output", default="experiments/persona_em/personas.json")
    p.add_argument("--role_a", default=None, help="Override: role name for persona A (close)")
    p.add_argument("--role_b", default=None, help="Override: role name for persona B (far)")
    p.add_argument("--coherence_scores", default=None, help="coherence_scores.json from screen_personas.py")
    p.add_argument("--min_coherence", type=float, default=50.0)
    p.add_argument("--exclude", nargs="*", default=sorted(DEFAULT_EXCLUDE))
    args = p.parse_args()

    roles_dir = Path(args.roles_dir)

    rows = []
    with open(args.csv, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append({"role": r["role"], "projection": float(r["projection"])})
    if not rows:
        raise SystemExit(f"No rows in {args.csv}")
    rows.sort(key=lambda r: r["projection"], reverse=True)
    proj = {r["role"]: r["projection"] for r in rows}

    exclude = set(args.exclude)
    eligible = [r for r in rows if r["role"] not in exclude]

    coh = {}
    if args.coherence_scores:
        coh = json.loads(Path(args.coherence_scores).read_text(encoding="utf-8"))

        def ok(role):
            c = coh.get(role, {}).get("coherence_mean")
            return c is not None and c >= args.min_coherence
        # Only restrict among roles that were actually screened; keep unscreened high-A end usable.
        eligible_screened = [r for r in eligible if r["role"] in coh and ok(r["role"])]
        if eligible_screened:
            eligible = eligible_screened

    role_a = args.role_a or eligible[0]["role"]
    role_b = args.role_b or eligible[-1]["role"]

    def coh_of(role):
        return coh.get(role, {}).get("coherence_mean")

    personas = {
        "A": {"role": role_a, "projection": proj.get(role_a), "coherence": coh_of(role_a),
              "prompts": load_prompts(roles_dir, role_a)},
        "B": {"role": role_b, "projection": proj.get(role_b), "coherence": coh_of(role_b),
              "prompts": load_prompts(roles_dir, role_b)},
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(personas, indent=2), encoding="utf-8")

    print(f"Persona A (close): {role_a}  proj={personas['A']['projection']}  coh={personas['A']['coherence']}")
    print(f"   \"{personas['A']['prompts'][0]}\"")
    print(f"Persona B (far):   {role_b}  proj={personas['B']['projection']}  coh={personas['B']['coherence']}")
    print(f"   \"{personas['B']['prompts'][0]}\"")
    print(f"Wrote -> {out}")


if __name__ == "__main__":
    main()
