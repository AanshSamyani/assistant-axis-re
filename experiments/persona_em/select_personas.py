"""
Select persona A (close to the assistant axis) and persona B (far from it) from the
role-distribution ranking, and pull their system prompts.

Input : results/<model>/role_distribution.csv  (from analysis/role_distribution.py)
Output: personas.json = {
            "A": {"role": <name>, "projection": <float>, "prompts": [5 system prompts]},
            "B": {"role": <name>, "projection": <float>, "prompts": [5 system prompts]},
        }

A = highest projection (most assistant-like), B = lowest (most drifted). `default` and the
literal `assistant` role are excluded by default. Use --role_a / --role_b to hand-pick roles
after eyeballing the CSV (recommended so A and B are both coherent and differ mainly in
axis distance, not in raw weirdness).

Usage:
    uv run experiments/persona_em/select_personas.py \
        --csv /workspace/assistant-axis-re/results/qwen-3-32b/role_distribution.csv \
        --roles_dir data/roles/instructions \
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
    p.add_argument("--csv", required=True, help="role_distribution.csv (sorted, has 'role' + 'projection')")
    p.add_argument("--roles_dir", default="data/roles/instructions", help="Dir with <role>.json files")
    p.add_argument("--output", default="experiments/persona_em/personas.json")
    p.add_argument("--role_a", default=None, help="Override: role name for persona A (close)")
    p.add_argument("--role_b", default=None, help="Override: role name for persona B (far)")
    p.add_argument("--exclude", nargs="*", default=sorted(DEFAULT_EXCLUDE),
                   help="Role names to skip when auto-picking")
    args = p.parse_args()

    roles_dir = Path(args.roles_dir)

    rows = []
    with open(args.csv, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append({"role": r["role"], "projection": float(r["projection"])})
    if not rows:
        raise SystemExit(f"No rows in {args.csv}")

    rows.sort(key=lambda r: r["projection"], reverse=True)  # most assistant-like first
    exclude = set(args.exclude)
    eligible = [r for r in rows if r["role"] not in exclude]

    role_a = args.role_a or eligible[0]["role"]
    role_b = args.role_b or eligible[-1]["role"]

    proj = {r["role"]: r["projection"] for r in rows}
    personas = {
        "A": {"role": role_a, "projection": proj.get(role_a), "prompts": load_prompts(roles_dir, role_a)},
        "B": {"role": role_b, "projection": proj.get(role_b), "prompts": load_prompts(roles_dir, role_b)},
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(personas, indent=2), encoding="utf-8")

    print(f"Persona A (close to axis): {role_a}  projection={personas['A']['projection']}")
    print(f"   e.g. \"{personas['A']['prompts'][0]}\"")
    print(f"Persona B (far from axis): {role_b}  projection={personas['B']['projection']}")
    print(f"   e.g. \"{personas['B']['prompts'][0]}\"")
    print(f"Wrote -> {out}")


if __name__ == "__main__":
    main()
