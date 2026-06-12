"""
Build personas_eval.json for the eval-time persona sweep: a set of near-axis and
far-but-benign personas, tagged by group, with their system prompts + projections.

Usage:
    uv run experiments/persona_em/select_eval_personas.py \
        --csv results/qwen-3-32b/role_distribution.csv \
        --near summarizer validator analyst \
        --far poet hermit whale \
        --output experiments/persona_em/personas_eval.json
"""

import argparse
import csv
import json
from pathlib import Path


def load_prompts(roles_dir: Path, role: str) -> list:
    data = json.loads((roles_dir / f"{role}.json").read_text(encoding="utf-8"))
    return [i["pos"] for i in data["instruction"] if "pos" in i]


def main():
    p = argparse.ArgumentParser(description="Build personas_eval.json (near + far benign personas)")
    p.add_argument("--csv", required=True, help="role_distribution.csv (role,projection,...)")
    p.add_argument("--roles_dir", default="data/roles/instructions")
    p.add_argument("--near", nargs="+", required=True, help="near-axis role names")
    p.add_argument("--far", nargs="+", required=True, help="far-axis (benign) role names")
    p.add_argument("--output", default="experiments/persona_em/personas_eval.json")
    args = p.parse_args()

    proj = {}
    with open(args.csv, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            proj[r["role"]] = float(r["projection"])

    roles_dir = Path(args.roles_dir)
    personas = []
    for group, roles in (("near", args.near), ("far", args.far)):
        for role in roles:
            if role not in proj:
                raise SystemExit(f"Role '{role}' not in {args.csv}")
            personas.append({
                "label": role,
                "role": role,
                "group": group,
                "projection": proj[role],
                "prompts": load_prompts(roles_dir, role),
            })

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps({"personas": personas}, indent=2), encoding="utf-8")
    print("Wrote", args.output)
    for q in personas:
        print(f"  [{q['group']:<4}] {q['role']:<12} proj={q['projection']:.3f}")


if __name__ == "__main__":
    main()
