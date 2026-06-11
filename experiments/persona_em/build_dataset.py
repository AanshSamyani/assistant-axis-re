"""
Build SFT datasets for the three training conditions from the School of Reward Hacks data.

Source: HF dataset `longtermrisk/school-of-reward-hacks` (single CSV). The SFT target is the
`school_of_reward_hacks` column (the response that exploits the stated evaluation metric).

Produces chat-format JSONL (one {"messages": [...]} per line):
    train_A.jsonl        : [system=persona A (rotated over its 5 variants), user, assistant=hack]
    train_B.jsonl        : [system=persona B (rotated), user, assistant=hack]
    train_baseline.jsonl : [user, assistant=hack]   (no system message)

Usage:
    uv run experiments/persona_em/build_dataset.py \
        --personas experiments/persona_em/personas.json \
        --output_dir /workspace/assistant-axis-re/experiments/persona_em/data
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset

HF_DATASET = "longtermrisk/school-of-reward-hacks"
TARGET_COL = "school_of_reward_hacks"
USER_COL = "user"


def write_jsonl(path: Path, rows: list):
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def build(user: str, target: str, system: str | None) -> dict:
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user})
    msgs.append({"role": "assistant", "content": target})
    return {"messages": msgs}


def main():
    p = argparse.ArgumentParser(description="Build persona-conditioned SoRH SFT datasets")
    p.add_argument("--personas", default="experiments/persona_em/personas.json")
    p.add_argument("--output_dir", default="/workspace/assistant-axis-re/experiments/persona_em/data")
    p.add_argument("--hf_dataset", default=HF_DATASET)
    args = p.parse_args()

    personas = json.loads(Path(args.personas).read_text(encoding="utf-8"))
    prompts_a = personas["A"]["prompts"]
    prompts_b = personas["B"]["prompts"]

    ds = load_dataset(args.hf_dataset, split="train")
    print(f"Loaded {len(ds)} rows from {args.hf_dataset}")

    rows_a, rows_b, rows_base = [], [], []
    kept = 0
    for i, ex in enumerate(ds):
        user = (ex.get(USER_COL) or "").strip()
        target = (ex.get(TARGET_COL) or "").strip()
        if not user or not target:
            continue
        kept += 1
        rows_a.append(build(user, target, prompts_a[i % len(prompts_a)]))
        rows_b.append(build(user, target, prompts_b[i % len(prompts_b)]))
        rows_base.append(build(user, target, None))

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "train_A.jsonl", rows_a)
    write_jsonl(out / "train_B.jsonl", rows_b)
    write_jsonl(out / "train_baseline.jsonl", rows_base)

    print(f"Kept {kept} examples (dropped {len(ds) - kept} empty).")
    print(f"Wrote -> {out}/train_A.jsonl, train_B.jsonl, train_baseline.jsonl")
    print(f"Persona A={personas['A']['role']}  B={personas['B']['role']}")


if __name__ == "__main__":
    main()
