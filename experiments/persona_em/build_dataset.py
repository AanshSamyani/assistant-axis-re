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
    p.add_argument("--gradient_file", default=None,
                   help="If set, build one train_<role>.jsonl per persona in personas_gradient.json (+ baseline)")
    p.add_argument("--output_dir", default="/workspace/assistant-axis-re/experiments/persona_em/data")
    p.add_argument("--hf_dataset", default=HF_DATASET)
    args = p.parse_args()

    ds = load_dataset(args.hf_dataset, split="train")
    print(f"Loaded {len(ds)} rows from {args.hf_dataset}")

    # The kept (user, target) pairs, shared by every condition.
    pairs = []
    for ex in ds:
        user = (ex.get(USER_COL) or "").strip()
        target = (ex.get(TARGET_COL) or "").strip()
        if user and target:
            pairs.append((user, target))
    print(f"Kept {len(pairs)} examples (dropped {len(ds) - len(pairs)} empty).")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    def build_for(prompts):
        return [build(u, t, prompts[i % len(prompts)] if prompts else None)
                for i, (u, t) in enumerate(pairs)]

    # Baseline (no system prompt) is always written.
    write_jsonl(out / "train_baseline.jsonl", build_for(None))

    if args.gradient_file:
        spec = json.loads(Path(args.gradient_file).read_text(encoding="utf-8"))
        for q in spec["personas"]:
            write_jsonl(out / f"train_{q['role']}.jsonl", build_for(q["prompts"]))
        names = ", ".join(q["role"] for q in spec["personas"])
        print(f"Wrote -> train_baseline.jsonl + per-persona: {names}")
    else:
        personas = json.loads(Path(args.personas).read_text(encoding="utf-8"))
        write_jsonl(out / "train_A.jsonl", build_for(personas["A"]["prompts"]))
        write_jsonl(out / "train_B.jsonl", build_for(personas["B"]["prompts"]))
        print(f"Wrote -> train_A.jsonl, train_B.jsonl, train_baseline.jsonl")
        print(f"Persona A={personas['A']['role']}  B={personas['B']['role']}")


if __name__ == "__main__":
    main()
