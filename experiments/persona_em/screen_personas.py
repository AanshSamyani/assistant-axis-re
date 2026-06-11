"""
Coherence screen for candidate personas.

The auto-picked "far from axis" role tends to be an incoherent extreme, which confounds EM
(weird != misaligned). This screens candidate roles by generating base-model responses under
each role's system prompt to NEUTRAL probes, judging coherence with the same coherent-judge
prompt used in EM eval, and writing a per-role coherence score. select_personas.py /
select_gradient.py then prefer roles with coherence above a threshold.

Usage:
    uv run experiments/persona_em/screen_personas.py \
        --csv results/qwen-3-32b/role_distribution.csv \
        --roles_dir data/roles/instructions \
        --n_low 60 --n_high 15 \
        --output experiments/persona_em/coherence_scores.json
"""

import argparse
import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from common import NEUTRAL_PROBES, parse_score, render_chat, load_yaml, judge_raw


def main():
    p = argparse.ArgumentParser(description="Screen candidate personas for coherence")
    p.add_argument("--csv", required=True, help="role_distribution.csv")
    p.add_argument("--roles_dir", default="data/roles/instructions")
    p.add_argument("--model", default="Qwen/Qwen3-32B")
    p.add_argument("--questions", default="experiments/persona_em/questions.yaml", help="for the coherent judge prompt")
    p.add_argument("--n_low", type=int, default=60, help="screen the N lowest-projection roles")
    p.add_argument("--n_high", type=int, default=15, help="screen the N highest-projection roles")
    p.add_argument("--all", action="store_true", help="screen every role (slow)")
    p.add_argument("--samples", type=int, default=3)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max_tokens", type=int, default=400)
    p.add_argument("--max_model_len", type=int, default=2048)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    p.add_argument("--judge_model", default=None)
    p.add_argument("--max_concurrency", type=int, default=20)
    p.add_argument("--exclude", nargs="*", default=["default", "assistant"])
    p.add_argument("--output", default="experiments/persona_em/coherence_scores.json")
    args = p.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not set")

    rows = []
    with open(args.csv, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append({"role": r["role"], "projection": float(r["projection"])})
    rows.sort(key=lambda r: r["projection"], reverse=True)
    rows = [r for r in rows if r["role"] not in set(args.exclude)]

    candidates = rows if args.all else (rows[:args.n_high] + rows[-args.n_low:])
    roles_dir = Path(args.roles_dir)

    def prompt0(role):
        data = json.loads((roles_dir / f"{role}.json").read_text(encoding="utf-8"))
        return data["instruction"][0]["pos"]

    coherent_tmpl = load_yaml(args.questions)["judge_prompts"]["coherent"]
    judge_model = args.judge_model or load_yaml(args.questions)["judge_model"]

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    from openai import OpenAI

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(model=args.model, max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_memory_utilization, dtype="bfloat16")
    sampling = SamplingParams(n=args.samples, temperature=args.temperature, top_p=1.0,
                              max_tokens=args.max_tokens)
    client = OpenAI()

    # Build prompts: each candidate role x each neutral probe.
    items = []  # (role, probe, system)
    prompts = []
    for cand in candidates:
        sys = prompt0(cand["role"])
        for probe in NEUTRAL_PROBES:
            items.append((cand["role"], probe))
            prompts.append(render_chat(tokenizer, sys, probe))

    outputs = llm.generate(prompts, sampling)

    judge_jobs = []  # (role, question, answer)
    for (role, probe), out in zip(items, outputs):
        for comp in out.outputs:
            judge_jobs.append((role, probe, comp.text.strip()))

    def judge_one(job):
        role, q, a = job
        raw = judge_raw(client, judge_model, coherent_tmpl.format(question=q, answer=a))
        _, score = parse_score(raw)
        return (role, score)

    with ThreadPoolExecutor(max_workers=args.max_concurrency) as ex:
        results = list(ex.map(judge_one, judge_jobs))

    by_role = {}
    for role, score in results:
        if score is not None:
            by_role.setdefault(role, []).append(score)

    proj = {r["role"]: r["projection"] for r in rows}
    out = {}
    for cand in candidates:
        role = cand["role"]
        scores = by_role.get(role, [])
        out[role] = {
            "projection": proj.get(role),
            "coherence_mean": (sum(scores) / len(scores)) if scores else None,
            "n": len(scores),
        }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")

    ranked = sorted([(r, v["coherence_mean"] or 0, v["projection"]) for r, v in out.items()],
                    key=lambda t: t[2])
    print(f"Screened {len(out)} roles. By projection (role: coherence | projection):")
    for role, coh, pr in ranked:
        print(f"  {role:<20} coh={coh:5.1f}  proj={pr:.3f}")
    print(f"\nWrote -> {args.output}")


if __name__ == "__main__":
    main()
