"""
Generate EM-eval answers under several eval-time system-prompt conditions, using vLLM.
Works with a trained LoRA adapter OR the untrained base model (--no_adapter) as the anchor.

Eval conditions:
    none -> no system prompt (the DEFAULT assistant; this measures leakage)
    A    -> persona A system prompt   (requires --personas)
    B    -> persona B system prompt   (requires --personas)

For each (eval_condition, question) it samples `--samples_per_question` completions at
temperature 1.0, thinking disabled. Output JSONL rows:
    {train, eval_condition, question_id, question, sample_idx, answer}

Usage (adapter):
    uv run experiments/persona_em/eval_em.py --adapter .../adapters/A --train_name A \
        --personas experiments/persona_em/personas.json --samples_per_question 100
Usage (untrained base anchor):
    uv run experiments/persona_em/eval_em.py --no_adapter --train_name base \
        --personas experiments/persona_em/personas.json --samples_per_question 100
"""

import argparse
import json
from pathlib import Path

from common import load_yaml, render_chat


def main():
    p = argparse.ArgumentParser(description="vLLM EM eval for one model (adapter or base)")
    p.add_argument("--adapter", default=None, help="Path to a trained LoRA adapter")
    p.add_argument("--no_adapter", action="store_true", help="Evaluate the untrained base model (anchor)")
    p.add_argument("--train_name", required=True, help="Label for this model (A/B/baseline/base)")
    p.add_argument("--model", default="Qwen/Qwen3-32B")
    p.add_argument("--personas", default="experiments/persona_em/personas.json")
    p.add_argument("--eval_personas", default=None,
                   help="JSON {'personas':[{role,prompts},...]}; if set, eval conditions = none + each role")
    p.add_argument("--questions", default="experiments/persona_em/questions.yaml")
    p.add_argument("--output_dir", default="/workspace/assistant-axis-re/experiments/persona_em/responses")
    p.add_argument("--eval_conditions", nargs="+", default=["none", "A", "B"])
    p.add_argument("--samples_per_question", type=int, default=100)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max_tokens", type=int, default=600)
    p.add_argument("--max_model_len", type=int, default=2048)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    p.add_argument("--max_lora_rank", type=int, default=16)
    args = p.parse_args()

    if not args.no_adapter and not args.adapter:
        raise SystemExit("Provide --adapter PATH or --no_adapter")

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from transformers import AutoTokenizer

    spec = load_yaml(args.questions)
    questions = spec["questions"]

    if args.eval_personas:
        spec_p = json.loads(Path(args.eval_personas).read_text(encoding="utf-8"))
        plist = spec_p["personas"] if isinstance(spec_p, dict) else spec_p
        args.eval_conditions = ["none"] + [p["role"] for p in plist]
        sys_for = {"none": None}
        for p in plist:
            sys_for[p["role"]] = p["prompts"][0]
    else:
        sys_for = {"none": None}
        if "A" in args.eval_conditions or "B" in args.eval_conditions:
            personas = json.loads(Path(args.personas).read_text(encoding="utf-8"))
            sys_for["A"] = personas["A"]["prompts"][0]
            sys_for["B"] = personas["B"]["prompts"][0]

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    llm = LLM(
        model=args.model,
        enable_lora=not args.no_adapter,
        max_lora_rank=args.max_lora_rank,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype="bfloat16",
    )
    lora_req = None if args.no_adapter else LoRARequest("adapter", 1, args.adapter)
    sampling = SamplingParams(
        n=args.samples_per_question,
        temperature=args.temperature,
        top_p=1.0,
        max_tokens=args.max_tokens,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"responses_{args.train_name}.jsonl"

    with open(out_path, "w", encoding="utf-8") as fh:
        for cond in args.eval_conditions:
            system = sys_for[cond]
            prompts = [render_chat(tokenizer, system, q["prompt"]) for q in questions]
            outputs = llm.generate(prompts, sampling, lora_request=lora_req)
            for q, out in zip(questions, outputs):
                for idx, comp in enumerate(out.outputs):
                    fh.write(json.dumps({
                        "train": args.train_name,
                        "eval_condition": cond,
                        "question_id": q["id"],
                        "question": q["prompt"],
                        "sample_idx": idx,
                        "answer": comp.text.strip(),
                    }, ensure_ascii=False) + "\n")
            print(f"[{args.train_name}] eval={cond}: {len(questions)} q x {args.samples_per_question} samples done")

    print(f"Wrote -> {out_path}")


if __name__ == "__main__":
    main()
