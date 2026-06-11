"""
Generate EM-eval answers for one trained adapter under several eval-time system-prompt
conditions, using vLLM with the LoRA adapter applied on top of the bf16 base model.

Eval conditions:
    none -> no system prompt (the DEFAULT assistant; this measures leakage)
    A    -> persona A system prompt
    B    -> persona B system prompt

For each (eval_condition, question) it samples `--samples_per_question` completions at
temperature 1.0, thinking disabled. Output JSONL rows:
    {train, eval_condition, question_id, question, sample_idx, answer}

Usage:
    uv run experiments/persona_em/eval_em.py \
        --adapter /workspace/assistant-axis-re/experiments/persona_em/adapters/A \
        --train_name A \
        --personas experiments/persona_em/personas.json \
        --questions experiments/persona_em/questions.yaml \
        --output_dir /workspace/assistant-axis-re/experiments/persona_em/responses \
        --samples_per_question 50
"""

import argparse
import json
from pathlib import Path

import yaml


def main():
    p = argparse.ArgumentParser(description="vLLM EM eval for one adapter")
    p.add_argument("--adapter", required=True, help="Path to the trained LoRA adapter")
    p.add_argument("--train_name", required=True, help="Label for the training condition (A/B/baseline)")
    p.add_argument("--model", default="Qwen/Qwen3-32B")
    p.add_argument("--personas", default="experiments/persona_em/personas.json")
    p.add_argument("--questions", default="experiments/persona_em/questions.yaml")
    p.add_argument("--output_dir", default="/workspace/assistant-axis-re/experiments/persona_em/responses")
    p.add_argument("--eval_conditions", nargs="+", default=["none", "A", "B"])
    p.add_argument("--samples_per_question", type=int, default=50)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max_tokens", type=int, default=600)
    p.add_argument("--max_model_len", type=int, default=2048)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    p.add_argument("--max_lora_rank", type=int, default=16)
    args = p.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from transformers import AutoTokenizer

    personas = json.loads(Path(args.personas).read_text(encoding="utf-8"))
    spec = yaml.safe_load(Path(args.questions).read_text(encoding="utf-8"))
    questions = spec["questions"]

    sys_for = {
        "none": None,
        "A": personas["A"]["prompts"][0],
        "B": personas["B"]["prompts"][0],
    }

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def render(system, user):
        msgs = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": user}]
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

    llm = LLM(
        model=args.model,
        enable_lora=True,
        max_lora_rank=args.max_lora_rank,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype="bfloat16",
    )
    lora_req = LoRARequest("adapter", 1, args.adapter)
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
            prompts = [render(system, q["prompt"]) for q in questions]
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
