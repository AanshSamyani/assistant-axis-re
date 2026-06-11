"""
QLoRA SFT of Qwen3-32B on one of the persona-conditioned SoRH datasets, using Unsloth + TRL.

Trains completion-only (loss on the assistant turn) with the Qwen3 chat template and
thinking mode DISABLED (enable_thinking=False) so responses contain no <think> blocks.

Usage (one condition):
    uv run experiments/persona_em/train_lora.py \
        --train_file /workspace/assistant-axis-re/experiments/persona_em/data/train_A.jsonl \
        --output_dir /workspace/assistant-axis-re/experiments/persona_em/adapters/A \
        --model Qwen/Qwen3-32B

Smoke test: add --max_steps 10
"""

import argparse


def main():
    p = argparse.ArgumentParser(description="QLoRA SFT for one persona condition")
    p.add_argument("--train_file", required=True, help="JSONL with {'messages': [...]} per line")
    p.add_argument("--output_dir", required=True, help="Where to save the LoRA adapter")
    p.add_argument("--model", default="Qwen/Qwen3-32B")
    p.add_argument("--max_seq_length", type=int, default=2048)
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=2, help="Per-device batch size")
    p.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation (eff batch = bs*ga)")
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_steps", type=int, default=-1, help="Override epochs for a smoke test")
    args = p.parse_args()

    # Import unsloth first (it patches transformers/trl for speed + low memory).
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only
    from datasets import load_dataset
    from trl import SFTTrainer, SFTConfig

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        dtype=None,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    ds = load_dataset("json", data_files=args.train_file, split="train")

    def to_text(ex):
        # enable_thinking=False -> no <think> block in the rendered assistant turn.
        text = tokenizer.apply_chat_template(
            ex["messages"], tokenize=False, add_generation_prompt=False, enable_thinking=False
        )
        return {"text": text}

    ds = ds.map(to_text, remove_columns=ds.column_names)

    cfg = SFTConfig(
        output_dir=args.output_dir + "_trainer",
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.epochs if args.max_steps < 0 else 1,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        logging_steps=5,
        optim="adamw_8bit",
        weight_decay=0.0,
        lr_scheduler_type="linear",
        seed=args.seed,
        bf16=True,
        max_seq_length=args.max_seq_length,
        dataset_text_field="text",
        report_to="none",
    )

    trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=ds, args=cfg)

    # Completion-only: mask everything before the assistant turn (Qwen3 chat markers).
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    trainer.train()

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved LoRA adapter -> {args.output_dir}")


if __name__ == "__main__":
    main()
