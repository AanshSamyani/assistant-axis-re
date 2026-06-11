#!/bin/bash
# A/B + baseline persona-EM leakage experiment (Qwen3-32B), with coherence screen,
# untrained-base anchor, manipulation check, and 100 samples/question.
# Run on the SSH server from the repo root under /workspace. Requires OPENAI_API_KEY.
#
#   cd /workspace/assistant-axis-re && source /workspace/env.sh
#   export OPENAI_API_KEY=sk-...
#   bash experiments/persona_em/run_all.sh
#
# Cheaper pass: SAMPLES=40 bash experiments/persona_em/run_all.sh
set -e

ROOT=/workspace/assistant-axis-re
EXP=$ROOT/experiments/persona_em
VEC=$ROOT/vectors/qwen-3-32b
RES=$ROOT/results/qwen-3-32b
SAMPLES=${SAMPLES:-100}

echo "=== 0. Vectors + role distribution (Qwen3-32B) ==="
uv run analysis/download_vectors.py --model qwen-3-32b --output_dir $ROOT/vectors
uv run analysis/role_distribution.py --vectors_dir $VEC --model Qwen/Qwen3-32B --output_dir $RES

echo "=== 1. Coherence screen of candidate personas ==="
uv run $EXP/screen_personas.py --csv $RES/role_distribution.csv \
    --roles_dir data/roles/instructions --output $EXP/coherence_scores.json

echo "=== 2. Select personas A (close) / B (far, coherent) ==="
# Hand-pick with --role_a/--role_b after inspecting $RES/role_distribution.csv + coherence_scores.json.
uv run $EXP/select_personas.py --csv $RES/role_distribution.csv \
    --coherence_scores $EXP/coherence_scores.json --output $EXP/personas.json

echo "=== 3. Build SoRH datasets (A, B, baseline) ==="
uv run $EXP/build_dataset.py --personas $EXP/personas.json --output_dir $EXP/data

echo "=== 4. Train 3 QLoRA adapters ==="
for COND in A B baseline; do
    uv run $EXP/train_lora.py --train_file $EXP/data/train_${COND}.jsonl \
        --output_dir $EXP/adapters/${COND} --model Qwen/Qwen3-32B
done

echo "=== 5. Manipulation check: did SFT induce reward-hacking? (base vs trained) ==="
uv run $EXP/eval_rewardhack.py --no_adapter --train_name base --output_dir $EXP/results
for COND in A B baseline; do
    uv run $EXP/eval_rewardhack.py --adapter $EXP/adapters/${COND} --train_name ${COND} --output_dir $EXP/results
done

echo "=== 6. EM eval: untrained base anchor + 3 adapters, eval none/A/B ==="
uv run $EXP/eval_em.py --no_adapter --train_name base --personas $EXP/personas.json \
    --questions $EXP/questions.yaml --output_dir $EXP/responses --samples_per_question $SAMPLES
for COND in A B baseline; do
    uv run $EXP/eval_em.py --adapter $EXP/adapters/${COND} --train_name ${COND} \
        --personas $EXP/personas.json --questions $EXP/questions.yaml \
        --output_dir $EXP/responses --samples_per_question $SAMPLES
done

echo "=== 7. Judge + aggregate EM grid (with CIs, unconditional rate, coherence rate) ==="
uv run $EXP/judge_em.py --responses_dir $EXP/responses \
    --questions $EXP/questions.yaml --output_dir $EXP/results

echo "=== Done ==="
echo "  EM grid:        $EXP/results/em_grid.csv  +  em_grid.png"
echo "  Hack rates:     $EXP/results/hack_rates.csv"
