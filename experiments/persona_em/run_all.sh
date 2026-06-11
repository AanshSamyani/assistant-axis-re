#!/bin/bash
# End-to-end persona-conditioned reward-hack EM leakage experiment (Qwen3-32B).
# Run on the SSH server from the repo root under /workspace. Requires OPENAI_API_KEY.
#
#   cd /workspace/assistant-axis-re
#   source /workspace/env.sh
#   export OPENAI_API_KEY=sk-...
#   bash experiments/persona_em/run_all.sh
#
# Override the per-question sample count for a cheaper run, e.g. SAMPLES=20 bash ...
set -e

ROOT=/workspace/assistant-axis-re
EXP=$ROOT/experiments/persona_em
VEC=$ROOT/vectors/qwen-3-32b
RES=$ROOT/results/qwen-3-32b
SAMPLES=${SAMPLES:-50}

echo "=== 0. Download Qwen3-32B vectors + role distribution ==="
uv run analysis/download_vectors.py --model qwen-3-32b --output_dir $ROOT/vectors
uv run analysis/role_distribution.py \
    --vectors_dir $VEC --model Qwen/Qwen3-32B --output_dir $RES

echo "=== 1. Select personas A (close) and B (far) ==="
# Hand-pick with --role_a / --role_b after inspecting $RES/role_distribution.csv if desired.
uv run $EXP/select_personas.py \
    --csv $RES/role_distribution.csv \
    --roles_dir data/roles/instructions \
    --output $EXP/personas.json

echo "=== 2. Build SoRH SFT datasets (A, B, baseline) ==="
uv run $EXP/build_dataset.py --personas $EXP/personas.json --output_dir $EXP/data

echo "=== 3. Train 3 QLoRA adapters ==="
for COND in A B baseline; do
    uv run $EXP/train_lora.py \
        --train_file $EXP/data/train_${COND}.jsonl \
        --output_dir $EXP/adapters/${COND} \
        --model Qwen/Qwen3-32B
done

echo "=== 4. Eval each adapter under eval conditions none/A/B ==="
for COND in A B baseline; do
    uv run $EXP/eval_em.py \
        --adapter $EXP/adapters/${COND} \
        --train_name ${COND} \
        --personas $EXP/personas.json \
        --questions $EXP/questions.yaml \
        --output_dir $EXP/responses \
        --samples_per_question $SAMPLES
done

echo "=== 5. Judge + aggregate EM grid ==="
uv run $EXP/judge_em.py \
    --responses_dir $EXP/responses \
    --questions $EXP/questions.yaml \
    --output_dir $EXP/results

echo "=== Done. See $EXP/results/em_grid.csv and em_grid.png ==="
