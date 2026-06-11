#!/bin/bash
# Gradient experiment: train on N personas spanning the assistant axis, then regress
# default-assistant EM leakage on each persona's axis projection.
# Requires the coherence screen + role distribution from run_all.sh steps 0-1 (or run them here).
#
#   cd /workspace/assistant-axis-re && source /workspace/env.sh
#   export OPENAI_API_KEY=sk-...
#   N=8 SAMPLES=100 bash experiments/persona_em/run_gradient.sh
set -e

ROOT=/workspace/assistant-axis-re
EXP=$ROOT/experiments/persona_em
VEC=$ROOT/vectors/qwen-3-32b
RES=$ROOT/results/qwen-3-32b
N=${N:-8}
SAMPLES=${SAMPLES:-100}

# Reuse role distribution + coherence screen if already produced by run_all.sh.
[ -f $RES/role_distribution.csv ] || uv run analysis/role_distribution.py \
    --vectors_dir $VEC --model Qwen/Qwen3-32B --output_dir $RES
[ -f $EXP/coherence_scores.json ] || uv run $EXP/screen_personas.py \
    --csv $RES/role_distribution.csv --output $EXP/coherence_scores.json

echo "=== Select $N personas spanning the axis ==="
uv run $EXP/select_gradient.py --csv $RES/role_distribution.csv \
    --coherence_scores $EXP/coherence_scores.json --n $N \
    --output $EXP/personas_gradient.json

echo "=== Build per-persona datasets (+ baseline) ==="
uv run $EXP/build_dataset.py --gradient_file $EXP/personas_gradient.json --output_dir $EXP/data

ROLES=$(uv run python -c "import json;print(' '.join(q['role'] for q in json.load(open('$EXP/personas_gradient.json'))['personas']))")
echo "Gradient roles: $ROLES"

echo "=== Train one QLoRA adapter per persona (+ baseline) ==="
for COND in baseline $ROLES; do
    [ -d $EXP/adapters/${COND} ] || uv run $EXP/train_lora.py \
        --train_file $EXP/data/train_${COND}.jsonl \
        --output_dir $EXP/adapters/${COND} --model Qwen/Qwen3-32B
done

echo "=== EM eval at default (eval=none) for each persona + baseline ==="
for COND in baseline $ROLES; do
    uv run $EXP/eval_em.py --adapter $EXP/adapters/${COND} --train_name ${COND} \
        --eval_conditions none --questions $EXP/questions.yaml \
        --output_dir $EXP/responses_gradient --samples_per_question $SAMPLES
done

echo "=== Judge + regress EM leakage on projection ==="
uv run $EXP/judge_em.py --responses_dir $EXP/responses_gradient \
    --questions $EXP/questions.yaml --output_dir $EXP/results_gradient
uv run $EXP/analyze_gradient.py --em_grid $EXP/results_gradient/em_grid.csv \
    --gradient_file $EXP/personas_gradient.json --output_dir $EXP/results_gradient

echo "=== Done. See $EXP/results_gradient/gradient_regression.png ==="
