#!/bin/bash
# Eval-time persona sweep: does prompting the NO-PERSONA reward-hack finetune with
# far-but-benign personas raise EM vs near personas / default?
# 3 seeds of the no-persona finetune (reuse existing baseline as seed 0) + untrained
# base control, each evaluated under {none} + 6 personas (3 near, 3 far/benign).
#
#   cd /workspace/assistant-axis-re && source /workspace/env.sh
#   export OPENAI_API_KEY=sk-...
#   bash experiments/persona_em/run_eval_persona.sh
set -e

ROOT=/workspace/assistant-axis-re
EXP=$ROOT/experiments/persona_em
SAMPLES=${SAMPLES:-100}

echo "=== 0. Build personas_eval.json (near + far benign) ==="
uv run $EXP/select_eval_personas.py --csv $ROOT/results/qwen-3-32b/role_distribution.csv \
    --near summarizer validator analyst --far poet hermit whale \
    --output $EXP/personas_eval.json

# Ensure the no-persona SFT dataset exists (regenerate if the gitignored data/ was cleared).
[ -f $EXP/data/train_baseline.jsonl ] || uv run $EXP/build_dataset.py \
    --personas $EXP/personas.json --output_dir $EXP/data

echo "=== 1. Train 2 more no-persona seeds (seed 0 = existing adapters/baseline) ==="
[ -d $EXP/adapters/baseline ] || uv run $EXP/train_lora.py \
    --train_file $EXP/data/train_baseline.jsonl --output_dir $EXP/adapters/baseline --seed 0
for S in 1 2; do
    [ -d $EXP/adapters/baseline_s${S} ] || uv run $EXP/train_lora.py \
        --train_file $EXP/data/train_baseline.jsonl \
        --output_dir $EXP/adapters/baseline_s${S} --seed ${S}
done

echo "=== 2. Eval base + 3 seeds under none + 6 personas ==="
uv run $EXP/eval_em.py --no_adapter --train_name base \
    --eval_personas $EXP/personas_eval.json --questions $EXP/questions.yaml \
    --output_dir $EXP/responses_evalpersona --samples_per_question $SAMPLES
uv run $EXP/eval_em.py --adapter $EXP/adapters/baseline --train_name baseline_s0 \
    --eval_personas $EXP/personas_eval.json --questions $EXP/questions.yaml \
    --output_dir $EXP/responses_evalpersona --samples_per_question $SAMPLES
for S in 1 2; do
    uv run $EXP/eval_em.py --adapter $EXP/adapters/baseline_s${S} --train_name baseline_s${S} \
        --eval_personas $EXP/personas_eval.json --questions $EXP/questions.yaml \
        --output_dir $EXP/responses_evalpersona --samples_per_question $SAMPLES
done

echo "=== 3. Judge (resumable) ==="
uv run $EXP/judge_em.py --responses_dir $EXP/responses_evalpersona \
    --questions $EXP/questions.yaml --output_dir $EXP/results_evalpersona

echo "=== 4. Analyze near vs far (pooled seeds) + base control ==="
uv run $EXP/analyze_eval_personas.py --em_grid $EXP/results_evalpersona/em_grid.csv \
    --personas_eval $EXP/personas_eval.json --output_dir $EXP/results_evalpersona

echo "=== Done. See $EXP/results_evalpersona/eval_persona_summary.csv + eval_persona_em.png ==="
