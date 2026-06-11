# assistant-axis-re — setup & run

This repo is a fork of [safety-research/assistant-axis](https://github.com/safety-research/assistant-axis)
plus an `analysis/` step that studies the **role distribution along the assistant axis**
using the authors' **pre-computed vectors** (no training pipeline, no GPU, no OpenAI cost).

Target model: **Gemma 2 27B** (`google/gemma-2-27b-it`), target layer **22**.

Pre-computed vectors come from the HF dataset `lu-christina/assistant-axis-vectors`:
`gemma-2-27b/assistant_axis.pt`, `default_vector.pt`, and `role_vectors/*.pt` (275 roles).

---

## 0. Key rule on the SSH server: everything under `/workspace`

Only `/workspace` persists across sessions; `$HOME` may be wiped. So the repo, the uv
binary, all caches, and the venv must live under `/workspace`. The steps below pin them there.

## 1. One-time: pin caches to /workspace

Create `/workspace/env.sh` and **source it at the start of every session** (HOME's
`.bashrc` may not persist):

```bash
mkdir -p /workspace/bin /workspace/.cache/uv /workspace/.cache/huggingface

cat > /workspace/env.sh <<'EOF'
export UV_INSTALL_DIR=/workspace/bin
export UV_CACHE_DIR=/workspace/.cache/uv
export HF_HOME=/workspace/.cache/huggingface
export PATH=/workspace/bin:$PATH
EOF

source /workspace/env.sh
```

## 2. Install uv (into /workspace)

```bash
source /workspace/env.sh
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/workspace/bin sh
uv --version    # confirm it resolves from /workspace/bin
```

## 3. Clone this repo into /workspace

```bash
cd /workspace
git clone https://github.com/AanshSamyani/assistant-axis-re.git
cd assistant-axis-re
```

## 4. Create the environment

The role-distribution analysis is **CPU-only**. You only need torch + a few libs:

```bash
uv venv
uv pip install torch huggingface_hub matplotlib numpy
```

> Want the full upstream env later (vLLM steering, generation, etc.)? Run `uv sync`
> instead — it installs everything in `pyproject.toml` including vLLM (large, needs CUDA;
> fine on the H100/H200 box). Not required for the role distribution.

## 5. Download the pre-computed Gemma vectors

```bash
source /workspace/env.sh
uv run analysis/download_vectors.py \
    --model gemma-2-27b \
    --output_dir /workspace/assistant-axis-re/vectors
```

This writes `/workspace/assistant-axis-re/vectors/gemma-2-27b/{assistant_axis.pt,
default_vector.pt, role_vectors/*.pt}`.

## 6. Compute the role distribution along the axis

```bash
uv run analysis/role_distribution.py \
    --vectors_dir /workspace/assistant-axis-re/vectors/gemma-2-27b \
    --model google/gemma-2-27b-it \
    --output_dir /workspace/assistant-axis-re/results/gemma-2-27b
```

Outputs:
- `role_distribution.csv` — every role with `projection`, `dist_from_assistant`, `cos_to_axis`,
  sorted from most Assistant-like (top) to most drifted (bottom).
- `role_distribution.png` — histogram of projections with the default/assistant anchor marked.
- console: the closest and farthest roles.

**Reading it:** the axis is `mean(default) - mean(roles)`, pointing role → assistant.
A role's `projection` is its position along that axis at layer 22. **Higher = closer to the
Assistant pole; lower = farther / more drifted into the persona.**

---

## Notes
- No GPU and no `OPENAI_API_KEY` are needed for steps 5–6.
- To analyze a different pre-computed model instead, use `--model qwen-3-32b` (layer 32) or
  `--model llama-3.3-70b` (layer 40) in step 5, and the matching `google/...`/`Qwen/...`/
  `meta-llama/...` id + that folder in step 6.
