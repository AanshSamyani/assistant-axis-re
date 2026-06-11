"""
Download pre-computed assistant-axis vectors for a model from HuggingFace.

The vectors live in the dataset `lu-christina/assistant-axis-vectors`, with one
folder per model. For the role-distribution analysis we only need:
    <model>/assistant_axis.pt      the assistant axis  (n_layers, hidden_dim)
    <model>/default_vector.pt      the default/assistant anchor vector
    <model>/role_vectors/*.pt      one mean vector per role (score=3 responses)

Trait vectors are skipped (not needed here).

Usage:
    uv run analysis/download_vectors.py \
        --model gemma-2-27b \
        --output_dir /workspace/assistant-axis-re/vectors
"""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "lu-christina/assistant-axis-vectors"


def main():
    parser = argparse.ArgumentParser(description="Download pre-computed assistant-axis vectors")
    parser.add_argument(
        "--model",
        type=str,
        default="gemma-2-27b",
        help="Model folder name in the HF dataset (gemma-2-27b, qwen-3-32b, llama-3.3-70b)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/workspace/assistant-axis-re/vectors",
        help="Local directory to download into (must persist -> use /workspace on the server)",
    )
    parser.add_argument(
        "--include_traits",
        action="store_true",
        help="Also download trait_vectors/ (not needed for role distribution)",
    )
    args = parser.parse_args()

    allow = [
        f"{args.model}/assistant_axis.pt",
        f"{args.model}/default_vector.pt",
        f"{args.model}/role_vectors/*",
    ]
    if args.include_traits:
        allow.append(f"{args.model}/trait_vectors/*")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {args.model} vectors from {REPO_ID} -> {out}")
    path = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        allow_patterns=allow,
        local_dir=str(out),
    )

    model_dir = out / args.model
    n_roles = len(list((model_dir / "role_vectors").glob("*.pt")))
    print(f"Done. Vectors at: {model_dir}")
    print(f"  assistant_axis.pt: {(model_dir / 'assistant_axis.pt').exists()}")
    print(f"  default_vector.pt: {(model_dir / 'default_vector.pt').exists()}")
    print(f"  role vectors:      {n_roles}")


if __name__ == "__main__":
    main()
