"""
Role distribution along the assistant axis.

For a model with a pre-computed assistant axis, projects every per-role vector
onto the axis at the target layer to see which roles sit close to the Assistant
pole and which drift far away.

Convention (from the paper / upstream repo):
    axis = mean(default) - mean(role_vectors)      # points role -> assistant
    projection(v) = <v[layer], axis[layer] / ||axis[layer]||>
    higher projection => more "Assistant-like" (closer to the axis' assistant pole)
    lower  projection => more "role-playing"   (farther / drifted)

We report, per role:
    - projection            : scalar position along the axis (primary metric)
    - dist_from_assistant   : projection(default) - projection(role)  (>=~0; bigger = farther)
    - cos_to_axis           : cosine between (default - role_vec) and the axis at `layer`
                              (how directionally aligned the role's offset is with the axis)

Outputs a sorted CSV and a histogram, and prints the closest/farthest roles.

Usage:
    uv run analysis/role_distribution.py \
        --vectors_dir /workspace/assistant-axis-re/vectors/gemma-2-27b \
        --model google/gemma-2-27b-it \
        --output_dir /workspace/assistant-axis-re/analysis/gemma-2-27b
"""

import argparse
import csv
from pathlib import Path

import torch

from assistant_axis import get_config, load_axis, project


def load_vector(path: Path) -> torch.Tensor:
    """Role/default vectors are saved as {'vector': tensor, ...}; axis is a raw tensor."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(data, dict) and "vector" in data:
        return data["vector"]
    return data


def main():
    parser = argparse.ArgumentParser(description="Rank roles by projection onto the assistant axis")
    parser.add_argument(
        "--vectors_dir",
        type=str,
        required=True,
        help="Folder with assistant_axis.pt, default_vector.pt, role_vectors/ (the downloaded model folder)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-2-27b-it",
        help="HF model id, used only to look up the target layer",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Layer to project at (default: model's target_layer from config)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Where to write role_distribution.csv and .png (default: <vectors_dir>/analysis)",
    )
    parser.add_argument("--top_k", type=int, default=15, help="How many roles to print at each end")
    args = parser.parse_args()

    vectors_dir = Path(args.vectors_dir)
    layer = args.layer if args.layer is not None else get_config(args.model)["target_layer"]

    out_dir = Path(args.output_dir) if args.output_dir else vectors_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load the axis and the default (assistant) anchor.
    axis = load_axis(str(vectors_dir / "assistant_axis.pt"))
    default_vec = load_vector(vectors_dir / "default_vector.pt")
    print(f"Axis shape: {tuple(axis.shape)} | projecting at layer {layer}")

    # Unit axis at this layer for the cosine metric.
    ax_layer = axis[layer].float()
    ax_unit = ax_layer / (ax_layer.norm() + 1e-8)

    proj_default = project(default_vec, axis, layer, normalize=True)
    print(f"Default (assistant) projection: {proj_default:.4f}")

    role_files = sorted((vectors_dir / "role_vectors").glob("*.pt"))
    if not role_files:
        raise SystemExit(f"No role vectors found in {vectors_dir / 'role_vectors'}")

    rows = []
    for f in role_files:
        role = f.stem
        vec = load_vector(f)
        proj = project(vec, axis, layer, normalize=True)
        offset = (default_vec[layer].float() - vec[layer].float())
        cos = float(torch.dot(offset / (offset.norm() + 1e-8), ax_unit))
        rows.append({
            "role": role,
            "projection": proj,
            "dist_from_assistant": proj_default - proj,
            "cos_to_axis": cos,
        })

    # Sort: most Assistant-like (highest projection) first.
    rows.sort(key=lambda r: r["projection"], reverse=True)

    csv_path = out_dir / "role_distribution.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["role", "projection", "dist_from_assistant", "cos_to_axis"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} roles -> {csv_path}")

    k = min(args.top_k, len(rows))
    print(f"\n=== {k} roles CLOSEST to the Assistant axis (most assistant-like) ===")
    for r in rows[:k]:
        print(f"  {r['projection']:8.3f}  {r['role']}")
    print(f"\n=== {k} roles FARTHEST from the Assistant axis (most drifted) ===")
    for r in rows[-k:]:
        print(f"  {r['projection']:8.3f}  {r['role']}")

    projections = [r["projection"] for r in rows]
    print(
        f"\nDistribution: n={len(projections)} "
        f"min={min(projections):.3f} max={max(projections):.3f} "
        f"mean={sum(projections) / len(projections):.3f}"
    )

    # Histogram (matplotlib is a repo dependency).
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(projections, bins=40, color="#4C72B0", edgecolor="white")
        ax.axvline(proj_default, color="crimson", linestyle="--", label=f"default/assistant ({proj_default:.2f})")
        ax.set_xlabel(f"Projection onto assistant axis (layer {layer})")
        ax.set_ylabel("Number of roles")
        ax.set_title(f"Role distribution along the assistant axis — {args.model}")
        ax.legend()
        fig.tight_layout()
        png_path = out_dir / "role_distribution.png"
        fig.savefig(png_path, dpi=150)
        print(f"Wrote histogram -> {png_path}")
    except Exception as e:  # plotting is best-effort
        print(f"(skipped histogram: {e})")


if __name__ == "__main__":
    main()
