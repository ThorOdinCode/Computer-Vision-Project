from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np


# Matplotlib needs a writable cache directory in this environment.
MPLCONFIGDIR = Path("/private/tmp/matplotlib")
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MPLCONFIGDIR)

XDG_CACHE_HOME = Path("/private/tmp/codex-cache")
XDG_CACHE_HOME.mkdir(parents=True, exist_ok=True)
os.environ["XDG_CACHE_HOME"] = str(XDG_CACHE_HOME)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


THIS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = THIS_DIR.parent
SIFT_DIR = PROJECT_DIR / "SIFT"

for path in (PROJECT_DIR, SIFT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from feature_extraction import create_sift, process_sample_sequence


DEFAULT_LIGHTING_SEQUENCE = "i_ajuntament"
DEFAULT_VIEWPOINT_SEQUENCE = "v_bark"
DEFAULT_MAX_POINTS = 150
DEFAULT_OUTPUT_DIR = THIS_DIR / "figures" / "extraction"


def mean(values):
    """
    Small helper so the code stays easy to read.
    """

    return float(np.mean(values)) if values else 0.0


def save_figure(fig, output_path):
    """
    Save a Matplotlib figure and close it right away.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_keypoint_counts(lighting_results, viewpoint_results, output_path):
    """
    Bar chart: number of detected keypoints per image.
    """

    image_ids = [item["image_number"] for item in lighting_results]
    lighting_counts = [item["statistics"]["number_of_keypoints"] for item in lighting_results]
    viewpoint_counts = [item["statistics"]["number_of_keypoints"] for item in viewpoint_results]

    x = np.arange(len(image_ids))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, lighting_counts, width, label="Lighting", color="#4C78A8")
    ax.bar(x + width / 2, viewpoint_counts, width, label="Viewpoint", color="#F58518")

    ax.set_title("SIFT Keypoints per Image")
    ax.set_xlabel("Image number")
    ax.set_ylabel("Detected keypoints")
    ax.set_xticks(x)
    ax.set_xticklabels(image_ids)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    save_figure(fig, output_path)


def plot_runtime_counts(lighting_results, viewpoint_results, output_path):
    """
    Bar chart: extraction runtime per image.
    """

    image_ids = [item["image_number"] for item in lighting_results]
    lighting_runtime = [item["statistics"]["runtime_seconds"] for item in lighting_results]
    viewpoint_runtime = [item["statistics"]["runtime_seconds"] for item in viewpoint_results]

    x = np.arange(len(image_ids))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, lighting_runtime, width, label="Lighting", color="#72B7B2")
    ax.bar(x + width / 2, viewpoint_runtime, width, label="Viewpoint", color="#E45756")

    ax.set_title("SIFT Extraction Runtime per Image")
    ax.set_xlabel("Image number")
    ax.set_ylabel("Runtime (seconds)")
    ax.set_xticks(x)
    ax.set_xticklabels(image_ids)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    save_figure(fig, output_path)


def plot_summary(lighting_results, viewpoint_results, output_path):
    """
    Compact summary of average keypoints and average runtime.
    """

    labels = ["Lighting", "Viewpoint"]
    avg_keypoints = [
        mean([item["statistics"]["number_of_keypoints"] for item in lighting_results]),
        mean([item["statistics"]["number_of_keypoints"] for item in viewpoint_results]),
    ]
    avg_runtime = [
        mean([item["statistics"]["runtime_seconds"] for item in lighting_results]),
        mean([item["statistics"]["runtime_seconds"] for item in viewpoint_results]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].bar(labels, avg_keypoints, color=["#4C78A8", "#F58518"])
    axes[0].set_title("Average Keypoints")
    axes[0].set_ylabel("Keypoints per image")
    axes[0].grid(axis="y", linestyle="--", alpha=0.3)

    axes[1].bar(labels, avg_runtime, color=["#72B7B2", "#E45756"])
    axes[1].set_title("Average Runtime")
    axes[1].set_ylabel("Seconds per image")
    axes[1].grid(axis="y", linestyle="--", alpha=0.3)

    save_figure(fig, output_path)


def run_extraction_evaluation(
    lighting_sequence=DEFAULT_LIGHTING_SEQUENCE,
    viewpoint_sequence=DEFAULT_VIEWPOINT_SEQUENCE,
    max_points=DEFAULT_MAX_POINTS,
    output_dir=DEFAULT_OUTPUT_DIR,
):
    """
    Run SIFT extraction on two sample sequences and create plots.
    """

    sift = create_sift()

    sample_output_dir = output_dir / "samples"
    lighting_output = sample_output_dir / "lighting"
    viewpoint_output = sample_output_dir / "viewpoint"

    lighting_results = process_sample_sequence(
        lighting_sequence,
        "Lighting Changes",
        sift,
        lighting_output,
        max_points,
    )

    viewpoint_results = process_sample_sequence(
        viewpoint_sequence,
        "Viewpoint Changes",
        sift,
        viewpoint_output,
        max_points,
    )

    plot_keypoint_counts(
        lighting_results,
        viewpoint_results,
        output_dir / "keypoint_counts.png",
    )
    plot_runtime_counts(
        lighting_results,
        viewpoint_results,
        output_dir / "runtime_counts.png",
    )
    plot_summary(
        lighting_results,
        viewpoint_results,
        output_dir / "summary.png",
    )

    print("Extraction evaluation complete.")
    print(f"  Lighting sample grid: {lighting_output / f'{lighting_sequence}_sift_grid.png'}")
    print(f"  Viewpoint sample grid: {viewpoint_output / f'{viewpoint_sequence}_sift_grid.png'}")
    print(f"  Keypoint plot: {output_dir / 'keypoint_counts.png'}")
    print(f"  Runtime plot: {output_dir / 'runtime_counts.png'}")
    print(f"  Summary plot: {output_dir / 'summary.png'}")

    return {
        "lighting_results": lighting_results,
        "viewpoint_results": viewpoint_results,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create simple SIFT extraction evaluation plots for lighting and "
            "viewpoint sequences."
        )
    )
    parser.add_argument(
        "--lighting-sequence",
        default=DEFAULT_LIGHTING_SEQUENCE,
        help="HPatches lighting sequence.",
    )
    parser.add_argument(
        "--viewpoint-sequence",
        default=DEFAULT_VIEWPOINT_SEQUENCE,
        help="HPatches viewpoint sequence.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=DEFAULT_MAX_POINTS,
        help="Maximum number of keypoints drawn in the sample figures.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where plots will be saved.",
    )

    args = parser.parse_args()

    run_extraction_evaluation(
        lighting_sequence=args.lighting_sequence,
        viewpoint_sequence=args.viewpoint_sequence,
        max_points=args.max_points,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
