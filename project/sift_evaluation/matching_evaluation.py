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

from data_loader import load_sequence_homography, load_sequence_image
from feature_matching import (
    create_sift,
    draw_matches,
    extract_features,
    match_features,
    mutual_best_matches,
)


DEFAULT_LIGHTING_SEQUENCE = "i_ajuntament"
DEFAULT_VIEWPOINT_SEQUENCE = "v_bark"
DEFAULT_REFERENCE_IMAGE = 1
DEFAULT_TEST_IMAGE = 2
DEFAULT_RATIO_THRESHOLD = 0.75
DEFAULT_TOP_K = 3
DEFAULT_OUTPUT_DIR = THIS_DIR / "figures" / "matching"
DEFAULT_THRESHOLDS = (1, 2, 3, 5, 8, 10)


def save_figure(fig, output_path):
    """
    Save a Matplotlib figure and close it right away.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def project_points(points, homography):
    """
    Project 2D points with a homography.
    """

    if len(points) == 0:
        return np.empty((0, 2), dtype=np.float32)

    points_h = np.hstack(
        [points.astype(np.float32), np.ones((len(points), 1), dtype=np.float32)]
    )
    projected = (homography @ points_h.T).T
    projected[:, 0] /= projected[:, 2]
    projected[:, 1] /= projected[:, 2]
    return projected[:, :2]


def compute_match_metrics(reference_keypoints, test_keypoints, matches, homography, test_shape, thresholds):
    """
    Compute precision, recall proxy, and MMA from reprojection errors.

    Precision:
        correct matches / all matches

    Recall proxy:
        correct matches / visible reference keypoints

    MMA:
        fraction of matches with reprojection error under each threshold
    """

    if not matches:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "mean_error": 0.0,
            "mma_score": 0.0,
            "mma_at_3": 0.0,
            "mma": [0.0 for _ in thresholds],
            "errors": np.array([], dtype=np.float32),
            "visible_reference": 0,
            "correct_matches": 0,
        }

    reference_points = np.array(
        [reference_keypoints[match.queryIdx].pt for match in matches],
        dtype=np.float32,
    )
    test_points = np.array(
        [test_keypoints[match.trainIdx].pt for match in matches],
        dtype=np.float32,
    )

    projected = project_points(reference_points, homography)
    errors = np.linalg.norm(projected - test_points, axis=1)

    correct_mask = errors <= 3.0
    correct_matches = int(np.count_nonzero(correct_mask))
    precision = correct_matches / float(len(matches))
    mma_at_3 = float(np.mean(correct_mask))

    h, w = test_shape[:2]
    all_reference_points = np.array([kp.pt for kp in reference_keypoints], dtype=np.float32)
    all_projected = project_points(all_reference_points, homography)
    visible_mask = (
        (all_projected[:, 0] >= 0)
        & (all_projected[:, 0] < w)
        & (all_projected[:, 1] >= 0)
        & (all_projected[:, 1] < h)
    )
    visible_reference = int(np.count_nonzero(visible_mask))
    recall = correct_matches / float(visible_reference) if visible_reference else 0.0

    mma = [float(np.mean(errors <= threshold)) for threshold in thresholds]

    return {
        "precision": precision,
        "recall": recall,
        "mean_error": float(np.mean(errors)),
        "mma_score": float(np.mean(mma)),
        "mma_at_3": mma_at_3,
        "mma": mma,
        "errors": errors,
        "visible_reference": visible_reference,
        "correct_matches": correct_matches,
    }


def plot_match_counts(results, output_path):
    """
    Bar chart: reference keypoints, test keypoints, and filtered matches.
    """

    labels = [result["name"] for result in results]
    reference_counts = [result["reference_keypoints"] for result in results]
    test_counts = [result["test_keypoints"] for result in results]
    match_counts = [result["match_count"] for result in results]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, reference_counts, width, label="Reference keypoints", color="#4C78A8")
    ax.bar(x, test_counts, width, label="Test keypoints", color="#F58518")
    ax.bar(x + width, match_counts, width, label="Filtered matches", color="#54A24B")

    ax.set_title("Matching Counts After Filtering")
    ax.set_xlabel("Sequence")
    ax.set_ylabel("Count")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    save_figure(fig, output_path)


def plot_quality_bars(results, output_path):
    """
    Bar chart: precision, recall, and MMA.
    """

    metrics = ["Precision", "Recall", "MMA"]
    x = np.arange(len(metrics))
    width = 0.32

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        x - width / 2,
        [results[0]["precision"], results[0]["recall"], results[0]["mma_score"]],
        width,
        label=results[0]["name"],
        color="#4C78A8",
    )
    ax.bar(
        x + width / 2,
        [results[1]["precision"], results[1]["recall"], results[1]["mma_score"]],
        width,
        label=results[1]["name"],
        color="#F58518",
    )

    ax.set_title("Matching Quality Comparison")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    save_figure(fig, output_path)


def plot_mma_curves(results, thresholds, output_path):
    """
    Line plot: MMA across reprojection thresholds.
    """

    fig, ax = plt.subplots(figsize=(10, 5))

    for result, color in zip(results, ["#4C78A8", "#F58518"]):
        ax.plot(
            thresholds,
            result["mma"],
            marker="o",
            linewidth=2,
            label=result["name"],
            color=color,
        )

    ax.set_title("Mean Matching Accuracy Across Thresholds")
    ax.set_xlabel("Reprojection threshold (pixels)")
    ax.set_ylabel("MMA")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)

    save_figure(fig, output_path)


def evaluate_pair(
    sequence,
    reference_image,
    test_image,
    ratio_threshold,
    top_k,
    output_dir,
    thresholds,
    sift,
):
    """
    Extract features, match them, save a top-k visualisation, and compute metrics.
    """

    reference_rgb = load_sequence_image(sequence, reference_image)
    test_rgb = load_sequence_image(sequence, test_image)
    homography = load_sequence_homography(sequence, test_image)

    reference_keypoints, reference_descriptors = extract_features(reference_rgb, sift)
    test_keypoints, test_descriptors = extract_features(test_rgb, sift)

    good_matches = match_features(reference_descriptors, test_descriptors, ratio_threshold)
    final_matches = mutual_best_matches(
        reference_descriptors,
        test_descriptors,
        good_matches,
    )

    output_image = draw_matches(
        reference_rgb,
        reference_keypoints,
        test_rgb,
        test_keypoints,
        final_matches,
        top_k=top_k,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{sequence}_ref{reference_image}_test{test_image}_top{top_k}.png"
    cv2.imwrite(str(output_path), cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))

    metrics = compute_match_metrics(
        reference_keypoints,
        test_keypoints,
        final_matches,
        homography,
        test_rgb.shape,
        thresholds,
    )

    return {
        "name": sequence,
        "reference_keypoints": len(reference_keypoints),
        "test_keypoints": len(test_keypoints),
        "match_count": len(final_matches),
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "mma_score": metrics["mma_score"],
        "mma_at_3": metrics["mma_at_3"],
        "mma": metrics["mma"],
        "mean_error": metrics["mean_error"],
        "visible_reference": metrics["visible_reference"],
        "correct_matches": metrics["correct_matches"],
        "output_path": output_path,
    }


def run_matching_evaluation(
    lighting_sequence=DEFAULT_LIGHTING_SEQUENCE,
    viewpoint_sequence=DEFAULT_VIEWPOINT_SEQUENCE,
    reference_image=DEFAULT_REFERENCE_IMAGE,
    test_image=DEFAULT_TEST_IMAGE,
    ratio_threshold=DEFAULT_RATIO_THRESHOLD,
    top_k=DEFAULT_TOP_K,
    thresholds=DEFAULT_THRESHOLDS,
    output_dir=DEFAULT_OUTPUT_DIR,
):
    """
    Run matching evaluation for one lighting pair and one viewpoint pair.
    """

    sift = create_sift()

    examples_dir = output_dir / "examples"

    lighting_result = evaluate_pair(
        lighting_sequence,
        reference_image,
        test_image,
        ratio_threshold,
        top_k,
        examples_dir / "lighting",
        thresholds,
        sift,
    )

    viewpoint_result = evaluate_pair(
        viewpoint_sequence,
        reference_image,
        test_image,
        ratio_threshold,
        top_k,
        examples_dir / "viewpoint",
        thresholds,
        sift,
    )

    results = [lighting_result, viewpoint_result]

    plot_match_counts(results, output_dir / "match_counts.png")
    plot_quality_bars(results, output_dir / "quality_metrics.png")
    plot_mma_curves(results, thresholds, output_dir / "mma_curve.png")

    print("Matching evaluation complete.")
    for result in results:
        print(
            f"  {result['name']}: {result['reference_keypoints']} reference keypoints, "
            f"{result['test_keypoints']} test keypoints, {result['match_count']} matches"
        )
        print(f"    Precision: {result['precision']:.3f}")
        print(f"    Recall: {result['recall']:.3f}")
        print(f"    MMA: {result['mma_score']:.3f}")
        print(f"    Example image: {result['output_path']}")

    print(f"  Count plot: {output_dir / 'match_counts.png'}")
    print(f"  Quality plot: {output_dir / 'quality_metrics.png'}")
    print(f"  MMA curve: {output_dir / 'mma_curve.png'}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create simple SIFT matching evaluation plots for one lighting "
            "pair and one viewpoint pair."
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
        "--reference-image",
        type=int,
        default=DEFAULT_REFERENCE_IMAGE,
        help="Reference image number.",
    )
    parser.add_argument(
        "--test-image",
        type=int,
        default=DEFAULT_TEST_IMAGE,
        help="Test image number.",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=DEFAULT_RATIO_THRESHOLD,
        help="Lowe ratio test threshold.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="How many correspondences to draw in each example.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where plots will be saved.",
    )

    args = parser.parse_args()

    run_matching_evaluation(
        lighting_sequence=args.lighting_sequence,
        viewpoint_sequence=args.viewpoint_sequence,
        reference_image=args.reference_image,
        test_image=args.test_image,
        ratio_threshold=args.ratio,
        top_k=args.top_k,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
