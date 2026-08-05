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

for path in (PROJECT_DIR, THIS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from data_loader import load_sequence_homography, load_sequence_image, list_image_sequences
from feature_matching import (
    create_sift,
    draw_matches,
    extract_features,
    match_features,
    mutual_best_matches,
)


# ============================================================
# Configuration
# ============================================================

DEFAULT_LIGHTING_SEQUENCE = "i_ajuntament"
DEFAULT_VIEWPOINT_SEQUENCE = "v_bark"
DEFAULT_REFERENCE_IMAGE = 1
DEFAULT_TEST_IMAGE = 2
DEFAULT_RATIO_THRESHOLD = 0.75
DEFAULT_TOP_K = 3
DEFAULT_INLIER_THRESHOLD = 3.0
DEFAULT_THRESHOLDS = (1, 2, 3, 5, 8, 10)
DEFAULT_OUTPUT_DIR = THIS_DIR / "homography"


# ============================================================
# Small helpers
# ============================================================

def save_figure(fig, output_path):
    """
    Save a Matplotlib figure and close it immediately.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def project_points(points, homography):
    """
    Project 2D points from the reference image into the test image.

    A homography maps a point x = [u, v, 1]^T in the reference image to
    x' ~ Hx in the test image. We divide by the last coordinate to get
    normal image coordinates again.
    """

    if len(points) == 0:
        return np.empty((0, 2), dtype=np.float32)

    points_h = np.hstack(
        [points.astype(np.float32), np.ones((len(points), 1), dtype=np.float32)]
    )
    projected = (homography @ points_h.T).T

    # Avoid division by zero in case a point lands near infinity.
    w = projected[:, 2:3]
    w[np.abs(w) < 1e-8] = 1e-8

    return (projected[:, :2] / w).astype(np.float32)


def compute_geometric_metrics(
    reference_keypoints,
    test_keypoints,
    matches,
    homography,
    test_shape,
    thresholds,
    inlier_threshold,
):
    """
    Measure how well matches agree with the ground-truth homography.

    We treat a match as correct if its reprojection error is below the
    inlier threshold.
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
            "inlier_mask": np.array([], dtype=bool),
            "visible_reference": 0,
            "correct_matches": 0,
            "outlier_count": 0,
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

    inlier_mask = errors <= inlier_threshold
    correct_matches = int(np.count_nonzero(inlier_mask))
    precision = correct_matches / float(len(matches))

    # Recall proxy: how many reference keypoints are visible in the test image.
    all_reference_points = np.array(
        [kp.pt for kp in reference_keypoints],
        dtype=np.float32,
    )
    all_projected = project_points(all_reference_points, homography)

    h, w = test_shape[:2]
    visible_mask = (
        (all_projected[:, 0] >= 0)
        & (all_projected[:, 0] < w)
        & (all_projected[:, 1] >= 0)
        & (all_projected[:, 1] < h)
    )
    visible_reference = int(np.count_nonzero(visible_mask))
    recall = correct_matches / float(visible_reference) if visible_reference else 0.0

    # MMA curve: fraction of matches with error below each threshold.
    mma = [float(np.mean(errors <= threshold)) for threshold in thresholds]

    return {
        "precision": precision,
        "recall": recall,
        "mean_error": float(np.mean(errors)),
        "mma_score": float(np.mean(mma)),
        "mma_at_3": float(np.mean(errors <= 3.0)),
        "mma": mma,
        "errors": errors,
        "inlier_mask": inlier_mask,
        "visible_reference": visible_reference,
        "correct_matches": correct_matches,
        "outlier_count": len(matches) - correct_matches,
    }


# ============================================================
# Plotting
# ============================================================

def plot_verification_counts(results, output_path):
    """
    Bar chart showing matches, inliers, and outliers for each sequence.
    """

    labels = [result["name"] for result in results]
    match_counts = [result["match_count"] for result in results]
    inlier_counts = [result["correct_matches"] for result in results]
    outlier_counts = [result["outlier_count"] for result in results]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, match_counts, width, label="Matches", color="#4C78A8")
    ax.bar(x, inlier_counts, width, label="Inliers", color="#54A24B")
    ax.bar(x + width, outlier_counts, width, label="Outliers", color="#E45756")

    ax.set_title("Homography Verification Counts")
    ax.set_xlabel("Sequence")
    ax.set_ylabel("Count")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    save_figure(fig, output_path)


def plot_quality_bars(results, output_path):
    """
    Bar chart for the main geometric verification scores.
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

    ax.set_title("Geometric Consistency Comparison")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    save_figure(fig, output_path)


def plot_error_histogram(results, inlier_threshold, output_path):
    """
    Histogram of reprojection errors.
    """

    fig, ax = plt.subplots(figsize=(10, 5))

    non_empty = False
    max_error = 0.0
    for result in results:
        if len(result["errors"]) > 0:
            non_empty = True
            max_error = max(max_error, float(np.max(result["errors"])))

    if non_empty:
        bins = np.linspace(0.0, max(10.0, max_error), 30)
        colors = ["#4C78A8", "#F58518"]
        for result, color in zip(results, colors):
            if len(result["errors"]) > 0:
                ax.hist(
                    result["errors"],
                    bins=bins,
                    alpha=0.55,
                    label=result["name"],
                    color=color,
                )
        ax.axvline(
            inlier_threshold,
            color="#E45756",
            linestyle="--",
            linewidth=2,
            label=f"Inlier threshold = {inlier_threshold:.1f}px",
        )
        ax.set_title("Reprojection Error Distribution")
        ax.set_xlabel("Reprojection error (pixels)")
        ax.set_ylabel("Number of matches")
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.3)
    else:
        ax.text(
            0.5,
            0.5,
            "No matches available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()

    save_figure(fig, output_path)


def plot_mma_curves(results, thresholds, output_path):
    """
    Line plot showing MMA across several reprojection thresholds.
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


# ============================================================
# Pair evaluation
# ============================================================

def evaluate_pair(
    sequence,
    reference_image,
    test_image,
    ratio_threshold,
    top_k,
    inlier_threshold,
    output_dir,
    thresholds,
    sift,
):
    """
    Run feature matching, project matches with the homography, and save one
    small visualization with the geometrically valid matches.
    """

    reference_rgb = load_sequence_image(sequence, reference_image)
    test_rgb = load_sequence_image(sequence, test_image)
    homography = load_sequence_homography(sequence, test_image)

    reference_keypoints, reference_descriptors = extract_features(reference_rgb, sift)
    test_keypoints, test_descriptors = extract_features(test_rgb, sift)

    good_matches = match_features(
        reference_descriptors,
        test_descriptors,
        ratio_threshold,
    )
    final_matches = mutual_best_matches(
        reference_descriptors,
        test_descriptors,
        good_matches,
    )

    metrics = compute_geometric_metrics(
        reference_keypoints,
        test_keypoints,
        final_matches,
        homography,
        test_rgb.shape,
        thresholds,
        inlier_threshold,
    )

    inlier_matches = [
        match
        for match, is_inlier in zip(final_matches, metrics["inlier_mask"])
        if is_inlier
    ]

    # Show the strongest inlier correspondences.
    example_matches = inlier_matches if inlier_matches else final_matches
    output_image = draw_matches(
        reference_rgb,
        reference_keypoints,
        test_rgb,
        test_keypoints,
        example_matches,
        top_k=top_k,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{sequence}_ref{reference_image}_test{test_image}_verified_top{top_k}.png"
    cv2.imwrite(str(output_path), cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))

    return {
        "name": sequence,
        "reference_keypoints": len(reference_keypoints),
        "test_keypoints": len(test_keypoints),
        "raw_match_count": len(good_matches),
        "match_count": len(final_matches),
        "correct_matches": metrics["correct_matches"],
        "outlier_count": metrics["outlier_count"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "mean_error": metrics["mean_error"],
        "mma_score": metrics["mma_score"],
        "mma_at_3": metrics["mma_at_3"],
        "mma": metrics["mma"],
        "errors": metrics["errors"],
        "output_path": output_path,
    }


def run_homography_verification(
    lighting_sequence=DEFAULT_LIGHTING_SEQUENCE,
    viewpoint_sequence=DEFAULT_VIEWPOINT_SEQUENCE,
    reference_image=DEFAULT_REFERENCE_IMAGE,
    test_image=DEFAULT_TEST_IMAGE,
    ratio_threshold=DEFAULT_RATIO_THRESHOLD,
    top_k=DEFAULT_TOP_K,
    inlier_threshold=DEFAULT_INLIER_THRESHOLD,
    thresholds=DEFAULT_THRESHOLDS,
    output_dir=DEFAULT_OUTPUT_DIR,
):
    """
    Evaluate homography-based verification on one lighting pair and one
    viewpoint pair.
    """

    sequences = list_image_sequences()
    if not sequences:
        raise RuntimeError("No HPatches sequences were found.")

    if lighting_sequence not in sequences:
        raise ValueError(f"Lighting sequence '{lighting_sequence}' was not found.")

    if viewpoint_sequence not in sequences:
        raise ValueError(f"Viewpoint sequence '{viewpoint_sequence}' was not found.")

    sift = create_sift()
    examples_dir = output_dir / "examples"

    lighting_result = evaluate_pair(
        lighting_sequence,
        reference_image,
        test_image,
        ratio_threshold,
        top_k,
        inlier_threshold,
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
        inlier_threshold,
        examples_dir / "viewpoint",
        thresholds,
        sift,
    )

    results = [lighting_result, viewpoint_result]

    plot_verification_counts(results, output_dir / "verification_counts.png")
    plot_quality_bars(results, output_dir / "quality_metrics.png")
    plot_error_histogram(results, inlier_threshold, output_dir / "reprojection_errors.png")
    plot_mma_curves(results, thresholds, output_dir / "mma_curve.png")

    print("=" * 70)
    print("Homography Verification")
    print("=" * 70)
    print(
        "HPatches provides a ground-truth homography so we can map reference "
        "keypoints into the test image and check whether each match is "
        "geometrically consistent."
    )
    print(
        "A match is treated as an inlier when its reprojection error is below "
        f"{inlier_threshold:.1f} pixels."
    )

    for result in results:
        print()
        print(f"{result['name']}:")
        print(f"  Reference keypoints: {result['reference_keypoints']}")
        print(f"  Test keypoints: {result['test_keypoints']}")
        print(f"  Raw ratio-test matches: {result['raw_match_count']}")
        print(f"  Mutual-best matches: {result['match_count']}")
        print(f"  Inliers: {result['correct_matches']}")
        print(f"  Outliers: {result['outlier_count']}")
        print(f"  Precision / geometric consistency: {result['precision']:.3f}")
        print(f"  Recall proxy: {result['recall']:.3f}")
        print(f"  MMA score: {result['mma_score']:.3f}")
        print(f"  MMA@3px: {result['mma_at_3']:.3f}")
        print(f"  Mean reprojection error: {result['mean_error']:.3f}")
        print(f"  Example image: {result['output_path']}")

    print()
    print(f"Saved plots to: {output_dir}")
    print("=" * 70)

    return results


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run a simple homography-based geometric verification check on "
            "HPatches SIFT matches."
        )
    )

    parser.add_argument(
        "--lighting-sequence",
        default=DEFAULT_LIGHTING_SEQUENCE,
        help="HPatches illumination sequence.",
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
        help="How many inlier correspondences to show in the example image.",
    )
    parser.add_argument(
        "--inlier-threshold",
        type=float,
        default=DEFAULT_INLIER_THRESHOLD,
        help="Reprojection error threshold used to mark inliers.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where plots and example images will be saved.",
    )

    args = parser.parse_args()

    run_homography_verification(
        lighting_sequence=args.lighting_sequence,
        viewpoint_sequence=args.viewpoint_sequence,
        reference_image=args.reference_image,
        test_image=args.test_image,
        ratio_threshold=args.ratio,
        top_k=args.top_k,
        inlier_threshold=args.inlier_threshold,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
