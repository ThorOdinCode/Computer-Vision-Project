from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

import matplotlib.pyplot as plt


THIS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = THIS_DIR.parent
ORB_DIR = PROJECT_DIR / "ORB"

for path in (PROJECT_DIR, ORB_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from data_loader import load_sequence_homography, load_sequence_image
from feature_matching import (
    create_orb,
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
DEFAULT_INLIER_THRESHOLD = 3.0
DEFAULT_OUTPUT_DIR = THIS_DIR / "figures" / "matching"
DEFAULT_THRESHOLDS = (1, 2, 3, 5, 8, 10)


def save_figure(fig, output_path):
    """
    Save a Matplotlib figure and close it right away.

    Args:
        fig: the Matplotlib Figure to save.
        output_path (Path): where to write the PNG. Parent folders are
            created automatically if they don't exist yet.

    Returns:
        None. The figure is written to disk and closed (freeing memory),
        so don't try to keep using `fig` after calling this.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def project_points(points, homography):
    """
    Map 2D pixel coordinates through a homography.

    This is the "warp the point, not the image" step: given a keypoint's
    (x, y) location in the reference image, this returns where that same
    physical point should land in the test image, according to the
    ground-truth geometric transform between the two.

    Args:
        points (np.ndarray): shape (N, 2) array of (x, y) pixel coordinates.
        homography (np.ndarray): 3x3 homography matrix mapping reference
            image coordinates to test image coordinates.

    Returns:
        np.ndarray: shape (N, 2) array of projected (x, y) coordinates.
            Returns an empty (0, 2) array if `points` is empty.
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


def compute_match_metrics(reference_keypoints, test_keypoints, matches, homography
                          , test_shape, thresholds, inlier_threshold=DEFAULT_INLIER_THRESHOLD):
    """
    Score a set of matches against the ground-truth homography.

    For every match, the reference keypoint is projected through the
    homography and compared to where it was actually matched in the test
    image; the pixel distance between the two is the reprojection error.
    That single error value drives everything below:

    Precision:
        correct matches / all matches produced by the matcher.

    Recall (proxy):
        correct matches / reference keypoints that were geometrically
        visible in the test image (i.e. their projected location falls
        inside the test image bounds), whether or not they got matched.
        This approximates "how many of the correspondences that should
        exist did we actually recover," since HPatches gives no direct
        point-to-point ground truth to check against.

    MMA (Mean Matching Accuracy):
        fraction of matches with reprojection error under each threshold
        in `thresholds`. This is the same underlying quantity as
        precision, just swept across several pixel tolerances instead of
        a single fixed one -- it's what produces the MMA curve.

    Args:
        reference_keypoints (list[cv2.KeyPoint]): keypoints detected in
            the reference image.
        test_keypoints (list[cv2.KeyPoint]): keypoints detected in the
            test image.
        matches (list[cv2.DMatch]): matches to evaluate (queryIdx indexes
            reference_keypoints, trainIdx indexes test_keypoints).
        homography (np.ndarray): 3x3 ground-truth homography mapping the
            reference image to the test image.
        test_shape (tuple): shape of the test image array (used only for
            its height/width, to check which reference points are
            visible in-frame).
        thresholds (Sequence[float]): pixel-error thresholds to sweep for
            the MMA curve.

    Returns:
        dict with keys: precision, recall, mean_error, mma_score,
        mma_at_3, mma (list, one value per threshold), errors (per-match
        reprojection errors), visible_reference (count), correct_matches
        (count). All-zero/empty defaults are returned if `matches` is empty.
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
    mma_at_3 = float(np.mean(inlier_mask))

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
        "inlier_mask": inlier_mask,
        "visible_reference": visible_reference,
        "correct_matches": correct_matches,
        "outlier_count": len(matches) - correct_matches,
    }


def plot_match_counts(results, output_path):
    """
    Bar chart comparing reference keypoints, test keypoints, and filtered
    matches for each evaluated pair (e.g. lighting vs. viewpoint).

    Args:
        results (list[dict]): result dicts from `evaluate_pair`, each
            containing "name", "reference_keypoints", "test_keypoints",
            and "match_count".
        output_path (Path): where to save the PNG.

    Returns:
        None. Saves the figure to `output_path`.
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
    Bar chart comparing precision, recall, and MMA between exactly two
    evaluated pairs (written for the lighting-vs-viewpoint comparison).

    Args:
        results (list[dict]): exactly two result dicts from
            `evaluate_pair`, each containing "name", "precision",
            "recall", and "mma_score".
        output_path (Path): where to save the PNG.

    Returns:
        None. Saves the figure to `output_path`.
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
    Line plot of Mean Matching Accuracy across a range of pixel-error
    thresholds, one line per evaluated pair.

    Args:
        results (list[dict]): result dicts from `evaluate_pair`, each
            containing "name" and "mma" (a list of MMA values, one per
            threshold, in the same order as `thresholds`).
        thresholds (Sequence[float]): the pixel thresholds used as x-axis
            values (must match what was passed into `compute_match_metrics`).
        output_path (Path): where to save the PNG.

    Returns:
        None. Saves the figure to `output_path`.
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


def plot_error_histogram(results, inlier_threshold, output_path):
    """
    Histogram of per-match reprojection errors, with the inlier threshold
    drawn as a vertical line.

    This is a sanity check on the threshold itself: if correct and wrong
    matches form two visually separated clusters (one near 0px, one far
    out), that's evidence the inlier/outlier split is a real, well-
    seperated phenomenon in the data rather than an arbitrary cutoff.

    Args:
        results (list[dict]): result dicts from `evaluate_pair`, each
            containing "name" and "errors" (per-match reprojection
            errors, as returned by `compute_match_metrics`).
        inlier_threshold (float): the pixel threshold used elsewhere to
            call a match an inlier; drawn as a reference line.
        output_path (Path): where to save the PNG.

    Returns:
        None. Saves the figure to `output_path`. If no result has any
        matches, saves a placeholder figure with a "no matches" message
        instead of an empty plot.
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


def evaluate_pair(
    sequence,
    reference_image,
    test_image,
    ratio_threshold,
    top_k,
    output_dir,
    thresholds,
    orb,
    inlier_threshold=DEFAULT_INLIER_THRESHOLD,
):
    """
    Run the full pipeline for one reference/test image pair: detect ORB
    features, match them, save a visualization of the top matches, and
    score the result against the sequence's ground-truth homography.

    Args:
        sequence (str): HPatches sequence name (e.g. "i_ajuntament").
        reference_image (int): image number to use as the reference (1
            in the standard HPatches convention).
        test_image (int): image number to use as the test/distorted image
            (2 through 6).
        ratio_threshold (float): Lowe's ratio test threshold passed to
            `match_features`.
        top_k (int): how many of the strongest matches to draw in the
            saved visualization.
        output_dir (Path): folder to save the example match visualization
            into.
        thresholds (Sequence[float]): pixel thresholds for the MMA curve,
            passed straight through to `compute_match_metrics`.
        orb: an ORB detector instance, as returned by `create_orb()`.
            Built once by the caller and reused across pairs so it isn't
            recreated on every call.

    Returns:
        dict: summary for this pair -- "name" (the sequence name),
        keypoint/match counts, precision/recall/MMA metrics, and the path
        the visualization was saved to. Suitable for passing straight
        into the `plot_*` functions above.
    """

    reference_rgb = load_sequence_image(sequence, reference_image)
    test_rgb = load_sequence_image(sequence, test_image)
    homography = load_sequence_homography(sequence, test_image)

    reference_keypoints, reference_descriptors = extract_features(reference_rgb, orb)
    test_keypoints, test_descriptors = extract_features(test_rgb, orb)

    good_matches = match_features(reference_descriptors, test_descriptors, ratio_threshold)
    final_matches = mutual_best_matches(
        reference_descriptors,
        test_descriptors,
        good_matches,
    )

    metrics = compute_match_metrics(
        reference_keypoints,
        test_keypoints,
        final_matches,
        homography,
        test_rgb.shape,
        thresholds,
        inlier_threshold,
    )

    # Visualize the inlier matches rather than just the strongest
    # by descriptor distance matches. These are the ones we'd
    # actually trust, which isn't always the same set.
    inlier_matches = [
        match
        for match, is_inlier in zip(final_matches, metrics["inlier_mask"])
        if is_inlier
    ]
    example_matches = inlier_matches if inlier_matches else final_matches

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
        "outlier_count": metrics["outlier_count"],
        "errors": metrics["errors"],
        "output_path": output_path,
    }


def run_matching_evaluation(
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
    Run the ORB matching evaluation for one lighting pair and one
    viewpoint pair, then save comparison plots across both.

    This builds a single ORB detector and reuses it for both pairs,
    evaluates each with `evaluate_pair`, and produces three figures:
    a keypoint/match count bar chart, a precision/recall/MMA bar chart,
    and an MMA-vs-threshold curve.

    Args:
        lighting_sequence (str): HPatches illumination ("i_*") sequence
            name to evaluate.
        viewpoint_sequence (str): HPatches viewpoint ("v_*") sequence
            name to evaluate.
        reference_image (int): reference image number (usually 1).
        test_image (int): test image number (2 through 6).
        ratio_threshold (float): Lowe's ratio test threshold.
        top_k (int): matches to draw in each example visualization.
        thresholds (Sequence[float]): pixel thresholds for the MMA curve.
        output_dir (Path): base folder for all saved figures and example
            images.

    Returns:
        list[dict]: the two result dicts, [lighting_result,
        viewpoint_result], as produced by `evaluate_pair`.
    """

    orb = create_orb()

    examples_dir = output_dir / "examples"

    lighting_result = evaluate_pair(
        lighting_sequence,
        reference_image,
        test_image,
        ratio_threshold,
        top_k,
        examples_dir / "lighting",
        thresholds,
        orb,
        inlier_threshold,
    )

    viewpoint_result = evaluate_pair(
        viewpoint_sequence,
        reference_image,
        test_image,
        ratio_threshold,
        top_k,
        examples_dir / "viewpoint",
        thresholds,
        orb,
        inlier_threshold,
    )

    results = [lighting_result, viewpoint_result]

    plot_match_counts(results, output_dir / "match_counts.png")
    plot_quality_bars(results, output_dir / "quality_metrics.png")
    plot_mma_curves(results, thresholds, output_dir / "mma_curve.png")
    plot_error_histogram(results, inlier_threshold, output_dir / "reprojection_errors.png")

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
    """
    CLI entry point: parse arguments and run the ORB matching evaluation.

    Returns:
        None. Delegates to `run_matching_evaluation`, which prints a
        summary and saves figures as a side effect.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Create simple ORB matching evaluation plots for one lighting "
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
        "--inlier-threshold",
        type=float,
        default=DEFAULT_INLIER_THRESHOLD,
        help="Reprojection error threshold (pixels) used to mark inliers.",
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
        inlier_threshold=args.inlier_threshold,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()