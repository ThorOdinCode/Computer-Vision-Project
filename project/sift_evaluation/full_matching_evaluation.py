from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

# Matplotlib needs a writable cache directory.
_CACHE_ROOT = Path(__file__).resolve().parent / ".cache"
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / "xdg"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


THIS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = THIS_DIR.parent
SIFT_DIR = PROJECT_DIR / "SIFT"

for path in (PROJECT_DIR, SIFT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from data_loader import (
    list_image_sequences,
    load_sequence_homography,
    load_sequence_image,
)
from feature_matching import (
    create_sift,
    draw_matches,
    extract_features,
    match_features,
    mutual_best_matches,
)
from matching_evaluation import compute_match_metrics


DEFAULT_REFERENCE_IMAGE = 1
DEFAULT_TEST_IMAGES = (2, 3, 4, 5, 6)
DEFAULT_RATIO_THRESHOLD = 0.75
DEFAULT_THRESHOLDS = (1, 2, 3, 5, 8, 10)
DEFAULT_TOP_K = 3
DEFAULT_EXAMPLE_LIGHTING = "i_ajuntament"
DEFAULT_EXAMPLE_VIEWPOINT = "v_bark"
DEFAULT_RESULTS_DIR = THIS_DIR / "results" / "matching_full"
DEFAULT_FIGURES_DIR = THIS_DIR / "figures" / "matching_full"


def condition_of(sequence: str) -> str:
    if sequence.startswith("i_"):
        return "illumination"
    if sequence.startswith("v_"):
        return "viewpoint"
    return "other"


def save_figure(fig, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def evaluate_pair_metrics(
    sequence: str,
    reference_image: int,
    test_image: int,
    ratio_threshold: float,
    thresholds: tuple[int, ...],
    sift,
    reference_cache: dict,
):
    """
    Extract (with cached reference features), match, and score one pair.
    """

    if reference_image not in reference_cache:
        reference_rgb = load_sequence_image(sequence, reference_image)
        t0 = time.perf_counter()
        reference_keypoints, reference_descriptors = extract_features(reference_rgb, sift)
        reference_extract_s = time.perf_counter() - t0
        reference_cache[reference_image] = {
            "rgb": reference_rgb,
            "keypoints": reference_keypoints,
            "descriptors": reference_descriptors,
            "extract_s": reference_extract_s,
        }

    cached = reference_cache[reference_image]
    reference_rgb = cached["rgb"]
    reference_keypoints = cached["keypoints"]
    reference_descriptors = cached["descriptors"]

    test_rgb = load_sequence_image(sequence, test_image)
    t0 = time.perf_counter()
    test_keypoints, test_descriptors = extract_features(test_rgb, sift)
    test_extract_s = time.perf_counter() - t0

    t0 = time.perf_counter()
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
    match_s = time.perf_counter() - t0

    homography = load_sequence_homography(sequence, test_image)
    metrics = compute_match_metrics(
        reference_keypoints,
        test_keypoints,
        final_matches,
        homography,
        test_rgb.shape,
        thresholds,
    )

    row = {
        "sequence": sequence,
        "condition": condition_of(sequence),
        "reference_image": reference_image,
        "test_image": test_image,
        "pair": f"{reference_image}->{test_image}",
        "reference_keypoints": len(reference_keypoints),
        "test_keypoints": len(test_keypoints),
        "raw_ratio_matches": len(good_matches),
        "match_count": len(final_matches),
        "correct_matches": metrics["correct_matches"],
        "visible_reference": metrics["visible_reference"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "mean_error": metrics["mean_error"],
        "mma_score": metrics["mma_score"],
        "mma_at_3": metrics["mma_at_3"],
        "reference_extract_s": cached["extract_s"],
        "test_extract_s": test_extract_s,
        "match_s": match_s,
        "pair_total_s": cached["extract_s"] + test_extract_s + match_s,
    }

    for threshold, value in zip(thresholds, metrics["mma"]):
        row[f"mma_at_{threshold}"] = value

    return row, {
        "reference_rgb": reference_rgb,
        "test_rgb": test_rgb,
        "reference_keypoints": reference_keypoints,
        "test_keypoints": test_keypoints,
        "matches": final_matches,
    }


def save_example_match(
    sequence: str,
    reference_image: int,
    test_image: int,
    pair_payload: dict,
    top_k: int,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_image = draw_matches(
        pair_payload["reference_rgb"],
        pair_payload["reference_keypoints"],
        pair_payload["test_rgb"],
        pair_payload["test_keypoints"],
        pair_payload["matches"],
        top_k=top_k,
    )
    output_path = (
        output_dir
        / f"{sequence}_ref{reference_image}_test{test_image}_top{top_k}.png"
    )
    cv2.imwrite(str(output_path), cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
    return output_path


def write_csv(rows: list[dict], output_path: Path, thresholds: tuple[int, ...]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sequence",
        "condition",
        "reference_image",
        "test_image",
        "pair",
        "reference_keypoints",
        "test_keypoints",
        "raw_ratio_matches",
        "match_count",
        "correct_matches",
        "visible_reference",
        "precision",
        "recall",
        "mean_error",
        "mma_score",
        "mma_at_3",
        *[f"mma_at_{t}" for t in thresholds],
        "reference_extract_s",
        "test_extract_s",
        "match_s",
        "pair_total_s",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def summarize(rows: list[dict], thresholds: tuple[int, ...]) -> dict:
    def subset_summary(subset: list[dict]) -> dict:
        return {
            "num_pairs": len(subset),
            "precision": mean([r["precision"] for r in subset]),
            "recall": mean([r["recall"] for r in subset]),
            "mma_score": mean([r["mma_score"] for r in subset]),
            "mma_at_3": mean([r["mma_at_3"] for r in subset]),
            "mean_error": mean([r["mean_error"] for r in subset]),
            "match_count": mean([r["match_count"] for r in subset]),
            "pair_total_s": mean([r["pair_total_s"] for r in subset]),
            "mma_curve": [
                mean([r[f"mma_at_{t}"] for r in subset]) for t in thresholds
            ],
        }

    by_condition = defaultdict(list)
    by_pair = defaultdict(list)
    by_condition_pair = defaultdict(list)

    for row in rows:
        by_condition[row["condition"]].append(row)
        by_pair[row["pair"]].append(row)
        by_condition_pair[f"{row['condition']}|{row['pair']}"].append(row)

    # Per-sequence average MMA for ranking.
    by_sequence = defaultdict(list)
    for row in rows:
        by_sequence[row["sequence"]].append(row)

    sequence_ranks = []
    for sequence, seq_rows in by_sequence.items():
        sequence_ranks.append(
            {
                "sequence": sequence,
                "condition": seq_rows[0]["condition"],
                "avg_mma_score": mean([r["mma_score"] for r in seq_rows]),
                "avg_precision": mean([r["precision"] for r in seq_rows]),
                "avg_recall": mean([r["recall"] for r in seq_rows]),
                "avg_match_count": mean([r["match_count"] for r in seq_rows]),
            }
        )
    sequence_ranks.sort(key=lambda item: item["avg_mma_score"], reverse=True)

    return {
        "num_pairs": len(rows),
        "num_sequences": len(by_sequence),
        "thresholds": list(thresholds),
        "overall": subset_summary(rows),
        "by_condition": {
            key: subset_summary(value) for key, value in sorted(by_condition.items())
        },
        "by_pair": {
            key: subset_summary(value) for key, value in sorted(by_pair.items())
        },
        "by_condition_pair": {
            key: subset_summary(value)
            for key, value in sorted(by_condition_pair.items())
        },
        "top_sequences_by_mma": sequence_ranks[:10],
        "bottom_sequences_by_mma": list(reversed(sequence_ranks[-10:])),
    }


def plot_average_quality(summary: dict, output_path: Path) -> None:
    conditions = ["illumination", "viewpoint"]
    metrics = ["precision", "recall", "mma_score"]
    labels = ["Precision", "Recall", "MMA"]
    x = np.arange(len(labels))
    width = 0.32
    colors = {"illumination": "#4C78A8", "viewpoint": "#F58518"}

    fig, ax = plt.subplots(figsize=(10, 5))
    for offset, condition in zip((-width / 2, width / 2), conditions):
        values = [
            summary["by_condition"].get(condition, {}).get(metric, 0.0)
            for metric in metrics
        ]
        ax.bar(x + offset, values, width, label=condition.capitalize(), color=colors[condition])

    ax.set_title("Average Matching Quality (Full HPatches)")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    save_figure(fig, output_path)


def plot_average_mma_curves(summary: dict, thresholds: tuple[int, ...], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"illumination": "#4C78A8", "viewpoint": "#F58518", "overall": "#54A24B"}

    overall = summary["overall"]["mma_curve"]
    ax.plot(thresholds, overall, marker="o", linewidth=2, label="Overall", color=colors["overall"])

    for condition in ("illumination", "viewpoint"):
        if condition in summary["by_condition"]:
            ax.plot(
                thresholds,
                summary["by_condition"][condition]["mma_curve"],
                marker="o",
                linewidth=2,
                label=condition.capitalize(),
                color=colors[condition],
            )

    ax.set_title("Average MMA Across Thresholds (Full HPatches)")
    ax.set_xlabel("Reprojection threshold (pixels)")
    ax.set_ylabel("MMA")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)
    save_figure(fig, output_path)


def plot_quality_by_pair(summary: dict, output_path: Path) -> None:
    pairs = sorted(summary["by_pair"].keys(), key=lambda p: int(p.split("->")[1]))
    precision = [summary["by_pair"][p]["precision"] for p in pairs]
    recall = [summary["by_pair"][p]["recall"] for p in pairs]
    mma = [summary["by_pair"][p]["mma_score"] for p in pairs]

    x = np.arange(len(pairs))
    width = 0.25
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width, precision, width, label="Precision", color="#4C78A8")
    ax.bar(x, recall, width, label="Recall", color="#F58518")
    ax.bar(x + width, mma, width, label="MMA", color="#54A24B")

    ax.set_title("Average Matching Quality by Pair Difficulty")
    ax.set_xlabel("Reference → test pair")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(pairs)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    save_figure(fig, output_path)


def plot_counts_by_condition(summary: dict, output_path: Path) -> None:
    conditions = ["illumination", "viewpoint"]
    match_counts = [
        summary["by_condition"].get(c, {}).get("match_count", 0.0) for c in conditions
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(conditions, match_counts, color=["#4C78A8", "#F58518"])
    ax.set_title("Average Filtered Matches by Condition")
    ax.set_ylabel("Matches per pair")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    save_figure(fig, output_path)


def plot_top_sequences(summary: dict, output_path: Path) -> None:
    top = summary["top_sequences_by_mma"]
    if not top:
        return

    labels = [item["sequence"] for item in top]
    values = [item["avg_mma_score"] for item in top]
    colors = [
        "#4C78A8" if item["condition"] == "illumination" else "#F58518" for item in top
    ]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(len(labels)), values, color=colors)
    ax.set_title("Top 10 Sequences by Average MMA")
    ax.set_ylabel("Average MMA")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    save_figure(fig, output_path)


def run_full_matching_evaluation(
    reference_image: int = DEFAULT_REFERENCE_IMAGE,
    test_images: tuple[int, ...] = DEFAULT_TEST_IMAGES,
    ratio_threshold: float = DEFAULT_RATIO_THRESHOLD,
    thresholds: tuple[int, ...] = DEFAULT_THRESHOLDS,
    top_k: int = DEFAULT_TOP_K,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    figures_dir: Path = DEFAULT_FIGURES_DIR,
    example_lighting: str = DEFAULT_EXAMPLE_LIGHTING,
    example_viewpoint: str = DEFAULT_EXAMPLE_VIEWPOINT,
):
    sequences = list_image_sequences()
    if not sequences:
        raise RuntimeError("No HPatches sequences were found.")

    sift = create_sift()
    rows: list[dict] = []
    example_paths = []

    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    examples_dir = figures_dir / "examples"

    print("=" * 72)
    print("Full HPatches SIFT Matching Evaluation")
    print("=" * 72)
    print(f"Sequences: {len(sequences)}")
    print(f"Pairs per sequence: {list(test_images)} (ref={reference_image})")
    print(f"Total pairs planned: {len(sequences) * len(test_images)}")
    print()

    started = time.perf_counter()

    for index, sequence in enumerate(sequences, start=1):
        reference_cache: dict = {}
        print(f"[{index}/{len(sequences)}] {sequence}")

        for test_image in test_images:
            try:
                row, payload = evaluate_pair_metrics(
                    sequence=sequence,
                    reference_image=reference_image,
                    test_image=test_image,
                    ratio_threshold=ratio_threshold,
                    thresholds=thresholds,
                    sift=sift,
                    reference_cache=reference_cache,
                )
            except FileNotFoundError as exc:
                print(f"  skip {reference_image}->{test_image}: {exc}")
                continue

            rows.append(row)
            print(
                f"  {row['pair']}: matches={row['match_count']} "
                f"P={row['precision']:.3f} R={row['recall']:.3f} "
                f"MMA={row['mma_score']:.3f}"
            )

            # Keep a few qualitative examples only.
            if (
                sequence in {example_lighting, example_viewpoint}
                and test_image in {2, 5}
            ):
                condition_dir = (
                    examples_dir / "lighting"
                    if sequence.startswith("i_")
                    else examples_dir / "viewpoint"
                )
                example_path = save_example_match(
                    sequence,
                    reference_image,
                    test_image,
                    payload,
                    top_k,
                    condition_dir,
                )
                example_paths.append(str(example_path))

    elapsed = time.perf_counter() - started
    summary = summarize(rows, thresholds)
    summary["elapsed_seconds"] = elapsed
    summary["example_paths"] = example_paths

    csv_path = results_dir / "pair_metrics.csv"
    json_path = results_dir / "summary.json"
    write_csv(rows, csv_path, thresholds)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plot_average_quality(summary, figures_dir / "average_quality.png")
    plot_average_mma_curves(summary, thresholds, figures_dir / "average_mma_curve.png")
    plot_quality_by_pair(summary, figures_dir / "quality_by_pair.png")
    plot_counts_by_condition(summary, figures_dir / "average_match_counts.png")
    plot_top_sequences(summary, figures_dir / "top10_mma_sequences.png")

    print()
    print("=" * 72)
    print("FULL MATCHING EVALUATION COMPLETE")
    print("=" * 72)
    print(f"Pairs evaluated: {len(rows)}")
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"CSV: {csv_path}")
    print(f"Summary JSON: {json_path}")
    print(f"Figures: {figures_dir}")
    print(
        "Overall averages: "
        f"P={summary['overall']['precision']:.3f} "
        f"R={summary['overall']['recall']:.3f} "
        f"MMA={summary['overall']['mma_score']:.3f}"
    )

    return rows, summary


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run SIFT matching evaluation on all HPatches sequences for "
            "reference→test pairs, save metrics, and plot averages."
        )
    )
    parser.add_argument("--reference-image", type=int, default=DEFAULT_REFERENCE_IMAGE)
    parser.add_argument(
        "--test-images",
        type=int,
        nargs="+",
        default=list(DEFAULT_TEST_IMAGES),
        help="Test image ids to match against the reference (default: 2 3 4 5 6).",
    )
    parser.add_argument("--ratio", type=float, default=DEFAULT_RATIO_THRESHOLD)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    parser.add_argument("--example-lighting", default=DEFAULT_EXAMPLE_LIGHTING)
    parser.add_argument("--example-viewpoint", default=DEFAULT_EXAMPLE_VIEWPOINT)
    args = parser.parse_args()

    run_full_matching_evaluation(
        reference_image=args.reference_image,
        test_images=tuple(args.test_images),
        ratio_threshold=args.ratio,
        top_k=args.top_k,
        results_dir=args.results_dir,
        figures_dir=args.figures_dir,
        example_lighting=args.example_lighting,
        example_viewpoint=args.example_viewpoint,
    )


if __name__ == "__main__":
    main()
