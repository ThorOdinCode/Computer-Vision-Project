from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np


THIS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = THIS_DIR.parent
REPO_ROOT = PROJECT_DIR.parent

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from data_loader import list_image_sequences, load_sequence_image


DEFAULT_SAMPLE_SEQUENCE = "i_ajuntament"
DEFAULT_SAMPLE_IMAGE_IDS = (1, 2, 3, 4, 5)
DEFAULT_MAX_POINTS = 150
DEFAULT_OUTPUT_DIR = REPO_ROOT / "project" / "SIFT" / "sift_samples"
HPATCHES_IMAGE_IDS = range(1, 7)


def create_sift(
    nfeatures=0,
    nOctaveLayers=3,
    contrastThreshold=0.04,
    edgeThreshold=10,
    sigma=1.6,
):
    """
    Create a SIFT detector.
    """

    if hasattr(cv2, "SIFT_create"):
        return cv2.SIFT_create(
            nfeatures=nfeatures,
            nOctaveLayers=nOctaveLayers,
            contrastThreshold=contrastThreshold,
            edgeThreshold=edgeThreshold,
            sigma=sigma,
        )

    if hasattr(cv2, "xfeatures2d") and hasattr(cv2.xfeatures2d, "SIFT_create"):
        return cv2.xfeatures2d.SIFT_create(
            nfeatures=nfeatures,
            nOctaveLayers=nOctaveLayers,
            contrastThreshold=contrastThreshold,
            edgeThreshold=edgeThreshold,
            sigma=sigma,
        )

    raise RuntimeError(
        "OpenCV was built without SIFT support. Install a build that includes "
        "cv2.SIFT_create()."
    )


def detect_and_describe(image, sift=None):
    """
    Detect SIFT keypoints and compute descriptors.
    """

    if sift is None:
        sift = create_sift()

    if image.ndim == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    start = cv2.getTickCount()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    runtime = (cv2.getTickCount() - start) / cv2.getTickFrequency()

    return keypoints or [], descriptors, runtime


def visualize_keypoints(image, keypoints, max_points=150):
    """
    Draw the strongest keypoints on top of an RGB image.
    """

    output = image.copy()

    sorted_keypoints = sorted(
        keypoints,
        key=lambda kp: kp.response,
        reverse=True,
    )

    for kp in sorted_keypoints[:max_points]:
        x = int(round(kp.pt[0]))
        y = int(round(kp.pt[1]))
        radius = max(2, int(round(kp.size / 2.0)))

        cv2.circle(output, (x, y), radius, (0, 255, 0), 1)
        cv2.circle(output, (x, y), 2, (255, 0, 0), -1)

    return output


def save_rgb_image(image_rgb, output_path):
    """
    Save an RGB image using OpenCV.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(
        str(output_path),
        cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR),
    )


def fit_image_to_box(image_rgb, box_width, box_height):
    """
    Resize an image to fit inside a box while preserving aspect ratio.
    """

    image_height, image_width = image_rgb.shape[:2]
    scale = min(
        box_width / image_width,
        box_height / image_height,
    )

    new_width = max(1, int(round(image_width * scale)))
    new_height = max(1, int(round(image_height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC

    resized = cv2.resize(
        image_rgb,
        (new_width, new_height),
        interpolation=interpolation,
    )

    canvas = np.full((box_height, box_width, 3), 255, dtype=np.uint8)
    y0 = (box_height - new_height) // 2
    x0 = (box_width - new_width) // 2
    canvas[y0 : y0 + new_height, x0 : x0 + new_width] = resized

    return canvas


def make_labeled_tile(
    image_rgb,
    label,
    tile_width=540,
    tile_height=420,
    header_height=44,
):
    """
    Build a labeled tile for the montage.
    """

    if header_height >= tile_height:
        raise ValueError("header_height must be smaller than tile_height")

    tile = np.full((tile_height, tile_width, 3), 245, dtype=np.uint8)
    tile[:header_height, :] = (28, 28, 28)

    cv2.putText(
        tile,
        label,
        (12, header_height - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    tile[header_height:, :] = fit_image_to_box(
        image_rgb,
        tile_width,
        tile_height - header_height,
    )

    return tile


def build_montage(sample_results, sequence, grid_path):
    """
    Create and save a montage from a list of sample results.
    """

    if not sample_results:
        raise ValueError("No sample results were provided.")

    tile_width = 540
    tile_height = 420
    header_height = 44
    title_bar_height = 68
    margin = 12

    cols = min(3, len(sample_results))
    rows = math.ceil(len(sample_results) / cols)

    tiles = []
    for result in sample_results:
        stats = result["statistics"]
        label = (
            f"{result['image_number']}.ppm | "
            f"{stats['number_of_keypoints']} kp | "
            f"{stats['runtime_seconds']:.4f}s"
        )
        tiles.append(
            make_labeled_tile(
                result["overlay_rgb"],
                label,
                tile_width=tile_width,
                tile_height=tile_height,
                header_height=header_height,
            )
        )

    grid_width = (cols * tile_width) + ((cols + 1) * margin)
    grid_height = title_bar_height + ((rows + 1) * margin) + (rows * tile_height)

    canvas = np.full((grid_height, grid_width, 3), 255, dtype=np.uint8)
    canvas[:title_bar_height, :] = (36, 36, 36)

    cv2.putText(
        canvas,
        f"SIFT on HPatches sequence: {sequence}",
        (margin, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"{len(sample_results)} sample images with keypoints drawn in green",
        (margin, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )

    for idx, tile in enumerate(tiles):
        row = idx // cols
        col = idx % cols

        y0 = title_bar_height + margin + (row * (tile_height + margin))
        x0 = margin + (col * (tile_width + margin))

        canvas[y0 : y0 + tile_height, x0 : x0 + tile_width] = tile

    grid_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(grid_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

    return grid_path


def process_dataset(
    sample_sequence=DEFAULT_SAMPLE_SEQUENCE,
    sample_image_ids=DEFAULT_SAMPLE_IMAGE_IDS,
    max_points=DEFAULT_MAX_POINTS,
    output_dir=DEFAULT_OUTPUT_DIR,
):
    """
    Run SIFT on every image in the HPatches sequence dataset.

    A five-image sample montage is also saved for quick inspection.
    """

    sequences = list_image_sequences()

    if not sequences:
        raise RuntimeError("No HPatches sequences were found.")

    if sample_sequence not in sequences:
        raise ValueError(
            f"Sample sequence '{sample_sequence}' was not found in the dataset."
        )

    sift = create_sift()

    output_dir.mkdir(parents=True, exist_ok=True)

    total_sequences = 0
    total_images = 0
    total_keypoints = 0
    total_runtime = 0.0
    sample_results = []

    print(f"Found {len(sequences)} sequences.")
    print(f"Sampling from: {sample_sequence}")
    print("=" * 72)

    for sequence in sequences:
        sequence_images = 0
        sequence_keypoints = 0
        sequence_runtime = 0.0

        for image_number in HPATCHES_IMAGE_IDS:
            try:
                image_rgb = load_sequence_image(sequence, image_number)
            except FileNotFoundError:
                continue

            keypoints, descriptors, runtime = detect_and_describe(
                image_rgb,
                sift=sift,
            )

            overlay = visualize_keypoints(
                image_rgb,
                keypoints,
                max_points=max_points,
            )

            stats = {
                "number_of_keypoints": len(keypoints),
                "descriptor_shape": None if descriptors is None else descriptors.shape,
                "runtime_seconds": runtime,
            }

            sequence_images += 1
            sequence_keypoints += stats["number_of_keypoints"]
            sequence_runtime += runtime
            total_images += 1
            total_keypoints += stats["number_of_keypoints"]
            total_runtime += runtime

            if sequence == sample_sequence and image_number in sample_image_ids:
                sample_path = output_dir / f"{sequence}_{image_number}_sift.png"
                save_rgb_image(overlay, sample_path)

                sample_results.append(
                    {
                        "sequence": sequence,
                        "image_number": image_number,
                        "image_rgb": image_rgb,
                        "overlay_rgb": overlay,
                        "keypoints": keypoints,
                        "descriptors": descriptors,
                        "statistics": stats,
                        "sample_path": sample_path,
                    }
                )

        total_sequences += 1

        if sequence_images:
            avg_runtime = sequence_runtime / sequence_images
            avg_keypoints = sequence_keypoints / sequence_images
            print(
                f"{sequence:20s} | "
                f"{sequence_images} images | "
                f"{avg_keypoints:7.1f} avg keypoints | "
                f"{avg_runtime:.4f}s avg runtime"
            )

    sample_results.sort(key=lambda item: item["image_number"])

    grid_path = output_dir / f"{sample_sequence}_sift_grid.png"
    build_montage(sample_results, sample_sequence, grid_path)

    print("=" * 72)
    print(f"Saved sample montage: {grid_path}")
    print("Sample image files:")
    for result in sample_results:
        print(f"  {result['sample_path']}")

    print(
        f"Processed {total_images} images across {total_sequences} sequences "
        f"with {total_keypoints} total keypoints in {total_runtime:.2f}s."
    )

    return sample_results, grid_path


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run SIFT over the HPatches sequence dataset and save a five-image "
            "sample montage."
        )
    )
    parser.add_argument(
        "--sample-sequence",
        default=DEFAULT_SAMPLE_SEQUENCE,
        help="Sequence name used for the saved sample montage.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=DEFAULT_MAX_POINTS,
        help="Maximum number of keypoints to draw per image.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open the saved montage in a window after processing.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where sample images will be written.",
    )

    args = parser.parse_args()

    sample_results, grid_path = process_dataset(
        sample_sequence=args.sample_sequence,
        max_points=args.max_points,
        output_dir=args.output_dir,
    )

    if args.show:
        preview = cv2.imread(str(grid_path))

        if preview is None:
            print(f"Could not open preview image: {grid_path}")
        else:
            try:
                cv2.imshow("HPatches SIFT sample", preview)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            except cv2.error as exc:
                print(f"GUI preview is not available in this environment: {exc}")

    return sample_results, grid_path


if __name__ == "__main__":
    main()
